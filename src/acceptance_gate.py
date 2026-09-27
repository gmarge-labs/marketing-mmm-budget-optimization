"""Automated acceptance gate for a fitted Bayesian MMM.

Turns the manual judgment calls from the model audit into explicit checks that
run without human review. Each check returns PASS, WARN, or FAIL with the
measured value and the threshold it was compared to. The model is assigned a
tier from the combined result:

    Tier A  every check passes              -> publish results
    Tier B  warnings only                   -> publish with caveats shown to the user
    Tier C  at least one failure            -> do not publish; route to review

The checks are split into pure functions on arrays (easy to test) and an
adapter, `evaluate_mmm`, that pulls those arrays out of a PyMC-Marketing fit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclass(frozen=True)
class Thresholds:
    # Sampler health
    max_divergences: int = 0
    max_rhat: float = 1.01
    min_ess_bulk: float = 400
    # Out-of-sample accuracy (test MAPE)
    mape_warn: float = 0.25
    mape_fail: float = 0.40
    # Decomposition plausibility
    max_prob_negative_baseline: float = 0.05
    min_baseline_share_warn: float = 0.10
    max_prob_media_over_100pct: float = 0.05
    # Uncertainty and identifiability
    max_median_roas_rel_width: float = 3.0  # (97% - 3%) / mean, median across channels
    prior_driven_sd_ratio: float = 0.8
    max_share_prior_driven: float = 0.5


@dataclass
class CheckResult:
    name: str
    status: str
    value: float | str
    threshold: str
    detail: str = ""


@dataclass
class GateReport:
    tier: str
    checks: list[CheckResult] = field(default_factory=list)
    channel_status: dict[str, str] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.tier in {"A", "B"}

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(c) for c in self.checks]).set_index("name")

    def to_dict(self) -> dict:
        return {"tier": self.tier, "checks": [asdict(c) for c in self.checks],
                "channel_status": self.channel_status}


# ---------------------------------------------------------------- pure checks

def check_divergences(n_divergent: int, t: Thresholds) -> CheckResult:
    status = PASS if n_divergent <= t.max_divergences else FAIL
    return CheckResult("divergences", status, int(n_divergent), f"<= {t.max_divergences}",
                       "Divergent transitions mean the sampler could not explore part of the posterior.")


def check_rhat(rhat: np.ndarray, t: Thresholds) -> CheckResult:
    worst = float(np.nanmax(rhat))
    return CheckResult("max R-hat", PASS if worst <= t.max_rhat else FAIL, round(worst, 3),
                       f"<= {t.max_rhat}", "Chains disagree if R-hat is above threshold.")


def check_ess(ess: np.ndarray, t: Thresholds) -> CheckResult:
    worst = float(np.nanmin(ess))
    return CheckResult("min bulk ESS", PASS if worst >= t.min_ess_bulk else FAIL, round(worst),
                       f">= {t.min_ess_bulk:g}", "Too few effective draws to summarize the posterior reliably.")


def check_out_of_sample(y_true: np.ndarray, y_pred_mean: np.ndarray, t: Thresholds) -> CheckResult:
    mape = float(np.mean(np.abs(y_pred_mean - y_true) / np.abs(y_true)))
    status = PASS if mape <= t.mape_warn else WARN if mape <= t.mape_fail else FAIL
    return CheckResult("test MAPE", status, round(mape, 3), f"warn > {t.mape_warn}, fail > {t.mape_fail}",
                       "Necessary but not sufficient: a model can predict well with an impossible decomposition.")


def check_baseline(baseline_share: np.ndarray, t: Thresholds) -> list[CheckResult]:
    """baseline_share: posterior draws of baseline / total sales."""
    p_neg = float(np.mean(baseline_share < 0))
    degenerate = CheckResult(
        "degenerate baseline", PASS if p_neg <= t.max_prob_negative_baseline else FAIL, round(p_neg, 3),
        f"P(baseline < 0) <= {t.max_prob_negative_baseline}",
        "A negative baseline means media is absorbing organic demand.")
    mean = float(np.mean(baseline_share))
    low = CheckResult(
        "baseline share", PASS if mean >= t.min_baseline_share_warn else WARN, round(mean, 3),
        f"warn < {t.min_baseline_share_warn}",
        "A baseline near zero is usually set by the prior or constraint, not the data; calibrate with an experiment.")
    return [degenerate, low]


def check_media_share(media_share: np.ndarray, t: Thresholds) -> CheckResult:
    p_over = float(np.mean(media_share > 1))
    return CheckResult("media share <= 100%", PASS if p_over <= t.max_prob_media_over_100pct else FAIL,
                       round(p_over, 3), f"P(media > 100% of sales) <= {t.max_prob_media_over_100pct}",
                       "Media cannot explain more than all of sales.")


def check_roas_width(roas_draws: dict[str, np.ndarray], t: Thresholds) -> CheckResult:
    widths = []
    for draws in roas_draws.values():
        lo, hi = np.percentile(draws, [3, 97])
        widths.append((hi - lo) / max(abs(float(np.mean(draws))), 1e-9))
    med = float(np.median(widths))
    return CheckResult("ROAS interval width", PASS if med <= t.max_median_roas_rel_width else WARN,
                       round(med, 2), f"median relative width <= {t.max_median_roas_rel_width}",
                       "Wide intervals: show ranges, not a channel ranking.")


def classify_channels(sd_ratios: pd.DataFrame, t: Thresholds) -> dict[str, str]:
    """sd_ratios: index=channel, columns include saturation_beta and saturation_lam (posterior sd / prior sd)."""
    worst = sd_ratios[["saturation_beta", "saturation_lam"]].max(axis=1)
    status = np.select(
        [sd_ratios["saturation_beta"] > 1.0, worst > t.prior_driven_sd_ratio, worst > 0.5],
        ["prior-data conflict", "prior-driven", "weakly identified"], default="data-driven")
    return dict(zip(sd_ratios.index, status))


def check_identifiability(channel_status: dict[str, str], t: Thresholds) -> CheckResult:
    bad = [c for c, s in channel_status.items() if s in {"prior-driven", "prior-data conflict"}]
    share = len(bad) / max(len(channel_status), 1)
    return CheckResult("identifiability", PASS if share <= t.max_share_prior_driven else WARN, round(share, 2),
                       f"share prior-driven <= {t.max_share_prior_driven}",
                       f"{len(bad)} of {len(channel_status)} channels are mostly prior-driven; show them as unmeasured.")


def assign_tier(checks: list[CheckResult]) -> str:
    statuses = {c.status for c in checks}
    return "C" if FAIL in statuses else "B" if WARN in statuses else "A"


def run_gate(*, n_divergent, rhat, ess, y_test, y_pred_mean, baseline_share, media_share,
             roas_draws, sd_ratios, thresholds: Thresholds | None = None) -> GateReport:
    t = thresholds or Thresholds()
    channel_status = classify_channels(sd_ratios, t)
    checks = [
        check_divergences(n_divergent, t),
        check_rhat(np.asarray(rhat), t),
        check_ess(np.asarray(ess), t),
        check_out_of_sample(np.asarray(y_test), np.asarray(y_pred_mean), t),
        *check_baseline(np.asarray(baseline_share), t),
        check_media_share(np.asarray(media_share), t),
        check_roas_width(roas_draws, t),
        check_identifiability(channel_status, t),
    ]
    return GateReport(tier=assign_tier(checks), checks=checks, channel_status=channel_status)


# ------------------------------------------------------ PyMC-Marketing adapter

def evaluate_mmm(mmm, X_train: pd.DataFrame, y_train: pd.Series, y_test: pd.Series,
                 test_pred, channels: list[str], thresholds: Thresholds | None = None) -> GateReport:
    """Grade a fitted PyMC-Marketing MMM.

    Requires `channel_contribution` and `intercept_contribution` to have been added
    as original-scale variables, and a prior group from `sample_prior_predictive`.
    `test_pred` is the posterior predictive `y_original_scale` for the test period
    with dims (chain, draw, date).
    """
    import arviz as az

    idata = mmm.idata
    post = idata.posterior
    summary = az.summary(idata, var_names=["intercept_contribution", "saturation_beta", "saturation_lam",
                                           "adstock_alpha", "gamma_control"])

    total = float(y_train.sum())
    base = post["intercept_contribution_original_scale"]
    base = base.sum("date") if "date" in base.dims else base * len(y_train)
    contrib = post["channel_contribution_original_scale"].sum("date")

    spend = X_train[channels].sum()
    roas = {ch: (contrib.sel(channel=ch) / spend[ch]).values.ravel() for ch in channels}

    ratios = pd.DataFrame({
        var: {ch: float(post[var].sel(channel=ch).std() / idata.prior[var].sel(channel=ch).std()) for ch in channels}
        for var in ["saturation_beta", "saturation_lam", "adstock_alpha"]
    })

    return run_gate(
        n_divergent=int(idata.sample_stats["diverging"].sum()),
        rhat=summary["r_hat"].to_numpy(),
        ess=summary["ess_bulk"].to_numpy(),
        y_test=y_test.to_numpy(),
        y_pred_mean=test_pred.mean(dim=("chain", "draw")).to_numpy(),
        baseline_share=(base / total).values.ravel(),
        media_share=(contrib.sum("channel") / total).values.ravel(),
        roas_draws=roas,
        sd_ratios=ratios,
        thresholds=thresholds,
    )
