"""Each gate check must pass on a healthy model and fire on the failure it targets."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from acceptance_gate import FAIL, PASS, WARN, Thresholds, run_gate  # noqa: E402

RNG = np.random.default_rng(0)
CHANNELS = ["search", "social", "display"]


def healthy(**overrides):
    """Inputs describing a well-behaved fit; override one field to break it."""
    inputs = dict(
        n_divergent=0,
        rhat=np.full(20, 1.001),
        ess=np.full(20, 2000.0),
        y_test=np.full(30, 100.0),
        y_pred_mean=np.full(30, 110.0),                       # 10% error
        baseline_share=RNG.normal(0.6, 0.05, 4000),
        media_share=RNG.normal(0.3, 0.05, 4000),
        roas_draws={c: RNG.normal(3.0, 0.3, 4000) for c in CHANNELS},
        sd_ratios=pd.DataFrame({"saturation_beta": [0.3, 0.4, 0.2],
                                "saturation_lam": [0.3, 0.4, 0.3],
                                "adstock_alpha": [0.5, 0.6, 0.4]}, index=CHANNELS),
    )
    inputs.update(overrides)
    return run_gate(**inputs)


def status(report, name):
    return report.to_frame().loc[name, "status"]


def test_healthy_model_is_tier_a():
    report = healthy()
    assert report.tier == "A"
    assert set(report.to_frame()["status"]) == {PASS}
    assert set(report.channel_status.values()) == {"data-driven"}


def test_divergences_fail():
    report = healthy(n_divergent=1)
    assert status(report, "divergences") == FAIL and report.tier == "C"


def test_rhat_fails():
    report = healthy(rhat=np.array([1.0, 1.02]))
    assert status(report, "max R-hat") == FAIL and report.tier == "C"


def test_low_ess_fails():
    report = healthy(ess=np.array([2000.0, 150.0]))
    assert status(report, "min bulk ESS") == FAIL


@pytest.mark.parametrize("pred, expected", [(130.0, WARN), (150.0, FAIL)])
def test_out_of_sample_thresholds(pred, expected):
    report = healthy(y_pred_mean=np.full(30, pred))
    assert status(report, "test MAPE") == expected


def test_negative_baseline_fails():
    report = healthy(baseline_share=RNG.normal(-0.28, 0.15, 4000))
    assert status(report, "degenerate baseline") == FAIL and report.tier == "C"


def test_baseline_near_zero_warns():
    report = healthy(baseline_share=np.abs(RNG.normal(0.05, 0.03, 4000)))
    assert status(report, "degenerate baseline") == PASS
    assert status(report, "baseline share") == WARN and report.tier == "B"


def test_media_over_100_percent_fails():
    report = healthy(media_share=RNG.normal(1.06, 0.15, 4000))
    assert status(report, "media share <= 100%") == FAIL


def test_wide_roas_intervals_warn():
    wide = {c: RNG.lognormal(1.0, 1.2, 4000) for c in CHANNELS}  # like the real ROAS posteriors
    report = healthy(roas_draws=wide)
    assert status(report, "ROAS interval width") == WARN


def test_prior_driven_channels_flagged():
    ratios = pd.DataFrame({"saturation_beta": [0.9, 1.3, 0.3],
                           "saturation_lam": [0.95, 0.5, 0.3],
                           "adstock_alpha": [0.9, 0.9, 0.4]}, index=CHANNELS)
    report = healthy(sd_ratios=ratios)
    assert report.channel_status == {"search": "prior-driven", "social": "prior-data conflict",
                                     "display": "data-driven"}
    assert status(report, "identifiability") == WARN


def test_thresholds_are_configurable():
    strict = Thresholds(mape_warn=0.05)
    report = healthy(thresholds=strict)
    assert status(report, "test MAPE") == WARN
