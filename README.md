# Bayesian Media Mix Model — Stunnerz Skateboards

**Author:** Halima Ladan · Data Scientist, Marketing Measurement & AI

> **How this was built:** I used AI-assisted tooling for implementation and code review. Problem framing, methodology, modeling decisions, and interpretation are my own, and I can walk through and defend every choice in this repo.

A Bayesian media mix model (MMM) built in PyMC-Marketing on weekly sales and spend across 12 paid media channels, followed by an audit of that model and an automated acceptance gate that grades any fit without an analyst in the loop. The point is not just fitting a model, but showing which of its conclusions the data actually supports: data window, baseline, identifiability, convergence, and out-of-sample validation.

## The question

Which channels actually drive sales, how quickly does their effect decay, where do returns diminish, and how much should the client trust each answer?

## Data

| | |
|---|---|
| Source | Daily sales, spend, and promotion flags, Jan 2021 – Sep 2024 |
| Media | 12 channels across Google, Bing, Facebook, and Pinterest |
| Controls | Promotion flags (big sale, holiday sale, other sale) |
| Modeling window | 166 complete weeks, Jan 2021 – Mar 2024 (after the audit) |

## Notebook 1 — Baseline MMM

`bayesian_mmm_budget_optimization.ipynb`

Geometric adstock, logistic saturation, yearly Fourier seasonality, and promotion controls, with channel priors scaled to spend share. Includes collinearity and sparsity screening, prior and posterior predictive checks, a chronological 80/20 holdout with adstock carried across the boundary, R-hat / ESS / divergence checks, CRPS, response curves, ROAS, and client-facing answers.

## Notebook 2 — Model audit

`02_model_audit.ipynb`

Treats every judgment call in notebook 1 as something to check.

| Finding | What it means |
|---|---|
| **Zero-sales days are a monthly reporting gap.** All 39 fall on the first Tuesday of a month while spend continues. | Dropping them created an artificial ~14% dip every month. Now imputed and flagged. |
| **Structural break on 17 Mar 2024.** Sales fall ~85% and decay toward zero while spend continues; values switch from full precision to 2 decimals. | A likely upstream feed change. It sat inside the original test set. Ending the window before it cut test CRPS from ~111K to ~72K. |
| **Channel priors depended on library version.** Spend shares were built alphabetically; current PyMC-Marketing keeps column order. | Re-running today would silently give each channel another channel's prior. Fixed by aligning by name; versions pinned. |
| **The original spec has a degenerate baseline.** Media explains 106% of sales and the baseline is −28%, yet it has the best test error. | Out-of-sample error alone cannot catch this. A positive-baseline spec fixes the impossibility but leaves the baseline near zero, so the media/baseline split is prior-driven. |
| **Most channel effects are prior-driven.** Posterior spread is close to prior spread for most channels; carryover is not identified for 9 of 12. | ROAS is reported with 94% credible intervals and each channel is labeled data-driven, weakly identified, prior-driven, or in prior-data conflict. |

**Recommendation for the client:** fix the reporting gap and the March 2024 feed issue at the source, and run a brand search holdout or geo test. That experiment would calibrate both the brand search effect and the baseline, which the observational data cannot separate.

All audit fits use 4 chains, 1,500 tuning and 1,000 draws, with target_accept 0.95, and seeded prior and posterior sampling.

## Notebook 3 — Automated acceptance gate

`03_acceptance_gate.ipynb` · `src/acceptance_gate.py`

The audit's judgment calls, turned into code. Each check returns PASS, WARN, or FAIL, and the model gets a tier: **A** publish, **B** publish with caveats shown, **C** do not publish.

| Check | Threshold |
|---|---|
| Divergences | 0 |
| Max R-hat | ≤ 1.01 |
| Min bulk ESS | ≥ 400 |
| Test MAPE | warn > 25%, fail > 40% |
| Degenerate baseline | P(baseline < 0) ≤ 5% |
| Baseline share | warn if mean < 10% |
| Media share | P(media > 100% of sales) ≤ 5% |
| ROAS interval width | warn if median (97% − 3%) / mean > 3 |
| Identifiability | warn if > 50% of channels are prior-driven |

| Spec | Tier | Why |
|---|---|---|
| A. Original priors | **C — reject** | 1 divergence, negative baseline, media > 100% of sales, despite the best test MAPE |
| B. Positive baseline | **B — caveats** | Healthy sampling; baseline near zero and 10 of 12 channels prior-driven |
| C. Brand search as control | **B — caveats** | Healthy sampling; baseline near zero and weaker test fit |

Every check has a unit test that breaks exactly one thing and asserts the check fires (`tests/test_acceptance_gate.py`, 12 tests).

## Repository

| File | Contents |
|---|---|
| `bayesian_mmm_budget_optimization.ipynb` | Baseline MMM: EDA, modeling, diagnostics, and client-facing answers |
| `02_model_audit.ipynb` | Audit: data window, prior alignment, baseline, identifiability, ROAS uncertainty |
| `03_acceptance_gate.ipynb` | Fits the three specifications and grades each with the gate |
| `src/data_prep.py` | Data preparation: reporting-gap imputation and structural-break window |
| `src/mmm_specs.py` | Model specifications, with priors aligned to the model's channel order |
| `src/acceptance_gate.py` | Automated acceptance gate: checks, thresholds, and tiers |
| `tests/test_acceptance_gate.py` | Unit tests proving each gate check fires |
| `stunnerz_skateboards_simulated_data.csv` | Simulated Stunnerz dataset |
| `requirements.txt` | Pinned dependencies |

## Run it

```bash
pip install -r requirements.txt
jupyter notebook 02_model_audit.ipynb
pytest tests
```

## Tools

Python, PyMC, PyMC-Marketing, ArviZ, pandas, NumPy, Matplotlib, Seaborn, pytest
