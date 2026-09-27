# Bayesian Media Mix Model — Stunnerz Skateboards

**Author:** Halima Ladan · Data Scientist, Marketing Measurement & AI

A Bayesian media mix model (MMM) built in PyMC-Marketing on weekly sales and spend across 12 paid media channels, followed by an audit of that model. The point is not just fitting a model, but showing which of its conclusions the data actually supports: data window, baseline, identifiability, convergence, and out-of-sample validation.

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

All audit fits use 4 chains, 1,500 tuning and 1,000 draws, with target_accept 0.95.

## Repository

| File | Contents |
|---|---|
| `bayesian_mmm_budget_optimization.ipynb` | Baseline MMM: EDA, modeling, diagnostics, and client-facing answers |
| `02_model_audit.ipynb` | Audit: data window, prior alignment, baseline, identifiability, ROAS uncertainty |
| `src/data_prep.py` | Data preparation: reporting-gap imputation and structural-break window |
| `stunnerz_skateboards_simulated_data.csv` | Simulated Stunnerz dataset |
| `requirements.txt` | Pinned dependencies |

## Run it

```bash
pip install -r requirements.txt
jupyter notebook 02_model_audit.ipynb
```

## Tools

Python, PyMC, PyMC-Marketing, ArviZ, pandas, NumPy, Matplotlib, Seaborn
