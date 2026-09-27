# Bayesian Media Mix Model — Stunnerz Skateboards

**Author:** Halima Ladan · Data Scientist, Marketing Measurement & AI

A Bayesian media mix model (MMM) built in PyMC-Marketing on four years of weekly sales and spend across 12 paid media channels. The focus is not just fitting a model, but showing every judgment call behind it: data quality, identifiability risk, prior choice, convergence, and out-of-sample validation. Results are then translated into guidance a non-technical client can act on.

## The question

Which channels actually drive sales, how quickly does their effect decay, where do returns diminish, and how much should the client trust each answer?

## Data

| | |
|---|---|
| Grain | Daily source, aggregated to weekly (194 weeks, Dec 2020 – Sep 2024) |
| Target | Total sales |
| Media | 12 channels across Google, Bing, Facebook, and Pinterest |
| Controls | Promotion flags (big sale, holiday sale, other sale) |

## Approach

1. **Data quality checks.** Required-column assertions, and removal of an incomplete zero-sales tail that would otherwise read as a business collapse.
2. **Identifiability screening before modeling.** Flags sparse channels (high share of zero-spend weeks) and collinear pairs, e.g. Google Search Brand ↔ Bing Search Brand at r = 0.815. These are the places where a contribution may be prior-driven rather than data-driven.
3. **Model specification.** Geometric adstock (8-week max lag), logistic saturation, yearly Fourier seasonality, and promotion controls. Channel priors are scaled to spend share so no channel is forced to look effective.
4. **Prior predictive check.** Confirms the priors can generate plausible sales before seeing the data.
5. **Chronological holdout.** 80/20 time-ordered split. Test predictions carry over the last training observations, so adstock is handled correctly at the boundary.
6. **Convergence diagnostics.** Divergences, R-hat, bulk ESS, and trace plots.
7. **Validation.** Posterior predictive checks, residual analysis, and CRPS in and out of sample.
8. **Interpretation.** Contribution decomposition, response curves, and ROAS, with explicit caveats where channel overlap limits attribution.

## Key results

- **Sampler health:** 0 divergent transitions, R-hat ≤ 1.01 and bulk ESS > 900 on all reported parameters.
- **Drivers:** Search (brand and non-brand) and Facebook retargeting account for most modeled media contribution. Promotions add material lift, so media effects are not interpreted without them.
- **Carryover:** Posterior adstock decay is very low (α ≈ 0.03), so media effects land almost entirely in the week of spend.
- **Out of sample:** CRPS rises from 63.9K (train) to 111.2K (test). The model captures the level of sales but misses the largest event-driven spikes.
- **Trust:** Brand search on Google and Bing moves together strongly, so their individual ROAS should be read as shared credit, not separate proof. This is a case for a geo or holdout lift test before large budget shifts.

## Repository

| File | Contents |
|---|---|
| `bayesian_mmm_budget_optimization.ipynb` | Full workflow: EDA, modeling, diagnostics, and client-facing answers |
| `requirements.txt` | Pinned dependencies |
| `data/` | Simulated Stunnerz dataset |

## Run it

```bash
pip install -r requirements.txt
jupyter notebook bayesian_mmm_budget_optimization.ipynb
```

## Tools

Python, PyMC, PyMC-Marketing, ArviZ, pandas, NumPy, Matplotlib, Seaborn
