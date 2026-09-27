"""Model specifications used in the audit and the acceptance gate."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pymc_extras.prior import Prior
from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.multidimensional import MMM

BRAND_SEARCH = ["google_search_brand", "bing_search_brand"]

INTERCEPT_PRIORS = {
    "A. Original priors": Prior("Normal", mu=0.2, sigma=0.1),
    "B. Positive baseline": Prior("HalfNormal", sigma=0.3),
    "C. Brand search as control": Prior("HalfNormal", sigma=0.3),
}


def spend_share_sigma(X: pd.DataFrame, channels: list[str]) -> np.ndarray:
    """Spend-share prior scale, built in the model's own channel order."""
    totals = X[channels].sum()
    return np.clip((totals / totals.sum()).to_numpy(), 0.01, None)


def fit_spec(name: str, X_train, y_train, X_test, channels, controls, seed: int,
             draws: int = 1_000, tune: int = 1_500, target_accept: float = 0.95) -> dict:
    Xtr, Xte, chans, ctrls = X_train.copy(), X_test.copy(), list(channels), list(controls)
    if name.startswith("C."):
        chans = [c for c in chans if c not in BRAND_SEARCH]
        for b in BRAND_SEARCH:
            scale = Xtr[b].max()
            Xtr[b], Xte[b] = Xtr[b] / scale, Xte[b] / scale
        ctrls += BRAND_SEARCH

    config = {
        "intercept": INTERCEPT_PRIORS[name],
        "saturation_beta": Prior("HalfNormal", sigma=spend_share_sigma(Xtr, chans), dims="channel"),
        "gamma_control": Prior("Normal", mu=0, sigma=1, dims="control"),
        "gamma_fourier": Prior("Laplace", mu=0, b=1, dims="fourier_mode"),
        "likelihood": Prior("TruncatedNormal", lower=0, sigma=Prior("HalfNormal", sigma=1)),
    }
    mmm = MMM(model_config=config, target_column="sales", date_column="date",
              channel_columns=chans, control_columns=ctrls,
              adstock=GeometricAdstock(l_max=8), saturation=LogisticSaturation(), yearly_seasonality=3)
    mmm.build_model(Xtr, y_train)
    assert list(mmm.model.coords["channel"]) == chans, "prior order must match model channel order"
    mmm.add_original_scale_contribution_variable(var=["channel_contribution", "intercept_contribution", "y"])
    mmm.sample_prior_predictive(Xtr, y_train, samples=4_000, random_seed=seed)
    mmm.fit(X=Xtr, y=y_train, target_accept=target_accept, chains=4, draws=draws, tune=tune,
            random_seed=seed, progressbar=False)
    pred = mmm.sample_posterior_predictive(Xte, include_last_observations=True, random_seed=seed,
                                           extend_idata=False, progressbar=False)
    pred = pred["y_original_scale"].unstack().transpose(..., "date")
    return {"name": name, "mmm": mmm, "X_train": Xtr, "channels": chans, "test_pred": pred}
