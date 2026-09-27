"""Data preparation for the Stunnerz MMM.

Two data issues are handled explicitly rather than by dropping rows:

1. Reporting gap: the first Tuesday of every month records zero sales while
   media spend continues. These days are imputed from the same weekday one
   week before and after, and flagged.
2. Structural break: from 2024-03-17 daily sales fall ~85% and keep decaying
   toward zero while spend continues, and sales values switch from full
   precision to two decimals (a likely change in the upstream feed). The
   modeling window ends before the break.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NON_CHANNEL = {"date", "total_sales", "promo", "weekday"}
BREAK_DATE = "2024-03-17"


def load_daily(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    missing = {"date", "total_sales", "promo", "weekday"} - set(df.columns)
    assert not missing, f"Missing required columns: {missing}"
    return df


def channel_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_CHANNEL and c != "sales_imputed"]


def impute_reporting_gaps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sales = df.set_index("date")["total_sales"]
    gaps = sales[sales <= 0].index
    for day in gaps:
        neighbours = [sales.get(day - pd.Timedelta(days=7)), sales.get(day + pd.Timedelta(days=7))]
        neighbours = [v for v in neighbours if v is not None and v > 0]
        sales.loc[day] = float(np.mean(neighbours))
    df["total_sales"] = sales.values
    df["sales_imputed"] = df["date"].isin(gaps).astype(int)
    return df


def to_weekly(df: pd.DataFrame, break_date: str = BREAK_DATE) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Aggregate to complete Tuesday-start weeks inside the modeling window."""
    channels = channel_columns(df)
    df = df[df["date"] < break_date].copy()
    df["week_start"] = df["date"].dt.to_period("W-MON").apply(lambda r: r.start_time)
    promo = pd.get_dummies(df["promo"], prefix="promo", drop_first=True, dtype=float)
    grouped = pd.concat([df[["week_start", "total_sales", *channels]], promo], axis=1).groupby("week_start")
    days = grouped.size()
    weekly = (
        grouped.sum()[days == 7]
        .reset_index()
        .rename(columns={"week_start": "date", "total_sales": "sales"})
    )
    controls = [c for c in weekly.columns if c.startswith("promo_")]
    return weekly, channels, controls


def prepare(path: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    return to_weekly(impute_reporting_gaps(load_daily(path)))
