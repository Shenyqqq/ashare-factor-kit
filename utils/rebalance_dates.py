"""Shared rebalance-date generation for ML, IC, and backtest."""
from __future__ import annotations

import pandas as pd


def horizon_to_rebalance_freq(horizon: int) -> str:
    """Map hold period (trading days) to pandas resample rule."""
    if horizon <= 3:
        return "3D"
    if horizon <= 7:
        return "W-FRI"
    if horizon <= 15:
        return "2W-FRI"
    return "ME"


def get_rebalance_dates(
    dates: pd.DatetimeIndex,
    rebalance_freq: str,
) -> pd.DatetimeIndex:
    """Last actual trading day in each resample bucket.

    Period labels from ``resample`` (calendar ME / W-FRI / …) may fall on
    weekends or holidays.  Those labels must not be used as rebalance dates:
    take the last *trading* timestamp in each bucket instead.
    """
    dates = pd.DatetimeIndex(dates).sort_values()
    if len(dates) == 0:
        return dates
    # Values are trading timestamps; resample index is the (possibly non-trading)
    # period label — keep the values, drop empty buckets.
    bucket_last = pd.Series(dates, index=dates).resample(rebalance_freq).last().dropna()
    return pd.DatetimeIndex(bucket_last.to_numpy()).sort_values()
