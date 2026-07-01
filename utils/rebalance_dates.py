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
    """Last actual trading day in each resample bucket."""
    dates = pd.DatetimeIndex(dates).sort_values()
    return (
        pd.Series(1, index=dates)
        .resample(rebalance_freq)
        .last()
        .index
        .intersection(dates)
    )
