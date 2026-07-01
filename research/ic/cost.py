"""Rough IC drag from rank-turnover proxy."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import COMMISSION_RATE, STAMP_DUTY, SLIPPAGE_RATE


def rank_autocorr_turnover(
    factor: pd.DataFrame,
    dates: pd.DatetimeIndex | None = None,
) -> float:
    """
    Turnover proxy: mean(1 - Spearman rank autocorr) over consecutive dates.

    Higher → more rank churn → higher implied trading cost.
    """
    if dates is not None:
        sub = factor.reindex(dates).dropna(how="all")
    else:
        sub = factor.dropna(how="all")
    if len(sub) < 2:
        return np.nan

    turnovers = []
    prev_rank = None
    for _, row in sub.iterrows():
        valid = row.dropna()
        if len(valid) < 10:
            prev_rank = None
            continue
        ranks = valid.rank(method="average")
        if prev_rank is not None:
            common = ranks.index.intersection(prev_rank.index)
            if len(common) >= 10:
                corr = ranks.loc[common].corr(prev_rank.loc[common], method="spearman")
                if np.isfinite(corr):
                    turnovers.append(1.0 - corr)
        prev_rank = ranks
    return float(np.mean(turnovers)) if turnovers else np.nan


def round_trip_cost_fraction() -> float:
    """One-way buy + one-way sell cost (commission + slippage + stamp on sell)."""
    buy = COMMISSION_RATE + SLIPPAGE_RATE
    sell = COMMISSION_RATE + STAMP_DUTY + SLIPPAGE_RATE
    return buy + sell


def estimate_ic_after_cost(ic_mean: float, turnover: float) -> float:
    """
    Simple linear drag: IC_after_cost ≈ IC_mean - turnover × round_trip_cost.

    turnover ∈ [0, 2] typical for rank autocorr proxy (0=no change, 1=full replace).
    """
    if not np.isfinite(ic_mean) or not np.isfinite(turnover):
        return np.nan
    drag = turnover * round_trip_cost_fraction()
    return float(ic_mean - drag)
