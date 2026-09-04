"""
backtest/benchmark.py — index cumulative NAV over rebalance windows.

Equal-weight universe benchmark is handled in quantile.py via the same
rebalance + simulate_period path as strategy portfolios.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def index_period_return(
    index_series: pd.Series,
    hold_dates: pd.DatetimeIndex,
    execution_date: pd.Timestamp,
) -> float:
    """
    Index cumulative return over hold window (daily chain, not point p1/p0).

    Uses close-to-close returns; first day vs prior close when execution day
    is not the first row.
    """
    idx = index_series.sort_index()
    sub = idx.reindex(hold_dates)
    arr = sub.to_numpy(dtype=np.float64)
    if len(arr) == 0 or not np.any(np.isfinite(arr)):
        return np.nan

    rets = np.zeros(len(arr))
    exec_loc = hold_dates.get_loc(execution_date)
    if isinstance(exec_loc, slice):
        exec_loc = exec_loc.start or 0

    if exec_loc > 0:
        prev = idx.asof(hold_dates[exec_loc - 1])
    else:
        prev = idx.asof(execution_date - pd.Timedelta(days=1))
    if prev and prev > 0 and np.isfinite(arr[exec_loc]):
        rets[exec_loc] = arr[exec_loc] / prev - 1.0

    for t in range(exec_loc + 1, len(arr)):
        if np.isfinite(arr[t - 1]) and arr[t - 1] > 0 and np.isfinite(arr[t]):
            rets[t] = arr[t] / arr[t - 1] - 1.0

    nav = np.cumprod(1.0 + rets)
    return float(nav[-1] - 1.0)
