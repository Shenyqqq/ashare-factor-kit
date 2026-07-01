"""
backtest/return_engine.py — per-stock buy-and-hold return paths.

Logic (2-stock sanity example in module docstring at bottom):
  - Buy at execution-day open (or prior close if no open).
  - Day-0 return: close/open − 1; thereafter close/close_prev − 1.
  - Suspension → daily return 0 (NAV flat), never fillna(0) on prices.
  - Portfolio NAV = Σ weight_i × stock_NAV_i (not daily cross-sectional mean).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.execution import TradeRules
from backtest.portfolio import equal_weights


def _suspension_matrix(
    hold_dates: pd.DatetimeIndex,
    stocks: list[str],
    close: pd.DataFrame,
    rules: TradeRules,
) -> np.ndarray:
    """(T, S) bool — True where stock is suspended (NAV flat)."""
    T, S = len(hold_dates), len(stocks)
    out = np.zeros((T, S), dtype=bool)
    for j, stock in enumerate(stocks):
        for i, dt in enumerate(hold_dates):
            out[i, j] = rules.is_suspended(stock, dt, close)
    return out


def period_daily_returns(
    close: pd.DataFrame,
    hold_dates: pd.DatetimeIndex,
    stocks: list[str],
    execution_date: pd.Timestamp,
    open_prices: pd.DataFrame | None,
    rules: TradeRules,
) -> np.ndarray:
    """
    On-demand (T, S) daily simple returns for one hold window.

    Does not cache full price matrix — only slices requested dates/columns.
    """
    if not stocks or len(hold_dates) == 0:
        return np.zeros((0, 0))

    sub_close = close.reindex(index=hold_dates, columns=stocks)
    arr = sub_close.to_numpy(dtype=np.float64)
    T, S = arr.shape
    rets = np.zeros((T, S), dtype=np.float64)
    susp = _suspension_matrix(hold_dates, stocks, close, rules)

    exec_idx = hold_dates.get_loc(execution_date)
    if isinstance(exec_idx, slice):
        exec_idx = exec_idx.start or 0

    use_open = open_prices is not None
    if use_open and execution_date in open_prices.index:
        open_row = open_prices.reindex(columns=stocks).loc[execution_date].to_numpy(dtype=np.float64)
        valid = np.isfinite(open_row) & (open_row > 0) & np.isfinite(arr[exec_idx]) & ~susp[exec_idx]
        rets[exec_idx, valid] = arr[exec_idx, valid] / open_row[valid] - 1.0
    elif exec_idx > 0:
        prev_day = hold_dates[exec_idx - 1]
        if prev_day in close.index:
            prev = close.reindex(columns=stocks).loc[prev_day].to_numpy(dtype=np.float64)
            valid = np.isfinite(prev) & (prev > 0) & np.isfinite(arr[exec_idx]) & ~susp[exec_idx]
            rets[exec_idx, valid] = arr[exec_idx, valid] / prev[valid] - 1.0
    else:
        loc = close.index.get_loc(execution_date)
        if isinstance(loc, int) and loc > 0:
            prev = close.iloc[loc - 1].reindex(stocks).to_numpy(dtype=np.float64)
            valid = np.isfinite(prev) & (prev > 0) & np.isfinite(arr[exec_idx]) & ~susp[exec_idx]
            rets[exec_idx, valid] = arr[exec_idx, valid] / prev[valid] - 1.0

    for t in range(exec_idx + 1, T):
        prev = arr[t - 1]
        curr = arr[t]
        valid = np.isfinite(prev) & (prev > 0) & np.isfinite(curr) & ~susp[t]
        rets[t, valid] = curr[valid] / prev[valid] - 1.0

    rets[susp] = 0.0
    return rets


def stock_nav_paths(daily_rets: np.ndarray) -> np.ndarray:
    """Cumulative product per column; each stock NAV starts at 1.0."""
    if daily_rets.size == 0:
        return daily_rets
    return np.cumprod(1.0 + daily_rets, axis=0)


def portfolio_nav_from_stock_navs(
    stock_navs: np.ndarray,
    weights: dict[str, float],
    stocks: list[str],
) -> np.ndarray:
    """portfolio_NAV[t] = Σ w_i × stock_NAV_i[t], weights sum to 1."""
    if stock_navs.size == 0:
        return np.array([])
    w = np.array([weights.get(s, 0.0) for s in stocks], dtype=np.float64)
    if w.sum() <= 0:
        w = np.ones(len(stocks)) / max(len(stocks), 1)
    else:
        w = w / w.sum()
    return stock_navs @ w


def simulate_period(
    close: pd.DataFrame,
    hold_dates: pd.DatetimeIndex,
    stocks: list[str],
    execution_date: pd.Timestamp,
    open_prices: pd.DataFrame | None,
    rules: TradeRules,
    cost_fraction: float,
) -> tuple[float, np.ndarray]:
    """
    Run one hold window; return (period_return, daily_portfolio_nav).

    Cost applied at period start: nav *= (1 − cost_fraction).
    period_return = final_nav − 1.
    """
    if not stocks or len(hold_dates) == 0:
        return np.nan, np.array([])

    daily_rets = period_daily_returns(
        close, hold_dates, stocks, execution_date, open_prices, rules,
    )
    stock_navs = stock_nav_paths(daily_rets)
    weights = equal_weights(stocks)
    port_nav = portfolio_nav_from_stock_navs(stock_navs, weights, stocks)

    if len(port_nav) == 0:
        return np.nan, port_nav

    port_nav = port_nav * (1.0 - cost_fraction)
    period_ret = float(port_nav[-1] - 1.0)
    return period_ret, port_nav


# ── Buy-hold sanity check (2-stock example) ──────────────────────────────────
# Stock A: open=10, close sequence 10→11 (+10% day0), then 11→11.55 (+5% day1)
# Stock B: open=20, close sequence 20→21 (+5% day0), then 21→21 (+0% day1)
# Equal weight → day0 port ret = 0.5*0.10 + 0.5*0.05 = 7.5%
# After day1: A_nav=1.155, B_nav=1.05 → port_nav = 0.5*1.155 + 0.5*1.05 = 1.1025
# With zero cost, period_return = 10.25%  (NOT mean of daily mean returns compounded)
