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
    """(T, S) bool — True where stock is suspended (NAV flat).

    Vectorized equivalent of ``TradeRules.is_suspended`` over the hold window:
    missing/non-positive close, or (when volume is provided) missing/non-positive
    volume. Semantics unchanged vs the former Python double loop.
    """
    if not stocks or len(hold_dates) == 0:
        return np.zeros((len(hold_dates), len(stocks)), dtype=bool)
    sub_close = close.reindex(index=hold_dates, columns=stocks)
    px = sub_close.to_numpy(dtype=np.float64, copy=False)
    susp = ~np.isfinite(px) | (px <= 0)
    vol_df = getattr(rules, "volume", None)
    if vol_df is not None:
        sub_vol = vol_df.reindex(index=hold_dates, columns=stocks)
        vol = sub_vol.to_numpy(dtype=np.float64, copy=False)
        susp |= ~np.isfinite(vol) | (vol <= 0)
    return susp


def period_daily_returns(
    close: pd.DataFrame,
    hold_dates: pd.DatetimeIndex,
    stocks: list[str],
    execution_date: pd.Timestamp,
    open_prices: pd.DataFrame | None,
    rules: TradeRules,
    stuck_exit_days: dict[str, int] | None = None,
) -> np.ndarray:
    """
    On-demand (T, S) daily simple returns for one hold window.

    Does not cache full price matrix — only slices requested dates/columns.

    stuck_exit_days : optional map of stuck stock → first sellable day index
        within hold_dates (used by Bug 2 fix to freeze a stuck stock's NAV
        after its mid-window exit). Computed by ``_stuck_exit_days`` and
        consumed here so the freeze lives next to the return computation.
    """
    if not stocks or len(hold_dates) == 0:
        return np.zeros((0, 0))

    sub_close = close.reindex(index=hold_dates, columns=stocks)
    arr = sub_close.to_numpy(dtype=np.float64)
    # Local column-wise ffill for prev-day denominator only.
    # Suspended days have NaN close; without ffill the resumption-day return
    # would divide by NaN and the gap return (e.g. +50% jump after halting)
    # would be silently dropped. arr_filled gives the last finite close per
    # stock so resumption-day rets = close[t] / last_finite_close - 1.
    # NOTE: only used as the *previous-day* denominator in the t-loop below;
    # `arr` (with NaN on susp days) is still used for `curr` and `valid`, and
    # `rets[susp] = 0` is re-applied at the end so susp days stay NAV-flat.
    arr_filled = sub_close.ffill().to_numpy(dtype=np.float64)
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
        # Use ffilled prev so a suspension gap's denominator is the last
        # finite close (resumption-day jump enters NAV). curr still uses
        # `arr` so susp days stay NaN-valid=False and rets[susp]=0 below
        # re-asserts NAV-flat on suspended days.
        prev = arr_filled[t - 1]
        curr = arr[t]
        valid = np.isfinite(prev) & (prev > 0) & np.isfinite(curr) & ~susp[t]
        rets[t, valid] = curr[valid] / prev[valid] - 1.0

    # Bug 2 fix: stuck (limit-down) stocks retried daily for sale. If a
    # sellable day `d` is found within the hold window, freeze the stock's
    # returns at 0 from d+1 onward (capital withdrawn at close[d], held as
    # cash thereafter). Simplified version per fix spec — weight stays equal
    # but that slice earns no further return after exit.
    if stuck_exit_days:
        stock_idx = {s: j for j, s in enumerate(stocks)}
        for s, d in stuck_exit_days.items():
            j = stock_idx.get(s)
            if j is None:
                continue
            if d + 1 < T:
                rets[d + 1:, j] = 0.0

    rets[susp] = 0.0
    return rets


def stock_nav_paths(daily_rets: np.ndarray) -> np.ndarray:
    """Cumulative product per column; each stock NAV starts at 1.0."""
    if daily_rets.size == 0:
        return daily_rets
    return np.cumprod(1.0 + daily_rets, axis=0)


def _stuck_exit_days(
    close: pd.DataFrame,
    hold_dates: pd.DatetimeIndex,
    stocks: list[str],
    execution_date: pd.Timestamp,
    rules: TradeRules,
    stuck_stocks: set[str] | None,
) -> dict[str, int]:
    """First sellable day index (within hold_dates, after exec) per stuck stock.

    Bug 2 fix: instead of carrying a one-word-limit-down stuck stock for the
    full hold window, retry selling each trading day after execution. Returns
    `{stock: day_index}` for stocks that became sellable before window end.

    Uses a vectorized suspension block + limit-down mask; falls back to
    ``rules.can_sell`` only if masks layout is unexpected (should be rare).
    """
    if not stuck_stocks or not stocks or len(hold_dates) == 0:
        return {}
    stock_set = set(stocks)
    candidates = [s for s in stuck_stocks if s in stock_set]
    if not candidates:
        return {}
    exec_idx = hold_dates.get_loc(execution_date)
    if isinstance(exec_idx, slice):
        exec_idx = exec_idx.start or 0
    T = len(hold_dates)
    if exec_idx + 1 >= T:
        return {}

    post_dates = hold_dates[exec_idx + 1:]
    susp = _suspension_matrix(post_dates, candidates, close, rules)
    # limit-down block (same semantics as TradeRules.can_sell)
    block = np.zeros_like(susp, dtype=bool)
    masks = getattr(rules, "masks", None)
    cfg = getattr(rules, "config", None)
    strict = bool(getattr(cfg, "strict_limit_mode", True)) if cfg is not None else True
    if masks is not None:
        key = "limit_down_open" if strict else "limit_down"
        ld = masks.get(key)
        if ld is not None:
            sub_ld = ld.reindex(index=post_dates, columns=candidates)
            # True / 1 → blocked; NaN → not blocked (same as missing .at → False)
            block = sub_ld.fillna(False).to_numpy(dtype=bool)
    sellable = ~(susp | block)
    out: dict[str, int] = {}
    # first True along time axis
    any_sell = sellable.any(axis=0)
    if any_sell.any():
        first = sellable.argmax(axis=0)  # 0 if all False; gated by any_sell
        for j, s in enumerate(candidates):
            if any_sell[j]:
                out[s] = exec_idx + 1 + int(first[j])
    return out


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
    stuck_stocks: set[str] | None = None,
    weights: dict[str, float] | None = None,
) -> tuple[float, np.ndarray, set[str]]:
    """
    Run one hold window; return (period_return, daily_portfolio_nav, sold_mid_window).

    Cost applied at period start: nav *= (1 − cost_fraction).
    period_return = final_nav − 1.

    stuck_stocks : names from ``rebalance_holdings`` that couldn't be sold on
        execution day (one-word limit down). Each is retried daily for sale
        within the hold window; the first sellable day `d` freezes that
        stock's NAV at close[d] (returns 0 from d+1). ``sold_mid_window`` is
        the subset of stuck_stocks that found a sellable day before window
        end — callers should drop them from next period's carried holdings.

    weights : optional {code: weight} summing to ~1 among *stocks*.
        ``None`` → ``equal_weights(stocks)``（默认路径，与旧实验 bit-identical）。
    """
    if not stocks or len(hold_dates) == 0:
        return np.nan, np.array([]), set()

    exit_days = _stuck_exit_days(
        close, hold_dates, stocks, execution_date, rules, stuck_stocks,
    )
    daily_rets = period_daily_returns(
        close, hold_dates, stocks, execution_date, open_prices, rules,
        stuck_exit_days=exit_days,
    )
    stock_navs = stock_nav_paths(daily_rets)
    # 默认等权：不经 optimize 模块，保持旧路径 bit-identical
    w = equal_weights(stocks) if weights is None else weights
    port_nav = portfolio_nav_from_stock_navs(stock_navs, w, stocks)

    if len(port_nav) == 0:
        return np.nan, port_nav, set(exit_days.keys())

    port_nav = port_nav * (1.0 - cost_fraction)
    period_ret = float(port_nav[-1] - 1.0)
    return period_ret, port_nav, set(exit_days.keys())


# ── Buy-hold sanity check (2-stock example) ──────────────────────────────────
# Stock A: open=10, close sequence 10→11 (+10% day0), then 11→11.55 (+5% day1)
# Stock B: open=20, close sequence 20→21 (+5% day0), then 21→21 (+0% day1)
# Equal weight → day0 port ret = 0.5*0.10 + 0.5*0.05 = 7.5%
# After day1: A_nav=1.155, B_nav=1.05 → port_nav = 0.5*1.155 + 0.5*1.05 = 1.1025
# With zero cost, period_return = 10.25%  (NOT mean of daily mean returns compounded)
