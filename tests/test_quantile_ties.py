"""
tests/test_quantile_ties.py — score ties must not skip a rebalance day.

LGBM ranker leaf outputs often produce many identical scores; the old
``pd.qcut(..., duplicates='drop')`` path raised when labels length no longer
matched the collapsed bins, and ``group_map is None`` dropped Q1–Q5 / TopN /
EW / HS300 for that period.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.execution import TradeRules
from backtest.portfolio import assign_quantile_groups, select_top_n
from backtest.quantile import run_quantile_backtest


def test_assign_quantile_groups_many_ties_still_five_bins():
    """All-equal scores must still map into Q1–Q5 when n is large enough."""
    n = 50
    codes = [f"{i:06d}" for i in range(n)]
    scores = pd.Series(1.0, index=codes)  # complete tie
    q_labels = [f"Q{i}" for i in range(1, 6)]
    group_map = assign_quantile_groups(scores, 5, q_labels, min_stocks=3)
    assert group_map is not None
    assert set(group_map) == set(q_labels)
    assert all(len(group_map[q]) > 0 for q in q_labels)
    assert sum(len(v) for v in group_map.values()) == n
    # Deterministic: lower codes land in lower quantiles under (score, code) order
    assert group_map["Q1"][0] == "000000"
    assert group_map["Q5"][-1] == "000049"


def test_assign_quantile_groups_insufficient_returns_none():
    scores = pd.Series([1.0, 1.0, 1.0], index=["a", "b", "c"])
    assert assign_quantile_groups(scores, 5, [f"Q{i}" for i in range(1, 6)], min_stocks=3) is None


def test_select_top_n_tie_stable_by_code():
    """Equal scores → Top-N prefers lower code (deterministic secondary key)."""
    scores = pd.Series(
        {"000003": 0.5, "000001": 0.5, "000002": 0.5, "000010": 0.9},
    )
    dates = pd.bdate_range("2023-01-02", periods=3)
    close = pd.DataFrame(100.0, index=dates, columns=scores.index)
    rules = TradeRules()
    selected = select_top_n(scores, n=3, rules=rules, execution_date=dates[1], close=close)
    assert selected[0] == "000010"  # unique high score
    assert selected[1:] == ["000001", "000002"]  # ties broken by code


def test_backtest_does_not_skip_rebalance_on_tied_scores():
    """Full engine: all-tied scores on a signal day must still produce Q/TopN NAV."""
    daily = pd.bdate_range("2023-01-02", periods=40)
    fridays = daily[daily.weekday == 4][:4]
    assert len(fridays) >= 3

    n = 20
    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.01, size=(len(daily), n))
    prices = pd.DataFrame(
        100 * np.cumprod(1 + rets, axis=0),
        index=daily,
        columns=[f"{i:06d}" for i in range(n)],
    )
    open_ = prices.shift(1).fillna(prices.iloc[0]) * 0.999

    # Every name gets the exact same score on every signal day
    scores = pd.DataFrame(1.0, index=fridays, columns=prices.columns)

    result = run_quantile_backtest(
        prices,
        scores,
        rebalance_freq="W-FRI",
        open_prices=open_,
        cost_bps=0.0,
        min_stocks=3,
        top_n=5,
        n_quantiles=5,
    )
    # Old bug: group_map is None → continue → empty period_meta / no NAV rows
    assert result.nav is not None and not result.nav.empty
    assert len(result.nav) >= 2
    for q in [f"Q{i}" for i in range(1, 6)]:
        assert q in result.nav.columns
        assert result.nav[q].notna().any(), f"{q} should have returns under ties"
    assert "Top5" in result.nav.columns
    assert result.nav["Top5"].notna().any()