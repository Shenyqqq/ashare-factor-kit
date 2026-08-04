"""
tests/test_portfolio_opt.py — 组合权重 v1（ew/score/rank/mv/invvol）

覆盖：
  - 权重和 ≈ 1、long-only、单票 cap
  - 默认 ew 路径与旧 equal_weights / simulate_period(None) bit-identical
  - quantile 默认 config 不改变 NAV（相对显式 ew）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.execution import BacktestConfig
from backtest.optimize import (
    equal_weight,
    inverse_vol_weight,
    mean_variance_lite,
    optimize_weights,
    rank_weight,
    score_weight,
)
from backtest.portfolio import equal_weights
from backtest.quantile import run_quantile_backtest
from backtest.return_engine import simulate_period
from backtest.execution import TradeRules


def _assert_weights_ok(w: dict[str, float], stocks: list[str], max_w: float | None = None):
    assert set(w.keys()) == set(stocks)
    arr = np.array([w[s] for s in stocks], dtype=np.float64)
    assert np.all(arr >= -1e-12)
    assert abs(arr.sum() - 1.0) < 1e-8
    if max_w is not None and 0 < max_w < 1 and len(stocks) * max_w >= 1.0 - 1e-9:
        assert arr.max() <= max_w + 1e-8


def test_equal_weight_matches_portfolio_helper():
    stocks = ["A", "B", "C"]
    w1 = equal_weight(stocks)
    w2 = equal_weights(stocks)
    assert w1 == w2
    _assert_weights_ok(w1, stocks)


def test_score_and_rank_sum_and_cap():
    stocks = ["A", "B", "C", "D"]
    scores = pd.Series({"A": 3.0, "B": 1.0, "C": 2.0, "D": 0.0})
    ws = score_weight(scores, stocks, max_weight=0.4)
    wr = rank_weight(scores, stocks, max_weight=0.4)
    _assert_weights_ok(ws, stocks, max_w=0.4)
    _assert_weights_ok(wr, stocks, max_w=0.4)
    # 高分应获得更大权重
    assert ws["A"] >= ws["B"] - 1e-12
    assert wr["A"] >= wr["D"] - 1e-12


def test_invvol_and_mv_synthetic():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2023-01-01", periods=80)
    stocks = [f"S{i}" for i in range(5)]
    # 不同波动：S0 低波 → invvol 权重大
    vols = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    rets = pd.DataFrame(
        rng.normal(0, 1, (len(dates), len(stocks))) * vols,
        index=dates,
        columns=stocks,
    )
    scores = pd.Series({s: float(i) for i, s in enumerate(stocks)})
    w_iv = inverse_vol_weight(stocks, returns=rets, asof=dates[-1], lookback=60)
    _assert_weights_ok(w_iv, stocks)
    assert w_iv["S0"] > w_iv["S4"]

    w_mv = mean_variance_lite(
        scores, stocks, returns=rets, asof=dates[-1], lookback=60, max_weight=0.35,
    )
    _assert_weights_ok(w_mv, stocks, max_w=0.35)

    w_rp = optimize_weights(
        "rp", stocks, scores=scores, returns=rets, asof=dates[-1], lookback=60, max_weight=0.4,
    )
    _assert_weights_ok(w_rp, stocks, max_w=0.4)


def test_simulate_period_default_weights_identical():
    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    close = pd.DataFrame(
        {"A": [10.0, 11.0, 11.55], "B": [20.0, 21.0, 21.0]},
        index=dates,
    )
    open_ = pd.DataFrame(
        {"A": [10.0, 11.0, 11.55], "B": [20.0, 21.0, 21.0]},
        index=dates,
    )
    rules = TradeRules(config=BacktestConfig(use_open_execution=True))
    ret0, nav0, _ = simulate_period(
        close, dates, ["A", "B"], dates[0], open_, rules, 0.0, weights=None,
    )
    ret1, nav1, _ = simulate_period(
        close, dates, ["A", "B"], dates[0], open_, rules, 0.0,
        weights=equal_weights(["A", "B"]),
    )
    assert ret0 == ret1
    np.testing.assert_array_equal(nav0, nav1)


def test_quantile_default_ew_matches_explicit_ew():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2023-01-01", periods=120)
    codes = [f"{i:06d}" for i in range(1, 41)]
    close = pd.DataFrame(
        10.0 * np.cumprod(1 + rng.normal(0.001, 0.02, (len(dates), len(codes))), axis=0),
        index=dates,
        columns=codes,
    )
    open_ = close.shift(1).bfill() * 0.999
    scores = pd.DataFrame(rng.normal(0, 1, close.shape), index=dates, columns=codes)

    r_default = run_quantile_backtest(
        close, scores, rebalance_freq="ME", open_prices=open_, top_n=8, min_stocks=3,
        config=BacktestConfig(),
    )
    r_ew = run_quantile_backtest(
        close, scores, rebalance_freq="ME", open_prices=open_, top_n=8, min_stocks=3,
        config=BacktestConfig(portfolio_opt="ew"),
    )
    pd.testing.assert_frame_equal(r_default.nav, r_ew.nav)

    r_score = run_quantile_backtest(
        close, scores, rebalance_freq="ME", open_prices=open_, top_n=8, min_stocks=3,
        config=BacktestConfig(portfolio_opt="score", max_weight=0.25),
        returns=close.pct_change(),
    )
    # score 路径应能跑通且 NAV 形状一致；通常与 EW 不同（允许偶然相等但不要求）
    assert r_score.nav.shape == r_default.nav.shape
    assert not r_score.nav.empty


def test_optimize_weights_dispatch_and_invalid():
    stocks = ["A", "B"]
    scores = pd.Series({"A": 1.0, "B": 2.0})
    w = optimize_weights("rank", stocks, scores=scores)
    _assert_weights_ok(w, stocks)
    with pytest.raises(ValueError):
        optimize_weights("unknown", stocks, scores=scores)
