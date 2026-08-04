"""
tests/test_backtest_calendar.py — 持有窗/执行日历与 scores 解耦

回归：稀疏周五 scores + 日频 prices → hold_dates 长度 ≈ 5（不是 1）；
指数区间收益接近真实 buy&hold 量级。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.execution import hold_dates_between, resolve_execution_date
from backtest.quantile import run_quantile_backtest
from backtest.benchmark import index_period_return


def _daily_calendar(start="2023-01-02", periods=60):
    return pd.bdate_range(start, periods=periods)


def test_hold_dates_use_full_daily_calendar_not_scores():
    """周频 scores 不得把持有窗裁成 1 日。"""
    daily = _daily_calendar()
    # 仅周五有信号
    fridays = daily[daily.weekday == 4]
    assert len(fridays) >= 3

    n = 20
    rng = np.random.default_rng(0)
    # 温和上涨价格
    rets = rng.normal(0.001, 0.01, size=(len(daily), n))
    prices = pd.DataFrame(
        100 * np.cumprod(1 + rets, axis=0),
        index=daily,
        columns=[f"{i:06d}" for i in range(n)],
    )
    open_ = prices.shift(1).fillna(prices.iloc[0]) * 0.999

    scores = pd.DataFrame(
        rng.normal(0, 1, size=(len(fridays), n)),
        index=fridays,
        columns=prices.columns,
    )

    # 直接测 hold_dates 辅助函数：周五→下周五之间日频 ≈ 5
    sig, nxt = fridays[0], fridays[1]
    exec_d = resolve_execution_date(sig, daily, use_open=True)
    assert exec_d is not None
    hdates = hold_dates_between(daily, exec_d, nxt)
    assert 4 <= len(hdates) <= 6, f"expected ~5 hold days, got {len(hdates)}"

    # 若误用 scores 日历，持有窗会塌成 1
    collapsed = hold_dates_between(fridays, exec_d if exec_d in fridays else nxt, nxt)
    assert len(collapsed) <= 2

    result = run_quantile_backtest(
        prices, scores, n_quantiles=5, rebalance_freq="W-FRI",
        open_prices=open_, cost_bps=0.0, min_stocks=3, top_n=5,
    )
    assert result.nav is not None and len(result.nav) >= 2


def test_index_period_return_near_buy_and_hold():
    """指数区间收益应接近同窗 buy&hold，而非接近 0。"""
    daily = _daily_calendar(periods=30)
    # 指数平稳上涨约 +1%/日 复利 → 5 日约 +5%
    idx = pd.Series(100 * (1.01 ** np.arange(len(daily))), index=daily)
    fridays = daily[daily.weekday == 4]
    sig, nxt = fridays[0], fridays[1]
    exec_d = resolve_execution_date(sig, daily, use_open=True)
    hdates = hold_dates_between(daily, exec_d, nxt)
    assert len(hdates) >= 4

    period_ret = index_period_return(idx, hdates, exec_d)
    # buy&hold over hold window
    bh = float(idx.loc[hdates[-1]] / idx.loc[hdates[0]] - 1.0)
    # 开盘执行模型与 close-to-close 略有差异，但量级应同向且接近
    assert period_ret > 0.02, f"period_ret={period_ret} too small (collapsed calendar?)"
    assert abs(period_ret - bh) < 0.03, f"period_ret={period_ret}, bh={bh}"


def test_sparse_scores_index_cumret_not_collapsed():
    """端到端：稀疏 scores + 日频指数 → 沪深300 累计接近真实涨幅量级。"""
    daily = _daily_calendar(start="2023-01-02", periods=80)
    fridays = daily[daily.weekday == 4]
    n = 15
    rng = np.random.default_rng(1)
    rets = rng.normal(0.0005, 0.01, size=(len(daily), n))
    prices = pd.DataFrame(
        100 * np.cumprod(1 + rets, axis=0),
        index=daily,
        columns=[f"{i:06d}" for i in range(n)],
    )
    open_ = prices.copy()
    scores = pd.DataFrame(
        rng.normal(0, 1, size=(len(fridays), n)),
        index=fridays,
        columns=prices.columns,
    )
    # 指数 +20% over full sample
    idx = pd.Series(
        np.linspace(100, 120, len(daily)), index=daily, name="沪深300",
    )
    result = run_quantile_backtest(
        prices, scores, n_quantiles=5, rebalance_freq="W-FRI",
        open_prices=open_, cost_bps=0.0, min_stocks=3, top_n=5,
        indices={"沪深300": idx},
    )
    assert "沪深300" in result.nav.columns
    cum = float(result.nav["沪深300"].iloc[-1] - 1.0)
    # 全样本约 +20%；若持有窗塌成 1 日/周，累计会远小于真实
    assert cum > 0.08, f"沪深300 cum={cum:.3%} too low (calendar bug?)"
    assert cum < 0.35, f"沪深300 cum={cum:.3%} unexpectedly high"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
