"""
Synthetic unit tests for the two backtest engine bug fixes.

Bug 1: 停牌复牌跳空收益丢失 — a 3-day suspension followed by a +50% gap
        must show up in the period return (was silently 0 before the fix).

Bug 2: 一字跌停卡仓扛一个完整调仓期 — a stuck (limit-down) stock must be
        retried daily for sale and its NAV frozen at the first sellable day,
        instead of riding close/close for the whole hold window.

Run:
    python -m pytest backtest/test_return_engine.py -v
    # or directly:
    python backtest/test_return_engine.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.execution import BacktestConfig, TradeRules
from backtest.return_engine import (
    period_daily_returns,
    simulate_period,
    stock_nav_paths,
    _stuck_exit_days,
    _suspension_matrix,
)


def _make_rules(masks: dict | None = None, volume: pd.DataFrame | None = None) -> TradeRules:
    cfg = BacktestConfig(use_open_execution=True, strict_limit_mode=True)
    return TradeRules(masks=masks, volume=volume, config=cfg)


# ── Bug 1: suspension gap recovery ───────────────────────────────────────────

def test_suspension_gap_recovery() -> None:
    """Halt 3 days → resume at +50%; period return must be ≈ +50%, not 0."""
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    # Stock A: 10 → [susp, susp, susp] → 15 (gap +50%)
    close = pd.DataFrame(
        {
            "A": [10.0, np.nan, np.nan, np.nan, 15.0],
            "B": [20.0, 20.0, 20.0, 20.0, 20.0],  # control: flat
        },
        index=dates,
    )
    open_ = pd.DataFrame({"A": [10.0, np.nan, np.nan, np.nan, 15.0],
                          "B": [20.0, 20.0, 20.0, 20.0, 20.0]}, index=dates)
    rules = _make_rules()
    hold = dates
    ret, nav, sold_mid = simulate_period(
        close, hold, ["A", "B"], dates[0], open_, rules, 0.0,
    )

    # A_nav should end at 1.5 (gap captured); B_nav stays 1.0.
    daily_rets = period_daily_returns(close, hold, ["A", "B"], dates[0], open_, rules)
    navs = stock_nav_paths(daily_rets)
    a_nav_final = navs[-1, 0]
    b_nav_final = navs[-1, 1]

    assert abs(a_nav_final - 1.5) < 1e-6, (
        f"A final NAV should be 1.5 (+50% gap), got {a_nav_final}"
    )
    assert abs(b_nav_final - 1.0) < 1e-6, f"B should stay flat at 1.0, got {b_nav_final}"

    # Suspended days (1,2,3) must still be NAV-flat for A: rets == 0 there.
    for t in (1, 2, 3):
        assert daily_rets[t, 0] == 0.0, (
            f"Suspended day {t} should have rets=0 (NAV flat), got {daily_rets[t, 0]}"
        )

    # Resumption day (t=4) captures the full gap: 15/10 - 1 = 0.5
    assert abs(daily_rets[4, 0] - 0.5) < 1e-6, (
        f"Resumption-day rets should be 0.5 (gap), got {daily_rets[4, 0]}"
    )

    # Portfolio (equal weight): 0.5*1.5 + 0.5*1.0 = 1.25 → period_ret = 0.25
    assert abs(ret - 0.25) < 1e-6, f"period_return should be 0.25, got {ret}"
    assert sold_mid == set(), "no stuck stocks in this test"

    print(f"Bug 1 OK: A_nav_final={a_nav_final:.6f}, period_ret={ret:.6%}")


def test_suspension_gap_no_open_prices() -> None:
    """Same gap scenario but without open_prices (close/close exec fallback)."""
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    close = pd.DataFrame(
        {"A": [10.0, np.nan, np.nan, np.nan, 15.0]},
        index=dates,
    )
    rules = TradeRules(config=BacktestConfig(use_open_execution=False))
    hold = dates
    # execution at dates[0]; prev-day close (day -1) doesn't exist → exec return 0
    daily_rets = period_daily_returns(close, hold, ["A"], dates[0], None, rules)
    navs = stock_nav_paths(daily_rets)
    assert abs(navs[-1, 0] - 1.5) < 1e-6, (
        f"A final NAV should be 1.5 via ffill prev, got {navs[-1, 0]}"
    )
    print(f"Bug 1 (no-open) OK: A_nav_final={navs[-1, 0]:.6f}")


# ── Bug 2: stuck limit-down retried daily, NAV frozen at exit ────────────────

def test_stuck_limit_down_daily_retry() -> None:
    """One-word limit down on exec day + day 1; sellable from day 2 onward.

    Stuck stock A must exit at close[day2]; its NAV must stay frozen from
    day3 onward (no further close/close P&L).
    """
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    close = pd.DataFrame(
        {
            # A: 10 → 9 (-10%) → 8 (-11.1%) → 12 (+50%) → 13 (+8.3%)
            # limit_down_open on days 0 and 1 (can't sell); sellable from day 2
            "A": [10.0, 9.0, 8.0, 12.0, 13.0],
            # B: control, always liquid, steadily up
            "B": [20.0, 21.0, 22.0, 23.0, 24.0],
        },
        index=dates,
    )
    open_ = pd.DataFrame({"A": [10.0, 9.0, 8.0, 12.0, 13.0],
                          "B": [20.0, 21.0, 22.0, 23.0, 24.0]}, index=dates)
    ld_open = pd.DataFrame(
        {
            "A": [True, True, False, False, False],  # limit_down_open on exec + day1
            "B": [False, False, False, False, False],
        },
        index=dates,
    )
    masks = {"limit_down_open": ld_open}
    rules = _make_rules(masks=masks)

    hold = dates
    stuck = {"A"}
    exit_days = _stuck_exit_days(close, hold, ["A", "B"], dates[0], rules, stuck)
    assert exit_days == {"A": 2}, (
        f"A should become sellable on day 2, got exit_days={exit_days}"
    )

    ret, nav, sold_mid = simulate_period(
        close, hold, ["A", "B"], dates[0], open_, rules, 0.0,
        stuck_stocks=stuck,
    )
    assert sold_mid == {"A"}, f"A should be reported as sold mid-window, got {sold_mid}"

    # Inspect the per-stock NAV path: A must freeze after day 2.
    daily_rets = period_daily_returns(
        close, hold, ["A", "B"], dates[0], open_, rules,
        stuck_exit_days=exit_days,
    )
    navs = stock_nav_paths(daily_rets)
    a_nav = navs[:, 0]
    # A_nav: day0=1.0 (open=close), day1=0.9, day2=0.9*8/9=0.8, day3=0.8, day4=0.8
    assert abs(a_nav[2] - 0.8) < 1e-6, f"A NAV at exit day 2 should be 0.8, got {a_nav[2]}"
    assert abs(a_nav[3] - 0.8) < 1e-6, (
        f"A NAV must freeze after exit: day3 should be 0.8, got {a_nav[3]}"
    )
    assert abs(a_nav[4] - 0.8) < 1e-6, (
        f"A NAV must freeze after exit: day4 should be 0.8, got {a_nav[4]}"
    )
    # Stuck-stock returns after exit day must be exactly 0
    assert daily_rets[3, 0] == 0.0 and daily_rets[4, 0] == 0.0, (
        "A rets after exit day must be 0 (cash flat)"
    )

    # Sanity: without the fix, A would ride to 13/10 = 1.3; with the fix it
    # freezes at 0.8 — clearly different outcomes.
    assert a_nav[4] < 1.0, (
        "Frozen-at-exit NAV (0.8) must differ from the no-fix ride (1.3)"
    )

    print(f"Bug 2 OK: A_nav={a_nav.tolist()}, period_ret={ret:.6%}")


def test_stuck_never_sellable_stays_full_window() -> None:
    """If a stuck stock is never sellable in the window, fall back to full hold."""
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    close = pd.DataFrame(
        {"A": [10.0, 9.0, 8.0, 7.0], "B": [20.0, 21.0, 22.0, 23.0]}, index=dates,
    )
    open_ = pd.DataFrame({"A": [10.0, 9.0, 8.0, 7.0],
                          "B": [20.0, 21.0, 22.0, 23.0]}, index=dates)
    ld_open = pd.DataFrame(
        {"A": [True, True, True, True], "B": [False, False, False, False]},
        index=dates,
    )
    masks = {"limit_down_open": ld_open}
    rules = _make_rules(masks=masks)

    exit_days = _stuck_exit_days(close, dates, ["A", "B"], dates[0], rules, {"A"})
    assert exit_days == {}, "A never sellable → no exit day"

    ret, nav, sold_mid = simulate_period(
        close, dates, ["A", "B"], dates[0], open_, rules, 0.0,
        stuck_stocks={"A"},
    )
    assert sold_mid == set(), "no mid-window sale when never sellable"
    # A rides full window: 7/10 = 0.7
    daily_rets = period_daily_returns(close, dates, ["A", "B"], dates[0], open_, rules)
    a_nav = stock_nav_paths(daily_rets)[:, 0]
    assert abs(a_nav[-1] - 0.7) < 1e-6, (
        f"Never-sellable A should ride to 0.7, got {a_nav[-1]}"
    )
    print(f"Bug 2 (never sellable) OK: A_nav_final={a_nav[-1]:.6f}")


def test_suspension_matrix_matches_scalar_is_suspended() -> None:
    """Vectorized `_suspension_matrix` ≡ nested `rules.is_suspended`."""
    dates = pd.date_range("2024-01-02", periods=6, freq="B")
    close = pd.DataFrame(
        {
            "A": [10.0, np.nan, 0.0, 11.0, 12.0, 13.0],
            "B": [20.0, 21.0, 22.0, np.nan, 24.0, 25.0],
            "C": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        },
        index=dates,
    )
    volume = pd.DataFrame(
        {
            "A": [100.0, 0.0, 50.0, 50.0, np.nan, 60.0],
            "B": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "C": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        },
        index=dates,
    )
    rules = _make_rules(volume=volume)
    stocks = ["A", "B", "C"]
    hold = dates
    vec = _suspension_matrix(hold, stocks, close, rules)
    ref = np.zeros((len(hold), len(stocks)), dtype=bool)
    for j, s in enumerate(stocks):
        for i, dt in enumerate(hold):
            ref[i, j] = rules.is_suspended(s, dt, close)
    assert np.array_equal(vec, ref), f"mismatch\nvec=\n{vec}\nref=\n{ref}"
    print("suspension_matrix vectorization OK")


def test_buyable_mask_matches_can_buy() -> None:
    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    close = pd.DataFrame(
        {"A": [10.0, 11.0, 12.0], "B": [20.0, 21.0, 22.0], "C": [np.nan, 1.0, 1.0]},
        index=dates,
    )
    lu = pd.DataFrame(
        {"A": [True, False, False], "B": [False, False, False], "C": [False, False, False]},
        index=dates,
    )
    rules = _make_rules(masks={"limit_up_open": lu})
    stocks = ["A", "B", "C"]
    dt = dates[0]
    mask = rules.buyable_mask(stocks, dt, close)
    ref = np.array([rules.can_buy(s, dt, close) for s in stocks], dtype=bool)
    assert np.array_equal(mask, ref), f"buyable mismatch mask={mask} ref={ref}"
    print("buyable_mask OK")


if __name__ == "__main__":
    test_suspension_gap_recovery()
    test_suspension_gap_no_open_prices()
    test_stuck_limit_down_daily_retry()
    test_stuck_never_sellable_stays_full_window()
    test_suspension_matrix_matches_scalar_is_suspended()
    test_buyable_mask_matches_can_buy()
    print("All return_engine tests passed.")
