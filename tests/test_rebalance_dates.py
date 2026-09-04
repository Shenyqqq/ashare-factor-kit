"""Unit tests for utils.rebalance_dates — ME / W-FRI last-trading-day semantics."""
from __future__ import annotations

import pandas as pd
import pytest

from utils.rebalance_dates import get_rebalance_dates, horizon_to_rebalance_freq


def _bday_calendar(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end)


def test_horizon_to_rebalance_freq():
    assert horizon_to_rebalance_freq(3) == "3D"
    assert horizon_to_rebalance_freq(5) == "W-FRI"
    assert horizon_to_rebalance_freq(10) == "2W-FRI"
    assert horizon_to_rebalance_freq(20) == "ME"
    assert horizon_to_rebalance_freq(60) == "ME"


def test_me_keeps_month_when_calendar_month_end_is_weekend():
    """2018-03-31 was Saturday — must still yield a March rebalance on last bday."""
    dates = _bday_calendar("2018-01-01", "2018-06-30")
    assert pd.Timestamp("2018-03-31") not in dates  # Saturday

    rb = get_rebalance_dates(dates, "ME")
    march = [d for d in rb if d.year == 2018 and d.month == 3]
    assert len(march) == 1
    assert march[0] == pd.Timestamp("2018-03-30")
    assert march[0] in dates


def test_me_one_per_calendar_month_2018_2026():
    """Every calendar month with trading days gets exactly one ME rebalance."""
    dates = _bday_calendar("2018-01-01", "2026-12-31")
    rb = get_rebalance_dates(dates, "ME")
    months = pd.DatetimeIndex(rb).to_period("M")
    expected = pd.period_range("2018-01", "2026-12", freq="M")
    assert len(rb) == len(expected)
    assert list(months) == list(expected)
    # All rebalance dates must be actual trading days
    assert set(rb).issubset(set(dates))


def test_me_old_intersection_bug_dropped_months():
    """Regression: intersection with calendar ME labels under-counts months."""
    dates = _bday_calendar("2018-01-01", "2026-12-31")
    buggy = (
        pd.Series(1, index=dates)
        .resample("ME")
        .last()
        .index
        .intersection(dates)
    )
    fixed = get_rebalance_dates(dates, "ME")
    assert len(buggy) < len(fixed)
    assert len(fixed) == 108  # 9 years × 12 months
    # March 2018 specifically dropped by buggy path
    assert not any(d.year == 2018 and d.month == 3 for d in buggy)
    assert any(d.year == 2018 and d.month == 3 for d in fixed)


def test_w_fri_uses_last_trading_day_not_friday_label():
    """If Friday is missing (holiday), keep last trading day in that week."""
    # Full week Mon–Thu only (Friday holiday)
    week = pd.DatetimeIndex([
        "2023-01-02",  # Mon
        "2023-01-03",
        "2023-01-04",
        "2023-01-05",  # Thu — Friday 2023-01-06 missing
    ])
    rb = get_rebalance_dates(week, "W-FRI")
    assert len(rb) == 1
    assert rb[0] == pd.Timestamp("2023-01-05")
    assert pd.Timestamp("2023-01-06") not in rb


def test_w_fri_regular_friday():
    dates = _bday_calendar("2023-01-02", "2023-01-27")
    rb = get_rebalance_dates(dates, "W-FRI")
    assert len(rb) >= 3
    assert set(rb).issubset(set(dates))
    # When Friday is a trading day, rebalance should be that Friday
    fridays = dates[dates.weekday == 4]
    for fri in fridays:
        assert fri in rb


def test_2w_fri_returns_trading_days():
    dates = _bday_calendar("2023-01-02", "2023-06-30")
    rb = get_rebalance_dates(dates, "2W-FRI")
    assert len(rb) >= 10
    assert set(rb).issubset(set(dates))


def test_empty_dates():
    rb = get_rebalance_dates(pd.DatetimeIndex([]), "ME")
    assert len(rb) == 0
