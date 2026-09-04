"""Data-architecture audit regression tests (P0/P1/P2)."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_codes_needing_refresh_skips_only_fresh(tmp_path):
    from data.download_shares import _codes_needing_refresh

    today = pd.Timestamp.now().normalize()
    existing = pd.DataFrame({
        "code": ["600000", "600000", "600519", "000001"],
        "announce_date": [
            today - pd.Timedelta(days=5),
            today - pd.Timedelta(days=100),
            today - pd.Timedelta(days=60),  # stale if refresh=30
            today - pd.Timedelta(days=10),
        ],
        "total_shares": [1e10, 1e10, 1e9, 1e10],
        "circ_shares": [1e10, 1e10, 1e9, 1e10],
    })
    codes = ["600000", "600519", "000001", "300750"]  # 300750 missing
    need, keep, kept = _codes_needing_refresh(
        codes, existing, refresh_stale_days=30, force_refresh=False,
    )
    assert "600000" in keep and "000001" in keep
    assert "600519" in need  # stale
    assert "300750" in need  # missing
    assert set(kept["code"].unique()) == {"600000", "000001"}


def test_codes_needing_refresh_force():
    from data.download_shares import _codes_needing_refresh

    existing = pd.DataFrame({
        "code": ["600000"],
        "announce_date": [pd.Timestamp.now()],
        "total_shares": [1.0],
        "circ_shares": [1.0],
    })
    need, keep, kept = _codes_needing_refresh(
        ["600000"], existing, refresh_stale_days=30, force_refresh=True,
    )
    assert need == ["600000"]
    assert keep == set()
    assert kept.empty


def test_pit_align_uses_ann_date_when_present():
    from utils import pit_align

    pit_align._statutory_warned = False
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    fin = pd.DataFrame({
        "trade_date": [pd.Timestamp("2024-03-31")],
        "code": ["600000"],
        "roe": [10.0],
        "ann_date": [pd.Timestamp("2024-04-10")],
    })
    out = pit_align.pit_pivot_ffill(fin, dates, value_cols=["roe"])
    # available from ann_date 2024-04-10, not statutory 03-31+30=04-30
    assert pd.isna(out.loc["2024-04-09", "600000"])
    assert out.loc["2024-04-10", "600000"] == 10.0


def test_pit_align_statutory_when_no_ann_date():
    from utils import pit_align

    pit_align._statutory_warned = False
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    fin = pd.DataFrame({
        "trade_date": [pd.Timestamp("2024-03-31")],
        "code": ["600000"],
        "roe": [10.0],
    })
    out = pit_align.pit_pivot_ffill(fin, dates, value_cols=["roe"])
    # statutory: 03-31 + 30 = 04-30
    assert pd.isna(out.loc["2024-04-29", "600000"])
    assert out.loc["2024-04-30", "600000"] == 10.0


def test_pit_disclosure_window_constants():
    from utils.pit_align import _DISCLOSURE_WINDOWS, disclosure_window

    assert _DISCLOSURE_WINDOWS == {3: 30, 6: 60, 9: 30, 12: 90}
    assert disclosure_window(pd.Timestamp("2024-03-31")) == 30
    assert disclosure_window(pd.Timestamp("2024-06-30")) == 60
    assert disclosure_window(pd.Timestamp("2024-09-30")) == 30
    assert disclosure_window(pd.Timestamp("2024-12-31")) == 90


def test_require_industry_panel_strict(tmp_path, monkeypatch):
    from research.ic import load_data as ld
    from data.industry import download_industry as di

    missing = tmp_path / "industry_map_panel.parquet"
    monkeypatch.setattr(di, "PANEL_PATH", missing)
    with pytest.raises(FileNotFoundError):
        ld.load_industry_panel(required=True)
    assert ld.load_industry_panel(required=False) is None


def test_st_fallback_source_label_and_list_date():
    from data.download_st_history import (
        SOURCE_SH_BJ_FALLBACK,
        _build_fallback_st_periods,
    )

    current = pd.DataFrame({
        "code": ["600000", "688001"],
        "name": ["ST浦发", "*ST科创"],
        "st_type": ["ST", "*ST"],
    })
    list_dates = {"600000": pd.Timestamp("2020-06-01")}
    fb = _build_fallback_st_periods(
        current,
        covered_codes=set(),
        start=pd.Timestamp("2018-01-01"),
        list_dates=list_dates,
    )
    assert set(fb["source"].unique()) == {SOURCE_SH_BJ_FALLBACK}
    row = fb.set_index("code").loc["600000"]
    assert row["start_date"] == pd.Timestamp("2020-06-01")  # tightened by list_date
    row2 = fb.set_index("code").loc["688001"]
    assert row2["start_date"] == pd.Timestamp("2018-01-01")


def test_download_market_cap_deprecated():
    import data.download_market_cap as dmc

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Don't actually download — just ensure warn path is reachable via message constant
        assert "废弃" in dmc._DEPRECATION_MSG or "deprecated" in dmc._DEPRECATION_MSG.lower()
        warnings.warn(dmc._DEPRECATION_MSG, DeprecationWarning)
        assert any(issubclass(x.category, DeprecationWarning) for x in w)


def test_volume_mul_is_100():
    from data.compute_market_cap import VOLUME_MUL
    assert VOLUME_MUL == 100


def test_report_raw_hfq_coverage_detects_gap(tmp_path):
    from data.download import report_raw_hfq_coverage

    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    hfq = pd.DataFrame({"600000": np.linspace(10, 11, len(idx))}, index=idx)
    raw = hfq.copy()
    raw.iloc[-3:] = np.nan  # raw lags
    hp = tmp_path / "close_hfq.parquet"
    rp = tmp_path / "prices_raw.parquet"
    hfq.to_parquet(hp)
    raw.to_parquet(rp)
    stats = report_raw_hfq_coverage(hfq_path=hp, raw_path=rp, max_lag_days=0)
    assert stats["stocks_raw_behind_hfq"] >= 1
    assert stats["cells_hfq_ok_raw_miss"] >= 1


def test_northbound_stop_constant():
    from data.download_northbound import NORTHBOUND_DISCLOSURE_STOP
    assert NORTHBOUND_DISCLOSURE_STOP >= pd.Timestamp("2024-08-01")
