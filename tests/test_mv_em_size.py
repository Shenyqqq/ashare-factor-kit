"""Size / 市值面板：东财主路径 + 自算兜底。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_mv_panels_resolve_prefers_em(tmp_path, monkeypatch):
    from data import mv_panels

    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setattr(mv_panels, "RAW_DIR", raw)
    monkeypatch.setattr(
        mv_panels,
        "PRIMARY",
        {"total_mv": raw / "total_mv.parquet", "circ_mv": raw / "circ_mv.parquet"},
    )
    monkeypatch.setattr(
        mv_panels,
        "COMPUTED",
        {
            "total_mv": raw / "total_mv_computed.parquet",
            "circ_mv": raw / "circ_mv_computed.parquet",
        },
    )

    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    em = pd.DataFrame({"600519": [1e12, 1.1e12, 1.2e12]}, index=dates)
    computed = pd.DataFrame({"600519": [9e11, 9.1e11, 9.2e11]}, index=dates)
    em.to_parquet(raw / "circ_mv.parquet")
    computed.to_parquet(raw / "circ_mv_computed.parquet")

    path = mv_panels.resolve_mv_path("circ_mv")
    assert path is not None and path.name == "circ_mv.parquet"
    loaded = mv_panels.load_mv_raw("circ_mv")
    assert loaded is not None
    assert float(loaded["600519"].iloc[-1]) == pytest.approx(1.2e12)


def test_mv_panels_fallback_to_computed(tmp_path, monkeypatch):
    from data import mv_panels

    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setattr(mv_panels, "RAW_DIR", raw)
    monkeypatch.setattr(
        mv_panels,
        "PRIMARY",
        {"total_mv": raw / "total_mv.parquet", "circ_mv": raw / "circ_mv.parquet"},
    )
    monkeypatch.setattr(
        mv_panels,
        "COMPUTED",
        {
            "total_mv": raw / "total_mv_computed.parquet",
            "circ_mv": raw / "circ_mv_computed.parquet",
        },
    )
    dates = pd.date_range("2024-01-01", periods=2, freq="B")
    computed = pd.DataFrame({"600000": [1e11, 1.1e11]}, index=dates)
    computed.to_parquet(raw / "total_mv_computed.parquet")

    path = mv_panels.resolve_mv_path("total_mv")
    assert path is not None and path.name == "total_mv_computed.parquet"


def test_barra_size_reads_em_circ_mv():
    """Barra_Size 主路径读传入的 circ_mv（模拟东财面板），不走 total_assets。"""
    from factors.barra_risk import barra_size, pick_market_cap

    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    codes = ["600519", "600000"]
    prices = pd.DataFrame(100.0, index=dates, columns=codes)
    circ_mv = pd.DataFrame(
        [[1e12, 2e11], [1.01e12, 2.1e11], [1.02e12, 2.2e11],
         [1.03e12, 2.3e11], [1.04e12, 2.4e11]],
        index=dates,
        columns=codes,
    )
    mv, src = pick_market_cap(prices, circ_mv=circ_mv, total_mv=None)
    assert src == "circ_mv"
    assert mv is not None
    size = barra_size(prices, circ_mv=circ_mv, total_mv=None, financial=None)
    assert size is not None
    # 标准化后面板非全 NaN；且大市值票截面秩应更低（Size 风格因子本身越高=越大）
    assert size.notna().any().any()
    # 原始 log 序：流通市值更大的 600519 在截面上 Size 原值更高（再 winsor/z）
    raw_log = np.log(circ_mv)
    assert raw_log.loc[dates[-1], "600519"] > raw_log.loc[dates[-1], "600000"]


def test_merge_wide_preserve_keeps_history_on_window_slice(tmp_path):
    """列完全重叠时，lookback 窗口切片也不得覆盖窗口外历史日期。"""
    from data.download_stock_value_em import _merge_wide_preserve

    path = tmp_path / "circ_mv.parquet"
    hist_idx = pd.date_range("2020-01-02", periods=5, freq="B")
    old = pd.DataFrame(
        {"600519": [1.0e12, 1.1e12, 1.2e12, 1.3e12, 1.4e12],
         "000001": [2.0e11, 2.1e11, 2.2e11, 2.3e11, 2.4e11]},
        index=hist_idx,
    )
    old.to_parquet(path)

    window = pd.DataFrame(
        {"600519": [1.41e12, 1.42e12], "000001": [2.41e11, 2.42e11]},
        index=pd.DatetimeIndex(["2020-01-09", "2020-01-10"]),
    )
    merged = _merge_wide_preserve(path, window)
    assert merged.index.min() == hist_idx.min()
    assert merged.index.max() == pd.Timestamp("2020-01-10")
    assert len(merged.index) == 7
    assert float(merged.loc[hist_idx[0], "600519"]) == pytest.approx(1.0e12)
    assert float(merged.loc[hist_idx[-1], "600519"]) == pytest.approx(1.4e12)
    assert float(merged.loc[pd.Timestamp("2020-01-10"), "600519"]) == pytest.approx(1.42e12)


def test_merge_wide_fill_missing_does_not_touch_existing(tmp_path):
    """补 92 列时沪深已有列的行数/格点必须原样保留。"""
    from data.download_stock_value_em import _merge_wide_preserve

    path = tmp_path / "circ_mv.parquet"
    hist_idx = pd.date_range("2020-01-02", periods=5, freq="B")
    old = pd.DataFrame(
        {"600519": [1.0e12, 1.1e12, 1.2e12, 1.3e12, 1.4e12],
         "000001": [2.0e11, 2.1e11, 2.2e11, 2.3e11, 2.4e11]},
        index=hist_idx,
    )
    old.to_parquet(path)

    incoming = pd.DataFrame(
        {
            "600519": [9.9e12, 9.9e12],  # 即使带沪深列也必须被跳过
            "920001": [1.1e9, 1.2e9],
        },
        index=pd.DatetimeIndex(["2020-01-03", "2020-01-06"]),
    )
    merged = _merge_wide_preserve(path, incoming, fill_missing_cols_only=True)
    assert list(old["600519"]) == list(merged["600519"].loc[hist_idx])
    assert list(old["000001"]) == list(merged["000001"].loc[hist_idx])
    assert "920001" in merged.columns
    assert float(merged.loc[pd.Timestamp("2020-01-06"), "920001"]) == pytest.approx(1.2e9)
    assert pd.isna(merged.loc[hist_idx[0], "920001"])


def test_drop_sparse_trailing_dates_keeps_body():
    """未完成批次的稀疏末日必须丢掉，历史完整日保留。"""
    from data.download_stock_value_em import _drop_sparse_trailing_dates

    idx = pd.date_range("2026-07-01", periods=30, freq="B")
    n = 100
    data = np.ones((len(idx), n))
    data[-3:, 10:] = np.nan  # 末日只剩 10 列
    data[-5:-3, 40:] = np.nan  # 再前两日 40 列
    wide = pd.DataFrame(data, index=idx, columns=[f"{i:06d}" for i in range(n)])
    out = _drop_sparse_trailing_dates(wide, min_frac=0.5)
    assert out.index.max() == idx[-6]
    assert len(out) == len(idx) - 5
    assert int(out.iloc[-1].notna().sum()) == n


def test_map_em_columns_units(monkeypatch):
    from data.download_stock_value_em import _map_em_columns

    raw = pd.DataFrame({
        "数据日期": ["2024-01-02", "2024-01-03"],
        "总市值": [1.5e12, 1.6e12],
        "流通市值": [1.4e12, 1.5e12],
        "总股本": [1.25e9, 1.25e9],
        "流通股本": [1.25e9, 1.25e9],
        "PE(TTM)": [20.0, 21.0],
        "市净率": [5.0, 5.1],
    })
    df = _map_em_columns(raw)
    assert list(df.columns)[:4] == ["date", "total_mv", "circ_mv", "total_shares"]
    assert float(df["total_mv"].iloc[0]) == pytest.approx(1.5e12)
    assert "pe_ttm" in df.columns and "pb" in df.columns
