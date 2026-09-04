"""因子面板缓存：industry_map DataFrame vs Series 指纹必须一致。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factors.factor_cache import (
    _industry_map_signature,
    _normalize_industry_id,
    _save_panel,
    build_input_signature,
    probe_factor_cache,
)


CODES = [f"{i:06d}" for i in range(1, 6)]


def _industry_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sw_l1": [f"L1_{i}" for i in range(5)],
            "sw_l2": [f"L2_{i}" for i in range(5)],
            "sw_l3": [f"L3_{i}" for i in range(5)],
        },
        index=CODES,
    )


def test_industry_df_and_series_same_signature():
    """IC 传 [N,3] DataFrame、ML 传 sw_l2 Series → 指纹相同。"""
    df = _industry_df()
    series = df["sw_l2"]
    assert list(df.shape) == [5, 3]
    assert list(series.shape) == [5]

    sig_df = _industry_map_signature(df)
    sig_s = _industry_map_signature(series)
    assert sig_df == sig_s
    assert sig_df["shape"] == [5]
    assert sig_df["index_first"] == "000001"
    assert sig_df["index_last"] == "000005"

    # 归一化后同为 sw_l2
    assert _normalize_industry_id(df).equals(series)


def test_build_input_signature_ic_ml_parity(tmp_path, monkeypatch):
    """IC kwargs(DataFrame) 与 ML kwargs(Series) 写入后互为 HIT。"""
    monkeypatch.setattr(
        "factors.factor_cache.FACTOR_CACHE_DIR", tmp_path / "factor_panels",
    )
    (tmp_path / "factor_panels").mkdir(parents=True)

    dates = pd.date_range("2024-01-02", periods=8, freq="B")
    prices = pd.DataFrame(
        np.random.randn(len(dates), len(CODES)),
        index=dates,
        columns=CODES,
    )
    ind_df = _industry_df()
    base = dict(
        prices=prices,
        financial=None,
        prices_raw=None,
        volume=None,
        amount=None,
        open_=None,
        high=None,
        low=None,
        clean_ret=prices.pct_change(),
        masks=None,
        market_prices=None,
        margin=None,
        moneyflow=None,
        northbound=None,
        institution=None,
        circ_mv=None,
        total_mv=None,
        walk_forward_hmm=False,
        include_regime=False,
    )
    # IC 路径：整表 DataFrame
    sig_ic = build_input_signature({**base, "industry_map": ind_df})
    # ML / run.py 路径：sw_l2 Series
    sig_ml = build_input_signature({**base, "industry_map": ind_df["sw_l2"]})
    assert sig_ic["industry_map"] == sig_ml["industry_map"]
    assert sig_ic == sig_ml

    name = "动量_20d"
    _save_panel(name, prices.astype(np.float32), sig_ic)
    hits, misses = probe_factor_cache([name], sig_ml)
    assert hits == [name]
    assert misses == []
