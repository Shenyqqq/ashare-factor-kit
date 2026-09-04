"""B 股默认剔除：filter_universe + tradable mask。"""
from __future__ import annotations

import pandas as pd

from data.download import filter_universe, is_b_share_code, is_excluded_universe_code
from research.ic.universe import build_ic_tradability_mask


def test_is_b_share_code_formats():
    assert is_b_share_code("200001")
    assert is_b_share_code("200001.SZ")
    assert is_b_share_code("900901")
    assert is_b_share_code("900901.SH")
    assert not is_b_share_code("600000")
    assert not is_b_share_code("000001")
    assert not is_b_share_code("002001")
    assert not is_b_share_code("300750")
    assert not is_b_share_code("301001")
    assert not is_b_share_code("601398")
    assert not is_b_share_code("603000")
    assert not is_b_share_code("605001")
    assert not is_b_share_code("688001")
    assert not is_b_share_code("920001")
    assert not is_b_share_code("430001")
    assert is_excluded_universe_code("830001")
    assert not is_excluded_universe_code("920001")


def test_filter_universe_drops_b_keeps_a():
    sl = pd.DataFrame({
        "code": ["200001", "900901", "600000", "000001", "300750", "688001"],
        "name": ["深B", "沪B", "浦发", "平安", "宁德", "科创"],
    })
    out = filter_universe(sl)
    codes = set(out["code"].astype(str).str.zfill(6))
    assert "200001" not in codes
    assert "900901" not in codes
    assert "600000" in codes
    assert "000001" in codes
    assert "300750" in codes
    assert "688001" in codes


def test_filter_universe_keeps_bj_92():
    sl = pd.DataFrame({
        "code": ["920001", "830001", "200001", "600000"],
        "name": ["北交", "新三板8", "深B", "浦发"],
    })
    out = filter_universe(sl)
    codes = set(out["code"].astype(str).str.zfill(6))
    assert "920001" in codes
    assert "830001" not in codes
    assert "200001" not in codes
    assert "600000" in codes


def test_tradable_mask_drops_b_keeps_a():
    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    cols = ["200001", "900901", "600000"]
    prices = pd.DataFrame(10.0, index=dates, columns=cols)
    volume = pd.DataFrame(1000.0, index=dates, columns=cols)
    tradable = build_ic_tradability_mask(prices, volume=volume)
    assert not tradable["200001"].any()
    assert not tradable["900901"].any()
    assert tradable["600000"].all()
