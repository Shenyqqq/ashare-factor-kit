"""
tests/test_size_alpha.py — 市值相关 alpha 单测

覆盖：
- 对数市值 / 市值分位：小市值高分
- 市值风格对齐：小盘强 + 小市值 → 高分
- get_factor_names / registry 可枚举与计算
- feature_neutralize 豁免 SIZE_ALPHA_FACTOR_NAMES
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factors.factor_size_alpha import (
    SIZE_ALPHA_FACTOR_NAMES,
    factor_log_mcap,
    factor_mcap_percentile,
    factor_size_style_align,
    get_size_alpha_factors,
)
from factors.factor import get_factor_names


DATES = pd.date_range("2024-01-02", periods=80, freq="B")
CODES = ["S1", "S2", "L1", "L2", "M1"]


def _synth():
    rng = np.random.default_rng(0)
    prices = pd.DataFrame(index=DATES, columns=CODES, dtype=float)
    for i, c in enumerate(CODES):
        rets = rng.normal(0.0005, 0.01, len(DATES))
        prices[c] = 10.0 * np.cumprod(1.0 + rets)
    clean_ret = prices.pct_change()
    # 小票 S* 市值小，大票 L* 市值大
    circ_mv = pd.DataFrame(
        {
            "S1": 1e9, "S2": 1.2e9,
            "L1": 5e10, "L2": 6e10,
            "M1": 8e9,
        },
        index=DATES,
    )
    return prices, clean_ret, circ_mv


def test_size_alpha_names_canonical():
    assert SIZE_ALPHA_FACTOR_NAMES == {
        "对数市值", "市值分位", "市值风格对齐_20d", "市值风格对齐_60d",
    }


def test_log_mcap_small_scores_higher():
    prices, _, circ_mv = _synth()
    panel = factor_log_mcap(prices, circ_mv=circ_mv)
    assert panel is not None
    last = panel.iloc[-1]
    assert last["S1"] > last["L1"]
    assert last["S2"] > last["L2"]


def test_mcap_percentile_small_scores_higher():
    prices, _, circ_mv = _synth()
    panel = factor_mcap_percentile(prices, circ_mv=circ_mv)
    assert panel is not None
    last = panel.iloc[-1]
    assert last["S1"] > last["L1"]


def test_size_style_align_sign_small_wins():
    prices, clean_ret, circ_mv = _synth()
    boost = clean_ret.copy()
    for c in ("S1", "S2"):
        boost[c] = 0.02
    for c in ("L1", "L2"):
        boost[c] = -0.02
    panel = factor_size_style_align(
        prices, circ_mv=circ_mv, clean_ret=boost, window=20,
    )
    assert panel is not None
    last = panel.iloc[-1].dropna()
    if {"S1", "L1"} <= set(last.index):
        assert last["S1"] > last["L1"]


def test_get_size_alpha_factors_subset():
    prices, clean_ret, circ_mv = _synth()
    out = get_size_alpha_factors(
        prices, circ_mv=circ_mv, clean_ret=clean_ret,
        factor_names={"对数市值", "市值风格对齐_20d"},
    )
    assert set(out) == {"对数市值", "市值风格对齐_20d"}
    for panel in out.values():
        assert panel.shape == prices.shape


def test_get_factor_names_includes_size_alpha():
    prices, _, circ_mv = _synth()
    names = get_factor_names(
        prices, financial=pd.DataFrame(),
        circ_mv=circ_mv, include_regime=False,
    )
    for n in SIZE_ALPHA_FACTOR_NAMES:
        assert n in names


def test_feature_neutralize_exempts_size_alpha(monkeypatch):
    import strategies.ml as ml

    prices, _, circ_mv = _synth()
    financial = pd.DataFrame()
    size_panel = pd.DataFrame(1.0, index=DATES, columns=CODES, dtype="float32")
    fake_alpha = {
        "动量_20d": prices.copy().astype("float32"),
        "对数市值": size_panel,
    }
    monkeypatch.setenv("FACTOR_CACHE_DISABLE", "1")

    monkeypatch.setattr(
        ml, "_load_or_compute_registry",
        lambda *a, **k: dict(fake_alpha),
    )
    monkeypatch.setattr(
        "factors.factor.cross_sectional_zscore",
        lambda panel: panel,
    )

    def fake_residualize(panel, *a, **k):
        return panel * 2.0

    monkeypatch.setattr("models.wf.labels.residualize_panel", fake_residualize)

    barra = {
        "Barra_Size": prices.copy().astype("float32"),
        "Barra_Beta": prices.copy().astype("float32"),
    }
    industry = pd.Series(
        ["银行", "银行", "地产", "地产", "制造"], index=CODES, name="sw_l2",
    )

    ds = ml.build_factor_dataset(
        prices, financial,
        circ_mv=circ_mv,
        hold_period=5,
        feature_neutralize=True,
        barra_factors=barra,
        industry_map=industry,
        rebalance_freq="W-FRI",
        apply_tradable_filter=False,
        fwd_return_winsor=False,
        use_factor_cache=False,
        include_regime=False,
    )
    assert "对数市值" in ds.feature_names
    assert np.allclose(ds.factor_panel["对数市值"].to_numpy(), 1.0)
    assert np.allclose(
        ds.factor_panel["动量_20d"].to_numpy(), prices.to_numpy() * 2.0,
    )
