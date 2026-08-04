"""分位小市值 mask + 因子缓存指纹与 universe 解耦。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factors.factor_cache import (
    build_input_signature,
    probe_factor_cache,
    _save_panel,
    _cache_paths,
)
from utils.universe import build_mcap_percentile_mask


DATES = pd.date_range("2024-01-02", periods=10, freq="B")
CODES = [f"{i:06d}" for i in range(1, 11)]  # 10 只


def _mv_linear() -> pd.DataFrame:
    """每日市值 = 1e8 * code_index（000001 最小 … 000010 最大）。"""
    row = np.arange(1, 11, dtype=float) * 1e8
    return pd.DataFrame(
        np.tile(row, (len(DATES), 1)),
        index=DATES,
        columns=CODES,
    )


def test_percentile_bottom_30_keeps_three():
    """q=0.30 → 最低 3/10 只（pct_rank≤0.30）。"""
    mv = _mv_linear()
    rb = DATES[[2, 5, 8]]
    mask = build_mcap_percentile_mask(
        mv, quantile=0.30, rebalance_dates=rb, trading_index=DATES,
    )
    # 调仓日后 ffill：第 2 日及之后应含 000001/002/003
    day = mask.loc[DATES[5]]
    kept = [c for c in CODES if bool(day[c])]
    assert kept == ["000001", "000002", "000003"]
    # 调仓日前无判定 → False
    assert not bool(mask.loc[DATES[0], "000001"])


def test_percentile_bottom_50():
    mv = _mv_linear()
    mask = build_mcap_percentile_mask(
        mv, quantile=0.50, rebalance_dates=None, trading_index=DATES,
    )
    kept = [c for c in CODES if bool(mask.iloc[-1][c])]
    assert kept == [f"{i:06d}" for i in range(1, 6)]


def test_median_count_shrinks_vs_full():
    mv = _mv_linear()
    mask = build_mcap_percentile_mask(mv, quantile=0.30, rebalance_dates=DATES)
    med = int(mask.sum(axis=1).median())
    assert med == 3
    assert med < mv.shape[1]


def test_fallback_total_mv():
    mv = _mv_linear()
    mask = build_mcap_percentile_mask(
        None, quantile=0.30, total_mv=mv, rebalance_dates=DATES,
    )
    assert int(mask.iloc[-1].sum()) == 3


def test_factor_cache_signature_ignores_universe_mask(tmp_path, monkeypatch):
    """换宇宙 mask 不改变输入指纹 → 同面板可 HIT，无需重算因子。"""
    monkeypatch.setattr(
        "factors.factor_cache.FACTOR_CACHE_DIR", tmp_path / "factor_panels",
    )
    (tmp_path / "factor_panels").mkdir(parents=True)

    prices = pd.DataFrame(
        np.random.randn(len(DATES), len(CODES)),
        index=DATES,
        columns=CODES,
    )
    kwargs = dict(
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
        industry_map=None,
        margin=None,
        moneyflow=None,
        northbound=None,
        institution=None,
        circ_mv=_mv_linear(),
        total_mv=None,
        walk_forward_hmm=False,
        include_regime=False,
    )
    sig = build_input_signature(kwargs)
    name = "动量_20d"
    panel = prices.astype(np.float32)
    _save_panel(name, panel, sig)

    hits, misses = probe_factor_cache([name], sig)
    assert hits == [name]
    assert misses == []

    # 模拟另一宇宙：只改 mask，不改 kwargs → 指纹不变 → 仍 HIT
    mask_a = build_mcap_percentile_mask(
        kwargs["circ_mv"], quantile=0.30, rebalance_dates=DATES,
    )
    mask_b = build_mcap_percentile_mask(
        kwargs["circ_mv"], quantile=0.50, rebalance_dates=DATES,
    )
    assert int(mask_a.iloc[-1].sum()) < int(mask_b.iloc[-1].sum())
    sig2 = build_input_signature(kwargs)
    hits2, misses2 = probe_factor_cache([name], sig2)
    assert hits2 == [name]
    assert misses2 == []
    # 确认缓存路径在 monkeypatched 目录
    pq, _ = _cache_paths(name)
    assert str(tmp_path) in str(pq)


def test_invalid_quantile():
    with pytest.raises(ValueError, match="quantile"):
        build_mcap_percentile_mask(_mv_linear(), quantile=0.0)
