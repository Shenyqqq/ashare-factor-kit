"""急切路径 neut 磁盘缓存：MISS→SAVE→HIT，且与直接 residualize 数值一致。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def tiny_panels(tmp_path, monkeypatch):
    from factors.factor_cache import FACTOR_CACHE_DIR
    monkeypatch.setattr(
        "factors.factor_cache.FACTOR_CACHE_DIR", tmp_path / "factor_panels",
    )
    # neut_cache imports FACTOR_CACHE_DIR lazily inside neut_cache_path;
    # also patch the symbol used after first import if already loaded.
    import research.rolling_pool.neut_cache as nc
    monkeypatch.setattr(
        "factors.factor_cache.FACTOR_CACHE_DIR", tmp_path / "factor_panels",
    )

    idx = pd.date_range("2020-01-03", periods=4, freq="W-FRI")
    cols = ["s0", "s1", "s2", "s3"]
    prices = pd.DataFrame(1.0, index=idx, columns=cols)
    rng = np.random.default_rng(0)
    factor = pd.DataFrame(
        rng.normal(size=(4, 4)), index=idx, columns=cols, dtype=np.float32,
    )
    barra = {
        "Barra_Size": pd.DataFrame(
            rng.normal(size=(4, 4)), index=idx, columns=cols, dtype=np.float32,
        ),
        "Barra_Beta": pd.DataFrame(
            rng.normal(size=(4, 4)), index=idx, columns=cols, dtype=np.float32,
        ),
    }
    ind = pd.Series(["A", "A", "B", "B"], index=cols)
    w = pd.DataFrame(1.0, index=idx, columns=cols)
    return prices, factor, barra, ind, w, tmp_path


def test_neut_cache_hit_skips_recompute(tiny_panels, monkeypatch):
    from research.rolling_pool.neut_cache import (
        barra_bundle_sig,
        neut_cache_path,
        neutralize_one_factor,
        save_neut_panel,
        try_load_neut_panel,
    )
    from research.rolling_pool.schedule_load import cs_zscore_sparse_rows

    prices, factor, barra, ind, w, tmp_path = tiny_panels
    dates = prices.index
    ctrl = barra_bundle_sig(barra, industry_map=ind, weight_panel=w)
    path = neut_cache_path(
        "动量_20d", prices,
        hold_period=5, rebalance_freq="W-FRI",
        rebalance_dates=dates, ctrl_sig=ctrl,
    )

    calls = {"n": 0}
    real_resid = __import__(
        "models.wf.labels", fromlist=["residualize_panel"],
    ).residualize_panel

    def counting_resid(*args, **kwargs):
        calls["n"] += 1
        return real_resid(*args, **kwargs)

    monkeypatch.setattr("models.wf.labels.residualize_panel", counting_resid)

    out1 = neutralize_one_factor(
        factor, "动量_20d",
        barra_factors=barra, industry_map=ind, dates_use=dates,
        weight_panel=w, zscore_fn=cs_zscore_sparse_rows,
    )
    save_neut_panel(path, out1, name="动量_20d")
    assert path.exists()
    assert calls["n"] == 1

    hit = try_load_neut_panel(path, prices=prices, name="动量_20d")
    assert hit is not None
    assert np.allclose(
        hit.to_numpy(), out1.to_numpy(), equal_nan=True, rtol=1e-5, atol=1e-5,
    )
    # HIT 路径不应再 residualize
    assert calls["n"] == 1


def test_neut_cache_path_differs_by_neut_controls(tiny_panels):
    """控制变量集合不同 → factor_panel_neut_* 路径不同，禁止与 9 风格残差共用。"""
    import hashlib

    from factors.factor_cache import FACTOR_CACHE_DIR
    from research.rolling_pool.neut_cache import (
        NEUT_CACHE_VERSION,
        _neut_controls_key_infix,
        barra_bundle_sig,
        dates_sig,
        neut_cache_path,
        universe_sig,
    )

    prices, _factor, barra, ind, w, _tmp = tiny_panels
    dates = prices.index
    kwargs = dict(
        hold_period=5, rebalance_freq="W-FRI", rebalance_dates=dates,
    )
    sig_all = barra_bundle_sig(barra, industry_map=ind, weight_panel=w)
    sig_size = barra_bundle_sig(
        {"Barra_Size": barra["Barra_Size"]},
        industry_map=ind, weight_panel=w,
    )
    p_barra = neut_cache_path(
        "动量_20d", prices, ctrl_sig=sig_all, neut_controls="barra", **kwargs,
    )
    p_size = neut_cache_path(
        "动量_20d", prices, ctrl_sig=sig_size, neut_controls="size_industry", **kwargs,
    )
    # 仅改 mode、控制面板相同也必须分键（防忘了子集化）
    p_size_same_panels = neut_cache_path(
        "动量_20d", prices, ctrl_sig=sig_all, neut_controls="size_industry", **kwargs,
    )
    assert p_barra != p_size
    assert p_barra != p_size_same_panels
    assert "factor_panel_neut_" in p_barra.name
    assert p_barra.name != p_size.name
    # 默认 barra 必须与「无 nc: 前缀」的旧 neut_v6 键相同，避免无意义失效
    p_barra_implicit = neut_cache_path(
        "动量_20d", prices, ctrl_sig=sig_all, **kwargs,
    )
    assert p_barra == p_barra_implicit
    assert _neut_controls_key_infix("barra") == ""
    assert _neut_controls_key_infix("size_industry") == "|nc:size_industry"
    old_raw = (
        f"动量_20d|{NEUT_CACHE_VERSION}"
        f"|h5|W-FRI|{universe_sig(prices)}|{dates_sig(dates)}|{sig_all}"
    )
    old_h = hashlib.md5(old_raw.encode("utf-8")).hexdigest()[:16]
    assert p_barra == FACTOR_CACHE_DIR / f"factor_panel_neut_{old_h}.parquet"


def test_neut_cache_path_mcap_tag_isolated(tiny_panels):
    """mcap universe_tag 进子目录且哈希不同，不得覆盖全市场 factor_panel_neut_*。"""
    from research.rolling_pool.neut_cache import neut_cache_path

    prices, *_rest = tiny_panels
    kwargs = dict(
        hold_period=5, rebalance_freq="W-FRI", rebalance_dates=prices.index,
        neut_controls="size_industry",
    )
    p_full = neut_cache_path("动量_20d", prices, **kwargs)
    p_mid = neut_cache_path(
        "动量_20d", prices, universe_tag="mcap30_100", **kwargs,
    )
    assert p_full != p_mid
    assert p_full.parent.name != "mcap30_100"
    assert p_mid.parent.name == "mcap30_100"
    assert p_full.name.startswith("factor_panel_neut_")
    assert p_mid.name.startswith("factor_panel_neut_")
    assert p_full.name != p_mid.name


def test_select_neut_control_factors_size_industry(tiny_panels):
    from models.wf.labels import (
        SIZE_NEUT_FACTOR,
        normalize_neut_controls,
        select_neut_control_factors,
    )

    _prices, _factor, barra, _ind, _w, _tmp = tiny_panels
    assert normalize_neut_controls(None, missing_warn=False) == "barra"
    subset = select_neut_control_factors(barra, "size_industry")
    assert list(subset.keys()) == [SIZE_NEUT_FACTOR]
    assert subset[SIZE_NEUT_FACTOR] is barra["Barra_Size"]
    full = select_neut_control_factors(barra, "barra")
    assert set(full.keys()) == set(barra.keys())
    size_only = select_neut_control_factors(barra, "size")
    assert list(size_only.keys()) == [SIZE_NEUT_FACTOR]


def test_live_neut_cache_path_differs_by_neut_controls(tiny_panels):
    import hashlib

    from factors.factor_cache import FACTOR_CACHE_DIR
    from research.rolling_pool.neut_cache import (
        LIVE_NEUT_CACHE_VERSION,
        live_neut_cache_path,
        universe_sig,
    )

    prices, *_rest = tiny_panels
    as_of = prices.index[-1]
    p_b = live_neut_cache_path("动量_20d", as_of, prices, neut_controls="barra")
    p_s = live_neut_cache_path(
        "动量_20d", as_of, prices, neut_controls="size_industry",
    )
    assert p_b != p_s
    as_of_str = as_of.strftime("%Y%m%d")
    old_raw = (
        f"动量_20d|{LIVE_NEUT_CACHE_VERSION}|{as_of_str}"
        f"|{universe_sig(prices)}|ctrl:na"
    )
    old_h = hashlib.md5(old_raw.encode("utf-8")).hexdigest()[:16]
    assert p_b == FACTOR_CACHE_DIR / f"live_neut_{old_h}.parquet"

