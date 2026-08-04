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
