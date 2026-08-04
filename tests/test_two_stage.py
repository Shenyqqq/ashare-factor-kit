"""Synthetic tests for two-stage ridge pool membership / in-pool transforms / stage1 cache."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models.trainer import MLDataset
from models.wf.stage1_cache import (
    build_pool_mask,
    load_stage1_cache,
    save_stage1_cache,
)
from models.wf.two_stage import (
    DEFAULT_STAGE2_POOL_FRAC,
    apply_two_stage_ridge,
    pool_cs_winsor_zscore,
    pool_cs_winsor_zscore_frame,
    top_frac_index,
)


def test_default_pool_frac_is_top20():
    assert DEFAULT_STAGE2_POOL_FRAC == 0.2


def test_top_frac_index_basic():
    s = pd.Series({"a": 0.1, "b": 0.9, "c": 0.5, "d": 0.2, "e": 0.8})
    pool = top_frac_index(s, 0.4)
    assert set(pool) == {"b", "e"}


def test_top_frac_index_pool_frac_0_2():
    """Top20% of 10 names → 2 names (ceil)."""
    s = pd.Series({f"s{i}": float(i) for i in range(10)})
    pool = top_frac_index(s, 0.2)
    assert set(pool) == {"s9", "s8"}
    assert len(pool) == 2


def test_pool_cs_winsor_zscore_mean_zero():
    """In-pool winsor→zscore should have mean≈0 and std≈1."""
    rng = np.random.default_rng(0)
    # Skewed pool with outliers
    y = pd.Series(rng.normal(loc=5.0, scale=2.0, size=80))
    y.iloc[-3:] = 50.0
    z = pool_cs_winsor_zscore(y)
    assert abs(float(np.mean(z))) < 1e-6
    assert abs(float(np.std(z)) - 1.0) < 1e-5


def test_pool_transform_differs_from_universe():
    """Pool-only path must differ from winsor→zscore on the full cross-section."""
    rng = np.random.default_rng(1)
    # Full universe: mostly low, top tail high — pool = top 20% ≈ right tail
    y_full = pd.Series(np.concatenate([
        rng.normal(-2.0, 0.5, size=80),
        rng.normal(3.0, 0.5, size=20),
    ]))
    scores = y_full.copy()  # S1 ranks by y for a clear top pool
    pool = top_frac_index(scores, 0.2)
    y_pool = y_full.loc[pool]

    z_pool = pd.Series(pool_cs_winsor_zscore(y_pool), index=pool)
    z_univ = pd.Series(pool_cs_winsor_zscore(y_full), index=y_full.index)
    z_univ_on_pool = z_univ.loc[pool]

    assert abs(float(z_pool.mean())) < 1e-6
    # Universe z-score restricted to pool is NOT mean-zero (pool is the right tail).
    assert abs(float(z_univ_on_pool.mean())) > 0.3
    # Paths differ on the same names.
    assert float((z_pool - z_univ_on_pool).abs().mean()) > 0.1


def test_pool_feature_frame_columnwise():
    rng = np.random.default_rng(2)
    X = pd.DataFrame({
        "f0": rng.normal(10, 3, size=40),
        "f1": rng.normal(-5, 1, size=40),
    })
    Z = pool_cs_winsor_zscore_frame(X)
    assert abs(float(Z["f0"].mean())) < 1e-5
    assert abs(float(Z["f1"].mean())) < 1e-5


def test_two_stage_scores_only_in_pool():
    """S2 finite only on S1 top-frac; out-of-pool is -inf on fitted dates."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-03", periods=40, freq="W-FRI")
    stocks = [f"S{i:03d}" for i in range(50)]
    # Two informative factors + noise; labels correlate with f0 among "good" names
    f0 = pd.DataFrame(rng.normal(size=(len(dates), len(stocks))), index=dates, columns=stocks)
    f1 = pd.DataFrame(rng.normal(size=(len(dates), len(stocks))), index=dates, columns=stocks)
    y = 0.05 * f0 + 0.01 * f1 + 0.02 * pd.DataFrame(
        rng.normal(size=(len(dates), len(stocks))), index=dates, columns=stocks,
    )

    dataset = MLDataset(
        factor_panel={"f0": f0, "f1": f1},
        forward_return=y,
        rebalance_dates=list(dates),
        feature_names=["f0", "f1"],
    )

    # Synthetic S1: rank by f0 (higher better) — available from date 10 onward
    s1 = f0.iloc[10:].copy()
    s2 = apply_two_stage_ridge(
        dataset,
        s1,
        hold_period=5,
        pool_frac=0.2,
        lookback_periods=16,
        min_train_samples=80,
        min_pool_size=10,
    )

    assert not s2.empty
    # Check a mid date that should have been fitted
    mid = s1.index[20]
    assert mid in s2.index
    row = s2.loc[mid]
    pool = top_frac_index(s1.loc[mid], 0.2)
    in_pool = row.reindex(pool)
    out_pool = row.drop(labels=pool, errors="ignore")
    assert np.isfinite(in_pool.to_numpy(dtype=float)).sum() >= 10
    # Out-of-pool must be -inf (keeps full universe for qcut), not NaN.
    assert out_pool.notna().all()
    assert np.isneginf(out_pool.to_numpy(dtype=float)).all()


def test_stage1_cache_roundtrip(tmp_path):
    """save → load preserves scores, mask, and meta (no X/y)."""
    dates = pd.date_range("2021-01-08", periods=5, freq="W-FRI")
    stocks = [f"S{i:02d}" for i in range(10)]
    rng = np.random.default_rng(7)
    s1 = pd.DataFrame(
        rng.normal(size=(len(dates), len(stocks))),
        index=dates,
        columns=stocks,
    )
    cache_dir = tmp_path / "stage1_cache"
    save_stage1_cache(
        cache_dir,
        s1,
        pool_frac=0.2,
        meta={
            "horizon": 5,
            "factor_config": "config/demo.yaml",
            "factor_config_hash": "abcd1234abcd1234",
            "feature_neutralize": True,
            "tag": "ridge_h5_demo",
        },
    )
    assert (cache_dir / "s1_scores.parquet").is_file()
    assert (cache_dir / "s1_pool_mask.parquet").is_file()
    assert (cache_dir / "meta.json").is_file()
    # Must not dump feature/label matrices
    assert not (cache_dir / "X.parquet").exists()
    assert not (cache_dir / "y.parquet").exists()

    scores2, mask2, meta2 = load_stage1_cache(cache_dir)
    s1_cmp = s1.copy()
    s1_cmp.index = pd.to_datetime(s1_cmp.index)
    s1_cmp.index.name = "date"
    pd.testing.assert_frame_equal(scores2, s1_cmp.astype(scores2.dtypes), check_freq=False)
    assert meta2["pool_frac"] == 0.2
    assert meta2["horizon"] == 5
    assert meta2["factor_config_hash"] == "abcd1234abcd1234"
    assert "rebalance_dates" in meta2
    assert len(meta2["rebalance_dates"]) == 5

    live_mask = build_pool_mask(s1, 0.2)
    live_mask.index = pd.to_datetime(live_mask.index)
    live_mask.index.name = "date"
    pd.testing.assert_frame_equal(
        mask2, live_mask, check_dtype=False, check_freq=False,
    )


def test_cache_pool_matches_live_top20(tmp_path):
    """Pool restored from cache equals live Top20% on the same S1 scores."""
    dates = pd.date_range("2021-01-08", periods=8, freq="W-FRI")
    stocks = [f"S{i:02d}" for i in range(25)]
    rng = np.random.default_rng(11)
    s1 = pd.DataFrame(
        rng.normal(size=(len(dates), len(stocks))),
        index=dates,
        columns=stocks,
    )
    cache_dir = tmp_path / "stage1_cache"
    save_stage1_cache(cache_dir, s1, pool_frac=0.2, meta={"horizon": 5})
    _, mask, meta = load_stage1_cache(cache_dir)
    assert meta["pool_frac"] == 0.2

    for d in s1.index:
        live = set(top_frac_index(s1.loc[d], 0.2))
        cached = set(mask.loc[d][mask.loc[d]].index)
        assert live == cached, f"mismatch on {d.date()}: {live ^ cached}"


def test_apply_two_stage_uses_cached_pool_mask():
    """Optional pool_mask drives membership (same Top20% as scores)."""
    rng = np.random.default_rng(3)
    dates = pd.date_range("2020-01-03", periods=36, freq="W-FRI")
    stocks = [f"S{i:03d}" for i in range(40)]
    f0 = pd.DataFrame(rng.normal(size=(len(dates), len(stocks))), index=dates, columns=stocks)
    f1 = pd.DataFrame(rng.normal(size=(len(dates), len(stocks))), index=dates, columns=stocks)
    y = 0.04 * f0 + 0.02 * pd.DataFrame(
        rng.normal(size=(len(dates), len(stocks))), index=dates, columns=stocks,
    )
    dataset = MLDataset(
        factor_panel={"f0": f0, "f1": f1},
        forward_return=y,
        rebalance_dates=list(dates),
        feature_names=["f0", "f1"],
    )
    s1 = f0.iloc[8:].copy()
    mask = build_pool_mask(s1, 0.2)
    s2 = apply_two_stage_ridge(
        dataset,
        s1,
        hold_period=5,
        pool_frac=0.2,
        lookback_periods=14,
        min_train_samples=60,
        min_pool_size=8,
        pool_mask=mask,
    )
    mid = s1.index[18]
    row = s2.loc[mid]
    pool = set(top_frac_index(s1.loc[mid], 0.2))
    finite = set(row.index[np.isfinite(row.to_numpy(dtype=float))])
    assert finite == pool


if __name__ == "__main__":
    test_default_pool_frac_is_top20()
    test_top_frac_index_basic()
    test_top_frac_index_pool_frac_0_2()
    test_pool_cs_winsor_zscore_mean_zero()
    test_pool_transform_differs_from_universe()
    test_pool_feature_frame_columnwise()
    test_two_stage_scores_only_in_pool()
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_stage1_cache_roundtrip(Path(td))
        test_cache_pool_matches_live_top20(Path(td) / "b")
    test_apply_two_stage_uses_cached_pool_mask()
    print("all ok")
