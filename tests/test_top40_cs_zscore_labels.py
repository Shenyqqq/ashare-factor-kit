"""top40_cs_zscore：全截面 z-score 后仅保留前 40% 幅度。"""
from __future__ import annotations

import numpy as np

from models.wf.labels import (
    cross_sectional_rank,
    cross_sectional_zscore,
    top_cs_zscore_label,
    transform_labels,
)


def test_top40_keeps_full_cs_zscore_on_long_side():
    rng = np.random.default_rng(42)
    y = rng.normal(size=100).astype(np.float32)
    out = transform_labels(y, "top40_cs_zscore")
    z = cross_sectional_zscore(y.astype(float)).astype(np.float32)
    rank = cross_sectional_rank(y.astype(float))
    keep = rank >= 0.60

    assert out.shape == y.shape
    # ~60% 置 0（ties 时可能略偏）
    zero_frac = float(np.mean(out == 0.0))
    assert 0.55 <= zero_frac <= 0.65, f"zero_frac={zero_frac}"
    # 阈上：与全截面 z 一致（含幅度），非区内再标准化
    np.testing.assert_allclose(out[keep], z[keep], rtol=1e-5, atol=1e-5)
    assert np.all(out[~keep] == 0.0)
    # 阈上非全零且有正负幅度
    assert np.any(out[keep] != 0.0)
    assert float(np.nanstd(out[keep])) > 0.1


def test_top_cs_zscore_label_custom_frac():
    y = np.arange(10, dtype=float)  # ranks 0..1 evenly
    out = top_cs_zscore_label(y, top_frac=0.3, fill_value=0.0)
    # top 30% of 10 → ranks >= 0.7 → last 3
    assert np.sum(out != 0.0) == 3
    assert np.all(out[:7] == 0.0)
