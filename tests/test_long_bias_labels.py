"""多头偏置 sample_weight + cs_rank_softlong 软截断标签。"""
from __future__ import annotations

import numpy as np

from models.wf.labels import (
    cross_sectional_rank,
    long_bias_sample_weights,
    soft_truncate_rank_label,
    transform_labels,
)


def test_long_bias_step_weights_shape_and_levels():
    y = np.arange(10, dtype=float)  # ranks 0..1 evenly
    w = long_bias_sample_weights(
        y, top_frac=0.4, bottom_weight=0.25, curve="step",
    )
    assert w.shape == y.shape
    # top 40% of 10 → ranks >= 0.6 → last 4
    assert np.allclose(w[-4:], 1.0)
    assert np.allclose(w[:6], 0.25)
    assert np.all(w > 0)


def test_long_bias_smooth_monotone_no_zero():
    y = np.linspace(-2, 2, 101)
    w = long_bias_sample_weights(
        y, top_frac=0.4, bottom_weight=0.3, curve="smooth", transition=0.10,
    )
    assert w.shape == y.shape
    assert np.all(w >= 0.3 - 1e-9)
    assert np.all(w <= 1.0 + 1e-9)
    # 单调非降（按原 y 升序已排）
    assert np.all(np.diff(w) >= -1e-12)
    # 两端贴近 bottom / 1
    assert w[0] == 0.3
    assert w[-1] == 1.0
    # 过渡区有介于其间的值（非硬阶跃）
    mid = w[(w > 0.3 + 1e-6) & (w < 1.0 - 1e-6)]
    assert len(mid) > 0


def test_soft_truncate_continuous_at_tau():
    y = np.arange(100, dtype=float)
    out = soft_truncate_rank_label(y, top_frac=0.4, floor_slope=0.25)
    r = cross_sectional_rank(y)
    tau = 0.60
    assert out.shape == y.shape
    # τ 处 ≈ 0；下侧负、上侧正
    near = np.abs(r - tau).argmin()
    assert abs(float(out[near])) < 0.05
    assert float(out[r < tau].max()) <= 1e-6
    assert float(out[r > tau].min()) >= -1e-6
    assert float(out.min()) >= -0.25 - 1e-5
    assert float(out.max()) <= 1.0 + 1e-5
    # 无大块硬零平台（常数 0 占比应远低于 50%）
    zero_frac = float(np.mean(np.abs(out) < 1e-8))
    assert zero_frac < 0.05


def test_transform_labels_cs_rank_softlong():
    y = np.random.default_rng(0).normal(size=50)
    a = transform_labels(y, "cs_rank_softlong", top_frac=0.4, floor_slope=0.2)
    b = soft_truncate_rank_label(y, top_frac=0.4, floor_slope=0.2)
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)
