"""barra_residual 标签：索引对齐，避免静默退化成 cs_zscore。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models.wf.labels import (
    cross_sectional_zscore,
    residual_return_label,
    transform_labels,
)


def test_transform_labels_barra_residual_aligns_ndarray_to_stock_index():
    """trainer 传 ndarray y + 股票代码索引控制矩阵时，残差化必须真正生效。"""
    rng = np.random.default_rng(0)
    stocks = pd.Index([f"{i:06d}" for i in range(80)])
    noise = rng.normal(size=80)
    size = rng.normal(size=80)
    # y 与 Size 强相关 → 正确残差化后应远离 cs_zscore
    y = (0.85 * size + 0.15 * noise).astype(np.float32)
    barra = pd.DataFrame({"Barra_Size": size.astype(np.float32)}, index=stocks)
    ind = pd.DataFrame(
        {"_ind_A": np.r_[np.ones(40), np.zeros(40)].astype(np.float32)},
        index=stocks,
    )

    out = transform_labels(
        y, "barra_residual", barra_factors=barra, industry_dummies=ind,
    )
    out_z = transform_labels(y, "cs_zscore")

    corr_z = float(np.corrcoef(out, out_z)[0, 1])
    assert corr_z < 0.95, (
        f"barra_residual 与 cs_zscore 相关过高 ({corr_z:.4f})，"
        "疑似索引错位导致控制变量全 0"
    )

    # 与显式 Series 路径一致
    resid = residual_return_label(pd.Series(y, index=stocks), barra, ind)
    expected = cross_sectional_zscore(resid.values.astype(float)).astype(np.float32)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_transform_labels_barra_residual_accepts_series_y():
    stocks = pd.Index(["a", "b", "c", "d"] + [f"s{i}" for i in range(40)])
    rng = np.random.default_rng(1)
    y = pd.Series(rng.normal(size=len(stocks)), index=stocks)
    barra = pd.DataFrame({"Barra_Size": rng.normal(size=len(stocks))}, index=stocks)
    out = transform_labels(
        y, "barra_residual", barra_factors=barra, industry_dummies=None,
    )
    assert out.shape == (len(stocks),)
    assert np.isfinite(out).all()
