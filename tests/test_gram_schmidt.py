"""
合成测试：Gram-Schmidt 正交选择
- A: 强 IC 因子
- B: 与 A 高相关，增量信号弱（B = 0.95*A + 微弱独立噪声）
- C: 独立 IC 因子
预期：A、C 入选；B 因正交后无增量被剔除。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.ic.orthogonalize import (
    cross_sectional_orthogonalize,
    gram_schmidt_select,
)


def _make_panel(values: np.ndarray, dates, codes) -> pd.DataFrame:
    return pd.DataFrame(values, index=pd.Index(dates, name="date"),
                        columns=pd.Index(codes, name="code"))


def test_gram_schmidt_synthetic():
    rng = np.random.default_rng(42)
    n_dates, n_stocks = 80, 300
    dates = pd.date_range("2022-01-31", periods=n_dates, freq="ME")
    codes = [f"S{i:03d}" for i in range(n_stocks)]

    # 两个独立 latent 信号
    latent1 = rng.standard_normal((n_dates, n_stocks))
    latent2 = rng.standard_normal((n_dates, n_stocks))
    # 前瞻收益同时依赖两个 latent
    forward_return = _make_panel(
        0.01 * (latent1 + 0.7 * latent2) + 0.005 * rng.standard_normal((n_dates, n_stocks)),
        dates, codes,
    )

    # A: 强信号 ≈ latent1（含少量测量噪声）
    A = _make_panel(latent1 + 0.1 * rng.standard_normal((n_dates, n_stocks)), dates, codes)
    # B: A 的近乎复制品（无独立信号路径），B = 0.98*A + 极小噪声
    #    正交于 A 后残差 ≈ 纯噪声 → IC ~ 0 → 应被剔除
    B = _make_panel(0.98 * A.values + 0.005 * rng.standard_normal((n_dates, n_stocks)),
                    dates, codes)
    # C: 独立信号 ≈ latent2
    C = _make_panel(latent2 + 0.1 * rng.standard_normal((n_dates, n_stocks)), dates, codes)

    registry = {"A": A, "B": B, "C": C}

    # summary_df：构造 IC 均值/ICIR/t（让 B 的 ICIR 略低于 A，确保 A 先入选）
    summary_df = pd.DataFrame(
        {
            "IC均值": [0.08, 0.076, 0.06],
            "ICIR": [0.85, 0.80, 0.65],
            "t统计量": [5.0, 4.8, 4.0],
            "NW_t统计量": [4.9, 4.7, 3.9],
            "|IC|均值": [0.08, 0.076, 0.06],
        },
        index=["A", "B", "C"],
    )
    summary_df.index.name = "因子"

    rebalance_dates = pd.DatetimeIndex(dates)
    selected, exclusions = gram_schmidt_select(
        summary_df=summary_df,
        factor_registry=registry,
        forward_return=forward_return,
        rebalance_dates=rebalance_dates,
        tradable=None,
        max_factors=3,
        ic_threshold=0.015,
        icir_threshold=0.15,
        pre_filter_ic=0.0,       # 跳过预筛（已在 summary 给定）
        pre_filter_icir=0.0,
        pre_filter_t=0.0,
        use_nw_t=True,
        verbose=False,
    )

    print("\n[合成测试] selected =", selected)
    print("[合成测试] exclusions =")
    for n, r in exclusions.items():
        print(f"   {n}: {r}")

    assert "A" in selected, "A 应入选（强 IC + 首批）"
    assert "C" in selected, "C 应入选（独立信号）"
    assert "B" not in selected, "B 应被剔除（正交后无增量）"
    assert len(selected) == 2
    # 顺序：A 先于 C（|ICIR| 降序）
    assert selected.index("A") < selected.index("C")
    print("[合成测试] PASS ✓")


def test_orthogonalize_residual_uncorrelated():
    """正交后残差应与基因子近似不相关（验证 OLS 几何）。"""
    rng = np.random.default_rng(7)
    n_dates, n_stocks = 30, 100
    dates = pd.date_range("2023-01-31", periods=n_dates, freq="ME")
    codes = [f"S{i:03d}" for i in range(n_stocks)]
    A = _make_panel(rng.standard_normal((n_dates, n_stocks)), dates, codes)
    # Y = 0.8*A + noise
    Y = _make_panel(0.8 * A.values + 0.3 * rng.standard_normal((n_dates, n_stocks)),
                    dates, codes)
    resid = cross_sectional_orthogonalize(Y, {"A": A}, list(dates))
    # 截面 corr(A, resid) 应接近 0
    corrs = []
    for d in dates:
        if d in resid.index and d in A.index:
            a = A.loc[d].values
            r = resid.loc[d].values
            mask = np.isfinite(a) & np.isfinite(r)
            if mask.sum() > 30:
                av, rv = a[mask], r[mask]
                av = av - av.mean()
                rv = rv - rv.mean()
                denom = np.sqrt((av ** 2).sum() * (rv ** 2).sum())
                if denom > 0:
                    corrs.append((av * rv).sum() / denom)
    mean_corr = float(np.mean(corrs))
    print(f"\n[正交几何] mean corr(A, residual) = {mean_corr:.4f}")
    assert abs(mean_corr) < 0.05, f"残差应与 A 近似不相关，got {mean_corr:.4f}"
    print("[正交几何] PASS ✓")


if __name__ == "__main__":
    try:
        from config.encoding_bootstrap import bootstrap_stdio_utf8
        bootstrap_stdio_utf8()
    except Exception:
        pass
    test_orthogonalize_residual_uncorrelated()
    test_gram_schmidt_synthetic()
    print("\n全部测试通过")
