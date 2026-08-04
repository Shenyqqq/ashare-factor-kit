"""人造面板：多头贡献 vs 空头贡献可区分；负 IC 对齐后可分类。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.ic.quantile_decomp import (
    classify_ls_source,
    compute_quantile_ls_from_resid_loop,
    ic_align_sign,
    quantile_ls_one_cross_section,
    summarize_quantile_ls,
)


def test_classify_ls_source_thresholds():
    assert classify_ls_source(0.8, 0.2) == "多头主导"  # share=0.8
    assert classify_ls_source(0.2, 0.8) == "空头主导"  # share=0.2
    assert classify_ls_source(0.5, 0.5) == "双边"
    assert classify_ls_source(-0.1, -0.1) == "无效"
    assert classify_ls_source(np.nan, 0.1) == "无效"


def test_quantile_ls_long_side_dominates():
    """Q5 显著跑赢截面，Q1 接近截面 → 多头主导。"""
    rng = np.random.default_rng(0)
    n = 200
    # 残差因子排序与收益弱相关，但仅最高组有超额
    resid_x = np.linspace(-2, 2, n)
    y = rng.normal(0, 0.01, n)
    y[-40:] += 0.05  # Q5 区域抬高
    out = quantile_ls_one_cross_section(resid_x, y, min_stocks=30)
    assert out is not None
    assert out["多头超额"] > out["空头贡献"]
    assert out["spread"] > 0
    src = classify_ls_source(out["多头超额"], out["空头贡献"])
    assert src == "多头主导"


def test_quantile_ls_short_side_dominates():
    """Q1 显著跑输截面，Q5 接近截面 → 空头主导。"""
    rng = np.random.default_rng(1)
    n = 200
    resid_x = np.linspace(-2, 2, n)
    y = rng.normal(0, 0.01, n)
    y[:40] -= 0.05  # Q1 区域压低
    out = quantile_ls_one_cross_section(resid_x, y, min_stocks=30)
    assert out is not None
    assert out["空头贡献"] > out["多头超额"]
    assert out["spread"] > 0
    src = classify_ls_source(out["多头超额"], out["空头贡献"])
    assert src == "空头主导"


def test_quantile_ls_bilateral():
    """Q5 抬高 + Q1 压低，两侧贡献接近 → 双边。"""
    rng = np.random.default_rng(2)
    n = 200
    resid_x = np.linspace(-2, 2, n)
    y = rng.normal(0, 0.005, n)
    y[-40:] += 0.03
    y[:40] -= 0.03
    out = quantile_ls_one_cross_section(resid_x, y, min_stocks=30)
    assert out is not None
    src = classify_ls_source(out["多头超额"], out["空头贡献"])
    assert src == "双边"
    assert 0.35 <= out["long_share"] <= 0.65


def _make_date_ctrl_and_factor(
    *,
    n_dates: int = 8,
    n_stocks: int = 120,
    mode: str = "long",
    seed: int = 42,
):
    """构造简易 date_ctrl（无真实 Barra）+ 因子面板。

    mode:
      long      — 高因子组收益高（多头，正 IC）
      short     — 低因子组收益低（空头贡献，正 IC）
      neg_corr  — 高因子组收益低（负 IC；对齐后应可分类）
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-31", periods=n_dates, freq="ME")
    codes = [f"{i:06d}" for i in range(n_stocks)]
    factor = pd.DataFrame(rng.normal(0, 1, (n_dates, n_stocks)), index=dates, columns=codes)

    date_ctrl = {}
    for d in dates:
        # 控制矩阵：常数以外加一列噪声（模拟风格）；OLS 后残差 ≈ 因子本身去均值
        ctrl = rng.normal(0, 0.1, (n_stocks, 1)).astype(np.float32)
        f = factor.loc[d].values.astype(np.float64)
        y = rng.normal(0, 0.01, n_stocks)
        order = np.argsort(f)
        if mode == "long":
            y[order[-n_stocks // 5:]] += 0.04
        elif mode == "neg_corr":
            # 高因子 → 低收益：负相关；对齐取反后 Q5 为原低分组
            y[order[-n_stocks // 5:]] -= 0.04
            y[order[: n_stocks // 5]] += 0.04
        else:
            y[order[: n_stocks // 5]] -= 0.04
        date_ctrl[d] = (ctrl, pd.Index(codes), y.astype(np.float32))
    return factor, date_ctrl, dates


def test_ic_align_sign():
    assert ic_align_sign(pd.Series([0.1, 0.2, 0.05])) == 1.0
    assert ic_align_sign(pd.Series([-0.1, -0.2, -0.05])) == -1.0
    assert ic_align_sign(pd.Series([], dtype=float)) == 1.0
    assert ic_align_sign(0.0) == 1.0


def test_negative_ic_cross_section_aligned_not_invalid():
    """负相关因子：未对齐 → 无效；对齐取反后 spread>0 且可分类。"""
    rng = np.random.default_rng(3)
    n = 200
    resid_x = np.linspace(-2, 2, n)
    y = -0.03 * resid_x + rng.normal(0, 0.005, n)

    raw = quantile_ls_one_cross_section(resid_x, y, min_stocks=30)
    assert raw is not None
    assert raw["spread"] < 0
    assert classify_ls_source(raw["多头超额"], raw["空头贡献"]) == "无效"

    aligned = quantile_ls_one_cross_section(-resid_x, y, min_stocks=30)
    assert aligned is not None
    assert aligned["spread"] > 0
    src = classify_ls_source(aligned["多头超额"], aligned["空头贡献"])
    assert src != "无效"
    assert src in ("多头主导", "空头主导", "双边")


def test_compute_loop_negative_ic_aligned_not_invalid():
    """负 IC 因子经 loop 内时序均值对齐后，不应标无效且 spread>0。"""
    fac, ctrl, dates = _make_date_ctrl_and_factor(mode="neg_corr", seed=99)
    pure, daily = compute_quantile_ls_from_resid_loop(
        fac, ctrl, dates, y_mode="raw", min_stocks=30
    )
    assert len(pure) > 0
    assert float(pure.mean()) < 0
    assert daily.attrs.get("ic_sign") == -1.0
    summary = summarize_quantile_ls(daily)
    assert summary["n_days"] >= 5
    assert summary["spread"] > 0
    assert summary["多空来源"] != "无效"


def test_compute_loop_distinguishes_long_vs_short():
    fac_l, ctrl_l, dates = _make_date_ctrl_and_factor(mode="long")
    fac_s, ctrl_s, _ = _make_date_ctrl_and_factor(mode="short", seed=43)

    _, daily_l = compute_quantile_ls_from_resid_loop(
        fac_l, ctrl_l, dates, y_mode="raw", min_stocks=30
    )
    _, daily_s = compute_quantile_ls_from_resid_loop(
        fac_s, ctrl_s, dates, y_mode="raw", min_stocks=30
    )
    sum_l = summarize_quantile_ls(daily_l)
    sum_s = summarize_quantile_ls(daily_s)

    assert sum_l["n_days"] >= 5
    assert sum_s["n_days"] >= 5
    assert sum_l["多空来源"] == "多头主导"
    assert sum_s["多空来源"] == "空头主导"
    assert sum_l["多头超额"] > sum_l["空头贡献"]
    assert sum_s["空头贡献"] > sum_s["多头超额"]


def test_residual_y_mode_runs():
    fac, ctrl, dates = _make_date_ctrl_and_factor(mode="long")
    pure, daily = compute_quantile_ls_from_resid_loop(
        fac, ctrl, dates, y_mode="residual", min_stocks=30
    )
    assert len(pure) > 0
    assert not daily.empty
    assert "多头超额" in daily.columns


def test_qcut_insufficient_stocks_returns_none():
    out = quantile_ls_one_cross_section(
        np.arange(10, dtype=float),
        np.random.randn(10),
        min_stocks=30,
    )
    assert out is None
