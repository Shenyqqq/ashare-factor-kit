"""
research/ic/orthogonalize.py — Gram-Schmidt 因子正交选择

按 ICIR 降序迭代：每个候选因子扣除已选因子的截面成分（OLS 残差），
如果残差仍有显著 IC → 保留（有增量信号）；否则剔除（信号被覆盖）。

机构量化标准做法：从候选池正交出 ≤30 个独立因子作为 ML 输入，
避免 51/60 因子全进白名单导致的过拟合。

算法参考：Yong Z. 等 "Informed Selection of Factors"（迭代残差 IC 检验），
等价于截面 Gram-Schmidt 正交化 + IC 显著性双门槛。
"""
from __future__ import annotations

import gc
from typing import Iterable

import numpy as np
import pandas as pd

from research.ic.ic_series import compute_ic_series
from research.ic.statistics import icir, prepare_ic_for_stats


# ══════════════════════════════════════════════════════════════════════════════
# 截面 Gram-Schmidt 正交化
# ══════════════════════════════════════════════════════════════════════════════

def cross_sectional_orthogonalize(
    factor: pd.DataFrame,
    basis_factors: dict[str, pd.DataFrame],
    dates: list,
    min_stocks_per_date: int = 30,
) -> pd.DataFrame:
    """
    对每个调仓日，OLS regress factor[date] ~ basis[date]，取残差作为正交化后的因子。

    Parameters
    ----------
    factor : pd.DataFrame
        待正交化的因子面板（date × code）。
    basis_factors : dict[str, pd.DataFrame]
        已选基因子面板字典（每个 shape 同 factor）。
    dates : list
        需要计算残差的调仓日列表（仅这些行会输出，其它行丢弃以省内存）。
    min_stocks_per_date : int
        单日有效股票数下限，不足则该日残差全 NaN。

    Returns
    -------
    pd.DataFrame
        残差面板，index ⊂ dates（与 factor.index 取交集），columns = 公共列。
        残差 = factor - (alpha + sum_k beta_k * basis_k)；NaN 股票保留 NaN。

    Notes
    -----
    - 向量化：单日内用 numpy.linalg.lstsq 一次求解；逐日循环无法避免（截面
      有效股票集每日不同），但单日 OLS 仅 ~ms 级。
    - 不做截面标准化：残差即增量成分，下游 compute_ic_series 会做 rank。
    - 基因子缺失某日时，该基因子该日视为全 0（等价于不参与当日回归）。
    """
    basis_names = list(basis_factors.keys())
    # 仅保留 dates 中实际在 factor.index 内的日期
    keep_dates = [d for d in dates if d in factor.index]
    if not keep_dates or not basis_names:
        # 无基因子 → 直接返回原因子在 keep_dates 上的切片
        return factor.loc[keep_dates].copy() if keep_dates else factor.iloc[0:0].copy()

    # 公共列：factor 与所有基因子的列交集
    common_cols = factor.columns
    for b in basis_factors.values():
        common_cols = common_cols.intersection(b.columns)
    common_cols = common_cols.sort_values()
    if len(common_cols) == 0:
        return pd.DataFrame(
            np.nan, index=pd.Index(keep_dates), columns=factor.columns[:0]
        )

    # 预对齐基因子：每个基因子切成 (keep_dates × common_cols)
    basis_aligned: list[pd.DataFrame] = []
    for bn in basis_names:
        b = basis_factors[bn]
        b_sub = b.reindex(index=keep_dates, columns=common_cols)
        basis_aligned.append(b_sub)

    out = pd.DataFrame(
        np.nan, index=pd.Index(keep_dates), columns=common_cols, dtype=np.float32
    )

    n_basis = len(basis_names)
    factor_sub = factor.reindex(index=keep_dates, columns=common_cols)

    # 逐日 OLS（向量化单日内）
    for i, date in enumerate(keep_dates):
        y = factor_sub.iloc[i].to_numpy(dtype=float)
        # 构造 X：shape (n_stocks, n_basis)
        X = np.empty((len(common_cols), n_basis), dtype=float)
        for k, b_sub in enumerate(basis_aligned):
            X[:, k] = b_sub.iloc[i].to_numpy(dtype=float)

        valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        n_valid = int(valid.sum())
        if n_valid < max(min_stocks_per_date, n_basis + 5):
            continue  # 该日残差保持 NaN

        Xv = X[valid]
        yv = y[valid]
        # 加截距项
        X_design = np.column_stack([np.ones(n_valid), Xv])
        try:
            beta, _, _, _ = np.linalg.lstsq(X_design, yv, rcond=None)
        except np.linalg.LinAlgError:
            continue
        resid = yv - X_design @ beta
        out_row = np.full(len(common_cols), np.nan, dtype=float)
        out_row[valid] = resid
        out.iloc[i] = out_row.astype(np.float32)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# IC 统计辅助
# ══════════════════════════════════════════════════════════════════════════════

def _ic_mean_icir(ic_series: pd.Series) -> tuple[float, float]:
    """返回 (IC 均值, ICIR)，使用 prepare_ic_for_stats 一致的预处理。"""
    s = prepare_ic_for_stats(ic_series)
    if len(s) == 0:
        return float("nan"), float("nan")
    mean = float(s.mean())
    ir = icir(ic_series)  # icir 内部已做 prepare_ic_for_stats
    return mean, float(ir)


# ══════════════════════════════════════════════════════════════════════════════
# 迭代 Gram-Schmidt 选择
# ══════════════════════════════════════════════════════════════════════════════

def _pre_filter_candidates(
    summary_df: pd.DataFrame,
    pre_filter_ic: float,
    pre_filter_icir: float,
    pre_filter_t: float,
    use_nw_t: bool,
) -> tuple[list, dict]:
    """
    预筛：IC、|ICIR|、|t|（NW_t 优先）三门槛。

    Returns (candidates_in_summary_order, exclusions_pre)
    """
    t_col = "NW_t统计量" if (use_nw_t and "NW_t统计量" in summary_df.columns) else "t统计量"
    candidates: list = []
    excl: dict = {}
    for name in summary_df.index:
        row = summary_df.loc[name]
        ic_mean = abs(row["IC均值"])
        icir_v = abs(row["ICIR"])
        t_v = abs(row.get(t_col, np.nan))
        reasons = []
        if ic_mean < pre_filter_ic:
            reasons.append(f"|IC均值|={ic_mean:.4f}<{pre_filter_ic}")
        if icir_v < pre_filter_icir:
            reasons.append(f"|ICIR|={icir_v:.4f}<{pre_filter_icir}")
        if not np.isnan(t_v) and t_v < pre_filter_t:
            reasons.append(f"|{t_col}|={t_v:.2f}<{pre_filter_t}")
        if reasons:
            excl[name] = "预筛剔除: " + ", ".join(reasons)
        else:
            candidates.append(name)
    return candidates, excl


def gram_schmidt_select(
    summary_df: pd.DataFrame,
    factor_registry,  # dict-like 或 _LazyFactorRegistry
    forward_return: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex | Iterable,
    tradable: pd.DataFrame | None = None,
    max_factors: int = 30,
    ic_threshold: float = 0.015,
    icir_threshold: float = 0.15,
    pre_filter_ic: float = 0.02,
    pre_filter_icir: float = 0.2,
    pre_filter_t: float = 2.0,
    use_nw_t: bool = True,
    pure_ic_means: dict | None = None,
    verbose: bool = True,
) -> tuple[list, dict]:
    """
    迭代 IC-weighted Gram-Schmidt 选择。

    流程：
      1. 预筛（IC / ICIR / NW_t 三门槛）→ 候选池
      2. 按 |ICIR| 降序排序候选
      3. 逐候选：扣除已选因子截面成分 → 残差 IC 显著则保留
      4. 至 max_factors 个停止

    Parameters
    ----------
    summary_df : pd.DataFrame
        含列 ``IC均值``、``ICIR``、``t统计量``、``NW_t统计量``（可选）。
    factor_registry : dict 或 _LazyFactorRegistry
        支持 ``__getitem__(name) → pd.DataFrame`` 与 ``__contains__``；
        LazyRegistry 时建议 cache=False（每因子仅加载一次，用完即释）。
    forward_return : pd.DataFrame
        N 日前瞻收益面板（与 compute_ic_series 同口径）。
    rebalance_dates : DatetimeIndex / Iterable
        调仓日（残差仅在这些日计算，下游 IC 也只取这些日）。
    tradable : pd.DataFrame | None
        可交易池 mask（与 compute_ic_series 同口径）。
    max_factors : int
        最终保留因子数上限（默认 30）。
    ic_threshold / icir_threshold : float
        残差 IC 显著性双门槛：|残差 IC 均值| > ic_threshold 且
        |残差 ICIR| > icir_threshold 才保留。
    pre_filter_* : float
        预筛门槛（原始 IC 指标）。
    use_nw_t : bool
        预筛优先用 NW_t 列；False 或列缺失则用经典 t。
    pure_ic_means : dict | None
        Barra 纯 IC 均值（可选，用于预筛门槛替代原始 IC；不参与正交）。
    verbose : bool
        打印每个候选的原始/正交后 IC 对比。

    Returns
    -------
    (selected_names, exclusions) : (list, dict)
        selected_names 按选择顺序排列；exclusions 含所有剔除原因（含正交后 IC）。
    """
    rebalance_list = list(rebalance_dates)
    exclusions: dict = {}

    # ── 1. 预筛 ──
    candidates, pre_excl = _pre_filter_candidates(
        summary_df, pre_filter_ic, pre_filter_icir, pre_filter_t, use_nw_t
    )
    exclusions.update(pre_excl)

    if not candidates:
        return [], exclusions

    # ── 2. 按 |ICIR| 降序排序（同 ICIR 则按 |IC均值| 降序） ──
    cand_summary = summary_df.loc[candidates]
    sorted_candidates = (
        cand_summary
        .assign(_abs_icir=cand_summary["ICIR"].abs(),
                _abs_ic=cand_summary["IC均值"].abs())
        .sort_values(["_abs_icir", "_abs_ic"], ascending=False)
        .index.tolist()
    )

    selected: list = []
    selected_panels: dict[str, pd.DataFrame] = {}
    rows_log: list = []  # 用于打印

    # ── 3. 迭代选择 ──
    for name in sorted_candidates:
        if len(selected) >= max_factors:
            break

        if name not in factor_registry:
            exclusions[name] = "因子面板不可用（registry 缺失）"
            continue

        try:
            panel = factor_registry[name]
        except KeyError:
            exclusions[name] = "因子面板加载失败（KeyError）"
            continue
        if panel is None or panel.empty:
            exclusions[name] = "因子面板为空"
            continue

        if not selected:
            # 第一个因子：原始 IC
            orth_ic = compute_ic_series(panel, forward_return, tradable=tradable)
            basis_str = "—（首批，无正交）"
        else:
            residual = cross_sectional_orthogonalize(
                panel, selected_panels, rebalance_list
            )
            orth_ic = compute_ic_series(residual, forward_return, tradable=tradable)
            basis_str = "正交于[" + ", ".join(selected[-3:]) + ("..." if len(selected) > 3 else "") + "]"
            del residual

        mean_o, icir_o = _ic_mean_icir(orth_ic)
        # 原始 IC 均值 / ICIR（来自 summary，便于对比）
        raw_ic = float(summary_df.loc[name, "IC均值"])
        raw_icir = float(summary_df.loc[name, "ICIR"])

        keep = (
            not np.isnan(mean_o)
            and not np.isnan(icir_o)
            and abs(mean_o) > ic_threshold
            and abs(icir_o) > icir_threshold
        )

        rows_log.append({
            "因子": name,
            "原始IC": raw_ic,
            "正交IC": mean_o,
            "原始ICIR": raw_icir,
            "正交ICIR": icir_o,
            "保留": keep,
            "basis": basis_str,
        })

        if keep:
            selected.append(name)
            selected_panels[name] = panel
        else:
            exclusions[name] = (
                f"Gram-Schmidt 正交后无增量：IC {raw_ic:.4f}→{mean_o:.4f}，"
                f"ICIR {raw_icir:.4f}→{icir_o:.4f}（门槛 "
                f"|IC|>{ic_threshold}, |ICIR|>{icir_threshold}）"
            )
            del panel
            if hasattr(factor_registry, "release_cache"):
                # LazyRegistry cache=False 时无操作；cache=True 时清掉该非选中因子
                # （注意：会同时清掉已 selected 的缓存，故 selected 路径已本地持有副本）
                factor_registry.release_cache()
            gc.collect()

    # ── 4. verbose 打印对比表 ──
    if verbose and rows_log:
        print(f"\n{'='*84}")
        print(f"  Gram-Schmidt 正交选择（候选 {len(rows_log)}，保留 {len(selected)}/{max_factors}）")
        print(f"{'='*84}")
        hdr = (f"  {'因子':<18}{'原始IC':>9}{'正交IC':>10}"
               f"{'原始ICIR':>10}{'正交ICIR':>11}  {'判定':<6}  正交基")
        print(hdr)
        print("-" * 84)
        for r in rows_log:
            flag = "OK" if r["保留"] else "X"
            color = "\033[92m" if r["保留"] else "\033[91m"
            reset = "\033[0m"
            def _fmt(v: float) -> str:
                return "nan" if np.isnan(v) else f"{v:>+.4f}"
            print(
                f"  {color}{r['因子']:<18}{reset}"
                f"{_fmt(r['原始IC']):>9}{_fmt(r['正交IC']):>10}"
                f"{_fmt(r['原始ICIR']):>10}{_fmt(r['正交ICIR']):>11}"
                f"  {color}{flag:<6}{reset}  {r['basis']}"
            )

    # 释放 LazyRegistry 缓存（selected_panels 本地副本已持有）
    if hasattr(factor_registry, "release_cache"):
        factor_registry.release_cache()
    del selected_panels
    gc.collect()

    return selected, exclusions
