"""Factor screening: thresholds, stability, correlation dedup, cost-adjusted IC.

扩展（稠密 / 稀疏 / 新兴观察）
-----------------------------
- **稠密（普通）因子**：|IC|∧|ICIR|（合取 AND；barra 用 pure 序列）+ t/FDR
  + **long_share**（默认 >0.4，符号对齐后；``--min-long-share 0`` 关闭）
  → 相关去重 →（可选 GS）→ ``dense_kept`` / ``factors``。
- **新兴因子**：全样本未过 IC∧ICIR 稠密门 + 近窗 pure 序列上
  BH-FDR(NW-t) ∧ |ICIR_recent| ∧ lift [∧ 三季度增强] → **仅标注**
  （``factors_emerging``），**不**并入 ``dense_kept``；ML 默认白名单不含新兴。
  近窗禁止 raw 救援。可选 holdout / asof 切断评效段。
  FDR 校正域 = 全体进入稠密筛选且有近窗 IC 的因子。
- **衰减因子 / 风格逆转**：仅标注（实盘谨慎），**不**从 factors 池剔除。
- **稀疏因子**：语义池见 ``factors.sparse_factors``；独立轨道，主门槛为
  **方向对齐** IC 胜率 + 触发日相对截面均值胜率（payoff_hit）；无 t/NW-t/FDR；
  IC/ICIR 软参考。负 IC 因子按 ``sign(mean_IC)`` 翻转后再算胜率/触发侧。
  胜率/payoff 过线后做 **corr-dedup**（默认阈值 0.70）：优先截面 Spearman+method
  （与稠密同逻辑）；截面不稳定时 fallback IC 序列相关。
- 稀疏入选写入 JSON ``factors_sparse``，经 ``--special-factors sparse`` 注入 ridge，
  **不**进 dynamic 轨道。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config.settings import (
    IC_CORR_METHOD,
    IC_DECAY_RECENT_IC_MAX,
    IC_DECAY_RECENT_ICIR_MAX,
    IC_DECAY_RECENT_MONTHS,
    IC_DECAY_RETENTION_MIN,
    IC_DECAY_RETENTION_MIN_SPARSE,
    IC_EMERGING_FDR_ALPHA,
    IC_EMERGING_HOLDOUT_MONTHS,
    IC_EMERGING_LIFT_MIN,
    IC_EMERGING_LOOKBACK,
    IC_EMERGING_RECENT_IC,
    IC_EMERGING_RECENT_ICIR,
    IC_EMERGING_REQUIRE_TREND,
    IC_EMERGING_TREND_EPS,
    IC_EMERGING_TREND_MONTHS,
    IC_EMERGING_TREND_SEGMENTS,
    IC_MIN_LONG_SHARE,
    IC_THRESHOLD,
    ICIR_THRESHOLD,
    IC_REVERSAL_ABS_IC,
    IC_REVERSAL_FRAC,
    IC_REVERSAL_MONTHS,
    IC_SPARSE_CORR_THRESHOLD,
    IC_SPARSE_IC_THRESHOLD,
    IC_SPARSE_ICIR_THRESHOLD,
    IC_SPARSE_PAYOFF_MIN,
    IC_SPARSE_WIN_RATE_MIN,
)
from factors.sparse_factors import (
    CAT_DECAYED,
    CAT_DENSE,
    CAT_EMERGING,
    CAT_REVERSAL,
    CAT_SPARSE,
    SPARSE_FACTOR_NAMES,
    partition_sparse,
)
from research.ic.cost import estimate_ic_after_cost, rank_autocorr_turnover
from research.ic.statistics import (
    benjamini_hochberg,
    ic_direction_sign,
    ic_stability_metrics,
    icir,
    newey_west_t,
    prepare_ic_for_stats,
    recent_past_icir_retention,
    recent_window_stats,
    segment_metric_trend,
    style_reversal_fraction,
    trigger_cs_payoff,
    win_rates,
)

# TODO: turnover / capacity / SHAP-based selection — not implemented in v2
# TODO: integrate holdings liquidity (ADV) filter from backtest v2


@dataclass
class FactorSelectionResult:
    """多轨筛选结果。``kept`` = 稠密轨道（不含新兴），供 ML YAML。

    ``emerging_kept``：新兴观察名单（``factors_emerging``），不进主池。
    ``categories``：主类别（普通 / 稀疏 / 新兴）。
    ``labels``：警示标签列表（衰减因子 / 风格逆转），可叠加，不剔除。
    """

    dense_kept: list[str] = field(default_factory=list)
    sparse_kept: list[str] = field(default_factory=list)
    emerging_kept: list[str] = field(default_factory=list)
    exclusions: dict = field(default_factory=dict)
    categories: dict[str, str] = field(default_factory=dict)
    labels: dict[str, list[str]] = field(default_factory=dict)

    @property
    def kept(self) -> list[str]:
        return self.dense_kept


def enrich_summary_with_cost(
    summary_df: pd.DataFrame,
    registry: dict,
    rebalance_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Add rank-turnover proxy and IC_after_cost columns."""
    turnovers = {}
    for name in summary_df.index:
        if name in registry:
            turnovers[name] = rank_autocorr_turnover(registry[name], rebalance_dates)
        else:
            turnovers[name] = np.nan
    summary_df = summary_df.copy()
    summary_df["rank_turnover"] = pd.Series(turnovers)
    summary_df["IC_after_cost"] = summary_df.apply(
        lambda r: estimate_ic_after_cost(r["IC均值"], r["rank_turnover"]),
        axis=1,
    )
    return summary_df


def overlay_pure_t_stats(
    summary_df: pd.DataFrame,
    pure_ic_series: dict | None,
) -> pd.DataFrame:
    """用 pure IC 序列重写 ``t统计量`` / ``NW_t统计量``。

    barra 模式下 IC/ICIR 门已用 pure，若仍读 summary 的 raw t/NW-t 会混用口径。
    无 pure 序列重算后，FDR / |t| 门槛与 pure IC 一致；无 pure 序列的因子保留 raw。
    """
    if not pure_ic_series:
        return summary_df
    out = summary_df.copy()
    n = 0
    for name, ic in pure_ic_series.items():
        if name not in out.index or ic is None:
            continue
        s = prepare_ic_for_stats(ic)
        if len(s) < 3:
            continue
        std0 = float(s.std(ddof=0))
        t_iid = float(s.mean() / (std0 / np.sqrt(len(s)))) if std0 > 0 else 0.0
        out.loc[name, "t统计量"] = round(t_iid, 2)
        out.loc[name, "NW_t统计量"] = round(float(newey_west_t(ic)), 2)
        n += 1
    if n:
        print(f"  [barra] 已用 pure IC 重算 t/NW_t：{n} 个因子（避免 pure IC + raw NW-t 混用）")
    return out


def _abs_t_for_gate(
    summary_df: pd.DataFrame,
    name: str,
    use_nw: bool,
    pure_ic_series: dict | None = None,
) -> float:
    """门控用 |t|：有 pure 序列时优先从 pure 算，否则读 summary。"""
    if pure_ic_series:
        ic = pure_ic_series.get(name)
        if ic is not None:
            s = prepare_ic_for_stats(ic)
            if len(s) >= 3:
                if use_nw:
                    return abs(float(newey_west_t(ic)))
                std0 = float(s.std(ddof=0))
                if std0 <= 0:
                    return 0.0
                return abs(float(s.mean() / (std0 / np.sqrt(len(s)))))
    col = "NW_t统计量" if use_nw else "t统计量"
    v = summary_df.loc[name].get(col, np.nan) if name in summary_df.index else np.nan
    try:
        return abs(float(v))
    except (TypeError, ValueError):
        return float("nan")


def _fdr_sig_map(
    summary_df: pd.DataFrame,
    use_nw: bool,
    fdr_alpha: float,
    pure_ic_series: dict | None = None,
) -> dict[str, bool]:
    """BH-FDR 显著性；barra 时用 pure 序列上的 t/NW-t。"""
    t_col = "NW_t统计量" if use_nw else "t统计量"
    if pure_ic_series:
        t_vals = np.array([
            _abs_t_for_gate(summary_df, name, use_nw, pure_ic_series)
            for name in summary_df.index
        ], dtype=float)
        t_vals = np.nan_to_num(t_vals, nan=0.0)
        sig_mask = benjamini_hochberg(t_vals, alpha=fdr_alpha)
        return dict(zip(summary_df.index, sig_mask))
    if t_col not in summary_df.columns:
        return {}
    t_vals = summary_df[t_col].fillna(0.0).values
    sig_mask = benjamini_hochberg(t_vals, alpha=fdr_alpha)
    return dict(zip(summary_df.index, sig_mask))


def _pairwise_corr_matrix(
    cand_registry,
    sample_dates: list,
    candidates: list | None = None,
) -> list[pd.DataFrame]:
    """
    流式构建逐调仓日截面 corr 矩阵，避免同时持有所有候选因子完整面板。

    ``cand_registry`` 可以是普通 dict 或 `_LazyFactorRegistry`：
      - 普通 dict：直接迭代 .items()（向后兼容；忽略 candidates 参数）
      - LazyRegistry：逐候选 __getitem__ 加载完整面板 → 仅取 sample_dates 行切片
        → 立即释放全面板（峰值 = 1 个完整面板 ~48MB + 全部候选切片 ~66MB，
        而非 |candidates| × 48MB ≈ 1.4GB）。此模式需传 candidates 列表。

    数值与原实现完全一致：corr 计算只用到 sample_dates 上的截面，切片取的
    正是这些行，不影响 spearman 结果。
    """
    is_lazy = hasattr(cand_registry, "release_cache") and hasattr(cand_registry, "__getitem__")

    if is_lazy:
        # 流式路径：逐候选加载 → 取 sample_dates 切片 → 释放全面板
        names_to_load = candidates if candidates is not None else (
            list(cand_registry._names) if hasattr(cand_registry, "_names") else []
        )
        slices: list[tuple[str, pd.DataFrame]] = []
        for name in names_to_load:
            if name not in cand_registry:
                continue
            try:
                panel = cand_registry[name]
            except KeyError:
                continue
            if panel is None or panel.empty:
                continue
            sub = panel.loc[panel.index.intersection(sample_dates)]
            slices.append((name, sub.astype(np.float32, copy=False)))
            del panel
        # 释放 LazyRegistry 内部缓存（cache=False 时无操作）
        cand_registry.release_cache()
        items_list = slices
    else:
        items_list = list(cand_registry.items())

    corr_list = []
    for date in sample_dates:
        row = {}
        for name, fdf in items_list:
            if date in fdf.index:
                row[name] = fdf.loc[date]
        if not row:
            continue
        # Pairwise min_periods — do NOT global dropna. With 60+ candidates,
        # requiring all factors non-null collapses the cross-section below 30
        # and silently skips corr dedup (seen on h20 overnight runs).
        df_slice = pd.DataFrame(row)
        if df_slice.shape[0] <= 30:
            continue
        c = df_slice.corr(method="spearman", min_periods=30)
        if c.isna().all().all():
            continue
        corr_list.append(c)
    return corr_list


def _aggregate_corr(corr_list: list[pd.DataFrame], method: str) -> pd.DataFrame:
    if not corr_list:
        return pd.DataFrame()
    stacked = pd.concat(corr_list)
    if method == "mean":
        return stacked.groupby(level=0).mean()
    if method == "p95":
        return stacked.groupby(level=0).quantile(0.95)
    # max: upper triangle max abs per pair across time
    # Use nanmax — pairwise corr matrices contain NaNs (min_periods), and
    # Python max([nan, 0.9]) is order-dependent / can return nan, which
    # silently disables corr-dedup (h20 overnight: 0–1 drops vs h5's ~25).
    names = corr_list[0].index.tolist()
    out = pd.DataFrame(np.nan, index=names, columns=names)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                out.loc[a, b] = 1.0
                continue
            vals = [
                float(abs(c.loc[a, b]))
                for c in corr_list
                if a in c.index and b in c.columns and np.isfinite(c.loc[a, b])
            ]
            if vals:
                out.loc[a, b] = max(vals)
    return out


def select_factors(
    summary_df: pd.DataFrame,
    factor_registry: dict,
    pure_ic_means: dict | None = None,
    pure_ic_series: dict | None = None,
    ic_threshold: float = IC_THRESHOLD,
    icir_threshold: float = ICIR_THRESHOLD,
    t_threshold: float = 2.5,
    nw_t_threshold: float | None = None,
    corr_threshold: float = 0.70,
    sample_step: int = 20,
    corr_method: str | None = None,
    rebalance_dates: pd.DatetimeIndex | None = None,
    use_fdr: bool = False,
    fdr_alpha: float = 0.05,
    regime_consistency_threshold: float | None = None,
    rolling_icir_threshold: float | None = None,
    worst_period_ic_threshold: float | None = None,
    corr_dedup: bool = True,
    min_long_share: float | None = IC_MIN_LONG_SHARE,
) -> tuple[list, dict]:
    """
    Auto factor selection:
      1. Drop unless |IC_eff|>th AND |ICIR_eff|>th (合取); use NW t when available
         - barra/pure: IC_eff/ICIR_eff from pure IC series (not summary raw ICIR)
         - If use_fdr=True: apply Benjamini-Hochberg FDR correction to NW_t
           (or IID t fallback) across all factors; non-rejected factors are
           excluded as statistically insignificant. Original |t|>threshold
           simple rule is preserved as fallback when use_fdr=False.
      2. long_share gate (default >0.4, sign-aligned); disable with min_long_share≤0
      3. Correlation dedup (max | p95 | mean pairwise corr) when corr_dedup=True
      4. Stability metrics logged in summary but not hard-filtered yet
    """
    exclusions = {}
    method = corr_method or IC_CORR_METHOD
    use_nw = nw_t_threshold is not None

    fdr_sig: dict[str, bool] = {}
    if use_fdr:
        fdr_sig = _fdr_sig_map(
            summary_df, use_nw=use_nw, fdr_alpha=fdr_alpha,
            pure_ic_series=pure_ic_series,
        )

    candidates = []
    for name in summary_df.index:
        row = summary_df.loc[name]
        t_stat = _abs_t_for_gate(summary_df, name, use_nw, pure_ic_series)
        thresh = nw_t_threshold if use_nw else t_threshold
        effective_ic, effective_icir = _effective_ic_icir(
            row, name, pure_ic_means, pure_ic_series,
        )
        use_pure = bool(pure_ic_series) or bool(pure_ic_means)

        if (
            (not np.isfinite(effective_ic) or effective_ic < ic_threshold)
            or (not np.isfinite(effective_icir) or effective_icir < icir_threshold)
        ):
            exclusions[name] = _ic_icir_gate_reason(
                effective_ic, effective_icir, ic_threshold, icir_threshold,
                use_pure=use_pure,
            )
        elif use_fdr and not fdr_sig.get(name, False):
            label = "NW_t" if use_nw else "t"
            exclusions[name] = f"{label}={t_stat:.2f} 未通过 BH-FDR(α={fdr_alpha})（多重检验不显著）"
        elif not use_fdr and not np.isnan(t_stat) and t_stat < thresh:
            label = "NW_t" if use_nw else "t"
            exclusions[name] = f"{label}={t_stat:.2f}<{thresh}（IC均值统计不显著）"
        elif (
            regime_consistency_threshold is not None
            and not np.isnan(row.get("同向年份占比", np.nan))
            and row["同向年份占比"] < regime_consistency_threshold
        ):
            val = row["同向年份占比"]
            exclusions[name] = (
                f"同向年份占比={val:.2f}<{regime_consistency_threshold}（regime 一致性不足）"
            )
        elif (
            rolling_icir_threshold is not None
            and not np.isnan(row.get("IC滚动ICIR", np.nan))
            and row["IC滚动ICIR"] < rolling_icir_threshold
        ):
            val = row["IC滚动ICIR"]
            exclusions[name] = (
                f"滚动ICIR={val:.3f}<{rolling_icir_threshold}（IC 时变稳定性不足）"
            )
        elif (
            worst_period_ic_threshold is not None
            and not np.isnan(row.get("最差12期IC均值", np.nan))
            and row["最差12期IC均值"] < worst_period_ic_threshold
        ):
            val = row["最差12期IC均值"]
            exclusions[name] = (
                f"最差12期IC={val:.4f}<{worst_period_ic_threshold}（regime 切换时反向过深）"
            )
        else:
            ls_reason = _long_share_gate_reason(row, min_long_share)
            if ls_reason:
                exclusions[name] = ls_reason
            else:
                candidates.append(name)

    if not candidates:
        return [], exclusions

    if not corr_dedup:
        return candidates, exclusions

    # 取首个候选的 index 推导 sample_dates（仅加载 1 个面板，用完即释）
    first_panel = factor_registry.get(candidates[0]) if hasattr(factor_registry, "get") else factor_registry[candidates[0]]
    if first_panel is None:
        print("  [warn] 相关去重：首候选面板为空，跳过 corr dedup（候选池未缩减）")
        return candidates, exclusions
    sample_dates = list(first_panel.index[::sample_step])
    del first_panel
    # 释放 LazyRegistry 缓存（cache=False 时无操作；确保首面板不残留）
    if hasattr(factor_registry, "release_cache"):
        factor_registry.release_cache()

    if len(candidates) <= 1:
        return candidates, exclusions

    # 流式 corr：传 LazyRegistry + candidates 列表，_pairwise_corr_matrix 内部
    # 逐候选加载 → 取 sample_dates 切片 → 释放全面板（峰值 1 个面板 + 切片集合）
    corr_list = _pairwise_corr_matrix(factor_registry, sample_dates, candidates=candidates)
    if not corr_list:
        print(
            f"  [warn] 相关去重：截面 corr 矩阵为空（{len(candidates)} 候选 / "
            f"{len(sample_dates)} 采样日），跳过 corr dedup（候选池未缩减）"
        )
        return candidates, exclusions

    agg_corr = _aggregate_corr(corr_list, method)

    icir_order = (
        summary_df.loc[candidates, "ICIR"]
        .abs().sort_values(ascending=False).index.tolist()
    )
    kept = []
    for name in icir_order:
        if name not in agg_corr.index:
            kept.append(name)
            continue
        drop = False
        for k in kept:
            if k in agg_corr.columns:
                c = abs(agg_corr.loc[name, k])
                if np.isfinite(c) and c > corr_threshold:
                    icir_name = abs(summary_df.loc[name, "ICIR"])
                    icir_k = abs(summary_df.loc[k, "ICIR"])
                    exclusions[name] = (
                        f"与{k}相关({method})={c:.2f}>{corr_threshold}，"
                        f"ICIR({icir_name:.3f})<ICIR({icir_k:.3f})"
                    )
                    drop = True
                    break
        if not drop:
            kept.append(name)

    return kept, exclusions


def select_factors_raw(
    summary_df: pd.DataFrame,
    all_ic: dict,
    pure_ic_means: dict | None = None,
    pure_ic_series: dict | None = None,
    ic_threshold: float = IC_THRESHOLD,
    icir_threshold: float = ICIR_THRESHOLD,
    t_threshold: float = 2.5,
    nw_t_threshold: float | None = None,
    corr_threshold: float = 0.70,
    use_fdr: bool = False,
    fdr_alpha: float = 0.05,
    regime_consistency_threshold: float | None = None,
    rolling_icir_threshold: float | None = None,
    worst_period_ic_threshold: float | None = None,
    corr_dedup: bool = True,
    min_long_share: float | None = IC_MIN_LONG_SHARE,
) -> tuple[list, dict]:
    """
    面板无关的轻量筛选（--raw-select 模式）：

      1. 阈值门（|IC|∧|ICIR| 合取 / t / NW_t / FDR / long_share）+ A/B/C regime 稳定性门
         —— 与 select_factors 完全一致，仅用 summary_df，不需要因子面板。
      2. IC 序列相关性去重（corr_dedup=True）：用 all_ic（每因子一条 IC Series）
         构建 IC-series corr 矩阵，按 |ICIR| 降序迭代，
         |corr| > corr_threshold 的剔除（保留 ICIR 更高者）。

    IC-series corr 是截面因子 corr 的标准近似：直接衡量两个因子信号
    同向程度，对"信号冗余"的刻画比截面 spearman 更贴近选股目的，
    且无需重算 132 个因子面板（省 ~30min）。

    Returns (kept, exclusions) —— 与 select_factors 同口径，可直接落
    selection checkpoint + JSON，下游 ML/backtest 无感。
    """
    exclusions: dict = {}
    use_nw = nw_t_threshold is not None

    fdr_sig: dict[str, bool] = {}
    if use_fdr:
        fdr_sig = _fdr_sig_map(
            summary_df, use_nw=use_nw, fdr_alpha=fdr_alpha,
            pure_ic_series=pure_ic_series,
        )

    candidates: list = []
    for name in summary_df.index:
        row = summary_df.loc[name]
        t_stat = _abs_t_for_gate(summary_df, name, use_nw, pure_ic_series)
        effective_ic, effective_icir = _effective_ic_icir(
            row, name, pure_ic_means, pure_ic_series,
        )
        use_pure = bool(pure_ic_series) or bool(pure_ic_means)

        # NaN-IC 因子（数据不足/全 NaN）直接剔除，避免漏过所有数值门
        if (not np.isfinite(effective_ic)) and (not np.isfinite(effective_icir)):
            exclusions[name] = "IC/ICIR 为 NaN（因子数据不足或不可计算）"
            continue

        thresh = nw_t_threshold if use_nw else t_threshold

        if (
            (not np.isfinite(effective_ic) or effective_ic < ic_threshold)
            or (not np.isfinite(effective_icir) or effective_icir < icir_threshold)
        ):
            exclusions[name] = _ic_icir_gate_reason(
                effective_ic, effective_icir, ic_threshold, icir_threshold,
                use_pure=use_pure,
            )
        elif use_fdr and not fdr_sig.get(name, False):
            label = "NW_t" if use_nw else "t"
            exclusions[name] = f"{label}={t_stat:.2f} 未通过 BH-FDR(α={fdr_alpha})（多重检验不显著）"
        elif not use_fdr and not np.isnan(t_stat) and t_stat < thresh:
            label = "NW_t" if use_nw else "t"
            exclusions[name] = f"{label}={t_stat:.2f}<{thresh}（IC均值统计不显著）"
        elif (
            regime_consistency_threshold is not None
            and not np.isnan(row.get("同向年份占比", np.nan))
            and row["同向年份占比"] < regime_consistency_threshold
        ):
            val = row["同向年份占比"]
            exclusions[name] = (
                f"同向年份占比={val:.2f}<{regime_consistency_threshold}（regime 一致性不足）"
            )
        elif (
            rolling_icir_threshold is not None
            and not np.isnan(row.get("IC滚动ICIR", np.nan))
            and row["IC滚动ICIR"] < rolling_icir_threshold
        ):
            val = row["IC滚动ICIR"]
            exclusions[name] = (
                f"滚动ICIR={val:.3f}<{rolling_icir_threshold}（IC 时变稳定性不足）"
            )
        elif (
            worst_period_ic_threshold is not None
            and not np.isnan(row.get("最差12期IC均值", np.nan))
            and row["最差12期IC均值"] < worst_period_ic_threshold
        ):
            val = row["最差12期IC均值"]
            exclusions[name] = (
                f"最差12期IC={val:.4f}<{worst_period_ic_threshold}（regime 切换时反向过深）"
            )
        else:
            ls_reason = _long_share_gate_reason(row, min_long_share)
            if ls_reason:
                exclusions[name] = ls_reason
            else:
                candidates.append(name)

    if not candidates:
        return [], exclusions
    if not corr_dedup or len(candidates) <= 1:
        return candidates, exclusions

    # IC-series 相关性去重：对齐所有候选 IC 序列 → corr 矩阵
    ic_frame = pd.DataFrame({n: all_ic.get(n) for n in candidates})
    corr_mat = ic_frame.corr(method="pearson")

    icir_order = (
        summary_df.loc[candidates, "ICIR"]
        .abs().sort_values(ascending=False).index.tolist()
    )
    kept: list = []
    for name in icir_order:
        if name not in corr_mat.index:
            kept.append(name)
            continue
        drop = False
        for k in kept:
            if k in corr_mat.columns:
                c = abs(corr_mat.loc[name, k])
                if c > corr_threshold:
                    icir_name = abs(summary_df.loc[name, "ICIR"])
                    icir_k = abs(summary_df.loc[k, "ICIR"])
                    exclusions[name] = (
                        f"IC序列相关={c:.2f}>{corr_threshold}，"
                        f"ICIR({icir_name:.3f})<ICIR({icir_k:.3f})"
                    )
                    drop = True
                    break
        if not drop:
            kept.append(name)

    return kept, exclusions


def stability_report(all_ic: dict) -> pd.DataFrame:
    """Per-factor stability metrics table."""
    rows = []
    for name, ic in all_ic.items():
        rows.append({"因子": name, **ic_stability_metrics(ic)})
    return pd.DataFrame(rows).set_index("因子") if rows else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# 衰减标注 / 新兴 / 稀疏多轨
# ══════════════════════════════════════════════════════════════════════════════


def _effective_ic_icir(
    row: pd.Series,
    name: str,
    pure_ic_means: dict | None,
    pure_ic_series: dict | None = None,
) -> tuple[float, float]:
    """稠密门用的 |IC| / |ICIR|（合取 AND）。

    - 有 ``pure_ic_series``（barra）：二者均来自 **pure IC 序列**（与衰减近窗同口径）。
      禁止用 summary raw ICIR 救援；该因子无可用 pure 序列 → ICIR=NaN（不过门）。
    - 仅有 ``pure_ic_means``（无序列，降级）：IC 用 pure 均值，ICIR 仍用 summary
      （仅兼容旧调用；生产 barra 路径应始终带序列）。
    - 无 pure：退回 summary raw，仍按 AND 使用。
    """
    raw_ic = abs(float(row.get("IC均值", np.nan)))
    raw_icir = abs(float(row.get("ICIR", np.nan)))

    if pure_ic_series:
        s = pure_ic_series.get(name)
        if s is not None:
            s2 = prepare_ic_for_stats(s)
            if len(s2) >= 3:
                return abs(float(s2.mean())), abs(float(icir(s2)))
        # barra 模式但缺该因子 pure 序列：不可用 raw ICIR 过门
        if pure_ic_means:
            pure_ic = abs(float(pure_ic_means.get(name, np.nan)))
            if np.isfinite(pure_ic):
                return pure_ic, np.nan
        return raw_ic, np.nan

    if pure_ic_means:
        pure_ic = abs(float(pure_ic_means.get(name, np.nan)))
        if np.isfinite(pure_ic):
            return pure_ic, raw_icir
    return raw_ic, raw_icir


def _ic_icir_gate_reason(
    effective_ic: float,
    effective_icir: float,
    ic_threshold: float,
    icir_threshold: float,
    *,
    use_pure: bool,
) -> str:
    """稠密 IC∧ICIR 硬门失败原因（供 emerging 识别 / 日志）。"""
    ic_lab = "纯IC" if use_pure else "IC"
    icir_lab = "纯ICIR" if use_pure else "ICIR"
    parts: list[str] = []
    if not np.isfinite(effective_ic) or effective_ic < ic_threshold:
        ic_s = f"{effective_ic:.4f}" if np.isfinite(effective_ic) else "NaN"
        parts.append(f"{ic_lab}={ic_s}<{ic_threshold}")
    if not np.isfinite(effective_icir) or effective_icir < icir_threshold:
        ir_s = f"{effective_icir:.4f}" if np.isfinite(effective_icir) else "NaN"
        parts.append(f"{icir_lab}={ir_s}<{icir_threshold}")
    body = ", ".join(parts) if parts else "IC/ICIR 未过线"
    return f"{body}（需 |IC|∧|ICIR| 同时过线）"


def _long_share_gate_reason(
    row: pd.Series,
    min_long_share: float | None,
) -> str | None:
    """稠密 long_share 门：要求符号对齐后 ``long_share > min``。

    ``min_long_share`` 为 None / ≤0 时关闭。缺失/NaN 在门开启时剔除并提示
    重跑 ``--barra``（分位分解默认开，已符号对齐）或 ``--long-share-csv``。
    """
    if min_long_share is None or float(min_long_share) <= 0:
        return None
    thr = float(min_long_share)
    raw = row.get("long_share", np.nan) if hasattr(row, "get") else np.nan
    try:
        ls = float(raw)
    except (TypeError, ValueError):
        ls = float("nan")
    if not np.isfinite(ls):
        return (
            f"long_share 缺失/NaN（需符号对齐分位分解：重跑 --barra 或 "
            f"--long-share-csv；门槛 long_share>{thr}）"
        )
    if not (ls > thr):
        return (
            f"long_share={ls:.4f}<={thr}"
            f"（需 long_share>{thr}，符号对齐后）"
        )
    return None


def overlay_long_share(
    summary_df: pd.DataFrame,
    *,
    quantile_df: pd.DataFrame | None = None,
    long_share_csv: str | None = None,
) -> pd.DataFrame:
    """把符号对齐后的 ``long_share``（及可选多空列）并入 summary。

    优先级：``long_share_csv``（显式 aligned 覆盖）> ``quantile_df`` >
    summary 已有列。历史未对齐 CSV 须重跑 ``--barra`` 或传 aligned csv。
    """
    out = summary_df.copy()
    q_cols = ("多头超额", "空头贡献", "long_share", "多空来源")

    if quantile_df is not None and not quantile_df.empty:
        join_cols = [c for c in q_cols if c in quantile_df.columns]
        if join_cols:
            # 先去掉旧列再 join，避免同名后缀
            drop = [c for c in join_cols if c in out.columns]
            if drop:
                out = out.drop(columns=drop)
            out = out.join(quantile_df[join_cols], how="left")

    if long_share_csv:
        from pathlib import Path

        path = Path(long_share_csv)
        if not path.exists():
            raise FileNotFoundError(f"long_share CSV 不存在: {path}")
        ls_df = pd.read_csv(path, encoding="utf-8-sig")
        if ls_df.empty:
            raise ValueError(f"long_share CSV 为空: {path}")
        # 因子名列：优先「因子」，否则首列
        name_col = "因子" if "因子" in ls_df.columns else ls_df.columns[0]
        if "long_share" not in ls_df.columns:
            raise ValueError(
                f"long_share CSV 缺 long_share 列: {path}；列={list(ls_df.columns)}"
            )
        overlay = ls_df.set_index(name_col)[["long_share"]].copy()
        # 可选一并覆盖多空诊断列
        for c in ("多头超额", "空头贡献", "多空来源"):
            if c in ls_df.columns:
                overlay[c] = ls_df.set_index(name_col)[c]
        drop = [c for c in overlay.columns if c in out.columns]
        if drop:
            out = out.drop(columns=drop)
        out = out.join(overlay, how="left")
        n_hit = int(out["long_share"].reindex(overlay.index).notna().sum()) if "long_share" in out.columns else 0
        print(
            f"  [long_share] 已从 CSV 覆盖对齐列: {path.name} "
            f"（命中 {n_hit}/{len(overlay)}）"
        )

    return out


def _is_ic_icir_gate_exclusion(reason: str) -> bool:
    """是否为稠密 IC/ICIR 硬门剔除（而非 t/FDR/corr/long_share 等）。"""
    if not reason:
        return False
    if any(
        k in reason
        for k in (
            "t=", "NW_t", "FDR", "同向年份", "滚动ICIR", "最差12期", "相关",
            "long_share",
        )
    ):
        return False
    return any(
        k in reason
        for k in ("IC=", "纯IC=", "ICIR=", "纯ICIR=", "|IC|∧|ICIR|")
    )


def _lookback_periods(hold_period: int, months: int) -> int:
    """日历月 → 调仓期数（与 Walk-Forward 同口径）。"""
    from models.trainer import months_to_rebalance_periods
    from utils.rebalance_dates import horizon_to_rebalance_freq

    freq = horizon_to_rebalance_freq(int(hold_period))
    return int(months_to_rebalance_periods(int(months), freq))


def _truncate_ic_for_emerging(
    ic: pd.Series | None,
    *,
    asof: pd.Timestamp | None = None,
    holdout_periods: int = 0,
) -> pd.Series | None:
    """新兴近窗统计截断：先按 asof，再去掉末尾 holdout 期（防评效段双重偷看）。"""
    if ic is None:
        return None
    s = ic
    if asof is not None:
        ts = pd.Timestamp(asof)
        s = s.loc[s.index <= ts]
    if holdout_periods > 0 and len(s) > holdout_periods:
        s = s.iloc[:-int(holdout_periods)]
    return s


def evaluate_decay_label(
    ic: pd.Series | None,
    *,
    recent_periods: int,
    retention_min: float = IC_DECAY_RETENTION_MIN,
    recent_icir_max: float = IC_DECAY_RECENT_ICIR_MAX,
    recent_ic_max: float = IC_DECAY_RECENT_IC_MAX,
) -> tuple[bool, dict[str, float]]:
    """同持仓期 IC 序列衰减标注（**不剔除**）。

    衰减 iff
    ``(R < retention_min ∧ |ICIR_recent| < recent_icir_max) ∧ |IC_recent| < recent_ic_max``
    （合取 AND；``R = |ICIR_recent|/|ICIR_past|``）。
    """
    stats = recent_past_icir_retention(ic, recent_periods)
    r = stats["retention"]
    ricir = stats["icir_recent"]
    ric = stats.get("ic_recent", np.nan)
    if not (np.isfinite(r) and np.isfinite(ricir) and np.isfinite(ric)):
        return False, stats
    decayed = (
        (r < retention_min)
        and (abs(float(ricir)) < recent_icir_max)
        and (abs(float(ric)) < recent_ic_max)
    )
    return decayed, stats


def evaluate_decay_gate(
    name: str,
    row: pd.Series,
    ic: pd.Series | None,
    *,
    decay_table: pd.DataFrame | None = None,
    half_life_min: float | None = None,
    short_long_min: float | None = None,
    residual_icir: float = 0.25,
    residual_ic: float = 0.015,
    pure_ic_means: dict | None = None,
    recent_periods: int = 52,
    retention_min: float = IC_DECAY_RETENTION_MIN,
    recent_icir_max: float = IC_DECAY_RECENT_ICIR_MAX,
    recent_ic_max: float = IC_DECAY_RECENT_IC_MAX,
) -> tuple[str | None, str | None]:
    """兼容旧接口：改为**仅标注**，永不返回 exclusion。

    Returns
    -------
    (CAT_DECAYED or None, None)
    """
    del name, row, decay_table, half_life_min, short_long_min
    del residual_icir, residual_ic, pure_ic_means
    decayed, _ = evaluate_decay_label(
        ic,
        recent_periods=recent_periods,
        retention_min=retention_min,
        recent_icir_max=recent_icir_max,
        recent_ic_max=recent_ic_max,
    )
    return (CAT_DECAYED if decayed else None), None


def _resolve_window_ic(
    name: str,
    all_ic: dict | None,
    pure_ic_series: dict | None,
    *,
    prefer_pure_only: bool = False,
) -> pd.Series | None:
    """近窗评估用的 IC 序列：有 Barra pure 时优先，与全样本纯因子同口径。

    ``prefer_pure_only=True``（新兴在 barra 模式下）：无该因子 pure 序列则返回
    None，**禁止**用 raw 近窗救援。无 barra / pure 字典为空时回退 raw。
    """
    has_pure_pool = bool(pure_ic_series)
    if has_pure_pool:
        s = pure_ic_series.get(name)
        if s is not None:
            s2 = s.dropna() if hasattr(s, "dropna") else s
            if len(s2) >= 3:
                return s
        if prefer_pure_only:
            return None
    if prefer_pure_only and has_pure_pool:
        return None
    if all_ic:
        return all_ic.get(name)
    return None


def _dedup_emerging_by_ic_corr(
    candidates: list[str],
    protected: list[str],
    ic_lookup: dict,
    *,
    corr_threshold: float = 0.70,
    score: dict[str, float] | None = None,
) -> tuple[list[str], dict]:
    """新兴名单 corr-dedup（只影响 emerging，不改 dense_kept）。

    方案（写入手册）：
    1. ``protected`` = 已入选稠密主池，始终保留、不淘汰；
    2. 新兴候选按 |score|（默认近窗 |ICIR|）降序；
    3. 若与 protected 或已选 emerging 的 IC 序列 |corr| > threshold → 丢弃该新兴；
    4. 从而在 emerging 内部 + 相对主池去重，避免市值簇重复标注一堆。
    """
    exclusions: dict = {}
    if not candidates:
        return [], exclusions
    if not corr_threshold or corr_threshold <= 0 or len(candidates) <= 1 and not protected:
        return list(candidates), exclusions

    names = list(dict.fromkeys([*protected, *candidates]))
    frame = pd.DataFrame({n: ic_lookup.get(n) for n in names if ic_lookup.get(n) is not None})
    if frame.shape[1] < 2:
        return list(candidates), exclusions
    corr_mat = frame.corr(method="pearson")

    scores = score or {}
    order = sorted(
        candidates,
        key=lambda n: abs(float(scores.get(n, 0.0))),
        reverse=True,
    )
    kept_emerging: list[str] = []
    kept_all = list(protected)
    for name in order:
        if name not in corr_mat.index:
            kept_emerging.append(name)
            kept_all.append(name)
            continue
        drop = False
        for k in kept_all:
            if k not in corr_mat.columns:
                continue
            c = abs(float(corr_mat.loc[name, k]))
            if np.isfinite(c) and c > corr_threshold:
                exclusions[name] = (
                    f"新兴观察相关去重: 与{k} IC序列相关={c:.2f}>{corr_threshold}"
                )
                drop = True
                break
        if not drop:
            kept_emerging.append(name)
            kept_all.append(name)
    return kept_emerging, exclusions


def _dedup_sparse_by_corr(
    candidates: list[str],
    summary_df: pd.DataFrame,
    *,
    factor_registry=None,
    all_ic: dict | None = None,
    corr_threshold: float = IC_SPARSE_CORR_THRESHOLD,
    corr_method: str | None = None,
    sample_step: int = 20,
) -> tuple[list[str], dict]:
    """稀疏名单 corr-dedup（只影响 ``sparse_kept`` / ``factors_sparse``）。

    相关度量（优先与稠密一致；见操作手册「稀疏 corr-dedup」）：
    1. 有 ``factor_registry``：截面 Spearman + ``corr_method`` 聚合
       （复用 ``_pairwise_corr_matrix`` / ``_aggregate_corr``，与 dense 同路径）；
    2. 截面矩阵为空或不稳定（稀疏触发导致有效截面不足 / 采样日无重叠）→
       fallback 到 IC 序列 Pearson 相关（与 emerging / ``select_factors_raw`` 同口径）。
       选用 fallback 的原因：稀疏事件日截面重叠常 < ``min_periods``，硬套截面会
       静默跳过去重；IC 序列相关仍能刻画信号时序冗余。

    保留规则：按 |ICIR| 降序迭代（与 dense「更高 |ICIR|」对齐）；
    |corr| > threshold 时剔除较弱者，理由写入 exclusions。
    """
    exclusions: dict = {}
    if not candidates or not corr_threshold or corr_threshold <= 0:
        return list(candidates), exclusions
    if len(candidates) <= 1:
        return list(candidates), exclusions

    method = corr_method or IC_CORR_METHOD
    agg_corr: pd.DataFrame | None = None
    metric_label = ""

    # ── 优先：截面 Spearman（与 dense select_factors 同逻辑）──
    if factor_registry is not None:
        try:
            first = (
                factor_registry.get(candidates[0])
                if hasattr(factor_registry, "get")
                else factor_registry[candidates[0]]
            )
        except (KeyError, TypeError):
            first = None
        if first is not None:
            sample_dates = list(first.index[::sample_step])
            del first
            if hasattr(factor_registry, "release_cache"):
                factor_registry.release_cache()
            corr_list = _pairwise_corr_matrix(
                factor_registry, sample_dates, candidates=candidates,
            )
            if corr_list:
                agg_corr = _aggregate_corr(corr_list, method)
                metric_label = f"截面相关({method})"
            if hasattr(factor_registry, "release_cache"):
                factor_registry.release_cache()

    # ── Fallback：IC 序列相关（稀疏截面不稳定时）──
    if agg_corr is None or agg_corr.empty:
        if not all_ic:
            return list(candidates), exclusions
        ic_frame = pd.DataFrame(
            {n: all_ic.get(n) for n in candidates if all_ic.get(n) is not None}
        )
        if ic_frame.shape[1] < 2:
            return list(candidates), exclusions
        agg_corr = ic_frame.corr(method="pearson")
        metric_label = "IC序列相关"

    icir_order = (
        summary_df.reindex(candidates)["ICIR"]
        .abs()
        .sort_values(ascending=False)
        .index.tolist()
    )
    # reindex 可能引入 NaN 索引外的名；保序补全未出现在 summary 的候选
    seen = set(icir_order)
    for n in candidates:
        if n not in seen:
            icir_order.append(n)

    kept: list[str] = []
    for name in icir_order:
        if name not in agg_corr.index:
            kept.append(name)
            continue
        drop = False
        for k in kept:
            if k not in agg_corr.columns:
                continue
            c = abs(float(agg_corr.loc[name, k]))
            if np.isfinite(c) and c > corr_threshold:
                icir_name = abs(float(summary_df.loc[name, "ICIR"])) if name in summary_df.index else 0.0
                icir_k = abs(float(summary_df.loc[k, "ICIR"])) if k in summary_df.index else 0.0
                exclusions[name] = (
                    f"稀疏相关去重: 与{k}{metric_label}={c:.2f}>{corr_threshold}，"
                    f"ICIR({icir_name:.3f})<ICIR({icir_k:.3f})"
                )
                drop = True
                break
        if not drop:
            kept.append(name)
    return kept, exclusions


def evaluate_style_reversal(
    ic: pd.Series | None,
    *,
    quarter_periods: int,
    frac_min: float = IC_REVERSAL_FRAC,
    abs_ic_min: float = IC_REVERSAL_ABS_IC,
) -> tuple[bool, float]:
    """最近一季风格逆转标注（**不剔除**）。

    最近 ``quarter_periods`` 期内，
    ``mean 1{ sign(IC_t) != sign(mean_IC) AND |IC_t| > abs_ic_min } > frac_min``。
    """
    frac = style_reversal_fraction(ic, quarter_periods, abs_ic_min=abs_ic_min)
    if not np.isfinite(frac):
        return False, frac
    return bool(frac > frac_min), float(frac)


def _compute_emerging_recent_fdr(
    factor_names,
    *,
    all_ic: dict | None,
    pure_ic_series: dict | None,
    prefer_pure_only: bool,
    lookback: int,
    alpha: float,
) -> dict[str, bool]:
    """近窗 NW-t 的 BH-FDR 显著性。

    校正域 = ``factor_names`` 中全部有可用近窗 IC 且 NW-t 有限的因子
    （生产传入稠密轨全体被测因子，而非仅新兴候选子集）。
    """
    names: list[str] = []
    t_vals: list[float] = []
    for name in factor_names:
        ic = _resolve_window_ic(
            name, all_ic, pure_ic_series, prefer_pure_only=prefer_pure_only,
        )
        if ic is None or lookback <= 0:
            continue
        s = ic.dropna()
        if len(s) < 3:
            continue
        tail = s.iloc[-lookback:] if len(s) >= lookback else s
        if len(tail) < 3:
            continue
        t = newey_west_t(tail)
        if not np.isfinite(t):
            continue
        names.append(str(name))
        t_vals.append(float(t))
    if not names:
        return {}
    sig = benjamini_hochberg(np.asarray(t_vals, dtype=float), alpha=alpha)
    return {n: bool(flag) for n, flag in zip(names, sig)}


def evaluate_emerging(
    name: str,
    row: pd.Series,
    ic: pd.Series | None,
    *,
    lookback: int,
    recent_icir_min: float = IC_EMERGING_RECENT_ICIR,
    recent_ic_min: float = IC_EMERGING_RECENT_IC,
    ic_threshold: float = IC_THRESHOLD,
    icir_threshold: float = ICIR_THRESHOLD,
    pure_ic_means: dict | None = None,
    pure_ic_series: dict | None = None,
    recent_fdr_sig: bool = False,
    lift_min: float = IC_EMERGING_LIFT_MIN,
    require_trend: bool = IC_EMERGING_REQUIRE_TREND,
    trend_segment_periods: int = 0,
    trend_segments: int = IC_EMERGING_TREND_SEGMENTS,
    trend_eps: float = IC_EMERGING_TREND_EPS,
) -> bool:
    """全样本未过 IC∧ICIR 稠密门 + 近窗 FDR∧ICIR∧lift[∧趋势] → 新兴（仅标注）。

    ``ic`` 须为近窗评估序列（调用方已按 asof/holdout 截断）：有 ``--barra`` 时传
    **pure IC 序列**；禁止传 raw 近窗做救援。
    ``lookback`` 为 **IC 期数**（调用方须先把日历月经 ``_lookback_periods`` 换算）。

    判定（合取）::

        全样本: 未同时满足 |IC_eff|≥ic_th ∧ |ICIR_eff|≥icir_th
        近窗:   recent_fdr_sig
                ∧ |ICIR_recent| ≥ recent_icir_min
                ∧ |ICIR_recent| / |ICIR_past| ≥ lift_min   （lift_min≤0 则跳过）
                ∧ [可选] 近 N 段 |ICIR| 单调不降且末段>首段

    ``recent_fdr_sig`` 须由调用方在**全体被测因子**近窗 NW-t 上做 BH-FDR 后传入。
    ``recent_ic_min`` 已弃用（不再用仅 |IC_recent| 过线的松 OR）。
    """
    del recent_ic_min  # deprecated: old OR on |IC_recent| removed
    if ic is None or lookback <= 0:
        return False
    if not recent_fdr_sig:
        return False
    eff_ic, eff_icir = _effective_ic_icir(
        row, name, pure_ic_means, pure_ic_series,
    )
    # 已过稠密 |IC|∧|ICIR| 硬门 → 不必标新兴
    if (
        np.isfinite(eff_ic) and eff_ic >= ic_threshold
        and np.isfinite(eff_icir) and eff_icir >= icir_threshold
    ):
        return False
    ret = recent_past_icir_retention(ic, lookback)
    ricir = abs(float(ret["icir_recent"])) if np.isfinite(ret["icir_recent"]) else 0.0
    if ricir < recent_icir_min:
        return False
    if lift_min > 0:
        lift = float(ret["retention"]) if np.isfinite(ret["retention"]) else np.nan
        if not (np.isfinite(lift) and lift >= lift_min):
            return False
    if require_trend and trend_segment_periods > 0:
        ok, _ = segment_metric_trend(
            ic,
            trend_segment_periods,
            n_segments=trend_segments,
            metric="icir",
            eps=trend_eps,
        )
        if not ok:
            return False
    return True


def _append_label(result: FactorSelectionResult, name: str, label: str) -> None:
    labs = result.labels.setdefault(name, [])
    if label not in labs:
        labs.append(label)


def select_sparse_factors(
    summary_df: pd.DataFrame,
    all_ic: dict | None,
    *,
    ic_threshold: float = IC_SPARSE_IC_THRESHOLD,
    icir_threshold: float = IC_SPARSE_ICIR_THRESHOLD,
    win_rate_min: float = IC_SPARSE_WIN_RATE_MIN,
    payoff_min: float = IC_SPARSE_PAYOFF_MIN,
    pure_ic_means: dict | None = None,
    sparse_names: frozenset[str] | None = None,
    factor_registry=None,
    forward_return: pd.DataFrame | None = None,
    rebalance_dates: pd.DatetimeIndex | None = None,
    payoff_hits: dict[str, float] | None = None,
    require_ic: bool = False,
    tradable: pd.DataFrame | None = None,
    corr_dedup: bool = True,
    corr_threshold: float = IC_SPARSE_CORR_THRESHOLD,
    corr_method: str | None = None,
    sample_step: int = 20,
    # deprecated kwargs kept for call-site compatibility (ignored)
    t_threshold: float | None = None,
    nw_t_threshold: float | None = None,
    use_fdr: bool = False,
    fdr_alpha: float = 0.05,
) -> tuple[list[str], dict]:
    """稀疏因子独立轨道。

    硬门槛（主）— 与全样本 IC 符号 ``s = sign(mean_IC)`` 对齐
    ----------------------------------------------------------
    - 同向 IC 胜率 ≥ ``win_rate_min``（默认 0.56）：
      ``mean(sign(IC_t) == s)``（见 summary「胜率」/ :func:`win_rates`）
    - ``payoff_hit`` ≥ ``payoff_min``（默认 0.55）：
      触发侧 ``f*s > 0``，``mean_t 1{ mean(y|f*s>0) > mean(y) }``
      （见 :func:`trigger_cs_payoff`；``tradable`` 与 IC 同口径）
    - 胜率与 payoff 均在**同一套 IC 有效交易日**上评估（``ic_series`` 索引），
      **不用** ``rebalance_dates``
    - ``s≈0``：无法确定方向 → **跳过稀疏门**（剔除并注明）

    软参考
    ------
    - IC/ICIR：默认不硬剔（``require_ic=False``）；仅当 ``require_ic=True`` 时
      要求 |IC|≥阈值 或 |ICIR|≥阈值。

    相关去重（``corr_dedup=True``，默认阈值 ``IC_SPARSE_CORR_THRESHOLD=0.70``）
    ----------------------------------------------------------------
    - 胜率/payoff 过线后，候选内部 corr-dedup（见 :func:`_dedup_sparse_by_corr`）
    - 优先截面 Spearman+method；不稳定时 fallback IC 序列相关
    - 冲突保留更高 |ICIR| 者

    明确不做
    --------
    - 无 t / NW-t / FDR 要求（稀疏事件因子上这些检验不合理）。
    """
    # rebalance_dates 曾误用于 payoff；保留参数以免旧调用方报错
    del t_threshold, nw_t_threshold, use_fdr, fdr_alpha, rebalance_dates
    pool = sparse_names if sparse_names is not None else SPARSE_FACTOR_NAMES
    exclusions: dict = {}
    kept: list[str] = []

    sub = summary_df.loc[[n for n in summary_df.index if n in pool]]
    for name in sub.index:
        row = sub.loc[name]
        eff_ic, eff_icir = _effective_ic_icir(row, name, pure_ic_means)
        if np.isnan(eff_ic) or np.isnan(eff_icir):
            exclusions[name] = "稀疏轨: IC/ICIR 为 NaN"
            continue
        if require_ic and eff_ic < ic_threshold and eff_icir < icir_threshold:
            exclusions[name] = (
                f"稀疏轨: IC={eff_ic:.4f}<{ic_threshold} 且 "
                f"ICIR={eff_icir:.4f}<{icir_threshold}（require_ic）"
            )
            continue

        # 全样本 IC 方向；≈0 时跳过稀疏门
        mean_ic = float(row.get("IC均值", np.nan))
        if (not np.isfinite(mean_ic)) and all_ic and name in all_ic:
            ic_s = all_ic[name].dropna()
            mean_ic = float(ic_s.mean()) if len(ic_s) else np.nan
        direction = ic_direction_sign(mean_ic)
        if direction == 0.0:
            exclusions[name] = (
                "稀疏轨: 全样本 IC 均值≈0，无法确定方向，跳过稀疏门"
            )
            continue

        # 胜率硬门槛：与 mean_IC 同向的 IC 期占比（非盲目 IC>0）
        wr = float(row.get("胜率", np.nan))
        if not np.isfinite(wr) and all_ic and name in all_ic:
            wr, _, _ = win_rates(all_ic[name])
        if not np.isfinite(wr) or wr < win_rate_min:
            exclusions[name] = (
                f"稀疏轨: 同向IC胜率={wr if np.isfinite(wr) else float('nan'):.2f}"
                f"<{win_rate_min}"
            )
            continue

        # 触发日相对截面均值胜率（主「盈亏」门槛；触发侧 f*s>0）
        # 日期 = 该因子 IC 有效日（与胜率同一索引）；禁止只在调仓日上算
        hit = np.nan
        if payoff_hits is not None and name in payoff_hits:
            hit = float(payoff_hits[name])
        elif factor_registry is not None and forward_return is not None:
            try:
                panel = (
                    factor_registry.get(name)
                    if hasattr(factor_registry, "get")
                    else factor_registry[name]
                )
            except (KeyError, TypeError):
                panel = None
            if panel is not None:
                ic_dates = None
                if all_ic and name in all_ic and all_ic[name] is not None:
                    ic_s = all_ic[name].dropna()
                    if len(ic_s) > 0:
                        ic_dates = ic_s.index
                stats = trigger_cs_payoff(
                    panel, forward_return,
                    dates=ic_dates,
                    direction=direction,
                    tradable=tradable,
                )
                hit = stats["payoff_hit"]
                if hasattr(factor_registry, "release_cache"):
                    factor_registry.release_cache()
        if not np.isfinite(hit):
            exclusions[name] = (
                "稀疏轨: 无法计算触发日截面胜率（缺因子面板/收益或触发日不足）"
            )
            continue
        if hit < payoff_min:
            exclusions[name] = (
                f"稀疏轨: 触发日截面胜率={hit:.2f}<{payoff_min}"
            )
            continue
        kept.append(name)

    if corr_dedup and len(kept) > 1:
        kept, dedup_ex = _dedup_sparse_by_corr(
            kept,
            summary_df,
            factor_registry=factor_registry,
            all_ic=all_ic,
            corr_threshold=corr_threshold,
            corr_method=corr_method,
            sample_step=sample_step,
        )
        exclusions.update(dedup_ex)

    return kept, exclusions


def select_factors_multi_track(
    summary_df: pd.DataFrame,
    factor_registry=None,
    all_ic: dict | None = None,
    pure_ic_means: dict | None = None,
    pure_ic_series: dict | None = None,
    ic_threshold: float = IC_THRESHOLD,
    icir_threshold: float = ICIR_THRESHOLD,
    t_threshold: float = 2.5,
    nw_t_threshold: float | None = None,
    corr_threshold: float = 0.70,
    sample_step: int = 20,
    corr_method: str | None = None,
    rebalance_dates: pd.DatetimeIndex | None = None,
    use_fdr: bool = False,
    fdr_alpha: float = 0.05,
    regime_consistency_threshold: float | None = None,
    rolling_icir_threshold: float | None = None,
    worst_period_ic_threshold: float | None = None,
    corr_dedup: bool = True,
    *,
    raw_mode: bool = False,
    hold_period: int = 20,
    min_long_share: float | None = IC_MIN_LONG_SHARE,
    enable_decay_gate: bool = True,
    decay_recent_months: int = IC_DECAY_RECENT_MONTHS,
    decay_retention_min: float = IC_DECAY_RETENTION_MIN,
    decay_retention_min_sparse: float = IC_DECAY_RETENTION_MIN_SPARSE,
    decay_recent_icir_max: float = IC_DECAY_RECENT_ICIR_MAX,
    decay_recent_ic_max: float = IC_DECAY_RECENT_IC_MAX,
    enable_reversal_label: bool = True,
    reversal_months: int = IC_REVERSAL_MONTHS,
    reversal_frac: float = IC_REVERSAL_FRAC,
    reversal_abs_ic: float = IC_REVERSAL_ABS_IC,
    # deprecated half-life kwargs (ignored)
    decay_half_life_min: float | None = None,
    decay_short_long_min: float | None = None,
    decay_residual_icir: float = 0.25,
    decay_residual_ic: float = 0.015,
    decay_table: pd.DataFrame | None = None,
    enable_emerging: bool = True,
    emerging_lookback: int = IC_EMERGING_LOOKBACK,  # 日历月；内部换算为 IC 期数
    emerging_recent_icir: float = IC_EMERGING_RECENT_ICIR,
    emerging_recent_ic: float = IC_EMERGING_RECENT_IC,  # deprecated no-op
    emerging_fdr_alpha: float = IC_EMERGING_FDR_ALPHA,
    emerging_lift_min: float = IC_EMERGING_LIFT_MIN,
    emerging_holdout_months: int = IC_EMERGING_HOLDOUT_MONTHS,
    emerging_asof: pd.Timestamp | str | None = None,
    emerging_require_trend: bool = IC_EMERGING_REQUIRE_TREND,
    emerging_trend_months: int = IC_EMERGING_TREND_MONTHS,
    emerging_trend_segments: int = IC_EMERGING_TREND_SEGMENTS,
    emerging_trend_eps: float = IC_EMERGING_TREND_EPS,
    enable_sparse_track: bool = True,
    sparse_ic_threshold: float = IC_SPARSE_IC_THRESHOLD,
    sparse_icir_threshold: float = IC_SPARSE_ICIR_THRESHOLD,
    sparse_t_threshold: float | None = None,
    sparse_win_rate_min: float = IC_SPARSE_WIN_RATE_MIN,
    sparse_payoff_min: float = IC_SPARSE_PAYOFF_MIN,
    sparse_require_ic: bool = False,
    sparse_corr_threshold: float = IC_SPARSE_CORR_THRESHOLD,
    forward_return: pd.DataFrame | None = None,
    payoff_hits: dict[str, float] | None = None,
    tradable: pd.DataFrame | None = None,
) -> FactorSelectionResult:
    """稠密 + 稀疏双轨筛选，并标注类别 / 警示标签。

    主类别
    ------
    - 普通因子 → ``dense_kept``（ML YAML ``factors``）
    - 新兴因子 → ``emerging_kept``（``factors_emerging``；**不**进 dense）
    - 稀疏因子 → ``sparse_kept``（不进 dense）

    警示标签（``labels``，可叠加，**不剔除**）
    ----------------------------------------
    - 衰减因子：ICIR 保留率塌缩 ∧ 近期 ICIR 弱 ∧ 近期 |IC| 弱（合取）
    - 风格逆转：近一季多数强 IC 与全样本符号相反

    近窗口径：有 ``pure_ic_series``（``--barra``）时，新兴/衰减/逆转均用 pure
    序列；新兴禁止 raw 近窗救援。新兴近窗另做 NW-t BH-FDR（校正域=稠密全体被测因子）。
    稠密硬门：|pure IC|∧|pure ICIR|（合取）[+ t/FDR] → long_share（默认>0.4）
    → corr-dedup；无 pure 时退回 raw 仍用 AND。
    """
    del decay_half_life_min, decay_short_long_min, decay_residual_icir
    del decay_residual_ic, decay_table, sparse_t_threshold
    del emerging_recent_ic  # deprecated: old |IC_recent| OR gate removed

    result = FactorSelectionResult()
    dense_names, sparse_names = partition_sparse(list(summary_df.index))
    dense_df = summary_df.loc[dense_names] if dense_names else summary_df.iloc[0:0]
    barra_pure_mode = bool(pure_ic_series)

    if (
        min_long_share is not None
        and float(min_long_share) > 0
        and "long_share" not in summary_df.columns
    ):
        print(
            f"  [warn] min_long_share={min_long_share} 但 summary 无 long_share 列："
            "稠密候选将因缺失被剔除。请重跑 --barra（分位分解默认开，符号对齐）"
            "或传 --long-share-csv 合并 aligned CSV。"
        )

    recent_periods = _lookback_periods(hold_period, decay_recent_months)
    quarter_periods = _lookback_periods(hold_period, reversal_months)
    # 新兴窗口：日历月 → IC 期数（与衰减同口径；勿把 52 当 raw periods）
    emerging_periods = (
        _lookback_periods(hold_period, emerging_lookback)
        if emerging_lookback > 0 else 0
    )
    emerging_holdout_periods = (
        _lookback_periods(hold_period, emerging_holdout_months)
        if emerging_holdout_months > 0 else 0
    )
    emerging_asof_ts = (
        pd.Timestamp(emerging_asof) if emerging_asof is not None else None
    )
    trend_seg_periods = (
        _lookback_periods(hold_period, emerging_trend_months)
        if emerging_require_trend and emerging_trend_months > 0 else 0
    )

    # ── 稀疏轨道（先跑；不参与稠密 corr；自身 corr-dedup）──
    if enable_sparse_track and sparse_names:
        sk, sex = select_sparse_factors(
            summary_df,
            all_ic,
            ic_threshold=sparse_ic_threshold,
            icir_threshold=sparse_icir_threshold,
            win_rate_min=sparse_win_rate_min,
            payoff_min=sparse_payoff_min,
            pure_ic_means=pure_ic_means,
            sparse_names=frozenset(sparse_names),
            factor_registry=factor_registry,
            forward_return=forward_return,
            payoff_hits=payoff_hits,
            require_ic=sparse_require_ic,
            tradable=tradable,
            corr_dedup=corr_dedup,
            corr_threshold=sparse_corr_threshold,
            corr_method=corr_method,
            sample_step=sample_step,
        )
        result.sparse_kept = sk
        result.exclusions.update(sex)
        for n in sk:
            result.categories[n] = CAT_SPARSE
    elif sparse_names:
        for n in sparse_names:
            result.exclusions[n] = "稀疏轨已关闭（--no-sparse-track）"

    if dense_df.empty:
        # 仍可为稀疏入选者打衰减/逆转标签
        _apply_caution_labels(
            result,
            all_ic,
            pure_ic_series=pure_ic_series,
            names=result.sparse_kept,
            enable_decay=enable_decay_gate,
            recent_periods=recent_periods,
            retention_min=decay_retention_min_sparse,
            recent_icir_max=decay_recent_icir_max,
            recent_ic_max=decay_recent_ic_max,
            enable_reversal=enable_reversal_label,
            quarter_periods=quarter_periods,
            reversal_frac=reversal_frac,
            reversal_abs_ic=reversal_abs_ic,
        )
        return result

    # ── 稠密基础门（与 select_factors / raw 同口径，仅稠密子集）──
    if raw_mode:
        base_kept, base_ex = select_factors_raw(
            dense_df,
            all_ic=all_ic or {},
            pure_ic_means=pure_ic_means,
            pure_ic_series=pure_ic_series,
            ic_threshold=ic_threshold,
            icir_threshold=icir_threshold,
            t_threshold=t_threshold,
            nw_t_threshold=nw_t_threshold,
            corr_threshold=corr_threshold,
            use_fdr=use_fdr,
            fdr_alpha=fdr_alpha,
            regime_consistency_threshold=regime_consistency_threshold,
            rolling_icir_threshold=rolling_icir_threshold,
            worst_period_ic_threshold=worst_period_ic_threshold,
            corr_dedup=corr_dedup,
            min_long_share=min_long_share,
        )
    else:
        if factor_registry is None:
            raise ValueError("select_factors_multi_track: 非 raw_mode 需要 factor_registry")
        base_kept, base_ex = select_factors(
            dense_df,
            factor_registry,
            pure_ic_means=pure_ic_means,
            pure_ic_series=pure_ic_series,
            ic_threshold=ic_threshold,
            icir_threshold=icir_threshold,
            t_threshold=t_threshold,
            nw_t_threshold=nw_t_threshold,
            corr_threshold=corr_threshold,
            sample_step=sample_step,
            corr_method=corr_method,
            rebalance_dates=rebalance_dates,
            use_fdr=use_fdr,
            fdr_alpha=fdr_alpha,
            regime_consistency_threshold=regime_consistency_threshold,
            rolling_icir_threshold=rolling_icir_threshold,
            worst_period_ic_threshold=worst_period_ic_threshold,
            corr_dedup=corr_dedup,
            min_long_share=min_long_share,
        )
    result.exclusions.update(base_ex)

    # ── 稠密入选（不含新兴）──
    dense_final: list[str] = list(base_kept)
    for name in base_kept:
        result.categories.setdefault(name, CAT_DENSE)
    result.dense_kept = dense_final

    # ── 新兴：仅标注观察，不进 dense_kept / factors ──
    # 近窗 BH-FDR 校正域 = 稠密轨全体被测因子（有近窗 IC 者），再与「全样本未过
    # IC∧ICIR 稠密门」合取；corr-dedup：emerging 内部 + 相对 dense_kept。
    # 近窗序列先按 asof / holdout 截断，避免用评效段定池双重偷看。
    emerging_cand: list[str] = []
    emerging_scores: dict[str, float] = {}
    if enable_emerging and emerging_periods > 0 and (all_ic or pure_ic_series):
        def _emerging_ic(name: str) -> pd.Series | None:
            ic0 = _resolve_window_ic(
                name, all_ic, pure_ic_series, prefer_pure_only=barra_pure_mode,
            )
            return _truncate_ic_for_emerging(
                ic0,
                asof=emerging_asof_ts,
                holdout_periods=emerging_holdout_periods,
            )

        # FDR 域也用截断后序列（与 emerging 判定同截止）；禁止 raw 救援
        trunc_lookup: dict = {
            n: s for n in dense_df.index
            if (s := _emerging_ic(n)) is not None
        }
        emerging_fdr = _compute_emerging_recent_fdr(
            list(dense_df.index),
            all_ic=None if barra_pure_mode else trunc_lookup,
            pure_ic_series=trunc_lookup if barra_pure_mode else None,
            prefer_pure_only=barra_pure_mode,
            lookback=emerging_periods,
            alpha=emerging_fdr_alpha,
        )
        for name, reason in list(base_ex.items()):
            if name not in dense_df.index:
                continue
            if not _is_ic_icir_gate_exclusion(reason):
                continue
            # barra 模式：近窗必须用 pure；无 pure 序列则跳过（禁止 raw 救援）
            ic_win = _emerging_ic(name)
            if ic_win is None:
                continue
            row = dense_df.loc[name]
            if evaluate_emerging(
                name, row, ic_win,
                lookback=emerging_periods,
                recent_icir_min=emerging_recent_icir,
                ic_threshold=ic_threshold,
                icir_threshold=icir_threshold,
                pure_ic_means=pure_ic_means,
                pure_ic_series=pure_ic_series,
                recent_fdr_sig=emerging_fdr.get(name, False),
                lift_min=emerging_lift_min,
                require_trend=emerging_require_trend,
                trend_segment_periods=trend_seg_periods,
                trend_segments=emerging_trend_segments,
                trend_eps=emerging_trend_eps,
            ):
                emerging_cand.append(name)
                st = recent_past_icir_retention(ic_win, emerging_periods)
                ricir = (
                    abs(float(st["icir_recent"]))
                    if np.isfinite(st["icir_recent"]) else 0.0
                )
                emerging_scores[name] = ricir

        ic_lookup: dict = {}
        for n in list(dense_final) + emerging_cand:
            s = _resolve_window_ic(
                n, all_ic, pure_ic_series, prefer_pure_only=False,
            )
            if s is not None:
                ic_lookup[n] = s
        if corr_dedup:
            emerging, emerg_ex = _dedup_emerging_by_ic_corr(
                emerging_cand,
                dense_final,
                ic_lookup,
                corr_threshold=corr_threshold,
                score=emerging_scores,
            )
            result.exclusions.update(emerg_ex)
        else:
            emerging = list(emerging_cand)

        for name in emerging:
            result.exclusions.pop(name, None)
            result.categories[name] = CAT_EMERGING
        result.emerging_kept = emerging

    label_names = (
        list(dense_final) + list(result.sparse_kept) + list(result.emerging_kept)
    )
    _apply_caution_labels(
        result,
        all_ic,
        pure_ic_series=pure_ic_series,
        names=label_names,
        enable_decay=enable_decay_gate,
        recent_periods=recent_periods,
        retention_min=decay_retention_min,
        retention_min_sparse=decay_retention_min_sparse,
        recent_icir_max=decay_recent_icir_max,
        recent_ic_max=decay_recent_ic_max,
        enable_reversal=enable_reversal_label,
        quarter_periods=quarter_periods,
        reversal_frac=reversal_frac,
        reversal_abs_ic=reversal_abs_ic,
    )
    return result


def _apply_caution_labels(
    result: FactorSelectionResult,
    all_ic: dict | None,
    *,
    names: list[str],
    enable_decay: bool,
    recent_periods: int,
    retention_min: float,
    recent_icir_max: float,
    enable_reversal: bool,
    quarter_periods: int,
    reversal_frac: float,
    reversal_abs_ic: float,
    retention_min_sparse: float | None = None,
    recent_ic_max: float = IC_DECAY_RECENT_IC_MAX,
    pure_ic_series: dict | None = None,
) -> None:
    """对入选/观察因子打衰减 / 风格逆转警示标签（不修改 kept / exclusions）。

    有 ``pure_ic_series`` 时优先用 pure（与新兴近窗同口径）。
    """
    if not names:
        return
    if not all_ic and not pure_ic_series:
        return
    for name in names:
        ic = _resolve_window_ic(name, all_ic, pure_ic_series, prefer_pure_only=False)
        if ic is None:
            continue
        r_min = retention_min
        if (
            retention_min_sparse is not None
            and result.categories.get(name) == CAT_SPARSE
        ):
            r_min = retention_min_sparse
        if enable_decay and recent_periods > 0:
            decayed, _ = evaluate_decay_label(
                ic,
                recent_periods=recent_periods,
                retention_min=r_min,
                recent_icir_max=recent_icir_max,
                recent_ic_max=recent_ic_max,
            )
            if decayed:
                _append_label(result, name, CAT_DECAYED)
        if enable_reversal and quarter_periods > 0:
            rev, _ = evaluate_style_reversal(
                ic,
                quarter_periods=quarter_periods,
                frac_min=reversal_frac,
                abs_ic_min=reversal_abs_ic,
            )
            if rev:
                _append_label(result, name, CAT_REVERSAL)
