"""Factor screening: thresholds, stability, correlation dedup, cost-adjusted IC."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import IC_CORR_METHOD
from research.ic.cost import estimate_ic_after_cost, rank_autocorr_turnover
from research.ic.statistics import benjamini_hochberg, ic_stability_metrics

# TODO: turnover / capacity / SHAP-based selection — not implemented in v2
# TODO: integrate holdings liquidity (ADV) filter from backtest v2


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


def _pairwise_corr_matrix(
    cand_registry: dict,
    sample_dates: list,
) -> list[pd.DataFrame]:
    corr_list = []
    for date in sample_dates:
        row = {}
        for name, fdf in cand_registry.items():
            if date in fdf.index:
                row[name] = fdf.loc[date]
        df_slice = pd.DataFrame(row).dropna()
        if len(df_slice) > 30:
            corr_list.append(df_slice.corr(method="spearman"))
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
    names = corr_list[0].index.tolist()
    out = pd.DataFrame(np.nan, index=names, columns=names)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                out.loc[a, b] = 1.0
                continue
            vals = [abs(c.loc[a, b]) for c in corr_list if a in c.index and b in c.columns]
            if vals:
                out.loc[a, b] = max(vals)
    return out


def select_factors(
    summary_df: pd.DataFrame,
    factor_registry: dict,
    pure_ic_means: dict | None = None,
    ic_threshold: float = 0.02,
    icir_threshold: float = 0.30,
    t_threshold: float = 2.5,
    nw_t_threshold: float | None = None,
    corr_threshold: float = 0.70,
    sample_step: int = 20,
    corr_method: str | None = None,
    rebalance_dates: pd.DatetimeIndex | None = None,
    use_fdr: bool = False,
    fdr_alpha: float = 0.05,
) -> tuple[list, dict]:
    """
    Auto factor selection:
      1. Drop weak IC / ICIR; use NW t when available else t-stat
         - If use_fdr=True: apply Benjamini-Hochberg FDR correction to NW_t
           (or IID t fallback) across all factors; non-rejected factors are
           excluded as statistically insignificant. Original |t|>threshold
           simple rule is preserved as fallback when use_fdr=False.
      2. Correlation dedup (max | p95 | mean pairwise corr)
      3. Stability metrics logged in summary but not hard-filtered yet
    """
    exclusions = {}
    method = corr_method or IC_CORR_METHOD
    use_nw = nw_t_threshold is not None

    t_col = "NW_t统计量" if use_nw else "t统计量"
    fdr_sig: dict[str, bool] = {}
    if use_fdr and t_col in summary_df.columns:
        t_vals = summary_df[t_col].fillna(0.0).values
        sig_mask = benjamini_hochberg(t_vals, alpha=fdr_alpha)
        fdr_sig = dict(zip(summary_df.index, sig_mask))

    candidates = []
    for name in summary_df.index:
        row = summary_df.loc[name]
        raw_ic = abs(row["IC均值"])
        icir = abs(row["ICIR"])
        t_stat = abs(row.get("NW_t统计量" if use_nw else "t统计量", np.nan))
        thresh = nw_t_threshold if use_nw else t_threshold
        pure_ic = abs(pure_ic_means.get(name, np.nan)) if pure_ic_means else raw_ic
        effective_ic = pure_ic if pure_ic_means and not np.isnan(pure_ic) else raw_ic

        if effective_ic < ic_threshold and icir < icir_threshold:
            reason = (
                f"纯IC={pure_ic:.4f}<{ic_threshold}, ICIR={icir:.4f}<{icir_threshold}"
                if pure_ic_means else
                f"IC={raw_ic:.4f}<{ic_threshold}, ICIR={icir:.4f}<{icir_threshold}"
            )
            exclusions[name] = reason
        elif use_fdr and not fdr_sig.get(name, False):
            label = "NW_t" if use_nw else "t"
            exclusions[name] = f"{label}={t_stat:.2f} 未通过 BH-FDR(α={fdr_alpha})（多重检验不显著）"
        elif not use_fdr and not np.isnan(t_stat) and t_stat < thresh:
            label = "NW_t" if use_nw else "t"
            exclusions[name] = f"{label}={t_stat:.2f}<{thresh}（IC均值统计不显著）"
        else:
            candidates.append(name)

    if not candidates:
        return [], exclusions

    cand_registry = {n: factor_registry[n] for n in candidates if n in factor_registry}
    if len(cand_registry) <= 1:
        return candidates, exclusions

    sample_dates = list(factor_registry[candidates[0]].index[::sample_step])
    corr_list = _pairwise_corr_matrix(cand_registry, sample_dates)
    if not corr_list:
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
                if c > corr_threshold:
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


def stability_report(all_ic: dict) -> pd.DataFrame:
    """Per-factor stability metrics table."""
    rows = []
    for name, ic in all_ic.items():
        rows.append({"因子": name, **ic_stability_metrics(ic)})
    return pd.DataFrame(rows).set_index("因子") if rows else pd.DataFrame()
