"""CLI orchestration for IC analysis v2."""
from __future__ import annotations

import argparse
import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*ConstantInput.*")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import (
    BARRA_IC_WORKERS,
    IC_APPLY_TRADABLE,
    IC_CLIP,
    IC_CORR_METHOD,
    IC_MAX_WORKERS,
    IC_MIN_LISTING_DAYS,
    IC_RANK_METHOD,
    INDUSTRY_REFERENCE,
)
from factors.factor import get_factor_registry
from research.ic.barra import run_barra_pure_ic
from research.ic.decay_corr import factor_corr_matrix, ic_decay_table
from research.ic.display import (
    plot_corr_matrix,
    plot_rolling_ic,
    print_barra_comparison,
    print_decay,
    print_selection_result,
    print_summary,
    print_yearly,
)
from research.ic.forward_return import build_forward_return, forward_return_label
from research.ic.ic_series import _to_float32_panel, compute_ic_series
from research.ic.industry import compute_ic_industry
from research.ic.io import save_results
from research.ic.load_data import load_ic_data, load_industry_panel, resolve_industry_map
from research.ic.parallel import run_bounded_parallel
from research.ic.selection import enrich_summary_with_cost, select_factors, stability_report
from research.ic.statistics import _default_nw_lags, ic_by_year, ic_stats
from research.ic.universe import build_ic_tradability_mask
from utils.rebalance_dates import get_rebalance_dates, horizon_to_rebalance_freq

_IC_SKIP_PREFIXES = ("市场", "HMM_")


def _is_ic_skippable(name: str) -> bool:
    return any(name.startswith(p) for p in _IC_SKIP_PREFIXES)


def _log_phase(label: str, t0: float) -> float:
    elapsed = time.perf_counter() - t0
    print(f"  [{label}] {elapsed:.1f}s", flush=True)
    return time.perf_counter()


def run(
    period: int = 20,
    top: int = 0,
    plot: bool = False,
    decay: bool = False,
    corr: bool = False,
    save: bool = False,
    industry: bool = False,
    barra: bool = False,
    lookback_years: int = 0,
    workers: int | None = None,
    barra_workers: int | None = None,
    sample: int = 0,
    corr_method: str | None = None,
    use_nw_t: bool = True,
    t_threshold: float = 2.5,
    use_fdr: bool = False,
):
    t_run = time.perf_counter()
    print("载入数据 (v2)...")
    t0 = time.perf_counter()
    bundle = load_ic_data()
    t0 = _log_phase("载入数据", t0)

    lookback_date = None
    if lookback_years > 0:
        lookback_date = bundle.prices.index.max() - pd.DateOffset(years=lookback_years)
        print(f"  [lookback_years={lookback_years}] IC 自 {lookback_date.date()} 起")

    tradable = None
    if IC_APPLY_TRADABLE:
        tradable = build_ic_tradability_mask(
            bundle.prices,
            volume=bundle.volume,
            masks=bundle.masks,
            stock_names=bundle.stock_names,
            is_st_current=bundle.is_st_current,
        )
        print("  可交易池 mask: ST / 涨跌停 / 停牌")

    print(f"计算因子（持仓期={period}日）...")
    registry = get_factor_registry(
        prices=bundle.prices,
        financial=bundle.financial,
        prices_raw=bundle.prices_raw,
        volume=bundle.volume,
        amount=bundle.amount,
        open_=bundle.open_,
        high=bundle.high,
        low=bundle.low,
        clean_ret=bundle.clean_ret,
        masks=bundle.masks,
        market_prices=bundle.market_prices,
        industry_map=bundle.industry_map_df,
        margin=bundle.margin,
        moneyflow=bundle.moneyflow,
        northbound=bundle.northbound,
        institution=bundle.institution,
    )
    skipped = [k for k in registry if _is_ic_skippable(k)]
    if skipped:
        registry = {k: v for k, v in registry.items() if not _is_ic_skippable(k)}
        print(f"  跳过 {len(skipped)} 个市场/HMM 特征")

    if sample > 0:
        keys = list(registry.keys())[:sample]
        registry = {k: registry[k] for k in keys}
        print(f"  [--sample {sample}] 仅计算前 {len(registry)} 个因子")

    registry = {k: _to_float32_panel(v) for k, v in registry.items()}
    t0 = _log_phase(f"计算因子（{len(registry)}个）", t0)

    print(f"  forward_return: {forward_return_label(bundle.open_)}")
    forward_return = build_forward_return(bundle.prices, bundle.open_, period).astype(np.float32)
    rebalance_dates = get_rebalance_dates(forward_return.index, horizon_to_rebalance_freq(period))
    print(f"  调仓日: {len(rebalance_dates)} 个")

    industry_map = resolve_industry_map(
        bundle.industry_map_df,
        need=industry or barra,
    )
    industry_panel = None
    if barra:
        # 优先用 PIT 行业面板（消除行业映射回填历史截面的未来信息泄漏）；
        # 文件不存在时 fallback 到静态 industry_map（向后兼容）。
        industry_panel = load_industry_panel()
        if industry_panel is not None:
            print(f"  [PIT] 使用 industry_map_panel.parquet（"
                  f"{len(industry_panel)} 条 / "
                  f"{industry_panel['code'].nunique()} 只股票）")
        elif industry_map is not None:
            print("  [PIT] industry_map_panel.parquet 不存在，"
                  "回退到静态 industry_map（行业哑变量可能含 PIT 泄漏）")

    ic_workers = IC_MAX_WORKERS if workers is None else max(1, workers)
    barra_ic_workers = BARRA_IC_WORKERS if barra_workers is None else max(1, barra_workers)
    n_factors = len(registry)
    mode = "串行" if ic_workers == 1 else f"最多{ic_workers}并发"
    print(f"计算IC（{n_factors}个因子，{mode}，ICIR std ddof=0）...")

    def _compute_one(name_fac):
        name, factor = name_fac
        return name, compute_ic_series(factor, forward_return, tradable=tradable)

    all_ic_full = {}
    for name, ic_full in run_bounded_parallel(
        _compute_one, list(registry.items()), ic_workers, progress_every=10
    ):
        all_ic_full[name] = ic_full
    t0 = _log_phase(f"计算IC（{n_factors}个因子）", t0)

    all_ic = {}
    summary_rows = []
    for name, ic_full in all_ic_full.items():
        ic = ic_full[ic_full.index >= lookback_date] if lookback_date is not None else ic_full
        all_ic[name] = ic
        summary_rows.append({"因子": name, **ic_stats(ic)})

    summary_df = (
        pd.DataFrame(summary_rows)
        .set_index("因子")
        .sort_values("|IC|均值", ascending=False)
    )
    summary_df = enrich_summary_with_cost(summary_df, registry, rebalance_dates)

    if top > 0:
        summary_df = summary_df.head(top)
        all_ic = {k: v for k, v in all_ic.items() if k in summary_df.index}

    print_summary(summary_df)

    stab = stability_report(all_ic)
    if not stab.empty:
        print(f"\n  IC 稳定性（滚动σ / 同向年份占比）:")
        print(stab.head(10).to_string())

    yearly_rows = []
    all_years = sorted({y for ic in all_ic_full.values() for y in ic.index.year})
    for name, ic in all_ic_full.items():
        by_year = ic_by_year(ic)
        row = {"因子": name, **{y: by_year.get(y, np.nan) for y in all_years}}
        yearly_rows.append(row)
    yearly_df = pd.DataFrame(yearly_rows).set_index("因子").loc[summary_df.index]
    print_yearly(yearly_df)

    if decay:
        print("\n计算IC衰减...")
        print_decay(ic_decay_table(registry, bundle.prices, bundle.open_, tradable=tradable))

    if corr:
        print("\n因子相关矩阵...")
        corr_mat = factor_corr_matrix(registry, bundle.prices)
        if not corr_mat.empty:
            if plot:
                plot_corr_matrix(corr_mat)
            else:
                print(corr_mat.round(3).to_string())

    pure_ic_means = {}
    barra_names_used: list = []
    if barra:
        print("\n计算 Barra 纯因子 IC...")
        try:
            pure_ic_means = run_barra_pure_ic(
                registry=registry,
                summary_index=summary_df.index,
                prices=bundle.prices,
                financial=bundle.financial,
                forward_return=forward_return,
                rebalance_dates=rebalance_dates,
                industry_map=industry_map,
                volume=bundle.volume,
                market_prices=bundle.market_prices,
                parallel_fn=run_bounded_parallel,
                barra_workers=barra_ic_workers,
                industry_panel=industry_panel,
                names_sink=barra_names_used,
                clean_ret=bundle.clean_ret,
            )
            if pure_ic_means:
                print_barra_comparison(summary_df, pure_ic_means)
            else:
                print("Barra 因子计算失败，跳过")
        except Exception as e:
            print(f"Barra 分析出错: {e}")
            import traceback
            traceback.print_exc()

    ind_ic_df = pd.DataFrame()
    if industry and industry_map is not None:
        print("\n分行业 IC...")
        ind_ic_df = compute_ic_industry(
            {k: v for k, v in registry.items() if k in summary_df.index},
            forward_return,
            industry_map,
            tradable=tradable,
        )

    if plot:
        plot_rolling_ic(all_ic, period)

    cm = corr_method or IC_CORR_METHOD
    kept, exclusions = select_factors(
        summary_df,
        registry,
        pure_ic_means=pure_ic_means or None,
        corr_method=cm,
        rebalance_dates=rebalance_dates,
        nw_t_threshold=t_threshold if use_nw_t else None,
        t_threshold=t_threshold,
        use_fdr=use_fdr,
    )
    print_selection_result(kept, exclusions)

    if save:
        # ── JSON 元数据（P1-7）：记录可追溯的样本/配置快照 ──
        rb_idx = forward_return.index.intersection(rebalance_dates)
        if len(rb_idx) > 0:
            universe_size = int(
                round(float(forward_return.loc[rb_idx].notna().sum(axis=1).mean()))
            )
        else:
            universe_size = 0
        ic_series_length = int(len(next(iter(all_ic_full.values())))) if all_ic_full else 0
        sample_period = (
            [str(forward_return.index.min()), str(forward_return.index.max())]
            if len(forward_return.index) else ["", ""]
        )
        nw_lag = int(_default_nw_lags(ic_series_length)) if ic_series_length > 0 else 0
        meta = {
            "universe_size": universe_size,
            "ic_series_length": ic_series_length,
            "sample_period": sample_period,
            "barra_factors_used": barra_names_used,
            "industry_reference": INDUSTRY_REFERENCE,
            "nw_lag": nw_lag,
            "config_snapshot": {
                "IC_CLIP": IC_CLIP,
                "IC_CORR_METHOD": IC_CORR_METHOD,
                "IC_RANK_METHOD": IC_RANK_METHOD,
                "IC_MIN_LISTING_DAYS": IC_MIN_LISTING_DAYS,
            },
        }
        json_path = save_results(
            period=period,
            summary_df=summary_df,
            yearly_df=yearly_df,
            kept_factors=kept,
            exclusion_reasons=exclusions,
            lookback_years=lookback_years,
            lookback_date=lookback_date,
            ind_ic_df=ind_ic_df,
            pure_ic_means=pure_ic_means or None,
            meta=meta,
        )
        print(f"\n结果已保存至 {json_path.parent}  ({json_path.name})")

    _log_phase("总耗时", t_run)
    gc.collect()
    return summary_df, yearly_df, all_ic, ind_ic_df


def main():
    from config.encoding_bootstrap import bootstrap_stdio_utf8

    bootstrap_stdio_utf8()
    parser = argparse.ArgumentParser(description="IC analysis v2 (modular)")
    parser.add_argument("--period", type=int, default=20)
    parser.add_argument("--top", type=int, default=0)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--decay", action="store_true")
    parser.add_argument("--corr", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--industry", action="store_true")
    parser.add_argument("--barra", action="store_true")
    parser.add_argument("--lookback-years", type=int, default=0, dest="lookback_years")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--barra-workers", type=int, default=None, dest="barra_workers")
    parser.add_argument("--sample", type=int, default=0, help="仅前 N 个因子（快速 smoke）")
    parser.add_argument(
        "--corr-method",
        choices=["max", "p95", "mean"],
        default=None,
        help=f"去冗余相关度聚合（默认 {IC_CORR_METHOD}）",
    )
    parser.add_argument(
        "--no-nw-t",
        action="store_true",
        help="筛选用经典 t 而非 Newey-West t",
    )
    parser.add_argument(
        "--t-threshold",
        type=float,
        default=2.5,
        help="t / NW_t 显著性阈值（默认 2.5，应对多重检验收紧）",
    )
    parser.add_argument(
        "--use-fdr",
        action="store_true",
        help="对 t 统计量做 Benjamini-Hochberg FDR 校正（多重检验控制）",
    )
    args = parser.parse_args()
    run(
        period=args.period,
        top=args.top,
        plot=args.plot,
        decay=args.decay,
        corr=args.corr,
        save=args.save,
        industry=args.industry,
        barra=args.barra,
        lookback_years=args.lookback_years,
        workers=args.workers,
        barra_workers=args.barra_workers,
        sample=args.sample,
        corr_method=args.corr_method,
        use_nw_t=not args.no_nw_t,
        t_threshold=args.t_threshold,
        use_fdr=args.use_fdr,
    )


if __name__ == "__main__":
    main()
