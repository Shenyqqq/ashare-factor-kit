"""CLI orchestration for IC analysis v2."""
from __future__ import annotations

import argparse
import gc
import os
import pickle
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
    CAP_BAND_DEFAULT,
    CAP_BANDS,
    FWD_RETURN_WINSOR,
    IC_APPLY_TRADABLE,
    IC_CLIP,
    IC_CORR_METHOD,
    IC_DECAY_HALF_LIFE_MIN,
    IC_DECAY_RECENT_IC_MAX,
    IC_DECAY_RECENT_ICIR_MAX,
    IC_DECAY_RECENT_MONTHS,
    IC_DECAY_RESIDUAL_IC,
    IC_DECAY_RESIDUAL_ICIR,
    IC_DECAY_RETENTION_MIN,
    IC_DECAY_RETENTION_MIN_SPARSE,
    IC_DECAY_SHORT_LONG_MIN,
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
    IC_MAX_WORKERS,
    IC_MIN_LISTING_DAYS,
    IC_MIN_LONG_SHARE,
    IC_QUANTILE_DECOMP,
    IC_QUANTILE_Y_MODE,
    IC_RANK_METHOD,
    IC_REVERSAL_ABS_IC,
    IC_REVERSAL_FRAC,
    IC_REVERSAL_MONTHS,
    IC_SPARSE_CORR_THRESHOLD,
    IC_SPARSE_IC_THRESHOLD,
    IC_SPARSE_ICIR_THRESHOLD,
    IC_SPARSE_PAYOFF_MIN,
    IC_SPARSE_T_THRESHOLD,
    IC_SPARSE_WIN_RATE_MIN,
    IC_THRESHOLD,
    ICIR_THRESHOLD,
    INDUSTRY_REFERENCE,
    SMALL_MCAP_QUANTILE,
    apply_label_exec_mask_for_mode,
    normalize_tradable_limit_mode,
    resolve_apply_exec_mask,
    resolve_exclude_limit_on_signal,
    tradable_ckpt_tag,
    tradable_mode_metadata,
)
from factors.factor import (
    compute_single_factor,
    get_factor_names,
    iter_factor_registry,
    _filter_none_emit,
)
from research.ic.barra import (
    barra_pure_cache_version,
    barra_pure_version_ok,
    merge_barra_pure_results,
    missing_barra_pure_names,
    pack_barra_pure_ckpt,
    run_barra_pure_ic,
    unpack_barra_pure_ckpt,
)
from research.ic.cost import estimate_ic_after_cost, rank_autocorr_turnover
from research.ic.decay_corr import factor_corr_matrix, ic_decay_table
from research.ic.display import (
    plot_corr_matrix,
    plot_rolling_ic,
    plot_rolling_ic_lines,
    print_barra_comparison,
    print_decay,
    print_quantile_ls,
    print_selection_result,
    print_summary,
    print_yearly,
)
from research.ic.forward_return import (
    build_forward_return,
    forward_return_label,
    winsorize_forward_return,
)
from research.ic.ic_series import _to_float32_panel, compute_ic_series
from research.ic.industry import compute_ic_industry
from research.ic.io import save_results
from research.ic.load_data import (
    load_ic_data,
    load_industry_panel,
    require_industry_panel,
    resolve_industry_map,
)
from research.ic.orthogonalize import gram_schmidt_select
from research.ic.parallel import run_bounded_parallel
from research.ic.selection import (
    overlay_long_share,
    overlay_pure_t_stats,
    select_factors_multi_track,
    stability_report,
)
from research.ic.statistics import _default_nw_lags, ic_by_year, ic_stats
from research.ic.universe import build_ic_tradability_mask, load_listing_dates, load_delist_dates
from utils.rebalance_dates import get_rebalance_dates, horizon_to_rebalance_freq
from utils.universe import (
    build_cap_band_mask,
    build_mcap_percentile_mask,
    load_universe_mask_file,
)

_IC_SKIP_PREFIXES = ("市场", "HMM_")

# IC 阶段 checkpoint 宇宙后缀（换 universe 不得复用全市场 IC）
_CKPT_TAG = ""
_CKPT_NEUT_TAG = ""
_NEUT_CKPT_STAGES = frozenset({"barra_pure", "selection", "gramschmidt"})


def _is_ic_skippable(name: str) -> bool:
    """IC 候选跳过：regime 前缀 + 事件 overlay（不进常规 IC 筛选池）。

    事件因子本身不在 ``get_factor_names`` / ML registry 枚举中，此处显式跳过
    作文档化防护，防止日后误入候选列表。唯一源：
    ``factors.factor.EVENT_OVERLAY_FACTOR_NAMES``。
    """
    from factors.factor import EVENT_OVERLAY_FACTOR_NAMES
    return (
        any(name.startswith(p) for p in _IC_SKIP_PREFIXES)
        or name in EVENT_OVERLAY_FACTOR_NAMES
    )


# ══════════════════════════════════════════════════════════════════════════════
# 阶段 checkpoint：崩溃后 --resume 跳过已完成阶段，避免重算 34min 因子 IC
# ══════════════════════════════════════════════════════════════════════════════

_CKPT_DIR_DEFAULT = Path("research/output/_checkpoints")
_CKPT_DIR = _CKPT_DIR_DEFAULT


def _universe_tag(
    universe: str | None,
    quantile: float,
    universe_mask: str | None,
    cap_band: str | None,
    mcap_min_yi: float | None = None,
    mcap_max_yi: float | None = None,
    calendar_fp: str = "",
) -> str:
    """Checkpoint / 落盘文件名后缀；全市场返回空串（兼容旧路径）。"""
    import hashlib

    if mcap_min_yi is not None or mcap_max_yi is not None:
        lo = int(mcap_min_yi) if mcap_min_yi is not None else 0
        hi = int(mcap_max_yi) if mcap_max_yi is not None else 0
        tag = f"mcap{lo}_{hi}"
        if calendar_fp:
            tag = f"{tag}_{calendar_fp}"
        return tag
    if universe_mask:
        h = hashlib.md5(str(universe_mask).encode("utf-8")).hexdigest()[:8]
        return f"umask_{h}"
    u = (universe or "all").strip().lower()
    if u == "small_mcap":
        return f"small_mcap_q{int(round(float(quantile) * 100)):02d}"
    if u not in ("all", "", None):
        return u.replace("-", "_")
    if cap_band not in ("all", None, ""):
        return f"cap_{cap_band}"
    return ""


def _calendar_fp(index) -> str:
    """交易日历指纹：首尾日期 + 长度。换数据区间不得 resume 旧 ckpt。"""
    import hashlib

    if index is None or len(index) == 0:
        return "nocal"
    idx = pd.DatetimeIndex(index)
    payload = f"{idx.min().date()}_{idx.max().date()}_{len(idx)}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:6]


def _set_ckpt_tag(tag: str) -> None:
    global _CKPT_TAG
    _CKPT_TAG = tag or ""


def _set_ckpt_dir(path: Path | None) -> None:
    """全市场用默认目录；亿元带落到 ``_checkpoints/mcap{lo}_{hi}/``，禁止同名覆盖。"""
    global _CKPT_DIR
    _CKPT_DIR = path if path is not None else _CKPT_DIR_DEFAULT


def _set_ckpt_neut_tag(tag: str) -> None:
    """raw / size / size_industry 与 9 风格 barra_pure 隔离；barra 保持空后缀。"""
    global _CKPT_NEUT_TAG
    _CKPT_NEUT_TAG = tag or ""


def _ckpt_path(period: int, stage: str) -> Path:
    tag = f"_{_CKPT_TAG}" if _CKPT_TAG else ""
    neut = ""
    if stage in _NEUT_CKPT_STAGES and _CKPT_NEUT_TAG:
        neut = f"_{_CKPT_NEUT_TAG}"
    return _CKPT_DIR / f"{stage}_h{period}{neut}{tag}.pkl"


def _save_ckpt(period: int, stage: str, obj) -> None:
    p = _ckpt_path(period, stage)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    try:
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        # 失败时清理 tmp，绝不让半截文件留下
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    # 原子替换：os.replace 在同盘上是原子操作；中断后 p 仍为上一次完整 checkpoint
    os.replace(tmp, p)


def _load_ckpt(period: int, stage: str):
    p = _ckpt_path(period, stage)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def _clear_ckpts(period: int) -> None:
    if not _CKPT_DIR.exists():
        return
    stages = ("ic_series", "summary", "yearly", "barra_pure", "selection", "gramschmidt")
    for stage in stages:
        p = _ckpt_path(period, stage)
        try:
            p.unlink()
        except OSError:
            pass


_DOWNSTREAM_STAGES = ("summary", "yearly", "barra_pure", "selection", "gramschmidt")
# 增量补录：保留 barra_pure，仅对新因子做纯化并 merge；FDR/selection 仍全表重算
_DOWNSTREAM_STAGES_KEEP_BARRA = ("summary", "yearly", "selection", "gramschmidt")


def _clear_downstream_ckpts(
    period: int, *, keep_barra_pure: bool = False,
) -> None:
    """删除 summary 及之后阶段 checkpoint，保留 ic_series（供增量合并）。

    ``keep_barra_pure=True``（``--only-new`` / ``--factors``）：不清 barra_pure，
    由后续阶段按指纹校验后只补算缺失因子并 merge。
    """
    if not _CKPT_DIR.exists():
        return
    stages = (
        _DOWNSTREAM_STAGES_KEEP_BARRA if keep_barra_pure else _DOWNSTREAM_STAGES
    )
    for stage in stages:
        p = _ckpt_path(period, stage)
        if p.exists():
            try:
                p.unlink()
                print(f"  [only-new] 清除下游 checkpoint: {p.name}")
            except OSError as e:
                print(f"  [only-new] 无法删除 {p.name}: {e}")


def _resolve_domain_mask(
    *,
    universe: str,
    universe_quantile: float,
    universe_mask_path: str | None,
    cap_band: str,
    circ_mv: pd.DataFrame | None,
    total_mv: pd.DataFrame | None,
    amount: pd.DataFrame | None,
    rebalance_dates: pd.DatetimeIndex,
    trading_index: pd.DatetimeIndex,
    mcap_min_yi: float | None = None,
    mcap_max_yi: float | None = None,
) -> tuple[pd.DataFrame | None, str]:
    """解析宇宙域 mask。优先级：--universe-mask > 亿元带 > --universe > --cap-band。

    返回 (mask_or_None, 人类可读描述)。因子面板列集不变，仅截面 AND。
    """
    if universe_mask_path:
        mask = load_universe_mask_file(universe_mask_path)
        mask = mask.reindex(index=trading_index).ffill().fillna(False).astype(bool)
        desc = f"universe-mask file={universe_mask_path}"
        return mask, desc

    if mcap_min_yi is not None or mcap_max_yi is not None:
        if circ_mv is None and total_mv is None:
            raise SystemExit(
                "--mcap-min-yi/--mcap-max-yi 需要 circ_mv 或 total_mv"
                "（请先 `python -m data.download_stock_value_em`）"
            )
        from utils.universe import build_mcap_yi_band_mask
        mask = build_mcap_yi_band_mask(
            circ_mv, min_yi=mcap_min_yi, max_yi=mcap_max_yi, total_mv=total_mv,
        )
        lo = "无" if mcap_min_yi is None else f"{float(mcap_min_yi):.0f}"
        hi = "无" if mcap_max_yi is None else f"{float(mcap_max_yi):.0f}"
        src = "circ_mv" if circ_mv is not None else "total_mv"
        desc = (
            f"mcap-yi-band: {src} ∈ [{lo}, {hi}] 亿元（含边界，单位元=亿×1e8；"
            f"无 20 日成交额过滤）"
        )
        return mask, desc

    u = (universe or "all").strip().lower()
    if u == "small_mcap":
        if circ_mv is None and total_mv is None:
            raise SystemExit(
                "--universe small_mcap 需要 circ_mv 或 total_mv（请先 "
                "`python -m data.download_stock_value_em`）"
            )
        mask = build_mcap_percentile_mask(
            circ_mv,
            quantile=universe_quantile,
            total_mv=total_mv,
            rebalance_dates=rebalance_dates,
            trading_index=trading_index,
        )
        src = "circ_mv" if circ_mv is not None else "total_mv"
        desc = (
            f"small_mcap: 调仓日截面 {src} 升序分位 ≤ {universe_quantile:.0%} "
            f"后 ffill（最低 {universe_quantile:.0%} 市值）"
        )
        return mask, desc

    if u not in ("all", ""):
        raise SystemExit(
            f"未知 --universe={universe!r}；可选: all, small_mcap "
            f"（或用 --universe-mask / --cap-band）"
        )

    if cap_band not in ("all", None, ""):
        if circ_mv is None and total_mv is None and amount is None:
            print(f"  [cap-band={cap_band}] circ_mv/total_mv/amount 均缺失，跳过市值带过滤")
            return None, "all"
        mask = build_cap_band_mask(
            cap_band, circ_mv, amount, total_mv=total_mv,
        )
        desc = f"cap-band={cap_band}"
        return mask, desc

    return None, "all"


def _log_mask_coverage(
    mask: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    label: str,
) -> None:
    """打印 mask 后每期股票数（中位数优先）。"""
    rb = pd.DatetimeIndex(rebalance_dates).intersection(mask.index)
    if len(rb) == 0:
        per = mask.sum(axis=1)
    else:
        per = mask.loc[rb].sum(axis=1)
    if len(per) == 0:
        print(f"  [{label}] mask 覆盖：无有效日期")
        return
    print(
        f"  [{label}] mask 后每期股票数: "
        f"median={int(per.median())} mean={int(per.mean())} "
        f"min={int(per.min())} max={int(per.max())} "
        f"（n_dates={len(per)}）"
    )


def _log_phase(label: str, t0: float) -> float:
    elapsed = time.perf_counter() - t0
    print(f"  [{label}] {elapsed:.1f}s", flush=True)
    return time.perf_counter()


# ══════════════════════════════════════════════════════════════════════════════
# 流式 IC 内存优化辅助（P0）
# ══════════════════════════════════════════════════════════════════════════════

def _build_data_kwargs(bundle) -> dict:
    """从 ICDataBundle 抽取 iter_factor_registry / compute_single_factor 所需 kwargs。"""
    return dict(
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
        circ_mv=bundle.circ_mv,
        total_mv=bundle.total_mv,
    )


def _stream_turnovers(names, data_kwargs: dict, rebalance_dates) -> dict:
    """
    流式计算每个因子的 rank_turnover（用完即释，避免持有全部面板）。

    通过 ``iter_factor_registry`` 逐因子生成 → 算 turnover → 释放，
    内存峰值仅为当前 1 个因子面板（~95MB），而非 names 全量面板。
    """
    name_set = set(names)
    turnovers: dict = {}
    for name, panel in _filter_none_emit(
        iter_factor_registry(factor_names=name_set, include_regime=False, **data_kwargs)
    ):
        if name in name_set:
            turnovers[name] = rank_autocorr_turnover(panel, rebalance_dates)
        del panel
    for name in names:
        turnovers.setdefault(name, np.nan)
    gc.collect()
    return turnovers


class _LazyFactorRegistry:
    """
    dict-like 懒加载因子 registry：``__getitem__`` 时才计算单个因子面板。

    用于 IC 下游需要按 ``registry[name]`` 访问单个因子、但不需同时持有全部面板的场景
    （Barra 纯因子 IC、select_factors 相关性去冗余）。

    cache=True  时缓存已计算的面板（用于 select_factors 需要同时访问多个候选因子
                做 corr 矩阵的场景；内存 = |已访问 names| × 48MB， transient）。
    cache=False 时不缓存，每次 __getitem__ 都重新计算并返回新面板，调用方用完即释
                （用于 Barra 逐因子 OLS 的场景；内存峰值 = 1 个面板）。
    """

    def __init__(self, computable_names, data_kwargs: dict, cache: bool = True):
        self._names = set(computable_names)
        self._kwargs = data_kwargs
        self._cache: dict = {}
        self._do_cache = cache

    def __contains__(self, name) -> bool:
        return name in self._names

    def __getitem__(self, name):
        if name not in self._names:
            raise KeyError(name)
        if self._do_cache and name in self._cache:
            return self._cache[name]
        panel = compute_single_factor(name, **self._kwargs)
        if panel is None:
            # 流式阶段曾成功计算过该因子但单因子重算失败（罕见，可能因数据竞争）；
            # 移出可计算集，下次 __contains__ 返回 False。
            self._names.discard(name)
            raise KeyError(name)
        if self._do_cache:
            self._cache[name] = panel
        return panel

    def get(self, name, default=None):
        try:
            return self[name]
        except KeyError:
            return default

    def release_cache(self):
        """显式释放缓存的面板（cache=False 时无操作）。"""
        self._cache.clear()
        gc.collect()



def _maybe_plot_rolling_ic(
    *,
    enabled: bool,
    period: int,
    all_ic: dict,
    pure_ic_series: dict | None,
    summary_df: pd.DataFrame,
    kept: list | None,
    sparse_kept: list | None,
    top_n: int,
    window: int | None,
    names_csv: str | None,
    source: str,
    tag: str,
) -> None:
    """Optional rolling-IC line chart from already-computed series (resume-safe)."""
    if not enabled:
        return
    name_list = None
    if names_csv:
        name_list = [n.strip() for n in str(names_csv).split(",") if n.strip()]
    selected = None
    if not name_list:
        selected = list(kept or []) + [n for n in (sparse_kept or []) if n not in (kept or [])]
        if not selected:
            selected = None
    src = (source or "auto").lower().strip()
    series_map = all_ic
    label = "raw"
    if src == "pure" or (src == "auto" and pure_ic_series):
        if pure_ic_series:
            series_map = pure_ic_series
            label = "pure"
        elif src == "pure":
            print("  [plot-rolling-ic] 无 pure IC 序列，回退 raw")
    plot_rolling_ic_lines(
        series_map,
        period,
        names=name_list,
        selected=selected,
        summary_df=summary_df,
        top_n=top_n,
        window=window,
        source_label=label,
        tag=tag,
        show=False,
    )


def run(
    period: int = 20,
    top: int = 0,
    plot: bool = False,
    plot_rolling_ic_flag: bool = False,
    plot_rolling_ic_top_n: int = 30,
    plot_rolling_ic_window: int | None = None,
    plot_rolling_ic_names: str | None = None,
    plot_rolling_ic_source: str = "auto",
    decay: bool = False,
    corr: bool = False,
    save: bool = False,
    industry: bool = False,
    allow_static_industry: bool = False,
    barra: bool = False,
    neut_controls: str | None = None,
    save_suffix: str = "",
    lookback_years: int = 0,
    workers: int | None = None,
    barra_workers: int | None = None,
    sample: int = 0,
    factor_prefix: str | None = None,
    batch_size: int = 0,
    corr_method: str | None = None,
    use_nw_t: bool = True,
    t_threshold: float = 2.5,
    use_fdr: bool = True,
    gram_schmidt: bool = False,
    max_factors: int = 30,
    gs_ic_threshold: float = 0.015,
    gs_icir_threshold: float = 0.15,
    regime_consistency: float | None = None,
    rolling_icir: float | None = None,
    worst_period_ic: float | None = None,
    raw_select: bool = False,
    resume: bool = False,
    fresh: bool = False,
    only_new: bool = False,
    factors: str | None = None,
    cap_band: str = CAP_BAND_DEFAULT,
    universe: str = "all",
    universe_quantile: float = SMALL_MCAP_QUANTILE,
    universe_mask: str | None = None,
    mcap_min_yi: float | None = None,
    mcap_max_yi: float | None = None,
    restan_in_universe: bool | None = None,
    min_industry_n: int | None = None,
    fwd_return_winsor: bool = True,
    corr_dedup: bool = True,
    corr_threshold: float = 0.70,
    ic_threshold: float = IC_THRESHOLD,
    icir_threshold: float = ICIR_THRESHOLD,
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
    # deprecated half-life / residual (ignored; CLI warns)
    decay_half_life_min: float | None = IC_DECAY_HALF_LIFE_MIN,
    decay_short_long_min: float | None = IC_DECAY_SHORT_LONG_MIN,
    decay_residual_icir: float = IC_DECAY_RESIDUAL_ICIR,
    decay_residual_ic: float = IC_DECAY_RESIDUAL_IC,
    enable_emerging: bool = True,
    emerging_lookback: int = IC_EMERGING_LOOKBACK,
    emerging_recent_icir: float = IC_EMERGING_RECENT_ICIR,
    emerging_recent_ic: float = IC_EMERGING_RECENT_IC,
    emerging_fdr_alpha: float = IC_EMERGING_FDR_ALPHA,
    emerging_lift_min: float = IC_EMERGING_LIFT_MIN,
    emerging_holdout_months: int = IC_EMERGING_HOLDOUT_MONTHS,
    emerging_asof: str | None = None,
    emerging_require_trend: bool = IC_EMERGING_REQUIRE_TREND,
    emerging_trend_months: int = IC_EMERGING_TREND_MONTHS,
    emerging_trend_segments: int = IC_EMERGING_TREND_SEGMENTS,
    emerging_trend_eps: float = IC_EMERGING_TREND_EPS,
    enable_sparse_track: bool = True,
    sparse_ic_threshold: float = IC_SPARSE_IC_THRESHOLD,
    sparse_icir_threshold: float = IC_SPARSE_ICIR_THRESHOLD,
    sparse_t_threshold: float = IC_SPARSE_T_THRESHOLD,
    sparse_win_rate_min: float = IC_SPARSE_WIN_RATE_MIN,
    sparse_payoff_min: float = IC_SPARSE_PAYOFF_MIN,
    sparse_require_ic: bool = False,
    sparse_corr_threshold: float = IC_SPARSE_CORR_THRESHOLD,
    quantile_decomp: bool | None = None,
    quantile_y_mode: str | None = None,
    min_long_share: float | None = IC_MIN_LONG_SHARE,
    long_share_csv: str | None = None,
    tradable_limit_mode: str | None = None,
    exclude_limit_on_signal: bool | None = None,
    apply_exec_mask: bool | None = None,
):
    t_run = time.perf_counter()
    ex_lim = resolve_exclude_limit_on_signal(
        exclude_limit_on_signal, tradable_limit_mode
    )
    ex_exec = resolve_apply_exec_mask(apply_exec_mask, tradable_limit_mode)
    limit_mode = normalize_tradable_limit_mode(tradable_limit_mode)
    tmr_meta = tradable_mode_metadata(
        exclude_limit_on_signal, apply_exec_mask, tradable_limit_mode
    )
    # --barra 默认开分位分解；显式 None 时跟 settings；False 关闭
    if quantile_decomp is None:
        quantile_decomp = bool(IC_QUANTILE_DECOMP)
    quantile_y_mode = (quantile_y_mode or IC_QUANTILE_Y_MODE or "residual").lower().strip()
    if quantile_y_mode not in ("residual", "raw"):
        raise ValueError(
            f"quantile_y_mode 须为 residual|raw，收到 {quantile_y_mode!r}"
        )
    # 宇宙 + 可交易口径标签：checkpoint 后缀在载入日历后才钉死（含 mcap 日历指纹）
    tmr_tag = tradable_ckpt_tag(
        exclude_limit_on_signal, apply_exec_mask, tradable_limit_mode
    )
    use_mcap_band = mcap_min_yi is not None or mcap_max_yi is not None
    # 每次 run 重置，避免同进程二次调用残留全市场 / 中盘目录
    _set_ckpt_tag("")
    _set_ckpt_neut_tag("")
    _set_ckpt_dir(None)
    if use_mcap_band:
        lo = int(mcap_min_yi) if mcap_min_yi is not None else 0
        hi = int(mcap_max_yi) if mcap_max_yi is not None else 0
        _set_ckpt_dir(_CKPT_DIR_DEFAULT / f"mcap{lo}_{hi}")
    if restan_in_universe is None:
        restan_in_universe = bool(use_mcap_band)
    if min_industry_n is None:
        min_industry_n = 10 if use_mcap_band else 0
    min_industry_n = int(min_industry_n)
    # 中性化口径：None=raw；size / size_industry / barra。--barra ≡ barra。
    from models.wf.labels import (
        NEUT_CONTROLS_CHOICES,
        NEUT_CONTROLS_SIZE,
        NEUT_CONTROLS_SIZE_INDUSTRY,
    )
    nc_raw = (str(neut_controls).strip().lower() if neut_controls else "")
    if barra and nc_raw and nc_raw not in ("barra",):
        raise ValueError(
            "--barra 是 9 风格纯 IC，与 --neut-controls size/size_industry 互斥"
        )
    if nc_raw in ("", "raw", "none"):
        neut_mode = "barra" if barra else None
    elif nc_raw in NEUT_CONTROLS_CHOICES:
        neut_mode = nc_raw
    else:
        raise ValueError(
            f"未知 --neut-controls={neut_controls!r}，可选: raw / "
            f"{list(NEUT_CONTROLS_CHOICES)}"
        )
    do_neut = neut_mode is not None
    if neut_mode and neut_mode != "barra":
        _set_ckpt_neut_tag(f"nc_{neut_mode}")
    elif not do_neut:
        _set_ckpt_neut_tag("nc_raw")
    else:
        _set_ckpt_neut_tag("")
    print(
        f"  [tradable] mode={tmr_meta['tradable_mode']} "
        f"exclude_limit_signal_day={ex_lim} "
        f"label_exec_mask={tmr_meta['label_exec_mask']}"
    )

    # 仅在用户显式 --fresh/--clear-ckpts 时清空；非 resume 默认保留旧 checkpoint，
    # 由 _save_ckpt 的原子替换保证中断后仍为上一次完整结果（避免 smoke test / 崩溃
    # 摧毁 34min 的真实 IC checkpoint）。
    if fresh:
        if only_new or factors:
            print("  [警告] --fresh 与 --only-new/--factors 互斥：--fresh 优先，将全量重算")
            only_new = False
            factors = None
    # 增量模式：保留 ic_series + barra_pure（指纹匹配时只补新区）；
    # summary/selection/GS 等下游仍强制重跑。
    incremental = bool(only_new or factors) and not fresh
    resume_downstream = bool(resume) and not incremental
    print("载入数据 (v2)...")
    t0 = time.perf_counter()
    bundle = load_ic_data()
    t0 = _log_phase("载入数据", t0)

    lookback_date = None
    if lookback_years > 0:
        lookback_date = bundle.prices.index.max() - pd.DateOffset(years=lookback_years)
        print(f"  [lookback_years={lookback_years}] IC 自 {lookback_date.date()} 起")

    # ════════════════════════════════════════════════════════════════════════
    # P0 内存优化：流式 IC + 按需重建小子集 registry + bundle 早期释放
    # ════════════════════════════════════════════════════════════════════════
    data_kwargs = _build_data_kwargs(bundle)

    # forward_return / 调仓日先建（域 mask 按调仓日截面判定，再 AND 进 tradable）
    print(f"  forward_return: {forward_return_label(bundle.open_)}")
    forward_return = build_forward_return(
        bundle.prices,
        bundle.open_,
        period,
        masks=bundle.masks,
        apply_exec_mask=ex_exec,
    ).astype(np.float32)
    rebalance_dates = get_rebalance_dates(
        forward_return.index, horizon_to_rebalance_freq(period)
    )
    print(f"  调仓日: {len(rebalance_dates)} 个")

    # 域 mask：只缩小截面，不改 data_kwargs / 因子缓存指纹（全市场面板可复用）
    domain_mask, domain_desc = _resolve_domain_mask(
        universe=universe,
        universe_quantile=universe_quantile,
        universe_mask_path=universe_mask,
        cap_band=cap_band,
        circ_mv=bundle.circ_mv,
        total_mv=bundle.total_mv,
        amount=bundle.amount,
        rebalance_dates=rebalance_dates,
        trading_index=bundle.prices.index,
        mcap_min_yi=mcap_min_yi,
        mcap_max_yi=mcap_max_yi,
    )
    if domain_mask is not None:
        print(f"  [universe] {domain_desc}")
        _log_mask_coverage(domain_mask, rebalance_dates, "universe-domain")

    tradable = None
    apply_tradable = bool(IC_APPLY_TRADABLE) or domain_mask is not None
    if apply_tradable:
        tradable = build_ic_tradability_mask(
            bundle.prices,
            volume=bundle.volume,
            masks=bundle.masks,
            stock_names=bundle.stock_names,
            is_st_current=bundle.is_st_current,
            listing_dates=load_listing_dates(),
            delist_dates=load_delist_dates(),
            small_cap_mask=domain_mask,
            st_history=bundle.st_history,
            exclude_limit_on_signal=ex_lim,
        )
        # 若关闭默认可交易过滤但仍有域 mask：仅保留域约束
        if not IC_APPLY_TRADABLE and domain_mask is not None:
            tradable = domain_mask.reindex(
                index=bundle.prices.index, columns=bundle.prices.columns,
            ).fillna(False)
        parts = []
        if IC_APPLY_TRADABLE:
            if limit_mode == "strict":
                parts.append("ST/涨跌停/停牌/次新/退市")
            else:
                parts.append("ST/停牌/次新/退市（research：信号日保留涨跌停）")
        if domain_mask is not None:
            parts.append(domain_desc)
        print("  可交易池 mask: " + " + ".join(parts))
        _log_mask_coverage(tradable, rebalance_dates, "tradable")

    # checkpoint 后缀：mcap 带必须含日历指纹，禁止 resume 全市场 barra_pure_h5
    cal_fp = _calendar_fp(bundle.prices.index) if use_mcap_band else ""
    u_tag = _universe_tag(
        universe, universe_quantile, universe_mask, cap_band,
        mcap_min_yi=mcap_min_yi, mcap_max_yi=mcap_max_yi, calendar_fp=cal_fp,
    )
    ckpt_tag = "_".join(x for x in (u_tag, tmr_tag) if x)
    _set_ckpt_tag(ckpt_tag)
    if fresh:
        _clear_ckpts(period)
    if ckpt_tag or _CKPT_NEUT_TAG or _CKPT_DIR != _CKPT_DIR_DEFAULT:
        print(
            f"  [checkpoint] dir={_CKPT_DIR.as_posix()} "
            f"后缀: _{ckpt_tag} neut={_CKPT_NEUT_TAG or 'barra/legacy'} "
            f"({tmr_meta})"
        )
    if use_mcap_band:
        print(
            f"  [midcap] restan_in_universe={bool(restan_in_universe)} "
            f"min_industry_n={min_industry_n} "
            f"Size=池内 log(circ_mv)  WLS=√circ_mv 池内归一  "
            f"禁止 --barra 9 风格"
        )
    if restan_in_universe and tradable is not None:
        print("  [restan] 全市场 winsor+zscore 面板将在当日宇宙上重做截面步")

    # 先 mask 再 winsorize：分位数只在可交易样本上算；未开启截尾时仍由
    # compute_ic_series(tradable=...) 按需屏蔽，避免改变 universe 统计口径。
    if fwd_return_winsor and FWD_RETURN_WINSOR is not None:
        if tradable is not None:
            t = tradable.reindex(
                index=forward_return.index, columns=forward_return.columns,
            ).fillna(False)
            forward_return = forward_return.where(t)
        lo, hi = FWD_RETURN_WINSOR
        forward_return = winsorize_forward_return(forward_return, lower=lo, upper=hi)
        print(f"  forward_return winsorize: 截面 [{lo:.0%}, {hi:.0%}] 截尾")

    industry_map = resolve_industry_map(
        bundle.industry_map_df,
        need=industry or do_neut,
    )
    industry_panel = None
    if do_neut and neut_mode != NEUT_CONTROLS_SIZE:
        # 严格默认：必须有 PIT 行业面板；禁止静默静态 fallback（PIT 泄漏）。
        # 仅 --allow-static-industry 允许退化。
        industry_panel = require_industry_panel(allow_static=allow_static_industry)
        if industry_panel is not None:
            print(f"  [PIT] 使用 industry_map_panel.parquet（"
                  f"{len(industry_panel)} 条 / "
                  f"{industry_panel['code'].nunique()} 只股票）")
        elif industry_map is not None:
            print("  [PIT] industry_map_panel.parquet 不存在，"
                  "已启用 --allow-static-industry → 静态 industry_map"
                  "（行业哑变量可能含 PIT 泄漏）")
    elif do_neut and neut_mode == NEUT_CONTROLS_SIZE:
        print("  [neut] size-only：不加载行业面板")

    # tradable mask 已构建完成，stock_names / is_st_current 不再需要 → 释放
    bundle.stock_names = None
    bundle.is_st_current = None

    # ── 流式 IC：逐因子计算 → IC series → 释放面板 ──
    # 用 get_factor_names 轻量枚举候选因子名（不构建面板），再过滤 IC-skippable / sample
    # include_regime=False：IC 阶段不算市场/HMM regime（它们被 _is_ic_skippable 跳过，
    # 且 HMM 拟合 ~4min，省去无谓 CPU/内存）
    candidate_names = get_factor_names(**data_kwargs, include_regime=False)
    candidate_names = [n for n in candidate_names if not _is_ic_skippable(n)]
    if factor_prefix:
        prefixes = [p.strip() for p in str(factor_prefix).split(",") if p.strip()]
        before = len(candidate_names)
        candidate_names = [
            n for n in candidate_names if any(n.startswith(p) for p in prefixes)
        ]
        print(
            f"  [--factor-prefix {factor_prefix}] "
            f"{before} → {len(candidate_names)} 个因子"
        )
    if factors:
        want = [n.strip() for n in str(factors).split(",") if n.strip()]
        want_set = set(want)
        before = len(candidate_names)
        candidate_names = [n for n in candidate_names if n in want_set]
        missing_reg = sorted(want_set - set(candidate_names))
        print(
            f"  [--factors] 指定 {len(want)} 个 → registry 命中 {len(candidate_names)}"
            + (f"（未注册/跳过: {missing_reg}）" if missing_reg else "")
        )
        if not candidate_names:
            raise SystemExit("--factors 无任何可计算因子（检查名称是否在 registry）")
    if sample > 0:
        candidate_names = candidate_names[:sample]
        print(f"  [--sample {sample}] 仅计算前 {len(candidate_names)} 个因子")

    ic_workers = IC_MAX_WORKERS if workers is None else max(1, workers)
    barra_ic_workers = BARRA_IC_WORKERS if barra_workers is None else max(1, barra_workers)
    n_factors = len(candidate_names)
    mode = "串行" if ic_workers == 1 else f"最多{ic_workers}并发"
    # batch_size>0：按批调用 registry，批间 gc + 写 checkpoint（防 OOM 卡死）
    _batch = int(batch_size) if batch_size and batch_size > 0 else 0
    if _batch:
        print(f"  [--batch-size {_batch}] 分批计算，批间写 ic_series checkpoint")

    # ── Stage 2 checkpoint：因子 IC series（最贵阶段，~34min） ──
    # 检查点格式：(n_factors_expected, all_ic_full)；旧格式为裸 dict（按 len 兼容判定）。
    # 加载时若期望因子数与当前 candidate_names 数量不符：
    #   - 前缀/分批/增量：若 checkpoint 与当前候选有交集，则合并（只算缺失）；
    #   - 否则视为陈旧，丢弃重算。
    # --only-new / --factors：始终尝试加载 ic_series 并合并；
    # 下游 summary/selection 强制重算；barra_pure 增量 merge（见 Stage 5）。
    load_ic_ckpt = bool(resume or incremental)
    ic_ckpt = _load_ckpt(period, "ic_series") if load_ic_ckpt else None
    all_ic_full = None
    skip_ic_compute = False
    if ic_ckpt is not None:
        if isinstance(ic_ckpt, tuple) and len(ic_ckpt) == 2:
            stored_n, all_ic_full = ic_ckpt
        elif isinstance(ic_ckpt, dict):
            # 旧格式：裸 dict
            stored_n, all_ic_full = len(ic_ckpt), ic_ckpt
        else:
            stored_n, all_ic_full = -1, None

        cand_set = set(candidate_names)
        if all_ic_full is None:
            ic_ckpt = None
        elif incremental:
            # 增量：保留完整库；只算缺失（--only-new）或指定且未缓存（--factors）
            missing = [n for n in candidate_names if n not in all_ic_full]
            if not missing:
                print(
                    f"  [only-new] ic_series 已覆盖本轮全部 {len(candidate_names)} 个因子"
                    f"（库内 {len(all_ic_full)}）；跳过 IC 计算，重跑下游报告"
                )
                computed_names = list(all_ic_full.keys())
                n_factors = len(all_ic_full)
                skip_ic_compute = True
            else:
                print(
                    f"  [only-new] 库内已有 {len(all_ic_full)} 个 IC，"
                    f"本轮补算 {len(missing)} 个: {missing[:12]}"
                    + ("..." if len(missing) > 12 else "")
                )
                candidate_names = missing
                n_factors = len(candidate_names)
                # fall-through 只算 missing，保留 all_ic_full 完整库
        elif stored_n == n_factors and set(all_ic_full.keys()) >= cand_set and len(all_ic_full) >= n_factors:
            # 精确或超集覆盖当前候选
            all_ic_full = {k: all_ic_full[k] for k in candidate_names if k in all_ic_full}
            computed_names = list(all_ic_full.keys())
            print(f"  [resume] 跳过阶段: 计算因子+IC (从 checkpoint 加载 {len(all_ic_full)} 个因子)")
            skip_ic_compute = True
        elif factor_prefix or _batch or set(all_ic_full.keys()) & cand_set:
            # 分批/前缀/部分 checkpoint：保留已有 IC（含其它前缀轮次），只补算缺失
            # 注意：勿按 cand_set 过滤掉其它前缀已算结果，否则多轮 --factor-prefix 会互相覆盖丢失。
            missing = [n for n in candidate_names if n not in all_ic_full]
            if not missing:
                all_ic_full_view = {k: all_ic_full[k] for k in candidate_names}
                computed_names = list(all_ic_full_view.keys())
                n_factors = len(computed_names)
                print(
                    f"  [resume] 跳过阶段: 计算因子+IC "
                    f"(checkpoint 已覆盖本轮 {n_factors} 个；库内共 {len(all_ic_full)} 个)"
                )
                # 前缀跑：本轮候选已齐则收窄视图；完整合并请用 --only-new（无 prefix）
                all_ic_full = all_ic_full_view
                skip_ic_compute = True
            else:
                print(
                    f"  [resume] 分批/前缀: 库内已有 {len(all_ic_full)} 个 IC，"
                    f"本轮补算 {len(missing)}/{len(candidate_names)} 个"
                )
                candidate_names = missing
                n_factors = len(candidate_names)
        else:
            print(
                f"  [resume] 警告: ic_series checkpoint 陈旧/不完整"
                f"（存储期望 {stored_n} 因子 vs 当前 {n_factors} 因子），"
                f"丢弃并重算。"
            )
            _clear_ckpts(period)
            all_ic_full = None
    elif incremental:
        print(
            "  [only-new] 未找到 ic_series checkpoint，"
            "将按当前候选全量计算（等同首次跑）"
        )
    if not skip_ic_compute:
        print(f"计算因子+IC（{n_factors}个因子，流式，{mode}，ICIR std ddof=0）...")
        # 因子面板缓存探测：universe 不进指纹，换宇宙应仍 HIT 全市场面板
        try:
            from factors.factor_cache import (
                FACTOR_CACHE_DIR,
                build_input_signature,
                probe_factor_cache,
            )
            _sig = build_input_signature(data_kwargs)
            _hits, _misses = probe_factor_cache(candidate_names, _sig)
            print(
                f"  [factor_panels] HIT={len(_hits)} MISS={len(_misses)} "
                f"dir={FACTOR_CACHE_DIR}"
                + (
                    f"  （首次将落盘；复跑同输入应全 HIT，universe 仅 mask）"
                    if _misses
                    else "  （复用已有全市场面板，universe 仅 mask）"
                )
            )
        except Exception as e:
            print(f"  [factor_panels] 缓存探测跳过: {e}")
        if all_ic_full is None:
            all_ic_full = {}
        # 分批/前缀/增量续跑时 all_ic_full 可能已含其它轮次结果，勿清空
        computed_names = list(all_ic_full.keys())

        def _compute_ic_for_panel(name, panel):
            panel_f32 = _to_float32_panel(panel)
            if restan_in_universe and tradable is not None:
                from research.ic.universe import restan_within_mask
                panel_f32 = restan_within_mask(panel_f32, tradable)
            ic = compute_ic_series(panel_f32, forward_return, tradable=tradable)
            all_ic_full[name] = ic
            computed_names.append(name)

        # 分批列表：默认整表一批；--batch-size 时切块
        name_batches: list[list[str]]
        if _batch and ic_workers == 1:
            name_batches = [
                candidate_names[i : i + _batch]
                for i in range(0, len(candidate_names), _batch)
            ]
        else:
            name_batches = [candidate_names]

        if ic_workers == 1:
            done = 0
            target = sum(len(b) for b in name_batches)
            for bi, batch_names in enumerate(name_batches, 1):
                if len(name_batches) > 1:
                    print(
                        f"  批次 {bi}/{len(name_batches)}："
                        f"{len(batch_names)} 个因子",
                        flush=True,
                    )
                for name, panel in _filter_none_emit(
                    iter_factor_registry(
                        factor_names=set(batch_names),
                        include_regime=False,
                        **data_kwargs,
                    )
                ):
                    if _is_ic_skippable(name):
                        del panel
                        continue
                    _compute_ic_for_panel(name, panel)
                    del panel
                    done += 1
                    if done % 10 == 0 or done == target:
                        print(f"  进度: {done}/{target}", flush=True)
                        gc.collect()
                if len(name_batches) > 1:
                    _save_ckpt(period, "ic_series", (len(all_ic_full), all_ic_full))
                    gc.collect()
            n_factors = len(all_ic_full)
        else:
            # 并行模式：仍需批量构建 registry 供线程池（线程间共享 GIL 但不共享 DataFrame
            # 拷贝；IC_MAX_WORKERS 默认 1，此分支仅在用户显式 --workers >1 时触发）。
            # 为保留并行能力，回退到一次性构建（内存代价同改造前）。
            # 有 --batch-size 时仍按批构建，降低峰值。
            from factors.factor import get_factor_registry
            for bi, batch_names in enumerate(name_batches, 1):
                if len(name_batches) > 1:
                    print(
                        f"  批次 {bi}/{len(name_batches)}（并行）："
                        f"{len(batch_names)} 个因子",
                        flush=True,
                    )
                registry_bulk = get_factor_registry(
                    factor_names=set(batch_names), include_regime=False, **data_kwargs
                )
                registry_bulk = {k: _to_float32_panel(v) for k, v in registry_bulk.items()}

                def _compute_one(name_fac):
                    name, factor = name_fac
                    return name, compute_ic_series(factor, forward_return, tradable=tradable)

                for name, ic_full in run_bounded_parallel(
                    _compute_one, list(registry_bulk.items()), ic_workers, progress_every=10
                ):
                    all_ic_full[name] = ic_full
                    computed_names.append(name)
                del registry_bulk
                gc.collect()
                if len(name_batches) > 1:
                    _save_ckpt(period, "ic_series", (len(all_ic_full), all_ic_full))
            n_factors = len(all_ic_full)
        # 存 (n_factors_expected, all_ic_full)：resume 时校验完整性，避免静默使用部分结果
        _save_ckpt(period, "ic_series", (n_factors, all_ic_full))

    if incremental:
        # 合并进完整库后必须重算 summary/selection；barra_pure 保留供增量 merge
        _clear_downstream_ckpts(period, keep_barra_pure=True)

    t0 = _log_phase(f"计算因子+IC（{len(all_ic_full)}个）", t0)

    # 流式 IC 完成：此时仅持有 IC series（每因子一条 Series，~几 KB），不持有任何因子面板。
    # 释放 data_kwargs 中下游重建不再需要的辅助数据（masks 仍需留给 limit 因子重建，
    # prices/open_/clean_ret/financial/market_prices/industry_map 等下游仍需，保留）。
    # 注：margin/moneyflow/northbound/institution 仅特定 alpha2 因子重建需要；
    # 若 summary 中无对应因子，build_factor_registry_subset 会自动跳过，故保留无害。

    # ── Stage 3 checkpoint：summary_df + all_ic（含成本富化） ──
    summary_ckpt = _load_ckpt(period, "summary") if resume_downstream else None
    if summary_ckpt is not None:
        summary_df, all_ic = summary_ckpt
        print(f"  [resume] 跳过阶段: 汇总+成本 (从 checkpoint 加载 {len(summary_df)} 行)")
    else:
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

        # ── enrich_summary_with_cost：流式算 turnover（避免持有全部面板） ──
        turnovers = _stream_turnovers(summary_df.index, data_kwargs, rebalance_dates)
        summary_df = summary_df.copy()
        summary_df["rank_turnover"] = pd.Series(turnovers)
        summary_df["IC_after_cost"] = summary_df.apply(
            lambda r: estimate_ic_after_cost(r["IC均值"], r["rank_turnover"]), axis=1
        )

        if top > 0:
            summary_df = summary_df.head(top)
            all_ic = {k: v for k, v in all_ic.items() if k in summary_df.index}

        _save_ckpt(period, "summary", (summary_df, all_ic))

    print_summary(summary_df)

    stab = stability_report(all_ic)
    if not stab.empty:
        print(f"\n  IC 稳定性（滚动σ / 同向年份占比）:")
        print(stab.head(10).to_string())

    # ── Stage 4 checkpoint：年度 IC 表 ──
    yearly_ckpt = _load_ckpt(period, "yearly") if resume_downstream else None
    if yearly_ckpt is not None:
        yearly_df = yearly_ckpt
        print(f"  [resume] 跳过阶段: 年度IC (从 checkpoint 加载)")
    else:
        yearly_rows = []
        all_years = sorted({y for ic in all_ic_full.values() for y in ic.index.year})
        for name, ic in all_ic_full.items():
            by_year = ic_by_year(ic)
            row = {"因子": name, **{y: by_year.get(y, np.nan) for y in all_years}}
            yearly_rows.append(row)
        yearly_df = pd.DataFrame(yearly_rows).set_index("因子").loc[summary_df.index]
        _save_ckpt(period, "yearly", yearly_df)
    print_yearly(yearly_df)

    # raw-select 通道在 Barra 之前分叉，此处先初始化默认值
    pure_ic_means: dict = {}
    pure_ic_series: dict = {}
    barra_names_used: list = []

    # ════════════════════════════════════════════════════════════════════════
    # --raw-select 快速通道：跳过所有需要因子面板的阶段（Barra / decay /
    # corr / select_factors 截面去重 / Gram-Schmidt），仅用已 checkpoint 的
    # summary_df + all_ic 做阈值门 + IC 序列相关性去重。秒级出结果。
    # 代价：放弃截面 spearman 去重与正交化（对"先跑通管线拿 baseline"非必需）。
    # ════════════════════════════════════════════════════════════════════════
    if raw_select:
        if barra or decay or corr or gram_schmidt:
            print("  [raw-select] 忽略 --barra/--decay/--corr/--gram-schmidt "
                  "（均需因子面板，raw-select 模式跳过）")
        sel_ckpt = _load_ckpt(period, "selection") if resume_downstream else None
        categories: dict = {}
        labels: dict = {}
        sparse_kept: list = []
        emerging_kept: list = []
        if sel_ckpt is not None:
            # 兼容旧 checkpoint (kept, exclusions) 与新 (result_dict,)
            if isinstance(sel_ckpt, tuple) and len(sel_ckpt) == 2 and isinstance(sel_ckpt[0], list):
                kept, exclusions = sel_ckpt
            elif isinstance(sel_ckpt, dict):
                kept = sel_ckpt.get("dense_kept", [])
                sparse_kept = sel_ckpt.get("sparse_kept", [])
                emerging_kept = sel_ckpt.get("emerging_kept", [])
                exclusions = sel_ckpt.get("exclusions", {})
                categories = sel_ckpt.get("categories", {})
                labels = sel_ckpt.get("labels", {})
                if not emerging_kept and categories:
                    emerging_kept = [
                        n for n, c in categories.items() if c == "新兴因子"
                    ]
            else:
                kept, exclusions = sel_ckpt[0], sel_ckpt[1]
            print(f"  [resume] 跳过阶段: 因子筛选-raw (从 checkpoint 加载 {len(kept)} 个保留)")
        else:
            print(f"\n因子筛选 (raw 多轨) — {len(summary_df)} 个因子...")
            # raw 无因子面板：稀疏轨无法算 payoff_hit，缺面板的稀疏因子会被剔除
            # （用户主路径不用 --raw-select；生产请走完整面板路径）
            print("  [raw-select] 稀疏轨：无因子面板，无法算 payoff_hit，"
                  "缺面板的稀疏因子将跳过/剔除")
            summary_for_sel = overlay_long_share(
                summary_df, long_share_csv=long_share_csv,
            )
            sel = select_factors_multi_track(
                summary_for_sel,
                all_ic=all_ic,
                pure_ic_means=pure_ic_means or None,
                pure_ic_series=pure_ic_series or None,
                ic_threshold=ic_threshold,
                icir_threshold=icir_threshold,
                nw_t_threshold=t_threshold if use_nw_t else None,
                t_threshold=t_threshold,
                use_fdr=use_fdr,
                regime_consistency_threshold=regime_consistency,
                rolling_icir_threshold=rolling_icir,
                worst_period_ic_threshold=worst_period_ic,
                corr_dedup=corr_dedup,
                corr_threshold=corr_threshold,
                raw_mode=True,
                hold_period=period,
                min_long_share=min_long_share,
                enable_decay_gate=enable_decay_gate,
                decay_recent_months=decay_recent_months,
                decay_retention_min=decay_retention_min,
                decay_retention_min_sparse=decay_retention_min_sparse,
                decay_recent_icir_max=decay_recent_icir_max,
                decay_recent_ic_max=decay_recent_ic_max,
                enable_reversal_label=enable_reversal_label,
                reversal_months=reversal_months,
                reversal_frac=reversal_frac,
                reversal_abs_ic=reversal_abs_ic,
                enable_emerging=enable_emerging,
                emerging_lookback=emerging_lookback,
                emerging_recent_icir=emerging_recent_icir,
                emerging_recent_ic=emerging_recent_ic,
                emerging_fdr_alpha=emerging_fdr_alpha,
                emerging_lift_min=emerging_lift_min,
                emerging_holdout_months=emerging_holdout_months,
                emerging_asof=emerging_asof,
                emerging_require_trend=emerging_require_trend,
                emerging_trend_months=emerging_trend_months,
                emerging_trend_segments=emerging_trend_segments,
                emerging_trend_eps=emerging_trend_eps,
                enable_sparse_track=enable_sparse_track,
                sparse_ic_threshold=sparse_ic_threshold,
                sparse_icir_threshold=sparse_icir_threshold,
                sparse_win_rate_min=sparse_win_rate_min,
                sparse_payoff_min=sparse_payoff_min,
                sparse_require_ic=sparse_require_ic,
                sparse_corr_threshold=sparse_corr_threshold,
                forward_return=None,
                tradable=tradable,
            )
            kept, exclusions = sel.dense_kept, sel.exclusions
            sparse_kept = sel.sparse_kept
            emerging_kept = sel.emerging_kept
            categories, labels = sel.categories, sel.labels
            _save_ckpt(period, "selection", {
                "dense_kept": kept,
                "sparse_kept": sparse_kept,
                "emerging_kept": emerging_kept,
                "exclusions": exclusions,
                "categories": categories,
                "labels": labels,
            })
        print_selection_result(
            kept, exclusions, categories=categories, sparse_kept=sparse_kept,
            emerging_kept=emerging_kept, labels=labels,
        )
        gs_meta = None
        # 跳到 save JSON（跳过 Stage 5/6/7panel/8）
        if save:
            ind_ic_df = pd.DataFrame()
            rb_idx = forward_return.index.intersection(rebalance_dates)
            universe_size = (
                int(round(float(forward_return.loc[rb_idx].notna().sum(axis=1).mean())))
                if len(rb_idx) > 0 else 0
            )
            ic_series_length = int(len(next(iter(all_ic_full.values())))) if all_ic_full else 0
            sample_period = (
                [str(forward_return.index.min()), str(forward_return.index.max())]
                if len(forward_return.index) else ["", ""]
            )
            nw_lag = int(_default_nw_lags(ic_series_length)) if ic_series_length > 0 else 0
            meta = {
                "universe_size": universe_size,
                "universe": domain_desc,
                "universe_tag": u_tag or "all",
                "ic_series_length": ic_series_length,
                "sample_period": sample_period,
                "barra_factors_used": [],
                "industry_reference": INDUSTRY_REFERENCE,
                "nw_lag": nw_lag,
                "config_snapshot": {
                    "IC_CLIP": IC_CLIP,
                    "IC_CORR_METHOD": IC_CORR_METHOD,
                    "IC_RANK_METHOD": IC_RANK_METHOD,
                    "IC_MIN_LISTING_DAYS": IC_MIN_LISTING_DAYS,
                    "barra": False,
                    "use_fdr": use_fdr,
                    "t_threshold": t_threshold,
                    "corr_dedup": corr_dedup,
                    "corr_threshold": corr_threshold,
                    "sparse_corr_threshold": sparse_corr_threshold,
                    "gram_schmidt": False,
                    "use_nw_t": use_nw_t,
                    "enable_decay_gate": enable_decay_gate,
                    "enable_emerging": enable_emerging,
                    "enable_sparse_track": enable_sparse_track,
                    "decay_recent_months": decay_recent_months,
                    "reversal_frac": reversal_frac,
                    "reversal_abs_ic": reversal_abs_ic,
                    "emerging_lookback": emerging_lookback,
                    "emerging_recent_icir": emerging_recent_icir,
                    "emerging_fdr_alpha": emerging_fdr_alpha,
                    "emerging_lift_min": emerging_lift_min,
                    "universe": universe,
                    "universe_quantile": universe_quantile,
                    "universe_mask": universe_mask,
                    "cap_band": cap_band,
                    "mcap_min_yi": mcap_min_yi,
                    "mcap_max_yi": mcap_max_yi,
                    "restan_in_universe": bool(restan_in_universe),
                    "min_industry_n": min_industry_n,
                    **tmr_meta,
                },
                "orthogonalization": None,
                "selection_mode": "raw_ic_series_corr_multitrack",
                **tmr_meta,
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
                sparse_factors=sparse_kept,
                emerging_factors=emerging_kept,
                categories=categories,
                labels=labels,
                name_suffix="_".join(x for x in (save_suffix, u_tag) if x),
            )
            print(f"\n结果已保存至 {json_path.parent}  ({json_path.name})")
        _maybe_plot_rolling_ic(
            enabled=plot_rolling_ic_flag,
            period=period,
            all_ic=all_ic,
            pure_ic_series=None,
            summary_df=summary_df,
            kept=kept,
            sparse_kept=sparse_kept,
            top_n=plot_rolling_ic_top_n,
            window=plot_rolling_ic_window,
            names_csv=plot_rolling_ic_names,
            source=plot_rolling_ic_source,
            tag=u_tag,
        )
        _log_phase("总耗时", t_run)
        gc.collect()
        return summary_df, yearly_df, all_ic, pd.DataFrame()

    # ── decay / corr / industry：流式 LazyRegistry（cache=False），不再重建全子集 ──
    # 原实现 build_factor_registry_subset(set(all_ic_full.keys())) 会同时构建
    # 60 个因子面板 ≈ 2.88GB，是 --decay/--corr/--industry 的内存 peak 元凶。
    # 改为传 LazyRegistry 给下游，下游函数逐因子 __getitem__ 加载 → 用完即释
    # （峰值 = 1 个面板 ~48MB）。
    decay_df: pd.DataFrame | None = None
    if decay:
        print("\n计算IC衰减...")
        decay_registry = _LazyFactorRegistry(
            set(all_ic_full.keys()), data_kwargs, cache=False
        )
        decay_df = ic_decay_table(
            decay_registry, bundle.prices, bundle.open_,
            tradable=tradable, masks=bundle.masks,
            names=list(all_ic_full.keys()),
            apply_exec_mask=ex_exec,
        )
        print_decay(decay_df)
        del decay_registry
        gc.collect()

    if corr:
        print("\n因子相关矩阵...")
        corr_registry = _LazyFactorRegistry(
            set(all_ic_full.keys()), data_kwargs, cache=False
        )
        corr_mat = factor_corr_matrix(
            corr_registry, bundle.prices, names=list(all_ic_full.keys())
        )
        del corr_registry
        gc.collect()
        if not corr_mat.empty:
            if plot:
                plot_corr_matrix(corr_mat)
            else:
                print(corr_mat.round(3).to_string())

    pure_ic_means = {}
    pure_ic_series = {}
    barra_names_used: list = []
    quantile_df = pd.DataFrame()
    # 分位分解仅在中性化路径有意义；raw 时忽略开关
    do_quantile = bool(do_neut) and bool(quantile_decomp)
    skip_neut_names: set[str] = set()
    if do_neut:
        from factors.sparse_factors import partition_sparse
        from factors.special_factors import should_skip_neutralize
        _dense_n, _sparse_n = partition_sparse(list(summary_df.index))
        skip_neut_names = set(_sparse_n) | {
            n for n in summary_df.index if should_skip_neutralize(n)
        }
        print(
            f"  [neut-controls] {neut_mode}；"
            f"跳过残差化 {len(skip_neut_names)} 个（sparse/special/size pack）"
        )
        # ── Stage 5 checkpoint：纯 IC（仅在成功时保存，失败则 resume 重试） ──
        # 格式演进：
        #   2-tuple → 无 series，重算
        #   3-tuple (means, names, series) → 无分位表；若需要 quantile 则重算
        #   4-tuple (+ quantile_df) → 完整（无版本元数据）
        #   5-tuple (+ meta{barra_version}) → 可增量 merge；版本不匹配则全量
        # 增量（--only-new/--factors）：resume_downstream=False，但仍加载 barra_pure
        # 做指纹校验 + 只算缺失因子并 merge；--fresh / 版本变 / 缺序列 → 全量。
        load_barra_ckpt = bool(resume_downstream or incremental)
        barra_ckpt = _load_ckpt(period, "barra_pure") if load_barra_ckpt else None
        barra_reuse = False
        barra_missing: list[str] = []
        if barra_ckpt is not None:
            unpacked = unpack_barra_pure_ckpt(barra_ckpt)
            if unpacked is None:
                print(
                    "  [resume] barra_pure checkpoint 无 pure IC 序列，将重算 Barra"
                    "（新兴/衰减近窗需与全样本纯因子同口径）"
                )
                barra_ckpt = None
            else:
                pure_ic_means, barra_names_used, pure_ic_series, quantile_df, bmeta = (
                    unpacked
                )
                ver_ok = barra_pure_version_ok(
                    bmeta, for_incremental=bool(incremental),
                )
                stored_nc = (bmeta or {}).get("neut_controls", "barra")
                stored_u = (bmeta or {}).get("universe_tag", "")
                if str(stored_nc) != str(neut_mode or "barra"):
                    print(
                        f"  [resume] barra_pure neut_controls 不匹配 "
                        f"（stored={stored_nc!r}, current={neut_mode!r}），将重算"
                    )
                    ver_ok = False
                if _CKPT_TAG and str(stored_u) != str(_CKPT_TAG):
                    if "mcap" in (_CKPT_TAG or "") or "mcap" in str(stored_u):
                        print(
                            f"  [resume] barra_pure universe_tag 不匹配 "
                            f"（stored={stored_u!r}, current={_CKPT_TAG!r}），"
                            f"禁止 resume 全市场/其它宇宙 ckpt"
                        )
                        ver_ok = False
                if not ver_ok:
                    stored = (bmeta or {}).get("barra_version")
                    print(
                        f"  [{'only-new' if incremental else 'resume'}] "
                        f"barra_pure 指纹失效"
                        f"（stored={stored!r}, current={barra_pure_cache_version()!r}），"
                        f"将全量重算 Barra"
                    )
                    barra_ckpt = None
                    pure_ic_means, pure_ic_series, quantile_df = {}, {}, pd.DataFrame()
                    barra_names_used = []
                else:
                    need_q_recompute = do_quantile and (
                        quantile_df is None or quantile_df.empty
                    )
                    if need_q_recompute:
                        print(
                            "  [resume] barra_pure checkpoint 无分位分解表，将重算 Barra"
                            "（含 Q1/Q5 多头空头贡献）"
                        )
                        barra_ckpt = None
                        pure_ic_means, pure_ic_series, quantile_df = (
                            {}, {}, pd.DataFrame()
                        )
                        barra_names_used = []
                    elif incremental:
                        barra_missing = missing_barra_pure_names(
                            summary_df.index, pure_ic_series,
                        )
                        if not barra_missing:
                            barra_reuse = True
                            print(
                                f"  [only-new] barra_pure 已覆盖本轮全部 "
                                f"{len(summary_df)} 个因子"
                                f"（库内 pure {len(pure_ic_series)}）；跳过 Barra"
                            )
                            if pure_ic_means:
                                print_barra_comparison(summary_df, pure_ic_means)
                            if do_quantile and not quantile_df.empty:
                                print_quantile_ls(
                                    quantile_df, y_mode=quantile_y_mode,
                                )
                        else:
                            print(
                                f"  [only-new] barra_pure 库内 {len(pure_ic_series)} 条，"
                                f"本轮补算 pure {len(barra_missing)} 个: "
                                f"{barra_missing[:12]}"
                                + ("..." if len(barra_missing) > 12 else "")
                            )
                    else:
                        # 普通 resume：整包复用
                        barra_reuse = True
                        print(
                            f"  [resume] 跳过阶段: Barra纯IC "
                            f"(从 checkpoint 加载 {len(pure_ic_means)} 个均值 / "
                            f"{len(pure_ic_series)} 条序列"
                            + (
                                f" / 分位表 {len(quantile_df)} 行"
                                if do_quantile and not quantile_df.empty
                                else ""
                            )
                            + ")"
                        )
                        if pure_ic_means:
                            print_barra_comparison(summary_df, pure_ic_means)
                        if do_quantile and not quantile_df.empty:
                            print_quantile_ls(
                                quantile_df, y_mode=quantile_y_mode,
                            )

        if not barra_reuse:
            q_msg = (
                f" + Q1/Q5 多空贡献(y={quantile_y_mode})" if do_quantile else ""
            )
            # 增量补区：只对缺失名跑 residualize+pure IC，再 merge
            partial = bool(incremental and barra_missing and pure_ic_series)
            target_index = (
                pd.Index(barra_missing) if partial else summary_df.index
            )
            mode_msg = (
                f"（增量 {len(barra_missing)} 个新因子）" if partial else ""
            )
            print(f"\n计算 Barra 纯因子 IC{q_msg}{mode_msg}...")
            try:
                # Barra 逐因子 OLS：cache=True 缓存面板（frac_diff 修复后内存充裕，
                # 2.86GB 缓存 + 2.2GB base ≈ 5GB，32GB 机器安全；no-cache 会因 Alpha101
                # __getitem__ 每次重算全部 10 个 WQ 因子导致极慢）
                names_for_reg = set(target_index)
                barra_registry = _LazyFactorRegistry(
                    names_for_reg, data_kwargs, cache=True
                )
                new_names_sink: list = []
                new_means, new_series, new_qdf = run_barra_pure_ic(
                    registry=barra_registry,
                    summary_index=target_index,
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
                    names_sink=new_names_sink,
                    clean_ret=bundle.clean_ret,
                    prices_raw=bundle.prices_raw,
                    quantile_decomp=do_quantile,
                    quantile_y_mode=quantile_y_mode,
                    # Size=log(流通市值)、Liquidity=换手率、WLS 权重=√市值
                    circ_mv=bundle.circ_mv,
                    total_mv=bundle.total_mv,
                    turnover_rate=bundle.turnover_rate,
                    amount=bundle.amount,
                    neut_controls=neut_mode or "barra",
                    skip_names=skip_neut_names,
                    membership_mask=tradable,
                    min_industry_n=min_industry_n,
                    restan_in_universe=bool(restan_in_universe),
                )
                del barra_registry
                gc.collect()
                if partial:
                    pure_ic_means, pure_ic_series, quantile_df = (
                        merge_barra_pure_results(
                            pure_ic_means,
                            pure_ic_series,
                            quantile_df,
                            new_means,
                            new_series,
                            new_qdf,
                        )
                    )
                    if new_names_sink:
                        barra_names_used = list(new_names_sink)
                    print(
                        f"  [only-new] barra_pure merge 完成："
                        f"库内 {len(pure_ic_series)} 条 pure"
                        f"（本轮新增 {len(new_series)}）"
                    )
                else:
                    pure_ic_means, pure_ic_series, quantile_df = (
                        new_means, new_series, new_qdf,
                    )
                    barra_names_used = list(new_names_sink)
                if pure_ic_means:
                    _save_ckpt(
                        period, "barra_pure",
                        pack_barra_pure_ckpt(
                            pure_ic_means,
                            barra_names_used,
                            pure_ic_series,
                            quantile_df,
                            neut_controls=neut_mode or "barra",
                            universe_tag=_CKPT_TAG,
                        ),
                    )
                    print_barra_comparison(summary_df, pure_ic_means)
                    if do_quantile and not quantile_df.empty:
                        print_quantile_ls(quantile_df, y_mode=quantile_y_mode)
                else:
                    print("Barra 因子计算失败，跳过")
            except Exception as e:
                print(f"Barra 分析出错: {e}")
                import traceback
                traceback.print_exc()

    # barra：用 pure IC 覆盖 summary 的 t/NW_t，供筛选门控与 JSON 落盘同口径
    if pure_ic_series:
        summary_df = overlay_pure_t_stats(summary_df, pure_ic_series)

    # 分位 long_share（符号对齐）并入 summary，供稠密门在 corr-dedup 前使用
    summary_df = overlay_long_share(
        summary_df,
        quantile_df=quantile_df if (do_quantile and not quantile_df.empty) else None,
        long_share_csv=long_share_csv,
    )

    ind_ic_df = pd.DataFrame()
    if industry and industry_map is not None:
        print("\n分行业 IC...")
        # 流式 LazyRegistry：逐因子加载算各行业 IC，峰值 = 1 个面板
        ind_registry = _LazyFactorRegistry(
            set(summary_df.index), data_kwargs, cache=True
        )
        ind_ic_df = compute_ic_industry(
            ind_registry,
            forward_return,
            industry_map,
            tradable=tradable,
            names=list(summary_df.index),
        )
        del ind_registry
        gc.collect()

    if plot:
        plot_rolling_ic(all_ic, period)

    # ── Stage 7 checkpoint：多轨筛选（稠密 + 稀疏；新兴/衰减/逆转仅标注）──
    # 注：selection checkpoint 对阈值敏感（--t-threshold/--regime-consistency 等），
    # 改阈值后请用 clean run（不带 --resume）以免使用陈旧筛选结果。
    sel_ckpt = _load_ckpt(period, "selection") if resume_downstream else None
    categories: dict = {}
    labels: dict = {}
    sparse_kept: list = []
    emerging_kept: list = []
    if sel_ckpt is not None:
        if isinstance(sel_ckpt, dict):
            kept = sel_ckpt.get("dense_kept", [])
            sparse_kept = sel_ckpt.get("sparse_kept", [])
            emerging_kept = sel_ckpt.get("emerging_kept", [])
            exclusions = sel_ckpt.get("exclusions", {})
            categories = sel_ckpt.get("categories", {})
            labels = sel_ckpt.get("labels", {})
            if not emerging_kept and categories:
                emerging_kept = [
                    n for n, c in categories.items() if c == "新兴因子"
                ]
        else:
            kept, exclusions = sel_ckpt
        print(f"  [resume] 跳过阶段: 因子筛选 (从 checkpoint 加载 {len(kept)} 个保留)")
    else:
        # ── 多轨：稠密（panel corr 去重）+ 稀疏独立轨道 + 新兴观察 ──
        cm = corr_method or IC_CORR_METHOD
        select_registry = _LazyFactorRegistry(
            set(all_ic_full.keys()), data_kwargs, cache=False
        )
        sel = select_factors_multi_track(
            summary_df,
            select_registry,
            all_ic=all_ic,
            pure_ic_means=pure_ic_means or None,
            pure_ic_series=pure_ic_series or None,
            ic_threshold=ic_threshold,
            icir_threshold=icir_threshold,
            corr_method=cm,
            rebalance_dates=rebalance_dates,
            nw_t_threshold=t_threshold if use_nw_t else None,
            t_threshold=t_threshold,
            use_fdr=use_fdr,
            regime_consistency_threshold=regime_consistency,
            rolling_icir_threshold=rolling_icir,
            worst_period_ic_threshold=worst_period_ic,
            corr_dedup=corr_dedup,
            corr_threshold=corr_threshold,
            raw_mode=False,
            hold_period=period,
            min_long_share=min_long_share,
            enable_decay_gate=enable_decay_gate,
            decay_recent_months=decay_recent_months,
            decay_retention_min=decay_retention_min,
            decay_retention_min_sparse=decay_retention_min_sparse,
            decay_recent_icir_max=decay_recent_icir_max,
            decay_recent_ic_max=decay_recent_ic_max,
            enable_reversal_label=enable_reversal_label,
            reversal_months=reversal_months,
            reversal_frac=reversal_frac,
            reversal_abs_ic=reversal_abs_ic,
            enable_emerging=enable_emerging,
            emerging_lookback=emerging_lookback,
            emerging_recent_icir=emerging_recent_icir,
            emerging_recent_ic=emerging_recent_ic,
            emerging_fdr_alpha=emerging_fdr_alpha,
            emerging_lift_min=emerging_lift_min,
            emerging_holdout_months=emerging_holdout_months,
            emerging_asof=emerging_asof,
            emerging_require_trend=emerging_require_trend,
            emerging_trend_months=emerging_trend_months,
            emerging_trend_segments=emerging_trend_segments,
            emerging_trend_eps=emerging_trend_eps,
            enable_sparse_track=enable_sparse_track,
            sparse_ic_threshold=sparse_ic_threshold,
            sparse_icir_threshold=sparse_icir_threshold,
            sparse_win_rate_min=sparse_win_rate_min,
            sparse_payoff_min=sparse_payoff_min,
            sparse_require_ic=sparse_require_ic,
            sparse_corr_threshold=sparse_corr_threshold,
            forward_return=forward_return,
            tradable=tradable,
        )
        kept, exclusions = sel.dense_kept, sel.exclusions
        sparse_kept = sel.sparse_kept
        emerging_kept = sel.emerging_kept
        categories, labels = sel.categories, sel.labels
        select_registry.release_cache()
        del select_registry
        gc.collect()
        _save_ckpt(period, "selection", {
            "dense_kept": kept,
            "sparse_kept": sparse_kept,
            "emerging_kept": emerging_kept,
            "exclusions": exclusions,
            "categories": categories,
            "labels": labels,
        })
    print_selection_result(
        kept, exclusions, categories=categories, sparse_kept=sparse_kept,
        emerging_kept=emerging_kept, labels=labels,
    )

    # ── Gram-Schmidt 正交精筛（可选）：从 corr 去重后的 kept 出发，
    #    按 |ICIR| 降序迭代，每个因子扣除已选因子的截面成分，残差 IC 仍显著才保留。
    #    机构量化标准做法：从候选池正交出 ≤max_factors 个独立因子作为 ML 输入，
    #    避免 51/60 因子全进白名单导致的过拟合。
    gs_meta: dict | None = None
    # 双轨制：kept 保持 pre-GS 完整集（给 ML），gs_selected 单独留作 dynamic 正交集
    gs_selected: list | None = None
    if gram_schmidt and kept:
        # ── Stage 8 checkpoint：Gram-Schmidt 正交精筛 ──
        # 同样对阈值敏感（gs_ic_threshold/gs_icir_threshold/max_factors），
        # 改参数后请用 clean run。
        gs_ckpt = _load_ckpt(period, "gramschmidt") if resume_downstream else None
        if gs_ckpt is not None:
            # 4-tuple：kept_pre_gs, gs_selected, exclusions, gs_meta
            kept, gs_selected, exclusions, gs_meta = gs_ckpt
            print(f"  [resume] 跳过阶段: Gram-Schmidt (从 checkpoint 加载 pre-GS {len(kept)} / 正交 {len(gs_selected)} 个)")
        else:
            print(f"\nGram-Schmidt 正交选择（候选池 = corr 去重后 {len(kept)} 个 → ≤{max_factors}）...")
            # cache=False：每个候选因子仅加载一次（选中→本地持有副本；未选中→即释），
            # 内存峰值 = 1 个因子面板 + |selected| 个面板副本。
            gs_registry = _LazyFactorRegistry(
                set(kept), data_kwargs, cache=False
            )
            # 精筛门槛比预筛更松（残差 IC 本就较小），保留有增量信号的因子
            gs_selected, gs_exclusions = gram_schmidt_select(
                summary_df=summary_df.loc[kept],
                factor_registry=gs_registry,
                forward_return=forward_return,
                rebalance_dates=rebalance_dates,
                tradable=tradable,
                max_factors=max_factors,
                ic_threshold=gs_ic_threshold,
                icir_threshold=gs_icir_threshold,
                pre_filter_ic=0.0,      # 预筛已在 select_factors 完成，此处不重复
                pre_filter_icir=0.0,
                pre_filter_t=0.0,
                use_nw_t=use_nw_t,
                pure_ic_means=pure_ic_means or None,
                verbose=True,
            )
            gs_registry.release_cache()
            del gs_registry
            gc.collect()

            # 合并 exclusions：corr 阶段已剔除的保留，Gram-Schmidt 阶段新剔除的追加
            # （gs_exclusions 仅含 kept 中被 Gram-Schmidt 剔除的）
            exclusions.update(gs_exclusions)
            gs_meta = {
                "method": "gram_schmidt",
                "max_factors": max_factors,
                "ic_threshold": gs_ic_threshold,
                "icir_threshold": gs_icir_threshold,
                "pre_filter_pool_size": len(kept),
                "selected_count": len(gs_selected),
                "rejected_in_orthogonalization": len(gs_exclusions),
            }
            print(f"\n  Gram-Schmidt 结果：正交集 {len(gs_selected)}/{len(kept)} → dynamic 白名单")
            print(f"  ML 白名单（pre-GS 完整 pure-IC 集）：{len(kept)} 个")
            # 不再 kept = gs_selected；kept 保持 pre-GS 完整集给 ML，gs_selected 给 dynamic
            _save_ckpt(period, "gramschmidt", (kept, gs_selected, exclusions, gs_meta))
        # 双轨打印：先 ML 完整集，再 dynamic 正交集
        print_selection_result(
            kept, exclusions, categories=categories, sparse_kept=sparse_kept,
            emerging_kept=emerging_kept, labels=labels,
        )
        if gs_selected is not None:
            print(f"\n  [dynamic 正交集] {len(gs_selected)} 个因子：")
            for n in gs_selected:
                print(f"    [ORTH] {n}")

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
            "universe": domain_desc,
            "universe_tag": u_tag or "all",
            "ic_series_length": ic_series_length,
            "sample_period": sample_period,
            "barra_factors_used": barra_names_used,
            "neut_controls": neut_mode or "raw",
            "industry_reference": INDUSTRY_REFERENCE,
            "nw_lag": nw_lag,
            "config_snapshot": {
                "IC_CLIP": IC_CLIP,
                "IC_CORR_METHOD": IC_CORR_METHOD,
                "IC_RANK_METHOD": IC_RANK_METHOD,
                "IC_MIN_LISTING_DAYS": IC_MIN_LISTING_DAYS,
                "barra": bool(do_neut and neut_mode == "barra"),
                "neut_controls": neut_mode or "raw",
                "use_fdr": use_fdr,
                "t_threshold": t_threshold,
                "corr_dedup": corr_dedup,
                "corr_threshold": corr_threshold,
                "sparse_corr_threshold": sparse_corr_threshold,
                "gram_schmidt": bool(gram_schmidt),
                "use_nw_t": use_nw_t,
                "universe": universe,
                "universe_quantile": universe_quantile,
                "universe_mask": universe_mask,
                "cap_band": cap_band,
                "mcap_min_yi": mcap_min_yi,
                "mcap_max_yi": mcap_max_yi,
                "restan_in_universe": bool(restan_in_universe),
                "min_industry_n": min_industry_n,
                "enable_decay_gate": enable_decay_gate,
                "enable_emerging": enable_emerging,
                "enable_sparse_track": enable_sparse_track,
                "ic_threshold": ic_threshold,
                "icir_threshold": icir_threshold,
                "ic_icir_gate": "pure_AND" if do_neut else "raw_AND",
                "decay_retention_min": decay_retention_min,
                "decay_recent_icir_max": decay_recent_icir_max,
                "decay_recent_ic_max": decay_recent_ic_max,
                "decay_recent_months": decay_recent_months,
                "reversal_frac": reversal_frac,
                "reversal_abs_ic": reversal_abs_ic,
                "emerging_lookback": emerging_lookback,  # 日历月；筛选时换算为 IC 期数
                "emerging_recent_icir": emerging_recent_icir,
                "emerging_fdr_alpha": emerging_fdr_alpha,
                "emerging_lift_min": emerging_lift_min,
                "emerging_holdout_months": emerging_holdout_months,
                "emerging_asof": emerging_asof,
                "emerging_require_trend": emerging_require_trend,
                "emerging_trend_months": emerging_trend_months,
                "emerging_trend_segments": emerging_trend_segments,
                "sparse_win_rate_min": sparse_win_rate_min,
                "sparse_payoff_min": sparse_payoff_min,
                "sparse_require_ic": sparse_require_ic,
                "min_long_share": min_long_share,
                "quantile_decomp": bool(do_quantile),
                "quantile_y_mode": quantile_y_mode if do_quantile else None,
                "tradable_limit_mode": limit_mode,
                **tmr_meta,
            },
            "orthogonalization": gs_meta,
            **tmr_meta,
        }
        # 分位贡献列已在筛选前 overlay；此处仅确保落盘 summary 含齐列
        summary_to_save = summary_df
        if do_quantile and not quantile_df.empty:
            q_join = [
                c for c in ("多头超额", "空头贡献", "long_share", "多空来源")
                if c in quantile_df.columns and c not in summary_to_save.columns
            ]
            if q_join:
                summary_to_save = summary_to_save.join(
                    quantile_df[q_join], how="left"
                )
        json_path = save_results(
            period=period,
            summary_df=summary_to_save,
            yearly_df=yearly_df,
            kept_factors=kept,
            exclusion_reasons=exclusions,
            lookback_years=lookback_years,
            lookback_date=lookback_date,
            ind_ic_df=ind_ic_df,
            pure_ic_means=pure_ic_means or None,
            meta=meta,
            orth_factors=gs_selected,
            sparse_factors=sparse_kept,
            emerging_factors=emerging_kept,
            categories=categories,
            labels=labels,
            quantile_df=quantile_df if do_quantile else None,
            name_suffix="_".join(x for x in (save_suffix, u_tag) if x),
        )
        print(f"\n结果已保存至 {json_path.parent}  ({json_path.name})")

    _maybe_plot_rolling_ic(
        enabled=plot_rolling_ic_flag,
        period=period,
        all_ic=all_ic,
        pure_ic_series=pure_ic_series or None,
        summary_df=summary_df,
        kept=kept,
        sparse_kept=sparse_kept,
        top_n=plot_rolling_ic_top_n,
        window=plot_rolling_ic_window,
        names_csv=plot_rolling_ic_names,
        source=plot_rolling_ic_source,
        tag=u_tag,
    )

    _log_phase("总耗时", t_run)
    gc.collect()
    return summary_df, yearly_df, all_ic, ind_ic_df


def main():
    from config.encoding_bootstrap import bootstrap_stdio_utf8
    from utils.cli_help import add_help_advanced, exit_if_help_advanced, help_text as _h

    bootstrap_stdio_utf8()
    parser = argparse.ArgumentParser(
        description="IC analysis v2（因子 IC 筛选）",
        epilog=(
            "日常最短: python -m research.ic_analysis_v2 --period 5 --barra --save\n"
            "默认已含: FDR / t=2.5 / corr-dedup / research tradable / "
            "decay·emerging·sparse 标注轨（GS 需 --gram-schmidt）。"
            "全部参数: --help-advanced 或 docs/CLI_QUICKSTART.md"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_help_advanced(parser)

    # ── 日常必用 ──────────────────────────────────────────────────────────
    parser.add_argument("--period", type=int, default=20, help="持仓期 / IC horizon（默认 20）")
    parser.add_argument("--barra", action="store_true", help="Barra 9 风格纯 IC（勿与 --neut-controls size* 同用）")
    parser.add_argument(
        "--neut-controls",
        dest="neut_controls",
        choices=("raw", "size", "size_industry", "barra"),
        default=None,
        help=(
            "中性化控制变量：默认 raw（不残差）；size=仅 Size；"
            "size_industry=Size+PIT行业；barra=9风格+行业（同 --barra）"
        ),
    )
    parser.add_argument("--save", action="store_true", help="写出 selected_factors / YAML 素材")
    parser.add_argument(
        "--save-suffix",
        dest="save_suffix",
        default="",
        help="落盘文件名后缀（如 raw_20260815），避免覆盖旗舰 JSON/YAML",
    )
    parser.add_argument("--sample", type=int, default=0, help="仅前 N 个因子（快速 smoke）")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从 checkpoint 续跑（改阈值后勿用；需全量重算请 --fresh）",
    )
    parser.add_argument(
        "--fresh",
        "--clear-ckpts",
        action="store_true",
        dest="fresh",
        help="清空本 period checkpoint 后全量重算",
    )
    parser.add_argument(
        "--only-new",
        action="store_true",
        dest="only_new",
        help="增量 IC：只算 registry 中尚未缓存的因子；barra_pure 指纹匹配时只补新区并 merge",
    )
    parser.add_argument(
        "--factors",
        type=str,
        default=None,
        help="仅计算逗号分隔因子名（须在 registry）",
    )
    parser.add_argument(
        "--cap-band",
        default=CAP_BAND_DEFAULT,
        choices=list(CAP_BANDS.keys()),
        dest="cap_band",
        help=(
            f"市值带预设（默认 {CAP_BAND_DEFAULT}）；"
            "micro_30/micro_lt30=circ_mv≤30亿无地板（≠micro 8~30亿）"
        ),
    )
    parser.add_argument(
        "--mcap-min-yi",
        type=float,
        default=None,
        dest="mcap_min_yi",
        help=_h("流通市值下限（亿元，含；单位元=亿×1e8）。与 --mcap-max-yi 构成每日宇宙，无成交额过滤", advanced=False),
    )
    parser.add_argument(
        "--mcap-max-yi",
        type=float,
        default=None,
        dest="mcap_max_yi",
        help=_h("流通市值上限（亿元，含）。例: 30–100 亿中盘", advanced=False),
    )
    parser.add_argument(
        "--restan-in-universe",
        dest="restan_in_universe",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=_h("在当日宇宙上重做截面 winsor+zscore（亿元带默认开）", advanced=True),
    )
    parser.add_argument(
        "--min-industry-n",
        type=int,
        default=None,
        dest="min_industry_n",
        help=_h("档内行业哑元最少有效样本，少则并入「其他」（亿元带默认 10；0=关闭）", advanced=True),
    )

    # ── 高级 ──────────────────────────────────────────────────────────────
    parser.add_argument("--top", type=int, default=0, help=_h("仅打印 Top-N", advanced=True))
    parser.add_argument("--plot", action="store_true", help=_h("交互式画图（旧）", advanced=True))
    parser.add_argument(
        "--plot-rolling-ic",
        action="store_true",
        dest="plot_rolling_ic",
        help=_h(
            "保存滚动 IC 折线图到 research/output/figs/（复用已算 IC 序列；默认入选/|IC| Top-N）",
            advanced=True,
        ),
    )
    parser.add_argument(
        "--plot-rolling-ic-top-n",
        type=int,
        default=30,
        dest="plot_rolling_ic_top_n",
        help=_h("折线图最多因子数（默认 30；有入选名单时先入选再截断）", advanced=True),
    )
    parser.add_argument(
        "--plot-rolling-ic-window",
        type=int,
        default=None,
        dest="plot_rolling_ic_window",
        help=_h("滚动窗期数（默认约半年：h5=26 / h20=6）", advanced=True),
    )
    parser.add_argument(
        "--plot-rolling-ic-names",
        type=str,
        default=None,
        dest="plot_rolling_ic_names",
        help=_h("逗号分隔因子名（优先于入选/Top-N）", advanced=True),
    )
    parser.add_argument(
        "--plot-rolling-ic-source",
        choices=["auto", "raw", "pure"],
        default="auto",
        dest="plot_rolling_ic_source",
        help=_h("折线序列口径 auto|raw|pure（默认 auto：有 barra pure 用 pure）", advanced=True),
    )
    parser.add_argument("--decay", action="store_true", help=_h("打印衰减表", advanced=True))
    parser.add_argument(
        "--corr",
        action="store_true",
        help=_h("打印全因子相关矩阵（仅展示；去重见 --corr-dedup）", advanced=True),
    )
    parser.add_argument("--industry", action="store_true", help=_h("行业 IC", advanced=True))
    parser.add_argument(
        "--allow-static-industry",
        action="store_true",
        default=False,
        help=_h("无 PIT 行业面板时回退静态 map（仅 debug）", advanced=True),
    )
    parser.add_argument(
        "--quantile-decomp",
        dest="quantile_decomp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=_h(
            f"Barra pure 下 Q1/Q5 分解（默认随 --barra 与 IC_QUANTILE_DECOMP={IC_QUANTILE_DECOMP}）",
            advanced=True,
        ),
    )
    parser.add_argument(
        "--quantile-y-mode",
        choices=["residual", "raw"],
        default=None,
        dest="quantile_y_mode",
        help=_h(f"分位收益口径 residual|raw（默认 {IC_QUANTILE_Y_MODE}）", advanced=True),
    )
    parser.add_argument(
        "--lookback-years", type=int, default=0, dest="lookback_years",
        help=_h("仅用最近 N 年样本（0=全样本）", advanced=True),
    )
    parser.add_argument("--workers", type=int, default=None, help=_h("IC 并行 workers", advanced=True))
    parser.add_argument(
        "--barra-workers", type=int, default=None, dest="barra_workers",
        help=_h("Barra 残差化并行 workers", advanced=True),
    )
    parser.add_argument(
        "--factor-prefix",
        type=str,
        default=None,
        dest="factor_prefix",
        help=_h("仅计算此前缀因子（逗号分隔）", advanced=True),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        dest="batch_size",
        help=_h("因子 IC 分批大小（0=不分批）", advanced=True),
    )
    parser.add_argument(
        "--corr-dedup",
        dest="corr_dedup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=_h("截面相关去冗余（默认 ON；--no-corr-dedup 关）", advanced=True),
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.70,
        dest="corr_threshold",
        help=_h("相关去重 |corr| 阈值（默认 0.70）", advanced=True),
    )
    parser.add_argument(
        "--corr-method",
        choices=["max", "p95", "mean"],
        default=None,
        help=_h(f"去冗余相关度聚合（默认 {IC_CORR_METHOD}）", advanced=True),
    )
    parser.add_argument(
        "--no-nw-t",
        action="store_true",
        help=_h("筛选用经典 t 而非 Newey-West t", advanced=True),
    )
    parser.add_argument(
        "--t-threshold",
        type=float,
        default=2.5,
        help=_h("t / NW_t 显著性阈值（默认 2.5）", advanced=True),
    )
    parser.add_argument(
        "--ic-threshold",
        type=float,
        default=IC_THRESHOLD,
        dest="ic_threshold",
        help=_h(f"稠密硬门 |IC| 下限（默认 {IC_THRESHOLD}）", advanced=True),
    )
    parser.add_argument(
        "--icir-threshold",
        type=float,
        default=ICIR_THRESHOLD,
        dest="icir_threshold",
        help=_h(f"稠密硬门 |ICIR| 下限（默认 {ICIR_THRESHOLD}）", advanced=True),
    )
    parser.add_argument(
        "--min-long-share",
        type=float,
        default=IC_MIN_LONG_SHARE,
        dest="min_long_share",
        help=_h(f"稠密门 long_share 下限（默认 {IC_MIN_LONG_SHARE}；0 关闭）", advanced=True),
    )
    parser.add_argument(
        "--long-share-csv",
        type=str,
        default=None,
        dest="long_share_csv",
        help=_h("从 CSV 合并符号对齐 long_share", advanced=True),
    )
    parser.add_argument(
        "--use-fdr",
        dest="use_fdr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=_h("BH-FDR 校正（默认 ON；--no-use-fdr 关）", advanced=True),
    )
    parser.add_argument(
        "--gram-schmidt",
        dest="gram_schmidt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=_h("Gram-Schmidt 正交精筛（默认 OFF；--gram-schmidt 开）", advanced=True),
    )
    parser.add_argument(
        "--max-factors",
        type=int,
        default=30,
        dest="max_factors",
        help=_h("Gram-Schmidt 保留上限（默认 30）", advanced=True),
    )
    parser.add_argument(
        "--gs-ic-threshold",
        type=float,
        default=0.015,
        dest="gs_ic_threshold",
        help=_h("GS 残差 IC 门槛（默认 0.015）", advanced=True),
    )
    parser.add_argument(
        "--gs-icir-threshold",
        type=float,
        default=0.15,
        dest="gs_icir_threshold",
        help=_h("GS 残差 ICIR 门槛（默认 0.15）", advanced=True),
    )
    parser.add_argument(
        "--regime-consistency",
        type=float,
        default=None,
        dest="regime_consistency",
        help=_h("A 门：同向年份占比（默认关）", advanced=True),
    )
    parser.add_argument(
        "--rolling-icir",
        type=float,
        default=None,
        dest="rolling_icir",
        help=_h("B 门：滚动 ICIR（默认关）", advanced=True),
    )
    parser.add_argument(
        "--worst-period-ic",
        type=float,
        default=None,
        dest="worst_period_ic",
        help=_h("C 门：最差12期 IC 均值（默认关）", advanced=True),
    )
    parser.add_argument(
        "--raw-select",
        action="store_true",
        dest="raw_select",
        help=_h("快速通道：跳过需面板阶段，仅 summary+IC 相关去重", advanced=True),
    )
    _q_default = int(round(SMALL_MCAP_QUANTILE * 100))
    parser.add_argument(
        "--universe",
        default="all",
        choices=["all", "small_mcap"],
        help=_h(
            f"IC 宇宙 all|small_mcap（小市值分位默认 {_q_default}%%）",
            advanced=True,
        ),
    )
    parser.add_argument(
        "--universe-quantile",
        type=float,
        default=SMALL_MCAP_QUANTILE,
        dest="universe_quantile",
        help=_h(f"small_mcap 分位（默认 {SMALL_MCAP_QUANTILE}）", advanced=True),
    )
    parser.add_argument(
        "--universe-mask",
        type=str,
        default=None,
        dest="universe_mask",
        help=_h("外部 universe mask parquet（覆盖 --universe/--cap-band）", advanced=True),
    )
    parser.add_argument(
        "--fwd-return-winsor",
        dest="fwd_return_winsor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=_h(f"forward_return 截尾（默认 FWD_RETURN_WINSOR={FWD_RETURN_WINSOR}）", advanced=True),
    )
    parser.add_argument(
        "--tradable-strict",
        action="store_true",
        dest="tradable_strict",
        help=_h("信号日剔除涨跌停（旧 strict）", advanced=True),
    )
    parser.add_argument(
        "--label-exec-mask",
        action="store_true",
        dest="label_exec_mask",
        help=_h("标签 execution mask（旧 strict）", advanced=True),
    )
    parser.add_argument(
        "--tradable-limit-mode",
        choices=("strict", "research"),
        default=None,
        dest="tradable_limit_mode",
        help=_h("涨跌停口径别名 research|strict", advanced=True),
    )
    # decay / reversal / emerging / sparse（默认 ON，阈值藏进 advanced）
    parser.add_argument(
        "--decay-gate",
        dest="enable_decay_gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=_h("衰减标注（默认 ON；--no-decay-gate）", advanced=True),
    )
    parser.add_argument(
        "--decay-recent-months", type=int, default=IC_DECAY_RECENT_MONTHS,
        dest="decay_recent_months",
        help=_h(f"衰减回溯月数（默认 {IC_DECAY_RECENT_MONTHS}）", advanced=True),
    )
    parser.add_argument(
        "--decay-retention-min", type=float, default=IC_DECAY_RETENTION_MIN,
        dest="decay_retention_min",
        help=_h(f"衰减保留率下限（默认 {IC_DECAY_RETENTION_MIN}）", advanced=True),
    )
    parser.add_argument(
        "--decay-retention-min-sparse", type=float, default=IC_DECAY_RETENTION_MIN_SPARSE,
        dest="decay_retention_min_sparse",
        help=_h(f"稀疏轨衰减保留率（默认 {IC_DECAY_RETENTION_MIN_SPARSE}）", advanced=True),
    )
    parser.add_argument(
        "--decay-recent-icir-max", type=float, default=IC_DECAY_RECENT_ICIR_MAX,
        dest="decay_recent_icir_max",
        help=_h(f"衰减 |ICIR_recent| 上限（默认 {IC_DECAY_RECENT_ICIR_MAX}）", advanced=True),
    )
    parser.add_argument(
        "--decay-recent-ic-max", type=float, default=IC_DECAY_RECENT_IC_MAX,
        dest="decay_recent_ic_max",
        help=_h(f"衰减 |IC_recent| 上限（默认 {IC_DECAY_RECENT_IC_MAX}）", advanced=True),
    )
    parser.add_argument(
        "--reversal-label",
        dest="enable_reversal_label",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=_h("风格逆转标注（默认 ON）", advanced=True),
    )
    parser.add_argument(
        "--reversal-months", type=int, default=IC_REVERSAL_MONTHS, dest="reversal_months",
        help=_h(f"逆转窗口月数（默认 {IC_REVERSAL_MONTHS}）", advanced=True),
    )
    parser.add_argument(
        "--reversal-frac", type=float, default=IC_REVERSAL_FRAC, dest="reversal_frac",
        help=_h(f"逆转占比门槛（默认 {IC_REVERSAL_FRAC}）", advanced=True),
    )
    parser.add_argument(
        "--reversal-abs-ic", type=float, default=IC_REVERSAL_ABS_IC, dest="reversal_abs_ic",
        help=_h(f"逆转 |IC| 门槛（默认 {IC_REVERSAL_ABS_IC}）", advanced=True),
    )
    parser.add_argument(
        "--emerging",
        dest="enable_emerging",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=_h("新兴因子标注（默认 ON；不进主池）", advanced=True),
    )
    parser.add_argument(
        "--emerging-lookback", type=int, default=IC_EMERGING_LOOKBACK,
        dest="emerging_lookback",
        help=_h(f"新兴近窗日历月（默认 {IC_EMERGING_LOOKBACK}）", advanced=True),
    )
    parser.add_argument(
        "--emerging-holdout-months", type=int, default=IC_EMERGING_HOLDOUT_MONTHS,
        dest="emerging_holdout_months",
        help=_h(f"新兴 holdout 月末数（默认 {IC_EMERGING_HOLDOUT_MONTHS}）", advanced=True),
    )
    parser.add_argument(
        "--emerging-asof", type=str, default=None, dest="emerging_asof",
        help=_h("新兴近窗截止日期 YYYY-MM-DD", advanced=True),
    )
    parser.add_argument(
        "--emerging-trend",
        dest="emerging_require_trend",
        action=argparse.BooleanOptionalAction,
        default=IC_EMERGING_REQUIRE_TREND,
        help=_h(
            f"新兴季度趋势门（默认 {'ON' if IC_EMERGING_REQUIRE_TREND else 'OFF'}）",
            advanced=True,
        ),
    )
    parser.add_argument(
        "--emerging-trend-months", type=int, default=IC_EMERGING_TREND_MONTHS,
        dest="emerging_trend_months",
        help=_h(f"新兴趋势段月数（默认 {IC_EMERGING_TREND_MONTHS}）", advanced=True),
    )
    parser.add_argument(
        "--emerging-trend-segments", type=int, default=IC_EMERGING_TREND_SEGMENTS,
        dest="emerging_trend_segments",
        help=_h(f"新兴趋势段数（默认 {IC_EMERGING_TREND_SEGMENTS}）", advanced=True),
    )
    parser.add_argument(
        "--emerging-trend-eps", type=float, default=IC_EMERGING_TREND_EPS,
        dest="emerging_trend_eps",
        help=_h(f"新兴趋势噪声容忍（默认 {IC_EMERGING_TREND_EPS}）", advanced=True),
    )
    parser.add_argument(
        "--emerging-recent-icir", type=float, default=IC_EMERGING_RECENT_ICIR,
        dest="emerging_recent_icir",
        help=_h(f"新兴 |ICIR| 门槛（默认 {IC_EMERGING_RECENT_ICIR}）", advanced=True),
    )
    parser.add_argument(
        "--emerging-fdr-alpha", type=float, default=IC_EMERGING_FDR_ALPHA,
        dest="emerging_fdr_alpha",
        help=_h(f"新兴 FDR α（默认 {IC_EMERGING_FDR_ALPHA}）", advanced=True),
    )
    parser.add_argument(
        "--emerging-lift-min", type=float, default=IC_EMERGING_LIFT_MIN,
        dest="emerging_lift_min",
        help=_h(f"新兴 lift 下限（默认 {IC_EMERGING_LIFT_MIN}）", advanced=True),
    )
    parser.add_argument(
        "--sparse-track",
        dest="enable_sparse_track",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=_h("稀疏因子轨道（默认 ON）", advanced=True),
    )
    parser.add_argument(
        "--sparse-ic-threshold", type=float, default=IC_SPARSE_IC_THRESHOLD,
        dest="sparse_ic_threshold",
        help=_h(f"稀疏 |IC| 软参考（默认 {IC_SPARSE_IC_THRESHOLD}）", advanced=True),
    )
    parser.add_argument(
        "--sparse-icir-threshold", type=float, default=IC_SPARSE_ICIR_THRESHOLD,
        dest="sparse_icir_threshold",
        help=_h(f"稀疏 |ICIR| 软参考（默认 {IC_SPARSE_ICIR_THRESHOLD}）", advanced=True),
    )
    parser.add_argument(
        "--sparse-require-ic",
        dest="sparse_require_ic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=_h("稀疏轨硬要求 IC/ICIR（默认 OFF）", advanced=True),
    )
    parser.add_argument(
        "--sparse-win-rate-min", type=float, default=IC_SPARSE_WIN_RATE_MIN,
        dest="sparse_win_rate_min",
        help=_h(f"稀疏同向胜率门槛（默认 {IC_SPARSE_WIN_RATE_MIN}）", advanced=True),
    )
    parser.add_argument(
        "--sparse-payoff-min", type=float, default=IC_SPARSE_PAYOFF_MIN,
        dest="sparse_payoff_min",
        help=_h(f"稀疏触发日截面胜率门槛（默认 {IC_SPARSE_PAYOFF_MIN}）", advanced=True),
    )
    parser.add_argument(
        "--sparse-corr-threshold", type=float, default=IC_SPARSE_CORR_THRESHOLD,
        dest="sparse_corr_threshold",
        help=_h(f"稀疏相关去重阈值（默认 {IC_SPARSE_CORR_THRESHOLD}）", advanced=True),
    )

    # ── deprecated no-op ─────────────────────────────────────────────────
    parser.add_argument(
        "--decay-half-life-min", type=float, default=IC_DECAY_HALF_LIFE_MIN,
        dest="decay_half_life_min",
        help=_h("旧 half-life 门已移除；请用 --decay-retention-min", deprecated=True),
    )
    parser.add_argument(
        "--decay-short-long-min", type=float, default=IC_DECAY_SHORT_LONG_MIN,
        dest="decay_short_long_min",
        help=_h("旧长短窗 |IC| 门已移除；请用 --decay-retention-min", deprecated=True),
    )
    parser.add_argument(
        "--emerging-recent-ic", type=float, default=IC_EMERGING_RECENT_IC,
        dest="emerging_recent_ic",
        help=_h("旧 |IC_recent| OR 门已移除", deprecated=True),
    )
    parser.add_argument(
        "--sparse-t-threshold", type=float, default=IC_SPARSE_T_THRESHOLD,
        dest="sparse_t_threshold",
        help=_h("稀疏轨已取消 t/NW-t/FDR 要求", deprecated=True),
    )

    args = parser.parse_args()
    exit_if_help_advanced(parser, args)
    # 旧 half-life / 稀疏 t 门：显式传入非默认时告警（已 no-op）
    if args.decay_half_life_min != IC_DECAY_HALF_LIFE_MIN:
        print("  [warn] --decay-half-life-min 已弃用（no-op）；请用 --decay-retention-min")
    if args.decay_short_long_min != IC_DECAY_SHORT_LONG_MIN:
        print("  [warn] --decay-short-long-min 已弃用（no-op）；请用 --decay-retention-min")
    if args.sparse_t_threshold != IC_SPARSE_T_THRESHOLD:
        print("  [warn] --sparse-t-threshold 已弃用（no-op）；稀疏轨无 t/FDR 要求")
    exclude_limit = True if args.tradable_strict else None
    apply_exec = True if args.label_exec_mask else None
    if args.tradable_limit_mode == "strict":
        exclude_limit = True
        apply_exec = True
    elif args.tradable_limit_mode == "research":
        exclude_limit = False
        apply_exec = False
    run(
        period=args.period,
        top=args.top,
        plot=args.plot,
        plot_rolling_ic_flag=args.plot_rolling_ic,
        plot_rolling_ic_top_n=args.plot_rolling_ic_top_n,
        plot_rolling_ic_window=args.plot_rolling_ic_window,
        plot_rolling_ic_names=args.plot_rolling_ic_names,
        plot_rolling_ic_source=args.plot_rolling_ic_source,
        decay=args.decay,
        corr=args.corr,
        save=args.save,
        industry=args.industry,
        allow_static_industry=args.allow_static_industry,
        barra=args.barra,
        neut_controls=args.neut_controls,
        save_suffix=args.save_suffix,
        lookback_years=args.lookback_years,
        workers=args.workers,
        barra_workers=args.barra_workers,
        sample=args.sample,
        factor_prefix=args.factor_prefix,
        batch_size=args.batch_size,
        corr_method=args.corr_method,
        use_nw_t=not args.no_nw_t,
        t_threshold=args.t_threshold,
        use_fdr=args.use_fdr,
        gram_schmidt=args.gram_schmidt,
        max_factors=args.max_factors,
        gs_ic_threshold=args.gs_ic_threshold,
        gs_icir_threshold=args.gs_icir_threshold,
        regime_consistency=args.regime_consistency,
        rolling_icir=args.rolling_icir,
        worst_period_ic=args.worst_period_ic,
        raw_select=args.raw_select,
        resume=args.resume,
        fresh=args.fresh,
        only_new=args.only_new,
        factors=args.factors,
        cap_band=args.cap_band,
        universe=args.universe,
        universe_quantile=args.universe_quantile,
        universe_mask=args.universe_mask,
        mcap_min_yi=args.mcap_min_yi,
        mcap_max_yi=args.mcap_max_yi,
        restan_in_universe=args.restan_in_universe,
        min_industry_n=args.min_industry_n,
        fwd_return_winsor=args.fwd_return_winsor,
        corr_dedup=args.corr_dedup,
        corr_threshold=args.corr_threshold,
        ic_threshold=args.ic_threshold,
        icir_threshold=args.icir_threshold,
        enable_decay_gate=args.enable_decay_gate,
        decay_recent_months=args.decay_recent_months,
        decay_retention_min=args.decay_retention_min,
        decay_retention_min_sparse=args.decay_retention_min_sparse,
        decay_recent_icir_max=args.decay_recent_icir_max,
        decay_recent_ic_max=args.decay_recent_ic_max,
        enable_reversal_label=args.enable_reversal_label,
        reversal_months=args.reversal_months,
        reversal_frac=args.reversal_frac,
        reversal_abs_ic=args.reversal_abs_ic,
        enable_emerging=args.enable_emerging,
        emerging_lookback=args.emerging_lookback,
        emerging_recent_icir=args.emerging_recent_icir,
        emerging_recent_ic=args.emerging_recent_ic,
        emerging_fdr_alpha=args.emerging_fdr_alpha,
        emerging_lift_min=args.emerging_lift_min,
        emerging_holdout_months=args.emerging_holdout_months,
        emerging_asof=args.emerging_asof,
        emerging_require_trend=args.emerging_require_trend,
        emerging_trend_months=args.emerging_trend_months,
        emerging_trend_segments=args.emerging_trend_segments,
        emerging_trend_eps=args.emerging_trend_eps,
        enable_sparse_track=args.enable_sparse_track,
        sparse_ic_threshold=args.sparse_ic_threshold,
        sparse_icir_threshold=args.sparse_icir_threshold,
        sparse_win_rate_min=args.sparse_win_rate_min,
        sparse_payoff_min=args.sparse_payoff_min,
        sparse_require_ic=args.sparse_require_ic,
        sparse_corr_threshold=args.sparse_corr_threshold,
        quantile_decomp=args.quantile_decomp,
        quantile_y_mode=args.quantile_y_mode,
        min_long_share=args.min_long_share,
        long_share_csv=args.long_share_csv,
        tradable_limit_mode=args.tradable_limit_mode,
        exclude_limit_on_signal=exclude_limit,
        apply_exec_mask=apply_exec,
    )



if __name__ == "__main__":
    main()
