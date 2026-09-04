"""
Build rolling factor-pool schedule from weekly pure-IC series.

因果性（无前视）
----------------
决策日 t 的池只由 **严格早于 t** 的 IC 观测决定（硬门统计、排序、IC 序列相关
去重全部走 ``index < t``，见 ``stats.compute_factor_stats`` / ``dedup``）。
理由：IC_t 需要 t → t+h 的前向收益，在 t 当日尚未实现。
因此消费侧 asof（``decision_date <= 调仓日``）不会引入未来信息，
**不需要**在消费侧再 shift 一期。
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from research.rolling_pool.corr_provider import FactorCorrProvider, NullFactorCorrProvider
from research.rolling_pool.dedup import dual_dedup
from research.rolling_pool.stats import (
    FactorStats,
    compute_factor_stats,
    passes_hard_gate,
    sort_key,
)


@dataclass
class PoolParams:
    window: int = 52
    abs_mean_min: float = 0.015
    abs_icir_min: float = 0.3
    min_periods: int = 52
    k_max: int = 50
    turnover_frac: float = 0.2
    ic_corr_thr: float = 0.7
    cs_corr_thr: float = 0.7
    cooldown_periods: int = 1
    ddof: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PeriodRecord:
    date: pd.Timestamp
    pool: list[str]
    n_pool: int
    n_fail: int
    n_trim: int
    n_out: int
    n_in: int
    turnover: float
    n_pass_universe: int
    dedup_meta: dict = field(default_factory=dict)


def _stats_universe(
    ic_dict: dict[str, pd.Series],
    end_date: pd.Timestamp,
    params: PoolParams,
) -> dict[str, FactorStats]:
    out: dict[str, FactorStats] = {}
    for name, s in ic_dict.items():
        out[name] = compute_factor_stats(
            s, end_date, window=params.window, ddof=params.ddof,
        )
    return out


def _passers(
    stats_map: dict[str, FactorStats],
    params: PoolParams,
    *,
    exclude: set[str] | None = None,
) -> list[str]:
    excl = exclude or set()
    names = [
        n for n, st in stats_map.items()
        if n not in excl and passes_hard_gate(
            st,
            abs_mean_min=params.abs_mean_min,
            abs_icir_min=params.abs_icir_min,
            min_periods=params.min_periods,
        )
    ]
    return sorted(names, key=lambda n: sort_key(stats_map[n]), reverse=True)


def _target_out(n_pool: int, n_fail: int, params: PoolParams) -> int:
    """强制换手相对上一期池大小 n：target_out = max(n_fail, floor(frac * n))."""
    if n_pool <= 0:
        return 0
    forced = int(math.floor(params.turnover_frac * n_pool))
    return max(int(n_fail), forced)


def _select_drop(
    old_pool: list[str],
    stats_map: dict[str, FactorStats],
    params: PoolParams,
) -> tuple[list[str], list[str], int, int]:
    """
    Returns (out_list, retained, n_fail, n_trim).

    1) 全部硬门失败者出局
    2) 若不足 target_out，在剩余合格者中按 |ICIR| 末位再踢
    """
    n = len(old_pool)
    fail = [
        n_ for n_ in old_pool
        if not passes_hard_gate(
            stats_map[n_],
            abs_mean_min=params.abs_mean_min,
            abs_icir_min=params.abs_icir_min,
            min_periods=params.min_periods,
        )
    ]
    n_fail = len(fail)
    target = _target_out(n, n_fail, params)
    out_set = set(fail)
    n_trim = 0
    if len(out_set) < target:
        survivors = [n_ for n_ in old_pool if n_ not in out_set]
        # ascending |ICIR| → kick weakest first
        survivors_asc = sorted(
            survivors,
            key=lambda n_: sort_key(stats_map[n_]),
            reverse=False,
        )
        need = target - len(out_set)
        for n_ in survivors_asc[:need]:
            out_set.add(n_)
            n_trim += 1
    out_list = [n_ for n_ in old_pool if n_ in out_set]
    retained = [n_ for n_ in old_pool if n_ not in out_set]
    return out_list, retained, n_fail, n_trim


def _fill_pool(
    retained: list[str],
    stats_map: dict[str, FactorStats],
    ic_dict: dict[str, pd.Series],
    end_date: pd.Timestamp,
    params: PoolParams,
    *,
    cooldown: set[str],
    cs_provider: FactorCorrProvider | None,
) -> tuple[list[str], dict]:
    """保留集 ∪ 池外过门新晋 → 双重去重 → Top K_max。"""
    exclude = set(retained) | set(cooldown)
    newcomers = _passers(stats_map, params, exclude=exclude)
    merged = list(dict.fromkeys(list(retained) + list(newcomers)))
    kept, dedup_meta = dual_dedup(
        merged,
        ic_dict,
        stats_map=stats_map,
        end_date=end_date,
        window=params.window,
        ic_corr_thr=params.ic_corr_thr,
        cs_corr_thr=params.cs_corr_thr,
        cs_provider=cs_provider,
    )
    # After dedup, re-sort and cap at K_max
    kept_sorted = sorted(kept, key=lambda n: sort_key(stats_map[n]), reverse=True)
    pool = kept_sorted[: params.k_max]
    return pool, dedup_meta


def infer_rebalance_dates(
    ic_dict: dict[str, pd.Series],
    *,
    coverage: float = 0.5,
    window: int = 52,
) -> list[pd.Timestamp]:
    """
    与 pure IC 索引对齐的周频决策日（覆盖率过滤 + warm-up）。

    warm-up 取 ``dates[window:]``：统计口径为「严格早于决策日」，位置 i 的
    决策日只有 i 个可用历史 IC 点，故 ``i >= window`` 才凑满 window 周。
    """
    from collections import Counter

    cnt: Counter = Counter()
    for s in ic_dict.values():
        if s is None or len(s) == 0:
            continue
        cnt.update(pd.DatetimeIndex(s.dropna().index))
    if not cnt:
        return []
    thr = max(1, int(np.ceil(coverage * len(ic_dict))))
    dates = sorted(pd.Timestamp(d) for d, c in cnt.items() if c >= thr)
    # warm-up：第 window 个位置起，历史（不含当日）才有 window 个点
    return dates[window:]


def build_pool_schedule(
    ic_dict: dict[str, pd.Series],
    rebalance_dates: list | pd.DatetimeIndex | None = None,
    *,
    params: PoolParams | None = None,
    cs_provider: FactorCorrProvider | None = None,
    progress_every: int = 25,
) -> tuple[pd.DataFrame, dict]:
    """
    构建轮动定池时间表。

    所有 IC 统计 / 去重口径为「严格早于决策日」（见模块 docstring）。

    Returns
    -------
    schedule_long : DataFrame
        长表列 ``date, factor``（另附可选 ``abs_icir, abs_mean``）
    meta : dict
        参数、并集、换手摘要、逐期记录等
    """
    params = params or PoolParams()
    if cs_provider is None:
        cs_provider = NullFactorCorrProvider()

    # normalize series
    clean: dict[str, pd.Series] = {}
    for k, v in ic_dict.items():
        if v is None:
            continue
        s = pd.Series(v).dropna().astype(float).sort_index()
        if len(s) > 0:
            clean[str(k)] = s
    if not clean:
        raise ValueError("ic_dict 为空")

    if rebalance_dates is None:
        dates = infer_rebalance_dates(clean, window=params.window)
    else:
        dates = [pd.Timestamp(d) for d in rebalance_dates]
    if not dates:
        raise ValueError("无可用 rebalance_dates")

    pool: list[str] = []
    # factor -> inclusive last period index that is blocked from re-entry
    # drop at i with cooldown_periods=1 → until=i+1（本轮 + 下一期）
    cooldown_until: dict[str, int] = {}
    records: list[PeriodRecord] = []
    union: set[str] = set()
    cs_applied_any = False
    cs_skip_reasons: set[str] = set()

    for i, dt in enumerate(dates):
        cool_block = {
            f for f, until in cooldown_until.items()
            if i <= int(until)
        }

        stats_map = _stats_universe(clean, dt, params)
        pass_u = _passers(stats_map, params)

        if i == 0 or not pool:
            # 初始化 / 空池重建
            kept, dedup_meta = dual_dedup(
                pass_u,
                clean,
                stats_map=stats_map,
                end_date=dt,
                window=params.window,
                ic_corr_thr=params.ic_corr_thr,
                cs_corr_thr=params.cs_corr_thr,
                cs_provider=cs_provider,
            )
            kept_sorted = sorted(
                kept, key=lambda n: sort_key(stats_map[n]), reverse=True,
            )
            new_pool = kept_sorted[: params.k_max]
            n_fail = n_trim = 0
            n_out = 0
            n_in = len(new_pool)
            turnover = 1.0 if new_pool else 0.0
        else:
            out_list, retained, n_fail, n_trim = _select_drop(
                pool, stats_map, params,
            )
            # 冷却：出局者在 [i, i+cooldown_periods] 内不可回流
            if params.cooldown_periods > 0:
                until = i + int(params.cooldown_periods)
                for f in out_list:
                    cooldown_until[f] = max(cooldown_until.get(f, until), until)
                cool_block = {
                    f for f, u in cooldown_until.items() if i <= int(u)
                }
            new_pool, dedup_meta = _fill_pool(
                retained,
                stats_map,
                clean,
                dt,
                params,
                cooldown=cool_block,
                cs_provider=cs_provider,
            )
            old_set = set(pool)
            new_set = set(new_pool)
            n_out = len(old_set - new_set)
            n_in = len(new_set - old_set)
            turnover = (n_out / len(pool)) if pool else 0.0

        cs_meta = (dedup_meta or {}).get("cs", {})
        if cs_meta.get("applied"):
            cs_applied_any = True
        elif cs_meta.get("skip_reason"):
            cs_skip_reasons.add(str(cs_meta["skip_reason"]))

        pool = list(new_pool)
        union.update(pool)

        # prune expired cooldown entries
        cooldown_until = {
            f: u for f, u in cooldown_until.items() if i < int(u)
        }

        records.append(
            PeriodRecord(
                date=dt,
                pool=list(pool),
                n_pool=len(pool),
                n_fail=n_fail,
                n_trim=n_trim,
                n_out=n_out,
                n_in=n_in,
                turnover=float(turnover),
                n_pass_universe=len(pass_u),
                dedup_meta=dedup_meta or {},
            )
        )
        if progress_every and (i + 1) % progress_every == 0:
            print(
                f"  [rolling_pool] {i + 1}/{len(dates)} {dt.date()} "
                f"n={len(pool)} turn={turnover:.2f} |U|={len(union)}"
            )

    # long table
    rows = []
    for rec in records:
        for fac in rec.pool:
            st = compute_factor_stats(
                clean[fac], rec.date, window=params.window, ddof=params.ddof,
            ) if fac in clean else None
            rows.append({
                "date": rec.date,
                "factor": fac,
                "abs_icir": st.abs_icir if st else np.nan,
                "abs_mean": st.abs_mean if st else np.nan,
            })
    schedule = pd.DataFrame(rows)
    if not schedule.empty:
        schedule["date"] = pd.to_datetime(schedule["date"])

    period_df = pd.DataFrame([
        {
            "date": r.date,
            "n_pool": r.n_pool,
            "n_fail": r.n_fail,
            "n_trim": r.n_trim,
            "n_out": r.n_out,
            "n_in": r.n_in,
            "turnover": r.turnover,
            "n_pass_universe": r.n_pass_universe,
        }
        for r in records
    ])

    turnovers = period_df["turnover"].astype(float)
    # skip init period (turnover=1 artificial) for mean turnover
    turn_sub = turnovers.iloc[1:] if len(turnovers) > 1 else turnovers

    def _seg_stats(mask: pd.Series) -> dict:
        sub = period_df.loc[mask]
        if sub.empty:
            return {}
        t = sub["turnover"].astype(float)
        # for segments, still exclude first global date if present
        return {
            "n_periods": int(len(sub)),
            "mean_n_pool": float(sub["n_pool"].mean()),
            "mean_turnover": float(t.mean()),
            "mean_n_out": float(sub["n_out"].mean()),
            "mean_n_fail": float(sub["n_fail"].mean()),
            "mean_n_trim": float(sub["n_trim"].mean()),
            "mean_n_pass_universe": float(sub["n_pass_universe"].mean()),
        }

    dates_idx = pd.to_datetime(period_df["date"])
    seg_2026 = _seg_stats(dates_idx.dt.year == 2026)

    cs_degraded = not cs_applied_any
    meta = {
        "params": params.to_dict(),
        "ic_window_excludes_decision_date": True,
        "schedule_causality": (
            "决策日 t 的池仅用 index < t 的 IC；消费侧 asof(<=) 安全，无需再 shift"
        ),
        "cs_provider": getattr(cs_provider, "label", "none"),
        "cs_corr_applied_any": cs_applied_any,
        "cs_corr_degraded": cs_degraded,
        "cs_skip_reasons": sorted(cs_skip_reasons),
        "n_factors_universe": len(clean),
        "n_rebalance_dates": len(dates),
        "date_start": str(dates[0].date()),
        "date_end": str(dates[-1].date()),
        "union_size": len(union),
        "union_factors": sorted(union),
        "mean_n_pool": float(period_df["n_pool"].mean()),
        "mean_turnover_ex_init": float(turn_sub.mean()) if len(turn_sub) else 0.0,
        "mean_turnover_all": float(turnovers.mean()) if len(turnovers) else 0.0,
        "segment_2026": seg_2026,
        "period_stats": period_df.to_dict(orient="list"),
        "smoke_first5": [
            {
                "date": str(r.date.date()),
                "n_pool": r.n_pool,
                "n_fail": r.n_fail,
                "n_trim": r.n_trim,
                "n_out": r.n_out,
                "turnover": r.turnover,
            }
            for r in records[:5]
        ],
    }
    meta["period_df"] = period_df  # attached for writers; stripped on JSON dump
    return schedule, meta
