"""Greedy correlation dedup for rolling pool candidates."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.rolling_pool.corr_provider import FactorCorrProvider, NullFactorCorrProvider
from research.rolling_pool.stats import FactorStats, sort_key


def _ordered_names(
    candidates: list[str],
    stats_map: dict[str, FactorStats],
) -> list[str]:
    return sorted(
        candidates,
        key=lambda n: sort_key(stats_map.get(n, FactorStats(
            mean=np.nan, icir=np.nan, abs_mean=np.nan, abs_icir=np.nan, n=0,
        ))),
        reverse=True,
    )


def dedup_by_ic_corr(
    candidates: list[str],
    ic_panel: pd.DataFrame | dict[str, pd.Series],
    thr: float = 0.7,
    *,
    stats_map: dict[str, FactorStats] | None = None,
    end_date: pd.Timestamp | str | None = None,
    window: int = 52,
) -> list[str]:
    """
    IC 序列相关贪心去重：按 ``|ICIR|`` 高→低，若与已选某因子
    在窗口内 IC 序列 ``|corr| > thr`` 则丢弃低 ICIR 者。

    Parameters
    ----------
    candidates : 已过硬门的因子名（顺序不重要）
    ic_panel   : ``DataFrame(date×factor)`` 或 ``dict[name, Series]``
    thr        : |corr| 阈值，默认 0.7
    stats_map  : 若提供则按其 ``|ICIR|`` 排序；否则用窗口内临时统计
    end_date   : 若给，只取 **严格早于** ``end_date`` 的末 ``window`` 周算 corr
        （与 ``compute_factor_stats`` 同口径：决策日当期 IC 尚未实现，禁止前视）
    window     : 与 stats 同口径的回看周数
    """
    if not candidates:
        return []
    if len(candidates) == 1:
        return list(candidates)

    if isinstance(ic_panel, dict):
        frame = pd.DataFrame({n: ic_panel[n] for n in candidates if n in ic_panel})
    else:
        cols = [n for n in candidates if n in ic_panel.columns]
        frame = ic_panel[cols].copy()

    if end_date is not None:
        end = pd.Timestamp(end_date)
        # 严格早于决策日：IC_t 在 t 当日未实现
        frame = frame.loc[frame.index < end]
        if window and window > 0:
            frame = frame.tail(int(window))

    if frame.shape[1] == 0:
        return list(candidates)

    if stats_map is None:
        # fallback: rank by |mean|/|icir| on the same window
        from research.rolling_pool.stats import compute_factor_stats

        # frame 已按 `< end_date` 裁剪；end_date 缺失时整段可用 → include_end
        ed = end_date if end_date is not None else frame.index.max()
        inc = end_date is None
        stats_map = {
            n: compute_factor_stats(frame[n], ed, window=window, include_end=inc)
            for n in frame.columns
        }

    order = _ordered_names([n for n in candidates if n in frame.columns], stats_map)
    # candidates missing from panel: append at end (keep, no corr info)
    missing = [n for n in candidates if n not in frame.columns]
    corr = frame[order].corr(method="pearson")

    kept: list[str] = []
    for name in order:
        drop = False
        for k in kept:
            if name not in corr.index or k not in corr.columns:
                continue
            c = abs(float(corr.loc[name, k]))
            if np.isfinite(c) and c > thr:
                drop = True
                break
        if not drop:
            kept.append(name)
    # missing-panel factors: keep (cannot verify redundancy via IC series)
    for n in missing:
        if n not in kept:
            kept.append(n)
    return kept


def dedup_by_factor_cs_corr(
    candidates: list[str],
    provider: FactorCorrProvider | None,
    asof: pd.Timestamp | str,
    thr: float = 0.7,
    *,
    stats_map: dict[str, FactorStats] | None = None,
) -> tuple[list[str], dict]:
    """
    因子值截面相关贪心去重（小集合）。

    Returns
    -------
    kept, meta
        meta 含 ``applied`` / ``n_pairs_checked`` / ``provider`` 等，
        便于 schedule 元数据记录是否降级。
    """
    meta: dict = {
        "applied": False,
        "provider": getattr(provider, "label", "none") if provider else "none",
        "n_input": len(candidates),
        "n_output": len(candidates),
        "n_dropped": 0,
    }
    if not candidates or len(candidates) == 1:
        return list(candidates), meta
    if provider is None or isinstance(provider, NullFactorCorrProvider):
        meta["skip_reason"] = "null_provider"
        return list(candidates), meta

    asof_ts = pd.Timestamp(asof)
    corr = provider.pairwise_abs_corr(list(candidates), asof_ts)
    if corr is None or corr.empty:
        meta["skip_reason"] = "empty_corr_matrix"
        return list(candidates), meta

    if stats_map is None:
        # no stats → keep input order
        order = list(candidates)
    else:
        order = _ordered_names(list(candidates), stats_map)

    kept: list[str] = []
    n_checked = 0
    for name in order:
        drop = False
        if name not in corr.index:
            kept.append(name)
            continue
        for k in kept:
            if k not in corr.columns:
                continue
            c = corr.loc[name, k]
            if not np.isfinite(c):
                continue
            n_checked += 1
            if float(c) > thr:
                drop = True
                break
        if not drop:
            kept.append(name)

    meta.update(
        applied=True,
        n_output=len(kept),
        n_dropped=len(candidates) - len(kept),
        n_pairs_checked=n_checked,
        n_corr_names=int(corr.shape[0]),
    )
    return kept, meta


def dual_dedup(
    candidates: list[str],
    ic_panel: pd.DataFrame | dict[str, pd.Series],
    *,
    stats_map: dict[str, FactorStats],
    end_date: pd.Timestamp | str,
    window: int = 52,
    ic_corr_thr: float = 0.7,
    cs_corr_thr: float = 0.7,
    cs_provider: FactorCorrProvider | None = None,
) -> tuple[list[str], dict]:
    """先 IC 序列去重，再截面去重；返回 (kept, dedup_meta)。"""
    after_ic = dedup_by_ic_corr(
        candidates,
        ic_panel,
        thr=ic_corr_thr,
        stats_map=stats_map,
        end_date=end_date,
        window=window,
    )
    after_cs, cs_meta = dedup_by_factor_cs_corr(
        after_ic,
        cs_provider,
        asof=end_date,
        thr=cs_corr_thr,
        stats_map=stats_map,
    )
    meta = {
        "n_in": len(candidates),
        "n_after_ic_corr": len(after_ic),
        "n_after_cs_corr": len(after_cs),
        "cs": cs_meta,
    }
    return after_cs, meta
