"""Optional IC decay and factor correlation utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.ic.forward_return import build_forward_return
from research.ic.ic_series import compute_ic_series
from research.ic.statistics import newey_west_t


def _half_life(periods: list, abs_ics: list) -> float:
    """
    Interpolated lag at which |IC| decays to half of its peak value.

    Searches after the peak; if |IC| never drops to half, returns the largest
    sampled period. Returns NaN if fewer than 2 valid samples.
    """
    pairs = [(float(p), float(v)) for p, v in zip(periods, abs_ics) if np.isfinite(v)]
    if len(pairs) < 2:
        return np.nan
    p_arr = np.array([p for p, _ in pairs])
    v_arr = np.array([v for _, v in pairs])
    peak = float(v_arr.max())
    if peak <= 0:
        return np.nan
    target = peak / 2.0
    idx_peak = int(np.argmax(v_arr))
    for i in range(idx_peak, len(pairs)):
        if v_arr[i] <= target:
            if i == 0:
                return float(p_arr[0])
            v0, v1 = float(v_arr[i - 1]), float(v_arr[i])
            p0, p1 = float(p_arr[i - 1]), float(p_arr[i])
            if v0 == v1:
                return float(p1)
            return float(p0 + (target - v0) / (v1 - v0) * (p1 - p0))
    return float(p_arr[-1])


def ic_decay_table(
    factor_registry,
    prices: pd.DataFrame,
    open_: pd.DataFrame | None = None,
    tradable: pd.DataFrame | None = None,
    masks: dict | None = None,
    periods: list | None = None,
    names: list | None = None,
    apply_exec_mask: bool | None = None,
    apply_label_exec_mask: bool | None = None,
) -> pd.DataFrame:
    """
    IC 衰减表。

    ``factor_registry`` 可以是普通 dict（向后兼容）或 `_LazyFactorRegistry`：
      - LazyRegistry 模式下逐因子 __getitem__ 加载面板 → 算 5 个 period 的 IC
        → 释放全面板（峰值 = 1 个面板 ~48MB），而非同时持有全部面板。
        此模式需传 names 列表（或从 LazyRegistry._names 推导）。
    """
    if periods is None:
        periods = [5, 10, 20, 40, 60]
    is_lazy = hasattr(factor_registry, "release_cache") and hasattr(factor_registry, "__getitem__")

    # 预构建各 period 的 forward_return（避免在内层循环重复构建）
    fwd_cache: dict[int, pd.DataFrame] = {}
    if apply_exec_mask is None:
        apply_exec_mask = apply_label_exec_mask
    for p in periods:
        fwd_cache[p] = build_forward_return(
            prices, open_, p, masks=masks, apply_exec_mask=apply_exec_mask,
        )

    if is_lazy:
        names_iter = names if names is not None else (
            list(factor_registry._names) if hasattr(factor_registry, "_names") else []
        )
    else:
        names_iter = list(factor_registry.keys())

    rows = []
    for name in names_iter:
        if is_lazy:
            if name not in factor_registry:
                continue
            try:
                factor = factor_registry[name]
            except KeyError:
                continue
        else:
            factor = factor_registry[name]
        if factor is None or factor.empty:
            continue
        row = {"因子": name}
        abs_ics = []
        for p in periods:
            fwd = fwd_cache[p]
            ic_s = compute_ic_series(factor, fwd, tradable=tradable)
            mean_ic = float(ic_s.mean()) if len(ic_s) else np.nan
            row[f"{p}日"] = round(mean_ic, 4)
            abs_ics.append(abs(mean_ic) if np.isfinite(mean_ic) else np.nan)
            if len(ic_s) > 1:
                std0 = float(ic_s.std(ddof=0))
                row[f"ICIR_lag{p}"] = round(mean_ic / std0, 4) if std0 and std0 > 0 else np.nan
            else:
                row[f"ICIR_lag{p}"] = np.nan
            row[f"t_lag{p}"] = round(newey_west_t(ic_s), 2) if len(ic_s) >= 3 else np.nan
        hl = _half_life(periods, abs_ics)
        row["half_life"] = round(float(hl), 1) if np.isfinite(hl) else np.nan
        ic_vals = {p: abs(row[f"{p}日"]) for p in periods if not np.isnan(row[f"{p}日"])}
        if ic_vals:
            row["最优期"] = f"{max(ic_vals, key=ic_vals.get)}日"
        rows.append(row)
        del factor

    if is_lazy:
        factor_registry.release_cache()
    del fwd_cache
    return pd.DataFrame(rows).set_index("因子") if rows else pd.DataFrame()


def factor_corr_matrix(
    factor_registry,
    prices: pd.DataFrame,
    sample_step: int = 20,
    names: list | None = None,
) -> pd.DataFrame:
    """
    因子相关矩阵（截面 spearman corr 在采样日上的均值）。

    ``factor_registry`` 可以是普通 dict（向后兼容）或 `_LazyFactorRegistry`：
      - LazyRegistry 模式下逐因子加载 → 取 sample_dates 切片 → 释放全面板
        （峰值 = 1 个面板 + 切片集合 ~66MB，而非 |names| × 48MB）。
        此模式需传 names 列表。
    """
    sample_dates = prices.index[::sample_step]
    is_lazy = hasattr(factor_registry, "release_cache") and hasattr(factor_registry, "__getitem__")

    if is_lazy:
        names_iter = names if names is not None else (
            list(factor_registry._names) if hasattr(factor_registry, "_names") else []
        )
        # 流式收集每个因子的 sample_dates 切片
        items_list: list[tuple[str, pd.DataFrame]] = []
        for name in names_iter:
            if name not in factor_registry:
                continue
            try:
                panel = factor_registry[name]
            except KeyError:
                continue
            if panel is None or panel.empty:
                continue
            sub = panel.loc[panel.index.intersection(sample_dates)]
            items_list.append((name, sub.astype(np.float32, copy=False)))
            del panel
        factor_registry.release_cache()
        # 用切片算 corr
        corr_list = []
        for date in sample_dates:
            row = {}
            for name, sub in items_list:
                if date in sub.index:
                    row[name] = sub.loc[date]
            if len(row) == len(items_list):
                df_slice = pd.DataFrame(row).dropna()
                if len(df_slice) > 30:
                    corr_list.append(df_slice.corr(method="spearman"))
        if not corr_list:
            return pd.DataFrame()
        return pd.concat(corr_list).groupby(level=0).mean()
    else:
        corr_list = []
        for date in sample_dates:
            row = {}
            for name, factor in factor_registry.items():
                if date in factor.index:
                    row[name] = factor.loc[date]
            if len(row) == len(factor_registry):
                df_slice = pd.DataFrame(row).dropna()
                if len(df_slice) > 30:
                    corr_list.append(df_slice.corr(method="spearman"))
        if not corr_list:
            return pd.DataFrame()
        return pd.concat(corr_list).groupby(level=0).mean()
