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
    factor_registry: dict,
    prices: pd.DataFrame,
    open_: pd.DataFrame | None = None,
    tradable: pd.DataFrame | None = None,
    periods: list | None = None,
) -> pd.DataFrame:
    if periods is None:
        periods = [5, 10, 20, 40, 60]
    rows = []
    for name, factor in factor_registry.items():
        row = {"因子": name}
        abs_ics = []
        for p in periods:
            fwd = build_forward_return(prices, open_, p)
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
        row["half_life"] = round(float(_half_life(periods, abs_ics)), 1) if np.isfinite(_half_life(periods, abs_ics)) else np.nan
        ic_vals = {p: abs(row[f"{p}日"]) for p in periods if not np.isnan(row[f"{p}日"])}
        if ic_vals:
            row["最优期"] = f"{max(ic_vals, key=ic_vals.get)}日"
        rows.append(row)
    return pd.DataFrame(rows).set_index("因子")


def factor_corr_matrix(
    factor_registry: dict,
    prices: pd.DataFrame,
    sample_step: int = 20,
) -> pd.DataFrame:
    sample_dates = prices.index[::sample_step]
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
