"""Per-date Spearman IC with min-stocks guard and configurable rank method."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from config.settings import IC_RANK_METHOD, MIN_IC_STOCKS
from research.ic.universe import apply_tradable_mask


def _to_float32_panel(df: pd.DataFrame) -> pd.DataFrame:
    if df.dtypes.apply(lambda d: d == np.float64).any():
        return df.astype(np.float32)
    return df


def _rank_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank; method from IC_RANK_METHOD config.

    返回 float32 以降低内存（pandas ``rank()`` 默认返回 float64，单面板 95MB → 48MB）。
    """
    method = IC_RANK_METHOD if IC_RANK_METHOD in ("average", "dense", "first") else "average"
    return df.rank(axis=1, method=method, na_option="keep").astype(np.float32)


def compute_ic_series(
    factor: pd.DataFrame,
    forward_return: pd.DataFrame,
    tradable: pd.DataFrame | None = None,
    min_stocks: int | None = None,
) -> pd.Series:
    """
    Vectorized Spearman IC per cross-section date（全日频有效交易日，非调仓日子集）。

    Skips dates with valid_stocks < min_stocks (default MIN_IC_STOCKS from settings).
    胜率 / payoff_hit 与本序列共用同一日期索引。
    """
    min_n = MIN_IC_STOCKS if min_stocks is None else min_stocks
    f, r = apply_tradable_mask(factor, forward_return, tradable)
    common = f.index.intersection(r.index)
    f = _to_float32_panel(f.loc[common])
    r = _to_float32_panel(r.loc[common])

    valid_count = (f.notna() & r.notna()).sum(axis=1)
    f_ranked = _rank_panel(f)
    r_ranked = _rank_panel(r)
    ic = f_ranked.corrwith(r_ranked, axis=1)
    ic = ic.where(valid_count >= min_n)
    return ic.dropna()


def spearman_ic_numpy(
    x: np.ndarray,
    y: np.ndarray,
    min_stocks: int | None = None,
) -> float:
    """Single cross-section Spearman IC (numpy path for Barra residuals)."""
    min_n = MIN_IC_STOCKS if min_stocks is None else min_stocks
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < min_n:
        return np.nan
    method = IC_RANK_METHOD if IC_RANK_METHOD in ("average", "dense", "first") else "average"
    rx = rankdata(x[mask], method=method)
    ry = rankdata(y[mask], method=method)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom <= 0:
        return np.nan
    return float((rx * ry).sum() / denom)
