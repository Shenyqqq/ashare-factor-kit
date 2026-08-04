"""
backtest/regime.py — 仓位体制（position-control regime）信号

市场级标量 → ``target_exposure ∈ [e_min, 1]``，用于回测总敞口缩放（现金收益记 0）。
**不**注入 ML 因子矩阵 X（旧 ``市场*`` / ``HMM_*`` / ``轮动_*`` 路径已退役）。

信号（PIT：全部 ``shift(1)``，调仓日 T 仅用 ≤T−1 信息）
------------------------------------------------
1. ``mkt_trend``  — 中证全指相对 MA60：``close/MA60 − 1``；``>0`` → risk-on
2. ``mkt_vol``    — 20d 实现波动 / 252d 滚动中位数；``<1`` → risk-on（波动未抬升）
3. ``mkt_breadth``— 个股站上 MA20 占比；``>0.5`` → risk-on
4. ``size_style`` — 可选 SMB（小盘 − 大盘 20d 收益，需 circ_mv）；仅记录，不进合成

合成规则（v0，简单计分）
----------------------
``score = I(trend_on) + I(vol_ok) + I(breadth_ok)`` ∈ {0,1,2,3}
``target_exposure = e_min + (1 − e_min) × score / 3``

可选 ``force_exposure`` 覆盖合成结果（人工降仓钩子）。

回测应用
--------
对非 benchmark track：``r_eff = exposure × r_invested``（现金收益 0）。
flag 关闭时不调用本模块，quantile 路径与旧实验 bit-identical。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class PositionRegimeConfig:
    """仓位体制参数。"""

    e_min: float = 0.30
    trend_ma: int = 60
    vol_window: int = 20
    vol_lookback: int = 252
    breadth_ma: int = 20
    size_window: int = 20
    size_quantile: float = 0.3
    force_exposure: float | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.e_min <= 1.0):
            raise ValueError(f"e_min 须在 [0,1]，got {self.e_min}")
        if self.force_exposure is not None and not (0.0 <= self.force_exposure <= 1.0):
            raise ValueError(
                f"force_exposure 须在 [0,1]，got {self.force_exposure}"
            )


def _as_close_series(market_prices: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(market_prices, pd.Series):
        s = market_prices.copy()
    else:
        if market_prices.shape[1] == 0:
            raise ValueError("market_prices 无列")
        col = "close" if "close" in market_prices.columns else market_prices.columns[0]
        s = market_prices[col]
    s = pd.to_numeric(s, errors="coerce")
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def compute_position_regime(
    market_prices: pd.DataFrame | pd.Series,
    prices: pd.DataFrame | None = None,
    clean_ret: pd.DataFrame | None = None,
    circ_mv: pd.DataFrame | None = None,
    config: PositionRegimeConfig | None = None,
) -> pd.DataFrame:
    """
    计算日频体制信号与 ``target_exposure``。

    Returns
    -------
    DataFrame
        index=交易日；列含 raw 信号、布尔 risk-on、``target_exposure``。
        所有列已 ``shift(1)``（PIT）。
    """
    cfg = config or PositionRegimeConfig()
    idx = _as_close_series(market_prices)
    idx_ret = idx.pct_change()

    # ── 1. trend ──
    ma = idx.rolling(cfg.trend_ma, min_periods=max(20, cfg.trend_ma // 2)).mean()
    mkt_trend = (idx / ma.replace(0, np.nan) - 1.0).replace([np.inf, -np.inf], np.nan)

    # ── 2. vol（相对历史中位数）──
    vol = idx_ret.rolling(cfg.vol_window, min_periods=max(5, cfg.vol_window // 2)).std()
    vol_med = vol.rolling(cfg.vol_lookback, min_periods=60).median().replace(0, np.nan)
    mkt_vol = (vol / vol_med).replace([np.inf, -np.inf], np.nan)

    # ── 3. breadth（% above MA）──
    mkt_breadth = pd.Series(np.nan, index=idx.index, dtype="float64")
    if prices is not None and not prices.empty:
        px = prices.reindex(index=idx.index)
        ma_b = px.rolling(cfg.breadth_ma, min_periods=max(5, cfg.breadth_ma // 2)).mean()
        above = (px > ma_b).sum(axis=1)
        n_valid = px.notna().sum(axis=1).replace(0, np.nan)
        mkt_breadth = (above / n_valid).astype("float64")
    else:
        logger.warning("position_regime: prices 缺失，mkt_breadth 置 NaN（该信号不计分）")

    # ── 4. size_style（SMB，可选，仅记录）──
    size_style = pd.Series(np.nan, index=idx.index, dtype="float64")
    if circ_mv is not None and clean_ret is not None and not circ_mv.empty:
        try:
            size_style = _smb_relative_return(
                clean_ret.reindex(index=idx.index),
                circ_mv.reindex(index=idx.index),
                window=cfg.size_window,
                q=cfg.size_quantile,
            )
        except Exception as e:
            logger.warning(f"position_regime: size_style 计算失败: {e}")

    # PIT：shift(1)
    mkt_trend = mkt_trend.shift(1)
    mkt_vol = mkt_vol.shift(1)
    mkt_breadth = mkt_breadth.shift(1)
    size_style = size_style.shift(1)

    trend_on = (mkt_trend > 0).astype("float64")
    vol_ok = (mkt_vol < 1.0).astype("float64")
    # breadth 缺失时该分量不参与计分（用 NaN → 合成时跳过）
    breadth_on = pd.Series(np.nan, index=idx.index, dtype="float64")
    valid_b = mkt_breadth.notna()
    breadth_on.loc[valid_b] = (mkt_breadth.loc[valid_b] > 0.5).astype("float64")

    score = trend_on.fillna(0.0) + vol_ok.fillna(0.0)
    n_parts = pd.Series(2.0, index=idx.index)
    has_b = breadth_on.notna()
    score = score.where(~has_b, score + breadth_on.fillna(0.0))
    n_parts = n_parts.where(~has_b, 3.0)

    target = cfg.e_min + (1.0 - cfg.e_min) * (score / n_parts)
    target = target.clip(cfg.e_min, 1.0)
    if cfg.force_exposure is not None:
        target = pd.Series(float(cfg.force_exposure), index=idx.index)

    out = pd.DataFrame(
        {
            "mkt_trend": mkt_trend,
            "mkt_vol": mkt_vol,
            "mkt_breadth": mkt_breadth,
            "size_style": size_style,
            "trend_on": trend_on,
            "vol_ok": vol_ok,
            "breadth_on": breadth_on,
            "score": score,
            "target_exposure": target,
        },
        index=idx.index,
    )
    return out


def _smb_relative_return(
    clean_ret: pd.DataFrame,
    circ_mv: pd.DataFrame,
    window: int = 20,
    q: float = 0.3,
) -> pd.Series:
    """小盘等权累计收益 − 大盘等权累计收益（window 日），PIT 用当日市值分位。"""
    common_cols = clean_ret.columns.intersection(circ_mv.columns)
    if len(common_cols) == 0:
        return pd.Series(np.nan, index=clean_ret.index)
    ret = clean_ret[common_cols]
    mv = circ_mv[common_cols]

    # 滚动窗口收益：Π(1+r)−1；涨跌停日 clean_ret=NaN → 当日该股不参与
    log1p = np.log1p(ret.clip(lower=-0.999999))
    # 用 rolling sum of log 近似；NaN 日跳过（min_periods）
    cum = log1p.rolling(window, min_periods=max(5, window // 2)).sum()
    window_ret = np.expm1(cum)

    rows = []
    for dt in window_ret.index:
        mv_row = mv.loc[dt].dropna()
        if len(mv_row) < 20:
            rows.append(np.nan)
            continue
        lo = mv_row.quantile(q)
        hi = mv_row.quantile(1.0 - q)
        small = mv_row[mv_row <= lo].index
        large = mv_row[mv_row >= hi].index
        wr = window_ret.loc[dt]
        s_ret = wr.reindex(small).mean(skipna=True)
        l_ret = wr.reindex(large).mean(skipna=True)
        if pd.isna(s_ret) or pd.isna(l_ret):
            rows.append(np.nan)
        else:
            rows.append(float(s_ret - l_ret))
    return pd.Series(rows, index=window_ret.index, dtype="float64")


def exposure_at(
    regime_df: pd.DataFrame,
    signal_date: pd.Timestamp,
    default: float = 1.0,
) -> float:
    """取调仓日 target_exposure；缺失则 default（通常 1.0）。"""
    if regime_df is None or regime_df.empty or "target_exposure" not in regime_df.columns:
        return default
    if signal_date not in regime_df.index:
        # 最近不超过 signal_date 的值（PIT）
        hist = regime_df.loc[:signal_date, "target_exposure"].dropna()
        if hist.empty:
            return default
        return float(hist.iloc[-1])
    val = regime_df.loc[signal_date, "target_exposure"]
    if isinstance(val, pd.Series):
        val = val.iloc[0]
    if not np.isfinite(val):
        return default
    return float(np.clip(val, 0.0, 1.0))


def apply_exposure(period_return: float, exposure: float) -> float:
    """
    敞口缩放：``r_eff = exposure × r_invested``，现金收益 0。

    ``NAV_end = exposure × NAV_stock + (1−exposure) × 1``
    → ``r_eff = exposure × (NAV_stock − 1) = exposure × r_invested``。
    """
    if period_return is None or (isinstance(period_return, float) and np.isnan(period_return)):
        return period_return
    exp = float(np.clip(exposure, 0.0, 1.0))
    return float(exp) * float(period_return)


def map_exposure_series(
    regime_df: pd.DataFrame,
    signal_dates: pd.DatetimeIndex | list,
) -> pd.Series:
    """将日频 regime 映射到调仓日序列。"""
    idx = pd.DatetimeIndex(signal_dates)
    return pd.Series(
        [exposure_at(regime_df, d, default=1.0) for d in idx],
        index=idx,
        name="target_exposure",
    )
