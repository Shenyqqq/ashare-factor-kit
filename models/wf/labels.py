"""
Label construction for walk-forward training.

Supports cross-sectional standardization of the regression target (Issue ⑤),
plus Barra+industry neutralized residual labels (P0-4).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from utils.wls import wls_residual

# ``--feature-neutralize`` 控制变量集合。``barra`` = 9 风格 + 行业（现状）；
# ``size_industry`` = 仅 Barra_Size（东财 circ_mv 的 log）+ 行业哑变量；
# ``size`` = 仅 Barra_Size，无行业（PIT 行业覆盖不足时的退路）。
NEUT_CONTROLS_BARRA = "barra"
NEUT_CONTROLS_SIZE_INDUSTRY = "size_industry"
NEUT_CONTROLS_SIZE = "size"
NEUT_CONTROLS_CHOICES = (
    NEUT_CONTROLS_BARRA,
    NEUT_CONTROLS_SIZE_INDUSTRY,
    NEUT_CONTROLS_SIZE,
)
SIZE_NEUT_FACTOR = "Barra_Size"


def normalize_neut_controls(
    value: str | None,
    *,
    missing_warn: bool = False,
) -> str:
    """规范化 ``neut_controls``；缺字段时默认 ``barra``。"""
    if value is None or (isinstance(value, str) and not str(value).strip()):
        if missing_warn:
            logger.warning(
                "manifest 无 neut_controls 字段（旧训练产物），按 barra 默认处理"
            )
        return NEUT_CONTROLS_BARRA
    v = str(value).strip().lower()
    if v not in NEUT_CONTROLS_CHOICES:
        raise ValueError(
            f"未知 neut_controls={value!r}，可选: {list(NEUT_CONTROLS_CHOICES)}"
        )
    return v


def select_neut_control_factors(
    barra_factors: dict[str, pd.DataFrame] | None,
    neut_controls: str | None = NEUT_CONTROLS_BARRA,
) -> dict[str, pd.DataFrame] | None:
    """按 ``neut_controls`` 从完整 Barra dict 取出残差化控制变量。

    ``barra``：原样返回 9 风格。
    ``size_industry`` / ``size``：只留 ``Barra_Size``（log 流通市值）。
    行业哑变量由 ``residualize_panel`` 的 ``industry_map`` / ``industry_panel``
    单独拼，不在此 dict（``size`` 模式不拼行业）。
    """
    mode = normalize_neut_controls(neut_controls)
    if barra_factors is None:
        return None
    if mode == NEUT_CONTROLS_BARRA:
        return barra_factors
    size = barra_factors.get(SIZE_NEUT_FACTOR)
    if size is None or getattr(size, "empty", False):
        raise ValueError(
            f"neut-controls={mode} 需要 Barra_Size（东财 circ_mv 的 log），"
            "但 get_barra_factors 未产出该面板"
        )
    return {SIZE_NEUT_FACTOR: size}


def cross_sectional_rank(y: np.ndarray) -> np.ndarray:
    """Percentile rank in [0, 1]."""
    if len(y) == 0:
        return y
    order = y.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(y))
    return ranks / max(len(y) - 1, 1)


def cross_sectional_zscore(y: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score; zero std → zeros."""
    if len(y) < 2:
        return np.zeros_like(y, dtype=float)
    mu, sigma = np.nanmean(y), np.nanstd(y)
    if sigma < 1e-12:
        return np.zeros_like(y, dtype=float)
    return (y - mu) / sigma


def top_cs_zscore_label(
    y: np.ndarray,
    top_frac: float = 0.4,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    截面前 ``top_frac`` 保留全截面 ``cs_zscore``，其余置常数。

    公式（每个调仓日截面、无泄漏）：
      z    = cs_zscore(y)                 # 全截面 μ/σ，与默认 cs_zscore 同口径
      keep = rank_pct(y) >= 1 - top_frac  # top_frac=0.4 → 分位 ≥ 0.60
      y'   = where(keep, z, fill_value)   # 默认 fill_value=0

    阈上保留的是全截面 z-score 幅度（非区内 re-rank / 区内再 z-score）。
    """
    y_f = np.asarray(y, dtype=float)
    if len(y_f) == 0:
        return y_f.astype(np.float32)
    if not (0.0 < float(top_frac) <= 1.0):
        raise ValueError(f"top_frac 须在 (0, 1]，收到 {top_frac}")
    z = cross_sectional_zscore(y_f)
    rank_pct = cross_sectional_rank(y_f)
    thr = 1.0 - float(top_frac)
    out = np.where(rank_pct >= thr, z, float(fill_value))
    return out.astype(np.float32)


def long_bias_sample_weights(
    y: np.ndarray,
    top_frac: float = 0.4,
    bottom_weight: float = 0.25,
    curve: str = "smooth",
    transition: float = 0.10,
) -> np.ndarray:
    """
    多头偏置样本权重（标签本身保持连续，不在此函数改 y）。

    按截面 ``rank_pct(y)`` 赋权，使 top 区 loss 权重大、bottom 区仍保留
    对比信息（权重 > 0，避免 60% 硬置 0 主导 MSE）。

    Parameters
    ----------
    y : ndarray
        原始截面收益（或任意可排序量）；权重只依赖其截面秩。
    top_frac : float
        多头区占比；``thr = 1 - top_frac``（默认 0.4 → thr=0.60）。
    bottom_weight : float
        下侧基准权重，须在 ``(0, 1]``；默认 0.25。
    curve : ``"smooth"`` | ``"step"``
        ``step``：rank≥thr → 1，否则 → bottom_weight。
        ``smooth``：在 thr 附近用 hermite smoothstep 过渡（带宽
        ``2*transition``），避免权重悬崖。
    transition : float
        仅 ``smooth``：过渡半宽（rank 单位），默认 0.10。
    """
    y_f = np.asarray(y, dtype=float)
    n = len(y_f)
    if n == 0:
        return np.array([], dtype=np.float64)
    if not (0.0 < float(top_frac) <= 1.0):
        raise ValueError(f"top_frac 须在 (0, 1]，收到 {top_frac}")
    bw = float(bottom_weight)
    if not (0.0 < bw <= 1.0):
        raise ValueError(f"bottom_weight 须在 (0, 1]，收到 {bottom_weight}")
    rank_pct = cross_sectional_rank(y_f)
    thr = 1.0 - float(top_frac)
    if curve == "step":
        return np.where(rank_pct >= thr, 1.0, bw).astype(np.float64)
    if curve == "smooth":
        half = max(float(transition), 1e-6)
        t = np.clip((rank_pct - (thr - half)) / (2.0 * half), 0.0, 1.0)
        s = t * t * (3.0 - 2.0 * t)  # hermite smoothstep
        return (bw + (1.0 - bw) * s).astype(np.float64)
    raise ValueError(f"未知 curve: {curve}，可选 smooth | step")


def rank_tail_sample_weights(
    y: np.ndarray,
    mid_weight: float = 0.6,
) -> np.ndarray:
    """
    截面分位 U 形样本权重（两端对比 + 头部区分）。

    每个训练截面用该样本自己的 y：已是 cs_rank ∈ [0, 1]（且几乎铺满
    单位区间）则直接当 r；否则先在该截面 ``cross_sectional_rank`` 到 0–1。

    公式（平滑抛物线，非三档阶跃）::

        w = mid + (1 - mid) * (2 * |r - 0.5|) ** 2

    mid=0.6 → w(0)=w(1)=1.0，w(0.5)=0.6；|r-0.5|≥0.4（r≤0.1 或 r≥0.9）
    时 w≥0.856，接近两端满权。mid≥1 → 全 1（关闭）。
    """
    y_f = np.asarray(y, dtype=float)
    n = len(y_f)
    if n == 0:
        return np.array([], dtype=np.float64)
    mid = float(mid_weight)
    if not np.isfinite(mid) or mid <= 0:
        raise ValueError(f"mid_weight 须为正有限值，收到 {mid_weight}")
    if mid >= 1.0 - 1e-12:
        return np.ones(n, dtype=np.float64)
    finite = y_f[np.isfinite(y_f)]
    use_as_rank = (
        finite.size >= 2
        and float(np.min(finite)) >= -1e-9
        and float(np.max(finite)) <= 1.0 + 1e-9
        and float(np.max(finite) - np.min(finite)) >= 0.8
    )
    if use_as_rank:
        r = np.clip(y_f, 0.0, 1.0)
    else:
        r = cross_sectional_rank(y_f)
    # w = mid + (1-mid) * (2*|r-0.5|)**2
    w = mid + (1.0 - mid) * np.square(2.0 * np.abs(r - 0.5))
    return w.astype(np.float64)


def soft_truncate_rank_label(
    y: np.ndarray,
    top_frac: float = 0.4,
    floor_slope: float = 0.25,
) -> np.ndarray:
    """
    软截断多头标签：连续、无 60% 常数零平台。

    公式（截面 rank_pct ∈ [0,1]，``τ = 1 - top_frac``）：
      r ≥ τ :  y' = (r - τ) / (1 - τ)          ∈ [0, 1]
      r < τ :  y' = floor_slope * (r - τ) / τ   ∈ [-floor_slope, 0)

    在 τ 处连续（y'=0）；下侧为小负斜率而非硬置 0，保留排序对比。
    """
    y_f = np.asarray(y, dtype=float)
    if len(y_f) == 0:
        return y_f.astype(np.float32)
    if not (0.0 < float(top_frac) <= 1.0):
        raise ValueError(f"top_frac 须在 (0, 1]，收到 {top_frac}")
    slope = float(floor_slope)
    if slope < 0.0:
        raise ValueError(f"floor_slope 须 ≥ 0，收到 {floor_slope}")
    r = cross_sectional_rank(y_f)
    tau = 1.0 - float(top_frac)
    above = (r - tau) / max(1.0 - tau, 1e-12)
    below = slope * (r - tau) / max(tau, 1e-12)
    out = np.where(r >= tau, above, below)
    return out.astype(np.float32)


def triple_barrier_label(
    prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    signal_dates: list,
    hold_period: int,
    vol_window: int = 20,
    upper_mult: float = 2.0,
    lower_mult: float = 1.5,
    label_type: str = "sign",
) -> pd.DataFrame:
    """
    AFML §3 triple-barrier 标签（路径依赖 + 波动率自适应）。

    对每个 signal_date 的每只股票：
      1. 入场价 = open[signal_date + 1]（次日开盘）
      2. σ = rolling std of daily returns, vol_window 天（截至 signal_date）
      3. 上障碍 = entry * (1 + upper_mult * σ)
         下障碍 = entry * (1 - lower_mult * σ)
      4. 从 signal_date+1 到 signal_date+hold_period 逐日检查累计收益是否触碰：
         - 先触碰上障碍 → +1（或触碰时实际收益）
         - 先触碰下障碍 → -1（或触碰时实际收益）
         - 都未触碰 → 0（或残余收益 close[end]/entry-1）
      5. 入场价/σ 缺失（停牌/退市/历史不足）→ 标签为 NaN

    实现说明：
      - 用 close 近似 high/low 检查障碍（无 high/low 时）；此处传入 prices 即可。
      - 对每个 signal_date 一次性向量化：取 hold_period 天 price 矩阵（days×stocks），
        计算相对 entry 的累计收益，用 argmax 找首个触碰日。
      - 同日同时触碰上下障碍时按"上障碍优先"判定（AFML 默认约定）。

    Parameters
    ----------
    prices : pd.DataFrame
        日频收盘价（index=日期, columns=股票），用于路径检查与 σ 计算。
    open_prices : pd.DataFrame
        日频开盘价，用于确定入场价。
    signal_dates : list
        调仓日列表（Timestamp 或可解析字符串）。
    hold_period : int
        持仓天数（时间障碍宽度，单位：交易日）。
    vol_window : int, default 20
        波动率回看窗口（交易日）。
    upper_mult, lower_mult : float
        上/下障碍的 σ 倍数。
    label_type : ``"sign"`` | ``"return"``
        ``sign`` → 标签 ∈ {+1, -1, 0}；``return`` → 触碰时实际收益（未触碰则残余收益）。

    Returns
    -------
    pd.DataFrame
        index=signal_dates, columns=stocks, values=label（NaN 表示该股该日无效）。
    """
    if prices is None or open_prices is None:
        return pd.DataFrame()
    prices = pd.DataFrame(prices).sort_index()
    open_prices = pd.DataFrame(open_prices).sort_index()
    common = prices.columns.intersection(open_prices.columns)
    if len(common) == 0:
        return pd.DataFrame()
    prices = prices[common]
    open_prices = open_prices[common]

    # σ：日收益的滚动标准差（min_periods 取窗口一半，避免开市初期全 NaN）
    daily_ret = prices.pct_change()
    min_per = max(5, vol_window // 2)
    vol = daily_ret.rolling(vol_window, min_periods=min_per).std()

    idx = prices.index
    pos = {pd.Timestamp(d): i for i, d in enumerate(idx)}
    n = len(idx)
    cols = prices.columns

    out: dict = {}
    for sd in signal_dates:
        sd = pd.Timestamp(sd)
        if sd not in pos:
            continue
        i_sig = pos[sd]
        i_entry = i_sig + 1
        if i_entry >= n:
            continue
        i_end = min(i_sig + hold_period, n - 1)
        if i_end < i_entry:
            continue

        entry = open_prices.iloc[i_entry]
        # asof 取截至 sd 的最近一次有效 σ，处理边界 NaN
        try:
            sigma = vol.asof(sd)
        except Exception:
            sigma = pd.Series(np.nan, index=cols)
        if sigma is None or not isinstance(sigma, pd.Series):
            sigma = pd.Series(np.nan, index=cols)

        window = prices.iloc[i_entry:i_end + 1]
        entry_arr = entry.reindex(cols).values.astype(np.float64)
        sigma_arr = sigma.reindex(cols).values.astype(np.float64)

        # 相对入场的累计收益（days × stocks）
        ret = window[cols].values / entry_arr[np.newaxis, :] - 1.0
        # 停牌日 NaN 当作 0（不触发任何障碍）
        ret = np.where(np.isnan(ret), 0.0, ret)
        # σ 缺失则阈值放到 ±inf（永不触发），后续 valid 标志会把标签设为 NaN
        up_thr = np.where(np.isnan(sigma_arr), np.inf, upper_mult * sigma_arr)
        dn_thr = np.where(np.isnan(sigma_arr), -np.inf, -lower_mult * sigma_arr)

        up_touch = ret >= up_thr[np.newaxis, :]
        dn_touch = ret <= dn_thr[np.newaxis, :]
        up_any = up_touch.any(axis=0)
        dn_any = dn_touch.any(axis=0)
        up_first = np.argmax(up_touch, axis=0)
        dn_first = np.argmax(dn_touch, axis=0)
        no_touch_d = window.shape[0]  # 哨兵：表示"未触碰"
        up_d = np.where(up_any, up_first, no_touch_d)
        dn_d = np.where(dn_any, dn_first, no_touch_d)

        # 同日同时触碰按上障碍优先
        upper_wins = up_any & (up_d <= dn_d)
        lower_wins = dn_any & (dn_d < up_d)

        valid = (~np.isnan(entry_arr)) & (entry_arr != 0.0) & (~np.isnan(sigma_arr))

        if label_type == "sign":
            lab = np.where(valid, 0.0, np.nan)
            lab = np.where(valid & upper_wins, 1.0, lab)
            lab = np.where(valid & lower_wins, -1.0, lab)
        elif label_type == "return":
            touched = upper_wins | lower_wins
            touch_day = np.where(upper_wins, up_d,
                                 np.where(lower_wins, dn_d, no_touch_d))
            safe_day = np.clip(touch_day, 0, window.shape[0] - 1)
            touch_px = np.take_along_axis(
                window[cols].values, safe_day[np.newaxis, :], axis=0,
            )[0]
            touch_ret = touch_px / np.where(entry_arr == 0, np.nan, entry_arr) - 1.0
            end_ret = (
                window[cols].iloc[-1].values
                / np.where(entry_arr == 0, np.nan, entry_arr) - 1.0
            )
            lab = np.where(valid & touched, touch_ret, end_ret)
            lab = np.where(valid, lab, np.nan)
        else:
            raise ValueError(f"未知 label_type: {label_type}，可选 sign | return")

        out[sd] = pd.Series(lab, index=cols)

    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out).T
    df.index.name = "signal_date"
    return df


def transform_labels(
    y: np.ndarray,
    mode: str = "raw",
    *,
    barra_factors: pd.DataFrame | None = None,
    industry_dummies: pd.DataFrame | None = None,
    top_frac: float = 0.4,
    floor_slope: float = 0.25,
) -> np.ndarray:
    """
    Transform forward-return labels for training.

    Parameters
    ----------
    mode : ``raw`` | ``cs_rank`` | ``cs_zscore`` | ``top40_cs_zscore``
          | ``cs_rank_softlong`` | ``barra_residual`` | ``triple_barrier``
    barra_factors, industry_dummies :
        仅 ``mode='barra_residual'`` 时使用，需为对齐到当期股票索引的
        DataFrame（stock × control）。省略时退化为 ``cs_zscore``。
    top_frac :
        ``top40_cs_zscore`` / ``cs_rank_softlong``：多头区占比（默认 0.4）。
    floor_slope :
        仅 ``cs_rank_softlong``：下侧负斜率幅度（默认 0.25）。

    ``mode='triple_barrier'`` 时，``y`` 应已是预计算的当期 triple-barrier
    标签（sign 或 return，由调用方从 ``triple_barrier_label`` 面板按当期
    行/股票索引取出）。此处仅做截面 z-score，保证与其它模式同尺度。
    """
    if mode == "raw":
        return y.astype(np.float32)
    if mode == "cs_rank":
        return cross_sectional_rank(y.astype(float)).astype(np.float32)
    if mode == "cs_zscore":
        return cross_sectional_zscore(y.astype(float)).astype(np.float32)
    if mode == "top40_cs_zscore":
        return top_cs_zscore_label(y, top_frac=top_frac, fill_value=0.0)
    if mode == "cs_rank_softlong":
        return soft_truncate_rank_label(
            y, top_frac=top_frac, floor_slope=floor_slope,
        )
    if mode == "triple_barrier":
        # y 已是当期预计算的 triple-barrier 标签；做截面标准化供回归训练
        return cross_sectional_zscore(y.astype(float)).astype(np.float32)
    if mode == "barra_residual":
        if barra_factors is None and industry_dummies is None:
            # 无控制变量时退化为截面 z-score（保留向后兼容）
            return cross_sectional_zscore(y.astype(float)).astype(np.float32)
        # trainer 传入的 y 是 ndarray（无股票索引）；控制矩阵已按 stock_idx 对齐，
        # 必须用控制矩阵的 index 挂到 y 上，否则 residual_return_label 的 reindex
        # 对不上股票代码 → 控制列全 0 → 静默退化成 cs_zscore（ablation 失效）。
        y_arr = np.asarray(y, dtype=float)
        if isinstance(y, pd.Series) and len(y) == len(y_arr):
            y_series = y.astype(float)
        elif barra_factors is not None and len(barra_factors) == len(y_arr):
            y_series = pd.Series(y_arr, index=barra_factors.index)
        elif industry_dummies is not None and len(industry_dummies) == len(y_arr):
            y_series = pd.Series(y_arr, index=industry_dummies.index)
        else:
            y_series = pd.Series(y_arr)
        resid = residual_return_label(y_series, barra_factors, industry_dummies)
        # 残差再做截面 z-score，保证训练稳定性（与 cs_zscore 同尺度）
        return cross_sectional_zscore(resid.values.astype(float)).astype(np.float32)
    raise ValueError(
        f"未知 label_mode: {mode}，可选 raw | cs_rank | cs_zscore | top40_cs_zscore "
        f"| cs_rank_softlong | triple_barrier | barra_residual"
    )


def _align_controls(ctrl: pd.DataFrame, target_index: pd.Index) -> pd.DataFrame:
    """按股票索引对齐控制矩阵；无标签重叠且长度相同时退化为位置对齐。"""
    aligned = ctrl.reindex(target_index)
    if aligned.isna().all().all() and len(ctrl) == len(target_index):
        # 索引完全错位（常见：RangeIndex vs 股票代码）→ 按位置挂目标索引
        aligned = pd.DataFrame(
            np.asarray(ctrl, dtype=np.float32),
            index=target_index,
            columns=ctrl.columns,
        )
    return aligned.fillna(0.0)


def residual_return_label(
    y: pd.Series,
    barra_factors: pd.DataFrame | None = None,
    industry_dummies: pd.DataFrame | None = None,
    weights: pd.Series | None = None,
) -> pd.Series:
    """
    Barra + industry neutralized residual return label.

    对单个截面做 WLS ``y ~ [const, Barra_*, industry_dummies_*]``（权重 = √市值），
    取残差作为剔除系统性风格/行业暴露后的"纯 alpha"标签。参考
    ``research/ic/barra.py`` 的纯 IC 残差化实现（同样的控制变量与权重口径）。

    Parameters
    ----------
    y : pd.Series
        当期 forward return，索引为股票代码。
    barra_factors : pd.DataFrame | None
        当期 Barra 风格因子值（stock × 9 Barra 因子，已截面 z-score）。
    industry_dummies : pd.DataFrame | None
        当期行业哑变量（stock × (n_industries - 1)，已 drop_first）。
    weights : pd.Series | None
        当期回归权重（= √市值，索引为股票代码）。None → 等权 OLS。

    Returns
    -------
    pd.Series
        残差，索引与 ``y.dropna()`` 一致。控制变量不足或样本过少时返回原 y。
    """
    if y is None:
        return pd.Series(dtype=float)
    y_s = y if isinstance(y, pd.Series) else pd.Series(y)
    y_s = y_s.dropna()
    if len(y_s) == 0:
        return y_s

    controls = []
    if barra_factors is not None and not barra_factors.empty:
        controls.append(_align_controls(barra_factors, y_s.index))
    if industry_dummies is not None and not industry_dummies.empty:
        controls.append(_align_controls(industry_dummies, y_s.index))

    if not controls:
        return y_s

    X_df = pd.concat(controls, axis=1).dropna(axis=1, how="all").fillna(0.0)
    # 至少需要 2×(控制变量数+1) 个样本，否则 OLS 不可靠
    if len(y_s) < 2 * (X_df.shape[1] + 1) or len(y_s) < 30:
        return y_s

    w_v = None
    if weights is not None:
        w_v = pd.Series(weights).reindex(y_s.index).values

    resid = wls_residual(y_s.values.astype(np.float64), X_df.values, w_v)
    if resid is None:
        return y_s

    return pd.Series(resid.astype(np.float32), index=y_s.index)


def _prepare_industry_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """规范化 PIT 行业长表，供逐日 as-of（避免每截面 copy）。"""
    pit = panel.copy()
    if "effective_date" not in pit.columns and "start_date" in pit.columns:
        pit = pit.rename(columns={"start_date": "effective_date"})
    pit["effective_date"] = pd.to_datetime(pit["effective_date"], errors="coerce")
    if "end_date" in pit.columns:
        pit["end_date"] = pd.to_datetime(pit["end_date"], errors="coerce")
    else:
        pit["end_date"] = pd.NaT
    pit["code"] = pit["code"].astype(str)
    return pit.dropna(subset=["effective_date"])


def _sw_l2_as_of(panel: pd.DataFrame, date) -> pd.Series:
    """``effective_date ≤ t`` 且 ``end_date`` 空或 ``≥ t``；同 code 取最晚生效。"""
    ts = pd.Timestamp(date)
    mask = (panel["effective_date"] <= ts) & (
        panel["end_date"].isna() | (panel["end_date"] >= ts)
    )
    sub = (
        panel.loc[mask]
        .sort_values("effective_date")
        .drop_duplicates(subset=["code"], keep="last")
    )
    if sub.empty or "sw_l2" not in sub.columns:
        return pd.Series(dtype=object)
    return sub.set_index("code")["sw_l2"]


def collapse_rare_industries(
    ind: pd.Series,
    min_n: int,
    other_label: str = "其他",
) -> pd.Series:
    """档内有效样本 < ``min_n`` 的行业并入 ``other_label``。

    ``min_n<=1`` 时原样返回。并完后若只剩一类，调用方应跳过行业哑元。
    """
    if ind is None or getattr(ind, "empty", True):
        return ind
    n_min = int(min_n or 0)
    if n_min <= 1:
        return ind.astype(str)
    s = ind.fillna("未分类").astype(str)
    vc = s.value_counts(dropna=False)
    rare = set(vc[vc < n_min].index)
    if not rare:
        return s
    return s.where(~s.isin(rare), other_label)


def inpool_log_mcap_control(
    circ_mv_row: pd.Series,
    members: pd.Index,
) -> pd.Series:
    """当日池内 ``log(circ_mv)`` 再截面 z-score，作 Size 控制。

    不用全市场 zscore 后的 ``Barra_Size``。池内方差≈0 时填 0（列仍在，
    与截距共线，等于「做了 Size 控制但无可估斜率」）。
    """
    mv = pd.to_numeric(circ_mv_row.reindex(members), errors="coerce")
    log_mv = np.log(mv.where(mv > 0))
    out = np.full(len(members), np.nan, dtype=np.float64)
    v = log_mv.to_numpy(dtype=np.float64, copy=False)
    ok = np.isfinite(v)
    if int(ok.sum()) < 2:
        return pd.Series(out, index=members, dtype=np.float32)
    sub = v[ok]
    std = float(sub.std())
    if std < 1e-12:
        out[ok] = 0.0
        return pd.Series(out, index=members, dtype=np.float32)
    out[ok] = (sub - float(sub.mean())) / std
    return pd.Series(out, index=members, dtype=np.float32)


def _industry_dummy_cols(
    ind: pd.Series,
    min_industry_n: int = 0,
) -> dict[str, pd.Series]:
    """行业哑变量（drop_first），与 IC ``_industry_dummies`` 同约定。"""
    ind_s = ind.fillna("未分类")
    if min_industry_n and int(min_industry_n) > 1:
        ind_s = collapse_rare_industries(ind_s, int(min_industry_n))
    cats = sorted(ind_s.astype(str).unique())
    if len(cats) <= 1:
        return {}
    ref = cats[0]
    return {
        f"_ind_{g}": (ind_s.astype(str) == g).astype(np.float32)
        for g in cats if g != ref
    }


def residualize_panel(
    factor_panel: pd.DataFrame,
    barra_factors: dict[str, pd.DataFrame] | None,
    industry_map: pd.Series | None,
    rebalance_dates: pd.DatetimeIndex,
    min_stocks: int = 30,
    weight_panel: pd.DataFrame | None = None,
    industry_panel: pd.DataFrame | None = None,
    membership_mask: pd.DataFrame | None = None,
    circ_mv: pd.DataFrame | None = None,
    min_industry_n: int = 0,
) -> pd.DataFrame:
    """
    逐截面 WLS: ``factor ~ [1, controls, industry_dummies]``，返回残差面板。

    默认控制变量 = Barra 9 风格 + 行业哑变量 drop_first，与
    ``research/ic/barra.py::compute_pure_ic_fast`` **及同一套回归权重（√市值）**
    对齐。``--neut-controls size_industry`` 时 ``barra_factors`` 只含
    ``Barra_Size``（东财 ``log(circ_mv)``），行业走 PIT ``industry_panel``
    （``effective_date ≤ t`` 当时申万二级）；无 panel 时才回退静态
    ``industry_map``。``size`` 模式不拼行业。
    调用方用 ``select_neut_control_factors`` 子集化，勿把 9 风格 dict 直接
    残差化后再靠缓存键区分。

    Parameters
    ----------
    factor_panel : pd.DataFrame
        单因子宽表 (date × stock)。
    barra_factors : dict[str, pd.DataFrame] | None
        Barra 风格因子名 -> (date × stock) DataFrame。
    industry_map : pd.Series | None
        stock code -> industry category（静态回填，仅无 PIT panel 时用）。
    rebalance_dates : pd.DatetimeIndex
        需要做残差化的截面日期；非这些日期的行返回 NaN。
    min_stocks : int, default 30
        截面有效股票数低于此值则该截面返回 NaN。
    weight_panel : pd.DataFrame | None
        截面回归权重面板 (date × stock)，本仓库口径 = **√市值**
        （见 ``factors/barra_risk.py::barra_regression_weights``）。
        None 时退化为等权 OLS（朴素全市场 OLS 会被小微盘噪声主导）。
    industry_panel : pd.DataFrame | None
        PIT 行业长表（code / effective_date / sw_l2 / end_date）。提供时
        逐调仓日 as-of，忽略静态 ``industry_map`` 的历史回填。
    membership_mask : pd.DataFrame | None
        逐日宇宙 bool 面板。提供时**先**把回归样本裁成当日成员，再估 β
        （禁止全市场 WLS 再切片）。``None`` = 全市场（默认，行为不变）。
    circ_mv : pd.DataFrame | None
        与 ``membership_mask`` 联用：Size 控制改为池内 ``log(circ_mv)`` z-score，
        不用全市场标准化后的 ``Barra_Size`` 列。
    min_industry_n : int, default 0
        档内某行业有效样本少于此值则并入「其他」；``<=1`` 关闭（默认全市场不变）。

    Returns
    -------
    pd.DataFrame
        残差面板，与 ``factor_panel`` 同 shape；非 ``rebalance_dates``
        行为 NaN，无法做 OLS 的截面也为 NaN。
    """
    if factor_panel is None or factor_panel.empty:
        return factor_panel
    has_ind = industry_panel is not None or industry_map is not None
    if (barra_factors is None or len(barra_factors) == 0) and not has_ind:
        # 无控制变量 → 不做残差化，原样返回
        return factor_panel

    out = pd.DataFrame(
        np.nan, index=factor_panel.index, columns=factor_panel.columns,
        dtype=np.float32,
    )
    f_cols = factor_panel.columns
    f_arr = factor_panel.to_numpy(dtype=np.float64, copy=False)
    date_to_row = {pd.Timestamp(d): i for i, d in enumerate(factor_panel.index)}

    use_pit = industry_panel is not None and not getattr(industry_panel, "empty", True)
    pit_prepared: pd.DataFrame | None = (
        _prepare_industry_panel(industry_panel) if use_pit else None
    )

    min_ind = int(min_industry_n or 0)
    use_members = membership_mask is not None

    # 静态路径：行业哑变量一次性构造；PIT / membership 路径每截面 as-of。
    ind_cols: dict | None = None
    if not use_pit and not use_members and industry_map is not None:
        ind_cols = _industry_dummy_cols(industry_map, min_industry_n=min_ind) or None

    for date in rebalance_dates:
        date = pd.Timestamp(date)
        row_i = date_to_row.get(date)
        if row_i is None:
            continue

        # 构造当期 Barra 控制矩阵
        ctrl_cols: dict = {}
        if barra_factors:
            for bname, bdf in barra_factors.items():
                if bdf is None or bdf.empty:
                    continue
                if date in bdf.index:
                    ctrl_cols[bname] = bdf.loc[date].astype(np.float32)
        if not ctrl_cols:
            continue  # 当期无 Barra 因子覆盖 → 跳过该截面
        barra_df = pd.DataFrame(ctrl_cols)
        if use_members:
            if date not in membership_mask.index:
                continue
            mem = membership_mask.loc[date].reindex(barra_df.index)
            mem = mem.fillna(False).astype(bool)
            barra_df = barra_df.loc[mem]
            if len(barra_df) < min_stocks:
                continue
        if (
            use_members
            and circ_mv is not None
            and SIZE_NEUT_FACTOR in barra_df.columns
            and date in circ_mv.index
        ):
            barra_df[SIZE_NEUT_FACTOR] = inpool_log_mcap_control(
                circ_mv.loc[date], barra_df.index,
            )
            other = [c for c in barra_df.columns if c != SIZE_NEUT_FACTOR]
            if other:
                barra_df[other] = barra_df[other].fillna(0.0)
            barra_df = barra_df.dropna(subset=[SIZE_NEUT_FACTOR])
            if len(barra_df) < min_stocks:
                continue
        else:
            barra_df = barra_df.fillna(0.0)

        # 行业哑变量对齐到 barra_df.index
        if use_pit and pit_prepared is not None:
            ind_s = _sw_l2_as_of(pit_prepared, date).reindex(barra_df.index)
            pit_cols = _industry_dummy_cols(ind_s, min_industry_n=min_ind)
            if pit_cols:
                ind_df = pd.DataFrame(pit_cols).reindex(barra_df.index).fillna(0.0)
                X_df = pd.concat([barra_df, ind_df], axis=1)
            else:
                X_df = barra_df
        elif use_members and industry_map is not None:
            ind_s = industry_map.reindex(barra_df.index)
            mem_cols = _industry_dummy_cols(ind_s, min_industry_n=min_ind)
            if mem_cols:
                ind_df = pd.DataFrame(mem_cols).reindex(barra_df.index).fillna(0.0)
                X_df = pd.concat([barra_df, ind_df], axis=1)
            else:
                X_df = barra_df
        elif ind_cols is not None:
            ind_df = pd.DataFrame(ind_cols).reindex(barra_df.index).fillna(0.0)
            X_df = pd.concat([barra_df, ind_df], axis=1)
        else:
            X_df = barra_df

        # factor 截面对齐到 X_df.index
        f_series = pd.Series(f_arr[row_i], index=f_cols).reindex(X_df.index)
        valid = np.isfinite(f_series.values)
        if valid.sum() < min_stocks:
            continue

        f_v = f_series.values[valid].astype(np.float64)
        X_v = X_df.values[valid].astype(np.float64)

        w_v = None
        if weight_panel is not None and date in weight_panel.index:
            w_v = (
                weight_panel.loc[date]
                .reindex(X_df.index)
                .values[valid]
                .astype(np.float64)
            )

        resid = wls_residual(f_v, X_v, w_v)
        if resid is None:
            continue

        # 把残差填回该截面（仅 valid 位置，其余 NaN）
        resid_full = np.full(len(X_df), np.nan, dtype=np.float64)
        resid_full[np.where(valid)[0]] = resid
        out.loc[date] = pd.Series(resid_full, index=X_df.index).reindex(f_cols).values.astype(np.float32)

    return out


def build_industry_dummies(
    industry_map: pd.Series,
    stock_index: pd.Index,
    reference: str | None = None,
) -> pd.DataFrame:
    """
    行业哑变量矩阵（drop_first 避免与常数项共线）。

    与 ``research/ic/barra.py::_industry_dummies`` 同一约定。
    """
    if industry_map is None:
        return pd.DataFrame(index=stock_index)
    ind = industry_map.reindex(stock_index).fillna("未分类")
    cats = sorted(ind.unique())
    if len(cats) <= 1:
        return pd.DataFrame(index=stock_index)
    ref = reference if reference and reference != "drop_first" else cats[0]
    if ref not in cats:
        ref = cats[0]
    cols = {f"_ind_{grp}": (ind == grp).astype(np.float32)
            for grp in cats if grp != ref}
    if not cols:
        return pd.DataFrame(index=stock_index)
    return pd.DataFrame(cols, index=stock_index)


def precompute_label_controls(
    barra_factors: dict[str, pd.DataFrame] | None,
    industry_map: pd.Series | None,
    dates,
) -> dict:
    """
    预计算每个调仓日的标签残差化控制矩阵。

    Returns
    -------
    dict
        ``{date: (barra_df, industry_dummies_df)}``，二者均按当期
        Barra 因子覆盖的股票并集索引。无 Barra 因子的日期不在 dict 中。
    """
    if barra_factors is None and industry_map is None:
        return {}

    controls: dict = {}
    for date in dates:
        cols = {}
        if barra_factors:
            for bname, bdf in barra_factors.items():
                if date in bdf.index:
                    cols[bname] = bdf.loc[date].astype(np.float32)
        if not cols:
            continue
        barra_df = pd.DataFrame(cols).fillna(0.0)
        ind_dummies = (
            build_industry_dummies(industry_map, barra_df.index)
            if industry_map is not None else pd.DataFrame(index=barra_df.index)
        )
        controls[date] = (barra_df, ind_dummies)
    return controls


def compute_return_overlap_weights(
    train_dates: list,
    hold_period_days: int,
    next_rebalance_date=None,
) -> np.ndarray:
    """
    AFML §4: 相邻训练样本的 forward_return 标签时间重叠时降权。

    样本权重 ∝ |标签独有天数| / |标签总天数|，乘以 time_decay。
    默认配置（调仓间隔≈hold_period）下 overlap≈0 → 权重≈1.0（无变化）。
    override 配置（调仓更频繁）下 overlap>0 → 降权。
    下限 0.1：即使 95% 重叠也保留 10% 权重。

    Parameters
    ----------
    train_dates : list
        训练调仓日列表（Timestamp 或可被 pd.to_datetime 解析的字符串）。
    hold_period_days : int
        持仓天数（与 forward_return 窗口一致）。
    next_rebalance_date : Timestamp / str / None
        预测日（可选），用于检查最后一个训练样本与预测日的标签重叠。

    Returns
    -------
    np.ndarray
        长度 = len(train_dates) 的逐日权重（每个调仓日一个权重，
        调用方需自行展开到逐样本）。
    """
    n = len(train_dates)
    if n == 0:
        return np.array([], dtype=np.float64)
    if hold_period_days is None or hold_period_days <= 0:
        return np.ones(n, dtype=np.float64)

    ts = pd.to_datetime(list(train_dates))
    hold_td = pd.Timedelta(days=int(hold_period_days))
    one_day = pd.Timedelta(days=1)

    weights = np.ones(n, dtype=np.float64)
    for i in range(n):
        # 当前训练样本标签区间 [d_i + 1, d_i + hold_period_days]
        start_i = ts[i] + one_day
        end_i = ts[i] + hold_td
        # 下一个参考日：相邻训练日；最后一个样本则用预测日（若提供）
        if i + 1 < n:
            ref = ts[i + 1]
        elif next_rebalance_date is not None:
            ref = pd.Timestamp(next_rebalance_date)
        else:
            continue  # 无后续参考日 → 权重保持 1.0
        start_ref = ref + one_day
        end_ref = ref + hold_td
        # 标签区间日历日重叠
        overlap_start = max(start_i, start_ref)
        overlap_end = min(end_i, end_ref)
        if overlap_end >= overlap_start:
            overlap_days = int((overlap_end - overlap_start).days) + 1
        else:
            overlap_days = 0
        overlap_ratio = overlap_days / float(hold_period_days)
        # 折中版：1 - overlap_ratio * 0.5，下限 0.1
        weights[i] = max(0.1, 1.0 - overlap_ratio * 0.5)
    return weights


def normalize_sample_weights_by_universe(
    weights: np.ndarray,
    dates_per_row: list,
    stocks_per_date: list[int],
) -> np.ndarray:
    """
    Per-date sample weight normalization for universe size changes (P1-2).

    A 股股票池随时间扩张，早期调仓日股票少、晚期股票多。原始 decay 权重
    对每只股票等同视之，导致晚期调仓日（股票多）的总权重远大于早期，
    训练样本被晚期主导。此处按 ``decay * (n_stocks_date / max_n_stocks)``
    缩放，让每个调仓日的总权重正比于其股票数占比而非绝对数量。

    Parameters
    ----------
    weights : np.ndarray
        已展开的逐样本权重（长度 = sum(stocks_per_date)），由 _stack_cached
        按 date 顺序 extend 而成。
    dates_per_row : list
        每个权重行对应的调仓日（长度 = len(stocks_per_date)，仅用于诊断/对齐）。
    stocks_per_date : list[int]
        每个调仓日的有效股票数。

    Returns
    -------
    np.ndarray
        归一化后的权重，长度与 ``weights`` 相同。
    """
    if len(stocks_per_date) == 0 or len(weights) == 0:
        return weights
    max_n = max(stocks_per_date)
    if max_n <= 0:
        return weights
    scale_per_date = np.array(
        [n / max_n for n in stocks_per_date], dtype=np.float64,
    )
    # 把逐日 scale 展开到逐样本
    scale_expanded = np.repeat(scale_per_date, stocks_per_date)
    if len(scale_expanded) != len(weights):
        # 长度不匹配时（理论上不应发生），安全降级返回原权重
        return weights
    return (weights * scale_expanded).astype(weights.dtype)
