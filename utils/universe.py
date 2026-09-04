"""
utils/universe.py  —  小盘策略 universe 过滤工具

构造时间序列 mask：每个交易日的小盘股池（PIT 安全，仅用当日及之前的数据）。

定义（每个截面日期 d）:
    eligible(c, d) = (circ_mv[c, d] ∈ [SHELL_CAP_LOWER, SMALL_CAP_UPPER])
                     AND (amount[c, d-20:d].mean() ≥ MIN_AMOUNT_20D)

参数（config/settings.py）:
    SMALL_CAP_UPPER = 150e8    # 流通市值上限 150亿
    SHELL_CAP_LOWER = 8e8      # 流通市值下限 8亿（剔壳股/微盘流动性陷阱）
    MIN_AMOUNT_20D  = 2000e4   # 20日均成交额 ≥ 2000万

降级模式：
    circ_mv 不可用时（AKShare fallback 路径仅有 total_mv），
    用 total_mv 近似（total_mv ≥ circ_mv，上界收紧、下界放宽，
    整体偏保守剔除小盘边缘股）。

────────────────────────────────────────────────────────────────────────────────
后续 wiring 接入点（本期不接，留给后续任务）：
────────────────────────────────────────────────────────────────────────────────
1. **IC 截面 mask** — `research/ic/universe.py::build_ic_tradability_mask`
   在该函数末尾追加 `tradable &= small_cap_mask`（reindex 对齐后），
   或新增 `small_cap_mask: pd.DataFrame | None` 参数传入。
   入口：`research/ic_analysis_v2.py` 调 `compute_factor_ic` 时
   把 `build_small_cap_universe(...)` 结果作为 `tradable` 的额外约束。

2. **ML `build_factor_dataset` 截面过滤** — `strategies/ml.py::build_factor_dataset`
   新增 `eligible_mask: pd.DataFrame | None` 参数，
   在 `MLDataset` 组装完 feature/label 后对每期截面应用 mask，
   被剔除的股票不进入训练/预测样本。
   注意：训练样本缩小可能影响 WF 窗口稳定性，建议先用 `--feature-neutralize`
   基线跑通后再开小盘过滤。

3. **回测 `run_quantile_backtest`** — `backtest/quantile.py::run_quantile_backtest`
   新增 `eligible_mask: pd.DataFrame | None` 参数，
   在 `compute_quantile_assignment` 之前对每期截面应用 mask，
   被剔除的股票不参与分组、不进入 Top-N 持仓。
   与 `st_schedule` / `delist_dates` 同口径，PIT 安全。

注意：三处接入都应让 mask 仅作"剔除"（AND），不改变现有可交易池逻辑。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import (
    SMALL_CAP_UPPER,
    SHELL_CAP_LOWER,
    MIN_AMOUNT_20D,
)


def _rolling_amount_mean(amount: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """20 日均成交额（含当日，min_periods=10 容忍月初挂牌新股）。"""
    return amount.rolling(window=window, min_periods=10).mean()


def resample_mv_to_daily(
    mv: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    ffill_limit: int = 7,
) -> pd.DataFrame:
    """把稀疏采样的市值面板重采样到交易日频率并前向填充。

    AKShare baidu 估值接口在 period='近十年' 下返回 5 日间隔采样，
    本函数把它对齐到交易日 index，并用 ffill(limit=ffill_limit) 填充中间缺失。

    Parameters
    ----------
    mv : 原始市值面板（index 可能是日历日/5日采样）
    trading_index : 目标交易日 DatetimeIndex（通常来自 amount.parquet.index）
    ffill_limit : 前向填充最大天数，默认 7（5 日采样 + 2 天容差）。
                  超过该窗口仍无新数据则保留 NaN（避免用陈旧市值）。

    Returns
    -------
    DataFrame(index=trading_index, columns=mv.columns)
    """
    mv = mv.copy()
    mv.index = pd.to_datetime(mv.index)
    # 先按日历日 reindex，ffill，再选交易日
    full_idx = pd.DatetimeIndex(sorted(set(mv.index) | set(trading_index)))
    mv_full = mv.reindex(full_idx).sort_index()
    mv_full = mv_full.ffill(limit=ffill_limit)
    return mv_full.reindex(trading_index)


def build_small_cap_universe(
    circ_mv: pd.DataFrame | None,
    amount: pd.DataFrame | None,
    dates: pd.DatetimeIndex | list | None = None,
    *,
    upper: float = SMALL_CAP_UPPER,
    lower: float = SHELL_CAP_LOWER,
    min_amount: float = MIN_AMOUNT_20D,
    amount_window: int = 20,
    total_mv: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    构造小盘 universe 时间序列 mask。

    Parameters
    ----------
    circ_mv : DataFrame(index=date, columns=code, 单位=元)
        流通市值。优先使用。None 时退化用 total_mv。
    amount : DataFrame(index=date, columns=code, 单位=元)
        成交额。None 时跳过流动性过滤（仅按市值过滤）。
    dates : 可选，限定输出 index；None = circ_mv/total_mv 的并集 index。
    upper : 流通市值上限（默认 SMALL_CAP_UPPER=150e8）。
    lower : 流通市值下限（默认 SHELL_CAP_LOWER=8e8）。
    min_amount : 20 日均成交额下限（默认 MIN_AMOUNT_20D=2000e4）。
    amount_window : 均成交额窗口，默认 20。
    total_mv : 当 circ_mv=None 时的 fallback，单位=元。

    Returns
    -------
    mask : DataFrame(index=date, columns=code, dtype=bool)
        True = 该日该股属于小盘 universe（PIT 安全：仅用当日及之前数据）。
    """
    # 选择市值面板
    if circ_mv is not None:
        mv = circ_mv
        mv_label = "circ_mv"
    elif total_mv is not None:
        mv = total_mv
        mv_label = "total_mv (circ_mv 不可用，近似)"
        logger.warning(
            f"build_small_cap_universe: circ_mv 缺失，用 total_mv 近似 "
            f"（上界 {upper/1e8:.0f}亿、下界 {lower/1e8:.0f}亿 按 total_mv 判定，"
            f"因 total_mv ≥ circ_mv，整体偏保守剔除小盘边缘股）"
        )
    else:
        raise ValueError("circ_mv 和 total_mv 至少需提供一个")

    if amount is None:
        logger.warning(
            "build_small_cap_universe: amount 缺失，跳过流动性过滤 "
            "（仅按市值过滤，可能纳入僵尸股）"
        )

    # 限定 index
    if dates is not None:
        dates = pd.DatetimeIndex(dates)
        mv = mv.reindex(index=dates)
        if amount is not None:
            amount = amount.reindex(index=dates)

    # 市值区间 mask（upper/lower 为 None 表示该侧无界）
    in_band = (mv >= lower) if lower is not None else pd.DataFrame(True, index=mv.index, columns=mv.columns)
    if upper is not None:
        in_band = in_band & (mv <= upper)
    mask = in_band.fillna(False)

    # 流动性 mask
    if amount is not None:
        amt_aligned = amount.reindex(index=mv.index, columns=mv.columns)
        roll_mean = _rolling_amount_mean(amt_aligned, window=amount_window)
        liquid = (roll_mean >= min_amount).fillna(False)
        mask = mask & liquid

    n_dates, n_codes = mask.shape
    avg_per_date = int(mask.sum(axis=1).mean()) if n_dates else 0
    upper_lbl = "无界" if upper is None else f"{upper/1e8:.0f}亿"
    lower_lbl = "无界" if lower is None else f"{lower/1e8:.0f}亿"
    logger.info(
        f"小盘 universe 构造完成 ({mv_label}): "
        f"{n_dates} 日 × {n_codes} 股，平均 {avg_per_date} 只/日，"
        f"上限={upper_lbl} 下限={lower_lbl} "
        f"min_amount={min_amount/1e4:.0f}万/{amount_window}d"
    )
    return mask


def small_cap_universe_size(mask: pd.DataFrame) -> pd.Series:
    """每期小盘股数量序列，便于统计时变性。"""
    return mask.sum(axis=1)


def apply_small_cap_mask(
    panel: pd.DataFrame,
    mask: pd.DataFrame | None,
) -> pd.DataFrame:
    """对宽表面板应用小盘 mask（被剔除的格子置 NaN）。

    与 research/ic/universe.py::apply_tradable_mask 同风格。
    panel/mask 自动 reindex 到公共 index/columns。
    """
    if mask is None:
        return panel
    common_idx = panel.index.intersection(mask.index)
    common_cols = panel.columns.intersection(mask.columns)
    m = mask.reindex(index=common_idx, columns=common_cols).fillna(False)
    return panel.reindex(index=common_idx, columns=common_cols).where(m)


def build_cap_band_mask(
    band: str | None,
    circ_mv: "pd.DataFrame | None",
    amount: "pd.DataFrame | None",
    total_mv: "pd.DataFrame | None" = None,
) -> "pd.DataFrame | None":
    """按市值带名构造 wide bool mask。

    band='all' 或 None 返回 None（不过滤，向后兼容）。
    band 查 ``config.settings.CAP_BANDS`` 得 ``(lower, upper)``，调
    :func:`build_small_cap_universe`。

    - ``lower is None`` → 回退 ``SHELL_CAP_LOWER``（历史兼容；真·无下限请用 ``0.0``，
      如 ``micro_small_100`` / ``micro_30`` / ``micro_lt30``）
    - ``upper is None`` → 无上限
    - 另附 20 日均成交额 ≥ ``MIN_AMOUNT_20D``（与现有 cap-band 一致）
    - ``micro_30`` / ``micro_lt30``：``circ_mv ≤ 30亿`` 且无壳股地板（≠ ``micro`` 的 8~30亿）

    PIT：逐日用当日 ``circ_mv``（优先）判定，不做期末成分回填。
    """
    if band in ("all", None):
        return None
    from config.settings import CAP_BANDS, MIN_AMOUNT_20D, SHELL_CAP_LOWER
    if band not in CAP_BANDS:
        raise ValueError(f"未知 cap-band: {band}，可选: {list(CAP_BANDS.keys())}")
    lower, upper = CAP_BANDS[band]
    if lower is None and upper is None:
        return None
    # None → SHELL_CAP_LOWER；显式 0.0 表示无下限（含微盘）
    eff_lower = SHELL_CAP_LOWER if lower is None else lower
    eff_upper = upper  # None 表示无上限
    return build_small_cap_universe(
        circ_mv=circ_mv,
        amount=amount,
        total_mv=total_mv,
        upper=eff_upper,
        lower=eff_lower,
        min_amount=MIN_AMOUNT_20D,
    )


def build_mcap_percentile_mask(
    circ_mv: pd.DataFrame | None,
    *,
    quantile: float = 0.30,
    total_mv: pd.DataFrame | None = None,
    rebalance_dates: pd.DatetimeIndex | list | None = None,
    trading_index: pd.DatetimeIndex | list | None = None,
) -> pd.DataFrame:
    """
    按截面市值分位构造小市值 universe mask（**仅 mask，不改因子面板列集**）。

    精确定义
    --------
    市值源：优先 ``circ_mv``（流通市值，元）；缺失则 ``total_mv``。

    在每个**调仓日** ``t``（若未传 ``rebalance_dates`` 则对每个交易日）：
      1. 取当日有限市值股票集合 ``S_t = {c : mv[c,t] 有限}``
      2. 升序百分位排名 ``pct_rank(c,t) = rank(mv, ascending=True, method='average', pct=True)``
         （最小市值 → 接近 0，最大 → 接近 1）
      3. ``mask[t,c] = True`` 当且仅当 ``c ∈ S_t`` 且 ``pct_rank(c,t) ≤ quantile``
         （默认 ``quantile=0.30`` → 截面流通市值最低 30%）

    若提供 ``rebalance_dates``：只在调仓日判定成员，再 ``ffill`` 到
    ``trading_index``（缺省 = 市值面板 index），使调仓区间内池子保持不变。

    PIT：仅用当日（及 ffill 的上期调仓判定）市值，无未来信息。
    """
    if not (0.0 < float(quantile) <= 1.0):
        raise ValueError(f"quantile 须在 (0, 1]，收到 {quantile!r}")

    if circ_mv is not None:
        mv = circ_mv
        mv_label = "circ_mv"
    elif total_mv is not None:
        mv = total_mv
        mv_label = "total_mv (circ_mv 不可用，近似)"
        logger.warning(
            "build_mcap_percentile_mask: circ_mv 缺失，用 total_mv 近似分位"
        )
    else:
        raise ValueError("circ_mv 和 total_mv 至少需提供一个")

    mv = mv.copy()
    mv.index = pd.to_datetime(mv.index)
    if trading_index is not None:
        trading_index = pd.DatetimeIndex(trading_index)
        mv = mv.reindex(trading_index)

    if rebalance_dates is not None:
        rb = pd.DatetimeIndex(rebalance_dates)
        rb = rb.intersection(mv.index)
        if len(rb) == 0:
            logger.warning(
                "build_mcap_percentile_mask: 调仓日与市值 index 无交集，返回全 False"
            )
            return pd.DataFrame(False, index=mv.index, columns=mv.columns)
        eval_idx = rb
    else:
        eval_idx = mv.index

    # 只在 eval 日算分位，再对齐到全日
    eval_mv = mv.reindex(eval_idx)
    # rank(axis=1, pct=True)：NaN 不参与排名，结果仍为 NaN
    pct = eval_mv.rank(axis=1, method="average", ascending=True, pct=True)
    eval_mask = (pct <= float(quantile)).fillna(False)

    if rebalance_dates is not None:
        # 调仓日判定 → ffill 到交易日（调仓前日期保持 False，避免用未来池）
        mask = eval_mask.reindex(mv.index)
        mask = mask.ffill().fillna(False).astype(bool)
    else:
        mask = eval_mask.astype(bool)

    n_dates = len(mask)
    per_day = mask.sum(axis=1)
    med = int(per_day.median()) if n_dates else 0
    logger.info(
        f"分位小市值 mask ({mv_label}, q≤{quantile:.0%}): "
        f"{n_dates} 日 × {mask.shape[1]} 股，"
        f"中位数 {med} 只/日"
        + (
            f"（调仓日判定 {len(eval_idx)} 个后 ffill）"
            if rebalance_dates is not None
            else "（逐日判定）"
        )
    )
    return mask


# 东财 circ_mv / total_mv 面板单位 = 元；1 亿 = 1e8 元。
YI_TO_YUAN = 1e8


def build_mcap_yi_band_mask(
    circ_mv: pd.DataFrame | None,
    min_yi: float = 30.0,
    max_yi: float = 100.0,
    *,
    total_mv: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """按流通市值亿元区间构造逐日 bool mask（**不含** 20 日成交额过滤）。

    产品口径：东财 ``circ_mv`` 单位=元，``[min_yi, max_yi]`` 亿元含边界
    （默认 30–100 亿 → ``[30e8, 100e8]`` 元）。缺市值格子 → False。

    与 :func:`build_cap_band_mask` / :func:`build_small_cap_universe` 的差别：
    那些函数会再 AND 20 日均成交额；本函数只做市值带，可交易（ST/停牌/零成交/
    次新）交给 ``build_ic_tradability_mask``。
    """
    if circ_mv is not None:
        mv = circ_mv
        mv_label = "circ_mv"
    elif total_mv is not None:
        mv = total_mv
        mv_label = "total_mv (circ_mv 不可用，近似)"
        logger.warning(
            "build_mcap_yi_band_mask: circ_mv 缺失，用 total_mv 近似亿元带"
        )
    else:
        raise ValueError("circ_mv 和 total_mv 至少需提供一个")

    lo = None if min_yi is None else float(min_yi) * YI_TO_YUAN
    hi = None if max_yi is None else float(max_yi) * YI_TO_YUAN
    in_band = pd.DataFrame(True, index=mv.index, columns=mv.columns)
    if lo is not None:
        in_band = in_band & (mv >= lo)
    if hi is not None:
        in_band = in_band & (mv <= hi)
    mask = in_band.fillna(False).astype(bool)

    n_dates = len(mask)
    per = mask.sum(axis=1)
    avg = int(per.mean()) if n_dates else 0
    lo_lbl = "无界" if lo is None else f"{lo / YI_TO_YUAN:.0f}亿"
    hi_lbl = "无界" if hi is None else f"{hi / YI_TO_YUAN:.0f}亿"
    logger.info(
        f"市值亿元带 ({mv_label}): {n_dates} 日 × {mask.shape[1]} 股，"
        f"平均 {avg} 只/日，区间=[{lo_lbl}, {hi_lbl}]（含边界，无成交额过滤）"
    )
    return mask


def load_universe_mask_file(path: str | Path) -> pd.DataFrame:
    """从 parquet 加载外部 universe mask（wide bool；True=纳入）。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"universe-mask 文件不存在: {path}")
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"universe-mask 为空: {path}")
    df.index = pd.to_datetime(df.index)
    # 兼容 0/1 与 bool
    return df.astype(bool)
