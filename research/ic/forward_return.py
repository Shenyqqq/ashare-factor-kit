"""Forward return construction (aligned with ML / backtest)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import resolve_apply_exec_mask


def build_forward_return(
    prices: pd.DataFrame,
    open_: pd.DataFrame | None,
    period: int,
    masks: dict | None = None,
    apply_exec_mask: bool | None = None,
    apply_label_exec_mask: bool | None = None,
) -> pd.DataFrame:
    """
    有开盘价 → close[t+N]/open[t+1]-1（信号日收盘后次日开盘买入）
    无开盘价 → close[t+N]/close[t]-1

    masks : dict, optional
        涨跌停 mask 矩阵（同 data.clean.make_limit_mask 输出），含
        limit_up_open / limit_down_open / any_limit 等 bool DataFrame。
        传入后屏蔽无法成交的样本，与因子侧 clean_ret 屏蔽涨跌停口径一致：
          - 买入日 t+1 一字涨停（limit_up_open[t+1]=True）→ forward_return 置 NaN
            （开盘即封死涨停板，次日开盘实际无法买入）
          - 卖出日 t+N 涨停/跌停（any_limit[t+N]=True）→ forward_return 置 NaN（保守）
            （涨跌停日无法保证按收盘价成交）
        不传 masks 时保持旧行为（向后兼容）。
    apply_exec_mask : bool | None
        是否在标签上屏蔽买日一字涨停 / 卖日涨跌停（默认 settings.FWD_RETURN_EXEC_MASK=False）。
        apply_label_exec_mask 为向后兼容别名。
    """
    if apply_exec_mask is None:
        apply_exec_mask = apply_label_exec_mask
    apply_exec_mask = resolve_apply_exec_mask(apply_exec_mask)
    # 强制 float64，避免 nullable/extension dtype 引入 pd.NA (NAType) 导致
    # 下游 .astype(np.float32) 报 TypeError: float() argument must be a string
    # or a real number, not 'NAType'.
    prices = prices.astype(np.float64)
    if open_ is not None:
        open_ = open_.astype(np.float64)

    if open_ is not None:
        buy = open_.shift(-1)
        sell = prices.shift(-period)
        # 用 np.nan 而非 pd.NA，保持 float64 dtype（pd.NA 会把数组转 object）
        fwd = sell / buy.replace(0.0, np.nan) - 1
    else:
        fwd = prices.pct_change(period).shift(-period)

    if masks and apply_exec_mask:
        # 买入日 t+1：limit_up_open.shift(-1) 把 t+1 的一字涨停状态对齐到信号日 t
        buy_mask = masks.get("limit_up_open")
        if buy_mask is not None and not buy_mask.empty:
            buy_block = (
                buy_mask.shift(-1)
                .reindex(index=fwd.index, columns=fwd.columns)
                .fillna(False)
                .astype(bool)
            )
            fwd = fwd.mask(buy_block)
        # 卖出日 t+N：any_limit.shift(-period) 把 t+N 的涨跌停状态对齐到信号日 t
        sell_mask = masks.get("any_limit")
        if sell_mask is not None and not sell_mask.empty:
            sell_block = (
                sell_mask.shift(-period)
                .reindex(index=fwd.index, columns=fwd.columns)
                .fillna(False)
                .astype(bool)
            )
            fwd = fwd.mask(sell_block)

    return fwd


def winsorize_forward_return(
    fwd: pd.DataFrame,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    """
    按行（每个 date）对 forward_return 做分位数截尾（winsorize），削弱妖股连板等
    极端持有收益对 IC / MSE 的污染。保留样本，只压幅度；全 NaN 行跳过。

    IC 与 ML 共用本函数。调用方应在 tradable / eligible mask 置 NaN **之后**
    再调用，使分位数只在可交易样本上计算。
    """
    if fwd.empty:
        return fwd

    def _win(row: pd.Series) -> pd.Series:
        if not row.notna().any():
            return row
        lo = row.quantile(lower)
        hi = row.quantile(upper)
        return row.clip(lo, hi)

    out = fwd.apply(_win, axis=1)
    # apply 常升为 float64；尽量保留原面板 dtype（ML 路径常用 float32）
    if len(fwd.dtypes):
        out = out.astype(fwd.dtypes.iloc[0])
    return out


def forward_return_label(open_: pd.DataFrame | None) -> str:
    return "open[t+1]→close[t+N]" if open_ is not None else "close→close"
