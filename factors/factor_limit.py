"""
factors/factor_limit.py  —  涨跌停信号因子

只负责从已有 masks 构建信号因子，不做 mask 计算。
masks 由 data.clean.clean_ohlcv() 统一生成，通过 get_factor_registry() 传入。

信号因子：
  涨停强度_20d     — 近20日涨停次数（强势度），排除一字板
  跌停弱势_20d     — 近20日跌停次数取负
  连板数           — 当前连续涨停天数
  涨跌停净强度_20d — 涨停次数 - 跌停次数（多空对比）
  涨跌停状态       — 信号日涨跌停序数（1=跌停/2=正常/3=涨停，截面 winsor+zscore）
  开板反转_5d      — 开板后N日收益取负（供应压力释放信号）
"""
import pandas as pd
import numpy as np
from loguru import logger

from factors.factor import _normalize


def factor_limit_state(masks: dict) -> pd.DataFrame:
    """
    涨跌停状态 dummy（越高越好：涨停=3 > 非涨跌停=2 > 跌停=1）。

    基于 masks["limit_up"] / masks["limit_down"] 逐日赋值；
    同日 limit_up 与 limit_down 均为 True 时优先 limit_up（→3）。
    """
    limit_up = masks["limit_up"].fillna(False).astype(bool)
    limit_down = masks["limit_down"].fillna(False).astype(bool)
    state = pd.DataFrame(
        2.0,
        index=limit_up.index,
        columns=limit_up.columns,
        dtype=np.float64,
    )
    state = state.mask(limit_down, 1.0)
    state = state.mask(limit_up, 3.0)
    both = limit_up & limit_down
    n_both = int(both.sum().sum())
    if n_both:
        logger.warning(
            f"涨跌停状态: {n_both} 格同日涨跌停，按涨停(3)优先编码"
        )
    return _normalize(state)


def factor_limit_up_count(
    masks: dict,
    window: int = 20,
    exclude_one_word: bool = True,
) -> pd.DataFrame:
    """
    近N日涨停次数（强势度）。
    exclude_one_word=True 排除一字板（一字板无法买入，信号失真）。
    """
    limit_up = masks["limit_up"].astype(float).copy()
    if exclude_one_word and "limit_up_open" in masks:
        limit_up[masks["limit_up_open"]] = 0.0
    count = limit_up.rolling(window, min_periods=1).sum()
    return _normalize(count)


def factor_limit_down_count(masks: dict, window: int = 20) -> pd.DataFrame:
    """近N日跌停次数取负（跌停次数少得高分）。"""
    count = masks["limit_down"].astype(float).rolling(window, min_periods=1).sum()
    return _normalize(-count)


def factor_consecutive_limit_up(masks: dict) -> pd.DataFrame:
    """
    当前连续涨停天数（连板数）。
    连板是A股短期动量强信号（3板以上有打板跟涨效应）。
    """
    arr = masks["limit_up"].values.astype(float)
    out = np.zeros_like(arr)
    for t in range(1, len(arr)):
        out[t] = np.where(arr[t] > 0, out[t - 1] + 1, 0)
    result = pd.DataFrame(out, index=masks["limit_up"].index,
                          columns=masks["limit_up"].columns)
    return _normalize(result)


def factor_limit_strength(masks: dict, window: int = 20) -> pd.DataFrame:
    """涨跌停净强度 = 涨停次数 - 跌停次数，反映近期多空力量对比。"""
    up   = masks["limit_up"].astype(float).rolling(window, min_periods=1).sum()
    down = masks["limit_down"].astype(float).rolling(window, min_periods=1).sum()
    return _normalize(up - down)


def factor_post_limit_reversion(
    close: pd.DataFrame,
    masks: dict,
    hold_days: int = 5,
    clean_ret: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    开板反转因子：开板日已实现的历史收益取负（供应压力释放信号）。

    语义：在 t 时刻若近期发生过开板，用开板信号滞后 hold_days 后，
    结合**截至 t 已实现**的收益（close[t]/close[t-hold_days]-1）刻画
    开板后的反转强度。绝不用未来收益，避免 look-ahead bias。

    连续涨停后开板，套牢盘解套抛售，往往大幅回调；已发生的回调幅度
    可作为后续继续承压的反向信号（收益越低 → 取负后分越高 → 规避）。

    clean_ret : 有则用屏蔽涨跌停日的日收益滚动复合（与动量/反转因子同口径）。
    """
    # 截至当前 t 的已实现收益（仅用 t 及之前的信息）
    if clean_ret is not None:
        realized_ret = (1 + clean_ret).rolling(
            hold_days, min_periods=max(1, hold_days // 2),
        ).apply(lambda x: np.nanprod(x) - 1, raw=True)
    else:
        realized_ret = close.pct_change(hold_days)
    # 开板信号滞后 hold_days，保证 t 时刻只用 t-hold_days 之前已知的开板事件
    broke_lagged = masks["broke_limit"].shift(hold_days)
    # 仅在滞后开板窗口激活，向前不填充（避免再次引入未来信息）
    signal = realized_ret.where(broke_lagged, other=np.nan)
    return _normalize(-signal)


def get_limit_factors(
    close: pd.DataFrame,
    masks: dict,
    clean_ret: pd.DataFrame | None = None,
) -> dict:
    """
    计算所有涨跌停信号因子，返回 {因子名: DataFrame}。
    masks 由 data.clean.clean_ohlcv() 提供，此处直接使用。
    """
    factors = {}
    try:
        factors["涨停强度_20d"]    = factor_limit_up_count(masks, window=20)
        factors["跌停弱势_20d"]    = factor_limit_down_count(masks, window=20)
        factors["连板数"]           = factor_consecutive_limit_up(masks)
        factors["涨跌停净强度_20d"] = factor_limit_strength(masks, window=20)
        factors["涨跌停状态"]       = factor_limit_state(masks)
        factors["开板反转_5d"]      = factor_post_limit_reversion(
            close, masks, hold_days=5, clean_ret=clean_ret,
        )
        logger.info(f"涨跌停因子: {list(factors.keys())}")
    except Exception as e:
        logger.error(f"涨跌停因子计算失败: {e}")
    return factors
