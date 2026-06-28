"""
factors/factor_limit.py  —  涨跌停信号因子

只负责从已有 masks 构建信号因子，不做 mask 计算。
masks 由 data.clean.clean_ohlcv() 统一生成，通过 get_factor_registry() 传入。

信号因子：
  涨停强度_20d     — 近20日涨停次数（强势度），排除一字板
  跌停弱势_20d     — 近20日跌停次数取负
  连板数           — 当前连续涨停天数
  涨跌停净强度_20d — 涨停次数 - 跌停次数（多空对比）
  开板反转_5d      — 开板后N日收益取负（供应压力释放信号）
"""
import pandas as pd
import numpy as np
from loguru import logger

from factors.factor import _normalize


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
) -> pd.DataFrame:
    """
    开板反转因子：开板日后N日预期收益取负。
    连续涨停后开板，套牢盘解套抛售，往往大幅回调。
    用于规避"接飞刀"风险。
    """
    fwd_ret = close.pct_change(hold_days).shift(-hold_days)
    # 只在开板日激活，前向填充整个持有期
    signal = fwd_ret.where(masks["broke_limit"], other=np.nan)
    signal = signal.ffill(limit=hold_days)
    return _normalize(-signal)


def get_limit_factors(close: pd.DataFrame, masks: dict) -> dict:
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
        factors["开板反转_5d"]      = factor_post_limit_reversion(close, masks, hold_days=5)
        logger.info(f"涨跌停因子: {list(factors.keys())}")
    except Exception as e:
        logger.error(f"涨跌停因子计算失败: {e}")
    return factors
