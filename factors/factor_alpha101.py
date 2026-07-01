"""
factors/factor_alpha101.py — 精选 10 个 WorldQuant Alpha101 价量因子

来源：Kakushadze (2016) "101 Formulaic Alphas"
选因依据：
  - arxiv:2507.07107 在 A 股 2022-2024 数据上验证持续有效的 9 个因子
  - arxiv:2602.14670 (FactorMiner) A 股因子重要性排名 (Alpha053 第13, Alpha061 第2)
  - 原始论文中被学术界引用最多的因子

输入数据要求：close/open/high/low/volume (日频 DataFrame，index=日期，columns=股票)
VWAP = amount / volume (需要 amount 参数)

命名规则：WQ_001 ... WQ_101（WQ = WorldQuant，避免与自研因子命名冲突）
方向约定：函数输出已经过验证或取反，尽量保持 "越高越好"。
IC 分析后若某因子方向与预期相反，在此文件里取反即可。
"""

import numpy as np
import pandas as pd
from loguru import logger


# ──────────────────────────────────────────────
# 内部工具函数
# ──────────────────────────────────────────────

def _cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """截面百分比排名 [0, 1]"""
    return df.rank(axis=1, pct=True)


def _ts_rank(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """时序百分比排名，min_periods=window*0.7"""
    mp = max(2, int(window * 0.7))
    return df.rolling(window, min_periods=mp).rank(pct=True)


def _delta(df: pd.DataFrame, period: int) -> pd.DataFrame:
    return df.diff(period)


def _ts_argmax(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """rolling 窗口内最大值的位置（0-indexed）"""
    return df.rolling(window, min_periods=window).apply(
        lambda x: float(np.argmax(x)), raw=True
    )


def _stddev(df: pd.DataFrame, window: int) -> pd.DataFrame:
    mp = max(2, int(window * 0.5))
    return df.rolling(window, min_periods=mp).std()


def _corr(x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
    """逐股票时序滚动相关系数"""
    mp = max(2, int(window * 0.7))
    return x.rolling(window, min_periods=mp).corr(y)


def _adv(volume: pd.DataFrame, window: int) -> pd.DataFrame:
    mp = max(2, int(window * 0.5))
    return volume.rolling(window, min_periods=mp).mean()


def _scale(df: pd.DataFrame) -> pd.DataFrame:
    """截面缩放：每行绝对值之和 = 1"""
    row_abs = df.abs().sum(axis=1).replace(0, np.nan)
    return df.div(row_abs, axis=0)


def _signed_power(df: pd.DataFrame, exp: float) -> pd.DataFrame:
    return df.abs().pow(exp) * np.sign(df)


# ──────────────────────────────────────────────
# 10 个精选因子
# ──────────────────────────────────────────────

def wq_alpha001(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha#001: rank(Ts_ArgMax(SignedPower((ret<0 ? stddev(ret,20) : close), 2), 5)) - 0.5

    逻辑：当收益为负时用波动率替代价格，找过去5期内上述指标峰值出现的位置。
    A股效果：捕捉波动率条件下的动量，熊市中高波动+反弹 → 买入信号。
    方向：rank越高=峰值越靠近今天=短期动量越强，高分 = 看多。
    """
    ret = close.pct_change()
    std20 = _stddev(ret, 20)
    x = std20.where(ret < 0, close)
    x2 = _signed_power(x, 2.0)
    arg = _ts_argmax(x2, 5)
    return _cs_rank(arg) - 0.5


def wq_alpha002(close: pd.DataFrame, open_: pd.DataFrame,
                volume: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha#002: -1 * corr(rank(delta(log(vol), 2)), rank((close-open)/open), 6)

    逻辑：成交量变化排名 与 日内涨幅排名 的相关性（6日）取负值。
    A股效果：追涨成交量（量价齐升）→ 相关性高 → alpha负（卖出）；量价背离 → 买入。
    方向：已取负，高分 = 量价背离 = 聪明资金买入迹象。
    """
    log_vol_chg = _cs_rank(_delta(np.log(volume.replace(0, np.nan)), 2))
    intraday_ret = _cs_rank((close - open_) / open_.replace(0, np.nan))
    return -1.0 * _corr(log_vol_chg, intraday_ret, 6)


def wq_alpha006(open_: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha#006: -1 * corr(open, volume, 10)

    逻辑：开盘价与成交量的相关性（10日）取负。
    A股效果：高开 + 大量 → 散户追涨 → 相关性正 → alpha负（卖出）；
             低开 + 缩量 → 机构吸筹 → 相关性负 → alpha正（买入）。
    方向：已取负，高分 = 开盘-成交量负相关 = 买入信号。
    """
    return -1.0 * _corr(open_, volume, 10)


def wq_alpha007(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha#007: (adv20 < volume) ? (-ts_rank(|delta(close,7)|, 60) * sign(delta(close,7))) : -1

    逻辑：当日成交量超过20日均量时，7日内大幅下跌 → 买入（恐慌出逃后反弹）。
    A股效果：A股散户情绪驱动的超卖反弹，放量暴跌后往往有技术性反弹。
    方向：高分 = 放量 + 大幅下跌 = 超卖反弹信号。
    """
    adv20 = _adv(volume, 20)
    d7 = _delta(close, 7)
    d7_sign = np.sign(d7)
    ts_r = _ts_rank(d7.abs(), 60)
    signal = -1.0 * ts_r * d7_sign
    # 当日成交量 <= 20日均量时，信号为 -1（中性偏空）
    result = signal.where(volume > adv20, -1.0)
    return result


def wq_alpha012(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha#012: sign(delta(vol, 1)) * (-1 * delta(close, 1))

    逻辑：成交量上升 + 价格下跌 → 主力吸筹 → 买入；成交量下降 + 价格上涨 → 出货 → 卖出。
    A股效果：A股吸筹规律相对明显，量价背离信号在短周期有效。
    方向：高分 = 放量下跌 = 机构吸筹 = 买入。
    """
    return np.sign(_delta(volume, 1)) * (-1.0 * _delta(close, 1))


def wq_alpha028(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame,
                volume: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha#028: scale(corr(adv20, low, 5) + (high+low)/2 - close)

    逻辑：均量与最低价的相关性 + 中间价 - 收盘价。
    均量-低价相关性高 = 放量时价格更低 = 在底部吸筹；收盘低于中间价 = 尾盘弱势。
    两者组合：scale后高分 = 尾盘下跌但有放量吸筹迹象 = 次日反弹。
    方向：高分 = 买入信号。
    """
    adv20 = _adv(volume, 20)
    c = _corr(adv20, low, 5)
    mid = (high + low) / 2.0
    return _scale(c + mid - close)


def wq_alpha034(close: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha#034: rank((1 - rank(std(ret,2)/std(ret,5))) + (1 - rank(delta(close,1))))

    逻辑：两个反转信号之和：
      1. 短期波动率下降（2日/5日波动率比小 → 近期趋于平静）
      2. 近一日价格下跌（一日反转）
    A股效果：短周期均值回归在A股显著（散户追涨杀跌）。
    方向：高分 = 波动缩窄 + 近日下跌 = 反转买入。
    """
    ret = close.pct_change()
    std2 = _stddev(ret, 2)
    std5 = _stddev(ret, 5).replace(0, np.nan)
    vol_ratio = std2 / std5
    d1 = _delta(close, 1)
    return _cs_rank((1.0 - _cs_rank(vol_ratio)) + (1.0 - _cs_rank(d1)))


def wq_alpha053(close: pd.DataFrame, high: pd.DataFrame,
                low: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha#053: -1 * delta((close-low-(high-close)) / (close-low), 9)
               = -1 * delta((2*close - high - low) / (close - low), 9)

    逻辑：价格在日内区间的位置变化。收盘越靠近当日最高价 → 位置越高 → 看多。
    取9日变化的负值 → 位置改善（连续走强）→ 高分。
    A股效果：FactorMiner 论文 A 股因子重要性第 13 位。
    方向：高分 = 收盘位置持续上移 = 上升趋势信号。
    """
    denom = (close - low).replace(0.0, np.nan)
    loc = (close - low - (high - close)) / denom
    return -1.0 * _delta(loc, 9)


def wq_alpha061(close: pd.DataFrame, volume: pd.DataFrame,
                amount: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha#061: rank(vwap - ts_min(vwap, 16)) < rank(corr(vwap, adv180, 17))
               → 返回 0/1 的布尔信号，转为连续版：差值

    原始公式返回布尔值，改用连续版（A股实证更稳定）：
    rank(vwap - ts_min(vwap, 16)) - rank(corr(vwap, adv180, 17))
    高分 = VWAP 相对近期低点强势 且 机构资金（大量）跟随买入

    FactorMiner 论文 A 股因子重要性排名第 2 位。
    方向：高分 = VWAP 动量强 + 机构资金支撑 = 买入。
    """
    vwap = amount / volume.replace(0, np.nan)
    adv180 = _adv(volume, 180)
    vwap_pos = _cs_rank(vwap - vwap.rolling(16, min_periods=10).min())
    vwap_vol_corr = _cs_rank(_corr(vwap, adv180, 17))
    return vwap_pos - vwap_vol_corr


def wq_alpha101(close: pd.DataFrame, open_: pd.DataFrame,
                high: pd.DataFrame, low: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha#101: (close - open) / (high - low + 0.001)

    逻辑：日内涨幅占全天振幅的比例。高分 = 收盘靠近当日高点 = 多头主导。
    A股效果：捕捉尾盘强势（机构护盘/主力拉升），短期延续性较好。
    方向：高分 = 日内多头强势 = 动量买入信号。
    """
    return (close - open_) / ((high - low) + 0.001)


# ──────────────────────────────────────────────
# 统一入口
# ──────────────────────────────────────────────

def get_alpha101_factors(
    prices: pd.DataFrame,
    open_: pd.DataFrame = None,
    high: pd.DataFrame = None,
    low: pd.DataFrame = None,
    volume: pd.DataFrame = None,
    amount: pd.DataFrame = None,
) -> dict:
    """
    返回精选 10 个 Alpha101 因子（均已按 "越高越好" 约定处理）。

    必须参数：prices (close)
    可选参数：open_, high, low, volume, amount（缺少时对应因子跳过）
    """
    factors = {}
    has_ohlcv = all(x is not None for x in [open_, high, low, volume])

    def _try(name, fn, *args):
        try:
            result = fn(*args)
            if result is not None and not result.isna().all(axis=None):
                factors[name] = result
        except Exception as e:
            logger.warning(f"Alpha101 {name} 计算失败: {e}")

    # 只需 close + volume
    if volume is not None:
        _try("WQ_001", wq_alpha001, prices, volume)
        _try("WQ_007", wq_alpha007, prices, volume)
        _try("WQ_012", wq_alpha012, prices, volume)

    # 需要 open
    if open_ is not None and volume is not None:
        _try("WQ_002", wq_alpha002, prices, open_, volume)
        _try("WQ_006", wq_alpha006, open_, volume)

    # 需要 high + low
    if has_ohlcv:
        _try("WQ_028", wq_alpha028, prices, high, low, volume)
        _try("WQ_053", wq_alpha053, prices, high, low)

    # 只需 close
    _try("WQ_034", wq_alpha034, prices)

    # 需要 OHLC (无需 volume)
    if open_ is not None and high is not None and low is not None:
        _try("WQ_101", wq_alpha101, prices, open_, high, low)

    # 需要 amount（用于计算 VWAP）
    if volume is not None and amount is not None:
        _try("WQ_061", wq_alpha061, prices, volume, amount)

    logger.info(f"Alpha101 精选因子: 计算完成 {len(factors)} 个 "
                f"({list(factors.keys())})")
    return factors
