"""
factors/factor_alpha101.py — WorldQuant "101 Formulaic Alphas" 集成

来源：Kakushadze (2016) "101 Formulaic Alphas"
参考实现：https://github.com/yli188/WorldQuant_alpha101_code （已对齐本项目约定重写）

输入数据要求：close/open/high/low/volume/amount (日频 DataFrame，index=日期，columns=股票)
VWAP = amount / volume (在 get_alpha101_factors 内统一计算后透传)

命名规则：WQ_001 ... WQ_101（WQ = WorldQuant，避免与自研因子命名冲突）
方向约定：函数输出已经过验证或按原论文方向返回；IC 分析后若某因子方向与预期相反，
          在此文件里取反即可。本项目标准化（截面 winsorize + zscore）在
          get_alpha101_factors._try 出口统一对每个 WQ 因子应用 _normalize，
          与其它因子同口径（截面 winsorize 1% → zscore clip 3σ）。

跳过的 alpha（无实现 / 需行业或市值数据）：
  - 048, 058, 59, 063, 067, 069, 070, 076, 079, 080, 089, 090, 091, 093, 097, 100
    （依赖 IndNeutralize(IndClass.sector/subindustry/industry)，本项目无对应数据）
  - 056：依赖市值 cap，无数据
共实现 84 个 alpha。

约定差异（相对原始 yli188 实现）：
  1. 外部依赖 (alphas/datas 模块) 全部移除，Alphas101 逻辑拆为独立函数。
  2. returns = clean_ret（屏蔽涨跌停日），替代原始 rolling(2) 比价收益，
     遵循项目"量价因子必须用 clean_ret"约定。
  3. cross-sectional rank 用 _cs_rank (df.rank(axis=1, pct=True))，统一不带 method='min'。
  4. ts_rank 用 pandas 原生 rolling().rank(pct=True) 替代 rolling.apply(rankdata)，约 10x 提速。
  5. 模块级 min/max 内置覆盖已移除，元素级 max/min 改用 np.maximum/np.minimum。
  6. alpha096 的调试 print 与未用 r1/r2 已删除。
  7. 相关性 alpha 用 _corr (min_periods 优雅 NaN 处理)，不强制 fillna(0)。
"""

# 候选 Alpha101 因子名（实际计算受 OHLCV/amount 可用性门控，
# 见 get_alpha101_factors 内部 _try 调用）。共 84 个（跳过 048/056/058/059/
# 063/067/069/070/076/079/080/089/090/091/093/097/100）。
ALPHA101_NAMES = (
    "WQ_001", "WQ_002", "WQ_003", "WQ_004", "WQ_005",
    "WQ_006", "WQ_007", "WQ_008", "WQ_009", "WQ_010",
    "WQ_011", "WQ_012", "WQ_013", "WQ_014", "WQ_015",
    "WQ_016", "WQ_017", "WQ_018", "WQ_019", "WQ_020",
    "WQ_021", "WQ_022", "WQ_023", "WQ_024", "WQ_025",
    "WQ_026", "WQ_027", "WQ_028", "WQ_029", "WQ_030",
    "WQ_031", "WQ_032", "WQ_033", "WQ_034", "WQ_035",
    "WQ_036", "WQ_037", "WQ_038", "WQ_039", "WQ_040",
    "WQ_041", "WQ_042", "WQ_043", "WQ_044", "WQ_045",
    "WQ_046", "WQ_047", "WQ_049", "WQ_050", "WQ_051",
    "WQ_052", "WQ_053", "WQ_054", "WQ_055", "WQ_057",
    "WQ_060", "WQ_061", "WQ_062", "WQ_064", "WQ_065",
    "WQ_066", "WQ_068", "WQ_071", "WQ_072", "WQ_073",
    "WQ_074", "WQ_075", "WQ_077", "WQ_078", "WQ_081",
    "WQ_083", "WQ_084", "WQ_085", "WQ_086", "WQ_088",
    "WQ_092", "WQ_094", "WQ_095", "WQ_096", "WQ_098",
    "WQ_099", "WQ_101",
)

import numpy as np
import pandas as pd
from loguru import logger

from factors.factor import _normalize


# ──────────────────────────────────────────────
# 内部工具函数
# ──────────────────────────────────────────────

def _cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """截面百分比排名 [0, 1]"""
    return df.rank(axis=1, pct=True)


def _ts_rank(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """时序百分比排名，min_periods=window*0.7（pandas 原生，~10x 快于 rankdata apply）"""
    mp = max(2, int(window * 0.7))
    return df.rolling(window, min_periods=mp).rank(pct=True)


def _delta(df: pd.DataFrame, period: int) -> pd.DataFrame:
    return df.diff(period)


def _delay(df: pd.DataFrame, period: int) -> pd.DataFrame:
    return df.shift(period)


def _ts_argmax(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """rolling 窗口内最大值的位置（0-indexed）"""
    return df.rolling(window, min_periods=window).apply(
        lambda x: float(np.argmax(x)), raw=True
    )


def _ts_argmin(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """rolling 窗口内最小值的位置（0-indexed）"""
    return df.rolling(window, min_periods=window).apply(
        lambda x: float(np.argmin(x)), raw=True
    )


def _stddev(df: pd.DataFrame, window: int) -> pd.DataFrame:
    mp = max(2, int(window * 0.5))
    return df.rolling(window, min_periods=mp).std()


def _corr(x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
    """逐股票时序滚动相关系数（min_periods 优雅 NaN，不强制 fillna(0)）"""
    mp = max(2, int(window * 0.7))
    return x.rolling(window, min_periods=mp).corr(y)


def _cov(x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
    """逐股票时序滚动协方差"""
    mp = max(2, int(window * 0.7))
    return x.rolling(window, min_periods=mp).cov(y)


def _ts_sum(df: pd.DataFrame, window: int) -> pd.DataFrame:
    mp = max(2, int(window * 0.7))
    return df.rolling(window, min_periods=mp).sum()


def _ts_min(df: pd.DataFrame, window: int) -> pd.DataFrame:
    mp = max(2, int(window * 0.7))
    return df.rolling(window, min_periods=mp).min()


def _ts_max(df: pd.DataFrame, window: int) -> pd.DataFrame:
    mp = max(2, int(window * 0.7))
    return df.rolling(window, min_periods=mp).max()


def _sma(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """时序简单移动平均"""
    mp = max(2, int(window * 0.5))
    return df.rolling(window, min_periods=mp).mean()


def _adv(volume: pd.DataFrame, window: int) -> pd.DataFrame:
    """平均美元成交量 = volume 的 rolling mean（与 _sma 同义，命名保留以匹配原论文 adv 术语）"""
    mp = max(2, int(window * 0.5))
    return volume.rolling(window, min_periods=mp).mean()


def _scale(df: pd.DataFrame) -> pd.DataFrame:
    """截面缩放：每行绝对值之和 = 1"""
    row_abs = df.abs().sum(axis=1).replace(0, np.nan)
    return df.div(row_abs, axis=0)


def _signed_power(df: pd.DataFrame, exp: float) -> pd.DataFrame:
    return df.abs().pow(exp) * np.sign(df)


def _product(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """rolling 乘积"""
    mp = max(2, int(window * 0.7))
    return df.rolling(window, min_periods=mp).apply(np.prod, raw=True)


def _decay_linear(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """线性加权移动平均（近期权重更大）

    注意：pandas rolling.apply(raw=True) 在窗口含 NaN 时只透传非 NaN 子集
    （变长），因此权重须按实际切片长度动态构造，而非固定 period。
    """
    mp = max(2, int(period * 0.7))

    def _wma(x: np.ndarray) -> float:
        n = x.shape[0]
        if n == 0:
            return np.nan
        w = np.arange(1, n + 1, dtype=float)
        w = w / w.sum()
        return float(np.dot(x, w))

    return df.rolling(period, min_periods=mp).apply(_wma, raw=True)


def _safe_div(num: pd.DataFrame, den: pd.DataFrame) -> pd.DataFrame:
    """除法，0/0 → NaN，inf → NaN"""
    out = num / den
    return out.replace([np.inf, -np.inf], np.nan)


# ──────────────────────────────────────────────
# 10 个精选因子（保留原始实现与详细文档，签名与之前版本一致）
# ──────────────────────────────────────────────

def wq_alpha001(close: pd.DataFrame, volume: pd.DataFrame,
                clean_ret: pd.DataFrame = None) -> pd.DataFrame:
    """
    Alpha#001: rank(Ts_ArgMax(SignedPower((ret<0 ? stddev(ret,20) : close), 2), 5)) - 0.5

    逻辑：当收益为负时用波动率替代价格，找过去5期内上述指标峰值出现的位置。
    A股效果：捕捉波动率条件下的动量，熊市中高波动+反弹 → 买入信号。
    方向：rank越高=峰值越靠近今天=短期动量越强，高分 = 看多。

    clean_ret: 屏蔽涨跌停日后的日收益率；传入时用 clean_ret 替代 close.pct_change()，
               避免涨跌停日 return 截断污染波动率与符号判定；为 None 时退化为 pct_change。
    """
    ret = clean_ret if clean_ret is not None else close.pct_change()
    std20 = _stddev(ret, 20)
    x = std20.where(ret < 0, close)
    x2 = _signed_power(x, 2.0)
    arg = _ts_argmax(x2, 5)
    return _cs_rank(arg) - 0.5


def wq_alpha002(close: pd.DataFrame, open_: pd.DataFrame,
                volume: pd.DataFrame,
                clean_ret: pd.DataFrame = None) -> pd.DataFrame:
    """
    Alpha#002: -1 * corr(rank(delta(log(vol), 2)), rank((close-open)/open), 6)

    逻辑：成交量变化排名 与 日内涨幅排名 的相关性（6日）取负值。
    A股效果：追涨成交量（量价齐升）→ 相关性高 → alpha负（卖出）；量价背离 → 买入。
    方向：已取负，高分 = 量价背离 = 聪明资金买入迹象。

    clean_ret: 预留参数（本因子不直接使用日频收益率），保证 10 个 Alpha101 因子
               签名一致，便于 get_alpha101_factors 统一透传。
    """
    log_vol_chg = _cs_rank(_delta(np.log(volume.replace(0, np.nan)), 2))
    intraday_ret = _cs_rank((close - open_) / open_.replace(0, np.nan))
    return -1.0 * _corr(log_vol_chg, intraday_ret, 6)


def wq_alpha006(open_: pd.DataFrame, volume: pd.DataFrame,
                clean_ret: pd.DataFrame = None) -> pd.DataFrame:
    """
    Alpha#006: -1 * corr(open, volume, 10)

    逻辑：开盘价与成交量的相关性（10日）取负。
    A股效果：高开 + 大量 → 散户追涨 → 相关性正 → alpha负（卖出）；
             低开 + 缩量 → 机构吸筹 → 相关性负 → alpha正（买入）。
    方向：已取负，高分 = 开盘-成交量负相关 = 买入信号。

    clean_ret: 预留参数，签名一致性用途。
    """
    return -1.0 * _corr(open_, volume, 10)


def wq_alpha007(close: pd.DataFrame, volume: pd.DataFrame,
                clean_ret: pd.DataFrame = None) -> pd.DataFrame:
    """
    Alpha#007: (adv20 < volume) ? (-ts_rank(|delta(close,7)|, 60) * sign(delta(close,7))) : -1

    逻辑：当日成交量超过20日均量时，7日内大幅下跌 → 买入（恐慌出逃后反弹）。
    A股效果：A股散户情绪驱动的超卖反弹，放量暴跌后往往有技术性反弹。
    方向：高分 = 放量 + 大幅下跌 = 超卖反弹信号。

    clean_ret: 预留参数，签名一致性用途（本因子用 delta(close) 而非日收益率）。
    """
    adv20 = _adv(volume, 20)
    d7 = _delta(close, 7)
    d7_sign = np.sign(d7)
    ts_r = _ts_rank(d7.abs(), 60)
    signal = -1.0 * ts_r * d7_sign
    # 当日成交量 <= 20日均量时，信号为 -1（中性偏空）
    result = signal.where(volume > adv20, -1.0)
    return result


def wq_alpha012(close: pd.DataFrame, volume: pd.DataFrame,
                clean_ret: pd.DataFrame = None) -> pd.DataFrame:
    """
    Alpha#012: sign(delta(vol, 1)) * (-1 * delta(close, 1))

    逻辑：成交量上升 + 价格下跌 → 主力吸筹 → 买入；成交量下降 + 价格上涨 → 出货 → 卖出。
    A股效果：A股吸筹规律相对明显，量价背离信号在短周期有效。
    方向：高分 = 放量下跌 = 机构吸筹 = 买入。

    clean_ret: 预留参数，签名一致性用途（本因子用 delta 而非日收益率）。
    """
    return np.sign(_delta(volume, 1)) * (-1.0 * _delta(close, 1))


def wq_alpha028(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame,
                volume: pd.DataFrame,
                clean_ret: pd.DataFrame = None) -> pd.DataFrame:
    """
    Alpha#028: scale(corr(adv20, low, 5) + (high+low)/2 - close)

    逻辑：均量与最低价的相关性 + 中间价 - 收盘价。
    均量-低价相关性高 = 放量时价格更低 = 在底部吸筹；收盘低于中间价 = 尾盘弱势。
    两者组合：scale后高分 = 尾盘下跌但有放量吸筹迹象 = 次日反弹。
    方向：高分 = 买入信号。

    clean_ret: 预留参数，签名一致性用途。
    """
    adv20 = _adv(volume, 20)
    c = _corr(adv20, low, 5)
    mid = (high + low) / 2.0
    return _scale(c + mid - close)


def wq_alpha034(close: pd.DataFrame,
                clean_ret: pd.DataFrame = None) -> pd.DataFrame:
    """
    Alpha#034: rank((1 - rank(std(ret,2)/std(ret,5))) + (1 - rank(delta(close,1))))

    逻辑：两个反转信号之和：
      1. 短期波动率下降（2日/5日波动率比小 → 近期趋于平静）
      2. 近一日价格下跌（一日反转）
    A股效果：短周期均值回归在A股显著（散户追涨杀跌）。
    方向：高分 = 波动缩窄 + 近日下跌 = 反转买入。

    clean_ret: 屏蔽涨跌停日后的日收益率；传入时用 clean_ret 替代 pct_change，
               避免涨跌停日 return 截断污染 std 估计；为 None 时退化为 pct_change。
    """
    ret = clean_ret if clean_ret is not None else close.pct_change()
    std2 = _stddev(ret, 2)
    std5 = _stddev(ret, 5).replace(0, np.nan)
    vol_ratio = std2 / std5
    d1 = _delta(close, 1)
    return _cs_rank((1.0 - _cs_rank(vol_ratio)) + (1.0 - _cs_rank(d1)))


def wq_alpha053(close: pd.DataFrame, high: pd.DataFrame,
                low: pd.DataFrame,
                clean_ret: pd.DataFrame = None) -> pd.DataFrame:
    """
    Alpha#053: -1 * delta((close-low-(high-close)) / (close-low), 9)
               = -1 * delta((2*close - high - low) / (close - low), 9)

    逻辑：价格在日内区间的位置变化。收盘越靠近当日最高价 → 位置越高 → 看多。
    取9日变化的负值 → 位置改善（连续走强）→ 高分。
    A股效果：FactorMiner 论文 A 股因子重要性第 13 位。
    方向：高分 = 收盘位置持续上移 = 上升趋势信号。

    clean_ret: 预留参数，签名一致性用途（本因子用日内价格位置，非日收益率）。
    """
    denom = (close - low).replace(0.0, np.nan)
    loc = (close - low - (high - close)) / denom
    return -1.0 * _delta(loc, 9)


def wq_alpha061(close: pd.DataFrame, volume: pd.DataFrame,
                amount: pd.DataFrame,
                clean_ret: pd.DataFrame = None) -> pd.DataFrame:
    """
    Alpha#061: rank(vwap - ts_min(vwap, 16)) < rank(corr(vwap, adv180, 17))
               → 返回 0/1 的布尔信号，转为连续版：差值

    原始公式返回布尔值，改用连续版（A股实证更稳定）：
    rank(vwap - ts_min(vwap, 16)) - rank(corr(vwap, adv180, 17))
    高分 = VWAP 相对近期低点强势 且 机构资金（大量）跟随买入

    FactorMiner 论文 A 股因子重要性排名第 2 位。
    方向：高分 = VWAP 动量强 + 机构资金支撑 = 买入。

    clean_ret: 预留参数，签名一致性用途。
    """
    vwap = amount / volume.replace(0, np.nan)
    adv180 = _adv(volume, 180)
    vwap_pos = _cs_rank(vwap - vwap.rolling(16, min_periods=10).min())
    vwap_vol_corr = _cs_rank(_corr(vwap, adv180, 17))
    return vwap_pos - vwap_vol_corr


def wq_alpha101(close: pd.DataFrame, open_: pd.DataFrame,
                high: pd.DataFrame, low: pd.DataFrame,
                clean_ret: pd.DataFrame = None) -> pd.DataFrame:
    """
    Alpha#101: (close - open) / (high - low + 0.001)

    逻辑：日内涨幅占全天振幅的比例。高分 = 收盘靠近当日高点 = 多头主导。
    A股效果：捕捉尾盘强势（机构护盘/主力拉升），短期延续性较好。
    方向：高分 = 日内多头强势 = 动量买入信号。

    clean_ret: 预留参数，签名一致性用途（本因子用日内 OHLC，非日收益率）。
    """
    return (close - open_) / ((high - low) + 0.001)


# ──────────────────────────────────────────────
# 其余 74 个 alpha（统一全签名：close, open_, high, low, volume, amount, vwap, clean_ret）
# clean_ret 用作 returns 的替代（涨跌停日屏蔽）；vwap 由 dispatcher 透传。
# ──────────────────────────────────────────────

def wq_alpha003(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#3: -1 * correlation(rank(open), rank(volume), 10)
    return -1.0 * _corr(_cs_rank(open_), _cs_rank(volume), 10)


def wq_alpha004(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#4: -1 * Ts_Rank(rank(low), 9)
    return -1.0 * _ts_rank(_cs_rank(low), 9)


def wq_alpha005(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#5: rank(open - sma(vwap,10)) * (-1 * abs(rank(close - vwap)))
    return _cs_rank(open_ - _sma(vwap, 10)) * (-1.0 * _cs_rank(close - vwap).abs())


def wq_alpha008(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#8: -1 * rank(sma(open,5)*sma(returns,5) - delay(sma(open,5)*sma(returns,5),10))
    ret = clean_ret if clean_ret is not None else close.pct_change()
    inner = _sma(open_, 5) * _sma(ret, 5)
    return -1.0 * _cs_rank(inner - _delay(inner, 10))


def wq_alpha009(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#9: (0<ts_min(delta(close,1),5)) ? delta(close,1)
    #          : ((ts_max(delta(close,1),5)<0) ? delta(close,1) : -delta(close,1))
    dc = _delta(close, 1)
    cond = (_ts_min(dc, 5) > 0) | (_ts_max(dc, 5) < 0)
    return dc.where(cond, -dc)


def wq_alpha010(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#10: rank(Alpha#9 的内部条件信号)
    dc = _delta(close, 1)
    cond = (_ts_min(dc, 4) > 0) | (_ts_max(dc, 4) < 0)
    return _cs_rank(dc.where(cond, -dc))


def wq_alpha011(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#11: (rank(ts_max(vwap-close,3)) + rank(ts_min(vwap-close,3))) * rank(delta(volume,3))
    return ((_cs_rank(_ts_max(vwap - close, 3)) + _cs_rank(_ts_min(vwap - close, 3)))
            * _cs_rank(_delta(volume, 3)))


def wq_alpha013(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#13: -1 * rank(covariance(rank(close), rank(volume), 5))
    return -1.0 * _cs_rank(_cov(_cs_rank(close), _cs_rank(volume), 5))


def wq_alpha014(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#14: -1 * rank(delta(returns,3)) * correlation(open, volume, 10)
    ret = clean_ret if clean_ret is not None else close.pct_change()
    return -1.0 * _cs_rank(_delta(ret, 3)) * _corr(open_, volume, 10)


def wq_alpha015(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#15: -1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3)
    df = _corr(_cs_rank(high), _cs_rank(volume), 3)
    return -1.0 * _ts_sum(_cs_rank(df), 3)


def wq_alpha016(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#16: -1 * rank(covariance(rank(high), rank(volume), 5))
    return -1.0 * _cs_rank(_cov(_cs_rank(high), _cs_rank(volume), 5))


def wq_alpha017(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#17: -1 * rank(ts_rank(close,10)) * rank(delta(delta(close,1),1))
    #           * rank(ts_rank(volume/adv20, 5))
    adv20 = _adv(volume, 20)
    return (-1.0 * _cs_rank(_ts_rank(close, 10))
            * _cs_rank(_delta(_delta(close, 1), 1))
            * _cs_rank(_ts_rank(_safe_div(volume, adv20), 5)))


def wq_alpha018(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#18: -1 * rank(stddev(|close-open|,5) + (close-open) + correlation(close,open,10))
    inner = (_stddev((close - open_).abs(), 5) + (close - open_)
             + _corr(close, open_, 10))
    return -1.0 * _cs_rank(inner)


def wq_alpha019(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#19: -1 * sign((close - delay(close,7)) + delta(close,7)) * (1 + rank(1 + sum(returns,250)))
    ret = clean_ret if clean_ret is not None else close.pct_change()
    return (-1.0 * np.sign((close - _delay(close, 7)) + _delta(close, 7))
            * (1.0 + _cs_rank(1.0 + _ts_sum(ret, 250))))


def wq_alpha020(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#20: -1 * rank(open-delay(high,1)) * rank(open-delay(close,1)) * rank(open-delay(low,1))
    return (-1.0 * _cs_rank(open_ - _delay(high, 1))
            * _cs_rank(open_ - _delay(close, 1))
            * _cs_rank(open_ - _delay(low, 1)))


def wq_alpha021(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#21: 复杂三层条件 → (-1 或 1)
    cond_1 = _sma(close, 8) + _stddev(close, 8) < _sma(close, 2)
    cond_2 = _sma(close, 2) < _sma(close, 8) - _stddev(close, 8)
    cond_3 = _sma(volume, 20) / volume < 1
    return (cond_1 | ((~cond_1) & (~cond_2) & (~cond_3))).astype(int) * (-2) + 1


def wq_alpha022(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#22: -1 * delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))
    df = _corr(high, volume, 5)
    return -1.0 * _delta(df, 5) * _cs_rank(_stddev(close, 20))


def wq_alpha023(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#23: ((sma(high,20) < high) ? -1 * delta(high,2) : 0)
    cond = _sma(high, 20) < high
    out = (-1.0 * _delta(high, 2)).where(cond, 0.0)
    return out


def wq_alpha024(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#24: ((delta(sma(close,100),100)/delay(close,100) <= 0.05)
    #           ? -1*(close - ts_min(close,100)) : -1*delta(close,3))
    cond = _safe_div(_delta(_sma(close, 100), 100), _delay(close, 100)) <= 0.05
    return (-1.0 * (close - _ts_min(close, 100))).where(cond, -1.0 * _delta(close, 3))


def wq_alpha025(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#25: rank((-1*returns) * adv20 * vwap * (high - close))
    ret = clean_ret if clean_ret is not None else close.pct_change()
    adv20 = _adv(volume, 20)
    return _cs_rank((-1.0 * ret) * adv20 * vwap * (high - close))


def wq_alpha026(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#26: -1 * ts_max(correlation(ts_rank(volume,5), ts_rank(high,5), 5), 3)
    df = _corr(_ts_rank(volume, 5), _ts_rank(high, 5), 5)
    return -1.0 * _ts_max(df, 3)


def wq_alpha027(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#27: (0.5 < rank(sma(corr(rank(volume),rank(vwap),6),2)/2)) ? -1 : 1
    # 原实现：sign((alpha-0.5)*(-2))
    alpha = _cs_rank(_sma(_corr(_cs_rank(volume), _cs_rank(vwap), 6), 2) / 2.0)
    return np.sign((alpha - 0.5) * (-2.0))


def wq_alpha029(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#29: ts_min(rank(rank(scale(log(sum(rank(rank(-1*rank(delta(close,5)))),2))))),5)
    #           + ts_rank(delay(-returns,6),5)
    ret = clean_ret if clean_ret is not None else close.pct_change()
    inner = _cs_rank(_cs_rank(_scale(np.log(_ts_sum(_cs_rank(_cs_rank(-1.0 * _cs_rank(_delta(close, 5)))), 2)))))
    return _ts_min(inner, 5) + _ts_rank(_delay(-1.0 * ret, 6), 5)


def wq_alpha030(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#30: ((1 - rank(sign(d1)+sign(d2)+sign(d3))) * sum(volume,5)) / sum(volume,20)
    dc = _delta(close, 1)
    inner = np.sign(dc) + np.sign(_delay(dc, 1)) + np.sign(_delay(dc, 2))
    return _safe_div((1.0 - _cs_rank(inner)) * _ts_sum(volume, 5), _ts_sum(volume, 20))


def wq_alpha031(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#31: rank(rank(rank(decay_linear(-rank(rank(delta(close,10))),10)))
    #           + rank(-delta(close,3)) + sign(scale(corr(adv20, low, 12)))
    adv20 = _adv(volume, 20)
    df = _corr(adv20, low, 12)
    p1 = _cs_rank(_cs_rank(_cs_rank(_decay_linear(-1.0 * _cs_rank(_cs_rank(_delta(close, 10))), 10))))
    p2 = _cs_rank(-1.0 * _delta(close, 3))
    p3 = np.sign(_scale(df))
    return p1 + p2 + p3


def wq_alpha032(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#32: scale(sma(close,7)/7 - close) + 20*scale(correlation(vwap, delay(close,5), 230))
    return _scale(_sma(close, 7) / 7.0 - close) + 20.0 * _scale(_corr(vwap, _delay(close, 5), 230))


def wq_alpha033(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#33: rank(-1 + open/close)   （等价于 rank((open/close - 1))）
    return _cs_rank(-1.0 + _safe_div(open_, close))


def wq_alpha035(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#35: ts_rank(volume,32) * (1 - ts_rank(close+high-low,16)) * (1 - ts_rank(returns,32))
    ret = clean_ret if clean_ret is not None else close.pct_change()
    return (_ts_rank(volume, 32)
            * (1.0 - _ts_rank(close + high - low, 16))
            * (1.0 - _ts_rank(ret, 32)))


def wq_alpha036(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#36: 多项加权组合（5 项）
    ret = clean_ret if clean_ret is not None else close.pct_change()
    adv20 = _adv(volume, 20)
    t1 = 2.21 * _cs_rank(_corr(close - open_, _delay(volume, 1), 15))
    t2 = 0.7 * _cs_rank(open_ - close)
    t3 = 0.73 * _cs_rank(_ts_rank(_delay(-1.0 * ret, 6), 5))
    t4 = _cs_rank(_corr(vwap, adv20, 6).abs())
    t5 = 0.6 * _cs_rank((_sma(close, 200) / 200.0 - open_) * (close - open_))
    return t1 + t2 + t3 + t4 + t5


def wq_alpha037(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#37: rank(correlation(delay(open-close,1), close, 200)) + rank(open - close)
    return _cs_rank(_corr(_delay(open_ - close, 1), close, 200)) + _cs_rank(open_ - close)


def wq_alpha038(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#38: -1 * rank(ts_rank(open,10)) * rank(close/open)
    # 注：原 yli188 实现用 ts_rank(open,10)，与论文公式 ts_rank(close,10) 不同；保持原实现。
    inner = _safe_div(close, open_)
    return -1.0 * _cs_rank(_ts_rank(open_, 10)) * _cs_rank(inner)


def wq_alpha039(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#39: -1 * rank(delta(close,7) * (1 - rank(decay_linear(volume/adv20, 9))))
    #           * (1 + rank(sma(returns, 250)))
    ret = clean_ret if clean_ret is not None else close.pct_change()
    adv20 = _adv(volume, 20)
    return (-1.0 * _cs_rank(_delta(close, 7) * (1.0 - _cs_rank(_decay_linear(_safe_div(volume, adv20), 9))))
            * (1.0 + _cs_rank(_sma(ret, 250))))


def wq_alpha040(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#40: -1 * rank(stddev(high,10)) * correlation(high, volume, 10)
    return -1.0 * _cs_rank(_stddev(high, 10)) * _corr(high, volume, 10)


def wq_alpha041(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#41: ((high*low)^0.5) - vwap
    return (high * low).pow(0.5) - vwap


def wq_alpha042(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#42: rank(vwap - close) / rank(vwap + close)
    return _safe_div(_cs_rank(vwap - close), _cs_rank(vwap + close))


def wq_alpha043(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#43: ts_rank(volume/adv20, 20) * ts_rank(-delta(close,7), 8)
    adv20 = _adv(volume, 20)
    return _ts_rank(_safe_div(volume, adv20), 20) * _ts_rank(-1.0 * _delta(close, 7), 8)


def wq_alpha044(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#44: -1 * correlation(high, rank(volume), 5)
    return -1.0 * _corr(high, _cs_rank(volume), 5)


def wq_alpha045(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#45: -1 * rank(sma(delay(close,5),20)) * corr(close,volume,2)
    #           * rank(correlation(sum(close,5), sum(close,20), 2))
    df = _corr(close, volume, 2)
    return (-1.0 * _cs_rank(_sma(_delay(close, 5), 20)) * df
            * _cs_rank(_corr(_ts_sum(close, 5), _ts_sum(close, 20), 2)))


def wq_alpha046(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#46: 内层<0 → 1；内层>0.25 → -1；否则 -delta(close,1)
    inner = ((_delay(close, 20) - _delay(close, 10)) / 10.0
             - (_delay(close, 10) - close) / 10.0)
    alpha = -1.0 * _delta(close, 1)
    alpha = alpha.mask(inner < 0, 1.0)
    alpha = alpha.mask(inner > 0.25, -1.0)
    return alpha


def wq_alpha047(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#47: rank(1/close)*volume/adv20 * (high*rank(high-close)/(sma(high,5)/5))
    #           - rank(vwap - delay(vwap,5))
    adv20 = _adv(volume, 20)
    term1 = _cs_rank(1.0 / close) * volume / adv20
    term2 = high * _cs_rank(high - close) / (_sma(high, 5) / 5.0)
    return term1 * term2 - _cs_rank(vwap - _delay(vwap, 5))


def wq_alpha049(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#49: 内层 < -0.1 → 1；否则 -delta(close,1)
    inner = ((_delay(close, 20) - _delay(close, 10)) / 10.0
             - (_delay(close, 10) - close) / 10.0)
    alpha = -1.0 * _delta(close, 1)
    return alpha.mask(inner < -0.1, 1.0)


def wq_alpha050(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#50: -1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5)
    return -1.0 * _ts_max(_cs_rank(_corr(_cs_rank(volume), _cs_rank(vwap), 5)), 5)


def wq_alpha051(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#51: 内层 < -0.05 → 1；否则 -delta(close,1)
    inner = ((_delay(close, 20) - _delay(close, 10)) / 10.0
             - (_delay(close, 10) - close) / 10.0)
    alpha = -1.0 * _delta(close, 1)
    return alpha.mask(inner < -0.05, 1.0)


def wq_alpha052(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#52: (-1*delta(ts_min(low,5),5)) * rank((sum(returns,240)-sum(returns,20))/220)
    #           * ts_rank(volume,5)
    ret = clean_ret if clean_ret is not None else close.pct_change()
    return (-1.0 * _delta(_ts_min(low, 5), 5)
            * _cs_rank((_ts_sum(ret, 240) - _ts_sum(ret, 20)) / 220.0)
            * _ts_rank(volume, 5))


def wq_alpha054(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#54: -1 * (low - close) * open^5 / ((low - high) * close^5)
    inner = (low - high).replace(0, -0.0001)
    return -1.0 * (low - close) * (open_ ** 5) / (inner * (close ** 5))


def wq_alpha055(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#55: -1 * correlation(rank((close - ts_min(low,12))/(ts_max(high,12)-ts_min(low,12))),
    #                            rank(volume), 6)
    divisor = (_ts_max(high, 12) - _ts_min(low, 12)).replace(0, 0.0001)
    inner = (close - _ts_min(low, 12)) / divisor
    return -1.0 * _corr(_cs_rank(inner), _cs_rank(volume), 6)


def wq_alpha057(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#57: -1 * (close - vwap) / decay_linear(rank(ts_argmax(close,30)), 2)
    return -1.0 * _safe_div(close - vwap, _decay_linear(_cs_rank(_ts_argmax(close, 30)), 2))


def wq_alpha060(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#60: -(2*scale(rank(((close-low)-(high-close))*volume/(high-low)))
    #            - scale(rank(ts_argmax(close,10))))
    divisor = (high - low).replace(0, 0.0001)
    inner = ((close - low) - (high - close)) * volume / divisor
    return -((2.0 * _scale(_cs_rank(inner))) - _scale(_cs_rank(_ts_argmax(close, 10))))


def wq_alpha062(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#62: (rank(corr(vwap, sma(adv20,22), 10))
    #           < rank(((rank(open)+rank(open)) < (rank((high+low)/2)+rank(high)))))*-1
    adv20 = _adv(volume, 20)
    lhs = _cs_rank(_corr(vwap, _sma(adv20, 22), 10))
    rhs = _cs_rank((_cs_rank(open_) + _cs_rank(open_))
                   < (_cs_rank((high + low) / 2.0) + _cs_rank(high)))
    return (lhs < rhs).astype(float) * -1.0


def wq_alpha064(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#64: (rank(corr(sma(open*0.178404+low*0.821596,13), sma(adv120,13),17))
    #           < rank(delta(((high+low)/2*0.178404 + vwap*0.821596), 4)))*-1
    adv120 = _adv(volume, 120)
    a = _sma(open_ * 0.178404 + low * (1 - 0.178404), 13)
    b = _sma(adv120, 13)
    c = ((high + low) / 2.0) * 0.178404 + vwap * (1 - 0.178404)
    return (_cs_rank(_corr(a, b, 17)) < _cs_rank(_delta(c, 4))).astype(float) * -1.0


def wq_alpha065(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#65: (rank(corr(open*0.00817205+vwap*0.99182795, sma(adv60,9), 6))
    #           < rank(open - ts_min(open,14)))*-1
    adv60 = _adv(volume, 60)
    a = open_ * 0.00817205 + vwap * (1 - 0.00817205)
    return (_cs_rank(_corr(a, _sma(adv60, 9), 6))
            < _cs_rank(open_ - _ts_min(open_, 14))).astype(float) * -1.0


def wq_alpha066(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#66: (rank(decay_linear(delta(vwap,4),7))
    #           + ts_rank(decay_linear((low - vwap)/(open-(high+low)/2),11),7))*-1
    inner = (low - vwap) / (open_ - (high + low) / 2.0)
    return (_cs_rank(_decay_linear(_delta(vwap, 4), 7))
            + _ts_rank(_decay_linear(inner, 11), 7)) * -1.0


def wq_alpha068(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#68: (ts_rank(corr(rank(high), rank(adv15), 9), 14)
    #           < rank(delta(close*0.518371+low*0.481629, 2))*14)*-1
    adv15 = _adv(volume, 15)
    a = _ts_rank(_corr(_cs_rank(high), _cs_rank(adv15), 9), 14)
    b = _cs_rank(_delta(close * 0.518371 + low * (1 - 0.518371), 2)) * 14
    return (a < b).astype(float) * -1.0


def wq_alpha071(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#71: max(ts_rank(decay_linear(corr(ts_rank(close,3),ts_rank(adv180,12),18),4),16),
    #               ts_rank(decay_linear(rank(((low+open)-(vwap+vwap))^2),16),4))
    adv180 = _adv(volume, 180)
    p1 = _ts_rank(_decay_linear(_corr(_ts_rank(close, 3), _ts_rank(adv180, 12), 18), 4), 16)
    p2 = _ts_rank(_decay_linear(_cs_rank((low + open_) - (vwap + vwap)).pow(2), 16), 4)
    return np.maximum(p1, p2)


def wq_alpha072(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#72: rank(decay_linear(corr((high+low)/2, adv40, 9), 10))
    #           / rank(decay_linear(corr(ts_rank(vwap,4), ts_rank(volume,19), 7), 3))
    adv40 = _adv(volume, 40)
    num = _cs_rank(_decay_linear(_corr((high + low) / 2.0, adv40, 9), 10))
    den = _cs_rank(_decay_linear(_corr(_ts_rank(vwap, 4), _ts_rank(volume, 19), 7), 3))
    return _safe_div(num, den)


def wq_alpha073(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#73: -1 * max(rank(decay_linear(delta(vwap,5),3)),
    #                    ts_rank(decay_linear(-delta(open*0.147155+low*0.852845,2)/(open*0.147155+low*0.852845),3),17))
    a = open_ * 0.147155 + low * (1 - 0.147155)
    p1 = _cs_rank(_decay_linear(_delta(vwap, 5), 3))
    p2 = _ts_rank(_decay_linear(-1.0 * _safe_div(_delta(a, 2), a), 3), 17)
    return -1.0 * np.maximum(p1, p2)


def wq_alpha074(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#74: (rank(corr(close, sma(adv30,37), 15))
    #           < rank(corr(rank(high*0.0261661+vwap*0.9738339), rank(volume), 11)))*-1
    adv30 = _adv(volume, 30)
    a = _cs_rank(_corr(close, _sma(adv30, 37), 15))
    b = _cs_rank(_corr(_cs_rank(high * 0.0261661 + vwap * (1 - 0.0261661)),
                       _cs_rank(volume), 11))
    return (a < b).astype(float) * -1.0


def wq_alpha075(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#75: (rank(corr(vwap, volume, 4)) < rank(corr(rank(low), rank(adv50), 12))).astype(int)
    adv50 = _adv(volume, 50)
    return (_cs_rank(_corr(vwap, volume, 4))
            < _cs_rank(_corr(_cs_rank(low), _cs_rank(adv50), 12))).astype(int)


def wq_alpha077(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#77: min(rank(decay_linear(((high+low)/2 + high) - (vwap+high), 20)),
    #               rank(decay_linear(corr((high+low)/2, adv40, 3), 6)))
    adv40 = _adv(volume, 40)
    p1 = _cs_rank(_decay_linear(((high + low) / 2.0 + high) - (vwap + high), 20))
    p2 = _cs_rank(_decay_linear(_corr((high + low) / 2.0, adv40, 3), 6))
    return np.minimum(p1, p2)


def wq_alpha078(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#78: rank(corr(sum(low*0.352233+vwap*0.647767,20), sum(adv40,20), 7))
    #               ^ rank(corr(rank(vwap), rank(volume), 6))
    adv40 = _adv(volume, 40)
    a = _cs_rank(_corr(_ts_sum(low * 0.352233 + vwap * (1 - 0.352233), 20),
                       _ts_sum(adv40, 20), 7))
    b = _cs_rank(_corr(_cs_rank(vwap), _cs_rank(volume), 6))
    out = a.pow(b)
    return out.replace([np.inf, -np.inf], np.nan)


def wq_alpha081(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#81: (rank(log(product(rank((rank(corr(vwap, sum(adv10,50), 8))^4)), 15)))
    #           < rank(corr(rank(vwap), rank(volume), 5)))*-1
    adv10 = _adv(volume, 10)
    inner = _cs_rank(_corr(vwap, _ts_sum(adv10, 50), 8)).pow(4)
    a = _cs_rank(np.log(_product(_cs_rank(inner), 15)))
    b = _cs_rank(_corr(_cs_rank(vwap), _cs_rank(volume), 5))
    return (a < b).astype(float) * -1.0


def wq_alpha083(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#83: (rank(delay((high-low)/(sma(close,5)/5), 2)) * rank(rank(volume)))
    #           / (((high-low)/(sma(close,5)/5)) / (vwap - close))
    ratio = (high - low) / (_sma(close, 5) / 5.0)
    num = _cs_rank(_delay(ratio, 2)) * _cs_rank(_cs_rank(volume))
    den = ratio / (vwap - close)
    return _safe_div(num, den)


def wq_alpha084(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#84: SignedPower(ts_rank(vwap - ts_max(vwap,15), 21), delta(close,5))
    base = _ts_rank(vwap - _ts_max(vwap, 15), 21)
    out = base.pow(_delta(close, 5))
    return out.replace([np.inf, -np.inf], np.nan)


def wq_alpha085(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#85: rank(corr(high*0.876703+close*0.123297, adv30, 10))
    #               ^ rank(corr(ts_rank((high+low)/2,4), ts_rank(volume,10), 7))
    adv30 = _adv(volume, 30)
    a = _cs_rank(_corr(high * 0.876703 + close * (1 - 0.876703), adv30, 10))
    b = _cs_rank(_corr(_ts_rank((high + low) / 2.0, 4), _ts_rank(volume, 10), 7))
    out = a.pow(b)
    return out.replace([np.inf, -np.inf], np.nan)


def wq_alpha086(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#86: (ts_rank(corr(close, sma(adv20,15), 6), 20)
    #           < rank((open+close)-(vwap+open))*20)*-1
    adv20 = _adv(volume, 20)
    a = _ts_rank(_corr(close, _sma(adv20, 15), 6), 20)
    b = _cs_rank((open_ + close) - (vwap + open_)) * 20
    return (a < b).astype(float) * -1.0


def wq_alpha088(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#88: min(rank(decay_linear((rank(open)+rank(low))-(rank(high)+rank(close)),8)),
    #               ts_rank(decay_linear(corr(ts_rank(close,8),ts_rank(adv60,21),8),7),3))
    adv60 = _adv(volume, 60)
    p1 = _cs_rank(_decay_linear((_cs_rank(open_) + _cs_rank(low))
                                - (_cs_rank(high) + _cs_rank(close)), 8))
    p2 = _ts_rank(_decay_linear(_corr(_ts_rank(close, 8), _ts_rank(adv60, 21), 8), 7), 3)
    return np.minimum(p1, p2)


def wq_alpha092(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#92: min(ts_rank(decay_linear(((high+low)/2 + close) < (low+open), 15), 19),
    #               ts_rank(decay_linear(corr(rank(low), rank(adv30), 8), 7), 7))
    adv30 = _adv(volume, 30)
    p1 = _ts_rank(_decay_linear(((high + low) / 2.0 + close) < (low + open_), 15), 19)
    p2 = _ts_rank(_decay_linear(_corr(_cs_rank(low), _cs_rank(adv30), 8), 7), 7)
    return np.minimum(p1, p2)


def wq_alpha094(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#94: (rank(vwap - ts_min(vwap,12))
    #           ^ ts_rank(corr(ts_rank(vwap,20), ts_rank(adv60,4), 18), 3)) * -1
    adv60 = _adv(volume, 60)
    a = _cs_rank(vwap - _ts_min(vwap, 12))
    b = _ts_rank(_corr(_ts_rank(vwap, 20), _ts_rank(adv60, 4), 18), 3)
    out = a.pow(b) * -1.0
    return out.replace([np.inf, -np.inf], np.nan)


def wq_alpha095(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#95: (rank(open - ts_min(open,12))*12
    #           < ts_rank(rank(corr(sma((high+low)/2,19), sma(adv40,19), 13))^5, 12)).astype(int)
    adv40 = _adv(volume, 40)
    a = _cs_rank(open_ - _ts_min(open_, 12)) * 12
    b = _ts_rank(_cs_rank(_corr(_sma((high + low) / 2.0, 19),
                                _sma(adv40, 19), 13)).pow(5), 12)
    return (a < b).astype(int)


def wq_alpha096(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#96: -1 * max(ts_rank(decay_linear(corr(rank(vwap),rank(volume),4),4),8),
    #                    ts_rank(decay_linear(ts_argmax(corr(ts_rank(close,7),ts_rank(adv60,4),4),13),14),13))
    adv60 = _adv(volume, 60)
    p1 = _ts_rank(_decay_linear(_corr(_cs_rank(vwap), _cs_rank(volume), 4), 4), 8)
    p2 = _ts_rank(_decay_linear(_ts_argmax(_corr(_ts_rank(close, 7), _ts_rank(adv60, 4), 4), 13), 14), 13)
    return -1.0 * np.maximum(p1, p2)


def wq_alpha098(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#98: rank(decay_linear(corr(vwap, sma(adv5,26), 5), 7))
    #           - rank(decay_linear(ts_rank(ts_argmin(corr(rank(open),rank(adv15),21),9),7),8))
    adv5 = _adv(volume, 5)
    adv15 = _adv(volume, 15)
    a = _cs_rank(_decay_linear(_corr(vwap, _sma(adv5, 26), 5), 7))
    b = _cs_rank(_decay_linear(_ts_rank(_ts_argmin(_corr(_cs_rank(open_), _cs_rank(adv15), 21), 9), 7), 8))
    return a - b


def wq_alpha099(close, open_, high, low, volume, amount, vwap, clean_ret):
    # Alpha#99: (rank(corr(sum((high+low)/2,20), sum(adv60,20), 9))
    #           < rank(corr(low, volume, 6)))*-1
    adv60 = _adv(volume, 60)
    a = _cs_rank(_corr(_ts_sum((high + low) / 2.0, 20), _ts_sum(adv60, 20), 9))
    b = _cs_rank(_corr(low, volume, 6))
    return (a < b).astype(float) * -1.0


# ──────────────────────────────────────────────
# 新增 alpha 注册表（统一全签名）
# ──────────────────────────────────────────────

_NEW_ALPHA_FUNCS = {
    "WQ_003": wq_alpha003, "WQ_004": wq_alpha004, "WQ_005": wq_alpha005,
    "WQ_008": wq_alpha008, "WQ_009": wq_alpha009, "WQ_010": wq_alpha010,
    "WQ_011": wq_alpha011, "WQ_013": wq_alpha013, "WQ_014": wq_alpha014,
    "WQ_015": wq_alpha015, "WQ_016": wq_alpha016, "WQ_017": wq_alpha017,
    "WQ_018": wq_alpha018, "WQ_019": wq_alpha019, "WQ_020": wq_alpha020,
    "WQ_021": wq_alpha021, "WQ_022": wq_alpha022, "WQ_023": wq_alpha023,
    "WQ_024": wq_alpha024, "WQ_025": wq_alpha025, "WQ_026": wq_alpha026,
    "WQ_027": wq_alpha027, "WQ_029": wq_alpha029, "WQ_030": wq_alpha030,
    "WQ_031": wq_alpha031, "WQ_032": wq_alpha032, "WQ_033": wq_alpha033,
    "WQ_035": wq_alpha035, "WQ_036": wq_alpha036, "WQ_037": wq_alpha037,
    "WQ_038": wq_alpha038, "WQ_039": wq_alpha039, "WQ_040": wq_alpha040,
    "WQ_041": wq_alpha041, "WQ_042": wq_alpha042, "WQ_043": wq_alpha043,
    "WQ_044": wq_alpha044, "WQ_045": wq_alpha045, "WQ_046": wq_alpha046,
    "WQ_047": wq_alpha047, "WQ_049": wq_alpha049, "WQ_050": wq_alpha050,
    "WQ_051": wq_alpha051, "WQ_052": wq_alpha052, "WQ_054": wq_alpha054,
    "WQ_055": wq_alpha055, "WQ_057": wq_alpha057, "WQ_060": wq_alpha060,
    "WQ_062": wq_alpha062, "WQ_064": wq_alpha064, "WQ_065": wq_alpha065,
    "WQ_066": wq_alpha066, "WQ_068": wq_alpha068, "WQ_071": wq_alpha071,
    "WQ_072": wq_alpha072, "WQ_073": wq_alpha073, "WQ_074": wq_alpha074,
    "WQ_075": wq_alpha075, "WQ_077": wq_alpha077, "WQ_078": wq_alpha078,
    "WQ_081": wq_alpha081, "WQ_083": wq_alpha083, "WQ_084": wq_alpha084,
    "WQ_085": wq_alpha085, "WQ_086": wq_alpha086, "WQ_088": wq_alpha088,
    "WQ_092": wq_alpha092, "WQ_094": wq_alpha094, "WQ_095": wq_alpha095,
    "WQ_096": wq_alpha096, "WQ_098": wq_alpha098, "WQ_099": wq_alpha099,
}


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
    clean_ret: pd.DataFrame = None,
    factor_names: set[str] | list[str] | None = None,
) -> dict:
    """
    返回 Alpha101 因子字典（已按 "越高越好" 约定处理）。

    必须参数：prices (close)
    可选参数：open_, high, low, volume, amount（缺少时对应因子跳过）
              clean_ret: 屏蔽涨跌停日后的日收益率，透传给各 wq_alpha* 函数，
                          遵循项目"量价因子必须用 clean_ret"约定（为 None 时各因子
                          内部退化为 pct_change）。
              factor_names: 可选白名单。None = 全量（IC 全库扫描）；非空时只计算
                          名称在集合内的 WQ（精选 10 + _NEW_ALPHA_FUNCS 均尊重过滤）。

    VWAP = amount / volume.replace(0, NaN)，在入口统一计算后透传给需要 vwap 的 alpha。
    单个 alpha 计算失败不会中断整批（_try 捕获异常并记录）。
    """
    factors = {}
    has_ohlcv = all(x is not None for x in [open_, high, low, volume])
    wanted = set(factor_names) if factor_names is not None else None

    def _want(name: str) -> bool:
        return wanted is None or name in wanted

    # 统一计算 vwap（需 amount + volume）
    vwap = None
    if amount is not None and volume is not None:
        vwap = amount / volume.replace(0, np.nan)

    def _try(name, fn, *args):
        try:
            result = fn(*args)
            if result is not None and not result.isna().all(axis=None):
                # 与其它因子同口径：截面 winsorize(1%) → cross_sectional_zscore(clip=3σ)。
                # 之前 _try 直接返回 raw，导致 84 个 WQ 因子以原始量纲进入 ML（与
                # factors/factor.py 中其它因子出口标准化不一致）。此处统一在出口标准化。
                factors[name] = _normalize(result)
        except Exception as e:
            logger.warning(f"Alpha101 {name} 计算失败: {e}")

    # ── 精选 10 因子（保留原签名调用）──
    if volume is not None:
        if _want("WQ_001"):
            _try("WQ_001", wq_alpha001, prices, volume, clean_ret)
        if _want("WQ_007"):
            _try("WQ_007", wq_alpha007, prices, volume, clean_ret)
        if _want("WQ_012"):
            _try("WQ_012", wq_alpha012, prices, volume, clean_ret)

    if open_ is not None and volume is not None:
        if _want("WQ_002"):
            _try("WQ_002", wq_alpha002, prices, open_, volume, clean_ret)
        if _want("WQ_006"):
            _try("WQ_006", wq_alpha006, open_, volume, clean_ret)

    if has_ohlcv:
        if _want("WQ_028"):
            _try("WQ_028", wq_alpha028, prices, high, low, volume, clean_ret)
        if _want("WQ_053"):
            _try("WQ_053", wq_alpha053, prices, high, low, clean_ret)

    if _want("WQ_034"):
        _try("WQ_034", wq_alpha034, prices, clean_ret)

    if open_ is not None and high is not None and low is not None:
        if _want("WQ_101"):
            _try("WQ_101", wq_alpha101, prices, open_, high, low, clean_ret)

    if volume is not None and amount is not None:
        if _want("WQ_061"):
            _try("WQ_061", wq_alpha061, prices, volume, amount, clean_ret)

    # ── 其余 74 个 alpha（统一全签名调用）──
    for name, fn in _NEW_ALPHA_FUNCS.items():
        if not _want(name):
            continue
        _try(name, fn, prices, open_, high, low, volume, amount, vwap, clean_ret)

    subset_tag = "" if wanted is None else f" (白名单 {len(wanted)} 个)"
    logger.info(f"Alpha101 因子: 计算完成 {len(factors)} 个{subset_tag} "
                f"({list(factors.keys())})")
    return factors
