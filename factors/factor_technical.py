"""
factors/factor_technical.py  —  技术分析类因子 + 行业中性量价因子

因子来源说明（所有公式均来自公开文献，非自行构造）：
  - BIAS/PSY/ARBR：《技术分析》教材标准公式；
    亦收录于国泰君安证券研究《基于技术分析的多因子选股体系》（2018）
  - 换手率系列变体：参考华泰证券研究《因子选股系列》（2017-2018）
    及东方证券《量价因子研究》系列
  - BIAS 在 A 股短期均值回归特性：
    Hou, Peng, Xiong (2006) "A Tale of Two Anomalies: The Implications..."
    以及 Liu, Stambaugh, Yuan (2019) "Size and Value in China" JFE

所有因子输出格式：
    DataFrame(index=日期, columns=股票) ，已经过 winsorize(1%) + z-score 归一化
    不预设方向（IC分析自动确定），模型自行学习方向系数

注意：行业中性化操作按列分组（vectorized），避免逐行 apply 性能瓶颈。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from factors.factor import _normalize, winsorize, cross_sectional_zscore


# ══════════════════════════════════════════════════════════════════════════════
# 辅助工具
# ══════════════════════════════════════════════════════════════════════════════

def _industry_demean(panel: pd.DataFrame, industry_map: pd.Series) -> pd.DataFrame:
    """
    截面行业去均值（vectorized）：
    对 panel 每一行（日期截面），减去同行业列的均值。
    industry_map: Series(stock_code → industry_label)
    """
    ind = industry_map.reindex(panel.columns).fillna("未分类")
    result = panel.copy()
    for grp_name in ind.unique():
        cols = ind[ind == grp_name].index.tolist()
        cols = [c for c in cols if c in panel.columns]
        if len(cols) < 2:
            continue
        grp = panel[cols]
        grp_mean = grp.mean(axis=1)          # Series: (date,)
        result[cols] = grp.sub(grp_mean, axis=0)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# BIAS — 乖离率
# 公式：BIAS_N = (Close - MA_N) / MA_N
# 来源：技术分析教材标准定义；国泰君安191因子研究报告
# 经济含义：价格偏离均线程度；A股短期表现为均值回归，中长期动量持续
# ══════════════════════════════════════════════════════════════════════════════

def factor_bias(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    ma = prices.rolling(window, min_periods=window // 2).mean()
    bias = (prices - ma) / ma.replace(0, np.nan)
    return _normalize(bias)


# ══════════════════════════════════════════════════════════════════════════════
# PSY — 心理线（Psychological Line）
# 公式：PSY_N = count(daily_ret > 0, past N days) / N × 100
# 来源：J. Welles Wilder (1978) "New Concepts in Technical Trading Systems"
#       以及技术分析教材广泛引用；国泰君安191因子报告
# 经济含义：上涨天数占比，反映市场情绪持续性
# ══════════════════════════════════════════════════════════════════════════════

def factor_psy(prices: pd.DataFrame, window: int = 12,
               clean_ret: pd.DataFrame = None) -> pd.DataFrame:
    ret = clean_ret if clean_ret is not None else prices.pct_change()
    psy = (ret > 0).rolling(window, min_periods=window // 2).mean() * 100
    return _normalize(psy)


# ══════════════════════════════════════════════════════════════════════════════
# AR / BR — 人气指标 / 意愿指标（ARBR）
# 来源：日本技术分析；国泰君安191因子报告；华泰证券量价因子研究
# AR 公式（N 日）：∑(High_i - Open_i) / ∑(Open_i - Low_i) × 100
#   AR > 100：买方力量强于卖方（日内主动买入意愿高）
# BR 公式（N 日）：∑max(0, High_i - Close_{i-1}) / ∑max(0, Close_{i-1} - Low_i) × 100
#   BR > 100：相对昨收的买方意愿强（隔夜买方力量强）
# ══════════════════════════════════════════════════════════════════════════════

def factor_ar(open_: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame,
              window: int = 26) -> pd.DataFrame:
    numer = (high - open_).rolling(window, min_periods=window // 2).sum()
    denom = (open_ - low).replace(0, np.nan).rolling(window, min_periods=window // 2).sum()
    ar = numer / denom * 100
    return _normalize(ar)


def factor_br(prices: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame,
              window: int = 26) -> pd.DataFrame:
    prev_close = prices.shift(1)
    numer = (high - prev_close).clip(lower=0).rolling(window, min_periods=window // 2).sum()
    denom = (prev_close - low).clip(lower=0).replace(0, np.nan).rolling(
        window, min_periods=window // 2).sum()
    br = numer / denom * 100
    return _normalize(br)


# ══════════════════════════════════════════════════════════════════════════════
# 换手率行业中性 — Industry-Adjusted Turnover
# 来源：华泰证券《因子选股系列之换手率因子》（2017）
#       东方证券《A股量价因子研究》
# 公式：Turnover_IA_i,t = Turnover_i,t - mean(Turnover_j,t | j∈industry_i)
# 经济含义：剔除行业整体资金活跃度后的个股相对异动
#           同行业内换手率明显偏高 → 个股被关注/炒作信号
# ══════════════════════════════════════════════════════════════════════════════

def factor_turnover_neutral(volume: pd.DataFrame,
                            industry_map: pd.Series,
                            window: int = 20,
                            industry_panel: pd.DataFrame = None) -> pd.DataFrame:
    """
    Industry-Adjusted Turnover：剔除行业整体资金活跃度后的个股相对异动。

    industry_map:     静态 Series(stock_code → industry_label)，向后兼容
    industry_panel:   PIT 行业长表；传入时按截面日期取当期行业做去均值，
                      消除用全样本静态行业映射的未来信息。优先于 industry_map。
    """
    avg_vol = volume.rolling(window, min_periods=window // 2).mean()
    if industry_panel is not None:
        try:
            from factors.factor_alpha import _pit_industry_wide, _pit_industry_demean
            ind_wide = _pit_industry_wide(industry_panel, avg_vol.index, level="sw_l2")
            neutral = _pit_industry_demean(avg_vol, ind_wide)
            logger.info("换手率行业中性_20d: 启用 PIT 行业面板")
            return _normalize(neutral)
        except Exception as e:
            logger.warning(f"换手率行业中性 PIT 路径失败，回退静态: {e}")
    # 行业内去均值（静态）
    neutral = _industry_demean(avg_vol, industry_map)
    # 不预设方向（高相对换手可能是买入信号也可能是过热反转）
    return _normalize(neutral)


# ══════════════════════════════════════════════════════════════════════════════
# 换手率加速度 — Turnover Acceleration
# 来源：东方证券《量价因子研究》（2019）
#       参考 Lou, Polk, Skouras (2019) "A Tug of War: Overnight vs Intraday..."
# 公式：Accel_i,t = MA(volume, 5d) / MA(volume, 20d)
# 经济含义：近期成交量相对中期均量的加速比；
#           加速 > 1 说明近期资金流入加速，可能是趋势延续或过热信号
# ══════════════════════════════════════════════════════════════════════════════

def factor_turnover_acceleration(volume: pd.DataFrame,
                                  short: int = 5,
                                  long: int = 20) -> pd.DataFrame:
    ma_short = volume.rolling(short, min_periods=short // 2).mean()
    ma_long  = volume.rolling(long,  min_periods=long  // 2).mean()
    accel = ma_short / ma_long.replace(0, np.nan)
    return _normalize(accel)


# ══════════════════════════════════════════════════════════════════════════════
# 价格相对行业强度 — Industry-Relative Price Strength
# 来源：参考 Moskowitz, Grinblatt (1999) "Do Industries Explain Momentum?" JF
#       华泰证券行业动量因子研究
# 公式：IRS_i,t = Return_i(N) - mean(Return_j(N) | j∈industry_i)
# 经济含义：个股动量剔除行业整体动量后的超额部分（纯个股选择能力信号）
# ══════════════════════════════════════════════════════════════════════════════

def factor_industry_relative_strength(prices: pd.DataFrame,
                                       industry_map: pd.Series,
                                       window: int = 20,
                                       clean_ret: pd.DataFrame = None,
                                       industry_panel: pd.DataFrame = None) -> pd.DataFrame:
    """
    Industry-Relative Price Strength：个股动量剔除行业整体动量后的超额部分。

    industry_map:     静态 Series(stock_code → industry_label)，向后兼容
    industry_panel:   PIT 行业长表；传入时按截面日期取当期行业做去均值，
                      消除用全样本静态行业映射的未来信息。优先于 industry_map。
    """
    if clean_ret is not None:
        ret = (1 + clean_ret).rolling(window, min_periods=window // 2).apply(
            lambda x: np.nanprod(x) - 1, raw=True)
    else:
        ret = prices.pct_change(window)
    if industry_panel is not None:
        try:
            from factors.factor_alpha import _pit_industry_wide, _pit_industry_demean
            ind_wide = _pit_industry_wide(industry_panel, ret.index, level="sw_l2")
            irs = _pit_industry_demean(ret, ind_wide)
            logger.info("行业相对强度_20d: 启用 PIT 行业面板")
            return _normalize(irs)
        except Exception as e:
            logger.warning(f"行业相对强度 PIT 路径失败，回退静态: {e}")
    irs = _industry_demean(ret, industry_map)
    return _normalize(irs)


# ══════════════════════════════════════════════════════════════════════════════
# 注册入口
# ══════════════════════════════════════════════════════════════════════════════

def get_technical_factors(
    prices: pd.DataFrame,
    volume: pd.DataFrame = None,
    industry_map=None,
    open_: pd.DataFrame = None,
    high: pd.DataFrame = None,
    low: pd.DataFrame = None,
    clean_ret: pd.DataFrame = None,
    industry_panel: pd.DataFrame = None,
) -> dict:
    """
    返回 {因子名: DataFrame} 字典，供 get_factor_registry() 合并。
    所有因子均有对应文献来源，可在 IC 分析中按需筛选。

    industry_panel: PIT 行业长表；传入时换手率行业中性 / 行业相对强度
                    按截面日期取当期行业做去均值（PIT 安全）。
    """
    reg = {}

    # BIAS 系列：5日（短期均值回归）& 20日（中期趋势）
    reg["BIAS_5d"]  = factor_bias(prices, window=5)
    reg["BIAS_20d"] = factor_bias(prices, window=20)

    # PSY 心理线：12日（标准参数）
    reg["PSY_12d"] = factor_psy(prices, window=12, clean_ret=clean_ret)

    # ARBR：需要 open/high/low
    if open_ is not None and high is not None and low is not None:
        reg["AR_26d"] = factor_ar(open_, high, low, window=26)
        reg["BR_26d"] = factor_br(prices, high, low, window=26)

    # 换手率变体：需要 volume
    if volume is not None:
        reg["换手率加速度"] = factor_turnover_acceleration(volume, short=5, long=20)
        if industry_map is not None or industry_panel is not None:
            if isinstance(industry_map, pd.DataFrame):
                ind_s = industry_map.get("sw_l2", industry_map.iloc[:, 0])
            else:
                ind_s = industry_map
            reg["换手率行业中性_20d"] = factor_turnover_neutral(
                volume, ind_s, window=20, industry_panel=industry_panel)

    # 行业相对强度：需要 industry_map 或 industry_panel
    if industry_map is not None or industry_panel is not None:
        if isinstance(industry_map, pd.DataFrame):
            ind_s = industry_map.get("sw_l2", industry_map.iloc[:, 0])
        else:
            ind_s = industry_map
        reg["行业相对强度_20d"] = factor_industry_relative_strength(
            prices, ind_s, window=20,
            clean_ret=clean_ret, industry_panel=industry_panel)

    return reg
