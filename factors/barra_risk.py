"""
factors/barra_risk.py  —  简化版 Barra CNE 风格因子

用途：
    在 ic_analysis.py --barra 中作为截面回归控制变量，
    计算"纯因子IC"（alpha因子剔除系统性风险敞口后的真实alpha信号）。
    不直接用于选股。

9个风格因子（参考 Barra CNE5/CNE6）：
    Barra_Size        市值对数（大市值 vs 小市值暴露）
    Barra_NonlinSize  非线性规模（市值^3/2 正交化，捕捉中盘效应）
    Barra_Beta        系统性风险（对沪深300的滚动252日 beta）
    Barra_Momentum    12-1月动量（跳过最近1月，避免短期反转污染）
    Barra_ResVol      残差波动率（剔除市场 beta 后的特质风险）
    Barra_Value       账面市值比 B/P = 1/PB
    Barra_Liquidity   流动性（20日对数换手量）
    Barra_Leverage    财务杠杆（资产负债率）
    Barra_Growth      成长性（营收同比增速）

用法（只在 ic_analysis.py 内部调用，不独立运行）：
    from factors.barra_risk import get_barra_factors
    barra = get_barra_factors(prices, financial, market_prices, volume)
"""
import numpy as np
import pandas as pd
from loguru import logger
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import RAW_DIR
from utils.pit_align import pit_pivot_ffill


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _cross_zscore(df: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """截面 z-score 标准化（clip=3σ），缺失值保持 NaN"""
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0).clip(-clip, clip)


def _pivot_ffill(financial: pd.DataFrame, col: str,
                 price_index: pd.Index) -> pd.DataFrame:
    """
    财务数据透视（长表→宽表）并前向填充到日频（PIT 安全）。

    把报告期 trade_date 按 A 股法定披露窗口（Q1/Q3 +45 天，半年报 +75 天，
    年报 +120 天）平移到「可用日下界」后再 pivot + ffill，
    消除用报告期日做 ffill 起点的 look-ahead bias。
    详见 utils/pit_align.py。
    """
    return pit_pivot_ffill(
        financial, pd.DatetimeIndex(price_index),
        date_col="trade_date", value_cols=[col],
    )


# ── 各因子实现 ────────────────────────────────────────────────────────────────

def barra_size(financial: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame | None:
    """规模：log(总市值)"""
    for col in ("total_mv", "total_assets"):
        if col in financial.columns:
            mv = _pivot_ffill(financial, col, prices.index)
            log_mv = np.log(mv.replace(0, np.nan).abs())
            logger.info(f"Barra_Size 使用列: {col}")
            return _cross_zscore(log_mv)
    logger.warning("Barra_Size: 无 total_mv/total_assets 列，跳过")
    return None


def barra_nonlin_size(size_df: pd.DataFrame) -> pd.DataFrame | None:
    """非线性规模：size^(3/2) 在每个截面正交化掉线性 Size"""
    if size_df is None:
        return None
    cubert = size_df.abs() ** 1.5 * np.sign(size_df)
    result = pd.DataFrame(np.nan, index=size_df.index, columns=size_df.columns)

    for date in size_df.index:
        s = size_df.loc[date].dropna()
        nl = cubert.loc[date].reindex(s.index).dropna()
        common = s.index.intersection(nl.index)
        if len(common) < 30:
            continue
        s_vals = s.loc[common].values
        nl_vals = nl.loc[common].values
        A = np.column_stack([np.ones(len(s_vals)), s_vals])
        try:
            coef, _, _, _ = np.linalg.lstsq(A, nl_vals, rcond=None)
            result.loc[date, common] = nl_vals - A @ coef
        except Exception:
            pass

    return _cross_zscore(result)


def barra_beta(prices: pd.DataFrame, market_prices: pd.DataFrame,
               window: int = 252,
               clean_ret: pd.DataFrame | None = None,
               mkt_clean_ret: pd.Series | None = None) -> pd.DataFrame:
    """Beta：滚动252日 OLS beta（向量化）

    优先使用 clean_ret（涨跌停日 return=NaN），避免涨跌停日强制 ±10%/±20%
    截断污染个股 beta 与市场相关系数。回退到 prices.pct_change() 以兼容旧调用。
    """
    stock_ret = clean_ret if clean_ret is not None else prices.pct_change()
    mkt_ret = mkt_clean_ret if mkt_clean_ret is not None else market_prices.squeeze().pct_change()
    common = stock_ret.index.intersection(mkt_ret.index)
    stock_ret = stock_ret.loc[common]
    mkt_ret = mkt_ret.loc[common]

    # 向量化：cov(ri, rm) / var(rm)
    cov = stock_ret.rolling(window, min_periods=window // 2).cov(mkt_ret)
    var = mkt_ret.rolling(window, min_periods=window // 2).var()
    beta_df = cov.div(var, axis=0)
    return _cross_zscore(beta_df)


def barra_momentum(prices: pd.DataFrame,
                   long_window: int = 240,
                   skip_window: int = 20,
                   clean_ret: pd.DataFrame | None = None) -> pd.DataFrame:
    """动量：12-1月跳过动量

    用 clean_ret 累乘得到区间收益，避免涨跌停日被强制截断污染动量信号。
    回退到 prices.pct_change() 以兼容旧调用。
    """
    if clean_ret is not None:
        ret_long = (1 + clean_ret).rolling(long_window, min_periods=long_window // 2).apply(
            np.prod, raw=True
        ) - 1
        ret_skip = (1 + clean_ret).rolling(skip_window, min_periods=skip_window // 2).apply(
            np.prod, raw=True
        ) - 1
    else:
        ret_long = prices.pct_change(long_window)
        ret_skip = prices.pct_change(skip_window)
    mom = (1 + ret_long) / (1 + ret_skip.replace(-1, np.nan)) - 1
    return _cross_zscore(mom)


def barra_res_vol(prices: pd.DataFrame, market_prices: pd.DataFrame,
                  window: int = 60,
                  clean_ret: pd.DataFrame | None = None,
                  mkt_clean_ret: pd.Series | None = None) -> pd.DataFrame:
    """残差波动率：vol × sqrt(1 - R²)，R = 与市场的滚动相关系数（向量化近似）

    优先使用 clean_ret，避免涨跌停日强制截断高估与市场的相关系数，
    进而低估残差波动率（特质风险）。回退到 prices.pct_change() 兼容旧调用。
    """
    stock_ret = clean_ret if clean_ret is not None else prices.pct_change()
    mkt_ret = mkt_clean_ret if mkt_clean_ret is not None else market_prices.squeeze().pct_change()
    common = stock_ret.index.intersection(mkt_ret.index)
    stock_ret = stock_ret.loc[common]
    mkt_ret = mkt_ret.loc[common]

    corr_with_mkt = stock_ret.rolling(window, min_periods=window // 2).corr(mkt_ret)
    total_vol = stock_ret.rolling(window, min_periods=window // 2).std()
    res_vol = total_vol * np.sqrt((1 - corr_with_mkt.clip(-0.999, 0.999) ** 2).clip(0))
    return _cross_zscore(res_vol)


def barra_value(financial: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame | None:
    """价值：B/P = 1/PB"""
    if "pb" not in financial.columns:
        logger.warning("Barra_Value: 无 pb 列，跳过")
        return None
    pb = _pivot_ffill(financial, "pb", prices.index)
    bp = 1.0 / pb.replace(0, np.nan)
    return _cross_zscore(bp)


def barra_liquidity(volume: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """流动性：log(20日均成交量)（越高流动性越好 → 流动性因子值越大）"""
    avg_vol = volume.rolling(window, min_periods=window // 2).mean()
    log_vol = np.log(avg_vol.replace(0, np.nan))
    return _cross_zscore(log_vol)


def barra_leverage(financial: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame | None:
    """杠杆：资产负债率"""
    for col in ("debt_to_assets", "debt_asset_ratio", "liabilities_to_assets"):
        if col in financial.columns:
            lev = _pivot_ffill(financial, col, prices.index)
            return _cross_zscore(lev)
    logger.warning("Barra_Leverage: 无资产负债率列，跳过")
    return None


def barra_growth(financial: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame | None:
    """成长：营收同比增速"""
    for col in ("revenue_yoy", "or_yoy", "total_revenue_yoy"):
        if col in financial.columns:
            yoy = _pivot_ffill(financial, col, prices.index)
            return _cross_zscore(yoy)
    # fallback：从 revenue 自算同比（季度频率，所以近似用4个季报间隔）
    for col in ("revenue", "total_revenue", "or"):
        if col in financial.columns:
            pivot = _pivot_ffill(financial, col, prices.index)
            # 用252交易日近似1年同比（已 ffill，所以是滚动的）
            yoy = pivot.pct_change(252)
            logger.info(f"Barra_Growth 用 {col}.pct_change(252) 近似同比")
            return _cross_zscore(yoy)
    logger.warning("Barra_Growth: 无营收相关列，跳过")
    return None


# ── 汇总入口 ──────────────────────────────────────────────────────────────────

def get_barra_factors(
    prices: pd.DataFrame,
    financial: pd.DataFrame = None,
    market_prices: pd.DataFrame = None,
    volume: pd.DataFrame = None,
    clean_ret: pd.DataFrame | None = None,
    mkt_clean_ret: pd.Series | None = None,
    industry_map: pd.Series | None = None,
) -> dict:
    """
    构建所有可用的简化 Barra 风格因子。

    返回 dict: {因子名: DataFrame(index=date, columns=stock)}
    数值为截面 z-score（±3σ clip），可直接用于截面回归控制变量。

    参数
    ----
    clean_ret : pd.DataFrame, optional
        涨跌停日 return=NaN 的日频收益（data.clean.clean_ohlcv 的输出）。
        传入后 Barra_Beta / Barra_ResVol / Barra_Momentum 用其替代
        prices.pct_change()，避免涨跌停截断污染。默认 None 时回退到 pct_change。
    mkt_clean_ret : pd.Series, optional
        市场指数的清洁收益（指数本身无涨跌停，一般可省）。默认 None 回退到
        market_prices.pct_change()。
    industry_map : pd.Series, optional
        index=stock, value=行业分类的截面映射。传入后对所有 Barra 因子做
        截面行业中性化（factor - 行业截面均值）再 z-score，剔除 Size/Liquidity/
        Leverage 等因子的行业残余成分，作为残差化控制变量更彻底。
        默认 None 时不动现有行为（向后兼容）。
    """
    factors = {}

    # ── 规模（尽早计算，NonlinSize 依赖它）
    if financial is not None:
        size_df = barra_size(financial, prices)
        if size_df is not None:
            factors["Barra_Size"] = size_df
            logger.info("计算 Barra_NonlinSize...")
            nl = barra_nonlin_size(size_df)
            if nl is not None:
                factors["Barra_NonlinSize"] = nl

    # ── 需要市场指数的因子
    if market_prices is not None:
        logger.info("计算 Barra_Beta（滚动252日，向量化）...")
        factors["Barra_Beta"] = barra_beta(
            prices, market_prices,
            clean_ret=clean_ret, mkt_clean_ret=mkt_clean_ret,
        )
        logger.info("计算 Barra_ResVol（滚动60日残差波动）...")
        factors["Barra_ResVol"] = barra_res_vol(
            prices, market_prices,
            clean_ret=clean_ret, mkt_clean_ret=mkt_clean_ret,
        )

    # ── 动量（纯价格）
    logger.info("计算 Barra_Momentum...")
    factors["Barra_Momentum"] = barra_momentum(prices, clean_ret=clean_ret)

    # ── 财务类因子
    if financial is not None:
        val = barra_value(financial, prices)
        if val is not None:
            factors["Barra_Value"] = val
        lev = barra_leverage(financial, prices)
        if lev is not None:
            factors["Barra_Leverage"] = lev
        gro = barra_growth(financial, prices)
        if gro is not None:
            factors["Barra_Growth"] = gro

    # ── 流动性
    if volume is not None:
        logger.info("计算 Barra_Liquidity...")
        factors["Barra_Liquidity"] = barra_liquidity(volume)

    factors = {k: v for k, v in factors.items() if v is not None}

    # ── 行业中性化（P1-3）：剔除 Barra 风格因子的行业残余成分
    # 控制变量之间更彻底正交，避免 Size/Liquidity/Leverage 等天然行业属性
    # 污染下游纯 IC 残差。默认 None 时跳过，保持向后兼容。
    if industry_map is not None and factors:
        ind_reindexed = industry_map.dropna()
        for name in list(factors.keys()):
            fdf = factors[name]
            # 按列（股票）对齐行业映射
            common_stocks = fdf.columns.intersection(ind_reindexed.index)
            if len(common_stocks) == 0:
                continue
            sub = fdf[common_stocks]
            ind_aligned = ind_reindexed.reindex(common_stocks)
            # 截面去行业均值：每行 factor - 该行各行业的截面均值
            # groupby(axis=1).transform('mean') 按列分组、对每行做组内均值
            industry_mean = sub.T.groupby(ind_aligned.values).transform("mean").T
            sub_neutral = sub - industry_mean
            # 行业哑变量列（未分类股票）保持原值
            if len(common_stocks) < fdf.shape[1]:
                neutralized = fdf.copy()
                neutralized[common_stocks] = sub_neutral
            else:
                neutralized = sub_neutral
            # 重新做截面 z-score，保持尺度一致
            mean = neutralized.mean(axis=1)
            std = neutralized.std(axis=1).replace(0, np.nan)
            factors[name] = neutralized.sub(mean, axis=0).div(std, axis=0).clip(-3.0, 3.0)
        logger.info("Barra 因子已完成行业中性化（截面去行业均值 + 重 z-score）")

    logger.info(f"Barra 风格因子就绪: {len(factors)} 个 → {list(factors.keys())}")
    return factors
