"""
factors/factor_alpha.py  —  Alpha因子库（第二批）

包含：
  行业动量因子         — 同行业过去N日平均涨幅，捕捉行业轮动
  特质波动率因子       — 剔除市场beta后的残差波动率（IVOL）
  融资余额变化率       — 融资买入净增量，反映杠杆资金情绪
  东财大单净流入       — 主力资金净流入，机构行为代理指标
  北向资金持股变化     — 外资对个股的净增减持
  机构持仓季度变化     — 基金重仓股季度增减持信号

所有因子输出：DataFrame(index=日期, columns=股票), 值越大越优先
"""
import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import RAW_DIR
from factors.factor import _normalize, cross_sectional_zscore, winsorize
from utils.pit_align import pit_reindex_ffill


# ══════════════════════════════════════════════════════════════════════════════
# 行业动量因子
# ══════════════════════════════════════════════════════════════════════════════

def factor_industry_momentum(
    prices: pd.DataFrame,
    industry_map: pd.DataFrame,
    window: int = 20,
    min_stocks: int = 5,
    clean_ret: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    行业动量因子：每只股票过去N日所在行业的平均涨幅（排除自身）。

    行业轮动在A股显著，牛市中强势行业有动量延续。
    "排除自身"避免自相关偏差（IDiosyncratic）。

    industry_map: DataFrame，index=code，列含 'sw_l2'（申万二级行业）
    clean_ret:   屏蔽涨跌停日后的日收益率；传入时用逐日复合，避免涨停日
                  return 截断污染行业均值；为 None 时退化为 pct_change(window)。
    """
    # 对齐股票列表
    common = prices.columns.intersection(industry_map.index)
    if len(common) == 0:
        logger.warning("行业动量：无公共股票，检查industry_map格式")
        return None

    prices_aligned = prices[common]
    if clean_ret is not None:
        # 用 clean_ret 逐日复合，涨跌停日 NaN 透明跳过
        ret = (1 + clean_ret[common]).rolling(
            window, min_periods=max(1, window // 2)
        ).apply(lambda x: np.nanprod(x) - 1, raw=True)
    else:
        ret = prices_aligned.pct_change(window)  # (date, stock) 收益率

    sw_l2 = industry_map.loc[common, "sw_l2"]
    industries = sw_l2.unique()

    result = pd.DataFrame(np.nan, index=ret.index, columns=common)

    for ind in industries:
        members = sw_l2[sw_l2 == ind].index.tolist()
        members_in_data = [c for c in members if c in ret.columns]
        if len(members_in_data) < min_stocks:
            continue

        ind_ret = ret[members_in_data]

        # 对每只股票：行业均值 = (行业总和 - 自身) / (N-1)
        ind_sum = ind_ret.sum(axis=1)           # 各日期行业总收益
        ind_count = ind_ret.notna().sum(axis=1)  # 有效股票数

        for code in members_in_data:
            # 行业均值排除自身
            self_ret = ind_ret[code]
            others_sum = ind_sum - self_ret.fillna(0)
            others_cnt = ind_count - self_ret.notna().astype(int)
            ind_avg = others_sum / others_cnt.replace(0, np.nan)
            result[code] = ind_avg

    # 补全不在industry_map里的股票（填NaN，标准化时会忽略）
    full_result = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    full_result[common] = result
    return _normalize(full_result)


# ══════════════════════════════════════════════════════════════════════════════
# 特质波动率（IVOL）
# ══════════════════════════════════════════════════════════════════════════════

def factor_idiosyncratic_vol(
    prices: pd.DataFrame,
    market_prices: pd.DataFrame,
    window: int = 60,
    min_obs: int = 30,
    clean_ret: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    特质波动率（IVOL）：用市场模型剔除系统性风险后的残差波动率，取反。

    IVOL = std(ε)，其中 r_i = α + β * r_m + ε
    用滚动OLS估计，window=60个交易日。

    A股中低IVOL股票往往有超额收益（与美股IVOL anomaly方向类似，
    但机制不同——A股高IVOL多为散户炒作标的，均值回归快）。

    market_prices: 市场指数价格序列（单列DataFrame或Series，推荐中证全指 sh000985，
                   覆盖全A股，比沪深300更适合做市场组合代理）
    clean_ret:    屏蔽涨跌停日后的日收益率。传入时 stock_ret 用 clean_ret（个股涨跌停
                   日 NaN），mkt_ret 仍用 market_prices 的指数收益（指数本身不受个股
                   涨跌停影响），保证 OLS 残差不被涨跌停日的截断收益污染；
                   为 None 时退化为原 pct_change 逻辑。
    """
    if clean_ret is not None:
        stock_ret = clean_ret
        # 市场收益用指数收益（指数本身不受个股涨跌停影响）
        mkt_ret = market_prices.squeeze().pct_change()
    else:
        stock_ret = prices.pct_change()
        mkt_ret = market_prices.squeeze().pct_change()

    # 对齐日期
    common_dates = stock_ret.index.intersection(mkt_ret.index)
    stock_ret = stock_ret.loc[common_dates]
    mkt_ret = mkt_ret.loc[common_dates]

    ivol = pd.DataFrame(np.nan, index=stock_ret.index, columns=stock_ret.columns)

    # 滚动回归：对每个窗口计算残差std
    # 向量化：先demean市场收益，用OLS公式一次性算所有股票
    mkt_arr = mkt_ret.values

    for end_idx in range(min_obs, len(common_dates)):
        start_idx = max(0, end_idx - window)
        if end_idx - start_idx < min_obs:
            continue

        date = common_dates[end_idx]
        rm = mkt_arr[start_idx:end_idx]
        ri = stock_ret.iloc[start_idx:end_idx].values  # (T, N)

        # OLS: β = cov(ri, rm) / var(rm)
        rm_dm = rm - rm.mean()
        var_rm = (rm_dm ** 2).mean()
        if var_rm < 1e-12:
            continue

        beta = (ri - ri.mean(axis=0)) * rm_dm[:, None]
        beta = beta.mean(axis=0) / var_rm  # (N,)

        resid = ri - (ri.mean(axis=0) + beta[None, :] * rm_dm[:, None])
        ivol_t = resid.std(axis=0)  # (N,)

        ivol.loc[date] = ivol_t

    return _normalize(-ivol)  # 取负：低IVOL得高分


# ══════════════════════════════════════════════════════════════════════════════
# 融资余额变化率
# ══════════════════════════════════════════════════════════════════════════════

def factor_margin_change(
    margin: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """
    融资余额变化率：过去N日融资余额增速，正向因子。

    margin: DataFrame(index=日期, columns=股票), 值=融资余额（元）
    融资余额快速增加 → 杠杆资金持续流入 → 短期动量强化信号。
    注意：这是"情绪加速器"型因子，牛市有效，熊市可能反向。

    数据来源：data/raw/margin_balance.parquet（需先运行 download_margin.py）
    """
    # 20日变化率
    margin_chg = margin.pct_change(window)
    margin_chg = margin_chg.replace([np.inf, -np.inf], np.nan)
    return _normalize(margin_chg)


# ══════════════════════════════════════════════════════════════════════════════
# 东财大单净流入
# ══════════════════════════════════════════════════════════════════════════════

def factor_moneyflow_large(
    moneyflow: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """
    大单净流入因子：过去N日大单净流入额均值（超大单+大单）。

    moneyflow: DataFrame(index=日期, columns=股票), 值=大单净流入额（元）
    持续大单净流入 → 主力建仓行为 → 正向预测短中期收益。
    注：东财数据大单阈值≥50万，超大单≥100万。

    数据来源：data/raw/moneyflow_large.parquet
    """
    avg_flow = moneyflow.rolling(window).mean()
    # 用成交额做标准化，消除股票间量级差异
    return _normalize(avg_flow)


# ══════════════════════════════════════════════════════════════════════════════
# 北向资金持股变化
# ══════════════════════════════════════════════════════════════════════════════

def factor_northbound_change(
    northbound: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """
    北向资金持股变化因子：过去N日北向资金持股量变化率。

    northbound: DataFrame(index=日期, columns=股票), 值=持股量（股）
    北向资金（沪深股通）代表外资聪明钱，持续增持是正向信号。
    窗口20日可以避免短期噪声，捕捉趋势性增减持。

    数据来源：data/raw/northbound_holding.parquet
    """
    nb_chg = northbound.pct_change(window)
    nb_chg = nb_chg.replace([np.inf, -np.inf], np.nan)
    return _normalize(nb_chg)


# ══════════════════════════════════════════════════════════════════════════════
# 机构持仓季度变化
# ══════════════════════════════════════════════════════════════════════════════

def factor_institution_change(
    institution: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    机构持仓季度变化因子：基金重仓股的季度增持信号。

    institution: DataFrame(index=季报日期, columns=股票), 值=持股市值或比例
    机构增持往往伴随深度调研，是信息优势的代理指标。
    季报数据前向填充到日频（PIT 安全：报告期按法定披露窗口平移后再 ffill，
    消除用季报日做 ffill 起点的 look-ahead bias）。

    数据来源：data/raw/institution_holding.parquet
    """
    # 季度变化：当季持仓 - 上季持仓
    inst_chg = institution.diff(1)
    # PIT 安全：报告期按法定披露窗口平移后再 ffill 到日频
    inst_chg_daily = pit_reindex_ffill(inst_chg, prices.index)
    return _normalize(inst_chg_daily)


# ══════════════════════════════════════════════════════════════════════════════
# 加载辅助函数（从 parquet 文件读取额外数据）
# ══════════════════════════════════════════════════════════════════════════════

def load_margin() -> pd.DataFrame | None:
    p = RAW_DIR / "margin_balance.parquet"
    if not p.exists():
        logger.warning(f"融资余额文件不存在: {p}，请先运行 data/download_margin.py")
        return None
    return pd.read_parquet(p)


def load_moneyflow() -> pd.DataFrame | None:
    p = RAW_DIR / "moneyflow_large.parquet"
    if not p.exists():
        logger.warning(f"大单净流入文件不存在: {p}，请先运行 data/download_moneyflow.py")
        return None
    return pd.read_parquet(p)


def load_northbound() -> pd.DataFrame | None:
    p = RAW_DIR / "northbound_holding.parquet"
    if not p.exists():
        logger.warning(f"北向资金文件不存在: {p}，请先运行 data/download_northbound.py")
        return None
    return pd.read_parquet(p)


def load_institution() -> pd.DataFrame | None:
    p = RAW_DIR / "institution_holding.parquet"
    if not p.exists():
        logger.warning(f"机构持仓文件不存在: {p}，请先运行 data/download_institution.py")
        return None
    return pd.read_parquet(p)


def load_industry_map() -> pd.DataFrame | None:
    p = RAW_DIR / "industry_map.parquet"
    if not p.exists():
        logger.warning(f"行业映射文件不存在: {p}，请先运行 data/industry/download_industry.py")
        return None
    return pd.read_parquet(p)
