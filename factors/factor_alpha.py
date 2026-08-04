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
# PIT 行业面板工具
# ══════════════════════════════════════════════════════════════════════════════

def _pit_industry_wide(
    industry_panel: pd.DataFrame,
    dates: pd.Index,
    level: str = "sw_l2",
) -> pd.DataFrame:
    """
    把 PIT 行业长表 (effective_date, code, level) 透视为 (dates × code) 宽表，
    并按 effective_date reindex 到目标 dates 后 ffill，得到每个截面的当期行业。

    严格 PIT：日期 t 的行业 = effective_date <= t 中最新的记录，
    消除用全样本静态行业映射带来的未来信息。
    """
    panel = industry_panel.copy()
    if "effective_date" not in panel.columns and "start_date" in panel.columns:
        panel = panel.rename(columns={"start_date": "effective_date"})
    panel["effective_date"] = pd.to_datetime(panel["effective_date"], errors="coerce")
    panel = panel.dropna(subset=["effective_date"])
    if level not in panel.columns:
        raise ValueError(f"industry_panel 不含列 {level}，已有: {list(panel.columns)}")

    wide = panel.pivot_table(
        index="effective_date", columns="code", values=level, aggfunc="last"
    ).sort_index()
    # reindex 到目标日期并前向填充：每个截面取 effective_date <= t 的最新行业
    wide = wide.reindex(dates).ffill()
    return wide


def _pit_industry_demean(values: pd.DataFrame,
                         ind_wide: pd.DataFrame) -> pd.DataFrame:
    """
    PIT 截面行业去均值（vectorized via stack + groupby）。

    values:   (date, stock) 因子值
    ind_wide: (date, stock) 行业标签，由 _pit_industry_wide 产出
    对每个 (date, industry) 组减去该组当期均值。
    """
    common = values.columns.intersection(ind_wide.columns)
    if len(common) == 0:
        return values
    values = values[common]
    ind_wide = ind_wide.reindex(index=values.index, columns=common)

    v_long = values.stack()
    i_long = ind_wide.stack()
    df = pd.DataFrame({"v": v_long, "ind": i_long}).reset_index()
    date_col = df.columns[0]  # stack 后的第一级（日期）
    code_col = df.columns[1]  # 第二级（股票代码）
    df = df.dropna(subset=["v"])
    df["ind"] = df["ind"].fillna("未分类")
    grp = df.groupby([date_col, "ind"])["v"]
    df["v_dm"] = df["v"] - grp.transform("mean")
    out = df.pivot(index=date_col, columns=code_col, values="v_dm")
    return out.reindex(index=values.index, columns=values.columns)


def _pit_industry_demean_exclude_self(
    values: pd.DataFrame,
    ind_wide: pd.DataFrame,
    min_stocks: int = 5,
) -> pd.DataFrame:
    """
    PIT 截面行业去均值（排除自身）：对每个 (date, industry) 组，
    每个股票取 (组内和 - 自身) / (组内有效数 - 1)。

    用于行业动量因子，避免自相关偏差。
    """
    common = values.columns.intersection(ind_wide.columns)
    if len(common) == 0:
        return pd.DataFrame(np.nan, index=values.index, columns=values.columns)
    values = values[common]
    ind_wide = ind_wide.reindex(index=values.index, columns=common)

    v_long = values.stack()
    i_long = ind_wide.stack()
    df = pd.DataFrame({"v": v_long, "ind": i_long}).reset_index()
    date_col = df.columns[0]
    code_col = df.columns[1]
    df["ind"] = df["ind"].fillna("未分类")

    is_valid = df["v"].notna()
    valid = df.dropna(subset=["v"])
    grp = valid.groupby([date_col, "ind"])["v"]
    sum_df = grp.transform("sum")
    group_size = grp.transform("size")

    # 把组内统计对齐回完整 df（按行位置）
    sum_full = sum_df.set_axis(valid.index).reindex(df.index)
    size_full = group_size.set_axis(valid.index).reindex(df.index)
    # 排除自身：others_mean = (sum - self) / (count - 1)
    others_sum = sum_full - df["v"].where(is_valid, 0)
    others_cnt = (size_full - is_valid.astype(int)).replace({0: np.nan})
    others_mean = others_sum / others_cnt
    # 小组（< min_stocks）置 NaN
    others_mean = others_mean.where(size_full >= min_stocks, np.nan)

    out = df.assign(others_mean=others_mean).pivot(
        index=date_col, columns=code_col, values="others_mean"
    )
    return out.reindex(index=values.index, columns=values.columns)


# ══════════════════════════════════════════════════════════════════════════════
# 行业动量因子
# ══════════════════════════════════════════════════════════════════════════════

def factor_industry_momentum(
    prices: pd.DataFrame,
    industry_map: pd.DataFrame,
    window: int = 20,
    min_stocks: int = 5,
    clean_ret: pd.DataFrame = None,
    industry_panel: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    行业动量因子：每只股票过去N日所在行业的平均涨幅（排除自身）。

    行业轮动在A股显著，牛市中强势行业有动量延续。
    "排除自身"避免自相关偏差（IDiosyncratic）。

    industry_map: DataFrame，index=code，列含 'sw_l2'（申万二级行业，静态快照）
    clean_ret:    屏蔽涨跌停日后的日收益率；传入时用逐日复合，避免涨停日
                   return 截断污染行业均值；为 None 时退化为 pct_change(window)。
    industry_panel: PIT 行业长表 (effective_date, code, sw_l2, ...)，
                   传入时按截面日期取当期行业做 PIT 行业动量（消除未来信息）；
                   为 None 时回退静态 industry_map（向后兼容）。

    注意：行业动量为时序滚动因子，这里采用「以 t 期行业归属近似整个滚动窗口」
    的 PIT 近似（行业变更罕见，影响可控）。
    """
    # ── PIT 路径：industry_panel 优先 ──
    if industry_panel is not None:
        try:
            ind_wide = _pit_industry_wide(industry_panel, prices.index, level="sw_l2")
            if clean_ret is not None:
                ret = (1 + clean_ret).rolling(
                    window, min_periods=max(1, window // 2)
                ).apply(lambda x: np.nanprod(x) - 1, raw=True)
            else:
                ret = prices.pct_change(window)
            pit_avg = _pit_industry_demean_exclude_self(
                ret, ind_wide, min_stocks=min_stocks
            )
            # 补全非 panel 内股票
            full = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
            full[pit_avg.columns] = pit_avg
            logger.info("行业动量_20d: 启用 PIT 行业面板 (industry_map_panel)")
            return _normalize(full)
        except Exception as e:
            logger.warning(
                f"行业动量 PIT 路径失败，回退静态 industry_map: {e}"
            )

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

    if industry_panel is not None:
        logger.warning(
            "行业动量_20d: industry_panel 传入但走静态 fallback，PIT 行业未真正启用"
        )
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
    # 可靠提取单列市场收益：squeeze 在多列 DataFrame 上不降维，会破坏后续 OLS 广播
    if isinstance(market_prices, pd.DataFrame):
        if market_prices.shape[1] == 1:
            mkt_s = market_prices.iloc[:, 0]
        elif "close" in market_prices.columns:
            mkt_s = market_prices["close"]
        else:
            # 多列且无 close：取首列并告警（csi_all 等应含 close）
            logger.warning(
                f"factor_idiosyncratic_vol: market_prices 有 {market_prices.shape[1]} 列且无 close，"
                f"取首列 '{market_prices.columns[0]}' 作市场代理"
            )
            mkt_s = market_prices.iloc[:, 0]
    else:
        mkt_s = market_prices

    if clean_ret is not None:
        stock_ret = clean_ret
        # 市场收益用指数收益（指数本身不受个股涨跌停影响）
        mkt_ret = mkt_s.pct_change()
    else:
        stock_ret = prices.pct_change()
        mkt_ret = mkt_s.pct_change()

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
    amount: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    大单净流入因子：过去N日大单净流入额均值（超大单+大单）。

    moneyflow: DataFrame(index=日期, columns=股票), 值=大单净流入额（元）
    amount:    DataFrame(index=日期, columns=股票), 值=成交额（元），可选；
               传入时按成交额标准化（大单净流入 / 成交额），消除股票间量级差异
               与规模效应；未传时回退到原始金额（会混入规模效应）。
    持续大单净流入 → 主力建仓行为 → 正向预测短中期收益。
    注：东财数据大单阈值≥50万，超大单≥100万。

    数据来源：data/raw/moneyflow_large.parquet
    """
    if amount is not None:
        # 按成交额标准化：大单净流入占比，消除量级与规模效应
        ratio = moneyflow / amount.replace(0, np.nan)
        avg_flow = ratio.rolling(window).mean()
    else:
        avg_flow = moneyflow.rolling(window).mean()
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

    ⚠️ 北向披露约 2024-08-19 停更；默认管线不注册本因子。若显式调用，
    停更日后截面会被置 NaN，避免把空档当成信号。

    northbound: DataFrame(index=日期, columns=股票), 值=持股量（股）
    数据来源：data/raw/northbound_holding.parquet
    """
    from data.download_northbound import NORTHBOUND_DISCLOSURE_STOP
    logger.warning(
        f"factor_northbound_change: 北向已停更（≥{NORTHBOUND_DISCLOSURE_STOP.date()} 置 NaN）；"
        "默认白名单不含本因子"
    )
    nb_chg = northbound.pct_change(window)
    nb_chg = nb_chg.replace([np.inf, -np.inf], np.nan)
    # 停更日后不可用
    stop = NORTHBOUND_DISCLOSURE_STOP
    if len(nb_chg.index) and nb_chg.index.max() >= stop:
        nb_chg.loc[nb_chg.index >= stop] = np.nan
    return _normalize(nb_chg)


def factor_northbound_flow(
    northbound_value: pd.DataFrame,
    short: int = 20,
    long: int = 60,
) -> pd.DataFrame:
    """
    北向资金净流入趋势因子：近 short 日北向持股市值日均净流入
    相对近 long 日均值的差值（净流入加速信号）。

    ⚠️ 北向披露约 2024-08-19 停更；默认管线不注册。停更日后置 NaN。

    northbound_value: DataFrame(index=日期, columns=股票), 值=北向持股市值（元）
                      宽表，由 data/download_northbound.py 落地
                      (data/raw/northbound_value.parquet)。

    逻辑：
      - 日净流入 ≈ 持股市值的一阶差分 value[t] - value[t-1]
      - trend = short_MA - long_MA > 0 → 北向加速流入 → 正向

    数据来源：data/raw/northbound_value.parquet
    """
    from data.download_northbound import NORTHBOUND_DISCLOSURE_STOP
    logger.warning(
        f"factor_northbound_flow: 北向已停更（≥{NORTHBOUND_DISCLOSURE_STOP.date()} 置 NaN）；"
        "默认白名单不含本因子"
    )
    if northbound_value is None or northbound_value.empty:
        return None
    # 日净流入（近似）：持股市值一阶差分
    daily_flow = northbound_value.diff()
    short_ma = daily_flow.rolling(short, min_periods=max(2, short // 2)).mean()
    long_ma = daily_flow.rolling(long, min_periods=max(2, long // 2)).mean()
    trend = short_ma - long_ma
    trend = trend.replace([np.inf, -np.inf], np.nan)
    stop = NORTHBOUND_DISCLOSURE_STOP
    if len(trend.index) and trend.index.max() >= stop:
        trend.loc[trend.index >= stop] = np.nan
    return _normalize(trend)


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


def load_northbound_value() -> pd.DataFrame | None:
    """加载北向持股市值宽表 (date × code)，用于北向净流入因子。"""
    p = RAW_DIR / "northbound_value.parquet"
    if not p.exists():
        logger.warning(f"北向持股市值文件不存在: {p}，请先运行 data/download_northbound.py")
        return None
    return pd.read_parquet(p)


def load_industry_panel() -> pd.DataFrame | None:
    """加载 PIT 行业长表 industry_map_panel.parquet（effective_date, code, sw_l2...）。"""
    p = RAW_DIR / "industry_map_panel.parquet"
    if not p.exists():
        logger.warning(
            f"PIT 行业面板文件不存在: {p}，请先运行 data/industry/download_industry.py"
        )
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
