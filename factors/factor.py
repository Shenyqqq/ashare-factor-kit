"""
factors/compute.py  —  因子计算库

所有因子返回格式统一：
    DataFrame, index=日期, columns=股票代码, 值=因子得分（越高越好）
    已经过 winsorize(1%) + cross_sectional_zscore(clip=3σ)

因子分类：
    价格/动量类（只需价格数据）：
        factor_momentum         中期动量
        factor_reversal         短期反转
        factor_momentum_skip    跳过最近1月的中长期动量（12-1月动量）
        factor_volatility       特质波动率取反（低波动得高分）
        factor_turnover         换手率（需要成交量数据）
        factor_amihud           Amihud非流动性（需要成交额数据）
        factor_high_low         振幅因子

    财务类（需要financial数据）：
        factor_value_pb         价值：1/PB
        factor_value_ep         价值：EP（盈利/市值）
        factor_quality_roe      质量：ROE水平
        factor_quality_roe_chg  质量：ROE季度环比变化
        factor_quality_gpm      质量：毛利率
        factor_quality_gpm_chg  质量：毛利率变化
        factor_quality_accrual  质量：应计项目（越低越好）
        factor_size             规模：-log(总资产)
        factor_leverage         财务杠杆取反

注册新因子：在 factor_map 字典里加一行即可被 compute_composite_factor 和 notebook 识别。
"""
import pandas as pd
import numpy as np
from scipy.stats import zscore as scipy_zscore
from loguru import logger
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import RAW_DIR, PROCESSED_DIR


# ══════════════════════════════════════════════════════════════════════════════
# 预处理工具
# ══════════════════════════════════════════════════════════════════════════════

def cross_sectional_zscore(df: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """截面标准化：每个日期对所有股票做 z-score，裁剪 ±clip σ"""
    result = df.apply(lambda row: pd.Series(
        np.clip(scipy_zscore(row.dropna()), -clip, clip),
        index=row.dropna().index
    ), axis=1)
    return result


def winsorize(df: pd.DataFrame, pct: float = 0.01) -> pd.DataFrame:
    """极值缩尾：截面最高/最低 pct 分位数处截断"""
    def _win(row):
        lo = row.quantile(pct)
        hi = row.quantile(1 - pct)
        return row.clip(lo, hi)
    return df.apply(_win, axis=1)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """winsorize + cross_sectional_zscore 的快捷方式"""
    return cross_sectional_zscore(winsorize(df))


def _pivot_financial(financial: pd.DataFrame, col: str,
                     prices: pd.DataFrame) -> pd.DataFrame:
    """将长表财务数据透视并前向填充到日频"""
    pivot = financial.pivot_table(index="trade_date", columns="code", values=col)
    return pivot.reindex(prices.index, method="ffill")


# ══════════════════════════════════════════════════════════════════════════════
# 价格/动量类因子（只需 prices_hfq）
# ══════════════════════════════════════════════════════════════════════════════

def factor_momentum(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    中期动量：过去 N 日涨幅。
    A股20-60日动量有正向预测力。
    """
    return _normalize(prices.pct_change(window))


def factor_reversal(prices: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    短期反转：过去 N 日涨幅取负值。
    A股5日反转效应显著，涨多了买入信号弱。
    """
    return _normalize(-prices.pct_change(window))


def factor_momentum_skip(prices: pd.DataFrame,
                         long_window: int = 240,
                         skip_window: int = 20) -> pd.DataFrame:
    """
    跳过最近1月的中长期动量（12-1月动量）。
    跳过最近1月是为了避免短期反转效应污染动量信号。
    long_window=240（约12月），skip_window=20（约1月）。
    """
    ret_long = prices.pct_change(long_window)
    ret_skip = prices.pct_change(skip_window)
    # 12月收益 / (1 + 1月收益) - 1，近似去掉最近1月
    mom_skip = (1 + ret_long) / (1 + ret_skip) - 1
    return _normalize(mom_skip)


def factor_volatility(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    特质波动率取反（低波动得高分）。
    用日收益率的滚动标准差近似特质波动率。
    A股中高波动率股票未来收益偏低（与美股相反）。
    """
    daily_ret = prices.pct_change()
    vol = daily_ret.rolling(window).std()
    return _normalize(-vol)  # 取负：低波动得高分


def factor_turnover(volume: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    换手率因子：过去N日平均换手率取反（低换手得高分）。
    低换手率说明筹码稳定，机构持仓比例高。
    需要传入 volume（成交量）数据。
    注：严格换手率 = 成交量/流通股本，这里用成交量均值近似排名。
    """
    avg_vol = volume.rolling(window).mean()
    return _normalize(-avg_vol)  # 取负：低换手得高分


def factor_amihud(prices: pd.DataFrame,
                  amount: pd.DataFrame,
                  window: int = 20) -> pd.DataFrame:
    """
    Amihud 非流动性因子（越大说明越难交易，流动性溢价越高）。
    ILLIQ = mean(|日收益率| / 日成交额)，窗口内均值。
    A股中小票流动性溢价真实存在，该因子正向预测收益。
    需要传入 amount（成交额，元）数据。
    """
    daily_ret = prices.pct_change().abs()
    amount_m = amount.replace(0, np.nan)  # 避免除以0
    illiq = (daily_ret / amount_m).rolling(window).mean()
    return _normalize(illiq)


def factor_high_low(prices: pd.DataFrame,
                    high: pd.DataFrame,
                    low: pd.DataFrame,
                    window: int = 20) -> pd.DataFrame:
    """
    振幅因子取反：过去N日 (high-low)/close 均值取反。
    低振幅说明价格走势平稳，通常对应机构持仓。
    需要传入 high、low 数据。
    """
    hl_ratio = (high - low) / prices.replace(0, np.nan)
    avg_hl = hl_ratio.rolling(window).mean()
    return _normalize(-avg_hl)


def factor_price_to_high(prices: pd.DataFrame, window: int = 52) -> pd.DataFrame:
    """
    价格相对N日高点的位置（52周新高因子）。
    价格接近历史高点往往意味着强势，正向预测动量延续。
    window=52时约等于1年。
    """
    roll_high = prices.rolling(window).max()
    ratio = prices / roll_high.replace(0, np.nan)
    return _normalize(ratio)


# ══════════════════════════════════════════════════════════════════════════════
# 财务类因子（需要 financial 数据）
# ══════════════════════════════════════════════════════════════════════════════

def factor_value_pb(financial: pd.DataFrame,
                    prices_raw: pd.DataFrame) -> pd.DataFrame:
    """
    价值因子：1/PB（低PB得高分）。
    PB = 不复权股价 / 每股净资产，1/PB = bvps / price。
    必须用不复权价格，因为bvps是绝对值。
    """
    bvps = _pivot_financial(financial, "bvps", prices_raw)
    price = prices_raw.reindex(columns=bvps.columns)
    inv_pb = bvps / price.replace(0, np.nan)
    inv_pb = inv_pb.replace([np.inf, -np.inf], np.nan)
    return _normalize(inv_pb)


def factor_value_ep(financial: pd.DataFrame,
                    prices_raw: pd.DataFrame) -> pd.DataFrame:
    """
    价值因子：EP = 每股收益 / 股价（盈利收益率）。
    用 roe * bvps 近似每股收益（EPS = ROE × 每股净资产）。
    EP 比 PB 更直接反映盈利能力，与未来收益相关性更强。
    """
    roe = _pivot_financial(financial, "roe", prices_raw) / 100  # 转为小数
    bvps = _pivot_financial(financial, "bvps", prices_raw)
    price = prices_raw.reindex(columns=roe.columns)

    eps = roe * bvps  # 近似EPS
    ep = eps / price.replace(0, np.nan)
    ep = ep.replace([np.inf, -np.inf], np.nan)
    return _normalize(ep)


def factor_quality_roe(financial: pd.DataFrame,
                       prices: pd.DataFrame) -> pd.DataFrame:
    """
    质量因子：ROE 水平（净资产收益率）。
    季报数据前向填充到日频。
    """
    roe = _pivot_financial(financial, "roe", prices)
    return _normalize(roe)


def factor_quality_roe_chg(financial: pd.DataFrame,
                            prices: pd.DataFrame) -> pd.DataFrame:
    """
    质量因子：ROE 季度环比变化（ROE改善信号）。
    ROE在提升的公司，往往伴随盈利能力改善，预示未来超额收益。
    用季报数据做差分，前向填充到日频。
    """
    roe_pivot = financial.pivot_table(
        index="trade_date", columns="code", values="roe"
    )
    roe_chg = roe_pivot.diff(1)  # 季度环比变化
    roe_chg = roe_chg.reindex(prices.index, method="ffill")
    return _normalize(roe_chg)


def factor_quality_gpm(financial: pd.DataFrame,
                        prices: pd.DataFrame) -> pd.DataFrame:
    """
    质量因子：毛利率水平。
    高毛利率说明产品竞争力强、护城河宽。
    需要财务数据中有 gross_profit_margin 列，若无则跳过。
    """
    if "gross_profit_margin" not in financial.columns:
        logger.warning("财务数据中无 gross_profit_margin 列，factor_quality_gpm 跳过")
        return None
    gpm = _pivot_financial(financial, "gross_profit_margin", prices)
    return _normalize(gpm)


def factor_quality_gpm_chg(financial: pd.DataFrame,
                             prices: pd.DataFrame) -> pd.DataFrame:
    """
    质量因子：毛利率季度环比变化。
    毛利率改善是上游成本下降或产品提价的信号。
    """
    if "gross_profit_margin" not in financial.columns:
        logger.warning("财务数据中无 gross_profit_margin 列，factor_quality_gpm_chg 跳过")
        return None
    gpm_pivot = financial.pivot_table(
        index="trade_date", columns="code", values="gross_profit_margin"
    )
    gpm_chg = gpm_pivot.diff(1)
    gpm_chg = gpm_chg.reindex(prices.index, method="ffill")
    return _normalize(gpm_chg)


def factor_quality_accrual(financial: pd.DataFrame,
                            prices: pd.DataFrame) -> pd.DataFrame:
    """
    质量因子：应计项目取反（越低越好）。
    应计项目 = 净利润 - 经营现金流（用总资产标准化）。
    高应计意味着利润质量差（更多来自会计调整而非真实现金）。
    需要财务数据中有 net_profit 和 operating_cashflow 列。
    """
    required = {"net_profit", "operating_cashflow", "total_assets"}
    missing = required - set(financial.columns)
    if missing:
        logger.warning(f"财务数据缺少列 {missing}，factor_quality_accrual 跳过")
        return None

    np_pivot  = _pivot_financial(financial, "net_profit", prices)
    ocf_pivot = _pivot_financial(financial, "operating_cashflow", prices)
    ta_pivot  = _pivot_financial(financial, "total_assets", prices)

    accrual = (np_pivot - ocf_pivot) / ta_pivot.replace(0, np.nan)
    accrual = accrual.replace([np.inf, -np.inf], np.nan)
    return _normalize(-accrual)  # 取负：应计越低得分越高


def factor_size(financial: pd.DataFrame,
                prices: pd.DataFrame) -> pd.DataFrame:
    """
    规模因子：-log(总资产)，小规模得高分。
    小市值效应在A股长期存在（尤其是小票）。
    """
    assets = _pivot_financial(financial, "total_assets", prices)
    neg_log = -np.log(assets.replace(0, np.nan))
    return _normalize(neg_log)


def factor_leverage(financial: pd.DataFrame,
                    prices: pd.DataFrame) -> pd.DataFrame:
    """
    财务杠杆取反：低杠杆得高分。
    杠杆 = 总资产 / 净资产 = 1 / (1 - 资产负债率)。
    用 total_assets / (bvps * 股本) 近似，或直接用总资产/净资产比。
    高杠杆公司在经济下行期风险更大。
    近似实现：用总资产/净资产（bvps）之比。
    """
    assets = _pivot_financial(financial, "total_assets", prices)
    bvps   = _pivot_financial(financial, "bvps", prices)

    # 净资产 ≈ bvps（每股净资产），量纲不同但截面排名有效
    leverage = assets / bvps.replace(0, np.nan).abs()
    leverage = leverage.replace([np.inf, -np.inf], np.nan)
    return _normalize(-leverage)  # 取负：低杠杆得高分


# ══════════════════════════════════════════════════════════════════════════════
# 因子注册表（供 notebook 和 run() 使用）
# ══════════════════════════════════════════════════════════════════════════════

def get_factor_registry(
    prices: pd.DataFrame,
    financial: pd.DataFrame,
    prices_raw: pd.DataFrame = None,
    volume: pd.DataFrame = None,
    amount: pd.DataFrame = None,
) -> dict:
    """
    返回所有可计算因子的字典 {因子名: DataFrame}。
    根据传入的数据自动跳过缺少数据源的因子。

    prices:     后复权收盘价（必须）
    financial:  财务季报数据（可选，无则跳过财务因子）
    prices_raw: 不复权收盘价（可选，无则跳过PB/EP因子）
    volume:     成交量（可选，无则跳过换手率因子）
    amount:     成交额（可选，无则跳过Amihud因子）
    """
    registry = {}

    # ── 价格因子（始终可算）──
    registry["动量_20d"]    = factor_momentum(prices, 20)
    registry["动量_60d"]    = factor_momentum(prices, 60)
    registry["动量_120d"]   = factor_momentum(prices, 120)
    registry["动量_skip"]   = factor_momentum_skip(prices, 240, 20)
    registry["反转_5d"]     = factor_reversal(prices, 5)
    registry["反转_20d"]    = factor_reversal(prices, 20)
    registry["波动率_20d"]  = factor_volatility(prices, 20)
    registry["波动率_60d"]  = factor_volatility(prices, 60)
    registry["52周新高"]    = factor_price_to_high(prices, 52 * 5)

    # ── 需要成交量 ──
    if volume is not None:
        registry["换手率_20d"] = factor_turnover(volume, 20)
    else:
        logger.debug("未传入 volume，跳过换手率因子")

    # ── 需要成交额 ──
    if amount is not None:
        registry["Amihud_20d"] = factor_amihud(prices, amount, 20)
    else:
        logger.debug("未传入 amount，跳过 Amihud 因子")

    # ── 需要财务数据 ──
    if financial is not None and not financial.empty:
        _pr = prices_raw if prices_raw is not None else prices

        registry["价值_PB"]      = factor_value_pb(financial, _pr)
        registry["价值_EP"]      = factor_value_ep(financial, _pr)
        registry["质量_ROE"]     = factor_quality_roe(financial, prices)
        registry["质量_ROE变化"] = factor_quality_roe_chg(financial, prices)
        registry["规模"]         = factor_size(financial, prices)
        registry["杠杆"]         = factor_leverage(financial, prices)

        # 毛利率和应计项目：数据列可能不存在，函数内部会处理
        gpm     = factor_quality_gpm(financial, prices)
        gpm_chg = factor_quality_gpm_chg(financial, prices)
        accrual = factor_quality_accrual(financial, prices)
        if gpm     is not None: registry["质量_毛利率"]     = gpm
        if gpm_chg is not None: registry["质量_毛利率变化"] = gpm_chg
        if accrual is not None: registry["质量_应计项目"]   = accrual
    else:
        logger.debug("未传入 financial，跳过所有财务因子")

    # 过滤掉计算失败（返回None）的因子
    registry = {k: v for k, v in registry.items() if v is not None}
    logger.info(f"因子库就绪: {len(registry)} 个因子 → {list(registry.keys())}")
    return registry


# ══════════════════════════════════════════════════════════════════════════════
# 合成多因子（线性加权，作为 ML 模型的基准）
# ══════════════════════════════════════════════════════════════════════════════

def compute_composite_factor(
    prices: pd.DataFrame,
    financial: pd.DataFrame,
    weights: dict,
    prices_raw: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    线性加权合成因子，作为 ML 模型的对照基准。
    weights 的 key 对应 get_factor_registry() 返回的因子名。
    权重正负编码因子方向（已在各因子函数里取反处理，此处正权重=越高越好）。
    """
    registry = get_factor_registry(prices, financial, prices_raw)

    composite = None
    for name, w in weights.items():
        if name not in registry:
            logger.warning(f"未知因子: {name}，跳过")
            continue
        logger.info(f"合成因子: {name} (权重={w})")
        f = registry[name] * w
        composite = f if composite is None else composite.add(f, fill_value=0)

    return composite


# ══════════════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════════════

def run():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("载入价格数据...")
    prices = pd.read_parquet(RAW_DIR / "prices_hfq.parquet")

    logger.info("载入财务数据...")
    financial = pd.read_parquet(RAW_DIR / "financial_indicators.parquet")

    try:
        prices_raw = pd.read_parquet(RAW_DIR / "prices_raw.parquet")
    except FileNotFoundError:
        prices_raw = None
        logger.warning("未找到不复权价格，PB/EP因子将跳过")

    from config.settings import FACTOR_WEIGHTS
    composite = compute_composite_factor(prices, financial, FACTOR_WEIGHTS, prices_raw)

    if composite is not None:
        out = PROCESSED_DIR / "composite_factor.parquet"
        composite.to_parquet(out)
        logger.info(f"综合因子保存至 {out}，shape={composite.shape}")


if __name__ == "__main__":
    run()
