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
from utils.pit_align import pit_pivot_ffill, pit_reindex_ffill


# ══════════════════════════════════════════════════════════════════════════════
# 预处理工具
# ══════════════════════════════════════════════════════════════════════════════

def cross_sectional_zscore(df: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """截面标准化：每个日期对所有股票做 z-score，裁剪 ±clip σ"""
    result = df.apply(lambda row: pd.Series(
        np.clip(scipy_zscore(row.dropna()), -clip, clip),
        index=row.dropna().index
    ), axis=1)
    return result.astype(np.float32)


def winsorize(df: pd.DataFrame, pct: float = 0.01) -> pd.DataFrame:
    """极值缩尾：截面最高/最低 pct 分位数处截断"""
    def _win(row):
        lo = row.quantile(pct)
        hi = row.quantile(1 - pct)
        return row.clip(lo, hi)
    return df.apply(_win, axis=1).astype(np.float32)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """winsorize + cross_sectional_zscore 的快捷方式"""
    return cross_sectional_zscore(winsorize(df))


def _pivot_financial(financial: pd.DataFrame, col: str,
                     prices: pd.DataFrame) -> pd.DataFrame:
    """
    将长表财务数据透视并前向填充到日频（PIT 安全）。

    把报告期 trade_date 按 A 股法定披露窗口（Q1/Q3 +45 天，半年报 +75 天，
    年报 +120 天）平移到「可用日下界」后再 pivot + ffill，
    消除用报告期日做 ffill 起点的 look-ahead bias。
    详见 utils/pit_align.py。
    """
    return pit_pivot_ffill(
        financial, prices.index, date_col="trade_date", value_cols=[col],
    )


# ══════════════════════════════════════════════════════════════════════════════
# 价格/动量类因子（只需 prices_hfq）
# ══════════════════════════════════════════════════════════════════════════════

def factor_momentum(prices: pd.DataFrame, window: int = 20,
                    clean_ret: pd.DataFrame = None) -> pd.DataFrame:
    """
    中期动量：过去 N 日复合收益率。
    用 clean_ret（屏蔽涨跌停日）滚动复合，避免涨停日 return 截断导致动量低估。
    若 clean_ret 为 None，退化为 pct_change(window)。
    """
    if clean_ret is not None:
        # 用 clean_ret 逐日复合，NaN 日透明跳过
        mom = (1 + clean_ret).rolling(window, min_periods=max(1, window // 2)).apply(
            lambda x: np.nanprod(x) - 1, raw=True
        )
    else:
        mom = prices.pct_change(window)
    return _normalize(mom)


def factor_reversal(prices: pd.DataFrame, window: int = 5,
                    clean_ret: pd.DataFrame = None) -> pd.DataFrame:
    """
    短期反转：过去 N 日复合收益率取负。
    A股5日反转效应显著。使用 clean_ret 屏蔽涨跌停日。
    """
    if clean_ret is not None:
        mom = (1 + clean_ret).rolling(window, min_periods=max(1, window // 2)).apply(
            lambda x: np.nanprod(x) - 1, raw=True
        )
    else:
        mom = prices.pct_change(window)
    return _normalize(-mom)


def factor_momentum_skip(prices: pd.DataFrame,
                         long_window: int = 240,
                         skip_window: int = 20,
                         clean_ret: pd.DataFrame = None) -> pd.DataFrame:
    """
    跳过最近1月的中长期动量（12-1月动量）。
    long_window=240（约12月），skip_window=20（约1月）。
    """
    if clean_ret is not None:
        _roll = lambda w: (1 + clean_ret).rolling(w, min_periods=w // 2).apply(
            lambda x: np.nanprod(x) - 1, raw=True
        )
        ret_long = _roll(long_window)
        ret_skip = _roll(skip_window)
    else:
        ret_long = prices.pct_change(long_window)
        ret_skip = prices.pct_change(skip_window)
    mom_skip = (1 + ret_long) / (1 + ret_skip) - 1
    return _normalize(mom_skip)


def factor_volatility(prices: pd.DataFrame, window: int = 20,
                      clean_ret: pd.DataFrame = None) -> pd.DataFrame:
    """
    波动率取反（低波动得高分）。
    必须用 clean_ret：涨跌停日 return 被强制截断，若不屏蔽会系统性低估波动率。
    """
    ret = clean_ret if clean_ret is not None else prices.pct_change()
    vol = ret.rolling(window, min_periods=window // 2).std()
    return _normalize(-vol)


def factor_turnover(volume: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    换手率因子取反（低换手得高分）。
    用成交量均值近似排名（无需 clean_ret，成交量在涨停日本身是信号）。
    """
    avg_vol = volume.rolling(window).mean()
    return _normalize(-avg_vol)


def factor_amihud(prices: pd.DataFrame,
                  amount: pd.DataFrame,
                  window: int = 20,
                  clean_ret: pd.DataFrame = None) -> pd.DataFrame:
    """
    Amihud 非流动性因子：ILLIQ = mean(|日收益率| / 日成交额)。
    涨跌停日成交额失真（买不到/卖不出），需用 clean_ret 屏蔽对应日。
    """
    ret = clean_ret if clean_ret is not None else prices.pct_change()
    amount_m = amount.replace(0, np.nan)
    illiq = (ret.abs() / amount_m).rolling(window, min_periods=window // 2).mean()
    return _normalize(illiq)


def factor_high_low(prices: pd.DataFrame,
                    high: pd.DataFrame,
                    low: pd.DataFrame,
                    window: int = 20,
                    masks: dict = None) -> pd.DataFrame:
    """
    振幅因子取反：过去N日 (high-low)/close 均值取反。
    一字板日（高==低==开==收）振幅为 0，会压低均值，需屏蔽。
    """
    hl_ratio = (high - low) / prices.replace(0, np.nan)
    if masks is not None:
        # 一字涨停/跌停日振幅为0，不具参考价值
        hl_ratio[masks["limit_up_open"] | masks["limit_down_open"]] = np.nan
    avg_hl = hl_ratio.rolling(window, min_periods=window // 2).mean()
    return _normalize(-avg_hl)


def factor_price_to_high(prices: pd.DataFrame, window: int = 52) -> pd.DataFrame:
    """
    52周新高因子：价格相对 N 日最高价的位置。
    不涉及 return 计算，无需 clean_ret。
    """
    roll_high = prices.rolling(window).max()
    ratio = prices / roll_high.replace(0, np.nan)
    return _normalize(ratio)


def factor_frac_diff_momentum(prices: pd.DataFrame, d: float = 0.4,
                              window: int = 20,
                              threshold: float = 1e-5) -> pd.DataFrame:
    """
    分数差分动量因子（AFML Ch5）。

    用 d∈(0,1) 部分差分价格序列，保留长期记忆同时使序列平稳，
    再取过去 window 日的差分值变化（动量信号）做截面 z-score。

    相比 pct_change（d=1 完全差分，丢失长期记忆），d=0.4 的分数差分
    能保留 AI 主题等股票的长期趋势信息，对动量/反转因子尤其重要。

    参数
    ----
    prices    : 后复权收盘价宽表
    d         : 分数差分阶数，默认 0.4（AFML 实证常用 0.3-0.5）
    window    : 动量窗口，默认 20 日；对差分后序列取 window 日变化
    threshold : 权重截断阈值，|w_k| < threshold 时停止扩展权重

    输出：截面 z-score 化的分数差分动量（越高越好）
    """
    from utils.fractional_diff import frac_diff_ffd

    fd = frac_diff_ffd(prices, d=d, threshold=threshold)
    # 取过去 window 日的分数差分变化作为动量信号
    # fd 已是平稳化的"价格水平"信号，window 日变化 = fd - fd.shift(window)
    mom = fd - fd.shift(window)
    return _normalize(mom)


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
    用季报数据做差分，前向填充到日频（PIT 安全：报告期按法定披露窗口
    平移后再 ffill）。
    """
    roe_pivot = financial.pivot_table(
        index="trade_date", columns="code", values="roe"
    )
    roe_pivot.index = pd.to_datetime(roe_pivot.index)
    roe_chg = roe_pivot.diff(1)  # 季度环比变化
    roe_chg = pit_reindex_ffill(roe_chg, prices.index)
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
    PIT 安全：报告期按法定披露窗口平移后再 ffill。
    """
    if "gross_profit_margin" not in financial.columns:
        logger.warning("财务数据中无 gross_profit_margin 列，factor_quality_gpm_chg 跳过")
        return None
    gpm_pivot = financial.pivot_table(
        index="trade_date", columns="code", values="gross_profit_margin"
    )
    gpm_pivot.index = pd.to_datetime(gpm_pivot.index)
    gpm_chg = gpm_pivot.diff(1)
    gpm_chg = pit_reindex_ffill(gpm_chg, prices.index)
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

def _fit_hmm_regime(idx_ret: pd.Series, n_states: int = 3) -> pd.DataFrame:
    """
    用 GaussianHMM 从指数日收益率序列识别市场隐藏状态，返回各状态的后验概率。

    n_states=3: 强势/震荡/弱势三状态，比二状态更能区分 A 股牛快熊慢的特征。
    状态按均值收益率排序后重新标记: 强势=idx 0, 震荡=idx 1, 弱势=idx 2。

    输出列: ['HMM_强势概率', 'HMM_弱势概率']（丢弃震荡，三列加和=1 线性相关）
    """
    from hmmlearn import hmm as _hmm

    returns = idx_ret.fillna(0).values.reshape(-1, 1)

    model = _hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        random_state=42,
    )
    model.fit(returns)

    # 后验概率矩阵，shape=(T, n_states)
    probs = model.predict_proba(returns)  # (T, 3)

    # 按各状态均值收益率排序：最高均值=强势(0)，最低=弱势(2)
    means = model.means_.flatten()
    order = np.argsort(means)[::-1]   # 降序：强势 → 震荡 → 弱势
    probs = probs[:, order]

    df = pd.DataFrame(probs, index=idx_ret.index,
                      columns=["HMM_强势概率", "HMM_震荡概率", "HMM_弱势概率"])
    return df[["HMM_强势概率", "HMM_弱势概率"]]  # 丢弃震荡（线性相关）


def _market_regime_features(market_prices: pd.DataFrame,
                             prices: pd.DataFrame) -> dict:
    """
    市场状态特征：截面内所有股票相同值，帮助树模型感知宏观 regime。

    不做截面 zscore（截面内同值 → zscore=0），改用时序滚动标准化。
    shift(1) 避免前视偏差（今天的特征用昨天的市场数据）。

    特征列表（共7个）：
      - 市场动量_20d/60d/120d  : 多周期趋势，时序zscore
      - 市场波动率_20d         : 恐慌/风险偏好信号，时序zscore
      - 市场MA偏离_60d         : 连续版牛熊程度，时序zscore
      - HMM_强势概率           : GaussianHMM 3态强势状态后验概率（数据驱动）
      - HMM_弱势概率           : GaussianHMM 3态弱势状态后验概率
    """
    idx = market_prices.iloc[:, 0].reindex(prices.index, method="ffill")
    idx_ret = idx.pct_change()

    def _ts_norm(s: pd.Series, window: int = 252) -> pd.Series:
        mu  = s.rolling(window, min_periods=60).mean()
        sig = s.rolling(window, min_periods=60).std().replace(0, np.nan)
        return ((s - mu) / sig).clip(-3, 3)

    def _broadcast(s: pd.Series) -> pd.DataFrame:
        arr = s.shift(1).reindex(prices.index).values.reshape(-1, 1)
        return pd.DataFrame(
            np.repeat(arr, len(prices.columns), axis=1),
            index=prices.index, columns=prices.columns
        )

    features = {}

    # 连续型技术特征
    for w in [20, 60, 120]:
        features[f"市场动量_{w}d"] = _broadcast(_ts_norm(idx.pct_change(w)))
    features["市场波动率_20d"] = _broadcast(_ts_norm(idx_ret.rolling(20).std()))
    ma60 = idx.rolling(60).mean()
    features["市场MA偏离_60d"] = _broadcast(_ts_norm(idx / ma60.replace(0, np.nan) - 1))

    # HMM 数据驱动状态概率（替代均线规则的离散状态）
    try:
        hmm_probs = _fit_hmm_regime(idx_ret)
        for col in hmm_probs.columns:
            features[col] = _broadcast(hmm_probs[col])
        logger.info(f"HMM 市场状态拟合完成，样本={len(idx_ret)}条")
    except Exception as e:
        logger.warning(f"HMM 拟合失败，跳过: {e}")

    return features


def _is_regime_factor(name: str) -> bool:
    return name.startswith("市场") or name.startswith("HMM_")


def _want_factor(name: str, factor_names: set | None) -> bool:
    if factor_names is None:
        return True
    return name in factor_names or _is_regime_factor(name)


def _section_needed(candidates: frozenset, factor_names: set | None) -> bool:
    if factor_names is None:
        return True
    return bool(factor_names & candidates)


_PRICE_FACTOR_NAMES = frozenset({
    "动量_3d", "动量_10d", "动量_20d", "动量_40d", "动量_60d", "动量_120d", "动量_skip",
    "反转_3d", "反转_5d", "反转_10d", "反转_20d",
    "波动率_20d", "波动率_60d", "52周新高", "换手率_20d", "Amihud_20d", "振幅_20d",
    "分数差分动量_20d",
})
_FIN_FACTOR_NAMES = frozenset({
    "价值_PB", "价值_EP", "质量_ROE", "质量_ROE变化", "规模", "杠杆",
    "质量_毛利率", "质量_毛利率变化", "质量_应计项目",
})
_ALPHA2_FACTOR_NAMES = frozenset({
    "行业动量_20d", "特质波动率_60d", "融资余额变化_20d",
    "大单净流入_5d", "北向持股变化_20d", "机构持仓变化",
})
_TECH_FACTOR_NAMES = frozenset({
    "BIAS_5d", "BIAS_20d", "PSY_12d", "AR_26d", "BR_26d",
    "换手率加速度", "换手率行业中性_20d", "行业相对强度_20d",
})
_LIMIT_FACTOR_NAMES = frozenset({
    "涨停强度_20d", "跌停弱势_20d", "连板数", "涨跌停净强度_20d", "开板反转_5d",
})


def get_factor_registry(
    prices: pd.DataFrame,
    financial: pd.DataFrame,
    prices_raw: pd.DataFrame = None,
    volume: pd.DataFrame = None,
    amount: pd.DataFrame = None,
    open_: pd.DataFrame = None,
    high: pd.DataFrame = None,
    low: pd.DataFrame = None,
    clean_ret: pd.DataFrame = None,
    masks: dict = None,
    market_prices: pd.DataFrame = None,
    industry_map: pd.DataFrame = None,
    margin: pd.DataFrame = None,
    moneyflow: pd.DataFrame = None,
    northbound: pd.DataFrame = None,
    institution: pd.DataFrame = None,
    factor_names: list | set | None = None,
) -> dict:
    """
    返回所有可计算因子的字典 {因子名: DataFrame}。
    根据传入的数据自动跳过缺少数据源的因子。

    factor_names: 仅计算指定因子 + 市场/HMM regime 特征；None 表示全量计算。

    prices:     后复权收盘价（必须）
    financial:  财务季报数据（可选）
    prices_raw: 不复权收盘价（可选）
    volume:     成交量（可选）
    amount:     成交额（可选）
    open_/high/low: 后复权 OHLC（可选）
    clean_ret:  屏蔽涨跌停日后的日收益率，由 clean.clean_ohlcv() 生成（强烈建议传入）
    masks:      涨跌停 mask dict，由 clean.clean_ohlcv() 生成（强烈建议传入）
    market_prices/industry_map/margin/moneyflow/northbound/institution: 可选 Alpha 数据源
    """
    fn_set = set(factor_names) if factor_names is not None else None
    registry = {}

    # ── 价格/动量类（全部传入 clean_ret，有则用，无则自动回退）──
    if _section_needed(_PRICE_FACTOR_NAMES, fn_set):
        if _want_factor("动量_3d", fn_set):
            registry["动量_3d"] = factor_momentum(prices, 3, clean_ret)
        if _want_factor("动量_10d", fn_set):
            registry["动量_10d"] = factor_momentum(prices, 10, clean_ret)
        if _want_factor("动量_20d", fn_set):
            registry["动量_20d"] = factor_momentum(prices, 20, clean_ret)
        if _want_factor("动量_40d", fn_set):
            registry["动量_40d"] = factor_momentum(prices, 40, clean_ret)
        if _want_factor("动量_60d", fn_set):
            registry["动量_60d"] = factor_momentum(prices, 60, clean_ret)
        if _want_factor("动量_120d", fn_set):
            registry["动量_120d"] = factor_momentum(prices, 120, clean_ret)
        if _want_factor("动量_skip", fn_set):
            registry["动量_skip"] = factor_momentum_skip(prices, 240, 20, clean_ret)
        if _want_factor("反转_3d", fn_set):
            registry["反转_3d"] = factor_reversal(prices, 3, clean_ret)
        if _want_factor("反转_5d", fn_set):
            registry["反转_5d"] = factor_reversal(prices, 5, clean_ret)
        if _want_factor("反转_10d", fn_set):
            registry["反转_10d"] = factor_reversal(prices, 10, clean_ret)
        if _want_factor("反转_20d", fn_set):
            registry["反转_20d"] = factor_reversal(prices, 20, clean_ret)
        if _want_factor("波动率_20d", fn_set):
            registry["波动率_20d"] = factor_volatility(prices, 20, clean_ret)
        if _want_factor("波动率_60d", fn_set):
            registry["波动率_60d"] = factor_volatility(prices, 60, clean_ret)
        if _want_factor("52周新高", fn_set):
            registry["52周新高"] = factor_price_to_high(prices, 52 * 5)
        if _want_factor("分数差分动量_20d", fn_set):
            registry["分数差分动量_20d"] = factor_frac_diff_momentum(
                prices, d=0.4, window=20
            )
        if volume is not None and _want_factor("换手率_20d", fn_set):
            registry["换手率_20d"] = factor_turnover(volume, 20)
        if amount is not None and _want_factor("Amihud_20d", fn_set):
            registry["Amihud_20d"] = factor_amihud(prices, amount, 20, clean_ret)
        if high is not None and low is not None and _want_factor("振幅_20d", fn_set):
            registry["振幅_20d"] = factor_high_low(prices, high, low, 20, masks)

    # ── 财务类 ──
    if financial is not None and not financial.empty and _section_needed(_FIN_FACTOR_NAMES, fn_set):
        _pr = prices_raw if prices_raw is not None else prices
        if _want_factor("价值_PB", fn_set):
            registry["价值_PB"] = factor_value_pb(financial, _pr)
        if _want_factor("价值_EP", fn_set):
            registry["价值_EP"] = factor_value_ep(financial, _pr)
        if _want_factor("质量_ROE", fn_set):
            registry["质量_ROE"] = factor_quality_roe(financial, prices)
        if _want_factor("质量_ROE变化", fn_set):
            registry["质量_ROE变化"] = factor_quality_roe_chg(financial, prices)
        if _want_factor("规模", fn_set):
            registry["规模"] = factor_size(financial, prices)
        if _want_factor("杠杆", fn_set):
            registry["杠杆"] = factor_leverage(financial, prices)
        for name, f in [
            ("质量_毛利率", factor_quality_gpm(financial, prices)),
            ("质量_毛利率变化", factor_quality_gpm_chg(financial, prices)),
            ("质量_应计项目", factor_quality_accrual(financial, prices)),
        ]:
            if f is not None and _want_factor(name, fn_set):
                registry[name] = f

    # ── 第二批 Alpha 因子 ──
    if _section_needed(_ALPHA2_FACTOR_NAMES, fn_set):
        try:
            from factors.factor_alpha import (
                factor_industry_momentum, factor_idiosyncratic_vol,
                factor_margin_change, factor_moneyflow_large,
                factor_northbound_change, factor_institution_change,
            )
            if industry_map is not None and _want_factor("行业动量_20d", fn_set):
                f = factor_industry_momentum(prices, industry_map, window=20, clean_ret=clean_ret)
                if f is not None:
                    registry["行业动量_20d"] = f
            if market_prices is not None and _want_factor("特质波动率_60d", fn_set):
                registry["特质波动率_60d"] = factor_idiosyncratic_vol(
                    prices, market_prices, window=60, clean_ret=clean_ret)
            if margin is not None and _want_factor("融资余额变化_20d", fn_set):
                registry["融资余额变化_20d"] = factor_margin_change(margin, window=20)
            if moneyflow is not None and _want_factor("大单净流入_5d", fn_set):
                registry["大单净流入_5d"] = factor_moneyflow_large(moneyflow, window=5)
            if northbound is not None and _want_factor("北向持股变化_20d", fn_set):
                registry["北向持股变化_20d"] = factor_northbound_change(northbound, window=20)
            if institution is not None and _want_factor("机构持仓变化", fn_set):
                registry["机构持仓变化"] = factor_institution_change(institution, prices)
        except ImportError as e:
            logger.warning(f"factor_alpha 导入失败: {e}")

    # ── 市场状态特征（白名单模式始终计算 regime）──
    if market_prices is not None:
        try:
            mkt_features = _market_regime_features(market_prices, prices)
            if fn_set is not None:
                mkt_features = {k: v for k, v in mkt_features.items() if _want_factor(k, fn_set)}
            registry.update(mkt_features)
        except Exception as e:
            logger.warning(f"市场状态特征计算失败: {e}")

    # ── 技术分析类因子（BIAS/PSY/ARBR/换手率变体/行业相对强度）──
    if _section_needed(_TECH_FACTOR_NAMES, fn_set):
        try:
            from factors.factor_technical import get_technical_factors
            tech_factors = get_technical_factors(
                prices=prices, volume=volume, industry_map=industry_map,
                open_=open_, high=high, low=low, clean_ret=clean_ret,
            )
            if fn_set is not None:
                tech_factors = {k: v for k, v in tech_factors.items() if k in fn_set}
            registry.update(tech_factors)
        except Exception as e:
            logger.warning(f"技术因子计算失败: {e}")

    # ── WorldQuant Alpha101 精选因子 ──
    need_wq = fn_set is None or any(n.startswith("WQ_") for n in fn_set)
    if need_wq:
        try:
            from factors.factor_alpha101 import get_alpha101_factors
            wq_factors = get_alpha101_factors(
                prices=prices, open_=open_, high=high, low=low,
                volume=volume, amount=amount,
            )
            if fn_set is not None:
                wq_factors = {k: v for k, v in wq_factors.items() if k in fn_set}
            registry.update(wq_factors)
        except Exception as e:
            logger.warning(f"Alpha101 因子计算失败: {e}")

    # ── 涨跌停信号因子（需要 masks，由 clean_ohlcv() 提供）──
    if (masks is not None and masks.get("limit_up") is not None
            and _section_needed(_LIMIT_FACTOR_NAMES, fn_set)):
        try:
            from factors.factor_limit import get_limit_factors
            limit_factors = get_limit_factors(prices, masks)
            if fn_set is not None:
                limit_factors = {k: v for k, v in limit_factors.items() if k in fn_set}
            registry.update(limit_factors)
        except Exception as e:
            logger.warning(f"涨跌停因子计算失败: {e}")

    registry = {k: v for k, v in registry.items() if v is not None}

    # 将所有因子 reindex 到 prices.index，保证日期维度完全对齐。
    # margin/moneyflow/northbound 等数据源历史较短，reindex 后缺失日期填 NaN，
    # 不影响有数据的时段，避免 set.intersection 因日期不重叠返回空集。
    for name in list(registry.keys()):
        f = registry[name]
        if not f.index.equals(prices.index):
            registry[name] = f.reindex(prices.index)

    logger.info(f"因子库就绪: {len(registry)} 个因子")
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
