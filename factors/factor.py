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

    市值 alpha（需 circ_mv/total_mv，见 factors/factor_size_alpha.py）：
        对数市值 / 市值分位 / 市值风格对齐_20d|60d
        —— 小市值高分；feature_neutralize 时豁免（SIZE_ALPHA_FACTOR_NAMES）

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
from factors.factor_cache import (
    FACTOR_CACHE_DIR,
    _FACTOR_LIB_VERSION,  # noqa: F401  — 暴露给外部 bump 检测
    build_input_signature,
    compute_single_factor_cached,
    clear_factor_cache,  # noqa: F401  — 便捷重导出，供 CLI/调试调用
    list_cached_factors,  # noqa: F401  — 便捷重导出
)


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

    理想口径：turnover = volume / 流通股本，剔除规模效应。
    本项目 `data/raw/` 未下载流通股本 / 总股本数据
    （`data/` 下无 free_float / total_shares / float_share 列），
    无法做流通股本归一化，当前以成交量均值近似排名，
    会混入规模效应（小盘股成交量绝对值更小 → 倾向高分）。

    使用建议：配合「规模」因子或在 IC/ML 流程中做行业 / 规模中性化；
    或后续补下载 AKShare 的 `stock_zh_a_spot` 流通股本字段后切到
    `volume / free_float_shares` 口径。
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

    用 financial 的 `eps` 列直接计算 EP = eps / price，
    替代此前 `roe * bvps` 的近似（ROE × 每股净资产 ≈ EPS），
    避免多重会计科目近似叠加的误差。

    PIT 对齐：eps 经 _pivot_financial 按法定披露日平移后再 ffill，
    不存在用报告期日做 ffill 起点的 look-ahead bias。
    """
    if "eps" not in financial.columns:
        logger.warning("财务数据中无 eps 列，factor_value_ep 跳过")
        return None
    eps = _pivot_financial(financial, "eps", prices_raw)
    price = prices_raw.reindex(columns=eps.columns)
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

    优先：应计 = (net_profit - operating_cashflow) / total_assets。
    若缺 ``net_profit``（financial_indicators 仅有每股字段）：退化为
    (eps - ocf_ps) / |eps|（与 OpenSourceAP PctAcc / ``应计占比`` 同口径）。
    """
    if "operating_cashflow" not in financial.columns:
        logger.warning("财务数据缺少 operating_cashflow，factor_quality_accrual 跳过")
        return None

    ocf_pivot = _pivot_financial(financial, "operating_cashflow", prices)

    if "net_profit" in financial.columns and "total_assets" in financial.columns:
        np_pivot = _pivot_financial(financial, "net_profit", prices)
        ta_pivot = _pivot_financial(financial, "total_assets", prices)
        accrual = (np_pivot - ocf_pivot) / ta_pivot.replace(0, np.nan)
    elif "eps" in financial.columns:
        eps = _pivot_financial(financial, "eps", prices)
        denom = eps.abs().where(eps.abs() >= 1e-6, 0.01)
        accrual = (eps - ocf_pivot) / denom
    else:
        logger.warning(
            "财务数据缺少 net_profit+total_assets 或 eps，factor_quality_accrual 跳过"
        )
        return None

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

    用 financial 的 debt_ratio（资产负债率，%）作杠杆代理：
        debt_ratio = 总负债 / 总资产 × 100
    量纲一致（分子分母均为「元」），截面可比，避免之前
    `total_assets / bvps` 把「元」与「元/股」混算的量纲错误。

    高杠杆公司在经济下行期风险更大，故对 debt_ratio 取负向。
    PIT 对齐：debt_ratio 经 _pivot_financial 按法定披露日平移后 ffill。
    """
    if "debt_ratio" not in financial.columns:
        logger.warning("财务数据中无 debt_ratio 列，factor_leverage 跳过")
        return None
    debt = _pivot_financial(financial, "debt_ratio", prices)
    return _normalize(-debt)  # 取负：低杠杆得高分


def factor_growth_revenue(financial: pd.DataFrame,
                          prices: pd.DataFrame) -> pd.DataFrame:
    """
    营收增长率因子。
    直接用 financial 的 `revenue_growth` 列（主营业务收入同比增长率，%），
    高增长得高分。PIT 对齐：经 _pivot_financial 按法定披露日平移后 ffill。
    """
    if "revenue_growth" not in financial.columns:
        logger.warning("财务数据中无 revenue_growth 列，factor_growth_revenue 跳过")
        return None
    g = _pivot_financial(financial, "revenue_growth", prices)
    return _normalize(g)


def factor_growth_profit(financial: pd.DataFrame,
                         prices: pd.DataFrame) -> pd.DataFrame:
    """
    净利润增长率因子。
    直接用 financial 的 `net_profit_growth` 列（净利润同比增长率，%），
    高增长得高分。PIT 对齐：经 _pivot_financial 按法定披露日平移后 ffill。
    """
    if "net_profit_growth" not in financial.columns:
        logger.warning("财务数据中无 net_profit_growth 列，factor_growth_profit 跳过")
        return None
    g = _pivot_financial(financial, "net_profit_growth", prices)
    return _normalize(g)


def factor_growth_eps(financial: pd.DataFrame,
                      prices: pd.DataFrame) -> pd.DataFrame:
    """
    EPS 增长率因子。

    financial 无 `eps_growth` 直接列，用季报 `eps` pivot 后做同比
    `pct_change(4)`（4 个季报期 = 1 年）计算 EPS YoY 增长率。

    PIT 安全：先 pivot 到季报截面，再做同比差分，最后通过
    `pit_reindex_ffill` 按法定披露日平移并 reindex 到日频。
    （若先按披露日平移再 diff，会引入相邻披露期之间的伪变化。）
    """
    if "eps" not in financial.columns:
        logger.warning("财务数据中无 eps 列，factor_growth_eps 跳过")
        return None
    eps_pivot = financial.pivot_table(index="trade_date", columns="code", values="eps")
    eps_pivot.index = pd.to_datetime(eps_pivot.index)
    eps_yoy = eps_pivot.pct_change(4)
    eps_yoy = eps_yoy.replace([np.inf, -np.inf], np.nan)
    eps_yoy = pit_reindex_ffill(eps_yoy, prices.index)
    return _normalize(eps_yoy)


# ══════════════════════════════════════════════════════════════════════════════
# 因子注册表（供 notebook 和 run() 使用）
# ══════════════════════════════════════════════════════════════════════════════

def _fit_hmm_regime(idx_ret: pd.Series, n_states: int = 3) -> pd.DataFrame:
    """
    用 GaussianHMM 从指数日收益率序列识别市场隐藏状态，返回各状态的后验概率。

    n_states=3: 强势/震荡/弱势三状态，比二状态更能区分 A 股牛快熊慢的特征。
    状态按均值收益率排序后重新标记: 强势=idx 0, 震荡=idx 1, 弱势=idx 2。

    输出列: ['HMM_强势概率', 'HMM_弱势概率']（丢弃震荡，三列加和=1 线性相关）

    注意：本函数在全样本上拟合，后验概率为 in-sample，理论轻微前视。
    若需消除前视，改用 _fit_hmm_regime_walk_forward（扩展窗口，逐段重拟合）。
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


def _fit_hmm_regime_walk_forward(
    idx_ret: pd.Series,
    n_states: int = 3,
    refit_every: int = 20,
    min_obs: int = 60,
    rolling_window: int = 0,
) -> pd.DataFrame:
    """
    Walk-forward HMM 市场状态识别（消除全样本拟合的前视偏差）。

    对每个日期 t，只用 [start, t] 的历史收益拟合 HMM（rolling_window>0 时改为
    [t-rolling_window+1, t] 的滚动窗口），预测 t 的状态后验概率。
    为兼顾性能，每 refit_every 个交易日才重拟合一次模型，中间沿用上次模型预测。

    严格 PIT：t 期的后验概率只依赖 [start, t] 的历史信息，不含未来收益。

    参数
    ----
    idx_ret:        指数日收益率 Series
    n_states:       HMM 状态数，默认 3（强势/震荡/弱势）
    refit_every:    重拟合间隔（日），默认 20（月频重拟合，中间沿用旧模型）
    min_obs:        启动拟合所需最小样本数，默认 60
    rolling_window: >0 时用滚动窗口 [t-w+1, t]；=0 时用扩展窗口 [0, t]
    """
    from hmmlearn import hmm as _hmm

    returns = idx_ret.fillna(0).values.reshape(-1, 1)
    T = len(returns)
    probs_out = np.full((T, n_states), np.nan)

    model = None
    order = None
    last_fit_idx = -refit_every  # 确保 min_obs 处首次拟合

    for t in range(T):
        if t < min_obs:
            continue
        need_refit = (t - last_fit_idx) >= refit_every or model is None
        if need_refit:
            if rolling_window > 0:
                start = max(0, t - rolling_window + 1)
                train = returns[start:t + 1]
            else:
                train = returns[:t + 1]
            if len(train) < min_obs:
                continue
            try:
                m = _hmm.GaussianHMM(
                    n_components=n_states,
                    covariance_type="full",
                    n_iter=100,
                    random_state=42,
                )
                m.fit(train)
                means = m.means_.flatten()
                order = np.argsort(means)[::-1]  # 强势 → 震荡 → 弱势
                model = m
                last_fit_idx = t
            except Exception:
                continue
        # 用当前模型预测 t 期后验概率（单点）
        try:
            p = model.predict_proba(returns[t:t + 1])[0]
            probs_out[t] = p[order]
        except Exception:
            continue

    df = pd.DataFrame(probs_out, index=idx_ret.index,
                      columns=["HMM_强势概率", "HMM_震荡概率", "HMM_弱势概率"])
    return df[["HMM_强势概率", "HMM_弱势概率"]]


def _fit_hmm_regime_multivariate(
    returns: pd.Series,
    breadth: pd.Series,
    liquidity: pd.Series,
    n_states: int = 3,
) -> pd.DataFrame:
    """
    多元 GaussianHMM 市场状态识别：用 [指数收益, 涨跌家数比, 成交额zscore] 三变量
    联合建模，相比单变量 HMM 引入广度与流动性信息，状态识别更稳健。

    状态按 idx_ret 均值排序：最高=强势(0)，最低=弱势(2)，与单变量 HMM 同口径。
    输出列: ['HMM_多元_强势概率', 'HMM_多元_弱势概率']（丢弃震荡）。
    全样本拟合，理论轻微前视；如需消除请用 _fit_hmm_regime_multivariate_walk_forward。
    """
    from hmmlearn import hmm as _hmm

    X = np.column_stack([returns.values, breadth.values, liquidity.values])
    model = _hmm.GaussianHMM(
        n_components=n_states, covariance_type="full",
        n_iter=200, random_state=42,
    )
    model.fit(X)
    probs = model.predict_proba(X)
    means_ret = model.means_[:, 0]
    order = np.argsort(means_ret)[::-1]
    probs = probs[:, order]
    df = pd.DataFrame(probs, index=returns.index,
                      columns=["HMM_多元_强势概率", "HMM_多元_震荡概率", "HMM_多元_弱势概率"])
    return df[["HMM_多元_强势概率", "HMM_多元_弱势概率"]]


def _fit_hmm_regime_multivariate_walk_forward(
    returns: pd.Series,
    breadth: pd.Series,
    liquidity: pd.Series,
    n_states: int = 3,
    refit_every: int = 20,
    min_obs: int = 60,
    rolling_window: int = 0,
) -> pd.DataFrame:
    """
    多元 HMM 的 walk-forward 版本（消除全样本拟合的前视偏差）。
    参数语义同 _fit_hmm_regime_walk_forward。
    """
    from hmmlearn import hmm as _hmm

    X = np.column_stack([returns.values, breadth.values, liquidity.values])
    T = len(X)
    probs_out = np.full((T, n_states), np.nan)
    model = None
    order = None
    last_fit_idx = -refit_every

    for t in range(T):
        if t < min_obs:
            continue
        need_refit = (t - last_fit_idx) >= refit_every or model is None
        if need_refit:
            if rolling_window > 0:
                start = max(0, t - rolling_window + 1)
                train = X[start:t + 1]
            else:
                train = X[:t + 1]
            if len(train) < min_obs:
                continue
            try:
                m = _hmm.GaussianHMM(
                    n_components=n_states, covariance_type="full",
                    n_iter=100, random_state=42,
                )
                m.fit(train)
                means_ret = m.means_[:, 0]
                order = np.argsort(means_ret)[::-1]
                model = m
                last_fit_idx = t
            except Exception:
                continue
        try:
            p = model.predict_proba(X[t:t + 1])[0]
            probs_out[t] = p[order]
        except Exception:
            continue

    df = pd.DataFrame(probs_out, index=returns.index,
                      columns=["HMM_多元_强势概率", "HMM_多元_震荡概率", "HMM_多元_弱势概率"])
    return df[["HMM_多元_强势概率", "HMM_多元_弱势概率"]]


def _market_regime_features(
    market_prices: pd.DataFrame,
    prices: pd.DataFrame,
    volume: pd.DataFrame | None = None,
    amount: pd.DataFrame | None = None,
    clean_ret: pd.DataFrame | None = None,
    masks: dict | None = None,
    northbound: pd.DataFrame | None = None,
    margin: pd.DataFrame | None = None,
    small_cap_index: pd.DataFrame | None = None,
    walk_forward_hmm: bool = False,
) -> dict:
    """
    **已退役（不再注入 ML 因子矩阵 X）**。

    历史实现：把市场标量广播成 date×stock 面板（同日全市场常数）。
    仓位控制请用 ``backtest.regime.compute_position_regime``；
    本函数仅保留供研究脚本 / 兼容旧调用，registry 默认不再 yield。

    不做截面 zscore（截面内同值 → zscore=0），改用时序滚动标准化。
    shift(1) 避免前视偏差。
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

    # ── 原有连续型技术特征（5个）──
    for w in [20, 60, 120]:
        features[f"市场动量_{w}d"] = _broadcast(_ts_norm(idx.pct_change(w)))
    features["市场波动率_20d"] = _broadcast(_ts_norm(idx_ret.rolling(20).std()))
    ma60 = idx.rolling(60).mean()
    features["市场MA偏离_60d"] = _broadcast(_ts_norm(idx / ma60.replace(0, np.nan) - 1))

    # ── 原有 HMM 单变量状态概率（2个）──
    try:
        if walk_forward_hmm:
            hmm_probs = _fit_hmm_regime_walk_forward(idx_ret)
            logger.info(f"HMM 市场状态拟合完成 (walk-forward 扩展窗口)，样本={len(idx_ret)}条")
        else:
            hmm_probs = _fit_hmm_regime(idx_ret)
            logger.info(f"HMM 市场状态拟合完成 (全样本)，样本={len(idx_ret)}条")
        for col in hmm_probs.columns:
            features[col] = _broadcast(hmm_probs[col])
    except Exception as e:
        logger.warning(f"HMM 拟合失败，跳过: {e}")

    # ── 广度（breadth）特征（4个）──
    # clean_ret 缺失时回退到 prices.pct_change()（涨跌停日不掩码，仅次优）
    ret_panel = clean_ret if clean_ret is not None else prices.pct_change()
    ret_panel = ret_panel.reindex(index=prices.index, columns=prices.columns)

    adv_dec_ratio = None
    try:
        up = ret_panel.gt(0).sum(axis=1)
        dn = ret_panel.lt(0).sum(axis=1).replace(0, np.nan)
        adv_dec_ratio = (up / dn).replace([np.inf, -np.inf], np.nan)
        for w in [20, 60]:
            features[f"市场_涨跌家数比_{w}d"] = _broadcast(
                _ts_norm(adv_dec_ratio.rolling(w).mean())
            )
    except Exception as e:
        logger.warning(f"涨跌家数比特征计算失败: {e}")

    try:
        rolling_max = prices.rolling(20).max()
        new_high_frac = (prices >= rolling_max).sum(axis=1) / prices.notna().sum(axis=1).replace(0, np.nan)
        features["市场_突破20日新高占比"] = _broadcast(
            _ts_norm(new_high_frac.rolling(20).mean())
        )
    except Exception as e:
        logger.warning(f"突破20日新高占比计算失败: {e}")

    try:
        ma20 = prices.rolling(20).mean()
        above_ma_frac = (prices > ma20).sum(axis=1) / prices.notna().sum(axis=1).replace(0, np.nan)
        features["市场_站上MA20占比"] = _broadcast(
            _ts_norm(above_ma_frac.rolling(20).mean())
        )
    except Exception as e:
        logger.warning(f"站上MA20占比计算失败: {e}")

    # ── 流动性特征（2个）──
    total_amount_norm = None
    if amount is not None:
        try:
            total_amount = amount.sum(axis=1)
            total_amount_norm = _ts_norm(total_amount)
            features["市场_成交额_zscore_20d"] = _broadcast(total_amount_norm)
        except Exception as e:
            logger.warning(f"成交额zscore计算失败: {e}")
        try:
            tot = amount.sum(axis=1).replace(0, np.nan)
            weights = amount.div(tot, axis=0)
            conc = (weights ** 2).sum(axis=1)
            features["市场_成交额集中度"] = _broadcast(_ts_norm(conc.rolling(20).mean()))
        except Exception as e:
            logger.warning(f"成交额集中度计算失败: {e}")

    # ── 资金流特征（融资；北向下线，不再生成市场_北向净流入_zscore_20d）──
    if margin is not None:
        try:
            margin_total = margin.sum(axis=1)
            margin_chg = margin_total.pct_change(5).replace([np.inf, -np.inf], np.nan)
            features["市场_融资余额变化_zscore_20d"] = _broadcast(_ts_norm(margin_chg))
        except Exception as e:
            logger.warning(f"融资余额变化特征计算失败: {e}")

    # ── 风格轮动（1个，需 small_cap_index，可选）──
    if small_cap_index is not None:
        try:
            small_idx = small_cap_index.iloc[:, 0].reindex(prices.index, method="ffill")
            small_ret = small_idx.pct_change(20)
            large_ret = idx.pct_change(20)
            rel = (small_ret - large_ret).replace([np.inf, -np.inf], np.nan)
            features["市场_小盘相对强度_20d"] = _broadcast(_ts_norm(rel))
        except Exception as e:
            logger.warning(f"小盘相对强度计算失败: {e}")

    # ── 波动率结构（1个）──
    try:
        cs_vol = ret_panel.std(axis=1)
        features["市场_截面波动率_20d"] = _broadcast(_ts_norm(cs_vol.rolling(20).mean()))
    except Exception as e:
        logger.warning(f"截面波动率特征计算失败: {e}")

    # ── 多元 HMM（2个，需 adv_dec_ratio + 成交额zscore）──
    try:
        if adv_dec_ratio is not None and total_amount_norm is not None:
            breadth_series = (
                adv_dec_ratio.rolling(20).mean()
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
            )
            liquidity_series = total_amount_norm.fillna(0)
            ret_series = idx_ret.fillna(0)
            breadth_series = breadth_series.reindex(ret_series.index).fillna(0)
            liquidity_series = liquidity_series.reindex(ret_series.index).fillna(0)
            if walk_forward_hmm:
                hmm_probs2 = _fit_hmm_regime_multivariate_walk_forward(
                    ret_series, breadth_series, liquidity_series
                )
                logger.info(f"多元 HMM 拟合完成 (walk-forward)，样本={len(ret_series)}条")
            else:
                hmm_probs2 = _fit_hmm_regime_multivariate(
                    ret_series, breadth_series, liquidity_series
                )
                logger.info(f"多元 HMM 拟合完成 (全样本)，样本={len(ret_series)}条")
            for col in hmm_probs2.columns:
                features[col] = _broadcast(hmm_probs2[col])
    except Exception as e:
        logger.warning(f"多元 HMM 拟合失败，跳过: {e}")

    return features


def _is_regime_factor(name: str) -> bool:
    """旧广播包名检测：``市场*`` / ``HMM_*``（同日全市场常数，已退役出 ML X）。"""
    return name.startswith("市场") or name.startswith("HMM_")


def _want_factor(name: str, factor_names: set | None) -> bool:
    if factor_names is None:
        return True
    return name in factor_names


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
    "营收增长率", "净利增长率", "EPS增长率",
})
# 事件驱动因子（业绩预告等）—— **事件 overlay 因子，不进入 ML 截面输入池**。
# 这类因子稀疏（仅在公告日有值 + window 日 ffill），不符合截面标准化假设
# （winsorize+zscore 在大量 NaN 行上失真），且语义上是离散事件信号而非连续
# 截面排序。详见 research/diagnose_sparse_factors.py 的稀疏因子诊断。
# 默认不进 ML / IC 候选枚举（get_factor_names / _iter_factor_registry_raw）。
# 用 `get_event_overlay_factors()` 单独计算；ML 通过
# `--special-factors event`（或 deprecated `--event-overlay`）在白名单过滤之后
# post-merge 注入（见 factors/special_factors.py + strategies/ml.py）。
# 后续新增事件因子只改本 frozenset，并确保 event pack 引用本集合。
EVENT_OVERLAY_FACTOR_NAMES = frozenset({
    "业绩预告_超预期",
    "业绩快报超预期",
})
_ALPHA2_FACTOR_NAMES = frozenset({
    "行业动量_20d", "特质波动率_60d", "融资余额变化_20d",
    "大单净流入_5d", "大单残差净流入_5d", "机构持仓变化",
    # 北向下线：北向持股变化_20d / 北向净流入_20d 不再默认注册
})
# A 股特色稠密增量（两融日频截面；大单残差在 Alpha2 段单独 yield）
_ASHARE_DENSE_FACTOR_NAMES = frozenset({
    "融资买入占成交额_5d",
    "融资净买入_5d",
    "融券卖出规避_5d",
    "融资余额流通市值比",
})
# A 股特色稀疏增量（评级/研报/龙虎榜机构/回购/大宗席位/板块资金流/解禁等）
_ASHARE_SPARSE_FACTOR_NAMES = frozenset({
    "评级上修_20d",
    "研报EPS上修次数_20d",
    "研报预期差",
    "龙虎榜机构净买入_20d",
    "龙虎榜机构买入强度_20d",
    "股份回购强度_60d",
    "大宗折价席位质量_20d",
    "板块资金流拥挤_5d",
    "龙虎榜净买占比_20d",
    "目标价上行空间",
    "研报覆盖热度_20d",
    "回购完成进度_60d",
    "解禁流动性压力_60d",
    "大宗机构接盘_20d",
    "评级下调规避_20d",
    "龙虎榜涨幅上榜_20d",
    "龙虎榜换手上榜_20d",
    "龙虎榜跌幅上榜规避_20d",
    "解禁定增压力_60d",
    "解禁激励压力_60d",
    "转债转股稀释_60d",
    "激励行权稀释_60d",
    "限售上市供给_60d",
    "研报EPS斜率",
    "研报EPS分歧度",
    "大宗卖方机构抛压_20d",
    "大宗折溢价波动_20d",
})
# get_ashare_factors 截面枚举（稠密两融 + 稀疏事件；不含 Alpha2 的大单残差）
_ASHARE_CS_FACTOR_NAMES = _ASHARE_DENSE_FACTOR_NAMES | _ASHARE_SPARSE_FACTOR_NAMES
_TECH_FACTOR_NAMES = frozenset({
    "BIAS_5d", "BIAS_20d", "PSY_12d", "AR_26d", "BR_26d",
    "换手率加速度", "换手率行业中性_20d", "行业相对强度_20d",
})
_LIMIT_FACTOR_NAMES = frozenset({
    "涨停强度_20d", "跌停弱势_20d", "连板数", "涨跌停净强度_20d", "涨跌停状态", "开板反转_5d",
})
_SMALLCAP_FACTOR_NAMES = frozenset({
    "股东户数变化率_季", "户均流通市值_季", "股东户数变化率_年",
    "龙虎榜上榜次数_20d", "龙虎榜净买额_20d", "龙虎榜连续上榜",
    "未来60日解禁市值占比", "未来30日解禁次数",
    "高管净增持额_60d", "高管增持次数_60d", "高管减持次数_60d", "增减持比_60d",
    "大宗交易折价率_20d", "大宗交易频次_20d",
    "个股融资余额变化_20d", "融券余额变化_20d", "融资买入额_5d",
})
# 市值相关 alpha（日频 circ_mv/total_mv）；进 IC/YAML 正常池，非强制注入。
# feature_neutralize 时豁免 —— 真源 SIZE_ALPHA_FACTOR_NAMES 在 factor_size_alpha.py。
_SIZE_ALPHA_FACTOR_NAMES = frozenset({
    "对数市值", "市值分位", "市值风格对齐_20d", "市值风格对齐_60d",
})
# OpenSourceAP Batch-1/2 因子（注册入池，不自动写入 factor_configs.yaml）
_OPENSOURCE_AP_FACTOR_NAMES = frozenset({
    "资产增长", "资产市值比", "现金流市值比",
    "权益增长", "盈利一致性", "营收增长秩",
    "盈利连增期数", "一年股本扩张", "五年股本扩张", "综合股权融资",
    "上市年龄", "年龄动量", "权益变化资产比", "应计占比",
    "现金流价格波动", "毛利资产比", "营收市值比",
    "净债务市值比", "综合债务融资",
    "月最大收益", "收益偏度", "行业集中度",
    # Batch-3
    "应计资产比", "经营利润权益比", "外部融资资产比",
    "资产周转变化", "经营杠杆",
    "协偏度", "残差动量", "季节动量",
})


def _iter_factor_registry_raw(
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
    circ_mv: pd.DataFrame = None,
    total_mv: pd.DataFrame = None,
    factor_names: list | set | None = None,
    walk_forward_hmm: bool = False,
    include_regime: bool = False,
):
    """
    流式因子生成器的**原始计算**实现（无缓存）。

    公开入口请用 ``iter_factor_registry``（带磁盘缓存）；本函数是其底层计算核，
    亦供 ``compute_single_factor``（已自带缓存层）直接调用，避免双重缓存。

    include_regime : 已退役。True 时仅打 warning，不再注入 ``市场*``/``HMM_*``。
    walk_forward_hmm : 已退役（随市场/HMM 注入一并停用），保留签名兼容。
    """
    _ = walk_forward_hmm
    if include_regime:
        logger.warning(
            "include_regime=True 已退役：市场/HMM 广播特征不再注入因子矩阵 X。"
            "仓位控制请用 --position-regime（backtest/regime.py）。"
        )
    fn_set = set(factor_names) if factor_names is not None else None

    def _emit(name, panel):
        """统一出口：过滤 None + reindex 到 prices.index。返回 None 表示跳过。"""
        if panel is None:
            return None
        if not panel.index.equals(prices.index):
            panel = panel.reindex(prices.index)
        return name, panel

    # ── 价格/动量类（全部传入 clean_ret，有则用，无则自动回退）──
    if _section_needed(_PRICE_FACTOR_NAMES, fn_set):
        if _want_factor("动量_3d", fn_set):
            yield _emit("动量_3d", factor_momentum(prices, 3, clean_ret))
        if _want_factor("动量_10d", fn_set):
            yield _emit("动量_10d", factor_momentum(prices, 10, clean_ret))
        if _want_factor("动量_20d", fn_set):
            yield _emit("动量_20d", factor_momentum(prices, 20, clean_ret))
        if _want_factor("动量_40d", fn_set):
            yield _emit("动量_40d", factor_momentum(prices, 40, clean_ret))
        if _want_factor("动量_60d", fn_set):
            yield _emit("动量_60d", factor_momentum(prices, 60, clean_ret))
        if _want_factor("动量_120d", fn_set):
            yield _emit("动量_120d", factor_momentum(prices, 120, clean_ret))
        if _want_factor("动量_skip", fn_set):
            yield _emit("动量_skip", factor_momentum_skip(prices, 240, 20, clean_ret))
        if _want_factor("反转_3d", fn_set):
            yield _emit("反转_3d", factor_reversal(prices, 3, clean_ret))
        if _want_factor("反转_5d", fn_set):
            yield _emit("反转_5d", factor_reversal(prices, 5, clean_ret))
        if _want_factor("反转_10d", fn_set):
            yield _emit("反转_10d", factor_reversal(prices, 10, clean_ret))
        if _want_factor("反转_20d", fn_set):
            yield _emit("反转_20d", factor_reversal(prices, 20, clean_ret))
        if _want_factor("波动率_20d", fn_set):
            yield _emit("波动率_20d", factor_volatility(prices, 20, clean_ret))
        if _want_factor("波动率_60d", fn_set):
            yield _emit("波动率_60d", factor_volatility(prices, 60, clean_ret))
        if _want_factor("52周新高", fn_set):
            yield _emit("52周新高", factor_price_to_high(prices, 52 * 5))
        if _want_factor("分数差分动量_20d", fn_set):
            yield _emit(
                "分数差分动量_20d",
                factor_frac_diff_momentum(prices, d=0.4, window=20),
            )
        if volume is not None and _want_factor("换手率_20d", fn_set):
            yield _emit("换手率_20d", factor_turnover(volume, 20))
        if amount is not None and _want_factor("Amihud_20d", fn_set):
            yield _emit("Amihud_20d", factor_amihud(prices, amount, 20, clean_ret))
        if high is not None and low is not None and _want_factor("振幅_20d", fn_set):
            yield _emit("振幅_20d", factor_high_low(prices, high, low, 20, masks))

    # ── 财务类 ──
    if financial is not None and not financial.empty and _section_needed(_FIN_FACTOR_NAMES, fn_set):
        _pr = prices_raw if prices_raw is not None else prices
        if _want_factor("价值_PB", fn_set):
            yield _emit("价值_PB", factor_value_pb(financial, _pr))
        if _want_factor("价值_EP", fn_set):
            yield _emit("价值_EP", factor_value_ep(financial, _pr))
        if _want_factor("质量_ROE", fn_set):
            yield _emit("质量_ROE", factor_quality_roe(financial, prices))
        if _want_factor("质量_ROE变化", fn_set):
            yield _emit("质量_ROE变化", factor_quality_roe_chg(financial, prices))
        if _want_factor("规模", fn_set):
            yield _emit("规模", factor_size(financial, prices))
        if _want_factor("杠杆", fn_set):
            yield _emit("杠杆", factor_leverage(financial, prices))
        for name, f in [
            ("质量_毛利率", factor_quality_gpm(financial, prices)),
            ("质量_毛利率变化", factor_quality_gpm_chg(financial, prices)),
            ("质量_应计项目", factor_quality_accrual(financial, prices)),
            ("营收增长率", factor_growth_revenue(financial, prices)),
            ("净利增长率", factor_growth_profit(financial, prices)),
            ("EPS增长率", factor_growth_eps(financial, prices)),
        ]:
            if _want_factor(name, fn_set):
                yield _emit(name, f)

    # ── 事件驱动因子（业绩预告等，PIT 用 announce_date）──
    # NOTE: 事件因子稀疏（仅公告日 + window 日 ffill），不适合截面标准化 + ML
    # 输入池。已从 ML 截面输入中移除，改由 `get_event_overlay_factors()` 单独
    # 计算供事件 overlay 接入。详见 EVENT_OVERLAY_FACTOR_NAMES 注释。
    # （保留块以显式说明此处刻意不 yield，避免被误以为是漏写。）

    # ── 第二批 Alpha 因子 ──
    if _section_needed(_ALPHA2_FACTOR_NAMES, fn_set):
        try:
            from factors.factor_alpha import (
                factor_industry_momentum, factor_idiosyncratic_vol,
                factor_margin_change, factor_moneyflow_large,
                factor_institution_change, load_industry_panel,
            )
            # PIT 行业面板：若调用方未提供则尝试自动加载
            industry_panel = None
            try:
                industry_panel = load_industry_panel()
            except Exception as e:
                logger.warning(f"加载 industry_map_panel 失败: {e}")
            if industry_map is not None and _want_factor("行业动量_20d", fn_set):
                f = factor_industry_momentum(
                    prices, industry_map, window=20,
                    clean_ret=clean_ret, industry_panel=industry_panel,
                )
                yield _emit("行业动量_20d", f)
            if market_prices is not None and _want_factor("特质波动率_60d", fn_set):
                yield _emit(
                    "特质波动率_60d",
                    factor_idiosyncratic_vol(prices, market_prices, window=60, clean_ret=clean_ret),
                )
            if margin is not None and _want_factor("融资余额变化_20d", fn_set):
                yield _emit("融资余额变化_20d", factor_margin_change(margin, window=20))
            # ── moneyflow 因子已弃用（akshare 资金流数据不足）──
            # 大单净流入_5d / 大单残差净流入_5d 不再计算；moneyflow 上游已不加载（恒 None）。
            # 保留 yield 守卫以兼容旧 YAML；新生产 YAML 不应包含这两个因子名。
            # 详见 docs/ASHARE_FACTOR_DATA_GAPS.md §1。
            if moneyflow is not None and _want_factor("大单净流入_5d", fn_set):
                yield _emit(
                    "大单净流入_5d",
                    factor_moneyflow_large(moneyflow, window=5, amount=amount),
                )
            if moneyflow is not None and _want_factor("大单残差净流入_5d", fn_set):
                from factors.factor_ashare import factor_moneyflow_residual
                turnover = None
                try:
                    tp = RAW_DIR / "turnover_rate.parquet"
                    if tp.exists():
                        turnover = pd.read_parquet(tp)
                except Exception:
                    turnover = None
                yield _emit(
                    "大单残差净流入_5d",
                    factor_moneyflow_residual(
                        moneyflow, amount=amount, clean_ret=clean_ret,
                        turnover=turnover, prices=prices, window=5,
                    ),
                )
            # 北向下线：不再 yield 北向持股变化_20d / 北向净流入_20d
            if institution is not None and _want_factor("机构持仓变化", fn_set):
                yield _emit("机构持仓变化", factor_institution_change(institution, prices))
        except ImportError as e:
            logger.warning(f"factor_alpha 导入失败: {e}")

    # ── A 股特色截面因子（稠密两融 + 稀疏评级/龙虎榜/回购/解禁等）──
    if _section_needed(_ASHARE_CS_FACTOR_NAMES, fn_set):
        try:
            from factors.factor_ashare import get_ashare_factors
            ashare = get_ashare_factors(
                prices,
                moneyflow=moneyflow,
                amount=amount,
                clean_ret=clean_ret,
                prices_raw=prices_raw,
                factor_names=(
                    (fn_set & _ASHARE_CS_FACTOR_NAMES)
                    if fn_set is not None
                    else set(_ASHARE_CS_FACTOR_NAMES)
                ),
            )
            for name, panel in ashare.items():
                if name in _ASHARE_CS_FACTOR_NAMES:
                    yield _emit(name, panel)
        except Exception as e:
            logger.warning(f"A股特色截面因子计算失败: {e}")

    # ── 市场状态特征：已退役，不再 yield 进 registry（见 backtest/regime.py）──

    # ── 技术分析类因子（BIAS/PSY/ARBR/换手率变体/行业相对强度）──
    if _section_needed(_TECH_FACTOR_NAMES, fn_set):
        try:
            from factors.factor_technical import get_technical_factors
            # 技术因子也尝试加载 PIT 行业面板
            tech_industry_panel = None
            try:
                from factors.factor_alpha import load_industry_panel
                tech_industry_panel = load_industry_panel()
            except Exception as e:
                logger.warning(f"技术因子加载 industry_map_panel 失败: {e}")
            tech_factors = get_technical_factors(
                prices=prices, volume=volume, industry_map=industry_map,
                open_=open_, high=high, low=low, clean_ret=clean_ret,
                industry_panel=tech_industry_panel,
            )
            if fn_set is not None:
                tech_factors = {k: v for k, v in tech_factors.items() if k in fn_set}
            for name, panel in tech_factors.items():
                yield _emit(name, panel)
        except Exception as e:
            logger.warning(f"技术因子计算失败: {e}")

    # ── WorldQuant Alpha101 精选因子 ──
    need_wq = fn_set is None or any(n.startswith("WQ_") for n in fn_set)
    if need_wq:
        try:
            from factors.factor_alpha101 import get_alpha101_factors
            # 白名单时只算请求的 WQ_*，避免整包 ~82 个（run.py top50 常见仅二三十个）
            wq_wanted = (
                None if fn_set is None
                else {n for n in fn_set if n.startswith("WQ_")}
            )
            wq_factors = get_alpha101_factors(
                prices=prices, open_=open_, high=high, low=low,
                volume=volume, amount=amount, clean_ret=clean_ret,
                factor_names=wq_wanted,
            )
            # ????? pop yield??????? dict ??
            for name in list(wq_factors.keys()):
                panel = wq_factors.pop(name)
                yield _emit(name, panel)
            del wq_factors
        except Exception as e:
            logger.warning(f"Alpha101 因子计算失败: {e}")

    # ── Qlib Alpha158 量价特征（A158_*；流式，避免整包驻留 OOM）──
    need_a158 = fn_set is None or any(n.startswith("A158_") for n in fn_set)
    if need_a158:
        try:
            from factors.factor_alpha158 import iter_alpha158_factors
            a158_wanted = (
                None if fn_set is None
                else {n for n in fn_set if n.startswith("A158_")}
            )
            for name, panel in iter_alpha158_factors(
                prices=prices, open_=open_, high=high, low=low,
                volume=volume, amount=amount, clean_ret=clean_ret,
                factor_names=a158_wanted,
            ):
                yield _emit(name, panel)
        except Exception as e:
            logger.warning(f"Alpha158 因子计算失败: {e}")

    # ── 国泰君安 Alpha191（GTJA_*；流式逐因子，避免整包驻留 OOM）──
    need_gtja = fn_set is None or any(n.startswith("GTJA_") for n in fn_set)
    if need_gtja:
        try:
            from factors.factor_alpha191 import iter_alpha191_factors
            gtja_wanted = (
                None if fn_set is None
                else {n for n in fn_set if n.startswith("GTJA_")}
            )
            for name, panel in iter_alpha191_factors(
                prices=prices, open_=open_, high=high, low=low,
                volume=volume, amount=amount, clean_ret=clean_ret,
                market_prices=market_prices, factor_names=gtja_wanted,
            ):
                yield _emit(name, panel)
        except Exception as e:
            logger.warning(f"Alpha191 因子计算失败: {e}")

    # ── 小盘股策略因子（筹码/事件/资金类，自加载 data/raw/*.parquet）──
    if _section_needed(_SMALLCAP_FACTOR_NAMES, fn_set):
        try:
            from factors.factor_smallcap import get_smallcap_factors
            sc_factors = get_smallcap_factors(prices, prices_raw=prices_raw)
            if fn_set is not None:
                sc_factors = {k: v for k, v in sc_factors.items() if k in fn_set}
            for name, panel in sc_factors.items():
                yield _emit(name, panel)
        except Exception as e:
            logger.warning(f"小盘股因子计算失败: {e}")

    # ── 市值 alpha（对数市值 / 分位 / 风格对齐；需 circ_mv 或 fallback）──
    if _section_needed(_SIZE_ALPHA_FACTOR_NAMES, fn_set):
        try:
            from factors.factor_size_alpha import get_size_alpha_factors
            size_factors = get_size_alpha_factors(
                prices,
                financial=financial,
                circ_mv=circ_mv,
                total_mv=total_mv,
                clean_ret=clean_ret,
                factor_names=fn_set,
            )
            for name, panel in size_factors.items():
                yield _emit(name, panel)
        except Exception as e:
            logger.warning(f"市值 alpha 因子计算失败: {e}")

    # ── OpenSourceAP Batch-1/2 因子（PIT；不自动进 YAML 白名单）──
    # 价量类（FirmAge/MaxRet 等）可不依赖 financial；会计类需要 financial。
    if _section_needed(_OPENSOURCE_AP_FACTOR_NAMES, fn_set):
        try:
            from factors.factor_opensource_ap import get_opensource_ap_factors
            osap = get_opensource_ap_factors(
                prices,
                financial=financial,
                prices_raw=prices_raw,
                circ_mv=circ_mv,
                total_mv=total_mv,
                clean_ret=clean_ret,
                market_prices=market_prices,
                industry_map=industry_map,
                factor_names=fn_set,
            )
            for name, panel in osap.items():
                yield _emit(name, panel)
        except Exception as e:
            logger.warning(f"OpenSourceAP 因子计算失败: {e}")

    # ── 涨跌停信号因子（需要 masks，由 clean_ohlcv() 提供）──
    if (masks is not None and masks.get("limit_up") is not None
            and _section_needed(_LIMIT_FACTOR_NAMES, fn_set)):
        try:
            from factors.factor_limit import get_limit_factors
            limit_factors = get_limit_factors(prices, masks, clean_ret=clean_ret)
            if fn_set is not None:
                limit_factors = {k: v for k, v in limit_factors.items() if k in fn_set}
            for name, panel in limit_factors.items():
                yield _emit(name, panel)
        except Exception as e:
            logger.warning(f"涨跌停因子计算失败: {e}")


def iter_factor_registry(
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
    circ_mv: pd.DataFrame = None,
    total_mv: pd.DataFrame = None,
    factor_names: list | set | None = None,
    walk_forward_hmm: bool = False,
    include_regime: bool = False,
):
    """
    流式因子生成器（**带磁盘缓存**）：逐个 yield (name, panel)。

    与 ``get_factor_registry`` 同输入语义，但**不一次性构建 dict**，
    内存峰值从「N 个因子面板同时驻留」降到「当前 1 个因子面板」。

    include_regime / walk_forward_hmm : 已退役，见 ``_iter_factor_registry_raw``。

    每个 yield 出的 panel 已 reindex 到 ``prices.index``（与批量版本一致）。
    数据源缺失 / 财务列缺失时该因子被跳过（不 yield）。

    缓存策略
    --------
    - 先用 ``get_factor_names(compute=False)`` 轻量枚举候选名（不构建面板），
      按 (name, 输入指纹) 查盘：命中且签名有效 → 直接读 parquet yield（跳过计算）；
      未命中 → 走 ``_iter_factor_registry_raw`` 计算，落盘后 yield。
    - 冷启动（首次运行 / 缓存空）：等价于原 raw 生成器 + 落盘开销（parquet 写盘，
      约占单因子计算耗时的 1-5%）。alpha101 按白名单子集计算；technical/limit
      等其它批处理 section 仍整组计算后过滤。
    - 热启动（同输入二次调用，如 IC 管线 Stage 3 turnover）：全部命中缓存，
      跳过 factor_momentum / alpha101 / HMM 等全部计算，仅 parquet 读盘。

    - 输入数据变化（prices/clean_ret/financial 追加 / 列变动）→ 签名失配 → 自动重算。
    - 环境变量 ``FACTOR_CACHE_DISABLE=1`` 跳过缓存，强制走 raw 重算（调试用）。
    """
    from factors.factor_cache import _load_meta, _load_panel, _cache_paths, _cache_disabled

    fn_set = set(factor_names) if factor_names is not None else None

    # 构建输入指纹（一次）
    sig_kwargs = dict(
        prices=prices, financial=financial, prices_raw=prices_raw,
        volume=volume, amount=amount, open_=open_, high=high, low=low,
        clean_ret=clean_ret, masks=masks, market_prices=market_prices,
        industry_map=industry_map, margin=margin, moneyflow=moneyflow,
        northbound=northbound, institution=institution,
        circ_mv=circ_mv, total_mv=total_mv,
        walk_forward_hmm=walk_forward_hmm, include_regime=include_regime,
    )
    signature = build_input_signature(sig_kwargs)

    cache_off = _cache_disabled()

    # 轻量枚举候选名（不触发面板计算）
    candidate_names = get_factor_names(
        prices=prices, financial=financial, prices_raw=prices_raw,
        volume=volume, amount=amount, open_=open_, high=high, low=low,
        clean_ret=clean_ret, masks=masks, market_prices=market_prices,
        industry_map=industry_map, margin=margin, moneyflow=moneyflow,
        northbound=northbound, institution=institution,
        circ_mv=circ_mv, total_mv=total_mv,
        factor_names=factor_names, walk_forward_hmm=walk_forward_hmm,
        include_regime=include_regime,
    )
    # 应用 fn_set 白名单（get_factor_names 内部已应用；再过一遍确保与 raw 一致）
    if fn_set is not None:
        candidate_names = [n for n in candidate_names if n in fn_set]

    # ── 缓存命中分区 ──
    cached_hits: list[str] = []
    uncached: list[str] = []
    if not cache_off:
        for n in candidate_names:
            pq, meta_p = _cache_paths(n)
            meta = _load_meta(meta_p)
            # 复用 factor_cache._signature_matches 的等价判定
            from factors.factor_cache import _signature_matches
            if meta is not None and _signature_matches(meta, signature) and pq.exists():
                cached_hits.append(n)
            else:
                uncached.append(n)
        logger.info(
            f"factor panel cache HIT={len(cached_hits)} MISS={len(uncached)} "
            f"(dir={FACTOR_CACHE_DIR})"
        )
    else:
        uncached = candidate_names
        logger.info(
            f"factor panel cache DISABLED (FACTOR_CACHE_DISABLE=1)；"
            f"将重算 {len(uncached)} 个因子"
        )

    # ── 1) 命中缓存：直接读盘 yield（跳过全部计算）──
    for n in cached_hits:
        pq, _ = _cache_paths(n)
        try:
            panel = _load_panel(pq)
        except Exception as e:
            logger.warning(f"因子缓存读取失败，回退重算: {n} ({e})")
            uncached.append(n)
            continue
        # 兜底 reindex（缓存已 reindex，但防御性保留）
        if not panel.index.equals(prices.index):
            panel = panel.reindex(prices.index)
        yield n, panel

    # ── 2) 未命中：走 raw 计算并落盘 ──
    if uncached:
        raw_kwargs = dict(
            prices=prices, financial=financial, prices_raw=prices_raw,
            volume=volume, amount=amount, open_=open_, high=high, low=low,
            clean_ret=clean_ret, masks=masks, market_prices=market_prices,
            industry_map=industry_map, margin=margin, moneyflow=moneyflow,
            northbound=northbound, institution=institution,
            circ_mv=circ_mv, total_mv=total_mv,
            factor_names=set(uncached), walk_forward_hmm=walk_forward_hmm,
            include_regime=include_regime,
        )
        from factors.factor_cache import _save_panel
        for name, panel in _filter_none_emit(_iter_factor_registry_raw(**raw_kwargs)):
            if not cache_off:
                try:
                    panel_f32 = panel.astype(np.float32, copy=False)
                    _save_panel(name, panel_f32, signature)
                    panel = panel_f32
                except Exception as e:
                    logger.warning(f"因子缓存落盘失败 ({name}): {e}")
            yield name, panel


def _filter_none_emit(pairs):
    """Drop None emits from the generator (None 表示因子数据源缺失被跳过)."""
    for item in pairs:
        if item is not None:
            yield item


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
    circ_mv: pd.DataFrame = None,
    total_mv: pd.DataFrame = None,
    factor_names: list | set | None = None,
    walk_forward_hmm: bool = False,
    include_regime: bool = False,
) -> dict:
    """
    返回所有可计算因子的字典 {因子名: DataFrame}。
    根据传入的数据自动跳过缺乏数据源的因子。

    factor_names: 仅计算指定因子；None 表示全量计算。
    include_regime: 已退役（默认 False）。True 时仅 warning，不再注入市场/HMM。

    实现说明：内部基于 ``iter_factor_registry`` generator 收集为 dict（向后兼容）。
    内存敏感的逐因子场景（IC 分析）请直接用 ``iter_factor_registry``，避免同时持有
    全部因子面板（67 个 float32 面板常驻约 2.86GB）。
    """
    registry = dict(_filter_none_emit(iter_factor_registry(
        prices=prices,
        financial=financial,
        prices_raw=prices_raw,
        volume=volume,
        amount=amount,
        open_=open_,
        high=high,
        low=low,
        clean_ret=clean_ret,
        masks=masks,
        market_prices=market_prices,
        industry_map=industry_map,
        margin=margin,
        moneyflow=moneyflow,
        northbound=northbound,
        institution=institution,
        circ_mv=circ_mv,
        total_mv=total_mv,
        factor_names=factor_names,
        walk_forward_hmm=walk_forward_hmm,
        include_regime=include_regime,
    )))
    logger.info(f"因子库就绪: {len(registry)} 个因子")
    return registry


def get_factor_names(
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
    circ_mv: pd.DataFrame = None,
    total_mv: pd.DataFrame = None,
    factor_names: list | set | None = None,
    walk_forward_hmm: bool = False,
    compute: bool = False,
    include_regime: bool = False,
) -> list:
    """
    返回当前数据条件下可计算的因子名列表。

    用于流式 IC 等"先知道有哪些因子、再逐个计算"的场景，避免为了拿名字
    而一次性构建全部面板。

    参数与 ``get_factor_registry`` / ``iter_factor_registry`` 一致。
    include_regime 已退役（不再枚举 ``市场*`` / ``HMM_*``）。

    compute: False（默认）时仅基于数据可用性 + factor_names 白名单枚举候选名
             （不触发实际面板计算，快）；True 时通过 ``iter_factor_registry``
             实跑一遍收集名字（严格准确但耗时高，仅在需要严格一致时使用）。
    """
    if compute:
        return [
            name for name, _ in _filter_none_emit(iter_factor_registry(
                prices=prices, financial=financial, prices_raw=prices_raw,
                volume=volume, amount=amount, open_=open_, high=high, low=low,
                clean_ret=clean_ret, masks=masks, market_prices=market_prices,
                industry_map=industry_map, margin=margin, moneyflow=moneyflow,
                northbound=northbound, institution=institution,
                circ_mv=circ_mv, total_mv=total_mv,
                factor_names=factor_names, walk_forward_hmm=walk_forward_hmm,
                include_regime=include_regime,
            ))
        ]

    # 轻量枚举：基于数据可用性 + factor_names 白名单列出候选因子名。
    fn_set = set(factor_names) if factor_names is not None else None
    names: list = []

    def _add(name):
        if _want_factor(name, fn_set):
            names.append(name)

    if _section_needed(_PRICE_FACTOR_NAMES, fn_set):
        for n in ("动量_3d", "动量_10d", "动量_20d", "动量_40d", "动量_60d",
                  "动量_120d", "动量_skip", "反转_3d", "反转_5d", "反转_10d",
                  "反转_20d", "波动率_20d", "波动率_60d", "52周新高",
                  "分数差分动量_20d"):
            _add(n)
        if volume is not None:
            _add("换手率_20d")
        if amount is not None:
            _add("Amihud_20d")
        if high is not None and low is not None:
            _add("振幅_20d")

    if financial is not None and not financial.empty and _section_needed(_FIN_FACTOR_NAMES, fn_set):
        for n in ("价值_PB", "价值_EP", "质量_ROE", "质量_ROE变化", "规模", "杠杆"):
            _add(n)
        fin_cols = set(financial.columns)
        if "gross_profit_margin" in fin_cols:
            _add("质量_毛利率")
            _add("质量_毛利率变化")
        if (
            {"net_profit", "operating_cashflow", "total_assets"} <= fin_cols
            or {"eps", "operating_cashflow"} <= fin_cols
        ):
            _add("质量_应计项目")
        if "revenue_growth" in fin_cols:
            _add("营收增长率")
        if "net_profit_growth" in fin_cols:
            _add("净利增长率")
        if "eps" in fin_cols:
            _add("EPS增长率")

    # 事件 overlay 因子（业绩预告等）不进入 ML 截面输入枚举；
    # 见 EVENT_OVERLAY_FACTOR_NAMES 注释 + get_event_overlay_factors()。

    if _section_needed(_ALPHA2_FACTOR_NAMES, fn_set):
        if industry_map is not None:
            _add("行业动量_20d")
        if market_prices is not None:
            _add("特质波动率_60d")
        if margin is not None:
            _add("融资余额变化_20d")
        if moneyflow is not None:
            _add("大单净流入_5d")
            _add("大单残差净流入_5d")
        # 北向下线：不加入 get_factor_names 默认枚举
        if institution is not None:
            _add("机构持仓变化")

    if _section_needed(_ASHARE_CS_FACTOR_NAMES, fn_set):
        for n in _ASHARE_CS_FACTOR_NAMES:
            _add(n)

    # 市场*/HMM_* 广播特征已退役，不再枚举（仓位体制见 backtest/regime.py）
    _ = include_regime

    if _section_needed(_TECH_FACTOR_NAMES, fn_set):
        for n in ("BIAS_5d", "BIAS_20d", "PSY_12d", "AR_26d", "BR_26d",
                  "换手率加速度", "换手率行业中性_20d", "行业相对强度_20d"):
            _add(n)

    need_wq = fn_set is None or any(n.startswith("WQ_") for n in fn_set)
    if need_wq:
        try:
            from factors.factor_alpha101 import ALPHA101_NAMES
            for n in ALPHA101_NAMES:
                if fn_set is None or n in fn_set:
                    names.append(n)
        except Exception:
            # alpha101 模块未导入或无 ALPHA101_NAMES 常量时跳过
            pass

    need_a158 = fn_set is None or any(n.startswith("A158_") for n in fn_set)
    if need_a158:
        try:
            from factors.factor_alpha158 import ALPHA158_NAMES
            for n in ALPHA158_NAMES:
                if fn_set is None or n in fn_set:
                    names.append(n)
        except Exception:
            pass

    need_gtja = fn_set is None or any(n.startswith("GTJA_") for n in fn_set)
    if need_gtja:
        try:
            from factors.factor_alpha191 import ALPHA191_NAMES, SKIP_NAMES
            for n in ALPHA191_NAMES:
                if n in SKIP_NAMES:
                    continue
                if fn_set is None or n in fn_set:
                    names.append(n)
        except Exception:
            pass

    if (masks is not None and masks.get("limit_up") is not None
            and _section_needed(_LIMIT_FACTOR_NAMES, fn_set)):
        for n in ("涨停强度_20d", "跌停弱势_20d", "连板数",
                  "涨跌停净强度_20d", "涨跌停状态", "开板反转_5d"):
            _add(n)

    if _section_needed(_SMALLCAP_FACTOR_NAMES, fn_set):
        for n in _SMALLCAP_FACTOR_NAMES:
            _add(n)

    if _section_needed(_SIZE_ALPHA_FACTOR_NAMES, fn_set):
        from factors.factor_size_alpha import mcap_data_available
        if mcap_data_available(circ_mv=circ_mv, total_mv=total_mv, financial=financial):
            for n in _SIZE_ALPHA_FACTOR_NAMES:
                _add(n)

    if _section_needed(_OPENSOURCE_AP_FACTOR_NAMES, fn_set):
        fin_cols = set(financial.columns) if financial is not None and not financial.empty else set()
        if "total_assets" in fin_cols:
            _add("资产增长")
            _add("资产市值比")
        if "operating_cashflow" in fin_cols:
            _add("现金流市值比")
            _add("现金流价格波动")
        if "bvps" in fin_cols:
            _add("权益增长")
        if "eps" in fin_cols:
            _add("盈利一致性")
            _add("盈利连增期数")
        if "revenue_growth" in fin_cols:
            _add("营收增长秩")
        if {"eps", "operating_cashflow"} <= fin_cols:
            _add("应计占比")
        if {"bvps", "total_assets"} <= fin_cols:
            _add("权益变化资产比")
        if {"gross_profit_margin", "eps", "net_profit_margin", "total_assets"} <= fin_cols:
            _add("毛利资产比")
        if {"eps", "net_profit_margin"} <= fin_cols:
            _add("营收市值比")
        if {"debt_ratio", "total_assets"} <= fin_cols:
            _add("净债务市值比")
            _add("综合债务融资")
        # Batch-3 accounting（现有字段；严格 BS 版见覆盖文档跳过项）
        if {"eps", "operating_cashflow", "total_assets"} <= fin_cols:
            _add("应计资产比")
        if {"gross_profit_margin", "eps", "net_profit_margin", "bvps"} <= fin_cols:
            _add("经营利润权益比")
        if {"debt_ratio", "total_assets"} <= fin_cols:
            _add("外部融资资产比")
        if {"eps", "net_profit_margin", "total_assets"} <= fin_cols:
            _add("资产周转变化")
        if {"gross_profit_margin", "eps", "net_profit_margin", "total_assets"} <= fin_cols:
            _add("经营杠杆")
        # 股本 / 上市 / 价量（数据文件存在即枚举；计算时再惰性加载）
        from config.settings import RAW_DIR as _RAW
        if (_RAW / "share_change.parquet").exists():
            _add("一年股本扩张")
            _add("五年股本扩张")
        if circ_mv is not None or total_mv is not None or (
            financial is not None and not financial.empty
        ):
            _add("综合股权融资")
        # 上市年龄：有 list_date 用 list_date，否则回退首次有效价
        _add("上市年龄")
        if clean_ret is not None:
            _add("年龄动量")
            _add("月最大收益")
            _add("收益偏度")
            _add("季节动量")
            if market_prices is not None:
                _add("协偏度")
                _add("残差动量")
        if industry_map is not None or (_RAW / "industry_map.parquet").exists():
            _add("行业集中度")

    return names


def get_event_overlay_factor_names(
    factor_names: list | set | None = None,
) -> list:
    """返回事件 overlay 因子名列表（默认**不进入** ML / IC 截面候选池）。

    与 `get_factor_names` 互斥：事件因子稀疏（仅公告日 + window 日 ffill），
    不适合截面 winsorize+zscore 标准化与默认 ML 截面排序输入。独立枚举供
    ``--special-factors event``（strategies/ml.py post-merge）或选股端事件叠加/过滤。
    """
    fn_set = set(factor_names) if factor_names is not None else None
    return [n for n in EVENT_OVERLAY_FACTOR_NAMES
            if fn_set is None or n in fn_set]


def get_event_overlay_factors(
    prices: pd.DataFrame,
    factor_names: list | set | None = None,
) -> dict:
    """计算事件 overlay 因子（默认**不进入** ML / IC 截面候选池）。

    返回 dict: {因子名: DataFrame(index=date, columns=stock)}。
    因子值保持事件信号原尺度（不做截面 winsorize+zscore），由调用方按事件
    overlay 语义自行处理。``--special-factors event`` 开启时由
    ``build_factor_dataset`` / ``special_factors.inject`` 在白名单过滤后 merge。

    Parameters
    ----------
    prices : 复权价 DataFrame（用于对齐日历 + 截面股票池）
    factor_names : 可选白名单，None 时计算全部事件 overlay 因子
    """
    overlay: dict = {}
    wanted = set(factor_names) if factor_names is not None else None
    if not _section_needed(EVENT_OVERLAY_FACTOR_NAMES, wanted):
        return overlay
    try:
        from factors.factor_event import factor_yjyg
        if _want_factor("业绩预告_超预期", wanted):
            f = factor_yjyg(prices)
            if f is not None and not f.isna().all(axis=None):
                if not f.index.equals(prices.index):
                    f = f.reindex(prices.index)
                overlay["业绩预告_超预期"] = f
    except Exception as e:
        logger.warning(f"事件 overlay 因子（业绩预告）计算失败: {e}")
    try:
        from factors.factor_ashare import factor_yjbb_surprise_raw
        if _want_factor("业绩快报超预期", wanted):
            f = factor_yjbb_surprise_raw(prices)
            if f is not None and not f.isna().all(axis=None):
                if not f.index.equals(prices.index):
                    f = f.reindex(prices.index)
                overlay["业绩快报超预期"] = f
    except Exception as e:
        logger.warning(f"事件 overlay 因子（业绩快报）计算失败: {e}")
    return overlay


def compute_single_factor(
    name: str,
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
    circ_mv: pd.DataFrame = None,
    total_mv: pd.DataFrame = None,
    walk_forward_hmm: bool = False,
    include_regime: bool = False,
) -> pd.DataFrame | None:
    """
    按因子名单独计算一个因子面板，返回 DataFrame（已 reindex 到 prices.index）或 None。

    用于 IC 下游按需重建小子集 registry（Barra 残差化、行业 IC、select_factors
    相关性去冗余等只用到少量因子的场景），避免为拿一两个因子而构建全部面板。

    实现通过 ``iter_factor_registry`` 流式生成、命中目标 name 即返回；未命中返回 None。
    内存代价仅为该单个因子面板。

    **磁盘缓存**：结果会落盘到 ``<DATA_ROOT>/processed/factor_panels/``，
    key=因子名。后续对同一 (name, 输入数据指纹) 的调用直接从 parquet 读取，
    跳过重算（IC 管线 Stage 2/3/5/7/8 共 4-5 遍重算 → 单遍）。
    输入指纹覆盖 prices/clean_ret/financial 等的 shape + index 首尾 +
    financial 列集 + 末报告期，任一变动即自动失效重算。
    环境变量 ``FACTOR_CACHE_DISABLE=1`` 可强制跳过缓存（调试 / 冷启动）。

    include_regime 默认 False：单因子重算通常针对 alpha 因子，无需重跑 HMM regime
    （省去 ~4min HMM 拟合开销）；如需计算 regime 因子本身，显式传 True。
    """
    kwargs = dict(
        prices=prices, financial=financial, prices_raw=prices_raw,
        volume=volume, amount=amount, open_=open_, high=high, low=low,
        clean_ret=clean_ret, masks=masks, market_prices=market_prices,
        industry_map=industry_map, margin=margin, moneyflow=moneyflow,
        northbound=northbound, institution=institution,
        circ_mv=circ_mv, total_mv=total_mv,
        walk_forward_hmm=walk_forward_hmm, include_regime=include_regime,
    )
    signature = build_input_signature(kwargs)

    def _real_compute(target_name: str):
        target = {target_name}
        # 直接走 raw 许算核（compute_single_factor_cached 已在上层管缓存，
        # 此处若调带缓存的 iter_factor_registry 会双重查盘）
        for n, panel in _filter_none_emit(_iter_factor_registry_raw(
            prices=prices, financial=financial, prices_raw=prices_raw,
            volume=volume, amount=amount, open_=open_, high=high, low=low,
            clean_ret=clean_ret, masks=masks, market_prices=market_prices,
            industry_map=industry_map, margin=margin, moneyflow=moneyflow,
            northbound=northbound, institution=institution,
            circ_mv=circ_mv, total_mv=total_mv,
            factor_names=target, walk_forward_hmm=walk_forward_hmm,
            include_regime=include_regime,
        )):
            if n == target_name:
                return panel
        return None

    return compute_single_factor_cached(name, _real_compute, signature)


def build_factor_registry_subset(
    names: list | set,
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
    circ_mv: pd.DataFrame = None,
    total_mv: pd.DataFrame = None,
    walk_forward_hmm: bool = False,
    include_regime: bool = False,
) -> dict:
    """
    按 ``names`` 子集构建因子 registry（dict），仅计算被请求的因子。

    用于 IC 下游需要同时持有一组因子面板的场景（如 select_factors 相关性去冗余、
    factor_corr_matrix、ic_decay_table），但只针对筛选后的小子集（通常 < 30 个），
    相比全量 registry 显著降内存。

    注：alpha101 已按白名单子集计算；technical/limit 等其它批量 section 仍可能
    整组计算再过滤，但被 factor_names 白名单限制后总体规模可控。


    include_regime 默认 False：IC 下游子集重建通常只针对 alpha 因子，无需重跑 HMM。
    """
    if not names:
        return {}
    return get_factor_registry(
        prices=prices, financial=financial, prices_raw=prices_raw,
        volume=volume, amount=amount, open_=open_, high=high, low=low,
        clean_ret=clean_ret, masks=masks, market_prices=market_prices,
        industry_map=industry_map, margin=margin, moneyflow=moneyflow,
        northbound=northbound, institution=institution,
        circ_mv=circ_mv, total_mv=total_mv,
        factor_names=set(names), walk_forward_hmm=walk_forward_hmm,
        include_regime=include_regime,
    )





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
