"""
factors/barra_risk.py  —  简化版 Barra CNE 风格因子

用途：
    在 ic_analysis.py --barra 中作为截面回归控制变量，
    计算"纯因子IC"（alpha因子剔除系统性风险敞口后的真实alpha信号）。
    不直接用于选股。

9个风格因子（参考 Barra CNE5/CNE6）：
    Barra_Size        LNCAP：log(流通市值)（无流通市值时用总市值）
    Barra_NonlinSize  非线性规模：Size² 对 Size 截面正交（捕捉中盘效应）
    Barra_Beta        对**中证全指**的加权回归 beta（半衰期 63 日指数加权）
    Barra_Momentum    RSTR：240 日窗口跳过最近 20 日，半衰期指数加权动量
    Barra_ResVol      HSIGMA：252 日口径的市场回归残差波动（特质风险）
    Barra_Value       账面市值比 B/P = 1/PB
    Barra_Liquidity   流动性：63 日 / 252 日**换手率**等权平均（STOQ + STOA）
    Barra_Leverage    财务杠杆 DTOA = 总负债 / 总资产（资产负债率）
    Barra_Growth      成长：营收 YoY 50% + 净利润 YoY 50%

钉死口径（勿在后续改动中漂移）
------------------------------
1. **Size = log(流通市值)**，缺流通市值才退到 log(总市值)。
   市值面板主源 = 东财 ``stock_value_em``（``download_stock_value_em``）；
   自算 shares×prices_raw 仅作校验/兜底。
   **不是** log(流通盘 / 总股本 比例)，也不再拿 total_assets 当主路径。
2. **Liquidity 用换手率**（成交量/流通股本 或 成交额/流通市值），不是成交量。
3. **Leverage 用单一 DTOA**（资产负债率），不做多杠杆口径合成。
4. **Growth = 营收 YoY 与净利润 YoY 各 50%**，两腿先各自截面 1% winsor + z-score
   再等权合成（量纲不同，必须先标准化再平均）。
5. **市场代理 = 中证全指**（``data/raw/csi_all.parquet``），由调用方传入。
6. Value 及其余未点名风格：沿用既有实现。

⚠️ 因子定义已于 2026-07 变更（Size/NonlinSize/Liquidity/Leverage/Growth/
   Beta/Momentum/ResVol 全部重写）。**旧的 Barra 纯 IC checkpoint 与
   `factor_panel_neut_*` 残差缓存必须清除后重跑**，否则新旧口径混用。

用法（只在 ic_analysis.py / run.py / strategies.ml 内部调用，不独立运行）：
    from factors.barra_risk import get_barra_factors, barra_regression_weights
    barra = get_barra_factors(prices, financial, market_prices,
                              circ_mv=circ_mv, total_mv=total_mv,
                              turnover_rate=turnover_rate)
    w = barra_regression_weights(prices, circ_mv=circ_mv, total_mv=total_mv)
"""
import numpy as np
import pandas as pd
from loguru import logger
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import RAW_DIR
from utils.pit_align import pit_pivot_ffill
# 与 factors/factor.py 共用同一套截面标准化工具，避免 ddof 不一致
# （factor.py 用 scipy.stats.zscore ddof=0；旧 Barra 私有 _cross_zscore 用 pandas
#  mean/std ddof=1，导致两套因子标准化口径不一致）。
from factors.factor import cross_sectional_zscore, winsorize as _cs_winsorize


# ── 默认参数（Barra CNE5 口径）────────────────────────────────────────────────

BETA_WINDOW = 252          # Beta / HSIGMA 有效回看窗口
BETA_HALFLIFE = 63         # Beta / HSIGMA 时间衰减半衰期（交易日）
MOMENTUM_WINDOW = 240      # RSTR 回看窗口
MOMENTUM_SKIP = 20         # RSTR 跳过最近 1 个月
MOMENTUM_HALFLIFE = 60     # RSTR 半衰期（≈窗口/4，同 CNE5 的 504/126 比例）
LIQUIDITY_WINDOWS = (63, 252)   # STOQ / STOA

# 磁盘缓存版本：改 Size/Liquidity/WLS/市值源等定义时 bump。
# 落盘：data/processed/factor_panels/barra_bundle_<hash>/
# 清缓存：删 barra_bundle_* 目录（或整 factor_panels/）；FACTOR_CACHE_DISABLE=1 跳过。
BARRA_CACHE_VERSION = "barra_v1"


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _normalize_barra(df: pd.DataFrame) -> pd.DataFrame:
    """Barra 因子出口标准化：winsorize(1%) → cross_sectional_zscore(clip=3σ)。

    与 factors/factor.py::_normalize 同口径（共用 winsorize + cross_sectional_zscore），
    消除旧私有 _cross_zscore（pandas ddof=1）与 factor.py（scipy ddof=0）的差异。
    """
    return cross_sectional_zscore(_cs_winsorize(df))


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


def _align_panel(df: pd.DataFrame | None,
                 prices: pd.DataFrame) -> pd.DataFrame | None:
    """把外部面板对齐到 prices 的 (index, columns)；空/None 返回 None。"""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out.columns = out.columns.astype(str)
    out = out.reindex(index=prices.index, columns=prices.columns.astype(str))
    out = out.apply(pd.to_numeric, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def _coverage(df: pd.DataFrame | None) -> float:
    if df is None or df.empty:
        return 0.0
    return float(df.notna().to_numpy().mean())


def pick_market_cap(
    prices: pd.DataFrame,
    circ_mv: pd.DataFrame | None = None,
    total_mv: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame | None, str]:
    """选定市值面板：**优先流通市值**，覆盖率不足才退到总市值。

    返回 ``(市值面板, 来源名)``；两者都不可用时返回 ``(None, "")``。
    不做逐格混填——同一截面混用流通/总市值会制造伪截面差异。
    """
    circ = _align_panel(circ_mv, prices)
    total = _align_panel(total_mv, prices)
    cov_c, cov_t = _coverage(circ), _coverage(total)

    if circ is not None and cov_c > 0 and cov_c >= 0.5 * cov_t:
        return circ.where(circ > 0), "circ_mv"
    if total is not None and cov_t > 0:
        if circ is not None:
            logger.warning(
                f"Barra: 流通市值覆盖率 {cov_c:.1%} 明显低于总市值 {cov_t:.1%}，"
                "改用总市值（口径为 log(总市值)）"
            )
        return total.where(total > 0), "total_mv"
    if circ is not None and cov_c > 0:
        return circ.where(circ > 0), "circ_mv"
    return None, ""


def barra_regression_weights(
    prices: pd.DataFrame,
    circ_mv: pd.DataFrame | None = None,
    total_mv: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """Barra 截面回归权重面板 = **√市值**（优先流通市值）。

    用于 IC 纯化（``research/ic/barra.py``）与 ML 特征中性化
    （``models/wf/labels.py::residualize_panel``）的 WLS。返回 None 时
    调用方须退化为等权 OLS 并 warning。
    """
    mv, src = pick_market_cap(prices, circ_mv=circ_mv, total_mv=total_mv)
    if mv is None:
        logger.warning(
            "Barra 回归权重：无 circ_mv/total_mv 面板 → 截面回归退化为等权 OLS"
        )
        return None
    logger.info(f"Barra 回归权重: √{src}（WLS）")
    return np.sqrt(mv).astype(np.float32)


def market_return(
    market_prices: pd.DataFrame | pd.Series | None,
    mkt_clean_ret: pd.Series | None = None,
) -> pd.Series | None:
    """从市场指数面板取**收盘价**日收益 Series（中证全指）。

    ``csi_all.parquet`` 是 OHLCV 5 列表，直接 ``.squeeze()`` 拿到的仍是
    DataFrame——旧实现据此算 rolling cov 会退化成 pairwise 结果，Beta/ResVol
    因此失真。这里显式取 ``close``（无该列时取最后一列数值列）。
    """
    if mkt_clean_ret is not None:
        return pd.Series(mkt_clean_ret).astype(float)
    if market_prices is None:
        return None
    mp = market_prices
    if isinstance(mp, pd.DataFrame):
        if mp.shape[1] == 1:
            s = mp.iloc[:, 0]
        else:
            col = next(
                (c for c in mp.columns
                 if str(c).lower() in ("close", "收盘", "收盘价", "adj_close")),
                None,
            )
            if col is None:
                num = mp.select_dtypes("number")
                if num.shape[1] == 0:
                    return None
                col = num.columns[-1]
                logger.warning(
                    f"市场指数面板无 close 列，退用 {col!r} 算市场收益"
                )
            s = mp[col]
    else:
        s = mp
    s = pd.Series(s).astype(float)
    s.index = pd.to_datetime(s.index)
    return s.sort_index().pct_change()


def _ewm_market_regression(
    stock_ret: pd.DataFrame,
    mkt_ret: pd.Series,
    halflife: int = BETA_HALFLIFE,
    window: int = BETA_WINDOW,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """指数加权的个股-市场回归，一次算出 Beta 与 HSIGMA。

    模型（每只股票、每个交易日的截至当日加权回归）::

        r_i,t = α_i + β_i · r_m,t + ε_i,t
        权重 w_s ∝ 0.5^((t-s)/halflife)

    用加权一阶/二阶矩闭式解，避免逐日 rolling-apply 慢循环：
        β      = cov_w(r_i, r_m) / var_w(r_m)
        HSIGMA = sqrt(var_w(r_i) - β² · var_w(r_m))     （含截距的 OLS 恒等式）

    半衰期 63 日在 252 日处权重衰减到 0.5⁴ ≈ 6%，等价于 CNE5「252 日窗口 +
    63 日半衰期」的口径。

    市场收益先按个股缺失（涨跌停日 ``clean_ret`` = NaN、停牌）逐格 mask，
    保证 β 与 HSIGMA 的加权矩来自**同一组有效交易日**。

    ``window`` 用于**有效性掩码**：``ewm`` 记忆无限长，退市/长期停牌后旧值
    会被无限期 carry forward；这里要求过去 ``window`` 个交易日内至少有
    ``window//2`` 个有效收益，否则置 NaN（等价于旧 rolling 的 ``min_periods``）。
    """
    common = stock_ret.index.intersection(mkt_ret.index)
    ri = stock_ret.loc[common].astype(np.float64)
    rm_s = mkt_ret.loc[common].astype(np.float64)

    # 市场收益广播成同形面板并按个股缺失 mask（矩的样本集必须一致）
    rm = pd.DataFrame(
        np.repeat(rm_s.to_numpy()[:, None], ri.shape[1], axis=1),
        index=ri.index, columns=ri.columns,
    ).where(ri.notna())
    ri = ri.where(rm.notna())

    min_obs = max(2, window // 2)
    # 有限窗口有效性掩码：ewm 无限记忆，退市后旧 beta 会被永久 carry forward
    enough = ri.notna().rolling(window, min_periods=1).sum() >= min_obs

    # 全市场面板很大（T×N），逐步计算并及时释放中间量，控制峰值内存
    kw = dict(halflife=float(halflife), min_periods=min_obs, ignore_na=False)
    e_i = ri.ewm(**kw).mean()
    e_m = rm.ewm(**kw).mean()

    var_i = ((ri * ri).ewm(**kw).mean() - e_i * e_i).clip(lower=0)
    var_m = ((rm * rm).ewm(**kw).mean() - e_m * e_m).clip(lower=0)
    cov_im = (ri * rm).ewm(**kw).mean() - e_i * e_m
    del ri, rm, e_i, e_m

    beta = cov_im / var_m.replace(0, np.nan)
    beta = beta.replace([np.inf, -np.inf], np.nan).where(enough)
    del cov_im

    hsigma = np.sqrt((var_i - beta.pow(2) * var_m).clip(lower=0))
    hsigma = hsigma.replace([np.inf, -np.inf], np.nan).where(enough)
    del var_i, var_m, enough

    return beta.astype(np.float32), hsigma.astype(np.float32)


def _halflife_weighted_mean(
    df: pd.DataFrame,
    window: int,
    halflife: float,
    min_frac: float = 0.5,
    col_chunk: int = 512,
) -> pd.DataFrame:
    """有限窗口的半衰期指数加权均值（向量化卷积，无 Python 逐日循环）。

    ``out[t, i] = Σ_{d=0}^{W-1} λ^d · x[t-d, i] / Σ_{d: x[t-d,i] 有效} λ^d``，
    其中 ``λ = 0.5^(1/halflife)``。

    用 ``fftconvolve`` 沿时间轴做卷积；分母对**有效观测的权重**求和，
    因此涨跌停/停牌造成的 NaN 不会被当成 0 收益拉低动量——这正是 A 股
    ``clean_ret`` 场景下必须用加权**均值**而非加权和的原因（最强势的股票
    恰恰在涨停日被置 NaN）。有效权重占比低于 ``min_frac`` 的格子置 NaN。

    按列分块做卷积，把 FFT 峰值内存压到 ``col_chunk`` 列规模。
    """
    from scipy.signal import fftconvolve

    lam = 0.5 ** (1.0 / float(halflife))
    w = (lam ** np.arange(int(window))).astype(np.float32)
    w_total = float(w.sum())

    arr = df.to_numpy(dtype=np.float32, copy=False)
    n_rows, n_cols = arr.shape
    out = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    kernel = w[:, None]

    for lo in range(0, n_cols, col_chunk):
        hi = min(lo + col_chunk, n_cols)
        block = arr[:, lo:hi]
        valid = np.isfinite(block)
        x = np.where(valid, block, 0.0).astype(np.float32)
        num = fftconvolve(x, kernel, mode="full", axes=0)[:n_rows]
        den = fftconvolve(valid.astype(np.float32), kernel, mode="full",
                          axes=0)[:n_rows]
        ok = den >= min_frac * w_total
        with np.errstate(divide="ignore", invalid="ignore"):
            res = np.where(ok, num / np.where(den > 0, den, np.nan), np.nan)
        out[:, lo:hi] = res

    return pd.DataFrame(out, index=df.index, columns=df.columns)


# ── 各因子实现 ────────────────────────────────────────────────────────────────

def barra_size(
    prices: pd.DataFrame,
    circ_mv: pd.DataFrame | None = None,
    total_mv: pd.DataFrame | None = None,
    financial: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """规模 LNCAP：**log(流通市值)**，缺流通市值时退到 log(总市值)。

    口径钉死：这里的「log 市值/流通」= ``log(流通市值)``，
    **不是** ``log(流通市值 / 总市值)`` 之类的流通盘比例。

    两个市值面板都缺失时，才 warning 降级到 ``log(total_assets)``——
    总资产不是市值，量纲与截面结构都不同，仅为避免流程中断，
    不应作为常规路径。
    """
    mv, src = pick_market_cap(prices, circ_mv=circ_mv, total_mv=total_mv)
    if mv is not None:
        logger.info(f"Barra_Size = log({src})  覆盖率={_coverage(mv):.1%}")
        return _normalize_barra(np.log(mv))

    if financial is not None and "total_assets" in getattr(financial, "columns", []):
        logger.warning(
            "Barra_Size: 无 circ_mv/total_mv 市值面板，降级为 log(total_assets)。"
            "总资产≠市值，请尽快跑 `python -m data.download_stock_value_em` 补市值面板"
        )
        ta = _pivot_ffill(financial, "total_assets", prices.index)
        return _normalize_barra(np.log(ta.where(ta > 0)))

    logger.warning("Barra_Size: 无市值面板也无 total_assets，跳过")
    return None


def barra_nonlin_size(size_df: pd.DataFrame | None,
                      power: float = 2.0) -> pd.DataFrame | None:
    """非线性规模：**Size² 在每个截面正交化掉线性 Size**。

    输入必须是新口径的 ``Barra_Size``（log 流通市值 z-score）。正交后
    保留的是市值的二次（中盘 vs 两端）成分。

    注：CNE5 原版用三次方（Size³）；此处按项目口径取平方，保持与线性
    Size 正交后仍是「中盘效应」代理。正交回归维持等权 OLS（单变量正交化，
    与 WLS 差异极小，且历史行为不变）。
    """
    if size_df is None:
        return None
    nonlin = size_df ** power
    result = pd.DataFrame(np.nan, index=size_df.index, columns=size_df.columns)

    for date in size_df.index:
        s = size_df.loc[date].dropna()
        nl = nonlin.loc[date].reindex(s.index).dropna()
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

    return _normalize_barra(result)


def barra_beta(prices: pd.DataFrame, market_prices: pd.DataFrame,
               window: int = BETA_WINDOW,
               clean_ret: pd.DataFrame | None = None,
               mkt_clean_ret: pd.Series | None = None,
               halflife: int = BETA_HALFLIFE) -> pd.DataFrame:
    """Beta：对**中证全指**的半衰期加权回归 beta。

    市场代理由调用方传入（``data/raw/csi_all.parquet`` = 中证全指）。
    时间权重为半衰期 ``halflife``（默认 63 日）的指数衰减，即加权最小二乘；
    ``window`` 决定预热样本数（``min_periods = window // 2``）。

    优先使用 clean_ret（涨跌停日 return=NaN），避免涨跌停日强制 ±10%/±20%
    截断污染个股 beta 与市场相关系数。回退到 prices.pct_change() 以兼容旧调用。
    """
    stock_ret = clean_ret if clean_ret is not None else prices.pct_change()
    mkt_ret = market_return(market_prices, mkt_clean_ret)
    beta_df, _ = _ewm_market_regression(
        stock_ret, mkt_ret, halflife=halflife, window=window,
    )
    return _normalize_barra(beta_df)


def barra_momentum(prices: pd.DataFrame,
                   long_window: int = MOMENTUM_WINDOW,
                   skip_window: int = MOMENTUM_SKIP,
                   clean_ret: pd.DataFrame | None = None,
                   halflife: float = MOMENTUM_HALFLIFE) -> pd.DataFrame:
    """动量 RSTR：240 日窗口跳过最近 20 日，**半衰期指数衰减加权**。

    RSTR = 对数收益 ``log(1+r)`` 在 [t-skip-239, t-skip] 上的半衰期加权
    平均（越久远权重越小，半衰期默认 60 日 ≈ 窗口/4，与 CNE5 的 504/126
    同比例）。用 ``_halflife_weighted_mean`` 的卷积实现，无逐日 apply 循环。

    用 clean_ret 而非 prices.pct_change()：涨跌停日收益被置 NaN，加权平均
    的分母只累计有效日权重，避免把涨停日当 0 收益稀释动量。
    """
    ret = clean_ret if clean_ret is not None else prices.pct_change()
    # log(1+r)：r<=-1（退市/异常）截断到 -99.9% 防 -inf
    log_ret = np.log1p(ret.clip(lower=-0.999))
    lagged = log_ret.shift(skip_window)
    rstr = _halflife_weighted_mean(lagged, long_window, halflife)
    return _normalize_barra(rstr)


def barra_res_vol(prices: pd.DataFrame, market_prices: pd.DataFrame,
                  window: int = BETA_WINDOW,
                  clean_ret: pd.DataFrame | None = None,
                  mkt_clean_ret: pd.Series | None = None,
                  halflife: int = BETA_HALFLIFE) -> pd.DataFrame:
    """残差波动率 HSIGMA：252 日口径下市场回归残差的加权标准差。

    与 ``barra_beta`` 同一个加权回归（半衰期 63 日、252 日有效窗口），
    HSIGMA = ``sqrt(var_w(r_i) - β²·var_w(r_m))``。替代旧的 60 日
    ``vol × sqrt(1-ρ²)`` 近似——旧口径窗口过短、且用相关系数近似残差。

    优先使用 clean_ret，避免涨跌停日强制截断高估与市场的相关系数。
    """
    stock_ret = clean_ret if clean_ret is not None else prices.pct_change()
    mkt_ret = market_return(market_prices, mkt_clean_ret)
    _, hsigma = _ewm_market_regression(
        stock_ret, mkt_ret, halflife=halflife, window=window,
    )
    return _normalize_barra(hsigma)


def barra_value(financial: pd.DataFrame, prices: pd.DataFrame,
                prices_raw: pd.DataFrame | None = None) -> pd.DataFrame | None:
    """价值：B/P = 1/PB

    优先用 `pb` 列直接取 1/PB；若财务表无 `pb` 但有 `bvps`（每股净资产）且
    调用方提供了 `prices_raw`（不复权股价），则按 `bvps / price_raw` 计算
    B/P（与 factors/factor.py::factor_value_pb 同口径，PIT 对齐 bvps 后除以
    当日不复权价）。两者都不可用则跳过。
    """
    if "pb" in financial.columns:
        pb = _pivot_ffill(financial, "pb", prices.index)
        bp = 1.0 / pb.replace(0, np.nan)
        return _normalize_barra(bp)

    if "bvps" in financial.columns and prices_raw is not None:
        bvps = _pivot_ffill(financial, "bvps", prices_raw.index)
        price = prices_raw.reindex(columns=bvps.columns)
        bp = bvps / price.replace(0, np.nan)
        bp = bp.replace([np.inf, -np.inf], np.nan)
        logger.info("Barra_Value 用 bvps / prices_raw 计算 B/P（同 factor_value_pb）")
        return _normalize_barra(bp)

    logger.warning("Barra_Value: 无 pb/bvps+prices_raw，跳过")
    return None


def resolve_turnover(
    prices: pd.DataFrame,
    turnover_rate: pd.DataFrame | None = None,
    amount: pd.DataFrame | None = None,
    circ_mv: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame | None, str]:
    """解析日频换手率面板，返回 ``(换手率, 来源)``。

    优先级：
      1. ``turnover_rate.parquet``（``data/compute_market_cap.py`` 产出；
         与 Size 主路径解耦，仍依赖 shares×volume；
         = 成交量×100 / 流通股本；PIT 安全）
      2. ``amount / circ_mv``（成交额 / 流通市值，等价口径）

    单位（小数 vs 百分比）无关紧要：下游取 log 再截面 z-score，
    常数倍缩放被 z-score 消掉。
    """
    tr = _align_panel(turnover_rate, prices)
    if tr is not None and _coverage(tr) > 0:
        return tr.where(tr > 0), "turnover_rate"

    amt = _align_panel(amount, prices)
    cmv = _align_panel(circ_mv, prices)
    if amt is not None and cmv is not None:
        ratio = amt / cmv.where(cmv > 0)
        ratio = ratio.replace([np.inf, -np.inf], np.nan)
        if _coverage(ratio) > 0:
            return ratio.where(ratio > 0), "amount/circ_mv"

    return None, ""


def barra_liquidity(
    prices: pd.DataFrame,
    turnover_rate: pd.DataFrame | None = None,
    amount: pd.DataFrame | None = None,
    circ_mv: pd.DataFrame | None = None,
    volume: pd.DataFrame | None = None,
    windows: tuple[int, ...] = LIQUIDITY_WINDOWS,
) -> pd.DataFrame | None:
    """流动性：**63 日与 252 日换手率等权平均**（≈ CNE5 的 STOQ + STOA）。

    每个窗口取 ``log(窗口内日均换手率)``（换手率右偏，取对数后截面分布近正态），
    各自 winsor(1%)+z-score 后**等权平均**，最后再标准化一次。
    等权平均前先标准化：63 日与 252 日均值的量纲/离散度不同，直接相加会让
    波动更大的短窗主导。

    换手率来源见 ``resolve_turnover``。**无换手率数据时不静默用 log 成交量**：
    会打 warning 明确说明已降级（成交量含股本规模信息，与 Size 高度共线）。
    """
    turnover, src = resolve_turnover(
        prices, turnover_rate=turnover_rate, amount=amount, circ_mv=circ_mv,
    )
    if turnover is None:
        vol = _align_panel(volume, prices)
        if vol is None:
            logger.warning(
                "Barra_Liquidity: 无 turnover_rate / amount+circ_mv / volume，跳过"
            )
            return None
        logger.warning(
            "Barra_Liquidity 降级：缺 turnover_rate 与 amount+circ_mv，"
            "改用 log(20日均成交量)。成交量未除流通股本，与 Barra_Size 高度共线，"
            "控制效果打折——请跑 `python -m data.compute_market_cap` 生成 "
            "turnover_rate.parquet（换手仍走自算；市值主路径见 download_stock_value_em）"
        )
        avg_vol = vol.rolling(20, min_periods=10).mean()
        return _normalize_barra(np.log(avg_vol.where(avg_vol > 0)))

    logger.info(f"Barra_Liquidity 换手率来源: {src}，窗口={windows}（等权平均）")
    legs = []
    for w in windows:
        avg = turnover.rolling(int(w), min_periods=max(2, int(w) // 2)).mean()
        legs.append(_normalize_barra(np.log(avg.where(avg > 0))))
    combined = sum(legs) / float(len(legs))
    return _normalize_barra(combined)


def barra_leverage(financial: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame | None:
    """杠杆 DTOA：**资产负债率 = 总负债 / 总资产**（单一口径，不做合成）。

    `debt_ratio` 为 0-100% 尺度，Barra 后续做截面 z-score 标准化，尺度对控制
    回归无影响，故不做缩放。财务表无现成比率时，用 总负债/总资产 自算。
    """
    for col in ("debt_to_assets", "debt_asset_ratio", "liabilities_to_assets",
                "debt_ratio"):
        if col in financial.columns:
            lev = _pivot_ffill(financial, col, prices.index)
            logger.info(f"Barra_Leverage = DTOA，使用列: {col}")
            return _normalize_barra(lev)

    liab_col = next(
        (c for c in ("total_liabilities", "total_liab", "liabilities")
         if c in financial.columns), None,
    )
    if liab_col is not None and "total_assets" in financial.columns:
        liab = _pivot_ffill(financial, liab_col, prices.index)
        assets = _pivot_ffill(financial, "total_assets", prices.index)
        dtoa = liab / assets.where(assets > 0)
        logger.info(f"Barra_Leverage = DTOA（{liab_col} / total_assets 自算）")
        return _normalize_barra(dtoa.replace([np.inf, -np.inf], np.nan))

    logger.warning("Barra_Leverage: 无资产负债率列，跳过")
    return None


# Growth 两腿的列名候选（各家数据源命名不一）
_REVENUE_YOY_COLS = ("revenue_growth", "revenue_yoy", "or_yoy",
                     "total_revenue_yoy", "operating_revenue_yoy")
_PROFIT_YOY_COLS = ("net_profit_growth", "netprofit_yoy", "net_profit_yoy",
                    "profit_yoy", "net_income_yoy")


def barra_growth(financial: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame | None:
    """成长：**营收 YoY 50% + 净利润 YoY 50%**。

    两腿各自先做截面 winsorize(1%) + z-score（与项目截面惯例一致），
    再等权合成、最后再标准化一次。先标准化再合成是必要的：净利润增速的
    截面离散度远大于营收增速（分母可接近 0），直接相加会让净利润腿吃掉
    全部权重。

    只有一腿可用时用单腿并 warning；两腿都缺则回退到 revenue.pct_change(252)。
    """
    legs: list[pd.DataFrame] = []
    used: list[str] = []

    for cols, label in ((_REVENUE_YOY_COLS, "营收YoY"),
                        (_PROFIT_YOY_COLS, "净利润YoY")):
        col = next((c for c in cols if c in financial.columns), None)
        if col is None:
            continue
        yoy = _pivot_ffill(financial, col, prices.index)
        legs.append(_normalize_barra(yoy))
        used.append(f"{label}={col}")

    if len(legs) == 2:
        logger.info(f"Barra_Growth = 0.5×{used[0]} + 0.5×{used[1]}")
        return _normalize_barra((legs[0] + legs[1]) / 2.0)

    if len(legs) == 1:
        logger.warning(
            f"Barra_Growth: 只找到一腿 {used[0]}，按单腿计算"
            "（营收 YoY 与净利润 YoY 应各占 50%）"
        )
        return legs[0]

    # fallback：从 revenue 自算同比（季度频率，所以近似用4个季报间隔）
    for col in ("revenue", "total_revenue", "or"):
        if col in financial.columns:
            pivot = _pivot_ffill(financial, col, prices.index)
            # 用252交易日近似1年同比（已 ffill，所以是滚动的）
            yoy = pivot.pct_change(252)
            logger.warning(f"Barra_Growth 用 {col}.pct_change(252) 近似同比（降级）")
            return _normalize_barra(yoy)

    logger.warning("Barra_Growth: 无营收/净利润增速列，跳过")
    return None


# ── Barra 磁盘缓存 ────────────────────────────────────────────────────────────

def _barra_input_sig(
    prices: pd.DataFrame,
    *,
    financial: pd.DataFrame | None,
    market_prices: pd.DataFrame | None,
    volume: pd.DataFrame | None,
    clean_ret: pd.DataFrame | None,
    industry_map: pd.Series | None,
    prices_raw: pd.DataFrame | None,
    circ_mv: pd.DataFrame | None,
    total_mv: pd.DataFrame | None,
    turnover_rate: pd.DataFrame | None,
    amount: pd.DataFrame | None,
) -> str:
    """Barra 输入指纹（轻量 shape/首尾 + 行业长度），供缓存键使用。"""
    import hashlib

    def _df_s(df: pd.DataFrame | None) -> str:
        if df is None or getattr(df, "empty", True):
            return "none"
        try:
            return (
                f"{df.shape[0]}x{df.shape[1]}_"
                f"{df.index[0]}_{df.index[-1]}_"
                f"{df.columns[0]}_{df.columns[-1]}"
            )
        except Exception:
            return f"{getattr(df, 'shape', '?')}"

    ind_s = "none"
    if industry_map is not None and len(industry_map):
        try:
            ind_s = f"{len(industry_map)}_{pd.Series(industry_map).nunique()}"
        except Exception:
            ind_s = str(len(industry_map))
    fin_s = "none"
    if financial is not None and not financial.empty:
        fin_s = f"{financial.shape}|{sorted(financial.columns.astype(str))[:12]}"
    raw = (
        f"{BARRA_CACHE_VERSION}|px:{_df_s(prices)}|fin:{fin_s}"
        f"|mkt:{_df_s(market_prices)}|vol:{_df_s(volume)}"
        f"|cret:{_df_s(clean_ret)}|raw:{_df_s(prices_raw)}"
        f"|cmv:{_df_s(circ_mv)}|tmv:{_df_s(total_mv)}"
        f"|to:{_df_s(turnover_rate)}|amt:{_df_s(amount)}|ind:{ind_s}"
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def _barra_cache_dir(sig: str) -> Path:
    from factors.factor_cache import FACTOR_CACHE_DIR
    return FACTOR_CACHE_DIR / f"barra_bundle_{sig}"


def _try_load_barra_bundle(cache_dir: Path) -> dict | None:
    import json
    import os as _os

    if _os.getenv("FACTOR_CACHE_DISABLE", "0") == "1":
        return None
    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        names = meta.get("names") or []
        if not names:
            return None
        out: dict = {}
        for name in names:
            pq = cache_dir / f"{name}.parquet"
            if not pq.exists():
                return None
            out[name] = pd.read_parquet(pq).astype(np.float32, copy=False)
        logger.info(f"Barra cache HIT: {cache_dir} ({len(out)} factors)")
        return out
    except Exception as e:
        logger.info(f"Barra cache MISS (read fail: {e}): {cache_dir}")
        return None


def _save_barra_bundle(cache_dir: Path, factors: dict) -> None:
    import json
    import os as _os

    if _os.getenv("FACTOR_CACHE_DISABLE", "0") == "1" or not factors:
        return
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        names = sorted(factors.keys())
        for name in names:
            panel = factors[name]
            if panel is None:
                continue
            tmp = cache_dir / f"{name}.tmp.parquet"
            final = cache_dir / f"{name}.parquet"
            panel.astype(np.float32, copy=False).to_parquet(tmp)
            _os.replace(tmp, final)
        meta_tmp = cache_dir / "meta.json.tmp"
        meta_tmp.write_text(
            json.dumps(
                {"version": BARRA_CACHE_VERSION, "names": names},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        _os.replace(meta_tmp, cache_dir / "meta.json")
        logger.info(f"Barra cache SAVE: {cache_dir} ({len(names)} factors)")
    except Exception as e:
        logger.warning(f"Barra cache SAVE fail: {e}")


# ── 汇总入口 ──────────────────────────────────────────────────────────────────

def get_barra_factors(
    prices: pd.DataFrame,
    financial: pd.DataFrame = None,
    market_prices: pd.DataFrame = None,
    volume: pd.DataFrame = None,
    clean_ret: pd.DataFrame | None = None,
    mkt_clean_ret: pd.Series | None = None,
    industry_map: pd.Series | None = None,
    prices_raw: pd.DataFrame | None = None,
    circ_mv: pd.DataFrame | None = None,
    total_mv: pd.DataFrame | None = None,
    turnover_rate: pd.DataFrame | None = None,
    amount: pd.DataFrame | None = None,
    use_cache: bool = True,
) -> dict:
    """
    构建所有可用的简化 Barra 风格因子。

    返回 dict: {因子名: DataFrame(index=date, columns=stock)}
    数值为截面 z-score（±3σ clip），可直接用于截面回归控制变量。

    同输入指纹下磁盘复用（``barra_bundle_<hash>/``）；清缓存见模块顶部
    ``BARRA_CACHE_VERSION`` 注释。``use_cache=False`` 或
    ``FACTOR_CACHE_DISABLE=1`` 强制重算。

    参数
    ----
    clean_ret : pd.DataFrame, optional
        涨跌停日 return=NaN 的日频收益（data.clean.clean_ohlcv 的输出）。
        传入后 Barra_Beta / Barra_ResVol / Barra_Momentum 用其替代
        prices.pct_change()，避免涨跌停截断污染。默认 None 时回退到 pct_change。
    mkt_clean_ret : pd.Series, optional
        市场指数的清洁收益（指数本身无涨跌停，一般可省）。默认 None 回退到
        market_prices.pct_change()。``market_prices`` 应为**中证全指**
        （``data/raw/csi_all.parquet``）。
    industry_map : pd.Series, optional
        index=stock, value=行业分类的截面映射。传入后对所有 Barra 因子做
        截面行业中性化（factor - 行业截面均值）再 z-score，剔除 Size/Liquidity/
        Leverage 等因子的行业残余成分，作为残差化控制变量更彻底。
        默认 None 时不动现有行为（向后兼容）。
    prices_raw : pd.DataFrame, optional
        不复权日频股价（data/raw/prices_raw.parquet）。Barra_Value 在财务表
        无 `pb` 列时，用 `bvps / prices_raw` 计算 B/P（与
        factors/factor.py::factor_value_pb 同口径）。
    circ_mv, total_mv : pd.DataFrame, optional
        日频流通市值 / 总市值面板（``data/raw/{circ_mv,total_mv}.parquet``，
        由 ``data/download_stock_value_em`` 产出；缺则可回退
        ``*_computed``）。**Barra_Size 的主路径**，优先 circ_mv。
        两者都缺才降级到 total_assets 并 warning。
    turnover_rate, amount : pd.DataFrame, optional
        日频换手率面板 / 成交额面板。``Barra_Liquidity`` 用换手率
        （turnover_rate 优先，其次 amount/circ_mv）；都缺才降级 log 成交量。
    use_cache : bool, default True
        是否读写磁盘 bundle 缓存。
    """
    cache_dir = None
    if use_cache:
        sig = _barra_input_sig(
            prices,
            financial=financial,
            market_prices=market_prices,
            volume=volume,
            clean_ret=clean_ret,
            industry_map=industry_map,
            prices_raw=prices_raw,
            circ_mv=circ_mv,
            total_mv=total_mv,
            turnover_rate=turnover_rate,
            amount=amount,
        )
        cache_dir = _barra_cache_dir(sig)
        cached = _try_load_barra_bundle(cache_dir)
        if cached is not None:
            return cached
        logger.info(f"Barra cache MISS: {cache_dir}")

    factors = _compute_barra_factors(
        prices,
        financial=financial,
        market_prices=market_prices,
        volume=volume,
        clean_ret=clean_ret,
        mkt_clean_ret=mkt_clean_ret,
        industry_map=industry_map,
        prices_raw=prices_raw,
        circ_mv=circ_mv,
        total_mv=total_mv,
        turnover_rate=turnover_rate,
        amount=amount,
    )
    if use_cache and cache_dir is not None:
        _save_barra_bundle(cache_dir, factors)
    return factors


def _compute_barra_factors(
    prices: pd.DataFrame,
    financial: pd.DataFrame = None,
    market_prices: pd.DataFrame = None,
    volume: pd.DataFrame = None,
    clean_ret: pd.DataFrame | None = None,
    mkt_clean_ret: pd.Series | None = None,
    industry_map: pd.Series | None = None,
    prices_raw: pd.DataFrame | None = None,
    circ_mv: pd.DataFrame | None = None,
    total_mv: pd.DataFrame | None = None,
    turnover_rate: pd.DataFrame | None = None,
    amount: pd.DataFrame | None = None,
) -> dict:
    """实际计算 Barra 9 风格（无缓存）。"""
    factors = {}

    # ── 规模（尽早计算，NonlinSize 依赖它）
    size_df = barra_size(
        prices, circ_mv=circ_mv, total_mv=total_mv, financial=financial,
    )
    if size_df is not None:
        factors["Barra_Size"] = size_df
        logger.info("计算 Barra_NonlinSize（Size² 正交 Size）...")
        nl = barra_nonlin_size(size_df)
        if nl is not None:
            factors["Barra_NonlinSize"] = nl

    # ── 需要市场指数的因子：Beta 与 HSIGMA 共用同一次加权回归
    if market_prices is not None:
        logger.info(
            f"计算 Barra_Beta / Barra_ResVol（中证全指，{BETA_WINDOW} 日窗口，"
            f"半衰期 {BETA_HALFLIFE} 日加权回归）..."
        )
        stock_ret = clean_ret if clean_ret is not None else prices.pct_change()
        mkt_ret = market_return(market_prices, mkt_clean_ret)
        if mkt_ret is None:
            logger.warning("Barra_Beta/ResVol: 市场指数无可用收盘价列，跳过")
        else:
            beta_raw, hsigma_raw = _ewm_market_regression(
                stock_ret, mkt_ret, halflife=BETA_HALFLIFE, window=BETA_WINDOW,
            )
            factors["Barra_Beta"] = _normalize_barra(beta_raw)
            factors["Barra_ResVol"] = _normalize_barra(hsigma_raw)
            del beta_raw, hsigma_raw

    # ── 动量（纯价格）
    logger.info(
        f"计算 Barra_Momentum（RSTR {MOMENTUM_WINDOW}/{MOMENTUM_SKIP}，"
        f"半衰期 {MOMENTUM_HALFLIFE} 日）..."
    )
    factors["Barra_Momentum"] = barra_momentum(prices, clean_ret=clean_ret)

    # ── 财务类因子
    if financial is not None:
        val = barra_value(financial, prices, prices_raw=prices_raw)
        if val is not None:
            factors["Barra_Value"] = val
        lev = barra_leverage(financial, prices)
        if lev is not None:
            factors["Barra_Leverage"] = lev
        gro = barra_growth(financial, prices)
        if gro is not None:
            factors["Barra_Growth"] = gro

    # ── 流动性（换手率）
    logger.info("计算 Barra_Liquidity（63/252 日换手率等权平均）...")
    liq = barra_liquidity(
        prices, turnover_rate=turnover_rate, amount=amount,
        circ_mv=circ_mv, volume=volume,
    )
    if liq is not None:
        factors["Barra_Liquidity"] = liq

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
            # 重新做截面 z-score，保持尺度一致（与 factor.py 同口径 ddof=0）
            factors[name] = cross_sectional_zscore(neutralized)
        logger.info("Barra 因子已完成行业中性化（截面去行业均值 + 重 z-score）")

    logger.info(f"Barra 风格因子就绪: {len(factors)} 个 → {list(factors.keys())}")
    return factors
