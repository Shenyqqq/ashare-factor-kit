"""
data/clean.py  —  原始数据清洗

在因子计算之前调用，清除数据源（AKShare/新浪财经）常见的错误值。
不修改磁盘上的原始 parquet，只在内存中返回干净的 DataFrame。

两层清洗体系：
  1. 本模块：绝对错误（零价、负资产、ROE=9999% 等数据源错误）
              + 涨跌停 mask（屏蔽日收益率中的虚假数据）
  2. factors/compute.py winsorize：截面相对异常值（极端但非错误的极值）

涨跌停清洗说明：
  涨跌停日的收盘价被强制锁定在涨跌停价，return 不反映真实供需。
  clean_ohlcv() 返回 clean_ret（涨跌停日 return 置为 NaN）+ masks 字典。
  因子计算中所有 rolling 的量价类计算必须用 clean_ret，
  而非直接 prices.pct_change()，否则波动率/动量等因子会被系统性低估。
"""
import os
import numpy as np
import pandas as pd
from loguru import logger


# ── 价格数据清洗 ──────────────────────────────────────────────────────────────

def _spike_mask(prices: pd.DataFrame) -> pd.DataFrame:
    """孤岛刺针 mask：当日价偏离前后两日均值 >3× 或 <1/3。"""
    neighbor_mean = (prices.shift(1) + prices.shift(-1)) / 2
    ratio = prices / neighbor_mean.replace(0, np.nan)
    return (ratio > 3.0) | (ratio < 1 / 3.0)


def clean_prices(
    prices: pd.DataFrame,
    label: str = "prices",
    *,
    ffill_limit: int | None = None,
    detect_spikes: bool = True,
) -> pd.DataFrame:
    """
    清洗日线价格 DataFrame（index=日期, columns=股票代码）。

    处理项:
      1. 去除重复日期（保留第一条）
      2. 零价、负价 → NaN
      3. 单日涨跌幅超过 ±100% → NaN（后复权价格几乎不可能，必为数据错误）
      4. "孤岛刺针"（可选）：某日价格是前后两日均值的 3 倍以上或 1/3 以下 → NaN
      5. 短缺口前向填充（默认关闭；传 ffill_limit>0 时启用）

    Parameters
    ----------
    ffill_limit : 前向填充最大交易日数；None/0 = 不 ffill（默认）。
    detect_spikes : 是否对本列独立做孤岛刺针。OHLC 联合清洗请用
        ``clean_ohlc_aligned``（以 close 判定刺针日，四价一并置 NaN）。
    """
    original_shape = prices.shape

    # 1. 去重日期
    if prices.index.duplicated().any():
        prices = prices[~prices.index.duplicated(keep="first")]

    # 2. 零价、负价
    prices = prices.where(prices > 0)

    # 3. 单日涨跌幅 >±100%（后复权价格不应出现，必为数据错误）
    daily_ret = prices.pct_change()
    prices = prices.where(daily_ret.abs() <= 1.0)

    # 4. 孤岛刺针（分列独立；OHLC 场景应关闭并改用 clean_ohlc_aligned）
    if detect_spikes:
        spike_mask = _spike_mask(prices)
        n_spikes = int(spike_mask.sum().sum())
        if n_spikes > 0:
            logger.warning(f"{label}: 发现 {n_spikes} 个孤岛刺针，已置为 NaN")
        prices = prices.where(~spike_mask)

    # 5. 短缺口前向填充（默认关闭，避免假成交日污染因子/回测）
    if ffill_limit is not None and ffill_limit > 0:
        prices = prices.ffill(limit=int(ffill_limit))

    n_nan = prices.isna().sum().sum()
    logger.info(
        f"{label} 清洗完成: {original_shape} → {prices.shape}，"
        f"剩余 NaN 格子 {n_nan}（停牌/退市正常）"
    )
    return prices.sort_index()


def clean_ohlc_aligned(
    close: pd.DataFrame,
    open_: pd.DataFrame | None = None,
    high: pd.DataFrame | None = None,
    low: pd.DataFrame | None = None,
    *,
    ffill_limit: int | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
]:
    """
    OHLC 联合清洗：基础校验分列做，孤岛刺针以 close 判定，刺针日四价一并 NaN。

    避免 open/high/low/close 分列独立刺针导致同 bar 部分价位被剜掉、
    涨跌停 mask / VWAP 口径错乱。
    """
    close_c = clean_prices(
        close, label="prices_hfq", ffill_limit=None, detect_spikes=False,
    )
    open_c = (
        clean_prices(open_, label="open_hfq", ffill_limit=None, detect_spikes=False)
        if open_ is not None else None
    )
    high_c = (
        clean_prices(high, label="high_hfq", ffill_limit=None, detect_spikes=False)
        if high is not None else None
    )
    low_c = (
        clean_prices(low, label="low_hfq", ffill_limit=None, detect_spikes=False)
        if low is not None else None
    )

    spike = _spike_mask(close_c)
    n_spikes = int(spike.sum().sum())
    if n_spikes > 0:
        logger.warning(
            f"OHLC 联合刺针: 以 close 判定 {n_spikes} 个刺针日，"
            f"当日 open/high/low/close 一并置 NaN"
        )
        close_c = close_c.where(~spike)

    # close 无效日（零价/±100%/刺针）→ 同 bar 的 open/high/low 一并 NaN
    valid = close_c.notna()
    if open_c is not None:
        v = valid.reindex(index=open_c.index, columns=open_c.columns).fillna(False)
        open_c = open_c.where(v)
    if high_c is not None:
        v = valid.reindex(index=high_c.index, columns=high_c.columns).fillna(False)
        high_c = high_c.where(v)
    if low_c is not None:
        v = valid.reindex(index=low_c.index, columns=low_c.columns).fillna(False)
        low_c = low_c.where(v)

    n_kill = int((~valid).sum().sum())
    if n_kill > 0 and n_spikes == 0:
        logger.info(
            f"OHLC 联合对齐: close 无效 {n_kill} 格，对应 open/high/low 已同步置 NaN"
        )

    if ffill_limit is not None and ffill_limit > 0:
        lim = int(ffill_limit)
        close_c = close_c.ffill(limit=lim)
        if open_c is not None:
            open_c = open_c.ffill(limit=lim)
        if high_c is not None:
            high_c = high_c.ffill(limit=lim)
        if low_c is not None:
            low_c = low_c.ffill(limit=lim)

    return close_c, open_c, high_c, low_c


def mask_post_delist(
    df: pd.DataFrame | None,
    delist_dates: dict[str, pd.Timestamp] | None,
) -> pd.DataFrame | None:
    """按 delist_date 将退市后行情置 NaN（date > delist_date）。"""
    if df is None or not delist_dates:
        return df
    out = df.copy()
    for code, d in delist_dates.items():
        if code not in out.columns or d is None or pd.isna(d):
            continue
        after = out.index > pd.Timestamp(d)
        if after.any():
            out.loc[after, code] = np.nan
    return out


def validate_amount_units(
    amount: pd.DataFrame,
    volume: pd.DataFrame,
    close: pd.DataFrame,
    *,
    lo: float | None = None,
    hi: float | None = None,
    strict: bool | None = None,
) -> float | None:
    """
    校验 amount / (volume × 100 × close) 截面中位数是否落在合理区间。

    A 股常见口径：amount=元，volume=手（100 股），则 ratio ≈ VWAP/close ≈ 1。
    偏离 [lo, hi] 时 WARNING；``AMOUNT_UNIT_STRICT=1`` 或 strict=True 时 raise。

    Returns
    -------
    float | None
        全局截面中位数；无法计算时返回 None。
    """
    if amount is None or volume is None or close is None:
        return None
    try:
        from config.settings import (
            AMOUNT_UNIT_STRICT,
            AMOUNT_UNIT_RATIO_LO,
            AMOUNT_UNIT_RATIO_HI,
        )
        if strict is None:
            strict = AMOUNT_UNIT_STRICT
        if lo is None:
            lo = AMOUNT_UNIT_RATIO_LO
        if hi is None:
            hi = AMOUNT_UNIT_RATIO_HI
    except Exception:
        if strict is None:
            strict = os.getenv("AMOUNT_UNIT_STRICT", "0").strip() in (
                "1", "true", "True", "yes",
            )
        if lo is None:
            lo = 0.5
        if hi is None:
            hi = 2.0

    common_idx = amount.index.intersection(volume.index).intersection(close.index)
    common_cols = amount.columns.intersection(volume.columns).intersection(close.columns)
    if len(common_idx) == 0 or len(common_cols) == 0:
        return None

    amt = amount.reindex(index=common_idx, columns=common_cols)
    vol = volume.reindex(index=common_idx, columns=common_cols)
    px = close.reindex(index=common_idx, columns=common_cols)
    denom = vol * 100.0 * px
    ratio = amt / denom.replace(0, np.nan)
    # 每日截面中位数，再对日期取中位数 → 稳健全局尺度
    daily_med = ratio.median(axis=1, skipna=True)
    med = float(daily_med.median(skipna=True))
    if np.isnan(med):
        logger.warning("amount 单位校验: 无法计算有效中位数（数据过稀）")
        return None

    msg = (
        f"amount 单位校验: median(amount/(volume×100×close))={med:.4f} "
        f"（期望约 1.0，合理区间 [{lo}, {hi}]）"
    )
    if lo <= med <= hi:
        logger.info(msg)
    else:
        logger.warning(msg + " — 可能 volume/amount 量纲不一致")
        if strict:
            raise ValueError(msg)
    return med


# ── 涨跌停 Mask ───────────────────────────────────────────────────────────────

def make_limit_mask(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    open_: pd.DataFrame = None,
    threshold_main: float = 0.095,   # 主板 ±10%，留 0.5% 容差
    threshold_star: float = 0.195,   # 科创板/创业板 ±20%，留 0.5% 容差
) -> dict:
    """
    构建涨跌停 mask 矩阵，供因子计算时屏蔽虚假数据。

    判断逻辑：
        涨停 = high == close  AND  日涨幅 ≥ threshold（股票代码区分主板/科创板）
        跌停 = low  == close  AND  日跌幅 ≤ -threshold
        用 high==close 而非纯比较涨幅，因为后复权涨幅计算存在累积误差，
        而涨停时收盘==最高这个等式几乎必然成立。

    返回 dict，键为以下 bool DataFrame（True = 该日该股处于对应状态）:
        limit_up        普通涨停（含一字板）
        limit_down      普通跌停（含一字板）
        any_limit       两者之一
        limit_up_open   一字涨停（开盘即封死，全天无法正常成交）
        limit_down_open 一字跌停
        broke_limit     开板（前日涨停 → 今日未涨停，供应压力释放）
    """
    # 科创板（688）、创业板（300/301）使用 ±20% 涨跌幅限制
    thresh_series = pd.Series(threshold_main, index=close.columns)
    for code in close.columns:
        if str(code).startswith(("688", "300", "301")):
            thresh_series[code] = threshold_star

    prev_close = close.shift(1)
    daily_ret = (close - prev_close) / prev_close.replace(0, np.nan)

    # 广播 threshold 到 (date, stock) 矩阵
    thresh_mat = pd.DataFrame(
        np.outer(np.ones(len(close)), thresh_series.values),
        index=close.index, columns=close.columns,
    )

    limit_up   = (high == close) & (daily_ret >=  thresh_mat * 0.98)
    limit_down = (low  == close) & (daily_ret <= -thresh_mat * 0.98)

    if open_ is not None:
        limit_up_open   = limit_up   & (open_ == close)
        limit_down_open = limit_down & (open_ == close)
    else:
        limit_up_open   = pd.DataFrame(False, index=close.index, columns=close.columns)
        limit_down_open = pd.DataFrame(False, index=close.index, columns=close.columns)

    # 开板：昨日涨停，今日未涨停
    broke_limit = limit_up.shift(1).fillna(False) & ~limit_up

    n_up   = int(limit_up.sum().sum())
    n_down = int(limit_down.sum().sum())
    n_one  = int(limit_up_open.sum().sum())
    logger.info(f"涨跌停 mask: 涨停={n_up:,} 条，跌停={n_down:,} 条，一字板={n_one:,} 条")

    return {
        "limit_up":        limit_up,
        "limit_down":      limit_down,
        "any_limit":       limit_up | limit_down,
        "limit_up_open":   limit_up_open,
        "limit_down_open": limit_down_open,
        "broke_limit":     broke_limit,
    }


def clean_ohlcv(
    close: pd.DataFrame,
    open_: pd.DataFrame = None,
    high: pd.DataFrame = None,
    low: pd.DataFrame = None,
) -> tuple:
    """
    对 OHLCV 数据做涨跌停清洗，返回 (clean_ret, masks)。

        clean_ret : DataFrame(index=日期, columns=股票)
                    日收益率，涨停/跌停日对应格子置为 NaN。
                    用于替代所有量价因子中的 prices.pct_change()。
                    rolling() 会自动跳过 NaN，不影响窗口长度计算。

        masks     : dict，同 make_limit_mask() 返回值，
                    供 factor_limit.py 直接使用，无需重复计算。

    当 open_/high/low 缺失时，退化为普通 pct_change()（无 mask），
    并返回空 masks，保证下游不因缺少 OHLCV 而崩溃。
    """
    if high is None or low is None:
        logger.warning("clean_ohlcv: 缺少 high/low，跳过涨跌停 mask，使用原始日收益率")
        clean_ret = close.pct_change()
        empty = pd.DataFrame(False, index=close.index, columns=close.columns)
        masks = {k: empty for k in
                 ("limit_up", "limit_down", "any_limit",
                  "limit_up_open", "limit_down_open", "broke_limit")}
        return clean_ret, masks

    masks = make_limit_mask(close, high, low, open_)
    clean_ret = close.pct_change()
    clean_ret[masks["any_limit"]] = np.nan
    logger.info(
        f"clean_ret 生成完成: shape={clean_ret.shape}，"
        f"屏蔽 {masks['any_limit'].sum().sum():,} 个涨跌停格子"
    )
    return clean_ret, masks


# ── 财务数据清洗 ──────────────────────────────────────────────────────────────

def clean_financial(fin: pd.DataFrame) -> pd.DataFrame:
    """
    清洗财务指标 DataFrame（列: trade_date, code, roe, bvps, total_assets,
    可选 eps / gross_profit_margin / debt_ratio / net_profit_growth /
    revenue_growth / net_profit_margin / operating_cashflow）。

    处理项:
      1. 去除重复 (trade_date, code) 行
      2. ROE 超出 ±300% → NaN（数据源录入错误，正常公司极少超过）
      3. 每股净资产 bvps ≤ 0 → NaN（负净资产导致 PB 计算无意义）
      4. 总资产 total_assets ≤ 0 → NaN（物理上不可能）
      5. 资产负债率 debt_ratio ∉ [0, 100] → NaN（百分比，超出必为录入错误）
      6. 毛利率 gpm ∉ [-100, 100] → NaN（极端负毛利罕见，>100% 必为错误）
      7. 主营收入增长率 revenue_growth ∉ [-1000, 1000] → NaN
         净利润增长率 net_profit_growth ∉ [-1000, 1000] → NaN
         （重组/扭亏可达数倍，但 1000% 以上多为数据错误）
      8. 每股收益 |eps| > 100 → NaN（A 股极少 eps 超 100 元）
      9. 未来数据防穿越：删除 trade_date > 今天的行（数据源偶发填错日期）

    所有可选列采用「存在才校验」原则，不影响旧数据加载。
    """
    n_before = len(fin)

    # 1. 去重
    fin = fin.drop_duplicates(subset=["trade_date", "code"], keep="last")

    def _clip(col: str, mask: pd.Series, desc: str):
        """存在列则按 mask 置 NaN 并告警。"""
        if col not in fin.columns:
            return
        n = int(mask.sum())
        if n > 0:
            logger.warning(f"financial: {desc} 共 {n} 行，已置 NaN")
            fin.loc[mask, col] = np.nan

    # 2. ROE 范围：[-300%, 300%]
    if "roe" in fin.columns:
        _clip("roe", fin["roe"].abs() > 300, "ROE 超出 ±300%")

    # 3. bvps ≤ 0
    if "bvps" in fin.columns:
        _clip("bvps", fin["bvps"] <= 0, "bvps ≤ 0")

    # 4. total_assets ≤ 0
    if "total_assets" in fin.columns:
        _clip("total_assets", fin["total_assets"] <= 0, "total_assets ≤ 0")

    # 5. debt_ratio ∉ [0, 100]
    if "debt_ratio" in fin.columns:
        _clip(
            "debt_ratio",
            (fin["debt_ratio"] < 0) | (fin["debt_ratio"] > 100),
            "debt_ratio 超出 [0, 100]",
        )

    # 6. gross_profit_margin ∉ [-100, 100]
    if "gross_profit_margin" in fin.columns:
        _clip(
            "gross_profit_margin",
            (fin["gross_profit_margin"] < -100) | (fin["gross_profit_margin"] > 100),
            "毛利率超出 [-100, 100]",
        )

    # 7. growth 阈值（保持既有口径，本次不做收紧）
    if "revenue_growth" in fin.columns:
        _clip(
            "revenue_growth",
            fin["revenue_growth"].abs() > 1000,
            "revenue_growth 超出 ±1000%",
        )
    if "net_profit_growth" in fin.columns:
        _clip(
            "net_profit_growth",
            fin["net_profit_growth"].abs() > 1000,
            "net_profit_growth 超出 ±1000%",
        )

    # 8. |eps| > 100
    if "eps" in fin.columns:
        _clip("eps", fin["eps"].abs() > 100, "|eps| > 100")

    # 9. 未来数据
    if "trade_date" in fin.columns:
        today = pd.Timestamp.today().normalize()
        future = pd.to_datetime(fin["trade_date"]) > today
        n_fut = int(future.sum())
        if n_fut > 0:
            logger.warning(f"financial: 删除 {n_fut} 行未来 trade_date")
            fin = fin.loc[~future]

    logger.info(f"financial 清洗完成: {n_before} → {len(fin)} 行")
    return fin


# ── volume / amount / aux / market_cap ───────────────────────────────────────

def clean_volume(
    volume: pd.DataFrame,
    name: str = "volume",
    spike_window: int = 20,
    spike_mult: float = 50.0,
) -> pd.DataFrame:
    """
    清洗成交量面板。

    处理项:
      1. 负值 / inf → NaN；0 保留（停牌语义）
      2. 突增检测：当日值 > 过去 spike_window 日均值 × spike_mult 时 warning
         （不自动置 NaN，避免误杀拆分等事件）
    """
    if volume is None or volume.empty:
        return volume
    out = volume.replace([np.inf, -np.inf], np.nan)
    neg = out < 0
    n_neg = int(neg.sum().sum())
    if n_neg > 0:
        logger.warning(f"{name}: {n_neg} 个负值格子已置 NaN")
        out = out.where(~neg)
    try:
        roll_mean = out.rolling(window=spike_window, min_periods=5).mean()
        spike_mask = out > (roll_mean * spike_mult)
        n_spike = int(spike_mask.sum().sum())
        if n_spike > 0:
            logger.warning(
                f"{name}: 发现 {n_spike} 个突增格子（>{spike_window}日均值的{spike_mult}倍），"
                f"仅告警不置 NaN"
            )
    except Exception as e:
        logger.debug(f"{name} 突增检测跳过: {e}")
    return out


def clean_amount(
    amount: pd.DataFrame,
    name: str = "amount",
    spike_window: int = 20,
    spike_mult: float = 50.0,
) -> pd.DataFrame:
    """
    清洗成交额面板，复用 clean_volume（amount 与 volume 同量纲同分布特征）。
    """
    return clean_volume(amount, name=name, spike_window=spike_window, spike_mult=spike_mult)


def clean_market_cap(
    df: pd.DataFrame,
    name: str = "total_mv",
    spike_window: int = 20,
    spike_mult: float = 10.0,
) -> pd.DataFrame:
    """
    清洗日频市值面板（index=trade_date, columns=股票代码，单位=元）。

    与 clean_volume 的差异：市值随股价日波动，单日 ±10% 是正常的，
    spike_mult 默认 10 倍（比 volume 的 50 倍更严格，因为市值时序更平滑）。

    处理项:
      1. 负值 / inf → NaN；0 → NaN（市值不应为 0）
      2. 突增检测：当日值 > 过去 spike_window 日均值 × spike_mult 时 warning
    """
    if df is None or df.empty:
        return df
    out = df.replace([np.inf, -np.inf], np.nan)
    bad = (out <= 0)
    n_bad = int(bad.sum().sum())
    if n_bad > 0:
        logger.warning(f"{name}: {n_bad} 个 ≤0 格子已置 NaN")
        out = out.where(~bad)
    try:
        roll_mean = out.rolling(window=spike_window, min_periods=5).mean()
        spike_mask = out > (roll_mean * spike_mult)
        n_spike = int(spike_mask.sum().sum())
        if n_spike > 0:
            logger.warning(
                f"{name}: 发现 {n_spike} 个突增格子（>{spike_window}日均值的{spike_mult}倍），"
                f"仅告警不置 NaN"
            )
    except Exception as e:
        logger.debug(f"{name} 突增检测跳过: {e}")
    return out


def clean_aux_panel(
    df: pd.DataFrame,
    name: str = "aux",
    spike_window: int = 20,
    spike_mult: float = 50.0,
) -> pd.DataFrame:
    """
    清洗资金流等辅助面板（允许负值，如净流入）。

    处理项:
      1. inf → NaN
      2. 突增检测：|当日值| > 过去 spike_window 日 |均值| × spike_mult 时
         warning（不置 NaN）
    """
    if df is None or df.empty:
        return df
    out = df.replace([np.inf, -np.inf], np.nan)
    try:
        abs_mean = out.abs().rolling(window=spike_window, min_periods=5).mean()
        spike_mask = out.abs() > (abs_mean * spike_mult)
        n_spike = int(spike_mask.sum().sum())
        if n_spike > 0:
            logger.warning(
                f"{name}: 发现 {n_spike} 个突增格子"
                f"（|当日|>{spike_window}日|均值|的{spike_mult}倍），"
                f"仅告警不置 NaN"
            )
    except Exception as e:
        logger.debug(f"{name} 突增检测跳过: {e}")
    return out
