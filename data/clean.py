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
import numpy as np
import pandas as pd
from loguru import logger


# ── 价格数据清洗 ──────────────────────────────────────────────────────────────

def clean_prices(prices: pd.DataFrame, label: str = "prices") -> pd.DataFrame:
    """
    清洗日线价格 DataFrame（index=日期, columns=股票代码）。

    处理项:
      1. 去除重复日期（保留第一条）
      2. 零价、负价 → NaN
      3. 单日涨跌幅超过 ±100% → NaN（后复权价格几乎不可能，必为数据错误）
      4. "孤岛刺针"：某日价格是前后两日均值的 3 倍以上或 1/3 以下 → NaN
      5. 短缺口前向填充（最多 5 个交易日，模拟停牌）
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

    # 4. 孤岛刺针：当日价格偏离前后两日均值超过 3 倍或低于 1/3
    #    shift(1) = 前一日, shift(-1) = 后一日
    neighbor_mean = (prices.shift(1) + prices.shift(-1)) / 2
    ratio = prices / neighbor_mean.replace(0, np.nan)
    spike_mask = (ratio > 3.0) | (ratio < 1 / 3.0)
    n_spikes = spike_mask.sum().sum()
    if n_spikes > 0:
        logger.warning(f"{label}: 发现 {n_spikes} 个孤岛刺针，已置为 NaN")
    prices = prices.where(~spike_mask)

    # 5. 短缺口前向填充（最多 5 天，模拟停牌复牌）
    prices = prices.ffill(limit=5)

    n_nan = prices.isna().sum().sum()
    logger.info(
        f"{label} 清洗完成: {original_shape} → {prices.shape}，"
        f"剩余 NaN 格子 {n_nan}（停牌/退市正常）"
    )
    return prices.sort_index()


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
    清洗财务指标 DataFrame（列: trade_date, code, roe, bvps, total_assets）。

    处理项:
      1. 去除重复 (trade_date, code) 行
      2. ROE 超出 ±300% → NaN（数据源录入错误，正常公司极少超过）
      3. 每股净资产 bvps ≤ 0 → NaN（负净资产导致 PB 计算无意义）
      4. 总资产 total_assets ≤ 0 → NaN（物理上不可能）
      5. 未来数据防穿越：删除 trade_date > 今天的行（数据源偶发填错日期）
    """
    n_before = len(fin)

    # 1. 去重
    fin = fin.drop_duplicates(subset=["trade_date", "code"], keep="last")

    # 2. ROE 范围：[-300%, 300%]
    #    正常优质公司 ROE 在 5%~30%；超过 ±100% 极罕见；超过 ±300% 基本是数据错误
    roe_bad = fin["roe"].abs() > 300
    if roe_bad.sum() > 0:
        logger.warning(f"financial: ROE 超出 ±300% 共 {roe_bad.sum()} 行，已置 NaN")
    fin.loc[roe_bad, "roe"] = np.nan

    # 3. bvps ≤ 0：负净资产股票的 PB 无意义，从 value_pb 因子中排除
    bvps_bad = fin["bvps"] <= 0
    if bvps_bad.sum() > 0:
        logger.warning(f"financial: bvps ≤ 0 共 {bvps_bad.sum()} 行，已置 NaN")
    fin.loc[bvps_bad, "bvps"] = np.nan

    # 4. total_assets ≤ 0
    assets_bad = fin["total_assets"] <= 0
    if assets_bad.sum() > 0:
        logger.warning(f"financial: total_assets ≤ 0 共 {assets_bad.sum()} 行，已置 NaN")
    fin.loc[assets_bad, "total_assets"] = np.nan

    # 5. 未来日期防穿越
    today = pd.Timestamp.today().normalize()
    future = fin["trade_date"] > today
    if future.sum() > 0:
        logger.warning(f"financial: 发现 {future.sum()} 行 trade_date 在未来，已删除")
    fin = fin[~future]

    n_after = len(fin)
    logger.info(
        f"financial 清洗完成: {n_before} → {n_after} 行"
        f"（删除 {n_before - n_after} 行重复/未来数据）"
    )
    return fin.sort_values(["code", "trade_date"]).reset_index(drop=True)
