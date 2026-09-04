"""
factors/factor_smallcap.py  —  小盘股策略因子库

针对小盘股策略补齐的一批事件/筹码/资金类因子，数据来自 data/raw/ 下
已下载的 parquet 文件（shareholder_count / lhb_detail / lockup_release /
holder_trade / block_trade / margin_detail）。

约定（遵循项目 CLAUDE.md）：
  - 因子函数内部已取反，输出「越高越好」
  - 截面 winsorize(1%) + cross_sectional_zscore(clip=3σ)（用 _normalize）
  - PIT 安全：股东户数/高管增减持用 **公告日**（announce_date），
    限售解禁用 **解禁日**（release_date，交易所提前公告的公开信息，
    T 日已知未来解禁计划，属合法前视信号），
    龙虎榜/大宗交易/融资融券用交易日（当日已知）
  - 数据自包含：函数内部 pd.read_parquet(RAW_DIR / ...) 自行加载，
    不依赖 run.py 透传，便于独立调试与 wiring 解耦
  - 清洗自包含：inf→NaN、code 补零、日期解析在本模块内完成

所有因子输出：DataFrame(index=trade_date, columns=code)，函数内调 _normalize。
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import RAW_DIR
from factors.factor import _normalize


# ══════════════════════════════════════════════════════════════════════════════
# 通用工具
# ══════════════════════════════════════════════════════════════════════════════

def _zero_pad_code(s: pd.Series) -> pd.Series:
    """把 code 统一成 6 位字符串（左补零），与 prices columns 对齐。"""
    return s.astype(str).str.zfill(6).str.strip()


def _forward_rolling_sum(panel: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    前向滚动求和：对每个 t，聚合 [t, t+window-1] 内的值。

    用于限售解禁等「未来 N 日累计」信号——解禁日期由交易所提前公告，
    T 日已知未来解禁计划，属合法前视信号（非 look-ahead bias）。

    实现：翻转序列后用标准 backward rolling，再翻转回来。
    """
    flipped = panel[::-1]
    fwd = flipped.rolling(window, min_periods=1).sum()[::-1]
    return fwd.reindex(panel.index)


def _reindex_to_prices(panel: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """把 (announce/trade_date × code) 面板 reindex 到 prices.index，列对齐 prices.columns。"""
    panel = panel.reindex(index=prices.index, columns=prices.columns)
    return panel


def _pivot_event_to_daily(
    long_df: pd.DataFrame,
    date_col: str,
    code_col: str = "code",
    value_col: str | None = None,
    prices: pd.DataFrame | None = None,
    agg: str = "sum",
) -> pd.DataFrame:
    """
    把事件长表透视成 (date × code) 日频面板，并 reindex 到 prices.index。

    value_col=None 时输出 0/1 事件指示面板（用于计数）；
    否则输出按 agg 聚合的数值面板（同日多事件求和/取末值）。
    """
    long_df = long_df.copy()
    long_df[code_col] = _zero_pad_code(long_df[code_col])
    long_df[date_col] = pd.to_datetime(long_df[date_col], errors="coerce")
    long_df = long_df.dropna(subset=[date_col, code_col])

    if value_col is None:
        long_df = long_df.assign(_evt=1)
        value_col = "_evt"

    panel = long_df.pivot_table(
        index=date_col, columns=code_col, values=value_col, aggfunc=agg,
    )
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()

    if prices is not None:
        panel = panel.reindex(index=prices.index, columns=prices.columns)
    return panel


# ══════════════════════════════════════════════════════════════════════════════
# 1. 股东户数（季频，筹码集中度）
# ══════════════════════════════════════════════════════════════════════════════

def _load_shareholder_count() -> pd.DataFrame:
    p = RAW_DIR / "shareholder_count.parquet"
    if not p.exists():
        logger.warning(f"股东户数文件不存在: {p}")
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["code"] = _zero_pad_code(df["code"])
    df["announce_date"] = pd.to_datetime(df["announce_date"], errors="coerce")
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    df = df.dropna(subset=["announce_date", "code"])
    # inf/异常值清洗
    for c in ("holder_count", "holder_count_prev", "avg_float_market_value",
              "total_market_value"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def factor_shareholder_count_change_quarterly(prices: pd.DataFrame) -> pd.DataFrame:
    """
    股东户数季度变化率：(上期户数 - 本期户数) / 上期户数。
    户数减少 = 筹码集中 = 利好 → 正方向。

    PIT：用 **公告日**（announce_date）而非统计截止日（report_date）作可用起点，
    reindex 到交易日 ffill，避免用 report_date 做 ffill 的 look-ahead bias。
    """
    df = _load_shareholder_count()
    if df.empty:
        return pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)

    # 用 announce_date 透视出 holder_count 季频面板
    panel = df.pivot_table(
        index="announce_date", columns="code", values="holder_count", aggfunc="last",
    ).sort_index()
    # 季度环比变化率：(prev - curr) / prev = -pct_change
    chg = -panel.pct_change(1)
    chg = chg.replace([np.inf, -np.inf], np.nan)

    # reindex 到交易日并 ffill（PIT：announce_date 之后才可用）
    chg = chg.reindex(prices.index).ffill()
    chg = _reindex_to_prices(chg, prices)
    return _normalize(chg)


def factor_holder_avg_float_mv_quarterly(prices: pd.DataFrame) -> pd.DataFrame:
    """
    户均流通市值（季频）：本期户均持股市值取对数取负。
    越小盘 + 越集中 → 越高分（筹码分散度代理，结合市值看）。

    PIT：用公告日作可用起点，reindex 到交易日 ffill。
    """
    df = _load_shareholder_count()
    if df.empty:
        return pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)

    panel = df.pivot_table(
        index="announce_date", columns="code", values="avg_float_market_value",
        aggfunc="last",
    ).sort_index()
    # 取对数取负：户均市值越小 → 分数越高
    neg_log = -np.log(panel.replace(0, np.nan).clip(lower=1e-8))
    neg_log = neg_log.replace([np.inf, -np.inf], np.nan)

    neg_log = neg_log.reindex(prices.index).ffill()
    neg_log = _reindex_to_prices(neg_log, prices)
    return _normalize(neg_log)


def factor_shareholder_count_change_yearly(prices: pd.DataFrame) -> pd.DataFrame:
    """
    股东户数年度变化率：4 期累计变化率 = (4期前户数 - 本期户数) / 4期前户数。
    户数年度净减少 = 长期筹码集中 = 利好 → 正方向。

    PIT：用公告日作可用起点。
    """
    df = _load_shareholder_count()
    if df.empty:
        return pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)

    panel = df.pivot_table(
        index="announce_date", columns="code", values="holder_count", aggfunc="last",
    ).sort_index()
    # 4 期累计变化率（同比）
    chg = -panel.pct_change(4)
    chg = chg.replace([np.inf, -np.inf], np.nan)

    chg = chg.reindex(prices.index).ffill()
    chg = _reindex_to_prices(chg, prices)
    return _normalize(chg)


# ══════════════════════════════════════════════════════════════════════════════
# 2. 龙虎榜（日频事件，妖股/打板信号）
# ══════════════════════════════════════════════════════════════════════════════

def _load_lhb() -> pd.DataFrame:
    p = RAW_DIR / "lhb_detail.parquet"
    if not p.exists():
        logger.warning(f"龙虎榜文件不存在: {p}")
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["code"] = _zero_pad_code(df["code"])
    df["lhb_date"] = pd.to_datetime(df["lhb_date"], errors="coerce")
    df = df.dropna(subset=["lhb_date", "code"])
    return df


def factor_lhb_count_20d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    龙虎榜上榜次数_20d：过去 20 日上榜次数。
    越高 = 越活跃 / 游资关注。当日上榜当日已知，无 PIT 问题。
    """
    df = _load_lhb()
    if df.empty:
        return pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)

    # 同一 (code, lhb_date) 可能多 reason 行，去重成单日事件指示
    daily = df.drop_duplicates(subset=["code", "lhb_date"])
    ind = _pivot_event_to_daily(daily, "lhb_date", value_col=None,
                                prices=prices, agg="sum")
    # 缺失日为 0 次上榜（事件面板 reindex 后 NaN → 0）
    ind = ind.fillna(0)
    cnt = ind.rolling(20, min_periods=5).sum()
    return _normalize(cnt)


def factor_lhb_net_buy_20d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    龙虎榜净买额_20d：过去 20 日龙虎榜净买额累计（正 = 游资净买入）。
    无 PIT 问题（当日已知）。
    """
    df = _load_lhb()
    if df.empty:
        return pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)

    # 同日多 reason 聚合 net_buy 求和（同一上榜日合并席位净额）
    daily = df.groupby(["lhb_date", "code"], as_index=False)["net_buy"].sum()
    panel = _pivot_event_to_daily(daily, "lhb_date", value_col="net_buy",
                                  prices=prices, agg="sum")
    # 非上榜日净买额 = 0
    panel = panel.fillna(0)
    cum = panel.rolling(20, min_periods=3).sum()
    return _normalize(cum)


def factor_lhb_consecutive(prices: pd.DataFrame) -> pd.DataFrame:
    """
    龙虎榜连续上榜强度：过去 3 日上榜天数（0-3 连续值）。
    连续上榜 = 强势游资接力，打板信号。无 PIT 问题。

    用连续计数而非 0/1 哑变量：二元哑变量经 winsorize(1%) 后稀有 1 会被
    截断成 0 导致信号全灭；改用 0-3 连续值可保留截面变异，标准化后有意义。
    """
    df = _load_lhb()
    if df.empty:
        return pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)

    daily = df.drop_duplicates(subset=["code", "lhb_date"])
    ind = _pivot_event_to_daily(daily, "lhb_date", value_col=None,
                                prices=prices, agg="sum").fillna(0)
    # 过去 3 日上榜天数（0-3 连续值，捕捉连续上榜强度）
    cnt3 = ind.rolling(3, min_periods=3).sum()
    return _normalize(cnt3)


# ══════════════════════════════════════════════════════════════════════════════
# 3. 限售解禁（事件，利空压力，前视信号合法）
# ══════════════════════════════════════════════════════════════════════════════

def _load_lockup() -> pd.DataFrame:
    p = RAW_DIR / "lockup_release.parquet"
    if not p.exists():
        logger.warning(f"限售解禁文件不存在: {p}")
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["code"] = _zero_pad_code(df["code"])
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df = df.dropna(subset=["release_date", "code"])
    return df


def factor_lockup_value_ratio_60d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    未来 60 日解禁市值占比：未来 60 日解禁市值 / 当前流通市值，**取负号**
    让「无解禁 = 高分」。

    前视合法性：解禁日期由交易所提前公告（限售股解禁计划是公开信息），
    T 日已知 [t, t+60] 内的解禁安排，属合法前视信号（不是 look-ahead bias）。
    数据无 announce_date 字段，直接用 release_date 作事件日（已公开的解禁日）。

    分母：circ_mv.parquet（流通市值宽表，元）；缺失时返回 NaN 面板。
    """
    df = _load_lockup()
    if df.empty:
        return pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)

    panel = _pivot_event_to_daily(
        df, "release_date", value_col="actual_release_value",
        prices=prices, agg="sum",
    ).fillna(0)
    # 前向 60 日累计解禁市值
    fwd60 = _forward_rolling_sum(panel, 60)

    # 流通市值分母
    mv_path = RAW_DIR / "circ_mv.parquet"
    if not mv_path.exists():
        logger.warning("circ_mv.parquet 不存在，限售解禁占比因子无法归一化，返回 NaN")
        return pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    circ_mv = pd.read_parquet(mv_path)
    circ_mv.index = pd.to_datetime(circ_mv.index)
    circ_mv = circ_mv.reindex(index=prices.index, columns=prices.columns)

    ratio = (fwd60 / circ_mv.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    # 取负：无解禁 = 高分
    return _normalize(-ratio)


def factor_lockup_count_30d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    未来 30 日解禁次数：未来 30 日解禁事件数，取负号让「无解禁 = 高分」。
    前视合法性同上。
    """
    df = _load_lockup()
    if df.empty:
        return pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)

    panel = _pivot_event_to_daily(
        df, "release_date", value_col=None, prices=prices, agg="sum",
    ).fillna(0)
    fwd30 = _forward_rolling_sum(panel, 30)
    return _normalize(-fwd30)


# ══════════════════════════════════════════════════════════════════════════════
# 4. 高管增减持（事件，内部人信号）
# ══════════════════════════════════════════════════════════════════════════════

def _load_holder_trade() -> pd.DataFrame:
    p = RAW_DIR / "holder_trade.parquet"
    if not p.exists():
        logger.warning(f"高管增减持文件不存在: {p}")
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["code"] = _zero_pad_code(df["code"])
    df["announce_date"] = pd.to_datetime(df["announce_date"], errors="coerce")
    df = df.dropna(subset=["announce_date", "code"])
    return df


def _holder_trade_amount(df: pd.DataFrame, prices_raw: pd.DataFrame) -> pd.DataFrame:
    """
    把增减持长表转成 (announce_date × code) 净增减持金额面板（元）。

    change_shares 单位为万股（参考 AKShare holder_trade 接口），
    净额 = change_shares × 10000 × 公告日不复权收盘价 × 方向(增持+1/减持-1)。
    价格缺失时返回 NaN（不参与累计）。
    """
    df = df.copy()
    # 方向：增持 → +1，减持 → -1，其他 → 0
    direction = df["change_direction"].str.contains("增", na=False).astype(int) \
        - df["change_direction"].str.contains("减", na=False).astype(int)
    df["direction"] = direction

    # 公告日不复权收盘价（近似成交价；高管增减持价格接近市价）
    # 按行 lookup 公告日不复权收盘价（vectorize 困难因 announce_date 与 code 均逐行不同）
    price_lookup = pd.Series(
        [prices_raw.loc[d, c] if d in prices_raw.index and c in prices_raw.columns
         else np.nan
         for d, c in zip(df["announce_date"], df["code"])],
        index=df.index,
    )
    # 净增持金额（元）：万股 × 10000 × 价格 × 方向
    df["net_amount"] = df["change_shares"] * 10000.0 * price_lookup * df["direction"]
    df["net_amount"] = df["net_amount"].replace([np.inf, -np.inf], np.nan)
    return df


def factor_holder_net_buy_60d(prices: pd.DataFrame,
                              prices_raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    高管净增持额_60d：过去 60 日高管净增持金额（正 = 内部人看好）。
    PIT：用公告日作事件日。
    """
    df = _load_holder_trade()
    if df.empty:
        return pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    if prices_raw is None:
        prices_raw = prices  # 退化为复权价近似

    df_amt = _holder_trade_amount(df, prices_raw)
    panel = _pivot_event_to_daily(
        df_amt, "announce_date", value_col="net_amount",
        prices=prices, agg="sum",
    ).fillna(0)
    cum = panel.rolling(60, min_periods=3).sum()
    return _normalize(cum)


def factor_holder_buy_count_60d(prices: pd.DataFrame) -> pd.DataFrame:
    """高管增持次数_60d：过去 60 日增持事件数。PIT：用公告日。"""
    df = _load_holder_trade()
    if df.empty:
        return pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    df_buy = df[df["change_direction"].str.contains("增", na=False)
                & ~df["change_direction"].str.contains("减", na=False)]
    panel = _pivot_event_to_daily(df_buy, "announce_date", value_col=None,
                                  prices=prices, agg="sum").fillna(0)
    cnt = panel.rolling(60, min_periods=3).sum()
    return _normalize(cnt)


def factor_holder_sell_count_60d(prices: pd.DataFrame) -> pd.DataFrame:
    """高管减持次数_60d：过去 60 日减持事件数（取负：减持越少 = 高分）。PIT：用公告日。"""
    df = _load_holder_trade()
    if df.empty:
        return pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    df_sell = df[df["change_direction"].str.contains("减", na=False)]
    panel = _pivot_event_to_daily(df_sell, "announce_date", value_col=None,
                                  prices=prices, agg="sum").fillna(0)
    cnt = panel.rolling(60, min_periods=3).sum()
    return _normalize(-cnt)


def factor_holder_buy_sell_ratio_60d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    增减持比_60d：增持次数 / (增持次数 + 减持次数)。
    越高 = 内部人越偏好看好。PIT：用公告日。
    """
    df = _load_holder_trade()
    if df.empty:
        return pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    is_buy = df["change_direction"].str.contains("增", na=False) \
        & ~df["change_direction"].str.contains("减", na=False)
    is_sell = df["change_direction"].str.contains("减", na=False)
    df_buy = df[is_buy]
    df_sell = df[is_sell]

    buy_panel = _pivot_event_to_daily(df_buy, "announce_date", value_col=None,
                                      prices=prices, agg="sum").fillna(0)
    sell_panel = _pivot_event_to_daily(df_sell, "announce_date", value_col=None,
                                       prices=prices, agg="sum").fillna(0)
    buy_cnt = buy_panel.rolling(60, min_periods=3).sum()
    sell_cnt = sell_panel.rolling(60, min_periods=3).sum()
    total = buy_cnt + sell_cnt
    ratio = (buy_cnt / total.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    return _normalize(ratio)


# ══════════════════════════════════════════════════════════════════════════════
# 5. 大宗交易（日频，折价信号）
# ══════════════════════════════════════════════════════════════════════════════

def _load_block_trade() -> pd.DataFrame:
    p = RAW_DIR / "block_trade.parquet"
    if not p.exists():
        logger.warning(f"大宗交易文件不存在: {p}")
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["code"] = _zero_pad_code(df["code"])
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df.dropna(subset=["trade_date", "code"])
    return df


def factor_block_discount_20d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    大宗交易折价率_20d：过去 20 日大宗交易平均折价率。
    discount_rate = (成交价 - 收盘) / 收盘（负值 = 折价）；
    本因子输出 -mean(discount_rate)，使「折价越大 → 分数越高」（利好承接解读）。

    注：深折价也可能是减持出货，需结合其他信号；方向由 IC 验证。
    无 PIT 问题（当日已知）。
    """
    df = _load_block_trade()
    if df.empty:
        return pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)

    df_disc = df.groupby(["trade_date", "code"], as_index=False)["discount_rate"].mean()
    panel = _pivot_event_to_daily(df_disc, "trade_date", value_col="discount_rate",
                                  prices=prices, agg="mean")
    avg = panel.rolling(20, min_periods=2).mean()
    # 取负：折价越大（discount_rate 越负）→ 分数越高
    return _normalize(-avg)


def factor_block_count_20d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    大宗交易频次_20d：过去 20 日大宗交易次数。无 PIT 问题。
    """
    df = _load_block_trade()
    if df.empty:
        return pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)

    daily = df.drop_duplicates(subset=["code", "trade_date"])
    panel = _pivot_event_to_daily(daily, "trade_date", value_col=None,
                                  prices=prices, agg="sum").fillna(0)
    cnt = panel.rolling(20, min_periods=3).sum()
    return _normalize(cnt)


# ══════════════════════════════════════════════════════════════════════════════
# 6. 个股融资融券（日频，杠杆情绪）
# ══════════════════════════════════════════════════════════════════════════════

def _load_margin_detail_wide(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    加载 margin_detail 长表并透视成 (date × code) 宽表字典。
    返回 {'margin_balance': ..., 'short_balance_amount': ..., 'margin_buy_amount': ...}，
    均已 reindex 到 prices.index/columns。
    """
    p = RAW_DIR / "margin_detail.parquet"
    if not p.exists():
        logger.warning(f"个股融资融券文件不存在: {p}")
        return {}
    df = pd.read_parquet(p)
    df["code"] = _zero_pad_code(df["code"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "code"])

    out = {}
    for col in ("margin_balance", "short_balance_amount", "margin_buy_amount"):
        sub = df.dropna(subset=[col]) if col in df.columns else pd.DataFrame()
        if sub.empty:
            out[col] = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
            continue
        wide = sub.pivot_table(index="date", columns="code", values=col, aggfunc="last")
        wide.index = pd.to_datetime(wide.index)
        wide = wide.sort_index().reindex(index=prices.index, columns=prices.columns)
        out[col] = wide
    return out


def factor_margin_balance_change_20d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    个股融资余额变化_20d：(当日融资余额 - 20日前) / 20日前融资余额。
    正 = 杠杆资金流入 = 看多情绪。命名「个股融资余额变化_20d」避免与
    ALPHA2 市场总量版「融资余额变化_20d」冲突。无 PIT 问题。
    """
    wide = _load_margin_detail_wide(prices).get("margin_balance")
    if wide is None or wide.empty:
        return pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    chg = wide.pct_change(20)
    chg = chg.replace([np.inf, -np.inf], np.nan)
    return _normalize(chg)


def factor_short_balance_change_20d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    个股融券余额变化_20d：20 日融券余额变化率，**取负**（融券增加 = 看空，
    融券减少 = 高分）。无 PIT 问题。
    """
    wide = _load_margin_detail_wide(prices).get("short_balance_amount")
    if wide is None or wide.empty:
        return pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    chg = wide.pct_change(20)
    chg = chg.replace([np.inf, -np.inf], np.nan)
    return _normalize(-chg)


def factor_margin_buy_amount_5d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    融资买入额_5d：过去 5 日融资买入额累计（正 = 杠杆资金主动买入）。
    无 PIT 问题。
    """
    wide = _load_margin_detail_wide(prices).get("margin_buy_amount")
    if wide is None or wide.empty:
        return pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    cum = wide.rolling(5, min_periods=2).sum()
    return _normalize(cum)


# ══════════════════════════════════════════════════════════════════════════════
# 批量入口（供 registry section 调用，惰性自加载）
# ══════════════════════════════════════════════════════════════════════════════

def get_smallcap_factors(prices: pd.DataFrame,
                         prices_raw: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """
    计算全部小盘股因子，返回 {因子名: 面板} 字典。
    每个因子函数内部已 _normalize，并 reindex 到 prices.index/columns。
    数据缺失时返回 NaN 面板（不抛错，registry 层会过滤 NaN）。
    """
    out: dict[str, pd.DataFrame] = {}
    try:
        out["股东户数变化率_季"] = factor_shareholder_count_change_quarterly(prices)
    except Exception as e:
        logger.warning(f"股东户数变化率_季 计算失败: {e}")
    try:
        out["户均流通市值_季"] = factor_holder_avg_float_mv_quarterly(prices)
    except Exception as e:
        logger.warning(f"户均流通市值_季 计算失败: {e}")
    try:
        out["股东户数变化率_年"] = factor_shareholder_count_change_yearly(prices)
    except Exception as e:
        logger.warning(f"股东户数变化率_年 计算失败: {e}")

    try:
        out["龙虎榜上榜次数_20d"] = factor_lhb_count_20d(prices)
    except Exception as e:
        logger.warning(f"龙虎榜上榜次数_20d 计算失败: {e}")
    try:
        out["龙虎榜净买额_20d"] = factor_lhb_net_buy_20d(prices)
    except Exception as e:
        logger.warning(f"龙虎榜净买额_20d 计算失败: {e}")
    try:
        out["龙虎榜连续上榜"] = factor_lhb_consecutive(prices)
    except Exception as e:
        logger.warning(f"龙虎榜连续上榜 计算失败: {e}")

    try:
        out["未来60日解禁市值占比"] = factor_lockup_value_ratio_60d(prices)
    except Exception as e:
        logger.warning(f"未来60日解禁市值占比 计算失败: {e}")
    try:
        out["未来30日解禁次数"] = factor_lockup_count_30d(prices)
    except Exception as e:
        logger.warning(f"未来30日解禁次数 计算失败: {e}")

    try:
        out["高管净增持额_60d"] = factor_holder_net_buy_60d(prices, prices_raw)
    except Exception as e:
        logger.warning(f"高管净增持额_60d 计算失败: {e}")
    try:
        out["高管增持次数_60d"] = factor_holder_buy_count_60d(prices)
    except Exception as e:
        logger.warning(f"高管增持次数_60d 计算失败: {e}")
    try:
        out["高管减持次数_60d"] = factor_holder_sell_count_60d(prices)
    except Exception as e:
        logger.warning(f"高管减持次数_60d 计算失败: {e}")
    try:
        out["增减持比_60d"] = factor_holder_buy_sell_ratio_60d(prices)
    except Exception as e:
        logger.warning(f"增减持比_60d 计算失败: {e}")

    try:
        out["大宗交易折价率_20d"] = factor_block_discount_20d(prices)
    except Exception as e:
        logger.warning(f"大宗交易折价率_20d 计算失败: {e}")
    try:
        out["大宗交易频次_20d"] = factor_block_count_20d(prices)
    except Exception as e:
        logger.warning(f"大宗交易频次_20d 计算失败: {e}")

    try:
        out["个股融资余额变化_20d"] = factor_margin_balance_change_20d(prices)
    except Exception as e:
        logger.warning(f"个股融资余额变化_20d 计算失败: {e}")
    try:
        out["融券余额变化_20d"] = factor_short_balance_change_20d(prices)
    except Exception as e:
        logger.warning(f"融券余额变化_20d 计算失败: {e}")
    try:
        out["融资买入额_5d"] = factor_margin_buy_amount_5d(prices)
    except Exception as e:
        logger.warning(f"融资买入额_5d 计算失败: {e}")

    return out


SMALLCAP_FACTOR_NAMES: tuple[str, ...] = (
    "股东户数变化率_季", "户均流通市值_季", "股东户数变化率_年",
    "龙虎榜上榜次数_20d", "龙虎榜净买额_20d", "龙虎榜连续上榜",
    "未来60日解禁市值占比", "未来30日解禁次数",
    "高管净增持额_60d", "高管增持次数_60d", "高管减持次数_60d", "增减持比_60d",
    "大宗交易折价率_20d", "大宗交易频次_20d",
    "个股融资余额变化_20d", "融券余额变化_20d", "融资买入额_5d",
)
