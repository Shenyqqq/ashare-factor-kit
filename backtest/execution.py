"""
backtest/execution.py — A-share trade constraint simulation.

Encapsulates buy/sell eligibility on execution day: limit-up open (no buy),
limit-down open (no sell), ST filter, listing-age filter, suspension detection.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config.settings import (
    BID_ASK_SPREAD_BPS,
    COMMISSION_RATE,
    EXCLUDE_ST,
    SLIPPAGE_RATE,
    STAMP_DUTY,
)


@dataclass
class BacktestConfig:
    """Execution & cost parameters (defaults from config/settings.py)."""

    commission_rate: float = COMMISSION_RATE
    stamp_duty: float = STAMP_DUTY
    slippage_rate: float = SLIPPAGE_RATE
    bid_ask_spread_bps: float = BID_ASK_SPREAD_BPS  # 单边 spread（bp），10bp=A股小盘股典型
    cost_bps: float = 3.5          # legacy single-side fallback (bp)
    exclude_st: bool = EXCLUDE_ST
    min_listing_days: int = 60     # 0 = disabled
    min_lot_100: bool = False      # stub: lot rounding not applied yet
    strict_limit_mode: bool = True # True=一字板; False=any limit day blocks trade
    use_open_execution: bool = True
    # ── Turnover 控制（任务3：解耦调仓频率与持仓期限）─────────────────────────
    # turnover_limit: 每期最大换手率，1.0=无限制，0.3=最多换 30% 仓位
    # rank_change_threshold: 只换掉排名下降超过此阈值的股票
    #   0.0=全换（标准 quantile 行为），0.2=排名跌出 top 20% 才换
    #   rank_pct < (1 - threshold) 视为「排名大幅下降」，触发换仓
    turnover_limit: float = 1.0
    rank_change_threshold: float = 0.0


@dataclass
class TradeRules:
    """Per-run trade constraints derived from masks and universe metadata."""

    masks: dict | None = None
    st_codes: set[str] = field(default_factory=set)
    # M4 修复：时间序列 ST 状态（wide bool DataFrame: index=date, columns=stock, True=当日 ST）
    # 优先于 st_codes 使用；为 None 时回退到 st_codes 静态集合（向后兼容）。
    st_schedule: pd.DataFrame | None = None
    listing_dates: dict[str, pd.Timestamp] | None = None
    # M4 修复：退市日期字典（code → delist_date），回测时按 date > delist_date 判断已退市
    delist_dates: dict[str, pd.Timestamp] | None = None
    volume: pd.DataFrame | None = None
    config: BacktestConfig = field(default_factory=BacktestConfig)

    def _limit_up_open(self, stock: str, date: pd.Timestamp) -> bool:
        if self.masks is None:
            return False
        lu = self.masks.get("limit_up_open")
        if lu is None or date not in lu.index or stock not in lu.columns:
            return False
        return bool(lu.at[date, stock])

    def _limit_down_open(self, stock: str, date: pd.Timestamp) -> bool:
        if self.masks is None:
            return False
        ld = self.masks.get("limit_down_open")
        if ld is None or date not in ld.index or stock not in ld.columns:
            return False
        return bool(ld.at[date, stock])

    def _any_limit_up(self, stock: str, date: pd.Timestamp) -> bool:
        if self.masks is None:
            return False
        lu = self.masks.get("limit_up")
        if lu is None or date not in lu.index or stock not in lu.columns:
            return False
        return bool(lu.at[date, stock])

    def _any_limit_down(self, stock: str, date: pd.Timestamp) -> bool:
        if self.masks is None:
            return False
        ld = self.masks.get("limit_down")
        if ld is None or stock not in ld.columns or date not in ld.index:
            return False
        return bool(ld.at[date, stock])

    def is_suspended(
        self,
        stock: str,
        date: pd.Timestamp,
        close: pd.DataFrame,
    ) -> bool:
        """Suspension: missing close or zero volume → no trade, NAV flat."""
        if date not in close.index or stock not in close.columns:
            return True
        px = close.at[date, stock]
        if pd.isna(px) or px <= 0:
            return True
        if self.volume is not None and date in self.volume.index and stock in self.volume.columns:
            vol = self.volume.at[date, stock]
            if pd.isna(vol) or vol <= 0:
                return True
        return False

    def passes_listing_filter(self, stock: str, date: pd.Timestamp) -> bool:
        min_days = self.config.min_listing_days
        if min_days <= 0 or not self.listing_dates:
            return True
        listed = self.listing_dates.get(stock)
        if listed is None:
            return True
        return (date - listed).days >= min_days

    def is_delisted(self, stock: str, date: pd.Timestamp) -> bool:
        """M4 修复：按日期判断股票是否已退市。date > delist_date 视为已退市。"""
        if not self.delist_dates:
            return False
        d = self.delist_dates.get(stock)
        if d is None or pd.isna(d):
            return False
        try:
            return pd.Timestamp(date) > pd.Timestamp(d)
        except Exception:
            return False

    def passes_st_filter(
        self,
        stock: str,
        date: pd.Timestamp | None = None,
    ) -> bool:
        """判断股票在 date 当日是否非 ST（可买入）。

        优先用时间序列 st_schedule（按日期精确查询）；
        无 schedule 或日期/股票不在表中时回退到 st_codes 静态集合（向后兼容）。
        """
        if not self.config.exclude_st:
            return True
        if self.st_schedule is not None and date is not None:
            try:
                if date in self.st_schedule.index and stock in self.st_schedule.columns:
                    return not bool(self.st_schedule.at[date, stock])
            except Exception:
                pass
        # 回退：静态 ST 集合
        if stock in self.st_codes:
            return False
        return True

    def can_buy(self, stock: str, date: pd.Timestamp, close: pd.DataFrame) -> bool:
        # M4 修复：先判断已退市（已退市股不能新买入，但已持仓的可继续持有到末日在市）
        if self.is_delisted(stock, date):
            return False
        if not self.passes_st_filter(stock, date):
            return False
        if not self.passes_listing_filter(stock, date):
            return False
        if self.is_suspended(stock, date, close):
            return False
        if self.config.strict_limit_mode:
            if self._limit_up_open(stock, date):
                return False
        elif self._any_limit_up(stock, date):
            return False
        return True

    def can_sell(self, stock: str, date: pd.Timestamp, close: pd.DataFrame) -> bool:
        if self.is_suspended(stock, date, close):
            return False
        if self.config.strict_limit_mode:
            if self._limit_down_open(stock, date):
                return False
        elif self._any_limit_down(stock, date):
            return False
        return True


def next_trading_day(date: pd.Timestamp, index: pd.DatetimeIndex) -> pd.Timestamp | None:
    """First trading day strictly after *date*."""
    later = index[index > date]
    return later[0] if len(later) > 0 else None


def resolve_execution_date(
    signal_date: pd.Timestamp,
    trading_index: pd.DatetimeIndex,
    use_open: bool,
) -> pd.Timestamp | None:
    """signal_date → execution_date (T+1 open when use_open else same-day close)."""
    if use_open:
        return next_trading_day(signal_date, trading_index)
    return signal_date if signal_date in trading_index else None


def hold_dates_between(
    trading_index: pd.DatetimeIndex,
    execution_date: pd.Timestamp,
    next_signal_date: pd.Timestamp,
) -> pd.DatetimeIndex:
    """Trading days in [execution_date, next_signal_date] inclusive."""
    return trading_index[
        (trading_index >= execution_date) & (trading_index <= next_signal_date)
    ]


def infer_st_codes(stock_names: pd.Series | None) -> set[str]:
    """Build ST set from name series (code → name). Stub when names unavailable."""
    if stock_names is None or stock_names.empty:
        return set()
    mask = stock_names.astype(str).str.contains("ST", case=False, na=False)
    return set(stock_names.index[mask].astype(str))


def build_st_schedule(
    stock_names: pd.Series | None,
    dates: pd.DatetimeIndex,
    is_st_current: pd.Series | None = None,
    delist_dates: dict[str, pd.Timestamp] | None = None,
) -> pd.DataFrame | None:
    """构建时间序列 ST 状态表（M4 修复：消除「所有历史日期都用当前 ST 状态」的失真）。

    当前为保守实现：对当前 ST 股，假设其在所有回测日期均为 ST。
    这比旧逻辑「所有历史日期都用当前状态」在方向上一致，但提供了 hook
    让未来接入真实 ST 历史接口（如 ak.stock_zh_a_st_em）时只需替换本函数实现。

    Parameters
    ----------
    stock_names : code → name（来自 universe/stock_list.parquet）
    dates : 回测全部交易日索引
    is_st_current : code → bool；可选，由 stock_list['is_st_current'] 提供
    delist_dates : code → delist_date；可选，已退市股在退市后不标记为 ST（已不可交易）

    Returns
    -------
    pd.DataFrame | None
        wide bool DataFrame：index=dates, columns=ST 股票代码, True=当日为 ST。
        无 ST 股时返回 None（下游回退到 st_codes 集合逻辑）。
    """
    st_codes: set[str] = set()
    if is_st_current is not None and not is_st_current.empty:
        st_codes = {
            str(c) for c, v in is_st_current.items()
            if bool(v) and not pd.isna(v)
        }
    if not st_codes and stock_names is not None and not stock_names.empty:
        st_codes = infer_st_codes(stock_names)
    if not st_codes:
        return None

    cols = sorted(c for c in st_codes)
    # 保守实现：当前 ST 股在所有回测日期均标记为 ST
    schedule = pd.DataFrame(True, index=pd.DatetimeIndex(dates), columns=cols)

    # 可选：退市股在退市后不再标记 ST（已不可交易，标记与否对回测无影响，但更准确）
    if delist_dates:
        for code in cols:
            d = delist_dates.get(code)
            if d is None or pd.isna(d):
                continue
            try:
                after = schedule.index > pd.Timestamp(d)
                schedule.loc[after, code] = False
            except Exception:
                pass
    return schedule


def build_delist_dates_from_stock_list(
    stock_list: pd.DataFrame,
) -> dict[str, pd.Timestamp]:
    """从 stock_list.parquet 提取 code → delist_date 字典（用于 TradeRules.delist_dates）。

    仅返回有有效 delist_date 的股票。
    """
    if stock_list is None or stock_list.empty or "delist_date" not in stock_list.columns:
        return {}
    sub = stock_list.dropna(subset=["delist_date"])
    if sub.empty:
        return {}
    out: dict[str, pd.Timestamp] = {}
    for _, row in sub.iterrows():
        code = str(row["code"]).zfill(6)
        d = pd.to_datetime(row["delist_date"], errors="coerce")
        if pd.notna(d):
            out[code] = d
    return out


def total_cost_fraction(turnover: float, cfg: BacktestConfig) -> float:
    """
    Total rebalance cost as fraction of NAV.

    turnover ∈ [0,1]: fraction of portfolio weight changed.
    Buy leg: turnover/2 × (commission + slippage)
    Sell leg: turnover/2 × (commission + stamp_duty + slippage)
    Spread:   turnover × bid_ask_spread_bps / 10000 / 2
              （单边 half-spread；买入吃 ask、卖出吃 bid，按换手权重一次性计提）
    Falls back to cost_bps when all rates zero.
    """
    if turnover <= 0:
        return 0.0
    half = turnover / 2.0
    buy_cost = half * (cfg.commission_rate + cfg.slippage_rate)
    sell_cost = half * (cfg.commission_rate + cfg.stamp_duty + cfg.slippage_rate)
    spread_cost = turnover * cfg.bid_ask_spread_bps / 10000.0 / 2.0
    total = buy_cost + sell_cost + spread_cost
    if total <= 0:
        total = turnover * cfg.cost_bps / 10000.0
    return total
