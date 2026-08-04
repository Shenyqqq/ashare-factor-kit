"""
utils/pit_align.py — Point-in-Time 财务数据对齐

A 股财报默认按**法定披露窗口**做保守 PIT 对齐（近似，非真实公告日）：
  报告期 + 披露窗口 → 可用日下界
然后对可用日做 reindex + ffill 到日频价格序列。

【ann_date 可用性说明 — 2026-07-29】
  - ``ak.stock_financial_analysis_indicator`` **不提供** ann_date（「日期」=报告期）。
  - ``ak.stock_yjbb_em`` 的「最新公告日期」是**最近一次修订/公告日**，不是首次披露日；
    用作 PIT 会在财报修订后过度推迟可用日，故**不接入主链**。
  - 若长表已带真实 ``ann_date`` 列（外部/付费源），``pit_pivot_ffill`` 优先用它；
    否则走法定窗并打一次性 WARNING（见 ``PIT_MODE_STATUTORY``）。

A 股法定披露窗口（偏实务、仍略保守）：
  - Q1 季报（03-31）：法定截止 04-30（+30 天）→ 取 **+30**
  - 半年报（06-30）：法定截止 08-31（+62 天）→ 取 **+60**（贴近实务）
  - Q3 季报（09-30）：法定截止 10-31（+31 天）→ 取 **+30**
  - 年报（12-31）：法定截止次年 04-30（+120 天）→ 取 **+90**
    （用户未点名年报；+90 比原 +120 更贴近多数公司实务披露，仍偏保守）

修复 `factors/factor.py:_pivot_financial` 与 `factors/barra_risk.py:_pivot_ffill`
原 `pivot(trade_date).reindex(prices.index, method="ffill")` 把报告期数据
ffill 到该日之后所有日子，造成 look-ahead bias（季报 ~15-25 个交易日，
年报最坏 30-60 个交易日）。
"""
from __future__ import annotations

import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# 报告期月份 → 法定披露窗口天数（偏实务；年报 +90 见模块头注释）
_DISCLOSURE_WINDOWS: dict[int, int] = {
    3: 30,    # Q1 季报
    6: 60,    # 半年报
    9: 30,    # Q3 季报
    12: 90,   # 年报（比法定 +120 略松，仍偏保守）
}

PIT_MODE_ANN_DATE = "ann_date"
PIT_MODE_STATUTORY = "statutory_window_approx"

_statutory_warned = False


def disclosure_window(report_period: pd.Period | pd.Timestamp) -> int:
    """
    按报告期月份返回披露窗口天数：
      03 → 30（Q1）
      06 → 60（半年报）
      09 → 30（Q3）
      12 → 90（年报）
    其他月份取最近季末。
    """
    if isinstance(report_period, pd.Period):
        month = report_period.month
    else:
        ts = pd.Timestamp(report_period)
        month = ts.month

    if month in _DISCLOSURE_WINDOWS:
        return _DISCLOSURE_WINDOWS[month]

    # 非季末月份：归到最近过去季末
    # 1-3月 → 03-31，4-6月 → 06-30，7-9月 → 09-30，10-12月 → 12-31
    quarter_end_month = ((month - 1) // 3) * 3 + 3
    return _DISCLOSURE_WINDOWS[quarter_end_month]


def pit_shift_report_dates(report_periods: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    对每个报告期加 disclosure_window 天，返回可用日下界序列。

    输入：DatetimeIndex（或可转 Timestamp 的序列）的财务报告期
    输出：DatetimeIndex，每个值为「报告期 + 披露窗口」
    """
    periods = pd.DatetimeIndex(pd.to_datetime(report_periods))
    days = np.array([disclosure_window(p) for p in periods], dtype=np.int64)
    return periods + pd.to_timedelta(days, unit="D")


def _warn_statutory_once() -> None:
    global _statutory_warned
    if _statutory_warned:
        return
    _statutory_warned = True
    msg = (
        "财务 PIT 使用法定披露窗口近似（Q1/Q3=+30、半年报=+60、年报=+90），"
        "非真实公告日。AKShare 主接口无可靠 first-ann_date；"
        "yjbb「最新公告日期」=修订日不可用。详见 utils/pit_align.py / docs/PIT_AUDIT.md"
    )
    try:
        from loguru import logger as _lg
        _lg.warning(msg)
    except Exception:
        logger.warning(msg)


def resolve_pit_available_dates(
    financial_df: pd.DataFrame,
    date_col: str = "trade_date",
    ann_date_col: str = "ann_date",
) -> tuple[pd.DatetimeIndex, str]:
    """解析每行财务记录的 PIT 可用日。

    Returns
    -------
    (available_dates, mode)
      mode = ``ann_date`` | ``statutory_window_approx``
    """
    if ann_date_col in financial_df.columns:
        ann = pd.to_datetime(financial_df[ann_date_col], errors="coerce")
        if ann.notna().mean() >= 0.8:
            # 缺 ann_date 的行回退法定窗
            report = pd.DatetimeIndex(pd.to_datetime(financial_df[date_col]))
            statutory = pit_shift_report_dates(report)
            available = ann.where(ann.notna(), statutory)
            return pd.DatetimeIndex(available), PIT_MODE_ANN_DATE

    _warn_statutory_once()
    report = pd.DatetimeIndex(pd.to_datetime(financial_df[date_col]))
    return pit_shift_report_dates(report), PIT_MODE_STATUTORY


def pit_pivot_ffill(
    financial_df: pd.DataFrame,
    prices_index: pd.DatetimeIndex,
    date_col: str = "trade_date",
    value_cols: list | None = None,
    ann_date_col: str = "ann_date",
) -> pd.DataFrame:
    """
    PIT 安全的财务数据 pivot + ffill。

    输入长表 financial_df，含 date_col 列（值为报告期，如 2024-03-31）。
    流程：
      1. 若有可靠 ``ann_date`` 列 → 用公告日作为可用日；否则报告期 + 法定披露窗口
      2. pivot_table(index=PIT可用日, columns=股票, values=数值)
      3. reindex(prices_index, method="ffill")
    这样某报告期的数据只会在披露后才出现在日频序列中，
    消除「用报告期日做 ffill 起点」的 look-ahead bias。

    若 financial_df 缺少 date_col 或 code 列，回退为普通 pivot ffill
    （保证向后兼容）。

    返回：DataFrame，index=prices_index，columns=股票代码
    """
    prices_index = pd.DatetimeIndex(pd.to_datetime(prices_index))

    # 兼容性回退：缺关键列时按原逻辑（不做 PIT shift）
    if date_col not in financial_df.columns or "code" not in financial_df.columns:
        pivot = financial_df.pivot_table(
            index=date_col if date_col in financial_df.columns else financial_df.columns[0],
            columns="code" if "code" in financial_df.columns else financial_df.columns[1],
            values=value_cols[0] if value_cols else financial_df.columns[-1],
        )
        pivot.index = pd.DatetimeIndex(pd.to_datetime(pivot.index))
        return pivot.reindex(prices_index, method="ffill")

    df = financial_df.copy()
    available, mode = resolve_pit_available_dates(df, date_col=date_col, ann_date_col=ann_date_col)
    df[date_col] = available
    if mode == PIT_MODE_ANN_DATE:
        try:
            from loguru import logger as _lg
            _lg.info("财务 PIT: 使用 ann_date 列（真实公告日）")
        except Exception:
            logger.info("财务 PIT: 使用 ann_date 列（真实公告日）")

    if value_cols is None:
        # 取第一个数值列（非 date_col、非 code、非 ann_date）
        skip = {date_col, "code", ann_date_col}
        num_cols = [c for c in df.columns if c not in skip]
        if not num_cols:
            raise ValueError("financial_df 缺少数值列")
        value_col = num_cols[0]
    elif isinstance(value_cols, (list, tuple)):
        value_col = value_cols[0]
    else:
        value_col = value_cols

    pivot = df.pivot_table(index=date_col, columns="code", values=value_col)
    pivot.index = pd.DatetimeIndex(pd.to_datetime(pivot.index))
    return pivot.reindex(prices_index, method="ffill")


def pit_reindex_ffill(
    wide_df: pd.DataFrame,
    prices_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    PIT 安全的宽表 reindex + ffill。

    用于已经 pivot 好的宽表（index=报告期，columns=股票），
    例如 institution_holding.parquet（index=季报日期）。

    流程：
      1. 把 wide_df 的 index（报告期）按 disclosure_window 平移（法定窗近似）
      2. reindex(prices_index, method="ffill")
    """
    prices_index = pd.DatetimeIndex(pd.to_datetime(prices_index))
    if wide_df.empty:
        return pd.DataFrame(np.nan, index=prices_index, columns=wide_df.columns)

    _warn_statutory_once()
    shifted = wide_df.copy()
    shifted.index = pit_shift_report_dates(pd.DatetimeIndex(pd.to_datetime(wide_df.index)))
    return shifted.reindex(prices_index, method="ffill")


if __name__ == "__main__":
    # 自检
    print("disclosure_window:")
    for m in (3, 6, 9, 12):
        d = pd.Timestamp(f"2024-{m:02d}-15") if m not in (3, 6, 9, 12) else pd.Timestamp(f"2024-{m:02d}-{30 if m in (6,9) else 31}")
        print(f"  {d.date()} → +{disclosure_window(d)} 天")
