"""
utils/pit_align.py — Point-in-Time 财务数据对齐

A 股财报按法定披露窗口做保守 PIT 对齐：
  报告期 + 披露窗口 → 可用日下界
然后对可用日做 reindex + ffill 到日频价格序列。

不依赖 AKShare 公告日（接口不提供 ann_date），用法定披露截止日作保守下界。

A 股法定披露窗口（保守取上限）：
  - Q1 季报（03-31）：截止 04-30（+30 天，实际常延迟，取 +45 天保守）
  - 半年报（06-30）：截止 08-31（+62 天，取 +75 天保守）
  - Q3 季报（09-30）：截止 10-31（+31 天，取 +45 天保守）
  - 年报（12-31）：截止 04-30（+120 天，取 +120 天保守）

修复 `factors/factor.py:_pivot_financial` 与 `factors/barra_risk.py:_pivot_ffill`
原 `pivot(trade_date).reindex(prices.index, method="ffill")` 把报告期数据
ffill 到该日之后所有日子，造成 look-ahead bias（季报 ~15-25 个交易日，
年报最坏 30-60 个交易日）。
"""
from __future__ import annotations

import pandas as pd
import numpy as np


# 报告期月份 → 法定披露窗口天数（保守上限）
_DISCLOSURE_WINDOWS: dict[int, int] = {
    3: 45,    # Q1 季报
    6: 75,    # 半年报
    9: 45,    # Q3 季报
    12: 120,  # 年报
}


def disclosure_window(report_period: pd.Period | pd.Timestamp) -> int:
    """
    按报告期月份返回披露窗口天数：
      03 → 45（Q1）
      06 → 75（半年报）
      09 → 45（Q3）
      12 → 120（年报）
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


def pit_pivot_ffill(
    financial_df: pd.DataFrame,
    prices_index: pd.DatetimeIndex,
    date_col: str = "trade_date",
    value_cols: list | None = None,
) -> pd.DataFrame:
    """
    PIT 安全的财务数据 pivot + ffill。

    输入长表 financial_df，含 date_col 列（值为报告期，如 2024-03-31）。
    流程：
      1. 把 date_col 替换为 报告期 + disclosure_window（PIT 可用日下界）
      2. pivot_table(index=PIT可用日, columns=股票, values=数值)
      3. reindex(prices_index, method="ffill")
    这样某报告期的数据只会在披露窗口后才出现在日频序列中，
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
    report_periods = pd.DatetimeIndex(pd.to_datetime(df[date_col]))
    df[date_col] = pit_shift_report_dates(report_periods)

    if value_cols is None:
        # 取第一个数值列（非 date_col、非 code）
        num_cols = [c for c in df.columns if c not in (date_col, "code")]
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
      1. 把 wide_df 的 index（报告期）按 disclosure_window 平移
      2. reindex(prices_index, method="ffill")
    """
    prices_index = pd.DatetimeIndex(pd.to_datetime(prices_index))
    if wide_df.empty:
        return pd.DataFrame(np.nan, index=prices_index, columns=wide_df.columns)

    shifted = wide_df.copy()
    shifted.index = pit_shift_report_dates(pd.DatetimeIndex(pd.to_datetime(wide_df.index)))
    return shifted.reindex(prices_index, method="ffill")


if __name__ == "__main__":
    # 自检
    print("disclosure_window:")
    for m in (3, 6, 9, 12):
        d = pd.Timestamp(f"2024-{m:02d}-15") if m not in (3, 6, 9, 12) else pd.Timestamp(f"2024-{m:02d}-{30 if m in (6,9) else 31}")
        print(f"  {d.date()} → +{disclosure_window(d)} 天")
