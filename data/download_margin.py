"""
data/download_margin.py  —  个股融资融券明细 + 融资余额宽表（合并脚本）

数据用途
    个股融资融券全字段（融资余额/融资买入额/融资偿还额/融券余额/融券卖出量/融券偿还量/
    融资融券余额）+ 派生的融资余额宽表（供 run.py 现有 wiring 使用）。

接口
    ak.stock_margin_detail_sse(date)   上交所融资融券明细（按交易日逐日拉）
    ak.stock_margin_detail_szse(date)  深交所融资融券明细（按交易日逐日拉）
    两交易所合并为统一长表。

输出（双产物，单次拉取派生）
    1. data/raw/margin_detail.parquet  —— 长表，完整字段（数据源 / source of truth）：
       date, code, name, exchange, margin_balance, margin_buy_amount, margin_repay_amount,
       short_balance_volume, short_sell_volume, short_repay_volume,
       short_balance_amount, total_margin_balance
       SSE/SZSE 缺失列以 NaN 填充，schema 统一。

    2. data/raw/margin_balance.parquet —— 宽表 DataFrame(index=date, columns=code)，
       值 = 融资余额（margin_balance）。**从长表 pivot 派生**，保证 run.py 现有 wiring
       （_load_opt("margin_balance.parquet") → clean_aux_panel → margin 变量）无缝兼容。

频率
    日频。仅交易日有数据（用交易日历 tool_trade_date_hist_sina 过滤）。

下载策略
    按交易日逐日拉取（SSE + SZSE 各一次），并发 4 线程，sleep 0.15s 避免限流。
    SSE/SZSE 限流敏感，建议保守。失败重试 MAX_RETRY=3，指数退避。

增量续传
    读已存在 margin_detail.parquet（source of truth），对已有 date 跳过；中断后重跑自动续。
    margin_balance.parquet 每次保存时从完整长表重新 pivot 派生，保证两文件同步。

用法
    python -m data.download_margin
    python -m data.download_margin --start 2018-01-01 --end 2026-07-03
    python -m data.download_margin --start 2018-01-01 --end 2026-07-03 --sample 10
"""
from __future__ import annotations

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

import akshare as ak
import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import RAW_DIR

MARGIN_WORKERS = 4
MAX_RETRY = 3
RETRY_DELAY = 2.0
SAVE_EVERY = 30
SLEEP_BETWEEN_EXCHANGES = 0.15

OUT_LONG_PATH = RAW_DIR / "margin_detail.parquet"
OUT_WIDE_PATH = RAW_DIR / "margin_balance.parquet"

SSE_COLS_MAP = {
    "信用交易日期": "date",
    "标的证券代码": "code",
    "标的证券简称": "name",
    "融资余额": "margin_balance",
    "融资买入额": "margin_buy_amount",
    "融资偿还额": "margin_repay_amount",
    "融券余量": "short_balance_volume",
    "融券卖出量": "short_sell_volume",
    "融券偿还量": "short_repay_volume",
}

SZSE_COLS_MAP = {
    "证券代码": "code",
    "证券简称": "name",
    "融资买入额": "margin_buy_amount",
    "融资余额": "margin_balance",
    "融券卖出量": "short_sell_volume",
    "融券余量": "short_balance_volume",
    "融券余额": "short_balance_amount",
    "融资融券余额": "total_margin_balance",
}

NUMERIC_COLS = [
    "margin_balance", "margin_buy_amount", "margin_repay_amount",
    "short_balance_volume", "short_sell_volume", "short_repay_volume",
    "short_balance_amount", "total_margin_balance",
]

# 统一 schema（长表）
ALL_COLS = [
    "date", "code", "name", "exchange",
    "margin_balance", "margin_buy_amount", "margin_repay_amount",
    "short_balance_volume", "short_sell_volume", "short_repay_volume",
    "short_balance_amount", "total_margin_balance",
]


def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)


_last_trade_date_cache = None


def get_trade_calendar() -> pd.Series:
    """获取 A 股交易日历（仅交易日发起请求，避免浪费）。"""
    global _last_trade_date_cache
    if _last_trade_date_cache is not None:
        return _last_trade_date_cache
    cal = ak.tool_trade_date_hist_sina()
    dates = pd.to_datetime(cal["trade_date"])
    _last_trade_date_cache = dates.sort_values().reset_index(drop=True)
    return _last_trade_date_cache


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """统一 schema：code 补零、数值列转 numeric + inf→NaN、补缺失列。"""
    if "code" in df.columns:
        df["code"] = df["code"].astype(str).str.zfill(6)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        else:
            df[col] = np.nan
    if "name" in df.columns:
        df["name"] = df["name"].astype("string")
    for col in ALL_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df[ALL_COLS]


def _clean_sse(df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=ALL_COLS)
    df = df.rename(columns=SSE_COLS_MAP)
    keep = [c for c in SSE_COLS_MAP.values() if c in df.columns]
    df = df[keep].copy()
    df["exchange"] = "SSE"
    df["date"] = pd.to_datetime(date_str, format="%Y%m%d", errors="coerce")
    return _finalize(df)


def _clean_szse(df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=ALL_COLS)
    df = df.rename(columns=SZSE_COLS_MAP)
    keep = [c for c in SZSE_COLS_MAP.values() if c in df.columns]
    df = df[keep].copy()
    df["exchange"] = "SZSE"
    df["date"] = pd.to_datetime(date_str, format="%Y%m%d", errors="coerce")
    return _finalize(df)


def fetch_one_date(date_ts: pd.Timestamp) -> pd.DataFrame | None:
    """拉取单日 SSE + SZSE 融资融券明细，合并为长表一行集合。"""
    date_str = date_ts.strftime("%Y%m%d")
    parts: list[pd.DataFrame] = []
    for ex, fn_name, cleaner in [
        ("SSE", "stock_margin_detail_sse", _clean_sse),
        ("SZSE", "stock_margin_detail_szse", _clean_szse),
    ]:
        success = False
        for attempt in range(MAX_RETRY + 1):
            try:
                fn = getattr(ak, fn_name)
                raw = fn(date=date_str)
                parts.append(cleaner(raw, date_str))
                success = True
                break
            except Exception as e:
                if attempt < MAX_RETRY:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    logger.warning(f"{ex} {date_str} 失败: {e}")
                    # 单边失败不影响另一边
                    parts.append(pd.DataFrame(columns=ALL_COLS))
        if success:
            time.sleep(SLEEP_BETWEEN_EXCHANGES)
    if not parts:
        return None
    try:
        return pd.concat(parts, ignore_index=True)
    except Exception:
        return None


def _pivot_balance(long_df: pd.DataFrame) -> pd.DataFrame:
    """从长表派生融资余额宽表：DataFrame(index=date, columns=code, values=margin_balance)。"""
    if long_df is None or long_df.empty:
        return pd.DataFrame()
    sub = long_df[["date", "code", "margin_balance"]].copy()
    sub["margin_balance"] = pd.to_numeric(sub["margin_balance"], errors="coerce")
    # 同一 (date, code) 可能因 SSE/SZSE 重复（理论不会，防御）取 last
    wide = (
        sub.drop_duplicates(subset=["date", "code"], keep="last")
        .pivot(index="date", columns="code", values="margin_balance")
        .sort_index()
    )
    wide.columns.name = None
    return wide


def _save_both(records: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """合并 records 与已有长表，落盘长表 + 派生宽表。返回 (long, wide)。"""
    if not records:
        return pd.DataFrame(), pd.DataFrame()
    long_df = pd.concat(records, ignore_index=True)
    long_df = long_df.drop_duplicates(
        subset=["date", "code", "exchange"], keep="last"
    )
    long_df = long_df.sort_values(["date", "code", "exchange"]).reset_index(drop=True)
    long_df.to_parquet(OUT_LONG_PATH)

    wide_df = _pivot_balance(long_df)
    wide_df.to_parquet(OUT_WIDE_PATH)
    return long_df, wide_df


def main(start: str, end: str, sample: int = 0):
    ensure_dirs()
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    logger.info("加载交易日历...")
    cal = get_trade_calendar()
    dates = cal[(cal >= s) & (cal <= e)].tolist()
    if sample:
        dates = dates[:sample]
        logger.info(f"调试模式：仅处理前 {sample} 个交易日")
    logger.info(f"融资融券下载：{len(dates)} 个交易日，并发={MARGIN_WORKERS}")

    # 长表为 source of truth
    existing_long = pd.read_parquet(OUT_LONG_PATH) if OUT_LONG_PATH.exists() else None
    done_dates: set[pd.Timestamp] = set()
    records: list[pd.DataFrame] = []
    if existing_long is not None and not existing_long.empty:
        existing_long["code"] = existing_long["code"].astype(str).str.zfill(6)
        if "date" in existing_long.columns:
            existing_long["date"] = pd.to_datetime(existing_long["date"], errors="coerce")
            done_dates = set(existing_long["date"].dropna().unique())
        records.append(existing_long)
        logger.info(f"已存在 {len(done_dates)} 个交易日，跳过")

    need = [d for d in dates if d not in done_dates]
    if not need:
        logger.info("全部已下载，跳过")
        # 即使无新增也重写一次宽表，保证两文件同步
        if records:
            long_df, wide_df = _save_both(records)
            logger.info(
                f"宽表 margin_balance.parquet 同步刷新: shape={wide_df.shape}"
            )
        return

    lock = threading.Lock()
    failed: list[str] = []
    success = empty = done = 0

    def _save():
        with lock:
            if not records:
                return
            long_df, wide_df = _save_both(records)
            logger.info(
                f"保存：长表 shape={long_df.shape}，宽表 shape={wide_df.shape}"
            )

    with ThreadPoolExecutor(max_workers=MARGIN_WORKERS) as executor:
        futures = {executor.submit(fetch_one_date, d): d for d in need}
        for future in as_completed(futures):
            d = futures[future]
            df = future.result()
            with lock:
                done += 1
                if df is None:
                    failed.append(d.strftime("%Y-%m-%d"))
                elif df.empty:
                    empty += 1
                else:
                    records.append(df)
                    success += 1
                if done % SAVE_EVERY == 0:
                    logger.info(
                        f"融资融券进度 {done}/{len(need)} "
                        f"成功={success} 空数据={empty} 失败={len(failed)}"
                    )
                    _save()

    _save()
    logger.info(f"完成：成功 {success}，空数据 {empty}，失败 {len(failed)}")
    if failed:
        logger.warning(f"失败日期(前10): {failed[:10]}")
    if OUT_LONG_PATH.exists():
        final_long = pd.read_parquet(OUT_LONG_PATH)
        logger.info(f"最终 {OUT_LONG_PATH.name}: shape={final_long.shape}")
    if OUT_WIDE_PATH.exists():
        final_wide = pd.read_parquet(OUT_WIDE_PATH)
        logger.info(f"最终 {OUT_WIDE_PATH.name}: shape={final_wide.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="下载个股融资融券明细（SSE+SZSE，按交易日）+ 派生融资余额宽表"
    )
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--sample", type=int, default=0,
                        help="调试：仅处理前 N 个交易日")
    args = parser.parse_args()
    main(args.start, args.end, args.sample)
