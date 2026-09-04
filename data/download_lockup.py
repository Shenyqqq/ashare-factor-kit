"""
data/download_lockup.py  —  限售解禁下载

数据用途
    小盘股策略的「供给冲击」因子：解禁市值占流通市值比例、解禁类型。
    大额解禁前 20 日存在系统性抛压（解禁前 20 日涨跌幅为负），解禁后压力释放。
    解禁市值/流通市值比例 > 阈值时回避，或做事件驱动。

接口
    ak.stock_restricted_release_detail_em(start_date, end_date)
        东方财富-限售解禁详情，按日期区间拉取，全市场一次返回。
    注：原候选 stock_released_em / stock_share_lock_em 在当前 akshare 版本不存在；
        stock_restricted_release_summary_em 仅返回日度汇总（无个股），故不采用。

频率
    日频（仅在有解禁的交易日才有数据；非解禁日无记录）。

输出 schema（长表，data/raw/lockup_release.parquet）
    code                    str     6位代码
    name                    str     股票简称
    release_date            datetime 解禁时间
    release_type            str     限售股类型（如「股权激励限售股份」「首发原股东限售股份」）
    release_shares          float   解禁数量（股）
    actual_release_shares   float   实际解禁数量（股）
    actual_release_value    float   实际解禁市值（元）
    pct_of_float            float   占解禁前流通市值比例
    close_prev               float   解禁前一交易日收盘价
    return_20d_before       float   解禁前 20 日涨跌幅（%）
    return_20d_after        float   解禁后 20 日涨跌幅（%，事后才有）

下载策略
    按月分块调用，6 年 ≈ 72 次请求。

增量续传
    读已存在 parquet，对已有 release_date（月份）整块跳过。

用法
    python -m data.download_lockup --start 2018-01-01 --end 2026-07-03
    python -m data.download_lockup --start 2018-01-01 --end 2026-07-03 --sample 100
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import RAW_DIR

MAX_RETRY = 3
RETRY_DELAY = 2.0

OUT_PATH = RAW_DIR / "lockup_release.parquet"

COLS_MAP = {
    "股票代码": "code",
    "股票简称": "name",
    "解禁时间": "release_date",
    "限售股类型": "release_type",
    "解禁数量": "release_shares",
    "实际解禁数量": "actual_release_shares",
    "实际解禁市值": "actual_release_value",
    "占解禁前流通市值比例": "pct_of_float",
    "解禁前一交易日收盘价": "close_prev",
    "解禁前20日涨跌幅": "return_20d_before",
    "解禁后20日涨跌幅": "return_20d_after",
}

NUMERIC_COLS = [
    "release_shares", "actual_release_shares", "actual_release_value",
    "pct_of_float", "close_prev", "return_20d_before", "return_20d_after",
]


def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def _clean_lockup(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=COLS_MAP)
    keep = [c for c in COLS_MAP.values() if c in df.columns]
    df = df[keep].copy()
    if "code" in df.columns:
        df["code"] = df["code"].astype(str).str.zfill(6)
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    # 解禁数量负值无意义，置 NaN
    for col in ("release_shares", "actual_release_shares", "actual_release_value"):
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan
    if "release_type" in df.columns:
        df["release_type"] = df["release_type"].astype("string")
    return df


def fetch_range(start_date: str, end_date: str) -> pd.DataFrame | None:
    for attempt in range(MAX_RETRY + 1):
        try:
            df = ak.stock_restricted_release_detail_em(
                start_date=start_date, end_date=end_date)
            time.sleep(0.3)
            return _clean_lockup(df)
        except Exception as e:
            if attempt < MAX_RETRY:
                wait = RETRY_DELAY * (attempt + 1)
                logger.debug(f"解禁 {start_date}~{end_date} 失败，{wait}s 后重试: {e}")
                time.sleep(wait)
            else:
                logger.warning(f"解禁 {start_date}~{end_date} 最终失败: {e}")
                return None


def month_chunks(start: str, end: str):
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    cur = s
    while cur <= e:
        nxt = (cur + pd.DateOffset(months=1)).replace(day=1) - pd.Timedelta(days=1)
        nxt = min(nxt, e)
        yield cur.strftime("%Y%m%d"), nxt.strftime("%Y%m%d"), cur.strftime("%Y-%m")
        cur = nxt + pd.Timedelta(days=1)


def main(start: str, end: str, sample: int = 0):
    ensure_dirs()
    chunks = list(month_chunks(start, end))
    if sample:
        chunks = chunks[:sample]
        logger.info(f"调试模式：仅处理前 {sample} 个月块")

    existing = pd.read_parquet(OUT_PATH) if OUT_PATH.exists() else None
    done_months: set[str] = set()
    records: list[pd.DataFrame] = []
    if existing is not None and not existing.empty:
        existing["code"] = existing["code"].astype(str).str.zfill(6)
        if "release_date" in existing.columns:
            existing["release_date"] = pd.to_datetime(
                existing["release_date"], errors="coerce")
            existing["_month"] = existing["release_date"].dt.to_period("M").astype(str)
            done_months = set(existing["_month"].unique())
            existing = existing.drop(columns=["_month"])
        records.append(existing)
        logger.info(f"已存在 {len(done_months)} 个月，跳过")

    need = [(s, e, m) for s, e, m in chunks if m not in done_months]
    if not need:
        logger.info("全部已下载，跳过")
        return

    logger.info(f"限售解禁下载：{len(need)} 个月块待下载")

    success = skip = failed = 0

    def _save():
        if not records:
            return
        result = pd.concat(records, ignore_index=True)
        result = result.drop_duplicates(
            subset=["code", "release_date", "release_type"], keep="last"
        )
        result = result.sort_values(["release_date", "code"]).reset_index(drop=True)
        result.to_parquet(OUT_PATH)

    for s, e, m in need:
        df = fetch_range(s, e)
        if df is not None and not df.empty:
            records.append(df)
            success += 1
            logger.info(f"  [{m}] {df.shape[0]} 条")
        elif df is None:
            failed += 1
        else:
            skip += 1
            logger.info(f"  [{m}] 无解禁数据")
        if (success + skip + failed) % 6 == 0:
            _save()

    _save()
    logger.info(f"完成：成功 {success}，无数据 {skip}，失败 {failed}")
    if OUT_PATH.exists():
        final = pd.read_parquet(OUT_PATH)
        logger.info(f"最终 {OUT_PATH.name}: shape={final.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载限售解禁详情（按月分块）")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--sample", type=int, default=0,
                        help="调试：仅处理前 N 个月块")
    args = parser.parse_args()
    main(args.start, args.end, args.sample)
