"""
data/download_shareholder.py  —  股东户数 / 筹码集中度下载

数据用途
    小盘股策略的「筹码集中度」因子：股东户数环比变化、户均流通市值/股数。
    股东户数下降 → 筹码集中 → 看多信号（户均市值上升同样代表筹码集中）。

接口
    ak.stock_zh_a_gdhs_detail_em(symbol=<6位代码>)   东方财富-股东户数详情
    按股票逐只拉取，返回该股票全部历史季频股东户数记录。

频率
    季频（每只股票 ~60 条历史，每季公告一次）。

输出 schema（长表，data/raw/shareholder_count.parquet）
    code                    str     6位代码（已 zfill）
    report_date             datetime 报告期（股东户数统计截止日）
    announce_date           datetime 股东户数公告日期
    holder_count            int     本次股东户数
    holder_count_prev       int     上次股东户数
    holder_count_change     int     股东户数增减
    holder_count_pct_change float   股东户数增减比例（%）
    avg_float_shares        float   户均持股数量（≈户均流通股）
    avg_float_market_value  float   户均持股市值（≈户均流通市值）
    total_market_value      float   总市值
    total_shares            float   总股本
    period_return           float   区间涨跌幅（%）
    share_change            float   股本变动
    share_change_reason     str     股本变动原因

并发 & 限流
    ThreadPoolExecutor, 4-8 线程；每股 sleep 0.1s；失败重试 2 次。
    5000 只 × ~0.5s/请求 / 8 线程 ≈ 5-8 分钟。

增量续传
    读已存在 parquet，对已有 code 整体跳过（一只股票的全部历史在单次 API 调用里返回）。
    删除 parquet 可强制重新下载。

用法
    python -m data.download_shareholder --start 2018-01-01
    python -m data.download_shareholder --start 2018-01-01 --sample 100
"""
from __future__ import annotations

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import RAW_DIR, UNIVERSE_DIR

WORKERS = 6
MAX_RETRY = 2
RETRY_DELAY = 1.0
SAVE_EVERY = 100

OUT_PATH = RAW_DIR / "shareholder_count.parquet"
STOCK_LIST_PATH = UNIVERSE_DIR / "stock_list.parquet"

COLS_MAP = {
    "代码": "code",
    "股东户数统计截止日": "report_date",
    "股东户数公告日期": "announce_date",
    "股东户数-本次": "holder_count",
    "股东户数-上次": "holder_count_prev",
    "股东户数-增减": "holder_count_change",
    "股东户数-增减比例": "holder_count_pct_change",
    "户均持股市值": "avg_float_market_value",
    "户均持股数量": "avg_float_shares",
    "总市值": "total_market_value",
    "总股本": "total_shares",
    "区间涨跌幅": "period_return",
    "股本变动": "share_change",
    "股本变动原因": "share_change_reason",
    "名称": "name",
}

NUMERIC_COLS = [
    "holder_count", "holder_count_prev", "holder_count_change",
    "holder_count_pct_change", "avg_float_market_value", "avg_float_shares",
    "total_market_value", "total_shares", "period_return", "share_change",
]


def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)


def load_codes(sample: int = 0) -> list[str]:
    if not STOCK_LIST_PATH.exists():
        raise FileNotFoundError(f"股票列表不存在: {STOCK_LIST_PATH}，先运行 python -m data.download")
    df = pd.read_parquet(STOCK_LIST_PATH)
    codes = df["code"].astype(str).str.zfill(6).tolist()
    if sample:
        codes = codes[:sample]
        logger.info(f"调试模式：仅处理 {sample} 只股票")
    return codes


def _clean_shareholder(df: pd.DataFrame) -> pd.DataFrame:
    """清洗单只股票的原始返回：列重命名 + 类型转换 + inf→NaN + code 补零。"""
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.rename(columns=COLS_MAP)
    keep = [c for c in COLS_MAP.values() if c in df.columns]
    df = df[keep].copy()

    if "code" in df.columns:
        df["code"] = df["code"].astype(str).str.zfill(6)
    for col in ("report_date", "announce_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    if "share_change_reason" in df.columns:
        df["share_change_reason"] = df["share_change_reason"].astype("string")
    return df


def fetch_one(code: str) -> pd.DataFrame | None:
    for attempt in range(MAX_RETRY + 1):
        try:
            df = ak.stock_zh_a_gdhs_detail_em(symbol=code)
            time.sleep(0.1)
            return _clean_shareholder(df)
        except Exception as e:
            if attempt < MAX_RETRY:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                logger.warning(f"股东户数失败 {code} (重试{MAX_RETRY}次): {e}")
                return None


def filter_by_start(df: pd.DataFrame, start: str) -> pd.DataFrame:
    if df.empty or "report_date" not in df.columns:
        return df
    cutoff = pd.Timestamp(start)
    return df[df["report_date"] >= cutoff].copy()


def main(start: str, sample: int = 0):
    ensure_dirs()
    codes = load_codes(sample)
    logger.info(f"股东户数下载：{len(codes)} 只股票，并发={WORKERS}")

    existing = pd.read_parquet(OUT_PATH) if OUT_PATH.exists() else None
    records: list[pd.DataFrame] = []
    done_codes: set[str] = set()
    if existing is not None and not existing.empty:
        existing["code"] = existing["code"].astype(str).str.zfill(6)
        existing = filter_by_start(existing, start)
        records.append(existing)
        done_codes = set(existing["code"].unique())
        logger.info(f"已存在 {len(done_codes)} 只，跳过")

    need = [c for c in codes if c not in done_codes]
    if not need:
        logger.info("全部已下载，跳过")
        return

    lock = threading.Lock()
    failed: list[str] = []
    success = skip = done = 0

    def _save():
        if not records:
            return
        result = pd.concat(records, ignore_index=True)
        result = filter_by_start(result, start)
        result = result.drop_duplicates(subset=["code", "report_date"], keep="last")
        result = result.sort_values(["code", "report_date"]).reset_index(drop=True)
        result.to_parquet(OUT_PATH)

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(fetch_one, c): c for c in need}
        for future in as_completed(futures):
            code = futures[future]
            df = future.result()
            with lock:
                done += 1
                if df is not None and not df.empty:
                    records.append(df)
                    success += 1
                else:
                    skip += 1
                    failed.append(code)
                if done % SAVE_EVERY == 0:
                    logger.info(
                        f"股东户数进度 {done}/{len(need)} 成功={success} 跳过={skip} 失败={len(failed)}"
                    )
                    _save()

    _save()
    logger.info(f"完成：成功 {success}，跳过 {skip}，失败 {len(failed)}")
    if failed:
        logger.warning(f"失败列表(前20): {failed[:20]}")
    if OUT_PATH.exists():
        final = pd.read_parquet(OUT_PATH)
        logger.info(f"最终 {OUT_PATH.name}: shape={final.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载股东户数（季频，按股票）")
    parser.add_argument("--start", default="2018-01-01",
                        help="报告期下限 YYYY-MM-DD")
    parser.add_argument("--sample", type=int, default=0,
                        help="调试：仅处理前 N 只股票")
    args = parser.parse_args()
    main(args.start, args.sample)
