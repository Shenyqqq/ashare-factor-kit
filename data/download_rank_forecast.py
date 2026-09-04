"""
data/download_rank_forecast.py  —  巨潮资讯按日评级变动

接口：ak.stock_rank_forecast_cninfo(date='YYYYMMDD')
存储：data/raw/rank_forecast.parquet（长表）

用法：
    python -m data.download_rank_forecast
    python -m data.download_rank_forecast --start 2024-01-01 --sample 20
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import akshare as ak
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import RAW_DIR

OUT_PATH = RAW_DIR / "rank_forecast.parquet"
MAX_RETRY = 3
SLEEP = 0.4

COLS_MAP = {
    "证券代码": "code",
    "证券简称": "name",
    "发布日期": "announce_date",
    "研究机构简称": "institute",
    "研究员名称": "analyst",
    "投资评级": "rating",
    "是否首次评级": "is_first",
    "评级变化": "rating_change",
    "前一次投资评级": "prev_rating",
    "目标价格-下限": "target_low",
    "目标价格-上限": "target_high",
}


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=COLS_MAP)
    keep = [c for c in COLS_MAP.values() if c in df.columns]
    df = df[keep].copy()
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["announce_date"] = pd.to_datetime(df["announce_date"], errors="coerce")
    for c in ("target_low", "target_high"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("name", "institute", "analyst", "rating", "is_first",
              "rating_change", "prev_rating"):
        if c in df.columns:
            df[c] = df[c].astype("string")
    return df.dropna(subset=["code", "announce_date"])


def download_rank_forecast(
    start: str = "2018-01-01",
    end: str | None = None,
    sample: int = 0,
    sleep: float = SLEEP,
) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    end_ts = pd.Timestamp(end or pd.Timestamp.today().normalize())
    dates = pd.bdate_range(start, end_ts)
    if sample:
        dates = dates[:sample]
        logger.info(f"调试：仅前 {sample} 个交易日")

    existing = pd.read_parquet(OUT_PATH) if OUT_PATH.exists() else pd.DataFrame()
    done: set[str] = set()
    if not existing.empty and "announce_date" in existing.columns:
        existing["announce_date"] = pd.to_datetime(existing["announce_date"], errors="coerce")
        done = set(existing["announce_date"].dropna().dt.strftime("%Y%m%d"))

    need = [d for d in dates if d.strftime("%Y%m%d") not in done]
    if not need:
        logger.info("评级变动已是最新，跳过")
        return existing

    logger.info(f"评级变动：需下载 {len(need)}/{len(dates)} 日")
    records = [existing] if not existing.empty else []
    failed: list[str] = []

    for i, d in enumerate(need):
        ds = d.strftime("%Y%m%d")
        ok = False
        for attempt in range(MAX_RETRY):
            try:
                raw = ak.stock_rank_forecast_cninfo(date=ds)
                df = _clean(raw)
                if not df.empty:
                    records.append(df)
                ok = True
                break
            except Exception as e:
                if attempt + 1 < MAX_RETRY:
                    time.sleep(SLEEP * (attempt + 2))
                else:
                    logger.warning(f"评级变动 {ds} 失败: {e}")
                    failed.append(ds)
        if ok and (i + 1) % 20 == 0:
            result = pd.concat(records, ignore_index=True)
            result = result.drop_duplicates(
                subset=["code", "announce_date", "institute", "rating"],
                keep="last",
            )
            result.to_parquet(OUT_PATH)
            logger.info(f"进度 {i + 1}/{len(need)}，已落盘 {len(result)} 行")
        time.sleep(sleep)

    if not records:
        return existing
    result = pd.concat(records, ignore_index=True)
    result = result.drop_duplicates(
        subset=["code", "announce_date", "institute", "rating"],
        keep="last",
    ).sort_values(["announce_date", "code"]).reset_index(drop=True)
    result.to_parquet(OUT_PATH)
    logger.info(f"评级变动保存: shape={result.shape} → {OUT_PATH}")
    if failed:
        logger.warning(f"失败 {len(failed)} 日: {failed[:10]}")
    return result


def main():
    p = argparse.ArgumentParser(description="下载巨潮按日评级变动")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--sleep", type=float, default=SLEEP)
    args = p.parse_args()
    download_rank_forecast(args.start, args.end, args.sample, args.sleep)


if __name__ == "__main__":
    main()
