"""
data/events/download_yjyg.py  —  下载全市场历史业绩预告

一次请求=全市场当期所有预告，按报告期循环拉取。
结果保存至 data/raw/yjyg.parquet

用法:
    python -m data.events.download_yjyg
"""
import time
from pathlib import Path
import sys

import akshare as ak
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import RAW_DIR

OUT_PATH = RAW_DIR / "yjyg.parquet"

# 2018Q1 ~ 2025Q2 所有报告期
REPORT_DATES = [
    f"{y}{q}"
    for y in range(2018, 2026)
    for q in ["0331", "0630", "0930", "1231"]
    if f"{y}{q}" <= "20250630"
]

# 列名映射（用列索引，避免编码问题）
COL_NAMES = [
    "seq", "code", "name", "indicator", "change_desc",
    "forecast_value", "change_pct", "change_reason",
    "forecast_type", "last_year_value", "announce_date"
]


def download():
    existing = pd.read_parquet(OUT_PATH) if OUT_PATH.exists() else pd.DataFrame()
    done_periods = set(existing["report_date"].unique()) if not existing.empty else set()

    todo = [d for d in REPORT_DATES if d not in done_periods]
    if not todo:
        logger.info("业绩预告数据已是最新，跳过下载")
        return existing

    logger.info(f"共 {len(REPORT_DATES)} 个报告期，需下载 {len(todo)} 个")

    records = [existing] if not existing.empty else []
    failed = []

    for i, date in enumerate(todo):
        try:
            df = ak.stock_yjyg_em(date=date)
            df.columns = COL_NAMES
            df["report_date"] = date
            df["announce_date"] = pd.to_datetime(df["announce_date"], errors="coerce")
            df["code"] = df["code"].astype(str).str.zfill(6)
            df["change_pct"] = pd.to_numeric(df["change_pct"], errors="coerce")
            df["forecast_value"] = pd.to_numeric(df["forecast_value"], errors="coerce")
            df["last_year_value"] = pd.to_numeric(df["last_year_value"], errors="coerce")
            records.append(df)
            logger.info(f"[{i+1}/{len(todo)}] {date}: {len(df)}条")
        except Exception as e:
            logger.warning(f"[{i+1}/{len(todo)}] {date}: 失败 {e}")
            failed.append(date)

        time.sleep(0.5)

    result = pd.concat(records, ignore_index=True)
    result = result.drop_duplicates(subset=["code", "report_date", "indicator"])
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUT_PATH)

    logger.info(f"完成: {len(result)}条，保存至 {OUT_PATH}")
    if failed:
        logger.warning(f"失败报告期: {failed}")
    return result


if __name__ == "__main__":
    download()
