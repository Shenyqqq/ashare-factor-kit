"""
data/download_repurchase.py  —  东财股份回购公告

接口：ak.stock_repurchase_em()（全市场截面，含历史公告）
存储：data/raw/repurchase.parquet（长表）

用法：
    python -m data.download_repurchase
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

OUT_PATH = RAW_DIR / "repurchase.parquet"
MAX_RETRY = 3

COLS_MAP = {
    "股票代码": "code",
    "股票简称": "name",
    "最新价": "price",
    "计划回购价格区间": "plan_price",
    "计划回购数量区间-下限": "plan_qty_lo",
    "计划回购数量区间-上限": "plan_qty_hi",
    "占公告前一日总股本比例-下限": "plan_pct_lo",
    "占公告前一日总股本比例-上限": "plan_pct_hi",
    "计划回购金额区间-下限": "plan_amt_lo",
    "计划回购金额区间-上限": "plan_amt_hi",
    "回购起始时间": "start_date",
    "实施进度": "progress",
    "已回购股份价格区间-下限": "done_price_lo",
    "已回购股份价格区间-上限": "done_price_hi",
    "已回购股份数量": "done_qty",
    "已回购金额": "done_amt",
    "最新公告日期": "announce_date",
}


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=COLS_MAP)
    keep = [c for c in COLS_MAP.values() if c in df.columns]
    df = df[keep].copy()
    df["code"] = df["code"].astype(str).str.zfill(6)
    for c in ("announce_date", "start_date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    num_cols = [
        "price", "plan_price", "plan_qty_lo", "plan_qty_hi",
        "plan_pct_lo", "plan_pct_hi", "plan_amt_lo", "plan_amt_hi",
        "done_price_lo", "done_price_hi", "done_qty", "done_amt",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["code", "announce_date"])


def download_repurchase() -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(MAX_RETRY):
        try:
            raw = ak.stock_repurchase_em()
            df = _clean(raw)
            if df.empty:
                logger.warning("回购接口返回空")
                return pd.DataFrame()
            # 与已有合并（按 code+announce_date+progress 去重）
            if OUT_PATH.exists():
                old = pd.read_parquet(OUT_PATH)
                df = pd.concat([old, df], ignore_index=True)
            df = df.drop_duplicates(
                subset=["code", "announce_date", "progress"],
                keep="last",
            ).sort_values(["announce_date", "code"]).reset_index(drop=True)
            df.to_parquet(OUT_PATH)
            logger.info(f"回购保存: shape={df.shape} → {OUT_PATH}")
            return df
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    logger.warning(f"回购下载失败: {last_err}")
    if OUT_PATH.exists():
        return pd.read_parquet(OUT_PATH)
    return pd.DataFrame()


def main():
    argparse.ArgumentParser(description="下载股份回购公告").parse_args()
    download_repurchase()


if __name__ == "__main__":
    main()
