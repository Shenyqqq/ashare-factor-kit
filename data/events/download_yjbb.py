"""
data/events/download_yjbb.py  —  东财业绩快报/正式稿（yjbb）

接口：ak.stock_yjbb_em(date='YYYYMMDD')  按报告期全市场
存储：data/raw/yjbb.parquet

注意：东财「最新公告日期」可能是修订日而非首次公告日；因子侧优先用该字段，
并在文档标 PIT 近似风险。

用法：
    python -m data.events.download_yjbb
    python -m data.events.download_yjbb --start-year 2018 --sample 4
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import akshare as ak
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import RAW_DIR

OUT_PATH = RAW_DIR / "yjbb.parquet"
MAX_RETRY = 3
SLEEP = 0.6

COLS_MAP = {
    "股票代码": "code",
    "股票简称": "name",
    "每股收益": "eps",
    "营业总收入-营业总收入": "revenue",
    "营业总收入-同比增长": "revenue_yoy",
    "营业总收入-季度环比增长": "revenue_qoq",
    "净利润-净利润": "net_profit",
    "净利润-同比增长": "net_profit_yoy",
    "净利润-季度环比增长": "net_profit_qoq",
    "每股净资产": "bvps",
    "净资产收益率": "roe",
    "每股经营现金流量": "ocf_ps",
    "销售毛利率": "gross_margin",
    "所处行业": "industry",
    "最新公告日期": "announce_date",
}


def _quarter_dates(start_year: int, end_year: int) -> list[str]:
    out = []
    today = pd.Timestamp.today()
    for y in range(start_year, end_year + 1):
        for q in ("0331", "0630", "0930", "1231"):
            d = f"{y}{q}"
            if pd.Timestamp(f"{y}-{q[:2]}-{q[2:]}") <= today:
                out.append(d)
    return out


def _clean(df: pd.DataFrame, report_date: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=COLS_MAP)
    keep = [c for c in COLS_MAP.values() if c in df.columns]
    out = df[keep].copy()
    out["code"] = out["code"].astype(str).str.zfill(6)
    out["report_date"] = report_date
    out["announce_date"] = pd.to_datetime(out.get("announce_date"), errors="coerce")
    for c in (
        "eps", "revenue", "revenue_yoy", "revenue_qoq",
        "net_profit", "net_profit_yoy", "net_profit_qoq",
        "bvps", "roe", "ocf_ps", "gross_margin",
    ):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(subset=["code"])


def download_yjbb(
    start_year: int = 2018,
    end_year: int | None = None,
    sample: int = 0,
) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    end_year = end_year or pd.Timestamp.today().year
    quarters = _quarter_dates(start_year, end_year)
    if sample:
        quarters = quarters[:sample]

    existing = pd.read_parquet(OUT_PATH) if OUT_PATH.exists() else pd.DataFrame()
    done = set(existing["report_date"].astype(str).unique()) if not existing.empty else set()
    need = [q for q in quarters if q not in done]
    if not need:
        logger.info("业绩快报/正式稿已是最新，跳过")
        return existing

    logger.info(f"yjbb：需下载 {len(need)}/{len(quarters)} 个报告期")
    records = [existing] if not existing.empty else []
    failed: list[str] = []

    for i, q in enumerate(need):
        ok = False
        for attempt in range(MAX_RETRY):
            try:
                raw = ak.stock_yjbb_em(date=q)
                df = _clean(raw, q)
                if not df.empty:
                    records.append(df)
                    logger.info(f"[{i + 1}/{len(need)}] {q}: {len(df)} 条")
                ok = True
                break
            except Exception as e:
                if attempt + 1 < MAX_RETRY:
                    time.sleep(SLEEP * (attempt + 2))
                else:
                    logger.warning(f"yjbb {q} 失败: {e}")
                    failed.append(q)
        time.sleep(SLEEP)

    if not records:
        return existing
    result = pd.concat(records, ignore_index=True)
    result = result.drop_duplicates(
        subset=["code", "report_date"], keep="last"
    ).sort_values(["announce_date", "code"]).reset_index(drop=True)
    result.to_parquet(OUT_PATH)
    logger.info(f"yjbb 保存: shape={result.shape} → {OUT_PATH}")
    if failed:
        logger.warning(f"失败报告期: {failed}")
    return result


def main():
    p = argparse.ArgumentParser(description="下载业绩快报/正式稿 yjbb")
    p.add_argument("--start-year", type=int, default=2018)
    p.add_argument("--end-year", type=int, default=None)
    p.add_argument("--sample", type=int, default=0)
    args = p.parse_args()
    download_yjbb(args.start_year, args.end_year, args.sample)


if __name__ == "__main__":
    main()
