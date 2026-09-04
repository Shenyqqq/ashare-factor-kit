"""
data/download_research_report.py  —  东财个股研报列表（伪一致预期）

接口：ak.stock_research_report_em(symbol=code)
存储：data/raw/research_report.parquet（长表）

用法：
    python -m data.download_research_report
    python -m data.download_research_report --sample 50
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
import sys

import akshare as ak
import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import RAW_DIR, UNIVERSE_DIR

OUT_PATH = RAW_DIR / "research_report.parquet"
MAX_RETRY = 3
SLEEP = 0.25


def _normalize_eps_cols(df: pd.DataFrame) -> pd.DataFrame:
    """把『YYYY-盈利预测-收益』动态列压成 nearest_eps + year 标记。"""
    eps_cols = [c for c in df.columns if re.match(r"^\d{4}-盈利预测-收益$", str(c))]
    if not eps_cols:
        df["eps_forecast"] = np.nan
        df["eps_year"] = pd.NA
        return df
    # 取列名年份升序，优先用最小年份（最近报告期预测）
    years = sorted(int(c[:4]) for c in eps_cols)
    nearest = f"{years[0]}-盈利预测-收益"
    df["eps_forecast"] = pd.to_numeric(df.get(nearest), errors="coerce")
    df["eps_year"] = years[0]
    # 保留原始各年预测列（若已存在）
    for y in years:
        col = f"{y}-盈利预测-收益"
        if col in df.columns:
            df[f"eps_{y}"] = pd.to_numeric(df[col], errors="coerce")
    return df


def _clean(df: pd.DataFrame, code: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    rename = {
        "股票代码": "code",
        "股票简称": "name",
        "报告名称": "title",
        "东财评级": "rating",
        "机构": "institute",
        "近一月个股研报数": "reports_1m",
        "行业": "industry",
        "日期": "announce_date",
        "报告PDF链接": "pdf_url",
    }
    df = df.rename(columns=rename)
    df["code"] = str(code).zfill(6)
    if "announce_date" in df.columns:
        df["announce_date"] = pd.to_datetime(df["announce_date"], errors="coerce")
    if "reports_1m" in df.columns:
        df["reports_1m"] = pd.to_numeric(df["reports_1m"], errors="coerce")
    df = _normalize_eps_cols(df)
    keep = [
        c for c in (
            "code", "name", "title", "rating", "institute", "reports_1m",
            "industry", "announce_date", "eps_forecast", "eps_year", "pdf_url",
        ) if c in df.columns
    ]
    # 附加动态年预测
    keep += [c for c in df.columns if c.startswith("eps_") and c not in keep]
    out = df[keep].copy()
    return out.dropna(subset=["announce_date"])


def download_research_report(
    codes: list[str] | None = None,
    sample: int = 0,
    save_every: int = 50,
    sleep: float = SLEEP,
    force: bool = False,
) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if codes is None:
        uni = pd.read_parquet(UNIVERSE_DIR / "stock_list.parquet")
        codes = uni["code"].astype(str).str.zfill(6).tolist()
    if sample:
        codes = codes[:sample]

    existing = pd.read_parquet(OUT_PATH) if OUT_PATH.exists() else pd.DataFrame()
    done: set[str] = set()
    if not force and not existing.empty and "code" in existing.columns:
        # 已有记录的股票跳过（resume）；--force 全量重拉
        done = set(existing["code"].astype(str).str.zfill(6).unique())

    need = [c for c in codes if c not in done]
    if not need:
        logger.info("研报列表已覆盖目标股票，跳过")
        return existing

    logger.info(f"研报列表：需下载 {len(need)}/{len(codes)} 只")
    records = [existing] if not existing.empty else []
    failed: list[str] = []

    for i, code in enumerate(need):
        ok = False
        for attempt in range(MAX_RETRY):
            try:
                raw = ak.stock_research_report_em(symbol=code)
                df = _clean(raw, code)
                if not df.empty:
                    records.append(df)
                ok = True
                break
            except Exception as e:
                if attempt + 1 < MAX_RETRY:
                    time.sleep(sleep * (attempt + 2))
                else:
                    logger.warning(f"研报 {code} 失败: {e}")
                    failed.append(code)
        if (i + 1) % save_every == 0 and records:
            result = pd.concat(records, ignore_index=True)
            result = result.drop_duplicates(
                subset=["code", "announce_date", "institute", "title"],
                keep="last",
            )
            result.to_parquet(OUT_PATH)
            logger.info(f"进度 {i + 1}/{len(need)}，已落盘 {len(result)} 行")
        time.sleep(sleep)

    if not records:
        return existing
    result = pd.concat(records, ignore_index=True)
    result = result.drop_duplicates(
        subset=["code", "announce_date", "institute", "title"],
        keep="last",
    ).sort_values(["announce_date", "code"]).reset_index(drop=True)
    result.to_parquet(OUT_PATH)
    logger.info(f"研报列表保存: shape={result.shape} → {OUT_PATH}")
    if failed:
        logger.warning(f"失败 {len(failed)} 只: {failed[:10]}")
    return result


def main():
    p = argparse.ArgumentParser(description="下载东财个股研报列表")
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--sleep", type=float, default=SLEEP)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    download_research_report(
        sample=args.sample,
        save_every=args.save_every,
        sleep=args.sleep,
        force=args.force,
    )


if __name__ == "__main__":
    main()
