"""
data/download_sector_fund_flow.py  —  行业/概念资金流（探索性）

优先东财历史：ak.stock_sector_fund_flow_hist(symbol=行业名)
兜底同花顺即时截面：ak.stock_fund_flow_industry / stock_fund_flow_concept

存储：
    data/raw/sector_fund_flow.parquet   行业日频长表
    data/raw/concept_fund_flow.parquet  概念日频/快照长表

用法：
    python -m data.download_sector_fund_flow
    python -m data.download_sector_fund_flow --sample 5
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

SECTOR_PATH = RAW_DIR / "sector_fund_flow.parquet"
CONCEPT_PATH = RAW_DIR / "concept_fund_flow.parquet"
MAX_RETRY = 2
SLEEP = 0.4


def _list_industries() -> list[str]:
    try:
        df = ak.stock_fund_flow_industry(symbol="即时")
        col = "行业" if "行业" in df.columns else df.columns[1]
        return df[col].astype(str).tolist()
    except Exception as e:
        logger.warning(f"获取行业列表失败: {e}")
        return []


def _list_concepts() -> list[str]:
    try:
        df = ak.stock_fund_flow_concept(symbol="即时")
        col = "行业" if "行业" in df.columns else (
            "概念" if "概念" in df.columns else df.columns[1]
        )
        return df[col].astype(str).tolist()
    except Exception as e:
        logger.warning(f"获取概念列表失败: {e}")
        return []


def _clean_hist(df: pd.DataFrame, name: str, kind: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    rename = {
        "日期": "date",
        "主力净流入-净额": "main_net",
        "主力净流入-净占比": "main_net_pct",
        "超大单净流入-净额": "super_net",
        "超大单净流入-净占比": "super_net_pct",
        "大单净流入-净额": "large_net",
        "大单净流入-净占比": "large_net_pct",
        "中单净流入-净额": "mid_net",
        "中单净流入-净占比": "mid_net_pct",
        "小单净流入-净额": "small_net",
        "小单净流入-净占比": "small_net_pct",
    }
    df = df.rename(columns=rename)
    keep = [c for c in rename.values() if c in df.columns]
    out = df[keep].copy()
    out["sector"] = name
    out["kind"] = kind
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for c in keep:
        if c != "date":
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(subset=["date"])


def _clean_spot(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """同花顺即时截面 → 伪日频（asof=today）。"""
    if df is None or df.empty:
        return pd.DataFrame()
    name_col = "行业" if "行业" in df.columns else (
        "概念" if "概念" in df.columns else df.columns[1]
    )
    net_col = "净额" if "净额" in df.columns else None
    out = pd.DataFrame({
        "date": pd.Timestamp.today().normalize(),
        "sector": df[name_col].astype(str),
        "kind": kind,
        "main_net": pd.to_numeric(df[net_col], errors="coerce") if net_col else pd.NA,
    })
    if "流入资金" in df.columns:
        out["inflow"] = pd.to_numeric(df["流入资金"], errors="coerce")
    if "流出资金" in df.columns:
        out["outflow"] = pd.to_numeric(df["流出资金"], errors="coerce")
    return out


def _merge_save(path: Path, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if new.empty:
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()
    if path.exists():
        old = pd.read_parquet(path)
        df = pd.concat([old, new], ignore_index=True)
    else:
        df = new
    df = df.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
    df.to_parquet(path)
    return df


def download_sector_fund_flow(sample: int = 0, use_hist: bool = True) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    names = _list_industries()
    if sample and names:
        names = names[:sample]

    parts: list[pd.DataFrame] = []
    hist_ok = 0
    if use_hist and names:
        # resume：已有 sector 跳过
        done: set[str] = set()
        if SECTOR_PATH.exists():
            old = pd.read_parquet(SECTOR_PATH)
            if "sector" in old.columns and "kind" in old.columns:
                done = set(old.loc[old["kind"] == "industry", "sector"].astype(str).unique())
        need = [n for n in names if n not in done]
        logger.info(f"行业资金流历史：需下载 {len(need)}/{len(names)}")
        for i, name in enumerate(need):
            ok = False
            for attempt in range(MAX_RETRY):
                try:
                    raw = ak.stock_sector_fund_flow_hist(symbol=name)
                    parts.append(_clean_hist(raw, name, "industry"))
                    hist_ok += 1
                    ok = True
                    break
                except Exception as e:
                    if attempt + 1 >= MAX_RETRY:
                        logger.debug(f"行业历史 {name} 失败: {e}")
                    time.sleep(SLEEP * (attempt + 1))
            if (i + 1) % 10 == 0:
                logger.info(f"行业进度 {i + 1}/{len(need)}（成功 {hist_ok}）")
            time.sleep(SLEEP)

    if hist_ok == 0:
        logger.warning("东财行业历史不可用，回退同花顺即时截面")
        try:
            spot = ak.stock_fund_flow_industry(symbol="即时")
            parts.append(_clean_spot(spot, "industry"))
        except Exception as e:
            logger.warning(f"同花顺行业截面失败: {e}")

    new = pd.concat([p for p in parts if not p.empty], ignore_index=True) if parts else pd.DataFrame()
    result = _merge_save(SECTOR_PATH, new, keys=["date", "sector", "kind"])
    logger.info(f"行业资金流保存: shape={result.shape} → {SECTOR_PATH}")
    return result


def download_concept_fund_flow(sample: int = 0) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    # 概念历史接口不稳定，默认只落即时截面（asof=today）便于探索
    try:
        spot = ak.stock_fund_flow_concept(symbol="即时")
        if sample:
            spot = spot.head(sample)
        new = _clean_spot(spot, "concept")
    except Exception as e:
        logger.warning(f"概念资金流失败: {e}")
        new = pd.DataFrame()
    result = _merge_save(CONCEPT_PATH, new, keys=["date", "sector", "kind"])
    logger.info(f"概念资金流保存: shape={result.shape} → {CONCEPT_PATH}")
    return result


def main():
    p = argparse.ArgumentParser(description="下载行业/概念资金流")
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--no-hist", action="store_true", help="跳过东财历史，仅即时截面")
    p.add_argument("--skip-concept", action="store_true")
    args = p.parse_args()
    download_sector_fund_flow(sample=args.sample, use_hist=not args.no_hist)
    if not args.skip_concept:
        download_concept_fund_flow(sample=args.sample)


if __name__ == "__main__":
    main()
