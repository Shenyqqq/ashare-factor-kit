"""
data/download_dzjy_yybph.py  —  大宗交易营业部排行（买方席位质量）

接口：ak.stock_dzjy_yybph(symbol='近一月'|…)
存储：data/raw/dzjy_yybph.parquet（长表，含 asof_date）

用法：
    python -m data.download_dzjy_yybph
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

OUT_PATH = RAW_DIR / "dzjy_yybph.parquet"
DEFAULT_WINDOWS = ["近一月", "近三月", "近六月", "近一年"]
MAX_RETRY = 3
SLEEP = 0.5


def _clean(df: pd.DataFrame, window: str, asof: pd.Timestamp) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    rename = {
        "营业部名称": "branch",
        "上榜后1天-买入次数": "buy_cnt_1d",
        "上榜后1天-平均涨幅": "avg_ret_1d",
        "上榜后1天-上涨概率": "win_rate_1d",
        "上榜后5天-买入次数": "buy_cnt_5d",
        "上榜后5天-平均涨幅": "avg_ret_5d",
        "上榜后5天-上涨概率": "win_rate_5d",
        "上榜后10天-买入次数": "buy_cnt_10d",
        "上榜后10天-平均涨幅": "avg_ret_10d",
        "上榜后10天-上涨概率": "win_rate_10d",
        "上榜后20天-买入次数": "buy_cnt_20d",
        "上榜后20天-平均涨幅": "avg_ret_20d",
        "上榜后20天-上涨概率": "win_rate_20d",
    }
    df = df.rename(columns=rename)
    keep = [c for c in rename.values() if c in df.columns]
    out = df[keep].copy()
    out["window"] = window
    out["asof_date"] = asof
    for c in keep:
        if c != "branch":
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def download_dzjy_yybph(windows: list[str] | None = None) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    windows = windows or DEFAULT_WINDOWS
    asof = pd.Timestamp.today().normalize()
    parts: list[pd.DataFrame] = []
    for w in windows:
        last_err = None
        for attempt in range(MAX_RETRY):
            try:
                raw = ak.stock_dzjy_yybph(symbol=w)
                parts.append(_clean(raw, w, asof))
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(SLEEP * (attempt + 1))
        if last_err is not None:
            logger.warning(f"大宗营业部排行 {w} 失败: {last_err}")
        time.sleep(SLEEP)

    new = pd.concat([p for p in parts if not p.empty], ignore_index=True) if parts else pd.DataFrame()
    if new.empty:
        logger.warning("大宗营业部排行无数据")
        return pd.read_parquet(OUT_PATH) if OUT_PATH.exists() else pd.DataFrame()
    if OUT_PATH.exists():
        old = pd.read_parquet(OUT_PATH)
        df = pd.concat([old, new], ignore_index=True)
    else:
        df = new
    df = df.drop_duplicates(
        subset=["branch", "window", "asof_date"], keep="last"
    ).reset_index(drop=True)
    df.to_parquet(OUT_PATH)
    logger.info(f"大宗营业部排行保存: shape={df.shape} → {OUT_PATH}")
    return df


def main():
    p = argparse.ArgumentParser(description="下载大宗交易营业部排行")
    p.add_argument("--windows", default=",".join(DEFAULT_WINDOWS))
    args = p.parse_args()
    wins = [w.strip() for w in args.windows.split(",") if w.strip()]
    download_dzjy_yybph(wins)


if __name__ == "__main__":
    main()
