"""
data/download_lhb_seats.py  —  龙虎榜营业部排行 + 机构席位统计（快照）

接口：
    ak.stock_lhb_yybph_em(symbol='近一月'|…)   营业部胜率榜
    ak.stock_lhb_jgstatistic_em(symbol=…)      个股机构席位统计

存储：
    data/raw/lhb_yybph.parquet          营业部排行长表（含 asof_date）
    data/raw/lhb_jgstatistic.parquet    个股机构席位长表（含 asof_date）

注：接口为滚动窗口快照，非完整历史流水。落盘带 asof_date 便于日后增量拼接；
因子侧主路径仍用 lhb_detail.interpretation 的「N家机构买入/卖出」（严格 PIT）。

用法：
    python -m data.download_lhb_seats
    python -m data.download_lhb_seats --windows 近一月,近三月
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

YYBPH_PATH = RAW_DIR / "lhb_yybph.parquet"
JGSTAT_PATH = RAW_DIR / "lhb_jgstatistic.parquet"
DEFAULT_WINDOWS = ["近一月", "近三月", "近六月", "近一年"]
MAX_RETRY = 3
SLEEP = 0.5


def _fetch(fn, **kwargs) -> pd.DataFrame:
    last_err = None
    for attempt in range(MAX_RETRY):
        try:
            df = fn(**kwargs)
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            last_err = e
            time.sleep(SLEEP * (attempt + 1))
    logger.warning(f"{fn.__name__} 失败: {last_err}")
    return pd.DataFrame()


def _clean_yybph(df: pd.DataFrame, window: str, asof: pd.Timestamp) -> pd.DataFrame:
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


def _clean_jgstat(df: pd.DataFrame, window: str, asof: pd.Timestamp) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    rename = {
        "代码": "code",
        "名称": "name",
        "收盘价": "close",
        "涨跌幅": "pct_change",
        "龙虎榜成交金额": "lhb_amount",
        "上榜次数": "list_count",
        "机构买入额": "inst_buy",
        "机构买入次数": "inst_buy_cnt",
        "机构卖出额": "inst_sell",
        "机构卖出次数": "inst_sell_cnt",
        "机构净买额": "inst_net_buy",
    }
    df = df.rename(columns=rename)
    keep = [c for c in rename.values() if c in df.columns]
    out = df[keep].copy()
    out["code"] = out["code"].astype(str).str.zfill(6)
    out["window"] = window
    out["asof_date"] = asof
    for c in keep:
        if c not in ("code", "name"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _append_dedup(path: Path, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
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


def download_lhb_seats(windows: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    windows = windows or DEFAULT_WINDOWS
    asof = pd.Timestamp.today().normalize()
    yyb_parts: list[pd.DataFrame] = []
    jg_parts: list[pd.DataFrame] = []

    for w in windows:
        raw = _fetch(ak.stock_lhb_yybph_em, symbol=w)
        yyb_parts.append(_clean_yybph(raw, w, asof))
        time.sleep(SLEEP)
        raw = _fetch(ak.stock_lhb_jgstatistic_em, symbol=w)
        jg_parts.append(_clean_jgstat(raw, w, asof))
        time.sleep(SLEEP)

    yyb = _append_dedup(
        YYBPH_PATH,
        pd.concat([p for p in yyb_parts if not p.empty], ignore_index=True)
        if any(not p.empty for p in yyb_parts) else pd.DataFrame(),
        keys=["branch", "window", "asof_date"],
    )
    jg = _append_dedup(
        JGSTAT_PATH,
        pd.concat([p for p in jg_parts if not p.empty], ignore_index=True)
        if any(not p.empty for p in jg_parts) else pd.DataFrame(),
        keys=["code", "window", "asof_date"],
    )
    logger.info(f"龙虎榜席位: yybph={yyb.shape}, jgstat={jg.shape}")
    return yyb, jg


def main():
    p = argparse.ArgumentParser(description="下载龙虎榜营业部/机构席位统计快照")
    p.add_argument("--windows", default=",".join(DEFAULT_WINDOWS))
    args = p.parse_args()
    wins = [w.strip() for w in args.windows.split(",") if w.strip()]
    download_lhb_seats(wins)


if __name__ == "__main__":
    main()
