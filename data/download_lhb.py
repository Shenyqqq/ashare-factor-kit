"""
data/download_lhb.py  —  龙虎榜下载

数据用途
    小盘股策略的「资金关注度/异常成交」因子：龙虎榜上榜频率、净买额、机构席位活跃度。
    高频上榜 + 净买额为正 → 短期资金关注度高的信号；上榜后 N 日收益可做事件研究。

接口
    ak.stock_lhb_detail_em(start_date, end_date)   东方财富-龙虎榜详情
    按日期区间拉取，全市场一次返回（覆盖该区间内所有上榜日/股票）。

频率
    日频（仅在有股票上榜的交易日才有数据）。

输出 schema（长表，data/raw/lhb_detail.parquet）
    code                str     6位代码
    name                str     名称
    lhb_date            datetime 上榜日
    reason              str     上榜原因
    interpretation      str     解读（含席位类型/成功率）
    close               float   收盘价
    pct_change          float   涨跌幅（%）
    net_buy             float   龙虎榜净买额
    buy_amount          float   龙虎榜买入额
    sell_amount         float   龙虎榜卖出额
    total_amount        float   龙虎榜成交额
    market_total_amount float   市场总成交额
    net_buy_pct         float   净买额占总成交比（%）
    total_pct           float   成交额占总成交比（%）
    turnover            float   换手率（%）
    float_market_value  float   流通市值
    return_1d           float   上榜后1日收益（%）
    return_2d           float   上榜后2日收益（%）
    return_5d           float   上榜后5日收益（%）
    return_10d          float   上榜后10日收益（%）

注：东方财富的 stock_lhb_detail_em 接口不直接返回「机构席位/营业部」明细列；
该信息需通过 stock_lhb_stock_detail_em(symbol, date) 按股票×日再查（请求量大），
本模块只做基础上榜表，下游如需席位明细可另起脚本。

下载策略
    按月分块调用（避免单次全量请求过大、便于增量续传）。
    6 年 ≈ 72 个月块 → ~72 次请求，每次 < 1s。

增量续传
    读已存在 parquet，对已有 lhb_date（月份）整块跳过。

用法
    python -m data.download_lhb --start 2018-01-01 --end 2026-07-03
    python -m data.download_lhb --start 2018-01-01 --end 2026-07-03 --sample 100
"""
from __future__ import annotations

import argparse
import time
from datetime import timedelta
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

OUT_PATH = RAW_DIR / "lhb_detail.parquet"

COLS_MAP = {
    "代码": "code",
    "名称": "name",
    "上榜日": "lhb_date",
    "上榜原因": "reason",
    "解读": "interpretation",
    "收盘价": "close",
    "涨跌幅": "pct_change",
    "龙虎榜净买额": "net_buy",
    "龙虎榜买入额": "buy_amount",
    "龙虎榜卖出额": "sell_amount",
    "龙虎榜成交额": "total_amount",
    "市场总成交额": "market_total_amount",
    "净买额占总成交比": "net_buy_pct",
    "成交额占总成交比": "total_pct",
    "换手率": "turnover",
    "流通市值": "float_market_value",
    "上榜后1日": "return_1d",
    "上榜后2日": "return_2d",
    "上榜后5日": "return_5d",
    "上榜后10日": "return_10d",
}

NUMERIC_COLS = [
    "close", "pct_change", "net_buy", "buy_amount", "sell_amount",
    "total_amount", "market_total_amount", "net_buy_pct", "total_pct",
    "turnover", "float_market_value",
    "return_1d", "return_2d", "return_5d", "return_10d",
]


def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def _clean_lhb(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=COLS_MAP)
    keep = [c for c in COLS_MAP.values() if c in df.columns]
    df = df[keep].copy()
    if "code" in df.columns:
        df["code"] = df["code"].astype(str).str.zfill(6)
    if "lhb_date" in df.columns:
        df["lhb_date"] = pd.to_datetime(df["lhb_date"], errors="coerce")
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    for col in ("reason", "interpretation"):
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df


def fetch_range(start_date: str, end_date: str) -> pd.DataFrame | None:
    for attempt in range(MAX_RETRY + 1):
        try:
            df = ak.stock_lhb_detail_em(start_date=start_date, end_date=end_date)
            time.sleep(0.3)
            return _clean_lhb(df)
        except Exception as e:
            if attempt < MAX_RETRY:
                wait = RETRY_DELAY * (attempt + 1)
                logger.debug(f"LHB {start_date}~{end_date} 失败，{wait}s 后重试: {e}")
                time.sleep(wait)
            else:
                logger.warning(f"LHB {start_date}~{end_date} 最终失败: {e}")
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
        if "lhb_date" in existing.columns:
            existing["lhb_date"] = pd.to_datetime(existing["lhb_date"], errors="coerce")
            existing["_month"] = existing["lhb_date"].dt.to_period("M").astype(str)
            done_months = set(existing["_month"].unique())
            existing = existing.drop(columns=["_month"])
        records.append(existing)
        logger.info(f"已存在 {len(done_months)} 个月，跳过")

    need = [(s, e, m) for s, e, m in chunks if m not in done_months]
    if not need:
        logger.info("全部已下载，跳过")
        return

    logger.info(f"龙虎榜下载：{len(need)} 个月块待下载")

    success = skip = failed = 0

    def _save():
        if not records:
            return
        result = pd.concat(records, ignore_index=True)
        result = result.drop_duplicates(
            subset=["code", "lhb_date", "reason"], keep="last"
        )
        result = result.sort_values(["lhb_date", "code"]).reset_index(drop=True)
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
            logger.info(f"  [{m}] 无数据（无股票上榜）")
        if (success + skip + failed) % 6 == 0:
            _save()

    _save()
    logger.info(f"完成：成功 {success}，无数据 {skip}，失败 {failed}")
    if OUT_PATH.exists():
        final = pd.read_parquet(OUT_PATH)
        logger.info(f"最终 {OUT_PATH.name}: shape={final.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载龙虎榜详情（按月分块）")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--sample", type=int, default=0,
                        help="调试：仅处理前 N 个月块")
    args = parser.parse_args()
    main(args.start, args.end, args.sample)
