"""
data/download_holder_trade.py  —  大股东/董监高增减持 + 大宗交易

数据用途
    小盘股策略的「内部人行为」因子：
      - 高管/股东增减持：大股东净增持为正信号（内部人私有信息），减持为负信号。
      - 大宗交易：折价率（折溢率为负 → 折价成交 → 大宗出货压力），成交量/成交额占流通市值比。

接口
    ak.stock_ggcg_em(symbol='全部')            东方财富-高管增减持（全市场全历史，一次返回）
        注：原候选 stock_ggcg_em(start_date, end_date) 不存在；当前签名是 symbol-only，
            全部历史在单次调用里返回（内部 ~290 页分页，~3-5 分钟）。
    ak.stock_dzjy_mrmx(symbol='A股', start_date, end_date)   东方财富-大宗交易每日明细
        按日期区间拉取，全市场一次返回。

频率
    高管增减持：事件驱动（不定期，全历史一次性拉）。
    大宗交易：日频（仅在大宗成交日有数据）。

输出 schema（长表）
    data/raw/holder_trade.parquet  （高管增减持）
        code                    str     6位代码
        name                    str     名称
        holder_name             str     股东名称（变动人）
        change_direction        str     增减（增持/减持）
        change_date             datetime 变动截止日（≈变动日；变动开始日另存）
        change_start_date       datetime 变动开始日
        announce_date           datetime 公告日
        change_shares           float   变动数量（万股）
        change_pct_of_total     float   变动占总股本比例（%）
        change_pct_of_float     float   变动占流通股比例（%）
        post_change_total_shares    float   变动后持股总数（万股）
        post_change_pct_of_total    float   变动后占总股本比例（%）
        post_change_float_shares    float   变动后持流通股数（万股）
        post_change_pct_of_float    float   变动后占流通股比例（%）

    data/raw/block_trade.parquet  （大宗交易）
        code                    str     6位代码
        name                    str     证券简称
        trade_date              datetime 交易日期
        close                   float   收盘价
        deal_price              float   成交价
        discount_rate           float   折溢率（%，负=折价）
        volume                  float   成交量（股）
        amount                  float   成交额（元）
        amount_to_float_ratio   float   成交额/流通市值（%）
        buyer_branch            str     买方营业部
        seller_branch           str     卖方营业部

下载策略
    高管增减持：单次 API 调用获取全历史 → 本地按 start/end 过滤 → 与已存在数据按
        (code, change_date, holder_name, change_direction) 去重合并。
        重新运行可覆盖最新增减持（API 返回全历史，每次都拿最新）。
    大宗交易：按月分块调用，6 年 ≈ 72 次请求。

增量续传
    holder_trade.parquet：按 (code, change_date, holder_name, change_direction) 去重。
    block_trade.parquet：按月份跳过已有数据。

用法
    python -m data.download_holder_trade --start 2018-01-01 --end 2026-07-03
    python -m data.download_holder_trade --start 2018-01-01 --end 2026-07-03 --no-block-trade
    python -m data.download_holder_trade --start 2018-01-01 --end 2026-07-03 --sample 12
"""
from __future__ import annotations

import argparse
import time
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

HOLDER_OUT_PATH = RAW_DIR / "holder_trade.parquet"
BLOCK_OUT_PATH = RAW_DIR / "block_trade.parquet"

GGCG_COLS_MAP = {
    "代码": "code",
    "名称": "name",
    "股东名称": "holder_name",
    "持股变动信息-增减": "change_direction",
    "持股变动信息-变动数量": "change_shares",
    "持股变动信息-占总股本比例": "change_pct_of_total",
    "持股变动信息-占流通股比例": "change_pct_of_float",
    "变动后持股情况-持股总数": "post_change_total_shares",
    "变动后持股情况-占总股本比例": "post_change_pct_of_total",
    "变动后持股情况-持流通股数": "post_change_float_shares",
    "变动后持股情况-占流通股比例": "post_change_pct_of_float",
    "变动开始日": "change_start_date",
    "变动截止日": "change_date",
    "公告日": "announce_date",
}

GGCG_NUMERIC_COLS = [
    "change_shares", "change_pct_of_total", "change_pct_of_float",
    "post_change_total_shares", "post_change_pct_of_total",
    "post_change_float_shares", "post_change_pct_of_float",
]

DZJY_COLS_MAP = {
    "证券代码": "code",
    "证券简称": "name",
    "交易日期": "trade_date",
    "收盘价": "close",
    "成交价": "deal_price",
    "折溢率": "discount_rate",
    "成交量": "volume",
    "成交额": "amount",
    "成交额/流通市值": "amount_to_float_ratio",
    "买方营业部": "buyer_branch",
    "卖方营业部": "seller_branch",
}

DZJY_NUMERIC_COLS = [
    "close", "deal_price", "discount_rate", "volume", "amount",
    "amount_to_float_ratio",
]


def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def _clean_ggcg(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=GGCG_COLS_MAP)
    keep = [c for c in GGCG_COLS_MAP.values() if c in df.columns]
    df = df[keep].copy()
    if "code" in df.columns:
        df["code"] = df["code"].astype(str).str.zfill(6)
    for col in ("change_start_date", "change_date", "announce_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in GGCG_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    for col in ("name", "holder_name", "change_direction"):
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df


def _clean_dzjy(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=DZJY_COLS_MAP)
    keep = [c for c in DZJY_COLS_MAP.values() if c in df.columns]
    df = df[keep].copy()
    if "code" in df.columns:
        df["code"] = df["code"].astype(str).str.zfill(6)
    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    for col in DZJY_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    for col in ("name", "buyer_branch", "seller_branch"):
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df


def fetch_ggcg(symbol: str = "全部") -> pd.DataFrame | None:
    """高管增减持：单次调用返回全历史，~3-5 分钟（内部 ~290 页分页）。"""
    for attempt in range(MAX_RETRY + 1):
        try:
            logger.info(f"拉取高管增减持 symbol={symbol}（请耐心等待 ~3-5 分钟）...")
            df = ak.stock_ggcg_em(symbol=symbol)
            return _clean_ggcg(df)
        except Exception as e:
            if attempt < MAX_RETRY:
                wait = RETRY_DELAY * (attempt + 1)
                logger.warning(f"高管增减持失败，{wait}s 后重试: {e}")
                time.sleep(wait)
            else:
                logger.error(f"高管增减持最终失败: {e}")
                return None


def fetch_dzjy_range(start_date: str, end_date: str) -> pd.DataFrame | None:
    for attempt in range(MAX_RETRY + 1):
        try:
            df = ak.stock_dzjy_mrmx(
                symbol="A股", start_date=start_date, end_date=end_date)
            time.sleep(0.3)
            return _clean_dzjy(df)
        except Exception as e:
            if attempt < MAX_RETRY:
                wait = RETRY_DELAY * (attempt + 1)
                logger.debug(f"大宗交易 {start_date}~{end_date} 失败，{wait}s 重试: {e}")
                time.sleep(wait)
            else:
                logger.warning(f"大宗交易 {start_date}~{end_date} 最终失败: {e}")
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


def filter_by_date(df: pd.DataFrame, date_col: str, start: str, end: str) -> pd.DataFrame:
    if df.empty or date_col not in df.columns:
        return df
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    return df[(df[date_col] >= s) & (df[date_col] <= e)].copy()


def download_holder_trade(start: str, end: str):
    """高管增减持：单次全历史调用 + 本地过滤 + 去重合并。"""
    df = fetch_ggcg("全部")
    if df is None or df.empty:
        logger.warning("高管增减持无数据")
        return
    df = filter_by_date(df, "change_date", start, end)
    logger.info(f"高管增减持过滤后 {df.shape[0]} 条")

    existing = pd.read_parquet(HOLDER_OUT_PATH) if HOLDER_OUT_PATH.exists() else None
    if existing is not None and not existing.empty:
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df
    combined = combined.drop_duplicates(
        subset=["code", "change_date", "holder_name", "change_direction"],
        keep="last",
    )
    combined = combined.sort_values(
        ["change_date", "code", "holder_name"]
    ).reset_index(drop=True)
    combined.to_parquet(HOLDER_OUT_PATH)
    logger.info(f"高管增减持保存: {HOLDER_OUT_PATH.name} shape={combined.shape}")


def download_block_trade(start: str, end: str, sample: int = 0):
    chunks = list(month_chunks(start, end))
    if sample:
        chunks = chunks[:sample]
        logger.info(f"调试模式：仅处理前 {sample} 个月块")

    existing = pd.read_parquet(BLOCK_OUT_PATH) if BLOCK_OUT_PATH.exists() else None
    done_months: set[str] = set()
    records: list[pd.DataFrame] = []
    if existing is not None and not existing.empty:
        existing["code"] = existing["code"].astype(str).str.zfill(6)
        if "trade_date" in existing.columns:
            existing["trade_date"] = pd.to_datetime(
                existing["trade_date"], errors="coerce")
            existing["_month"] = existing["trade_date"].dt.to_period("M").astype(str)
            done_months = set(existing["_month"].unique())
            existing = existing.drop(columns=["_month"])
        records.append(existing)
        logger.info(f"大宗交易已存在 {len(done_months)} 个月，跳过")

    need = [(s, e, m) for s, e, m in chunks if m not in done_months]
    if not need:
        logger.info("大宗交易全部已下载，跳过")
        return

    logger.info(f"大宗交易下载：{len(need)} 个月块待下载")
    success = skip = failed = 0

    def _save():
        if not records:
            return
        result = pd.concat(records, ignore_index=True)
        result = result.drop_duplicates(
            subset=["code", "trade_date", "buyer_branch", "seller_branch", "deal_price"],
            keep="last",
        )
        result = result.sort_values(["trade_date", "code"]).reset_index(drop=True)
        result.to_parquet(BLOCK_OUT_PATH)

    for s, e, m in need:
        df = fetch_dzjy_range(s, e)
        if df is not None and not df.empty:
            records.append(df)
            success += 1
            logger.info(f"  [{m}] {df.shape[0]} 条")
        elif df is None:
            failed += 1
        else:
            skip += 1
            logger.info(f"  [{m}] 无大宗交易数据")
        if (success + skip + failed) % 6 == 0:
            _save()

    _save()
    logger.info(f"大宗交易完成：成功 {success}，无数据 {skip}，失败 {failed}")
    if BLOCK_OUT_PATH.exists():
        final = pd.read_parquet(BLOCK_OUT_PATH)
        logger.info(f"最终 {BLOCK_OUT_PATH.name}: shape={final.shape}")


def main(start: str, end: str, sample: int = 0, no_block_trade: bool = False):
    ensure_dirs()
    download_holder_trade(start, end)
    if not no_block_trade:
        download_block_trade(start, end, sample=sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="下载大股东/董监高增减持 + 大宗交易")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--sample", type=int, default=0,
                        help="调试：大宗交易仅处理前 N 个月块")
    parser.add_argument("--no-block-trade", action="store_true",
                        help="跳过大宗交易，仅下载高管增减持")
    args = parser.parse_args()
    main(args.start, args.end, args.sample, args.no_block_trade)
