"""
data/download_margin.py  —  融资余额数据下载

接口：
    ak.stock_margin_detail_sse(date)   沪市个股融资明细（按日期）
    ak.stock_margin_detail_szse(date)  深市个股融资明细（按日期）
存储：
    data/raw/margin_balance.parquet   融资余额宽表（index=日期, columns=股票代码）

策略：按交易日逐日拉取沪深两市，合并后存宽表，支持断点续传。

用法：
    python -m data.download_margin
    python -m data.download_margin --start 2020-01-01
"""
import argparse
import time
from pathlib import Path
import sys

import akshare as ak
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import RAW_DIR


def _load_existing(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def download_margin(start: str, end: str, save_every: int = 60) -> pd.DataFrame:
    out_path = RAW_DIR / "margin_balance.parquet"
    existing = _load_existing(out_path)

    # 生成目标交易日列表
    try:
        cal = ak.tool_trade_date_hist_sina()
        all_dates = pd.to_datetime(cal["trade_date"])
    except Exception:
        all_dates = pd.bdate_range(start, end)

    target_dates = all_dates[
        (all_dates >= pd.Timestamp(start)) &
        (all_dates <= pd.Timestamp(end))
    ]

    done = set(existing.index) if not existing.empty else set()
    need = [d for d in target_dates if d not in done]
    if not need:
        logger.info("融资余额数据已是最新，跳过")
        return existing

    logger.info(f"融资余额: 需下载 {len(need)} 个交易日")
    records = {}
    failed = []

    for i, date in enumerate(need):
        date_str = date.strftime("%Y%m%d")
        try:
            sh = ak.stock_margin_detail_sse(date=date_str)
            sz = ak.stock_margin_detail_szse(date=date_str)

            # 沪市：列顺序固定，取代码和融资余额
            # cols: 交易日期, 证券代码, 证券名称, 融资余额, 融资买入额, 融资偿还额, 融券余量, 融券余量金额, 融券偿还量
            sh_code = sh.columns[1]
            sh_bal  = sh.columns[3]
            sh_s = sh[[sh_code, sh_bal]].copy()
            sh_s.columns = ["code", "balance"]
            sh_s["code"] = sh_s["code"].astype(str).str.zfill(6)

            # 深市：列顺序固定，取代码和融资余额
            # cols: 证券代码, 证券名称, 融资余额, 融资买入额, 融券余量金额, 融券余量, 融券卖出额, 融资融券余额
            sz_code = sz.columns[0]
            sz_bal  = sz.columns[2]
            sz_s = sz[[sz_code, sz_bal]].copy()
            sz_s.columns = ["code", "balance"]
            sz_s["code"] = sz_s["code"].astype(str).str.zfill(6)

            combined = pd.concat([sh_s, sz_s], ignore_index=True)
            combined["balance"] = pd.to_numeric(combined["balance"], errors="coerce")
            records[date] = combined.set_index("code")["balance"]

            time.sleep(0.5)

        except Exception as e:
            logger.warning(f"融资余额失败 {date_str}: {e}")
            failed.append(date_str)

        if (i + 1) % save_every == 0:
            logger.info(f"进度 {i+1}/{len(need)}")
            _merge_and_save(records, existing, out_path)

    if failed:
        logger.warning(f"失败 {len(failed)} 个日期: {failed[:5]}")

    return _merge_and_save(records, existing, out_path)


def _merge_and_save(new_records: dict, existing: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not new_records:
        return existing
    new_df = pd.DataFrame(new_records).T.sort_index()
    new_df.index = pd.to_datetime(new_df.index)
    if not existing.empty:
        result = pd.concat([existing, new_df])
        result = result[~result.index.duplicated(keep="last")].sort_index()
    else:
        result = new_df
    result.to_parquet(path)
    logger.info(f"融资余额保存: shape={result.shape}")
    return result


def main(start: str, end: str):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    download_margin(start, end)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end",   default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    args = parser.parse_args()
    main(args.start, args.end)
