"""
data/download_northbound.py  —  北向资金个股持股数据下载

数据：沪深股通外资对A股各股票的持股量（股）和持股市值（元）
接口：
    ak.stock_hsgt_individual_em()  — 个股北向资金持股数据（东财）
    ak.stock_hsgt_hold_stock_em()  — 沪深港通持股明细（按日期）
存储：
    data/raw/northbound_holding.parquet   持股量宽表（index=日期, columns=股票）
    data/raw/northbound_value.parquet     持股市值宽表

策略：
    按日期批量拉取（接口支持按日查询），比逐股拉取效率高得多。
    历史数据从2016年开始，沪股通2014年，深股通2016年。

用法：
    python -m data.download_northbound
    python -m data.download_northbound --start 2018-01-01
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


def download_northbound_by_date(
    start: str,
    end: str,
    save_every: int = 60,  # 每60个交易日保存一次
) -> pd.DataFrame:
    """
    按日期逐日拉取北向持股明细，汇总成宽表。

    接口：ak.stock_hsgt_hold_stock_em(market="北向", date="20231201")
    返回列：股票代码, 股票名称, 持股数量, 持股市值, 持股数量占A股, 持股市值占总市值
    """
    hold_path = RAW_DIR / "northbound_holding.parquet"
    val_path  = RAW_DIR / "northbound_value.parquet"

    # 加载已有数据
    existing_hold = pd.read_parquet(hold_path) if hold_path.exists() else pd.DataFrame()
    existing_val  = pd.read_parquet(val_path)  if val_path.exists()  else pd.DataFrame()

    # 生成待下载日期列表（交易日历）
    try:
        cal = ak.tool_trade_date_hist_sina()
        all_dates = pd.to_datetime(cal["trade_date"])
    except Exception:
        all_dates = pd.bdate_range(start, end)

    target_dates = all_dates[
        (all_dates >= pd.Timestamp(start)) &
        (all_dates <= pd.Timestamp(end))
    ]

    # 跳过已有的日期
    if not existing_hold.empty:
        done_dates = set(existing_hold.index)
        target_dates = [d for d in target_dates if d not in done_dates]

    if not target_dates:
        logger.info("北向资金数据已是最新，跳过")
        return existing_hold

    logger.info(f"北向资金: 需下载 {len(target_dates)} 个交易日")
    hold_records = {}
    val_records  = {}
    failed = []

    for i, date in enumerate(target_dates):
        date_str = date.strftime("%Y%m%d")
        try:
            df = ak.stock_hsgt_hold_stock_em(market="北向", date=date_str)
            if df is None or df.empty:
                continue

            df.columns = [c.strip() for c in df.columns]
            code_col = [c for c in df.columns if "代码" in c or "股票代码" in c]
            hold_col = [c for c in df.columns if "持股数量" in c and "占" not in c]
            val_col  = [c for c in df.columns if "持股市值" in c and "占" not in c]

            if not code_col:
                logger.debug(f"{date_str} 列名: {list(df.columns)}")
                continue

            df["code"] = df[code_col[0]].astype(str).str.zfill(6)
            df = df.set_index("code")

            if hold_col:
                s = pd.to_numeric(df[hold_col[0]], errors="coerce")
                hold_records[date] = s
            if val_col:
                s = pd.to_numeric(df[val_col[0]], errors="coerce")
                val_records[date] = s

            time.sleep(0.3)

        except Exception as e:
            logger.warning(f"北向资金失败 {date_str}: {e}")
            failed.append(date_str)

        if (i + 1) % save_every == 0:
            logger.info(f"进度 {i+1}/{len(target_dates)}")
            _merge_and_save(hold_records, val_records, existing_hold, existing_val,
                            hold_path, val_path)

    if failed:
        logger.warning(f"失败 {len(failed)} 个日期: {failed[:5]}")

    return _merge_and_save(hold_records, val_records, existing_hold, existing_val,
                           hold_path, val_path)


def _merge_and_save(hold_new, val_new, existing_hold, existing_val, hold_path, val_path):
    def _merge(new_dict, existing):
        if not new_dict:
            return existing
        new_df = pd.DataFrame(new_dict).T.sort_index()  # (date, stock)
        new_df.index = pd.to_datetime(new_df.index)
        if existing.empty:
            result = new_df
        else:
            result = pd.concat([existing, new_df])
            result = result[~result.index.duplicated(keep="last")].sort_index()
        return result

    hold_df = _merge(hold_new, existing_hold)
    val_df  = _merge(val_new,  existing_val)
    hold_df.to_parquet(hold_path)
    val_df.to_parquet(val_path)
    logger.info(f"北向资金保存: holding={hold_df.shape}, value={val_df.shape}")
    return hold_df


def main(start: str, end: str):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    download_northbound_by_date(start, end)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end",   default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    args = parser.parse_args()
    main(args.start, args.end)
