"""
data/download_institution.py  —  机构（基金）持仓季报数据下载

接口：ak.stock_report_fund_hold(symbol='基金持仓', date='20241231')
      返回全市场所有被基金持仓股票的汇总数据（持股数量、持股市值等）

存储：data/raw/institution_holding.parquet
      宽表：index=季报日期, columns=股票代码, 值=持股市值（元）

季报日期：03-31 / 06-30 / 09-30 / 12-31

用法：
    python -m data.download_institution
    python -m data.download_institution --start-year 2018
    python -m data.download_institution --start-year 2018 --sample 4
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

MAX_RETRY = 3
SLEEP = 1.0


def _quarter_dates(start_year: int, end_year: int) -> list[str]:
    dates = []
    for y in range(start_year, end_year + 1):
        for q in ["0331", "0630", "0930", "1231"]:
            dates.append(f"{y}{q}")
    return dates


def download_institution(
    start_year: int = 2018,
    end_year: int | None = None,
    sample: int = 0,
    sleep: float = SLEEP,
) -> pd.DataFrame:
    """回补基金持仓季报；默认自 2018 起，已存在季报期 resume 跳过。"""
    out_path = RAW_DIR / "institution_holding.parquet"
    if end_year is None:
        end_year = pd.Timestamp.today().year

    quarters = _quarter_dates(start_year, end_year)
    quarters = [
        q for q in quarters
        if pd.Timestamp(q[:4] + "-" + q[4:6] + "-" + q[6:]) <= pd.Timestamp.today()
    ]
    if sample:
        quarters = quarters[:sample]
        logger.info(f"调试：仅前 {sample} 个季报期")

    existing = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
    done: set[str] = set()
    if not existing.empty:
        done = {d.strftime("%Y%m%d") for d in existing.index}

    need = [q for q in quarters if q not in done]
    if not need:
        logger.info("机构持仓数据已是最新，跳过")
        return existing

    logger.info(f"机构持仓: 需下载 {len(need)} 个季报期（目标覆盖 {start_year}+）")
    records = {}
    failed = []

    for q in need:
        date_str = f"{q[:4]}-{q[4:6]}-{q[6:]}"
        ok = False
        for attempt in range(MAX_RETRY):
            try:
                df = ak.stock_report_fund_hold(symbol="基金持仓", date=q)
                if df is None or df.empty:
                    logger.warning(f"{q} 返回空数据")
                    ok = True
                    break

                code_col = df.columns[1]
                value_col = df.columns[5]

                s = df[[code_col, value_col]].copy()
                s.columns = ["code", "value"]
                s["code"] = s["code"].astype(str).str.zfill(6)
                s["value"] = pd.to_numeric(s["value"], errors="coerce")
                s = s.set_index("code")["value"]

                records[pd.Timestamp(date_str)] = s
                logger.info(f"{q}: {len(s)} 只股票有基金持仓")
                ok = True
                break
            except Exception as e:
                if attempt + 1 < MAX_RETRY:
                    time.sleep(sleep * (attempt + 1))
                else:
                    logger.warning(f"机构持仓失败 {q}: {e}")
                    failed.append(q)
        time.sleep(sleep)

        # 每季落盘，便于中断 resume
        if records:
            new_df = pd.DataFrame(records).T.sort_index()
            if not existing.empty:
                result = pd.concat([existing, new_df])
                result = result[~result.index.duplicated(keep="last")].sort_index()
            else:
                result = new_df
            result.to_parquet(out_path)
            existing = result
            records = {}

    if failed:
        logger.warning(f"失败季报期: {failed}")

    if out_path.exists():
        result = pd.read_parquet(out_path)
        logger.info(f"机构持仓保存: shape={result.shape}")
        return result
    logger.warning("未下载到任何机构持仓数据")
    return existing


def main(start_year: int = 2018, sample: int = 0):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    download_institution(start_year=start_year, sample=sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载基金持仓季报（默认回补 2018+）")
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--sample", type=int, default=0,
                        help="调试：仅下载前 N 个待补季报期")
    args = parser.parse_args()
    main(args.start_year, args.sample)
