"""
data/download_institution.py  —  机构（基金）持仓季报数据下载

接口：ak.stock_report_fund_hold(symbol='基金持仓', date='20241231')
      返回全市场所有被基金持仓股票的汇总数据（持股数量、持股市值等）

存储：data/raw/institution_holding.parquet
      宽表：index=季报日期, columns=股票代码, 值=持股市值（元）

季报日期：03-31 / 06-30 / 09-30 / 12-31

用法：
    python -m data.download_institution
    python -m data.download_institution --start-year 2020
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


def _quarter_dates(start_year: int, end_year: int) -> list[str]:
    dates = []
    for y in range(start_year, end_year + 1):
        for q in ["0331", "0630", "0930", "1231"]:
            dates.append(f"{y}{q}")
    return dates


def download_institution(start_year: int = 2018, end_year: int = None) -> pd.DataFrame:
    out_path = RAW_DIR / "institution_holding.parquet"
    if end_year is None:
        end_year = pd.Timestamp.today().year

    quarters = _quarter_dates(start_year, end_year)
    # 只下载已过的季报期
    quarters = [q for q in quarters
                if pd.Timestamp(q[:4] + "-" + q[4:6] + "-" + q[6:]) <= pd.Timestamp.today()]

    existing = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
    done = set()
    if not existing.empty:
        done = {d.strftime("%Y%m%d") for d in existing.index}

    need = [q for q in quarters if q not in done]
    if not need:
        logger.info("机构持仓数据已是最新，跳过")
        return existing

    logger.info(f"机构持仓: 需下载 {len(need)} 个季报期")
    records = {}
    failed = []

    for q in need:
        # date 格式：yyyymmdd → akshare 需要 yyyy-mm-dd
        date_str = f"{q[:4]}-{q[4:6]}-{q[6:]}"
        try:
            df = ak.stock_report_fund_hold(symbol="基金持仓", date=q)
            if df is None or df.empty:
                logger.warning(f"{q} 返回空数据")
                continue

            # 列：序, 股票代码, 股票名称, 持有基金家数, 持股总数, 持股市值, 持股变化, ...
            code_col  = df.columns[1]   # 股票代码
            value_col = df.columns[5]   # 持股市值

            s = df[[code_col, value_col]].copy()
            s.columns = ["code", "value"]
            s["code"] = s["code"].astype(str).str.zfill(6)
            s["value"] = pd.to_numeric(s["value"], errors="coerce")
            s = s.set_index("code")["value"]

            records[pd.Timestamp(date_str)] = s
            logger.info(f"{q}: {len(s)} 只股票有基金持仓")
            time.sleep(1.0)

        except Exception as e:
            logger.warning(f"机构持仓失败 {q}: {e}")
            failed.append(q)

    if failed:
        logger.warning(f"失败季报期: {failed}")

    if not records:
        logger.warning("未下载到任何机构持仓数据")
        return existing

    new_df = pd.DataFrame(records).T.sort_index()
    if not existing.empty:
        result = pd.concat([existing, new_df])
        result = result[~result.index.duplicated(keep="last")].sort_index()
    else:
        result = new_df

    result.to_parquet(out_path)
    logger.info(f"机构持仓保存: shape={result.shape}")
    return result


def main(start_year: int = 2018):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    download_institution(start_year=start_year)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2018)
    args = parser.parse_args()
    main(args.start_year)
