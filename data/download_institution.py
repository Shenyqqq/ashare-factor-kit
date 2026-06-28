"""
data/download_institution.py  —  机构持仓季报数据下载

数据：基金重仓股季度披露数据（每季报披露前10大持股）
接口：
    ak.fund_portfolio_hold_em()  — 东财基金持仓明细（按季报期）
    ak.stock_institute_hold_detail_em() — 个股机构持仓
存储：
    data/raw/institution_holding.parquet  (宽表: index=季报日期, columns=股票)
    值 = 机构持股总量（股）或持股比例（%）

策略：
    逐季报期拉取所有基金的持仓，汇总每只股票的机构总持仓。
    季报披露时间：一季报4月底、半年报8月底、三季报10月底、年报4月初。

用法：
    python -m data.download_institution
    python -m data.download_institution --start-year 2018
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


def _get_quarters(start_year: int, end_year: int) -> list:
    """生成季报期列表，格式如 '2023-09-30'"""
    quarters = []
    for year in range(start_year, end_year + 1):
        quarters.extend([
            f"{year}-03-31",
            f"{year}-06-30",
            f"{year}-09-30",
            f"{year}-12-31",
        ])
    return quarters


def download_institution_holding(
    start_year: int = 2018,
    end_year: int = None,
) -> pd.DataFrame:
    """
    按季报期批量下载机构持仓，汇总成宽表。

    接口：ak.stock_institute_hold_detail_em(quarter="20231231", symbol=None)
    此接口返回某季报期所有被机构持仓的股票。
    """
    out_path = RAW_DIR / "institution_holding.parquet"
    if end_year is None:
        end_year = pd.Timestamp.today().year

    quarters = _get_quarters(start_year, end_year)

    # 加载已有数据
    existing = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
    done_quarters = set()
    if not existing.empty:
        done_quarters = {q.strftime("%Y-%m-%d") for q in existing.index}

    need = [q for q in quarters if q not in done_quarters
            and pd.Timestamp(q) <= pd.Timestamp.today()]

    if not need:
        logger.info("机构持仓数据已是最新，跳过")
        return existing

    logger.info(f"机构持仓: 需下载 {len(need)} 个季报期")
    records = {}
    failed = []

    for q in need:
        q_str = q.replace("-", "")
        try:
            # 东财个股机构持仓汇总接口
            df = ak.stock_institute_hold_detail_em(quarter=q_str)
            if df is None or df.empty:
                logger.debug(f"{q} 无数据")
                continue

            df.columns = [c.strip() for c in df.columns]
            code_col  = [c for c in df.columns if "代码" in c]
            share_col = [c for c in df.columns if "持股数量" in c and "变化" not in c]
            ratio_col = [c for c in df.columns if "持股比例" in c and "变化" not in c]

            if not code_col:
                logger.debug(f"{q} 列名: {list(df.columns)}")
                failed.append(q)
                continue

            df["code"] = df[code_col[0]].astype(str).str.zfill(6)

            # 优先用持股数量，无则用持股比例
            value_col = share_col[0] if share_col else (ratio_col[0] if ratio_col else None)
            if value_col is None:
                failed.append(q)
                continue

            agg = df.groupby("code")[value_col].sum()
            agg.name = pd.Timestamp(q)
            records[pd.Timestamp(q)] = pd.to_numeric(agg, errors="coerce")

            logger.info(f"{q}: {len(agg)} 只股票有机构持仓")
            time.sleep(1.0)  # 季报接口，频率要低一些

        except Exception as e:
            logger.warning(f"机构持仓失败 {q}: {e}")
            failed.append(q)

    if failed:
        logger.warning(f"失败 {len(failed)} 个季报期: {failed}")

    if not records:
        logger.warning("未下载到任何机构持仓数据")
        return existing

    new_df = pd.DataFrame(records).T.sort_index()
    new_df.index = pd.to_datetime(new_df.index)

    if not existing.empty:
        result = pd.concat([existing, new_df])
        result = result[~result.index.duplicated(keep="last")].sort_index()
    else:
        result = new_df

    result.to_parquet(out_path)
    logger.info(f"机构持仓保存: {out_path.name}, shape={result.shape}")
    return result


def main(start_year: int = 2018):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    download_institution_holding(start_year=start_year)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2018)
    args = parser.parse_args()
    main(args.start_year)
