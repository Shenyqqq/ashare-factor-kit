"""
data/download_moneyflow.py  —  东财大单净流入数据下载

数据：个股每日大单（≥50万）和超大单（≥100万）净流入额
接口：ak.stock_individual_fund_flow()  东财个股资金流向
存储：
    data/raw/moneyflow_large.parquet      大单净流入（元）
    data/raw/moneyflow_superlarge.parquet 超大单净流入（元）

用法：
    python -m data.download_moneyflow
    python -m data.download_moneyflow --sample 200
"""
import argparse
import time
from pathlib import Path
import sys

import akshare as ak
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import RAW_DIR, UNIVERSE_DIR


def _load_existing(path: Path) -> dict:
    if path.exists():
        df = pd.read_parquet(path)
        return {c: df[c].dropna() for c in df.columns}
    return {}


def download_moneyflow(
    codes: list,
    market: str = "sh",
    save_every: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    下载大单和超大单净流入数据。
    东财接口只支持按市场查询（sh/sz），按代码逐个拉取。

    注意：东财免费接口有时只返回近期数据（约60-90个交易日），
    历史数据需要逐日拉取或用付费接口。此处按股票拉取全量。
    """
    large_path = RAW_DIR / "moneyflow_large.parquet"
    super_path = RAW_DIR / "moneyflow_superlarge.parquet"

    large_data = _load_existing(large_path)
    super_data = _load_existing(super_path)

    # 判断哪些股票需要更新（以近30日为准）
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=30)
    need = []
    for code in codes:
        s = large_data.get(code)
        if s is None or len(s) == 0 or s.index[-1] < cutoff:
            need.append(code)

    if not need:
        logger.info("大单净流入数据已是最新，跳过")
        return pd.DataFrame(large_data), pd.DataFrame(super_data)

    logger.info(f"大单净流入: 需下载 {len(need)}/{len(codes)} 只")
    failed = []

    for i, code in enumerate(need):
        # 判断市场
        mkt = "sh" if code.startswith(("6", "9")) else "sz"
        try:
            df = ak.stock_individual_fund_flow(stock=code, market=mkt)
            if df is None or df.empty:
                failed.append(code)
                continue

            df.columns = [c.strip() for c in df.columns]
            # 列名模式：'日期', '主力净流入-净额', '超大单净流入-净额', '大单净流入-净额', ...
            date_col = [c for c in df.columns if "日期" in c]
            large_col = [c for c in df.columns if "大单净流入" in c and "超大单" not in c and "净额" in c]
            super_col = [c for c in df.columns if "超大单净流入" in c and "净额" in c]

            if not date_col:
                logger.debug(f"{code} 列名: {list(df.columns)}")
                failed.append(code)
                continue

            df["date"] = pd.to_datetime(df[date_col[0]])
            df = df.set_index("date").sort_index()

            if large_col:
                s = pd.to_numeric(df[large_col[0]], errors="coerce")
                existing = large_data.get(code)
                if existing is not None:
                    combined = pd.concat([existing, s])
                    large_data[code] = combined[~combined.index.duplicated(keep="last")].sort_index()
                else:
                    large_data[code] = s

            if super_col:
                s = pd.to_numeric(df[super_col[0]], errors="coerce")
                existing = super_data.get(code)
                if existing is not None:
                    combined = pd.concat([existing, s])
                    super_data[code] = combined[~combined.index.duplicated(keep="last")].sort_index()
                else:
                    super_data[code] = s

            time.sleep(0.2)

        except Exception as e:
            logger.warning(f"大单净流入失败 {code}: {e}")
            failed.append(code)

        if (i + 1) % save_every == 0:
            logger.info(f"进度 {i+1}/{len(need)}")
            _save_both(large_data, super_data, large_path, super_path)

    if failed:
        logger.warning(f"失败 {len(failed)} 只: {failed[:10]}")

    return _save_both(large_data, super_data, large_path, super_path)


def _save_both(large_data, super_data, large_path, super_path):
    df_large = pd.DataFrame(large_data).sort_index()
    df_super = pd.DataFrame(super_data).sort_index()
    df_large.to_parquet(large_path)
    df_super.to_parquet(super_path)
    logger.info(f"大单净流入保存: large={df_large.shape}, super={df_super.shape}")
    return df_large, df_super


def main(sample: int = 0):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    universe = pd.read_parquet(UNIVERSE_DIR / "stock_list.parquet")
    codes = universe["code"].tolist()
    if sample:
        codes = codes[:sample]
    download_moneyflow(codes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0)
    args = parser.parse_args()
    main(args.sample)
