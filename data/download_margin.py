"""
data/download_margin.py  —  融资余额数据下载

数据：沪深两市各股票的融资余额（元）
接口：ak.stock_margin_detail_em()  东财个股融资融券数据
存储：data/raw/margin_balance.parquet  (宽表: index=日期, columns=股票)

用法：
    python -m data.download_margin
    python -m data.download_margin --start 2020-01-01 --sample 200
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


def download_margin(
    codes: list,
    start: str,
    end: str,
    save_every: int = 100,
) -> pd.DataFrame:
    """
    逐股下载融资余额，保存宽表parquet。
    断点续传：已有且数据较新的股票直接跳过。
    """
    out_path = RAW_DIR / "margin_balance.parquet"
    data = _load_existing(out_path)

    last_trade = pd.Timestamp(end)
    need = []
    for code in codes:
        s = data.get(code)
        if s is None or len(s) == 0 or s.index[-1] < last_trade - pd.Timedelta(days=10):
            need.append(code)

    if not need:
        logger.info("融资余额数据已是最新，跳过")
        return pd.DataFrame(data).sort_index()

    logger.info(f"融资余额: 需下载 {len(need)}/{len(codes)} 只")
    failed = []

    for i, code in enumerate(need):
        try:
            df = ak.stock_margin_detail_em(symbol=code)
            if df is None or df.empty:
                failed.append(code)
                continue

            # 东财接口列名：'日期', '融资余额', '融资买入额', ...
            date_col = [c for c in df.columns if "日期" in c or "date" in c.lower()]
            bal_col  = [c for c in df.columns if "融资余额" in c]

            if not date_col or not bal_col:
                logger.debug(f"{code} 列名: {list(df.columns)}")
                failed.append(code)
                continue

            df = df[[date_col[0], bal_col[0]]].copy()
            df.columns = ["date", "margin_balance"]
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()

            # 过滤日期范围
            df = df.loc[start:end, "margin_balance"]
            df = pd.to_numeric(df, errors="coerce")

            existing = data.get(code)
            if existing is not None and len(existing) > 0:
                combined = pd.concat([existing, df])
                data[code] = combined[~combined.index.duplicated(keep="last")].sort_index()
            else:
                data[code] = df

            time.sleep(0.15)

        except Exception as e:
            logger.warning(f"融资余额失败 {code}: {e}")
            failed.append(code)

        if (i + 1) % save_every == 0:
            logger.info(f"进度 {i+1}/{len(need)}，保存中间结果...")
            _save(data, out_path)

    if failed:
        logger.warning(f"失败 {len(failed)} 只: {failed[:10]}")

    return _save(data, out_path)


def _save(data: dict, path: Path) -> pd.DataFrame:
    df = pd.DataFrame(data).sort_index()
    df.to_parquet(path)
    logger.info(f"融资余额保存: {path.name}, shape={df.shape}")
    return df


def main(start: str, end: str, sample: int = 0):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    universe = pd.read_parquet(UNIVERSE_DIR / "stock_list.parquet")
    codes = universe["code"].tolist()
    if sample:
        codes = codes[:sample]
    download_margin(codes, start, end)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",  default="2018-01-01")
    parser.add_argument("--end",    default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--sample", type=int, default=0)
    args = parser.parse_args()
    main(args.start, args.end, args.sample)
