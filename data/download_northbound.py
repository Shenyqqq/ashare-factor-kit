"""
data/download_northbound.py  —  北向资金个股持股数据下载

⚠️ 北向持股披露已于 2024-08 起实质停更（监管口径调整）。
本模块产物仅作历史归档；默认因子/IC/白名单路径**不再加载**北向数据
（见 run.py / research.ic.load_data）。勿把停更后的稀疏/空值当成有效信号。

接口：ak.stock_hsgt_individual_em(symbol)
      按股票代码拉取沪深股通历史持股数量（全量历史，约从2017年开始）

存储：
    data/raw/northbound_holding.parquet  持股量宽表（index=日期, columns=股票）
    data/raw/northbound_value.parquet    持股市值宽表

特点：
  - 逐股拉取，每只股票返回完整历史（约1500-2000行）
  - 断点续传：已有且数据较新的股票跳过
  - 只下载沪深股通成分股（600/601/603/605 沪市 + 000/001/002/300 深市中的纳通标的）

用法：
    python -m data.download_northbound
    python -m data.download_northbound --sample 20
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

# 北向个股持股公开披露实质停更日（约）；此日后数据不可信/常为空
NORTHBOUND_DISCLOSURE_STOP = pd.Timestamp("2024-08-19")


def _warn_northbound_deprecated() -> None:
    logger.warning(
        f"北向持股披露已于约 {NORTHBOUND_DISCLOSURE_STOP.date()} 停更；"
        "本下载仅归档历史。默认管线不加载 northbound（勿误用停更后区间做因子/IC）。"
    )


def _load_existing(path: Path) -> dict:
    if path.exists():
        df = pd.read_parquet(path)
        return {c: df[c].dropna() for c in df.columns}
    return {}


def download_northbound(codes: list, save_every: int = 100) -> tuple:
    _warn_northbound_deprecated()
    hold_path = RAW_DIR / "northbound_holding.parquet"
    val_path  = RAW_DIR / "northbound_value.parquet"

    hold_data = _load_existing(hold_path)
    val_data  = _load_existing(val_path)

    # 跳过近30日内已更新的股票
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=30)
    need = []
    for code in codes:
        s = hold_data.get(code)
        if s is None or len(s) == 0 or s.index[-1] < cutoff:
            need.append(code)

    if not need:
        logger.info("北向资金数据已是最新，跳过")
        return pd.DataFrame(hold_data), pd.DataFrame(val_data)

    logger.info(f"北向资金: 需下载 {len(need)}/{len(codes)} 只")
    failed = []

    for i, code in enumerate(need):
        try:
            df = ak.stock_hsgt_individual_em(symbol=code)
            if df is None or df.empty:
                continue

            # 列顺序固定：持股日期, 收盘价, 涨跌幅, 持股数量, 持股市值, 持股占A股%, 当日增减, 当日净买入, 近一周持股市值变化
            df.columns = ["date", "close", "pct_chg", "holding",
                          "value", "hold_pct", "chg", "net_buy", "weekly_chg"]
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()

            hold_s = pd.to_numeric(df["holding"], errors="coerce")
            val_s  = pd.to_numeric(df["value"],   errors="coerce")

            for data_dict, s in [(hold_data, hold_s), (val_data, val_s)]:
                existing = data_dict.get(code)
                if existing is not None and len(existing) > 0:
                    combined = pd.concat([existing, s])
                    data_dict[code] = combined[~combined.index.duplicated(keep="last")].sort_index()
                else:
                    data_dict[code] = s

            time.sleep(0.2)

        except Exception as e:
            logger.warning(f"北向资金失败 {code}: {e}")
            failed.append(code)

        if (i + 1) % save_every == 0:
            logger.info(f"进度 {i+1}/{len(need)}")
            _save(hold_data, val_data, hold_path, val_path)

    if failed:
        logger.warning(f"失败 {len(failed)} 只: {failed[:10]}")

    return _save(hold_data, val_data, hold_path, val_path)


def _save(hold_data, val_data, hold_path, val_path):
    hold_df = pd.DataFrame(hold_data).sort_index()
    val_df  = pd.DataFrame(val_data).sort_index()
    hold_df.to_parquet(hold_path)
    val_df.to_parquet(val_path)
    logger.info(f"北向资金保存: holding={hold_df.shape}, value={val_df.shape}")
    return hold_df, val_df


def main(sample: int = 0):
    _warn_northbound_deprecated()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    universe = pd.read_parquet(UNIVERSE_DIR / "stock_list.parquet")
    codes = universe["code"].tolist()
    if sample:
        codes = codes[:sample]
    download_northbound(codes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0)
    args = parser.parse_args()
    main(args.sample)
