"""
data/download.py  —  下载历史数据，存为 parquet

存储文件（全部宽表：index=日期，columns=股票代码）:
    prices_hfq.parquet   — 后复权收盘价（回测/动量因子）
    open_hfq.parquet     — 后复权开盘价
    high_hfq.parquet     — 后复权最高价
    low_hfq.parquet      — 后复权最低价
    volume.parquet       — 成交量（股）
    amount.parquet       — 成交额（元）
    prices_raw.parquet   — 不复权收盘价（计算PB用）
    financial_indicators.parquet — 季报财务指标

特性：
  - 断点续传：中断后重跑自动跳过已下载的股票
  - 增量更新：已有数据只补充最新部分，不重复下载历史
  - OHLCV 一次请求全取，不重复调用 API

用法:
    python -m data.download                    # 全量（首次）
    python -m data.download --update           # 增量更新到今天
    python -m data.download --sample 100       # 调试：只下载前100只
"""
import argparse
import time
from pathlib import Path

import akshare as ak
import pandas as pd
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import RAW_DIR, UNIVERSE_DIR

# OHLCV 字段映射：AKShare列名 → 文件名前缀
OHLCV_FIELDS = {
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
}


# ── 工具 ──────────────────────────────────────────────────────────────────────

def ensure_dirs():
    for d in [RAW_DIR, UNIVERSE_DIR]:
        d.mkdir(parents=True, exist_ok=True)


_last_trade_date_cache: pd.Timestamp | None = None

def get_last_trade_date() -> pd.Timestamp:
    global _last_trade_date_cache
    if _last_trade_date_cache is not None:
        return _last_trade_date_cache
    try:
        cal = ak.tool_trade_date_hist_sina()
        dates = pd.to_datetime(cal["trade_date"])
        _last_trade_date_cache = dates[dates <= pd.Timestamp.today()].max()
        logger.info(f"最近交易日: {_last_trade_date_cache.date()}")
    except Exception as e:
        logger.warning(f"获取交易日历失败，使用今天: {e}")
        _last_trade_date_cache = pd.Timestamp.today().normalize()
    return _last_trade_date_cache


def _load_parquet(path) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_parquet(p) if p.exists() else None


def _save_wide(data: dict, path: Path):
    """dict{code: Series} → 宽表 parquet"""
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index)
    df.sort_index().to_parquet(path)
    return df


# ── 股票列表 ──────────────────────────────────────────────────────────────────

def get_stock_list() -> pd.DataFrame:
    logger.info("获取股票列表...")
    df = ak.stock_info_a_code_name()
    df.columns = ["code", "name"]
    df["code"] = df["code"].astype(str).str.zfill(6)
    return df


def filter_universe(stock_list: pd.DataFrame) -> pd.DataFrame:
    mask = ~stock_list["name"].str.contains("ST|退", na=False)
    mask &= ~stock_list["code"].str.startswith("8")
    filtered = stock_list[mask].copy()
    logger.info(f"过滤后股票池: {len(filtered)} 只")
    return filtered


# ── OHLCV 下载（一次请求取全部字段）────────────────────────────────────────────

def download_ohlcv(
    codes: list,
    start: str,
    end: str,
    adjust: str = "hfq",
    save_every: int = 200,
) -> dict[str, pd.DataFrame]:
    """
    下载日线 OHLCV，一次 API 请求取 open/close/high/low/volume/amount。

    adjust: "hfq"=后复权, ""=不复权
    返回: {field_name: DataFrame(date×stock)}
        后复权: open_hfq, close_hfq, high_hfq, low_hfq, volume, amount
        不复权: close_raw（仅 close）
    """
    suffix = "_hfq" if adjust == "hfq" else "_raw"
    # 文件路径映射
    paths = {}
    if adjust == "hfq":
        for ak_col, fname in OHLCV_FIELDS.items():
            key = fname if fname in ("volume", "amount") else f"{fname}{suffix}"
            paths[ak_col] = (key, RAW_DIR / f"{key}.parquet")
    else:
        # 不复权只存收盘价
        paths["收盘"] = ("close_raw", RAW_DIR / "prices_raw.parquet")

    # 加载已有数据
    data: dict[str, dict] = {ak_col: {} for ak_col in paths}
    for ak_col, (key, path) in paths.items():
        existing = _load_parquet(path)
        if existing is not None:
            data[ak_col] = {c: existing[c].dropna()
                            for c in existing.columns if c in existing}

    last_trade = get_last_trade_date()

    # 判断哪些股票需要更新（以 close 为准）
    close_col = "收盘"
    need = []
    for code in codes:
        series = data[close_col].get(code)
        if series is None or len(series) == 0 or series.index[-1] < last_trade:
            need.append(code)

    if not need:
        label = "后复权" if adjust == "hfq" else "不复权"
        logger.info(f"OHLCV ({label}) 已是最新，跳过下载")
        return {key: pd.DataFrame(data[ak_col]).sort_index()
                for ak_col, (key, _) in paths.items()}

    label = "后复权" if adjust == "hfq" else "不复权"
    logger.info(f"OHLCV ({label}): 需下载/更新 {len(need)}/{len(codes)} 只")
    failed = []

    for i, code in enumerate(need):
        # 增量起始日
        series = data[close_col].get(code)
        if series is not None and len(series) > 0:
            code_start = (series.index[-1] + pd.Timedelta(days=1)).strftime("%Y%m%d")
        else:
            code_start = start.replace("-", "")

        try:
            df = ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date=code_start,
                end_date=end.replace("-", ""),
                adjust=adjust,
            )
            if not df.empty:
                df["日期"] = pd.to_datetime(df["日期"])
                df = df.set_index("日期")

                for ak_col in paths:
                    if ak_col not in df.columns:
                        continue
                    new_s = df[ak_col].astype(float)
                    existing_s = data[ak_col].get(code)
                    if existing_s is not None and len(existing_s) > 0:
                        combined = pd.concat([existing_s, new_s])
                        data[ak_col][code] = combined[
                            ~combined.index.duplicated(keep="last")
                        ].sort_index()
                    else:
                        data[ak_col][code] = new_s

            time.sleep(0.05)

        except Exception as e:
            failed.append(code)
            logger.warning(f"OHLCV 下载失败 {code}: {e}")

        if (i + 1) % save_every == 0:
            logger.info(f"进度 {i+1}/{len(need)}，保存中间结果...")
            for ak_col, (key, path) in paths.items():
                if data[ak_col]:
                    _save_wide(data[ak_col], path)

    if failed:
        logger.warning(f"失败 {len(failed)} 只: {failed[:10]}")

    # 最终保存
    result = {}
    for ak_col, (key, path) in paths.items():
        df_out = _save_wide(data[ak_col], path)
        result[key] = df_out
        logger.info(f"  {path.name}: {df_out.shape}")

    return result


# ── 财务指标（支持断点续传 + 增量更新）────────────────────────────────────────

def _fetch_one_financial(code: str, start_year: str, cols: dict) -> pd.DataFrame | None:
    df = ak.stock_financial_analysis_indicator(symbol=code, start_year=start_year)
    if df is None or df.empty:
        return None

    col_map = {}
    actual_cols = list(df.columns)

    KEYWORDS = {
        "trade_date":          ["日期"],
        "roe":                 ["净资产收益率(%)"],
        "bvps":                ["每股净资产_调整后"],
        "total_assets":        ["总资产(元)"],
        "eps":                 ["摊薄每股收益"],
        "gross_profit_margin": ["销售毛利率"],
        "operating_cashflow":  ["每股经营性现金流"],
        "debt_ratio":          ["资产负债率(%)"],
        "net_profit_growth":   ["净利润增长率"],
        "revenue_growth":      ["主营业务收入增长率"],
        "net_profit_margin":   ["销售净利率"],
    }

    for target, keywords in KEYWORDS.items():
        exact = {k: v for k, v in cols.items() if v == target}
        matched = [k for k in exact if k in actual_cols]
        if matched:
            col_map[matched[0]] = target
            continue
        for col in actual_cols:
            if any(kw in col for kw in keywords):
                col_map[col] = target
                break

    if "trade_date" not in col_map.values():
        logger.debug(f"{code} 找不到日期列，实际列名: {actual_cols[:8]}")
        return None

    df = df[list(col_map.keys())].rename(columns=col_map).copy()
    df["code"] = code

    numeric_cols = [
        "roe", "bvps", "total_assets", "eps", "gross_profit_margin",
        "operating_cashflow", "debt_ratio", "net_profit_growth",
        "revenue_growth", "net_profit_margin",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df if not df.empty else None


def download_financial_indicators(
    codes: list,
    start_year: str = "2015",
    out_path: Path = None,
    save_every: int = 50,
    max_retry: int = 2,
    retry_delay: float = 1.0,
) -> pd.DataFrame:
    COLS = {
        "日期":                    "trade_date",
        "净资产收益率(%)":           "roe",
        "每股净资产_调整后(元)":      "bvps",
        "总资产(元)":               "total_assets",
        "摊薄每股收益(元)":          "eps",
        "销售毛利率(%)":            "gross_profit_margin",
        "每股经营性现金流(元)":       "operating_cashflow",
        "资产负债率(%)":            "debt_ratio",
        "净利润增长率(%)":           "net_profit_growth",
        "主营业务收入增长率(%)":      "revenue_growth",
        "销售净利率(%)":            "net_profit_margin",
    }

    existing = _load_parquet(out_path) if out_path else None
    fresh_codes: set = set()
    records: list = []

    if existing is not None and not existing.empty:
        cutoff = pd.Timestamp.now() - pd.DateOffset(months=4)
        for code, grp in existing.groupby("code"):
            if grp["trade_date"].max() >= cutoff:
                fresh_codes.add(code)
                records.append(grp)
        logger.info(f"财务数据: {len(fresh_codes)} 只已是最新，跳过")

    need = [c for c in codes if c not in fresh_codes]
    if not need:
        logger.info("财务数据已全部最新，跳过下载")
        return existing

    logger.info(f"需下载/更新 {len(need)}/{len(codes)} 只股票财务数据")

    failed_codes = []
    success = skip = 0

    for i, code in enumerate(need):
        df = None
        for attempt in range(max_retry + 1):
            try:
                df = _fetch_one_financial(code, start_year, COLS)
                time.sleep(0.1)
                break
            except Exception as e:
                if attempt < max_retry:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    logger.warning(f"财务指标失败 {code} (重试{max_retry}次): {e}")
                    failed_codes.append(code)

        if df is not None:
            records.append(df)
            success += 1
        else:
            skip += 1

        if (i + 1) % save_every == 0:
            logger.info(
                f"财务进度 {i+1}/{len(need)}  "
                f"成功={success} 跳过={skip} 失败={len(failed_codes)}"
            )
            if records:
                _save_financial(records, out_path)

    if failed_codes:
        logger.warning(f"最终失败 {len(failed_codes)} 只: {failed_codes[:10]}")

    if not records:
        return pd.DataFrame()

    return _save_financial(records, out_path)


def _save_financial(records: list, out_path: Path) -> pd.DataFrame:
    result = pd.concat(records, ignore_index=True)
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    result = result.drop_duplicates(subset=["trade_date", "code"], keep="last")
    if out_path:
        result.to_parquet(out_path)
    return result


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main(start: str, end: str, sample: int = 0):
    ensure_dirs()

    stock_list = get_stock_list()
    universe   = filter_universe(stock_list)
    universe.to_parquet(UNIVERSE_DIR / "stock_list.parquet")

    codes = universe["code"].tolist()
    if sample:
        codes = codes[:sample]
        logger.info(f"调试模式：仅处理 {sample} 只股票")

    fin_path = RAW_DIR / "financial_indicators.parquet"

    # 后复权 OHLCV（回测 + 大部分因子）
    logger.info("=== 后复权 OHLCV ===")
    hfq = download_ohlcv(codes, start, end, adjust="hfq")
    # 向后兼容：prices_hfq.parquet = 后复权收盘价
    if "close_hfq" in hfq:
        hfq["close_hfq"].to_parquet(RAW_DIR / "prices_hfq.parquet")

    # 不复权收盘价（计算 PB 用）
    logger.info("=== 不复权收盘价 ===")
    download_ohlcv(codes, start, end, adjust="")
    # 向后兼容：prices_raw.parquet 已在 download_ohlcv 里保存

    # 财务季报
    logger.info("=== 财务指标（季报）===")
    fin = download_financial_indicators(codes, start_year=start[:4], out_path=fin_path)
    if fin is not None and not fin.empty:
        logger.info(f"财务数据: {fin.shape}")

    logger.info("数据下载完成！")
    logger.info("生成的文件:")
    for f in sorted(RAW_DIR.glob("*.parquet")):
        size_mb = f.stat().st_size / 1024 / 1024
        logger.info(f"  {f.name:35s} {size_mb:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",  default="2018-01-01")
    parser.add_argument("--end",    default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--sample", type=int, default=0)
    args = parser.parse_args()
    main(args.start, args.end, args.sample)
