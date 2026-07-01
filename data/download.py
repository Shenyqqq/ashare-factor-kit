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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import akshare as ak
import pandas as pd
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import RAW_DIR, UNIVERSE_DIR

# 并发线程数：I/O密集型，线程数 > CPU核数也有效
# 太高会被东财限流（429），8-12 是经验值
OHLCV_WORKERS = 8
FIN_WORKERS   = 8

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
        import datetime
        now = datetime.datetime.now()
        today = pd.Timestamp.today().normalize()
        # A股 15:00 收盘后数据才发布；盘前/盘中用上一交易日
        if now.hour >= 15:
            _last_trade_date_cache = dates[dates <= today].max()
        else:
            _last_trade_date_cache = dates[dates < today].max()
        logger.info(f"最近已发布交易日: {_last_trade_date_cache.date()}")
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
# M4 修复：股票池须包含退市股（消除幸存者偏差），并补充 list_date/delist_date/
# is_st_current 元数据，供回测按日期动态过滤。下游不再按当前名字剔除 ST/退市。

# 统一的股票元数据列
STOCK_META_COLUMNS = ["code", "name", "list_date", "delist_date", "is_st_current"]


def _normalize_meta_table(df: pd.DataFrame) -> pd.DataFrame:
    """SH/SZ 股票列表表 → 统一列名 (code/name/list_date/delist_date)。

    实际列名随接口版本变化，按关键词匹配。注意源表可能同时含『证券简称』
    和『公司简称』两个含"简称"的列 —— 旧实现用 ``"name" not in col_map``
    检查 key（永远是中文列名）导致守卫恒真、两列都被映射成 `name`，concat
    时抛 InvalidIndexError。这里改为检查 ``col_map.values()`` 并对结果去重。
    """
    col_map = {}
    used: set[str] = set()
    for c in df.columns:
        cs = str(c)
        # 先判 delist_date：SH 用『暂停上市日期』、SZ 用『终止上市日期』、
        # 也有接口叫『退市日期』；都含"上市"+"日期"，必须在 list_date 之前匹配
        if "日期" in cs and ("终止" in cs or "退市" in cs or "暂停" in cs) and "delist_date" not in used:
            col_map[c] = "delist_date"; used.add("delist_date")
        elif "代码" in cs and "code" not in used:
            col_map[c] = "code"; used.add("code")
        elif "简称" in cs and "name" not in used:
            col_map[c] = "name"; used.add("name")
        elif "上市" in cs and "日期" in cs and "list_date" not in used:
            col_map[c] = "list_date"; used.add("list_date")
        elif "退市" in cs and "日期" in cs and "delist_date" not in used:
            col_map[c] = "delist_date"; used.add("delist_date")
    df = df.rename(columns=col_map)
    # 兜底：即使上面的守卫漏判，也强制去掉重复列（保留首次出现）
    df = df.loc[:, ~df.columns.duplicated()]
    keep = [c for c in ("code", "name", "list_date", "delist_date") if c in df.columns]
    out = df[keep].copy()
    for col in ("list_date", "delist_date"):
        if col not in out.columns:
            out[col] = pd.NaT
        else:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    if "name" not in out.columns:
        out["name"] = ""
    return out


# 兼容旧调用名（外部可能仍引用）
_normalize_sh_table = _normalize_meta_table
_normalize_sz_table = _normalize_meta_table


def _fetch_stock_list_with_metadata() -> pd.DataFrame:
    """优先用上交所/深交所官方接口（含上市日期），失败回退到 stock_info_a_code_name。

    返回列：code, name, list_date, delist_date, is_st_current

    注：akshare 1.18.x 起 ``stock_info_sz_name_code`` 的合法 ``symbol`` 仅剩
    ``'A股列表'`` / ``'B股列表'``（中小板 2021 年并入主板，创业板由 A股列表
    里的『板块』列区分）；``stock_info_sh_name_code`` 仍接受 主板A股/主板B股/
    科创板。各源表先经 ``_normalize_meta_table`` 去重列再 concat，避免
    InvalidIndexError。
    """
    frames: list[pd.DataFrame] = []

    # ── 上交所：主板A / 主板B / 科创板 ──
    for sym in ("主板A股", "主板B股", "科创板"):
        try:
            sh = ak.stock_info_sh_name_code(symbol=sym)
            if sh is not None and not sh.empty:
                frames.append(_normalize_meta_table(sh))
        except Exception as e:
            logger.warning(f"SH 接口({sym}) 失败: {e}")

    # ── 深交所：A股列表（含主板+创业板，按『板块』列区分）/ B股列表 ──
    for sym in ("A股列表", "B股列表"):
        try:
            sz = ak.stock_info_sz_name_code(symbol=sym)
            if sz is not None and not sz.empty:
                frames.append(_normalize_meta_table(sz))
        except Exception as e:
            logger.warning(f"SZ 接口({sym}) 失败: {e}")

    if frames:
        # 二次兜底：每个 frame 强制列名唯一，避免任意源引入重复列导致 concat 失败
        frames = [f.loc[:, ~f.columns.duplicated()] for f in frames]
        df = pd.concat(frames, ignore_index=True, sort=False)
        df["code"] = df["code"].astype(str).str.zfill(6)
        df = df.drop_duplicates(subset="code", keep="last").reset_index(drop=True)
        df["is_st_current"] = df["name"].astype(str).str.contains(
            "ST", case=False, na=False
        )
        logger.info(
            f"SH/SZ 接口取股票列表: {len(df)} 只 "
            f"(含退市 {df['delist_date'].notna().sum()} 只)"
        )
        return df

    # ── fallback：老接口（无上市/退市日期，仅当前在市）──
    logger.warning("回退到 ak.stock_info_a_code_name（无 list_date/delist_date 元数据）")
    df = ak.stock_info_a_code_name()
    df.columns = ["code", "name"]
    df["list_date"] = pd.NaT
    df["delist_date"] = pd.NaT
    df["is_st_current"] = df["name"].astype(str).str.contains(
        "ST", case=False, na=False
    )
    return df


def fetch_delisted_stocks() -> pd.DataFrame:
    """获取已退市股票清单（不在当前 SH/SZ 主接口里的）。

    优先调用 AKShare 退市接口（``stock_info_sh_delist`` / ``stock_info_sz_delist``，
    含上市日期 + 终止上市日期）；若接口不存在或失败，则从已下载的 raw parquet
    中扫描「曾经出现过但当前股票列表中已不存在」的代码作为退市股候选。

    返回列与 STOCK_META_COLUMNS 一致；失败时返回空 DataFrame。
    """
    frames: list[pd.DataFrame] = []
    # SH 退市：symbol='全部'；列=公司代码/公司简称/上市日期/终止上市日期
    sh_fn = getattr(ak, "stock_info_sh_delist", None)
    if sh_fn is not None:
        for sym in ("全部", "终止上市公司", "退市公司"):
            try:
                df = sh_fn(symbol=sym) if "symbol" in sh_fn.__code__.co_varnames else sh_fn()
                if df is not None and not df.empty:
                    frames.append(_normalize_meta_table(df))
                    logger.info(f"stock_info_sh_delist({sym}) 返回 {len(df)} 只")
                    break
            except Exception as e:
                logger.warning(f"stock_info_sh_delist({sym}) 失败: {e}")
    # SZ 退市：symbol='终止上市公司'；列=证券代码/证券简称/上市日期/终止上市日期
    sz_fn = getattr(ak, "stock_info_sz_delist", None)
    if sz_fn is not None:
        for sym in ("终止上市公司", "终止上市", "全部"):
            try:
                df = sz_fn(symbol=sym) if "symbol" in sz_fn.__code__.co_varnames else sz_fn()
                if df is not None and not df.empty:
                    frames.append(_normalize_meta_table(df))
                    logger.info(f"stock_info_sz_delist({sym}) 返回 {len(df)} 只")
                    break
            except Exception as e:
                logger.warning(f"stock_info_sz_delist({sym}) 失败: {e}")

    if frames:
        frames = [f.loc[:, ~f.columns.duplicated()] for f in frames]
        df = pd.concat(frames, ignore_index=True, sort=False)
        if "code" not in df.columns:
            logger.warning("退市接口返回无 code 列，回退到 raw parquet 反推")
            return _scan_delisted_from_raw()
        df["code"] = df["code"].astype(str).str.zfill(6)
        if "name" not in df.columns:
            df["name"] = ""
        for col in ("list_date", "delist_date"):
            if col not in df.columns:
                df[col] = pd.NaT
            else:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        df["is_st_current"] = df["name"].astype(str).str.contains(
            "ST", case=False, na=False
        )
        df = df.drop_duplicates(subset="code", keep="last").reset_index(drop=True)
        logger.info(f"退市接口合计 {len(df)} 只 (含 delist_date {df['delist_date'].notna().sum()})")
        return df[STOCK_META_COLUMNS]

    # fallback：扫描已下载 raw parquet 中曾出现过的代码
    logger.info("退市接口不可用，回退到 raw parquet 反推退市股候选")
    return _scan_delisted_from_raw()


def _scan_delisted_from_raw() -> pd.DataFrame:
    """从已下载的 raw/close_hfq.parquet 反推退市股：当前股票列表没有的代码视为退市候选。"""
    path = RAW_DIR / "close_hfq.parquet"
    if not path.exists():
        # 兼容旧文件名 prices_hfq.parquet
        path = RAW_DIR / "prices_hfq.parquet"
    if not path.exists():
        logger.info("无 raw parquet 可用于反推，退市清单为空")
        return pd.DataFrame(columns=STOCK_META_COLUMNS)

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        logger.warning(f"读取 {path} 失败: {e}")
        return pd.DataFrame(columns=STOCK_META_COLUMNS)

    all_codes = set(df.columns.astype(str).str.zfill(6))
    # 当前股票列表：直接调用底层接口拿当前在市股票代码集合
    try:
        cur_df = ak.stock_info_a_code_name()
        cur = set(cur_df.iloc[:, 0].astype(str).str.zfill(6))
    except Exception:
        cur = set()

    delist_codes = sorted(all_codes - cur)
    if not delist_codes:
        return pd.DataFrame(columns=STOCK_META_COLUMNS)

    # 用每只股票的「最后出现日期」近似 delist_date
    rows = []
    for code in delist_codes:
        if code not in df.columns:
            continue
        s = pd.to_numeric(df[code], errors="coerce").dropna()
        if s.empty:
            continue
        rows.append({
            "code": code,
            "name": "",
            "list_date": pd.NaT,
            "delist_date": s.index.max(),
            "is_st_current": False,
        })
    out = pd.DataFrame(rows, columns=STOCK_META_COLUMNS)
    logger.info(f"从 raw parquet 反推退市候选 {len(out)} 只")
    return out


def get_stock_list(include_delisted: bool = True) -> pd.DataFrame:
    """获取股票列表（含 list_date/delist_date/is_st_current 元数据）。

    Parameters
    ----------
    include_delisted : bool
        True（默认）= 合并退市股清单，消除幸存者偏差；
        False = 仅返回当前在市股票（向后兼容旧行为）。
    """
    df = _fetch_stock_list_with_metadata()
    if include_delisted:
        try:
            delisted = fetch_delisted_stocks()
            if not delisted.empty:
                # 仅补充当前列表里没有的代码
                existing = set(df["code"].astype(str))
                new_rows = delisted[~delisted["code"].astype(str).isin(existing)]
                if not new_rows.empty:
                    df = pd.concat([df, new_rows], ignore_index=True, sort=False)
                    logger.info(f"合并退市股后股票池: {len(df)} 只")
        except Exception as e:
            logger.warning(f"合并退市股失败（忽略，仅用当前列表）: {e}")
    df["code"] = df["code"].astype(str).str.zfill(6)
    return df


def filter_universe(stock_list: pd.DataFrame) -> pd.DataFrame:
    """构建回测股票池：保留所有曾上市股票（含已退市、当前 ST），仅剔除北交所。

    M4 修复：不再按当前名字剔除 ST/退市。
      - ST 状态由回测时按日期查询 st_schedule 决定（见 backtest.execution.build_st_schedule）
      - 退市股由回测时按 delist_date > date 判断仍在市（见 TradeRules.is_delisted）
    历史回测需要这些股票在退市前的价格数据，否则会引入幸存者偏差。
    """
    filtered = stock_list.copy()
    # 仅剔除北交所（8 开头代码）；保留 6/0/3 开头主板/创业板/科创板
    if "code" in filtered.columns:
        mask = ~filtered["code"].astype(str).str.startswith("8")
        filtered = filtered[mask].copy()
    # 确保元数据列存在（接口失败时补 NaT / False）
    for col in ("list_date", "delist_date"):
        if col not in filtered.columns:
            filtered[col] = pd.NaT
        else:
            filtered[col] = pd.to_datetime(filtered[col], errors="coerce")
    if "is_st_current" not in filtered.columns:
        filtered["is_st_current"] = (
            filtered["name"].astype(str).str.contains("ST", case=False, na=False)
            if "name" in filtered.columns else False
        )
    n_delist = filtered["delist_date"].notna().sum() if "delist_date" in filtered.columns else 0
    n_st = filtered["is_st_current"].sum() if "is_st_current" in filtered.columns else 0
    logger.info(
        f"过滤后股票池: {len(filtered)} 只（含退市 {n_delist}，当前 ST {n_st}）"
    )
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

    last_trade = get_last_trade_date()  # 已排除今日（盘后数据未发布）
    target_end = min(last_trade, pd.Timestamp(end))

    # 判断哪些股票需要更新（以 close 为准）
    close_col = "收盘"
    need = []
    for code in codes:
        series = data[close_col].get(code)
        if series is None or len(series) == 0 or series.index[-1] < target_end:
            need.append(code)

    if not need:
        label = "后复权" if adjust == "hfq" else "不复权"
        logger.info(f"OHLCV ({label}) 已是最新，跳过下载")
        return {key: pd.DataFrame(data[ak_col]).sort_index()
                for ak_col, (key, _) in paths.items()}

    label = "后复权" if adjust == "hfq" else "不复权"
    logger.info(f"OHLCV ({label}): 需下载/更新 {len(need)}/{len(codes)} 只（并发={OHLCV_WORKERS}线程）")

    lock = threading.Lock()
    failed = []
    done_count = 0

    def _fetch_ohlcv(code: str):
        series = data[close_col].get(code)
        code_start = (
            (series.index[-1] + pd.Timedelta(days=1)).strftime("%Y%m%d")
            if series is not None and len(series) > 0
            else start.replace("-", "")
        )
        try:
            df = ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date=code_start,
                end_date=end.replace("-", ""),
                adjust=adjust,
            )
            time.sleep(0.05)
            return code, df, None
        except Exception as e:
            return code, None, e

    with ThreadPoolExecutor(max_workers=OHLCV_WORKERS) as executor:
        futures = {executor.submit(_fetch_ohlcv, code): code for code in need}
        for future in as_completed(futures):
            code, df, err = future.result()
            nonlocal_done = 0
            with lock:
                done_count += 1
                nonlocal_done = done_count
                if err is not None:
                    failed.append(code)
                    logger.warning(f"OHLCV 下载失败 {code}: {err}")
                elif df is not None and not df.empty:
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

                if nonlocal_done % save_every == 0:
                    logger.info(f"OHLCV进度 {nonlocal_done}/{len(need)}，保存中间结果...")
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

    logger.info(f"需下载/更新 {len(need)}/{len(codes)} 只股票财务数据（并发={FIN_WORKERS}线程）")

    lock = threading.Lock()
    failed_codes = []
    success = skip = done_count = 0

    def _fetch_fin(code: str):
        for attempt in range(max_retry + 1):
            try:
                df = _fetch_one_financial(code, start_year, COLS)
                time.sleep(0.1)
                return code, df
            except Exception as e:
                if attempt < max_retry:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    logger.warning(f"财务指标失败 {code} (重试{max_retry}次): {e}")
                    return code, None

    with ThreadPoolExecutor(max_workers=FIN_WORKERS) as executor:
        futures = {executor.submit(_fetch_fin, code): code for code in need}
        for future in as_completed(futures):
            code, df = future.result()
            with lock:
                done_count += 1
                if df is not None:
                    records.append(df)
                    success += 1
                else:
                    skip += 1
                    failed_codes.append(code)

                if done_count % save_every == 0:
                    logger.info(
                        f"财务进度 {done_count}/{len(need)}  "
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
