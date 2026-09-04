"""
data/download.py  —  下载历史数据，存为 parquet

存储文件（全部宽表：index=日期，columns=股票代码）:
    prices_hfq.parquet   — 后复权收盘价（回测/动量因子）
    open_hfq.parquet     — 后复权开盘价
    high_hfq.parquet     — 后复权最高价
    low_hfq.parquet      — 后复权最低价
    volume.parquet       — 成交量（**手**；1 手 = 100 股；AKShare stock_zh_a_hist 原口径）
    amount.parquet       — 成交额（元）
    prices_raw.parquet   — 不复权收盘价（计算 PB / 自算市值用）
    financial_indicators.parquet — 季报财务指标

特性：
  - 断点续传：中断后重跑自动跳过已下载的股票
  - 增量更新：已有数据只补充最新部分，不重复下载历史
  - OHLCV 一次请求全取，不重复调用 API
  - **hfq / raw 对齐**：不复权下载会对照 close_hfq 末日，强制补齐落后股票；
    主流程结束后校验覆盖率，防止 raw 相对 hfq 静默塌缩（拖垮 circ_mv）

用法:
    python -m data.download                    # 增量补齐到今天（已有则只补新区间）
    python -m data.download --start 2018-01-01 # 指定起始（默认即此）
    python -m data.download --sample 100       # 调试：只下载前100只
    # 注意：没有 --update 开关；日常直接跑本模块即增量。详见 data/DATA_UPDATE.md
"""
import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import akshare as ak
import pandas as pd
import requests
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import RAW_DIR, UNIVERSE_DIR

# 并发线程数：I/O密集型，线程数 > CPU核数也有效
# 太高会被东财限流（429），8-12 是经验值；push2his 被掐时可 OHLCV_WORKERS=2
OHLCV_WORKERS = int(os.environ.get("OHLCV_WORKERS", "8"))
FIN_WORKERS   = 8

_EM_KLINE_URLS = (
    "https://push2delay.eastmoney.com/api/qt/stock/kline/get",
    "https://push2his.eastmoney.com/api/qt/stock/kline/get",
)
_EM_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def _fetch_em_kline(code: str, start_ymd: str, end_ymd: str, adjust: str) -> pd.DataFrame | None:
    """东财日 K。push2his 常 RemoteDisconnected，回退 push2delay（同字段、同复权口径）。"""
    market_code = 1 if str(code).startswith("6") else 0
    fqt = {"qfq": "1", "hfq": "2", "": "0"}.get(adjust, "2")
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "klt": "101",
        "fqt": fqt,
        "secid": f"{market_code}.{str(code).zfill(6)}",
        "beg": start_ymd.replace("-", ""),
        "end": end_ymd.replace("-", ""),
    }
    headers = {
        "User-Agent": _EM_UA,
        "Referer": "https://quote.eastmoney.com/",
        "Connection": "close",
    }
    last_err = None
    for url in _EM_KLINE_URLS:
        s = requests.Session()
        s.trust_env = False
        s.proxies = {"http": None, "https": None}
        try:
            r = s.get(url, params=params, timeout=8, headers=headers)
            r.raise_for_status()
            payload = r.json()
            klines = (payload.get("data") or {}).get("klines") or []
            if not klines:
                continue
            temp = pd.DataFrame([item.split(",") for item in klines])
            cols = ["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额",
                    "振幅", "涨跌幅", "涨跌额", "换手率"]
            temp = temp.iloc[:, :len(cols)]
            temp.columns = cols[: temp.shape[1]]
            temp["日期"] = pd.to_datetime(temp["日期"])
            for c in ("开盘", "收盘", "最高", "最低", "成交量", "成交额"):
                if c in temp.columns:
                    temp[c] = pd.to_numeric(temp[c], errors="coerce")
            return temp
        except Exception as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    return None

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


def _last_valid_by_code(path: Path) -> dict[str, pd.Timestamp]:
    """宽表 parquet → {code: last_valid_index}；文件不存在则 {}。"""
    if not path.exists():
        return {}
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        logger.warning(f"读取 peer 面板失败 {path}: {e}")
        return {}
    df.index = pd.to_datetime(df.index)
    df.columns = df.columns.astype(str).str.zfill(6)
    out: dict[str, pd.Timestamp] = {}
    for c in df.columns:
        s = df[c].dropna()
        if len(s):
            out[c] = pd.Timestamp(s.index[-1])
    return out


def report_raw_hfq_coverage(
    hfq_path: Path | None = None,
    raw_path: Path | None = None,
    max_lag_days: int = 3,
    raise_on_gap: bool = False,
) -> dict:
    """对比 close_hfq vs prices_raw 覆盖；raw 落后时 WARNING（可选 raise）。

    根因场景：hfq / raw 分两次增量下载，raw 中断或被限流后，约一半股票停在旧末日，
    而 ``circ_mv = prices_raw × circ_shares`` 会随 raw 塌缩。
    """
    hfq_path = hfq_path or (RAW_DIR / "close_hfq.parquet")
    raw_path = raw_path or (RAW_DIR / "prices_raw.parquet")
    if not hfq_path.exists() or not raw_path.exists():
        logger.warning("raw/hfq 覆盖校验跳过：文件缺失")
        return {}
    hfq = pd.read_parquet(hfq_path)
    raw = pd.read_parquet(raw_path)
    hfq.index = pd.to_datetime(hfq.index)
    raw.index = pd.to_datetime(raw.index)
    hfq.columns = hfq.columns.astype(str).str.zfill(6)
    raw.columns = raw.columns.astype(str).str.zfill(6)
    common = hfq.columns.intersection(raw.columns)
    if len(common) == 0:
        logger.warning("raw/hfq 无共同列，覆盖校验跳过")
        return {}
    hn = hfq[common].notna()
    rn = raw[common].notna()
    miss = hn & ~rn
    last_h = hfq[common].apply(lambda s: s.last_valid_index())
    last_r = raw[common].apply(lambda s: s.last_valid_index())
    lag = (last_h - last_r).dt.days
    behind = int((lag > max_lag_days).sum())
    stats = {
        "hfq_overall": float(hn.mean().mean()),
        "raw_overall": float(rn.mean().mean()),
        "cells_hfq_ok_raw_miss": int(miss.sum().sum()),
        "stocks_raw_behind_hfq": behind,
        "max_lag_days_allowed": max_lag_days,
    }
    msg = (
        f"prices_raw vs hfq 覆盖: overall hfq={stats['hfq_overall']:.4f} "
        f"raw={stats['raw_overall']:.4f}; "
        f"hfq有值raw缺={stats['cells_hfq_ok_raw_miss']}; "
        f"raw落后>{max_lag_days}日的股票={behind}"
    )
    if behind > 0 or stats["cells_hfq_ok_raw_miss"] > 0:
        logger.warning(msg + " — 请重跑 `python -m data.download` 补齐不复权")
        if raise_on_gap:
            raise RuntimeError(msg)
    else:
        logger.info(msg + " ✓")
    return stats


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


def _fetch_bj_stock_list() -> pd.DataFrame:
    """北交所在市名单（``ak.stock_info_bj_name_code``），统一为 STOCK_META_COLUMNS。

    失败返回空表。仅保留 ``92xxxx``（与 ``filter_universe`` 一致；8 开头不收录）。
    """
    fn = getattr(ak, "stock_info_bj_name_code", None)
    if fn is None:
        logger.warning("akshare 无 stock_info_bj_name_code，跳过北交所名单")
        return pd.DataFrame(columns=STOCK_META_COLUMNS)
    try:
        raw = fn()
    except Exception as e:
        logger.warning(f"BJ 接口 stock_info_bj_name_code 失败: {e}")
        return pd.DataFrame(columns=STOCK_META_COLUMNS)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=STOCK_META_COLUMNS)
    df = _normalize_meta_table(raw)
    df["code"] = df["code"].astype(str).str.zfill(6)
    df = df[df["code"].str.startswith("92")].copy()
    df = df.drop_duplicates(subset="code", keep="last")
    df["is_st_current"] = df["name"].astype(str).str.contains("ST", case=False, na=False)
    keep = [c for c in STOCK_META_COLUMNS if c in df.columns]
    out = df[keep].copy()
    for col in STOCK_META_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NaT if col.endswith("date") else (False if col == "is_st_current" else "")
    logger.info(f"BJ 接口取股票列表: {len(out)} 只 92xxxx")
    return out[STOCK_META_COLUMNS]


def _bj_codes_from_price_panels() -> list[str]:
    """从 ``prices_hfq`` / ``close_hfq`` 列名取 92xxxx（只读 schema，不加载整表）。"""
    codes: set[str] = set()
    try:
        import pyarrow.parquet as pq
    except ImportError:
        pq = None  # type: ignore[assignment]
    for fname in ("prices_hfq.parquet", "close_hfq.parquet"):
        p = RAW_DIR / fname
        if not p.exists():
            continue
        try:
            if pq is not None:
                names = pq.read_schema(p).names
                cols = [c for c in names if c not in ("date", "__index_level_0__")]
            else:
                cols = pd.read_parquet(p).columns.tolist()
            codes.update(
                c.zfill(6) for c in map(str, cols) if str(c).zfill(6).startswith("92")
            )
        except Exception as e:
            logger.warning(f"从 {fname} 读北交所列失败: {e}")
    return sorted(codes)


def collect_bj_stock_codes(*, use_api: bool = True) -> list[str]:
    """北交所 ``92xxxx`` 代码：``prices_hfq`` 列 ∪ 可选 ``stock_info_bj_name_code``。

    与 ``filter_universe`` 同口径：保留 92，剔除 200/900/8。
    """
    codes: set[str] = set(_bj_codes_from_price_panels())
    if use_api:
        bj = _fetch_bj_stock_list()
        if not bj.empty:
            codes.update(bj["code"].astype(str).str.zfill(6).tolist())
    out = sorted(c for c in codes if c.startswith("92") and not is_excluded_universe_code(c))
    logger.info(
        f"北交所 92xxxx 代码 {len(out)} 只"
        f"（prices_hfq{' ∪ BJ 接口' if use_api else ''}）"
    )
    return out


def append_bj_codes_to_stock_list(codes: list[str] | None = None) -> int:
    """只追加缺失的 92xxxx 到 ``stock_list.parquet``，不删历史 B 股/退市行。

    Returns
    -------
    int
        新追加只数。
    """
    p = UNIVERSE_DIR / "stock_list.parquet"
    if not p.exists():
        logger.warning(f"找不到 {p}，跳过追加北交所")
        return 0
    sl = pd.read_parquet(p)
    sl["code"] = sl["code"].astype(str).str.zfill(6)
    have = set(sl["code"])
    n_b_before = int(sl["code"].map(is_b_share_code).sum())
    want = codes if codes is not None else collect_bj_stock_codes()
    want = [
        c.zfill(6) for c in want
        if str(c).zfill(6).startswith("92") and not is_excluded_universe_code(c)
    ]
    new_codes = [c for c in want if c not in have]
    if not new_codes:
        logger.info("stock_list 已含全部待追加 92xxxx，不改写")
        return 0

    meta = _fetch_bj_stock_list()
    meta_map: dict[str, dict] = {}
    if not meta.empty:
        for _, row in meta.iterrows():
            meta_map[str(row["code"]).zfill(6)] = row.to_dict()

    rows = []
    for c in new_codes:
        m = meta_map.get(c, {})
        name = str(m.get("name", "") or "")
        rows.append({
            "code": c,
            "name": name,
            "list_date": m.get("list_date", pd.NaT),
            "delist_date": m.get("delist_date", pd.NaT),
            "is_st_current": bool(m.get("is_st_current", False))
            or ("ST" in name.upper()),
        })
    extra = pd.DataFrame(rows)
    for col in sl.columns:
        if col not in extra.columns:
            extra[col] = pd.NA
    extra = extra[list(sl.columns)]
    out = pd.concat([sl, extra], ignore_index=True)
    n_b_after = int(out["code"].astype(str).str.zfill(6).map(is_b_share_code).sum())
    if n_b_after != n_b_before:
        raise RuntimeError(
            f"追加 92 后 B 股行数变化 {n_b_before} → {n_b_after}，拒绝写盘"
        )
    UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(p)
    logger.info(
        f"stock_list 追加北交所 {len(new_codes)} 只（未删历史 B/退市行），"
        f"现 {len(out)} 只；B 股仍 {n_b_after} 行"
    )
    return len(new_codes)


def _fetch_stock_list_with_metadata() -> pd.DataFrame:
    """优先用上交所/深交所官方接口（含上市日期），失败回退到 stock_info_a_code_name。

    返回列：code, name, list_date, delist_date, is_st_current

    注：akshare 1.18.x 起 ``stock_info_sz_name_code`` 的合法 ``symbol`` 仅剩
    ``'A股列表'`` / ``'B股列表'``；``stock_info_sh_name_code`` 仍接受
    主板A股/主板B股/科创板。本函数**不再拉取 B 股列表**；``filter_universe``
    再兜底去掉 ``200xxxx`` / ``900xxxx`` 与北交所 ``8xxxxx``。
    北交所 ``92xxxx`` 经 ``stock_info_bj_name_code`` 并入（与 filter_universe 一致：保留 92）。
    各源表先经 ``_normalize_meta_table`` 去重列再 concat，避免 InvalidIndexError。
    """
    frames: list[pd.DataFrame] = []

    # ── 上交所：主板A / 科创板（B 股不拉，filter_universe 再兜底）──
    for sym in ("主板A股", "科创板"):
        try:
            sh = ak.stock_info_sh_name_code(symbol=sym)
            if sh is not None and not sh.empty:
                frames.append(_normalize_meta_table(sh))
        except Exception as e:
            logger.warning(f"SH 接口({sym}) 失败: {e}")

    # ── 深交所：A股列表（含主板+创业板，按『板块』列区分）──
    for sym in ("A股列表",):
        try:
            sz = ak.stock_info_sz_name_code(symbol=sym)
            if sz is not None and not sz.empty:
                frames.append(_normalize_meta_table(sz))
        except Exception as e:
            logger.warning(f"SZ 接口({sym}) 失败: {e}")

    # ── 北交所：92xxxx（保留；8 开头由 filter_universe / is_excluded 剔除）──
    bj = _fetch_bj_stock_list()
    if bj is not None and not bj.empty:
        frames.append(bj)

    if frames:
        # 二次兜底：每个 frame 强制列名唯一，避免任意源引入重复列导致 concat 失败
        frames = [f.loc[:, ~f.columns.duplicated()] for f in frames]
        df = pd.concat(frames, ignore_index=True, sort=False)
        df["code"] = df["code"].astype(str).str.zfill(6)
        df = df.drop_duplicates(subset="code", keep="last").reset_index(drop=True)
        df["is_st_current"] = df["name"].astype(str).str.contains(
            "ST", case=False, na=False
        )
        n_bj = int(df["code"].astype(str).str.zfill(6).str.startswith("92").sum())
        logger.info(
            f"SH/SZ/BJ 接口取股票列表: {len(df)} 只 "
            f"(含退市 {df['delist_date'].notna().sum()} 只，北交所92 {n_bj} 只)"
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


# 深市 B 股 200xxxx / 沪市 B 股 900xxxx。92/43 不是 B，startswith("8") 也不会误伤它们。
B_SHARE_PREFIXES = ("200", "900")


def normalize_stock_code(code: object) -> str:
    """6 位数字代码；去掉 ``.SZ`` / ``.SH`` / ``.BJ`` 后缀。非股票列名原样返回。"""
    s = str(code).strip()
    if not s or s.lower() in {"nan", "none"}:
        return ""
    if "." in s:
        s = s.split(".", 1)[0]
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) < 5:
        return s
    return digits.zfill(6)


def is_b_share_code(code: object) -> bool:
    """深市 B 股 ``200xxxx``、沪市 B 股 ``900xxxx``（可带交易所后缀）。"""
    return normalize_stock_code(code).startswith(B_SHARE_PREFIXES)


def is_excluded_universe_code(code: object) -> bool:
    """默认股票池剔除：北交所 ``8xxxxx`` + 沪深 B 股。

    不含北交所 ``92xxxx``、新三板 ``43xxxx``。
    """
    c = normalize_stock_code(code)
    return c.startswith("8") or c.startswith(B_SHARE_PREFIXES)


def drop_excluded_universe_columns(
    df: pd.DataFrame | None,
    *,
    name: str | None = None,
) -> pd.DataFrame | None:
    """从宽表（columns=code）去掉北交所 8 开头与 B 股，供 IC / run.py / live 加载口共用。"""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    keep = [c for c in df.columns if not is_excluded_universe_code(c)]
    n_drop = len(df.columns) - len(keep)
    if n_drop <= 0:
        return df
    if name:
        logger.info(f"{name}: 剔除北交所8开头/B股 {n_drop} 列，余 {len(keep)}")
    return df.loc[:, keep]


def filter_universe(stock_list: pd.DataFrame) -> pd.DataFrame:
    """构建回测股票池：保留曾上市 A 股（含已退市、当前 ST），剔除北交所 8 开头与沪深 B 股。

    M4 修复：不再按当前名字剔除 ST/退市。
      - ST 状态由回测时按日期查询 st_schedule 决定（见 backtest.execution.build_st_schedule）
      - 退市股由回测时按 delist_date > date 判断仍在市（见 TradeRules.is_delisted）
    历史回测需要这些股票在退市前的价格数据，否则会引入幸存者偏差。

    B 股（深 200xxxx / 沪 900xxxx）以美元或港元计价，与 A 股截面不可比；默认剔除。
    北交所 ``92xxxx`` / 新三板 ``43xxxx`` 不在此列（``startswith('8')`` 不会误伤）。
    """
    filtered = stock_list.copy()
    # 剔除北交所 8 开头 + B 股；保留 000/002/300/301/600/601/603/605/688 与 92
    if "code" in filtered.columns:
        mask = ~filtered["code"].map(is_excluded_universe_code)
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
    peer_last: dict[str, pd.Timestamp] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    下载日线 OHLCV，一次 API 请求取 open/close/high/low/volume/amount。

    adjust: "hfq"=后复权, ""=不复权
    peer_last: 可选对照面板末日 {code: Timestamp}。用于不复权下载时强制补齐
        相对 close_hfq 落后的股票（防止 raw/hfq 分次下载导致覆盖塌缩）。
    返回: {field_name: DataFrame(date×stock)}
        后复权: open_hfq, close_hfq, high_hfq, low_hfq, volume, amount
        不复权: close_raw（仅 close）

    单位：volume = **手**（1手=100股）；amount = 元。
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

    # 判断哪些股票需要更新（以 close 为准；不复权额外对照 peer/hfq）
    close_col = "收盘"
    need = []
    peer_forced = 0
    for code in codes:
        series = data[close_col].get(code)
        last = (
            pd.Timestamp(series.index[-1])
            if series is not None and len(series) > 0
            else None
        )
        behind_calendar = last is None or last < target_end
        peer_ts = peer_last.get(code) if peer_last else None
        behind_peer = (
            peer_ts is not None and (last is None or last < pd.Timestamp(peer_ts))
        )
        if behind_calendar or behind_peer:
            need.append(code)
            if behind_peer and not behind_calendar:
                peer_forced += 1

    if not need:
        label = "后复权" if adjust == "hfq" else "不复权"
        logger.info(f"OHLCV ({label}) 已是最新，跳过下载")
        return {key: pd.DataFrame(data[ak_col]).sort_index()
                for ak_col, (key, _) in paths.items()}

    label = "后复权" if adjust == "hfq" else "不复权"
    extra = f"，其中对照 hfq 强制补齐 {peer_forced} 只" if peer_forced else ""
    logger.info(
        f"OHLCV ({label}): 需下载/更新 {len(need)}/{len(codes)} 只"
        f"（并发={OHLCV_WORKERS}线程{extra}）"
    )

    lock = threading.Lock()
    failed = []
    done_count = 0
    em_fail_streak = 0
    em_skip = False
    EM_SKIP_AFTER = 6

    def _code_to_sina_symbol(code: str) -> str:
        c = str(code).zfill(6)
        if c.startswith(("6", "9")):
            return f"sh{c}"
        if c.startswith(("4", "8")):
            return f"bj{c}"
        return f"sz{c}"

    def _fetch_raw_via_sina(code: str, code_start: str, code_end: str) -> pd.DataFrame | None:
        """东财不可用时：新浪 stock_zh_a_daily(adjust='') 补不复权收盘价。"""
        sym = _code_to_sina_symbol(code)
        sdf = ak.stock_zh_a_daily(
            symbol=sym,
            start_date=code_start,
            end_date=code_end,
            adjust="",
        )
        if sdf is None or sdf.empty:
            return None
        out = pd.DataFrame({
            "日期": pd.to_datetime(sdf["date"]),
            "收盘": pd.to_numeric(sdf["close"], errors="coerce"),
        })
        return out.dropna(subset=["日期"])

    def _fetch_hfq_via_sina_splice(
        code: str, em_last_close: float, em_last_date, end_ymd: str,
    ) -> pd.DataFrame | None:
        """用新浪 hfq 日收益接到东财复权价上，避免混用两套复权基数。

        新浪 volume 为股，东财 volume.parquet 为手，这里 /100。
        """
        if em_last_close is None or not pd.notna(em_last_close) or float(em_last_close) <= 0:
            return None
        sina_start = pd.Timestamp(em_last_date).strftime("%Y%m%d")
        sdf = ak.stock_zh_a_daily(
            symbol=_code_to_sina_symbol(code),
            start_date=sina_start,
            end_date=end_ymd.replace("-", ""),
            adjust="hfq",
        )
        if sdf is None or sdf.empty:
            return None
        sdf = sdf.copy()
        sdf["date"] = pd.to_datetime(sdf["date"])
        sdf = sdf.set_index("date").sort_index()
        overlap = sdf.index[sdf.index <= pd.Timestamp(em_last_date)]
        if len(overlap) == 0:
            return None
        base = sdf.loc[overlap[-1]]
        base_c = float(base["close"])
        if not pd.notna(base_c) or base_c <= 0:
            return None
        scale = float(em_last_close) / base_c
        new = sdf.loc[sdf.index > pd.Timestamp(em_last_date)]
        if new.empty:
            return None
        out = pd.DataFrame({
            "日期": new.index,
            "开盘": pd.to_numeric(new["open"], errors="coerce") * scale,
            "收盘": pd.to_numeric(new["close"], errors="coerce") * scale,
            "最高": pd.to_numeric(new["high"], errors="coerce") * scale,
            "最低": pd.to_numeric(new["low"], errors="coerce") * scale,
            "成交量": pd.to_numeric(new["volume"], errors="coerce") / 100.0,
            "成交额": pd.to_numeric(new["amount"], errors="coerce"),
        })
        return out.dropna(subset=["日期", "收盘"])

    def _fetch_ohlcv(code: str):
        nonlocal em_fail_streak, em_skip
        series = data[close_col].get(code)
        last_dt = (
            series.index[-1]
            if series is not None and len(series) > 0
            else None
        )
        code_start = (
            (last_dt + pd.Timedelta(days=1)).strftime("%Y%m%d")
            if last_dt is not None
            else start.replace("-", "")
        )
        code_end = end.replace("-", "")
        df = None
        err = None
        if not em_skip:
            try:
                df = _fetch_em_kline(code, code_start, code_end, adjust)
            except Exception as e:
                err = e
                df = None
                with lock:
                    em_fail_streak += 1
                    if em_fail_streak >= EM_SKIP_AFTER and not em_skip:
                        em_skip = True
                        logger.warning(
                            f"东财 K 线连续失败 {em_fail_streak} 次，"
                            f"本批改走新浪拼接/不复权回退"
                        )
        if df is None or df.empty:
            if adjust == "" :
                try:
                    df = _fetch_raw_via_sina(code, code_start, code_end)
                    err = None if df is not None and not df.empty else err
                except Exception as e2:
                    return code, None, e2
            elif adjust == "hfq" and series is not None and len(series) > 0:
                try:
                    df = _fetch_hfq_via_sina_splice(
                        code, float(series.iloc[-1]), last_dt, code_end,
                    )
                    err = None if df is not None and not df.empty else err
                except Exception as e3:
                    return code, None, e3
        if df is None or (hasattr(df, "empty") and df.empty):
            return code, None, err or RuntimeError("empty OHLCV")
        time.sleep(0.05)
        return code, df, None

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
                    date_col = "日期" if "日期" in df.columns else "date"
                    df[date_col] = pd.to_datetime(df[date_col])
                    df = df.set_index(date_col)
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

def main(start: str, end: str, sample: int = 0, quality_report: bool = False):
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

    # 不复权收盘价（PB / 自算市值）：对照 hfq 末日强制对齐，避免覆盖塌缩
    logger.info("=== 不复权收盘价（对照 hfq 对齐）===")
    peer_last = _last_valid_by_code(RAW_DIR / "close_hfq.parquet")
    download_ohlcv(codes, start, end, adjust="", peer_last=peer_last)
    # 向后兼容：prices_raw.parquet 已在 download_ohlcv 里保存
    report_raw_hfq_coverage()

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

    # P1-3: 下载完成后生成数据质量报告
    if quality_report:
        logger.info("=== 生成数据质量报告 ===")
        try:
            from data.quality_report import generate_quality_report
            report_path = generate_quality_report(data_dir=str(RAW_DIR))
            if report_path:
                logger.info(f"质量报告: {report_path}")
        except Exception as e:
            logger.warning(f"质量报告生成失败（忽略）: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",  default="2018-01-01")
    parser.add_argument("--end",    default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--quality-report", action="store_true",
                        help="下载完成后生成 data/quality_report_YYYYMMDD.md")
    args = parser.parse_args()
    main(args.start, args.end, args.sample, quality_report=args.quality_report)
