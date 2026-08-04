"""
data/download_stock_value_em.py — 东财日频估值/市值面板（Size 主路径）

接口：``ak.stock_value_em(symbol=<6位代码>)``
  https://data.eastmoney.com/gzfx/detail/{code}.html

返回列（实测 akshare 1.18.x）：
  数据日期 / 当日收盘价 / 当日涨跌幅 /
  总市值 / 流通市值 / 总股本 / 流通股本 /
  PE(TTM) / PE(静) / 市净率 / PEG值 / 市现率 / 市销率

单位（实测）：
  - 总市值 / 流通市值 → **元**（量级 ~1e11–1e12；勿再 ×1e8）
  - 总股本 / 流通股本 → **股**
  - PE / PB 等为比率

产出（wide，index=交易日，columns=code）：
  data/raw/total_mv.parquet       — 总市值（元）★ Size / WLS 主路径
  data/raw/circ_mv.parquet        — 流通市值（元）★ 优先
  data/raw/pe_ttm.parquet         — PE(TTM)（估值 follow-up 用）
  data/raw/pb.parquet             — 市净率（估值 follow-up 用）
  data/raw/total_shares_em.parquet / circ_shares_em.parquet — 股本日频（可选对照）

中间缓存（可 resume）：
  data/raw/_cache/stock_value_em/{code}.parquet — 单票长表
  data/raw/stock_value_em_long.parquet          — 合并长表（组装 wide 前）

特性：
  - 增量：缺票 / max(date) < today-stale_days → 重拉；``--force-refresh`` 全量
  - 中断可 resume（单票缓存 + 定期组装 wide）
  - ``--sample`` / ``--codes`` 调试
  - 不碰 Tushare

用法：
    python -m data.download_stock_value_em
    python -m data.download_stock_value_em --sample 20
    python -m data.download_stock_value_em --codes 600519,600000,000001
    python -m data.download_stock_value_em --refresh-stale-days 5 --force-refresh
    python -m data.download_stock_value_em --start 2018-01-01
    python -m data.download_stock_value_em --assemble-only   # 仅从缓存重装 wide
"""
from __future__ import annotations

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd
from loguru import logger

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import RAW_DIR, UNIVERSE_DIR

WORKERS = 2
SLEEP = 0.12
MAX_RETRY = 2
RETRY_DELAY = 1.0
SAVE_EVERY = 30
DEFAULT_REFRESH_STALE_DAYS = 5

CACHE_DIR = RAW_DIR / "_cache" / "stock_value_em"
LONG_PATH = RAW_DIR / "stock_value_em_long.parquet"

OUT_TOTAL_MV = RAW_DIR / "total_mv.parquet"
OUT_CIRC_MV = RAW_DIR / "circ_mv.parquet"
OUT_PE_TTM = RAW_DIR / "pe_ttm.parquet"
OUT_PB = RAW_DIR / "pb.parquet"
OUT_TOTAL_SHARES = RAW_DIR / "total_shares_em.parquet"
OUT_CIRC_SHARES = RAW_DIR / "circ_shares_em.parquet"

# 长表列（统一英文）
LONG_COLS = [
    "date",
    "code",
    "total_mv",
    "circ_mv",
    "total_shares",
    "circ_shares",
    "pe_ttm",
    "pe_static",
    "pb",
    "peg",
    "pcf",
    "ps",
    "close",
]


def _load_stock_codes(sample: int = 0, codes_csv: str | None = None) -> list[str]:
    if codes_csv:
        codes = [c.strip().zfill(6) for c in codes_csv.split(",") if c.strip()]
        logger.info(f"使用指定代码列表: {len(codes)} 只 → {codes}")
        return codes
    p = UNIVERSE_DIR / "stock_list.parquet"
    if not p.exists():
        raise FileNotFoundError(f"找不到 {p}，请先运行 `python -m data.download` 生成股票列表")
    df = pd.read_parquet(p)
    codes = df["code"].astype(str).str.zfill(6).tolist()
    codes = [c for c in codes if not c.startswith("8")]
    if sample:
        codes = codes[:sample]
        logger.info(f"调试模式：仅取前 {sample} 只 → {codes}")
    logger.info(f"股票代码总数: {len(codes)}")
    return codes


def _map_em_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """按关键词把东财中文列映射为统一英文列。"""
    col_map: dict = {}
    for c in raw.columns:
        cs = str(c)
        if "数据日期" in cs or cs in ("日期", "date"):
            col_map[c] = "date"
        elif "当日收盘价" in cs or cs in ("收盘价", "close"):
            col_map[c] = "close"
        elif "总市值" in cs and "流通" not in cs:
            col_map[c] = "total_mv"
        elif "流通市值" in cs:
            col_map[c] = "circ_mv"
        elif "总股本" in cs and "流通" not in cs:
            col_map[c] = "total_shares"
        elif "流通股本" in cs:
            col_map[c] = "circ_shares"
        elif "PE(TTM)" in cs.upper() or "PE（TTM）" in cs.upper():
            col_map[c] = "pe_ttm"
        elif "PE(静)" in cs or "PE（静）" in cs:
            col_map[c] = "pe_static"
        elif "市净率" in cs:
            col_map[c] = "pb"
        elif "PEG" in cs.upper():
            col_map[c] = "peg"
        elif "市现率" in cs:
            col_map[c] = "pcf"
        elif "市销率" in cs:
            col_map[c] = "ps"
    if "date" not in col_map.values():
        raise ValueError(f"stock_value_em 无日期列: {list(raw.columns)}")
    df = raw.rename(columns=col_map)
    keep = [c for c in LONG_COLS if c in df.columns and c != "code"]
    df = df[keep].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in df.columns:
        if c == "date":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # 市值单位自检：中位数若落在「亿元」量级则 ×1e8（接口偶发改口径）
    for c in ("total_mv", "circ_mv"):
        if c not in df.columns:
            continue
        med = float(df[c].median()) if df[c].notna().any() else float("nan")
        if med == med and 0 < med < 1e6:
            logger.warning(f"市值列 {c} 中位数={med:.4g} 疑似亿元口径 → ×1e8 换算为元")
            df[c] = df[c] * 1e8
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def _cache_path(code: str) -> Path:
    return CACHE_DIR / f"{code}.parquet"


def _fetch_one(code: str) -> tuple[str, pd.DataFrame | None]:
    for attempt in range(MAX_RETRY + 1):
        try:
            raw = ak.stock_value_em(symbol=code)
            time.sleep(SLEEP)
            if raw is None or raw.empty:
                return code, None
            df = _map_em_columns(raw)
            df["code"] = code
            df = df[[c for c in LONG_COLS if c in df.columns]]
            return code, df
        except Exception as e:
            if attempt < MAX_RETRY:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                logger.warning(f"stock_value_em 失败 {code}（重试{MAX_RETRY}次）: {e}")
                return code, None


def _codes_needing_refresh(
    codes: list[str],
    refresh_stale_days: int,
    force_refresh: bool,
) -> tuple[list[str], set[str]]:
    """返回 (need, fresh_codes)。基于单票缓存末日判断。"""
    if force_refresh:
        logger.info(f"force_refresh → 全量下载 {len(codes)} 只")
        return list(codes), set()

    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=int(refresh_stale_days))
    need: list[str] = []
    fresh: set[str] = set()
    n_missing = 0
    n_stale = 0
    for code in codes:
        p = _cache_path(code)
        if not p.exists():
            need.append(code)
            n_missing += 1
            continue
        try:
            # 只读 date 列末值，避免整表进内存
            cached = pd.read_parquet(p, columns=["date"])
            last = pd.to_datetime(cached["date"], errors="coerce").max()
            if pd.isna(last) or last < cutoff:
                need.append(code)
                n_stale += 1
            else:
                fresh.add(code)
        except Exception:
            need.append(code)
            n_missing += 1
    logger.info(
        f"stock_value_em 增量: 跳过新鲜 {len(fresh)} 只 "
        f"(date≥{cutoff.date()}, stale_days={refresh_stale_days})；"
        f"需下载 {len(need)} 只（缺 {n_missing} + 过期 {n_stale}）"
    )
    return need, fresh


def _write_cache(code: str, df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_cache_path(code), index=False)


def _load_all_cached(codes: list[str] | None = None) -> pd.DataFrame:
    """从单票缓存拼成长表。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(CACHE_DIR.glob("*.parquet"))
    if codes is not None:
        want = {c.zfill(6) for c in codes}
        files = [f for f in files if f.stem in want]
    if not files:
        return pd.DataFrame(columns=LONG_COLS)
    parts: list[pd.DataFrame] = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            if "code" not in df.columns:
                df["code"] = f.stem
            df["code"] = df["code"].astype(str).str.zfill(6)
            parts.append(df)
        except Exception as e:
            logger.warning(f"读缓存失败 {f.name}: {e}")
    if not parts:
        return pd.DataFrame(columns=LONG_COLS)
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date", "code"])
    out = out.drop_duplicates(subset=["code", "date"], keep="last")
    out = out.sort_values(["code", "date"]).reset_index(drop=True)
    return out


def _pivot_wide(long_df: pd.DataFrame, value_col: str, start: str | None) -> pd.DataFrame:
    if long_df.empty or value_col not in long_df.columns:
        return pd.DataFrame()
    sub = long_df[["date", "code", value_col]].dropna(subset=[value_col])
    if start is not None:
        sub = sub[sub["date"] >= pd.Timestamp(start)]
    if sub.empty:
        return pd.DataFrame()
    wide = sub.pivot_table(index="date", columns="code", values=value_col, aggfunc="last")
    wide.index = pd.DatetimeIndex(pd.to_datetime(wide.index))
    wide = wide.sort_index()
    wide.columns = wide.columns.astype(str).str.zfill(6)
    wide.index.name = "date"
    return wide


def _merge_wide_preserve(existing_path: Path, new_wide: pd.DataFrame) -> pd.DataFrame:
    """把新东财列合并进已有 wide，避免局部下载截断全市场列。"""
    if new_wide is None or new_wide.empty:
        if existing_path.exists():
            old = pd.read_parquet(existing_path)
            old.index = pd.to_datetime(old.index)
            old.columns = old.columns.astype(str).str.zfill(6)
            return old
        return new_wide
    new_wide = new_wide.copy()
    new_wide.columns = new_wide.columns.astype(str).str.zfill(6)
    if not existing_path.exists():
        return new_wide
    try:
        old = pd.read_parquet(existing_path)
        old.index = pd.to_datetime(old.index)
        old.columns = old.columns.astype(str).str.zfill(6)
    except Exception as e:
        logger.warning(f"读取已有 {existing_path.name} 失败，将整表替换: {e}")
        return new_wide
    if old.empty:
        return new_wide
    # 未在本轮东财缓存中的旧列保留；重叠列用新值覆盖
    keep_cols = [c for c in old.columns if c not in new_wide.columns]
    if keep_cols:
        logger.warning(
            f"{existing_path.name}: 保留尚未被东财缓存覆盖的 {len(keep_cols)} 列"
            f"（本轮仅更新 {new_wide.shape[1]} 列）。请尽快跑全市场 "
            f"`download_stock_value_em` 以统一口径。"
        )
        union_idx = old.index.union(new_wide.index).sort_values()
        merged = old.reindex(union_idx)
        for c in new_wide.columns:
            merged[c] = new_wide[c].reindex(union_idx)
        return merged
    return new_wide


def assemble_wide_panels(
    codes: list[str] | None = None,
    start: str | None = "2018-01-01",
) -> dict[str, pd.DataFrame]:
    """从单票缓存组装 wide 面板并落盘。

    始终读取 ``_cache/stock_value_em/`` 下**全部**已缓存票，避免
    ``--sample`` / ``--codes`` 局部下载时把全市场 wide 面板截断成子集。
    若磁盘上已有更宽的主面板，未覆盖列会被保留并 warning。
    ``codes`` 仅用于日志提示。
    """
    long_df = _load_all_cached(None)
    if long_df.empty:
        logger.warning("无缓存可组装")
        return {}
    if codes is not None:
        have = set(long_df["code"].astype(str).str.zfill(6))
        miss = [c for c in codes if c not in have]
        logger.info(
            f"组装 wide：缓存共 {len(have)} 只；本次目标 {len(codes)} 只"
            + (f"；其中尚未缓存 {len(miss)} 只" if miss else "")
        )
    LONG_PATH.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_parquet(LONG_PATH, index=False)
    logger.info(
        f"stock_value_em_long: {long_df.shape}  "
        f"覆盖 {long_df['code'].nunique()} 只  "
        f"{long_df['date'].min().date()} → {long_df['date'].max().date()}"
    )

    mapping = {
        "total_mv": OUT_TOTAL_MV,
        "circ_mv": OUT_CIRC_MV,
        "pe_ttm": OUT_PE_TTM,
        "pb": OUT_PB,
        "total_shares": OUT_TOTAL_SHARES,
        "circ_shares": OUT_CIRC_SHARES,
    }
    out: dict[str, pd.DataFrame] = {}
    for col, path in mapping.items():
        wide = _pivot_wide(long_df, col, start=start)
        if wide.empty:
            logger.warning(f"{col}: 空面板，跳过写出")
            continue
        # 清洗：市值负/0 → NaN
        if col in ("total_mv", "circ_mv", "total_shares", "circ_shares"):
            wide = wide.replace([np.inf, -np.inf], np.nan)
            wide = wide.where(wide > 0)
        else:
            wide = wide.replace([np.inf, -np.inf], np.nan)
        # 主市值面板：与已有宽表合并，防止 sample 截断
        if col in ("total_mv", "circ_mv"):
            wide = _merge_wide_preserve(path, wide)
        wide.to_parquet(path)
        out[col] = wide
        logger.info(
            f"写出 {path.name}: shape={wide.shape}  "
            f"{wide.index.min().date()} → {wide.index.max().date()}"
        )

    # 茅台量级自检
    if "total_mv" in out and "600519" in out["total_mv"].columns:
        s = out["total_mv"]["600519"].dropna()
        if not s.empty:
            last = float(s.iloc[-1])
            logger.info(
                f"[量级自检] 600519 total_mv 最新={last:.3e} 元 "
                f"({last / 1e8:.1f} 亿元 = {last / 1e12:.3f} 万亿)"
            )
    return out


def download_stock_value_em(
    codes: list[str],
    start: str | None = "2018-01-01",
    refresh_stale_days: int = DEFAULT_REFRESH_STALE_DAYS,
    force_refresh: bool = False,
    save_every: int = SAVE_EVERY,
    assemble: bool = True,
    workers: int = WORKERS,
) -> dict[str, pd.DataFrame]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    need, _fresh = _codes_needing_refresh(codes, refresh_stale_days, force_refresh)
    n_workers = max(1, int(workers))

    if need:
        logger.info(f"需下载 {len(need)}/{len(codes)} 只（并发={n_workers}，sleep={SLEEP}s）")
        lock = threading.Lock()
        success = 0
        failed: list[str] = []
        done = 0

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_fetch_one, c): c for c in need}
            for fut in as_completed(futures):
                code, df = fut.result()
                with lock:
                    done += 1
                    if df is not None and not df.empty:
                        _write_cache(code, df)
                        success += 1
                    else:
                        failed.append(code)
                    if done % save_every == 0:
                        logger.info(
                            f"进度 {done}/{len(need)}  成功={success} 失败={len(failed)}"
                        )
                        if assemble:
                            assemble_wide_panels(codes=codes, start=start)

        if failed:
            logger.warning(f"最终失败 {len(failed)} 只: {failed[:30]}")
        logger.info(f"本轮下载成功 {success}/{len(need)}")
    else:
        logger.info("全部股票 stock_value_em 仍新鲜，跳过下载")

    if assemble:
        return assemble_wide_panels(codes=codes, start=start)
    return {}


def main():
    parser = argparse.ArgumentParser(
        description="东财 stock_value_em → 日频市值/估值面板（Size 主路径）"
    )
    parser.add_argument("--start", default="2018-01-01", help="wide 面板日期下界")
    parser.add_argument("--sample", type=int, default=0, help="仅取 universe 前 N 只")
    parser.add_argument("--codes", default=None, help="逗号分隔代码，优先于 --sample")
    parser.add_argument(
        "--refresh-stale-days",
        type=int,
        default=DEFAULT_REFRESH_STALE_DAYS,
        help="单票缓存末日早于 today-N 则天则重拉（默认 5）",
    )
    parser.add_argument("--force-refresh", action="store_true", help="忽略新鲜度全量重拉")
    parser.add_argument(
        "--assemble-only",
        action="store_true",
        help="不下载，仅从 _cache/stock_value_em 重装 wide 面板",
    )
    parser.add_argument("--workers", type=int, default=WORKERS, help="并发线程数")
    args = parser.parse_args()

    codes = _load_stock_codes(sample=args.sample, codes_csv=args.codes)
    if args.assemble_only:
        assemble_wide_panels(codes=codes, start=args.start)
        return

    download_stock_value_em(
        codes,
        start=args.start,
        refresh_stale_days=args.refresh_stale_days,
        force_refresh=args.force_refresh,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
