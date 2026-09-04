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
  - 中断可 resume（单票缓存）。wide 只在**整轮下载结束后**原子替换，
    禁止把未完成批次 assemble 进官方 parquet（否则末日行只剩先跑到的 600xxx）
  - ``--sample`` / ``--codes`` 调试
  - 不碰 Tushare

用法：
    python -m data.download_stock_value_em
    python -m data.download_stock_value_em --sample 20
    python -m data.download_stock_value_em --codes 600519,600000,000001
    python -m data.download_stock_value_em --refresh-stale-days 5 --force-refresh
    python -m data.download_stock_value_em --start 2018-01-01
    python -m data.download_stock_value_em --assemble-only   # 仅从缓存重装 wide
    python -m data.download_stock_value_em --bj-only --workers 1  # 只补北交所 92xxxx 缺失列
"""
from __future__ import annotations

import argparse
import shutil
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


def _load_stock_codes(
    sample: int = 0,
    codes_csv: str | None = None,
    *,
    bj_only: bool = False,
) -> list[str]:
    from data.download import collect_bj_stock_codes, is_excluded_universe_code

    if codes_csv:
        codes = [c.strip().zfill(6) for c in codes_csv.split(",") if c.strip()]
        logger.info(f"使用指定代码列表: {len(codes)} 只 → {codes}")
        return codes

    if bj_only:
        codes = collect_bj_stock_codes()
        if sample:
            codes = codes[:sample]
            logger.info(f"调试模式：北交所仅取前 {sample} 只 → {codes}")
        logger.info(f"北交所 92xxxx 下载名单: {len(codes)} 只")
        return codes

    p = UNIVERSE_DIR / "stock_list.parquet"
    if not p.exists():
        raise FileNotFoundError(f"找不到 {p}，请先运行 `python -m data.download` 生成股票列表")
    df = pd.read_parquet(p)
    codes = df["code"].astype(str).str.zfill(6).tolist()
    codes = [c for c in codes if not is_excluded_universe_code(c)]
    # stock_list 历史可能缺 92：从 prices_hfq / BJ 接口补入，不依赖整表重写名单
    have = set(codes)
    extra = [c for c in collect_bj_stock_codes(use_api=False) if c not in have]
    if extra:
        logger.info(f"stock_list 未收录北交所 {len(extra)} 只，已并入下载名单")
        codes = list(codes) + extra
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


def _fetch_one(code: str, sleep: float | None = None) -> tuple[str, pd.DataFrame | None]:
    pause = SLEEP if sleep is None else float(sleep)
    for attempt in range(MAX_RETRY + 1):
        try:
            raw = ak.stock_value_em(symbol=code)
            time.sleep(pause)
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


def _backup_parquet(path: Path) -> None:
    """写前 ``*.parquet.bak``，避免窗口切片覆盖后无法回滚。"""
    if not path.exists():
        return
    bak = path.with_suffix(path.suffix + ".bak")
    try:
        shutil.copy2(path, bak)
    except OSError as e:
        logger.warning(f".bak 备份失败（继续）: {path.name} -> {e}")


def _atomic_to_parquet(df: pd.DataFrame, path: Path) -> None:
    """先写 ``*.tmp`` 再 replace，避免 kill 半截把官方 parquet 写坏。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp)
    tmp.replace(path)


def _drop_sparse_trailing_dates(
    wide: pd.DataFrame,
    *,
    min_frac: float = 0.5,
    lookback: int = 40,
) -> pd.DataFrame:
    """丢掉末尾覆盖过低的新日期（未完成批次 assemble 的典型症状）。

    用「去掉最近 10 日」后的 lookback 窗口中位数作典型覆盖；从末日往回
    连续低于 ``typical * min_frac`` 的日期一律不写入官方面板。
    历史中段低覆盖（次新上市爬升）不动。
    """
    if wide is None or wide.empty or len(wide.index) < 10:
        return wide
    cov = wide.notna().sum(axis=1)
    body = cov.iloc[:-10] if len(cov) > 20 else cov
    window = body.iloc[-int(lookback) :] if len(body) else cov
    typical = float(window.median()) if len(window) else float(cov.median())
    if not np.isfinite(typical) or typical <= 0:
        return wide
    thresh = typical * float(min_frac)
    drop: list = []
    for d in reversed(list(cov.index)):
        if float(cov.loc[d]) < thresh:
            drop.append(d)
        else:
            break
    if not drop:
        return wide
    drop_sorted = sorted(drop)
    logger.warning(
        f"丢弃覆盖过低的末尾 {len(drop_sorted)} 日"
        f"（typical={typical:.0f}, thresh={thresh:.0f}）: "
        f"{pd.Timestamp(drop_sorted[0]).date()} → {pd.Timestamp(drop_sorted[-1]).date()} "
        f"cov={int(cov.loc[drop_sorted[0]])}..{int(cov.loc[drop_sorted[-1]])}。"
        f"未完成批次不写进官方 parquet"
    )
    return wide.drop(index=drop_sorted)


def _norm_wide(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index))
    out.columns = out.columns.astype(str).str.zfill(6)
    out.index.name = "date"
    return out


def _merge_wide_preserve(
    existing_path: Path,
    new_wide: pd.DataFrame,
    *,
    fill_missing_cols_only: bool = False,
) -> pd.DataFrame:
    """新 wide 与已有面板按 index/columns outer 合并。

    默认：重叠格新值优先（全市场增量）。
    ``fill_missing_cols_only=True``：只追加旧面板没有的列，已有沪深列格点完全不动
    （北交所 92xxxx 补洞用；禁止用缓存重写主链）。

    禁止用 start~end 窗口切片整表覆盖全历史。列完全重叠时也必须保留
    旧面板中窗口外的日期（live ``lookback`` 曾因此把市值砍成最近一个月）。
    """
    if new_wide is None or new_wide.empty:
        if existing_path.exists():
            return _norm_wide(pd.read_parquet(existing_path))
        return new_wide if new_wide is not None else pd.DataFrame()

    new_wide = _norm_wide(new_wide)
    if not existing_path.exists():
        return new_wide
    try:
        old = _norm_wide(pd.read_parquet(existing_path))
    except Exception as e:
        logger.warning(f"读取已有 {existing_path.name} 失败，将整表替换: {e}")
        return new_wide
    if old.empty:
        return new_wide

    if fill_missing_cols_only:
        add_cols = [c for c in new_wide.columns if c not in old.columns]
        skipped = [c for c in new_wide.columns if c in old.columns]
        if skipped:
            logger.info(
                f"{existing_path.name}: 跳过已有 {len(skipped)} 列（沪深主链不动），"
                f"仅补 {len(add_cols)} 列"
            )
        if not add_cols:
            return old
        add = new_wide.loc[:, add_cols]
        extra_idx = add.index.difference(old.index)
        if len(extra_idx) > 0:
            logger.info(
                f"{existing_path.name}: 新列带来 {len(extra_idx)} 个新交易日 "
                f"{extra_idx.min().date()} → {extra_idx.max().date()}"
            )
        logger.info(f"{existing_path.name}: 追加缺失列 {len(add_cols)} 只")
        merged = old.join(add, how="outer")
        merged.index.name = "date"
        return merged.sort_index()

    extra_cols = [c for c in old.columns if c not in new_wide.columns]
    if extra_cols:
        logger.warning(
            f"{existing_path.name}: 保留尚未被东财缓存覆盖的 {len(extra_cols)} 列"
            f"（本轮仅更新 {new_wide.shape[1]} 列）。请尽快跑全市场 "
            f"`download_stock_value_em` 以统一口径。"
        )
    extra_idx = old.index.difference(new_wide.index)
    if len(extra_idx) > 0:
        logger.info(
            f"{existing_path.name}: outer 合并保留历史 "
            f"{extra_idx.min().date()} → {extra_idx.max().date()} "
            f"共 {len(extra_idx)} 日（本轮新窗口 {len(new_wide.index)} 日）"
        )

    # 新非 NA 覆盖旧值；窗口外日期与未覆盖列由旧面板补上
    return new_wide.combine_first(old).sort_index()


def assemble_wide_panels(
    codes: list[str] | None = None,
    start: str | None = "2018-01-01",
    *,
    fill_missing_cols_only: bool = False,
) -> dict[str, pd.DataFrame]:
    """从单票缓存组装 wide 面板并落盘。

    默认读取 ``_cache/stock_value_em/`` 下**全部**已缓存票，避免
    ``--sample`` / ``--codes`` 局部下载时把全市场 wide 面板截断成子集。
    写出前与磁盘已有面板按 index/columns outer 合并（新值优先），
    避免 ``start`` 窗口切片覆盖全历史。``codes`` 仅用于日志提示。

    ``fill_missing_cols_only=True``（北交所补洞）：只从缓存读取 ``codes``，
    且只把旧面板**没有的列** outer join 进去，沪深已有列格点保持不动。
    """
    load_codes = codes if fill_missing_cols_only else None
    long_df = _load_all_cached(load_codes)
    if long_df.empty:
        logger.warning("无缓存可组装")
        return {}
    if codes is not None:
        have = set(long_df["code"].astype(str).str.zfill(6))
        miss = [c for c in codes if c not in have]
        logger.info(
            f"组装 wide：缓存共 {len(have)} 只；本次目标 {len(codes)} 只"
            + (f"；其中尚未缓存 {len(miss)} 只" if miss else "")
            + ("；仅补缺失列" if fill_missing_cols_only else "")
        )
    LONG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not fill_missing_cols_only:
        long_df.to_parquet(LONG_PATH, index=False)
        logger.info(
            f"stock_value_em_long: {long_df.shape}  "
            f"覆盖 {long_df['code'].nunique()} 只  "
            f"{long_df['date'].min().date()} → {long_df['date'].max().date()}"
        )
    else:
        logger.info(
            f"北交所补洞长表: {long_df.shape}  "
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
        # 北交所补洞：只 outer join 缺失列。全市场增量仍走新值优先 merge。
        wide = _merge_wide_preserve(
            path, wide, fill_missing_cols_only=fill_missing_cols_only
        )
        if not fill_missing_cols_only:
            wide = _drop_sparse_trailing_dates(wide)
        if wide.empty:
            logger.warning(f"{col}: 丢弃稀疏末日后为空，跳过写出")
            continue
        _backup_parquet(path)
        _atomic_to_parquet(wide, path)
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
    *,
    fill_missing_cols_only: bool = False,
    sleep: float | None = None,
) -> dict[str, pd.DataFrame]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    need, _fresh = _codes_needing_refresh(codes, refresh_stale_days, force_refresh)
    n_workers = max(1, int(workers))
    pause = SLEEP if sleep is None else float(sleep)

    if need:
        logger.info(
            f"需下载 {len(need)}/{len(codes)} 只（并发={n_workers}，sleep={pause}s"
            f"{'，仅补缺失列' if fill_missing_cols_only else ''}）"
        )
        lock = threading.Lock()
        success = 0
        failed: list[str] = []
        done = 0

        def _fetch_paused(code: str) -> tuple[str, pd.DataFrame | None]:
            return _fetch_one(code, sleep=pause)

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_fetch_paused, c): c for c in need}
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
                            f"（单票缓存已落盘；wide 等整轮结束后再 assemble）"
                        )

        if failed:
            logger.warning(f"最终失败 {len(failed)} 只: {failed[:30]}")
        logger.info(f"本轮下载成功 {success}/{len(need)}")
    else:
        logger.info("全部股票 stock_value_em 仍新鲜，跳过下载")

    if assemble:
        return assemble_wide_panels(
            codes=codes,
            start=start,
            fill_missing_cols_only=fill_missing_cols_only,
        )
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
    parser.add_argument(
        "--bj-only",
        action="store_true",
        help="只下载 92xxxx 北交所，并仅把缺失列写入现有 circ_mv/total_mv/pe_ttm/pb",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=None,
        help="单票请求间隔秒（默认 0.12；--bj-only 建议 0.25）",
    )
    args = parser.parse_args()

    fill_missing = bool(args.bj_only)
    bj_codes: list[str] | None = None
    if args.bj_only:
        from data.download import append_bj_codes_to_stock_list, collect_bj_stock_codes

        bj_codes = collect_bj_stock_codes(use_api=True)
        n_appended = append_bj_codes_to_stock_list(bj_codes)
        logger.info(f"stock_list 追加北交所 {n_appended} 只（未改沪深/B 股历史行）")

    codes = bj_codes if bj_codes is not None else _load_stock_codes(
        sample=args.sample, codes_csv=args.codes, bj_only=False
    )
    if args.bj_only and args.sample:
        codes = codes[: args.sample]
        logger.info(f"调试模式：北交所仅取前 {args.sample} 只")
    if args.bj_only:
        logger.info(f"北交所 92xxxx 下载名单: {len(codes)} 只")
    if args.assemble_only:
        assemble_wide_panels(
            codes=codes,
            start=args.start,
            fill_missing_cols_only=fill_missing,
        )
        return

    sleep = args.sleep
    if sleep is None and args.bj_only:
        sleep = 0.25
    workers = 1 if args.bj_only and args.workers == WORKERS else args.workers
    save_every = 500 if args.bj_only else SAVE_EVERY

    download_stock_value_em(
        codes,
        start=args.start,
        refresh_stale_days=args.refresh_stale_days,
        force_refresh=args.force_refresh,
        workers=workers,
        fill_missing_cols_only=fill_missing,
        sleep=sleep,
        save_every=save_every,
    )


if __name__ == "__main__":
    main()
