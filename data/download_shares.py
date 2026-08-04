"""
data/download_shares.py — 下载全市场股本变动历史（巨潮 cninfo 接口）

接口：ak.stock_share_change_cninfo(symbol=<6位代码>)
  每只股票返回一张股本变动长表（含 44 列细分股东结构），关键字段：
    证券代码 / 公告日期（披露日，PIT 安全用这个）/ 变动日期（生效日）
    总股本 / 已流通股份（=流通股本）/ 变动原因

单位：巨潮返回的股本数字单位为「万股」，本模块统一换算成「股」（×1e4）。
  交叉验证：
    - 600519 茅台 总股本=125619.78 万股 ×1e4 = 1.256e9 股 ≈ 1.256 亿股 ✓
      （茅台 2024 年总股本 ~1.256 亿股，总市值 ~1.5 万亿）
    - 600000 浦发 总股本=2.935e6 万股 ×1e4 = 2.935e10 股 ≈ 293.5 亿股 ✓
      （浦发银行历史总股本 ~293 亿股）

输出：data/raw/share_change.parquet（长表）
  列：code, announce_date, change_date, total_shares, circ_shares, change_reason
  - total_shares / circ_shares 单位均为「股」
  - announce_date = 公告日期（PIT 披露日，下游 ffill 起点）
  - change_date  = 变动日期（生效日，仅作记录）

特性：
  - **增量刷新**（非按 code 永久跳过）：
      * 缺记录的 code → 下载
      * 已有但 max(announce_date) 早于 ``today - refresh_stale_days`` → 重拉全历史并替换
      * ``--force-refresh`` → 全市场重拉
  - 失败重试 max_retry=2，失败 code 收集
  - 4 线程并发 + sleep 0.1（巨潮限流比东财宽松，仍保守）
  - --start 年份过滤：仅保留 announce_date >= start 的记录，
    但**保留每只股票最早一条 < start 的记录**作为 ffill 起点
    （否则早期市值会全 NaN）

用法：
    python -m data.download_shares                       # 增量刷新
    python -m data.download_shares --refresh-stale-days 30
    python -m data.download_shares --force-refresh       # 全量重拉
    python -m data.download_shares --sample 100          # 调试：前 100 只
    python -m data.download_shares --codes 600519,600000 # 指定代码
    python -m data.download_shares --start 2015-01-01    # 仅 2015 起（保留 ffill 起点）
"""
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

WORKERS = 2  # cninfo + mini_racer：过高并发易崩；1–2 稳妥
SLEEP = 0.15
MAX_RETRY = 2
RETRY_DELAY = 1.0
SAVE_EVERY = 50
DEFAULT_REFRESH_STALE_DAYS = 30

# 巨潮返回列名 → 目标列名
COL_MAP = {
    "证券代码": "code",
    "公告日期": "announce_date",
    "变动日期": "change_date",
    "总股本":   "total_shares",
    "已流通股份": "circ_shares",
    "变动原因": "change_reason",
}
# 万股 → 股
SHARES_UNIT_MUL = 1e4

OUT_PATH = RAW_DIR / "share_change.parquet"


def _load_stock_codes(sample: int = 0, codes_csv: str | None = None) -> list[str]:
    """从 universe/stock_list.parquet 读取股票代码列表（zfill 6 位）。"""
    if codes_csv:
        codes = [c.strip().zfill(6) for c in codes_csv.split(",") if c.strip()]
        logger.info(f"使用指定代码列表: {len(codes)} 只 → {codes}")
        return codes
    p = UNIVERSE_DIR / "stock_list.parquet"
    if not p.exists():
        raise FileNotFoundError(f"找不到 {p}，请先运行 `python -m data.download` 生成股票列表")
    df = pd.read_parquet(p)
    codes = df["code"].astype(str).str.zfill(6).tolist()
    # 剔除北交所（8 开头）
    codes = [c for c in codes if not c.startswith("8")]
    if sample:
        codes = codes[:sample]
        logger.info(f"调试模式：仅取前 {sample} 只 → {codes}")
    logger.info(f"股票代码总数: {len(codes)}")
    return codes


def _clean_one(raw: pd.DataFrame, code: str) -> pd.DataFrame | None:
    """单只股票的原始 44 列表 → 6 列清洗长表（单位换算成「股」）。"""
    if raw is None or raw.empty:
        return None
    # 仅保留存在的目标列
    keep = {k: v for k, v in COL_MAP.items() if k in raw.columns}
    keep_vals = set(keep.values())
    if "code" not in keep_vals or "announce_date" not in keep_vals:
        logger.debug(f"{code}: 缺 code/announce_date 列，跳过")
        return None
    df = raw[list(keep.keys())].rename(columns=keep).copy()

    # code 兜底（接口返回的 证券代码 可能空，用入参补）
    df["code"] = df["code"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
    empty_mask = (df["code"].isin(["", "nan", "None"])) | df["code"].isna()
    df.loc[empty_mask, "code"] = code
    df["code"] = df["code"].astype(str).str.zfill(6)

    # 日期解析
    df["announce_date"] = pd.to_datetime(df["announce_date"], errors="coerce")
    df["change_date"] = pd.to_datetime(df["change_date"], errors="coerce")

    # 数值列：万股 → 股
    for col in ("total_shares", "circ_shares"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") * SHARES_UNIT_MUL
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            # 负值/0 股本置 NaN
            df[col] = df[col].where(df[col] > 0)

    # 变动原因兜底
    if "change_reason" in df.columns:
        df["change_reason"] = df["change_reason"].astype(str).replace("nan", "")
    else:
        df["change_reason"] = ""

    # 丢弃 announce_date 为 NaT 的记录（无法做 PIT 对齐）
    df = df.dropna(subset=["announce_date"])
    if df.empty:
        return None

    # 列顺序
    cols_order = ["code", "announce_date", "change_date",
                  "total_shares", "circ_shares", "change_reason"]
    df = df[[c for c in cols_order if c in df.columns]]
    return df


def _fetch_one(code: str) -> tuple[str, pd.DataFrame | None]:
    for attempt in range(MAX_RETRY + 1):
        try:
            raw = ak.stock_share_change_cninfo(symbol=code)
            time.sleep(SLEEP)
            return code, _clean_one(raw, code)
        except Exception as e:
            if attempt < MAX_RETRY:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                logger.warning(f"股本变动下载失败 {code}（重试{MAX_RETRY}次）: {e}")
                return code, None


def _save(records: list[pd.DataFrame], out_path: Path) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.concat(records, ignore_index=True)
    df = df.drop_duplicates(subset=["code", "announce_date"], keep="last")
    df = df.sort_values(["code", "announce_date"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    return df


def _apply_start_filter(df: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    """仅保留 announce_date >= start 的记录，但**保留每只股票最早一条 < start 的记录**
    作为 ffill 起点（否则 --start 2018 之后早期市值会全 NaN）。
    """
    if df.empty:
        return df
    keep_mask = df["announce_date"] >= start
    # 每只股票最早一条（无论是否 >= start）
    earliest_idx = df.groupby("code")["announce_date"].idxmin()
    keep_mask = keep_mask | df.index.isin(earliest_idx)
    return df[keep_mask].reset_index(drop=True)


def _codes_needing_refresh(
    codes: list[str],
    existing: pd.DataFrame | None,
    refresh_stale_days: int,
    force_refresh: bool,
) -> tuple[list[str], set[str], pd.DataFrame]:
    """返回 (need, keep_codes, existing_kept)。

    - force_refresh：全部重拉，existing_kept 空
    - 否则：缺 code 或 max(announce_date) < cutoff → need；其余 keep
    """
    if existing is None or existing.empty or force_refresh:
        reason = "force_refresh" if force_refresh else "无已有文件"
        logger.info(f"股本刷新策略: {reason} → 全量下载 {len(codes)} 只")
        return list(codes), set(), pd.DataFrame()

    ex = existing.copy()
    ex["code"] = ex["code"].astype(str).str.zfill(6)
    ex["announce_date"] = pd.to_datetime(ex["announce_date"], errors="coerce")
    last_ann = ex.groupby("code")["announce_date"].max()
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=int(refresh_stale_days))
    fresh = set(last_ann[last_ann >= cutoff].index.astype(str).str.zfill(6))
    have = set(last_ann.index.astype(str).str.zfill(6))
    need = [c for c in codes if c not in fresh]
    keep_codes = {c for c in codes if c in fresh}
    existing_kept = ex[ex["code"].isin(keep_codes)].copy()
    n_missing = sum(1 for c in codes if c not in have)
    n_stale = sum(1 for c in codes if c in have and c not in fresh)
    logger.info(
        f"股本增量刷新: 跳过新鲜 {len(keep_codes)} 只 "
        f"(announce≥{cutoff.date()}, stale_days={refresh_stale_days})；"
        f"需下载 {len(need)} 只（缺 {n_missing} + 过期 {n_stale}）"
    )
    return need, keep_codes, existing_kept


def download_shares(
    codes: list[str],
    start: str = "2015-01-01",
    out_path: Path = OUT_PATH,
    save_every: int = SAVE_EVERY,
    refresh_stale_days: int = DEFAULT_REFRESH_STALE_DAYS,
    force_refresh: bool = False,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing: pd.DataFrame | None = None
    if out_path.exists() and not force_refresh:
        try:
            existing = pd.read_parquet(out_path)
            if existing is not None and not existing.empty:
                logger.info(
                    f"已有 share_change: {existing.shape}  "
                    f"覆盖 {existing['code'].astype(str).str.zfill(6).nunique()} 只"
                )
        except Exception as e:
            logger.warning(f"读取已有 {out_path} 失败（将重下）: {e}")
            existing = None

    need, keep_codes, existing_kept = _codes_needing_refresh(
        codes, existing, refresh_stale_days, force_refresh,
    )
    existing_records: list[pd.DataFrame] = []
    if existing_kept is not None and not existing_kept.empty:
        existing_records.append(existing_kept)

    # 全量/过期刷新时 existing_kept 可能为空；中间落盘必须保留「尚未刷新」旧记录，
    # 否则崩溃/中断会把 share_change.parquet 截成只含已下完的子集。
    stale_existing = pd.DataFrame()
    if existing is not None and not existing.empty and need:
        ex = existing.copy()
        ex["code"] = ex["code"].astype(str).str.zfill(6)
        # 待刷新 code 的旧行：成功刷新后剔除；失败/未完成时兜底保留
        stale_existing = ex[ex["code"].isin(need)].copy()

    # 开始刷新前备份，防止中断丢库
    if out_path.exists() and need:
        bak = out_path.with_suffix(".parquet.bak")
        try:
            import shutil
            shutil.copy2(out_path, bak)
            logger.info(f"已备份 share_change → {bak.name}")
        except Exception as e:
            logger.warning(f"备份 share_change 失败（继续）: {e}")

    if not need:
        logger.info("全部股票股本数据仍新鲜，仅做 --start 过滤后写回")
        merged = (
            pd.concat(existing_records, ignore_index=True)
            if existing_records else pd.DataFrame()
        )
        if not merged.empty:
            merged = _apply_start_filter(merged, start_ts)
            merged.to_parquet(out_path)
        return merged

    logger.info(f"需下载 {len(need)}/{len(codes)} 只（并发={WORKERS} 线程，sleep={SLEEP}s）")

    fresh_records: list[pd.DataFrame] = []
    success = 0
    refreshed_codes: set[str] = set()

    def _persist(extra_failed: list[str] | None = None) -> None:
        """中间/最终落盘：已刷新 + 未刷新旧行 + 失败旧行。"""
        parts: list[pd.DataFrame] = list(existing_records) + list(fresh_records)
        pending_old = stale_existing
        if not pending_old.empty and refreshed_codes:
            pending_old = pending_old[~pending_old["code"].isin(refreshed_codes)]
        fail_set = set(extra_failed or [])
        # 已成功刷新的不再保留旧行；失败的保留旧行
        if not pending_old.empty:
            # 从 pending 去掉已刷新；失败码保留在 pending 里
            parts.append(pending_old)
        if parts:
            _save(parts, out_path)

    # 预热 py_mini_racer V8 isolate：cninfo 接口用 JS 加密，首次调用会初始化
    # ConfigurablePool；多线程并发初始化会触发
    # `Check failed: !IsConfigurablePoolInitialized()` 致命崩溃。
    # 在主线程串行跑一只，确保 isolate pool 已初始化后再开线程池。
    try:
        logger.info("预热 mini_racer isolate（cninfo JS 加密）...")
        _warm_code = need[0]
        _wraw = ak.stock_share_change_cninfo(symbol=_warm_code)
        _wdf = _clean_one(_wraw, _warm_code)
        if _wdf is not None and not _wdf.empty:
            fresh_records.append(_wdf)
            refreshed_codes.add(_warm_code)
            success += 1
        logger.info(f"预热完成：{_warm_code} → {_wdf.shape if _wdf is not None else 'None'}")
        need = need[1:]  # 已下，从待下载列表移除
        time.sleep(SLEEP)
    except Exception as e:
        logger.warning(f"预热失败（继续多线程）: {e}")

    if not need:
        # 只有一只或预热已耗尽
        _persist()
        merged = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
        if not merged.empty:
            merged = _apply_start_filter(merged, start_ts)
            merged.to_parquet(out_path)
        return merged

    lock = threading.Lock()
    failed: list[str] = []
    done = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(_fetch_one, c): c for c in need}
        for fut in as_completed(futures):
            code, df = fut.result()
            with lock:
                done += 1
                if df is not None and not df.empty:
                    fresh_records.append(df)
                    refreshed_codes.add(code)
                    success += 1
                else:
                    failed.append(code)
                if done % save_every == 0:
                    logger.info(f"股本进度 {done}/{len(need)}  成功={success} 失败={len(failed)}")
                    _persist(failed)

    if failed:
        logger.warning(f"最终失败 {len(failed)} 只（保留旧记录若有）: {failed[:20]}")

    _persist(failed)
    if not out_path.exists():
        return pd.DataFrame()

    merged = pd.read_parquet(out_path)
    # --start 过滤（保留 ffill 起点）
    merged = _apply_start_filter(merged, start_ts)
    merged = merged.drop_duplicates(subset=["code", "announce_date"], keep="last")
    merged = merged.sort_values(["code", "announce_date"]).reset_index(drop=True)
    merged.to_parquet(out_path)
    logger.info(f"share_change.parquet 最终: {merged.shape}  "
                f"覆盖 {merged['code'].nunique()} 只  "
                f"日期 {merged['announce_date'].min().date()} → "
                f"{merged['announce_date'].max().date()}  "
                f"本轮刷新 {len(refreshed_codes)} 只")
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",  default="2015-01-01",
                        help="仅保留 announce_date >= start 的记录（保留每只股票最早一条作为 ffill 起点）")
    parser.add_argument("--sample", type=int, default=0,
                        help="调试：仅取 universe 前 N 只")
    parser.add_argument("--codes",  default=None,
                        help="调试：指定逗号分隔代码列表（如 600519,600000），优先于 --sample")
    parser.add_argument(
        "--refresh-stale-days", type=int, default=DEFAULT_REFRESH_STALE_DAYS,
        help="已有记录的 max(announce_date) 早于 today-N 天则重拉（默认 30；"
             "禁止按 code 永久跳过）",
    )
    parser.add_argument(
        "--force-refresh", action="store_true",
        help="忽略新鲜度，全市场重拉股本变动",
    )
    args = parser.parse_args()

    codes = _load_stock_codes(sample=args.sample, codes_csv=args.codes)
    df = download_shares(
        codes,
        start=args.start,
        refresh_stale_days=args.refresh_stale_days,
        force_refresh=args.force_refresh,
    )
    if df is not None and not df.empty:
        # 单位自检：打印茅台最新一条
        m = df[df["code"] == "600519"]
        if not m.empty:
            last = m.sort_values("announce_date").iloc[-1]
            logger.info(
                f"[单位自检] 600519 茅台最新："
                f"announce={last['announce_date'].date()} "
                f"total_shares={last['total_shares']:.0f} 股 "
                f"({last['total_shares']/1e8:.4f} 亿股) "
                f"circ_shares={last['circ_shares']:.0f} 股"
            )


if __name__ == "__main__":
    main()
