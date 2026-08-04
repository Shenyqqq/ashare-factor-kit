"""
data/download_delisted.py — 独立下载退市股历史 OHLCV

避免幸存者偏差：补齐当前股票列表中已不存在的退市股的历史数据。
不破坏 data/download.py 的主流程，单独运行后会把退市股数据合并写入
raw/close_hfq.parquet / open_hfq.parquet / ... 等宽表。

用法:
    python -m data.download_delisted --start 2018-01-01
    python -m data.download_delisted --start 2018-01-01 --sample 20   # 调试
    python -m data.download_delisted --start 2018-01-01 --scan-raw    # 仅从 raw 反推
"""
from __future__ import annotations

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
from data.download import (
    OHLCV_FIELDS,
    OHLCV_WORKERS,
    _load_parquet,
    _save_wide,
    ensure_dirs,
    fetch_delisted_stocks,
    get_stock_list,
)

# 网络重试相关异常（akshare 内部用 requests，requests 依赖 urllib3）
try:
    import requests.exceptions as _req_exc
    import urllib3.exceptions as _url3_exc

    _NETWORK_RETRY_EXCEPTIONS: tuple = (
        _req_exc.ProxyError,
        _req_exc.ConnectionError,
        _req_exc.Timeout,
        _req_exc.SSLError,
        _req_exc.ChunkedEncodingError,
        _req_exc.RequestException,
        _url3_exc.MaxRetryError,
        ConnectionError,  # builtin，ProxyError 是其子类
        TimeoutError,     # builtin
    )
except ImportError:  # pragma: no cover
    _NETWORK_RETRY_EXCEPTIONS = (ConnectionError, TimeoutError)

# 网络错误重试退避（秒），指数退避：2s -> 4s -> 8s，最后一次失败后归类为网络失败
_RETRY_DELAYS: tuple[int, ...] = (2, 4, 8)


def _classify_error(e: Exception) -> str:
    """错误分类：'network'（可重试）vs 'data'（跳过不重试）。

    网络错误：ProxyError / ConnectionError / Timeout / MaxRetries / SSLError 等。
    数据错误：其余一律视为不可重试（如接口返回字段异常、symbol 不存在）。
    """
    if isinstance(e, _NETWORK_RETRY_EXCEPTIONS):
        return "network"
    msg = str(e).lower()
    net_keywords = ("proxy", "max retries", "timeout", "connection",
                    "remotedisconnected", "remote end closed", "sserror",
                    "handshake", "read timed out", "connectionpool")
    if any(k in msg for k in net_keywords):
        return "network"
    return "data"


def _current_codes() -> set[str]:
    """当前在市股票代码集合（不调用包含退市股的 get_stock_list，避免循环）。"""
    try:
        df = ak.stock_info_a_code_name()
        return set(df.iloc[:, 0].astype(str).str.zfill(6))
    except Exception as e:
        logger.warning(f"获取当前股票列表失败: {e}")
        return set()


def collect_delisted_codes(scan_raw: bool = False) -> list[str]:
    """汇总退市股候选代码清单。

    来源：
      1. fetch_delisted_stocks()：尝试 AKShare 退市接口
      2. _scan_delisted_from_raw()：从已下载的 raw parquet 反推（scan_raw=True 时强制启用）
    """
    ensure_dirs()

    # 先尝试 AKShare 退市接口
    try:
        delist_df = fetch_delisted_stocks()
    except Exception as e:
        logger.warning(f"fetch_delisted_stocks 失败: {e}")
        delist_df = pd.DataFrame()

    codes: set[str] = set()
    if not delist_df.empty and "code" in delist_df.columns:
        codes |= set(delist_df["code"].astype(str).str.zfill(6))

    # raw parquet 反推（接口失败或 --scan-raw 时启用）
    if scan_raw or not codes:
        from data.download import _scan_delisted_from_raw
        scanned = _scan_delisted_from_raw()
        if not scanned.empty and "code" in scanned.columns:
            codes |= set(scanned["code"].astype(str).str.zfill(6))

    # 剔除仍在市的股票（双重保险）
    cur = _current_codes()
    if cur:
        codes -= cur

    # 剔除北交所 8 开头
    codes = {c for c in codes if not c.startswith("8")}

    codes_list = sorted(codes)
    logger.info(f"退市股候选: {len(codes_list)} 只")
    return codes_list


def download_delisted_ohlcv(
    codes: list[str],
    start: str,
    end: str,
    adjust: str = "hfq",
    save_every: int = 100,
) -> None:
    """下载退市股历史 OHLCV，合并写入 raw 宽表 parquet。

    复用 data/download.py 的 OHLCV_FIELDS / 文件命名规范，把退市股数据
    merge 到已有 parquet（按 index 对齐，列追加）。
    """
    if not codes:
        logger.info("无退市股候选，跳过下载")
        return

    suffix = "_hfq" if adjust == "hfq" else "_raw"
    paths = {}
    if adjust == "hfq":
        for ak_col, fname in OHLCV_FIELDS.items():
            key = fname if fname in ("volume", "amount") else f"{fname}{suffix}"
            paths[ak_col] = (key, RAW_DIR / f"{key}.parquet")
    else:
        paths["收盘"] = ("close_raw", RAW_DIR / "prices_raw.parquet")

    # 加载已有数据
    data: dict[str, dict] = {ak_col: {} for ak_col in paths}
    for ak_col, (key, path) in paths.items():
        existing = _load_parquet(path)
        if existing is not None:
            data[ak_col] = {c: existing[c].dropna()
                            for c in existing.columns if c in existing}

    # 仅下载尚未有数据的退市股
    need = [c for c in codes if c not in data.get("收盘", {})]
    if not need:
        logger.info(f"退市股 OHLCV ({adjust}) 已全部下载，跳过")
        return

    logger.info(
        f"下载退市股 OHLCV ({adjust}): {len(need)} 只，并发={OHLCV_WORKERS} 线程"
    )

    lock = threading.Lock()
    failed: list[str] = []           # 网络最终失败
    skipped_no_data: list[str] = []  # 数据不存在（不重试）
    success_count = 0
    done = 0

    def _fetch(code: str):
        """单股下载，带网络错误重试。

        返回 (code, df, err, kind)：
          kind='network' 表示网络错误（已重试仍失败）；
          kind='data'    表示数据错误（不重试，直接跳过）；
          err=None       表示成功。
        """
        last_err: Exception | None = None
        last_kind = "data"
        # 首次尝试 + _RETRY_DELAYS 次重试
        for attempt, delay in enumerate([0] + list(_RETRY_DELAYS)):
            if delay > 0:
                time.sleep(delay)
            try:
                df = ak.stock_zh_a_hist(
                    symbol=code, period="daily",
                    start_date=start.replace("-", ""),
                    end_date=end.replace("-", ""),
                    adjust=adjust,
                )
                time.sleep(0.05)
                return code, df, None, None
            except Exception as e:
                last_err = e
                last_kind = _classify_error(e)
                if last_kind == "data":
                    # 数据错误不重试
                    return code, None, e, "data"
                # 网络错误：log 后继续下一次重试
                if attempt < len(_RETRY_DELAYS):
                    logger.debug(
                        f"退市股 {code} 网络错误（第 {attempt+1} 次失败，"
                        f"{_RETRY_DELAYS[attempt]}s 后重试）: {e}"
                    )
                continue
        return code, None, last_err, "network"

    with ThreadPoolExecutor(max_workers=OHLCV_WORKERS) as ex:
        futures = {ex.submit(_fetch, c): c for c in need}
        for fut in as_completed(futures):
            code, df, err, kind = fut.result()
            with lock:
                done += 1
                if err is not None:
                    if kind == "data":
                        skipped_no_data.append(code)
                        logger.info(f"退市股无数据/跳过 {code}: {err}")
                    else:
                        failed.append(code)
                        logger.warning(
                            f"退市股网络失败（已重试 {_RETRY_DELAYS} 次）"
                            f" {code}: {err}"
                        )
                elif df is None or df.empty:
                    # 接口返回空 DataFrame：视为无数据，跳过不重试
                    skipped_no_data.append(code)
                    logger.info(f"退市股无数据 {code}: 接口返回空")
                else:
                    success_count += 1
                    df["日期"] = pd.to_datetime(df["日期"])
                    df = df.set_index("日期")
                    for ak_col in paths:
                        if ak_col not in df.columns:
                            continue
                        data[ak_col][code] = df[ak_col].astype(float)
                if done % save_every == 0:
                    logger.info(f"退市股进度 {done}/{len(need)}，保存中间结果...")
                    for ak_col, (key, path) in paths.items():
                        if data[ak_col]:
                            _save_wide(data[ak_col], path)

    # 失败统计汇总
    logger.info(
        f"退市股下载：成功 {success_count} 只，"
        f"跳过 {len(skipped_no_data)} 只（无数据），"
        f"失败 {len(failed)} 只（网络）"
    )
    if skipped_no_data:
        logger.info(f"  跳过列表（前 10）: {skipped_no_data[:10]}")
    if failed:
        logger.warning(f"  网络失败列表（前 10）: {failed[:10]}")

    # 合并保存：与已有宽表对齐，新列追加
    for ak_col, (key, path) in paths.items():
        if not data[ak_col]:
            continue
        existing = _load_parquet(path)
        new_df = pd.DataFrame(data[ak_col]).sort_index()
        if existing is not None and not existing.empty:
            # 按列合并（已有列保留，新列追加；相同 index 取新值补充）
            combined = existing.reindex(
                index=existing.index.union(new_df.index)
            )
            # 把 new_df 对齐到 combined.index，避免逐列赋值触发碎片化警告
            new_df_aligned = new_df.reindex(index=combined.index)
            new_cols = [c for c in new_df.columns if c not in existing.columns]
            overlap_cols = [c for c in new_df.columns if c in existing.columns]

            # 批量 concat 新列（一次性追加，避免 frame.insert 多次触发 PerformanceWarning）
            if new_cols:
                combined = pd.concat(
                    [combined, new_df_aligned[new_cols]], axis=1
                )
            # 仅对重叠列做 fillna 补齐（退市股场景一般无重叠，循环成本极低）
            for col in overlap_cols:
                combined[col] = combined[col].where(
                    combined[col].notna(), new_df_aligned[col]
                )

            combined = combined.sort_index()
            # 官方建议：concat 后 copy() 一次性去碎片化
            combined = combined.copy()
            combined.to_parquet(path)
            logger.info(f"  {path.name}: 合并后 {combined.shape}")
        else:
            new_df.to_parquet(path)
            logger.info(f"  {path.name}: 新建 {new_df.shape}")


def main(start: str, end: str, sample: int = 0, scan_raw: bool = False) -> None:
    ensure_dirs()

    codes = collect_delisted_codes(scan_raw=scan_raw)
    if sample:
        codes = codes[:sample]
        logger.info(f"调试模式：仅下载前 {sample} 只退市股")

    if not codes:
        logger.info("无可下载的退市股，退出")
        return

    # 后复权 OHLCV
    logger.info("=== 退市股后复权 OHLCV ===")
    download_delisted_ohlcv(codes, start, end, adjust="hfq")

    # 向后兼容：prices_hfq.parquet = 后复权收盘价合并
    hfq_path = RAW_DIR / "close_hfq.parquet"
    if hfq_path.exists():
        close_df = pd.read_parquet(hfq_path)
        close_df.to_parquet(RAW_DIR / "prices_hfq.parquet")
        logger.info(f"已同步 prices_hfq.parquet: {close_df.shape}")

    # 不复权收盘价（PB 计算用）
    logger.info("=== 退市股不复权收盘价 ===")
    download_delisted_ohlcv(codes, start, end, adjust="")

    # 把退市股元数据合并到 stock_list.parquet
    try:
        sl_path = UNIVERSE_DIR / "stock_list.parquet"
        if sl_path.exists():
            sl = pd.read_parquet(sl_path)
            delist_df = fetch_delisted_stocks()
            if not delist_df.empty:
                existing = set(sl["code"].astype(str).str.zfill(6))
                new = delist_df[
                    ~delist_df["code"].astype(str).str.zfill(6).isin(existing)
                ]
                if not new.empty:
                    merged = pd.concat([sl, new], ignore_index=True, sort=False)
                    merged["code"] = merged["code"].astype(str).str.zfill(6)
                    merged.to_parquet(sl_path)
                    logger.info(f"stock_list.parquet 合并退市股: {len(merged)} 只")
    except Exception as e:
        logger.warning(f"合并 stock_list.parquet 失败（不影响数据下载）: {e}")

    logger.info("退市股数据下载完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument(
        "--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"),
    )
    parser.add_argument("--sample", type=int, default=0, help="调试：仅下载前 N 只")
    parser.add_argument(
        "--scan-raw", action="store_true",
        help="仅从已下载的 raw parquet 反推退市股候选（不调 AKShare 退市接口）",
    )
    args = parser.parse_args()
    main(args.start, args.end, args.sample, args.scan_raw)
