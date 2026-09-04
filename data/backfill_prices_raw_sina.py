"""One-off / emergency: backfill prices_raw via Sina when Eastmoney is down.

Usage:
  python -m data.backfill_prices_raw_sina
  python -m data.backfill_prices_raw_sina --workers 2 --max-codes 500
"""
from __future__ import annotations

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

import akshare as ak
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import RAW_DIR, UNIVERSE_DIR
from data.download import _last_valid_by_code, report_raw_hfq_coverage, _save_wide


def _sina_sym(code: str) -> str:
    c = str(code).zfill(6)
    return f"sh{c}" if c.startswith(("6", "9")) else f"sz{c}"


def _fetch(code: str, start: str, end: str) -> tuple[str, pd.Series | None, str | None]:
    try:
        df = ak.stock_zh_a_daily(
            symbol=_sina_sym(code),
            start_date=start,
            end_date=end,
            adjust="",
        )
        time.sleep(0.15)
        if df is None or df.empty:
            return code, None, "empty"
        s = pd.Series(
            pd.to_numeric(df["close"], errors="coerce").values,
            index=pd.to_datetime(df["date"]),
            name=code,
        ).dropna().sort_index()
        return code, s, None
    except Exception as e:
        return code, None, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-codes", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=100)
    args = parser.parse_args()

    hfq_path = RAW_DIR / "close_hfq.parquet"
    raw_path = RAW_DIR / "prices_raw.parquet"
    peer = _last_valid_by_code(hfq_path)
    existing = pd.read_parquet(raw_path)
    existing.index = pd.to_datetime(existing.index)
    existing.columns = existing.columns.astype(str).str.zfill(6)
    data = {c: existing[c].dropna() for c in existing.columns}

    u = pd.read_parquet(UNIVERSE_DIR / "stock_list.parquet")
    from data.download import is_excluded_universe_code
    codes = [
        c for c in u["code"].astype(str).str.zfill(6)
        if not is_excluded_universe_code(c)
    ]
    need = []
    for c in codes:
        last = data[c].index[-1] if c in data and len(data[c]) else None
        peer_ts = peer.get(c)
        if peer_ts is not None and (last is None or last < peer_ts):
            need.append(c)
    if args.max_codes:
        need = need[: args.max_codes]
    logger.info(f"Sina backfill prices_raw: {len(need)} codes, workers={args.workers}")

    lock = threading.Lock()
    ok = fail = 0
    done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for c in need:
            series = data.get(c)
            start = (
                (series.index[-1] + pd.Timedelta(days=1)).strftime("%Y%m%d")
                if series is not None and len(series) else "20180101"
            )
            end = pd.Timestamp.today().strftime("%Y%m%d")
            futs[ex.submit(_fetch, c, start, end)] = c
        for fut in as_completed(futs):
            code, s, err = fut.result()
            with lock:
                done += 1
                if err or s is None or s.empty:
                    fail += 1
                    if fail <= 10 or done % 200 == 0:
                        logger.warning(f"fail {code}: {err}")
                else:
                    old = data.get(code)
                    if old is not None and len(old):
                        comb = pd.concat([old, s])
                        data[code] = comb[~comb.index.duplicated(keep="last")].sort_index()
                    else:
                        data[code] = s
                    ok += 1
                if done % args.save_every == 0:
                    logger.info(f"progress {done}/{len(need)} ok={ok} fail={fail}")
                    _save_wide(data, raw_path)

    _save_wide(data, raw_path)
    logger.info(f"done ok={ok} fail={fail}")
    print(report_raw_hfq_coverage())


if __name__ == "__main__":
    main()
