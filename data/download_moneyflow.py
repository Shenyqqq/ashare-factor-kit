"""
data/download_moneyflow.py  —  东财大单净流入数据下载

数据：个股每日大单（≥50万）和超大单（≥100万）净流入额
主接口：东财公开 HTTP（优先 kline，daykline 为 AKShare 同源备选）
    GET https://push2his.eastmoney.com/api/qt/stock/fflow/kline/get
    GET https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get
    （ak.stock_individual_fund_flow 只用 daykline；本机常被掐，kline 更稳）

存储：
    data/raw/moneyflow_large.parquet      大单净流入（元） wide
    data/raw/moneyflow_superlarge.parquet 超大单净流入（元） wide
    data/raw/_cache/moneyflow/{code}.parquet  单票长表（resume）

注意：
    - 历史深度约近数月（实测常 ~120 个交易日），非全历史
    - push2his 易 Connection aborted / 经本机代理(7890) 易 ProxyError
    - 默认 trust_env=False（绕过 WinIE 代理）；需要代理时加 --use-proxy
    - 绝不允许用更短序列覆盖已有更长历史（防 push2delay 单日污染）
    - 全市场请低并发：--sleep 1.0+，workers=1

用法：
    python -m data.download_moneyflow --preflight
    python -m data.download_moneyflow --sample 20 --sleep 1.5
    python -m data.download_moneyflow --force --sample 50
    python -m data.download_moneyflow --codes 000001,600519,300750
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import pandas as pd
import requests
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import RAW_DIR, UNIVERSE_DIR

MAX_RETRY = 3
DEFAULT_SLEEP = 1.5
STALE_DAYS = 30
CIRCUIT_BREAK_N = 5  # consecutive failures → abort batch (fflow 限流后重试无意义)
CIRCUIT_COOLDOWN = 90.0

CACHE_DIR = RAW_DIR / "_cache" / "moneyflow"
LARGE_PATH = RAW_DIR / "moneyflow_large.parquet"
SUPER_PATH = RAW_DIR / "moneyflow_superlarge.parquet"

# kline 优先（实测 daykline 易 Connection aborted；kline 同结构可返回 ~120 日）
FFLOW_URLS = (
    "https://push2his.eastmoney.com/api/qt/stock/fflow/kline/get",
    "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get",
)
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def _market_of(code: str) -> str:
    if code.startswith(("6", "9")):
        return "sh"
    if code.startswith(("4", "8")):
        return "bj"
    return "sz"


def _session(use_proxy: bool) -> requests.Session:
    s = requests.Session()
    s.trust_env = bool(use_proxy)
    if not use_proxy:
        s.proxies = {"http": None, "https": None}
    s.headers.update(
        {
            "User-Agent": UA,
            "Referer": "https://data.eastmoney.com/zjlx/detail.html",
            "Accept": "*/*",
            "Connection": "close",
        }
    )
    return s


def _parse_klines(klines: list) -> pd.DataFrame:
    rows = []
    for item in klines:
        parts = str(item).split(",")
        if len(parts) < 6:
            continue
        rows.append(
            {
                "date": parts[0],
                "large_net": pd.to_numeric(parts[4], errors="coerce"),
                "super_net": pd.to_numeric(parts[5], errors="coerce"),
                "main_net": pd.to_numeric(parts[1], errors="coerce"),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["date", "large_net", "super_net", "main_net"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    return df.reset_index(drop=True)


def fetch_fflow_history(
    code: str,
    session: requests.Session | None = None,
    use_proxy: bool = False,
    max_retry: int = MAX_RETRY,
    base_sleep: float = 1.0,
) -> pd.DataFrame:
    """
    直连东财 push2his 个股资金流日 K。
    返回长表 columns=[date, large_net, super_net, main_net]；失败抛异常。
    """
    market_map = {"sh": 1, "sz": 0, "bj": 0}
    mkt = _market_of(code)
    params = {
        "lmt": "0",
        "klt": "101",
        "secid": f"{market_map[mkt]}.{code}",
        "fields1": "f1,f2,f3,f7",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
        "ut": "b2884a393a59ad64002292a3e90d46a5",
    }
    own = session is None
    sess = session or _session(use_proxy)
    last_err: Exception | None = None
    try:
        for attempt in range(max_retry):
            # 前几次只用 kline；最后一次才回退 daykline（AKShare 同源，常被掐）
            url = FFLOW_URLS[0] if attempt < max_retry - 1 else FFLOW_URLS[-1]
            params["_"] = int(time.time() * 1000)
            try:
                r = sess.get(url, params=params, timeout=30)
                r.raise_for_status()
                data = r.json()
                klines = (data.get("data") or {}).get("klines") or []
                if not klines:
                    raise ValueError(f"empty klines for {code}: data={data.get('data')}")
                df = _parse_klines(klines)
                if df.empty:
                    raise ValueError(f"parsed empty for {code}")
                if len(df) < 5:
                    raise ValueError(
                        f"suspiciously short history for {code}: rows={len(df)} via {url}"
                    )
                return df
            except Exception as e:
                last_err = e
                ep = url.rsplit("/", 2)[-2] + "/" + url.rsplit("/", 1)[-1]
                # Connection aborted = 典型限流，快速放弃本票，交给上层 circuit breaker
                aborted = "Remote end closed" in str(e) or "Connection aborted" in str(e)
                wait = 0.8 if aborted else base_sleep * (2**attempt)
                logger.debug(
                    f"{code} attempt {attempt + 1}/{max_retry} via {ep} "
                    f"{type(e).__name__}: {e}; sleep {wait:.1f}s"
                )
                if aborted and attempt >= 1:
                    break
                time.sleep(wait)
        assert last_err is not None
        raise last_err
    finally:
        if own:
            sess.close()


def preflight(use_proxy: bool = False, code: str = "000001") -> bool:
    """连通性探测：成功拿到 ≥20 行历史才算通路打开。"""
    logger.info(f"预检 push2his fflow code={code} use_proxy={use_proxy}")
    try:
        df = fetch_fflow_history(code, use_proxy=use_proxy, max_retry=3, base_sleep=1.5)
        ok = len(df) >= 20
        logger.info(
            f"预检 {'通过' if ok else '偏短'}: rows={len(df)} "
            f"{df['date'].min().date()}->{df['date'].max().date()}"
        )
        return ok
    except Exception as e:
        logger.error(f"预检失败: {type(e).__name__}: {e}")
        logger.error(
            "提示: 1) 确认本机代理 127.0.0.1:7890 未劫持东财 "
            "（默认已 bypass；若必须走代理加 --use-proxy）；"
            "2) push2his 限流时 Connection aborted，隔数分钟再试；"
            "3) stock_value_em 仍可用不代表 fflow 可用。"
        )
        return False


def _load_existing_wide(path: Path) -> dict[str, pd.Series]:
    if path.exists():
        df = pd.read_parquet(path)
        return {str(c).zfill(6): df[c].dropna() for c in df.columns}
    return {}


def _load_cache(code: str) -> pd.DataFrame | None:
    p = CACHE_DIR / f"{code}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if "date" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


def _save_cache(code: str, df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_DIR / f"{code}.parquet", index=False)


def _merge_series(existing: pd.Series | None, new: pd.Series, force: bool) -> pd.Series:
    """合并时禁止用更短序列整体替换更长序列。"""
    new = new.dropna().sort_index()
    if existing is None or existing.empty or force:
        # force 仍拒绝「新序列显著更短」的整体替换，避免误用单日接口污染
        if existing is not None and len(existing) > max(5, len(new) * 2) and not force:
            logger.warning(
                f"拒绝用短序列覆盖: existing={len(existing)} new={len(new)}"
            )
            return existing
        if existing is not None and len(existing) > len(new) * 2 and force:
            # force 也做保护：合并而非替换
            combined = pd.concat([existing, new])
            return combined[~combined.index.duplicated(keep="last")].sort_index()
        return new
    if len(new) < 5 and len(existing) > 20:
        # 疑似 push2delay 单日：只补最新点
        combined = pd.concat([existing, new])
        return combined[~combined.index.duplicated(keep="last")].sort_index()
    combined = pd.concat([existing, new])
    return combined[~combined.index.duplicated(keep="last")].sort_index()


def _save_both(large_data: dict, super_data: dict):
    df_large = pd.DataFrame(large_data).sort_index()
    df_super = pd.DataFrame(super_data).sort_index()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df_large.to_parquet(LARGE_PATH)
    df_super.to_parquet(SUPER_PATH)
    logger.info(
        f"大单净流入保存: large={df_large.shape}, super={df_super.shape} "
        f"range={df_large.index.min()}->{df_large.index.max()}"
    )
    return df_large, df_super


def download_moneyflow(
    codes: list[str],
    save_every: int = 50,
    sleep: float = DEFAULT_SLEEP,
    force: bool = False,
    stale_days: int = STALE_DAYS,
    use_proxy: bool = False,
    skip_preflight: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    下载大单和超大单净流入。逐票直连 push2his；失败不中断；支持 cache resume。
    """
    codes = [str(c).zfill(6) for c in codes]
    large_data = _load_existing_wide(LARGE_PATH)
    super_data = _load_existing_wide(SUPER_PATH)

    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=stale_days)
    need = []
    for code in codes:
        if force:
            need.append(code)
            continue
        s = large_data.get(code)
        cache = _load_cache(code)
        latest = None
        if s is not None and len(s):
            latest = pd.Timestamp(s.index.max())
        if cache is not None and len(cache):
            cmax = pd.Timestamp(cache["date"].max())
            latest = cmax if latest is None else max(latest, cmax)
        if latest is None or latest < cutoff:
            need.append(code)
        elif code not in large_data and cache is not None:
            # 缓存有、wide 无 → 组装时补上
            need.append(code)

    if not need:
        logger.info("大单净流入数据已是最新，跳过")
        return pd.DataFrame(large_data), pd.DataFrame(super_data)

    if not skip_preflight:
        if not preflight(use_proxy=use_proxy):
            logger.error("预检未通过，中止批量下载（可用 --skip-preflight 强制）")
            return _save_both(large_data, super_data)

    logger.info(f"大单净流入: 需下载 {len(need)}/{len(codes)} 只 sleep={sleep}s proxy={use_proxy}")
    failed: list[str] = []
    consecutive_fail = 0
    sess = _session(use_proxy)

    try:
        for i, code in enumerate(need):
            ok = False
            try:
                df = fetch_fflow_history(
                    code, session=sess, use_proxy=use_proxy, base_sleep=max(sleep, 0.8)
                )
                _save_cache(code, df)
                s_large = df.set_index("date")["large_net"]
                s_super = df.set_index("date")["super_net"]
                large_data[code] = _merge_series(large_data.get(code), s_large, force=force)
                super_data[code] = _merge_series(super_data.get(code), s_super, force=force)
                ok = True
                consecutive_fail = 0
            except Exception as e:
                logger.warning(f"大单净流入失败 {code}: {type(e).__name__}: {e}")
                failed.append(code)
                consecutive_fail += 1
                # 回退：若有足够长的 cache，写入 wide
                cache = _load_cache(code)
                if cache is not None and len(cache) >= 20:
                    large_data[code] = _merge_series(
                        large_data.get(code),
                        cache.set_index("date")["large_net"],
                        force=False,
                    )
                    super_data[code] = _merge_series(
                        super_data.get(code),
                        cache.set_index("date")["super_net"],
                        force=False,
                    )

            if consecutive_fail >= CIRCUIT_BREAK_N:
                logger.error(
                    f"连续失败 {consecutive_fail} 次，疑似 push2his 限流；"
                    f"冷却 {CIRCUIT_COOLDOWN:.0f}s 后中止本批，已保存进度。"
                )
                time.sleep(CIRCUIT_COOLDOWN)
                break

            time.sleep(sleep)
            if (i + 1) % save_every == 0:
                logger.info(f"进度 {i + 1}/{len(need)} ok_so_far={i + 1 - len(failed)}")
                _save_both(large_data, super_data)
    finally:
        sess.close()

    if failed:
        logger.warning(f"失败 {len(failed)} 只: {failed[:20]}")

    return _save_both(large_data, super_data)


def _load_codes(sample: int = 0, codes_csv: str | None = None) -> list[str]:
    if codes_csv:
        return [c.strip().zfill(6) for c in codes_csv.split(",") if c.strip()]
    universe = pd.read_parquet(UNIVERSE_DIR / "stock_list.parquet")
    codes = universe["code"].astype(str).str.zfill(6).tolist()
    if sample:
        codes = codes[:sample]
    return codes


def main():
    parser = argparse.ArgumentParser(description="下载东财大单/超大单净流入")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--codes", type=str, default=None, help="逗号分隔代码")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP)
    parser.add_argument(
        "--use-proxy",
        action="store_true",
        help="使用系统/环境代理（默认关闭：本机 7890 代理常导致东财 ProxyError）",
    )
    parser.add_argument("--preflight", action="store_true", help="仅预检后退出")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--stale-days", type=int, default=STALE_DAYS)
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if args.preflight:
        ok = preflight(use_proxy=args.use_proxy)
        raise SystemExit(0 if ok else 2)

    codes = _load_codes(sample=args.sample, codes_csv=args.codes)
    download_moneyflow(
        codes,
        force=args.force,
        sleep=args.sleep,
        use_proxy=args.use_proxy,
        skip_preflight=args.skip_preflight,
        stale_days=args.stale_days,
    )


if __name__ == "__main__":
    main()
