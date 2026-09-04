"""
data/download_moneyflow_ths.py  —  同花顺资金流下载（可自动化截面 + 日频归档）

限制（诚实）：
    AKShare 同花顺个股资金流 ``stock_fund_flow_individual`` **没有**稳定的
    「单票全历史日 K」接口，仅提供截面排行：
        即时 / 3日排行 / 5日排行 / 10日排行 / 20日排行
    行业/概念同理（``stock_fund_flow_industry`` / ``stock_fund_flow_concept``）。
    「N日排行」= 近 N 日累计净流入排行（当日截面），不是可回放的逐日序列。

可重复落盘策略：
    1) 拉取全部窗口截面 → 最新快照 parquet（带 ths_ 前缀，不覆盖东财 moneyflow_*）
    2) 按 asof_date 追加进 *_hist.parquet（同日同窗去重续传）
    3) 可选 big_deal 即时大单成交明细

口径差异：
    THS「净额」≠ 东财大单(≥50万)/超大单(≥100万)净流入；勿直接喂
    ``moneyflow_large`` / ``大单净流入_5d``。

存储：
    data/raw/moneyflow_ths_individual_spot.parquet      即时最新
    data/raw/moneyflow_ths_individual_w{3,5,10,20}.parquet
    data/raw/moneyflow_ths_individual_hist.parquet      日归档长表
    data/raw/moneyflow_ths_industry_{spot|w*|hist}.parquet
    data/raw/moneyflow_ths_concept_{spot|w*|hist}.parquet
    data/raw/moneyflow_ths_big_deal_spot.parquet
    data/raw/_cache/moneyflow_ths/{kind}_{window}_{asof}.ok

用法：
    python -m data.download_moneyflow_ths --smoke
    python -m data.download_moneyflow_ths --sleep 1.5
    python -m data.download_moneyflow_ths --kinds individual,industry --windows 即时,5日排行
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import akshare as ak
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import RAW_DIR

MAX_RETRY = 3
DEFAULT_SLEEP = 1.5
CACHE_DIR = RAW_DIR / "_cache" / "moneyflow_ths"
STATUS_PATH = Path(__file__).resolve().parent.parent / ".tmp" / "moneyflow_ths_download_status.txt"

WINDOWS_ALL = ("即时", "3日排行", "5日排行", "10日排行", "20日排行")
KINDS_ALL = ("individual", "industry", "concept", "big_deal")

WINDOW_TAG = {
    "即时": "spot",
    "3日排行": "w3",
    "5日排行": "w5",
    "10日排行": "w10",
    "20日排行": "w20",
}

_CN_AMOUNT_RE = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s*(万亿|亿|万|元)?\s*$"
)


def _write_status(text: str) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(text, encoding="utf-8")


def _append_status(line: str) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATUS_PATH.open("a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def parse_cn_amount(x) -> float:
    """解析同花顺金额字符串，如 '1.26亿' / '-8306.28万' / '3.35亿' → 元。"""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return float("nan")
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(",", "").replace("%", "")
    if not s or s in {"-", "--", "None", "nan"}:
        return float("nan")
    m = _CN_AMOUNT_RE.match(s)
    if not m:
        try:
            return float(s)
        except ValueError:
            return float("nan")
    val = float(m.group(1))
    unit = m.group(2) or "元"
    mul = {"万亿": 1e12, "亿": 1e8, "万": 1e4, "元": 1.0}[unit]
    return val * mul


def parse_pct(x) -> float:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return float("nan")
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace("%", "").replace(",", "")
    if not s or s in {"-", "--"}:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _zero_pad_code(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)


def _cache_ok(kind: str, window: str, asof: str) -> Path:
    tag = WINDOW_TAG.get(window, window)
    return CACHE_DIR / f"{kind}_{tag}_{asof}.ok"


def _mark_ok(kind: str, window: str, asof: str, n: int) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _cache_ok(kind, window, asof)
    p.write_text(f"n={n}\nts={datetime.now().isoformat(timespec='seconds')}\n", encoding="utf-8")


def _latest_path(kind: str, window: str) -> Path:
    tag = WINDOW_TAG[window]
    if kind == "individual" and tag == "spot":
        # 与先前探测落盘名对齐
        return RAW_DIR / "moneyflow_ths_individual_spot.parquet"
    return RAW_DIR / f"moneyflow_ths_{kind}_{tag}.parquet"


def _hist_path(kind: str) -> Path:
    return RAW_DIR / f"moneyflow_ths_{kind}_hist.parquet"


def _fetch_with_retry(fn, *, symbol: str | None, sleep: float) -> pd.DataFrame:
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRY + 1):
        try:
            if symbol is None:
                df = fn()
            else:
                df = fn(symbol=symbol)
            if df is None or df.empty:
                raise RuntimeError("empty dataframe")
            return df
        except Exception as e:
            last_err = e
            wait = sleep * attempt
            logger.warning(f"{fn.__name__}({symbol!r}) attempt {attempt}/{MAX_RETRY}: {e}; sleep {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"{fn.__name__}({symbol!r}) failed: {last_err}")


def _clean_individual(raw: pd.DataFrame, window: str, asof: pd.Timestamp) -> pd.DataFrame:
    # 即时：净额/流入资金/流出资金；N日排行：资金流入净额/阶段涨跌幅/连续换手率
    colmap = {
        "股票代码": "code",
        "股票简称": "name",
        "最新价": "price",
        "涨跌幅": "pct_change",
        "阶段涨跌幅": "pct_change",
        "换手率": "turnover",
        "连续换手率": "turnover",
        "流入资金": "inflow",
        "流出资金": "outflow",
        "净额": "net",
        "资金流入净额": "net",
        "成交额": "amount",
    }
    df = raw.rename(columns={k: v for k, v in colmap.items() if k in raw.columns})
    # rename 可能重复映射到同一英文名，保留首次
    df = df.loc[:, ~df.columns.duplicated()]
    if "code" not in df.columns or "net" not in df.columns:
        raise KeyError(f"individual missing col code/net; got {list(raw.columns)}")
    def _amt(series: pd.Series) -> pd.Series:
        # object/string（含「亿/万」）→ parse_cn_amount；纯数值原样
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors="coerce")
        return series.map(parse_cn_amount)

    out = pd.DataFrame({
        "asof_date": asof,
        "window": window,
        "code": _zero_pad_code(df["code"]),
        "name": df["name"].astype(str) if "name" in df.columns else "",
        "price": pd.to_numeric(df["price"], errors="coerce") if "price" in df.columns else pd.NA,
        "pct_change": df["pct_change"].map(parse_pct) if "pct_change" in df.columns else pd.NA,
        "turnover": df["turnover"].map(parse_pct) if "turnover" in df.columns else pd.NA,
        "inflow": _amt(df["inflow"]) if "inflow" in df.columns else pd.NA,
        "outflow": _amt(df["outflow"]) if "outflow" in df.columns else pd.NA,
        "net": _amt(df["net"]),
        "amount": _amt(df["amount"]) if "amount" in df.columns else pd.NA,
        "source": "ths",
    })
    return out.dropna(subset=["code"]).drop_duplicates(["asof_date", "window", "code"], keep="last")


def _board_amount_to_yuan(s: pd.Series) -> pd.Series:
    """
    同花顺行业/概念资金流：AKShare 常直接给 float（单位=亿元），
    偶发带「亿/万」字符串；统一成元。
    """
    if s.dtype == object:
        return s.map(parse_cn_amount)
    num = pd.to_numeric(s, errors="coerce")
    # 数值面板口径为亿元（如净额 50.86 → 5.086e9 元）
    return num * 1e8


def _clean_board(raw: pd.DataFrame, window: str, asof: pd.Timestamp, kind: str) -> pd.DataFrame:
    # 行业/概念列名均为「行业」
    name_col = "行业" if "行业" in raw.columns else ("概念" if "概念" in raw.columns else None)
    if name_col is None:
        raise KeyError(f"{kind} missing name col; got {list(raw.columns)}")
    out = pd.DataFrame({
        "asof_date": asof,
        "window": window,
        "kind": kind,
        "sector": raw[name_col].astype(str),
        "index_price": pd.to_numeric(raw["行业指数"], errors="coerce") if "行业指数" in raw.columns else pd.NA,
        "pct_change": raw["行业-涨跌幅"].map(parse_pct) if "行业-涨跌幅" in raw.columns else (
            raw["涨跌幅"].map(parse_pct) if "涨跌幅" in raw.columns else pd.NA
        ),
        "inflow": _board_amount_to_yuan(raw["流入资金"]) if "流入资金" in raw.columns else pd.NA,
        "outflow": _board_amount_to_yuan(raw["流出资金"]) if "流出资金" in raw.columns else pd.NA,
        "net": _board_amount_to_yuan(raw["净额"]) if "净额" in raw.columns else pd.NA,
        "company_count": pd.to_numeric(raw["公司家数"], errors="coerce") if "公司家数" in raw.columns else pd.NA,
        "source": "ths",
    })
    return out.dropna(subset=["sector"]).drop_duplicates(
        ["asof_date", "window", "kind", "sector"], keep="last"
    )


def _clean_big_deal(raw: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    colmap = {
        "成交时间": "trade_time",
        "股票代码": "code",
        "股票简称": "name",
        "成交价格": "price",
        "成交量": "volume",
        "成交额": "amount",
        "大单性质": "deal_type",
        "涨跌幅": "pct_change",
        "涨跌额": "chg",
    }
    df = raw.rename(columns={k: v for k, v in colmap.items() if k in raw.columns})
    if "code" not in df.columns:
        raise KeyError(f"big_deal missing code; got {list(raw.columns)}")
    if "amount" in df.columns:
        if df["amount"].dtype == object:
            amount = df["amount"].map(parse_cn_amount)
        else:
            # AKShare 数值口径多为万元
            amount = pd.to_numeric(df["amount"], errors="coerce") * 1e4
    else:
        amount = pd.NA
    out = pd.DataFrame({
        "asof_date": asof,
        "trade_time": df["trade_time"].astype(str) if "trade_time" in df.columns else "",
        "code": _zero_pad_code(df["code"]),
        "name": df["name"].astype(str) if "name" in df.columns else "",
        "price": pd.to_numeric(df["price"], errors="coerce") if "price" in df.columns else pd.NA,
        "volume": pd.to_numeric(df["volume"], errors="coerce") if "volume" in df.columns else pd.NA,
        "amount": amount,
        "deal_type": df["deal_type"].astype(str) if "deal_type" in df.columns else "",
        "pct_change": df["pct_change"].map(parse_pct) if "pct_change" in df.columns else pd.NA,
        "source": "ths",
    })
    return out.dropna(subset=["code"])


def _merge_hist(path: Path, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if new.empty:
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()
    if path.exists():
        old = pd.read_parquet(path)
        df = pd.concat([old, new], ignore_index=True)
    else:
        df = new
    df = df.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def download_one_window(
    kind: str,
    window: str,
    asof: pd.Timestamp,
    sleep: float,
    force: bool,
) -> dict:
    asof_s = asof.strftime("%Y-%m-%d")
    ok_path = _cache_ok(kind, window, asof_s)
    if ok_path.exists() and not force:
        logger.info(f"skip {kind}/{window} asof={asof_s} (cache ok)")
        return {"kind": kind, "window": window, "status": "skipped", "n": 0}

    if kind == "individual":
        raw = _fetch_with_retry(ak.stock_fund_flow_individual, symbol=window, sleep=sleep)
        cleaned = _clean_individual(raw, window, asof)
        keys = ["asof_date", "window", "code"]
    elif kind == "industry":
        raw = _fetch_with_retry(ak.stock_fund_flow_industry, symbol=window, sleep=sleep)
        cleaned = _clean_board(raw, window, asof, "industry")
        keys = ["asof_date", "window", "kind", "sector"]
    elif kind == "concept":
        raw = _fetch_with_retry(ak.stock_fund_flow_concept, symbol=window, sleep=sleep)
        cleaned = _clean_board(raw, window, asof, "concept")
        keys = ["asof_date", "window", "kind", "sector"]
    else:
        raise ValueError(kind)

    latest = _latest_path(kind, window)
    cleaned.to_parquet(latest, index=False)
    hist = _merge_hist(_hist_path(kind), cleaned, keys=keys)
    _mark_ok(kind, window, asof_s, len(cleaned))
    logger.info(
        f"OK {kind}/{window}: n={len(cleaned)} → {latest.name}; hist={hist.shape}"
    )
    return {
        "kind": kind,
        "window": window,
        "status": "ok",
        "n": len(cleaned),
        "latest": str(latest),
        "hist_rows": len(hist),
    }


def download_big_deal(asof: pd.Timestamp, sleep: float, force: bool) -> dict:
    asof_s = asof.strftime("%Y-%m-%d")
    window = "即时"
    ok_path = _cache_ok("big_deal", window, asof_s)
    if ok_path.exists() and not force:
        logger.info(f"skip big_deal asof={asof_s}")
        return {"kind": "big_deal", "window": window, "status": "skipped", "n": 0}
    raw = _fetch_with_retry(ak.stock_fund_flow_big_deal, symbol=None, sleep=sleep)
    cleaned = _clean_big_deal(raw, asof)
    path = RAW_DIR / "moneyflow_ths_big_deal_spot.parquet"
    cleaned.to_parquet(path, index=False)
    hist = _merge_hist(
        RAW_DIR / "moneyflow_ths_big_deal_hist.parquet",
        cleaned,
        keys=["asof_date", "trade_time", "code", "price", "volume", "amount", "deal_type"],
    )
    _mark_ok("big_deal", window, asof_s, len(cleaned))
    logger.info(f"OK big_deal: n={len(cleaned)} → {path.name}; hist={hist.shape}")
    return {
        "kind": "big_deal",
        "window": window,
        "status": "ok",
        "n": len(cleaned),
        "latest": str(path),
        "hist_rows": len(hist),
    }


def download_moneyflow_ths(
    kinds: list[str] | None = None,
    windows: list[str] | None = None,
    sleep: float = DEFAULT_SLEEP,
    force: bool = False,
    smoke: bool = False,
) -> list[dict]:
    """
    下载同花顺资金流截面并归档。

    smoke=True：仅 individual×即时 + industry×即时（快速验证）。
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    asof = pd.Timestamp.today().normalize()

    if smoke:
        kinds = ["individual", "industry"]
        windows = ["即时"]
    else:
        kinds = list(kinds or ["individual", "industry", "concept", "big_deal"])
        windows = list(windows or list(WINDOWS_ALL))

    for w in windows:
        if w not in WINDOW_TAG:
            raise ValueError(f"unknown window {w!r}; choose from {WINDOWS_ALL}")

    started = datetime.now().isoformat(timespec="seconds")
    _write_status(
        f"# moneyflow_ths download status\n"
        f"started: {started}\n"
        f"asof: {asof.date()}\n"
        f"kinds: {kinds}\n"
        f"windows: {windows}\n"
        f"smoke: {smoke}\n"
        f"sleep: {sleep}\n"
        f"note: THS has no per-stock daily history via AKShare; "
        f"windows are cross-section lookbacks; hist = daily archive.\n"
        f"note: THS net ≠ EM large/superlarge; EM full-market history NOT ready.\n"
        f"---\n"
    )

    results: list[dict] = []
    tasks: list[tuple[str, str | None]] = []
    for kind in kinds:
        if kind == "big_deal":
            tasks.append(("big_deal", None))
        else:
            for w in windows:
                tasks.append((kind, w))

    for i, (kind, window) in enumerate(tasks):
        try:
            if kind == "big_deal":
                r = download_big_deal(asof, sleep=sleep, force=force)
            else:
                assert window is not None
                r = download_one_window(kind, window, asof, sleep=sleep, force=force)
            results.append(r)
            _append_status(
                f"{datetime.now().isoformat(timespec='seconds')} "
                f"{r['kind']}/{r.get('window','')} status={r['status']} n={r.get('n',0)}"
            )
        except Exception as e:
            logger.exception(f"FAIL {kind}/{window}: {e}")
            results.append({"kind": kind, "window": window, "status": "fail", "error": str(e)})
            _append_status(
                f"{datetime.now().isoformat(timespec='seconds')} "
                f"{kind}/{window} status=fail error={e}"
            )
        if i + 1 < len(tasks):
            time.sleep(sleep)

    ok_n = sum(1 for r in results if r.get("status") == "ok")
    skip_n = sum(1 for r in results if r.get("status") == "skipped")
    fail_n = sum(1 for r in results if r.get("status") == "fail")
    finished = datetime.now().isoformat(timespec="seconds")
    _append_status("---")
    _append_status(f"finished: {finished}")
    _append_status(f"summary: ok={ok_n} skipped={skip_n} fail={fail_n} total={len(results)}")
    for kind in {r["kind"] for r in results}:
        hp = _hist_path(kind) if kind != "big_deal" else RAW_DIR / "moneyflow_ths_big_deal_hist.parquet"
        if hp.exists():
            try:
                h = pd.read_parquet(hp)
                _append_status(f"hist {hp.name}: rows={len(h)} cols={list(h.columns)[:8]}")
            except Exception as e:
                _append_status(f"hist {hp.name}: read_error={e}")
    logger.info(f"done ok={ok_n} skipped={skip_n} fail={fail_n} → {STATUS_PATH}")
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="同花顺资金流下载（截面+日归档）")
    p.add_argument("--smoke", action="store_true", help="仅 individual+industry 即时，冒烟")
    p.add_argument(
        "--kinds",
        type=str,
        default="",
        help="comma: individual,industry,concept,big_deal",
    )
    p.add_argument(
        "--windows",
        type=str,
        default="",
        help="comma: 即时,3日排行,5日排行,10日排行,20日排行",
    )
    p.add_argument("--sleep", type=float, default=DEFAULT_SLEEP)
    p.add_argument("--force", action="store_true", help="忽略当日 cache，强制重拉")
    args = p.parse_args()

    kinds = [x.strip() for x in args.kinds.split(",") if x.strip()] or None
    windows = [x.strip() for x in args.windows.split(",") if x.strip()] or None
    download_moneyflow_ths(
        kinds=kinds,
        windows=windows,
        sleep=args.sleep,
        force=args.force,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
