"""
data/validate_market_cap.py — 东财主面板 vs 自算市值小样本校验

主路径：``download_stock_value_em`` → ``total_mv`` / ``circ_mv``。
本模块抽若干股票，把东财面板与自算 ``*_computed``（或现拉 em API）对比。

用法：
    python -m data.validate_market_cap
    python -m data.validate_market_cap --codes 600519,600000,000001 --days 60
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import RAW_DIR
from data.mv_panels import COMPUTED, PRIMARY

DEFAULT_CODES = ["600519", "600000", "000001", "000858", "300750"]


def _load_wide(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"缺少 {path}")
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df.columns = df.columns.astype(str).str.zfill(6)
    return df


def _fetch_em_value(code: str) -> pd.DataFrame:
    import akshare as ak
    raw = ak.stock_value_em(symbol=code)
    col_map = {}
    for c in raw.columns:
        cs = str(c)
        if "数据日期" in cs or cs in ("日期", "date"):
            col_map[c] = "date"
        elif "总市值" in cs and "流通" not in cs:
            col_map[c] = "total_mv_em"
        elif "流通市值" in cs:
            col_map[c] = "circ_mv_em"
    if "date" not in col_map.values():
        raise ValueError(f"{code}: stock_value_em 无日期列: {list(raw.columns)}")
    df = raw.rename(columns=col_map)
    keep = [c for c in ("date", "total_mv_em", "circ_mv_em") if c in df.columns]
    df = df[keep].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ("total_mv_em", "circ_mv_em"):
        if c in df.columns:
            # 东财 stock_value_em「总市值/流通市值」实测单位=元
            df[c] = pd.to_numeric(df[c], errors="coerce")
            med = float(df[c].median()) if df[c].notna().any() else float("nan")
            if med == med and 0 < med < 1e6:
                df[c] = df[c] * 1e8
    return df.dropna(subset=["date"]).set_index("date").sort_index()


def compare_one(
    code: str,
    primary_total: pd.DataFrame,
    primary_circ: pd.DataFrame | None,
    computed_total: pd.DataFrame | None,
    computed_circ: pd.DataFrame | None,
    days: int = 60,
) -> dict:
    if code not in primary_total.columns:
        return {"code": code, "ok": False, "error": "not in primary total_mv"}
    out: dict = {"code": code, "ok": True}

    # 主面板 vs 现拉 API（一致性）
    try:
        em = _fetch_em_value(code)
        ours_t = primary_total[code].dropna().tail(days)
        common = ours_t.index.intersection(em.index)
        if len(common) >= 5:
            t_ours = ours_t.reindex(common)
            t_em = em["total_mv_em"].reindex(common)
            rel_t = ((t_ours - t_em) / t_em.replace(0, np.nan)).abs()
            out["panel_vs_api_med_abs_rel_err"] = float(rel_t.median())
            out["n_api"] = int(len(common))
            out["panel_last"] = float(t_ours.iloc[-1])
            out["api_last"] = float(t_em.iloc[-1])
        else:
            out["api_note"] = f"api_overlap={len(common)}"
    except Exception as e:
        out["api_error"] = str(e)

    # 主面板 vs 自算（若有）
    if computed_total is not None and code in computed_total.columns:
        a = primary_total[code].dropna().tail(days)
        b = computed_total[code].dropna().tail(days)
        common = a.index.intersection(b.index)
        if len(common) >= 5:
            ra = a.reindex(common)
            rb = b.reindex(common)
            rel = ((ra - rb) / rb.replace(0, np.nan)).abs()
            out["em_vs_computed_total_med_abs_rel_err"] = float(rel.median())
            out["n_computed"] = int(len(common))
        if (
            primary_circ is not None
            and computed_circ is not None
            and code in primary_circ.columns
            and code in computed_circ.columns
        ):
            a = primary_circ[code].dropna().tail(days)
            b = computed_circ[code].dropna().tail(days)
            common = a.index.intersection(b.index)
            if len(common) >= 5:
                rel = (
                    (a.reindex(common) - b.reindex(common))
                    / b.reindex(common).replace(0, np.nan)
                ).abs()
                out["em_vs_computed_circ_med_abs_rel_err"] = float(rel.median())
    else:
        out["computed_note"] = "无 *_computed 面板（可先跑 compute_market_cap）"
    return out


def validate_market_cap(
    codes: list[str] | None = None,
    days: int = 60,
) -> pd.DataFrame:
    codes = codes or DEFAULT_CODES
    primary_total = _load_wide(PRIMARY["total_mv"])
    primary_circ = (
        _load_wide(PRIMARY["circ_mv"]) if PRIMARY["circ_mv"].exists() else None
    )
    computed_total = (
        _load_wide(COMPUTED["total_mv"]) if COMPUTED["total_mv"].exists() else None
    )
    computed_circ = (
        _load_wide(COMPUTED["circ_mv"]) if COMPUTED["circ_mv"].exists() else None
    )
    rows = []
    for code in codes:
        code = str(code).zfill(6)
        logger.info(f"校验 {code} ...")
        rows.append(
            compare_one(
                code, primary_total, primary_circ,
                computed_total, computed_circ, days=days,
            )
        )
    df = pd.DataFrame(rows)
    ok = df[df["ok"] == True]  # noqa: E712
    if not ok.empty and "em_vs_computed_total_med_abs_rel_err" in ok.columns:
        s = ok["em_vs_computed_total_med_abs_rel_err"].dropna()
        if not s.empty:
            logger.info(
                f"em vs computed total_mv |中位相对误差| 中位数={s.median():.4%}  "
                f"最差={s.max():.4%}"
            )
    fail = df[df["ok"] == False]  # noqa: E712
    if not fail.empty:
        logger.warning(f"校验失败 {len(fail)} 只: {fail[['code','error']].to_dict('records')}")
    return df


def main():
    parser = argparse.ArgumentParser(description="东财主市值 vs 自算/API 小样本校验")
    parser.add_argument("--codes", default=None, help="逗号分隔代码")
    parser.add_argument("--days", type=int, default=60, help="最近 N 个重叠交易日")
    args = parser.parse_args()
    codes = (
        [c.strip().zfill(6) for c in args.codes.split(",") if c.strip()]
        if args.codes else None
    )
    df = validate_market_cap(codes=codes, days=args.days)
    print(df.to_string(index=False))
    out = RAW_DIR.parent / "processed" / "market_cap_validation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info(f"写出 {out}")


if __name__ == "__main__":
    main()
