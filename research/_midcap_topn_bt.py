"""One-shot: Top10/Top30 production-engine backtest on saved midcap dense scores."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.execution import (
    BacktestConfig,
    build_delist_dates_from_stock_list,
    build_listing_dates_from_stock_list,
    build_st_schedule,
)
from backtest.quantile import run_quantile_backtest
from backtest.risk_metrics import compute_risk_metrics, period_returns_from_nav
from config.settings import (
    BACKTEST_END,
    BACKTEST_START,
    RAW_DIR,
    RISK_FREE_RATE,
    UNIVERSE_DIR,
)
from research.ic.universe import (
    build_ic_tradability_mask,
    load_delist_dates,
    load_is_st_current,
    load_listing_dates,
    load_st_history,
    load_stock_names,
    mask_scores_for_backtest,
)
from utils.universe import build_mcap_yi_band_mask

OUT = Path("results/lgbm_h5_midcap30_100_sizeind_dense_w104_decay0_20260817")
SCORES = OUT / "factor_scores_lgbm_h5_w104_p_mcap30_100_rt4.parquet"
NAV100 = OUT / "backtest_lgbm_h5_w104_p_mcap30_100_rt4_nav.csv"


def year_returns(nav: pd.Series) -> pd.Series:
    """Calendar-year compounded period returns; same as quantile annual.csv."""
    s = pd.to_numeric(nav, errors="coerce")
    s.index = pd.DatetimeIndex(s.index)
    rets = period_returns_from_nav(s)
    out = {}
    for y, g in rets.groupby(rets.index.year):
        out[int(y)] = float((1.0 + g.fillna(0.0)).prod() - 1.0)
    return pd.Series(out)


def excess_vs_bm(nav: pd.DataFrame, col: str) -> dict:
    r_s = year_returns(nav[col])
    r_b = year_returns(nav["benchmark"])
    years = sorted(set(r_s.index) & set(r_b.index))
    xs = {y: float(r_s[y] - r_b[y]) for y in years}

    def cagr_span(lo, hi):
        sub_s = nav[col].dropna()
        sub_b = nav["benchmark"].dropna()
        idx = sub_s.index.intersection(sub_b.index)
        idx = idx[(idx >= f"{lo}-01-01") & (idx <= f"{hi}-12-31")]
        if len(idx) < 2:
            return np.nan
        n_y = (idx[-1] - idx[0]).days / 365.25
        if n_y <= 0:
            return np.nan
        cagr_s = (float(sub_s.loc[idx[-1]] / sub_s.loc[idx[0]])) ** (1 / n_y) - 1
        cagr_b = (float(sub_b.loc[idx[-1]] / sub_b.loc[idx[0]])) ** (1 / n_y) - 1
        return float(cagr_s - cagr_b)

    xs["2022-24_cagr"] = cagr_span(2022, 2024)
    return xs


def summarize_track(nav: pd.DataFrame, col: str, freq: str = "W-FRI") -> dict:
    m = compute_risk_metrics(nav, rebalance_freq=freq, rf=RISK_FREE_RATE)
    row = m.loc[col] if col in m.index else None
    xs = excess_vs_bm(nav, col)
    out = {
        "ann": float(row["年化收益"]) if row is not None else np.nan,
        "sharpe": float(row["Sharpe"]) if row is not None else np.nan,
        "mdd": float(row["最大回撤"]) if row is not None else np.nan,
        "xs": xs,
    }
    return out


def main() -> None:
    from run import _load_data

    print("load data...")
    (
        prices, prices_raw, financial, volume, amount,
        open_, high, low, clean_ret, masks,
        market_prices, industry_map,
        margin, moneyflow, northbound, institution,
        total_mv, circ_mv, turnover_rate,
    ) = _load_data(skip_download=True, sample=0)

    scores = pd.read_parquet(SCORES)
    mcap = build_mcap_yi_band_mask(circ_mv, min_yi=30.0, max_yi=100.0, total_mv=total_mv)
    eligible = build_ic_tradability_mask(
        prices, volume=volume, masks=masks,
        stock_names=load_stock_names(),
        listing_dates=load_listing_dates(),
        delist_dates=load_delist_dates(),
        is_st_current=load_is_st_current(),
        st_history=load_st_history(),
        small_cap_mask=mcap,
        exclude_limit_on_signal=False,
    )
    per = eligible.sum(axis=1)
    print(f"eligible (mcap∩tradable) mean={per.mean():.0f} min={int(per.min())} max={int(per.max())}")

    sl_path = UNIVERSE_DIR / "stock_list.parquet"
    sl_df = pd.read_parquet(sl_path)
    stock_names = sl_df.set_index("code")["name"]
    stock_names.index = stock_names.index.astype(str).str.zfill(6)
    delist_dates = build_delist_dates_from_stock_list(sl_df) or None
    listing_dates = build_listing_dates_from_stock_list(sl_df) or None
    is_st = sl_df.set_index("code")["is_st_current"] if "is_st_current" in sl_df.columns else None
    if is_st is not None:
        is_st.index = is_st.index.astype(str).str.zfill(6)
    st_hist = pd.read_parquet(RAW_DIR / "st_history.parquet") if (RAW_DIR / "st_history.parquet").exists() else None
    st_schedule = build_st_schedule(
        stock_names, prices.index, is_st_current=is_st,
        delist_dates=delist_dates, st_history=st_hist,
    )

    bt_scores = mask_scores_for_backtest(
        scores, prices, open_=open_, hold_period=5, volume=volume, masks=masks,
        stock_names=stock_names, listing_dates=listing_dates,
        delist_dates=delist_dates, is_st_current=is_st, st_history=st_hist,
        score_universe="strict",
    )
    cfg = BacktestConfig(bid_ask_spread_bps=10.0)

    report = {}
    nav100 = pd.read_csv(NAV100, index_col=0, parse_dates=True)
    report["Top100"] = summarize_track(nav100, "Top100")
    report["Q5"] = summarize_track(nav100, "Q5")
    report["benchmark"] = summarize_track(nav100, "benchmark")

    for n in (10, 30):
        print(f"backtest Top{n}...")
        result = run_quantile_backtest(
            prices, bt_scores,
            n_quantiles=5, rebalance_freq="W-FRI",
            start=BACKTEST_START, end=BACKTEST_END,
            open_prices=open_, masks=masks,
            config=cfg, stock_names=stock_names,
            listing_dates=listing_dates, volume=volume,
            st_schedule=st_schedule, delist_dates=delist_dates,
            eligible_mask=eligible, top_n=n,
            returns=clean_ret, hold_period=5,
        )
        col = f"Top{n}"
        result.nav.to_csv(OUT / f"backtest_top{n}_nav.csv", encoding="utf-8-sig")
        report[col] = summarize_track(result.nav, col)
        bm_n = int(result.nav["benchmark"].notna().sum())
        print(f"  Top{n} ann={report[col]['ann']:.2%} sharpe={report[col]['sharpe']:.2f} "
              f"mdd={report[col]['mdd']:.2%} bm_periods={bm_n}")

    outp = OUT / "_topn_vs_midcap_ew.json"
    def _ser(d):
        if isinstance(d, dict):
            return {str(k): _ser(v) for k, v in d.items()}
        if isinstance(d, float):
            return None if (d != d) else round(d, 6)
        return d
    outp.write_text(json.dumps(_ser(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", outp)
    print(json.dumps(_ser(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
