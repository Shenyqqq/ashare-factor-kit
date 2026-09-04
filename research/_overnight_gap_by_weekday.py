"""一次性研究：A 股高开/低开门槛扫描 × T+1 可卖路径（不入库、不改生产代码）。

主分类：该股自己的隔夜 vs θ（不是当日 EW 隔夜）。
主收益：open[t]→open[t+1]、open[t]→close[t+1]（T+1 能卖的路径）。
白天 close[t]/open[t] 仅诊断，不作追低开依据。

用法（仓库根目录）:
    .venv\\Scripts\\python.exe research/_overnight_gap_by_weekday.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import RAW_DIR  # noqa: E402
from data.clean import clean_ohlc_aligned, clean_volume, mask_post_delist  # noqa: E402
from research.ic.universe import (  # noqa: E402
    build_ic_tradability_mask,
    load_delist_dates,
    load_is_st_current,
    load_listing_dates,
    load_st_history,
    load_stock_names,
)

THETAS_BP = (10, 20, 50, 100, 200, 300, 500)
WEEKDAY_THETA = 100  # 星期精简表默认 θ；若扫描分叉再在解读里换
WEEKDAY_NAMES = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
REGIME_A = ("2022-01-01", "2024-12-31")
REGIME_B = ("2025-01-01", "2026-12-31")
BP = 1e4
SPARSE_N_DAYS = 30
SPARSE_AVG_N = 50
GAPS = ("高开", "平开", "低开")
OUT_PATH = ROOT / "research" / "_overnight_gap_t1_out.txt"


class _Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            try:
                f.write(data)
            except UnicodeEncodeError:
                try:
                    f.write(data.encode("utf-8", errors="replace").decode("utf-8", errors="replace"))
                except Exception:
                    pass
            try:
                f.flush()
            except Exception:
                pass

    def flush(self):
        for f in self.files:
            f.flush()


def _opt_parquet(name: str) -> pd.DataFrame | None:
    p = RAW_DIR / name
    return pd.read_parquet(p) if p.exists() else None


def _align(df: pd.DataFrame, index, columns) -> pd.DataFrame:
    return df.reindex(index=index, columns=columns)


def _daily_ew(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """截面等权。返回 (mu[T], n[T])；n=0 → mu=nan。"""
    x = np.where(mask, arr, np.nan)
    n = np.isfinite(x).sum(axis=1).astype(np.float64)
    with np.errstate(all="ignore"):
        mu = np.nanmean(x, axis=1)
    mu = np.where(n > 0, mu, np.nan)
    return mu, n


def _ts_stats(mu: np.ndarray, dates: pd.DatetimeIndex, sel: np.ndarray | None = None) -> dict:
    if sel is None:
        s = mu
        idx = dates
    else:
        s = mu[sel]
        idx = dates[sel]
    ok = np.isfinite(s)
    s = s[ok]
    n = int(s.size)
    if n == 0:
        return {"n_days": 0, "mean_bp": np.nan, "t": np.nan, "win": np.nan, "end": None}
    mean = float(s.mean())
    if n >= 2 and float(s.std(ddof=1)) > 0:
        tstat = float(stats.ttest_1samp(s, 0.0, nan_policy="omit").statistic)
    else:
        tstat = np.nan
    return {
        "n_days": n,
        "mean_bp": mean * BP,
        "t": tstat,
        "win": float((s > 0).mean()),
        "end": pd.Timestamp(idx[ok][-1]) if ok.any() else None,
    }


def _fmt(r: dict, *, sparse: bool = False) -> str:
    if r["n_days"] == 0:
        return f"{'—':>5} {'—':>7} {'—':>6} {'—':>6}"
    star = "*" if sparse else " "
    t = "   n/a" if not np.isfinite(r["t"]) else f"{r['t']:+6.2f}"
    return f"{r['n_days']:5d}{star}{r['mean_bp']:+7.1f} {t} {100*r['win']:5.1f}%"


def _sparse(n_days: int, avg_n: float) -> bool:
    return (n_days < SPARSE_N_DAYS) or (np.isfinite(avg_n) and avg_n < SPARSE_AVG_N)


def classify_stock(ovn: np.ndarray, theta_bp: float) -> dict[str, np.ndarray]:
    thr = theta_bp / BP
    return {
        "高开": ovn >= thr,
        "平开": np.abs(ovn) < thr,
        "低开": ovn <= -thr,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    out_f = open(OUT_PATH, "w", encoding="utf-8")
    sys.stdout = out_f

    logger.info("加载 hfq OHLC / volume …")
    close_raw = pd.read_parquet(RAW_DIR / "prices_hfq.parquet")
    open_raw = _opt_parquet("open_hfq.parquet")
    high_raw = _opt_parquet("high_hfq.parquet")
    low_raw = _opt_parquet("low_hfq.parquet")
    if open_raw is None:
        raise SystemExit("缺少 open_hfq.parquet")
    vol_raw = _opt_parquet("volume.parquet")

    close, open_, high, low = clean_ohlc_aligned(close_raw, open_raw, high_raw, low_raw)
    volume = clean_volume(vol_raw, name="volume") if vol_raw is not None else None
    del high, low  # 本轮不看缺口回补

    delist_dates = load_delist_dates()
    if delist_dates:
        close = mask_post_delist(close, delist_dates)
        open_ = mask_post_delist(open_, delist_dates)
        volume = mask_post_delist(volume, delist_dates)

    cols = close.columns.intersection(open_.columns)
    idx = close.index.intersection(open_.index)
    close, open_ = close.loc[idx, cols], open_.loc[idx, cols]
    if volume is not None:
        volume = _align(volume, idx, cols)

    logger.info("构建 tradable mask（research：非 ST/停牌/零成交/次新/退市后，保留涨跌停）…")
    tradable = build_ic_tradability_mask(
        close,
        volume=volume,
        stock_names=load_stock_names(),
        listing_dates=load_listing_dates(),
        delist_dates=delist_dates,
        is_st_current=load_is_st_current(),
        st_history=load_st_history(),
        exclude_limit_on_signal=False,
    )

    dates = close.index
    dow = dates.dayofweek.to_numpy()
    prev_close = close.shift(1)
    overnight = (open_ / prev_close - 1.0).to_numpy(dtype=np.float64)
    open_a = open_.to_numpy(dtype=np.float64)
    close_a = close.to_numpy(dtype=np.float64)

    open_n1 = np.roll(open_a, -1, axis=0)
    open_n1[-1] = np.nan
    close_n1 = np.roll(close_a, -1, axis=0)
    close_n1[-1] = np.nan
    close_n2 = np.roll(close_a, -2, axis=0)
    close_n2[-2:] = np.nan

    with np.errstate(all="ignore"):
        o2o = open_n1 / open_a - 1.0
        o2c1 = close_n1 / open_a - 1.0
        daytime = close_a / open_a - 1.0
        o2c2 = close_n2 / open_a - 1.0

    trad_a = tradable.to_numpy(dtype=bool)
    buy_ok = (
        trad_a
        & np.isfinite(overnight)
        & np.isfinite(open_a)
        & (open_a > 0)
        & np.isfinite(prev_close.to_numpy(dtype=np.float64))
        & (prev_close.to_numpy(dtype=np.float64) > 0)
    )
    win_ok = {
        "o2o": buy_ok & np.isfinite(o2o),
        "o2c1": buy_ok & np.isfinite(o2c1),
        "day": buy_ok & np.isfinite(daytime),
        "o2c2": buy_ok & np.isfinite(o2c2),
    }
    panels = {"o2o": o2o, "o2c1": o2c1, "day": daytime, "o2c2": o2c2}

    n_days_all = int(len(dates))
    start, end = dates.min(), dates.max()
    n_valid_med = float(buy_ok.sum(axis=1).mean())

    print()
    print("=" * 92)
    print("A 股 高开/低开门槛扫描 × T+1 可卖路径（股票级隔夜 vs θ）")
    print("=" * 92)
    print(f"样本         : {start.date()} → {end.date()}  （{n_days_all} 个交易日）")
    print("复权         : hfq（prices_hfq + open_hfq，clean_ohlc_aligned）")
    print("隔夜类型     : 该股 open[t]/close[t-1]-1 对照 θ；高开≥θ，低开≤−θ，平开|ovn|<θ")
    print("买入         : 当天 open[t]（开完、类型已知）")
    print("主窗口       : open[t]→open[t+1]（T+1 最早可卖）  |  open[t]→close[t+1]（次日收盘卖）")
    print("诊断         : close[t]/open[t] 白天 —— 不能作为是否追低开的依据")
    print("实盘对照     : 若 t=周一，Mon open→Wed close；其余星期 open[t]→close[t+2]")
    print("截面         : 每日组内等权，再对日期做均值/t/胜率（胜率=该日 EW>0 的交易日占比）")
    print(f"可交易       : research mask；median 买入日有效约 {n_valid_med:.0f} 只")
    print(f"稀疏         : 天数<{SPARSE_N_DAYS} 或日均入组<{SPARSE_AVG_N} 标 *")
    print("未扣成本     : 表内收益未扣约 10bp 价差/佣金")
    print()

    # 每个 θ 预计算各组每日 EW
    cache: dict[int, dict] = {}
    for theta in THETAS_BP:
        cls = classify_stock(overnight, theta)
        cache[theta] = {}
        for gap in GAPS:
            cache[theta][gap] = {"n_buy": None}
            m_buy = buy_ok & cls[gap]
            n_buy = m_buy.sum(axis=1).astype(np.float64)
            n_buy = np.where(n_buy > 0, n_buy, np.nan)
            cache[theta][gap]["n_buy"] = n_buy
            cache[theta][gap]["avg_n"] = float(np.nanmean(n_buy))
            for wname, parr in panels.items():
                mu, n_w = _daily_ew(parr, win_ok[wname] & cls[gap])
                cache[theta][gap][wname] = mu
                cache[theta][gap][f"n_{wname}"] = n_w

    def row_of(theta, gap, wname, sel=None):
        mu = cache[theta][gap][wname]
        r = _ts_stats(mu, dates, sel)
        avg_n = cache[theta][gap]["avg_n"]
        if sel is not None:
            nb = cache[theta][gap]["n_buy"][sel]
            avg_n = float(np.nanmean(nb)) if np.isfinite(nb).any() else np.nan
        r["avg_n"] = avg_n
        r["sparse"] = _sparse(r["n_days"], avg_n)
        return r

    # ── 主表：θ × 类型，两个 T+1 窗口 ────────────────────────────────────────
    print("【主表】全样本  θ × 开盘类型   股票级分类")
    print("列：n日*=天数(稀疏)  日均N  |  o→o+1 = open[t]→open[t+1]  |  o→c+1 = open[t]→close[t+1]")
    print("-" * 92)
    hdr = (
        f"{'θbp':>5} {'类型':<4} {'日均N':>6}  "
        f"{'n日':>5} {'o→o+1':>7} {'t':>6} {'胜率':>6}  "
        f"{'n日':>5} {'o→c+1':>7} {'t':>6} {'胜率':>6}"
    )
    print(hdr)
    print("-" * 92)
    for theta in THETAS_BP:
        for gap in GAPS:
            r1 = row_of(theta, gap, "o2o")
            r2 = row_of(theta, gap, "o2c1")
            sp = "*" if (r1["sparse"] or r2["sparse"]) else " "
            t1 = "   n/a" if not np.isfinite(r1["t"]) else f"{r1['t']:+6.2f}"
            t2 = "   n/a" if not np.isfinite(r2["t"]) else f"{r2['t']:+6.2f}"
            print(
                f"{theta:5d} {gap:<4} {r1['avg_n']:6.0f}{sp} "
                f"{r1['n_days']:5d} {r1['mean_bp']:+7.1f} {t1} {100*r1['win']:5.1f}%  "
                f"{r2['n_days']:5d} {r2['mean_bp']:+7.1f} {t2} {100*r2['win']:5.1f}%"
            )
        print("-" * 92)

    # ── 诊断：白天（不作决策）────────────────────────────────────────────────
    print()
    print("【诊断·白天】close[t]/open[t]  —— 不能作为是否追低开的依据")
    print("-" * 72)
    print(f"{'θbp':>5} {'类型':<4} {'日均N':>6} {'n日':>6} {'白天bp':>8} {'t':>7} {'胜率':>7}")
    print("-" * 72)
    for theta in THETAS_BP:
        for gap in GAPS:
            r = row_of(theta, gap, "day")
            t = "    n/a" if not np.isfinite(r["t"]) else f"{r['t']:+7.2f}"
            star = "*" if r["sparse"] else " "
            print(
                f"{theta:5d} {gap:<4} {r['avg_n']:6.0f}{star}{r['n_days']:6d} "
                f"{r['mean_bp']:+8.1f} {t} {100*r['win']:6.1f}%"
            )
    print("-" * 72)
    print("对照：若白天低开为正、但 o→o+1 为负，说明当日收正被次日开盘吃掉。")

    # ── 星期精简表（θ=100bp）────────────────────────────────────────────────
    print()
    print(f"【星期】θ={WEEKDAY_THETA}bp  股票级   主窗口 1/2 + open[t]→close[t+2]（周一=Mon→Wed）")
    print("-" * 100)
    print(
        f"{'星期':<5} {'类型':<4} {'日均N':>6}  "
        f"{'o→o+1':>7} {'t':>6} {'胜率':>6}  "
        f"{'o→c+1':>7} {'t':>6} {'胜率':>6}  "
        f"{'o→c+2':>7} {'t':>6} {'胜率':>6}"
    )
    print("-" * 100)
    for d in range(5):
        sel = dow == d
        wd = WEEKDAY_NAMES[d]
        for gap in GAPS:
            r0 = row_of(WEEKDAY_THETA, gap, "o2o", sel)
            r1 = row_of(WEEKDAY_THETA, gap, "o2c1", sel)
            r2 = row_of(WEEKDAY_THETA, gap, "o2c2", sel)
            t0 = "   n/a" if not np.isfinite(r0["t"]) else f"{r0['t']:+6.2f}"
            t1 = "   n/a" if not np.isfinite(r1["t"]) else f"{r1['t']:+6.2f}"
            t2 = "   n/a" if not np.isfinite(r2["t"]) else f"{r2['t']:+6.2f}"
            star = "*" if r0["sparse"] else " "
            print(
                f"{wd:<5} {gap:<4} {r0['avg_n']:6.0f}{star} "
                f"{r0['mean_bp']:+7.1f} {t0} {100*r0['win']:5.1f}%  "
                f"{r1['mean_bp']:+7.1f} {t1} {100*r1['win']:5.1f}%  "
                f"{r2['mean_bp']:+7.1f} {t2} {100*r2['win']:5.1f}%"
            )
        print("-" * 100)

    # ── 周一 × 全部 θ：对着实盘 ────────────────────────────────────────────
    mon = dow == 0
    print()
    print("【周一开盘买】全 θ    o→o+1（周二开最早可卖） / o→c+1（周二收） / Mon open→Wed close")
    print("-" * 100)
    print(
        f"{'θbp':>5} {'类型':<4} {'日均N':>6}  "
        f"{'o→o+1':>7} {'t':>6} {'胜率':>6}  "
        f"{'o→c+1':>7} {'t':>6} {'胜率':>6}  "
        f"{'→Wed':>7} {'t':>6} {'胜率':>6}"
    )
    print("-" * 100)
    for theta in THETAS_BP:
        for gap in GAPS:
            r0 = row_of(theta, gap, "o2o", mon)
            r1 = row_of(theta, gap, "o2c1", mon)
            r2 = row_of(theta, gap, "o2c2", mon)
            t0 = "   n/a" if not np.isfinite(r0["t"]) else f"{r0['t']:+6.2f}"
            t1 = "   n/a" if not np.isfinite(r1["t"]) else f"{r1['t']:+6.2f}"
            t2 = "   n/a" if not np.isfinite(r2["t"]) else f"{r2['t']:+6.2f}"
            star = "*" if r0["sparse"] else " "
            print(
                f"{theta:5d} {gap:<4} {r0['avg_n']:6.0f}{star} "
                f"{r0['mean_bp']:+7.1f} {t0} {100*r0['win']:5.1f}%  "
                f"{r1['mean_bp']:+7.1f} {t1} {100*r1['win']:5.1f}%  "
                f"{r2['mean_bp']:+7.1f} {t2} {100*r2['win']:5.1f}%"
            )
        print("-" * 100)

    # ── 阶段 2022–24 vs 2025–26：主窗口 o2o / o2c1 ─────────────────────────
    def _sel_range(lo, hi):
        return np.asarray((dates >= lo) & (dates <= hi))

    sel_a = _sel_range(*REGIME_A)
    sel_b = _sel_range(*REGIME_B)
    d_a = dates[sel_a]
    d_b = dates[sel_b]
    print()
    print("【阶段】2022–24 vs 2025–26    主窗口 open[t]→open[t+1] / open[t]→close[t+1]")
    print(f"  A: {d_a.min().date()} → {d_a.max().date()}  ({int(sel_a.sum())} 日)")
    print(f"  B: {d_b.min().date()} → {d_b.max().date()}  ({int(sel_b.sum())} 日)")
    print("-" * 100)
    print(
        f"{'θbp':>5} {'类型':<4}  "
        f"{'A o→o+1':>8} {'t':>6} {'胜率':>6}  "
        f"{'B o→o+1':>8} {'t':>6} {'胜率':>6}  flip  "
        f"{'A o→c+1':>8} {'B o→c+1':>8}"
    )
    print("-" * 100)
    for theta in THETAS_BP:
        for gap in GAPS:
            ra0 = row_of(theta, gap, "o2o", sel_a)
            rb0 = row_of(theta, gap, "o2o", sel_b)
            ra1 = row_of(theta, gap, "o2c1", sel_a)
            rb1 = row_of(theta, gap, "o2c1", sel_b)
            flip = (
                np.isfinite(ra0["mean_bp"])
                and np.isfinite(rb0["mean_bp"])
                and np.sign(ra0["mean_bp"]) != np.sign(rb0["mean_bp"])
                and abs(ra0["mean_bp"]) > 1e-9
                and abs(rb0["mean_bp"]) > 1e-9
            )
            ta = "   n/a" if not np.isfinite(ra0["t"]) else f"{ra0['t']:+6.2f}"
            tb = "   n/a" if not np.isfinite(rb0["t"]) else f"{rb0['t']:+6.2f}"
            print(
                f"{theta:5d} {gap:<4}  "
                f"{ra0['mean_bp']:+8.1f} {ta} {100*ra0['win']:5.1f}%  "
                f"{rb0['mean_bp']:+8.1f} {tb} {100*rb0['win']:5.1f}%  "
                f"{'翻转' if flip else '同号'}  "
                f"{ra1['mean_bp']:+8.1f} {rb1['mean_bp']:+8.1f}"
            )
        print("-" * 100)

    # 周一阶段翻转（o2o + Wed）
    print()
    print("【阶段·仅周一】o→o+1 与 Mon→Wed")
    print("-" * 92)
    mon_a = sel_a & mon
    mon_b = sel_b & mon
    print(
        f"{'θbp':>5} {'类型':<4}  "
        f"{'A o→o+1':>8} {'B o→o+1':>8} flip  "
        f"{'A→Wed':>8} {'B→Wed':>8} flip"
    )
    print("-" * 92)
    for theta in THETAS_BP:
        for gap in GAPS:
            ra0 = row_of(theta, gap, "o2o", mon_a)
            rb0 = row_of(theta, gap, "o2o", mon_b)
            ra2 = row_of(theta, gap, "o2c2", mon_a)
            rb2 = row_of(theta, gap, "o2c2", mon_b)

            def _flip(a, b):
                return (
                    np.isfinite(a["mean_bp"])
                    and np.isfinite(b["mean_bp"])
                    and np.sign(a["mean_bp"]) != np.sign(b["mean_bp"])
                    and abs(a["mean_bp"]) > 1e-9
                    and abs(b["mean_bp"]) > 1e-9
                )

            print(
                f"{theta:5d} {gap:<4}  "
                f"{ra0['mean_bp']:+8.1f} {rb0['mean_bp']:+8.1f} "
                f"{'翻转' if _flip(ra0, rb0) else '同号'}  "
                f"{ra2['mean_bp']:+8.1f} {rb2['mean_bp']:+8.1f} "
                f"{'翻转' if _flip(ra2, rb2) else '同号'}"
                f"{' *' if ra0['sparse'] or rb0['sparse'] else ''}"
            )
        print("-" * 92)

    # ── 市场级（当日 EW 隔夜 vs θ）附一行，不当主表 ─────────────────────────
    print()
    print("【附·市场级】当日全部可交易 EW 隔夜相对 θ 给「这一天」贴标签，再看全市场 EW 的 o→o+1")
    print("（不是主结论；主结论是上面的股票自己的隔夜。）")
    ew_ovn, _ = _daily_ew(overnight, buy_ok)
    mkt_o2o, _ = _daily_ew(o2o, win_ok["o2o"])
    print("-" * 72)
    print(f"{'θbp':>5} {'日类型':<4} {'n日':>6} {'o→o+1':>8} {'t':>7} {'胜率':>7}")
    print("-" * 72)
    for theta in (50, 100, 200):
        thr = theta / BP
        mcls = {
            "高开": ew_ovn >= thr,
            "平开": np.abs(ew_ovn) < thr,
            "低开": ew_ovn <= -thr,
        }
        for gap in GAPS:
            r = _ts_stats(np.where(mcls[gap], mkt_o2o, np.nan), dates)
            star = "*" if r["n_days"] < SPARSE_N_DAYS else " "
            t = "    n/a" if not np.isfinite(r["t"]) else f"{r['t']:+7.2f}"
            print(
                f"{theta:5d} {gap:<4} {r['n_days']:6d}{star}{r['mean_bp']:+8.1f} {t} {100*r['win']:6.1f}%"
            )
        print("-" * 72)

    print()
    print("脚本结束。全市场可交易等权 ≠ Top30；* = 稀疏。未扣约 10bp 价差。")
    print(f"完整输出: {OUT_PATH}")
    out_f.close()
    sys.stdout = sys.__stdout__
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
