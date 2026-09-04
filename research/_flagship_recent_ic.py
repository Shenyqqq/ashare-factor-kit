"""旗舰最近几期 W-FRI 截面 IC（不重训、不全量 quantile）。

用 last-window xgb_w156_20260731 对 8/07、8/14、8/21 出分；
标签口径与训练一致：Rank IC = spearman_ic(score, raw forward_return)，
forward_return = close[t+5]/open[t+1]-1，research 可交易池（信号日保留涨跌停）。
8/21 出分但不编 IC。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TRAIN_MAX_WORKERS", "1")
os.environ.setdefault("IC_MAX_WORKERS", "1")

from config.encoding_bootstrap import add_utf8_file_sink, bootstrap_stdio_utf8, configure_loguru

bootstrap_stdio_utf8()
configure_loguru()

import numpy as np
import pandas as pd
from loguru import logger

from data.download import is_b_share_code
from models.wf.metrics import spearman_ic
from research.ic.forward_return import build_forward_return
from research.ic.universe import (
    build_ic_tradability_mask,
    load_delist_dates,
    load_is_st_current,
    load_listing_dates,
    load_st_history,
    load_stock_names,
)
from utils.rebalance_dates import get_rebalance_dates as _rebalance_dates

OLD_SCORE_DIR = ROOT / "results" / "xgb_h5_sizeind_sparse_w156_decay0_nobshare_20260817"
OLD_SCORE_FILE = OLD_SCORE_DIR / "factor_scores_xgb_h5_w156_p_sparse_rt4.parquet"
OLD_IC = OLD_SCORE_DIR / "ic_series_xgb_h5_w156_p_sparse_rt4.csv"
OLD_METRICS = OLD_SCORE_DIR / "model_metrics_xgb_h5_w156_p_sparse_rt4.json"

OUT = ROOT / "results" / "xgb_h5_sizeind_w156_nob_20260823"
MODEL_DIR = OUT
DENSE_YAML = ROOT / "config" / "factor_configs_h5_sizeind_20260815.yaml"
BT_FREQ = "W-FRI"
HOLD = 5
TOP_N = 100
# 至少覆盖这些 W-FRI；实际以行情日历为准
WANT_DATES = [
    pd.Timestamp("2026-07-17"),
    pd.Timestamp("2026-07-24"),
    pd.Timestamp("2026-07-31"),
    pd.Timestamp("2026-08-07"),
    pd.Timestamp("2026-08-14"),
    pd.Timestamp("2026-08-21"),
]


def pearson_ic(pred, actual) -> float:
    if len(pred) < 5:
        return float("nan")
    return float(pd.Series(pred).corr(pd.Series(actual), method="pearson"))


def _drop_b(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in df.columns if not is_b_share_code(c)]
    n_b = df.shape[1] - len(keep)
    if n_b:
        logger.warning(f"剔除 B 股 {n_b} 列")
    return df.loc[:, keep]


def _label_window(idx: pd.DatetimeIndex, date, n: int = HOLD):
    """返回 (complete, t1, tN)。complete 当且仅当 t 在日历上且 t+1、t+N 都存在。"""
    d = pd.Timestamp(date).normalize()
    if d not in idx:
        return False, None, None
    loc = int(idx.get_loc(d))
    if loc + n >= len(idx):
        return False, None, None
    t1, tN = idx[loc + 1], idx[loc + n]
    return True, t1, tN


def score_dates(dates, panels, data_end) -> pd.DataFrame:
    from live.daily_update import (
        DEFAULT_WARMUP_DAYS,
        append_factor_panels,
        build_feature_matrix,
        compute_barra_warmup,
        load_latest_models,
        neutralize_as_of,
        predict_scores,
        resolve_feature_names,
        slice_warmup,
    )

    model_entries, manifest_fn, manifest_nc = load_latest_models(
        str(MODEL_DIR), prefer_model="xgb", prefer_window=156,
    )
    feature_names, fn_source = resolve_feature_names(
        model_entries, str(DENSE_YAML), horizon=5,
    )
    logger.info(f"特征 {len(feature_names)} 个（{fn_source}） neut={manifest_nc}")

    warmup = slice_warmup(panels, data_end, DEFAULT_WARMUP_DAYS)
    factor_panels = append_factor_panels(feature_names, warmup, data_end)
    barra, weights = compute_barra_warmup(warmup)
    universe = panels["prices"].columns.astype(str).str.zfill(6)

    rows = []
    for d in dates:
        logger.info(f"last-window 出分 {pd.Timestamp(d).date()}")
        neut_rows = None
        if manifest_fn:
            neut_rows = neutralize_as_of(
                factor_panels, barra, weights, panels["industry_map"], d,
                universe=universe, prices=panels["prices"],
                neut_controls=manifest_nc,
            )
        X = build_feature_matrix(
            feature_names, neut_rows or {}, factor_panels, d, universe=universe,
        )
        s = predict_scores(model_entries, X)
        s.name = pd.Timestamp(d)
        n = int(pd.to_numeric(s, errors="coerce").notna().sum())
        logger.info(f"  {pd.Timestamp(d).date()} n_scored={n}")
        rows.append(s)
    extra = pd.DataFrame(rows)
    extra.index = pd.DatetimeIndex([r.name for r in rows])
    extra.columns = extra.columns.astype(str).str.zfill(6)
    return extra


def _fmt_pct(x, nd=1) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x * 100:.{nd}f}%"


def _pctile(hist: pd.Series, x: float) -> float:
    h = hist.dropna()
    if not np.isfinite(x) or h.empty:
        return float("nan")
    return float((h <= x).mean())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "recent_ic.log"
    add_utf8_file_sink(log_path)
    logger.info("旗舰最近 W-FRI 截面 IC（last-window，不重训全 WF）")

    from live.daily_update import load_clean_panels

    panels = load_clean_panels(sample=0)
    prices = panels["prices"]
    open_ = panels["open_"]
    volume = panels["volume"]
    masks = panels["masks"]
    prices.columns = prices.columns.astype(str).str.zfill(6)
    if open_ is not None:
        open_.columns = open_.columns.astype(str).str.zfill(6)
    if volume is not None:
        volume.columns = volume.columns.astype(str).str.zfill(6)

    data_end = pd.Timestamp(prices.index.max()).normalize()
    idx = pd.DatetimeIndex(prices.index).sort_values()
    cal = _rebalance_dates(prices.index, BT_FREQ)
    cal = [pd.Timestamp(d).normalize() for d in cal if d <= data_end]
    logger.info(f"数据末日={data_end.date()} 最近调仓日={[d.date() for d in cal[-8:]]}")

    # 最近若干 W-FRI：历史得分覆盖的 + 待 last-window 出分的
    target = []
    for d in WANT_DATES:
        if d in cal:
            target.append(d)
        else:
            logger.warning(f"{d.date()} 不在 W-FRI 调仓日历（或无行情）")
    # 再补最近 6 个日历调仓日，避免漏期
    for d in cal[-6:]:
        if d not in target:
            target.append(d)
    target = sorted(set(target))
    logger.info(f"目标调仓日={[d.date() for d in target]}")

    old = pd.read_parquet(OLD_SCORE_FILE)
    old.index = pd.to_datetime(old.index).normalize()
    old.columns = old.columns.astype(str).str.zfill(6)
    old = _drop_b(old)
    last_old = pd.Timestamp(old.index.max()).normalize()
    logger.info(f"旧得分末日={last_old.date()} n_dates={len(old)}")

    need_score = [d for d in target if d > last_old and d <= data_end]
    logger.info(f"last-window 待出分={[d.date() for d in need_score]}")

    extra = pd.DataFrame()
    if need_score:
        extra = score_dates(need_score, panels, data_end)
        extra = _drop_b(extra)
        extra.to_parquet(OUT / "recent_scores_lastwindow.parquet")
        logger.info(f"已写 {OUT / 'recent_scores_lastwindow.parquet'}  {extra.shape}")

    # 研究口径标签（不做 execution mask）+ 可交易池
    fwd = build_forward_return(prices, open_, HOLD, masks=masks, apply_exec_mask=False)
    tradable = build_ic_tradability_mask(
        prices,
        volume=volume,
        masks=masks,
        stock_names=load_stock_names(),
        listing_dates=load_listing_dates(),
        delist_dates=load_delist_dates(),
        is_st_current=load_is_st_current(),
        st_history=load_st_history(),
        tradable_limit_mode="research",
    )

    hist_ic = pd.read_csv(OLD_IC, index_col=0).iloc[:, 0]
    hist_ic.index = pd.to_datetime(hist_ic.index)
    hist_ic = hist_ic.astype(float)
    hist_1y = hist_ic.loc[hist_ic.index >= (data_end - pd.Timedelta(days=365))]
    mu = float(hist_ic.mean())
    sig = float(hist_ic.std())
    mu_1y = float(hist_1y.mean()) if len(hist_1y) else float("nan")
    sig_1y = float(hist_1y.std()) if len(hist_1y) else float("nan")
    logger.info(
        f"历史 Rank IC mean={mu:.4f} std={sig:.4f} n={len(hist_ic)} | "
        f"近1年 mean={mu_1y:.4f} std={sig_1y:.4f} n={len(hist_1y)}"
    )

    rows = []
    for d in target:
        complete, t1, tN = _label_window(idx, d, HOLD)
        if d in extra.index:
            s = extra.loc[d]
            src = "last-window xgb_w156_20260731"
        elif d in old.index:
            s = old.loc[d]
            src = "full-WF scores"
        else:
            logger.warning(f"{d.date()} 无得分，跳过")
            continue
        s = pd.to_numeric(s, errors="coerce")
        s.index = s.index.astype(str).str.zfill(6)

        rec = {
            "signal_date": str(d.date()),
            "score_source": src,
            "label_complete": bool(complete),
            "buy_open": str(t1.date()) if t1 is not None else None,
            "label_close": str(tN.date()) if tN is not None else None,
            "n_scored": int(s.notna().sum()),
            "rank_ic": np.nan,
            "pearson_ic": np.nan,
            "n_ic": 0,
            "hist_ic_stored": float(hist_ic.loc[d]) if d in hist_ic.index else np.nan,
            "pctile_all": np.nan,
            "pctile_1y": np.nan,
            "below_mean_minus_1s": None,
            "below_1y_mean_minus_1s": None,
            "top100_fwd": np.nan,
            "ew_fwd": np.nan,
            "top100_vs_ew": np.nan,
            "n_top": 0,
            "n_ew": 0,
            "note": "",
        }
        if not complete:
            rec["note"] = (
                f"数据末日 {data_end.date()}，h=5 标签未满"
                f"（需 close[t+5]，最早约再过 5 个交易日）"
            )
            rows.append(rec)
            logger.info(f"{d.date()} 出分但无完整标签，不计算 IC")
            continue

        y = fwd.loc[d] if d in fwd.index else pd.Series(dtype=float)
        y = pd.to_numeric(y, errors="coerce")
        y.index = y.index.astype(str).str.zfill(6)
        tmask = tradable.loc[d] if d in tradable.index else pd.Series(dtype=bool)
        tmask.index = tmask.index.astype(str).str.zfill(6)
        common = s.dropna().index.intersection(y.dropna().index)
        if len(tmask):
            trad_ok = tmask.reindex(common).fillna(False)
            trad_ok = trad_ok.astype(bool)
            common = common[trad_ok.reindex(common).fillna(False).astype(bool).values]
        # 无 B（双保险）
        common = pd.Index([c for c in common if not is_b_share_code(c)])
        rec["n_ic"] = int(len(common))
        if len(common) < 30:
            rec["note"] = f"可交易交集仅 {len(common)}，跳过 IC"
            rows.append(rec)
            continue

        sv = s.reindex(common).astype(float)
        yv = y.reindex(common).astype(float)
        ric = spearman_ic(sv.values, yv.values)
        pic = pearson_ic(sv.values, yv.values)
        rec["rank_ic"] = float(ric) if ric is not None else np.nan
        rec["pearson_ic"] = float(pic) if pic is not None else np.nan
        rec["pctile_all"] = _pctile(hist_ic, rec["rank_ic"])
        rec["pctile_1y"] = _pctile(hist_1y, rec["rank_ic"])
        if np.isfinite(rec["rank_ic"]) and np.isfinite(mu) and np.isfinite(sig) and sig > 0:
            rec["below_mean_minus_1s"] = bool(rec["rank_ic"] < (mu - sig))
        if np.isfinite(rec["rank_ic"]) and np.isfinite(mu_1y) and np.isfinite(sig_1y) and sig_1y > 0:
            rec["below_1y_mean_minus_1s"] = bool(rec["rank_ic"] < (mu_1y - sig_1y))

        ranked = sv.sort_values(ascending=False)
        top = ranked.head(TOP_N)
        rec["n_top"] = int(len(top))
        rec["n_ew"] = int(len(yv))
        rec["top100_fwd"] = float(yv.reindex(top.index).mean())
        rec["ew_fwd"] = float(yv.mean())
        rec["top100_vs_ew"] = rec["top100_fwd"] - rec["ew_fwd"]
        rec["note"] = "标签已满；Rank IC=训练口径 Spearman"
        rows.append(rec)
        logger.info(
            f"{d.date()} RankIC={rec['rank_ic']:.4f} Pearson={rec['pearson_ic']:.4f} "
            f"n={rec['n_ic']} Top100-EW={rec['top100_vs_ew']:.4f}"
        )

    ic_df = pd.DataFrame(rows)
    ic_path = OUT / "recent_ic.csv"
    ic_df.to_csv(ic_path, index=False, encoding="utf-8-sig")
    logger.info(f"IC 表 → {ic_path}")

    # 8/17 周一 live 出分：标签未满，明确不写 IC
    live_817 = OUT.parent / "xgb_h5_sizeind_w156_nob_20260817" / "candidates_20260817.csv"
    note_817 = ""
    if live_817.exists():
        note_817 = (
            "另：2026-08-17（周一）曾 live 出分，但非 W-FRI 调仓日，"
            "且 h=5 标签在数据末日 8/21 尚未满（t+5≈8/24），不计算 IC。"
        )

    realized = ic_df[ic_df["label_complete"] & ic_df["rank_ic"].notna()].copy()
    latest = realized.iloc[-1] if len(realized) else None

    lines = []
    lines.append("旗舰最近一周 Rank IC（不重训、不全量回测）")
    lines.append("模型：xgb_w156_20260731  last-window；h=5 W-FRI；size_industry PIT；无 B")
    lines.append("IC 口径：与 models/wf/metrics.spearman_ic 相同 = Rank IC（Spearman）")
    lines.append("标签：close[t+5]/open[t+1]-1，research 可交易池（ST/停牌/次新/退市/无B，信号日保留涨跌停）")
    lines.append("cs_rank 训练不 winsor 标签，故 Rank IC 对 raw fwd 与 cs_rank(y) 等价")
    lines.append(f"数据末日：{data_end.date()}")
    lines.append(f"历史训练 IC：mean={_fmt_pct(mu, 2)}  ICIR={mu / sig:.2f}  n={len(hist_ic)}"
                 if sig else f"历史训练 IC：mean={_fmt_pct(mu, 2)}")
    lines.append(f"近1年 IC：mean={_fmt_pct(mu_1y, 2)}  std={_fmt_pct(sig_1y, 2)}  n={len(hist_1y)}")
    lines.append("")

    if latest is not None:
        lines.append(
            f"最新可算 IC 的信号日：{latest['signal_date']}  "
            f"Rank IC={_fmt_pct(latest['rank_ic'], 2)}  "
            f"（vs 历史均值 {_fmt_pct(mu, 2)}，差 {_fmt_pct(latest['rank_ic'] - mu, 2)}）"
        )
        lines.append(
            f"  Pearson IC={_fmt_pct(latest['pearson_ic'], 2)}  n={int(latest['n_ic'])}  "
            f"近1年分位={latest['pctile_1y'] * 100:.0f}%  "
            f"全样本分位={latest['pctile_all'] * 100:.0f}%"
        )
        if latest.get("below_1y_mean_minus_1s"):
            lines.append("  低于近1年均值−1σ：是")
        else:
            lines.append("  低于近1年均值−1σ：否")
        lines.append(
            f"  Top100 一周收益={_fmt_pct(latest['top100_fwd'], 2)}  "
            f"等权={_fmt_pct(latest['ew_fwd'], 2)}  "
            f"超额={_fmt_pct(latest['top100_vs_ew'], 2)}"
        )
    else:
        lines.append("最新可算 IC 的信号日：无（标签均未满）")

    lines.append("")
    lines.append("各期（含尚未满标签）")
    lines.append(
        f"{'信号日':<12} {'标签':<6} {'买开':<12} {'卖收':<12} "
        f"{'RankIC':>8} {'Pearson':>8} {'vs8.0':>8} {'1y分位':>8} "
        f"{'Top100':>8} {'EW':>8} {'超额':>8} 来源"
    )
    for _, r in ic_df.iterrows():
        lab = "已满" if r["label_complete"] else "未满"
        ric = _fmt_pct(r["rank_ic"], 2) if r["label_complete"] else "—"
        pic = _fmt_pct(r["pearson_ic"], 2) if r["label_complete"] else "—"
        vs = _fmt_pct(r["rank_ic"] - mu, 2) if r["label_complete"] and np.isfinite(r["rank_ic"]) else "—"
        p1 = f"{r['pctile_1y'] * 100:.0f}%" if r["label_complete"] and np.isfinite(r["pctile_1y"]) else "—"
        t100 = _fmt_pct(r["top100_fwd"], 2) if r["label_complete"] else "—"
        ew = _fmt_pct(r["ew_fwd"], 2) if r["label_complete"] else "—"
        xs = _fmt_pct(r["top100_vs_ew"], 2) if r["label_complete"] else "—"
        lines.append(
            f"{r['signal_date']:<12} {lab:<6} {str(r['buy_open'] or '—'):<12} "
            f"{str(r['label_close'] or '—'):<12} {ric:>8} {pic:>8} {vs:>8} {p1:>8} "
            f"{t100:>8} {ew:>8} {xs:>8} {r['score_source']}"
        )

    lines.append("")
    lines.append("8/21 为何还没有 IC：")
    lines.append("  信号日 2026-08-21（周五）→ 次日开盘买入约 8/24，h=5 卖出收盘约 8/28。")
    lines.append("  行情末日是 8/21，close[t+5] 不存在，不能算这期截面 IC。")
    if note_817:
        lines.append(note_817)

    # 一句话结论：对齐 8/14
    row_814 = ic_df[ic_df["signal_date"] == "2026-08-14"]
    row_807 = ic_df[ic_df["signal_date"] == "2026-08-07"]
    verdict = "无法判断（8/14 无完整 IC）"
    if len(row_814) and bool(row_814.iloc[0]["label_complete"]) and np.isfinite(row_814.iloc[0]["rank_ic"]):
        ic814 = float(row_814.iloc[0]["rank_ic"])
        ic807 = float(row_807.iloc[0]["rank_ic"]) if len(row_807) and np.isfinite(row_807.iloc[0]["rank_ic"]) else np.nan
        # 「掉 IC」：明显低于历史均值，或跌破均值−1σ
        dropped = ic814 < (mu - 0.5 * sig) if np.isfinite(sig) else ic814 < mu * 0.5
        hard = bool(row_814.iloc[0].get("below_mean_minus_1s"))
        if hard:
            verdict = (
                f"新一周（8/14）Rank IC={_fmt_pct(ic814, 2)}，低于历史均值−1σ"
                f"（{_fmt_pct(mu - sig, 2)}），相对训练 8.0% 明显掉了。"
            )
        elif dropped:
            verdict = (
                f"新一周（8/14）Rank IC={_fmt_pct(ic814, 2)}，低于训练均值 {_fmt_pct(mu, 2)}，"
                f"未破 1σ，算走弱但未崩。"
            )
        else:
            verdict = (
                f"新一周（8/14）Rank IC={_fmt_pct(ic814, 2)}，仍贴近训练均值 {_fmt_pct(mu, 2)}，"
                f"没有掉出历史正常带。"
            )
        if np.isfinite(ic807):
            verdict += f" 前一期 8/07 为 {_fmt_pct(ic807, 2)}。"
    lines.append("")
    lines.append("一句话：" + verdict)
    lines.append("")
    lines.append(f"log: {log_path}")
    lines.append(f"表: {ic_path}")
    lines.append("未覆盖 candidates_20260821.csv")

    text = "\n".join(lines) + "\n"
    report_path = OUT / "recent_ic_report.txt"
    report_path.write_text(text, encoding="utf-8")
    summary = {
        "data_end": str(data_end.date()),
        "ic_metric": "Rank IC (Spearman), models.wf.metrics.spearman_ic",
        "pearson_also": True,
        "label": "close[t+5]/open[t+1]-1",
        "tradable": "research (ST/halt/ipo252/delist/no B; keep limit on signal day)",
        "hist_mean": mu,
        "hist_std": sig,
        "hist_icir": (mu / sig) if sig else None,
        "hist_1y_mean": mu_1y,
        "hist_1y_std": sig_1y,
        "verdict": verdict,
        "rows": rows,
    }
    (OUT / "recent_ic_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    print(text)
    logger.info("完成")


if __name__ == "__main__":
    main()
