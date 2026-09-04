"""research/diagnose_sparse_factors.py — 稀疏因子诊断与归类建议

对当前因子库（`factors.factor.get_factor_names` 枚举的 ML 输入池 + 事件 overlay
池）逐个评估：

  * 每个调仓日的非 NaN 数 / 全市场截面大小 → sparsity ratio
  * 跨调仓日 sparsity 的中位数 / 分布
  * 在「有值的调仓日」上计算的 Spearman IC 稳定性（mean IC, ICIR, t）
  * 推荐：ml_input / event_overlay / drop

判定规则（`--sparse-threshold` 默认 0.30）：
  * 中位数 sparsity ≥ threshold → ml_input（与现有 ML 截面输入框架兼容）
  * 中位数 sparsity < threshold 且 |IC_mean| 在 populated dates 上显著
    （|mean| ≥ `--ic-mean-min`，默认 0.02）→ event_overlay
  * 中位数 sparsity < threshold 且 IC 接近 0 → drop

输出：
  * `research/output/sparse_factor_report.md` — Markdown 报告
  * `research/output/sparse_factor_report.csv` — CSV 表格

用法
----
    # 全市场诊断（首次运行会触发因子计算 / 缓存写入，耗时较长）
    python -m research.diagnose_sparse_factors --horizon 5

    # 快速冒烟（取前 100 只股票，仅调仓日抽样）
    python -m research.diagnose_sparse_factors --horizon 5 --sample 100

    # 自定义稀疏阈值
    python -m research.diagnose_sparse_factors --horizon 5 --sparse-threshold 0.20

注意：本工具不做 IC 筛选 / 回测，只评估稀疏性 + populated-date IC 稳定性，
供人工决定每个因子应进 ML 截面输入、做事件 overlay、还是丢弃。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# 确保仓库根在 sys.path（python -m research.diagnose_sparse_factors 时本应已就绪，
# 但脚本直接运行时兜底）
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config.settings import RAW_DIR
from factors.factor import (
    EVENT_OVERLAY_FACTOR_NAMES,
    _ALPHA2_FACTOR_NAMES,
    _FIN_FACTOR_NAMES,
    _LIMIT_FACTOR_NAMES,
    _PRICE_FACTOR_NAMES,
    _SMALLCAP_FACTOR_NAMES,
    _TECH_FACTOR_NAMES,
    get_event_overlay_factors,
    get_factor_names,
    iter_factor_registry,
)
from research.ic.forward_return import build_forward_return
from research.ic.ic_series import compute_ic_series
from research.ic.load_data import load_ic_data
from utils.rebalance_dates import get_rebalance_dates, horizon_to_rebalance_freq


# ── 因子分类（用于报告展示；与 factor.py 内部 frozenset 同步）──────────────────
def _factor_category(name: str) -> str:
    if name in _PRICE_FACTOR_NAMES:
        return "price"
    if name in _FIN_FACTOR_NAMES:
        return "financial"
    if name in _ALPHA2_FACTOR_NAMES:
        return "alpha2"
    if name in _TECH_FACTOR_NAMES:
        return "technical"
    if name in _LIMIT_FACTOR_NAMES:
        return "limit"
    if name in _SMALLCAP_FACTOR_NAMES:
        return "smallcap"
    if name in EVENT_OVERLAY_FACTOR_NAMES:
        return "event_overlay"
    if name.startswith("WQ_"):
        return "alpha101"
    if name.startswith("市场") or name.startswith("HMM_"):
        return "regime"
    return "other"


def _classify_recommendation(
    median_sparsity: float,
    ic_mean: float,
    ic_t: float,
    sparse_threshold: float,
    ic_mean_min: float,
    ic_t_min: float,
) -> str:
    """根据稀疏度 + populated-date IC 稳定性给出推荐归类。"""
    if median_sparsity >= sparse_threshold:
        return "ml_input"
    # 稀疏因子：看 populated-date IC 是否有信号
    if abs(ic_mean) >= ic_mean_min and abs(ic_t) >= ic_t_min:
        return "event_overlay"
    return "drop"


def _sparsity_per_date(panel: pd.DataFrame, rebalance_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """对每个调仓日计算 (non_nan, universe_size, sparsity_ratio)。

    universe_size 取该调仓日 panel 列数（全市场截面大小，含全 NaN 列），
    sparsity_ratio = non_nan / universe_size。
    """
    sub = panel.loc[panel.index.intersection(rebalance_dates)]
    universe_size = sub.shape[1]
    non_nan = sub.notna().sum(axis=1)
    ratio = non_nan / universe_size if universe_size > 0 else non_nan * 0
    return pd.DataFrame({"non_nan": non_nan, "universe_size": universe_size, "sparsity": ratio})


def _ic_on_populated(
    panel: pd.DataFrame,
    forward_return: pd.DataFrame,
    sparse_threshold: float,
    min_stocks: int = 30,
) -> tuple[float, float, float, int]:
    """在「有值的调仓日」上计算 Spearman IC 稳定性统计。

    只保留 non_nan ≥ max(min_stocks, 30) 的调仓日，避免极少数样本噪声主导 IC。
    返回 (ic_mean, ic_ir, ic_t, n_dates_used)。
    """
    # 复用 compute_ic_series（含 tradable mask=None 路径），它在内部按
    # MIN_IC_STOCKS 过滤；这里额外用 sparsity 过滤，仅保留 panel 当日有足够样本的日期。
    ic = compute_ic_series(panel, forward_return, tradable=None, min_stocks=min_stocks)
    if ic.empty:
        return (np.nan, np.nan, np.nan, 0)
    # 进一步过滤：只保留 panel 当日非 NaN 数 ≥ min_stocks 的日期
    valid_count = panel.notna().sum(axis=1).reindex(ic.index).fillna(0)
    keep = valid_count >= min_stocks
    ic = ic.where(keep).dropna()
    if ic.empty:
        return (np.nan, np.nan, np.nan, 0)
    ic_mean = float(ic.mean())
    ic_std = float(ic.std(ddof=1))
    ic_ir = ic_mean / ic_std if ic_std > 0 else np.nan
    ic_t = ic_mean / (ic_std / np.sqrt(len(ic))) if ic_std > 0 else np.nan
    return (ic_mean, ic_ir, ic_t, int(len(ic)))


def _subset_columns(df: pd.DataFrame, sample: int | None) -> pd.DataFrame:
    """取前 sample 列（按列顺序），用于 --sample 快速冒烟。"""
    if df is None or sample is None or sample <= 0:
        return df
    cols = list(df.columns[:sample])
    return df[cols]


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="稀疏因子诊断：评估每个因子的 sparsity + populated-date IC，"
                    "推荐 ml_input / event_overlay / drop。",
    )
    p.add_argument("--horizon", type=int, default=5,
                   help="持有期（交易日），用于 forward_return + 调仓频率（默认 5）")
    p.add_argument("--sparse-threshold", type=float, default=0.30,
                   help="中位数 sparsity < threshold 判为稀疏（默认 0.30）")
    p.add_argument("--ic-mean-min", type=float, default=0.02,
                   help="稀疏因子判为 event_overlay 的 |IC mean| 下限（默认 0.02）")
    p.add_argument("--ic-t-min", type=float, default=2.0,
                   help="稀疏因子判为 event_overlay 的 |IC t| 下限（默认 2.0）")
    p.add_argument("--sample", type=int, default=None,
                   help="仅取前 N 只股票列做快速冒烟（默认全市场）")
    p.add_argument("--include-regime", action="store_true", default=False,
                   help="纳入市场/HMM regime 特征（默认排除——regime 是 TS-zscore 全市场单值，"
                        "sparsity 语义不适用）")
    p.add_argument("--out-dir", type=str, default="research/output",
                   help="报告输出目录（默认 research/output）")
    p.add_argument("--max-dates", type=int, default=None,
                   help="仅取前 N 个调仓日做快速冒烟（默认全部）")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"加载数据 (sample={args.sample})...")
    bundle = load_ic_data()

    # 子集化股票列（冒烟）
    prices = _subset_columns(bundle.prices, args.sample)
    financial = bundle.financial
    if financial is not None and args.sample is not None:
        # financial 是长表，按 code 子集过滤
        try:
            codes = set(prices.columns)
            financial = financial[financial["code"].isin(codes)] if "code" in financial.columns else financial
        except Exception:
            pass
    prices_raw = _subset_columns(bundle.prices_raw, args.sample)
    volume = _subset_columns(bundle.volume, args.sample)
    amount = _subset_columns(bundle.amount, args.sample)
    open_ = _subset_columns(bundle.open_, args.sample)
    high = _subset_columns(bundle.high, args.sample)
    low = _subset_columns(bundle.low, args.sample)
    margin = _subset_columns(bundle.margin, args.sample)
    moneyflow = _subset_columns(bundle.moneyflow, args.sample)
    northbound = _subset_columns(bundle.northbound, args.sample)
    institution = _subset_columns(bundle.institution, args.sample)
    market_prices = bundle.market_prices  # 市场指数是单时序，不子集
    industry_map = bundle.industry_map_df
    clean_ret = _subset_columns(bundle.clean_ret, args.sample)
    masks = bundle.masks

    # 调仓日 + forward_return
    rebalance_freq = horizon_to_rebalance_freq(args.horizon)
    rebalance_dates = get_rebalance_dates(pd.DatetimeIndex(prices.index), rebalance_freq)
    if args.max_dates is not None and args.max_dates > 0:
        rebalance_dates = rebalance_dates[: args.max_dates]
    logger.info(f"调仓日: {len(rebalance_dates)} 个 (freq={rebalance_freq}, horizon={args.horizon})")

    forward_return = build_forward_return(prices, open_, args.horizon, masks=masks)

    # 枚举 ML 输入因子名
    ml_names = get_factor_names(
        prices=prices, financial=financial, prices_raw=prices_raw,
        volume=volume, amount=amount, open_=open_, high=high, low=low,
        clean_ret=clean_ret, masks=masks, market_prices=market_prices,
        industry_map=industry_map, margin=margin, moneyflow=moneyflow,
        northbound=northbound, institution=institution,
        factor_names=None, walk_forward_hmm=False,
        include_regime=args.include_regime,
    )
    # 排除 regime（除非 --include-regime）
    if not args.include_regime:
        ml_names = [n for n in ml_names
                    if not (n.startswith("市场") or n.startswith("HMM_"))]

    logger.info(f"ML 输入候选因子: {len(ml_names)} 个；事件 overlay 因子: "
                f"{len(EVENT_OVERLAY_FACTOR_NAMES)} 个 → {sorted(EVENT_OVERLAY_FACTOR_NAMES)}")

    rows: list[dict] = []

    # ── 1) ML 输入池因子：iter_factor_registry（带缓存）逐个评估 ──
    computed: dict[str, pd.DataFrame] = {}
    for name, panel in iter_factor_registry(
        prices=prices, financial=financial, prices_raw=prices_raw,
        volume=volume, amount=amount, open_=open_, high=high, low=low,
        clean_ret=clean_ret, masks=masks, market_prices=market_prices,
        industry_map=industry_map, margin=margin, moneyflow=moneyflow,
        northbound=northbound, institution=institution,
        factor_names=set(ml_names), walk_forward_hmm=False,
        include_regime=args.include_regime,
    ):
        computed[name] = panel

    logger.info(f"已计算 {len(computed)} 个 ML 输入因子面板，开始稀疏 + IC 评估...")

    for name in ml_names:
        panel = computed.get(name)
        if panel is None:
            rows.append({
                "factor": name,
                "category": _factor_category(name),
                "n_rebalance_dates": 0,
                "median_sparsity": np.nan,
                "mean_sparsity": np.nan,
                "p10_sparsity": np.nan,
                "p90_sparsity": np.nan,
                "ic_mean_populated": np.nan,
                "ic_ir_populated": np.nan,
                "ic_t_populated": np.nan,
                "ic_dates_used": 0,
                "recommendation": "drop",
                "note": "panel 未生成（数据缺失或计算失败）",
            })
            continue
        sp = _sparsity_per_date(panel, rebalance_dates)
        if sp.empty:
            med_sp = np.nan
        else:
            med_sp = float(sp["sparsity"].median())
        ic_mean, ic_ir, ic_t, ic_n = _ic_on_populated(panel, forward_return, args.sparse_threshold)
        rec = _classify_recommendation(
            med_sp if not np.isnan(med_sp) else 0.0,
            ic_mean if not np.isnan(ic_mean) else 0.0,
            ic_t if not np.isnan(ic_t) else 0.0,
            args.sparse_threshold, args.ic_mean_min, args.ic_t_min,
        )
        rows.append({
            "factor": name,
            "category": _factor_category(name),
            "n_rebalance_dates": int(len(sp)),
            "median_sparsity": med_sp,
            "mean_sparsity": float(sp["sparsity"].mean()) if not sp.empty else np.nan,
            "p10_sparsity": float(sp["sparsity"].quantile(0.10)) if not sp.empty else np.nan,
            "p90_sparsity": float(sp["sparsity"].quantile(0.90)) if not sp.empty else np.nan,
            "ic_mean_populated": ic_mean,
            "ic_ir_populated": ic_ir,
            "ic_t_populated": ic_t,
            "ic_dates_used": ic_n,
            "recommendation": rec,
            "note": "",
        })

    # ── 2) 事件 overlay 因子：单独计算（不在 ML 输入枚举内）──
    overlay_panels = get_event_overlay_factors(prices, factor_names=None)
    for name in EVENT_OVERLAY_FACTOR_NAMES:
        panel = overlay_panels.get(name)
        if panel is None:
            rows.append({
                "factor": name,
                "category": _factor_category(name),
                "n_rebalance_dates": 0,
                "median_sparsity": np.nan,
                "mean_sparsity": np.nan,
                "p10_sparsity": np.nan,
                "p90_sparsity": np.nan,
                "ic_mean_populated": np.nan,
                "ic_ir_populated": np.nan,
                "ic_t_populated": np.nan,
                "ic_dates_used": 0,
                "recommendation": "event_overlay",
                "note": "事件 overlay 因子（数据缺失或计算失败）",
            })
            continue
        sp = _sparsity_per_date(panel, rebalance_dates)
        med_sp = float(sp["sparsity"].median()) if not sp.empty else np.nan
        ic_mean, ic_ir, ic_t, ic_n = _ic_on_populated(panel, forward_return, args.sparse_threshold)
        # 事件 overlay 因子默认推荐 event_overlay（除非 populated-date IC 完全为 0
        # 且稀疏度也极低 → drop）
        if abs(ic_mean if not np.isnan(ic_mean) else 0.0) < args.ic_mean_min and ic_n == 0:
            rec = "drop"
        else:
            rec = "event_overlay"
        rows.append({
            "factor": name,
            "category": _factor_category(name),
            "n_rebalance_dates": int(len(sp)),
            "median_sparsity": med_sp,
            "mean_sparsity": float(sp["sparsity"].mean()) if not sp.empty else np.nan,
            "p10_sparsity": float(sp["sparsity"].quantile(0.10)) if not sp.empty else np.nan,
            "p90_sparsity": float(sp["sparsity"].quantile(0.90)) if not sp.empty else np.nan,
            "ic_mean_populated": ic_mean,
            "ic_ir_populated": ic_ir,
            "ic_t_populated": ic_t,
            "ic_dates_used": ic_n,
            "recommendation": rec,
            "note": "事件 overlay 因子（不在 ML 截面输入池）",
        })

    # ── 输出 ──
    df = pd.DataFrame(rows).sort_values("median_sparsity", ascending=True, na_position="last")
    csv_path = out_dir / "sparse_factor_report.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"CSV 报告: {csv_path}")

    md_path = out_dir / "sparse_factor_report.md"
    _write_markdown(df, md_path, args)
    logger.info(f"Markdown 报告: {md_path}")

    # 控制台摘要
    sparse_flagged = df[df["median_sparsity"] < args.sparse_threshold]
    print("\n═══════════════════════════════════════════════════════════════")
    print(f"稀疏因子诊断完成（threshold={args.sparse_threshold}, horizon={args.horizon}）")
    print(f"  评估因子总数: {len(df)}")
    print(f"  稀疏因子 (median sparsity < {args.sparse_threshold}): {len(sparse_flagged)}")
    if len(sparse_flagged):
        print("\n  稀疏因子明细 (top 20 by sparsity asc):")
        for _, r in sparse_flagged.head(20).iterrows():
            print(f"    {r['factor']:<25s} cat={r['category']:<12s} "
                  f"med_sp={r['median_sparsity']:.3f} "
                  f"ic_mean={r['ic_mean_populated'] if not np.isnan(r['ic_mean_populated']) else float('nan'):+.4f} "
                  f"→ {r['recommendation']}")
    rec_counts = df["recommendation"].value_counts().to_dict()
    print(f"\n  推荐分布: {rec_counts}")
    print(f"  报告: {md_path} / {csv_path}")
    print("═══════════════════════════════════════════════════════════════")
    return 0


def _write_markdown(df: pd.DataFrame, path: Path, args) -> None:
    """写 Markdown 报告。"""
    sparse_thr = args.sparse_threshold
    flagged = df[df["median_sparsity"] < sparse_thr]
    rec_counts = df["recommendation"].value_counts().to_dict()

    lines: list[str] = []
    lines.append("# 稀疏因子诊断报告")
    lines.append("")
    lines.append(f"- 生成参数: `--horizon {args.horizon}` "
                 f"`--sparse-threshold {sparse_thr}` "
                 f"`--ic-mean-min {args.ic_mean_min}` "
                 f"`--ic-t-min {args.ic_t_min}` "
                 f"`--sample {args.sample if args.sample else '全市场'}` "
                 f"`--include-regime {args.include_regime}`")
    lines.append(f"- 评估因子总数: {len(df)}")
    lines.append(f"- 稀疏因子数 (median sparsity < {sparse_thr}): {len(flagged)}")
    lines.append(f"- 推荐分布: {rec_counts}")
    lines.append("")
    lines.append("## 判定规则")
    lines.append("")
    lines.append(f"- `median_sparsity ≥ {sparse_thr}` → **ml_input**（与现有 ML 截面输入框架兼容）")
    lines.append(f"- `median_sparsity < {sparse_thr}` 且 `|IC mean| ≥ {args.ic_mean_min}` 且 "
                 f"`|IC t| ≥ {args.ic_t_min}`（populated dates）→ **event_overlay**")
    lines.append(f"- `median_sparsity < {sparse_thr}` 且 IC 接近 0 → **drop**")
    lines.append("")
    lines.append("## 稀疏因子明细（按 median sparsity 升序）")
    lines.append("")
    if flagged.empty:
        lines.append("_(无稀疏因子)_")
    else:
        lines.append("| factor | category | median_sparsity | mean_sparsity | "
                     "ic_mean_pop | ic_ir_pop | ic_t_pop | ic_dates | rec | note |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for _, r in flagged.iterrows():
            lines.append(
                f"| {r['factor']} | {r['category']} | "
                f"{r['median_sparsity']:.4f} | {r['mean_sparsity']:.4f} | "
                f"{r['ic_mean_populated'] if not np.isnan(r['ic_mean_populated']) else float('nan'):+.4f} | "
                f"{r['ic_ir_populated'] if not np.isnan(r['ic_ir_populated']) else float('nan'):+.3f} | "
                f"{r['ic_t_populated'] if not np.isnan(r['ic_t_populated']) else float('nan'):+.2f} | "
                f"{r['ic_dates_used']} | **{r['recommendation']}** | {r['note']} |"
            )
    lines.append("")
    lines.append("## 全因子表（按 median sparsity 升序，前 50）")
    lines.append("")
    lines.append("| factor | category | median_sparsity | mean_sparsity | "
                 "ic_mean_pop | ic_ir_pop | ic_t_pop | ic_dates | rec |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in df.head(50).iterrows():
        lines.append(
            f"| {r['factor']} | {r['category']} | "
            f"{r['median_sparsity']:.4f} | {r['mean_sparsity']:.4f} | "
            f"{r['ic_mean_populated'] if not np.isnan(r['ic_mean_populated']) else float('nan'):+.4f} | "
            f"{r['ic_ir_populated'] if not np.isnan(r['ic_ir_populated']) else float('nan'):+.3f} | "
            f"{r['ic_t_populated'] if not np.isnan(r['ic_t_populated']) else float('nan'):+.2f} | "
            f"{r['ic_dates_used']} | {r['recommendation']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
