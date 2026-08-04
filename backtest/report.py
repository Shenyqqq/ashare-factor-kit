"""
backtest/report.py — charts and CSV helpers for quantile backtest results.

Functions mirror backtest/quantile.py so run.py can import from one place
when using the v2 engine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from backtest.quantile import QuantileResult
from backtest.risk_metrics import (
    compute_risk_metrics,
    export_risk_metrics,
    format_risk_metrics_table,
)


def plot_quantile_result(
    result: QuantileResult,
    title: str = "分组回测（Q1-Q5）",
    save_path: str = None,
    rebalance_freq: str = "ME",
    rf: float = 0.0,
):
    """4-panel chart: NAV, long-short, calendar-year bars, annualized-return monotonicity.

    rebalance_freq / rf 用于年化收益条形图与 Top-N / benchmark 净值子图标注
    （Sharpe/Sortino/最大回撤）；年化口径与 risk_metrics.compute_risk_metrics 一致。
    """
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.family"] = "SimHei"
    matplotlib.rcParams["axes.unicode_minus"] = False

    q_labels = [c for c in result.nav.columns if c.startswith("Q")]
    top_label = next((c for c in result.nav.columns if c.startswith("Top")), None)
    n = len(q_labels)
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, n))

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    ax = axes[0, 0]
    for i, q in enumerate(q_labels):
        result.nav[q].plot(ax=ax, color=colors[i], lw=1.2, alpha=0.7, label=q)
    if top_label and top_label in result.nav.columns:
        result.nav[top_label].plot(ax=ax, color="darkorange", lw=2.2, label=top_label, zorder=5)
    if "benchmark" in result.nav.columns:
        result.nav["benchmark"].plot(ax=ax, color="gray", lw=1.5, ls="--", label="等权基准")
    for idx_name, style in {"沪深300": ("black", "-.", 1.8), "创业板指": ("purple", ":", 1.8)}.items():
        if idx_name in result.nav.columns:
            result.nav[idx_name].plot(ax=ax, color=style[0], ls=style[1], lw=style[2], label=idx_name)
    ax.set_title("各分组净值走势（橙色=实操Top-N）")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylabel("净值")

    # 风险指标标注（Top-N / benchmark）；与右下角年化条形图共用同一 rm
    rm = None
    try:
        rm = compute_risk_metrics(
            result.nav, rebalance_freq=rebalance_freq, rf=rf,
        )
        annotate_cols = [c for c in (top_label, "benchmark") if c]
        ann_parts = []
        for c in annotate_cols:
            if c in rm.index and not np.isnan(rm.loc[c, "Sharpe"]):
                sharpe = rm.loc[c, "Sharpe"]
                sortino = rm.loc[c, "Sortino"]
                mdd = abs(rm.loc[c, "最大回撤"]) * 100
                sortino_str = f"{sortino:.2f}" if not np.isnan(sortino) else "  nan"
                ann_parts.append(
                    f"{c}: Sharpe={sharpe:.2f}  Sortino={sortino_str}  "
                    f"最大回撤={mdd:.1f}%"
                )
        if ann_parts:
            ax.text(
                0.02, 0.02, "\n".join(ann_parts),
                transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.85),
            )
    except Exception:
        # 标注失败不影响主图
        pass

    ax = axes[0, 1]
    result.long_short_nav.plot(ax=ax, color="steelblue", lw=1.5)
    ax.axhline(1, color="gray", ls="--", lw=0.8)
    ls = result.long_short_nav.copy()
    ax.fill_between(ls.index, 1, ls, where=(ls > 1), alpha=0.3, color="green", label="Q5>Q1")
    ax.fill_between(ls.index, 1, ls, where=(ls < 1), alpha=0.3, color="red", label="Q5<Q1")
    mono = result.ic_monotonicity
    mono_color = "green" if mono > 0.8 else "orange" if mono > 0.5 else "red"
    ax.set_title(f"多空组合 (Q5 - Q1)  |  单调性={mono:.3f}", color=mono_color)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    plot_cols = q_labels + ([top_label] if top_label and top_label in result.annual_returns.columns else [])
    x = np.arange(len(result.annual_returns))
    width = 0.8 / max(len(plot_cols), 1)
    bar_colors = list(colors) + (["darkorange"] if top_label else [])
    for i, col in enumerate(plot_cols):
        if col in result.annual_returns.columns:
            vals = result.annual_returns[col].values
            ax.bar(x + i * width - 0.4 + width / 2, vals * 100, width,
                   label=col, color=bar_colors[i], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(result.annual_returns.index, rotation=45)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("逐年收益（%）")
    ax.legend(fontsize=8, ncol=len(plot_cols))
    ax.grid(alpha=0.3, axis="y")

    # 右下：年化收益条形图（跨 OOS 长度可比；口径=risk_metrics 几何年化）
    ax = axes[1, 1]
    if rm is None:
        try:
            rm = compute_risk_metrics(
                result.nav, rebalance_freq=rebalance_freq, rf=rf,
            )
        except Exception:
            rm = None
    bar_labels = list(q_labels)
    bar_cols = list(colors)
    if top_label and top_label in result.nav.columns:
        bar_labels.append(top_label)
        bar_cols.append("darkorange")
    if rm is not None:
        bar_vals = [
            float(rm.loc[lab, "年化收益"]) * 100 if lab in rm.index else float("nan")
            for lab in bar_labels
        ]
    else:
        # 回退：日历年几何年化（与旧摘要表一致）
        n_years = max((result.nav.index[-1] - result.nav.index[0]).days / 365.25, 0.5)
        bar_vals = [
            ((float(result.nav[lab].iloc[-1]) ** (1.0 / n_years)) - 1.0) * 100
            for lab in bar_labels
        ]
    xs = np.arange(len(bar_labels))
    ax.bar(xs, bar_vals, color=bar_cols, alpha=0.85)
    for i, v in enumerate(bar_vals):
        if np.isnan(v):
            continue
        ax.text(i, v + (0.3 if v >= 0 else -0.8), f"{v:.1f}%",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(bar_labels, rotation=20)
    ax.set_ylabel("年化收益（%）")
    if rm is not None and "benchmark" in rm.index and not np.isnan(rm.loc["benchmark", "年化收益"]):
        bm = float(rm.loc["benchmark", "年化收益"]) * 100
        ax.axhline(bm, color="gray", ls="--", lw=1.2, label=f"基准 {bm:.1f}%")
    ax.set_title("各组年化收益（单调性）")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"图表已保存: {save_path}")
    else:
        plt.show()
    plt.close()


def export_holdings(result: QuantileResult, save_path: str = None) -> pd.DataFrame:
    if not result.top_holdings:
        logger.info("无持仓数据")
        return pd.DataFrame()
    rows = [
        {"信号日": d.strftime("%Y-%m-%d"), "标的": " | ".join(stocks), "数量": len(stocks)}
        for d, stocks in sorted(result.top_holdings.items())
    ]
    df = pd.DataFrame(rows).set_index("信号日")
    if save_path:
        df.to_csv(save_path, encoding="utf-8-sig")
        logger.info(f"持仓已保存: {save_path}")
    return df


def export_turnover_detail(result: QuantileResult, save_path: str) -> None:
    if result.turnover_detail is None or result.turnover_detail.empty:
        logger.info("无 turnover 明细")
        return
    result.turnover_detail.to_csv(save_path, index=False, encoding="utf-8-sig")
    logger.info(f"Turnover 明细已保存: {save_path}")


def print_holdings(result: QuantileResult, last_n: int = None):
    if not result.top_holdings:
        logger.info("无持仓数据")
        return
    items = sorted(result.top_holdings.items())
    if last_n:
        items = items[-last_n:]
    n = len(items[0][1]) if items else 0
    lines = [f"{'=' * 60}", f"  Top-{n} 每期选股", f"{'=' * 60}"]
    for sig_date, stocks in items:
        lines.append(f"  [{sig_date.strftime('%Y-%m-%d')}]")
        for j in range(0, len(stocks), 6):
            lines.append("    " + "  ".join(stocks[j:j + 6]))
    logger.info("\n" + "\n".join(lines))


def print_quantile_summary(
    result: QuantileResult,
    rebalance_freq: str = "ME",
    rf: float = 0.0,
):
    q_labels = [c for c in result.nav.columns if c.startswith("Q")]
    top_label = next((c for c in result.nav.columns if c.startswith("Top")), None)

    # 年化优先（与图表 / risk_metrics 同口径）；累计仅作辅列
    rm = None
    try:
        rm = compute_risk_metrics(
            result.nav, rebalance_freq=rebalance_freq, rf=rf,
        )
    except Exception:
        pass

    def _ann_pct(col: str) -> float:
        if rm is not None and col in rm.index and not np.isnan(rm.loc[col, "年化收益"]):
            return float(rm.loc[col, "年化收益"]) * 100
        n_years = max((result.nav.index[-1] - result.nav.index[0]).days / 365.25, 0.5)
        return ((float(result.nav[col].iloc[-1]) ** (1.0 / n_years)) - 1.0) * 100

    def _mdd_pct(col: str) -> float:
        if rm is not None and col in rm.index and not np.isnan(rm.loc[col, "最大回撤"]):
            return abs(float(rm.loc[col, "最大回撤"])) * 100
        return float("nan")

    lines = [
        "=" * 60,
        "  分组回测摘要 (v2 engine)",
        "=" * 60,
        f"  {'组别':<10} {'年化收益':>10} {'累计收益':>10} {'最大回撤':>10}",
        "-" * 60,
    ]
    for q in q_labels:
        cum = (result.nav[q].iloc[-1] - 1) * 100
        mdd = _mdd_pct(q)
        mdd_s = f"{mdd:>9.1f}%" if not np.isnan(mdd) else f"{'nan':>10}"
        lines.append(f"  {q:<10} {_ann_pct(q):>9.1f}%  {cum:>9.1f}%  {mdd_s}")
    if top_label and top_label in result.nav.columns:
        top_cum = (result.nav[top_label].iloc[-1] - 1) * 100
        mdd = _mdd_pct(top_label)
        mdd_s = f"{mdd:>9.1f}%" if not np.isnan(mdd) else f"{'nan':>10}"
        lines.append(
            f"  {top_label:<10} {_ann_pct(top_label):>9.1f}%  {top_cum:>9.1f}%  "
            f"{mdd_s}  ← 实操参考"
        )
    if "benchmark" in result.nav.columns:
        bm = (result.nav["benchmark"].iloc[-1] - 1) * 100
        mdd = _mdd_pct("benchmark")
        mdd_s = f"{mdd:>9.1f}%" if not np.isnan(mdd) else f"{'nan':>10}"
        lines.append(
            f"  {'等权基准':<10} {_ann_pct('benchmark'):>9.1f}%  {bm:>9.1f}%  {mdd_s}"
        )

    # 多空：有足够期数时给年化，否则只打累计
    ls_cum = (result.long_short_nav.iloc[-1] - 1) * 100
    try:
        ls_rm = compute_risk_metrics(
            result.long_short_nav.to_frame("LS"),
            rebalance_freq=rebalance_freq, rf=rf,
        )
        ls_ann = float(ls_rm.loc["LS", "年化收益"]) * 100
        lines.append(f"  多空年化: {ls_ann:.1f}%  (累计 {ls_cum:.1f}%)")
    except Exception:
        lines.append(f"  多空累计: {ls_cum:.1f}%")
    lines.append(f"  单调性:   {result.ic_monotonicity:.3f}")
    lines.append("=" * 60)

    # ── 风险调整指标表 ────────────────────────────────────────────────────────
    if rm is not None:
        lines.append(
            "  风险调整指标（年化由调仓频率 {} 推算, rf={:.2%}）".format(
                rebalance_freq, rf
            )
        )
        lines.append("-" * 82)
        lines.append(format_risk_metrics_table(rm))
        lines.append("-" * 82)
    else:
        lines.append("  (风险指标计算失败)")

    logger.info("\n" + "\n".join(lines))
