"""
backtest/report.py — charts and CSV helpers for quantile backtest results.

Functions mirror backtest/quantile.py so run.py can import from one place
when using the v2 engine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.quantile import QuantileResult


def plot_quantile_result(
    result: QuantileResult,
    title: str = "分组回测（Q1-Q5）",
    save_path: str = None,
):
    """4-panel chart: NAV, long-short, annual bars, total return monotonicity."""
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

    ax = axes[1, 1]
    total_rets = (result.nav[q_labels].iloc[-1] - 1) * 100
    ax.bar(q_labels, total_rets.values, color=colors, alpha=0.85)
    for i, v in enumerate(total_rets.values):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    if "benchmark" in result.nav.columns:
        bm = (result.nav["benchmark"].iloc[-1] - 1) * 100
        ax.axhline(bm, color="gray", ls="--", lw=1.2, label=f"基准 {bm:.1f}%")
    ax.set_title("各组总收益（单调性）")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图表已保存: {save_path}")
    else:
        plt.show()
    plt.close()


def export_holdings(result: QuantileResult, save_path: str = None) -> pd.DataFrame:
    if not result.top_holdings:
        print("无持仓数据")
        return pd.DataFrame()
    rows = [
        {"信号日": d.strftime("%Y-%m-%d"), "标的": " | ".join(stocks), "数量": len(stocks)}
        for d, stocks in sorted(result.top_holdings.items())
    ]
    df = pd.DataFrame(rows).set_index("信号日")
    if save_path:
        df.to_csv(save_path, encoding="utf-8-sig")
        print(f"持仓已保存: {save_path}")
    return df


def export_turnover_detail(result: QuantileResult, save_path: str) -> None:
    if result.turnover_detail is None or result.turnover_detail.empty:
        print("无 turnover 明细")
        return
    result.turnover_detail.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"Turnover 明细已保存: {save_path}")


def print_holdings(result: QuantileResult, last_n: int = None):
    if not result.top_holdings:
        print("无持仓数据")
        return
    items = sorted(result.top_holdings.items())
    if last_n:
        items = items[-last_n:]
    n = len(items[0][1]) if items else 0
    print(f"\n{'=' * 60}\n  Top-{n} 每期选股\n{'=' * 60}")
    for sig_date, stocks in items:
        print(f"\n  [{sig_date.strftime('%Y-%m-%d')}]")
        for j in range(0, len(stocks), 6):
            print("    " + "  ".join(stocks[j:j + 6]))


def print_quantile_summary(result: QuantileResult):
    q_labels = [c for c in result.nav.columns if c.startswith("Q")]
    top_label = next((c for c in result.nav.columns if c.startswith("Top")), None)
    total_rets = (result.nav[q_labels].iloc[-1] - 1) * 100
    n_years = (result.nav.index[-1] - result.nav.index[0]).days / 365

    print("\n" + "=" * 60)
    print("  分组回测摘要 (v2 engine)")
    print("=" * 60)
    print(f"  {'组别':<10} {'累计收益':>10} {'年化收益':>10}")
    print("-" * 60)
    for q in q_labels:
        cum = total_rets[q]
        ann = ((1 + cum / 100) ** (1 / max(n_years, 0.5)) - 1) * 100
        print(f"  {q:<10} {cum:>9.1f}%  {ann:>9.1f}%")
    if top_label and top_label in result.nav.columns:
        top_cum = (result.nav[top_label].iloc[-1] - 1) * 100
        top_ann = ((1 + top_cum / 100) ** (1 / max(n_years, 0.5)) - 1) * 100
        print(f"\n  {top_label:<10} {top_cum:>9.1f}%  {top_ann:>9.1f}%  ← 实操参考")
    if "benchmark" in result.nav.columns:
        bm = (result.nav["benchmark"].iloc[-1] - 1) * 100
        bm_ann = ((1 + bm / 100) ** (1 / max(n_years, 0.5)) - 1) * 100
        print(f"  {'等权基准':<10} {bm:>9.1f}%  {bm_ann:>9.1f}%")
    print(f"\n  多空累计: {(result.long_short_nav.iloc[-1] - 1) * 100:.1f}%")
    print(f"  单调性:   {result.ic_monotonicity:.3f}")
    print("=" * 60)
