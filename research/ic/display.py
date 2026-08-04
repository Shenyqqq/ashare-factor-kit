"""Terminal / matplotlib output for IC analysis v2."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.ic.statistics import ic_by_year

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _color_ic(val: float) -> str:
    if np.isnan(val):
        return f"{'nan':>8}"
    color = GREEN if abs(val) >= 0.05 else YELLOW if abs(val) >= 0.03 else RED
    return f"{color}{val:>8.4f}{RESET}"


def _color_icir(val: float) -> str:
    if np.isnan(val):
        return f"{'nan':>7}"
    color = GREEN if abs(val) >= 0.5 else YELLOW if abs(val) >= 0.3 else ""
    return f"{color}{val:>7.4f}{RESET}"


def print_summary(summary_df: pd.DataFrame):
    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}  全周期 IC 汇总 (v2){RESET}")
    print(f"  {GREEN}绿=有效(|IC|>0.05){RESET}  {YELLOW}黄=弱(>0.03){RESET}  {RED}红=无效{RESET}")
    print(f"{'='*80}")
    extra = "IC_after_cost" in summary_df.columns
    hdr = (f"  {'因子':<16} {'IC均值':>8} {'ICIR':>7} {'胜率':>7} "
           f"{'NW_t':>6} {'|IC|':>8}")
    if extra:
        hdr += f" {'IC_扣费':>8}"
    hdr += f" {'N':>5}"
    print(hdr)
    print("-" * 80)
    for name, row in summary_df.iterrows():
        ic_abs = row["|IC|均值"]
        color = GREEN if ic_abs >= 0.05 else YELLOW if ic_abs >= 0.03 else RED
        nw = row.get("NW_t统计量", np.nan)
        line = (
            f"  {color}{name:<16}{RESET}"
            f"{_color_ic(row['IC均值'])}"
            f"{_color_icir(row['ICIR'])}"
            f"  {row['胜率']:>7.1%}"
            f"  {nw:>6.2f}"
            f"{_color_ic(ic_abs)}"
        )
        if extra:
            line += f"  {_color_ic(row.get('IC_after_cost', np.nan))}"
        line += f"  {int(row['样本数']):>5}"
        print(line)
    valid = (summary_df["|IC|均值"] >= 0.05).sum()
    weak = ((summary_df["|IC|均值"] >= 0.03) & (summary_df["|IC|均值"] < 0.05)).sum()
    invalid = (summary_df["|IC|均值"] < 0.03).sum()
    print(f"\n  {GREEN}有效: {valid}{RESET}  {YELLOW}弱: {weak}{RESET}  {RED}无效: {invalid}{RESET}")


def print_yearly(yearly_df: pd.DataFrame):
    years = [c for c in yearly_df.columns if str(c).isdigit()]
    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}  逐年 IC 均值{RESET}")
    print(f"{'='*80}")
    header = f"  {'因子':<16}" + "".join(f"  {y}" for y in years) + "  趋势"
    print(header)
    print("-" * 80)
    for name, row in yearly_df.iterrows():
        vals = [row.get(y, np.nan) for y in years]
        recent = np.nanmean(vals[-3:]) if len(vals) >= 3 else np.nan
        early = np.nanmean(vals[:3]) if len(vals) >= 3 else np.nan
        if not np.isnan(recent) and not np.isnan(early):
            trend = ("↓衰减" if recent < early - 0.01
                     else "↑增强" if recent > early + 0.01 else "→稳定")
            trend_color = RED if "衰减" in trend else GREEN if "增强" in trend else ""
        else:
            trend, trend_color = "", ""
        row_str = f"  {name:<16}"
        for v in vals:
            if np.isnan(v):
                row_str += "    nan"
            else:
                c = GREEN if abs(v) >= 0.05 else YELLOW if abs(v) >= 0.03 else RED
                row_str += f"  {c}{v:>5.3f}{RESET}"
        row_str += f"  {trend_color}{trend}{RESET}"
        print(row_str)


def print_decay(decay_df: pd.DataFrame):
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  IC 衰减表{RESET}")
    print(decay_df.to_string())


def print_selection_result(
    kept: list,
    exclusions: dict,
    categories: dict | None = None,
    sparse_kept: list | None = None,
    emerging_kept: list | None = None,
    labels: dict | None = None,
):
    n_sparse = len(sparse_kept or [])
    n_emerg = len(emerging_kept or [])
    print(f"\n{BOLD}{'='*72}{RESET}")
    print(
        f"{BOLD}  因子筛选 (v2){RESET}  "
        f"稠密 {GREEN}{len(kept)}{RESET} 稀疏 {GREEN}{n_sparse}{RESET} "
        f"新兴观察 {YELLOW}{n_emerg}{RESET} "
        f"剔除 {RED}{len(exclusions)}{RESET}"
    )
    print(f"{'='*72}")

    def _tag(name: str, default: str = "") -> str:
        cat = (categories or {}).get(name, default)
        labs = (labels or {}).get(name) or []
        parts = [p for p in [cat, *labs] if p]
        return f" ({', '.join(parts)})" if parts else ""

    for n in kept:
        print(f"    [OK] {n}{_tag(n)}")
    if sparse_kept:
        print(f"  --- 稀疏轨道 ---")
        for n in sparse_kept:
            print(f"    [SPARSE] {n}{_tag(n, '稀疏因子')}")
    if emerging_kept:
        print(f"  --- 新兴观察（不进 factors 主池）---")
        for n in emerging_kept:
            print(f"    [EMERGING] {n}{_tag(n, '新兴因子')}")
    for n, reason in exclusions.items():
        print(f"    [X]  {n:<20} {reason}")


def print_barra_comparison(summary_df: pd.DataFrame, pure_ic_means: dict):
    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}  Barra 纯因子 IC{RESET}")
    print(f"{'='*80}")
    for name in summary_df.index:
        raw_ic = summary_df.loc[name, "IC均值"]
        pure_ic = pure_ic_means.get(name, np.nan)
        if np.isnan(raw_ic) or np.isnan(pure_ic) or abs(raw_ic) < 1e-6:
            ret_str, judge, color = "  nan", "数据不足", ""
        else:
            retention = pure_ic / raw_ic
            ret_str = f"{retention:.1%}"
            if retention >= 0.8:
                judge, color = "真实alpha", GREEN
            elif retention >= 0.5:
                judge, color = "部分alpha", YELLOW
            elif retention > 0:
                judge, color = "主要风格", RED
            else:
                judge, color = "方向反转!", RED
        pure_s = f"{pure_ic:>10.4f}" if not np.isnan(pure_ic) else f"{'nan':>10}"
        print(f"  {color}{name:<18}{RESET}  {raw_ic:>8.4f}  {pure_s}  {ret_str:>8}  {color}{judge}{RESET}")


def print_quantile_ls(quantile_df: pd.DataFrame, *, y_mode: str = "residual", top: int = 15):
    """打印 Barra pure 路径下的 Q1/Q5 多头空头贡献摘要。"""
    if quantile_df is None or quantile_df.empty:
        return
    print(f"\n{BOLD}{'='*80}{RESET}")
    print(
        f"{BOLD}  多头/空头贡献（Q5 vs Q1，已 IC 方向对齐，y_mode={y_mode}）{RESET}"
    )
    print(f"{'='*80}")
    print(
        f"  {'因子':<18} {'多头超额':>10} {'空头贡献':>10} "
        f"{'long_share':>10} {'多空来源':<8} {'spread':>10}"
    )
    # 按 |spread| 降序展示，便于先看多空拉开的因子
    view = quantile_df.copy()
    if "spread" in view.columns:
        view = view.reindex(view["spread"].abs().sort_values(ascending=False).index)
    n_show = 0
    for name, row in view.iterrows():
        if top > 0 and n_show >= top:
            break
        le = row.get("多头超额", np.nan)
        sc = row.get("空头贡献", np.nan)
        share = row.get("long_share", np.nan)
        src = row.get("多空来源", "无效")
        sp = row.get("spread", np.nan)
        if src == "多头主导":
            color = GREEN
        elif src == "空头主导":
            color = YELLOW
        elif src == "双边":
            color = ""
        else:
            color = RED
        le_s = f"{le:>10.4f}" if np.isfinite(le) else f"{'nan':>10}"
        sc_s = f"{sc:>10.4f}" if np.isfinite(sc) else f"{'nan':>10}"
        sh_s = f"{share:>10.2f}" if np.isfinite(share) else f"{'nan':>10}"
        sp_s = f"{sp:>10.4f}" if np.isfinite(sp) else f"{'nan':>10}"
        print(
            f"  {color}{str(name):<18}{RESET}  {le_s}  {sc_s}  {sh_s}  "
            f"{color}{str(src):<8}{RESET}  {sp_s}"
        )
        n_show += 1
    if top > 0 and len(view) > top:
        print(f"  ... 共 {len(view)} 个因子（仅展示 |spread| Top {top}）")


def plot_rolling_ic(all_ic: dict, period: int, window: int = 12):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams["font.family"] = "SimHei"
    matplotlib.rcParams["axes.unicode_minus"] = False

    valid = {k: v for k, v in all_ic.items() if v.abs().mean() >= 0.03}
    if not valid:
        print("无IC>0.03的因子，跳过图表")
        return

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle(f"因子IC分析 v2（持仓期={period}日）", fontsize=13, fontweight="bold")
    ax = axes[0]
    for name, ic in valid.items():
        ic.rolling(window).mean().plot(ax=ax, label=name, alpha=0.8, lw=1.2)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(0.05, color="green", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(-0.05, color="green", lw=0.8, ls="--", alpha=0.5)
    ax.set_title(f"{window}期滚动IC均值")
    ax.legend(fontsize=8, ncol=4)
    ax.grid(alpha=0.3)

    ax = axes[1]
    yearly = pd.DataFrame({name: ic_by_year(ic) for name, ic in valid.items()}).T
    yearly.plot(kind="bar", ax=ax, width=0.8, alpha=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("逐年IC均值对比")
    ax.legend(fontsize=8, ncol=5, loc="upper right")
    plt.xticks(rotation=45)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()


def plot_corr_matrix(corr: pd.DataFrame):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams["font.family"] = "SimHei"
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(max(8, len(corr) * 0.7), max(6, len(corr) * 0.6)))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    ax.set_title("因子截面相关矩阵")
    plt.tight_layout()
    plt.show()
