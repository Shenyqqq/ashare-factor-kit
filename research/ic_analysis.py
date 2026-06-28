"""
research/ic_analysis.py  —  全功能因子IC分析

用法:
    python -m research.ic_analysis                   # 全因子，月度持仓
    python -m research.ic_analysis --period 5        # 周频持仓
    python -m research.ic_analysis --decay           # IC衰减曲线（多持仓期）
    python -m research.ic_analysis --corr            # 因子相关矩阵
    python -m research.ic_analysis --plot            # 输出图表
    python -m research.ic_analysis --save            # 保存结果到 research/output/
    python -m research.ic_analysis --top 10          # 只看前N个因子

输出：
    1. 全周期IC汇总（按|IC|均值排序，颜色标注有效性）
    2. 逐年IC分解（最重要：发现因子时序衰减）
    3. IC衰减曲线（5/10/20/40/60日，--decay）
    4. 因子相关矩阵（--corr）
    5. 滚动IC图表（--plot）
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import RAW_DIR
from factors.factor import get_factor_registry

OUTPUT_DIR = Path(__file__).parent / "output"

# ANSI颜色
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


# ── 核心计算 ──────────────────────────────────────────────────────────────────

def compute_ic_series(factor: pd.DataFrame,
                      forward_return: pd.DataFrame) -> pd.Series:
    """每个截面日计算因子与下期收益的Spearman IC"""
    common = factor.index.intersection(forward_return.index)
    return (
        factor.loc[common]
        .corrwith(forward_return.loc[common], axis=1, method="spearman")
        .dropna()
    )


def ic_stats(ic: pd.Series) -> dict:
    """全周期IC统计量"""
    if len(ic) == 0:
        return {"IC均值": np.nan, "IC标准差": np.nan, "ICIR": np.nan,
                "胜率": np.nan, "|IC|均值": np.nan, "样本数": 0}
    icir = ic.mean() / ic.std() if ic.std() > 0 else 0
    return {
        "IC均值":   round(ic.mean(), 4),
        "IC标准差": round(ic.std(), 4),
        "ICIR":     round(icir, 4),
        "胜率":     round((ic > 0).mean(), 4),
        "|IC|均值": round(ic.abs().mean(), 4),
        "样本数":   len(ic),
    }


def ic_by_year(ic: pd.Series) -> pd.Series:
    """逐年IC均值"""
    return ic.groupby(ic.index.year).mean().round(4)


def ic_decay_table(factor_registry: dict,
                   prices: pd.DataFrame,
                   periods: list = None) -> pd.DataFrame:
    """
    IC衰减表：每个因子在不同持仓期下的IC均值。
    periods: 持仓期列表（交易日）
    """
    if periods is None:
        periods = [5, 10, 20, 40, 60]
    rows = []
    for name, factor in factor_registry.items():
        row = {"因子": name}
        for p in periods:
            fwd = prices.pct_change(p).shift(-p)
            ic = compute_ic_series(factor, fwd).mean()
            row[f"{p}日"] = round(ic, 4)
        # 最优持仓期
        ic_vals = {p: abs(row[f"{p}日"]) for p in periods
                   if not np.isnan(row[f"{p}日"])}
        if ic_vals:
            row["最优期"] = f"{max(ic_vals, key=ic_vals.get)}日"
        rows.append(row)
    return pd.DataFrame(rows).set_index("因子")


def factor_corr_matrix(factor_registry: dict,
                       prices: pd.DataFrame,
                       sample_step: int = 20) -> pd.DataFrame:
    """
    因子截面相关矩阵（Spearman，时序均值）。
    sample_step: 每隔N个交易日取一个截面，加速计算。
    """
    sample_dates = prices.index[::sample_step]
    corr_list = []
    for date in sample_dates:
        row = {}
        for name, factor in factor_registry.items():
            if date in factor.index:
                row[name] = factor.loc[date]
        if len(row) == len(factor_registry):
            df_slice = pd.DataFrame(row).dropna()
            if len(df_slice) > 30:
                corr_list.append(df_slice.corr(method="spearman"))
    if not corr_list:
        return pd.DataFrame()
    return pd.concat(corr_list).groupby(level=0).mean()


# ── 打印工具 ──────────────────────────────────────────────────────────────────

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
    print(f"\n{BOLD}{'='*72}{RESET}")
    print(f"{BOLD}  全周期 IC 汇总{RESET}")
    print(f"  {GREEN}绿=有效(|IC|>0.05){RESET}  "
          f"{YELLOW}黄=弱信号(>0.03){RESET}  "
          f"{RED}红=无效(<0.03){RESET}")
    print(f"{'='*72}")
    header = (f"  {'因子':<16} {'IC均值':>8} {'IC标准差':>8} "
              f"{'ICIR':>7} {'胜率':>7} {'|IC|均值':>8} {'样本数':>6}")
    print(header)
    print("-" * 72)
    for name, row in summary_df.iterrows():
        ic_abs = row["|IC|均值"]
        color = GREEN if ic_abs >= 0.05 else YELLOW if ic_abs >= 0.03 else RED
        print(
            f"  {color}{name:<16}{RESET}"
            f"{_color_ic(row['IC均值'])}"
            f"  {row['IC标准差']:>8.4f}"
            f"{_color_icir(row['ICIR'])}"
            f"  {row['胜率']:>7.1%}"
            f"{_color_ic(ic_abs)}"
            f"  {row['样本数']:>6}"
        )
    valid   = (summary_df["|IC|均值"] >= 0.05).sum()
    weak    = ((summary_df["|IC|均值"] >= 0.03) &
               (summary_df["|IC|均值"] < 0.05)).sum()
    invalid = (summary_df["|IC|均值"] < 0.03).sum()
    print(f"\n  {GREEN}有效: {valid}{RESET}  "
          f"{YELLOW}弱信号: {weak}{RESET}  "
          f"{RED}无效: {invalid}{RESET}")


def print_yearly(yearly_df: pd.DataFrame):
    years = [c for c in yearly_df.columns if str(c).isdigit()]
    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}  逐年 IC 均值（发现时序衰减）{RESET}")
    print(f"  持续下降说明因子alpha被套利，需要替换")
    print(f"{'='*80}")
    header = f"  {'因子':<16}" + "".join(f"  {y}" for y in years) + "  趋势"
    print(header)
    print("-" * 80)
    for name, row in yearly_df.iterrows():
        vals = [row.get(y, np.nan) for y in years]
        # 趋势：最近3年 vs 最早3年
        recent = np.nanmean(vals[-3:]) if len(vals) >= 3 else np.nan
        early  = np.nanmean(vals[:3])  if len(vals) >= 3 else np.nan
        if not np.isnan(recent) and not np.isnan(early):
            trend = ("↓衰减" if recent < early - 0.01
                     else "↑增强" if recent > early + 0.01
                     else "→稳定")
            trend_color = (RED if "衰减" in trend
                           else GREEN if "增强" in trend else "")
        else:
            trend = ""
            trend_color = ""

        row_str = f"  {name:<16}"
        for v in vals:
            if np.isnan(v):
                row_str += "    nan"
            else:
                color = GREEN if abs(v) >= 0.05 else YELLOW if abs(v) >= 0.03 else RED
                row_str += f"  {color}{v:>5.3f}{RESET}"
        row_str += f"  {trend_color}{trend}{RESET}"
        print(row_str)


def print_decay(decay_df: pd.DataFrame):
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  IC 衰减表（持仓期 → IC均值）{RESET}")
    print(f"  快速衰减→适合短频调仓，缓慢衰减→适合月频调仓")
    print(f"{'='*70}")
    print(decay_df.to_string())


# ── 图表 ─────────────────────────────────────────────────────────────────────

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
    fig.suptitle(f"因子IC分析（持仓期={period}日）", fontsize=13, fontweight="bold")

    # 上：滚动IC均值
    ax = axes[0]
    for name, ic in valid.items():
        ic.rolling(window).mean().plot(ax=ax, label=name, alpha=0.8, lw=1.2)
    ax.axhline(0,     color="black", lw=0.8)
    ax.axhline( 0.05, color="green", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(-0.05, color="green", lw=0.8, ls="--", alpha=0.5)
    ax.set_title(f"{window}期滚动IC均值")
    ax.legend(fontsize=8, ncol=4)
    ax.grid(alpha=0.3)

    # 下：逐年IC均值柱状图（堆叠对比）
    ax = axes[1]
    yearly = pd.DataFrame({
        name: ic_by_year(ic) for name, ic in valid.items()
    }).T
    yearly.plot(kind="bar", ax=ax, width=0.8, alpha=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline( 0.05, color="green", lw=0.8, ls="--", alpha=0.5)
    ax.set_title("逐年IC均值对比")
    ax.legend(fontsize=8, ncol=5, loc="upper right")
    ax.set_xlabel("")
    plt.xticks(rotation=45)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()


def plot_corr_matrix(corr: pd.DataFrame):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams["font.family"] = "SimHei"
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(max(8, len(corr) * 0.7),
                                    max(6, len(corr) * 0.6)))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7,
                    color="white" if abs(v) > 0.6 else "black")
    ax.set_title("因子截面相关矩阵（Spearman，时序均值）")
    plt.tight_layout()
    plt.show()

    # 打印高相关对
    high = [(corr.index[i], corr.columns[j], round(corr.iloc[i, j], 3))
            for i in range(len(corr))
            for j in range(i + 1, len(corr))
            if abs(corr.iloc[i, j]) > 0.7]
    if high:
        print(f"\n{YELLOW}高相关因子对（|corr|>0.7，存在冗余，ML训练时可剔除一个）:{RESET}")
        for a, b, c in high:
            print(f"  {a} vs {b}: {c}")
    else:
        print(f"\n{GREEN}无高相关因子对（|corr|<0.7）{RESET}")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def neutralize_factor(factor: pd.DataFrame,
                      industry_map: pd.Series) -> pd.DataFrame:
    """
    行业中性化：在每个截面日，把因子得分替换为行业内 z-score。
    industry_map: Series(code → industry_label)
    """
    result = factor.copy()
    for date in factor.index:
        row = factor.loc[date].dropna()
        ind = industry_map.reindex(row.index).fillna("未分类")
        neutralized = row.copy()
        for grp, stocks in ind.groupby(ind):
            vals = row[stocks.index]
            if len(vals) < 3:
                continue
            std = vals.std()
            if std > 0:
                neutralized[stocks.index] = (vals - vals.mean()) / std
            else:
                neutralized[stocks.index] = 0.0
        result.loc[date] = neutralized
    return result


def run(period: int = 20, top: int = 0, plot: bool = False,
        decay: bool = False, corr: bool = False, save: bool = False,
        neutralize: bool = False):

    print("载入数据...")
    prices    = pd.read_parquet(RAW_DIR / "prices_hfq.parquet")
    financial = (pd.read_parquet(RAW_DIR / "financial_indicators.parquet")
                 if (RAW_DIR / "financial_indicators.parquet").exists() else None)

    def _opt(fname):
        p = RAW_DIR / fname
        return pd.read_parquet(p) if p.exists() else None

    prices_raw  = _opt("prices_raw.parquet")
    volume      = _opt("volume.parquet")
    amount      = _opt("amount.parquet")
    open_       = _opt("open_hfq.parquet")
    high        = _opt("high_hfq.parquet")
    low         = _opt("low_hfq.parquet")
    institution = _opt("institution_holding.parquet")

    from data.clean import clean_ohlcv
    clean_ret, masks = clean_ohlcv(prices, open_, high, low)

    print(f"计算因子（持仓期={period}日）...")
    registry = get_factor_registry(
        prices=prices, financial=financial, prices_raw=prices_raw,
        volume=volume, amount=amount, open_=open_, high=high, low=low,
        clean_ret=clean_ret, masks=masks, institution=institution,
    )
    forward_return = prices.pct_change(period).shift(-period)

    # ── 行业中性化（可选）────────────────────────────────────────────────────
    if neutralize:
        try:
            from data.industry.download_industry import load_industry_map
            industry_map = load_industry_map()["sw_l2"]
            print("应用行业中性化（申万二级）...")
            registry = {
                name: neutralize_factor(factor, industry_map)
                for name, factor in registry.items()
            }
        except FileNotFoundError:
            print("行业映射文件不存在，跳过中性化。"
                  "请先运行: python -m data.industry.download_industry")

    # ── 1. 全周期汇总 ────────────────────────────────────────────────────────
    print("计算IC...")
    all_ic = {}
    summary_rows = []
    for name, factor in registry.items():
        ic = compute_ic_series(factor, forward_return)
        all_ic[name] = ic
        summary_rows.append({"因子": name, **ic_stats(ic)})

    summary_df = (pd.DataFrame(summary_rows)
                  .set_index("因子")
                  .sort_values("|IC|均值", ascending=False))
    if top > 0:
        summary_df = summary_df.head(top)
        all_ic = {k: v for k, v in all_ic.items() if k in summary_df.index}

    print_summary(summary_df)

    # ── 2. 逐年IC分解 ────────────────────────────────────────────────────────
    yearly_rows = []
    all_years = sorted(set(
        y for ic in all_ic.values() for y in ic.index.year
    ))
    for name, ic in all_ic.items():
        by_year = ic_by_year(ic)
        row = {"因子": name}
        for y in all_years:
            row[y] = by_year.get(y, np.nan)
        yearly_rows.append(row)

    yearly_df = pd.DataFrame(yearly_rows).set_index("因子")
    # 按全周期|IC|均值排序
    yearly_df = yearly_df.loc[summary_df.index]
    print_yearly(yearly_df)

    # ── 3. IC衰减表（可选）──────────────────────────────────────────────────
    if decay:
        print("\n计算IC衰减（需要一些时间）...")
        decay_df = ic_decay_table(registry, prices)
        print_decay(decay_df)

    # ── 4. 因子相关矩阵（可选）──────────────────────────────────────────────
    if corr:
        print("\n计算因子相关矩阵...")
        corr_mat = factor_corr_matrix(registry, prices)
        if not corr_mat.empty:
            if plot:
                plot_corr_matrix(corr_mat)
            else:
                print(corr_mat.round(3).to_string())

    # ── 5. 图表（可选）──────────────────────────────────────────────────────
    if plot:
        plot_rolling_ic(all_ic, period)

    # ── 6. 保存结果（可选）──────────────────────────────────────────────────
    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(OUTPUT_DIR / f"ic_summary_h{period}.csv",
                          encoding="utf-8-sig")
        yearly_df.to_csv(OUTPUT_DIR / f"ic_yearly_h{period}.csv",
                         encoding="utf-8-sig")
        print(f"\n结果已保存至 {OUTPUT_DIR}")

    return summary_df, yearly_df, all_ic


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", type=int, default=20)
    parser.add_argument("--top",    type=int, default=0)
    parser.add_argument("--plot",   action="store_true")
    parser.add_argument("--decay",  action="store_true")
    parser.add_argument("--corr",   action="store_true")
    parser.add_argument("--save",      action="store_true")
    parser.add_argument("--neutralize", action="store_true",
                        help="行业中性化后再计算IC（诊断用，实盘不建议）")
    args = parser.parse_args()
    run(period=args.period, top=args.top, plot=args.plot,
        decay=args.decay, corr=args.corr, save=args.save,
        neutralize=args.neutralize)
