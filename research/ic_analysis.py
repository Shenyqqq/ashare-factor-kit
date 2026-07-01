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
    python -m research.ic_analysis --industry        # 分申万二级行业IC分析
    python -m research.ic_analysis --workers 1       # 串行（32GB 推荐，默认）
    python -m research.ic_analysis --workers 2       # 最多 2 个因子同时算 IC

并行说明：
    每个因子 IC 需对全历史截面做 rank+corr，多线程同时跑会复制多份中间矩阵，
    默认 --workers=1（串行）。勿与 logs/driver_ic_parallel.sh 叠加大进程并行。

输出：
    1. 全周期IC汇总（按|IC|均值排序，颜色标注有效性）
    2. 逐年IC分解（最重要：发现因子时序衰减）
    3. IC衰减曲线（5/10/20/40/60日，--decay）
    4. 因子相关矩阵（--corr）
    5. 滚动IC图表（--plot）
    6. 分行业IC（--industry，需要industry_map.parquet）
"""
import argparse
import gc
import sys
import time
from pathlib import Path

import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*ConstantInput.*")

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import RAW_DIR, IC_MAX_WORKERS, BARRA_IC_WORKERS
from factors.factor import get_factor_registry

OUTPUT_DIR = Path(__file__).parent / "output"

# 截面常数特征（市场/HMM），Spearman IC 无意义，不参与 IC 汇总与筛选
_IC_SKIP_PREFIXES = ("市场", "HMM_")


def _is_ic_skippable(name: str) -> bool:
    return any(name.startswith(p) for p in _IC_SKIP_PREFIXES)


def _log_phase(label: str, t0: float) -> float:
    """打印阶段耗时，返回新的计时起点。"""
    elapsed = time.perf_counter() - t0
    print(f"  [{label}] {elapsed:.1f}s", flush=True)
    return time.perf_counter()


def _get_rebalance_dates(dates: pd.DatetimeIndex, period: int) -> pd.DatetimeIndex:
    """与 build_ml_dataset / quantile / run.py 一致：resample 取周期末交易日。"""
    from utils.rebalance_dates import get_rebalance_dates, horizon_to_rebalance_freq

    return get_rebalance_dates(dates, horizon_to_rebalance_freq(period))


def _to_float32_panel(df: pd.DataFrame) -> pd.DataFrame:
    """IC 阶段用 float32 减半内存，rank/corr 精度足够。"""
    if df.dtypes.apply(lambda d: d == np.float64).any():
        return df.astype(np.float32)
    return df


def _build_forward_return(
    prices: pd.DataFrame,
    open_: pd.DataFrame | None,
    period: int,
) -> pd.DataFrame:
    """
    与 strategies/ml.py、回测一致：
    有开盘价 → close[t+N]/open[t+1]-1（信号日收盘后次日开盘买入）
    无开盘价 → close[t+N]/close[t]-1（退化为收收收益）
    """
    if open_ is not None:
        buy = open_.shift(-1)
        sell = prices.shift(-period)
        return sell / buy.replace(0, np.nan) - 1
    return prices.pct_change(period).shift(-period)

# ANSI颜色
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


# ── 核心计算 ──────────────────────────────────────────────────────────────────

def compute_ic_series(factor: pd.DataFrame,
                      forward_return: pd.DataFrame) -> pd.Series:
    """每个截面日计算因子与下期收益的Spearman IC（向量化实现）

    等价于逐日 spearmanr，但先整体 rank 再算 Pearson，速度快 10-20x。
    """
    common = factor.index.intersection(forward_return.index)
    f = _to_float32_panel(factor.loc[common])
    r = _to_float32_panel(forward_return.loc[common])
    # 截面排名（axis=1），NaN自动排到最后不影响有效股票
    f_ranked = f.rank(axis=1, na_option="keep")
    r_ranked = r.rank(axis=1, na_option="keep")
    return f_ranked.corrwith(r_ranked, axis=1).dropna()


def ic_stats(ic: pd.Series) -> dict:
    """全周期IC统计量（含t检验显著性）"""
    if len(ic) == 0:
        return {"IC均值": np.nan, "IC标准差": np.nan, "ICIR": np.nan,
                "胜率": np.nan, "|IC|均值": np.nan, "t统计量": np.nan, "样本数": 0}
    icir = ic.mean() / ic.std() if ic.std() > 0 else 0
    # t检验：H0: IC均值=0，t = IC均值 / (IC标准差 / sqrt(N))
    t_stat = ic.mean() / (ic.std() / np.sqrt(len(ic))) if ic.std() > 0 else 0
    return {
        "IC均值":   round(ic.mean(), 4),
        "IC标准差": round(ic.std(), 4),
        "ICIR":     round(icir, 4),
        "胜率":     round((ic > 0).mean(), 4),
        "|IC|均值": round(ic.abs().mean(), 4),
        "t统计量":  round(t_stat, 2),   # |t|>2 表示95%置信水平显著
        "样本数":   len(ic),
    }


def ic_by_year(ic: pd.Series) -> pd.Series:
    """逐年IC均值"""
    return ic.groupby(ic.index.year).mean().round(4)


def _run_bounded_parallel(fn, items: list, n_workers: int, progress_every: int = 0):
    """
    有界并行：最多 n_workers 个任务同时在飞（不会一次性 submit 全部因子）。

    n_workers=1 时完全串行，适合 32GB 内存机器。
    fn(item) -> result
    """
    n_workers = max(1, min(n_workers, len(items) or 1))
    if n_workers == 1:
        for i, item in enumerate(items, 1):
            yield fn(item)
            if progress_every and (i % progress_every == 0 or i == len(items)):
                print(f"  进度: {i}/{len(items)}", flush=True)
        return

    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

    it = iter(items)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        pending: set = set()
        for _ in range(n_workers):
            try:
                pending.add(pool.submit(fn, next(it)))
            except StopIteration:
                break
        done_count = 0
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                yield fut.result()
                done_count += 1
                if progress_every and (done_count % progress_every == 0):
                    print(f"  进度: {done_count}/{len(items)}", flush=True)
                try:
                    pending.add(pool.submit(fn, next(it)))
                except StopIteration:
                    pass
        if progress_every and done_count:
            print(f"  进度: {done_count}/{len(items)}", flush=True)


def ic_decay_table(factor_registry: dict,
                   prices: pd.DataFrame,
                   open_: pd.DataFrame | None = None,
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
            fwd = _build_forward_return(prices, open_, p)
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

def compute_ic_industry(factor_registry: dict,
                        forward_return: pd.DataFrame,
                        industry_map: pd.Series,
                        min_stocks: int = 20) -> pd.DataFrame:
    """
    分申万二级行业计算IC均值。

    返回 DataFrame: index=行业代码, columns=因子名,
                    值=该行业内各截面IC的时序均值。
    只保留平均截面股票数 >= min_stocks 的行业。
    """
    # 统计每个行业的股票列表（取有记录的股票）
    industry_groups = industry_map.groupby(industry_map).apply(lambda x: x.index.tolist())

    # 所有因子的列集合并集（即所有有数据的股票）
    all_factor_stocks = set()
    for factor in factor_registry.values():
        all_factor_stocks.update(factor.columns)

    rows = []
    for ind_code, stocks in industry_groups.items():
        available = [s for s in stocks if s in all_factor_stocks]
        if len(available) < min_stocks:
            continue

        row = {"行业": ind_code, "股票数": len(available)}
        for fname, factor in factor_registry.items():
            cols = [s for s in available if s in factor.columns]
            if len(cols) < min_stocks:
                row[fname] = np.nan
                continue
            f_sub = factor[cols]
            fwd_sub = forward_return.reindex(columns=cols)
            ic = compute_ic_series(f_sub, fwd_sub)
            row[fname] = round(ic.mean(), 4) if len(ic) > 0 else np.nan
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("行业")
    return df


def select_factors(
    summary_df: pd.DataFrame,
    factor_registry: dict,
    pure_ic_means: dict = None,
    ic_threshold: float = 0.02,
    icir_threshold: float = 0.30,
    t_threshold: float = 2.0,
    corr_threshold: float = 0.70,
    sample_step: int = 20,
) -> tuple[list, dict]:
    """
    三步自动筛选因子：
      1. 剔除 |IC| < ic_threshold 且 ICIR < icir_threshold 的弱因子
         （若 pure_ic_means 存在，用纯因子IC代替原始IC作为主判据）
      2. 剔除 t统计量 < t_threshold 的统计不显著因子（默认 |t|<2.0，即95%置信）
      3. 对高相关对（|corr| > corr_threshold），保留 ICIR 更高的一个

    返回 (selected_names, exclusion_reasons)
    """
    exclusions = {}

    # ── Step 1：按IC/ICIR剔除弱因子 ─────────────────────────────────────
    candidates = []
    for name in summary_df.index:
        row = summary_df.loc[name]
        raw_ic   = abs(row["IC均值"])
        icir     = abs(row["ICIR"])
        t_stat   = abs(row.get("t统计量", np.nan))
        pure_ic  = abs(pure_ic_means.get(name, np.nan)) if pure_ic_means else raw_ic

        # 纯IC存在时以纯IC为主，原始IC为辅
        effective_ic = pure_ic if pure_ic_means and not np.isnan(pure_ic) else raw_ic

        if effective_ic < ic_threshold and icir < icir_threshold:
            reason = (f"纯IC={pure_ic:.4f}<{ic_threshold}, ICIR={icir:.4f}<{icir_threshold}"
                      if pure_ic_means else
                      f"IC={raw_ic:.4f}<{ic_threshold}, ICIR={icir:.4f}<{icir_threshold}")
            exclusions[name] = reason
        elif not np.isnan(t_stat) and t_stat < t_threshold:
            exclusions[name] = f"t统计量={t_stat:.2f}<{t_threshold}（IC均值统计不显著）"
        else:
            candidates.append(name)

    if not candidates:
        return [], exclusions

    # ── Step 2：按相关矩阵去冗余 ─────────────────────────────────────────
    # 只对候选因子计算相关性，按ICIR从高到低贪心保留
    cand_registry = {n: factor_registry[n] for n in candidates if n in factor_registry}
    if len(cand_registry) <= 1:
        return candidates, exclusions

    sample_dates = list(factor_registry[candidates[0]].index[::sample_step])
    corr_list = []
    for date in sample_dates:
        row = {}
        for name, fdf in cand_registry.items():
            if date in fdf.index:
                row[name] = fdf.loc[date]
        df_slice = pd.DataFrame(row).dropna()
        if len(df_slice) > 30:
            corr_list.append(df_slice.corr(method="spearman"))

    if not corr_list:
        return candidates, exclusions

    avg_corr = pd.concat(corr_list).groupby(level=0).mean()

    # 按 ICIR 降序排，贪心保留
    icir_order = (summary_df.loc[candidates, "ICIR"]
                  .abs().sort_values(ascending=False).index.tolist())
    kept = []
    for name in icir_order:
        if name not in avg_corr.index:
            kept.append(name)
            continue
        # 检查与已保留因子的相关性
        drop = False
        for k in kept:
            if k in avg_corr.columns:
                c = abs(avg_corr.loc[name, k])
                if c > corr_threshold:
                    icir_name = abs(summary_df.loc[name, "ICIR"])
                    icir_k    = abs(summary_df.loc[k, "ICIR"])
                    exclusions[name] = (
                        f"与{k}相关系数={c:.2f}>{corr_threshold}，"
                        f"ICIR({icir_name:.3f})<ICIR({icir_k:.3f})"
                    )
                    drop = True
                    break
        if not drop:
            kept.append(name)

    return kept, exclusions


def print_selection_result(kept: list, exclusions: dict):
    print(f"\n{BOLD}{'='*72}{RESET}")
    print(f"{BOLD}  因子自动筛选结果{RESET}  "
          f"保留 {GREEN}{len(kept)}{RESET} 个，剔除 {RED}{len(exclusions)}{RESET} 个")
    print(f"{'='*72}")
    if kept:
        print(f"\n  {GREEN}保留因子:{RESET}")
        for n in kept:
            print(f"    [OK] {n}")
    if exclusions:
        print(f"\n  {RED}剔除因子:{RESET}")
        for n, reason in exclusions.items():
            print(f"    [X]  {n:<20} {reason}")


def print_industry_ic(ind_df: pd.DataFrame, factor_names: list):
    """打印分行业IC热力图（文本版）"""
    if ind_df.empty:
        print("无满足条件的行业（股票数<20）")
        return

    factor_cols = [f for f in factor_names if f in ind_df.columns]
    print(f"\n{BOLD}{'='*90}{RESET}")
    print(f"{BOLD}  分申万二级行业 IC 均值{RESET}  "
          f"({len(ind_df)}个行业，股票数≥20)")
    print(f"  {GREEN}绿=|IC|≥0.05{RESET}  {YELLOW}黄=≥0.03{RESET}  {RED}红=<0.03{RESET}")
    print(f"{'='*90}")

    # 表头
    hdr = f"  {'行业':>8}  {'股票数':>4}"
    for f in factor_cols:
        hdr += f"  {f[:8]:>8}"
    print(hdr)
    print("-" * 90)

    # 按第一个因子的绝对IC排序
    sort_col = factor_cols[0] if factor_cols else None
    sort_key = ind_df[sort_col].abs() if sort_col else ind_df.iloc[:, 0].abs()
    sorted_df = ind_df.loc[sort_key.sort_values(ascending=False).index]

    for ind_code, row in sorted_df.iterrows():
        line = f"  {str(ind_code):>8}  {int(row['股票数']):>4}"
        for f in factor_cols:
            v = row.get(f, np.nan)
            line += f"  {_color_ic(v)}"
        print(line)

    # 最后一行：跨行业标准差（衡量因子行业间一致性）
    print("-" * 90)
    std_line = f"  {'行业间σ':>8}  {'':>4}"
    for f in factor_cols:
        v = ind_df[f].std()
        std_line += f"  {v:>8.4f}"
    print(std_line)
    print(f"  行业间σ越大说明因子有明显行业偏向，分行业模型收益更大")


def compute_pure_ic_series(
    factor: pd.DataFrame,
    barra_factors: dict,
    forward_return: pd.DataFrame,
    industry_map: pd.Series = None,
    min_stocks: int = 30,
) -> pd.Series:
    """
    纯因子 IC：每个截面日，将 alpha 因子对 Barra 风格因子（+行业哑变量）做截面 OLS，
    取残差（正交化后的纯 alpha），再计算与远期收益的 Spearman IC。

    IC 下降幅度 = 因子中系统性风险敞口的比例。
    IC 保留比例 = 真实 alpha 含量。
    """
    common_dates = factor.index.intersection(forward_return.index)
    results = {}

    for date in common_dates:
        f_row = factor.loc[date].dropna()
        if len(f_row) < min_stocks:
            continue

        # 构建控制变量矩阵（Barra 因子 + 可选行业哑变量）
        ctrl_cols = {}
        for bname, bdf in barra_factors.items():
            if date in bdf.index:
                b_row = bdf.loc[date].reindex(f_row.index)
                ctrl_cols[bname] = b_row

        if industry_map is not None:
            ind = industry_map.reindex(f_row.index).fillna("未分类")
            cats = ind.unique()
            if len(cats) > 1:
                for grp in cats[1:]:  # 删掉一个避免多重共线
                    ctrl_cols[f"_ind_{grp}"] = (ind == grp).astype(float)

        if not ctrl_cols:
            continue

        ctrl_df = pd.DataFrame(ctrl_cols, index=f_row.index).fillna(0)
        common_stocks = f_row.index.intersection(ctrl_df.index)
        if len(common_stocks) < min_stocks:
            continue

        f_vals = f_row.loc[common_stocks].values
        X = ctrl_df.loc[common_stocks].values
        A = np.column_stack([np.ones(len(f_vals)), X])

        try:
            coef, _, _, _ = np.linalg.lstsq(A, f_vals, rcond=None)
            resid = f_vals - A @ coef
        except Exception:
            continue

        resid_s = pd.Series(resid, index=common_stocks)
        fwd_row = forward_return.loc[date].reindex(common_stocks).dropna()
        valid = resid_s.index.intersection(fwd_row.index)
        if len(valid) < min_stocks:
            continue

        ic = resid_s.loc[valid].rank().corr(fwd_row.loc[valid].rank())
        if not np.isnan(ic):
            results[date] = ic

    return pd.Series(results)


def precompute_ctrl_matrices(
    barra_factors: dict,
    forward_return: pd.DataFrame,
    industry_map: pd.Series = None,
    dates: pd.DatetimeIndex | None = None,
    industry_panel: pd.DataFrame | None = None,
) -> dict:
    """
    预计算调仓日控制变量矩阵（Barra + 行业哑变量），供纯化 OLS 共享。

    仅在 dates（默认调仓日）上缓存，避免 2056 日 × 全市场 × 行业哑变量爆内存。

    PIT 支持：industry_panel 不为 None 时，按当期行业构建哑变量（消除 PIT 泄漏）；
              为 None 时回退到静态 industry_map 一次构建的快速路径。

    返回 dict: date -> (ctrl_arr, ctrl_idx, fwd_arr)
      ctrl_arr  (N, K) float32
      ctrl_idx  股票代码 Index
      fwd_arr     与 ctrl_idx 对齐的远期收益 float32（NaN 表示无效）
    """
    target_dates = dates if dates is not None else forward_return.index
    use_pit = industry_panel is not None

    # 静态路径：行业哑变量只建一次
    ind_arr = None
    ind_index = None
    if not use_pit and industry_map is not None:
        ind_full = industry_map.fillna("未分类")
        cats = sorted(ind_full.unique())
        ind_cols = {
            f"_ind_{grp}": (ind_full == grp).astype(np.float32)
            for grp in cats[1:]
        }
        if ind_cols:
            ind_df = pd.DataFrame(ind_cols)
            ind_index = ind_df.index
            ind_arr = ind_df.values.astype(np.float32)

    if use_pit:
        from data.industry.download_industry import load_industry_as_of

    date_ctrl = {}
    for date in target_dates:
        if date not in forward_return.index:
            continue

        ctrl_cols = {}
        for bname, bdf in barra_factors.items():
            if date in bdf.index:
                ctrl_cols[bname] = bdf.loc[date].astype(np.float32)

        if not ctrl_cols:
            continue

        barra_df = pd.DataFrame(ctrl_cols).fillna(0.0)

        if use_pit:
            ind_series = load_industry_as_of(industry_panel, date, level="sw_l2")
            ind = ind_series.reindex(barra_df.index).fillna("未分类")
            cats_d = sorted(ind.unique())
            if len(cats_d) > 1:
                cols_d = {
                    f"_ind_{g}": (ind == g).astype(np.float32)
                    for g in cats_d[1:]
                }
                if cols_d:
                    ind_part = (
                        pd.DataFrame(cols_d, index=barra_df.index)
                        .values.astype(np.float32)
                    )
                    ctrl_arr = np.hstack([
                        barra_df.values.astype(np.float32), ind_part
                    ])
                else:
                    ctrl_arr = barra_df.values.astype(np.float32)
            else:
                ctrl_arr = barra_df.values.astype(np.float32)
        elif ind_arr is not None:
            positions = ind_index.get_indexer(barra_df.index)
            valid = positions >= 0
            if not valid.all():
                barra_df = barra_df.iloc[valid]
                positions = positions[valid]
            ind_part = ind_arr[positions]
            ctrl_arr = np.hstack([barra_df.values.astype(np.float32), ind_part])
        else:
            ctrl_arr = barra_df.values.astype(np.float32)

        ctrl_idx = barra_df.index
        fwd_arr = (
            forward_return.loc[date]
            .reindex(ctrl_idx)
            .values.astype(np.float32)
        )
        date_ctrl[date] = (ctrl_arr, ctrl_idx, fwd_arr)

    return date_ctrl


def _spearman_ic_numpy(x: np.ndarray, y: np.ndarray, min_stocks: int) -> float:
    """Spearman IC，纯 numpy，避免 pandas rank/corr 开销。"""
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < min_stocks:
        return np.nan
    rx = rankdata(x[mask], method="average")
    ry = rankdata(y[mask], method="average")
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom <= 0:
        return np.nan
    return float((rx * ry).sum() / denom)


def compute_pure_ic_fast(
    factor: pd.DataFrame,
    date_ctrl: dict,
    rebalance_dates: pd.DatetimeIndex,
    min_stocks: int = 30,
) -> pd.Series:
    """
    纯因子 IC（预计算控制矩阵 + 调仓日截面 OLS）。

    仅在 rebalance_dates 上计算（与 ML/回测调仓频率一致），
    相比逐日 2056 次 OLS 约快 20×、内存占用更低。
    """
    results = {}
    factor_f32 = _to_float32_panel(factor)
    if factor_f32.empty or len(factor_f32.columns) == 0:
        return pd.Series(dtype=float)
    factor_arr = factor_f32.to_numpy()
    date_to_row = {d: i for i, d in enumerate(factor_f32.index)}
    factor_cols = factor_f32.columns

    for date in rebalance_dates:
        cached = date_ctrl.get(date)
        if cached is None:
            continue
        ctrl_arr, ctrl_idx, fwd_arr = cached

        row_i = date_to_row.get(date)
        if row_i is None:
            continue

        f_row = factor_arr[row_i]
        if len(f_row) == 0 or len(ctrl_idx) == 0:
            continue

        col_pos = factor_cols.get_indexer(ctrl_idx)
        f_vals = f_row[np.maximum(col_pos, 0)]
        valid = (col_pos >= 0) & np.isfinite(f_vals) & np.isfinite(fwd_arr)
        if valid.sum() < min_stocks:
            continue

        f_v = f_vals[valid].astype(np.float32, copy=False)
        X = ctrl_arr[valid]
        y_fwd = fwd_arr[valid]
        A = np.column_stack([np.ones(len(f_v), dtype=np.float32), X])

        try:
            coef, _, _, _ = np.linalg.lstsq(A, f_v, rcond=None)
            resid = f_v - A @ coef
        except Exception:
            continue

        ic = _spearman_ic_numpy(resid, y_fwd, min_stocks)
        if np.isfinite(ic):
            results[date] = ic

    return pd.Series(results)


def print_barra_comparison(summary_df: pd.DataFrame, pure_ic_df: pd.DataFrame):
    """打印原始 IC vs 纯因子 IC 对比表"""
    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}  Barra 纯因子 IC 分析（剔除系统性风险后的真实 alpha）{RESET}")
    print(f"  保留率 = 纯IC / 原始IC，越高说明因子 alpha 越独立于系统性风险")
    print(f"  {GREEN}绿=保留≥80%{RESET}  {YELLOW}黄=50-80%{RESET}  {RED}红=<50%（主要是风格暴露）{RESET}")
    print(f"{'='*80}")
    print(f"  {'因子':<18} {'原始IC':>8} {'纯因子IC':>10} {'保留率':>8} {'判断':>12}")
    print("-" * 80)

    for name in summary_df.index:
        raw_ic = summary_df.loc[name, "IC均值"]
        pure_ic = pure_ic_df.get(name, np.nan)
        if np.isnan(raw_ic) or np.isnan(pure_ic):
            retention = np.nan
            judge = "数据不足"
            color = ""
        elif abs(raw_ic) < 1e-6:
            retention = np.nan
            judge = "原始IC≈0"
            color = RED
        else:
            retention = pure_ic / raw_ic  # 方向应一致
            if retention >= 0.8:
                judge = "真实alpha"
                color = GREEN
            elif retention >= 0.5:
                judge = "部分alpha"
                color = YELLOW
            elif retention > 0:
                judge = "主要风格"
                color = RED
            else:
                judge = "方向反转!"
                color = RED

        ret_str = f"{retention:.1%}" if not np.isnan(retention) else "  nan"
        pure_s = f"{pure_ic:>10.4f}" if not np.isnan(pure_ic) else f"{'nan':>10}"
        print(
            f"  {color}{name:<18}{RESET}"
            f"  {raw_ic:>8.4f}"
            f"  {pure_s}"
            f"  {ret_str:>8}"
            f"  {color}{judge:>12}{RESET}"
        )


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
        neutralize: bool = False, industry: bool = False, barra: bool = False,
        lookback_years: int = 0, workers: int = None, barra_workers: int = None):

    t_run = time.perf_counter()
    print("载入数据...")
    t0 = time.perf_counter()
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
    margin      = _opt("margin_balance.parquet")
    moneyflow   = _opt("moneyflow_large.parquet")
    northbound  = _opt("northbound_holding.parquet")
    institution = _opt("institution_holding.parquet")
    market_prices = _opt("csi_all.parquet")
    if market_prices is None:
        market_prices = _opt("index_000300.parquet")
    if market_prices is None:
        market_prices = _opt("csi300.parquet")
    industry_map_df = _opt("industry_map.parquet")

    from data.clean import clean_ohlcv
    clean_ret, masks = clean_ohlcv(prices, open_, high, low)
    t0 = _log_phase("载入数据", t0)

    # ── lookback 截断：只用最近 N 年数据计算 IC（解决 regime shift 稀释问题）──
    lookback_date = None
    if lookback_years > 0:
        lookback_date = prices.index.max() - pd.DateOffset(years=lookback_years)
        print(f"  [lookback_years={lookback_years}] 只计算 {lookback_date.date()} 之后的 IC")

    print(f"计算因子（持仓期={period}日）...")
    registry = get_factor_registry(
        prices=prices, financial=financial, prices_raw=prices_raw,
        volume=volume, amount=amount, open_=open_, high=high, low=low,
        clean_ret=clean_ret, masks=masks,
        market_prices=market_prices, industry_map=industry_map_df,
        margin=margin, moneyflow=moneyflow,
        northbound=northbound, institution=institution,
    )
    skipped = [k for k in registry if _is_ic_skippable(k)]
    if skipped:
        registry = {k: v for k, v in registry.items() if not _is_ic_skippable(k)}
        print(f"  跳过 {len(skipped)} 个市场状态特征（截面常数，IC 无意义）")

    # IC 阶段 float32，约减半因子面板内存
    registry = {k: _to_float32_panel(v) for k, v in registry.items()}
    t0 = _log_phase(f"计算因子（{len(registry)}个）", t0)

    fwd_label = "open[t+1]→close[t+N]" if open_ is not None else "close→close"
    print(f"  forward_return: {fwd_label}")
    forward_return = _build_forward_return(prices, open_, period).astype(np.float32)

    rebalance_dates = _get_rebalance_dates(forward_return.index, period)
    print(f"  调仓日: {len(rebalance_dates)} 个（period={period}，"
          f"{'周频' if period <= 15 else '月频'}）")

    # ── 加载行业映射（Barra纯化/中性化/分行业分析都需要）────────────────────
    industry_map_series = None
    industry_panel = None
    if industry_map_df is not None and "sw_l2" in industry_map_df.columns:
        industry_map_series = industry_map_df["sw_l2"]
    elif neutralize or industry or barra:
        try:
            from data.industry.download_industry import load_industry_map
            industry_map_series = load_industry_map()["sw_l2"]
        except FileNotFoundError:
            print("行业映射文件不存在，请先运行: python -m data.industry.download_industry")
            if barra:
                print("  警告: --barra 将不含行业哑变量控制")
            neutralize = industry = False

    # 尝试加载 PIT 行业面板（消除行业映射回填历史截面的未来信息泄漏）；
    # 文件不存在时 fallback 到静态 industry_map_series（向后兼容）。
    if barra:
        try:
            from data.industry.download_industry import (
                load_industry_panel as _load_panel,
                PANEL_PATH,
            )
            if PANEL_PATH.exists():
                industry_panel = _load_panel()
                print(f"  [PIT] 使用 industry_map_panel.parquet（"
                      f"{len(industry_panel)} 条 / "
                      f"{industry_panel['code'].nunique()} 只股票）")
            elif industry_map_series is not None:
                print("  [PIT] industry_map_panel.parquet 不存在，"
                      "回退到静态 industry_map（行业哑变量可能含 PIT 泄漏）")
        except Exception as e:
            print(f"  [PIT] 加载行业面板失败 ({e})，回退到静态映射")

    # ── 行业中性化（可选）────────────────────────────────────────────────────
    if neutralize and industry_map_series is not None:
        print("应用行业中性化（申万二级）...")
        registry = {
            name: neutralize_factor(factor, industry_map_series)
            for name, factor in registry.items()
        }

    # ── 1. 全周期汇总（有界并行计算 IC）──────────────────────────────────────
    ic_workers = IC_MAX_WORKERS if workers is None else max(1, workers)
    barra_ic_workers = BARRA_IC_WORKERS if barra_workers is None else max(1, barra_workers)

    n_factors = len(registry)
    mode = "串行" if ic_workers == 1 else f"最多{ic_workers}并发"
    print(f"计算IC（{n_factors}个因子，{mode}）...")

    def _compute_one_ic(name_fac):
        name, factor = name_fac
        return name, compute_ic_series(factor, forward_return)

    all_ic_full = {}
    for name, ic_full in _run_bounded_parallel(
        _compute_one_ic, list(registry.items()), ic_workers, progress_every=10
    ):
        all_ic_full[name] = ic_full
    t0 = _log_phase(f"计算IC（{n_factors}个因子）", t0)

    # 从全历史IC衍生lookback-filtered IC（避免重复计算）
    all_ic = {}
    summary_rows = []
    for name, ic_full in all_ic_full.items():
        ic = ic_full[ic_full.index >= lookback_date] if lookback_date is not None else ic_full
        all_ic[name] = ic
        summary_rows.append({"因子": name, **ic_stats(ic)})

    summary_df = (pd.DataFrame(summary_rows)
                  .set_index("因子")
                  .sort_values("|IC|均值", ascending=False))
    if top > 0:
        summary_df = summary_df.head(top)
        all_ic = {k: v for k, v in all_ic.items() if k in summary_df.index}

    print_summary(summary_df)

    # ── 2. 逐年IC分解（复用全历史IC，无需重算）─────────────────────────────
    yearly_rows = []
    all_years = sorted(set(
        y for ic in all_ic_full.values() for y in ic.index.year
    ))
    for name, ic in all_ic_full.items():
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
        decay_df = ic_decay_table(registry, prices, open_=open_)
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

    # ── 5. Barra 纯因子 IC（可选）───────────────────────────────────────────
    pure_ic_means = {}
    if barra:
        print("\n计算 Barra 纯因子 IC...")
        t_barra = time.perf_counter()
        try:
            from factors.barra_risk import get_barra_factors
            # 加载市场指数（优先沪深300，fallback创业板）
            mkt_prices = _opt("index_000300.parquet")
            if mkt_prices is None:
                mkt_prices = _opt("index_399006.parquet")
            barra_factors = get_barra_factors(
                prices=prices,
                financial=financial,
                market_prices=mkt_prices,
                volume=volume,
                clean_ret=clean_ret,
                industry_map=industry_map_series,
            )
            t_barra = _log_phase("Barra 风格因子", t_barra)

            if barra_factors:
                barra_names = [n for n in summary_df.index if n in registry]

                # 仅在调仓日预计算控制矩阵（非 2056 全交易日）
                print(f"  预计算调仓日控制矩阵（{len(rebalance_dates)}日，"
                      f"非全样本 {len(forward_return.index)} 日）...")
                date_ctrl = precompute_ctrl_matrices(
                    barra_factors,
                    forward_return,
                    industry_map_series,
                    dates=rebalance_dates,
                    industry_panel=industry_panel,
                )
                del barra_factors
                gc.collect()
                t_barra = _log_phase(
                    f"控制矩阵缓存 {len(date_ctrl)} 日", t_barra
                )

                bw = max(1, min(barra_ic_workers, len(barra_names)))
                bw_mode = "串行" if bw == 1 else f"最多{bw}并发"
                print(f"  纯化 OLS（{len(barra_names)}个因子，{bw_mode}）...")

                def _pure_ic_one(name):
                    pure_ic = compute_pure_ic_fast(
                        registry[name], date_ctrl, rebalance_dates
                    )
                    mean_val = pure_ic.mean() if len(pure_ic) > 0 else np.nan
                    return name, mean_val

                for name, mean_val in _run_bounded_parallel(
                    _pure_ic_one, barra_names, bw, progress_every=10
                ):
                    pure_ic_means[name] = mean_val

                del date_ctrl
                gc.collect()
                _log_phase("Barra 纯 IC", t_barra)
                print_barra_comparison(summary_df, pure_ic_means)
            else:
                print("Barra 因子计算失败，跳过纯因子IC")
        except Exception as e:
            print(f"Barra 分析出错: {e}")
            import traceback
            traceback.print_exc()

    # ── 6. 分行业IC（可选）──────────────────────────────────────────────────
    ind_ic_df = pd.DataFrame()
    if industry and industry_map_series is not None:
        print("\n计算分行业IC（可能需要1-2分钟）...")
        ind_ic_df = compute_ic_industry(
            {k: v for k, v in registry.items()
             if k in summary_df.index},  # 只算已筛选的因子
            forward_return,
            industry_map_series,
        )
        print_industry_ic(ind_ic_df, list(summary_df.index))

    # ── 7. 图表（可选）──────────────────────────────────────────────────────
    if plot:
        plot_rolling_ic(all_ic, period)

    # ── 8. 因子自动筛选 ─────────────────────────────────────────────────────
    kept_factors, exclusion_reasons = select_factors(
        summary_df, registry,
        pure_ic_means=pure_ic_means if pure_ic_means else None,
    )
    print_selection_result(kept_factors, exclusion_reasons)

    # ── 9. 保存结果（可选）──────────────────────────────────────────────────
    if save:
        import json
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(OUTPUT_DIR / f"ic_summary_h{period}.csv",
                          encoding="utf-8-sig")
        yearly_df.to_csv(OUTPUT_DIR / f"ic_yearly_h{period}.csv",
                         encoding="utf-8-sig")
        if not ind_ic_df.empty:
            ind_ic_df.to_csv(OUTPUT_DIR / f"ic_industry_h{period}.csv",
                             encoding="utf-8-sig")
        if pure_ic_means:
            pure_df = pd.DataFrame.from_dict(
                pure_ic_means, orient="index", columns=["纯因子IC均值"]
            )
            pure_df["原始IC均值"] = summary_df["IC均值"].reindex(pure_df.index)
            pure_df["保留率"] = pure_df["纯因子IC均值"] / pure_df["原始IC均值"]
            pure_df.to_csv(OUTPUT_DIR / f"ic_barra_pure_h{period}.csv",
                           encoding="utf-8-sig")
        # 保存筛选结果 JSON（供 run.py --factor-config 读取）
        selection = {
            "horizon": period,
            "lookback_years": lookback_years if lookback_years > 0 else "full",
            "ic_start_date": str(lookback_date.date()) if lookback_date else "all",
            "factors": kept_factors,
            "excluded": exclusion_reasons,
            "ic_stats": summary_df[["IC均值", "ICIR", "胜率"]].to_dict(),
        }
        json_path = OUTPUT_DIR / f"selected_factors_h{period}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(selection, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存至 {OUTPUT_DIR}")

    _log_phase("总耗时", t_run)
    return summary_df, yearly_df, all_ic, ind_ic_df


if __name__ == "__main__":
    from config.encoding_bootstrap import bootstrap_stdio_utf8

    bootstrap_stdio_utf8()
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", type=int, default=20)
    parser.add_argument("--top",    type=int, default=0)
    parser.add_argument("--plot",   action="store_true")
    parser.add_argument("--decay",  action="store_true")
    parser.add_argument("--corr",   action="store_true")
    parser.add_argument("--save",       action="store_true")
    parser.add_argument("--neutralize", action="store_true",
                        help="行业中性化后再计算IC（诊断用，实盘不建议）")
    parser.add_argument("--industry",   action="store_true",
                        help="按申万二级分行业计算IC（需要industry_map.parquet）")
    parser.add_argument("--barra",      action="store_true",
                        help="计算 Barra 纯因子IC（剔除系统性风险，显示真实alpha含量）")
    parser.add_argument("--lookback-years", type=int, default=0, dest="lookback_years",
                        help="只用最近 N 年 IC 评估因子（0=全历史）。"
                             "解决 regime shift 问题：旧市场负相关、新市场正相关的因子不会被稀释筛掉。"
                             "例：--lookback-years 3 只用 2023-2026 的 IC 做筛选，"
                             "逐年分解仍展示全部年份供对比。")
    parser.add_argument("--workers", type=int, default=None,
                        help=f"因子 IC 并行度（默认 {IC_MAX_WORKERS}，32GB 建议 1）")
    parser.add_argument("--barra-workers", type=int, default=None, dest="barra_workers",
                        help=f"Barra 纯 IC 并行度（默认 {BARRA_IC_WORKERS}）")
    args = parser.parse_args()
    run(period=args.period, top=args.top, plot=args.plot,
        decay=args.decay, corr=args.corr, save=args.save,
        neutralize=args.neutralize, industry=args.industry, barra=args.barra,
        lookback_years=args.lookback_years,
        workers=args.workers, barra_workers=args.barra_workers)
