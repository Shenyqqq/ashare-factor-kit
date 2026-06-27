"""
research/ic_analysis.py  —  命令行快速 IC 分析

不用开 Jupyter，直接打印所有因子的 IC 汇总表。

用法:
    python -m research.ic_analysis                # 全部因子，月度持仓
    python -m research.ic_analysis --period 5     # 5日持仓
    python -m research.ic_analysis --top 10       # 只展示 IC 最高的10个
    python -m research.ic_analysis --plot         # 额外输出 IC 对比折线图
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import RAW_DIR
from factors.factor import get_factor_registry


def compute_ic_series(factor: pd.DataFrame, forward_return: pd.DataFrame) -> pd.Series:
    common = factor.index.intersection(forward_return.index)
    return (
        factor.loc[common]
        .corrwith(forward_return.loc[common], axis=1, method="spearman")
        .dropna()
    )


def ic_summary(ic: pd.Series, name: str) -> dict:
    icir = ic.mean() / ic.std() if ic.std() > 0 else 0
    return {
        "因子":        name,
        "IC均值":      round(ic.mean(), 4),
        "IC标准差":    round(ic.std(), 4),
        "ICIR":        round(icir, 4),
        "IC>0胜率":    round((ic > 0).mean(), 4),
        "|IC|均值":    round(ic.abs().mean(), 4),
        "样本数":      len(ic),
    }


def run(period: int = 20, top: int = 0, plot: bool = False):
    print(f"载入数据...")
    prices    = pd.read_parquet(RAW_DIR / "prices_hfq.parquet")
    financial = pd.read_parquet(RAW_DIR / "financial_indicators.parquet")

    prices_raw = None
    if (RAW_DIR / "prices_raw.parquet").exists():
        prices_raw = pd.read_parquet(RAW_DIR / "prices_raw.parquet")

    print(f"计算因子（持仓期={period}日）...")
    registry = get_factor_registry(
        prices=prices,
        financial=financial,
        prices_raw=prices_raw,
    )

    forward_return = prices.pct_change(period).shift(-period)

    rows = []
    all_ic = {}
    for name, factor in registry.items():
        ic = compute_ic_series(factor, forward_return)
        all_ic[name] = ic
        rows.append(ic_summary(ic, name))

    df = pd.DataFrame(rows).set_index("因子")
    df = df.sort_values("|IC|均值", ascending=False)

    if top > 0:
        df = df.head(top)

    # 颜色标记（终端 ANSI）
    def _mark(val, col):
        if col == "|IC|均值":
            if val >= 0.05: return f"\033[92m{val}\033[0m"   # 绿
            if val >= 0.03: return f"\033[93m{val}\033[0m"   # 黄
            return f"\033[91m{val}\033[0m"                   # 红
        if col == "ICIR":
            if abs(val) >= 0.5: return f"\033[92m{val}\033[0m"
            if abs(val) >= 0.3: return f"\033[93m{val}\033[0m"
        return str(val)

    print(f"\n{'='*65}")
    print(f"  IC 分析汇总（持仓期={period}日，共{len(df)}个因子）")
    print(f"{'='*65}")
    print(f"  绿=有效(>0.05)  黄=弱信号(>0.03)  红=无效(<0.03)")
    print(f"{'-'*65}")

    header = f"{'因子':<16} {'IC均值':>8} {'IC标准差':>8} {'ICIR':>8} {'胜率':>8} {'|IC|均值':>8}"
    print(header)
    print("-" * 65)

    for name, row in df.iterrows():
        ic_abs = row["|IC|均值"]
        color = "\033[92m" if ic_abs >= 0.05 else "\033[93m" if ic_abs >= 0.03 else "\033[91m"
        reset = "\033[0m"
        print(
            f"  {color}{name:<14}{reset}"
            f"  {row['IC均值']:>8.4f}"
            f"  {row['IC标准差']:>8.4f}"
            f"  {row['ICIR']:>8.4f}"
            f"  {row['IC>0胜率']:>8.1%}"
            f"  {color}{row['|IC|均值']:>8.4f}{reset}"
        )

    valid   = (df["|IC|均值"] >= 0.05).sum()
    weak    = ((df["|IC|均值"] >= 0.03) & (df["|IC|均值"] < 0.05)).sum()
    invalid = (df["|IC|均值"] < 0.03).sum()
    print(f"\n  有效因子: {valid}个  弱信号: {weak}个  无效: {invalid}个")

    if plot:
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.rcParams["font.family"] = "SimHei"
        matplotlib.rcParams["axes.unicode_minus"] = False

        valid_factors = df[df["|IC|均值"] >= 0.03].index.tolist()
        if not valid_factors:
            print("无 IC>0.03 的因子，跳过图表")
            return df

        fig, ax = plt.subplots(figsize=(16, 5))
        for name in valid_factors:
            all_ic[name].rolling(6).mean().plot(
                ax=ax, label=name, alpha=0.8, lw=1.2
            )
        ax.axhline(0,     color="black", lw=0.8)
        ax.axhline( 0.05, color="green", lw=0.8, ls="--", alpha=0.5)
        ax.axhline(-0.05, color="green", lw=0.8, ls="--", alpha=0.5)
        ax.set_title(f"各因子 6期滚动IC均值（持仓期={period}日）")
        ax.legend(fontsize=8, ncol=4)
        plt.tight_layout()
        plt.show()

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", type=int, default=20, help="持仓期（交易日）")
    parser.add_argument("--top",    type=int, default=0,  help="只显示前N个因子（0=全部）")
    parser.add_argument("--plot",   action="store_true",  help="输出IC对比折线图")
    args = parser.parse_args()
    run(period=args.period, top=args.top, plot=args.plot)
