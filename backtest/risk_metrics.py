"""
backtest/risk_metrics.py — 风险调整指标（Sharpe / Sortino / Calmar / IR / 胜率 / 回撤）

独立模块，便于多策略对比复用。输入 nav（累计净值 DataFrame，index=signal_date，
列=track 名，如 Q1-Q5/Top100/benchmark/指数），输出每条 track 的年化收益、年化波动、
Sharpe、Sortino、最大回撤、Calmar、信息比率（vs benchmark）、胜率。

年化因子由 rebalance_freq 推断（W-FRI→52, ME→12, 2W-FRI→26, 3D→84），与
backtest/quantile.py 的调仓周期一致。无风险利率 rf 默认 0（A 股短期简化），
可在 config/settings.py 改 RISK_FREE_RATE。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


_FREQ_TO_PERIODS_PER_YEAR: dict[str, int] = {
    "W-FRI": 52,
    "W-MON": 52,
    "W-TUE": 52,
    "W-WED": 52,
    "W-THU": 52,
    "W": 52,
    "2W-FRI": 26,
    "ME": 12,        # 月末
    "MS": 12,        # 月初
    "QE": 4,         # 季末
    "3D": 84,        # 252/3
    "5D": 50,        # 252/5 ≈ 50 (周频，按交易日)
    "10D": 25,       # 双周
    "20D": 12,       # 月频（按交易日）
    "60D": 4,        # 季频
}


def _periods_per_year(freq: str) -> int:
    """从 rebalance_freq 字符串推年化期数；未知 freq 回退 12（保守月频）。"""
    if freq is None:
        return 12
    key = str(freq).strip().upper()
    if key in _FREQ_TO_PERIODS_PER_YEAR:
        return _FREQ_TO_PERIODS_PER_YEAR[key]
    # 形如 "5D" / "20D" 这种 business-day 串：252/N
    if key.endswith("D") and key[:-1].isdigit():
        n = int(key[:-1])
        if n > 0:
            return max(1, round(252 / n))
    return 12


def _safe_div(num: float, den: float) -> float:
    """除零保护：den=0/NaN → NaN，避免 inf 污染输出。"""
    if den is None or np.isnan(den) or den == 0:
        return float("nan")
    return float(num) / float(den)


def compute_risk_metrics(
    nav: pd.DataFrame,
    rebalance_freq: str = "ME",
    benchmark_col: str = "benchmark",
    rf: float = 0.0,
) -> pd.DataFrame:
    """
    计算每条 track 的风险调整指标。

    Parameters
    ----------
    nav : pd.DataFrame
        累计净值，index=DatetimeIndex（signal_date），每列一条 track
        （Q1-Q5 / Top100 / benchmark / 指数）。首行通常为 1.0。
    rebalance_freq : str
        调仓频率，用于推年化因子（W-FRI→52, ME→12, ...）。
    benchmark_col : str
        benchmark 列名；不存在时 IR 全 NaN。
    rf : float
        年化无风险利率，默认 0。

    Returns
    -------
    pd.DataFrame
        index=track 名，columns=
        [年化收益, 年化波动, Sharpe, Sortino, 最大回撤, Calmar, IR, 胜率]
        全 NaN/常数列 track 不崩溃，输出 NaN。
    """
    if nav is None or nav.empty:
        return pd.DataFrame(
            columns=["年化收益", "年化波动", "Sharpe", "Sortino",
                     "最大回撤", "Calmar", "IR", "胜率"]
        )

    ppy = _periods_per_year(rebalance_freq)
    cols = ["年化收益", "年化波动", "Sharpe", "Sortino",
            "最大回撤", "Calmar", "IR", "胜率"]
    rows: dict[str, dict] = {}

    # benchmark 的 period_rets 预先取出，供 IR 用
    bm_rets: pd.Series | None = None
    if benchmark_col in nav.columns:
        bm_rets = nav[benchmark_col].pct_change().dropna()

    for track in nav.columns:
        s = nav[track].dropna()
        if len(s) < 2:
            rows[track] = {c: float("nan") for c in cols}
            continue

        period_rets = s.pct_change().dropna()
        n = len(period_rets)
        if n == 0:
            rows[track] = {c: float("nan") for c in cols}
            continue

        with np.errstate(divide="ignore", invalid="ignore"):
            # 年化收益（几何）
            growth = float((1 + period_rets).prod())
            ann_return = float(growth ** (ppy / n) - 1) if growth > 0 else float("nan")

            # 年化波动（样本标准差 × sqrt(期数)）
            vol = float(period_rets.std(ddof=1))
            ann_vol = vol * np.sqrt(ppy) if not np.isnan(vol) else float("nan")

            # Sharpe
            sharpe = _safe_div(ann_return - rf, ann_vol)

            # 下行波动 & Sortino
            downside = period_rets[period_rets < 0]
            if len(downside) > 0:
                dvol = float(downside.std(ddof=1)) * np.sqrt(ppy)
            else:
                dvol = 0.0 if (period_rets > 0).any() else float("nan")
            sortino = _safe_div(ann_return - rf, dvol)

            # 最大回撤（负数，输出绝对值百分比由格式化处理）
            cummax = s.cummax()
            drawdown = (s / cummax - 1)
            max_dd = float(drawdown.min())  # 负数
            max_dd_abs = abs(max_dd)

            # Calmar = 年化收益 / |最大回撤|
            calmar = _safe_div(ann_return, max_dd_abs)

            # 信息比率（vs benchmark）
            ir = float("nan")
            if bm_rets is not None and track != benchmark_col:
                aligned = pd.concat([period_rets, bm_rets], axis=1, keys=["s", "b"]).dropna()
                if len(aligned) > 1:
                    excess = aligned["s"] - aligned["b"]
                    ex_std = float(excess.std(ddof=1))
                    if not np.isnan(ex_std) and ex_std > 0:
                        ir = float(excess.mean() / ex_std * np.sqrt(ppy))

            # 胜率
            win_rate = float((period_rets > 0).mean())

        rows[track] = {
            "年化收益": ann_return,
            "年化波动": ann_vol,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "最大回撤": max_dd,         # 负数（格式化时取绝对值百分比）
            "Calmar": calmar,
            "IR": ir,
            "胜率": win_rate,
        }

    df = pd.DataFrame.from_dict(rows, orient="index", columns=cols)
    return df


def format_risk_metrics_table(metrics: pd.DataFrame) -> str:
    """把 compute_risk_metrics 输出格式化成对齐的文本表（用于 print_quantile_summary）。"""
    if metrics is None or metrics.empty:
        return "  (无风险指标数据)"

    fmt = metrics.copy()
    # 最大回撤：负数 → 绝对值百分比；胜率存 0–1 小数 → 百分比
    fmt["最大回撤"] = fmt["最大回撤"].abs() * 100
    fmt["年化收益"] = fmt["年化收益"] * 100
    fmt["年化波动"] = fmt["年化波动"] * 100
    fmt["胜率"] = fmt["胜率"] * 100

    fmt_str = {
        "年化收益": "{:>8.2f}%",
        "年化波动": "{:>8.2f}%",
        "Sharpe": "{:>7.2f}",
        "Sortino": "{:>7.2f}",
        "最大回撤": "{:>8.2f}%",
        "Calmar": "{:>7.2f}",
        "IR": "{:>7.2f}",
        "胜率": "{:>6.1f}%",
    }

    header = (
        f"  {'track':<10}"
        f"{'年化收益':>9}"
        f"{'年化波动':>9}"
        f"{'Sharpe':>8}"
        f"{'Sortino':>8}"
        f"{'最大回撤':>9}"
        f"{'Calmar':>8}"
        f"{'IR':>8}"
        f"{'胜率':>7}"
    )
    sep = "  " + "-" * (len(header) - 2)
    lines = [header, sep]
    for track, row in fmt.iterrows():
        cells = [f"  {track:<10}"]
        for col, f in fmt_str.items():
            v = row[col]
            cells.append(f.format(v) if not (isinstance(v, float) and np.isnan(v)) else " " * len(f.format(0)))
        lines.append("".join(cells))
    return "\n".join(lines)


def export_risk_metrics(
    nav: pd.DataFrame,
    save_path: str,
    rebalance_freq: str = "ME",
    benchmark_col: str = "benchmark",
    rf: float = 0.0,
) -> pd.DataFrame:
    """
    计算风险指标并导出 CSV，供多策略对比合并用。

    Returns
    -------
    pd.DataFrame
        与 compute_risk_metrics 相同，便于调用方继续打印或合并。
    """
    metrics = compute_risk_metrics(
        nav, rebalance_freq=rebalance_freq,
        benchmark_col=benchmark_col, rf=rf,
    )
    if save_path:
        # CSV 用英文字段名（便于 pandas 跨策略合并），另存一份带中文表头
        en_cols = {
            "年化收益": "ann_return",
            "年化波动": "ann_vol",
            "Sharpe": "sharpe",
            "Sortino": "sortino",
            "最大回撤": "max_drawdown",
            "Calmar": "calmar",
            "IR": "ir",
            "胜率": "win_rate",
        }
        metrics_en = metrics.rename(columns=en_cols)
        metrics_en.index.name = "track"
        metrics_en.to_csv(save_path, encoding="utf-8-sig", float_format="%.6f")
        logger.info(f"风险指标已保存: {save_path}")
    return metrics


if __name__ == "__main__":
    # 简单冒烟：合成 nav 跑一遍
    rng = np.random.default_rng(0)
    dates = pd.date_range("2022-01-31", periods=24, freq="ME")
    nav = pd.DataFrame(index=dates)
    for track, mu, sd in [("Q1", 0.005, 0.04), ("Q5", 0.015, 0.04),
                          ("Top100", 0.012, 0.035), ("benchmark", 0.008, 0.038)]:
        rets = rng.normal(mu, sd, len(dates))
        nav[track] = (1 + rets).cumprod()
    m = compute_risk_metrics(nav, rebalance_freq="ME", benchmark_col="benchmark")
    print(format_risk_metrics_table(m))
