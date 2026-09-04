"""
research/pbo.py — Probability of Backtest Overfitting (PBO) + Deflated Sharpe

AFML Ch11-15: 在多个策略变体中，最优策略有多大概率是过拟合的？

PBO（Probability of Backtest Overfitting）：
  - Combinatorial Purged CV 思想：把样本分成 N 份，取 J 份做 IS（样本内），N-J 份做 OOS
  - 对每个组合，在 IS 上选最优策略，看它在 OOS 上排名如何
  - PBO = P(最优 IS 策略在 OOS 中排名低于中位数)
  - PBO > 50% → 过拟合严重，IS 最优不代表 OOS 好

Deflated Sharpe Ratio（DSR）：
  - 修正多次试验后的 Sharpe Ratio 显著性
  - 考虑：试验次数 N、样本长度 T、偏度 γ3、非正态性
  - DSR = P(SR* > SR_observed | SR*=0)，即"观测 Sharpe 来自零 SR 的概率"
  - DSR < 0.95 → Sharpe 可能不显著

参考：
  - Bailey & López de Prado (2014) "The Deflated Sharpe Ratio"
  - López de Prado (2018) AFML Ch15
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

EULER_GAMMA = 0.5772156649015329  # Euler–Mascheroni 常数，DSR 期望极大值用


def probabilistic_sharpe_ratio(
    sr_observed: float,
    sr_benchmark: float = 0.0,
    T: int = 252,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """Probabilistic Sharpe Ratio (Bailey & López de Prado 2012, AFML Eq 15.1).

    返回 P(true SR > SR_benchmark | observed SR)，已对偏度/峰度做非正态修正。

    Parameters
    ----------
    sr_observed, sr_benchmark : 年化 Sharpe（按 sqrt(T) 缩放回单期）
    T : 观测期数（如 252*years for daily, 52*years for weekly）
    skew : 收益偏度
    kurt : 收益峰度（注意：非超额峰度，正态=3）
    """
    sr_obs = sr_observed / np.sqrt(T)
    sr_star = sr_benchmark / np.sqrt(T)
    # 方差修正项（Bailey-LdP 2012）
    denom = np.sqrt(max(1e-12, 1.0 - skew * sr_obs + (kurt - 1.0) / 4.0 * sr_obs ** 2))
    z = (sr_obs - sr_star) * np.sqrt(T - 1) / denom
    return float(norm.cdf(z))


def deflated_sharpe_ratio(
    sr_observed: float,
    sr_benchmark: float = 0.0,
    n_trials: int = 1,
    T: int = 252,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio (Bailey & López de Prado 2014, AFML Ch15).

    DSR = PSR，但 benchmark 替换为「n_trials 次独立试验在零 SR 原假设下的期望极大 SR」。

    Parameters
    ----------
    sr_observed : 观测到的（年化）Sharpe Ratio
    sr_benchmark : 用户指定的基准 SR（默认 0）
    n_trials : 独立试验次数（即尝试过的策略变体数）
    T : 样本长度（观测期数）
    skew, kurt : 收益分布的偏度/峰度

    Returns
    -------
    DSR ∈ [0, 1]，>0.95 表示在多次试验校正后仍统计显著。
    """
    if n_trials < 1:
        n_trials = 1
    if T < 2:
        raise ValueError("T 必须 >= 2 以计算 SR 标准误")

    sr_obs = sr_observed / np.sqrt(T)
    sr_bench_per = sr_benchmark / np.sqrt(T)

    # 零假设下 n_trials 次独立试验的期望极大 SR（per-observation 单位）
    # E[max_n] ≈ sqrt(2 ln N / (T-1)) * (1 - γ / (2 ln N))
    if n_trials > 1:
        ln_n = np.log(n_trials)
        sr0 = np.sqrt(2.0 * ln_n / (T - 1)) * (1.0 - EULER_GAMMA / (2.0 * ln_n))
    else:
        sr0 = 0.0

    # Deflated benchmark：取用户基准与「期望极大值」的较大者
    sr_star = max(sr_bench_per, sr0)

    denom = np.sqrt(max(1e-12, 1.0 - skew * sr_obs + (kurt - 1.0) / 4.0 * sr_obs ** 2))
    z = (sr_obs - sr_star) * np.sqrt(T - 1) / denom
    return float(norm.cdf(z))


def probability_of_backtest_overfitting(
    is_sharpes: np.ndarray,
    oos_sharpes: np.ndarray,
    n_partitions: int = 16,
    n_is: int = 8,
) -> dict:
    """Probability of Backtest Overfitting (AFML Ch11-12).

    对每个 trial（一组 IS/OOS 分割）：
      - idx_best = argmax(is_sharpes[:, trial])   # IS 上选最优策略
      - r = rank(oos_sharpes[idx_best, trial]) / (n_strategies + 1)   # OOS 相对排名 ∈ (0,1)
      - 若 r <= 0.5（OOS 排名低于中位数）→ 视为过拟合

    PBO = mean(r <= 0.5 across trials)

    Parameters
    ----------
    is_sharpes, oos_sharpes : shape (n_strategies, n_trials)
        每个策略在每组 IS/OOS 分割上的 Sharpe（或 ICIR）。
    n_partitions, n_is : 仅作记录用，描述原始 combinatorial purged CV 的 N/J 配置。

    Returns
    -------
    {"pbo": float, "logit_pbo": float, "ranks": list[float]}
    """
    is_s = np.asarray(is_sharpes, dtype=float)
    oos_s = np.asarray(oos_sharpes, dtype=float)
    if is_s.shape != oos_s.shape:
        raise ValueError(
            f"is_sharpes 与 oos_sharpes 形状不一致: {is_s.shape} vs {oos_s.shape}"
        )
    if is_s.ndim != 2:
        raise ValueError("输入必须是 2D (n_strategies, n_trials)")

    n_strat, n_trials = is_s.shape

    ranks: list[float] = []
    overfit_cnt = 0
    for t in range(n_trials):
        is_col = is_s[:, t]
        oos_col = oos_s[:, t]
        valid = ~(np.isnan(is_col) | np.isnan(oos_col))
        if valid.sum() < 2:
            continue
        idx_local = int(np.argmax(np.where(valid, is_col, -np.inf)))
        # rank: 1=最低, n_strat=最高
        order = np.argsort(oos_col, kind="mergesort")
        rank_of = np.empty_like(order)
        rank_of[order] = np.arange(1, len(order) + 1)
        rel_rank = rank_of[idx_local] / (n_strat + 1.0)
        ranks.append(float(rel_rank))
        if rel_rank <= 0.5:
            overfit_cnt += 1

    if not ranks:
        return {"pbo": float("nan"), "logit_pbo": float("nan"), "ranks": []}

    pbo = overfit_cnt / len(ranks)
    eps = 1e-6
    p_clipped = min(max(pbo, eps), 1.0 - eps)
    logit = float(np.log(p_clipped / (1.0 - p_clipped)))

    return {
        "pbo": float(pbo),
        "logit_pbo": logit,
        "ranks": ranks,
        "n_strategies": int(n_strat),
        "n_trials": int(n_trials),
        "n_partitions": int(n_partitions),
        "n_is": int(n_is),
    }


def evaluate_experiments(results_dir: str = "results", metric: str = "ICIR") -> dict:
    """扫描 results/ 下所有 model_metrics_*.json，对最优实验计算 DSR。

    将 ICIR 视作 Sharpe-like 指标（IC 均值 / IC 标准差，本身就是信息系数的 SR）。
    """
    base = Path(results_dir)
    if not base.is_absolute():
        base = Path(__file__).resolve().parent.parent / results_dir

    rows: list[dict] = []
    for fpath in sorted(base.rglob("model_metrics_*.json")):
        try:
            with open(fpath, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        if metric not in d or "预测期数" not in d:
            continue
        try:
            metric_val = float(d[metric])
            n_obs = int(d.get("预测期数", 0))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(metric_val) or n_obs < 2:
            continue
        rows.append(
            {
                "tag": str(d.get("tag", fpath.stem)),
                "metric": metric_val,
                "ic_mean": float(d.get("IC均值", float("nan"))),
                "ic_std": float(d.get("IC标准差", float("nan"))),
                "n_obs": n_obs,
                "path": str(fpath),
            }
        )

    n_exp = len(rows)
    if n_exp == 0:
        return {
            "n_experiments": 0,
            "best_icir": None,
            "best_tag": None,
            "dsr": None,
            "interpretation": "未找到包含 ICIR 与预测期数的实验产物",
            "all_metrics": [],
        }

    rows.sort(key=lambda r: r["metric"], reverse=True)
    best = rows[0]

    # ICIR 本身是「IC 均值 / IC 标准差」——可视为 per-observation 的 SR。
    # 这里直接将其作为 sr_observed（年化等价：n_obs 期已含全部信息），
    # 同时把 n_exp 当作 n_trials 做 Bonferroni/Bailey 极大值校正。
    sr_obs = best["metric"]
    T = max(best["n_obs"], 2)
    dsr = deflated_sharpe_ratio(
        sr_observed=sr_obs,
        sr_benchmark=0.0,
        n_trials=n_exp,
        T=T,
        skew=0.0,
        kurt=3.0,
    )

    if dsr < 0.95:
        interp = (
            f"[WARN] DSR={dsr:.3f} < 0.95: 最优 {metric}={best['metric']:.3f} "
            f"(tag={best['tag']}, T={T}) 在 {n_exp} 次试验下不显著，疑似噪声/过拟合"
        )
    elif dsr < 0.99:
        interp = (
            f"[OK]  DSR={dsr:.3f} in [0.95, 0.99): 最优 {metric}={best['metric']:.3f} "
            f"在 {n_exp} 次试验下边缘显著，需谨慎"
        )
    else:
        interp = (
            f"[OK]  DSR={dsr:.3f} >= 0.99: 最优 {metric}={best['metric']:.3f} "
            f"在 {n_exp} 次试验下统计显著"
        )

    return {
        "n_experiments": n_exp,
        "best_icir": best["metric"],
        "best_tag": best["tag"],
        "best_n_obs": T,
        "dsr": dsr,
        "interpretation": interp,
        "all_metrics": rows,
    }


def _cli() -> None:
    print("=" * 70)
    print("【PBO + Deflated Sharpe Ratio 评估】")
    print("=" * 70)

    result = evaluate_experiments()
    if result["n_experiments"] == 0:
        print(result["interpretation"])
        return

    print(f"实验总数        : {result['n_experiments']}")
    print(
        f"最优 ICIR       : {result['best_icir']:.4f}  "
        f"(tag={result['best_tag']}, T={result['best_n_obs']})"
    )
    print(f"Deflated Sharpe : {result['dsr']:.4f}")
    print(f"解读            : {result['interpretation']}")

    print("\nTop 10 实验排名:")
    for i, r in enumerate(result["all_metrics"][:10]):
        print(
            f"  {i + 1:2d}. {r['tag']:50s}  ICIR={r['metric']:+.4f}  "
            f"IC={r['ic_mean']:+.4f}  T={r['n_obs']}"
        )

    # ── PBO 演示（基于已有实验做组合 CV 估计） ─────────────────────────────
    rows = result["all_metrics"]
    if len(rows) >= 2:
        # 把每个实验当作一个「策略」，n_obs 当作其 IS/OOS 共有的 trial 估计
        # 这里我们没有逐 trial 的 IS/OOS Sharpe，只能给出退化 PBO 提示
        print("\n【PBO 提示】")
        print(
            "  当前 results/ 只保存了汇总指标（IC均值/标准差/ICIR/预测期数），"
            "未保留逐 trial 的 IS/OOS Sharpe，无法直接计算 PBO。"
        )
        print("  如需 PBO，请保存每个 walk-forward 窗口的 IS/OOS ICIR 序列，")
        print("  再调用 probability_of_backtest_overfitting(is_s, oos_s)。")
        print(
            f"  当前可作为替代：DSR={result['dsr']:.3f}，"
            f"{'[WARN] 多试验校正后不显著' if result['dsr'] < 0.95 else '[OK] 显著'}。"
        )


if __name__ == "__main__":
    _cli()
