"""
backtest/optimize.py — 经典组合权重（v1 practical）

在已选股票集合（quantile / TopN）上分配权重；**不**做选股。

方法
----
- ``ew``      : 等权（默认，与旧路径 bit-identical）
- ``score``   : 得分正比加权（shift 到非负）
- ``rank``    : 截面 rank 正比加权
- ``invvol``  : 逆波动率加权（Σ 对角）
- ``mv``      : 均值-方差 lite（μ←得分，Σ←Ledoit-Wolf / 对角 shrink）
- ``rp``      : 风险平价（可选；失败则回退 invvol）

约束（v1）
----------
long-only；``sum(w)=target_sum``（默认 1）；可选单票 ``max_weight``。
换手相对上期权重的硬约束 v1 不做（已有 ``apply_turnover_control`` 控持仓集合）。

与仓位体制组合
--------------
先在 invested book 上优化（权重和为 1），再由 ``apply_exposure`` 缩放
``r_eff = exposure × r_invested``。勿把 ``target_exposure`` 同时折进权重和敞口缩放。
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger

PortfolioOptMethod = Literal["ew", "score", "rank", "mv", "invvol", "rp"]
VALID_METHODS: tuple[str, ...] = ("ew", "score", "rank", "mv", "invvol", "rp")


def normalize_method(method: str | None) -> str:
    if method is None or str(method).strip() == "":
        return "ew"
    m = str(method).strip().lower()
    aliases = {
        "equal": "ew",
        "equal_weight": "ew",
        "score_weight": "score",
        "rank_weight": "rank",
        "mean_variance": "mv",
        "mean-variance": "mv",
        "inverse_vol": "invvol",
        "inv_vol": "invvol",
        "vol_inverse": "invvol",
        "risk_parity": "rp",
        "riskparity": "rp",
    }
    m = aliases.get(m, m)
    if m not in VALID_METHODS:
        raise ValueError(
            f"未知 portfolio_opt={method!r}，可选: {', '.join(VALID_METHODS)}"
        )
    return m


def _apply_cap_and_renorm(
    w: np.ndarray,
    max_weight: float | None,
    target_sum: float,
) -> np.ndarray:
    """Long-only cap + renormalize to target_sum (iterative water-fill)."""
    n = len(w)
    if n == 0:
        return w
    w = np.asarray(w, dtype=np.float64)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    if w.sum() <= 0:
        w = np.ones(n, dtype=np.float64)
    w = w / w.sum() * float(target_sum)

    # max_weight ∈ (0,1)：占 invested book 比例；相对 target_sum 缩放
    if max_weight is None or not np.isfinite(max_weight) or not (0.0 < float(max_weight) < 1.0):
        return w
    cap = float(max_weight) * float(target_sum)
    # 若 n * cap < target_sum，无法满足 → 放宽到等权
    if n * cap + 1e-12 < float(target_sum):
        return np.full(n, float(target_sum) / n)

    for _ in range(n + 2):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        free = ~over
        if not free.any() or excess <= 0:
            break
        free_sum = float(w[free].sum())
        if free_sum <= 0:
            w[free] = excess / int(free.sum())
        else:
            w[free] = w[free] + excess * (w[free] / free_sum)
    w = np.clip(w, 0.0, None)
    s = w.sum()
    if s > 0:
        w = w / s * float(target_sum)
    return w


def _to_dict(stocks: list[str], w: np.ndarray) -> dict[str, float]:
    return {s: float(wi) for s, wi in zip(stocks, w)}


def equal_weight(
    stocks: list[str],
    target_sum: float = 1.0,
    max_weight: float | None = None,
) -> dict[str, float]:
    """等权；与 ``portfolio.equal_weights`` 同口径（无 cap 时）。"""
    if not stocks:
        return {}
    w = np.full(len(stocks), 1.0 / len(stocks), dtype=np.float64)
    w = _apply_cap_and_renorm(w, max_weight, target_sum)
    return _to_dict(stocks, w)


def score_weight(
    scores: pd.Series,
    stocks: list[str],
    target_sum: float = 1.0,
    max_weight: float | None = None,
) -> dict[str, float]:
    """得分正比：``w ∝ score - min(score) + eps``（候选集内非负）。"""
    if not stocks:
        return {}
    s = scores.reindex(stocks).astype("float64")
    vals = s.to_numpy(dtype=np.float64)
    if not np.isfinite(vals).any():
        return equal_weight(stocks, target_sum=target_sum, max_weight=max_weight)
    finite = vals[np.isfinite(vals)]
    base = float(np.min(finite))
    vals = np.where(np.isfinite(vals), vals - base + 1e-8, 0.0)
    w = _apply_cap_and_renorm(vals, max_weight, target_sum)
    return _to_dict(stocks, w)


def rank_weight(
    scores: pd.Series,
    stocks: list[str],
    target_sum: float = 1.0,
    max_weight: float | None = None,
) -> dict[str, float]:
    """Rank 正比：候选集内 ``rank(method='average')``，越高越好。"""
    if not stocks:
        return {}
    s = scores.reindex(stocks).astype("float64")
    if s.notna().sum() == 0:
        return equal_weight(stocks, target_sum=target_sum, max_weight=max_weight)
    ranks = s.rank(method="average", ascending=True).fillna(0.0).to_numpy(dtype=np.float64)
    w = _apply_cap_and_renorm(ranks, max_weight, target_sum)
    return _to_dict(stocks, w)


def _recent_return_matrix(
    returns: pd.DataFrame | None,
    stocks: list[str],
    asof: pd.Timestamp | None,
    lookback: int,
) -> np.ndarray | None:
    """(T, N) 近期收益矩阵；不足则 None。"""
    if returns is None or returns.empty or not stocks:
        return None
    cols = [c for c in stocks if c in returns.columns]
    if len(cols) < len(stocks):
        # 缺列用 NaN 列补齐以保持对齐
        sub = returns.reindex(columns=stocks)
    else:
        sub = returns[stocks]
    if asof is not None:
        sub = sub.loc[:asof]
    if lookback > 0:
        sub = sub.iloc[-int(lookback) :]
    arr = sub.to_numpy(dtype=np.float64)
    if arr.shape[0] < 5:
        return None
    return arr


def _asset_vols(ret_mat: np.ndarray) -> np.ndarray:
    """逐列样本波动；全 NaN → nan。"""
    with np.errstate(invalid="ignore"):
        vol = np.nanstd(ret_mat, axis=0, ddof=1)
    return vol


def inverse_vol_weight(
    stocks: list[str],
    returns: pd.DataFrame | None = None,
    asof: pd.Timestamp | None = None,
    lookback: int = 60,
    target_sum: float = 1.0,
    max_weight: float | None = None,
) -> dict[str, float]:
    """逆波动率：``w ∝ 1 / σ``；波动缺失则回退等权。"""
    if not stocks:
        return {}
    ret_mat = _recent_return_matrix(returns, stocks, asof, lookback)
    if ret_mat is None:
        return equal_weight(stocks, target_sum=target_sum, max_weight=max_weight)
    vol = _asset_vols(ret_mat)
    inv = np.where(np.isfinite(vol) & (vol > 1e-12), 1.0 / vol, np.nan)
    if not np.isfinite(inv).any():
        return equal_weight(stocks, target_sum=target_sum, max_weight=max_weight)
    inv = np.where(np.isfinite(inv), inv, 0.0)
    w = _apply_cap_and_renorm(inv, max_weight, target_sum)
    return _to_dict(stocks, w)


def _shrinkage_cov(ret_mat: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf；失败则对角 + 轻度 shrink 向等权方差。"""
    n = ret_mat.shape[1]
    # 列向填 0（涨跌停/停牌 NaN）以便协方差估计；样本过少时对角兜底
    filled = np.where(np.isfinite(ret_mat), ret_mat, 0.0)
    try:
        from sklearn.covariance import LedoitWolf

        lw = LedoitWolf().fit(filled)
        cov = np.asarray(lw.covariance_, dtype=np.float64)
        if cov.shape == (n, n) and np.isfinite(cov).all():
            # 保证 PSD
            cov = 0.5 * (cov + cov.T)
            return cov
    except Exception as e:
        logger.debug(f"LedoitWolf 失败，改用对角 shrink: {e}")

    var = np.nanvar(ret_mat, axis=0, ddof=1)
    var = np.where(np.isfinite(var) & (var > 1e-12), var, np.nanmedian(var))
    if not np.isfinite(var).any():
        var = np.ones(n, dtype=np.float64)
    else:
        med = float(np.nanmedian(var))
        var = np.where(np.isfinite(var), var, med)
    # shrink 向平均方差
    target = float(np.mean(var))
    alpha = 0.2
    var_s = (1 - alpha) * var + alpha * target
    return np.diag(var_s)


def _mu_from_scores(scores: pd.Series, stocks: list[str]) -> np.ndarray:
    """μ：候选集内 z-score（越高越好）；缺失 → 0。"""
    s = scores.reindex(stocks).astype("float64")
    vals = s.to_numpy(dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    if len(finite) < 2:
        return np.zeros(len(stocks), dtype=np.float64)
    mu = (vals - float(np.mean(finite))) / (float(np.std(finite, ddof=0)) + 1e-12)
    return np.where(np.isfinite(mu), mu, 0.0)


def mean_variance_lite(
    scores: pd.Series,
    stocks: list[str],
    returns: pd.DataFrame | None = None,
    asof: pd.Timestamp | None = None,
    lookback: int = 60,
    risk_aversion: float = 1.0,
    target_sum: float = 1.0,
    max_weight: float | None = None,
) -> dict[str, float]:
    """
    均值-方差 lite：``max μ'w − (λ/2) w'Σw``，long-only + 权重和 + 单票 cap。

    μ 来自得分截面 z-score；Σ 为近期收益 Ledoit-Wolf。
    无 cvxpy / 求解失败 → 回退 ``inverse_vol_weight``。
    """
    if not stocks:
        return {}
    ret_mat = _recent_return_matrix(returns, stocks, asof, lookback)
    if ret_mat is None:
        return score_weight(scores, stocks, target_sum=target_sum, max_weight=max_weight)

    mu = _mu_from_scores(scores, stocks)
    cov = _shrinkage_cov(ret_mat)
    lam = max(float(risk_aversion), 1e-6)
    n = len(stocks)
    cap = None
    if max_weight is not None and np.isfinite(max_weight) and max_weight > 0:
        cap = float(max_weight) * float(target_sum) if max_weight <= 1.0 else float(max_weight)
        if n * cap + 1e-12 < float(target_sum):
            return equal_weight(stocks, target_sum=target_sum, max_weight=None)

    try:
        import cvxpy as cp

        w = cp.Variable(n)
        objective = cp.Maximize(mu @ w - 0.5 * lam * cp.quad_form(w, cp.psd_wrap(cov)))
        cons = [w >= 0, cp.sum(w) == float(target_sum)]
        if cap is not None:
            cons.append(w <= cap)
        prob = cp.Problem(objective, cons)
        prob.solve(solver=cp.OSQP, warm_start=False, verbose=False)
        if w.value is None or not np.isfinite(w.value).all():
            # 再试 SCS
            prob.solve(solver=cp.SCS, verbose=False)
        if w.value is not None and np.isfinite(w.value).all():
            arr = np.asarray(w.value, dtype=np.float64).ravel()
            arr = _apply_cap_and_renorm(arr, max_weight, target_sum)
            return _to_dict(stocks, arr)
    except Exception as e:
        logger.debug(f"MV cvxpy 失败，回退 invvol: {e}")

    return inverse_vol_weight(
        stocks,
        returns=returns,
        asof=asof,
        lookback=lookback,
        target_sum=target_sum,
        max_weight=max_weight,
    )


def risk_parity_weight(
    stocks: list[str],
    returns: pd.DataFrame | None = None,
    asof: pd.Timestamp | None = None,
    lookback: int = 60,
    target_sum: float = 1.0,
    max_weight: float | None = None,
) -> dict[str, float]:
    """
    简易风险平价：最小化风险贡献偏离；失败则 ``inverse_vol_weight``。

    目标：``Σ_i (RC_i − target_sum/n)²``，``RC_i = w_i (Σw)_i``。
    """
    if not stocks:
        return {}
    ret_mat = _recent_return_matrix(returns, stocks, asof, lookback)
    if ret_mat is None:
        return equal_weight(stocks, target_sum=target_sum, max_weight=max_weight)

    cov = _shrinkage_cov(ret_mat)
    n = len(stocks)
    # 先试 invvol 作初值 / 快速路径
    try:
        from scipy.optimize import minimize

        def _obj(x: np.ndarray) -> float:
            x = np.clip(x, 1e-12, None)
            x = x / x.sum() * float(target_sum)
            m = cov @ x
            port_var = float(x @ m)
            if port_var <= 1e-18:
                return 0.0
            rc = x * m
            target = float(target_sum) / n
            return float(np.sum((rc - target) ** 2))

        x0 = np.full(n, float(target_sum) / n)
        bounds = [(1e-8, None)] * n
        cons = {"type": "eq", "fun": lambda x: float(np.sum(x) - float(target_sum))}
        res = minimize(_obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 200, "ftol": 1e-12})
        if res.success and np.isfinite(res.x).all():
            arr = _apply_cap_and_renorm(np.asarray(res.x, dtype=np.float64), max_weight, target_sum)
            return _to_dict(stocks, arr)
    except Exception as e:
        logger.debug(f"risk_parity 失败，回退 invvol: {e}")

    return inverse_vol_weight(
        stocks,
        returns=returns,
        asof=asof,
        lookback=lookback,
        target_sum=target_sum,
        max_weight=max_weight,
    )


def optimize_weights(
    method: str,
    stocks: list[str],
    scores: pd.Series | None = None,
    returns: pd.DataFrame | None = None,
    asof: pd.Timestamp | None = None,
    lookback: int = 60,
    risk_aversion: float = 1.0,
    target_sum: float = 1.0,
    max_weight: float | None = None,
) -> dict[str, float]:
    """
    统一入口。``method='ew'`` 时走等权（与历史 ``equal_weights`` 一致）。

    Parameters
    ----------
    max_weight : float | None
        单票上限（相对 ``target_sum=1`` 的比例，如 0.1=10%）。
        ``None`` / ``<=0`` / ``>=1`` 视为不设 cap（等权时 ``1/n`` 可能超过小 n 的「常识 cap」）。
    """
    m = normalize_method(method)
    if not stocks:
        return {}
    # max_weight ∈ (0,1) 生效；None / ≤0 / ≥1 → 不设单票 cap
    if max_weight is not None and np.isfinite(max_weight) and 0.0 < float(max_weight) < 1.0:
        cap: float | None = float(max_weight)
    else:
        cap = None

    empty_scores = pd.Series(dtype="float64")
    sc = scores if scores is not None else empty_scores

    if m == "ew":
        return equal_weight(stocks, target_sum=target_sum, max_weight=cap)
    if m == "score":
        return score_weight(sc, stocks, target_sum=target_sum, max_weight=cap)
    if m == "rank":
        return rank_weight(sc, stocks, target_sum=target_sum, max_weight=cap)
    if m == "invvol":
        return inverse_vol_weight(
            stocks, returns=returns, asof=asof, lookback=lookback,
            target_sum=target_sum, max_weight=cap,
        )
    if m == "mv":
        return mean_variance_lite(
            sc, stocks, returns=returns, asof=asof, lookback=lookback,
            risk_aversion=risk_aversion, target_sum=target_sum, max_weight=cap,
        )
    if m == "rp":
        return risk_parity_weight(
            stocks, returns=returns, asof=asof, lookback=lookback,
            target_sum=target_sum, max_weight=cap,
        )
    return equal_weight(stocks, target_sum=target_sum, max_weight=cap)
