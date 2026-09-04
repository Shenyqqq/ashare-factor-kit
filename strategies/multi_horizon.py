"""
strategies/multi_horizon.py — 多期限集成选股

运行 h5 + h10 + h20 三个 WalkForwardTrainer，对每期预测做 rank 平均：
  composite_score = Σ w_h * rank(predictions_h)

权重 w_h 可配置（默认等权，或按各 horizon 的 OOS IC 加权）。

背景：
  不同因子在不同期限生效。h5 捕捉短期反转/资金面，h20 捕捉中期动量/基本面。
  集成多期限预测比单一期限更稳健——分散了 horizon-specific 噪声，并融合
  短/中/长期信号。

用法（CLI）：
  python run.py --multi-horizon 5,10,20 --mh-weights equal
  python run.py --multi-horizon 5,10,20 --mh-weights ic_weighted

用法（Python）：
  from strategies.multi_horizon import run_multi_horizon
  composite, info = run_multi_horizon(
      horizons=[5, 10, 20],
      weights="ic_weighted",
      prices=prices, financial=financial,
      extra_kwargs=..., cache_kwargs=...,
      model_types=["lgbm", "xgb"],
      factor_whitelist=...,
      tag="mh_5-10-20",
  )
  # composite: DataFrame(index=调仓日, columns=股票)，越大越优先
  # info: {"weights": {h: w}, "ic_per_h": {h: ic_mean}, "trainers": {h: trainer}}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger


from models.trainer import LABEL_MODE_DEFAULT


def _resolve_rebalance_freq(horizon: int) -> str:
    """与 run.py._horizon_to_rebalance_freq 对齐。"""
    if horizon <= 3:
        return "3D"
    elif horizon <= 7:
        return "W-FRI"
    elif horizon <= 15:
        return "2W-FRI"
    else:
        return "ME"


def _resolve_weights(
    horizons: list[int],
    weights,
    ic_per_h: dict[int, float],
) -> dict[int, float]:
    """
    解析权重方式：
      - None / "equal"        → 等权
      - "ic_weighted"         → 按 OOS IC 均值加权（负 IC 截断为 0）
      - list[float]           → 自定义权重（长度需与 horizons 一致）
    """
    n = len(horizons)
    if weights is None or weights == "equal":
        w = {h: 1.0 / n for h in horizons}
    elif weights == "ic_weighted":
        ic_vals = {h: max(ic_per_h.get(h, 0.0), 0.0) for h in horizons}
        total = sum(ic_vals.values())
        if total <= 0:
            logger.warning(
                "ic_weighted: 所有 horizon OOS IC ≤ 0，回退为等权"
            )
            w = {h: 1.0 / n for h in horizons}
        else:
            w = {h: ic_vals[h] / total for h in horizons}
    elif isinstance(weights, (list, tuple)):
        if len(weights) != n:
            raise ValueError(
                f"weights 长度 {len(weights)} 与 horizons 长度 {n} 不一致"
            )
        total = float(sum(weights))
        if total <= 0:
            raise ValueError("weights 之和必须为正")
        w = {h: float(weights[i]) / total for i, h in enumerate(horizons)}
    else:
        raise ValueError(
            f"未知 weights 类型: {type(weights)}，可选 None|'equal'|'ic_weighted'|list[float]"
        )
    return w


def run_multi_horizon(
    horizons=(5, 10, 20),
    weights=None,
    *,
    prices: pd.DataFrame,
    financial: pd.DataFrame,
    extra_kwargs: dict | None = None,
    cache_kwargs: dict | None = None,
    model_types: list | None = None,
    factor_whitelist: list | None = None,
    train_windows: list | None = None,
    train_window_units: str = "months",
    val_window: int | None = None,
    artifact_dir=None,
    wf_selection: str = "ic_weighted",
    label_mode: str = LABEL_MODE_DEFAULT,
    ensemble_method: str = "zscore",
    save_models: bool = False,
    objective: str = "regression",
    tag: str | None = None,
    device: str = "cpu",
    barra_factors: dict[str, pd.DataFrame] | None = None,
    industry_map: pd.Series | None = None,
    regime_states: pd.Series | None = None,
    show_report: bool = False,
    enable_shap: bool = False,
    shap_top: int = 20,
    shap_max_samples: int = 500,
    shap_max_dates: int = 12,
) -> tuple[pd.DataFrame, dict]:
    """
    多期限集成训练。

    对每个 horizon 调用 ``strategies.ml.run``（复用 WalkForwardTrainer /
    RegimeConditionalTrainer 流程），收集 factor_scores，再对每期做
    rank(pct=True) 加权平均得到 composite_score。

    Parameters
    ----------
    horizons : list[int]
        持仓期限组合（如 [5, 10, 20]）。
    weights : None | "equal" | "ic_weighted" | list[float]
        集成权重方式。
    prices, financial : 行情/财务数据（透传给 strategies.ml.run）。
    extra_kwargs, cache_kwargs : dict
        与 run.py 一致的共享参数（prices_raw / volume / masks / use_factor_cache 等）。
    regime_states : pd.Series, optional
        若提供，则每个 horizon 的训练使用 RegimeConditionalTrainer（任务2）。
    tag : str, optional
        总标签；每个 horizon 子训练会附加 _h{N} 后缀，避免产物互相覆盖。

    Returns
    -------
    composite : pd.DataFrame
        index=调仓日, columns=股票，越大越优先。
    info : dict
        包含 ``weights`` / ``ic_per_h`` / ``trainers``。
    """
    from strategies.ml import run as ml_run

    extra_kwargs = extra_kwargs or {}
    cache_kwargs = cache_kwargs or {}

    horizons = list(horizons)
    if len(horizons) < 2:
        raise ValueError(
            f"multi_horizon 至少需要 2 个 horizon，得到 {horizons}"
        )

    per_h_scores: dict[int, pd.DataFrame] = {}
    ic_per_h: dict[int, float] = {}
    trainers: dict = {}

    base_tag = tag or "mh"

    for h in horizons:
        logger.info(f"=== Multi-Horizon: 训练 h={h} ===")
        rebalance_freq = _resolve_rebalance_freq(h)
        sub_tag = f"{base_tag}_h{h}"
        sub_artifact = (
            Path(artifact_dir) / f"h{h}" if artifact_dir else None
        )
        if sub_artifact is not None:
            sub_artifact.mkdir(parents=True, exist_ok=True)

        score_df, trainer = ml_run(
            prices, financial,
            model_types=model_types,
            hold_period=h,
            show_report=show_report,
            factor_whitelist=factor_whitelist,
            train_windows=train_windows,
            train_window_units=train_window_units,
            val_window=val_window,
            artifact_dir=sub_artifact,
            rebalance_freq=rebalance_freq,
            wf_selection=wf_selection,
            label_mode=label_mode,
            ensemble_method=ensemble_method,
            save_models=save_models,
            objective=objective,
            tag=sub_tag,
            barra_factors=barra_factors,
            industry_map=industry_map,
            device=device,
            regime_states=regime_states,
            enable_shap=enable_shap,
            shap_top=shap_top,
            shap_max_samples=shap_max_samples,
            shap_max_dates=shap_max_dates,
            **extra_kwargs,
            **cache_kwargs,
        )
        per_h_scores[h] = score_df
        trainers[h] = trainer
        if (
            hasattr(trainer, "ic_series")
            and trainer.ic_series is not None
            and not trainer.ic_series.empty
        ):
            ic_per_h[h] = float(trainer.ic_series.mean())
        else:
            ic_per_h[h] = 0.0
        logger.info(
            f"h={h} 训练完成: shape={score_df.shape}, "
            f"OOS IC 均值={ic_per_h[h]:.4f}"
        )

    # ── 解析权重 ─────────────────────────────────────────────────────────────
    w = _resolve_weights(horizons, weights, ic_per_h)
    logger.info(
        "Multi-Horizon 权重: "
        + ", ".join(f"h{h}={w[h]:.3f}(IC={ic_per_h[h]:.4f})" for h in horizons)
    )

    # ── 对齐日期，做 rank(pct=True) 加权平均 ────────────────────────────────
    common_dates: pd.DatetimeIndex | None = None
    for h, df in per_h_scores.items():
        if df is None or df.empty:
            continue
        idx = df.index
        common_dates = (
            idx if common_dates is None else common_dates.intersection(idx)
        )
    if common_dates is None or len(common_dates) == 0:
        raise ValueError(
            "多期限无公共预测日，请检查各 horizon 数据范围是否重叠"
        )
    common_dates = pd.DatetimeIndex(sorted(common_dates))

    composite: pd.DataFrame | None = None
    for h, df in per_h_scores.items():
        if df is None or df.empty:
            continue
        rank_df = df.loc[common_dates].rank(axis=1, pct=True)
        contribution = (w[h] * rank_df).astype("float32")
        if composite is None:
            composite = contribution
        else:
            composite = composite.add(contribution, fill_value=0.0)

    if composite is None:
        raise ValueError("所有 horizon 训练均未产生有效得分")

    composite.index.name = "date"

    info = {
        "weights": w,
        "ic_per_h": ic_per_h,
        "trainers": trainers,
        "horizons": horizons,
    }
    return composite, info


def _parse_horizons(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="多期限集成选股（CLI smoke — 完整流程请走 run.py --multi-horizon）"
    )
    parser.add_argument(
        "--horizons", default="5,10,20",
        help="期限组合，逗号分隔（默认 5,10,20）",
    )
    parser.add_argument(
        "--mh-weights", default="equal",
        choices=["equal", "ic_weighted"],
        help="集成权重方式（默认等权）",
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument(
        "--factor-config", default=None,
        help="因子白名单 YAML/JSON（按 horizon 选 h{N} 段）",
    )
    parser.add_argument(
        "--models", default="lgbm,xgb",
        help="ensemble 使用的模型子集（默认 lgbm,xgb）",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="实验产物目录（默认 results/multi_horizon_<weights>/）",
    )
    args = parser.parse_args()

    # 复用 run.py 的数据加载逻辑
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from run import _load_data, _load_factor_config, _resolve_output_dir
    from config.settings import BACKTEST_START, BACKTEST_END

    horizons = _parse_horizons(args.horizons)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    weights = args.mh_weights

    logger.info(f"Multi-Horizon CLI: horizons={horizons}, weights={weights}")

    (prices, prices_raw, financial, volume, amount,
     open_, high, low, clean_ret, masks,
     market_prices, industry_map,
     margin, moneyflow, northbound, institution) = _load_data(
        args.skip_download, args.sample,
    )

    extra_kwargs = dict(
        prices_raw=prices_raw, volume=volume, amount=amount,
        open_=open_, high=high, low=low,
        clean_ret=clean_ret, masks=masks,
        market_prices=market_prices, industry_map=industry_map,
        margin=margin, moneyflow=moneyflow,
        northbound=northbound, institution=institution,
    )
    cache_kwargs = dict(use_factor_cache=True)

    # 每个 horizon 加载各自的因子白名单（YAML 中 h5/h10/h20 段）
    # 此处简化：传 None，由 ml.run 自行处理
    factor_whitelist = (
        _load_factor_config(args.factor_config, horizons[0])
        if args.factor_config else None
    )

    tag = f"multi_horizon_{weights}"
    out_dir = _resolve_output_dir(args.output_dir, tag)

    composite, info = run_multi_horizon(
        horizons=horizons,
        weights=weights,
        prices=prices, financial=financial,
        extra_kwargs=extra_kwargs, cache_kwargs=cache_kwargs,
        model_types=models,
        factor_whitelist=factor_whitelist,
        tag=tag,
        artifact_dir=out_dir,
    )

    composite.to_parquet(out_dir / f"factor_scores_{tag}.parquet")
    logger.info(
        f"composite scores saved → {out_dir / f'factor_scores_{tag}.parquet'}"
    )
    logger.info(
        "权重: " + ", ".join(
            f"h{h}={info['weights'][h]:.3f}" for h in info["horizons"]
        )
    )
