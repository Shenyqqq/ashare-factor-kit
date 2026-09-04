"""
models/wf/tuning.py — Optuna 轻量超参搜索（LGBM / XGB / RF / Cat）

机构级做法：离线搜索 + 锁定生产。
- `--tune` 触发搜索，用前 N 折 purged walk-forward 做 CV objective
- 结果存 config/tuned_params.json
- 正常训练读 JSON（有则用调过的，无则用 params.py 默认值）
- 不在每次训练时搜索（避免过拟合 CV + 节省时间）

支持模型：
  - lgbm: max_depth, num_leaves, learning_rate, n_estimators, min_child_samples, subsample, colsample_bytree
  - xgb:  max_depth, learning_rate, n_estimators, min_child_weight, subsample, colsample_bytree
  - rf:   n_estimators, max_depth, max_features, min_samples_leaf
  - cat:  depth, learning_rate, l2_leaf_reg, iterations（CatBoost ordered boosting，抗过拟合）

不支持：
  - ridge: 已用 RidgeCV 内置 alpha 选择
  - mlp: 小样本上调参过拟合风险高
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import optuna

from models.trainer import LABEL_MODE_DEFAULT, TIME_DECAY

from models.wf.metrics import spearman_ic
from models.wf.splits import (
    embargo_train_end,
    get_window_splits,
    hold_period_to_embargo_periods,
    purge_train_indices,
)
from models.wf.labels import transform_labels

optuna.logging.set_verbosity(optuna.logging.WARNING)

TUNABLE_MODELS = {"lgbm", "xgb", "rf", "cat"}
TUNED_PARAMS_PATH = Path("config/tuned_params.json")


# ── 搜索空间 ──────────────────────────────────────────────────────────────

def define_search_space(trial: optuna.Trial, model_type: str) -> dict:
    """用 trial.suggest_* 定义各模型搜索空间，返回超参 dict。"""
    if model_type == "lgbm":
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 40),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
    if model_type == "xgb":
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "min_child_weight": trial.suggest_int("min_child_weight", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
    if model_type == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 40, 120),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "max_features": trial.suggest_float("max_features", 0.4, 0.8),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 30),
        }
    if model_type == "cat":
        # CatBoost ordered boosting：深度(对称树)、学习率、L2 叶正则、迭代轮数
        return {
            "depth": trial.suggest_int("depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "iterations": trial.suggest_int("iterations", 100, 400),
        }
    raise ValueError(f"不支持的调参模型: {model_type}，可选: {TUNABLE_MODELS}")


def _build_model_with_params(model_type: str, params: dict, n_jobs: int = -1):
    """用给定超参 build 模型（带固定 random_state / 静默 / 正则默认值）。"""
    if model_type == "lgbm":
        import lightgbm as lgb
        merged = dict(
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbose=-1, n_jobs=n_jobs,
        )
        merged.update(params)
        return lgb.LGBMRegressor(**merged)
    if model_type == "xgb":
        import xgboost as xgb_lib
        merged = dict(
            reg_alpha=0.1, reg_lambda=1.0,
            early_stopping_rounds=30,
            random_state=42, verbosity=0, n_jobs=n_jobs,
        )
        merged.update(params)
        return xgb_lib.XGBRegressor(**merged)
    if model_type == "rf":
        from sklearn.ensemble import RandomForestRegressor
        merged = dict(random_state=42, n_jobs=n_jobs)
        merged.update(params)
        return RandomForestRegressor(**merged)
    if model_type == "cat":
        import catboost as cb
        # 固定 rsm/subsample/random_strength/random_seed/verbose 等非搜索项，
        # 仅让 depth/learning_rate/l2_leaf_reg/iterations 进入搜索空间。
        merged = dict(
            subsample=0.8, rsm=0.8, random_strength=1,
            random_seed=42, verbose=False, thread_count=n_jobs,
        )
        merged.update(params)
        return cb.CatBoostRegressor(**merged)
    raise ValueError(f"不支持的调参模型: {model_type}")


def _fit_predict_ic(model_type: str, params: dict,
                    X_tr, y_tr, w_tr, X_va, y_va) -> float:
    """build + fit + predict，返回验证集 Spearman IC。"""
    model = _build_model_with_params(model_type, params, n_jobs=-1)
    if model_type == "lgbm":
        model.fit(
            X_tr, y_tr, sample_weight=w_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[
                __import__("lightgbm").early_stopping(30, verbose=False),
                __import__("lightgbm").log_evaluation(-1),
            ],
        )
    elif model_type == "xgb":
        model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)], verbose=False)
    elif model_type == "cat":
        model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=(X_va, y_va), early_stopping_rounds=30)
    else:  # rf
        model.fit(X_tr, y_tr, sample_weight=w_tr)
    pred = model.predict(X_va)
    return spearman_ic(pred, y_va)


# ── 单模型搜索 ────────────────────────────────────────────────────────────

def tune_one_model(model_type: str,
                   X_train, y_train, w_train,
                   X_val, y_val,
                   n_trials: int = 15,
                   random_state: int = 42) -> dict:
    """对单模型做 Optuna 搜索，返回 best_params。"""
    if model_type not in TUNABLE_MODELS:
        raise ValueError(f"不支持的调参模型: {model_type}，可选: {TUNABLE_MODELS}")

    def objective(trial: optuna.Trial) -> float:
        params = define_search_space(trial, model_type)
        try:
            ic = _fit_predict_ic(model_type, params,
                                 X_train, y_train, w_train,
                                 X_val, y_val)
        except Exception:
            return 0.0
        if not np.isfinite(ic):
            return 0.0
        return float(ic)

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = dict(study.best_params)
    best["_best_ic"] = float(study.best_value) if study.best_value is not None else np.nan
    return best


# ── 数据集 → 单折训练/验证集 ──────────────────────────────────────────────

def _stack_sections(
    dataset, date_list, label_mode: str = LABEL_MODE_DEFAULT,
    time_decay: float = TIME_DECAY,
):
    """把多个调仓日的截面堆成 (X, y, w)；y 走 transform_labels，w 用时间衰减。"""
    X_list, y_list, w_list = [], [], []
    for i, d in enumerate(date_list):
        X, y = dataset.get_cross_section(d)
        if X is None or len(X) < 5:
            continue
        y_t = transform_labels(y.values.astype(np.float32), label_mode)
        decay = float(np.exp(float(time_decay) * i))
        X_list.append(X.values.astype(np.float32))
        y_list.append(y_t)
        w_list.extend([decay] * len(y_t))
    if not X_list:
        return None
    return (
        np.vstack(X_list),
        np.concatenate(y_list),
        np.array(w_list, dtype=np.float32),
    )


def _extract_first_fold(dataset, train_windows: list, val_window: int,
                        hold_period: int, label_mode: str = LABEL_MODE_DEFAULT,
                        time_decay: float = TIME_DECAY):
    """
    取第一个可用 walk-forward 折的训练+验证集（purged + embargo）。
    返回 (X_tr, y_tr, w_tr, X_va, y_va) 或 None。
    """
    dates = dataset.rebalance_dates
    n_dates = len(dates)
    windows = sorted(train_windows)
    min_w = windows[0]
    embargo_periods = hold_period_to_embargo_periods(hold_period, dates)
    date_to_pos = {d: i for i, d in enumerate(dates)}

    # 共用近期 val：首折起点 = max(W)+V（不再加旧错位 offset）
    predict_start = max(windows) + val_window
    for idx in range(predict_start, n_dates):
        for window in windows:
            ts, te, vs, ve = get_window_splits(
                idx, window, val_window, n_dates,
                min_train_window=min_w, window_specific_val=True,
            )
            train_dates = dates[ts:te]
            val_dates = dates[vs:ve]
            if te > 0:
                eff_te = embargo_train_end(te, embargo_periods)
                train_dates = dates[ts:eff_te]
            train_dates = purge_train_indices(
                train_dates, val_dates, dates[idx], dates, hold_period,
                date_pos_map=date_to_pos,
            )
            no_val = val_window == 0
            if len(train_dates) < max(8, window // 3):
                continue
            if not no_val and len(val_dates) < 2:
                continue
            tr = _stack_sections(
                dataset, train_dates, label_mode, time_decay=time_decay,
            )
            if tr is None:
                continue
            if no_val:
                # Optuna 需要 eval 指标；无独立 val 时从 train 尾部切一小段作 surrogate
                # （仅 tuning；正式 WF 无 val 时关闭 early stop）
                X_tr, y_tr, w_tr = tr
                n = len(y_tr)
                cut = max(int(n * 0.85), 1)
                if cut >= n:
                    continue
                return X_tr[:cut], y_tr[:cut], w_tr[:cut], X_tr[cut:], y_tr[cut:]
            va = _stack_sections(
                dataset, val_dates, label_mode, time_decay=time_decay,
            )
            if va is None:
                continue
            X_tr, y_tr, w_tr = tr
            X_va, y_va, _ = va
            return X_tr, y_tr, w_tr, X_va, y_va
    return None


# ── 多模型搜索 ────────────────────────────────────────────────────────────

def tune_all_models(model_types: list, dataset,
                    train_windows: list, val_window: int, hold_period: int,
                    n_trials: int = 15, label_mode: str = LABEL_MODE_DEFAULT,
                    time_decay: float = TIME_DECAY) -> dict:
    """
    对多个模型做 Optuna 搜索。
    用 dataset 第一个 walk-forward 折作为 CV 数据。
    返回 {model_type: best_params}。
    """
    fold = _extract_first_fold(
        dataset, train_windows, val_window,
        hold_period, label_mode=label_mode, time_decay=time_decay,
    )
    if fold is None:
        raise RuntimeError("无法从 dataset 构建首个 walk-forward 折（数据不足）")
    X_tr, y_tr, w_tr, X_va, y_va = fold
    print(f"[tune] 首折样本: train={X_tr.shape}, val={X_va.shape}")

    results: dict[str, dict] = {}
    for mt in model_types:
        if mt not in TUNABLE_MODELS:
            print(f"[tune] 跳过 {mt}（不在可调参列表 {sorted(TUNABLE_MODELS)}）")
            continue
        print(f"[tune] 搜索 {mt} ... (n_trials={n_trials})")
        best = tune_one_model(mt, X_tr, y_tr, w_tr, X_va, y_va, n_trials=n_trials)
        ic = best.pop("_best_ic", np.nan)
        print(f"[tune] {mt} best IC={ic:.4f}, params={best}")
        results[mt] = best
    return results


# ── JSON 读写 ─────────────────────────────────────────────────────────────

def save_tuned_params(params_dict: dict, path: str | Path = TUNED_PARAMS_PATH) -> Path:
    """存 JSON。剔除 _best_ic 等内部字段。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    clean = {}
    for k, v in params_dict.items():
        clean[k] = {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
    p.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_tuned_params(path: str | Path = TUNED_PARAMS_PATH) -> dict | None:
    """读 JSON，不存在或为空返回 None。"""
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not data:
        return None
    return data


# ── CLI 入口 ──────────────────────────────────────────────────────────────

def _cli_main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(description="Optuna 超参搜索（LGBM/XGB/RF/Cat）")
    parser.add_argument("--models", default="lgbm,xgb,rf,cat",
                        help="待调参模型，逗号分隔（默认 lgbm,xgb,rf,cat）")
    parser.add_argument("--horizon", type=int, default=20,
                        help="持仓期（用于 embargo 换算）")
    parser.add_argument("--n-trials", type=int, default=15,
                        help="每模型 Optuna 搜索轮数（默认 15）")
    parser.add_argument("--train-windows", default="6,12",
                        help="训练窗口月数，逗号分隔（默认 6,12）")
    parser.add_argument("--val-window", type=int, default=6,
                        help="验证窗口月数（默认 6；两窗共用近期 val）")
    parser.add_argument("--factor-config", default=None,
                        help="因子白名单 YAML/JSON（与 run.py 一致）")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--label-mode", default=LABEL_MODE_DEFAULT,
                        choices=["raw", "cs_rank", "cs_zscore"],
                        help="标签模式（barra_residual 需额外依赖，此处不开放）")
    parser.add_argument("--output", default=str(TUNED_PARAMS_PATH),
                        help="输出 JSON 路径")
    args = parser.parse_args(argv)

    model_types = [m.strip() for m in args.models.split(",") if m.strip()]
    train_windows = [int(x) for x in args.train_windows.split(",")]
    hold_period = args.horizon

    # 数据加载 + 数据集构建（复用 run.py 与 strategies.ml）
    from run import _load_data, _horizon_to_rebalance_freq, _load_factor_config
    from strategies.ml import build_factor_dataset
    from models.trainer import resolve_train_windows

    rebalance_freq = _horizon_to_rebalance_freq(hold_period)
    (prices, prices_raw, financial, volume, amount,
     open_, high, low, clean_ret, masks,
     market_prices, industry_map,
     margin, moneyflow, northbound, institution) = _load_data(
        args.skip_download, args.sample)

    factor_whitelist = (_load_factor_config(args.factor_config, hold_period)
                        if args.factor_config else None)
    extra_kwargs = dict(
        prices_raw=prices_raw, volume=volume, amount=amount,
        open_=open_, high=high, low=low,
        clean_ret=clean_ret, masks=masks,
        market_prices=market_prices, industry_map=industry_map,
        margin=margin, moneyflow=moneyflow,
        northbound=northbound, institution=institution,
    )
    dataset = build_factor_dataset(
        prices, financial, hold_period=hold_period,
        factor_whitelist=factor_whitelist,
        rebalance_freq=rebalance_freq,
        **extra_kwargs,
    )

    # 月 → 调仓期数
    win_periods, val_periods = resolve_train_windows(
        train_windows, args.val_window, rebalance_freq, units="months")

    best = tune_all_models(
        model_types, dataset,
        train_windows=win_periods, val_window=val_periods,
        hold_period=hold_period, n_trials=args.n_trials,
        label_mode=args.label_mode,
    )
    out = save_tuned_params(best, path=args.output)
    print(f"[tune] 超参搜索完成，结果存 {out}")


if __name__ == "__main__":
    _cli_main()
