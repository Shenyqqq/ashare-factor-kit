"""
models/trainer.py  —  Walk-Forward 训练器

WalkForwardTrainer 在每个调仓日用历史数据滚动训练，
输出样本外预测分矩阵（与回测引擎直接对接）。

支持模型：ridge | lgbm | xgb | cat
Ensemble：多窗口 × 多模型 → Rank Averaging
"""
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass
from loguru import logger

import lightgbm as lgb
import xgboost as xgb_lib
import catboost as cb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import PROCESSED_DIR

# ── 超参数（集中配置）────────────────────────────────────────────────────────

TRAIN_WINDOWS_MONTHS = [12, 24, 36]
VAL_WINDOW_MONTHS    = 6
TIME_DECAY           = 0.005
MIN_STOCKS_PER_DATE  = 30
REBALANCE_FREQ       = "ME"
MODEL_TYPES          = ["ridge", "lgbm", "xgb", "cat"]

RIDGE_PARAMS = dict(alpha=10.0)

LGBM_PARAMS = dict(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    num_leaves=15, subsample=0.8, colsample_bytree=0.8,
    min_child_samples=30, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, verbose=-1, n_jobs=-1,
)

XGB_PARAMS = dict(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=30,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, verbosity=0, n_jobs=-1,
)

CAT_PARAMS = dict(
    iterations=300, depth=4, learning_rate=0.05,
    l2_leaf_reg=3, subsample=0.8,
    random_seed=42, verbose=False, thread_count=-1,
)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class MLDataset:
    """
    多因子截面数据集。
    factor_panel:   {因子名: DataFrame(index=日期, columns=股票)}
    forward_return: DataFrame(index=日期, columns=股票)
    """
    factor_panel:    dict
    forward_return:  pd.DataFrame
    rebalance_dates: list
    feature_names:   list

    def get_cross_section(self, date):
        rows = {name: df.loc[date] for name, df in self.factor_panel.items()
                if date in df.index}
        if not rows:
            return None, None
        # 用 0 填充 NaN（z-score 空间里 0 = 截面中性），避免稀疏因子剔除大量股票
        X = pd.DataFrame(rows).fillna(0)
        # 至少要有一个因子有真实值的股票才保留
        has_data = pd.DataFrame(rows).notna().any(axis=1)
        X = X.loc[has_data]
        if date not in self.forward_return.index:
            return None, None
        y = self.forward_return.loc[date].reindex(X.index).dropna()
        X = X.loc[X.index.intersection(y.index)]
        return X, y.loc[X.index]

    def stack_dates(self, dates):
        X_list, y_list, w_list = [], [], []
        for i, date in enumerate(dates):
            X, y = self.get_cross_section(date)
            if X is None or len(X) < MIN_STOCKS_PER_DATE:
                continue
            decay = np.exp(TIME_DECAY * i)
            X_list.append(X.values)
            y_list.append(y.values)
            w_list.extend([decay] * len(y))
        if not X_list:
            return None, None, None
        return np.vstack(X_list), np.concatenate(y_list), np.array(w_list)


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def build_ml_dataset(factor_dict, forward_return, rebalance_freq=REBALANCE_FREQ):
    if not factor_dict:
        raise ValueError("factor_dict 为空，没有任何因子被计算")
    all_dates = sorted(
        set.intersection(*[set(df.index) for df in factor_dict.values()])
        & set(forward_return.index)
    )
    if not all_dates:
        ranges = {n: f"[{df.index.min()}, {df.index.max()}]" if len(df) > 0 else "EMPTY"
                  for n, df in factor_dict.items()}
        raise ValueError(
            f"all_dates 为空：各因子日期无公共交集。\n"
            f"forward_return: [{forward_return.index.min()}, {forward_return.index.max()}]\n"
            f"各因子范围: {ranges}"
        )
    rebalance_dates = [
        d for d in pd.date_range(all_dates[0], all_dates[-1], freq=rebalance_freq)
        if d in all_dates
    ]
    logger.info(f"数据集: {len(factor_dict)}个因子, {len(rebalance_dates)}个调仓日")
    return MLDataset(
        factor_panel=factor_dict,
        forward_return=forward_return,
        rebalance_dates=rebalance_dates,
        feature_names=list(factor_dict.keys()),
    )


def to_rank(arr):
    if len(arr) == 0:
        return arr
    order = arr.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(arr))
    return ranks / max(len(arr) - 1, 1)


def rank_average(rank_list):
    return np.mean(rank_list, axis=0)


def spearman_ic(pred, actual):
    if len(pred) < 5:
        return np.nan
    return pd.Series(pred).rank().corr(pd.Series(actual).rank())


# ── 单模型构造与训练 ──────────────────────────────────────────────────────────

def _build_model(model_type):
    if model_type == "ridge":
        return Pipeline([("scaler", StandardScaler()), ("model", Ridge(**RIDGE_PARAMS))])
    elif model_type == "lgbm":
        return lgb.LGBMRegressor(**LGBM_PARAMS)
    elif model_type == "xgb":
        return xgb_lib.XGBRegressor(**XGB_PARAMS)
    elif model_type == "cat":
        return cb.CatBoostRegressor(**CAT_PARAMS)
    raise ValueError(f"未知模型: {model_type}")


def _fit_model(model_type, X_tr, y_tr, w_tr, X_va, y_va):
    model = _build_model(model_type)
    if model_type == "ridge":
        model.fit(X_tr, y_tr, **{"model__sample_weight": w_tr})
    elif model_type == "lgbm":
        model.fit(X_tr, y_tr, sample_weight=w_tr,
                  eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
    elif model_type == "xgb":
        model.fit(X_tr, y_tr, sample_weight=w_tr,
                  eval_set=[(X_va, y_va)], early_stopping_rounds=30, verbose=False)
    elif model_type == "cat":
        model.fit(X_tr, y_tr, sample_weight=w_tr,
                  eval_set=(X_va, y_va), early_stopping_rounds=30)
    return model


# ── Walk-Forward 训练器 ───────────────────────────────────────────────────────

class WalkForwardTrainer:
    """
    每个调仓日：多窗口 × 多模型 → Rank Averaging → 最终得分。

    调用：
        trainer = WalkForwardTrainer(model_types=["lgbm", "xgb", "cat"])
        score_df = trainer.fit_predict(dataset)
    """

    def __init__(
        self,
        train_windows=TRAIN_WINDOWS_MONTHS,
        val_window=VAL_WINDOW_MONTHS,
        model_types=MODEL_TYPES,
    ):
        self.train_windows = train_windows
        self.val_window    = val_window
        self.model_types   = model_types
        self.score_df:  pd.DataFrame = None
        self.ic_series: pd.Series    = None
        self.models:    dict         = {}
        self.model_ic:  dict         = {m: {} for m in model_types}
        self._dataset:  MLDataset    = None

    def fit_predict(self, dataset: MLDataset) -> pd.DataFrame:
        self._dataset = dataset
        dates = dataset.rebalance_dates
        min_history  = max(self.train_windows) + self.val_window
        predict_start = min_history

        all_scores = {}
        n_predict  = len(dates) - predict_start
        logger.info(f"Walk-Forward: 预测{n_predict}个调仓日, 模型={self.model_types}")

        for idx in range(predict_start, len(dates)):
            pred_date = dates[idx]
            X_pred, y_true = dataset.get_cross_section(pred_date)
            if X_pred is None or len(X_pred) < MIN_STOCKS_PER_DATE:
                continue

            rank_collector = {m: [] for m in self.model_types}

            for window in self.train_windows:
                val_end    = idx
                val_start  = val_end - self.val_window
                train_end  = val_start
                train_start = max(0, train_end - window)

                train_dates = dates[train_start:train_end]
                val_dates   = dates[val_start:val_end]
                if len(train_dates) < 12:
                    continue

                X_tr, y_tr, w_tr = dataset.stack_dates(train_dates)
                X_va, y_va, _    = dataset.stack_dates(val_dates)
                if X_tr is None or X_va is None:
                    continue

                for model_type in self.model_types:
                    try:
                        model = _fit_model(model_type, X_tr, y_tr, w_tr, X_va, y_va)
                        pred  = model.predict(X_pred.values)
                        rank_collector[model_type].append(to_rank(pred))
                        self.models[(window, model_type)] = model
                    except Exception as e:
                        logger.warning(f"训练失败({model_type}, w={window}, {pred_date.date()}): {e}")

            model_scores = {}
            for model_type in self.model_types:
                ranks = rank_collector[model_type]
                if ranks:
                    model_scores[model_type] = rank_average(ranks)
                    ic = spearman_ic(model_scores[model_type], y_true.values)
                    self.model_ic[model_type][pred_date] = ic

            if not model_scores:
                continue

            final_rank = rank_average(list(model_scores.values()))
            all_scores[pred_date] = pd.Series(final_rank, index=X_pred.index)

            done = idx - predict_start + 1
            if done % 6 == 0 or done == n_predict:
                logger.info(f"进度 {done}/{n_predict}: {pred_date.date()}, 股票={len(X_pred)}")

        self.score_df = pd.DataFrame(all_scores).T
        self.score_df.index.name = "date"

        # 样本外 IC
        ic_dict = {}
        for date in self.score_df.index:
            _, y = dataset.get_cross_section(date)
            if y is None:
                continue
            s = self.score_df.loc[date].dropna()
            y = y.reindex(s.index).dropna()
            s = s.loc[y.index]
            if len(s) >= MIN_STOCKS_PER_DATE:
                ic_dict[date] = spearman_ic(s.values, y.values)
        self.ic_series = pd.Series(ic_dict)

        for m in self.model_types:
            self.model_ic[m] = pd.Series(self.model_ic[m])

        ic_mean = self.ic_series.mean()
        icir    = ic_mean / self.ic_series.std()
        logger.info(f"完成: IC均值={ic_mean:.4f}, ICIR={icir:.4f}, 胜率={(self.ic_series > 0).mean():.1%}")

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        self.score_df.to_parquet(PROCESSED_DIR / "ml_factor_scores.parquet")
        return self.score_df

    def ic_summary(self):
        ic = self.ic_series.dropna()
        return pd.Series({
            "IC均值":   round(ic.mean(), 4),
            "IC标准差": round(ic.std(), 4),
            "ICIR":     round(ic.mean() / ic.std(), 4),
            "IC>0胜率": round((ic > 0).mean(), 4),
        })
