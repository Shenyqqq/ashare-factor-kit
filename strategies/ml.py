"""
strategies/ml.py  —  ML 多因子策略

调用 WalkForwardTrainer 用所有可用因子训练，输出样本外预测得分。
支持单模型（lgbm/xgb/cat/ridge）和 ensemble（所有模型 Rank Averaging）。
"""
import pandas as pd
from loguru import logger
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from factors.factor import get_factor_registry
from models.trainer import WalkForwardTrainer, build_ml_dataset, MODEL_TYPES
from models.analyzer import MLAnalyzer


def run(
    prices: pd.DataFrame,
    financial: pd.DataFrame,
    prices_raw: pd.DataFrame = None,
    volume: pd.DataFrame = None,
    amount: pd.DataFrame = None,
    model_types: list = None,
    hold_period: int = 20,
    show_report: bool = False,
) -> tuple[pd.DataFrame, WalkForwardTrainer]:
    """
    训练 ML 策略并返回样本外预测得分。

    model_types: 传 None 使用全部 ["ridge","lgbm","xgb","cat"]，
                 传单个列表如 ["lgbm"] 则只用该模型（不做 ensemble）。
    hold_period: 预测未来 N 日收益，决定 forward_return 的计算窗口。
    show_report: 训练完是否立即展示 IC / 分组净值 / SHAP 图表。

    返回 (score_df, trainer)
        score_df: DataFrame(index=调仓日, columns=股票)，越大越优先
        trainer:  训练完成的 WalkForwardTrainer，可用于后续分析
    """
    if model_types is None:
        model_types = MODEL_TYPES

    # 1. 计算所有因子
    registry = get_factor_registry(
        prices=prices,
        financial=financial,
        prices_raw=prices_raw,
        volume=volume,
        amount=amount,
    )
    logger.info(f"ML 策略使用 {len(registry)} 个因子，模型={model_types}")

    # 2. 构建数据集
    forward_return = prices.pct_change(hold_period).shift(-hold_period)
    dataset = build_ml_dataset(registry, forward_return)

    # 3. Walk-Forward 训练
    trainer = WalkForwardTrainer(model_types=model_types)
    score_df = trainer.fit_predict(dataset)

    # 4. 可选：展示分析报告
    if show_report:
        analyzer = MLAnalyzer(trainer)
        analyzer.full_report(prices)

    return score_df, trainer
