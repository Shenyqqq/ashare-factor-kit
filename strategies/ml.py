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


def build_factor_dataset(
    prices: pd.DataFrame,
    financial: pd.DataFrame,
    prices_raw: pd.DataFrame = None,
    volume: pd.DataFrame = None,
    amount: pd.DataFrame = None,
    open_: pd.DataFrame = None,
    high: pd.DataFrame = None,
    low: pd.DataFrame = None,
    clean_ret: pd.DataFrame = None,
    masks: dict = None,
    market_prices: pd.DataFrame = None,
    industry_map: pd.DataFrame = None,
    margin: pd.DataFrame = None,
    moneyflow: pd.DataFrame = None,
    northbound: pd.DataFrame = None,
    institution: pd.DataFrame = None,
    hold_period: int = 20,
):
    """
    构建 MLDataset，供 IndustryWalkForwardTrainer 等复用。

    forward_return 定义：
        有 open_（次日开盘价）时 → close[t+N] / open[t+1] - 1
            即信号日收盘后，次日开盘买入，持有 N 日到收盘卖出的真实收益。
            这与实盘执行完全一致（次日开盘手动买入）。
        无 open_ 时 → close[t+N] / close[t] - 1（退化为收收收益，含隔夜跳空）
    """
    registry = get_factor_registry(
        prices=prices, financial=financial,
        prices_raw=prices_raw, volume=volume, amount=amount,
        open_=open_, high=high, low=low,
        clean_ret=clean_ret, masks=masks,
        market_prices=market_prices, industry_map=industry_map,
        margin=margin, moneyflow=moneyflow,
        northbound=northbound, institution=institution,
    )
    if open_ is not None:
        # 买入价 = 次日开盘，卖出价 = N 日后收盘
        buy_price  = open_.shift(-1)                  # open[t+1]
        sell_price = prices.shift(-hold_period)        # close[t+N]
        forward_return = sell_price / buy_price.replace(0, float("nan")) - 1
    else:
        forward_return = prices.pct_change(hold_period).shift(-hold_period)
    return build_ml_dataset(registry, forward_return)


def run(
    prices: pd.DataFrame,
    financial: pd.DataFrame,
    prices_raw: pd.DataFrame = None,
    volume: pd.DataFrame = None,
    amount: pd.DataFrame = None,
    open_: pd.DataFrame = None,
    high: pd.DataFrame = None,
    low: pd.DataFrame = None,
    clean_ret: pd.DataFrame = None,
    masks: dict = None,
    market_prices: pd.DataFrame = None,
    industry_map: pd.DataFrame = None,
    margin: pd.DataFrame = None,
    moneyflow: pd.DataFrame = None,
    northbound: pd.DataFrame = None,
    institution: pd.DataFrame = None,
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
        prices=prices, financial=financial,
        prices_raw=prices_raw, volume=volume, amount=amount,
        open_=open_, high=high, low=low,
        clean_ret=clean_ret, masks=masks,
        market_prices=market_prices, industry_map=industry_map,
        margin=margin, moneyflow=moneyflow,
        northbound=northbound, institution=institution,
    )
    logger.info(f"ML 策略使用 {len(registry)} 个因子，模型={model_types}")

    # 2. 构建数据集（forward_return 从次日开盘算起，与实盘执行一致）
    if open_ is not None:
        buy_price  = open_.shift(-1)
        sell_price = prices.shift(-hold_period)
        forward_return = sell_price / buy_price.replace(0, float("nan")) - 1
    else:
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
