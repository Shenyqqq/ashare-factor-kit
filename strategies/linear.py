"""
strategies/linear.py  —  线性加权多因子策略

把因子字典 + 权重字典合成一张得分表，直接喂给回测引擎。
这是最简单的基准策略，用于和 ML 策略对比。
"""
import pandas as pd
from loguru import logger
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from factors.factor import get_factor_registry


def run(
    prices: pd.DataFrame,
    financial: pd.DataFrame,
    weights: dict,
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
) -> pd.DataFrame:
    """
    计算线性加权合成因子得分。

    weights: {因子名: 权重}，key 对应 get_factor_registry() 返回的因子名。
    权重正负已在各因子函数内处理（越高越好），这里直接加权求和。

    返回 DataFrame(index=日期, columns=股票)，值越大选股优先级越高。
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

    composite = None
    for name, w in weights.items():
        if name not in registry:
            logger.warning(f"因子 '{name}' 不在注册表中，跳过")
            continue
        f = registry[name] * w
        composite = f if composite is None else composite.add(f, fill_value=0)

    if composite is None:
        raise ValueError("没有任何有效因子，请检查 weights 配置")

    logger.info(f"线性策略合成完成，使用因子: {[k for k in weights if k in registry]}")
    return composite
