"""
strategies/ml.py  —  ML 多因子策略

调用 WalkForwardTrainer 训练（模块化 WF：purged split + embargo + IC 加权 ensemble），
输出样本外预测得分。支持单模型（lgbm/xgb/cat/ridge/rf/mlp）和 ensemble。
"""
import pandas as pd
from loguru import logger
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import BACKTEST_START, BACKTEST_END
from factors.factor import get_factor_registry
from factors.factor_cache import (
    cache_exists, factor_cache_path, load_factor_panel, save_factor_panel,
)
from models.trainer import (
    WalkForwardTrainer, RegimeConditionalTrainer,
    build_ml_dataset, MODEL_TYPES, REBALANCE_FREQ,
)


def _compute_forward_return(
    prices: pd.DataFrame,
    hold_period: int,
    open_: pd.DataFrame = None,
) -> pd.DataFrame:
    if open_ is not None:
        buy_price = open_.shift(-1)
        sell_price = prices.shift(-hold_period)
        return (
            sell_price / buy_price.replace(0, float("nan")) - 1
        ).astype("float32")
    return (
        prices.pct_change(hold_period).shift(-hold_period)
    ).astype("float32")


def _registry_kwargs(
    prices, financial, prices_raw, volume, amount,
    open_, high, low, clean_ret, masks,
    market_prices, industry_map, margin, moneyflow,
    northbound, institution,
) -> dict:
    return dict(
        prices=prices, financial=financial,
        prices_raw=prices_raw, volume=volume, amount=amount,
        open_=open_, high=high, low=low,
        clean_ret=clean_ret, masks=masks,
        market_prices=market_prices, industry_map=industry_map,
        margin=margin, moneyflow=moneyflow,
        northbound=northbound, institution=institution,
    )


def _load_or_compute_registry(
    factor_whitelist: list | None,
    use_factor_cache: bool,
    skip_factor_build: bool,
    rebuild_factor_cache: bool,
    hold_period: int,
    rebalance_freq: str | None,
    registry_kwargs: dict,
) -> dict:
    freq = rebalance_freq or REBALANCE_FREQ
    cache_path = factor_cache_path(
        hold_period, freq, factor_whitelist, BACKTEST_START, BACKTEST_END,
    )

    if use_factor_cache and not rebuild_factor_cache and cache_exists(cache_path):
        return load_factor_panel(cache_path)

    if skip_factor_build:
        raise FileNotFoundError(
            f"因子缓存不存在且指定了 --skip-factor-build: {cache_path}"
        )

    factor_names = factor_whitelist if factor_whitelist else None
    registry = get_factor_registry(**registry_kwargs, factor_names=factor_names)

    if factor_whitelist:
        before = len(registry)
        regime_keys = {k for k in registry if k.startswith("市场") or k.startswith("HMM_")}
        barra_keys = {k for k in registry if k.startswith("Barra_")}
        registry = {
            k: v for k, v in registry.items()
            if k in factor_whitelist or k in regime_keys or k in barra_keys
        }
        logger.info(
            f"因子白名单过滤: {before} → {len(registry)} 个因子"
            f"（含 {len(regime_keys)} 个市场状态特征）"
        )
    else:
        logger.info(f"因子库就绪: {len(registry)} 个因子（全量）")

    if use_factor_cache:
        save_factor_panel(
            cache_path, registry,
            hold_period=hold_period,
            rebalance_freq=freq,
            factor_whitelist=factor_whitelist,
            start=BACKTEST_START,
            end=BACKTEST_END,
        )
    return registry


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
    factor_whitelist: list = None,
    rebalance_freq: str = None,
    use_factor_cache: bool = True,
    skip_factor_build: bool = False,
    rebuild_factor_cache: bool = False,
    feature_neutralize: bool = False,
    barra_factors: dict[str, pd.DataFrame] | None = None,
):
    """
    构建 MLDataset，供 IndustryWalkForwardTrainer 等复用。

    forward_return 定义：
        有 open_（次日开盘价）时 → close[t+N] / open[t+1] - 1
            即信号日收盘后，次日开盘买入，持有 N 日到收盘卖出的真实收益。
            这与实盘执行完全一致（次日开盘手动买入）。
        无 open_ 时 → close[t+N] / close[t] - 1（退化为收收收益，含隔夜跳空）

    feature_neutralize : bool
        若为 True 且 ``barra_factors`` 与 ``industry_map`` 同时提供，则在因子
        registry 构建完成后、``build_ml_dataset`` 之前，对每个因子面板按
        ``rebalance_dates`` 做截面 OLS 残差化（控制变量 = Barra 9 风格 + 行业
        哑变量），与 IC 筛选阶段 ``research/ic/barra.py`` 用同一套口径。
        市场/HMM/Barra 自身特征不中性化（它们是有意保留的系统状态信号）。
    """
    rk = _registry_kwargs(
        prices, financial, prices_raw, volume, amount,
        open_, high, low, clean_ret, masks,
        market_prices, industry_map, margin, moneyflow,
        northbound, institution,
    )
    registry = _load_or_compute_registry(
        factor_whitelist, use_factor_cache, skip_factor_build,
        rebuild_factor_cache, hold_period, rebalance_freq, rk,
    )
    forward_return = _compute_forward_return(prices, hold_period, open_)

    if feature_neutralize and barra_factors is not None and industry_map is not None:
        from models.wf.labels import residualize_panel
        from utils.rebalance_dates import get_rebalance_dates

        # industry_map 可能是 DataFrame（多列），残差化需要 Series
        ind_map_series = (
            industry_map["sw_l2"]
            if isinstance(industry_map, pd.DataFrame) and "sw_l2" in industry_map.columns
            else industry_map
        )
        freq = rebalance_freq or REBALANCE_FREQ
        rebalance_dates = get_rebalance_dates(
            pd.DatetimeIndex(forward_return.index), freq,
        )

        # 不中性化：市场状态 / HMM regime / Barra 自身特征（系统状态信号）
        def _excluded(name: str) -> bool:
            return (
                name.startswith("市场")
                or name.startswith("HMM_")
                or name.startswith("Barra_")
            )

        excluded = 0
        new_registry: dict = {}
        for name, panel in registry.items():
            if _excluded(name):
                new_registry[name] = panel
                excluded += 1
                continue
            new_registry[name] = residualize_panel(
                panel, barra_factors, ind_map_series, rebalance_dates,
            )
        registry = new_registry
        logger.info(
            f"feature_neutralize: {len(registry) - excluded} 个因子已 Barra+行业残差化"
            f"（保留 {excluded} 个市场/HMM/Barra 特征不中性化）"
        )

    ds_kwargs = {}
    if rebalance_freq is not None:
        ds_kwargs["rebalance_freq"] = rebalance_freq
    return build_ml_dataset(registry, forward_return, **ds_kwargs)


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
    factor_whitelist: list = None,
    train_windows: list = None,
    train_window_units: str = "months",
    artifact_dir=None,
    rebalance_freq: str = None,
    use_factor_cache: bool = True,
    skip_factor_build: bool = False,
    rebuild_factor_cache: bool = False,
    trainer_engine: str = "v2",
    wf_selection: str = "ic_weighted",
    label_mode: str = "cs_zscore",
    ensemble_method: str = "zscore",
    save_models: bool = False,
    objective: str = "regression",
    tag: str | None = None,
    barra_factors: dict[str, pd.DataFrame] | None = None,
    device: str = "cpu",
    tb_prices: pd.DataFrame | None = None,
    tb_open: pd.DataFrame | None = None,
    triple_barrier_params: dict | None = None,
    regime_states: pd.Series | None = None,
    feature_neutralize: bool = False,
) -> tuple[pd.DataFrame, object]:
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

    dataset = build_factor_dataset(
        prices, financial,
        prices_raw=prices_raw, volume=volume, amount=amount,
        open_=open_, high=high, low=low,
        clean_ret=clean_ret, masks=masks,
        market_prices=market_prices, industry_map=industry_map,
        margin=margin, moneyflow=moneyflow,
        northbound=northbound, institution=institution,
        hold_period=hold_period,
        factor_whitelist=factor_whitelist,
        rebalance_freq=rebalance_freq,
        use_factor_cache=use_factor_cache,
        skip_factor_build=skip_factor_build,
        rebuild_factor_cache=rebuild_factor_cache,
        feature_neutralize=feature_neutralize,
        barra_factors=barra_factors,
    )
    logger.info(f"ML 策略使用 {len(dataset.feature_names)} 个因子，模型={model_types}")

    trainer_kwargs = dict(
        model_types=model_types,
        rebalance_freq=rebalance_freq or REBALANCE_FREQ,
        train_window_units=train_window_units,
    )
    if train_windows is not None:
        trainer_kwargs["train_windows"] = train_windows
    if artifact_dir is not None:
        trainer_kwargs["artifact_dir"] = artifact_dir

    # Regime-conditional：regime_states 提供时切换到 RegimeConditionalTrainer
    # （子类，与 WalkForwardTrainer 输出格式完全一致；regime_states=None 退化为父类）
    if regime_states is not None:
        logger.info("regime_states 已提供 → 使用 RegimeConditionalTrainer")
        trainer = RegimeConditionalTrainer(
            hold_period=hold_period,
            wf_selection=wf_selection,
            label_mode=label_mode,
            ensemble_method=ensemble_method,
            save_models=save_models,
            objective=objective,
            tag=tag,
            barra_factors=barra_factors,
            industry_map=industry_map,
            device=device,
            prices=tb_prices,
            open_prices=tb_open,
            triple_barrier_params=triple_barrier_params,
            regime_states=regime_states,
            **trainer_kwargs,
        )
    else:
        trainer = WalkForwardTrainer(
            hold_period=hold_period,
            wf_selection=wf_selection,
            label_mode=label_mode,
            ensemble_method=ensemble_method,
            save_models=save_models,
            objective=objective,
            tag=tag,
            barra_factors=barra_factors,
            industry_map=industry_map,
            device=device,
            prices=tb_prices,
            open_prices=tb_open,
            triple_barrier_params=triple_barrier_params,
            **trainer_kwargs,
        )
    score_df = trainer.fit_predict(dataset)

    if show_report:
        from models.analyzer import MLAnalyzer
        analyzer = MLAnalyzer(trainer)
        analyzer.full_report(prices)

    return score_df, trainer
