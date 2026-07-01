"""
run.py  —  一键运行完整流程

PowerShell 重定向日志（避免 ANSI 乱码与 GBK 中文乱码）:
    chcp 65001
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    $OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    $env:LOGURU_COLORIZE='0'
    $env:PYTHONIOENCODING='utf-8'
    $env:PYTHONUTF8='1'
    $env:PYTHONUNBUFFERED='1'
    python -u run.py ... 2>&1 | Tee-Object -FilePath logs/run_xxx.log -Encoding utf8

run.py 启动时会自动 bootstrap UTF-8（config/encoding_bootstrap.py）。

用法:
    python run.py --mode linear                      # 线性加权基准，月频
    python run.py --mode ensemble                    # ML ensemble，月频（推荐）
    python run.py --mode ensemble --horizon 5        # 周频（5日持仓）
    python run.py --mode ensemble --horizon 3        # 超短（3日，研究用）
    python run.py --mode lgbm --horizon 10           # 单模型，10日持仓
    python run.py --skip-download                    # 跳过数据下载
    python run.py --sample 50                        # 调试模式
    python run.py --mode ensemble --report           # 训练后展示分析报告

horizon 说明：
    3  — 超短线，研究连板后动量延续性用，不建议实盘
    5  — 周频，T+1影响小，可实盘
    10 — 双周频
    20 — 月频（默认），最稳定
    60 — 季频

注意：A股T+1制度，horizon<3的结果不具备实盘参考价值。
"""
import argparse
import os
import sys
from pathlib import Path

from config.encoding_bootstrap import bootstrap_stdio_utf8, configure_loguru

bootstrap_stdio_utf8()

from loguru import logger

configure_loguru()

import pandas as pd

from config.settings import (
    RAW_DIR,
    BACKTEST_START, BACKTEST_END,
    FACTOR_WEIGHTS,
)

PRICES_PATH     = RAW_DIR / "prices_hfq.parquet"
PRICES_RAW_PATH = RAW_DIR / "prices_raw.parquet"
FIN_PATH        = RAW_DIR / "financial_indicators.parquet"

ML_MODES    = {"lgbm", "xgb", "cat", "ridge", "ensemble", "rf", "mlp"}
ALL_MODES   = {"linear", "industry", "dynamic"} | ML_MODES
VALID_MODELS = {"lgbm", "xgb", "cat", "ridge", "rf", "mlp"}



def _resolve_dynamic_lookback(dynamic_lookback: int | None, rebalance_freq: str) -> int:
    from models.trainer import months_to_rebalance_periods
    if dynamic_lookback is not None:
        return dynamic_lookback
    return months_to_rebalance_periods(6, rebalance_freq)

from models.trainer import MODEL_TYPES  # 分行业模式默认模型列表

INDEX_MAP = {"沪深300": "000300", "创业板指": "399006"}


def _load_indices() -> dict:
    """加载沪深300和创业板指收盘价，优先读本地缓存，否则从AKShare拉取。"""
    import akshare as ak
    result = {}
    for name, code in INDEX_MAP.items():
        cache = RAW_DIR / f"index_{code}.parquet"
        if cache.exists():
            s = pd.read_parquet(cache).squeeze()
        else:
            try:
                df = ak.index_zh_a_hist(
                    symbol=code, period="daily",
                    start_date=BACKTEST_START.replace("-", ""),
                    end_date=BACKTEST_END.replace("-", ""),
                )
                df["日期"] = pd.to_datetime(df["日期"])
                s = df.set_index("日期")["收盘"].rename(name)
                s.to_frame().to_parquet(cache)
                logger.info(f"指数 {name}({code}) 下载完成，shape={s.shape}")
            except Exception as e:
                logger.warning(f"指数 {name} 下载失败: {e}，跳过")
                continue
        result[name] = s
    return result


def _load_data(skip_download, sample):
    if not skip_download:
        logger.info("Step 1: 检查并更新数据")
        from data.download import main as download_main
        download_main(BACKTEST_START, BACKTEST_END, sample=sample)
    else:
        logger.info("Step 1: 跳过下载（--skip-download）")
        if not PRICES_PATH.exists():
            raise FileNotFoundError(f"必须文件不存在: {PRICES_PATH}")
        for p in [PRICES_RAW_PATH, FIN_PATH]:
            if not p.exists():
                logger.warning(f"可选文件不存在，对应因子将跳过: {p.name}")

    logger.info("Step 2: 加载并清洗数据")
    from data.clean import clean_prices, clean_financial, clean_ohlcv

    prices     = clean_prices(pd.read_parquet(PRICES_PATH), label="prices_hfq")
    prices_raw = (clean_prices(pd.read_parquet(PRICES_RAW_PATH), label="prices_raw")
                  if PRICES_RAW_PATH.exists() else None)
    financial  = (clean_financial(pd.read_parquet(FIN_PATH))
                  if FIN_PATH.exists() else None)

    def _load_opt(fname):
        p = RAW_DIR / fname
        if p.exists():
            logger.debug(f"加载 {fname}")
            return pd.read_parquet(p)
        return None

    volume      = _load_opt("volume.parquet")
    amount      = _load_opt("amount.parquet")
    open_       = _load_opt("open_hfq.parquet")
    high        = _load_opt("high_hfq.parquet")
    low         = _load_opt("low_hfq.parquet")
    margin      = _load_opt("margin_balance.parquet")
    moneyflow   = _load_opt("moneyflow_large.parquet")
    northbound  = _load_opt("northbound_holding.parquet")
    institution = _load_opt("institution_holding.parquet")
    market_prices = _load_opt("csi_all.parquet")
    if market_prices is None:
        market_prices = _load_opt("csi300.parquet")
    if market_prices is None:
        market_prices = _load_opt("index_000300.parquet")
    industry_map  = _load_opt("industry_map.parquet")

    # 涨跌停清洗：生成 clean_ret（屏蔽涨跌停日的日收益率）和 masks
    # 所有量价因子必须使用 clean_ret 而非 prices.pct_change()
    logger.info("Step 2b: 涨跌停清洗（生成 clean_ret）")
    clean_ret, masks = clean_ohlcv(prices, open_, high, low)

    return (prices, prices_raw, financial, volume, amount,
            open_, high, low, clean_ret, masks,
            market_prices, industry_map,
            margin, moneyflow, northbound, institution)


def _horizon_to_rebalance_freq(horizon: int) -> str:
    """根据持仓期推断调仓频率"""
    if horizon <= 3:
        return "3D"
    elif horizon <= 7:
        return "W-FRI"   # 每周五
    elif horizon <= 15:
        return "2W-FRI"  # 每两周
    else:
        return "ME"       # 月末


def _load_factor_config(config_path: str, horizon: int) -> list | None:
    """
    从 YAML 或 JSON 读取指定 horizon 的因子白名单。
    YAML 结构：{ h5: {factors: [...]}, h20: {factors: [...]} }
    JSON 结构：{ horizon: 5, factors: [...] }
    返回 None 表示不做过滤（使用全部因子）。
    """
    import json
    p = Path(config_path)
    if not p.exists():
        logger.warning(f"因子配置文件不存在: {config_path}，使用全部因子")
        return None
    if p.suffix in (".yaml", ".yml"):
        import yaml
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
        key = f"h{horizon}"
        if key not in cfg:
            logger.warning(f"YAML 中无 {key} 配置，使用全部因子")
            return None
        return cfg[key].get("factors")
    elif p.suffix == ".json":
        cfg = json.loads(p.read_text(encoding="utf-8"))
        return cfg.get("factors")
    logger.warning(f"不支持的配置格式: {p.suffix}，使用全部因子")
    return None


def _resolve_output_dir(output_dir: str | Path | None, tag: str) -> Path:
    """实验产物目录，默认 results/<tag>/。"""
    out = Path(output_dir) if output_dir else Path("results") / tag
    out.mkdir(parents=True, exist_ok=True)
    return out


def main(mode="linear", skip_download=False, sample=0,
         show_report=False, horizon=20,
         factor_config: str = None, show_holdings: bool = False,
         train_windows: list = None, train_window_units: str = "months",
         models: list = None,
         blend_dynamic: bool = False, output_dir: str = None,
         backtest_freq: str = None,
         use_factor_cache: bool = True, skip_factor_build: bool = False,
         rebuild_factor_cache: bool = False,
         dynamic_lookback: int = None,
         backtest_engine: str = "v2",
         trainer_engine: str = "v2",
         wf_selection: str = "ic_weighted",
         label_mode: str = "cs_zscore",
         ensemble_method: str = "zscore",
       save_models: bool = False,
       objective: str = "regression",
       device: str = "cpu",
       tune: bool = False,
      tune_trials: int = 15,
      triple_barrier_params: dict | None = None,
      multi_horizon: list | None = None,
      mh_weights: str = "equal",
      regime_conditional: bool = False,
      turnover_limit: float = 1.0,
      rank_change_threshold: float = 0.0,
      bid_ask_spread_bps: float | None = None,
      feature_neutralize: bool = False):

    if horizon < 3:
        logger.warning(
            f"horizon={horizon} 在A股T+1制度下不具备实盘参考价值，"
            "结果仅供研究"
        )

    rebalance_freq = _horizon_to_rebalance_freq(horizon)
    bt_freq = backtest_freq or rebalance_freq
    win_tag = ("_w" + "-".join(str(w) for w in train_windows)) if train_windows else ""
    units_tag = "_p" if train_window_units == "periods" else ""
    bt_tag = f"_bt{bt_freq.replace('-', '')}" if backtest_freq else ""
    mdl_tag = ("_m" + "-".join(sorted(models))) if models else ""
    dyn_tag = "_blend" if blend_dynamic else ""
    dyn_lb_tag = f"_lb{dynamic_lookback}" if dynamic_lookback is not None else ""
    mh_tag = ("_mh" + "-".join(str(h) for h in multi_horizon)) if multi_horizon else ""
    mh_w_tag = f"_{mh_weights}" if multi_horizon else ""
    reg_tag = "_regime" if regime_conditional else ""
    to_tag = (f"_to{turnover_limit}" if turnover_limit < 1.0 else "") + (
        f"_rc{rank_change_threshold}" if rank_change_threshold > 0.0 else ""
    )
    tag = (
        f"{mode}_h{horizon}{win_tag}{units_tag}{bt_tag}{mdl_tag}{dyn_tag}{dyn_lb_tag}"
        f"{mh_tag}{mh_w_tag}{reg_tag}{to_tag}"
    )
    out_dir = _resolve_output_dir(output_dir, tag)
    logger.info(
        f"模式={mode}, 持仓期={horizon}日, ML调仓={rebalance_freq}, "
        f"回测调仓={bt_freq}, 训练窗单位={train_window_units}, 输出={out_dir}"
    )
    if multi_horizon:
        logger.info(
            f"Multi-Horizon: horizons={multi_horizon}, weights={mh_weights}"
        )
    if regime_conditional:
        logger.info("Regime-Conditional: 启用按 regime 过滤训练样本")
    if turnover_limit < 1.0 or rank_change_threshold > 0.0:
        logger.info(
            f"Turnover 控制: turnover_limit={turnover_limit}, "
            f"rank_change_threshold={rank_change_threshold}"
        )

    (prices, prices_raw, financial, volume, amount,
     open_, high, low, clean_ret, masks,
     market_prices, industry_map,
     margin, moneyflow, northbound, institution) = _load_data(skip_download, sample)

    # 因子白名单（从 YAML/JSON 读取，None=使用全部因子）
    factor_whitelist = _load_factor_config(factor_config, horizon) if factor_config else None
    if factor_whitelist:
        logger.info(f"因子白名单: {len(factor_whitelist)} 个因子 → {factor_whitelist}")

    # 共享的额外数据关键字参数（传给 get_factor_registry）
    extra_kwargs = dict(
        prices_raw=prices_raw, volume=volume, amount=amount,
        open_=open_, high=high, low=low,
        clean_ret=clean_ret, masks=masks,
        market_prices=market_prices, industry_map=industry_map,
        margin=margin, moneyflow=moneyflow,
        northbound=northbound, institution=institution,
    )
    cache_kwargs = dict(
        use_factor_cache=use_factor_cache,
        skip_factor_build=skip_factor_build,
        rebuild_factor_cache=rebuild_factor_cache,
    )

    # ── Optuna 超参搜索（离线步骤：搜索完存 JSON 后退出，不继续训练）─────
    if tune:
        from models.wf.tuning import tune_all_models, save_tuned_params
        from strategies.ml import build_factor_dataset
        from models.trainer import resolve_train_windows
        logger.info("Step 3 (tune): Optuna 超参搜索（LGBM/XGB/RF）")
        tune_models = models or ["lgbm", "xgb", "rf"]
        tune_windows = train_windows or [6, 12]
        val_window_months = 6
        win_periods, val_periods = resolve_train_windows(
            tune_windows, val_window_months, rebalance_freq, units="months")
        ds = build_factor_dataset(
            prices, financial, hold_period=horizon,
            factor_whitelist=factor_whitelist,
            rebalance_freq=rebalance_freq,
            **extra_kwargs, **cache_kwargs,
        )
        best = tune_all_models(
            tune_models, ds,
            train_windows=win_periods, val_window=val_periods,
            hold_period=horizon, n_trials=tune_trials,
            label_mode=label_mode,
        )
        out = save_tuned_params(best)
        logger.info(f"超参搜索完成，结果存 {out}")
        return

    # ── Regime-conditional：加载市场状态序列（任务2）─────────────────────────
    regime_states_arg = None
    if regime_conditional:
        from strategies.market_state import get_market_state
        logger.info("加载市场状态序列（regime_conditional=True）...")
        regime_states_arg = get_market_state(method="ma")
        logger.info(
            f"市场状态就绪: {len(regime_states_arg)} 天, "
            f"distribution={regime_states_arg.value_counts().to_dict()}"
        )

    # ── Multi-Horizon 集成（任务1）：优先于单 horizon 路由 ────────────────────
    if multi_horizon:
        logger.info(
            f"Step 3: Multi-Horizon 集成（horizons={multi_horizon}, "
            f"weights={mh_weights}, regime={regime_conditional}）"
        )
        from strategies.multi_horizon import run_multi_horizon

        # model_types 解析（与单 horizon ML 模式一致）
        if models:
            invalid = set(models) - VALID_MODELS
            if invalid:
                raise ValueError(
                    f"--models 包含无效模型: {invalid}，可选: {VALID_MODELS}"
                )
            mh_model_types = models
        else:
            mh_model_types = ["lgbm", "xgb"]

        # Barra 残差化标签（按需计算）
        mh_barra = None
        if label_mode == "barra_residual":
            from factors.barra_risk import get_barra_factors
            mh_barra = get_barra_factors(
                prices=prices, financial=financial,
                market_prices=market_prices, volume=volume,
                clean_ret=clean_ret,
            )

        # industry_map → Series
        mh_ind_map = None
        if industry_map is not None:
            mh_ind_map = (
                industry_map["sw_l2"]
                if isinstance(industry_map, pd.DataFrame)
                and "sw_l2" in industry_map.columns
                else industry_map
            )

        factor_scores, mh_info = run_multi_horizon(
            horizons=list(multi_horizon),
            weights=mh_weights,
            prices=prices, financial=financial,
            extra_kwargs=extra_kwargs, cache_kwargs=cache_kwargs,
            model_types=mh_model_types,
            factor_whitelist=factor_whitelist,
            train_windows=train_windows,
            train_window_units=train_window_units,
            artifact_dir=out_dir,
            wf_selection=wf_selection,
            label_mode=label_mode,
            ensemble_method=ensemble_method,
            save_models=save_models,
            objective=objective,
            tag=tag,
            device=device,
            barra_factors=mh_barra,
            industry_map=mh_ind_map,
            regime_states=regime_states_arg,
        )
        # 保存集成权重 / IC 摘要
        import json as _json
        mh_summary = {
            "tag": tag,
            "horizons": list(multi_horizon),
            "weights": {str(h): mh_info["weights"][h] for h in multi_horizon},
            "ic_per_h": {str(h): mh_info["ic_per_h"][h] for h in multi_horizon},
            "regime_conditional": regime_conditional,
        }
        with open(out_dir / f"multi_horizon_summary_{tag}.json", "w", encoding="utf-8") as f:
            _json.dump(mh_summary, f, ensure_ascii=False, indent=2)

    # ── 生成因子得分 ──────────────────────────────────────────────────────────
    elif mode == "linear":
        logger.info("Step 3: 线性加权策略（基准）")
        from strategies.linear import run as linear_run
        factor_scores = linear_run(prices, financial, FACTOR_WEIGHTS, **extra_kwargs)

    elif mode == "industry":
        if models:
            invalid = set(models) - VALID_MODELS
            if invalid:
                raise ValueError(f"--models 包含无效模型: {invalid}，可选: {VALID_MODELS}")
            model_types = models
        else:
            model_types = list(MODEL_TYPES)
        logger.info(
            f"Step 3: 分行业ML策略（申万二级，模型={model_types}，horizon={horizon}日）"
        )
        from strategies.ml import build_factor_dataset
        from models.industry_trainer import IndustryWalkForwardTrainer
        dataset = build_factor_dataset(
            prices, financial, hold_period=horizon,
            factor_whitelist=factor_whitelist,
            rebalance_freq=rebalance_freq,
            **extra_kwargs, **cache_kwargs,
        )
        trainer = IndustryWalkForwardTrainer(
            model_types    = model_types,
            train_windows  = train_windows,
            rebalance_freq = rebalance_freq,
            hold_period    = horizon,
            label_mode     = label_mode,
            wf_selection   = wf_selection,
            ensemble_method= ensemble_method,
        )
        factor_scores = trainer.fit_predict(dataset, industry_map=industry_map)
        # 保存 IC 指标（与其他模式一致）
        trainer.save_metrics(tag, output_dir=out_dir)
        if trainer.ic_series is not None:
            trainer.ic_series.to_csv(out_dir / f"ic_series_{tag}.csv", header=True)

    elif mode in ML_MODES:
        if models:
            invalid = set(models) - VALID_MODELS
            if invalid:
                raise ValueError(f"--models 包含无效模型: {invalid}，可选: {VALID_MODELS}")
            model_types = models
        else:
            model_types = list(ML_MODES - {"ensemble"}) if mode == "ensemble" else [mode]
        logger.info(
            f"Step 3: ML 策略（{mode}，模型={model_types}，horizon={horizon}日，"
            f"trainer={trainer_engine}）"
        )
        from strategies.ml import run as ml_run

        # Barra 残差化标签：按需计算 Barra 风格因子并透传给 trainer
        # feature_neutralize 也需要 Barra 因子（对特征做残差化），二者复用同一份
        # industry_map 可能是 DataFrame（多列），先转成 Series 以便透传给 Barra 行业中性化
        ind_map_arg = None
        if industry_map is not None:
            ind_map_arg = (
                industry_map["sw_l2"] if isinstance(industry_map, pd.DataFrame)
                and "sw_l2" in industry_map.columns else industry_map
            )

        barra_factors_arg = None
        if label_mode == "barra_residual" or feature_neutralize:
            from factors.barra_risk import get_barra_factors
            if label_mode == "barra_residual":
                logger.info("label_mode=barra_residual: 计算 Barra 风格因子用于标签残差化...")
            if feature_neutralize:
                logger.info("feature_neutralize=True: 计算 Barra 风格因子用于特征残差化...")
            barra_factors_arg = get_barra_factors(
                prices=prices,
                financial=financial,
                market_prices=market_prices,
                volume=volume,
                clean_ret=clean_ret,
                industry_map=ind_map_arg,
            )
            logger.info(f"Barra 因子就绪: {len(barra_factors_arg)} 个")

        # triple_barrier 需要把日频 prices/open_ 透传给 trainer 用于预计算标签面板
        tb_prices_arg = prices if label_mode == "triple_barrier" else None
        tb_open_arg = open_ if label_mode == "triple_barrier" else None

        # extra_kwargs 含 industry_map(DataFrame)，此处显式传 ind_map_arg(Series)，弹出避免 kwarg 冲突
        ml_extra_kwargs = {k: v for k, v in extra_kwargs.items() if k != "industry_map"}
        factor_scores, trainer = ml_run(
            prices, financial,
            model_types=model_types,
            hold_period=horizon,
            show_report=show_report,
            factor_whitelist=factor_whitelist,
            train_windows=train_windows,
            train_window_units=train_window_units,
            artifact_dir=out_dir,
            rebalance_freq=rebalance_freq,
            trainer_engine=trainer_engine,
            wf_selection=wf_selection,
            label_mode=label_mode,
            ensemble_method=ensemble_method,
            save_models=save_models,
            objective=objective,
            tag=tag,
            barra_factors=barra_factors_arg,
            industry_map=ind_map_arg,
            device=device,
            tb_prices=tb_prices_arg,
            tb_open=tb_open_arg,
            triple_barrier_params=triple_barrier_params,
            regime_states=regime_states_arg,
            feature_neutralize=feature_neutralize,
            **ml_extra_kwargs, **cache_kwargs,
        )
        if hasattr(trainer, "save_metrics"):
            trainer.save_metrics(tag, output_dir=out_dir)
        elif hasattr(trainer, "ic_series") and trainer.ic_series is not None:
            ic = trainer.ic_series.dropna()
            import json
            ic_metrics = {
                "tag": tag, "IC均值": round(ic.mean(), 4),
                "IC标准差": round(ic.std(), 4),
                "ICIR": round(ic.mean() / ic.std(), 4),
                "IC>0胜率": round((ic > 0).mean(), 4),
                "预测期数": len(ic),
            }
            with open(out_dir / f"model_metrics_{tag}.json", "w", encoding="utf-8") as f:
                json.dump(ic_metrics, f, ensure_ascii=False, indent=2)
        if hasattr(trainer, "ic_series") and trainer.ic_series is not None:
            trainer.ic_series.to_csv(out_dir / f"ic_series_{tag}.csv", header=True)

        # ── Dynamic 混合（rank-average ML + Dynamic）────────────────────────
        if blend_dynamic:
            logger.info("blend_dynamic: 追加 DynamicFactorTrainer，与 ML 得分 rank-average")
            from strategies.ml import build_factor_dataset
            from models.dynamic_trainer import DynamicFactorTrainer
            import numpy as np
            dyn_dataset = build_factor_dataset(
                prices, financial, hold_period=horizon,
                factor_whitelist=factor_whitelist,
                rebalance_freq=rebalance_freq,
                **extra_kwargs, **cache_kwargs,
            )
            dyn_lb = _resolve_dynamic_lookback(dynamic_lookback, rebalance_freq)
            dyn_trainer = DynamicFactorTrainer(
                lookback=dyn_lb, min_lookback=max(3, dyn_lb // 2), method="icir",
            )
            dyn_scores = dyn_trainer.fit_predict(dyn_dataset)

            # 对齐日期，各自转百分位 rank，再等权平均
            common_dates = factor_scores.index.intersection(dyn_scores.index)
            ml_rank  = factor_scores.loc[common_dates].rank(axis=1, pct=True)
            dyn_rank = dyn_scores.loc[common_dates].rank(axis=1, pct=True)
            factor_scores = (ml_rank.add(dyn_rank, fill_value=0) / 2)
            logger.info(f"blend_dynamic 完成: {len(common_dates)} 个预测日")

    elif mode == "dynamic":
        dyn_lb = _resolve_dynamic_lookback(dynamic_lookback, rebalance_freq)
        logger.info(
            f"Step 3: 因子动态加权策略（ICIR 权重，lookback={dyn_lb}期，horizon={horizon}日）"
        )
        from strategies.ml import build_factor_dataset
        from models.dynamic_trainer import DynamicFactorTrainer
        from models.trainer import spearman_ic
        import json
        dataset = build_factor_dataset(
            prices, financial, hold_period=horizon,
            factor_whitelist=factor_whitelist,
            rebalance_freq=rebalance_freq,
            **extra_kwargs, **cache_kwargs,
        )
        trainer = DynamicFactorTrainer(
            lookback=dyn_lb, min_lookback=max(3, dyn_lb // 2), method="icir",
        )
        factor_scores = trainer.fit_predict(dataset)
        trainer.print_weight_evolution()
        ic_dict = {}
        for date in factor_scores.index:
            _, y = dataset.get_cross_section(date)
            if y is None:
                continue
            s = factor_scores.loc[date].dropna()
            common = s.index.intersection(y.dropna().index)
            if len(common) < 20:
                continue
            ic_dict[date] = spearman_ic(s.loc[common].values, y.loc[common].values)
        if ic_dict:
            ic = pd.Series(ic_dict).dropna()
            ic_metrics = {
                "tag": tag, "IC均值": round(ic.mean(), 4),
                "IC标准差": round(ic.std(), 4),
                "ICIR": round(ic.mean() / ic.std(), 4),
                "IC>0胜率": round((ic > 0).mean(), 4),
                "预测期数": len(ic),
                "dynamic_lookback": dyn_lb,
            }
            with open(out_dir / f"model_metrics_{tag}.json", "w", encoding="utf-8") as f:
                json.dump(ic_metrics, f, ensure_ascii=False, indent=2)
            ic.to_csv(out_dir / f"ic_series_{tag}.csv", header=True)

    else:
        raise ValueError(f"未知 mode: {mode}，可选: {ALL_MODES}")

    factor_scores.to_parquet(out_dir / f"factor_scores_{tag}.parquet")

    # ── 回测（Q1-Q5 分组，模块化 quantile 引擎）────────────────────────────────
    logger.info("Step 4: 回测（quantile 模块化引擎）")

    from backtest.quantile import run_quantile_backtest
    from backtest.report import (
        plot_quantile_result, print_quantile_summary,
        print_holdings, export_holdings, export_turnover_detail,
    )
    from backtest.execution import (
        BacktestConfig,
        build_st_schedule,
        build_delist_dates_from_stock_list,
    )
    indices = _load_indices()
    # bid_ask_spread_bps：None=读 settings 默认，CLI 显式覆盖
    bt_kwargs = dict(
        turnover_limit=turnover_limit,
        rank_change_threshold=rank_change_threshold,
    )
    if bid_ask_spread_bps is not None:
        bt_kwargs["bid_ask_spread_bps"] = bid_ask_spread_bps
        logger.info(f"bid-ask spread override: {bid_ask_spread_bps} bp")
    bt_config = BacktestConfig(**bt_kwargs)

    # M4 修复：从 stock_list.parquet 构建时间序列 ST 状态 + 退市日期字典
    stock_names_ser: pd.Series | None = None
    st_schedule: pd.DataFrame | None = None
    delist_dates: dict[str, pd.Timestamp] | None = None
    try:
        from config.settings import UNIVERSE_DIR
        sl_path = UNIVERSE_DIR / "stock_list.parquet"
        if sl_path.exists():
            sl_df = pd.read_parquet(sl_path)
            if "code" in sl_df.columns and "name" in sl_df.columns:
                stock_names_ser = sl_df.set_index("code")["name"]
                stock_names_ser.index = stock_names_ser.index.astype(str).str.zfill(6)
            delist_dates = build_delist_dates_from_stock_list(sl_df) or None
            is_st_ser = (
                sl_df.set_index("code")["is_st_current"]
                if "is_st_current" in sl_df.columns else None
            )
            if is_st_ser is not None:
                is_st_ser.index = is_st_ser.index.astype(str).str.zfill(6)
            st_schedule = build_st_schedule(
                stock_names_ser,
                prices.index,
                is_st_current=is_st_ser,
                delist_dates=delist_dates,
            )
            if st_schedule is not None:
                logger.info(
                    f"M4 ST 时间序列: {st_schedule.shape[1]} 只 ST 股 × "
                    f"{st_schedule.shape[0]} 交易日"
                )
            if delist_dates:
                logger.info(f"M4 退市日期字典: {len(delist_dates)} 只")
    except Exception as e:
        logger.warning(f"M4 ST/退市元数据加载失败（忽略，回退旧行为）: {e}")

    result = run_quantile_backtest(
        prices, factor_scores,
        n_quantiles=5,
        rebalance_freq=bt_freq,
        start=BACKTEST_START,
        end=BACKTEST_END,
        open_prices=open_,   # 次日开盘执行，一字涨停自动剔除
        masks=masks,
        indices=indices,
        config=bt_config,
        stock_names=stock_names_ser,
        volume=volume,
        st_schedule=st_schedule,
        delist_dates=delist_dates,
    )
    print_quantile_summary(result)
    plot_quantile_result(
        result,
        title=f"Q1-Q5 分组回测  |  mode={mode}  horizon={horizon}日",
        save_path=str(out_dir / f"backtest_{tag}.png"),
    )
    # 保存原始回测数据
    result.nav.to_csv(out_dir / f"backtest_{tag}_nav.csv", encoding="utf-8-sig")
    result.annual_returns.to_csv(out_dir / f"backtest_{tag}_annual.csv", encoding="utf-8-sig")
    result.long_short_nav.to_csv(out_dir / f"backtest_{tag}_longshort.csv", header=True)
    export_holdings(result, save_path=str(out_dir / f"holdings_top30_{tag}.csv"))
    export_turnover_detail(
        result, save_path=str(out_dir / f"turnover_detail_{tag}.csv"),
    )
    if show_holdings:
        print_holdings(result, last_n=12)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", default="linear",
        choices=sorted(ALL_MODES),
        help="策略模式: linear | ridge | lgbm | xgb | cat | ensemble",
    )
    parser.add_argument(
        "--horizon", type=int, default=20,
        help="持仓期（交易日）: 3=超短 5=周频 10=双周 20=月频(默认) 60=季频",
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument(
        "--report", action="store_true",
        help="ML 模式训练完后展示 IC / SHAP 分析报告",
    )
    parser.add_argument(
        "--factor-config", default=None,
        help="因子白名单配置文件路径（YAML 或 JSON，由 ic_analysis --save 生成）",
    )
    parser.add_argument(
        "--train-windows", default=None,
        help="训练窗口月数，逗号分隔，如 '6,12'（默认 6,12）",
    )
    parser.add_argument(
        "--train-window-units", default="months",
        choices=["months", "periods"],
        help="训练/验证窗口单位：months=日历月（默认）；periods=调仓期数（历史 bug 复现）",
    )
    parser.add_argument(
        "--backtest-freq", default=None,
        help="回测调仓频率覆盖（如 ME），ML 仍按 horizon 推断；用于复现训练/回测频率错位",
    )
    parser.add_argument(
        "--holdings", action="store_true",
        help="回测完打印并导出 Q5 每期持仓（保存为 holdings_Q5_<tag>.csv）",
    )
    parser.add_argument(
        "--models", default=None,
        help="ensemble 使用的模型子集，逗号分隔，如 'lgbm,xgb'（默认全部4个模型）",
    )
    parser.add_argument(
        "--blend-dynamic", action="store_true",
        help="将 DynamicFactorTrainer 得分与 ML 得分 rank-average 混合（3模型集成）",
    )
    parser.add_argument(
        "--dynamic-lookback", type=int, default=None,
        help="dynamic / blend-dynamic 的 ICIR 回看调仓期数（默认 6 月换算为 rebalance 期数）",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="实验输出目录（默认 results/<tag>/），同批次可共用如 results/v5_window612_h5/",
    )
    parser.add_argument(
        "--skip-factor-build", action="store_true",
        help="仅从磁盘缓存加载因子面板；缓存不存在则报错",
    )
    parser.add_argument(
        "--rebuild-factor-cache", action="store_true",
        help="忽略已有因子面板缓存，强制重新计算并覆盖",
    )
    parser.add_argument(
        "--no-factor-cache", action="store_true",
        help="禁用因子面板磁盘缓存（不读不写）",
    )
    parser.add_argument(
        "--backtest-engine", default="v2",
        help="[deprecated] 仅保留模块化 quantile 引擎；该参数已无效果，向后兼容保留",
    )
    parser.add_argument(
        "--trainer-engine", default="v2",
        help="[deprecated] 仅保留模块化 WF trainer；该参数已无效果，向后兼容保留",
    )
    parser.add_argument(
        "--wf-selection", default="ic_weighted",
        choices=["average", "best_window", "best_model", "ic_weighted"],
        help="v2: 多窗口/模型验证 IC 加权方式",
    )
    parser.add_argument(
        "--label-mode", default="cs_zscore",
        choices=["raw", "cs_rank", "cs_zscore", "triple_barrier", "barra_residual"],
        help="v2: 训练标签截面标准化；triple_barrier 走 AFML §3 路径依赖标签；"
             "barra_residual 需要 Barra 因子+行业映射",
    )
    parser.add_argument(
        "--ensemble-method", default="zscore", choices=["rank", "zscore"],
        help="v2: 多模型集成为 rank 平均或 z-score 加权平均",
    )
    parser.add_argument(
        "--tb-upper", type=float, default=2.0,
        help="triple_barrier: 上障碍 = +upper_mult * σ（默认 2.0）",
    )
    parser.add_argument(
        "--tb-lower", type=float, default=1.5,
        help="triple_barrier: 下障碍 = -lower_mult * σ（默认 1.5）",
    )
    parser.add_argument(
        "--tb-vol-window", type=int, default=20,
        help="triple_barrier: 波动率回看窗口（交易日，默认 20）",
    )
    parser.add_argument(
        "--tb-label-type", default="sign", choices=["sign", "return"],
        help="triple_barrier: 标签类型 sign(+1/-1/0) 或 return(触碰时实际收益)",
    )
    parser.add_argument(
        "--save-models", action="store_true",
        help="v2: 保存每折模型到 results/<tag>/models/",
    )
    parser.add_argument(
        "--objective", default="regression", choices=["regression", "rank"],
        help="训练目标：regression（MSE 回归，默认）或 rank（Learning-to-Rank，"
             "LGBMRanker/XGBRanker/CatBoostRanker；rank objective 自动配 cs_rank 标签，"
             "ridge/rf/mlp 不支持 rank 自动回退 regression）",
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "gpu"],
        help="LGBM/XGB 训练设备：cpu（默认）或 gpu（需 lightgbm[gpu] / xgboost≥2.0+cuda）",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="运行 Optuna 超参搜索（LGBM/XGB/RF/Cat），结果存 config/tuned_params.json 后退出",
    )
    parser.add_argument(
        "--tune-trials", type=int, default=15,
        help="Optuna 每模型搜索轮数（默认 15，仅 --tune 时生效）",
    )
    # ── 任务1：Multi-Horizon 集成 ─────────────────────────────────────────────
    parser.add_argument(
        "--multi-horizon", default=None,
        help=(
            "多期限集成：逗号分隔的 horizon 列表，如 '5,10,20'。"
            "指定后忽略 --horizon 单期限路径，对每个 horizon 各跑一次 WF，"
            "再 rank 加权平均得到 composite_score。"
        ),
    )
    parser.add_argument(
        "--mh-weights", default="equal",
        choices=["equal", "ic_weighted"],
        help="Multi-Horizon 集成权重方式：equal=等权（默认），ic_weighted=按各 horizon OOS IC 加权",
    )
    # ── 任务2：Regime-conditional 训练 ─────────────────────────────────────────
    parser.add_argument(
        "--regime-conditional", action="store_true",
        help=(
            "启用 RegimeConditionalTrainer：每个 pred_date 只用同 regime 历史样本训练，"
            "样本不足时 fallback 到全量。需要 market_state.parquet（无则自动下载沪深300）。"
        ),
    )
    # ── 任务3：Turnover 控制（解耦调仓/持仓）──────────────────────────────────
    parser.add_argument(
        "--turnover-limit", type=float, default=1.0,
        help=(
            "每期最大换手率（0-1），1.0=无限制（默认），0.3=最多换 30%% 仓位。"
            "约束 |sells|+|buys| ≤ 2 × turnover_limit × target_size。"
        ),
    )
    parser.add_argument(
        "--rank-change-threshold", type=float, default=0.0,
        help=(
            "排名变动阈值（0-1），0.0=不启用（默认），0.2=排名跌出 top 20%% 才换。"
            "上期持仓中 rank_pct ≥ (1-threshold) 的股票强制保留，避免无谓换仓。"
        ),
    )
    # ── 任务2：交易成本 bid-ask spread ──────────────────────────────────────────
    parser.add_argument(
        "--bid-ask-spread", type=float, default=None,
        help=(
            "bid-ask spread（单边 bp），覆盖 settings.BID_ASK_SPREAD_BPS（默认 10bp）。"
            "大盘股 ~2-5bp，小盘股 ~10-20bp；A股不能做空、用户资金量 ~200 万 market impact 可忽略。"
        ),
    )
    # ── P0-2: ML 特征中性化（与 IC 筛选口径一致）──────────────────────────────
    parser.add_argument(
        "--feature-neutralize", action="store_true",
        help=(
            "对 ML 特征做 Barra 9 风格 + 行业哑变量残差化，与 IC 筛选阶段"
            "（research/ic/barra.py）用同一套控制变量，消除 Size/Beta 系统性敞口。"
            "市场/HMM/Barra 自身特征不中性化。默认 False。"
        ),
    )
    args = parser.parse_args()
    train_windows = (
        [int(x) for x in args.train_windows.split(",")]
        if args.train_windows else None
    )
    models = (
        [m.strip() for m in args.models.split(",")]
        if args.models else None
    )
    multi_horizon_arg = (
        [int(x.strip()) for x in args.multi_horizon.split(",") if x.strip()]
        if args.multi_horizon else None
    )
    main(
        mode=args.mode,
        skip_download=args.skip_download,
        sample=args.sample,
        show_report=args.report,
        horizon=args.horizon,
        factor_config=args.factor_config,
        show_holdings=args.holdings,
        train_windows=train_windows,
        train_window_units=args.train_window_units,
        models=models,
        blend_dynamic=args.blend_dynamic,
        output_dir=args.output_dir,
        backtest_freq=args.backtest_freq,
        use_factor_cache=not args.no_factor_cache,
        skip_factor_build=args.skip_factor_build,
        rebuild_factor_cache=args.rebuild_factor_cache,
        dynamic_lookback=args.dynamic_lookback,
        backtest_engine=args.backtest_engine,
        trainer_engine=args.trainer_engine,
        wf_selection=args.wf_selection,
        label_mode=args.label_mode,
        ensemble_method=args.ensemble_method,
        save_models=args.save_models,
        objective=args.objective,
        device=args.device,
        tune=args.tune,
        tune_trials=args.tune_trials,
        triple_barrier_params={
            "vol_window": args.tb_vol_window,
            "upper_mult": args.tb_upper,
            "lower_mult": args.tb_lower,
            "label_type": args.tb_label_type,
        },
        multi_horizon=multi_horizon_arg,
        mh_weights=args.mh_weights,
        regime_conditional=args.regime_conditional,
        turnover_limit=args.turnover_limit,
        rank_change_threshold=args.rank_change_threshold,
        bid_ask_spread_bps=args.bid_ask_spread,
        feature_neutralize=args.feature_neutralize,
    )
