"""
run.py  —  一键运行完整流程

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
from pathlib import Path
from loguru import logger
import pandas as pd

from config.settings import (
    RAW_DIR, PROCESSED_DIR,
    BACKTEST_START, BACKTEST_END,
    FACTOR_WEIGHTS,
)

PRICES_PATH     = RAW_DIR / "prices_hfq.parquet"
PRICES_RAW_PATH = RAW_DIR / "prices_raw.parquet"
FIN_PATH        = RAW_DIR / "financial_indicators.parquet"

ML_MODES    = {"lgbm", "xgb", "cat", "ridge", "ensemble"}
ALL_MODES   = {"linear", "industry"} | ML_MODES


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
    market_prices = _load_opt("csi300.parquet")
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


def main(mode="linear", skip_download=False, sample=0,
         show_report=False, horizon=20, backtest="quantile"):

    if horizon < 3:
        logger.warning(
            f"horizon={horizon} 在A股T+1制度下不具备实盘参考价值，"
            "结果仅供研究"
        )

    rebalance_freq = _horizon_to_rebalance_freq(horizon)
    tag = f"{mode}_h{horizon}"  # 输出文件名带horizon，不同版本不互相覆盖
    logger.info(f"模式={mode}, 持仓期={horizon}日, 调仓频率={rebalance_freq}")

    (prices, prices_raw, financial, volume, amount,
     open_, high, low, clean_ret, masks,
     market_prices, industry_map,
     margin, moneyflow, northbound, institution) = _load_data(skip_download, sample)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 共享的额外数据关键字参数（传给 get_factor_registry）
    extra_kwargs = dict(
        prices_raw=prices_raw, volume=volume, amount=amount,
        open_=open_, high=high, low=low,
        clean_ret=clean_ret, masks=masks,
        market_prices=market_prices, industry_map=industry_map,
        margin=margin, moneyflow=moneyflow,
        northbound=northbound, institution=institution,
    )

    # ── 生成因子得分 ──────────────────────────────────────────────────────────
    if mode == "linear":
        logger.info("Step 3: 线性加权策略（基准）")
        from strategies.linear import run as linear_run
        factor_scores = linear_run(prices, financial, FACTOR_WEIGHTS, **extra_kwargs)

    elif mode == "industry":
        logger.info(f"Step 3: 分行业ML策略（申万二级，horizon={horizon}日）")
        from strategies.ml import build_factor_dataset
        from models.industry_trainer import IndustryWalkForwardTrainer
        dataset = build_factor_dataset(
            prices, financial, hold_period=horizon, **extra_kwargs,
        )
        trainer = IndustryWalkForwardTrainer()
        factor_scores = trainer.fit_predict(dataset)

    elif mode in ML_MODES:
        model_types = list(ML_MODES - {"ensemble"}) if mode == "ensemble" else [mode]
        logger.info(f"Step 3: ML 策略（{mode}，模型={model_types}，horizon={horizon}日）")
        from strategies.ml import run as ml_run
        factor_scores, trainer = ml_run(
            prices, financial,
            model_types=model_types,
            hold_period=horizon,
            show_report=show_report,
            **extra_kwargs,
        )

    else:
        raise ValueError(f"未知 mode: {mode}，可选: {ALL_MODES}")

    factor_scores.to_parquet(PROCESSED_DIR / f"factor_scores_{tag}.parquet")

    # ── 回测 ─────────────────────────────────────────────────────────────────
    logger.info(f"Step 4: 回测（{backtest}）")

    if backtest == "quantile":
        from backtest.quantile import (
            run_quantile_backtest, plot_quantile_result, print_quantile_summary,
        )
        result = run_quantile_backtest(
            prices, factor_scores,
            n_quantiles=5,
            rebalance_freq=rebalance_freq,
            start=BACKTEST_START,
            end=BACKTEST_END,
            open_prices=open_,   # 次日开盘执行，一字涨停自动剔除
            masks=masks,
        )
        print_quantile_summary(result)
        plot_quantile_result(
            result,
            title=f"Q1-Q5 分组回测  |  mode={mode}  horizon={horizon}日",
            save_path=f"backtest_{tag}.png",
        )

    elif backtest == "topN":
        from backtest.engine import run_backtest, plot_result
        result = run_backtest(prices, factor_scores, rebalance_freq=rebalance_freq)
        plot_result(result, save_path=f"backtest_{tag}.png")

    else:
        raise ValueError(f"未知回测模式: {backtest}，可选: quantile | topN")

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
        "--backtest", default="quantile",
        choices=["quantile", "topN"],
        help="回测模式: quantile=Q1-Q5分组(默认) | topN=top30等权",
    )
    args = parser.parse_args()
    main(
        mode=args.mode,
        skip_download=args.skip_download,
        sample=args.sample,
        show_report=args.report,
        horizon=args.horizon,
        backtest=args.backtest,
    )
