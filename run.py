"""
run.py  —  一键运行完整流程

用法:
    python run.py                          # 默认：线性策略
    python run.py --mode linear            # 线性加权（基准）
    python run.py --mode lgbm             # 单模型 LightGBM
    python run.py --mode xgb             # 单模型 XGBoost
    python run.py --mode cat             # 单模型 CatBoost
    python run.py --mode ridge            # 单模型 Ridge（线性 ML）
    python run.py --mode ensemble         # 全模型 Rank Averaging（推荐）
    python run.py --skip-download         # 跳过数据下载
    python run.py --sample 50             # 调试：只用50只股票
    python run.py --mode ensemble --report  # 训练后展示 IC / SHAP 分析图
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
ALL_MODES   = {"linear"} | ML_MODES


def _load_data(skip_download, sample):
    if not skip_download:
        logger.info("Step 1: 检查并更新数据")
        from data.download import main as download_main
        download_main(BACKTEST_START, BACKTEST_END, sample=sample)
    else:
        logger.info("Step 1: 跳过下载（--skip-download）")
        for p in [PRICES_PATH, PRICES_RAW_PATH, FIN_PATH]:
            if not p.exists():
                raise FileNotFoundError(f"数据文件不存在: {p}")

    logger.info("Step 2: 加载并清洗数据")
    from data.clean import clean_prices, clean_financial

    prices     = clean_prices(pd.read_parquet(PRICES_PATH),     label="prices_hfq")
    prices_raw = clean_prices(pd.read_parquet(PRICES_RAW_PATH), label="prices_raw")
    financial  = clean_financial(pd.read_parquet(FIN_PATH))

    # 可选：成交量/成交额（换手率/Amihud 因子需要）
    volume = amount = None
    if (RAW_DIR / "volume.parquet").exists():
        volume = pd.read_parquet(RAW_DIR / "volume.parquet")
    if (RAW_DIR / "amount.parquet").exists():
        amount = pd.read_parquet(RAW_DIR / "amount.parquet")

    return prices, prices_raw, financial, volume, amount


def main(mode="linear", skip_download=False, sample=0, show_report=False):

    prices, prices_raw, financial, volume, amount = _load_data(skip_download, sample)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── 生成因子得分 ──────────────────────────────────────────────────────────
    if mode == "linear":
        logger.info("Step 3: 线性加权策略（基准）")
        from strategies.linear import run as linear_run
        factor_scores = linear_run(
            prices, financial, FACTOR_WEIGHTS,
            prices_raw=prices_raw, volume=volume, amount=amount,
        )
        factor_scores.to_parquet(PROCESSED_DIR / "factor_scores_linear.parquet")

    elif mode in ML_MODES:
        model_types = list(ML_MODES - {"ensemble"}) if mode == "ensemble" else [mode]
        logger.info(f"Step 3: ML 策略（{mode}，模型={model_types}）")
        from strategies.ml import run as ml_run
        factor_scores, trainer = ml_run(
            prices, financial,
            prices_raw=prices_raw, volume=volume, amount=amount,
            model_types=model_types,
            show_report=show_report,
        )
        factor_scores.to_parquet(PROCESSED_DIR / f"factor_scores_{mode}.parquet")

    else:
        raise ValueError(f"未知 mode: {mode}，可选: {ALL_MODES}")

    # ── 回测 ──────────────────────────────────────────────────────────────────
    logger.info("Step 4: 运行回测")
    from backtest.engine import run_backtest, plot_result

    result = run_backtest(prices, factor_scores)
    plot_result(result, save_path=f"backtest_{mode}.png")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", default="linear",
        choices=sorted(ALL_MODES),
        help="策略模式: linear | ridge | lgbm | xgb | cat | ensemble",
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument(
        "--report", action="store_true",
        help="ML 模式训练完后展示 IC / SHAP 分析报告",
    )
    args = parser.parse_args()
    main(
        mode=args.mode,
        skip_download=args.skip_download,
        sample=args.sample,
        show_report=args.report,
    )
