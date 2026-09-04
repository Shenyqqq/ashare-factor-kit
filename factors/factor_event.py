"""
factors/factor_event.py  —  事件驱动因子

目前实现：
    factor_yjyg(prices, window=60)
        业绩预告因子。预增/扭亏→正分，预减/首亏→负分；
        magnitude 放大，pre-drift 惩罚已提前炒作；
        信号持续 window 个交易日后衰减至 0。

接口规范：返回 DataFrame(index=date, columns=stock, value=factor_score)
            高分 = 正向信号（买入），低分 = 负向信号
            数值已经过截面 winsorize+zscore，但本因子不做，由调用方统一处理
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import RAW_DIR

YJYG_PATH = RAW_DIR / "yjyg.parquet"

# ── 预告类型得分映射 ──────────────────────────────────────────────────────────
# 正向：盈利增长或扭亏；负向：亏损或业绩下滑
TYPE_BASE_SCORE: dict[str, float] = {
    "扭亏":  3.0,   # 从亏到盈，最强正信号
    "预增":  2.0,   # 预期大幅增长
    "略增":  1.0,   # 预期小幅增长
    "首盈":  2.5,   # 首次扭亏（类似扭亏）
    "续盈":  0.3,   # 持续盈利，弱正信号
    "不确定": 0.0,  # 不确定
    "略减": -1.0,   # 预期小幅下降
    "预减": -2.0,   # 预期大幅下降
    "续亏": -1.5,   # 持续亏损
    "预亏": -2.5,   # 预期亏损
    "首亏": -3.0,   # 从盈到亏，最强负信号
}


def _safe_log_magnitude(change_pct: pd.Series) -> pd.Series:
    """
    将变动幅度（%）映射为对数放大系数。
    扭亏/首亏 change_pct 可能为 NaN 或 inf，用 1.0 填充。
    """
    pct = pd.to_numeric(change_pct, errors="coerce").fillna(0)
    pct = pct.clip(-1000, 1000)   # 防止极端值
    return np.log1p(pct.abs() / 100).clip(0, 3)   # [0, 3]


def factor_yjyg(
    prices: pd.DataFrame,
    window: int = 60,
    pre_drift_period: int = 60,
    pre_drift_penalty: float = 0.3,
) -> pd.DataFrame:
    """
    业绩预告因子

    参数：
        prices: 复权价 (date x stock)
        window: 信号持续窗口（交易日），超出后信号衰减至0
        pre_drift_period: 预公告前的价格漂移计算窗口（交易日）
        pre_drift_penalty: pre-drift 惩罚系数

    返回：
        factor_df: (date x stock) DataFrame，高分=正向信号

    信号设计：
        score = base_score * (1 + log_magnitude) - penalty * pre_drift_zscore
    """
    if not YJYG_PATH.exists():
        raise FileNotFoundError(
            f"业绩预告数据不存在: {YJYG_PATH}\n"
            "请先运行: python -m data.events.download_yjyg"
        )

    yjyg = pd.read_parquet(YJYG_PATH)

    # 只保留有announce_date的记录
    yjyg = yjyg.dropna(subset=["announce_date", "code", "forecast_type"])
    yjyg["announce_date"] = pd.to_datetime(yjyg["announce_date"], errors="coerce")
    yjyg = yjyg.dropna(subset=["announce_date"])

    # 一家公司同一报告期可能有多条（不同指标），取 forecast_type 最重要的一条
    # 优先级：type_base_score 绝对值最大的
    yjyg["type_score_base"] = yjyg["forecast_type"].map(TYPE_BASE_SCORE).fillna(0)
    yjyg = (
        yjyg.assign(_abs_score=lambda df: df["type_score_base"].abs())
        .sort_values("_abs_score", ascending=False)
        .drop_duplicates(subset=["code", "report_date"])
        .drop(columns="_abs_score")
    )

    # ── 计算最终信号得分 ──────────────────────────────────────────────────────
    yjyg["log_mag"]    = _safe_log_magnitude(yjyg["change_pct"])
    yjyg["raw_score"]  = yjyg["type_score_base"] * (1 + yjyg["log_mag"])

    # pre-drift：公告前 pre_drift_period 个交易日的价格涨跌
    # 提前涨得越多，可能alpha已经被耗尽，惩罚正信号
    trading_dates = prices.index
    announce_series = yjyg.set_index("code")["announce_date"]

    drift_records = {}
    for code, ann_date in announce_series.items():
        if code not in prices.columns:
            continue
        # 找公告日的前一个交易日
        before = trading_dates[trading_dates < ann_date]
        if len(before) == 0:
            continue
        end_idx   = before[-1]
        start_arr = before[max(0, len(before) - pre_drift_period):]
        if len(start_arr) == 0:
            continue
        start_idx = start_arr[0]
        p0 = prices.loc[start_idx, code] if start_idx in prices.index else np.nan
        p1 = prices.loc[end_idx,   code] if end_idx   in prices.index else np.nan
        if np.isnan(p0) or np.isnan(p1) or p0 == 0:
            continue
        drift_records[code] = (p1 - p0) / p0  # 前期收益率

    drift_s = pd.Series(drift_records, name="pre_drift")
    yjyg = yjyg.join(drift_s, on="code", how="left")
    yjyg["pre_drift"] = yjyg["pre_drift"].fillna(0)

    # pre_drift 截面标准化（简单clip+zscore，不做完整截面，因为公告日不统一）
    drift_std = yjyg["pre_drift"].std()
    if drift_std > 0:
        yjyg["drift_z"] = (yjyg["pre_drift"] / drift_std).clip(-3, 3)
    else:
        yjyg["drift_z"] = 0.0

    yjyg["final_score"] = (
        yjyg["raw_score"] - pre_drift_penalty * yjyg["drift_z"]
    )

    # ── 把事件信号展开到日频 factor_df ──────────────────────────────────────
    # 方法：在 announce_date 放入信号，然后 forward-fill 最多 window 天
    all_dates  = prices.index
    all_stocks = prices.columns

    # pivot: date x stock (只有公告日有值)
    event_pivot = (
        yjyg[["announce_date", "code", "final_score"]]
        .pivot_table(index="announce_date", columns="code",
                     values="final_score", aggfunc="last")
    )

    # 重建到完整交易日历
    event_daily = event_pivot.reindex(all_dates)

    # forward fill 最多 window 天
    # 同一只股票后续公告会覆盖旧信号
    factor_df = event_daily.ffill(limit=window).reindex(columns=all_stocks)

    return factor_df


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    print("载入价格数据...")
    prices = pd.read_parquet(RAW_DIR / "prices_hfq.parquet")

    print("计算业绩预告因子...")
    factor_df = factor_yjyg(prices)

    print(f"\n因子维度: {factor_df.shape}")
    print(f"时间范围: {factor_df.index[0].date()} ~ {factor_df.index[-1].date()}")

    coverage = factor_df.notna().mean(axis=1)
    print(f"\n每日覆盖率（均值={coverage.mean():.1%}）:")
    print(coverage.resample("QE").mean().round(3).to_string())

    # 简单IC测试
    fwd_ret = prices.pct_change(20).shift(-20)
    from research.ic_analysis import compute_ic_series
    ic = compute_ic_series(factor_df, fwd_ret)
    print(f"\n20日IC: mean={ic.mean():.4f}, std={ic.std():.4f}, ICIR={ic.mean()/ic.std():.4f}")
