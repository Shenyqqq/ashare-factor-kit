"""
strategies/market_state.py  —  市场状态识别

用途：
    1. 作为额外特征加入ML模型（state_encoded = {bull:1, neutral:0, bear:-1}）
    2. 高层路由：在熊市降低仓位或切换为防御性因子
    3. 分状态IC分析：看因子在牛熊市中是否表现不同

方法：
    基于沪深300（sh000300）的多均线状态机：
    - Bull:    price > MA20 > MA60，且MA20斜率 > 0
    - Bear:    price < MA20 < MA60，且MA20斜率 < 0
    - Neutral: 其他（震荡/过渡期）

    可选：HMM（2/3态隐马尔科夫），更平滑但依赖hmmlearn库。

用法:
    from strategies.market_state import get_market_state
    state = get_market_state()      # 返回 Series(date→'bull'/'bear'/'neutral')
    state_encoded = encode_state(state)  # 1/0/-1

    python -m strategies.market_state   # 下载数据并打印状态分布
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import RAW_DIR

STATE_PATH = RAW_DIR / "market_state.parquet"
INDEX_PATH = RAW_DIR / "csi300.parquet"
CSI_ALL_PATH = RAW_DIR / "csi_all.parquet"

# 沪深300 AKShare symbol
CSI300_SYMBOL = "sh000300"
# 中证全指 AKShare symbol（覆盖全A股，更适合做市场组合代理）
CSI_ALL_SYMBOL = "sh000985"


# ── 数据获取 ──────────────────────────────────────────────────────────────────

def download_csi300(start: str = "2017-01-01",
                    end: str = None,
                    force: bool = False) -> pd.Series:
    """下载沪深300日收盘价，缓存到 data/raw/csi300.parquet"""
    if INDEX_PATH.exists() and not force:
        series = pd.read_parquet(INDEX_PATH)["close"]
        return series

    import akshare as ak
    from loguru import logger

    logger.info("下载沪深300指数(sh000300)...")
    # 优先东方财富源（数据更新），失败回退到原源
    try:
        df = ak.stock_zh_index_daily_em(symbol=CSI300_SYMBOL)
    except Exception:
        df = ak.stock_zh_index_daily(symbol=CSI300_SYMBOL)
    df.index = pd.to_datetime(df["date"])
    df = df.sort_index()

    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index <= end]

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[["open", "high", "low", "close", "volume"]].to_parquet(INDEX_PATH)
    return df["close"]


def download_csi_all(start: str = "2017-01-01",
                     end: str = None,
                     force: bool = False) -> pd.Series:
    """
    下载中证全指日收盘价，缓存到 data/raw/csi_all.parquet。

    中证全指（000985）覆盖全A股，比沪深300（大盘股）更适合做市场组合代理，
    用于 IVOL / Barra beta 等需要市场收益的因子。
    """
    if CSI_ALL_PATH.exists() and not force:
        series = pd.read_parquet(CSI_ALL_PATH)["close"]
        return series

    import akshare as ak
    from loguru import logger

    logger.info("下载中证全指指数(sh000985)...")
    # 优先东方财富源（数据更新），失败回退到原源
    try:
        df = ak.stock_zh_index_daily_em(symbol=CSI_ALL_SYMBOL)
    except Exception:
        df = ak.stock_zh_index_daily(symbol=CSI_ALL_SYMBOL)
    df.index = pd.to_datetime(df["date"])
    df = df.sort_index()

    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index <= end]

    CSI_ALL_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[["open", "high", "low", "close", "volume"]].to_parquet(CSI_ALL_PATH)
    return df["close"]


# ── 状态计算 ──────────────────────────────────────────────────────────────────

def _ma_state(close: pd.Series,
              fast: int = 20,
              slow: int = 60,
              slope_window: int = 5) -> pd.Series:
    """
    多均线状态机：
    bull:    price > MA_fast > MA_slow AND MA_fast_slope > 0
    bear:    price < MA_fast < MA_slow AND MA_fast_slope < 0
    neutral: 其他
    """
    ma_fast  = close.rolling(fast).mean()
    ma_slow  = close.rolling(slow).mean()
    slope    = ma_fast.diff(slope_window)   # 均线斜率（简化为差分）

    cond_bull = (
        (close > ma_fast) &
        (ma_fast > ma_slow) &
        (slope > 0)
    )
    cond_bear = (
        (close < ma_fast) &
        (ma_fast < ma_slow) &
        (slope < 0)
    )

    state = pd.Series("neutral", index=close.index, name="market_state")
    state[cond_bull] = "bull"
    state[cond_bear] = "bear"

    return state


def _trend_state(close: pd.Series,
                 ma: int = 120,
                 threshold: float = 0.05) -> pd.Series:
    """
    简单趋势跟踪：收盘价相对MA120的偏离度
    > +5%: bull
    < -5%: bear
    其他: neutral
    """
    deviation = (close / close.rolling(ma).mean() - 1)
    state = pd.Series("neutral", index=close.index, name="market_state")
    state[deviation > threshold]  = "bull"
    state[deviation < -threshold] = "bear"
    return state


def compute_market_state(
    close: pd.Series,
    method: str = "ma",
    **kwargs,
) -> pd.Series:
    """
    计算市场状态序列

    method:
        "ma"    — 多均线状态机（默认，直觉清晰）
        "trend" — MA120偏离度（更滞后但稳定）
    """
    if method == "ma":
        return _ma_state(close, **kwargs)
    elif method == "trend":
        return _trend_state(close, **kwargs)
    else:
        raise ValueError(f"未知方法: {method}，可选: ma / trend")


# ── 公开接口 ──────────────────────────────────────────────────────────────────

def get_market_state(
    method: str = "ma",
    force_download: bool = False,
    reindex: pd.DatetimeIndex = None,
) -> pd.Series:
    """
    获取市场状态序列（自动下载+缓存）

    参数：
        method: "ma" 或 "trend"
        force_download: 强制重新下载沪深300数据
        reindex: 如果提供，将状态对齐到这组日期（ffill填充）

    返回：
        Series(index=date, values=['bull','bear','neutral'])
    """
    if STATE_PATH.exists() and not force_download:
        state = pd.read_parquet(STATE_PATH)["market_state"]
    else:
        close = download_csi300(force=force_download)
        state = compute_market_state(close, method=method)
        pd.DataFrame({"market_state": state}).to_parquet(STATE_PATH)

    if reindex is not None:
        state = state.reindex(reindex).ffill()

    return state


def encode_state(state: pd.Series) -> pd.Series:
    """
    将文字状态编码为数值特征（ML训练时用）
    bull=1, neutral=0, bear=-1
    """
    mapping = {"bull": 1, "neutral": 0, "bear": -1}
    return state.map(mapping).astype(float)


def get_state_mask(state: pd.Series,
                   which: str) -> pd.Series:
    """返回布尔掩码，用于分状态回测/IC分析"""
    return state == which


# ── 分析工具 ──────────────────────────────────────────────────────────────────

def print_state_summary(state: pd.Series):
    counts = state.value_counts()
    total  = len(state)
    print("\n市场状态分布")
    print("=" * 40)
    for s in ["bull", "neutral", "bear"]:
        n = counts.get(s, 0)
        bar = "█" * (n * 30 // total)
        print(f"  {s:<8}: {n:>4}日 ({n/total:>5.1%})  {bar}")
    print(f"\n  时间范围: {state.index[0].date()} ~ {state.index[-1].date()}")

    # 逐年分布
    print("\n逐年状态占比（%）:")
    by_year = state.groupby(state.index.year).value_counts(normalize=True).unstack(fill_value=0)
    print((by_year * 100).round(1).to_string())


def plot_market_state(close: pd.Series, state: pd.Series,
                      save_path: str = None):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams["font.family"] = "SimHei"
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(16, 5))
    close.plot(ax=ax, color="steelblue", lw=1.2, label="沪深300")

    # 背景颜色标注状态
    state_aligned = state.reindex(close.index).ffill()
    prev = None
    start = None
    for date, s in state_aligned.items():
        if s != prev:
            if prev is not None and start is not None:
                color = {"bull": "limegreen", "bear": "salmon",
                         "neutral": "lightyellow"}[prev]
                ax.axvspan(start, date, alpha=0.25, color=color)
            start = date
            prev = s
    if prev is not None and start is not None:
        color = {"bull": "limegreen", "bear": "salmon",
                 "neutral": "lightyellow"}[prev]
        ax.axvspan(start, close.index[-1], alpha=0.25, color=color)

    # 均线
    close.rolling(20).mean().plot(ax=ax, color="orange",  lw=1, ls="--", label="MA20")
    close.rolling(60).mean().plot(ax=ax, color="red",     lw=1, ls="--", label="MA60")

    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color="limegreen", alpha=0.5, label="牛市(bull)"),
        Patch(color="lightyellow", alpha=0.8, label="震荡(neutral)"),
        Patch(color="salmon",    alpha=0.5, label="熊市(bear)"),
    ]
    ax.legend(handles=legend_patches + ax.get_lines()[:3], fontsize=9)
    ax.set_title("沪深300市场状态识别（MA多均线法）")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


# ── 分状态IC分析 ──────────────────────────────────────────────────────────────

def ic_by_state(factor: pd.DataFrame,
                forward_return: pd.DataFrame,
                state: pd.Series) -> pd.DataFrame:
    """
    计算各市场状态下的IC均值。
    返回 DataFrame(index=['bull','neutral','bear'], columns=['IC均值','ICIR','胜率','样本数'])
    """
    from research.ic_analysis import compute_ic_series

    ic_full = compute_ic_series(factor, forward_return)
    rows = []
    for s in ["bull", "neutral", "bear"]:
        mask = state.reindex(ic_full.index).ffill() == s
        ic_s = ic_full[mask]
        if len(ic_s) == 0:
            rows.append({"状态": s, "IC均值": np.nan, "ICIR": np.nan,
                          "胜率": np.nan, "样本数": 0})
            continue
        icir = ic_s.mean() / ic_s.std() if ic_s.std() > 0 else 0
        rows.append({
            "状态":   s,
            "IC均值": round(ic_s.mean(), 4),
            "ICIR":   round(icir, 4),
            "胜率":   round((ic_s > 0).mean(), 4),
            "样本数": len(ic_s),
        })
    return pd.DataFrame(rows).set_index("状态")


if __name__ == "__main__":
    print("下载中证全指与沪深300指数...")
    # 中证全指用于 IVOL / Barra beta 等市场组合代理
    close_all = download_csi_all(force=False)
    # 沪深300用于市场状态计算（保持原行为）
    close = download_csi300(force=False)
    state = compute_market_state(close, method="ma")

    # 保存缓存
    pd.DataFrame({"market_state": state}).to_parquet(STATE_PATH)
    print(f"状态已保存: {STATE_PATH}")

    print_state_summary(state)

    try:
        plot_market_state(close, state)
    except Exception as e:
        print(f"绘图失败（非致命）: {e}")
