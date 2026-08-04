"""
data/compute_market_cap.py — 自算日频市值 / 换手（校验与兜底；非 Size 默认源）

【定位 · 2026-07-30】
  Size / WLS √市值 / 对数市值的**主路径**已改为东财
  ``python -m data.download_stock_value_em`` → ``total_mv`` / ``circ_mv``。
  本模块默认只写出：
    - ``total_mv_computed.parquet`` / ``circ_mv_computed.parquet``（对照/兜底）
    - ``turnover_rate.parquet``（Barra_Liquidity 仍依赖；与 Size 解耦）
  仅当显式 ``--promote-main`` 时，才覆盖主文件 ``total_mv`` / ``circ_mv``
 （东财不可用时的应急）。

输入：
  data/raw/share_change.parquet  — 股本变动长表（download_shares.py 产出）
  data/raw/prices_raw.parquet    — 不复权收盘价（用于市值计算）
                                    ⚠️ 必须用不复权价：后复权已调整 splits，
                                    会扭曲「实际股本 × 实际价格」的市值
  data/raw/volume.parquet        — 成交量（用于算 turnover_rate）

逻辑（PIT 安全）：
  1. 对每只股票，按 announce_date（公告日）排序股本变动记录。
  2. 构造日频 total_shares / circ_shares 面板：
     - 用 announce_date 作为 ffill 起点——只有当 trade_date >= announce_date
       后，新股本数才「已知」（披露日才公开）。
     - trade_date < 首条 announce_date 的填 NaN（不向后填）。
     - 实现：pivot 到 (announce_date × code) 稀疏表 → reindex 到
       trade_dates ∪ announce_dates 联合索引 → ffill（按列）→ reindex 回 trade_dates。
  3. total_mv    = prices_raw × total_shares   （单位：元）
  4. circ_mv     = prices_raw × circ_shares    （单位：元）
  5. turnover_rate = (volume×100) / circ_shares （小数；circ_shares<=0/NaN 时置 NaN）

用法：
    python -m data.compute_market_cap                          # 自算校验 + 换手
    python -m data.compute_market_cap --promote-main           # 应急覆盖主面板
    python -m data.compute_market_cap --start 2018-01-01
    python -m data.compute_market_cap --sample 50 --codes 600519,600000
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import RAW_DIR

SHARES_PATH  = RAW_DIR / "share_change.parquet"
PRICES_PATH  = RAW_DIR / "prices_raw.parquet"
VOLUME_PATH  = RAW_DIR / "volume.parquet"

# volume.parquet 单位为「手」（akshare stock_zh_a_hist 成交量列），
# 1 手 = 100 股。换手率需用「股」口径：turnover = (volume×100) / circ_shares。
VOLUME_MUL = 100

# 默认写出到 *_computed（不覆盖东财主面板）；--promote-main 才写主文件
OUT_TOTAL_MV_COMPUTED = RAW_DIR / "total_mv_computed.parquet"
OUT_CIRC_MV_COMPUTED = RAW_DIR / "circ_mv_computed.parquet"
OUT_TOTAL_MV = RAW_DIR / "total_mv.parquet"
OUT_CIRC_MV = RAW_DIR / "circ_mv.parquet"
OUT_TURNOVER = RAW_DIR / "turnover_rate.parquet"
BACKUP_TOTAL_MV = RAW_DIR / "total_mv_baidu_backup.parquet"

SPIKE_WINDOW = 20
SPIKE_MULT   = 10.0


def _load_shares(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}，请先运行 `python -m data.download_shares`")
    df = pd.read_parquet(path)
    needed = ["code", "announce_date", "total_shares", "circ_shares"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{path} 缺列: {missing}")
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["announce_date"] = pd.to_datetime(df["announce_date"], errors="coerce")
    df = df.dropna(subset=["announce_date"])
    for c in ("total_shares", "circ_shares"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).where(df[c] > 0)
    # 同一 (code, announce_date) 多条 → 保留最后一条
    df = df.sort_values(["code", "announce_date"])
    df = df.drop_duplicates(subset=["code", "announce_date"], keep="last")
    return df


def _build_shares_panel(
    shares_long: pd.DataFrame,
    trade_dates: pd.DatetimeIndex,
    col: str,
) -> pd.DataFrame:
    """长表 → wide 面板（index=trade_date, columns=code），按 announce_date ffill。

    PIT 安全：只有 trade_date >= announce_date 后股本才「已知」。
    向后填到无限远（直到下一条 announce_date 替换）；不向前填。
    """
    sub = shares_long[["code", "announce_date", col]].dropna(subset=[col])
    if sub.empty:
        return pd.DataFrame(index=trade_dates, columns=[])
    # pivot：index=announce_date（各股混合），columns=code，values=col
    pivot = sub.pivot_table(
        index="announce_date", columns="code", values=col, aggfunc="last"
    )
    pivot.index = pd.to_datetime(pivot.index)
    # 联合索引：trade_dates ∪ announce_dates，按时间排序
    union_idx = trade_dates.union(pivot.index).unique().sort_values()
    # reindex 到联合索引后 ffill（按列/按股票），再 reindex 回 trade_dates
    full = pivot.reindex(union_idx)
    full = full.ffill()
    panel = full.reindex(trade_dates)
    panel.index.name = "date"
    return panel


def _clean_market_cap_panel(
    df: pd.DataFrame, name: str
) -> pd.DataFrame:
    """市值面板清洗（自包含，不改 data/clean.py）：
       inf→NaN、负值→NaN、0→NaN、突增告警（保留原值）。
    """
    if df is None or df.empty:
        return df
    out = df.apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    n_neg = int((out < 0).sum().sum())
    if n_neg:
        logger.warning(f"{name}: {n_neg} 个负值 → NaN")
    out = out.where(out >= 0)
    n_zero = int((out == 0).sum().sum())
    if n_zero:
        logger.warning(f"{name}: {n_zero} 个 0 值 → NaN（市值 0 无意义）")
    out = out.where(out != 0)
    try:
        roll = out.rolling(window=SPIKE_WINDOW, min_periods=5).mean()
        n_spike = int((out > roll * SPIKE_MULT).sum().sum())
        if n_spike:
            logger.warning(
                f"{name}: {n_spike} 个突增格子（>{SPIKE_WINDOW}日均值的{SPIKE_MULT}倍），"
                f"保留原值，请人工核查（可能是真实拆股/增发）"
            )
    except Exception as e:
        logger.debug(f"{name}: 突增检测跳过 ({e})")
    logger.info(f"{name}: 清洗完成 shape={out.shape} NaN格={int(out.isna().sum().sum())}")
    return out


def compute_market_cap(
    shares_path: Path = SHARES_PATH,
    prices_path: Path = PRICES_PATH,
    volume_path: Path = VOLUME_PATH,
    start: str | None = "2018-01-01",
    end: str | None = None,
    sample_codes: list[str] | None = None,
    promote_main: bool = False,
) -> dict[str, pd.DataFrame]:
    """自算流程。返回 {total_mv, circ_mv, turnover_rate}。

    默认写出 ``*_computed.parquet`` + ``turnover_rate.parquet``；
    ``promote_main=True`` 时额外覆盖东财主面板 ``total_mv`` / ``circ_mv``。
    """
    # ── 1. 读输入 ──
    shares_long = _load_shares(shares_path)
    logger.info(f"share_change: {shares_long.shape}  "
                f"覆盖 {shares_long['code'].nunique()} 只  "
                f"announce {shares_long['announce_date'].min().date()} → "
                f"{shares_long['announce_date'].max().date()}")

    prices = pd.read_parquet(prices_path)
    prices.index = pd.to_datetime(prices.index)
    prices.columns = prices.columns.astype(str).str.zfill(6)
    logger.info(f"prices_raw: {prices.shape}  "
                f"{prices.index.min().date()} → {prices.index.max().date()}")

    volume = pd.read_parquet(volume_path)
    volume.index = pd.to_datetime(volume.index)
    volume.columns = volume.columns.astype(str).str.zfill(6)
    logger.info(f"volume: {volume.shape}")

    # ── 2. 截取输出日期范围（在 ffill 之后再切片，避免破坏 ffill 起点）──
    trade_dates = prices.index
    # ── 3. 构造股本面板 ──
    if sample_codes is not None:
        sample_set = set(sample_codes)
        shares_long = shares_long[shares_long["code"].isin(sample_set)]
        # prices/volume 仍保留全部列即可（运算时自动对齐 NaN），但为省内存可截
        keep_cols = [c for c in prices.columns if c in sample_set]
        prices = prices[keep_cols]
        keep_cols_v = [c for c in volume.columns if c in sample_set]
        volume = volume[keep_cols_v]
        logger.info(f"sample 模式：仅用 {len(sample_codes)} 只 → {sorted(sample_codes)}")

    logger.info("构造 total_shares / circ_shares 日频面板（PIT ffill）...")
    total_shares_panel = _build_shares_panel(shares_long, trade_dates, "total_shares")
    circ_shares_panel  = _build_shares_panel(shares_long, trade_dates, "circ_shares")
    logger.info(f"total_shares_panel: {total_shares_panel.shape}  "
                f"circ_shares_panel: {circ_shares_panel.shape}")

    # 对齐到 prices 的列
    cols = prices.columns
    total_shares_panel = total_shares_panel.reindex(columns=cols)
    circ_shares_panel  = circ_shares_panel.reindex(columns=cols)

    # ── 4. 计算市值 ──
    total_mv = prices * total_shares_panel
    circ_mv  = prices * circ_shares_panel
    # turnover_rate = (volume[手] × 100) / circ_shares[股]
    with np.errstate(divide="ignore", invalid="ignore"):
        turnover_rate = (volume * VOLUME_MUL) / circ_shares_panel
    turnover_rate = turnover_rate.replace([np.inf, -np.inf], np.nan)
    # 0 流通股本对应的换手率置 NaN（已被 circ_shares<=0→NaN 自然处理，兜底再清一次）
    turnover_rate = turnover_rate.where(circ_shares_panel > 0)

    # ── 5. 日期切片 ──
    if start is not None or end is not None:
        lo = pd.Timestamp(start) if start else None
        hi = pd.Timestamp(end)   if end   else None
        total_mv      = total_mv.loc[lo:hi]
        circ_mv       = circ_mv.loc[lo:hi]
        turnover_rate = turnover_rate.loc[lo:hi]

    # ── 6. 清洗 ──
    total_mv      = _clean_market_cap_panel(total_mv,      "total_mv")
    circ_mv       = _clean_market_cap_panel(circ_mv,       "circ_mv")
    # turnover_rate 用轻清洗（0 是合法值，不置 NaN）
    if turnover_rate is not None and not turnover_rate.empty:
        turnover_rate = turnover_rate.apply(pd.to_numeric, errors="coerce")
        turnover_rate = turnover_rate.replace([np.inf, -np.inf], np.nan)
        n_neg = int((turnover_rate < 0).sum().sum())
        if n_neg:
            logger.warning(f"turnover_rate: {n_neg} 个负值 → NaN")
        turnover_rate = turnover_rate.where(turnover_rate >= 0)

    # ── 7. 写出：默认 *_computed + turnover；可选 promote 主面板 ──
    write_list: list[tuple[Path, pd.DataFrame, str]] = [
        (OUT_TOTAL_MV_COMPUTED, total_mv, "total_mv_computed"),
        (OUT_CIRC_MV_COMPUTED, circ_mv, "circ_mv_computed"),
        (OUT_TURNOVER, turnover_rate, "turnover_rate"),
    ]
    if promote_main:
        logger.warning(
            "--promote-main：用自算覆盖东财主面板 total_mv/circ_mv "
            "（仅应急；日常请跑 download_stock_value_em）"
        )
        if OUT_TOTAL_MV.exists() and not BACKUP_TOTAL_MV.exists():
            shutil.copy2(OUT_TOTAL_MV, BACKUP_TOTAL_MV)
            logger.info(f"已备份旧主面板 → {BACKUP_TOTAL_MV.name}")
        write_list.extend([
            (OUT_TOTAL_MV, total_mv, "total_mv"),
            (OUT_CIRC_MV, circ_mv, "circ_mv"),
        ])
    else:
        logger.info(
            "自算市值写入 *_computed.parquet（不覆盖东财主面板）；"
            "换手率仍写 turnover_rate.parquet"
        )

    for path, panel, name in write_list:
        if panel is None or panel.empty:
            logger.warning(f"{name}: 空，跳过写出")
            continue
        panel.to_parquet(path)
        logger.info(
            f"写出 {path.name}: shape={panel.shape}  "
            f"{panel.index.min().date()} → {panel.index.max().date()}"
        )

    return dict(total_mv=total_mv, circ_mv=circ_mv, turnover_rate=turnover_rate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",  default="2018-01-01",
                        help="输出面板日期下界（含），ffill 仍用更早的股本记录作起点")
    parser.add_argument("--end",    default=None,
                        help="输出面板日期上界（含），默认到 prices_raw 末日")
    parser.add_argument("--sample", type=int, default=0,
                        help="调试：仅用 share_change 中前 N 只股票（按 code 排序）")
    parser.add_argument("--codes",  default=None,
                        help="调试：指定逗号分隔代码列表（如 600519,600000），优先于 --sample")
    parser.add_argument(
        "--promote-main", action="store_true",
        help="应急：用自算覆盖 total_mv/circ_mv 主面板（默认只写 *_computed）",
    )
    args = parser.parse_args()

    sample_codes = None
    if args.codes:
        sample_codes = [c.strip().zfill(6) for c in args.codes.split(",") if c.strip()]
    elif args.sample:
        # 取 share_change 中前 N 个 code
        if SHARES_PATH.exists():
            df = pd.read_parquet(SHARES_PATH)
            df["code"] = df["code"].astype(str).str.zfill(6)
            sample_codes = sorted(df["code"].unique())[:args.sample]
        else:
            logger.error(f"--sample 需要 {SHARES_PATH} 已存在")

    res = compute_market_cap(
        start=args.start, end=args.end, sample_codes=sample_codes,
        promote_main=args.promote_main,
    )
    # ── 摘要 ──
    tm, cm, tr = res["total_mv"], res["circ_mv"], res["turnover_rate"]
    print("\n" + "=" * 70)
    print("摘要")
    print("=" * 70)
    for name, p in [("total_mv", tm), ("circ_mv", cm), ("turnover_rate", tr)]:
        if p is None or p.empty:
            print(f"{name:15s} EMPTY"); continue
        print(f"{name:15s} shape={p.shape}  "
              f"date {p.index.min().date()} → {p.index.max().date()}")
    # 茅台量级校验
    if tm is not None and "600519" in tm.columns:
        moutai_tm = tm["600519"].dropna()
        if not moutai_tm.empty:
            last = moutai_tm.iloc[-1]
            print(f"\n[量级校验] 600519 茅台 total_mv 最新 = {last:.3e} 元 "
                  f"({last/1e8:.1f} 亿元 = {last/1e12:.3f} 万亿)")
    # circ_mv <= total_mv 校验
    if tm is not None and cm is not None and not tm.empty and not cm.empty:
        common = tm.columns.intersection(cm.columns)
        ratio = (cm[common] / tm[common])
        bad = (ratio > 1.0001).sum().sum()
        print(f"[校验] circ_mv > total_mv 的格子数: {int(bad)}（应为 0）")
        print(f"[校验] circ_mv/total_mv 中位数: "
              f"{float(ratio.stack().median()):.4f}（应 ≤ 1）")
    # turnover_rate 区间
    if tr is not None and not tr.empty:
        s = tr.stack()
        print(f"[校验] turnover_rate: min={s.min():.4f}  median={s.median():.4f}  "
              f"max={s.max():.4f}  (>0.5 占比={float((s>0.5).mean()):.4%})")


if __name__ == "__main__":
    main()
