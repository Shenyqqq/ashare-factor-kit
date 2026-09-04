"""
中盘 30–100 亿 × sizeind 全量重筛入口（``python -m research.midcap_ic``）。

产品语义（每日宇宙，禁止用全市场截面冒充）
========================================
1. **每日股票池**：东财 ``circ_mv`` ∈ **[30 亿, 100 亿]**（含边界；面板单位=元，
   即 ``[30e8, 100e8]``）。缺市值 → 当日剔除。
2. **再交可交易**：现有 ``build_ic_tradability_mask``（非 ST / 停牌 / 零成交 /
   次新 252 日；research 默认信号日保留涨跌停）。**不**另做 20 日成交额过滤，
   **不**自造更松的 tradable。
3. **因子值**：量价时序（动量/波动）可用该股自己的历史（不必历史上也在 30–100）。
   磁盘面板已是全市场 winsor+zscore，**不能直接拿去 IC**。本入口打开
   ``restan_in_universe``：在当日 30–100∩可交易 上重做截面 winsor(1%)+zscore(3σ)。
4. **Neut**：``--neut-controls size_industry``（仅 Size + PIT 行业），WLS √市值，
   **仅用当日池内股票估 β**（``membership_mask``）。**禁止** 9 风格 ``--barra``。
   Size 控制 = 池内 ``log(circ_mv)``，不用全市场 z 后的 ``Barra_Size``。
   行业哑元 PIT as-of；档内有效样本 < ``min_industry_n``（默认 10）并入「其他」。
5. **IC / sparse 胜率 / corr-dedup**：全部在当日宇宙；``MIN_IC_STOCKS`` 不足则当日
   IC=NaN。稀疏因子仍 skip 中性化，但 IC/胜率/payoff 在池内。
6. **缓存**：IC / barra_pure checkpoint 后缀含 ``mcap30_100`` + ``nc_size_industry``
   + 交易日历指纹，**禁止 resume** 全市场 ``barra_pure_h5``。
7. **全量重筛**：不写死历史 45/53 名单。
8. **基准/EW**：本脚本只做到 ``--save`` 出 YAML/JSON。回测等权必须是当日宇宙等权
   （不是全 A）——留给训练/回测阶段，见下方「不跑」。

默认与旗舰 nolongshare 对齐：``--min-long-share 0``。

不跑（留给用户一声令下）
------------------------
- ``run.py`` lgbm/ensemble 训练
- 分组回测 / 候选股输出
- 全量 IC 本体（本模块 ``--dry-run`` 只验 mask）

用法
----
    python -m research.midcap_ic --dry-run
    python -m research.midcap_ic --period 5 --save --mcap-min-yi 30 --mcap-max-yi 100
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 禁止 --save 落到这些全市场 YAML
_PROTECTED_YAML_NAMES = frozenset({
    "factor_configs.yaml",
    "factor_configs_h5_nolongshare_20260804.yaml",
    "factor_configs_h5_sizeind_20260815.yaml",
})


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "中盘 30–100 亿每日宇宙 → 池内截面标准化 → 池内 sizeind → 池内 IC/sparse。"
            "禁止 --barra。--dry-run 只打印日均池子，不跑 IC。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mcap-min-yi", type=float, default=30.0, dest="mcap_min_yi")
    p.add_argument("--mcap-max-yi", type=float, default=100.0, dest="mcap_max_yi")
    p.add_argument("--period", type=int, default=5)
    p.add_argument("--save", action="store_true")
    p.add_argument(
        "--save-suffix",
        dest="save_suffix",
        default="",
        help="JSON/YAML 文件名后缀；默认 midcap{lo}_{hi}_sizeind_YYYYMMDD",
    )
    p.add_argument(
        "--min-long-share",
        type=float,
        default=0.0,
        dest="min_long_share",
        help="稠密门 long_share 下限；默认 0（与旗舰 nolongshare 一致）",
    )
    p.add_argument(
        "--min-industry-n",
        type=int,
        default=10,
        dest="min_industry_n",
        help="档内行业哑元最少有效样本，少则并入「其他」（默认 10）",
    )
    p.add_argument("--dry-run", action="store_true", dest="dry_run")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--lookback-years", type=int, default=0, dest="lookback_years")
    return p.parse_args(argv)


def _load_universe_inputs():
    """dry-run 轻量载入：市值 + 价量 + ST/上市，不碰财务/因子面板。"""
    from config.settings import RAW_DIR
    from data.clean import clean_market_cap, clean_prices, clean_volume, mask_post_delist
    from data.mv_panels import load_mv_raw
    from research.ic.universe import load_delist_dates, load_is_st_current, load_stock_names

    _circ = load_mv_raw("circ_mv")
    if _circ is None:
        raise SystemExit(
            "缺少 circ_mv（请先 python -m data.download_stock_value_em，不要在本脚本下载）"
        )
    circ_mv = clean_market_cap(_circ, name="circ_mv")

    prices = clean_prices(
        pd.read_parquet(RAW_DIR / "prices_hfq.parquet"), label="prices_hfq",
    )
    from data.download import drop_excluded_universe_columns
    prices = drop_excluded_universe_columns(prices, name="prices")
    vol_raw = (
        pd.read_parquet(RAW_DIR / "volume.parquet")
        if (RAW_DIR / "volume.parquet").exists()
        else None
    )
    volume = clean_volume(vol_raw, name="volume") if vol_raw is not None else None
    volume = drop_excluded_universe_columns(volume)

    delist = load_delist_dates()
    if delist:
        prices = mask_post_delist(prices, delist)
        if volume is not None:
            volume = mask_post_delist(volume, delist)

    st_history = None
    st_path = RAW_DIR / "st_history.parquet"
    if st_path.exists():
        st_history = pd.read_parquet(st_path)

    return {
        "circ_mv": circ_mv,
        "prices": prices,
        "volume": volume,
        "stock_names": load_stock_names(),
        "is_st_current": load_is_st_current(),
        "listing_dates": None,  # filled below
        "delist_dates": delist,
        "st_history": st_history,
    }


def dry_run_universe(*, mcap_min_yi: float, mcap_max_yi: float) -> dict:
    """打印日均池子只数、覆盖、缺 circ_mv 比例；不跑 IC。"""
    from research.ic.universe import build_ic_tradability_mask, load_listing_dates
    from utils.universe import YI_TO_YUAN, build_mcap_yi_band_mask

    t0 = time.perf_counter()
    print(
        f"  [dry-run] 市值带 [{mcap_min_yi:.0f}, {mcap_max_yi:.0f}] 亿元 "
        f"（{mcap_min_yi * YI_TO_YUAN:.3g}–{mcap_max_yi * YI_TO_YUAN:.3g} 元，含边界）",
        flush=True,
    )
    data = _load_universe_inputs()
    data["listing_dates"] = load_listing_dates()
    circ_mv = data["circ_mv"]
    prices = data["prices"]
    circ_mv = circ_mv.reindex(index=prices.index, columns=prices.columns)

    mv_arr = circ_mv.to_numpy(dtype=np.float64, copy=False)
    finite_mv = np.isfinite(mv_arr)
    miss_ratio = float(1.0 - finite_mv.mean()) if finite_mv.size else 1.0
    per_miss = (~finite_mv).mean(axis=1) if finite_mv.size else np.array([1.0])
    print(
        f"  circ_mv 缺失格子比例={miss_ratio:.1%}  "
        f"日均缺失列占比={float(per_miss.mean()):.1%}"
    )

    mcap_mask = build_mcap_yi_band_mask(
        circ_mv, min_yi=mcap_min_yi, max_yi=mcap_max_yi,
    )
    mcap_mask = mcap_mask.reindex(index=prices.index, columns=prices.columns).fillna(False)
    tradable = build_ic_tradability_mask(
        prices,
        volume=data["volume"],
        masks=None,
        stock_names=data["stock_names"],
        is_st_current=data["is_st_current"],
        listing_dates=data["listing_dates"],
        delist_dates=data["delist_dates"],
        small_cap_mask=mcap_mask,
        st_history=data["st_history"],
        exclude_limit_on_signal=False,
    )
    per_mcap = mcap_mask.sum(axis=1)
    per_pool = tradable.sum(axis=1)
    stats = {
        "mcap_mean": float(per_mcap.mean()),
        "mcap_median": float(per_mcap.median()),
        "mcap_min": float(per_mcap.min()),
        "mcap_max": float(per_mcap.max()),
        "pool_mean": float(per_pool.mean()),
        "pool_median": float(per_pool.median()),
        "pool_min": float(per_pool.min()),
        "pool_max": float(per_pool.max()),
        "n_dates": int(len(per_pool)),
        "circ_mv_missing_cell_ratio": miss_ratio,
        "elapsed_s": time.perf_counter() - t0,
    }
    print(
        f"  市值带每日只数: mean={stats['mcap_mean']:.0f} median={stats['mcap_median']:.0f} "
        f"min={stats['mcap_min']:.0f} max={stats['mcap_max']:.0f}"
    )
    print(
        f"  市值带∩可交易 每日只数: mean={stats['pool_mean']:.0f} "
        f"median={stats['pool_median']:.0f} "
        f"min={stats['pool_min']:.0f} max={stats['pool_max']:.0f} "
        f"（n_dates={stats['n_dates']}）",
        flush=True,
    )
    n_zero = int((per_pool == 0).sum())
    if n_zero:
        print(
            f"  其中 {n_zero} 日池子为 0（多为 circ_mv 尚未覆盖的早期交易日）",
            flush=True,
        )
    print(f"  [dry-run] {stats['elapsed_s']:.1f}s  （未跑 IC / 训练 / 回测）")
    return stats


def _default_save_suffix(*, mcap_min_yi: float, mcap_max_yi: float) -> str:
    lo = int(mcap_min_yi)
    hi = int(mcap_max_yi)
    return f"midcap{lo}_{hi}_sizeind_{datetime.now():%Y%m%d}"


def _horizon_to_rebalance_freq(period: int) -> str:
    if period <= 5:
        return "W-FRI"
    if period <= 10:
        return "2W-FRI"
    return "ME"


def write_midcap_yaml(
    json_path: Path,
    yaml_path: Path,
    *,
    period: int,
) -> Path:
    """把 selected_factors JSON 写成独立 YAML，绝不覆盖全市场旗舰文件。"""
    if yaml_path.name in _PROTECTED_YAML_NAMES:
        raise SystemExit(f"拒绝覆盖全市场 YAML: {yaml_path}")
    try:
        import yaml
    except ImportError as e:
        raise SystemExit("缺少 PyYAML，请运行: pip install pyyaml") from e

    sel = json.loads(json_path.read_text(encoding="utf-8"))
    dense = list(sel.get("factors") or [])
    sparse = list(sel.get("factors_sparse") or [])
    key = f"h{int(period)}"
    cfg = {
        key: {
            "horizon": int(period),
            "rebalance_freq": _horizon_to_rebalance_freq(int(period)),
            "factors": dense,
        }
    }
    if sparse:
        cfg[key]["factors_sparse"] = sparse
    header = (
        f"# h{period} midcap size_industry {json_path.stem}: "
        f"dense={len(dense)} sparse={len(sparse)}\n"
        f"# gates: IC AND ICIR AND t/FDR AND corr_dedup; min_long_share=0; "
        f"research tradable; neut=size_industry (池内 log(circ_mv)+PIT sw_l2); "
        f"GS off; sparse-track on (not residualized)\n"
        f"# sparse inject: --special-factors sparse --sparse-from-ic {json_path.as_posix()}\n"
    )
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        header
        + yaml.dump(cfg, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    print(f"  [yaml] {yaml_path}")
    return yaml_path


def run_screen(args: argparse.Namespace) -> None:
    from research.ic.cli import run
    from research.ic.io import OUTPUT_DIR

    print(
        "中盘 sizeind 筛选：每日宇宙 → 池内 restan → 池内 Size+PIT行业 WLS → 池内 IC/sparse。"
        " 不跑 lgbm/回测。"
    )
    if args.mcap_min_yi >= args.mcap_max_yi:
        raise SystemExit("--mcap-min-yi 必须 < --mcap-max-yi")
    save_suffix = (args.save_suffix or "").strip() or _default_save_suffix(
        mcap_min_yi=args.mcap_min_yi, mcap_max_yi=args.mcap_max_yi,
    )
    yaml_path = (
        Path("config") / f"factor_configs_h{args.period}_{save_suffix}.yaml"
    )
    if yaml_path.name in _PROTECTED_YAML_NAMES:
        raise SystemExit(f"拒绝覆盖全市场 YAML: {yaml_path}")
    print(
        f"  [save] suffix={save_suffix}  JSON→research/output/  "
        f"YAML→{yaml_path.as_posix()}  workers=1"
    )
    run(
        period=args.period,
        save=args.save,
        barra=False,
        neut_controls="size_industry",
        min_long_share=args.min_long_share,
        mcap_min_yi=args.mcap_min_yi,
        mcap_max_yi=args.mcap_max_yi,
        restan_in_universe=True,
        min_industry_n=args.min_industry_n,
        resume=args.resume,
        fresh=args.fresh,
        sample=args.sample,
        lookback_years=args.lookback_years,
        cap_band="all",
        universe="all",
        save_suffix=save_suffix,
        workers=1,
        barra_workers=1,
    )
    if not args.save:
        return
    json_hits = sorted(
        OUTPUT_DIR.glob(f"selected_factors_h{args.period}_{save_suffix}*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not json_hits:
        print(f"  [yaml] 未找到 selected_factors_h{args.period}_{save_suffix}*.json，跳过 YAML")
        return
    write_midcap_yaml(json_hits[0], yaml_path, period=args.period)


def main(argv: list[str] | None = None) -> None:
    from config.encoding_bootstrap import bootstrap_stdio_utf8

    bootstrap_stdio_utf8()
    args = _parse_args(argv)
    if args.dry_run:
        dry_run_universe(
            mcap_min_yi=args.mcap_min_yi, mcap_max_yi=args.mcap_max_yi,
        )
        return
    run_screen(args)


if __name__ == "__main__":
    main()
