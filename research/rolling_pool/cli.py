"""CLI: ``python -m research.rolling_pool``."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from research.rolling_pool.corr_provider import (
    CachedPanelCorrProvider,
    NullFactorCorrProvider,
)
from research.rolling_pool.io import (
    DEFAULT_CKPT,
    OUTPUT_DIR,
    load_barra_pure_ic,
    write_schedule_outputs,
)
from research.rolling_pool.schedule import PoolParams, build_pool_schedule


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="周频 Barra pure IC 轮动定池 → schedule 长表",
    )
    p.add_argument(
        "--ckpt",
        type=str,
        default=str(DEFAULT_CKPT),
        help="barra_pure checkpoint（默认 research/output/_checkpoints/barra_pure_h5.pkl）",
    )
    p.add_argument("--horizon", type=int, default=5, help="输出文件名 h{N} 标签（默认 5）")
    p.add_argument(
        "--out-prefix",
        type=str,
        default="",
        help="输出前缀（默认 research/output/rolling_pool_schedule_h{horizon}）",
    )
    p.add_argument("--window", type=int, default=52)
    p.add_argument("--abs-mean-min", type=float, default=0.015)
    p.add_argument("--abs-icir-min", type=float, default=0.3)
    p.add_argument("--min-periods", type=int, default=52)
    p.add_argument("--k-max", type=int, default=50)
    p.add_argument("--turnover-frac", type=float, default=0.2)
    p.add_argument("--ic-corr-thr", type=float, default=0.7)
    p.add_argument("--cs-corr-thr", type=float, default=0.7)
    p.add_argument("--cooldown", type=int, default=1, help="出局冷却期数（0=关闭）")
    p.add_argument("--ddof", type=int, default=0, help="ICIR std ddof（默认 0）")
    p.add_argument(
        "--no-cs-corr",
        action="store_true",
        help="关闭第二道截面相关去重（仅 IC 序列相关）",
    )
    p.add_argument(
        "--cs-sample-step",
        type=int,
        default=20,
        help="截面相关：因子面板采样步长",
    )
    p.add_argument(
        "--cs-max-dates",
        type=int,
        default=13,
        help="截面相关：决策日前最多采样日数",
    )
    p.add_argument(
        "--smoke-only",
        action="store_true",
        help="只跑前 5 个决策日（冒烟）",
    )
    p.add_argument("--progress-every", type=int, default=25)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    t0 = time.perf_counter()

    print("=" * 60)
    print("[rolling_pool] load pure IC")
    print(f"  ckpt: {args.ckpt}")
    ic_dict = load_barra_pure_ic(args.ckpt)
    print(f"  n_factors={len(ic_dict)}")

    params = PoolParams(
        window=args.window,
        abs_mean_min=args.abs_mean_min,
        abs_icir_min=args.abs_icir_min,
        min_periods=args.min_periods,
        k_max=args.k_max,
        turnover_frac=args.turnover_frac,
        ic_corr_thr=args.ic_corr_thr,
        cs_corr_thr=args.cs_corr_thr,
        cooldown_periods=args.cooldown,
        ddof=args.ddof,
    )
    print(f"  params: {params.to_dict()}")

    if args.no_cs_corr:
        cs_provider = NullFactorCorrProvider()
        print("  cs_provider: NULL（第二道截面去重关闭）")
    else:
        cs_provider = CachedPanelCorrProvider(
            sample_step=args.cs_sample_step,
            max_sample_dates=args.cs_max_dates,
        )
        avail = cs_provider.available_names()
        overlap = len(avail & set(ic_dict))
        print(
            f"  cs_provider: {cs_provider.label} "
            f"cached={len(avail)} overlap_with_ic={overlap}"
        )
        if overlap < 10:
            print(
                "  [WARN] 缓存与 IC 宇宙重叠过少，第二道截面去重可能整段降级"
            )

    from research.rolling_pool.schedule import infer_rebalance_dates

    dates = infer_rebalance_dates(ic_dict, window=params.window)
    if args.smoke_only:
        dates = dates[:5]
        print(f"  smoke-only: {len(dates)} dates")
    else:
        print(f"  rebalance_dates: {len(dates)} "
              f"({dates[0].date()} → {dates[-1].date()})")

    print("[rolling_pool] build schedule …")
    schedule, meta = build_pool_schedule(
        ic_dict,
        dates,
        params=params,
        cs_provider=cs_provider,
        progress_every=args.progress_every,
    )

    out_prefix = args.out_prefix or str(
        OUTPUT_DIR / f"rolling_pool_schedule_h{args.horizon}"
    )
    if args.smoke_only:
        out_prefix = str(Path(out_prefix).with_name(
            Path(out_prefix).name + "_smoke"
        ))

    paths = write_schedule_outputs(
        schedule, meta, out_prefix=out_prefix, horizon=args.horizon,
    )

    dt = time.perf_counter() - t0
    print("=" * 60)
    print("[rolling_pool] done")
    print(f"  |U|={meta.get('union_size')}  "
          f"mean_n_pool={meta.get('mean_n_pool'):.2f}  "
          f"mean_turnover(ex_init)={meta.get('mean_turnover_ex_init'):.4f}")
    if meta.get("cs_corr_degraded"):
        print("  [WARN] CS 截面去重已降级 / 未应用:", meta.get("cs_skip_reasons"))
    else:
        print("  CS 截面去重: 已启用")
    print("  smoke_first5:")
    for r in meta.get("smoke_first5") or []:
        print(
            f"    {r['date']} n={r['n_pool']} fail={r['n_fail']} "
            f"trim={r['n_trim']} out={r['n_out']} turn={r['turnover']:.3f}"
        )
    print("  outputs:")
    for k, v in paths.items():
        print(f"    {k}: {v}")
    print(f"  elapsed: {dt:.1f}s")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
