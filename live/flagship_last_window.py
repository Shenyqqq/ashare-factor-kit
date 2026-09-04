"""旗舰最后一窗重训 + 增量日更出 Top-N（辅助人工选股）。

流程：
  1. 记录 circ_mv / close 末日，再 ``incremental_download``（lookback 不当 circ_mv start）
  2. 用下载前日期切片构建因子/neut（缓存 HIT），对最新调仓日训 **一个** xgb（156 期, val=0）
  3. ``live.daily_update --no-download`` 对数据末日出分，写 candidates_YYYYMMDD.csv

配置：``config/flagship_xgb_h5_sizeind_w156_nob.yaml``
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("TRAIN_MAX_WORKERS", "1")

from config.encoding_bootstrap import add_utf8_file_sink, bootstrap_stdio_utf8, configure_loguru

bootstrap_stdio_utf8()
configure_loguru()

FLAGSHIP_NAME = "xgb_h5_sizeind_w156_nob"
DEFAULT_CFG = ROOT / "config" / "flagship_xgb_h5_sizeind_w156_nob.yaml"
DEFAULT_OUT = ROOT / "results" / "xgb_h5_sizeind_w156_nob_20260817"

STOCK_PANEL_KEYS = (
    "prices", "prices_raw", "open_", "high", "low", "volume", "amount",
    "clean_ret", "margin", "moneyflow", "institution",
    "total_mv", "circ_mv", "turnover_rate",
)
DATE_ONLY_KEYS = ("market_prices",)


def _load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _slice_to(df, end, columns=None):
    """切日期（及可选列）以对齐下载前 neut/因子缓存指纹。"""
    if df is None or not isinstance(df, pd.DataFrame):
        return df
    idx = pd.DatetimeIndex(pd.to_datetime(df.index))
    df = df.copy()
    df.index = idx
    df = df.loc[df.index <= pd.Timestamp(end)]
    if columns is not None:
        cols = [str(c).zfill(6) for c in columns]
        have = [c for c in cols if c in df.columns.astype(str).str.zfill(6)]
        # 宽表列名可能未 zfill；按原列对齐
        colmap = {str(c).zfill(6): c for c in df.columns}
        keep = [colmap[c] for c in have if c in colmap]
        if keep:
            df = df.loc[:, keep]
    return df


def _slice_panels(panels: dict, end, columns=None) -> dict:
    end = pd.Timestamp(end)
    out = dict(panels)
    for k in STOCK_PANEL_KEYS:
        if k in out:
            out[k] = _slice_to(out[k], end, columns=columns)
    for k in DATE_ONLY_KEYS:
        if k in out:
            out[k] = _slice_to(out[k], end, columns=None)
    masks = out.get("masks")
    if masks:
        out["masks"] = {
            mk: _slice_to(mv, end, columns=columns) if isinstance(mv, pd.DataFrame) else mv
            for mk, mv in masks.items()
        }
    return out


def _log_universe(prices: pd.DataFrame, tag: str) -> None:
    from data.download import is_b_share_code

    cols = [str(c).zfill(6) for c in prices.columns]
    n_b = sum(is_b_share_code(c) for c in cols)
    n_92 = sum(c.startswith("92") for c in cols)
    n_8 = sum(c.startswith("8") and not c.startswith("92") for c in cols)
    logger.info(
        f"宇宙[{tag}]: n={len(cols)}  B股200/900={n_b}  北交所92={n_92}  8开头={n_8}"
    )
    if n_b:
        sample = [c for c in cols if is_b_share_code(c)][:8]
        logger.error(f"宇宙仍含 B 股: {sample}")
    if n_92 == 0:
        logger.warning("宇宙无 92xxxx，北交所可能未并入")


def _next_trading_day(as_of, prices_index) -> pd.Timestamp:
    as_of = pd.Timestamp(as_of).normalize()
    idx = pd.DatetimeIndex(pd.to_datetime(prices_index)).normalize().unique().sort_values()
    after = idx[idx > as_of]
    if len(after):
        return pd.Timestamp(after[0])
    return as_of + pd.offsets.BDay(1)


def _enrich_signal_buy(top: pd.DataFrame, as_of, buy_date) -> pd.DataFrame:
    top = top.copy()
    top["signal_date"] = pd.Timestamp(as_of).strftime("%Y-%m-%d")
    top["suggested_buy_date"] = pd.Timestamp(buy_date).strftime("%Y-%m-%d")
    preferred = [
        "signal_date", "suggested_buy_date", "as_of_date",
        "rank", "code", "name", "sw_l2", "circ_mv", "circ_mv_yi", "score",
    ]
    cols = [c for c in preferred if c in top.columns]
    extra = [c for c in top.columns if c not in cols]
    return top[cols + extra]


def train_last_window(
    *,
    panels: dict,
    train_end,
    train_columns,
    cfg: dict,
    out_dir: Path,
    tag: str,
):
    """对 train_end 切片上的最新 W-FRI 调仓日，用过去 156 期训一个 xgb 并 --save-models。"""
    from data.download import is_b_share_code
    from factors.barra_risk import get_barra_factors
    from models.trainer import WalkForwardTrainer
    from run import _load_factor_config
    from strategies.ml import build_factor_dataset

    sliced = _slice_panels(panels, train_end, columns=train_columns)
    prices = sliced["prices"]
    _log_universe(prices, "train_slice")
    b_cols = [c for c in prices.columns.astype(str).str.zfill(6) if is_b_share_code(c)]
    if b_cols:
        raise SystemExit(f"训练切片仍含 B 股: {b_cols[:10]}")

    model_cfg = cfg["model"]
    neut_cfg = cfg["neutralize"]
    fac_cfg = cfg["factors"]
    horizon = int(model_cfg["horizon"])
    factor_whitelist, _ = _load_factor_config(str(ROOT / fac_cfg["dense_yaml"]), horizon)
    if not factor_whitelist or len(factor_whitelist) != int(fac_cfg["dense_n"]):
        logger.warning(
            f"dense 白名单数量={len(factor_whitelist or [])}，期望 {fac_cfg['dense_n']}"
        )

    ind_map = sliced["industry_map"]
    ind_map_arg = None
    if ind_map is not None:
        ind_map_arg = (
            ind_map["sw_l2"]
            if isinstance(ind_map, pd.DataFrame) and "sw_l2" in ind_map.columns
            else ind_map
        )

    logger.info(
        "feature_neutralize=True neut_controls=size_industry: "
        "计算 Barra 后仅用 Size+PIT 行业残差化"
    )
    barra = get_barra_factors(
        prices=prices,
        financial=sliced["financial"],
        market_prices=sliced["market_prices"],
        volume=sliced["volume"],
        clean_ret=sliced["clean_ret"],
        industry_map=ind_map_arg,
        prices_raw=sliced["prices_raw"],
        circ_mv=sliced["circ_mv"],
        total_mv=sliced["total_mv"],
        turnover_rate=sliced["turnover_rate"],
        amount=sliced["amount"],
    )

    dataset = build_factor_dataset(
        prices, sliced["financial"],
        prices_raw=sliced["prices_raw"],
        volume=sliced["volume"],
        amount=sliced["amount"],
        open_=sliced["open_"],
        high=sliced["high"],
        low=sliced["low"],
        clean_ret=sliced["clean_ret"],
        masks=sliced["masks"],
        market_prices=sliced["market_prices"],
        industry_map=ind_map,
        margin=sliced["margin"],
        moneyflow=sliced["moneyflow"],
        northbound=None,
        institution=sliced["institution"],
        circ_mv=sliced["circ_mv"],
        total_mv=sliced["total_mv"],
        hold_period=int(model_cfg["hold_period"]),
        factor_whitelist=factor_whitelist,
        rebalance_freq=model_cfg["rebalance_freq"],
        use_factor_cache=True,
        skip_factor_build=False,
        rebuild_factor_cache=False,
        feature_neutralize=bool(neut_cfg["feature_neutralize"]),
        neut_controls=str(neut_cfg["neut_controls"]),
        special_factors=fac_cfg["special_factors"],
        sparse_from_ic=str(ROOT / fac_cfg["sparse_from_ic"]),
        barra_factors=barra,
        fwd_return_winsor=bool(model_cfg.get("fwd_return_winsor", True)),
        cs_rank_winsor=bool(model_cfg.get("cs_rank_winsor", False)),
        label_mode=model_cfg["label_mode"],
    )
    n_feat = len(dataset.feature_names)
    logger.info(f"特征数={n_feat}（dense+sparse）；neut_controls={neut_cfg['neut_controls']}")

    w = int(model_cfg["train_windows"][0])
    val = int(model_cfg["val_window"])
    dates = list(dataset.rebalance_dates)
    fr = dataset.forward_return
    valid_pred = []
    for d in dates:
        if d not in fr.index:
            continue
        n_y = int(pd.to_numeric(fr.loc[d], errors="coerce").notna().sum())
        if n_y >= 100:
            valid_pred.append(d)
    if not valid_pred:
        raise SystemExit("切片内无带标签的调仓日（forward_return 全空）")
    last = valid_pred[-1]
    idx = dates.index(last)
    need = w + val + 1
    if idx + 1 < need:
        raise SystemExit(f"调仓日不足: idx={idx} < {need - 1}")
    dataset.rebalance_dates = dates[idx + 1 - need: idx + 1]
    logger.info(
        f"最后一窗: 调仓日 {dataset.rebalance_dates[0].date()} → "
        f"{dataset.rebalance_dates[-1].date()} "
        f"(n={len(dataset.rebalance_dates)}, train={w}, val={val}, "
        f"pred标签非空={int(pd.to_numeric(fr.loc[last], errors='coerce').notna().sum())})"
    )

    trainer = WalkForwardTrainer(
        hold_period=int(model_cfg["hold_period"]),
        wf_selection=model_cfg.get("wf_selection", "ic_weighted"),
        label_mode=model_cfg["label_mode"],
        save_models=True,
        objective=model_cfg.get("objective", "regression"),
        tag=tag,
        barra_factors=barra,
        industry_map=ind_map_arg,
        device=model_cfg.get("device", "cpu"),
        model_types=["xgb"],
        rebalance_freq=model_cfg["rebalance_freq"],
        train_window_units=model_cfg["train_window_units"],
        feature_neutralize=bool(neut_cfg["feature_neutralize"]),
        neut_controls=str(neut_cfg["neut_controls"]),
        train_windows=list(model_cfg["train_windows"]),
        val_window=val,
        artifact_dir=out_dir,
        retrain_every=int(model_cfg["retrain_every"]),
        time_decay=float(model_cfg["time_decay"]),
    )
    trainer.fit_predict(dataset)
    manifest = out_dir / "models" / "models_manifest.json"
    if not manifest.exists():
        raise SystemExit(f"未写出 manifest: {manifest}")
    logger.info(f"已存最后一窗模型: {manifest}")
    return trainer


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="旗舰 xgb_h5_sizeind_w156_nob 最后一窗 + Top100")
    p.add_argument("--config", default=str(DEFAULT_CFG))
    p.add_argument("--as-of-date", default=None)
    p.add_argument(
        "--output-dir", default=None,
        help="覆盖 yaml output_dir；写新日期目录，避免覆盖旧旗舰落盘",
    )
    p.add_argument(
        "--train-end", default=None,
        help="训练切片末日（对齐已有 neut 缓存日历；默认=下载前 close 末日）",
    )
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--skip-train", action="store_true", help="已有 manifest 时只做出分")
    args = p.parse_args(argv)

    cfg = _load_yaml(Path(args.config))
    name = cfg.get("name", FLAGSHIP_NAME)
    out_dir = ROOT / str(args.output_dir or cfg.get("output_dir", DEFAULT_OUT))
    out_dir.mkdir(parents=True, exist_ok=True)
    add_utf8_file_sink(out_dir / "run.log")

    as_of = pd.Timestamp(args.as_of_date or cfg.get("as_of") or pd.Timestamp.today()).normalize()
    lookback = int(cfg.get("lookback_days", 14))
    top_n = int(cfg.get("top_n", 100))
    logger.info(
        f"旗舰={name} as_of={as_of.date()} out={out_dir} "
        f"lookback_days={lookback} TRAIN_MAX_WORKERS={os.environ.get('TRAIN_MAX_WORKERS')}"
    )
    logger.info(
        "参数: mode=xgb horizon=5 train=156 periods val=0 retrain_every=4 "
        "time_decay=0 cs_rank_winsor=False feature_neutralize size_industry "
        "bid-ask=10bp 无B股 保留北交所92"
    )

    from config.settings import RAW_DIR
    from live.daily_update import incremental_download, load_clean_panels, main as live_main

    close_path = RAW_DIR / "close_hfq.parquet"
    pre_max = None
    pre_cols = None
    if close_path.exists():
        pre_close = pd.read_parquet(close_path)
        pre_idx = pd.DatetimeIndex(pd.to_datetime(pre_close.index))
        pre_max = pre_idx.max()
        pre_cols = list(pre_close.columns.astype(str).str.zfill(6))
        logger.info(
            f"下载前 close_hfq 末日={pre_max.date()} "
            f"rows={len(pre_idx)} cols={len(pre_cols)}"
        )

    n0, d0, d1 = None, None, None
    from live.daily_update import _log_circ_mv_shape
    n0, d0, d1 = _log_circ_mv_shape("pipeline 前")

    if not args.no_download:
        incremental_download(as_of, lookback, sample=0)
        n1, a0, a1 = _log_circ_mv_shape("pipeline 后")
        if n0 and n1 is not None and n1 < n0:
            raise SystemExit(
                f"circ_mv 行数 {n0} → {n1}，历史被截断！禁止用 lookback 当 start"
            )
        logger.info(f"circ_mv 行数 {n0} → {n1}  区间 {a0} → {a1}")
    else:
        logger.info("[Step 1] --no-download")

    panels = load_clean_panels(sample=0)
    prices_full = panels["prices"]
    _log_universe(prices_full, "download_后")
    data_end = pd.Timestamp(prices_full.index.max())
    logger.info(f"数据末日={data_end.date()}（请求 as_of={as_of.date()}）")
    as_of_eff = min(as_of, data_end)

    train_end = (
        pd.Timestamp(args.train_end) if args.train_end
        else (pre_max if pre_max is not None else data_end)
    )
    # 切片保留下载前日历以命中 neut 缓存；标签窗口仍覆盖最后一窗
    logger.info(f"训练切片末日={pd.Timestamp(train_end).date()}（避免 neut 日历指纹 MISS）")

    if not args.skip_train:
        train_last_window(
            panels=panels,
            train_end=train_end,
            train_columns=pre_cols,
            cfg=cfg,
            out_dir=out_dir,
            tag=name,
        )
    else:
        logger.info("--skip-train: 使用已有 models_manifest")

    csv_name = f"candidates_{as_of_eff.strftime('%Y%m%d')}.csv"
    csv_path = out_dir / csv_name
    top = live_main(
        as_of=as_of_eff,
        lookback_days=lookback,
        model_dir=str(out_dir),
        top_n=top_n,
        cap_band="all",
        output=str(csv_path),
        factor_config=str(ROOT / cfg["factors"]["dense_yaml"]),
        horizon=int(cfg["model"]["horizon"]),
        feature_neutralize=True,
        no_download=True,
        sample=0,
        prefer_model="xgb",
        prefer_window=int(cfg["model"]["train_windows"][0]),
    )

    from data.download import is_b_share_code
    codes = top["code"].astype(str).str.zfill(6)
    n_b = int(codes.map(is_b_share_code).sum())
    n_92 = int(codes.str.startswith("92").sum())
    logger.info(f"Top{len(top)} 中 B股={n_b}  北交所92={n_92}")
    if n_b:
        raise SystemExit(f"候选含 B 股: {codes[codes.map(is_b_share_code)].tolist()}")

    buy = _next_trading_day(as_of_eff, prices_full.index)
    top = _enrich_signal_buy(top, as_of_eff, buy)
    top.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(
        f"信号日={as_of_eff.date()} 建议买入日={buy.date()} "
        f"(次日开盘，与 forward_return 一致) → {csv_path}"
    )
    logger.info(f"--- Top30 / {len(top)} ---")
    for _, r in top.head(30).iterrows():
        yi = r.get("circ_mv_yi", float("nan"))
        yi_s = f"{float(yi):.1f}亿" if pd.notna(yi) else "NA"
        logger.info(
            f"  {int(r['rank']):>3}. {r['code']} {str(r.get('name') or ''):<10} "
            f"{yi_s}  score={float(r['score']):.4f}"
        )


if __name__ == "__main__":
    main()
