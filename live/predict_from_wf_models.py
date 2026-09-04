"""按 fold schedule 加载全 WF 留存模型，对信号日截面出 Top-N（与回测严格同口径）。

与 ``live.daily_update`` 的区别 —— **同口径**：
  ``daily_update`` 在 warmup 窗上重算 Barra+残差，Barra bundle 指纹与全量回测不同
  → 残差化特征与回测不一致（实测 08-21 spearman≈0.09、Top100 重合 33%）。

  本脚本复用 ``run.py`` 全量数据加载 + ``strategies.ml.build_factor_dataset``：
  - 全量 prices/circ_mv/turnover → Barra cache HIT 回测 ``barra_bundle_*``；
  - 全量 rebalance_dates（含信号日）→ neut cache HIT 回测 ``factor_panel_neut_*``；
  - 直接从 ``dataset.factor_panel`` 取信号日截面 X（绕开 ``get_cross_section``
    对 forward_return 的 dropna —— 信号日标签未实现时它返回 None）；
  - 加载 manifest 中服务该信号日的 fold 模型（<= 信号日的最新 fit 日）predict；
  - strict 宇宙 mask + Top-N，与回测 ``mask_scores_for_backtest`` 同口径。

fold schedule（retrain_every=4，W-FRI 周频）：
  fit 日 = pred_offset ≡ 0 (mod 4) 的调仓日；中间 3 期复用上一个 fit 模型。
  信号日 08-28 的 pred_offset=18 → 复用 offset=16 的 fit 日 08-14 模型。

入口:
    python -m live.predict_from_wf_models --as-of-date 2026-08-28 \
        --model-dir results/xgb_h5_sizeind_w156_nob_wf_20260830 --top-n 100

同口径校验：把 --as-of-date 设成回测已出分的日期（如 2026-08-21），
对比 candidates 与回测 ml_factor_scores 该日 Top100 重合度（应 >90）。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TRAIN_MAX_WORKERS", "1")

from config.encoding_bootstrap import bootstrap_stdio_utf8, configure_loguru

bootstrap_stdio_utf8()
configure_loguru()


def _load_manifest(model_dir, prefer_model="xgb", prefer_window=156):
    import json
    mp = Path(model_dir) / "models" / "models_manifest.json"
    if not mp.exists():
        raise FileNotFoundError(f"未找到模型 manifest: {mp}")
    entries = json.loads(mp.read_text(encoding="utf-8"))
    cand = entries
    if prefer_model:
        cand = [e for e in cand if e.get("model") == prefer_model]
    if prefer_window:
        cand = [e for e in cand if int(e.get("window", -1)) == int(prefer_window)]
    if not cand:
        raise ValueError(f"manifest 无匹配模型 (prefer_model={prefer_model}, prefer_window={prefer_window})")
    return cand


def _model_for_as_of(entries, as_of):
    """返回服务 as_of 的 fold 模型 entry：<= as_of 的最新 fit 日。"""
    as_of = pd.Timestamp(as_of)
    pairs = [(pd.Timestamp(e["date"]), e) for e in entries if pd.Timestamp(e["date"]) <= as_of]
    if not pairs:
        raise ValueError(f"manifest 无 fit_date <= {as_of.date()}")
    pairs.sort(key=lambda x: x[0])
    return pairs[-1][1], pairs[-1][0]


def _build_dataset_same_caliber(panels, cfg):
    """复用 run.py 全量数据 + build_factor_dataset，命中回测 neut/Barra 缓存。"""
    from factors.barra_risk import get_barra_factors
    from strategies.ml import build_factor_dataset
    from run import _load_factor_config

    (prices, prices_raw, financial, volume, amount,
     open_, high, low, clean_ret, masks,
     market_prices, industry_map,
     margin, moneyflow, northbound, institution,
     total_mv, circ_mv, turnover_rate) = panels

    ind_map_arg = None
    if industry_map is not None:
        ind_map_arg = (industry_map["sw_l2"]
                       if isinstance(industry_map, pd.DataFrame) and "sw_l2" in industry_map.columns
                       else industry_map)

    horizon = int(cfg["model"]["horizon"])
    factor_whitelist, _ = _load_factor_config(str(ROOT / cfg["factors"]["dense_yaml"]), horizon)
    logger.info(f"dense 白名单: {len(factor_whitelist or [])} 个（期望 {cfg['factors']['dense_n']}）")

    logger.info("计算 Barra 风格因子（全量 → cache HIT 回测 bar bundle）...")
    barra = get_barra_factors(
        prices=prices, financial=financial, market_prices=market_prices,
        volume=volume, clean_ret=clean_ret, industry_map=ind_map_arg,
        prices_raw=prices_raw, circ_mv=circ_mv, total_mv=total_mv,
        turnover_rate=turnover_rate, amount=amount,
    )

    mcfg = cfg["model"]
    ncfg = cfg["neutralize"]
    fcfg = cfg["factors"]
    dataset = build_factor_dataset(
        prices, financial,
        prices_raw=prices_raw, volume=volume, amount=amount,
        open_=open_, high=high, low=low, clean_ret=clean_ret, masks=masks,
        market_prices=market_prices, industry_map=industry_map,
        margin=margin, moneyflow=moneyflow, northbound=None,
        institution=institution, circ_mv=circ_mv, total_mv=total_mv,
        hold_period=int(mcfg["hold_period"]),
        factor_whitelist=factor_whitelist,
        rebalance_freq=mcfg["rebalance_freq"],
        use_factor_cache=True, skip_factor_build=False, rebuild_factor_cache=False,
        feature_neutralize=bool(ncfg["feature_neutralize"]),
        neut_controls=str(ncfg["neut_controls"]),
        special_factors=fcfg["special_factors"],
        sparse_from_ic=str(ROOT / fcfg["sparse_from_ic"]),
        barra_factors=barra,
        fwd_return_winsor=bool(mcfg.get("fwd_return_winsor", True)),
        cs_rank_winsor=bool(mcfg.get("cs_rank_winsor", False)),
        label_mode=mcfg["label_mode"],
    )
    logger.info(
        f"dataset 就绪: feature_names={len(dataset.feature_names)} "
        f"rebalance_dates={len(dataset.rebalance_dates)} "
        f"末日={pd.Timestamp(dataset.rebalance_dates[-1]).date()}"
    )
    return dataset, prices, volume, masks


def _cross_section_X(dataset, as_of, feature_names):
    """从 dataset.factor_panel 取信号日截面 X，绕开 forward_return dropna。

    与 MLDataset.get_cross_section 同口径：reindex 到 feature_names、fillna(0)、
    保留有任意数据的行；但不要求 forward_return 非空（信号日标签未实现）。
    """
    dt = pd.Timestamp(as_of)
    rows = {}
    for name in feature_names:
        df = dataset.factor_panel.get(name)
        if df is not None and dt in df.index:
            rows[name] = df.loc[dt]
    if not rows:
        return None
    X_raw = pd.DataFrame(rows).reindex(columns=list(feature_names))
    has_data = X_raw.notna().any(axis=1)
    return X_raw.fillna(0).loc[has_data]


def summarize_candidates(top) -> dict:
    """候选表 ST / B 股 / 北交所 92 计数（供 CLI 日志与 UI 共用，不读盘）。

    - B 股：代码前缀 200 / 900
    - 北交所 92：代码前缀 92
    - ST：名称含 ``ST``（启发式；精确历史仍看 ``st_history``）
    """
    codes = top["code"].astype(str).str.replace(r"\.\w+$", "", regex=True).str.zfill(6)
    is_b = codes.str.startswith(("200", "900"))
    is_92 = codes.str.startswith("92")
    if "name" in top.columns:
        is_st = top["name"].astype(str).str.contains("ST", case=False, na=False)
        st_codes = codes.loc[is_st].tolist()
        n_st = int(is_st.sum())
    else:
        st_codes = []
        n_st = 0
    return {
        "n": int(len(top)),
        "n_b": int(is_b.sum()),
        "n_92": int(is_92.sum()),
        "n_st_name": n_st,
        "b_codes": codes.loc[is_b].tolist(),
        "st_codes": st_codes,
    }


def predict_candidates(*, as_of, model_dir, top_n=100, cfg_path=None,
                       prefer_model="xgb", prefer_window=156, output=None):
    """按 fold schedule 加载全 WF 留存模型，对信号日出 Top-N（与回测同口径）。

    CLI（``python -m live.predict_from_wf_models``）与 UI 共用此函数，
    **不训练**、不下单；需本地全市场数据 + ``models_manifest.json``。
    返回 ``(candidates_df, fit_date)``。
    """
    import joblib
    from models.wf.models import predict_model
    from live.daily_update import strict_universe_mask, output_topn
    from live.flagship_last_window import _load_yaml, _next_trading_day

    as_of = pd.Timestamp(as_of).normalize()
    model_dir = Path(model_dir)
    cfg = _load_yaml(Path(cfg_path) if cfg_path
                     else ROOT / "config" / "flagship_xgb_h5_sizeind_w156_nob.yaml")

    from run import _load_data
    logger.info(f"[same-caliber] 全量加载数据 as_of={as_of.date()}")
    panels = _load_data(skip_download=True, sample=0)

    dataset, prices, volume, masks = _build_dataset_same_caliber(panels, cfg)

    entries = _load_manifest(model_dir, prefer_model, prefer_window)
    entry, fit_date = _model_for_as_of(entries, as_of)
    feature_names = entry.get("feature_names") or dataset.feature_names
    logger.info(
        f"信号日={as_of.date()} 服务模型 fit_date={fit_date.date()} "
        f"feature_neutralize={entry.get('feature_neutralize')} "
        f"neut_controls={entry.get('neut_controls')} 特征={len(feature_names)}"
    )

    X = _cross_section_X(dataset, as_of, feature_names)
    if X is None or len(X) == 0:
        raise SystemExit(f"信号日 {as_of.date()} 无因子截面（数据未覆盖？）")
    logger.info(f"信号日截面 X: {X.shape}")

    X_np = X.reindex(columns=feature_names).to_numpy(dtype=np.float32, copy=False)
    X_np = np.where(np.isfinite(X_np), X_np, 0.0)
    model = joblib.load(entry["path"])
    pred = predict_model(model, X_np, entry["model"])
    scores = pd.Series(pred, index=X.index, dtype=np.float32)

    mask = strict_universe_mask(prices, volume, masks, as_of)
    csv_path = Path(output) if output else model_dir / f"candidates_{as_of.strftime('%Y%m%d')}.csv"
    from research.ic.universe import load_stock_names
    top = output_topn(
        scores, mask, top_n, "all", as_of, str(csv_path), stock_names=load_stock_names(),
        circ_mv=panels[17],
    )

    buy = _next_trading_day(as_of, prices.index)
    top = top.copy()
    top["code"] = top["code"].astype(str).str.zfill(6)
    top["signal_date"] = as_of.strftime("%Y-%m-%d")
    top["suggested_buy_date"] = pd.Timestamp(buy).strftime("%Y-%m-%d")
    top["fit_date"] = fit_date.strftime("%Y-%m-%d")
    preferred = ["signal_date", "suggested_buy_date", "fit_date", "rank",
                 "code", "name", "sw_l2", "circ_mv", "circ_mv_yi", "score"]
    cols = [c for c in preferred if c in top.columns] + [c for c in top.columns if c not in preferred]
    top = top[cols]
    top.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"已写出（含 fit_date）: {csv_path}  共 {len(top)} 行")

    flags = summarize_candidates(top)
    logger.info(
        f"Top{flags['n']} 中 B股={flags['n_b']}  北交所92={flags['n_92']}  "
        f"名称含ST={flags['n_st_name']}"
    )
    if flags["n_b"]:
        raise SystemExit(f"候选含 B 股: {flags['b_codes']}")

    logger.info(f"--- Top30 / {len(top)}（信号日 {as_of.date()}，fit {fit_date.date()}）---")
    for _, r in top.head(30).iterrows():
        yi = r.get("circ_mv_yi", float("nan"))
        yi_s = f"{float(yi):.1f}亿" if pd.notna(yi) else "NA"
        logger.info(f"  {int(r['rank']):>3}. {r['code']} {str(r.get('name') or ''):<10} "
                    f"{yi_s}  score={float(r['score']):.4f}")
    return top, fit_date


predict = predict_candidates  # 旧名，CLI / 调用方兼容


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="按 fold schedule 加载全 WF 留存模型，对信号日出 Top-N（与回测同口径）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python -m live.predict_from_wf_models --as-of-date 2026-08-28 \\\n"
            "      --model-dir results/xgb_h5_sizeind_w156_nob_wf_20260830 --top-n 100\n"
            "\n需先有全 WF --save-models 产物（results/<tag>/models/models_manifest.json）。"
        ),
    )
    p.add_argument("--as-of-date", required=True, help="信号日 YYYY-MM-DD")
    p.add_argument("--model-dir", required=True, help="全 WF --save-models 产物目录（results/<tag>/）")
    p.add_argument("--top-n", type=int, default=100)
    p.add_argument("--cfg-path", default=None, help="旗舰 YAML（默认 config/flagship_xgb_h5_sizeind_w156_nob.yaml）")
    p.add_argument("--prefer-model", default="xgb")
    p.add_argument("--prefer-window", type=int, default=156)
    p.add_argument("--output", default=None, help="输出 CSV（默认 <model-dir>/candidates_<date>.csv）")
    return p.parse_args(argv)


def main(argv=None):
    a = _parse_args(argv)
    predict_candidates(
        as_of=a.as_of_date, model_dir=a.model_dir, top_n=a.top_n,
        cfg_path=a.cfg_path, prefer_model=a.prefer_model,
        prefer_window=a.prefer_window, output=a.output,
    )


if __name__ == "__main__":
    main()
