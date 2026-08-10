"""live/daily_update.py — 实盘日更增量链路（辅助人工选股，非自动交易）

每日增量四步：
  1. 行情增量：复用 data.download.download_ohlcv / download_stock_value_em /
     download_shares / compute_market_cap，只拉最近 N 日 append 到 data/raw 的
     OHLCV / 市值 / 换手等 parquet，保留旧行，启动前 .bak。
  2. 因子面板 append：对模型用到的每个因子，在「热身窗」（最近 ~450 日）上重算，
     只取新增交易日行 concat 到 data/processed/factor_panels/factor_panel_*.parquet；
     指纹策略保留——append 失败回退全量重算并 warning。
  3. 当日中性化：对新截面做 WLS Barra+行业残差（复用 residualize_panel），
     只产出当日 neut 行，不整盘重算。
  4. 用已训模型出分：加载 results/<tag>/models/models_manifest.json 中最近一次
     重训模型，对当日截面 predict，输出 Top-N 候选清单（strict 宇宙 + 涨跌停/停牌 mask）。

入口:
    python -m live.daily_update --model-dir results/lgbm_h5_... --top-n 30

与现有研究口径一致：clean_ret、PIT、tradable strict、barra WLS（√市值）。
不破坏研究回测路径：增量是新入口，不替换现有全量 run.py。
已知限制见 docs/LIVE_DAILY.md。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import RAW_DIR, IC_MIN_LISTING_DAYS
from factors.factor_cache import (
    FACTOR_CACHE_DIR,
    _cache_paths,
    _load_meta,
    _load_panel,
    _save_panel,
)

DEFAULT_WARMUP_DAYS = 450
DEFAULT_LOOKBACK_DAYS = 7


def _backup_parquet(path: Path) -> None:
    """写前 .bak 备份（与 download_shares 习惯一致）。"""
    if not path.exists():
        return
    bak = path.with_suffix(path.suffix + ".bak")
    try:
        shutil.copy2(path, bak)
    except OSError as e:
        logger.warning(f".bak 备份失败（继续）: {path.name} -> {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 1: 行情增量下载
# ══════════════════════════════════════════════════════════════════════════════

def incremental_download(as_of, lookback_days, sample=0):
    """增量拉取最近 lookback_days 日行情/市值/换手，append 到 data/raw。

    复用现有下载函数（本身已是 per-stock date-append 增量）。
    """
    from data.download import download_ohlcv, get_stock_list

    as_of = pd.Timestamp(as_of)
    start = (as_of - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end = as_of.strftime("%Y-%m-%d")
    logger.info(f"[Step 1] 行情增量下载 {start} ~ {end}")

    stock_list = get_stock_list(include_delisted=True)
    codes = list(stock_list["code"].astype(str).str.zfill(6))
    if sample and sample > 0:
        codes = codes[:sample]
        logger.info(f"--sample {sample}: 仅前 {len(codes)} 只")

    for fname in ("close_hfq", "open_hfq", "high_hfq", "low_hfq",
                  "volume", "amount", "prices_raw"):
        _backup_parquet(RAW_DIR / f"{fname}.parquet")

    try:
        download_ohlcv(codes, start=start, end=end, adjust="hfq")
    except Exception as e:
        logger.error(f"OHLCV 增量下载失败: {e}")
        raise

    try:
        from data.download_stock_value_em import download_stock_value_em
        download_stock_value_em(codes, start=start, refresh_stale_days=5)
    except Exception as e:
        logger.warning(f"stock_value_em 增量下载失败（市值因子将退化）: {e}")

    try:
        from data.download_shares import download_shares
        download_shares(codes, start=start, refresh_stale_days=30)
    except Exception as e:
        logger.warning(f"download_shares 增量下载失败（换手率将退化）: {e}")

    try:
        from data.compute_market_cap import compute_market_cap
        _backup_parquet(RAW_DIR / "turnover_rate.parquet")
        compute_market_cap(
            shares_path=RAW_DIR / "share_change.parquet",
            prices_path=RAW_DIR / "prices_raw.parquet",
            volume_path=RAW_DIR / "volume.parquet",
            start=start, end=end,
            sample_codes=codes if sample else None,
        )
    except Exception as e:
        logger.warning(f"compute_market_cap（turnover_rate）失败: {e}")

    logger.info("[Step 1] 行情增量下载完成")


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: 加载 + 清洗 raw 面板（与 run.py._load_data 同口径，简化版）
# ══════════════════════════════════════════════════════════════════════════════

def _load_opt(fname):
    p = RAW_DIR / fname
    if p.exists():
        return pd.read_parquet(p)
    return None


def load_clean_panels(sample=0):
    """加载并清洗 raw 面板，返回与 factors.factor registry 同口径的 kwargs dict。

    与 run.py::_load_data 同口径：clean_ohlc_aligned / clean_ohlcv / clean_* /
    mask_post_delist / 退市股保留。北向停更不加载。
    sample>0 时截取前 N 只股票（冒烟，与 run.py --sample 同口径）。
    """
    from data.clean import (
        clean_prices, clean_financial, clean_ohlcv, clean_ohlc_aligned,
        clean_volume, clean_amount, clean_aux_panel, clean_market_cap,
        mask_post_delist, validate_amount_units,
    )

    logger.info("[Step 2a] 加载 + 清洗 raw 面板")
    _close_raw = pd.read_parquet(RAW_DIR / "prices_hfq.parquet")
    _open_raw = _load_opt("open_hfq.parquet")
    _high_raw = _load_opt("high_hfq.parquet")
    _low_raw = _load_opt("low_hfq.parquet")
    prices, open_, high, low = clean_ohlc_aligned(_close_raw, _open_raw, _high_raw, _low_raw)

    prices_raw_path = RAW_DIR / "prices_raw.parquet"
    prices_raw = (
        clean_prices(pd.read_parquet(prices_raw_path), label="prices_raw")
        if prices_raw_path.exists() else None
    )
    fin_path = RAW_DIR / "financial_indicators.parquet"
    financial = clean_financial(pd.read_parquet(fin_path)) if fin_path.exists() else None

    _vol_raw = _load_opt("volume.parquet")
    volume = clean_volume(_vol_raw, name="volume") if _vol_raw is not None else None
    _amt_raw = _load_opt("amount.parquet")
    amount = clean_amount(_amt_raw, name="amount") if _amt_raw is not None else None
    if amount is not None and volume is not None:
        try:
            validate_amount_units(amount, volume, prices)
        except Exception as e:
            logger.warning(f"amount 量纲校验失败（继续）: {e}")

    try:
        from research.ic.universe import load_delist_dates
        _delist = load_delist_dates()
    except Exception:
        _delist = None
    if _delist:
        prices = mask_post_delist(prices, _delist)
        if prices_raw is not None:
            prices_raw = mask_post_delist(prices_raw, _delist)
        open_ = mask_post_delist(open_, _delist)
        high = mask_post_delist(high, _delist)
        low = mask_post_delist(low, _delist)
        if volume is not None:
            volume = mask_post_delist(volume, _delist)
        if amount is not None:
            amount = mask_post_delist(amount, _delist)

    _margin_raw = _load_opt("margin_balance.parquet")
    margin = clean_aux_panel(_margin_raw, name="margin") if _margin_raw is not None else None
    _moneyflow_raw = _load_opt("moneyflow_large.parquet")
    moneyflow = clean_aux_panel(_moneyflow_raw, name="moneyflow") if _moneyflow_raw is not None else None
    institution = _load_opt("institution_holding.parquet")

    from data.mv_panels import load_mv_raw
    _total_mv_raw = load_mv_raw("total_mv")
    total_mv = clean_market_cap(_total_mv_raw, name="total_mv") if _total_mv_raw is not None else None
    _circ_mv_raw = load_mv_raw("circ_mv")
    circ_mv = clean_market_cap(_circ_mv_raw, name="circ_mv") if _circ_mv_raw is not None else None
    _turnover_raw = _load_opt("turnover_rate.parquet")
    turnover_rate = clean_aux_panel(_turnover_raw, name="turnover_rate") if _turnover_raw is not None else None

    market_prices = _load_opt("csi_all.parquet")
    if market_prices is None:
        market_prices = _load_opt("csi300.parquet")
    industry_map = _load_opt("industry_map.parquet")

    logger.info("[Step 2b] 涨跌停清洗（生成 clean_ret）")
    clean_ret, masks = clean_ohlcv(prices, open_, high, low)

    panels = dict(
        prices=prices, financial=financial, prices_raw=prices_raw,
        volume=volume, amount=amount, open_=open_, high=high, low=low,
        clean_ret=clean_ret, masks=masks, market_prices=market_prices,
        industry_map=industry_map, margin=margin, moneyflow=moneyflow,
        northbound=None, institution=institution,
        circ_mv=circ_mv, total_mv=total_mv, turnover_rate=turnover_rate,
    )
    if sample and sample > 0:
        panels = _apply_sample(panels, sample)
    return panels


def _apply_sample(panels, sample):
    """截取前 N 只股票（与 run.py --sample 同口径，冒烟用）。"""
    prices = panels["prices"]
    codes = list(prices.columns[:sample])
    def _col(df):
        if df is None:
            return None
        if isinstance(df, pd.DataFrame) and df.columns.isin(codes).any():
            keep = [c for c in codes if c in df.columns]
            return df.loc[:, keep]
        return df
    out = dict(panels)
    for k in ("prices", "prices_raw", "open_", "high", "low", "volume",
              "amount", "clean_ret", "margin", "moneyflow", "institution",
              "total_mv", "circ_mv", "turnover_rate"):
        if k in out:
            out[k] = _col(out[k])
    if out.get("masks"):
        out["masks"] = {mk: _col(mv) for mk, mv in out["masks"].items()}
    logger.info(f"--sample {sample}: 截取 {len(codes)} 只股票")
    return out


def slice_warmup(panels, as_of, warmup_days):
    """把所有面板切到 [as_of - warmup_days, as_of] 热身窗，供增量因子计算。

    截面列（股票）保持全量——winsorize/zscore 是 per-date 的，需保留当日全截面。
    长表（financial/industry）不按日期截。
    """
    prices = panels["prices"]
    end = pd.Timestamp(as_of)
    if end not in prices.index:
        le = prices.index[prices.index <= end]
        if len(le) == 0:
            raise ValueError(f"as_of={as_of} 早于数据首日 {prices.index[0]}")
        end = le[-1]
    start = end - pd.Timedelta(days=warmup_days)
    idx = prices.index[(prices.index >= start) & (prices.index <= end)]
    if len(idx) < 20:
        raise ValueError(
            f"热身窗太短：{len(idx)} 日（warmup_days={warmup_days}, as_of={as_of}）"
        )

    def _sl(df):
        if df is None:
            return None
        if isinstance(df, pd.DataFrame) and df.index.equals(prices.index):
            return df.loc[idx]
        return df

    out = {}
    for k, v in panels.items():
        if k == "masks" and v is not None:
            out[k] = {mk: _sl(mv) for mk, mv in v.items()}
        else:
            out[k] = _sl(v)
    out["_warmup_idx"] = idx
    out["_as_of"] = end
    logger.info(f"热身窗: {idx[0].date()} ~ {idx[-1].date()} ({len(idx)} 日)")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: 因子面板 append（热身窗重算 → 取新行 concat）
# ══════════════════════════════════════════════════════════════════════════════

def _compute_factors_warmup(names, warmup_panels):
    """在热身窗输入上重算指定因子（绕过磁盘缓存，避免签名失配）。

    直接调 _iter_factor_registry_raw（无缓存计算核），与 compute_single_factor
    同口径。返回 {name: panel}，panel 已 reindex 到热身窗 prices.index。
    """
    from factors.factor import _iter_factor_registry_raw, _filter_none_emit

    kwargs = dict(
        prices=warmup_panels["prices"], financial=warmup_panels["financial"],
        prices_raw=warmup_panels["prices_raw"], volume=warmup_panels["volume"],
        amount=warmup_panels["amount"], open_=warmup_panels["open_"],
        high=warmup_panels["high"], low=warmup_panels["low"],
        clean_ret=warmup_panels["clean_ret"], masks=warmup_panels["masks"],
        market_prices=warmup_panels["market_prices"],
        industry_map=warmup_panels["industry_map"],
        margin=warmup_panels["margin"], moneyflow=warmup_panels["moneyflow"],
        northbound=warmup_panels["northbound"],
        institution=warmup_panels["institution"],
        circ_mv=warmup_panels["circ_mv"], total_mv=warmup_panels["total_mv"],
        factor_names=set(names), include_regime=False,
    )
    out = {}
    for n, panel in _filter_none_emit(_iter_factor_registry_raw(**kwargs)):
        out[n] = panel
    missing = [n for n in names if n not in out]
    if missing:
        logger.warning(
            f"热身窗重算缺 {len(missing)} 个因子: "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
        )
    return out


def append_factor_panels(names, warmup_panels, as_of):
    """对每个因子：加载已有 factor_panel_<hash>.parquet，热身窗重算，取新行 concat。

    - 优先 append：已有面板 + 新行（index > 已有末日）concat，dedup keep='last'，sort
    - 失败回退：全量重算（get_factor_registry 全输入）并 warning
    - 写前 .bak；写后更新 .meta.json（标记 live_appended）
    - 返回 {name: full_panel}（含历史 + 新行），供中性化 / 打分用

    注：append 后 .meta.json 的输入指纹仍是旧值，下次全量研究管线会因签名
    失配自动重算——这是预期行为（append 是捷径，不替代全量校准）。
    """
    as_of = pd.Timestamp(as_of)
    fresh = _compute_factors_warmup(names, warmup_panels)
    result = {}

    for name in names:
        if name not in fresh:
            continue
        fresh_panel = fresh[name]
        pq_path, meta_path = _cache_paths(name)

        if pq_path.exists():
            try:
                existing = _load_panel(pq_path)
                last_existing = existing.index[-1]
                new_rows = fresh_panel.loc[fresh_panel.index > last_existing]
                if new_rows.empty:
                    logger.info(f"[append] {name}: 无新行（已有至 {last_existing.date()}）")
                    result[name] = existing
                    continue
                # 列对齐：union(existing, new)，缺失填 NaN
                all_cols = existing.columns.union(new_rows.columns)
                existing_a = existing.reindex(columns=all_cols)
                new_a = new_rows.reindex(columns=all_cols)
                merged = pd.concat([existing_a, new_a])
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                merged = merged.astype(np.float32, copy=False)
                _backup_parquet(pq_path)
                _save_panel_live(name, merged, meta_path, as_of, appended=len(new_rows))
                result[name] = merged
                logger.info(
                    f"[append] {name}: +{len(new_rows)} 行 "
                    f"({last_existing.date()} -> {merged.index[-1].date()})"
                )
                continue
            except Exception as e:
                logger.warning(
                    f"[append] {name}: append 失败 ({e})，回退全量重算"
                )
        # 无已有面板 或 append 失败 → 用热身窗结果作为该因子面板（冷启动 / 兜底）
        _backup_parquet(pq_path)
        _save_panel_live(name, fresh_panel, meta_path, as_of, appended=len(fresh_panel))
        result[name] = fresh_panel
        logger.info(
            f"[append] {name}: 冷启动/兜底，写热身窗 {len(fresh_panel)} 行"
        )
    return result


def _save_panel_live(name, panel, meta_path, as_of, appended=0):
    """落盘因子面板 + 更新 .meta.json（标记 live_appended，保留旧指纹）。"""
    from factors.factor_cache import _cache_paths, _save_panel
    pq_path, _ = _cache_paths(name)
    # 复用 _save_panel 写 parquet（原子 tmp+rename），但 meta 用旧指纹 + live 标记
    old_meta = _load_meta(meta_path)
    sig = old_meta.get("signature") if old_meta else None
    # 直接调 _save_panel 会用 sig 重建 meta；若 sig=None 则写空指纹 meta
    if sig is not None:
        _save_panel(name, panel, sig)
    else:
        # 冷启动：写一个最小 meta（无指纹，下次研究管线会重算）
        panel.astype(np.float32, copy=False).to_parquet(pq_path)
    # 追加 live 标记
    meta = _load_meta(meta_path) or {}
    meta["live_appended"] = True
    meta["live_last_as_of"] = pd.Timestamp(as_of).isoformat()
    meta["live_last_appended_rows"] = int(appended)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# Step 4: 当日中性化（WLS Barra+行业残差，只产当日 neut 行）
# ══════════════════════════════════════════════════════════════════════════════

def compute_barra_warmup(warmup_panels):
    """在热身窗上算 Barra 9 风格因子 + WLS 权重面板（√市值）。

    复用 factors.barra_risk.get_barra_factors / barra_regression_weights。
    返回 (barra_factors: dict, neut_weights: DataFrame|None)。
    """
    from factors.barra_risk import get_barra_factors, barra_regression_weights

    prices = warmup_panels["prices"]
    ind_map_arg = None
    if warmup_panels["industry_map"] is not None:
        im = warmup_panels["industry_map"]
        if isinstance(im, pd.DataFrame) and "sw_l2" in im.columns:
            ind_map_arg = im["sw_l2"]
        else:
            ind_map_arg = im
    barra = get_barra_factors(
        prices=prices,
        financial=warmup_panels["financial"],
        market_prices=warmup_panels["market_prices"],
        volume=warmup_panels["volume"],
        clean_ret=warmup_panels["clean_ret"],
        industry_map=ind_map_arg,
        prices_raw=warmup_panels["prices_raw"],
        circ_mv=warmup_panels["circ_mv"],
        total_mv=warmup_panels["total_mv"],
        turnover_rate=warmup_panels["turnover_rate"],
        amount=warmup_panels["amount"],
    )
    weights = barra_regression_weights(
        prices, circ_mv=warmup_panels["circ_mv"], total_mv=warmup_panels["total_mv"],
    )
    logger.info(f"[Step 4] Barra 因子就绪: {len(barra)} 个；WLS 权重={'有' if weights is not None else '无(等权OLS)'}")
    return barra, weights


def neutralize_as_of(factor_panels, barra, weights, industry_map, as_of, universe=None):
    """对 as_of 截面做 WLS Barra+行业残差，返回 {name: neut_row_series}。

    - 复用 models.wf.labels.residualize_panel（与训练同口径）
    - rebalance_dates=[as_of]：只算当日一行
    - should_skip_neutralize 的因子（Barra_*/special）原样返回 raw 行
    - 输出每项是 pd.Series(index=universe)（当日截面残差 + re-zscore）
    - universe: 统一股票宇宙（prices.columns）；各因子面板列集可能不同
      （如融资因子仅覆盖两融标的），统一 reindex 到 universe 避免空 DataFrame。
    """
    from models.wf.labels import residualize_panel
    from factors.special_factors import should_skip_neutralize

    as_of = pd.Timestamp(as_of)
    if universe is None:
        # 取所有面板列的并集
        cols = []
        for p in factor_panels.values():
            if p is not None:
                cols.extend(list(p.columns))
        universe = pd.Index(sorted(set(cols)))
    universe = universe.astype(str).str.zfill(6)

    ind_series = None
    if industry_map is not None:
        if isinstance(industry_map, pd.DataFrame) and "sw_l2" in industry_map.columns:
            ind_series = industry_map["sw_l2"]
        else:
            ind_series = industry_map

    def _zscore_keep_nan(series):
        """截面 zscore 但保留 NaN（不 dropna），与 cross_sectional_zscore 在
        非全 NaN 时等价；全 NaN 时返回全 NaN（index 不丢）。"""
        s = series.astype(np.float64)
        finite = np.isfinite(s.values)
        if finite.sum() < 2:
            return pd.Series(np.nan, index=s.index, dtype=np.float32)
        mu = s.values[finite].mean()
        sd = s.values[finite].std()
        if sd == 0 or not np.isfinite(sd):
            return pd.Series(np.nan, index=s.index, dtype=np.float32)
        z = np.clip((s.values - mu) / sd, -3.0, 3.0)
        z[~finite] = np.nan
        return pd.Series(z, index=s.index, dtype=np.float32)

    out = {}
    n_skip = 0
    n_neut = 0
    n_allnan = 0
    for name, panel in factor_panels.items():
        if should_skip_neutralize(name):
            if as_of in panel.index:
                out[name] = panel.loc[as_of].reindex(universe)
            else:
                out[name] = pd.Series(np.nan, index=universe, dtype=np.float32)
            n_skip += 1
            continue
        resid = residualize_panel(
            panel, barra, ind_series, pd.DatetimeIndex([as_of]),
            weight_panel=weights,
        )
        if as_of in resid.index:
            row = resid.loc[as_of]
        else:
            row = pd.Series(np.nan, index=panel.columns, dtype=np.float32)
        row = row.reindex(universe)
        if not np.isfinite(row.values).any():
            # 残差全 NaN（min_stocks 未达 / 数据未更新）→ 退化用 raw 因子行
            if as_of in panel.index:
                row = panel.loc[as_of].reindex(universe)
            n_allnan += 1
        out[name] = _zscore_keep_nan(row)
        n_neut += 1
    logger.info(
        f"[Step 4] 当日中性化: {n_neut} 残差 + {n_skip} 豁免"
        + (f"（{n_allnan} 个残差全 NaN 退化为 raw）" if n_allnan else "")
    )
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Step 5: 加载已训模型 + predict + strict 宇宙 + Top-N 输出
# ══════════════════════════════════════════════════════════════════════════════

def load_latest_models(model_dir, prefer_model=None, prefer_window=None):
    """读 results/<tag>/models/models_manifest.json，返回最近重训日的模型 entries。

    - 按 date 降序取最近重训日（同一日可能有多个 window×model）
    - prefer_model/prefer_window: 优先选指定模型/窗口（None=全部）
    - 返回 list[dict]（每条含 path/model/window/date/feature_names/...）
    - 需要 --save-models 产物；无 manifest 报错并提示
    """
    model_dir = Path(model_dir)
    manifest_path = model_dir / "models" / "models_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"未找到模型 manifest: {manifest_path}\n"
            f"请用 --save-models 重训，或指定正确的 --model-dir（指向 results/<tag>/）"
        )
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not entries:
        raise ValueError(f"manifest 为空: {manifest_path}")
    # 过滤
    cand = entries
    if prefer_model:
        cand = [e for e in cand if e.get("model") == prefer_model]
    if prefer_window:
        cand = [e for e in cand if int(e.get("window", -1)) == int(prefer_window)]
    if not cand:
        raise ValueError(
            f"manifest 无匹配模型 (prefer_model={prefer_model}, "
            f"prefer_window={prefer_window})"
        )
    # 最近重训日
    latest_date = max(e["date"] for e in cand)
    batch = [e for e in cand if e["date"] == latest_date]
    logger.info(
        f"[Step 5] 加载模型: {len(batch)} 个 fold（重训日 {latest_date}，"
        f"models={[e['model'] + '_w' + str(e['window']) for e in batch]}）"
    )
    return batch


def build_feature_matrix(feature_names, neut_rows, raw_panels, as_of, universe=None):
    """组装当日特征矩阵 X (n_stocks × n_features)。

    - feature_names: 模型训练时的特征列序（来自 manifest）
    - neut_rows: {name: Series} 当日中性化后的截面（对非豁免因子）
    - raw_panels: {name: DataFrame} 全量因子面板（对豁免因子取当日 raw 行）
    - as_of: 当日
    - universe: 统一股票宇宙；None 时取 neut_rows/raw_panels 列并集
    返回 (X: DataFrame[index=stock, columns=feature_names], stock_index)

    保证 X 非空（至少 universe 行数），NaN 保留（树模型原生支持 NaN 分裂）。
    """
    as_of = pd.Timestamp(as_of)
    if universe is None:
        cols = []
        for s in neut_rows.values():
            if s is not None:
                cols.extend(list(s.index))
        for p in raw_panels.values():
            if p is not None:
                cols.extend(list(p.columns))
        universe = pd.Index(sorted(set(cols))) if cols else pd.Index(["__empty__"])
    universe = universe.astype(str).str.zfill(6)

    cols = {}
    for fn in feature_names:
        if fn in neut_rows:
            cols[fn] = neut_rows[fn].reindex(universe)
        elif fn in raw_panels:
            panel = raw_panels[fn]
            if as_of in panel.index:
                cols[fn] = panel.loc[as_of].reindex(universe)
            else:
                cols[fn] = pd.Series(np.nan, index=universe, dtype=np.float32)
        else:
            logger.warning(f"特征 {fn} 缺面板/neut 行 → 整列 NaN")
            cols[fn] = pd.Series(np.nan, index=universe, dtype=np.float32)
    X = pd.DataFrame(cols, index=universe)
    X.index = X.index.astype(str).str.zfill(6)
    return X


def predict_scores(model_entries, X):
    """对当日截面 X 出分，多 fold IC 加权 Z-score 平均（与训练 ensemble 同口径）。

    返回 pd.Series(index=stock, values=score)。单模型直接 predict。
    """
    import joblib
    from models.wf.models import predict_model

    # 对齐 X 列序到 manifest feature_names
    feature_names = model_entries[0].get("feature_names")
    if feature_names:
        X = X.reindex(columns=feature_names)
    X_np = X.to_numpy(dtype=np.float32, copy=False)
    # inf -> NaN -> 0（与训练 get_cross_section 同口径 fillna(0)）
    X_np = np.where(np.isfinite(X_np), X_np, 0.0)

    fold_scores = []
    fold_ics = []
    for e in model_entries:
        model = joblib.load(e["path"])
        pred = predict_model(model, X_np, e["model"])
        s = pd.Series(pred, index=X.index, dtype=np.float32)
        fold_scores.append(s)
        # IC 权重：训练时记录的 fold IC（若有）；无则等权
        ic = e.get("fold_ic") or e.get("pred_ic")
        fold_ics.append(ic if ic is not None and np.isfinite(ic) else 0.0)

    if len(fold_scores) == 1:
        return fold_scores[0]
    # IC 加权 Z-score 平均（与 ensemble.combine_model_scores 同语义）
    zscores = []
    weights = []
    for s, ic in zip(fold_scores, fold_ics):
        std = s.std()
        if std == 0 or not np.isfinite(std):
            continue
        zscores.append((s - s.mean()) / std)
        weights.append(max(ic, 0.0))  # 负 IC 折半/丢弃与训练一致：这里取 max(0)
    if not zscores:
        return fold_scores[0]
    w = np.array(weights, dtype=np.float64)
    if w.sum() == 0:
        w = np.ones(len(zscores))
    w = w / w.sum()
    zarr = np.zeros_like(zscores[0].values, dtype=np.float64)
    for zi, wi in zip(zscores, w):
        zarr += wi * zi.values
    return pd.Series(zarr, index=X.index, dtype=np.float32)


def strict_universe_mask(prices, volume, masks, as_of):
    """当日 strict 可买宇宙（与 mask_scores_for_backtest strict 同口径）。

    信号日剔涨跌停 + ST + 停牌 + 次新 + 退市。返回 bool Series(index=stock)。
    """
    from research.ic.universe import (
        build_ic_tradability_mask, load_stock_names, load_is_st_current,
        load_listing_dates, load_delist_dates, load_st_history,
    )

    as_of = pd.Timestamp(as_of)
    # 取单日面板（build_ic_tradability_mask 需 DataFrame）
    if as_of in prices.index:
        p1 = prices.loc[[as_of]]
        v1 = volume.loc[[as_of]] if volume is not None and as_of in volume.index else None
    else:
        le = prices.index[prices.index <= as_of]
        if len(le) == 0:
            return pd.Series(False, index=prices.columns)
        d = le[-1]
        p1 = prices.loc[[d]]
        v1 = volume.loc[[d]] if volume is not None and d in volume.index else None
    m1 = None
    if masks is not None:
        m1 = {k: (v.loc[[as_of]] if as_of in v.index else v.iloc[0:0])
              for k, v in masks.items() if v is not None}

    sn = load_stock_names()
    ist = load_is_st_current()
    ld = load_listing_dates()
    dd = load_delist_dates()
    sth = load_st_history()
    tradable = build_ic_tradability_mask(
        p1, volume=v1, masks=m1,
        stock_names=sn, listing_dates=ld, delist_dates=dd,
        is_st_current=ist, st_history=sth,
        exclude_limit_on_signal=True,  # strict
        min_listing_days=IC_MIN_LISTING_DAYS,
    )
    row = tradable.iloc[0] if len(tradable) else pd.Series(False, index=prices.columns)
    return row.reindex(prices.columns).fillna(False)


def output_topn(scores, mask, top_n, cap_band, as_of, output_path, stock_names=None):
    """输出 Top-N 候选清单（strict 宇宙内按得分降序）。

    - scores: pd.Series(stock -> score)
    - mask: bool Series(stock -> 可买)
    - cap_band: 可选市值带过滤（'all' 关闭）；当前仅 'all'，其它值 warning
    - output_path: .csv 或 .md（按扩展名决定格式）
    返回 Top-N DataFrame。
    """
    as_of = pd.Timestamp(as_of)
    s = scores.reindex(mask.index).where(mask)
    s = s.dropna()
    s = s.sort_values(ascending=False)
    top = s.head(top_n).reset_index()
    top.columns = ["code", "score"]
    if stock_names is not None:
        top = top.merge(
            stock_names.rename("name").reset_index(), on="code", how="left",
        )
    top.insert(0, "as_of_date", as_of.strftime("%Y-%m-%d"))
    top.insert(len(top.columns), "rank", range(1, len(top) + 1))
    if cap_band and cap_band != "all":
        logger.warning(
            f"cap_band={cap_band} 在 live 暂未实现市值带过滤（用 'all'）；"
            f"可在训练时用 --cap-band 限定训练池以间接约束"
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".md":
        _write_md(top, out, as_of)
    else:
        top.to_csv(out, index=False, encoding="utf-8-sig")
    logger.info(f"[Step 5] Top-{len(top)} 候选清单 -> {out}")
    return top


def _write_md(top, out, as_of):
    """写 Markdown 候选清单。"""
    lines = [
        f"# 实盘候选清单 {as_of.strftime('%Y-%m-%d')}",
        "",
        f"共 {len(top)} 只（strict 可买宇宙，按模型得分降序）。",
        "",
        "| rank | code | name | score |",
        "|------|------|------|-------|",
    ]
    for _, r in top.iterrows():
        name = r.get("name", "")
        name = "" if pd.isna(name) else name
        lines.append(f"| {int(r['rank'])} | {r['code']} | {name} | {r['score']:.4f} |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# 特征名解析 + 主编排
# ══════════════════════════════════════════════════════════════════════════════

def resolve_feature_names(model_entries, factor_config, horizon=None):
    """确定要计算哪些因子。

    优先级：manifest.feature_names > factor_config YAML/JSON > Barra_* 补集 > 全量。
    返回 (feature_names: list, source: str)
    """
    fn = model_entries[0].get("feature_names")
    if fn:
        return list(fn), "manifest"
    # lazy rolling-pool：用并集元数据兜底
    union = model_entries[0].get("feature_names_union_metadata")
    if union:
        return list(union), "manifest_union"
    if factor_config:
        wl = _load_factor_whitelist(factor_config, horizon)
        if wl:
            return wl, "factor_config"
    return None, "all"


def _load_factor_whitelist(config_path, horizon=None):
    """从 YAML/JSON 读取因子白名单（与 run.py::_load_factor_config 同口径，简化）。

    YAML: { h5: {factors: [...]} }（按 horizon 选 key，无 horizon 则取第一个）
    JSON: { horizon: 5, factors: [...] }
    """
    p = Path(config_path)
    if not p.exists():
        logger.warning(f"因子配置文件不存在: {config_path}")
        return None
    import json
    if p.suffix in (".yaml", ".yml"):
        import yaml
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            return None
        if horizon is not None:
            key = f"h{horizon}"
            if key in cfg:
                return cfg[key].get("factors")
        # 无 horizon：取第一个含 factors 的 section
        for v in cfg.values():
            if isinstance(v, dict) and v.get("factors"):
                return v["factors"]
        return None
    if p.suffix == ".json":
        cfg = json.loads(p.read_text(encoding="utf-8"))
        return cfg.get("factors")
    logger.warning(f"不支持的配置格式: {p.suffix}")
    return None


def main(as_of=None, lookback_days=DEFAULT_LOOKBACK_DAYS,
         model_dir=None, top_n=30, cap_band="all", output=None,
         factor_config=None, horizon=None, warmup_days=DEFAULT_WARMUP_DAYS,
         feature_neutralize=True, no_download=False, sample=0,
         prefer_model=None, prefer_window=None):
    """实盘日更增量主流程。"""
    as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.today().normalize()
    logger.info(f"=== live.daily_update as_of={as_of.date()} ===")

    if model_dir is None:
        raise ValueError("必须指定 --model-dir（指向 results/<tag>/）")

    # Step 1: 行情增量下载
    if not no_download:
        incremental_download(as_of, lookback_days, sample=sample)
    else:
        logger.info("[Step 1] --no-download: 跳过下载")

    # Step 2: 加载 + 清洗 raw 面板
    panels = load_clean_panels(sample=sample)

    # Step 5a: 加载模型 manifest（先于因子计算，以确定 feature_names）
    model_entries = load_latest_models(
        model_dir, prefer_model=prefer_model, prefer_window=prefer_window,
    )
    feature_names, fn_source = resolve_feature_names(model_entries, factor_config, horizon)
    if feature_names is None:
        # 全量：枚举所有可计算因子名
        from factors.factor import get_factor_names
        feature_names = get_factor_names(
            prices=panels["prices"], financial=panels["financial"],
            prices_raw=panels["prices_raw"], volume=panels["volume"],
            amount=panels["amount"], open_=panels["open_"],
            high=panels["high"], low=panels["low"],
            clean_ret=panels["clean_ret"], masks=panels["masks"],
            market_prices=panels["market_prices"],
            industry_map=panels["industry_map"],
            circ_mv=panels["circ_mv"], total_mv=panels["total_mv"],
        )
        fn_source = "all"
    logger.info(f"特征集: {len(feature_names)} 个（来源={fn_source}）")

    # Step 2b: 切热身窗（as_of 对齐到数据中 <= as_of 的最近交易日）
    warmup = slice_warmup(panels, as_of, warmup_days)
    as_of = warmup["_as_of"]  # 用对齐后的实际交易日作为当日
    logger.info(f"当日（对齐后）= {as_of.date()}")

    # Step 3: 因子面板 append（热身窗重算 + concat）
    factor_panels = append_factor_panels(feature_names, warmup, as_of)

    # Step 4: 当日中性化（若训练用了 feature_neutralize）
    neut_rows = None
    universe = panels["prices"].columns.astype(str).str.zfill(6)
    if feature_neutralize:
        barra, weights = compute_barra_warmup(warmup)
        neut_rows = neutralize_as_of(
            factor_panels, barra, weights, panels["industry_map"], as_of,
            universe=universe,
        )

    # Step 5b: 组装特征矩阵 + predict
    X = build_feature_matrix(feature_names, neut_rows or {}, factor_panels, as_of, universe=universe)
    scores = predict_scores(model_entries, X)

    # Step 5c: strict 宇宙 + Top-N 输出
    mask = strict_universe_mask(
        panels["prices"], panels["volume"], panels["masks"], as_of,
    )
    if output is None:
        out_dir = Path(model_dir)
        output = out_dir / f"candidates_{as_of.strftime('%Y%m%d')}.csv"
    from research.ic.universe import load_stock_names
    sn = load_stock_names()
    top = output_topn(scores, mask, top_n, cap_band, as_of, output, stock_names=sn)

    # 控制台打印 Top-N
    logger.info(f"--- Top-{len(top)} 候选（{as_of.date()}）---")
    for _, r in top.iterrows():
        name = r.get("name", "")
        name = "" if pd.isna(name) else name
        logger.info(f"  {int(r['rank']):>3}. {r['code']} {name:<8} score={r['score']:.4f}")
    return top


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="实盘日更增量链路：行情增量 → 因子 append → 当日中性化 → 模型出分 → Top-N",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python -m live.daily_update --model-dir results/lgbm_h5_w6-12 --top-n 30\n"
            "  python -m live.daily_update --as-of-date 2026-08-08 --no-download \\\n"
            "      --model-dir results/lgbm_h5_w6-12 --output candidates.md\n"
            "\n"
            "需先有 --save-models 训练产物（results/<tag>/models/models_manifest.json）。\n"
            "详见 docs/LIVE_DAILY.md"
        ),
    )
    p.add_argument("--as-of-date", default=None, help="当日（YYYY-MM-DD，默认今天）")
    p.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS,
                   help=f"行情增量回看日历日数（默认 {DEFAULT_LOOKBACK_DAYS}）")
    p.add_argument("--model-dir", required=True,
                   help="模型目录（指向 results/<tag>/，需含 models/models_manifest.json）")
    p.add_argument("--top-n", type=int, default=30, help="Top-N 候选数（默认 30）")
    p.add_argument("--cap-band", default="all", help="市值带（当前仅 'all'）")
    p.add_argument("--output", default=None, help="输出路径 .csv 或 .md（默认 <model-dir>/candidates_<date>.csv）")
    p.add_argument("--factor-config", default=None, help="因子白名单 YAML（manifest 无 feature_names 时兜底）")
    p.add_argument("--horizon", type=int, default=None, help="持仓期（仅提示，不参与计算）")
    p.add_argument("--warmup-days", type=int, default=DEFAULT_WARMUP_DAYS,
                   help=f"热身窗日历日数（默认 {DEFAULT_WARMUP_DAYS}，覆盖 Barra 252d EWM）")
    p.add_argument("--no-feature-neutralize", dest="feature_neutralize",
                   action="store_false", default=True,
                   help="训练未用 --feature-neutralize 时加此开关（用 raw 因子出分）")
    p.add_argument("--no-download", action="store_true", help="跳过行情增量下载")
    p.add_argument("--sample", type=int, default=0, help="仅前 N 只股票（冒烟）")
    p.add_argument("--prefer-model", default=None, help="多模型 ensemble 时优先选某模型（如 lgbm）")
    p.add_argument("--prefer-window", type=int, default=None, help="优先选某训练窗口（如 6）")
    return p.parse_args(argv)


def _cli():
    try:
        from config.encoding_bootstrap import configure_loguru
        configure_loguru()
    except Exception:
        pass
    args = _parse_args()
    main(
        as_of=args.as_of_date,
        lookback_days=args.lookback_days,
        model_dir=args.model_dir,
        top_n=args.top_n,
        cap_band=args.cap_band,
        output=args.output,
        factor_config=args.factor_config,
        horizon=args.horizon,
        warmup_days=args.warmup_days,
        feature_neutralize=args.feature_neutralize,
        no_download=args.no_download,
        sample=args.sample,
        prefer_model=args.prefer_model,
        prefer_window=args.prefer_window,
    )


if __name__ == "__main__":
    _cli()
