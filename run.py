"""
run.py  —  一键运行完整流程

日志文件：run.py 会向 --output-dir/run.log 写入 UTF-8（loguru encoding=utf-8）。
勿用 PowerShell Tee-Object / ``>`` 另存日志——默认 UTF-16 LE，且常把 UTF-8 管道误读成 GBK。

可选控制台环境（仅终端显示）:
    chcp 65001
    $env:LOGURU_COLORIZE='0'
    $env:PYTHONIOENCODING='utf-8'
    $env:PYTHONUTF8='1'
    $env:PYTHONUNBUFFERED='1'
    python -u run.py ...

run.py 启动时会自动 bootstrap UTF-8（config/encoding_bootstrap.py）。

用法（最短日常）:
    python run.py --skip-download --mode lgbm --horizon 5 \\
      --factor-config config/factor_configs_h5_sizeind_20260815.yaml \\
      --neut-controls size_industry --train-windows 104 --train-window-units periods --val-window 0
    python run.py --sample 100
    python run.py --help-advanced   # 全部参数（含高级/deprecated）

默认已含: feature-neutralize、bid-ask(settings=10bp)、research tradable、label=cs_rank。
--neut-controls 代码默认仍是 barra；日常/旗舰请显式 size_industry。
旗舰见 config/flagship_xgb_h5_sizeind_w156_nob.yaml（xgb / 156 期）。
详见 docs/操作手册.md。
"""
import argparse
import os
import sys
from pathlib import Path

from config.encoding_bootstrap import (
    add_utf8_file_sink,
    bootstrap_stdio_utf8,
    configure_loguru,
)

bootstrap_stdio_utf8()

from loguru import logger

configure_loguru()

import pandas as pd

from config.settings import (
    RAW_DIR,
    BACKTEST_START, BACKTEST_END,
    FACTOR_WEIGHTS,
    RISK_FREE_RATE,
    N_STOCKS,
    RETRAIN_EVERY,
)

PRICES_PATH     = RAW_DIR / "prices_hfq.parquet"
PRICES_RAW_PATH = RAW_DIR / "prices_raw.parquet"
FIN_PATH        = RAW_DIR / "financial_indicators.parquet"

ML_MODES    = {"lgbm", "xgb", "cat", "ridge", "ensemble", "rf", "mlp"}
ALL_MODES   = {"linear", "industry", "dynamic"} | ML_MODES
VALID_MODELS = {"lgbm", "xgb", "cat", "ridge", "rf", "mlp"}



def _resolve_dynamic_lookback(dynamic_lookback: int | None, rebalance_freq: str) -> int:
    from models.trainer import months_to_rebalance_periods
    if dynamic_lookback is not None:
        return dynamic_lookback
    return months_to_rebalance_periods(6, rebalance_freq)

from models.trainer import LABEL_MODE_DEFAULT, MODEL_TYPES, TIME_DECAY  # 分行业模式默认模型列表

INDEX_MAP = {"沪深300": "000300", "创业板指": "399006"}


def _load_indices() -> dict:
    """加载沪深300和创业板指收盘价，优先读本地缓存，否则从AKShare拉取。

    增量更新：缓存末日 < BACKTEST_END 时用 ak.index_zh_a_hist 从 last+1 拉到
    BACKTEST_END，concat + drop_duplicates 后写回 parquet（写前 .bak）。
    失败 warning 不 raise，保留旧缓存。
    """
    import akshare as ak
    import shutil

    result = {}
    for name, code in INDEX_MAP.items():
        cache = RAW_DIR / f"index_{code}.parquet"
        if cache.exists():
            s = pd.read_parquet(cache).squeeze()
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            last = pd.Timestamp(s.index.max())
            end_ts = pd.Timestamp(BACKTEST_END)
            if last >= end_ts:
                result[name] = s.rename(name) if s.name is None else s
                continue
            # 增量：从 last+1 拉到 BACKTEST_END
            start_inc = (last + pd.Timedelta(days=1)).strftime("%Y%m%d")
            end_inc = end_ts.strftime("%Y%m%d")
            try:
                df = ak.index_zh_a_hist(
                    symbol=code, period="daily",
                    start_date=start_inc, end_date=end_inc,
                )
                df["日期"] = pd.to_datetime(df["日期"])
                inc = df.set_index("日期")["收盘"].rename(name)
                merged = pd.concat([s, inc])
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                merged.name = name
                try:
                    shutil.copy2(cache, cache.with_suffix(cache.suffix + ".bak"))
                except OSError:
                    pass
                merged.to_frame().to_parquet(cache)
                logger.info(f"指数 {name}({code}) 增量更新 {start_inc}~{end_inc}: +{len(inc)} 行 → 末日 {merged.index.max().date()}")
                result[name] = merged
            except Exception as e:
                logger.warning(f"指数 {name}({code}) 增量更新失败: {e}，沿用旧缓存（末日 {last.date()}）")
                result[name] = s.rename(name) if s.name is None else s
        else:
            try:
                df = ak.index_zh_a_hist(
                    symbol=code, period="daily",
                    start_date=BACKTEST_START.replace("-", ""),
                    end_date=BACKTEST_END.replace("-", ""),
                )
                df["日期"] = pd.to_datetime(df["日期"])
                s = df.set_index("日期")["收盘"].rename(name)
                s.to_frame().to_parquet(cache)
                logger.info(f"指数 {name}({code}) 下载完成，shape={s.shape}")
            except Exception as e:
                logger.warning(f"指数 {name} 下载失败: {e}，跳过")
                continue
            result[name] = s
    return result


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
    from data.clean import (
        clean_prices, clean_financial, clean_ohlcv, clean_ohlc_aligned,
        clean_volume, clean_amount, clean_aux_panel,
        clean_market_cap, mask_post_delist, validate_amount_units,
    )

    prices_raw_path_exists = PRICES_RAW_PATH.exists()
    _close_raw = pd.read_parquet(PRICES_PATH)

    def _load_opt(fname):
        p = RAW_DIR / fname
        if p.exists():
            logger.debug(f"加载 {fname}")
            return pd.read_parquet(p)
        return None

    _open_raw = _load_opt("open_hfq.parquet")
    _high_raw = _load_opt("high_hfq.parquet")
    _low_raw = _load_opt("low_hfq.parquet")
    # OHLC 联合清洗：刺针以 close 判定，四价一并置 NaN；默认不 ffill
    prices, open_, high, low = clean_ohlc_aligned(
        _close_raw, _open_raw, _high_raw, _low_raw,
    )
    prices_raw = (
        clean_prices(pd.read_parquet(PRICES_RAW_PATH), label="prices_raw")
        if prices_raw_path_exists else None
    )
    financial = (clean_financial(pd.read_parquet(FIN_PATH))
                  if FIN_PATH.exists() else None)

    # P0-3: volume/amount 异常值清洗（负值置 NaN、inf 置 NaN、突增告警、0 保留）
    _vol_raw = _load_opt("volume.parquet")
    volume   = clean_volume(_vol_raw, name="volume") if _vol_raw is not None else None
    _amt_raw = _load_opt("amount.parquet")
    amount   = clean_amount(_amt_raw, name="amount") if _amt_raw is not None else None
    if amount is not None and volume is not None:
        validate_amount_units(amount, volume, prices)

    # 退市后行情置 NaN（与 TradeRules.is_delisted 同口径）
    try:
        from research.ic.universe import load_delist_dates
        _delist = load_delist_dates()
    except Exception:
        _delist = None
    if _delist:
        n_d = len(_delist)
        prices = mask_post_delist(prices, _delist)
        prices_raw = mask_post_delist(prices_raw, _delist)
        open_ = mask_post_delist(open_, _delist)
        high = mask_post_delist(high, _delist)
        low = mask_post_delist(low, _delist)
        volume = mask_post_delist(volume, _delist)
        amount = mask_post_delist(amount, _delist)
        logger.info(f"退市后行情已置 NaN: {n_d} 只股票")

    # P1-5: 资金流辅助面板清洗（inf 置 NaN、突增告警、保留负值）
    # 北向已下线：仍可读 parquet 归档，但不进默认因子/白名单；此处不再加载以省内存
    _margin_raw     = _load_opt("margin_balance.parquet")
    margin          = clean_aux_panel(_margin_raw, name="margin") if _margin_raw is not None else None
    # moneyflow 已弃用：akshare 东财大单资金流数据不足（全市场限流、单票历史短），
    # 因子不可用。强制 None 跳过大单净流入/残差因子计算。北向/资金流缺口见操作手册 §9.2。
    if _load_opt("moneyflow_large.parquet") is not None:
        logger.warning("moneyflow_large 已弃用（akshare 资金流数据不足），跳过加载；大单净流入/残差因子不计算。")
    moneyflow       = None
    northbound      = None  # 北向下线：默认不算、不进白名单生成路径（parquet 保留）
    institution = _load_opt("institution_holding.parquet")
    # 日频市值：东财 stock_value_em 主路径；缺则回退自算 *_computed
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
    if market_prices is None:
        market_prices = _load_opt("index_000300.parquet")
    industry_map  = _load_opt("industry_map.parquet")

    # 涨跌停清洗：生成 clean_ret（屏蔽涨跌停日的日收益率）和 masks
    # 所有量价因子必须使用 clean_ret 而非 prices.pct_change()
    logger.info("Step 2b: 涨跌停清洗（生成 clean_ret）")
    clean_ret, masks = clean_ohlcv(prices, open_, high, low)

    # 与 filter_universe 同口径：已落盘宽表仍可能含 B 股 / 8 开头，截面标准化前去掉
    from data.download import drop_excluded_universe_columns

    prices = drop_excluded_universe_columns(prices, name="prices")
    prices_raw = drop_excluded_universe_columns(prices_raw)
    open_ = drop_excluded_universe_columns(open_)
    high = drop_excluded_universe_columns(high)
    low = drop_excluded_universe_columns(low)
    volume = drop_excluded_universe_columns(volume)
    amount = drop_excluded_universe_columns(amount)
    clean_ret = drop_excluded_universe_columns(clean_ret)
    margin = drop_excluded_universe_columns(margin)
    moneyflow = drop_excluded_universe_columns(moneyflow)
    institution = drop_excluded_universe_columns(institution)
    total_mv = drop_excluded_universe_columns(total_mv)
    circ_mv = drop_excluded_universe_columns(circ_mv)
    turnover_rate = drop_excluded_universe_columns(turnover_rate)
    if masks:
        masks = {
            k: (drop_excluded_universe_columns(v) if isinstance(v, pd.DataFrame) else v)
            for k, v in masks.items()
        }

    # --sample N：即使 --skip-download 也截取前 N 只股票（冒烟 / 内存友好）
    if sample and sample > 0:
        codes = list(prices.columns[:sample])
        def _col(df):
            if df is None:
                return None
            if isinstance(df, pd.DataFrame) and df.columns.isin(codes).any():
                keep = [c for c in codes if c in df.columns]
                return df.loc[:, keep]
            return df
        prices = _col(prices)
        prices_raw = _col(prices_raw)
        open_ = _col(open_)
        high = _col(high)
        low = _col(low)
        volume = _col(volume)
        amount = _col(amount)
        clean_ret = _col(clean_ret)
        margin = _col(margin)
        moneyflow = _col(moneyflow)
        institution = _col(institution)
        total_mv = _col(total_mv)
        circ_mv = _col(circ_mv)
        turnover_rate = _col(turnover_rate)
        if masks:
            masks = {
                k: (_col(v) if isinstance(v, pd.DataFrame) else v)
                for k, v in masks.items()
            }
        if financial is not None and "code" in getattr(financial, "columns", []):
            financial = financial[financial["code"].isin(codes)]
        logger.info(f"--sample={sample}: 截取 {len(codes)} 只股票（skip-download 兼容）")

    return (prices, prices_raw, financial, volume, amount,
            open_, high, low, clean_ret, masks,
            market_prices, industry_map,
            margin, moneyflow, northbound, institution,
            total_mv, circ_mv, turnover_rate)


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


def _load_factor_config(config_path: str, horizon: int) -> tuple[list | None, list | None]:
    """
    从 YAML 或 JSON 读取指定 horizon 的因子白名单（双轨制）。
    YAML 结构：{ h5: {factors: [...], factors_orth: [...]} }
    JSON 结构：{ horizon: 5, factors: [...], factors_orth: [...] }
    返回 (factors, factors_orth)，任一为 None 表示该轨道不过滤。
      - factors       → ML 轨道（pre-GS 完整 pure-IC 集，~65）
      - factors_orth  → dynamic 轨道（Gram-Schmidt 正交集，≤30；未跑 GS 时 None，回退用 factors）
    """
    import json
    p = Path(config_path)
    if not p.exists():
        logger.warning(f"因子配置文件不存在: {config_path}，使用全部因子")
        return None, None
    if p.suffix in (".yaml", ".yml"):
        import yaml
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
        key = f"h{horizon}"
        if key not in cfg:
            logger.warning(f"YAML 中无 {key} 配置，使用全部因子")
            return None, None
        return cfg[key].get("factors"), cfg[key].get("factors_orth")
    elif p.suffix == ".json":
        cfg = json.loads(p.read_text(encoding="utf-8"))
        return cfg.get("factors"), cfg.get("factors_orth")
    logger.warning(f"不支持的配置格式: {p.suffix}，使用全部因子")
    return None, None


def _resolve_output_dir(output_dir: str | Path | None, tag: str) -> Path:
    """实验产物目录，默认 results/<tag>/。"""
    out = Path(output_dir) if output_dir else Path("results") / tag
    out.mkdir(parents=True, exist_ok=True)
    return out


def main(mode="linear", skip_download=False, sample=0,
         show_report=False, horizon=20,
         hold_period: int | None = None,
         factor_config: str = None, show_holdings: bool = False,
         train_windows: list = None, train_window_units: str = "months",
         val_window: int | None = None,
         models: list = None,
         blend_dynamic: bool = False, output_dir: str = None,
         backtest_freq: str = None,
         use_factor_cache: bool = True, skip_factor_build: bool = False,
         rebuild_factor_cache: bool = False,
         dynamic_lookback: int = None,
         backtest_engine: str = "v2",
         trainer_engine: str = "v2",
         wf_selection: str = "ic_weighted",
         label_mode: str = LABEL_MODE_DEFAULT,
         ensemble_method: str = "zscore",
       save_models: bool = False,
       objective: str = "regression",
       device: str = "cpu",
       tune: bool = False,
      tune_trials: int = 15,
      triple_barrier_params: dict | None = None,
      multi_horizon: list | None = None,
      mh_weights: str = "equal",
      regime_conditional: bool = False,
      turnover_limit: float = 1.0,
      rank_change_threshold: float = 0.0,
      bid_ask_spread_bps: float | None = None,
      feature_neutralize: bool = False,
      neut_controls: str = "barra",
      barra_features: bool = False,
      special_factors: str | list | None = None,
      event_overlay: bool = False,
      sparse_from_ic: str | None = None,
      regime_cs: bool = False,
      cap_band: str = "all",
      mcap_min_yi: float | None = None,
      mcap_max_yi: float | None = None,
      ridge_drop_regime: bool = False,
      include_regime: bool = False,
      position_regime: bool = False,
      force_exposure: float | None = None,
      fwd_return_winsor: bool = True,
      cs_rank_winsor: bool = False,
      top_n: int | None = None,
      portfolio_opt: str = "ew",
      max_weight: float | None = None,
      cov_lookback: int = 60,
      risk_aversion: float = 1.0,
      two_stage: bool = False,
      stage2_pool_frac: float = 0.2,
      stage2_lookback: int | None = None,
      save_stage1_cache: bool = False,
      stage1_cache: str | None = None,
      enable_shap: bool = False,
      shap_top: int = 20,
      shap_max_samples: int = 500,
      shap_max_dates: int = 12,
      rolling_pool_schedule: str | None = None,
      rolling_pool_lazy: bool | None = None,
      rolling_pool_max_cached: int = 160,
      rolling_pool_strict: bool = True,
      long_weight_top: float | None = None,
      long_weight_ratio: float = 0.25,
      long_weight_curve: str = "smooth",
      rank_weight_mid: float = 1.0,
      softlong_floor_slope: float = 0.25,
      retrain_every: int = RETRAIN_EVERY,
      time_decay: float = TIME_DECAY,
      tradable_limit_mode: str | None = None,
      exclude_limit_on_signal: bool | None = None,
      apply_exec_mask: bool | None = None,
      bt_score_universe: str = "strict"):

    if horizon < 3:
        logger.warning(
            f"horizon={horizon} 在A股T+1制度下不具备实盘参考价值，"
            "结果仅供研究"
        )

    rebalance_freq = _horizon_to_rebalance_freq(horizon)
    bt_freq = backtest_freq or rebalance_freq
    # --hold-period / --label-horizon: 只改标签窗口与回测出场，不改调仓日历
    # （--horizon 5 仍为 W-FRI；勿把 --horizon 改成 3 否则会变成 3D）。
    label_hold = int(hold_period) if hold_period is not None else int(horizon)
    win_tag = ("_w" + "-".join(str(w) for w in train_windows)) if train_windows else ""
    units_tag = "_p" if train_window_units == "periods" else ""
    bt_tag = f"_bt{bt_freq.replace('-', '')}" if backtest_freq else ""
    mdl_tag = ("_m" + "-".join(sorted(models))) if models else ""
    dyn_tag = "_blend" if blend_dynamic else ""
    dyn_lb_tag = f"_lb{dynamic_lookback}" if dynamic_lookback is not None else ""
    mh_tag = ("_mh" + "-".join(str(h) for h in multi_horizon)) if multi_horizon else ""
    mh_w_tag = f"_{mh_weights}" if multi_horizon else ""
    reg_tag = "_regime" if regime_conditional else ""
    to_tag = (f"_to{turnover_limit}" if turnover_limit < 1.0 else "") + (
        f"_rc{rank_change_threshold}" if rank_change_threshold > 0.0 else ""
    )
    cap_tag = f"_{cap_band}" if cap_band and cap_band != "all" else ""
    mcap_tag = ""
    if mcap_min_yi is not None or mcap_max_yi is not None:
        _lo = int(mcap_min_yi) if mcap_min_yi is not None else 0
        _hi = int(mcap_max_yi) if mcap_max_yi is not None else 0
        mcap_tag = f"_mcap{_lo}_{_hi}"
    barra_feat_tag = "_barrafeat" if barra_features else ""
    from factors.special_factors import resolve_special_factors
    sf_req = resolve_special_factors(
        special_factors,
        event_overlay=event_overlay,
        sparse_from_ic=sparse_from_ic,
    )
    # 下游只传已解析规格，避免重复 DeprecationWarning；保留因子名子集
    if not sf_req:
        sf_pass = None
    elif sf_req.names is not None:
        sf_pass = sorted(sf_req.all_factor_names())
    else:
        sf_pass = list(sf_req.packs)
    special_tag = sf_req.tag_suffix()
    posreg_tag = "_posreg" if position_regime else ""
    _popt = (portfolio_opt or "ew").strip().lower()
    opt_tag = f"_opt{_popt}" if _popt and _popt != "ew" else ""
    # --stage1-cache implies two-stage (before tag so suffix is correct).
    if stage1_cache:
        two_stage = True
    twostage_tag = (
        f"_twostage{int(round(stage2_pool_frac * 100))}" if two_stage else ""
    )
    # 滚动定池可辨识片段：避免与同参数固定池实验共用 results/<tag>/ 目录
    rp_tag = ""
    if rolling_pool_schedule:
        from research.rolling_pool.schedule_load import schedule_tag
        rp_tag = schedule_tag(rolling_pool_schedule)
        if rolling_pool_lazy is False:
            rp_tag += "-eager"
        if not rolling_pool_strict:
            rp_tag += "-lax"
    lw_tag = ""
    if long_weight_top is not None and float(long_weight_top) > 0:
        # e.g. _lw40s025 → top40% + smooth + bottom_weight=0.25
        _curve_ch = "s" if long_weight_curve == "smooth" else "t"
        lw_tag = (
            f"_lw{int(round(float(long_weight_top) * 100))}"
            f"{_curve_ch}{int(round(float(long_weight_ratio) * 100)):03d}"
        )
    tailw_tag = ""
    if float(rank_weight_mid) < 1.0 - 1e-12:
        tailw_tag = f"_tailw{int(round(float(rank_weight_mid) * 100)):02d}"
    softlong_tag = (
        f"_sl{int(round(float(softlong_floor_slope) * 100))}"
        if label_mode == "cs_rank_softlong" else ""
    )
    rt_tag = f"_rt{int(retrain_every)}" if int(retrain_every) > 1 else ""
    hp_tag = f"_hp{label_hold}" if hold_period is not None else ""
    tag = (
        f"{mode}_h{horizon}{win_tag}{units_tag}{bt_tag}{mdl_tag}{dyn_tag}{dyn_lb_tag}"
        f"{mh_tag}{mh_w_tag}{reg_tag}{to_tag}{cap_tag}{mcap_tag}{barra_feat_tag}"
        f"{special_tag}{posreg_tag}{opt_tag}{twostage_tag}{rp_tag}"
        f"{lw_tag}{tailw_tag}{softlong_tag}{rt_tag}{hp_tag}"
    )
    out_dir = _resolve_output_dir(output_dir, tag)
    add_utf8_file_sink(out_dir / "run.log")
    logger.info(
        f"模式={mode}, 持仓期={label_hold}日, ML调仓={rebalance_freq}, "
        f"回测调仓={bt_freq}, 训练窗单位={train_window_units}, 输出={out_dir}"
    )
    if hold_period is not None:
        logger.info(
            f"hold_period={label_hold}: 标签/回测出场=close[t+{label_hold}]/"
            f"open[t+1]（调仓日历仍 {rebalance_freq}，由 --horizon {horizon} 推断）"
        )
    if multi_horizon:
        logger.info(
            f"Multi-Horizon: horizons={multi_horizon}, weights={mh_weights}"
        )
    if regime_conditional:
        logger.info("Regime-Conditional: 启用按 regime 过滤训练样本")
    if ridge_drop_regime:
        logger.warning(
            "--ridge-drop-regime 已退役（市场/HMM 不再注入 ML X）；忽略"
        )
    if sf_req and mode == "dynamic":
        logger.warning(
            f"--mode dynamic 忽略 special-factors={list(sf_req.packs)}："
            "动态加权（ICIR）轨道禁止 special/sparse 注入；请用 ridge/ensemble 注入"
        )
        sf_req = resolve_special_factors(None)
        sf_pass = None
        special_tag = ""
    elif sf_req:
        logger.info(
            f"special_factors={list(sf_req.packs)}：白名单之后 post-merge 注入"
            f"（tag{special_tag}；feature_neutralize 时按 pack 豁免残差化；"
            f"IC 筛选路径不自动纳入；dynamic 轨道禁止）"
        )
    if regime_cs:
        logger.warning(
            "--regime-cs 已退役（轮动_* 不再注入 ML X）；忽略。"
            "仓位控制请用 --position-regime"
        )
    if position_regime:
        logger.info(
            f"position_regime=True：回测按市场体制缩放总敞口"
            f"（force_exposure={force_exposure}）"
        )
    if stage1_cache:
        logger.info(
            f"stage1_cache={stage1_cache}：跳过 S1，用缓存 universe 跑 S2 "
            f"（pool_frac={stage2_pool_frac:.0%}；二级因子按当前 factor-config 现算）"
        )
    if two_stage:
        logger.info(
            f"two_stage=True：S1 全市场得分 → top {stage2_pool_frac:.0%} 池内 "
            f"winsor→cs_zscore + rolling ridge（lookback={stage2_lookback}）；"
            f"Top20%≈Q5，池 EW 可对照单段 Q5；回测建议 --top-n"
        )
    if turnover_limit < 1.0 or rank_change_threshold > 0.0:
        logger.info(
            f"Turnover 控制: turnover_limit={turnover_limit}, "
            f"rank_change_threshold={rank_change_threshold}"
        )

    (prices, prices_raw, financial, volume, amount,
     open_, high, low, clean_ret, masks,
     market_prices, industry_map,
     margin, moneyflow, northbound, institution,
     total_mv, circ_mv, turnover_rate) = _load_data(skip_download, sample)

    # 因子白名单（双轨制，从 YAML/JSON 读取，None=使用全部因子）
    #   factor_whitelist       → ML 轨道（pre-GS 完整 pure-IC 集）
    #   factor_whitelist_orth  → dynamic 轨道（GS 正交集；未跑 GS 时 None，回退用 ML 集）
    if factor_config:
        factor_whitelist, factor_whitelist_orth = _load_factor_config(factor_config, horizon)
    else:
        factor_whitelist, factor_whitelist_orth = None, None
    if factor_whitelist:
        logger.info(f"ML 因子白名单: {len(factor_whitelist)} 个因子 → {factor_whitelist}")
    if factor_whitelist_orth:
        logger.info(f"dynamic 正交白名单: {len(factor_whitelist_orth)} 个因子 → {factor_whitelist_orth}")
    if rolling_pool_schedule:
        _lazy = rolling_pool_lazy if rolling_pool_lazy is not None else True
        logger.info(
            f"rolling_pool_schedule={rolling_pool_schedule}："
            f"lazy={_lazy}, max_cached={rolling_pool_max_cached}, "
            f"strict={rolling_pool_strict}, tag_suffix={rp_tag}；"
            "每期只用当日 pool_t（禁止窗内并集 / 一次性 materialize |U|）"
            "（优先于 --factor-config 白名单）"
        )
        if not rolling_pool_strict:
            logger.warning(
                "--no-rolling-pool-strict（仅 debug）：schedule 因子缺面板不再 "
                "fail-fast，缺失列会整列 NaN→fillna(0) 变常数特征"
            )
        if not _lazy:
            logger.warning(
                "--no-rolling-pool-lazy：全 U materialize + 按日 mask 为过时路径；"
                "正确语义是每期 train/val/pred 共用当日 pool_t（请用默认 lazy）"
            )

    # ── cap-band universe mask（小盘/小中盘/中盘策略）──────────────────────────────
    # cap_band != "all" 时，用 circ_mv + amount 构造 wide bool mask，后续透传给
    # build_factor_dataset（ML/industry/dynamic 训练截面过滤）和 run_quantile_backtest
    # （回测 eligible 过滤）。mask=None（all）时全市场，向后兼容。
    eligible_mask = None
    if cap_band and cap_band != "all":
        from utils.universe import build_cap_band_mask
        if circ_mv is not None or total_mv is not None:
            eligible_mask = build_cap_band_mask(
                cap_band, circ_mv=circ_mv, amount=amount, total_mv=total_mv,
            )
            if eligible_mask is not None:
                cov = float(eligible_mask.mean().mean())
                logger.info(
                    f"cap-band={cap_band}: mask 就绪, shape={eligible_mask.shape}, "
                    f"平均覆盖率={cov:.3f} (~{int(cov*eligible_mask.shape[1])} 只/日)"
                )
        else:
            logger.warning(
                f"cap-band={cap_band} 但 circ_mv/total_mv 均缺失，退化为全市场（mask=None）"
            )

    # ML / dynamic 可交易池元数据（与 IC build_ic_tradability_mask 同口径）
    tradable_kwargs: dict = {}
    try:
        from research.ic.universe import (
            load_stock_names, load_is_st_current, load_listing_dates,
            load_delist_dates, load_st_history,
        )
        from config.settings import MIN_LISTING_DAYS
        _sn = load_stock_names()
        _ist = load_is_st_current()
        _ld = load_listing_dates()
        _dd = load_delist_dates()
        _sth = load_st_history()
        tradable_kwargs = dict(
            stock_names=_sn, is_st_current=_ist,
            listing_dates=_ld, delist_dates=_dd,
            st_history=_sth,
            tradable_limit_mode=tradable_limit_mode,
            exclude_limit_on_signal=exclude_limit_on_signal,
            apply_exec_mask=apply_exec_mask,
        )
        if _ld:
            logger.info(
                f"ML tradable meta: listing_dates={len(_ld)} 只 "
                f"(min_listing_days={MIN_LISTING_DAYS})"
            )
        if _dd:
            logger.info(f"ML tradable meta: delist_dates={len(_dd)} 只")
        if _sth is not None:
            logger.info(f"ML tradable meta: st_history={len(_sth)} 段")
    except Exception as e:
        logger.warning(f"universe meta 加载失败（build_factor_dataset 将自动回退加载）: {e}")

    restan_in_universe = False
    min_industry_n = 0
    universe_tag = ""
    if mcap_min_yi is not None or mcap_max_yi is not None:
        if (
            mcap_min_yi is not None
            and mcap_max_yi is not None
            and float(mcap_min_yi) >= float(mcap_max_yi)
        ):
            raise ValueError("--mcap-min-yi 必须 < --mcap-max-yi")
        if circ_mv is None and total_mv is None:
            raise ValueError(
                "--mcap-min-yi/--mcap-max-yi 需要 circ_mv 或 total_mv"
                "（请先 python -m data.download_stock_value_em）"
            )
        from research.ic.universe import build_ic_tradability_mask
        from utils.universe import build_mcap_yi_band_mask

        mcap_mask = build_mcap_yi_band_mask(
            circ_mv, min_yi=mcap_min_yi, max_yi=mcap_max_yi, total_mv=total_mv,
        )
        eligible_mask = build_ic_tradability_mask(
            prices,
            volume=volume,
            masks=masks,
            stock_names=tradable_kwargs.get("stock_names"),
            listing_dates=tradable_kwargs.get("listing_dates"),
            delist_dates=tradable_kwargs.get("delist_dates"),
            is_st_current=tradable_kwargs.get("is_st_current"),
            st_history=tradable_kwargs.get("st_history"),
            small_cap_mask=mcap_mask,
            exclude_limit_on_signal=exclude_limit_on_signal,
            tradable_limit_mode=tradable_limit_mode,
        )
        restan_in_universe = True
        min_industry_n = 10
        lo = int(mcap_min_yi) if mcap_min_yi is not None else 0
        hi = int(mcap_max_yi) if mcap_max_yi is not None else 0
        universe_tag = f"mcap{lo}_{hi}"
        per = eligible_mask.sum(axis=1)
        logger.info(
            f"mcap-yi-band [{lo}, {hi}] 亿 ∩ 可交易: 日均 {int(per.mean())} 只 "
            f"(min={int(per.min())} max={int(per.max())})；"
            f"档内 restan + membership WLS universe_tag={universe_tag}"
        )
        if cap_band and cap_band != "all":
            logger.warning(
                f"--mcap-min-yi/--mcap-max-yi 优先于 --cap-band={cap_band}（已忽略 cap-band）"
            )

    midcap_ds_kwargs = dict(
        restan_in_universe=restan_in_universe,
        min_industry_n=min_industry_n,
        universe_tag=universe_tag,
    )

    # 共享的额外数据关键字参数（传给 get_factor_registry）
    # circ_mv / total_mv：市值 alpha（对数市值/分位/风格对齐）与 cap-band 共用
    extra_kwargs = dict(
        prices_raw=prices_raw, volume=volume, amount=amount,
        open_=open_, high=high, low=low,
        clean_ret=clean_ret, masks=masks,
        market_prices=market_prices, industry_map=industry_map,
        margin=margin, moneyflow=moneyflow,
        northbound=northbound, institution=institution,
        circ_mv=circ_mv, total_mv=total_mv,
    )
    cache_kwargs = dict(
        use_factor_cache=use_factor_cache,
        skip_factor_build=skip_factor_build,
        rebuild_factor_cache=rebuild_factor_cache,
    )

    # ── Optuna 超参搜索（离线步骤：搜索完存 JSON 后退出，不继续训练）─────
    if tune:
        from models.wf.tuning import tune_all_models, save_tuned_params
        from strategies.ml import build_factor_dataset
        from models.trainer import resolve_train_windows
        logger.info("Step 3 (tune): Optuna 超参搜索（LGBM/XGB/RF）")
        tune_models = models or ["lgbm", "xgb", "rf"]
        from models.trainer import VAL_WINDOW_MONTHS
        tune_windows = train_windows or [6, 12]
        val_window_months = val_window if val_window is not None else VAL_WINDOW_MONTHS
        win_periods, val_periods = resolve_train_windows(
            tune_windows, val_window_months, rebalance_freq, units="months")
        ds = build_factor_dataset(
            prices, financial, hold_period=label_hold,
            factor_whitelist=factor_whitelist,
            rebalance_freq=rebalance_freq,
            eligible_mask=eligible_mask,
            special_factors=sf_pass,
            regime_cs=False,
            fwd_return_winsor=fwd_return_winsor,
            cs_rank_winsor=cs_rank_winsor,
            label_mode=label_mode,
            **tradable_kwargs, **extra_kwargs, **cache_kwargs, **midcap_ds_kwargs,
        )
        best = tune_all_models(
            tune_models, ds,
            train_windows=win_periods, val_window=val_periods,
            hold_period=label_hold, n_trials=tune_trials,
            label_mode=label_mode,
            time_decay=time_decay,
        )
        out = save_tuned_params(best)
        logger.info(f"超参搜索完成，结果存 {out}")
        return

    # ── Regime-conditional：加载市场状态序列（任务2）─────────────────────────
    regime_states_arg = None
    if regime_conditional:
        from strategies.market_state import get_market_state
        logger.info("加载市场状态序列（regime_conditional=True）...")
        regime_states_arg = get_market_state(method="ma")
        logger.info(
            f"市场状态就绪: {len(regime_states_arg)} 天, "
            f"distribution={regime_states_arg.value_counts().to_dict()}"
        )

    # ── Multi-Horizon 集成（任务1）：优先于单 horizon 路由 ────────────────────
    if multi_horizon:
        logger.info(
            f"Step 3: Multi-Horizon 集成（horizons={multi_horizon}, "
            f"weights={mh_weights}, regime={regime_conditional}）"
        )
        from strategies.multi_horizon import run_multi_horizon

        # model_types 解析（与单 horizon ML 模式一致）
        if models:
            invalid = set(models) - VALID_MODELS
            if invalid:
                raise ValueError(
                    f"--models 包含无效模型: {invalid}，可选: {VALID_MODELS}"
                )
            mh_model_types = models
        else:
            mh_model_types = ["lgbm", "xgb"]

        # Barra 残差化标签（按需计算）
        mh_barra = None
        if label_mode == "barra_residual":
            from factors.barra_risk import get_barra_factors
            mh_barra = get_barra_factors(
                prices=prices, financial=financial,
                market_prices=market_prices, volume=volume,
                clean_ret=clean_ret,
                prices_raw=prices_raw,
                circ_mv=circ_mv, total_mv=total_mv,
                turnover_rate=turnover_rate, amount=amount,
            )

        # industry_map → Series
        mh_ind_map = None
        if industry_map is not None:
            mh_ind_map = (
                industry_map["sw_l2"]
                if isinstance(industry_map, pd.DataFrame)
                and "sw_l2" in industry_map.columns
                else industry_map
            )

        factor_scores, mh_info = run_multi_horizon(
            horizons=list(multi_horizon),
            weights=mh_weights,
            prices=prices, financial=financial,
            extra_kwargs={
                **extra_kwargs, **tradable_kwargs, **midcap_ds_kwargs,
                "special_factors": sf_pass,
                "regime_cs": False,
                "include_regime": False,
            },
            cache_kwargs=cache_kwargs,
            model_types=mh_model_types,
            factor_whitelist=factor_whitelist,
            train_windows=train_windows,
            train_window_units=train_window_units,
            val_window=val_window,
            artifact_dir=out_dir,
            wf_selection=wf_selection,
            label_mode=label_mode,
            ensemble_method=ensemble_method,
            save_models=save_models,
            objective=objective,
            tag=tag,
            device=device,
            barra_factors=mh_barra,
            industry_map=mh_ind_map,
            regime_states=regime_states_arg,
            enable_shap=enable_shap,
            shap_top=shap_top,
            shap_max_samples=shap_max_samples,
            shap_max_dates=shap_max_dates,
        )
        # 保存集成权重 / IC 摘要
        import json as _json
        mh_summary = {
            "tag": tag,
            "horizons": list(multi_horizon),
            "weights": {str(h): mh_info["weights"][h] for h in multi_horizon},
            "ic_per_h": {str(h): mh_info["ic_per_h"][h] for h in multi_horizon},
            "regime_conditional": regime_conditional,
        }
        with open(out_dir / f"multi_horizon_summary_{tag}.json", "w", encoding="utf-8") as f:
            _json.dump(mh_summary, f, ensure_ascii=False, indent=2)

    # ── 生成因子得分 ──────────────────────────────────────────────────────────
    elif mode == "linear":
        logger.info("Step 3: 线性加权策略（基准）")
        from strategies.linear import run as linear_run
        factor_scores = linear_run(prices, financial, FACTOR_WEIGHTS, **extra_kwargs)

    elif mode == "industry":
        if models:
            invalid = set(models) - VALID_MODELS
            if invalid:
                raise ValueError(f"--models 包含无效模型: {invalid}，可选: {VALID_MODELS}")
            model_types = models
        else:
            model_types = list(MODEL_TYPES)
        logger.info(
            f"Step 3: 分行业ML策略（申万二级，模型={model_types}，horizon={horizon}日，"
            f"hold={label_hold}日）"
        )
        from strategies.ml import build_factor_dataset
        from models.industry_trainer import IndustryWalkForwardTrainer
        dataset = build_factor_dataset(
            prices, financial, hold_period=label_hold,
            factor_whitelist=factor_whitelist,
            rebalance_freq=rebalance_freq,
            eligible_mask=eligible_mask,
            special_factors=sf_pass,
            regime_cs=False,
            fwd_return_winsor=fwd_return_winsor,
            cs_rank_winsor=cs_rank_winsor,
            label_mode=label_mode,
            **tradable_kwargs, **extra_kwargs, **cache_kwargs, **midcap_ds_kwargs,
        )
        ind_kwargs = dict(
            model_types    = model_types,
            train_windows  = train_windows,
            rebalance_freq = rebalance_freq,
            hold_period    = label_hold,
            label_mode     = label_mode,
            wf_selection   = wf_selection,
            ensemble_method= ensemble_method,
        )
        if val_window is not None:
            ind_kwargs["val_window"] = val_window
        trainer = IndustryWalkForwardTrainer(**ind_kwargs)
        factor_scores = trainer.fit_predict(dataset, industry_map=industry_map)
        # 保存 IC 指标（与其他模式一致）
        trainer.save_metrics(tag, output_dir=out_dir)
        if trainer.ic_series is not None:
            trainer.ic_series.to_csv(out_dir / f"ic_series_{tag}.csv", header=True)

    elif mode in ML_MODES:
        if models:
            invalid = set(models) - VALID_MODELS
            if invalid:
                raise ValueError(f"--models 包含无效模型: {invalid}，可选: {VALID_MODELS}")
            model_types = models
        else:
            model_types = list(ML_MODES - {"ensemble"}) if mode == "ensemble" else [mode]
        logger.info(
            f"Step 3: ML 策略（{mode}，模型={model_types}，horizon={horizon}日，"
            f"hold={label_hold}日，trainer={trainer_engine}）"
        )
        from strategies.ml import run as ml_run

        # Barra 残差化标签：按需计算 Barra 风格因子并透传给 trainer
        # feature_neutralize 也需要 Barra 因子（对特征做残差化），二者复用同一份
        # industry_map 可能是 DataFrame（多列），先转成 Series 以便透传给 Barra 行业中性化
        ind_map_arg = None
        if industry_map is not None:
            ind_map_arg = (
                industry_map["sw_l2"] if isinstance(industry_map, pd.DataFrame)
                and "sw_l2" in industry_map.columns else industry_map
            )

        barra_factors_arg = None
        if label_mode == "barra_residual" or feature_neutralize or barra_features:
            from factors.barra_risk import get_barra_factors
            if label_mode == "barra_residual":
                logger.info("label_mode=barra_residual: 计算 Barra 风格因子用于标签残差化...")
            if feature_neutralize:
                from models.wf.labels import NEUT_CONTROLS_SIZE_INDUSTRY, normalize_neut_controls
                _nc = normalize_neut_controls(neut_controls)
                if _nc == NEUT_CONTROLS_SIZE_INDUSTRY:
                    logger.info(
                        "feature_neutralize=True neut_controls=size_industry: "
                        "计算 Barra 风格因子后仅用 Size+行业残差化..."
                    )
                else:
                    logger.info("feature_neutralize=True: 计算 Barra 风格因子用于特征残差化...")
            if barra_features:
                logger.info("barra_features=True: 计算 Barra 风格因子作为 ML 输入特征...")
            barra_factors_arg = get_barra_factors(
                prices=prices,
                financial=financial,
                market_prices=market_prices,
                volume=volume,
                clean_ret=clean_ret,
                industry_map=ind_map_arg,
                prices_raw=prices_raw,
                circ_mv=circ_mv, total_mv=total_mv,
                turnover_rate=turnover_rate, amount=amount,
            )
            logger.info(f"Barra 因子就绪: {len(barra_factors_arg)} 个")

        # triple_barrier 需要把日频 prices/open_ 透传给 trainer 用于预计算标签面板
        tb_prices_arg = prices if label_mode == "triple_barrier" else None
        tb_open_arg = open_ if label_mode == "triple_barrier" else None

        # extra_kwargs 含 industry_map(DataFrame)，此处显式传 ind_map_arg(Series)，弹出避免 kwarg 冲突
        ml_extra_kwargs = {k: v for k, v in extra_kwargs.items() if k != "industry_map"}
        from models.wf.stage1_cache import hash_file
        stage1_meta = {
            "horizon": horizon,
            "hold_period": label_hold,
            "factor_config": factor_config,
            "factor_config_hash": hash_file(factor_config),
            "feature_neutralize": bool(feature_neutralize),
            "neut_controls": neut_controls,
            "sparse_from_ic": sparse_from_ic,
            "special_factors": sf_pass,
            "pool_frac": float(stage2_pool_frac),
            "tag": tag,
            "mode": mode,
            "models": list(model_types),
        }
        factor_scores, trainer = ml_run(
            prices, financial,
            model_types=model_types,
            hold_period=label_hold,
            show_report=show_report,
            factor_whitelist=factor_whitelist,
            train_windows=train_windows,
            train_window_units=train_window_units,
            val_window=val_window,
            artifact_dir=out_dir,
            rebalance_freq=rebalance_freq,
            trainer_engine=trainer_engine,
            wf_selection=wf_selection,
            label_mode=label_mode,
            ensemble_method=ensemble_method,
            save_models=save_models,
            objective=objective,
            tag=tag,
            barra_factors=barra_factors_arg,
            industry_map=ind_map_arg,
            device=device,
            tb_prices=tb_prices_arg,
            tb_open=tb_open_arg,
            triple_barrier_params=triple_barrier_params,
            regime_states=regime_states_arg,
            feature_neutralize=feature_neutralize,
            neut_controls=neut_controls,
            barra_features=barra_features,
            special_factors=sf_pass,
            regime_cs=False,
            eligible_mask=eligible_mask,
            ridge_drop_regime=False,
            include_regime=False,
            fwd_return_winsor=fwd_return_winsor,
            cs_rank_winsor=cs_rank_winsor,
            two_stage=two_stage,
            stage2_pool_frac=stage2_pool_frac,
            stage2_lookback=stage2_lookback,
            save_stage1_cache=save_stage1_cache,
            stage1_cache=stage1_cache,
            stage1_cache_meta=stage1_meta,
            rolling_pool_schedule=rolling_pool_schedule,
            rolling_pool_lazy=rolling_pool_lazy,
            rolling_pool_max_cached=rolling_pool_max_cached,
            rolling_pool_strict=rolling_pool_strict,
            enable_shap=enable_shap,
            shap_top=shap_top,
            shap_max_samples=shap_max_samples,
            shap_max_dates=shap_max_dates,
            long_weight_top=long_weight_top,
            long_weight_ratio=long_weight_ratio,
            long_weight_curve=long_weight_curve,
            rank_weight_mid=rank_weight_mid,
            softlong_floor_slope=softlong_floor_slope,
            retrain_every=retrain_every,
            time_decay=time_decay,
            **tradable_kwargs, **ml_extra_kwargs, **cache_kwargs, **midcap_ds_kwargs,
        )
        if hasattr(trainer, "save_metrics"):
            trainer.save_metrics(tag, output_dir=out_dir)
        elif hasattr(trainer, "ic_series") and trainer.ic_series is not None:
            ic = trainer.ic_series.dropna()
            import json
            ic_metrics = {
                "tag": tag, "IC均值": round(ic.mean(), 4),
                "IC标准差": round(ic.std(), 4),
                "ICIR": round(ic.mean() / ic.std(), 4),
                "IC>0胜率": round((ic > 0).mean(), 4),
                "预测期数": len(ic),
            }
            with open(out_dir / f"model_metrics_{tag}.json", "w", encoding="utf-8") as f:
                json.dump(ic_metrics, f, ensure_ascii=False, indent=2)
        if hasattr(trainer, "ic_series") and trainer.ic_series is not None:
            trainer.ic_series.to_csv(out_dir / f"ic_series_{tag}.csv", header=True)

        # ── Dynamic 混合（rank-average ML + Dynamic）────────────────────────
        if blend_dynamic:
            logger.info("blend_dynamic: 追加 DynamicFactorTrainer，与 ML 得分 rank-average")
            from strategies.ml import build_factor_dataset
            from models.dynamic_trainer import DynamicFactorTrainer
            import numpy as np
            # dynamic 轨道：用正交集 + Barra 残差化（与 IC 阶段同口径）
            # 禁止 special/sparse 注入（与 --mode dynamic 一致）
            dyn_dataset = build_factor_dataset(
                prices, financial, hold_period=label_hold,
                factor_whitelist=factor_whitelist_orth or factor_whitelist,
                rebalance_freq=rebalance_freq,
                eligible_mask=eligible_mask,
                feature_neutralize=feature_neutralize,
                neut_controls=neut_controls,
                special_factors=None,
                deny_special_inject=True,
                regime_cs=False,
                barra_factors=barra_factors_arg,
                fwd_return_winsor=fwd_return_winsor,
                cs_rank_winsor=cs_rank_winsor,
                label_mode=label_mode,
                include_regime=False,
                **tradable_kwargs, **extra_kwargs, **cache_kwargs, **midcap_ds_kwargs,
            )
            dyn_lb = _resolve_dynamic_lookback(dynamic_lookback, rebalance_freq)
            dyn_trainer = DynamicFactorTrainer(
                lookback=dyn_lb, min_lookback=max(3, dyn_lb // 2), method="icir",
            )
            dyn_scores = dyn_trainer.fit_predict(dyn_dataset)

            # 对齐日期，各自转百分位 rank，再等权平均
            common_dates = factor_scores.index.intersection(dyn_scores.index)
            ml_rank  = factor_scores.loc[common_dates].rank(axis=1, pct=True)
            dyn_rank = dyn_scores.loc[common_dates].rank(axis=1, pct=True)
            factor_scores = (ml_rank.add(dyn_rank, fill_value=0) / 2)
            logger.info(f"blend_dynamic 完成: {len(common_dates)} 个预测日")

    elif mode == "dynamic":
        dyn_lb = _resolve_dynamic_lookback(dynamic_lookback, rebalance_freq)
        logger.info(
            f"Step 3: 因子动态加权策略（ICIR 权重，lookback={dyn_lb}期，horizon={horizon}日，"
            f"hold={label_hold}日）"
        )
        from strategies.ml import build_factor_dataset
        from models.dynamic_trainer import DynamicFactorTrainer
        from models.trainer import spearman_ic
        import json
        # dynamic 轨道：正交集 + Barra+行业残差化（与 IC 阶段 research/ic/barra.py 同口径）
        ind_map_arg = None
        if industry_map is not None:
            ind_map_arg = (
                industry_map["sw_l2"] if isinstance(industry_map, pd.DataFrame)
                and "sw_l2" in industry_map.columns else industry_map
            )
        barra_factors_arg = None
        if feature_neutralize:
            from factors.barra_risk import get_barra_factors
            logger.info("dynamic feature_neutralize=True: 计算 Barra 风格因子用于特征残差化...")
            barra_factors_arg = get_barra_factors(
                prices=prices,
                financial=financial,
                market_prices=market_prices,
                volume=volume,
                clean_ret=clean_ret,
                industry_map=ind_map_arg,
                prices_raw=prices_raw,
                circ_mv=circ_mv, total_mv=total_mv,
                turnover_rate=turnover_rate, amount=amount,
            )
            logger.info(f"Barra 因子就绪: {len(barra_factors_arg)} 个")
        dataset = build_factor_dataset(
            prices, financial, hold_period=label_hold,
            factor_whitelist=factor_whitelist_orth or factor_whitelist,
            rebalance_freq=rebalance_freq,
            eligible_mask=eligible_mask,
            feature_neutralize=feature_neutralize,
            neut_controls=neut_controls,
            special_factors=None,
            deny_special_inject=True,
            regime_cs=False,
            barra_factors=barra_factors_arg,
            fwd_return_winsor=fwd_return_winsor,
            cs_rank_winsor=cs_rank_winsor,
            label_mode=label_mode,
            include_regime=False,
            **tradable_kwargs, **extra_kwargs, **cache_kwargs, **midcap_ds_kwargs,
        )
        trainer = DynamicFactorTrainer(
            lookback=dyn_lb, min_lookback=max(3, dyn_lb // 2), method="icir",
        )
        factor_scores = trainer.fit_predict(dataset)
        trainer.print_weight_evolution()
        ic_dict = {}
        for date in factor_scores.index:
            _, y = dataset.get_cross_section(date)
            if y is None:
                continue
            s = factor_scores.loc[date].dropna()
            common = s.index.intersection(y.dropna().index)
            if len(common) < 20:
                continue
            ic_dict[date] = spearman_ic(s.loc[common].values, y.loc[common].values)
        if ic_dict:
            ic = pd.Series(ic_dict).dropna()
            ic_metrics = {
                "tag": tag, "IC均值": round(ic.mean(), 4),
                "IC标准差": round(ic.std(), 4),
                "ICIR": round(ic.mean() / ic.std(), 4),
                "IC>0胜率": round((ic > 0).mean(), 4),
                "预测期数": len(ic),
                "dynamic_lookback": dyn_lb,
            }
            with open(out_dir / f"model_metrics_{tag}.json", "w", encoding="utf-8") as f:
                json.dump(ic_metrics, f, ensure_ascii=False, indent=2)
            ic.to_csv(out_dir / f"ic_series_{tag}.csv", header=True)

    else:
        raise ValueError(f"未知 mode: {mode}，可选: {ALL_MODES}")

    factor_scores.to_parquet(out_dir / f"factor_scores_{tag}.parquet")

    # ── 回测（Q1-Q5 分组，模块化 quantile 引擎）────────────────────────────────
    logger.info("Step 4: 回测（quantile 模块化引擎）")

    from backtest.quantile import run_quantile_backtest
    from backtest.report import (
        plot_quantile_result, print_quantile_summary,
        print_holdings, export_holdings, export_turnover_detail,
    )
    from backtest.risk_metrics import export_risk_metrics as _export_risk_metrics
    from backtest.execution import (
        BacktestConfig,
        build_st_schedule,
        build_delist_dates_from_stock_list,
        build_listing_dates_from_stock_list,
    )
    from research.ic.universe import mask_scores_for_backtest
    indices = _load_indices()
    # bid_ask_spread_bps：None=读 settings 默认，CLI 显式覆盖
    bt_kwargs = dict(
        turnover_limit=turnover_limit,
        rank_change_threshold=rank_change_threshold,
        portfolio_opt=portfolio_opt or "ew",
        max_weight=max_weight,
        cov_lookback=int(cov_lookback),
        risk_aversion=float(risk_aversion),
    )
    if bid_ask_spread_bps is not None:
        bt_kwargs["bid_ask_spread_bps"] = bid_ask_spread_bps
        logger.info(f"bid-ask spread override: {bid_ask_spread_bps} bp")
    bt_config = BacktestConfig(**bt_kwargs)
    if (portfolio_opt or "ew").strip().lower() not in ("", "ew", "equal", "equal_weight"):
        logger.info(
            f"portfolio_opt={bt_config.portfolio_opt} "
            f"max_weight={bt_config.max_weight} "
            f"cov_lookback={bt_config.cov_lookback} "
            f"risk_aversion={bt_config.risk_aversion}"
        )

    # M4 修复：从 stock_list.parquet 构建时间序列 ST 状态 + 退市/上市日期字典
    stock_names_ser: pd.Series | None = None
    st_schedule: pd.DataFrame | None = None
    delist_dates: dict[str, pd.Timestamp] | None = None
    listing_dates: dict[str, pd.Timestamp] | None = None
    is_st_ser: pd.Series | None = None
    # P0-2：真实 ST 历史长表（data/download_st_history.py 产出）。文件不存在
    # 时为 None，build_st_schedule 回退到 M4 保守实现（向后兼容）。
    st_history_df: pd.DataFrame | None = None
    try:
        from config.settings import UNIVERSE_DIR, RAW_DIR
        st_hist_path = RAW_DIR / "st_history.parquet"
        if st_hist_path.exists():
            st_history_df = pd.read_parquet(st_hist_path)
            logger.info(
                f"P0-2 加载真实 ST 历史: {len(st_history_df)} 段, "
                f"{st_history_df['code'].nunique()} 只股票（{st_hist_path.name}）"
            )
    except Exception as e:
        logger.warning(f"P0-2 ST 历史加载失败（忽略，回退保守实现）: {e}")
        st_history_df = None
    try:
        from config.settings import UNIVERSE_DIR
        sl_path = UNIVERSE_DIR / "stock_list.parquet"
        if sl_path.exists():
            sl_df = pd.read_parquet(sl_path)
            if "code" in sl_df.columns and "name" in sl_df.columns:
                stock_names_ser = sl_df.set_index("code")["name"]
                stock_names_ser.index = stock_names_ser.index.astype(str).str.zfill(6)
            delist_dates = build_delist_dates_from_stock_list(sl_df) or None
            listing_dates = build_listing_dates_from_stock_list(sl_df) or None
            is_st_ser = (
                sl_df.set_index("code")["is_st_current"]
                if "is_st_current" in sl_df.columns else None
            )
            if is_st_ser is not None:
                is_st_ser.index = is_st_ser.index.astype(str).str.zfill(6)
            st_schedule = build_st_schedule(
                stock_names_ser,
                prices.index,
                is_st_current=is_st_ser,
                delist_dates=delist_dates,
                st_history=st_history_df,
            )
            if st_schedule is not None:
                src = "真实历史" if st_history_df is not None else "保守实现"
                logger.info(
                    f"M4 ST 时间序列 ({src}): {st_schedule.shape[1]} 只 ST 股 × "
                    f"{st_schedule.shape[0]} 交易日"
                )
            if delist_dates:
                logger.info(f"M4 退市日期字典: {len(delist_dates)} 只")
            if listing_dates:
                logger.info(
                    f"次新过滤 listing_dates: {len(listing_dates)} 只 "
                    f"(min_listing_days={bt_config.min_listing_days})"
                )
    except Exception as e:
        logger.warning(f"M4 ST/退市元数据加载失败（忽略，回退旧行为）: {e}")

    top_n_eff = int(top_n) if top_n is not None else int(N_STOCKS)

    # Research 训练可保留信号日涨跌停标签；默认回测得分宇宙仍用 strict，
    # 避免 label 池扩张泄漏进等权基准 / Q 分组（execution 本身未关涨停拦截）。
    n_scored_train = int(factor_scores.notna().sum().sum())
    bt_scores = mask_scores_for_backtest(
        factor_scores,
        prices,
        open_=open_,
        hold_period=int(label_hold),
        volume=volume,
        masks=masks,
        stock_names=stock_names_ser,
        listing_dates=listing_dates,
        delist_dates=delist_dates,
        is_st_current=is_st_ser,
        st_history=st_history_df,
        score_universe=bt_score_universe,
    )
    n_scored_bt = int(bt_scores.notna().sum().sum())
    if n_scored_bt != n_scored_train:
        logger.info(
            f"bt_score_universe={bt_score_universe}: 回测得分格子 "
            f"{n_scored_train} → {n_scored_bt} "
            f"(Δ={n_scored_train - n_scored_bt}; 训练分数仍落盘未改)"
        )
    else:
        logger.info(f"bt_score_universe={bt_score_universe}: 回测得分覆盖未裁剪")

    # 仓位体制（可选）：市场级标量 → target_exposure，缩放非 benchmark 收益
    pos_regime_df = None
    if position_regime:
        if market_prices is None:
            logger.warning(
                "position_regime=True 但 market_prices 缺失，跳过敞口缩放"
            )
        else:
            from backtest.regime import PositionRegimeConfig, compute_position_regime
            pos_cfg = PositionRegimeConfig(force_exposure=force_exposure)
            pos_regime_df = compute_position_regime(
                market_prices=market_prices,
                prices=prices,
                clean_ret=clean_ret,
                circ_mv=circ_mv,
                config=pos_cfg,
            )
            logger.info(
                f"position_regime 就绪: exposure "
                f"mean={pos_regime_df['target_exposure'].mean():.3f} "
                f"min={pos_regime_df['target_exposure'].min():.3f} "
                f"max={pos_regime_df['target_exposure'].max():.3f}"
            )

    result = run_quantile_backtest(
        prices, bt_scores,
        n_quantiles=5,
        rebalance_freq=bt_freq,
        start=BACKTEST_START,
        end=BACKTEST_END,
        open_prices=open_,   # 次日开盘执行，一字涨停自动剔除
        masks=masks,
        indices=indices,
        config=bt_config,
        stock_names=stock_names_ser,
        listing_dates=listing_dates,
        volume=volume,
        st_schedule=st_schedule,
        delist_dates=delist_dates,
        eligible_mask=eligible_mask,
        top_n=top_n_eff,
        position_regime=pos_regime_df,
        returns=clean_ret,
        hold_period=int(label_hold) if hold_period is not None else None,
    )
    print_quantile_summary(result, rebalance_freq=bt_freq, rf=RISK_FREE_RATE)
    plot_quantile_result(
        result,
        title=f"Q1-Q5 分组回测  |  mode={mode}  horizon={horizon}日 hold={label_hold}日",
        save_path=str(out_dir / f"backtest_{tag}.png"),
        rebalance_freq=bt_freq,
        rf=RISK_FREE_RATE,
    )
    # 保存原始回测数据
    result.nav.to_csv(out_dir / f"backtest_{tag}_nav.csv", encoding="utf-8-sig")
    result.annual_returns.to_csv(out_dir / f"backtest_{tag}_annual.csv", encoding="utf-8-sig")
    result.long_short_nav.to_csv(out_dir / f"backtest_{tag}_longshort.csv", header=True)
    _export_risk_metrics(
        result.nav,
        save_path=str(out_dir / f"backtest_{tag}_risk_metrics.csv"),
        rebalance_freq=bt_freq,
        rf=RISK_FREE_RATE,
    )
    export_holdings(result, save_path=str(out_dir / f"holdings_top{top_n_eff}_{tag}.csv"))
    export_turnover_detail(
        result, save_path=str(out_dir / f"turnover_detail_{tag}.csv"),
    )
    if result.position_exposure is not None:
        result.position_exposure.to_csv(
            out_dir / f"position_exposure_{tag}.csv", header=True,
        )
    if result.position_regime is not None:
        result.position_regime.to_csv(
            out_dir / f"position_regime_{tag}.csv", encoding="utf-8-sig",
        )
    if show_holdings:
        print_holdings(result, last_n=12)

    return result


if __name__ == "__main__":
    from utils.cli_help import add_help_advanced, exit_if_help_advanced, help_text as _h

    parser = argparse.ArgumentParser(
        description="A股多因子选股流水线（数据→因子→训练→回测）",
        epilog=(
            "日常最短: python run.py --skip-download --mode ridge --horizon 5 "
            "--factor-config config/factor_configs.yaml\n"
            "默认已含: feature-neutralize / bid-ask(settings) / research tradable / "
            "label=cs_rank。全部参数: --help-advanced 或 docs/操作手册.md"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_help_advanced(parser)

    # ── 日常必用 ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--mode", default="linear",
        choices=sorted(ALL_MODES),
        help="策略模式: linear | ridge | lgbm | xgb | cat | ensemble | dynamic | industry",
    )
    parser.add_argument(
        "--horizon", type=int, default=20,
        help="调仓频率锚点（交易日）: 5=周频 10=双周 20=月频(默认) 60=季频；"
             "标签/回测持有默认同此，可用 --hold-period 覆盖",
    )
    parser.add_argument("--skip-download", action="store_true", help="跳过数据下载")
    parser.add_argument("--sample", type=int, default=0, help="仅前 N 只股票（调试）")
    parser.add_argument(
        "--factor-config", default=None,
        help="因子白名单 YAML/JSON（ic_analysis --save / driver 生成）",
    )
    parser.add_argument(
        "--models", default=None,
        help="ensemble 模型子集，逗号分隔，如 lgbm,xgb（默认全部）",
    )
    parser.add_argument(
        "--blend-dynamic", action="store_true",
        help="ML 得分与 DynamicFactorTrainer rank-average 混合",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="训练后展示 IC / SHAP 分析报告",
    )
    parser.add_argument(
        "--holdings", action="store_true",
        help="回测完打印并导出 Top-N 每期持仓 CSV",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="实验输出目录（默认 results/<tag>/）",
    )
    parser.add_argument(
        "--shap", action="store_true",
        help="训练期计算 SHAP 因子贡献（写 results/<tag>/shap_*.csv）",
    )
    parser.add_argument(
        "--rolling-pool-schedule",
        default=None,
        metavar="PATH",
        help="rolling-pool 长表；WF 每期特征=当日 pool_t（优先于 --factor-config）",
    )
    parser.add_argument(
        "--special-factors",
        "--inject-factors",
        dest="special_factors",
        default=None,
        help="特殊因子 pack（event/size/sparse），白名单后再注入；详见 docs/操作手册.md §5.4",
    )
    parser.add_argument(
        "--cap-band",
        default="all",
        help=(
            "市值带: all|small|small_mid|small_mid_wide|mid|micro|"
            "micro_small_100|micro_30(=micro_lt30)（默认 all）。"
            "micro_30/micro_lt30: circ_mv∈(0,30亿]、无8亿地板，20d均额≥2000万；"
            "勿与 micro(8~30亿) 混淆"
        ),
    )
    parser.add_argument(
        "--mcap-min-yi",
        type=float,
        default=None,
        dest="mcap_min_yi",
        help=(
            "流通市值下限（亿元，含；单位元=亿×1e8）。与 --mcap-max-yi 构成每日宇宙，"
            "无 20 日成交额过滤；档内 restan + membership WLS（与 research.midcap_ic 同口径）"
        ),
    )
    parser.add_argument(
        "--mcap-max-yi",
        type=float,
        default=None,
        dest="mcap_max_yi",
        help="流通市值上限（亿元，含）。例: --mcap-min-yi 30 --mcap-max-yi 100",
    )

    # ── 高级（默认 --help 隐藏）──────────────────────────────────────────
    parser.add_argument(
        "--shap-top", type=int, default=20,
        help=_h("SHAP Top-N 特征数（默认 20）", advanced=True),
    )
    parser.add_argument(
        "--shap-max-samples", type=int, default=500,
        help=_h("每折 SHAP 最大样本行数（默认 500）", advanced=True),
    )
    parser.add_argument(
        "--shap-max-dates", type=int, default=12,
        help=_h("仅最近 N 个预测日算 SHAP（默认 12；<=0=全部）", advanced=True),
    )
    parser.add_argument(
        "--rolling-pool-lazy",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=_h(
            "rolling-pool 按需加载（有 schedule 时默认开；--no-rolling-pool-lazy 全 U）",
            advanced=True,
        ),
    )
    parser.add_argument(
        "--rolling-pool-max-cached",
        type=int,
        default=160,
        help=_h("lazy PanelStore LRU 常驻上限（默认 160）", advanced=True),
    )
    parser.add_argument(
        "--rolling-pool-strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=_h("schedule 缺面板 fail-fast（默认开）", advanced=True),
    )
    parser.add_argument(
        "--train-windows", default=None,
        help=_h("训练窗口月数，逗号分隔，如 6,12（默认 6,12）", advanced=True),
    )
    parser.add_argument(
        "--val-window", type=int, default=None,
        help=_h(
            "验证窗口月数（默认 6；0=无独立 val，train 贴 pred；多窗须 --wf-selection average；"
            "两窗共用近期 val，与 --train-windows 同单位）",
            advanced=True,
        ),
    )
    parser.add_argument(
        "--train-window-units", default="months",
        choices=["months", "periods"],
        help=_h("训练窗口单位：months（默认）| periods", advanced=True),
    )
    parser.add_argument(
        "--retrain-every", type=int, default=RETRAIN_EVERY,
        help=_h(
            f"每 N 个调仓期重训一次（默认 {RETRAIN_EVERY}=周频≈月度；"
            "每期重训用 1；周频约一季用 13）；"
            "中间预测日复用最近重训日模型",
            advanced=True,
        ),
    )
    parser.add_argument(
        "--time-decay",
        type=float,
        default=TIME_DECAY,
        dest="time_decay",
        help=_h(
            f"训练样本指数时间衰减（默认 {TIME_DECAY}，与历史实验一致；"
            "0=窗内等权，仍保留 universe 归一化与 overlap 权重）",
            advanced=True,
        ),
    )
    parser.add_argument(
        "--backtest-freq", default=None,
        help=_h("回测调仓频率覆盖（如 ME）；用于复现训练/回测频率错位", advanced=True),
    )
    parser.add_argument(
        "--hold-period", "--label-horizon",
        type=int, default=None, dest="hold_period",
        help=_h(
            "标签与回测持有交易日（如 3=close[t+3]/open[t+1]）。"
            "不改 --horizon 推断的调仓日历（W-FRI 等）；默认=--horizon。"
            "embargo/purge 按此换算",
            advanced=True,
        ),
    )
    parser.add_argument(
        "--top-n", type=int, default=None,
        help=_h(f"回测 Top-N（默认 settings.N_STOCKS={N_STOCKS}）", advanced=True),
    )
    parser.add_argument(
        "--dynamic-lookback", type=int, default=None,
        help=_h("dynamic/blend ICIR 回看调仓期数（默认 6 月换算）", advanced=True),
    )
    parser.add_argument(
        "--skip-factor-build", action="store_true",
        help=_h("仅从磁盘缓存加载因子面板；缺失则报错", advanced=True),
    )
    parser.add_argument(
        "--rebuild-factor-cache", action="store_true",
        help=_h("忽略已有因子面板缓存，强制重算", advanced=True),
    )
    parser.add_argument(
        "--no-factor-cache", action="store_true",
        help=_h("禁用因子面板磁盘缓存", advanced=True),
    )
    parser.add_argument(
        "--wf-selection", default="ic_weighted",
        choices=["average", "best_window", "best_model", "ic_weighted"],
        help=_h("多窗口/模型验证 IC 加权方式（默认 ic_weighted）", advanced=True),
    )
    parser.add_argument(
        "--label-mode", default=LABEL_MODE_DEFAULT,
        choices=[
            "raw", "cs_rank", "cs_zscore", "top40_cs_zscore",
            "cs_rank_softlong", "triple_barrier", "barra_residual",
        ],
        help=_h("训练标签截面标准化（默认 cs_rank）", advanced=True),
    )
    parser.add_argument(
        "--long-weight-top", type=float, default=None,
        help=_h("多头偏置 sample_weight：top 区占比（如 0.4）", advanced=True),
    )
    parser.add_argument(
        "--long-weight-ratio", type=float, default=0.25,
        help=_h("多头偏置：bottom 区权重（默认 0.25）", advanced=True),
    )
    parser.add_argument(
        "--long-weight-curve", default="smooth", choices=["smooth", "step"],
        help=_h("多头偏置权重曲线：smooth|step", advanced=True),
    )
    parser.add_argument(
        "--rank-weight-mid",
        type=float,
        default=1.0,
        dest="rank_weight_mid",
        help=_h(
            "截面分位 U 形 sample_weight 的中间权重（默认 1.0=关闭；"
            "0.6=中间 0.6、两端 1.0，乘到 time-decay/universe/overlap 上）",
            advanced=True,
        ),
    )
    parser.add_argument(
        "--rank-tail-weight",
        action="store_true",
        dest="rank_tail_weight",
        help=_h("开启两端加权（等价 --rank-weight-mid 0.6）", advanced=True),
    )
    parser.add_argument(
        "--softlong-floor-slope", type=float, default=0.25,
        help=_h("cs_rank_softlong 下侧负斜率（默认 0.25）", advanced=True),
    )
    parser.add_argument(
        "--ensemble-method", default="zscore", choices=["rank", "zscore"],
        help=_h("多模型集成：rank|zscore（默认 zscore）", advanced=True),
    )
    parser.add_argument(
        "--tb-upper", type=float, default=2.0,
        help=_h("triple_barrier 上障碍倍数（默认 2.0）", advanced=True),
    )
    parser.add_argument(
        "--tb-lower", type=float, default=1.5,
        help=_h("triple_barrier 下障碍倍数（默认 1.5）", advanced=True),
    )
    parser.add_argument(
        "--tb-vol-window", type=int, default=20,
        help=_h("triple_barrier 波动率窗口（默认 20）", advanced=True),
    )
    parser.add_argument(
        "--tb-label-type", default="sign", choices=["sign", "return"],
        help=_h("triple_barrier 标签类型 sign|return", advanced=True),
    )
    parser.add_argument(
        "--save-models", action="store_true",
        help=_h("保存每折模型到 results/<tag>/models/", advanced=True),
    )
    parser.add_argument(
        "--objective", default="regression", choices=["regression", "rank"],
        help=_h("训练目标 regression|rank（默认 regression）", advanced=True),
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "gpu"],
        help=_h("LGBM/XGB 设备 cpu|gpu（默认 cpu）", advanced=True),
    )
    parser.add_argument(
        "--tune", action="store_true",
        help=_h("Optuna 超参搜索后退出", advanced=True),
    )
    parser.add_argument(
        "--tune-trials", type=int, default=15,
        help=_h("Optuna 每模型轮数（默认 15）", advanced=True),
    )
    parser.add_argument(
        "--multi-horizon", default=None,
        help=_h("多期限集成，如 5,10,20（忽略单 --horizon）", advanced=True),
    )
    parser.add_argument(
        "--mh-weights", default="equal",
        choices=["equal", "ic_weighted"],
        help=_h("Multi-Horizon 权重 equal|ic_weighted", advanced=True),
    )
    parser.add_argument(
        "--regime-conditional", action="store_true",
        help=_h("按市场 regime 过滤训练样本（需 market_state）", advanced=True),
    )
    parser.add_argument(
        "--turnover-limit", type=float, default=1.0,
        help=_h("每期最大换手率 0-1（默认 1=无限制）", advanced=True),
    )
    parser.add_argument(
        "--rank-change-threshold", type=float, default=0.0,
        help=_h("排名变动阈值，跌出 top 比例才换仓（默认 0=关）", advanced=True),
    )
    parser.add_argument(
        "--bid-ask-spread", type=float, default=None,
        help=_h(
            "bid-ask spread 单边 bp，覆盖 settings.BID_ASK_SPREAD_BPS（默认 10）",
            advanced=True,
        ),
    )
    parser.add_argument(
        "--feature-neutralize",
        dest="feature_neutralize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=_h(
            "ML/dynamic 特征残差化（默认开；--no-feature-neutralize 关）。"
            "控制变量集合见 --neut-controls",
            advanced=True,
        ),
    )
    parser.add_argument(
        "--neut-controls",
        dest="neut_controls",
        choices=("barra", "size_industry", "size"),
        default="barra",
        help=_h(
            "feature-neutralize 控制变量：barra=9风格+行业（默认）；"
            "size_industry=仅 log(流通市值)+PIT行业哑变量；"
            "size=仅 Size（无行业）；WLS 仍√市值",
            advanced=True,
        ),
    )
    parser.add_argument(
        "--fwd-return-winsor",
        dest="fwd_return_winsor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=_h("forward_return 截面 1%/99% 截尾（默认开；cs_rank 仍默认跳过）", advanced=True),
    )
    parser.add_argument(
        "--cs-rank-winsor",
        dest="cs_rank_winsor",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=_h("cs_rank 先截面 winsor 再 rank（默认关；消融用）", advanced=True),
    )
    parser.add_argument(
        "--tradable-strict",
        action="store_true",
        dest="tradable_strict",
        help=_h("信号日可交易池剔除涨跌停（旧 strict）", advanced=True),
    )
    parser.add_argument(
        "--label-exec-mask",
        action="store_true",
        dest="label_exec_mask",
        help=_h("标签屏蔽买日一字涨停/卖日涨跌停（旧 strict）", advanced=True),
    )
    parser.add_argument(
        "--tradable-limit-mode",
        choices=("strict", "research"),
        default=None,
        dest="tradable_limit_mode",
        help=_h("涨跌停口径别名 research（默认）|strict", advanced=True),
    )
    parser.add_argument(
        "--bt-score-universe",
        choices=("strict", "train"),
        default="strict",
        dest="bt_score_universe",
        help=_h(
            "回测得分宇宙：strict=信号日剔涨跌停∩label-exec门控（默认，基准不随 research 训练池膨胀）；"
            "train=与训练可交易池一致",
            advanced=True,
        ),
    )
    parser.add_argument(
        "--barra-features",
        action="store_true",
        default=False,
        help=_h("Barra 9 风格作树模型输入特征（tag _barrafeat）", advanced=True),
    )
    parser.add_argument(
        "--sparse-from-ic",
        dest="sparse_from_ic",
        default=None,
        help=_h("IC JSON 路径；sparse pack 仅注入 factors_sparse", advanced=True),
    )
    parser.add_argument(
        "--position-regime",
        "--pos-regime",
        dest="position_regime",
        action="store_true",
        default=False,
        help=_h("仓位体制：按市场缩放总敞口（docs/操作手册.md §5.1）", advanced=True),
    )
    parser.add_argument(
        "--force-exposure",
        type=float,
        default=None,
        help=_h("配合 --position-regime 强制固定敞口 0~1", advanced=True),
    )
    parser.add_argument(
        "--portfolio-opt",
        default="ew",
        choices=["ew", "score", "rank", "mv", "invvol", "rp"],
        help=_h("组合权重 ew|score|rank|mv|invvol|rp（默认 ew）", advanced=True),
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=None,
        help=_h("单票权重上限 0~1", advanced=True),
    )
    parser.add_argument(
        "--cov-lookback",
        type=int,
        default=60,
        help=_h("mv/invvol/rp 协方差回看交易日（默认 60）", advanced=True),
    )
    parser.add_argument(
        "--risk-aversion",
        type=float,
        default=1.0,
        help=_h("MV 风险厌恶 λ（默认 1.0）", advanced=True),
    )
    parser.add_argument(
        "--two-stage",
        action="store_true",
        help=_h("两阶段 ridge：S1 全市场 → S2 池内（models/wf/two_stage.py）", advanced=True),
    )
    parser.add_argument(
        "--stage2-pool-frac",
        type=float,
        default=0.2,
        help=_h("两阶段 S1 候选池截面分位（默认 0.2）", advanced=True),
    )
    parser.add_argument(
        "--stage2-lookback",
        type=int,
        default=None,
        help=_h("两阶段 S2 训练回看调仓期数", advanced=True),
    )
    parser.add_argument(
        "--save-stage1-cache",
        action="store_true",
        help=_h("写出 stage1_cache（universe only）", advanced=True),
    )
    parser.add_argument(
        "--stage1-cache",
        type=str,
        default=None,
        help=_h("加载已有 stage1_cache，跳过 S1（隐含 --two-stage）", advanced=True),
    )

    # ── deprecated：仍接受，默认 --help 隐藏 ─────────────────────────────
    parser.add_argument(
        "--backtest-engine", default="v2",
        help=_h("仅模块化 quantile；无效果", deprecated=True),
    )
    parser.add_argument(
        "--trainer-engine", default="v2",
        help=_h("仅模块化 WF trainer；无效果", deprecated=True),
    )
    parser.add_argument(
        "--event-overlay",
        action="store_true",
        default=False,
        help=_h("请改用 --special-factors event", deprecated=True),
    )
    parser.add_argument(
        "--regime-cs",
        dest="regime_cs",
        action="store_true",
        default=False,
        help=_h("轮动_* CS 注入已移除；请用 --position-regime", deprecated=True),
    )
    parser.add_argument(
        "--ridge-drop-regime",
        action="store_true",
        default=False,
        help=_h("市场/HMM 不再注入 ML X，无需 drop", deprecated=True),
    )
    parser.add_argument(
        "--no-regime",
        dest="no_regime_flag",
        action="store_true",
        default=False,
        help=_h("ML 默认不再注入市场/HMM，可省略", deprecated=True),
    )

    args = parser.parse_args()
    exit_if_help_advanced(parser, args)
    if getattr(args, "no_regime_flag", False):
        logger.warning("--no-regime 已退役 no-op：ML 默认不再注入市场/HMM，可省略")
    if args.regime_cs:
        logger.warning("--regime-cs 已退役 no-op：请用 --position-regime")
    if args.ridge_drop_regime:
        logger.warning("--ridge-drop-regime 已退役 no-op")
    train_windows = (
        [int(x) for x in args.train_windows.split(",")]
        if args.train_windows else None
    )
    models = (
        [m.strip() for m in args.models.split(",")]
        if args.models else None
    )
    multi_horizon_arg = (
        [int(x.strip()) for x in args.multi_horizon.split(",") if x.strip()]
        if args.multi_horizon else None
    )
    exclude_limit = True if args.tradable_strict else None
    apply_exec = True if args.label_exec_mask else None
    if args.tradable_limit_mode == "strict":
        exclude_limit = True
        apply_exec = True
    elif args.tradable_limit_mode == "research":
        exclude_limit = False
        apply_exec = False
    main(
        mode=args.mode,
        skip_download=args.skip_download,
        sample=args.sample,
        show_report=args.report,
        horizon=args.horizon,
        hold_period=args.hold_period,
        factor_config=args.factor_config,
        show_holdings=args.holdings,
        train_windows=train_windows,
        train_window_units=args.train_window_units,
        val_window=args.val_window,
        models=models,
        blend_dynamic=args.blend_dynamic,
        output_dir=args.output_dir,
        backtest_freq=args.backtest_freq,
        use_factor_cache=not args.no_factor_cache,
        skip_factor_build=args.skip_factor_build,
        rebuild_factor_cache=args.rebuild_factor_cache,
        dynamic_lookback=args.dynamic_lookback,
        backtest_engine=args.backtest_engine,
        trainer_engine=args.trainer_engine,
        wf_selection=args.wf_selection,
        label_mode=args.label_mode,
        long_weight_top=args.long_weight_top,
        long_weight_ratio=args.long_weight_ratio,
        long_weight_curve=args.long_weight_curve,
        rank_weight_mid=(
            0.6 if args.rank_tail_weight and abs(float(args.rank_weight_mid) - 1.0) < 1e-12
            else float(args.rank_weight_mid)
        ),
        softlong_floor_slope=args.softlong_floor_slope,
        retrain_every=args.retrain_every,
        time_decay=args.time_decay,
        ensemble_method=args.ensemble_method,
        save_models=args.save_models,
        objective=args.objective,
        device=args.device,
        tune=args.tune,
        tune_trials=args.tune_trials,
        triple_barrier_params={
            "vol_window": args.tb_vol_window,
            "upper_mult": args.tb_upper,
            "lower_mult": args.tb_lower,
            "label_type": args.tb_label_type,
        },
        multi_horizon=multi_horizon_arg,
        mh_weights=args.mh_weights,
        regime_conditional=args.regime_conditional,
        turnover_limit=args.turnover_limit,
        rank_change_threshold=args.rank_change_threshold,
        bid_ask_spread_bps=args.bid_ask_spread,
        feature_neutralize=args.feature_neutralize,
        neut_controls=args.neut_controls,
        cap_band=args.cap_band,
        mcap_min_yi=args.mcap_min_yi,
        mcap_max_yi=args.mcap_max_yi,
        barra_features=args.barra_features,
        special_factors=args.special_factors,
        event_overlay=args.event_overlay,
        sparse_from_ic=args.sparse_from_ic,
        regime_cs=args.regime_cs,
        ridge_drop_regime=args.ridge_drop_regime,
        include_regime=False,
        position_regime=args.position_regime,
        force_exposure=args.force_exposure,
        fwd_return_winsor=args.fwd_return_winsor,
        cs_rank_winsor=args.cs_rank_winsor,
        tradable_limit_mode=args.tradable_limit_mode,
        exclude_limit_on_signal=exclude_limit,
        apply_exec_mask=apply_exec,
        bt_score_universe=args.bt_score_universe,
        top_n=args.top_n,
        portfolio_opt=args.portfolio_opt,
        max_weight=args.max_weight,
        cov_lookback=args.cov_lookback,
        risk_aversion=args.risk_aversion,
        two_stage=args.two_stage,
        stage2_pool_frac=args.stage2_pool_frac,
        stage2_lookback=args.stage2_lookback,
        save_stage1_cache=args.save_stage1_cache,
        stage1_cache=args.stage1_cache,
        enable_shap=args.shap,
        shap_top=args.shap_top,
        shap_max_samples=args.shap_max_samples,
        shap_max_dates=args.shap_max_dates,
        rolling_pool_schedule=args.rolling_pool_schedule,
        rolling_pool_lazy=args.rolling_pool_lazy,
        rolling_pool_max_cached=args.rolling_pool_max_cached,
        rolling_pool_strict=args.rolling_pool_strict,
    )
