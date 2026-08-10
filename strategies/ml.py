"""
strategies/ml.py  —  ML 多因子策略

调用 WalkForwardTrainer 训练（模块化 WF：purged split + embargo + IC 加权 ensemble），
输出样本外预测得分。支持单模型（lgbm/xgb/cat/ridge/rf/mlp）和 ensemble。
"""
import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    BACKTEST_START,
    BACKTEST_END,
    FWD_RETURN_WINSOR,
    RETRAIN_EVERY,
    resolve_apply_exec_mask,
    resolve_exclude_limit_on_signal,
)
from factors.factor import get_factor_registry
from factors.factor_cache import (
    cache_exists, factor_cache_path, load_factor_panel, save_factor_panel,
)
from factors.special_factors import (
    inject_special_factors,
    resolve_special_factors,
    should_skip_neutralize,
)
from models.trainer import (
    WalkForwardTrainer, RegimeConditionalTrainer,
    build_ml_dataset, MODEL_TYPES, REBALANCE_FREQ,
)
from research.ic.forward_return import build_forward_return, winsorize_forward_return


def _registry_kwargs(
    prices, financial, prices_raw, volume, amount,
    open_, high, low, clean_ret, masks,
    market_prices, industry_map, margin, moneyflow,
    northbound, institution,
    circ_mv=None, total_mv=None,
) -> dict:
    return dict(
        prices=prices, financial=financial,
        prices_raw=prices_raw, volume=volume, amount=amount,
        open_=open_, high=high, low=low,
        clean_ret=clean_ret, masks=masks,
        market_prices=market_prices, industry_map=industry_map,
        margin=margin, moneyflow=moneyflow,
        northbound=northbound, institution=institution,
        circ_mv=circ_mv, total_mv=total_mv,
    )


def _load_or_compute_registry(
    factor_whitelist: list | None,
    use_factor_cache: bool,
    skip_factor_build: bool,
    rebuild_factor_cache: bool,
    hold_period: int,
    rebalance_freq: str | None,
    registry_kwargs: dict,
    include_regime: bool = False,
    prefer_disk_panels: bool = False,
) -> dict:
    if include_regime:
        logger.warning(
            "include_regime=True 已退役：市场/HMM 不再注入 ML X；"
            "仓位控制请用 --position-regime"
        )
    # 缓存 key 固定 reg=0（市场广播已退役）；保留参数仅为签名兼容
    freq = rebalance_freq or REBALANCE_FREQ
    cache_path = factor_cache_path(
        hold_period, freq, factor_whitelist, BACKTEST_START, BACKTEST_END,
        include_regime=False,
    )

    if use_factor_cache and not rebuild_factor_cache and cache_exists(cache_path):
        return load_factor_panel(cache_path)

    if skip_factor_build and not prefer_disk_panels:
        raise FileNotFoundError(
            f"因子缓存不存在且指定了 --skip-factor-build: {cache_path}"
        )

    # rolling-pool 大并集：优先单因子 parquet（factor_panels/），避免全量重算
    if prefer_disk_panels and factor_whitelist:
        from research.rolling_pool.schedule_load import load_panels_prefer_cache
        prices = registry_kwargs.get("prices")
        if prices is None:
            raise ValueError("prefer_disk_panels 需要 registry_kwargs['prices']")
        if skip_factor_build:
            registry = load_panels_prefer_cache(
                factor_whitelist, prices, registry_kwargs, compute_missing=False,
            )
        else:
            registry = load_panels_prefer_cache(
                factor_whitelist, prices, registry_kwargs, compute_missing=True,
            )
        if not registry:
            raise FileNotFoundError(
                "rolling-pool 并集因子面板为空（disk miss 且无法计算）"
            )
        logger.info(f"rolling_pool 因子库就绪: {len(registry)} 个（并集 U）")
    else:
        if skip_factor_build:
            raise FileNotFoundError(
                f"因子缓存不存在且指定了 --skip-factor-build: {cache_path}"
            )
        factor_names = factor_whitelist if factor_whitelist else None
        registry = get_factor_registry(
            **registry_kwargs, factor_names=factor_names, include_regime=False,
        )

        if factor_whitelist:
            before = len(registry)
            barra_keys = {k for k in registry if k.startswith("Barra_")}
            wl_set = set(factor_whitelist)
            registry = {
                k: v for k, v in registry.items()
                if k in wl_set or k in barra_keys
            }
            logger.info(
                f"因子白名单过滤: {before} → {len(registry)} 个因子"
                f"（Barra 额外保留 {len(barra_keys)} 个）"
            )
        else:
            logger.info(f"因子库就绪: {len(registry)} 个因子（全量）")

    # 大并集（rolling-pool）已有单因子 parquet；整包 pkl 可能极大，跳过落盘
    n_wl = len(factor_whitelist) if factor_whitelist else 0
    if use_factor_cache and not (prefer_disk_panels and n_wl > 80):
        save_factor_panel(
            cache_path, registry,
            hold_period=hold_period,
            rebalance_freq=freq,
            factor_whitelist=factor_whitelist,
            start=BACKTEST_START,
            end=BACKTEST_END,
            include_regime=False,
        )
    elif prefer_disk_panels and n_wl > 80:
        logger.info(
            f"rolling_pool: 跳过 registry 整包缓存（|U|={n_wl}，沿用 factor_panels/）"
        )
    return registry


def _build_factor_dataset_lazy(
    *,
    prices: pd.DataFrame,
    financial: pd.DataFrame,
    open_,
    masks,
    volume,
    hold_period: int,
    rebalance_freq: str | None,
    feature_neutralize: bool,
    barra_features: bool,
    special_factors,
    event_overlay: bool,
    deny_special_inject: bool,
    sparse_from_ic: str | None,
    barra_factors,
    industry_map,
    eligible_mask,
    stock_names,
    is_st_current,
    listing_dates,
    delist_dates,
    st_history,
    apply_tradable_filter: bool,
    fwd_return_winsor: bool,
    tradable_limit_mode: str | None,
    exclude_limit_on_signal: bool | None,
    apply_exec_mask: bool | None,
    schedule_df: pd.DataFrame,
    schedule_union_names: list[str],
    registry_kwargs: dict,
    skip_factor_build: bool,
    max_cached: int,
    strict: bool,
    circ_mv,
    total_mv,
    clean_ret,
    prices_raw,
    amount,
    high,
    low,
    market_prices,
    margin,
    moneyflow,
    northbound,
    institution,
):
    """rolling-pool lazy：不拼并集 U 宽表；面板经 RollingPoolPanelStore 按需加载。"""
    from research.rolling_pool.lazy import (
        RollingPoolPanelStore,
        assert_panel_files_exist,
        log_rss,
    )
    from research.rolling_pool.schedule_load import active_factors_by_rebalance
    from utils.rebalance_dates import get_rebalance_dates

    log_rss("lazy_dataset_start")
    logger.info(
        f"rolling_pool lazy: |U|={len(schedule_union_names)} 仅作存在性检查，"
        f"禁止一次性 materialize；LRU max_cached={max_cached}, strict={strict}"
    )
    # strict（默认）：缺面板直接 raise（因子名对不上会静默变 NaN→0 常数列）
    present, missing = assert_panel_files_exist(
        schedule_union_names, allow_missing=not strict,
    )
    if missing and skip_factor_build:
        raise FileNotFoundError(
            f"rolling_pool lazy + skip_factor_build：缺失 {len(missing)} 个面板 "
            f"（示例: {missing[:8]}）"
        )
    if missing and not skip_factor_build:
        logger.warning(
            f"rolling_pool lazy: {len(missing)} 个面板将在首次使用时计算 "
            f"（示例: {missing[:8]}）"
        )
    logger.info(
        f"rolling_pool lazy: disk present={len(present)}/{len(schedule_union_names)}"
    )

    forward_return = build_forward_return(
        prices,
        open_,
        hold_period,
        masks=masks,
        apply_exec_mask=resolve_apply_exec_mask(
            apply_exec_mask, tradable_limit_mode
        ),
    ).astype("float32")

    freq = rebalance_freq or REBALANCE_FREQ
    rebalance_dates = get_rebalance_dates(
        pd.DatetimeIndex(forward_return.index), freq,
    )
    af_map = active_factors_by_rebalance(schedule_df, rebalance_dates.tolist())

    # always-on：Barra-as-features / special packs（常驻 seed，不进 U 并集统计）
    seed_panels: dict[str, pd.DataFrame] = {}
    always_on: list[str] = []

    if barra_features and barra_factors:
        for name, panel in barra_factors.items():
            if panel is None:
                continue
            seed_panels[name] = panel
            always_on.append(name)
        if always_on:
            logger.info(
                f"barra_features (lazy): {len(always_on)} 个 Barra 作为 always-on 特征"
            )

    if deny_special_inject and (special_factors or event_overlay or sparse_from_ic):
        logger.warning(
            "deny_special_inject=True（dynamic 轨道）：忽略 special-factors / "
            f"event_overlay / sparse_from_ic（收到 special_factors={special_factors!r}）"
        )
        sf_req = resolve_special_factors(None, event_overlay=False)
    else:
        sf_req = resolve_special_factors(
            special_factors,
            event_overlay=event_overlay,
            sparse_from_ic=sparse_from_ic,
        )
    if sf_req:
        # 注入到临时 registry，再作为 seed always-on
        tmp: dict = {}
        inject_special_factors(
            tmp, sf_req,
            prices=prices,
            financial=financial,
            circ_mv=circ_mv,
            total_mv=total_mv,
            clean_ret=clean_ret,
            masks=masks,
            prices_raw=prices_raw,
            volume=volume,
            amount=amount,
            open_=open_,
            high=high,
            low=low,
            market_prices=market_prices,
            industry_map=industry_map,
            margin=margin,
            moneyflow=moneyflow,
            northbound=northbound,
            institution=institution,
        )
        for name, panel in tmp.items():
            if panel is None:
                continue
            seed_panels[name] = panel
            if name not in always_on:
                always_on.append(name)

    # 中性化：pool_t 训练需历史调仓日真实残差 → 全 rebalance_dates OLS（非入选日稀疏）
    ind_map_series = None
    neut_weights = None
    do_neut = (
        feature_neutralize
        and barra_factors is not None
        and industry_map is not None
    )
    if do_neut:
        from factors.barra_risk import barra_regression_weights
        ind_map_series = (
            industry_map["sw_l2"]
            if isinstance(industry_map, pd.DataFrame) and "sw_l2" in industry_map.columns
            else industry_map
        )
        # WLS 权重 = √市值，与 IC 纯化（research/ic/barra.py）同口径
        neut_weights = barra_regression_weights(
            prices, circ_mv=circ_mv, total_mv=total_mv,
        )
        logger.info(
            "feature_neutralize (lazy): 残差化推迟到 PanelStore.get；"
            "全调仓日 WLS（权重=√市值；pool_t 历史截面需要真实值）"
        )

    store = RollingPoolPanelStore(
        prices,
        registry_kwargs,
        compute_missing=not skip_factor_build,
        feature_neutralize=do_neut,
        barra_factors=barra_factors if do_neut else None,
        industry_map=ind_map_series if do_neut else None,
        weight_panel=neut_weights if do_neut else None,
        active_dates_by_factor=None,
        rebalance_dates=rebalance_dates,
        max_cached=max_cached,
        seed_panels=seed_panels or None,
        # neut 磁盘缓存键必须区分 horizon + 调仓频率（h20 不得命中 h5 残差面板）
        hold_period=hold_period,
        rebalance_freq=freq,
        strict=strict,
    )

    if apply_tradable_filter:
        from research.ic.universe import (
            build_ic_tradability_mask,
            load_stock_names,
            load_is_st_current,
            load_listing_dates,
            load_delist_dates,
            load_st_history,
        )
        sn = stock_names if stock_names is not None else load_stock_names()
        ist = is_st_current if is_st_current is not None else load_is_st_current()
        ld = listing_dates if listing_dates is not None else load_listing_dates()
        dd = delist_dates if delist_dates is not None else load_delist_dates()
        sth = st_history if st_history is not None else load_st_history()
        tradable = build_ic_tradability_mask(
            prices,
            volume=volume,
            masks=masks,
            stock_names=sn,
            listing_dates=ld,
            delist_dates=dd,
            is_st_current=ist,
            st_history=sth,
            exclude_limit_on_signal=resolve_exclude_limit_on_signal(
                exclude_limit_on_signal, tradable_limit_mode
            ),
        )
        t = tradable.reindex(
            index=forward_return.index, columns=forward_return.columns,
        ).fillna(False)
        forward_return = forward_return.where(t)
        ex_lim = resolve_exclude_limit_on_signal(
            exclude_limit_on_signal, tradable_limit_mode
        )
        limit_desc = (
            "ST/涨跌停/停牌/次新/退市"
            if ex_lim
            else "ST/停牌/次新/退市（research：信号日保留涨跌停）"
        )
        logger.info(
            f"tradable_filter: {int(t.sum().sum())} 个 (date×stock) 可交易格 "
            f"/ {t.shape[0]} 日（{limit_desc}已从标签排除）"
        )

    if eligible_mask is not None:
        em = eligible_mask.reindex(
            index=forward_return.index, columns=forward_return.columns,
        ).fillna(False)
        forward_return = forward_return.where(em)
        logger.info(
            f"eligible_mask 应用: {int(em.sum().sum())} 个 (date×stock) 有效格 "
            f"/ {em.shape[0]} 日"
        )

    if fwd_return_winsor and FWD_RETURN_WINSOR is not None:
        lo, hi = FWD_RETURN_WINSOR
        forward_return = winsorize_forward_return(forward_return, lower=lo, upper=hi)
        logger.info(f"fwd_return_winsor: 截面 [{lo:.0%}, {hi:.0%}] 截尾（与 IC 同函数）")

    log_rss("lazy_dataset_ready")
    return build_ml_dataset(
        {},
        forward_return,
        rebalance_freq=freq,
        active_factors=af_map,
        lazy_rolling_pool=True,
        lazy_store=store,
        always_on_features=always_on or None,
        feature_names=list(schedule_union_names) + [
            n for n in always_on if n not in schedule_union_names
        ],
        rebalance_dates=rebalance_dates.tolist(),
    )


def build_factor_dataset(
    prices: pd.DataFrame,
    financial: pd.DataFrame,
    prices_raw: pd.DataFrame = None,
    volume: pd.DataFrame = None,
    amount: pd.DataFrame = None,
    open_: pd.DataFrame = None,
    high: pd.DataFrame = None,
    low: pd.DataFrame = None,
    clean_ret: pd.DataFrame = None,
    masks: dict = None,
    market_prices: pd.DataFrame = None,
    industry_map: pd.DataFrame = None,
    margin: pd.DataFrame = None,
    moneyflow: pd.DataFrame = None,
    northbound: pd.DataFrame = None,
    institution: pd.DataFrame = None,
    circ_mv: pd.DataFrame = None,
    total_mv: pd.DataFrame = None,
    hold_period: int = 20,
    factor_whitelist: list = None,
    rebalance_freq: str = None,
    use_factor_cache: bool = True,
    skip_factor_build: bool = False,
    rebuild_factor_cache: bool = False,
    feature_neutralize: bool = False,
    barra_features: bool = False,
    special_factors: str | list | None = None,
    event_overlay: bool = False,
    deny_special_inject: bool = False,
    sparse_from_ic: str | None = None,
    regime_cs: bool = False,
    barra_factors: dict[str, pd.DataFrame] | None = None,
    eligible_mask: pd.DataFrame | None = None,
    include_regime: bool = False,
    stock_names: pd.Series | None = None,
    is_st_current: pd.Series | None = None,
    listing_dates: dict[str, pd.Timestamp] | None = None,
    delist_dates: dict[str, pd.Timestamp] | None = None,
    st_history: pd.DataFrame | None = None,
    apply_tradable_filter: bool = True,
    fwd_return_winsor: bool = True,
    rolling_pool_schedule: str | Path | None = None,
    rolling_pool_lazy: bool | None = None,
    rolling_pool_max_cached: int = 160,
    rolling_pool_strict: bool = True,
    tradable_limit_mode: str | None = None,
    exclude_limit_on_signal: bool | None = None,
    apply_exec_mask: bool | None = None,
):
    """
    构建 MLDataset，供 IndustryWalkForwardTrainer 等复用。

    forward_return 定义：
        有 open_（次日开盘价）时 → close[t+N] / open[t+1] - 1
            即信号日收盘后，次日开盘买入，持有 N 日到收盘卖出的真实收益。
            这与实盘执行完全一致（次日开盘手动买入）。
        无 open_ 时 → close[t+N] / close[t] - 1（退化为收收收益，含隔夜跳空）
        经 ``build_forward_return`` 构建；strict 模式下经 masks 屏蔽买日一字涨停 /
        卖日涨跌停；research 默认不做 execution mask（回测层仍拦截）。

    tradable_limit_mode : strict | research
        与 IC 同口径（默认 settings.TRADABLE_LIMIT_MODE）。research 时信号日
        可交易池保留涨跌停股；strict 为旧口径。

    feature_neutralize : bool
        若为 True 且 ``barra_factors`` 与 ``industry_map`` 同时提供，则在因子
        registry 构建完成后、``build_ml_dataset`` 之前，对每个因子面板按
        ``rebalance_dates`` 做截面 OLS 残差化（控制变量 = Barra 9 风格 + 行业
        哑变量），与 IC 筛选阶段 ``research/ic/barra.py`` 用同一套口径。
        Barra / special-factor packs（event、size 等，见
        ``factors.special_factors``）自身特征默认不中性化。

    barra_features : bool
        若为 True 且 ``barra_factors`` 提供，则在 registry 白名单过滤之后把
        Barra 9 风格因子面板 merge 进 registry，使树模型直接以 Barra 为输入
        特征（不做残差化，与 ``feature_neutralize`` 互斥语义：前者把 Barra
        当输入，后者把 Barra 当控制变量去残差化 alpha）。Barra 面板已截面
        z-score（±3σ clip），无需额外标准化。

    special_factors : str | list | None
        特殊因子 pack / 因子名，白名单过滤后 post-merge 注入（不进因子缓存）。
        例：``"event,size,sparse"`` / ``["event", "size"]``。见
        ``factors.special_factors`` / ``docs/SPECIAL_FACTORS.md``。
        IC 路径仍不自动纳入；仅训练 / 打分注入。
        ``sparse`` pack 注入时做方差对齐（供 ridge）；**dynamic 轨道禁止注入**。

    event_overlay : bool
        **Deprecated**：等价于把 ``event`` 并入 ``special_factors``。
        请改用 ``special_factors="event"``。

    deny_special_inject : bool
        若为 True（``run.py --mode dynamic`` / blend-dynamic 的 dynamic 半边），
        忽略 ``special_factors`` / ``event_overlay`` 并 warning。

    sparse_from_ic : str | None
        IC ``selected_factors_h*.json`` 路径；启用 sparse pack 且仅注入
        JSON 中 ``factors_sparse``。

    regime_cs : bool
        **已退役**（原 ``轮动_*`` CS 注入）。True 时仅 warning，不注入。

    include_regime : bool
        **已退役**（原 ``市场*``/``HMM_*`` 广播注入）。True 时仅 warning，不注入。
        仓位控制见 ``--position-regime`` / ``backtest.regime``。

    apply_tradable_filter : bool
        若为 True，用 ``build_ic_tradability_mask``（与 IC 同口径）把不可交易
        格子在 forward_return 上置 NaN，使 ``get_cross_section`` 的 dropna 排除
        ST / 停牌 / 次新 / 退市（research 模式信号日保留涨跌停）。
        stock_names / is_st_current / listing_dates / delist_dates 未传入时从
        stock_list.parquet 自动加载。

    fwd_return_winsor : bool
        若为 True 且 ``FWD_RETURN_WINSOR`` 非 None，在 tradable / eligible mask
        置 NaN 之后、``build_ml_dataset`` 之前，对 forward_return 做截面
        分位截尾（与 IC 共用 ``winsorize_forward_return``）。默认 True。

    rolling_pool_schedule : str | Path | None
        ``research.rolling_pool`` 产出的长表 parquet/csv（date×factor）。
        若提供：并集 U 仅作存在性检查；WF 每期特征列 = 该调仓日 pool_t。
        与 ``--factor-config`` 可并存，schedule 优先。

    rolling_pool_lazy : bool | None
        若为 True（有 schedule 时默认 True）：不一次性加载并集 U；
        WF 每期 ``ensure(pool_t)``（约 ≤50 列）。False 为旧路径（全 U
        materialize，易 OOM，且按日 mask 语义已过时）。None → 有 schedule 时 True。

    rolling_pool_max_cached : int
        lazy store LRU 上限（常驻面板数），默认 160；ensure 时抬到 |pool_t|。

    rolling_pool_strict : bool
        默认 True（fail-fast）：schedule 中的因子缺磁盘面板、或运行时 ``store.get``
        拿不到面板 / 日期对不上 → 直接 raise，禁止整列 NaN→``fillna(0)`` 静默变
        常数特征。False **仅 debug**：退回 warning + NaN 列。
    """
    rk = _registry_kwargs(
        prices, financial, prices_raw, volume, amount,
        open_, high, low, clean_ret, masks,
        market_prices, industry_map, margin, moneyflow,
        northbound, institution,
        circ_mv=circ_mv, total_mv=total_mv,
    )
    ex_lim = resolve_exclude_limit_on_signal(
        exclude_limit_on_signal, tradable_limit_mode
    )
    ex_exec = resolve_apply_exec_mask(apply_exec_mask, tradable_limit_mode)
    if regime_cs:
        logger.warning(
            "regime_cs / --regime-cs 已退役：轮动_* 不再注入 ML X；"
            "行业/风格截面信号请用 alpha 因子（如 行业相对强度）；"
            "仓位控制请用 --position-regime"
        )

    schedule_df = None
    schedule_union_names: list | None = None
    if rolling_pool_schedule:
        from research.rolling_pool.schedule_load import (
            load_pool_schedule, schedule_union,
        )
        schedule_df = load_pool_schedule(rolling_pool_schedule)
        schedule_union_names = schedule_union(schedule_df)
        if factor_whitelist:
            logger.info(
                f"rolling_pool_schedule 优先于 factor_whitelist: "
                f"|U|={len(schedule_union_names)} "
                f"(原 whitelist={len(factor_whitelist)})"
            )
        else:
            logger.info(
                f"rolling_pool_schedule: |U|={len(schedule_union_names)} "
                f"factors, path={rolling_pool_schedule}"
            )
        factor_whitelist = schedule_union_names

    if rolling_pool_lazy is None:
        rolling_pool_lazy = schedule_df is not None
    if rolling_pool_lazy and schedule_df is None:
        logger.warning("rolling_pool_lazy=True 但未提供 schedule，忽略 lazy")
        rolling_pool_lazy = False

    # ── lazy 路径：不 materialize U；仅校验磁盘 / WF 每期 ensure(pool_t) ─────
    if rolling_pool_lazy:
        return _build_factor_dataset_lazy(
            prices=prices,
            financial=financial,
            open_=open_,
            masks=masks,
            volume=volume,
            hold_period=hold_period,
            rebalance_freq=rebalance_freq,
            feature_neutralize=feature_neutralize,
            barra_features=barra_features,
            special_factors=special_factors,
            event_overlay=event_overlay,
            deny_special_inject=deny_special_inject,
            sparse_from_ic=sparse_from_ic,
            barra_factors=barra_factors,
            industry_map=industry_map,
            eligible_mask=eligible_mask,
            stock_names=stock_names,
            is_st_current=is_st_current,
            listing_dates=listing_dates,
            delist_dates=delist_dates,
            st_history=st_history,
            apply_tradable_filter=apply_tradable_filter,
            fwd_return_winsor=fwd_return_winsor,
            tradable_limit_mode=tradable_limit_mode,
            exclude_limit_on_signal=exclude_limit_on_signal,
            apply_exec_mask=apply_exec_mask,
            schedule_df=schedule_df,
            schedule_union_names=schedule_union_names or [],
            registry_kwargs=rk,
            skip_factor_build=skip_factor_build,
            max_cached=rolling_pool_max_cached,
            strict=rolling_pool_strict,
            circ_mv=circ_mv,
            total_mv=total_mv,
            clean_ret=clean_ret,
            prices_raw=prices_raw,
            amount=amount,
            high=high,
            low=low,
            market_prices=market_prices,
            margin=margin,
            moneyflow=moneyflow,
            northbound=northbound,
            institution=institution,
        )

    registry = _load_or_compute_registry(
        factor_whitelist, use_factor_cache, skip_factor_build,
        rebuild_factor_cache, hold_period, rebalance_freq, rk,
        include_regime=include_regime,
        prefer_disk_panels=bool(schedule_df is not None),
    )
    # 与 IC 同口径：research 默认不在标签上屏蔽涨跌停 execution
    forward_return = build_forward_return(
        prices,
        open_,
        hold_period,
        masks=masks,
        apply_exec_mask=ex_exec,
    ).astype("float32")

    # 把 Barra 9 风格因子作为 ML 输入特征 merge 进 registry（不残差化）。
    # 在白名单过滤之后做，避免被白名单（不含 Barra）筛掉。
    if barra_features and barra_factors:
        merged = 0
        for name, panel in barra_factors.items():
            if panel is None or name in registry:
                continue
            registry[name] = panel
            merged += 1
        if merged:
            logger.info(
                f"barra_features: 把 {merged} 个 Barra 风格因子作为 ML 输入"
                f"（不残差化）merge 进 registry → 共 {len(registry)} 个特征"
            )

    # 特殊因子 post-merge（仿 barra_features）：白名单过滤之后注入，
    # 不进 `_load_or_compute_registry` 缓存，避免被白名单筛掉。
    # dynamic 轨道：禁止 special/sparse 注入（ICIR 加权不应混入稀疏事件包）。
    if deny_special_inject and (special_factors or event_overlay or sparse_from_ic):
        logger.warning(
            "deny_special_inject=True（dynamic 轨道）：忽略 special-factors / "
            f"event_overlay / sparse_from_ic（收到 special_factors={special_factors!r}）"
        )
        sf_req = resolve_special_factors(None, event_overlay=False)
    else:
        sf_req = resolve_special_factors(
            special_factors,
            event_overlay=event_overlay,
            sparse_from_ic=sparse_from_ic,
        )
    if sf_req:
        inject_special_factors(
            registry, sf_req,
            prices=prices,
            financial=financial,
            circ_mv=circ_mv,
            total_mv=total_mv,
            clean_ret=clean_ret,
            # 涨跌停类稀疏因子（跌停弱势/涨跌停净强度等）依赖 masks
            masks=masks,
            prices_raw=prices_raw,
            volume=volume,
            amount=amount,
            open_=open_,
            high=high,
            low=low,
            market_prices=market_prices,
            industry_map=industry_map,
            margin=margin,
            moneyflow=moneyflow,
            northbound=northbound,
            institution=institution,
        )

    # 源头 inf 清洗：树模型（lgbm/xgb/cat）原生支持 NaN 分裂，但 XGBoost 对 inf
    # 直接抛 "Input data contains inf"；Barra_Beta = cov/var 在 var=0 时产生 inf，
    # 部分原始因子（Amihud/资金流比率）也可能残留 inf。此处统一在数据集出口把
    # 所有因子面板的 ±inf 替换为 NaN，确保下游任意模型都不会因 inf 失败。
    inf_cleaned = 0
    for name, panel in registry.items():
        if panel is None:
            continue
        # 快速检查：仅当面板含 inf 时才做 replace（避免无 inf 时的全量拷贝）
        try:
            arr = panel.to_numpy(dtype=np.float32, copy=False)
            if not np.isfinite(arr).all() and np.isinf(arr).any():
                registry[name] = panel.replace([np.inf, -np.inf], np.nan)
                inf_cleaned += 1
        except Exception:
            registry[name] = panel.replace([np.inf, -np.inf], np.nan)
            inf_cleaned += 1
    if inf_cleaned:
        logger.info(f"inf 清洗: {inf_cleaned} 个因子面板含 ±inf，已替换为 NaN")

    # rolling-pool：在中性化之前按日 mask（非活跃行已是 NaN → residualize 自然跳过，省算力）
    if schedule_df is not None:
        from research.rolling_pool.schedule_load import apply_schedule_mask
        union_set = set(schedule_union_names or [])
        registry = apply_schedule_mask(registry, schedule_df, only_names=union_set)

    if feature_neutralize and barra_factors is not None and industry_map is not None:
        from factors.barra_risk import barra_regression_weights
        from research.rolling_pool.neut_cache import (
            barra_bundle_sig,
            neut_cache_path,
            neutralize_one_factor,
            save_neut_panel,
            try_load_neut_panel,
        )
        from research.rolling_pool.schedule_load import cs_zscore_sparse_rows
        from utils.rebalance_dates import get_rebalance_dates

        # industry_map 可能是 DataFrame（多列），残差化需要 Series
        ind_map_series = (
            industry_map["sw_l2"]
            if isinstance(industry_map, pd.DataFrame) and "sw_l2" in industry_map.columns
            else industry_map
        )
        # WLS 权重 = √市值，与 IC 纯化（research/ic/barra.py）同口径
        neut_weights = barra_regression_weights(
            prices, circ_mv=circ_mv, total_mv=total_mv,
        )
        freq = rebalance_freq or REBALANCE_FREQ
        rebalance_dates = get_rebalance_dates(
            pd.DatetimeIndex(forward_return.index), freq,
        )

        # rolling-pool 急切 materialize：仅对「该因子曾入选」的调仓日残差化
        # （lazy 路径用全调仓日；缓存键含 dates_sig，二者不会串味）
        active_dates_by_factor: dict[str, pd.DatetimeIndex] | None = None
        if schedule_df is not None:
            from research.rolling_pool.schedule_load import active_factors_by_rebalance
            af_map = active_factors_by_rebalance(schedule_df, rebalance_dates.tolist())
            inv: dict[str, list] = {}
            for d, facs in af_map.items():
                for f in facs:
                    inv.setdefault(f, []).append(d)
            active_dates_by_factor = {
                f: pd.DatetimeIndex(ds) for f, ds in inv.items()
            }

        ctrl_sig = barra_bundle_sig(
            barra_factors,
            industry_map=ind_map_series,
            weight_panel=neut_weights,
        )
        # schedule 稀疏面板用 cs_zscore_sparse_rows；固定池仍用 cross_sectional_zscore
        # （dense 上二者同口径；lazy store 一律 sparse，缓存键含 dates_sig 区分）
        if schedule_df is not None:
            _zscore_fn = cs_zscore_sparse_rows
        else:
            from factors.factor import cross_sectional_zscore as _zscore_fn

        excluded = 0
        n_hit = 0
        n_miss = 0
        new_registry: dict = {}
        n_resid = sum(1 for n in registry if not should_skip_neutralize(n))
        done_resid = 0
        # 逐因子：HIT 则跳过 WLS；MISS 则残差化+re-zscore 后落盘。
        # 清缓存：删 data/processed/factor_panels/factor_panel_neut_*.parquet
        for name, panel in registry.items():
            if should_skip_neutralize(name):
                new_registry[name] = panel
                excluded += 1
                continue
            dates_use = rebalance_dates
            if active_dates_by_factor is not None:
                dates_use = active_dates_by_factor.get(name, pd.DatetimeIndex([]))
            path = neut_cache_path(
                name, prices,
                hold_period=int(hold_period),
                rebalance_freq=str(freq),
                rebalance_dates=dates_use,
                ctrl_sig=ctrl_sig,
            )
            cached = try_load_neut_panel(path, prices=prices, name=name)
            if cached is not None:
                new_registry[name] = cached
                n_hit += 1
            else:
                new_registry[name] = neutralize_one_factor(
                    panel, name,
                    barra_factors=barra_factors,
                    industry_map=ind_map_series,
                    dates_use=dates_use,
                    weight_panel=neut_weights,
                    zscore_fn=_zscore_fn,
                )
                save_neut_panel(path, new_registry[name], name=name)
                n_miss += 1
            done_resid += 1
            if done_resid % 50 == 0:
                logger.info(
                    f"feature_neutralize: {done_resid}/{n_resid} "
                    f"(HIT={n_hit}, MISS={n_miss})"
                )
            # 峰值内存：算完一个即可丢掉原 panel 引用（registry 稍后整体替换）
            registry[name] = None

        registry = new_registry
        logger.info(
            f"feature_neutralize: {len(registry) - excluded} 个因子已 Barra+行业残差化"
            f"（保留 {excluded} 个 Barra/special 特征不中性化）；"
            f"neut cache HIT={n_hit} MISS={n_miss}；"
            f"残差面板已 re-zscore（per-date mean≈0 std≈1）"
        )

    # 与 IC 同口径的可交易池：信号日 ST / 停牌 / 次新 / 退市 → forward_return NaN
    # research 模式保留涨跌停；get_cross_section 的 y.dropna() 自然排除。
    if apply_tradable_filter:
        from research.ic.universe import (
            build_ic_tradability_mask,
            load_stock_names,
            load_is_st_current,
            load_listing_dates,
            load_delist_dates,
            load_st_history,
        )
        sn = stock_names if stock_names is not None else load_stock_names()
        ist = is_st_current if is_st_current is not None else load_is_st_current()
        ld = listing_dates if listing_dates is not None else load_listing_dates()
        dd = delist_dates if delist_dates is not None else load_delist_dates()
        sth = st_history if st_history is not None else load_st_history()
        tradable = build_ic_tradability_mask(
            prices,
            volume=volume,
            masks=masks,
            stock_names=sn,
            listing_dates=ld,
            delist_dates=dd,
            is_st_current=ist,
            st_history=sth,
            exclude_limit_on_signal=ex_lim,
        )
        t = tradable.reindex(
            index=forward_return.index, columns=forward_return.columns,
        ).fillna(False)
        forward_return = forward_return.where(t)
        limit_desc = (
            "ST/涨跌停/停牌/次新/退市"
            if ex_lim
            else "ST/停牌/次新/退市（research：信号日保留涨跌停）"
        )
        logger.info(
            f"tradable_filter: {int(t.sum().sum())} 个 (date×stock) 可交易格 "
            f"/ {t.shape[0]} 日（{limit_desc}已从标签排除）"
        )

    # cap-band / 自定义 universe mask：在 build_ml_dataset 前，把非 eligible 格子置 NaN。
    # get_cross_section 的 dropna 会自然排除这些股票，使训练/预测/IC 都在 cap-band 池内。
    # mask 为 wide bool DataFrame(index=date, columns=code)；None 时不过滤（全市场）。
    if eligible_mask is not None:
        em = eligible_mask.reindex(index=forward_return.index, columns=forward_return.columns).fillna(False)
        forward_return = forward_return.where(em)
        masked_registry = {}
        for name, panel in registry.items():
            masked_registry[name] = panel.where(em.reindex(index=panel.index, columns=panel.columns).fillna(False))
        registry = masked_registry
        n_dates = em.shape[0]
        logger.info(f"eligible_mask 应用: {int(em.sum().sum())} 个 (date×stock) 有效格 / {n_dates} 日")

    # 标签截面截尾：在 tradable / eligible 置 NaN 之后，分位数只在可交易样本上算
    if fwd_return_winsor and FWD_RETURN_WINSOR is not None:
        lo, hi = FWD_RETURN_WINSOR
        forward_return = winsorize_forward_return(forward_return, lower=lo, upper=hi)
        logger.info(f"fwd_return_winsor: 截面 [{lo:.0%}, {hi:.0%}] 截尾（与 IC 同函数）")

    ds_kwargs = {}
    if rebalance_freq is not None:
        ds_kwargs["rebalance_freq"] = rebalance_freq
    if schedule_df is not None:
        from research.rolling_pool.schedule_load import active_factors_by_rebalance
        from utils.rebalance_dates import get_rebalance_dates
        freq = rebalance_freq or REBALANCE_FREQ
        # 与 build_ml_dataset 同口径预览调仓日，写入 active_factors 元数据
        all_dates = sorted(
            set.intersection(*[set(df.index) for df in registry.values()])
            & set(forward_return.index)
        )
        rb_dates = get_rebalance_dates(
            pd.DatetimeIndex(all_dates), freq,
        ).tolist() if all_dates else []
        ds_kwargs["active_factors"] = active_factors_by_rebalance(
            schedule_df, rb_dates,
        )
    return build_ml_dataset(registry, forward_return, **ds_kwargs)


def run(
    prices: pd.DataFrame,
    financial: pd.DataFrame,
    prices_raw: pd.DataFrame = None,
    volume: pd.DataFrame = None,
    amount: pd.DataFrame = None,
    open_: pd.DataFrame = None,
    high: pd.DataFrame = None,
    low: pd.DataFrame = None,
    clean_ret: pd.DataFrame = None,
    masks: dict = None,
    market_prices: pd.DataFrame = None,
    industry_map: pd.DataFrame = None,
    margin: pd.DataFrame = None,
    moneyflow: pd.DataFrame = None,
    northbound: pd.DataFrame = None,
    institution: pd.DataFrame = None,
    circ_mv: pd.DataFrame = None,
    total_mv: pd.DataFrame = None,
    model_types: list = None,
    hold_period: int = 20,
    show_report: bool = False,
    factor_whitelist: list = None,
    train_windows: list = None,
    train_window_units: str = "months",
    val_window: int | None = None,
    artifact_dir=None,
    rebalance_freq: str = None,
    use_factor_cache: bool = True,
    skip_factor_build: bool = False,
    rebuild_factor_cache: bool = False,
    trainer_engine: str = "v2",
    wf_selection: str = "ic_weighted",
    label_mode: str = "cs_zscore",
    ensemble_method: str = "zscore",
    save_models: bool = False,
    objective: str = "regression",
    tag: str | None = None,
    barra_factors: dict[str, pd.DataFrame] | None = None,
    device: str = "cpu",
    tb_prices: pd.DataFrame | None = None,
    tb_open: pd.DataFrame | None = None,
    triple_barrier_params: dict | None = None,
    regime_states: pd.Series | None = None,
    feature_neutralize: bool = False,
    barra_features: bool = False,
    special_factors: str | list | None = None,
    event_overlay: bool = False,
    regime_cs: bool = False,
    eligible_mask: pd.DataFrame | None = None,
    ridge_drop_regime: bool = False,
    include_regime: bool = False,
    stock_names: pd.Series | None = None,
    is_st_current: pd.Series | None = None,
    listing_dates: dict[str, pd.Timestamp] | None = None,
    delist_dates: dict[str, pd.Timestamp] | None = None,
    st_history: pd.DataFrame | None = None,
    apply_tradable_filter: bool = True,
    fwd_return_winsor: bool = True,
    tradable_limit_mode: str | None = None,
    exclude_limit_on_signal: bool | None = None,
    apply_exec_mask: bool | None = None,
    two_stage: bool = False,
    stage2_pool_frac: float = 0.2,
    stage2_lookback: int | None = None,
    save_stage1_cache: bool = False,
    stage1_cache: str | Path | None = None,
    stage1_cache_meta: dict | None = None,
    enable_shap: bool = False,
    shap_top: int = 20,
    shap_max_samples: int = 500,
    shap_max_dates: int = 12,
    rolling_pool_schedule: str | Path | None = None,
    rolling_pool_lazy: bool | None = None,
    rolling_pool_max_cached: int = 160,
    rolling_pool_strict: bool = True,
    long_weight_top: float | None = None,
    long_weight_ratio: float = 0.25,
    long_weight_curve: str = "smooth",
    softlong_floor_slope: float = 0.25,
    retrain_every: int = RETRAIN_EVERY,
) -> tuple[pd.DataFrame, object]:
    """
    训练 ML 策略并返回样本外预测得分。

    model_types: 传 None 使用全部 ["ridge","lgbm","xgb","cat"]，
                 传单个列表如 ["lgbm"] 则只用该模型（不做 ensemble）。
    hold_period: 预测未来 N 日收益，决定 forward_return 的计算窗口。
    show_report: 训练完是否立即展示 IC / 分组净值 / SHAP 图表。
    two_stage: 若 True，在 S1 WF 得分上再跑 in-pool ridge（见 models.wf.two_stage）；
               S2 对池内 label/特征按日做 winsor→cs_zscore 后再 Ridge。
    stage2_pool_frac: S1 截面 top 分位作为 stage-2 候选池（默认 0.2 ≈ Q5）。
    stage2_lookback: stage-2 训练回看调仓期数（默认由 hold_period 推算）。
    save_stage1_cache: 写出 S1 universe cache（scores+pool mask+meta；不含 X/y）。
    stage1_cache: 若提供，跳过 S1 训练，从 cache 读池跑 S2（隐含 two_stage）。
    stage1_cache_meta: 写入 cache 的额外 meta（horizon / factor-config 哈希等）。
    enable_shap: 训练期对最近 OOS 折算 SHAP 并写入 artifact_dir（默认关）。
    shap_top / shap_max_samples / shap_max_dates: SHAP 汇总与采样控制。
    rolling_pool_schedule: rolling-pool 长表路径；见 ``build_factor_dataset``。
    rolling_pool_lazy: 有 schedule 时默认 True；见 ``build_factor_dataset``。
    rolling_pool_strict: 缺面板 fail-fast（默认 True）；见 ``build_factor_dataset``。

    返回 (score_df, trainer)
        score_df: DataFrame(index=调仓日, columns=股票)，越大越优先
        trainer:  训练完成的 WalkForwardTrainer，可用于后续分析
    """
    if model_types is None:
        model_types = MODEL_TYPES

    if stage1_cache is not None:
        two_stage = True

    dataset = build_factor_dataset(
        prices, financial,
        prices_raw=prices_raw, volume=volume, amount=amount,
        open_=open_, high=high, low=low,
        clean_ret=clean_ret, masks=masks,
        market_prices=market_prices, industry_map=industry_map,
        margin=margin, moneyflow=moneyflow,
        northbound=northbound, institution=institution,
        circ_mv=circ_mv, total_mv=total_mv,
        hold_period=hold_period,
        factor_whitelist=factor_whitelist,
        rebalance_freq=rebalance_freq,
        use_factor_cache=use_factor_cache,
        skip_factor_build=skip_factor_build,
        rebuild_factor_cache=rebuild_factor_cache,
        feature_neutralize=feature_neutralize,
        barra_features=barra_features,
        special_factors=special_factors,
        event_overlay=event_overlay,
        regime_cs=regime_cs,
        barra_factors=barra_factors,
        eligible_mask=eligible_mask,
        include_regime=include_regime,
        stock_names=stock_names,
        is_st_current=is_st_current,
        listing_dates=listing_dates,
        delist_dates=delist_dates,
        st_history=st_history,
        apply_tradable_filter=apply_tradable_filter,
        fwd_return_winsor=fwd_return_winsor,
        tradable_limit_mode=tradable_limit_mode,
        exclude_limit_on_signal=exclude_limit_on_signal,
        apply_exec_mask=apply_exec_mask,
        rolling_pool_schedule=rolling_pool_schedule,
        rolling_pool_lazy=rolling_pool_lazy,
        rolling_pool_max_cached=rolling_pool_max_cached,
        rolling_pool_strict=rolling_pool_strict,
    )
    n_feat = len(dataset.feature_names)
    lazy_tag = " [lazy]" if getattr(dataset, "lazy_rolling_pool", False) else ""
    logger.info(f"ML 策略使用 {n_feat} 个因子{lazy_tag}，模型={model_types}")

    trainer_kwargs = dict(
        model_types=model_types,
        rebalance_freq=rebalance_freq or REBALANCE_FREQ,
        train_window_units=train_window_units,
    )
    if train_windows is not None:
        trainer_kwargs["train_windows"] = train_windows
    if val_window is not None:
        trainer_kwargs["val_window"] = val_window
    if artifact_dir is not None:
        trainer_kwargs["artifact_dir"] = artifact_dir
    trainer_kwargs["retrain_every"] = int(retrain_every)

    # Regime-conditional：regime_states 提供时切换到 RegimeConditionalTrainer
    # （子类，与 WalkForwardTrainer 输出格式完全一致；regime_states=None 退化为父类）
    if regime_states is not None:
        logger.info("regime_states 已提供 → 使用 RegimeConditionalTrainer")
        trainer = RegimeConditionalTrainer(
            hold_period=hold_period,
            wf_selection=wf_selection,
            label_mode=label_mode,
            ensemble_method=ensemble_method,
            save_models=save_models,
            objective=objective,
            tag=tag,
            barra_factors=barra_factors,
            industry_map=industry_map,
            device=device,
            prices=tb_prices,
            open_prices=tb_open,
            triple_barrier_params=triple_barrier_params,
            regime_states=regime_states,
            ridge_drop_regime=ridge_drop_regime,
            enable_shap=enable_shap,
            shap_top=shap_top,
            shap_max_samples=shap_max_samples,
            shap_max_dates=shap_max_dates,
            long_weight_top=long_weight_top,
            long_weight_ratio=long_weight_ratio,
            long_weight_curve=long_weight_curve,
            softlong_floor_slope=softlong_floor_slope,
            **trainer_kwargs,
        )
    else:
        trainer = WalkForwardTrainer(
            hold_period=hold_period,
            wf_selection=wf_selection,
            label_mode=label_mode,
            ensemble_method=ensemble_method,
            save_models=save_models,
            objective=objective,
            tag=tag,
            barra_factors=barra_factors,
            industry_map=industry_map,
            device=device,
            prices=tb_prices,
            open_prices=tb_open,
            triple_barrier_params=triple_barrier_params,
            ridge_drop_regime=ridge_drop_regime,
            enable_shap=enable_shap,
            shap_top=shap_top,
            shap_max_samples=shap_max_samples,
            shap_max_dates=shap_max_dates,
            long_weight_top=long_weight_top,
            long_weight_ratio=long_weight_ratio,
            long_weight_curve=long_weight_curve,
            softlong_floor_slope=softlong_floor_slope,
            **trainer_kwargs,
        )
    pool_mask = None
    s1_from_cache = stage1_cache is not None
    if s1_from_cache:
        from models.wf.stage1_cache import load_stage1_cache
        s1_scores, pool_mask, cache_meta = load_stage1_cache(stage1_cache)
        cached_frac = cache_meta.get("pool_frac")
        if cached_frac is not None and abs(float(cached_frac) - float(stage2_pool_frac)) > 1e-9:
            logger.warning(
                f"stage1_cache pool_frac={cached_frac} ≠ --stage2-pool-frac="
                f"{stage2_pool_frac}; using cached pool mask for membership"
            )
        logger.info(
            f"Stage1 skipped: loaded cache from {stage1_cache} "
            f"(dates={len(s1_scores)}, pool_frac={cached_frac})"
        )
        score_df = s1_scores
        trainer.score_df = score_df
        trainer.s1_score_df = s1_scores
    else:
        score_df = trainer.fit_predict(dataset)

    # Persist S1 universe cache (scores + mask + meta; never X/y).
    # Auto-write when two_stage runs a fresh S1; also honor --save-stage1-cache.
    # Must run before S2 overwrites score_df.
    if (
        artifact_dir is not None
        and not s1_from_cache
        and (save_stage1_cache or two_stage)
    ):
        from models.wf.stage1_cache import default_stage1_cache_dir, save_stage1_cache as _save_s1
        cache_dir = default_stage1_cache_dir(artifact_dir)
        meta = dict(stage1_cache_meta or {})
        meta.setdefault("horizon", hold_period)
        meta.setdefault("pool_frac", float(stage2_pool_frac))
        meta.setdefault("feature_neutralize", bool(feature_neutralize))
        meta.setdefault("tag", tag)
        meta.setdefault("model_types", list(model_types))
        _save_s1(
            cache_dir,
            score_df,
            pool_frac=float(stage2_pool_frac),
            meta=meta,
        )

    if two_stage:
        from models.wf.two_stage import apply_two_stage_ridge
        logger.info(
            f"Two-stage enabled: refining S1 scores with in-pool ridge "
            f"(pool_frac={stage2_pool_frac}, lookback={stage2_lookback}"
            f"{', from_cache' if s1_from_cache else ''})"
        )
        s1_scores = score_df.copy() if not s1_from_cache else score_df
        # Persist S1 for ablation / debugging alongside final S2 scores.
        if artifact_dir is not None:
            out = Path(artifact_dir)
            out.mkdir(parents=True, exist_ok=True)
            tag_s1 = tag or "wf"
            s1_scores.to_parquet(out / f"ml_factor_scores_s1_{tag_s1}.parquet")
        score_df = apply_two_stage_ridge(
            dataset,
            s1_scores,
            hold_period=hold_period,
            pool_frac=stage2_pool_frac,
            lookback_periods=stage2_lookback,
            pool_mask=pool_mask,
        )
        trainer.score_df = score_df
        trainer.s1_score_df = s1_scores
        # Recompute IC on S2 in-pool only (finite scores; -inf = out of pool).
        from models.trainer import MIN_STOCKS_PER_DATE
        from models.wf.metrics import spearman_ic
        ic_dict = {}
        for date in score_df.index:
            _, y = dataset.get_cross_section(date)
            if y is None:
                continue
            s = score_df.loc[date]
            s = s[np.isfinite(s.to_numpy(dtype=float))]
            y = y.reindex(s.index).dropna()
            s = s.loc[y.index]
            if len(s) >= MIN_STOCKS_PER_DATE:
                ic_dict[date] = spearman_ic(s.values, y.values)
        trainer.ic_series = pd.Series(ic_dict)
        if artifact_dir is not None:
            tag_s2 = tag or "wf"
            score_df.to_parquet(
                Path(artifact_dir) / f"ml_factor_scores_{tag_s2}.parquet"
            )
            trainer.ic_series.to_csv(
                Path(artifact_dir) / "ic_series.csv", header=True
            )

    if show_report:
        from models.analyzer import MLAnalyzer
        analyzer = MLAnalyzer(trainer)
        analyzer.full_report(prices)

    return score_df, trainer
