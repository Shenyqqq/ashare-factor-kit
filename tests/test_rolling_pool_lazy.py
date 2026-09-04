"""rolling-pool lazy：当期 pool_t 特征语义（不 materialize U，禁止窗内并集）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.rolling_pool.lazy import (
    MissingFactorPanelError,
    RollingPoolPanelStore,
    _neut_cache_path,
    assert_panel_files_exist,
    build_cross_section_from_store,
    pool_features_for_date,
    union_active_factors,
)
from research.rolling_pool.schedule_load import schedule_tag


def test_union_active_factors_multi_date_for_metadata_only():
    af = {
        pd.Timestamp("2020-01-03"): ["f1", "f2"],
        pd.Timestamp("2020-01-10"): ["f2", "f3"],
        pd.Timestamp("2020-01-17"): ["f3", "f4"],
    }
    # 多日并集仅用于元数据 / 存在性检查，不是 WF 特征列
    u = union_active_factors(af, list(af), always_on=["Barra_Size"])
    assert u == ["f1", "f2", "f3", "f4", "Barra_Size"]


def test_pool_features_for_date_is_period_pool():
    af = {
        pd.Timestamp("2020-01-03"): ["f1", "f2"],
        pd.Timestamp("2020-01-10"): ["f2", "f3"],
        pd.Timestamp("2020-01-17"): ["f3", "f4"],
    }
    assert pool_features_for_date(af, "2020-01-10") == ["f2", "f3"]
    assert pool_features_for_date(
        af, pd.Timestamp("2020-01-10"), always_on=["Barra_Size"],
    ) == ["f2", "f3", "Barra_Size"]
    # 不等于窗内 train∪val∪pred 并集
    window = [
        pd.Timestamp("2020-01-03"),
        pd.Timestamp("2020-01-10"),
        pd.Timestamp("2020-01-17"),
    ]
    assert union_active_factors(af, window) == ["f1", "f2", "f3", "f4"]
    assert pool_features_for_date(af, "2020-01-17") == ["f3", "f4"]


def test_cross_section_pool_t_uses_real_values_on_history():
    """pool_t=[a,b] 训历史日：即使用 b 当时未入池，也取面板真实值（不二次 mask）。"""
    idx = pd.DatetimeIndex(["2020-01-03", "2020-01-10"])
    cols = ["s1", "s2", "s3"]
    panel_a = pd.DataFrame(
        [[1.0, 2.0, np.nan], [3.0, 4.0, 5.0]], index=idx, columns=cols, dtype=np.float32,
    )
    panel_b = pd.DataFrame(
        [[9.0, 8.0, 7.0], [1.0, 1.0, 1.0]], index=idx, columns=cols, dtype=np.float32,
    )
    prices = pd.DataFrame(1.0, index=idx, columns=cols)
    store = RollingPoolPanelStore(
        prices, {},
        compute_missing=False,
        max_cached=10,
        seed_panels={"a": panel_a, "b": panel_b},
    )
    fwd = pd.DataFrame(0.01, index=idx, columns=cols, dtype=np.float32)

    # 预测日 pool_t = [a,b]；历史日 2020-01-03 也用同列，取真实值
    X, y = build_cross_section_from_store(
        store, fwd, "2020-01-03", ["a", "b"],
        active_set=None,
    )
    assert X is not None
    assert list(X.columns) == ["a", "b"]
    assert float(X.loc["s1", "b"]) == 9.0
    assert set(X.index) == {"s1", "s2", "s3"}


def test_cross_section_optional_active_set_still_masks():
    idx = pd.DatetimeIndex(["2020-01-03", "2020-01-10"])
    cols = ["s1", "s2", "s3"]
    panel_a = pd.DataFrame(
        [[1.0, 2.0, np.nan], [3.0, 4.0, 5.0]], index=idx, columns=cols, dtype=np.float32,
    )
    panel_b = pd.DataFrame(
        [[9.0, 8.0, 7.0], [1.0, 1.0, 1.0]], index=idx, columns=cols, dtype=np.float32,
    )
    prices = pd.DataFrame(1.0, index=idx, columns=cols)
    store = RollingPoolPanelStore(
        prices, {},
        compute_missing=False,
        max_cached=10,
        seed_panels={"a": panel_a, "b": panel_b},
    )
    fwd = pd.DataFrame(0.01, index=idx, columns=cols, dtype=np.float32)

    X, y = build_cross_section_from_store(
        store, fwd, "2020-01-03", ["a", "b"],
        active_set={"a"},
    )
    assert X is not None
    assert (X["b"] == 0).all()  # fillna 0
    assert set(X.index) == {"s1", "s2"}  # s3 在 a 上是 NaN


# ── A. neut 缓存键必须区分 horizon / 调仓频率 ────────────────────────────────

def _prices(n_dates: int = 4, n_stocks: int = 3) -> pd.DataFrame:
    idx = pd.date_range("2020-01-03", periods=n_dates, freq="W-FRI")
    cols = [f"s{i}" for i in range(n_stocks)]
    return pd.DataFrame(1.0, index=idx, columns=cols)


def test_neut_cache_path_separates_horizon_and_freq():
    px = _prices()
    p_h5 = _neut_cache_path("f1", px, hold_period=5, rebalance_freq="W-FRI")
    p_h20 = _neut_cache_path("f1", px, hold_period=20, rebalance_freq="ME")
    p_h5_me = _neut_cache_path("f1", px, hold_period=5, rebalance_freq="ME")
    assert p_h5 != p_h20, "h5 残差面板绝不可被 h20 命中"
    assert p_h5 != p_h5_me, "调仓频率不同 → 调仓日集合不同 → 缓存键必须不同"
    # 日历 / 控制变量指纹也必须进键
    d1 = pd.DatetimeIndex(["2020-01-03", "2020-01-10"])
    d2 = pd.DatetimeIndex(["2020-01-03", "2020-01-17"])
    p_cal1 = _neut_cache_path(
        "f1", px, hold_period=5, rebalance_freq="W-FRI", rebalance_dates=d1,
    )
    p_cal2 = _neut_cache_path(
        "f1", px, hold_period=5, rebalance_freq="W-FRI", rebalance_dates=d2,
    )
    assert p_cal1 != p_cal2
    # 与历史版本键不同 → 旧 factor_panel_neut_*.parquet 自动失效
    import hashlib

    from research.rolling_pool.lazy import NEUT_CACHE_VERSION, _universe_sig

    legacy_v2 = hashlib.md5(
        f"f1|neut_v2|{_universe_sig(px)}".encode("utf-8"),
    ).hexdigest()[:16]
    legacy_v5 = hashlib.md5(
        f"f1|neut_v5|h5|W-FRI|{_universe_sig(px)}".encode("utf-8"),
    ).hexdigest()[:16]
    assert legacy_v2 not in p_h5.name
    assert legacy_v5 not in p_h5.name
    assert NEUT_CACHE_VERSION == "neut_v6"

def test_store_requires_horizon_when_neut_disk_cache_on():
    px = _prices()
    with pytest.raises(ValueError, match="hold_period"):
        RollingPoolPanelStore(
            px, {}, feature_neutralize=True, neut_disk_cache=True,
            barra_factors={}, industry_map=pd.Series(dtype=object),
        )
    # 显式关掉落盘则允许
    RollingPoolPanelStore(
        px, {}, feature_neutralize=True, neut_disk_cache=False,
    )


def test_store_accepts_horizon_and_freq_like_ml_pipeline():
    """strategies.ml._build_factor_dataset_lazy 的构造参数不应报错。"""
    px = _prices()
    store = RollingPoolPanelStore(
        px, {},
        feature_neutralize=True,
        neut_disk_cache=True,
        hold_period=20,
        rebalance_freq="ME",
    )
    assert store._neut_path("f1") != _neut_cache_path(
        "f1", px, hold_period=5, rebalance_freq="W-FRI",
    )


# ── C. seed / always_on 面板必须过 clean ─────────────────────────────────────

def test_seed_panels_are_cleaned_aligned_and_typed():
    px = _prices(n_dates=3, n_stocks=3)
    # seed 面板故意：多一列、少一行、float64、含 ±inf
    raw = pd.DataFrame(
        [[1.0, np.inf, 2.0, 7.0], [-np.inf, 3.0, 4.0, 8.0]],
        index=px.index[:2],
        columns=["s0", "s1", "s2", "s_extra"],
        dtype=np.float64,
    )
    store = RollingPoolPanelStore(
        px, {}, compute_missing=False, seed_panels={"Barra_Size": raw},
    )
    # 未 get 之前不应直接进缓存
    assert "Barra_Size" not in store
    panel = store.get("Barra_Size")
    assert panel is not None
    assert panel.index.equals(px.index) and list(panel.columns) == list(px.columns)
    assert panel.dtypes.unique().tolist() == [np.dtype(np.float32)]
    assert not np.isinf(panel.to_numpy(dtype=np.float32)).any()
    assert np.isnan(panel.iloc[0]["s1"]) and np.isnan(panel.iloc[1]["s0"])


def test_seed_skip_neutralize_names_are_not_double_zscored():
    px = _prices(n_dates=2, n_stocks=3)
    raw = pd.DataFrame(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        index=px.index, columns=px.columns, dtype=np.float32,
    )
    store = RollingPoolPanelStore(
        px, {},
        compute_missing=False,
        feature_neutralize=True,
        neut_disk_cache=False,
        barra_factors={"Barra_Size": raw},
        industry_map=pd.Series("A", index=px.columns),
        seed_panels={"Barra_Size": raw},
    )
    out = store.get("Barra_Size")
    # Barra_* 跳过残差化 / re-zscore：数值原样（只做 dtype/inf 清洗）
    assert np.allclose(out.to_numpy(), raw.to_numpy())


# ── D. 因子名对不上 → fail-fast，禁止静默 0 列 ──────────────────────────────

def test_strict_store_raises_on_missing_panel():
    px = _prices()
    store = RollingPoolPanelStore(px, {}, compute_missing=False, strict=True)
    with pytest.raises(MissingFactorPanelError):
        store.get("因子名对不上_xxx")
    # 第二次仍 raise（不因 _failed 缓存而静默）
    with pytest.raises(MissingFactorPanelError):
        store.get("因子名对不上_xxx")


def test_non_strict_store_warns_and_returns_none():
    px = _prices()
    store = RollingPoolPanelStore(px, {}, compute_missing=False, strict=False)
    assert store.get("因子名对不上_xxx") is None


def test_strict_cross_section_raises_instead_of_nan_column():
    idx = pd.DatetimeIndex(["2020-01-03", "2020-01-10"])
    cols = ["s1", "s2"]
    px = pd.DataFrame(1.0, index=idx, columns=cols)
    panel_a = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]], index=idx, columns=cols, dtype=np.float32,
    )
    fwd = pd.DataFrame(0.01, index=idx, columns=cols, dtype=np.float32)
    store = RollingPoolPanelStore(
        px, {}, compute_missing=False, strict=True, seed_panels={"a": panel_a},
    )
    with pytest.raises(MissingFactorPanelError):
        build_cross_section_from_store(store, fwd, idx[0], ["a", "missing_b"])


def test_assert_panel_files_exist_raises_by_default():
    with pytest.raises(MissingFactorPanelError):
        assert_panel_files_exist(["绝不存在的因子_zzz"])
    present, missing = assert_panel_files_exist(
        ["绝不存在的因子_zzz"], allow_missing=True,
    )
    assert present == [] and missing == ["绝不存在的因子_zzz"]


# ── sticky LRU：跨期保留，减少重复读盘 ──────────────────────────────────────

class _CountingStore(RollingPoolPanelStore):
    """用内存面板冒充磁盘，统计 ``_load_raw``（=读盘）次数。"""

    def __init__(self, panels: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._panels = panels
        self.n_raw_loads = 0

    def _load_raw(self, name):
        if name not in self._panels:
            return None
        self.n_raw_loads += 1
        return self._panels[name].copy()


def _counting_store(names, px, **kwargs) -> _CountingStore:
    rs = np.random.RandomState(7)
    panels = {
        n: pd.DataFrame(
            rs.randn(len(px.index), len(px.columns)).astype(np.float32),
            index=px.index, columns=px.columns,
        )
        for n in names
    }
    return _CountingStore(panels, px, {}, compute_missing=False, **kwargs)


def test_sticky_lru_overlapping_pools_hit_cache():
    px = _prices(n_dates=4, n_stocks=3)
    store = _counting_store(["a", "b", "c", "d"], px, max_cached=8)

    store.ensure(["a", "b", "c"])
    assert store.n_raw_loads == 3
    # 第二期与上期重叠 a,b → 只应新读 d
    store.ensure(["a", "b", "d"])
    assert store.n_raw_loads == 4, "重叠因子不应重复读盘（sticky LRU）"
    # 第三期回到首期池 → 全命中（c 未被淘汰，容量够）
    store.ensure(["a", "b", "c"])
    assert store.n_raw_loads == 4


def test_lru_still_bounded_by_max_cached():
    px = _prices(n_dates=4, n_stocks=3)
    store = _counting_store(["a", "b", "c", "d"], px, max_cached=8)
    store.max_cached = 2  # 模拟 --rolling-pool-max-cached 压内存
    store.ensure(["a", "b"])
    assert len(store) == 2
    store.ensure(["c", "d"])
    assert len(store) == 2, "常驻数必须受 max_cached 封顶，不能无限涨"
    # a/b 已被淘汰 → 再要就得重读
    store.ensure(["a", "b"])
    assert store.n_raw_loads == 6


def test_walk_forward_lazy_does_not_reload_every_period():
    """端到端：固定池跑完整 WF，每个因子只读盘一次（此前每期 release 会读 N 次）。"""
    from models.trainer import WalkForwardTrainer, build_ml_dataset
    from utils.rebalance_dates import get_rebalance_dates

    rs = np.random.RandomState(0)
    idx = pd.date_range("2018-01-05", periods=160, freq="B")
    cols = [f"s{i:02d}" for i in range(30)]
    px = pd.DataFrame(1.0, index=idx, columns=cols)
    names = ["f0", "f1", "f2", "f3"]
    store = _counting_store(names, px, max_cached=16)
    fwd = pd.DataFrame(
        rs.randn(len(idx), len(cols)).astype(np.float32) * 0.02,
        index=idx, columns=cols,
    )
    rb = get_rebalance_dates(pd.DatetimeIndex(idx), "W-FRI")
    active = {pd.Timestamp(d): list(names) for d in rb}

    ds = build_ml_dataset(
        {}, fwd, rebalance_freq="W-FRI", active_factors=active,
        lazy_rolling_pool=True, lazy_store=store,
        feature_names=list(names), rebalance_dates=rb.tolist(),
    )
    trainer = WalkForwardTrainer(
        model_types=["ridge"], train_windows=[8], train_window_units="periods",
        rebalance_freq="W-FRI", hold_period=5, tag="sticky_lru_test",
    )
    scores = trainer.fit_predict(ds)
    assert not scores.empty
    n_periods = scores.shape[0]
    assert n_periods > 3
    assert store.n_raw_loads == len(names), (
        f"sticky LRU 失效：{n_periods} 期读盘 {store.n_raw_loads} 次，"
        f"应为 {len(names)} 次"
    )


# ── E. 产物可辨识性 ─────────────────────────────────────────────────────────

def test_schedule_tag_is_stable_and_distinguishes_files():
    t1 = schedule_tag("research/output/rolling_pool_schedule_h5_ic_corr_only.parquet")
    t2 = schedule_tag("research/output/rolling_pool_schedule_h20.parquet")
    assert t1.startswith("_rp") and t2.startswith("_rp")
    assert t1 != t2
    assert t1 == schedule_tag(
        "research/output/rolling_pool_schedule_h5_ic_corr_only.parquet",
    )
    assert "h5iccorronly" in t1
    # 文件名安全：无路径分隔符 / 点
    assert not set(t1) & set("/\\. ")
