"""
tests/test_event_overlay.py

最小单测：``event_overlay=True``（deprecated → ``special_factors=event``）时把
EVENT_OVERLAY_FACTOR_NAMES（当前仅「业绩预告_超预期」）post-merge 进 ML
registry；False 时不含。feature_neutralize 时 event 豁免残差化。
更完整覆盖见 ``tests/test_special_factors.py``。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factors.factor import EVENT_OVERLAY_FACTOR_NAMES
from research.ic.cli import _is_ic_skippable


EVENT_NAME = "业绩预告_超预期"
DATES = pd.date_range("2024-01-02", periods=40, freq="B")
CODES = ["000001", "000002", "000003"]


def _prices() -> pd.DataFrame:
    base = np.linspace(10.0, 20.0, len(DATES))
    return pd.DataFrame(
        {c: base * (1.0 + 0.01 * i) for i, c in enumerate(CODES)},
        index=DATES,
    )


@pytest.fixture
def patched_registry(monkeypatch):
    """跳过真实因子计算；mock event overlay 面板。"""
    import strategies.ml as ml

    prices = _prices()
    fake_alpha = {"动量_20d": prices.copy().astype("float32")}
    event_panel = pd.DataFrame(1.0, index=DATES, columns=CODES, dtype="float32")

    monkeypatch.setattr(
        ml, "_load_or_compute_registry",
        lambda *a, **k: dict(fake_alpha),
    )
    monkeypatch.setattr(
        "factors.special_factors.get_event_overlay_factors",
        lambda prices, factor_names=None: {EVENT_NAME: event_panel},
    )
    return prices, event_panel


def test_event_name_in_canonical_set():
    assert EVENT_NAME in EVENT_OVERLAY_FACTOR_NAMES


def test_ic_skippable_documents_event_overlay():
    assert _is_ic_skippable(EVENT_NAME)
    assert _is_ic_skippable("市场_牛熊")
    assert not _is_ic_skippable("动量_20d")


def test_event_overlay_true_merges_into_registry(patched_registry):
    import warnings
    import strategies.ml as ml

    prices, _ = patched_registry
    financial = pd.DataFrame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        ds = ml.build_factor_dataset(
            prices, financial,
            hold_period=5,
            event_overlay=True,
            apply_tradable_filter=False,
            fwd_return_winsor=False,
            use_factor_cache=False,
            include_regime=False,
        )
    assert EVENT_NAME in ds.feature_names
    assert "动量_20d" in ds.feature_names


def test_event_overlay_false_excludes_event(patched_registry):
    import strategies.ml as ml

    prices, _ = patched_registry
    financial = pd.DataFrame()
    ds = ml.build_factor_dataset(
        prices, financial,
        hold_period=5,
        event_overlay=False,
        apply_tradable_filter=False,
        fwd_return_winsor=False,
        use_factor_cache=False,
        include_regime=False,
    )
    assert EVENT_NAME not in ds.feature_names
    assert "动量_20d" in ds.feature_names


def test_feature_neutralize_exempts_event(monkeypatch, patched_registry):
    """event 因子应跳过 residualize_panel（与市场/HMM/Barra 同口径）。"""
    import strategies.ml as ml

    prices, event_panel = patched_registry
    financial = pd.DataFrame()
    residualized: list[str] = []
    monkeypatch.setenv("FACTOR_CACHE_DISABLE", "1")

    def fake_residualize(panel, *a, **k):
        # 通过 identity 标记：调用方传哪个 panel 我们记名字较难，
        # 改为返回 panel * 2，再断言 event 面板未被放大。
        residualized.append(id(panel))
        return panel * 2.0

    monkeypatch.setattr("models.wf.labels.residualize_panel", fake_residualize)
    # 跳过 re-zscore 对断言的干扰：保留 identity 检查用原值
    monkeypatch.setattr(
        "factors.factor.cross_sectional_zscore",
        lambda panel: panel,
    )

    barra = {
        "Barra_Size": prices.copy().astype("float32"),
        "Barra_Beta": prices.copy().astype("float32"),
    }
    industry = pd.Series(
        ["银行", "银行", "地产"], index=CODES, name="sw_l2",
    )

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        ds = ml.build_factor_dataset(
            prices, financial,
            hold_period=5,
            event_overlay=True,
            feature_neutralize=True,
            barra_factors=barra,
            industry_map=industry,
            rebalance_freq="W-FRI",
            apply_tradable_filter=False,
            fwd_return_winsor=False,
            use_factor_cache=False,
            include_regime=False,
        )
    assert EVENT_NAME in ds.feature_names
    # event 面板应仍为原始 1.0（未 *2）；alpha 因子应被 residualize → *2
    event_out = ds.factor_panel[EVENT_NAME]
    alpha_out = ds.factor_panel["动量_20d"]
    assert np.allclose(event_out.to_numpy(), 1.0)
    assert np.allclose(alpha_out.to_numpy(), prices.to_numpy() * 2.0)
