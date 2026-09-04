"""
tests/test_special_factors.py

统一特殊因子注入：parse / inject / neutralize skip / CLI alias（event_overlay）。
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
import pytest

from factors.factor import EVENT_OVERLAY_FACTOR_NAMES
from factors.factor_size_alpha import SIZE_ALPHA_FACTOR_NAMES
from factors.special_factors import (
    SPECIAL_FACTOR_PACKS,
    inject_special_factors,
    resolve_special_factors,
    should_skip_neutralize,
)


EVENT_NAME = "业绩预告_超预期"
DATES = pd.date_range("2024-01-02", periods=40, freq="B")
CODES = ["000001", "000002", "000003"]


def _prices() -> pd.DataFrame:
    base = np.linspace(10.0, 20.0, len(DATES))
    return pd.DataFrame(
        {c: base * (1.0 + 0.01 * i) for i, c in enumerate(CODES)},
        index=DATES,
    )


def test_packs_contain_canonical_names():
    assert EVENT_OVERLAY_FACTOR_NAMES <= SPECIAL_FACTOR_PACKS["event"].factor_names
    assert SIZE_ALPHA_FACTOR_NAMES <= SPECIAL_FACTOR_PACKS["size"].factor_names


def test_resolve_packs_and_aliases():
    req = resolve_special_factors("event,size")
    assert req.packs == ("event", "size")
    assert req.names is None
    assert req.tag_suffix() == "_event_size"

    req2 = resolve_special_factors("overlay,市值")
    assert req2.packs == ("event", "size")


def test_resolve_inject_factors_alias_order():
    req = resolve_special_factors(["size", "event"])
    assert req.packs == ("size", "event")
    assert req.tag_suffix() == "_size_event"


def test_resolve_factor_name_subset():
    req = resolve_special_factors("对数市值")
    assert req.packs == ("size",)
    assert req.names == frozenset({"对数市值"})


def test_resolve_pack_plus_name_keeps_full_pack():
    req = resolve_special_factors("event,对数市值")
    assert set(req.packs) == {"event", "size"}
    assert EVENT_NAME in req.all_factor_names()
    assert "对数市值" in req.all_factor_names()
    # event 为 pack 级 → 全量；size 仅显式名
    assert req.names is not None
    assert EVENT_NAME in req.names
    assert "市值分位" not in req.names


def test_event_overlay_deprecated_maps_to_event():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        req = resolve_special_factors(None, event_overlay=True)
    assert req.packs == ("event",)
    assert any(issubclass(x.category, DeprecationWarning) for x in w)


def test_unknown_token_raises():
    with pytest.raises(ValueError, match="未知"):
        resolve_special_factors("not_a_pack")


def test_should_skip_neutralize():
    assert should_skip_neutralize("Barra_Size")
    assert should_skip_neutralize(EVENT_NAME)
    assert should_skip_neutralize("对数市值")
    assert not should_skip_neutralize("动量_20d")


@pytest.fixture
def patched_registry(monkeypatch):
    import strategies.ml as ml

    prices = _prices()
    fake_alpha = {"动量_20d": prices.copy().astype("float32")}
    event_panel = pd.DataFrame(1.0, index=DATES, columns=CODES, dtype="float32")
    size_panel = pd.DataFrame(0.5, index=DATES, columns=CODES, dtype="float32")

    monkeypatch.setattr(
        ml, "_load_or_compute_registry",
        lambda *a, **k: dict(fake_alpha),
    )
    def _fake_size(*a, factor_names=None, **k):
        want = set(factor_names) if factor_names is not None else set(SIZE_ALPHA_FACTOR_NAMES)
        want &= SIZE_ALPHA_FACTOR_NAMES
        out = {}
        for n in want:
            out[n] = size_panel if n == "对数市值" else size_panel * 2
        return out

    monkeypatch.setattr(
        "factors.special_factors.get_event_overlay_factors",
        lambda prices, factor_names=None: {EVENT_NAME: event_panel},
    )
    monkeypatch.setattr(
        "factors.special_factors.get_size_alpha_factors",
        _fake_size,
    )
    return prices, event_panel, size_panel


def test_inject_event_and_size(patched_registry):
    import strategies.ml as ml

    prices, _, _ = patched_registry
    ds = ml.build_factor_dataset(
        prices, pd.DataFrame(),
        hold_period=5,
        special_factors="event,size",
        apply_tradable_filter=False,
        fwd_return_winsor=False,
        use_factor_cache=False,
        include_regime=False,
    )
    assert EVENT_NAME in ds.feature_names
    assert "对数市值" in ds.feature_names
    assert "动量_20d" in ds.feature_names


def test_inject_off_excludes_specials(patched_registry):
    import strategies.ml as ml

    prices, _, _ = patched_registry
    ds = ml.build_factor_dataset(
        prices, pd.DataFrame(),
        hold_period=5,
        special_factors=None,
        event_overlay=False,
        apply_tradable_filter=False,
        fwd_return_winsor=False,
        use_factor_cache=False,
        include_regime=False,
    )
    assert EVENT_NAME not in ds.feature_names
    assert "对数市值" not in ds.feature_names


def test_event_overlay_alias_still_injects(patched_registry):
    import strategies.ml as ml

    prices, _, _ = patched_registry
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        ds = ml.build_factor_dataset(
            prices, pd.DataFrame(),
            hold_period=5,
            event_overlay=True,
            apply_tradable_filter=False,
            fwd_return_winsor=False,
            use_factor_cache=False,
            include_regime=False,
        )
    assert EVENT_NAME in ds.feature_names


def test_feature_neutralize_skips_specials(monkeypatch, patched_registry):
    import strategies.ml as ml

    prices, _, _ = patched_registry
    monkeypatch.setenv("FACTOR_CACHE_DISABLE", "1")

    monkeypatch.setattr(
        "factors.factor.cross_sectional_zscore",
        lambda panel: panel,
    )

    def fake_residualize(panel, *a, **k):
        return panel * 2.0

    monkeypatch.setattr("models.wf.labels.residualize_panel", fake_residualize)

    barra = {
        "Barra_Size": prices.copy().astype("float32"),
        "Barra_Beta": prices.copy().astype("float32"),
    }
    industry = pd.Series(
        ["银行", "银行", "地产"], index=CODES, name="sw_l2",
    )

    ds = ml.build_factor_dataset(
        prices, pd.DataFrame(),
        hold_period=5,
        special_factors="event,size",
        feature_neutralize=True,
        barra_factors=barra,
        industry_map=industry,
        rebalance_freq="W-FRI",
        apply_tradable_filter=False,
        fwd_return_winsor=False,
        use_factor_cache=False,
        include_regime=False,
    )
    assert np.allclose(ds.factor_panel[EVENT_NAME].to_numpy(), 1.0)
    assert np.allclose(ds.factor_panel["对数市值"].to_numpy(), 0.5)
    assert np.allclose(
        ds.factor_panel["动量_20d"].to_numpy(), prices.to_numpy() * 2.0,
    )


def test_inject_special_factors_direct(monkeypatch):
    prices = _prices()
    registry: dict = {}
    event_panel = pd.DataFrame(1.0, index=DATES, columns=CODES)
    req = resolve_special_factors("event")
    monkeypatch.setattr(
        "factors.special_factors.get_event_overlay_factors",
        lambda prices, factor_names=None: {EVENT_NAME: event_panel},
    )
    merged = inject_special_factors(registry, req, prices=prices)
    assert merged == [EVENT_NAME]
    assert EVENT_NAME in registry
    assert req.tag_suffix() == "_event"


def test_cli_special_factors_and_deprecated_alias():
    """argparse：--special-factors / --inject-factors / --event-overlay。"""
    parser = argparse.ArgumentParser()
    # 复用 run 里同类参数定义（抽一段最小 parser）
    parser.add_argument(
        "--special-factors", "--inject-factors",
        dest="special_factors", default=None,
    )
    parser.add_argument(
        "--event-overlay", action="store_true", default=False,
    )
    args = parser.parse_args(["--inject-factors", "event,size"])
    assert args.special_factors == "event,size"

    args2 = parser.parse_args(["--special-factors", "size", "--event-overlay"])
    req = resolve_special_factors(
        args2.special_factors, event_overlay=args2.event_overlay,
        warn_deprecated=False,
    )
    assert req.packs == ("size", "event")
    assert req.tag_suffix() == "_size_event"
