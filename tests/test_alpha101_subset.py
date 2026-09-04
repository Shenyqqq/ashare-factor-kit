"""Alpha101 白名单子集计算：只跑请求的 WQ，不整包 ~82 个。"""
from __future__ import annotations

import numpy as np
import pandas as pd

import factors.factor_alpha101 as a101


def _tiny_ohlcv(n_dates: int = 80, n_stocks: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-04", periods=n_dates, freq="B")
    codes = [f"S{i:03d}" for i in range(n_stocks)]
    close = pd.DataFrame(
        10 + rng.uniform(0, 5, (n_dates, n_stocks)).cumsum(axis=0),
        index=dates, columns=codes,
    )
    open_ = close * (1 + rng.uniform(-0.01, 0.01, close.shape))
    high = np.maximum(close, open_) * (1 + rng.uniform(0, 0.01, close.shape))
    low = np.minimum(close, open_) * (1 - rng.uniform(0, 0.01, close.shape))
    volume = pd.DataFrame(
        rng.uniform(1e5, 1e6, close.shape), index=dates, columns=codes,
    )
    amount = volume * close
    clean_ret = close.pct_change()
    return close, open_, high, low, volume, amount, clean_ret


def test_get_alpha101_factors_subset_only_invokes_requested(monkeypatch):
    """请求 {"WQ_001","WQ_013"} 时只调用这两个函数，返回恰好这 2 个。"""
    close, open_, high, low, volume, amount, clean_ret = _tiny_ohlcv()
    invoked: list[str] = []

    def _spy(name: str, orig):
        def wrapped(*args, **kwargs):
            invoked.append(name)
            return orig(*args, **kwargs)
        return wrapped

    # 精选 10（模块级直接引用）
    curated = {
        "WQ_001": a101.wq_alpha001,
        "WQ_002": a101.wq_alpha002,
        "WQ_006": a101.wq_alpha006,
        "WQ_007": a101.wq_alpha007,
        "WQ_012": a101.wq_alpha012,
        "WQ_028": a101.wq_alpha028,
        "WQ_034": a101.wq_alpha034,
        "WQ_053": a101.wq_alpha053,
        "WQ_061": a101.wq_alpha061,
        "WQ_101": a101.wq_alpha101,
    }
    for name, fn in curated.items():
        monkeypatch.setattr(a101, fn.__name__, _spy(name, fn))

    # _NEW_ALPHA_FUNCS 内的函数（含 WQ_013）
    spied_new = {name: _spy(name, fn) for name, fn in a101._NEW_ALPHA_FUNCS.items()}
    monkeypatch.setattr(a101, "_NEW_ALPHA_FUNCS", spied_new)

    wanted = {"WQ_001", "WQ_013"}
    out = a101.get_alpha101_factors(
        prices=close, open_=open_, high=high, low=low,
        volume=volume, amount=amount, clean_ret=clean_ret,
        factor_names=wanted,
    )

    assert set(out.keys()) == wanted
    assert set(invoked) == wanted
    assert len(invoked) == 2


def test_get_alpha101_factors_none_still_full():
    """factor_names=None 时仍全量计算（IC 全库扫描路径）。"""
    close, open_, high, low, volume, amount, clean_ret = _tiny_ohlcv()
    out = a101.get_alpha101_factors(
        prices=close, open_=open_, high=high, low=low,
        volume=volume, amount=amount, clean_ret=clean_ret,
        factor_names=None,
    )
    # 全量应远多于白名单 2 个；允许个别 alpha 因数据/数值失败而缺失
    assert len(out) > 50
    assert "WQ_001" in out
    assert "WQ_013" in out
