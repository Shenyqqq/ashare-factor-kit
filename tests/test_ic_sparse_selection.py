"""IC 多轨筛选：新兴 / 衰减标注 / 风格逆转 / 稀疏胜率+截面胜率 / dynamic 拒注入。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factors.sparse_factors import (
    CAT_DECAYED,
    CAT_DENSE,
    CAT_EMERGING,
    CAT_REVERSAL,
    CAT_SPARSE,
    SPARSE_FACTOR_NAMES,
    variance_align_panel,
)
from factors.special_factors import (
    SPECIAL_FACTOR_PACKS,
    resolve_special_factors,
)
from research.ic.selection import (
    _dedup_emerging_by_ic_corr,
    _dedup_sparse_by_corr,
    _lookback_periods,
    _resolve_window_ic,
    evaluate_decay_gate,
    evaluate_decay_label,
    evaluate_emerging,
    evaluate_style_reversal,
    select_factors_multi_track,
    select_sparse_factors,
)
from research.ic.statistics import (
    ic_direction_sign,
    ic_payoff_ratio,
    recent_past_icir_retention,
    style_reversal_fraction,
    trigger_cs_payoff,
    win_rates,
)


def _ic_series(values, start="2020-01-03", freq="W-FRI"):
    idx = pd.date_range(start, periods=len(values), freq=freq)
    return pd.Series(values, index=idx, dtype=float)


def _summary_row(ic_mean, icir, t=3.0, nw_t=3.0, pos_rate=0.55, n=100):
    return {
        "IC均值": ic_mean,
        "ICIR": icir,
        "t统计量": t,
        "NW_t统计量": nw_t,
        "胜率": pos_rate,
        "正IC占比": pos_rate,
        "负IC占比": 1.0 - pos_rate,
        "样本数": n,
        "同向年份占比": np.nan,
        "IC滚动ICIR": np.nan,
        "最差12期IC均值": np.nan,
    }


def test_sparse_pack_registered():
    assert "sparse" in SPECIAL_FACTOR_PACKS
    assert SPECIAL_FACTOR_PACKS["sparse"].variance_align is True
    assert "龙虎榜连续上榜" in SPARSE_FACTOR_NAMES
    assert "开板反转_5d" in SPARSE_FACTOR_NAMES
    assert "业绩预告_超预期" in SPARSE_FACTOR_NAMES
    req = resolve_special_factors("sparse")
    assert req.packs == ("sparse",)


def test_ic_payoff_ratio_legacy():
    # 旧公式保留兼容；稀疏轨已改用 trigger_cs_payoff
    ic = _ic_series([0.04, 0.04, -0.02, -0.02])
    assert abs(ic_payoff_ratio(ic) - 2.0) < 1e-9


def test_trigger_cs_payoff():
    dates = pd.date_range("2024-01-05", periods=4, freq="W-FRI")
    codes = ["a", "b", "c", "d", "e", "f"]
    # 日1-3：f>0 股票收益高于截面均值；日4：反之
    f = pd.DataFrame(0.0, index=dates, columns=codes)
    y = pd.DataFrame(0.0, index=dates, columns=codes)
    for dt in dates[:3]:
        f.loc[dt, ["a", "b", "c"]] = 1.0
        y.loc[dt, ["a", "b", "c"]] = 0.03
        y.loc[dt, ["d", "e", "f"]] = -0.01
    f.loc[dates[3], ["a", "b", "c"]] = 1.0
    y.loc[dates[3], ["a", "b", "c"]] = -0.02
    y.loc[dates[3], ["d", "e", "f"]] = 0.02
    stats = trigger_cs_payoff(f, y, min_trigger=3, direction=1.0)
    assert stats["n_days"] == 4
    assert abs(stats["payoff_hit"] - 0.75) < 1e-9


def _neg_ic_payoff_panels():
    """负 IC 场景：低因子值侧收益更高（默认 min_trigger=5，每侧 ≥5 只）。"""
    dates = pd.date_range("2024-01-05", periods=4, freq="W-FRI")
    low = [f"l{i}" for i in range(5)]
    high = [f"h{i}" for i in range(5)]
    codes = low + high
    f = pd.DataFrame(0.0, index=dates, columns=codes)
    y = pd.DataFrame(0.0, index=dates, columns=codes)
    for dt in dates[:3]:
        f.loc[dt, low] = -1.0
        f.loc[dt, high] = 1.0
        y.loc[dt, low] = 0.03
        y.loc[dt, high] = -0.01
    f.loc[dates[3], low] = -1.0
    f.loc[dates[3], high] = 1.0
    y.loc[dates[3], low] = -0.02
    y.loc[dates[3], high] = 0.02
    return f, y


def test_trigger_cs_payoff_negative_ic_direction():
    """负 IC：触发侧为 f<0；unsigned f>0 会得到相反/失败的 hit。"""
    f, y = _neg_ic_payoff_panels()
    signed = trigger_cs_payoff(f, y, direction=-1.0)
    unsigned = trigger_cs_payoff(f, y, direction=1.0)
    assert signed["n_days"] == 4
    assert abs(signed["payoff_hit"] - 0.75) < 1e-9
    # unsigned f>0 触发侧是高因子值 → hit 反过来 = 0.25
    assert abs(unsigned["payoff_hit"] - 0.25) < 1e-9


def test_trigger_cs_payoff_respects_tradable_mask():
    """edge_t 截面宇宙必须与 IC tradable 同口径；不可交易股不得拉高 CS 均值。"""
    dates = pd.date_range("2024-01-05", periods=3, freq="W-FRI")
    # 可交易：3 触发高收益 + 2 非触发低收益；ST：5 只极高收益非触发
    # 有 mask：trig > CS(good) → hit=1；无 mask：CS 被 ST 抬高 → hit=0
    good_trig = [f"t{i}" for i in range(3)]
    good_rest = [f"r{i}" for i in range(2)]
    st = [f"s{i}" for i in range(5)]
    codes = good_trig + good_rest + st
    f = pd.DataFrame(0.0, index=dates, columns=codes)
    y = pd.DataFrame(0.0, index=dates, columns=codes)
    tradable = pd.DataFrame(True, index=dates, columns=codes)
    tradable.loc[:, st] = False
    for dt in dates:
        f.loc[dt, good_trig] = 1.0
        f.loc[dt, good_rest] = -1.0
        f.loc[dt, st] = -1.0
        y.loc[dt, good_trig] = 0.05
        y.loc[dt, good_rest] = 0.00
        y.loc[dt, st] = 0.20
    masked = trigger_cs_payoff(f, y, min_trigger=3, direction=1.0, tradable=tradable)
    unmasked = trigger_cs_payoff(f, y, min_trigger=3, direction=1.0)
    assert masked["n_days"] == 3
    assert abs(masked["payoff_hit"] - 1.0) < 1e-9
    assert abs(unmasked["payoff_hit"] - 0.0) < 1e-9


def test_payoff_dates_align_ic_not_rebalance():
    """payoff 必须用 IC 有效日，禁止只在 rebalance_dates 上算。"""
    # 日频 10 天；「调仓日」仅取 2 天；IC 日 = 全部 10 天
    # 每侧 ≥5 只（select_sparse 默认 min_trigger=5）
    dates = pd.date_range("2024-01-02", periods=10, freq="B")
    rebalance = dates[[4, 9]]  # 模拟调仓日子集
    trig = [f"t{i}" for i in range(5)]
    rest = [f"r{i}" for i in range(5)]
    codes = trig + rest
    f = pd.DataFrame(0.0, index=dates, columns=codes)
    y = pd.DataFrame(0.0, index=dates, columns=codes)
    # 非调仓日：触发侧持续赢（hit=1）；调仓日：触发侧输（hit=0）
    for dt in dates:
        f.loc[dt, trig] = 1.0
        f.loc[dt, rest] = -1.0
        if dt in rebalance:
            y.loc[dt, trig] = -0.02
            y.loc[dt, rest] = 0.02
        else:
            y.loc[dt, trig] = 0.03
            y.loc[dt, rest] = -0.01

    ic_dates = dates  # 与全日频 IC 对齐
    full = trigger_cs_payoff(f, y, min_trigger=5, direction=1.0, dates=ic_dates)
    rb_only = trigger_cs_payoff(f, y, min_trigger=5, direction=1.0, dates=rebalance)
    assert full["n_days"] == 10
    assert rb_only["n_days"] == 2
    assert abs(full["payoff_hit"] - 0.8) < 1e-9  # 8/10
    assert abs(rb_only["payoff_hit"] - 0.0) < 1e-9

    # select_sparse_factors 应走 IC 索引，而非 rebalance_dates
    name = "龙虎榜连续上榜"
    assert name in SPARSE_FACTOR_NAMES
    ic = pd.Series(0.05, index=ic_dates)  # 正 IC → direction=+1；胜率=1
    row = _summary_row(0.05, 0.40, pos_rate=1.0, n=len(ic))
    summary = pd.DataFrame([row], index=[name])
    kept, ex = select_sparse_factors(
        summary, {name: ic},
        win_rate_min=0.56,
        payoff_min=0.55,
        factor_registry={name: f},
        forward_return=y,
        rebalance_dates=rebalance,  # 若误用会因 hit=0 剔除
        require_ic=False,
    )
    assert name in kept, ex


def test_emerging_lookback_months_conversion():
    """新兴/衰减近窗：日历月 → IC 期数（与衰减同口径）。"""
    assert _lookback_periods(20, 6) == 6    # 月频 · 半年（默认）
    assert _lookback_periods(5, 6) == 26    # 周频 · 半年
    assert _lookback_periods(10, 6) == 13   # 双周 · 半年
    assert _lookback_periods(20, 12) == 12  # 月频 · 一年（换算仍成立）
    assert _lookback_periods(5, 12) == 52


def _emerging_strong_recent_ic(n_past: int = 40, n_recent: int = 12):
    """Past 弱 ICIR、recent 强 ICIR（有方差）→ lift≫1.5 且近窗 NW-t 显著。"""
    past = np.tile([0.003, -0.001], max(1, n_past // 2))[:n_past]
    recent = np.tile([0.08, 0.05], max(1, (n_recent + 1) // 2))[:n_recent]
    return _ic_series(np.concatenate([past, recent]))


def test_sparse_negative_ic_signed_gates_pass():
    """负 mean IC 稀疏因子：同向胜率 + f*s>0 payoff 过线；盲目 IC>0/f>0 会误杀。"""
    name = "开板反转_5d"
    assert name in SPARSE_FACTOR_NAMES
    f, y = _neg_ic_payoff_panels()
    # IC 索引必须与因子面板日期对齐（生产里二者同源日历）
    # 多数期 IC<0 → mean IC 负；正 IC 占比仅 0.30，同向胜率 0.70
    ic_vals = [-0.06] * 42 + [0.02] * 18
    ic = pd.Series(ic_vals, index=pd.date_range(y.index[0], periods=len(ic_vals), freq="W-FRI"))
    # payoff 只在面板有数据的日期上算；把 IC 截到面板索引以模拟「IC 有效日」
    ic_for_payoff = ic.reindex(y.index).fillna(-0.06)
    assert float(ic.mean()) < 0
    aligned, pos_rate, _ = win_rates(ic)
    assert aligned >= 0.56
    assert pos_rate < 0.56  # 盲目正 IC 胜率会失败
    assert ic_direction_sign(float(ic.mean())) == -1.0

    signed_hit = trigger_cs_payoff(f, y, direction=-1.0)["payoff_hit"]
    unsigned_hit = trigger_cs_payoff(f, y, direction=1.0)["payoff_hit"]
    assert signed_hit >= 0.55
    assert unsigned_hit < 0.55  # 旧 unsigned f>0 门槛会误杀

    # summary「胜率」= 同向；「正IC占比」故意写成会失败的正占比
    row = _summary_row(
        float(ic.mean()), 0.40,
        pos_rate=pos_rate,  # 写入正IC占比
        n=len(ic),
    )
    row["胜率"] = aligned  # 同向胜率（筛选应读这个）
    row["正IC占比"] = pos_rate
    row["负IC占比"] = 1.0 - pos_rate
    summary = pd.DataFrame([row], index=[name])

    kept, ex = select_sparse_factors(
        summary, {name: ic_for_payoff},
        win_rate_min=0.56,
        payoff_min=0.55,
        factor_registry={name: f},
        forward_return=y,
        require_ic=False,
    )
    assert name in kept, ex
    assert name not in ex

    # 若误用正IC占比作胜率门：构造「胜率」=正占比 → 应被剔除（对照）
    row_bad = dict(row)
    row_bad["胜率"] = pos_rate  # 模拟旧逻辑读到的值
    summary_bad = pd.DataFrame([row_bad], index=[name])
    kept_bad, ex_bad = select_sparse_factors(
        summary_bad, {name: ic},
        win_rate_min=0.56,
        payoff_min=0.55,
        payoff_hits={name: float(signed_hit)},
    )
    assert name not in kept_bad
    assert "胜率" in ex_bad[name]


def test_emerging_inclusion():
    ic = _emerging_strong_recent_ic(40, 20)
    row = pd.Series(_summary_row(0.005, 0.10))
    assert evaluate_emerging(
        "弱因子", row, ic,
        lookback=20,
        recent_icir_min=0.3,
        ic_threshold=0.02,
        icir_threshold=0.30,
        recent_fdr_sig=True,
        lift_min=1.5,
        require_trend=False,
    )
    # 无 FDR 显著 → 不标
    assert not evaluate_emerging(
        "弱因子", row, ic,
        lookback=20,
        recent_icir_min=0.3,
        recent_fdr_sig=False,
        lift_min=1.5,
        require_trend=False,
    )
    row2 = pd.Series(_summary_row(0.03, 0.40))
    assert not evaluate_emerging(
        "强因子", row2, ic,
        lookback=20,
        recent_icir_min=0.3,
        recent_fdr_sig=True,
        lift_min=1.5,
        require_trend=False,
    )


def _decayed_ic_series(n_recent: int = 20):
    """Past 强 ICIR，recent 均值≈0（交替正负）→ R≈0 且 |ICIR_recent|≈0。"""
    rng = np.random.default_rng(0)
    past = 0.06 + 0.02 * rng.standard_normal(40)
    recent = np.tile([0.03, -0.03], max(1, (n_recent + 1) // 2))[:n_recent]
    return _ic_series(np.concatenate([past, recent]))


def test_decay_label_not_exclusion():
    ic = _decayed_ic_series(20)
    decayed, stats = evaluate_decay_label(
        ic, recent_periods=20, retention_min=0.40,
        recent_icir_max=0.20, recent_ic_max=0.015,
    )
    assert decayed
    assert stats["retention"] < 0.40
    assert abs(stats["ic_recent"]) < 0.015

    # 兼容旧接口：永不返回 exclusion
    row = pd.Series(_summary_row(0.01, 0.10))
    cat, excl = evaluate_decay_gate(
        "衰", row, ic, recent_periods=20, retention_min=0.40,
        recent_icir_max=0.20, recent_ic_max=0.015,
    )
    assert cat == CAT_DECAYED
    assert excl is None


def test_decay_requires_weak_recent_ic_and():
    """高 |IC_recent| 即使 R 与 ICIR 弱也不标衰减（合取，非 OR）。"""
    # past 强；recent = [0.20, -0.14]*10 → mean=0.03、ICIR≈0.176、|IC|>0.015
    past = np.tile([0.08, 0.06], 20)
    recent = np.tile([0.20, -0.14], 10)
    ic = _ic_series(np.concatenate([past, recent]))
    decayed, stats = evaluate_decay_label(
        ic, recent_periods=20, retention_min=0.40,
        recent_icir_max=0.20, recent_ic_max=0.015,
    )
    assert abs(stats["ic_recent"]) >= 0.015
    assert abs(stats["icir_recent"]) < 0.20
    assert stats["retention"] < 0.40
    assert not decayed


def test_decay_does_not_exclude_from_pool():
    name = "动量_20d"
    # hold_period=5 + recent_months=12 → ~52 weekly periods；造足够长的塌缩尾部
    ic = _decayed_ic_series(52)
    summary = pd.DataFrame([_summary_row(0.04, 0.50)], index=[name])
    all_ic = {name: ic}
    sel = select_factors_multi_track(summary,
        all_ic=all_ic,
        raw_mode=True,
        corr_dedup=False,
        hold_period=5,
        enable_decay_gate=True,
        decay_recent_months=12,
        decay_retention_min=0.40,
        decay_recent_icir_max=0.20,
        enable_emerging=False,
        enable_sparse_track=False,
        enable_reversal_label=False,
        use_fdr=False,
        t_threshold=2.0,
        nw_t_threshold=None,
        min_long_share=0,
    )
    assert name in sel.dense_kept
    assert name not in sel.exclusions
    assert CAT_DECAYED in sel.labels.get(name, [])


def test_style_reversal_label():
    # 全样本正 IC，近 13 期几乎全负且 |IC|>0.015（默认门槛）
    vals = [0.05] * 40 + [-0.04] * 13
    ic = _ic_series(vals)
    frac = style_reversal_fraction(ic, 13, abs_ic_min=0.015)
    assert frac > 0.75
    rev, _ = evaluate_style_reversal(
        ic, quarter_periods=13, frac_min=0.75, abs_ic_min=0.015,
    )
    assert rev


def test_sparse_no_t_test_gates():
    name = "龙虎榜连续上榜"
    # t 极低、FDR 会失败，但胜率+payoff 够 → 仍过线
    summary = pd.DataFrame(
        [_summary_row(0.01, 0.10, t=0.5, nw_t=0.5, pos_rate=0.60)],
        index=[name],
    )
    ic = _ic_series([0.06] * 36 + [-0.02] * 24)
    kept, ex = select_sparse_factors(
        summary, {name: ic},
        win_rate_min=0.56,
        payoff_min=0.55,
        payoff_hits={name: 0.60},
        require_ic=False,
        use_fdr=True,  # ignored
        t_threshold=2.5,  # ignored
    )
    assert name in kept
    assert name not in ex


def test_sparse_win_rate_and_payoff_gates():
    name = "龙虎榜连续上榜"
    assert name in SPARSE_FACTOR_NAMES
    summary = pd.DataFrame([_summary_row(0.03, 0.40, pos_rate=0.45)], index=[name])
    ic_bad = _ic_series([0.05] * 4 + [-0.04] * 6)
    kept, ex = select_sparse_factors(
        summary, {name: ic_bad},
        win_rate_min=0.56,
        payoff_min=0.55,
        payoff_hits={name: 0.80},
    )
    assert name not in kept
    assert "胜率" in ex[name]

    summary2 = pd.DataFrame(
        [_summary_row(0.03, 0.40, pos_rate=0.60)], index=[name],
    )
    ic_good = _ic_series([0.06] * 6 + [-0.02] * 4)
    kept2, ex2 = select_sparse_factors(
        summary2, {name: ic_good},
        win_rate_min=0.56,
        payoff_min=0.55,
        payoff_hits={name: 0.60},
    )
    assert name in kept2
    assert name not in ex2

    # payoff 不够 → 剔除
    kept3, ex3 = select_sparse_factors(
        summary2, {name: ic_good},
        win_rate_min=0.56,
        payoff_min=0.55,
        payoff_hits={name: 0.40},
    )
    assert name not in kept3
    assert "截面胜率" in ex3[name]


def test_multi_track_separates_sparse_from_dense():
    dense = "动量_20d"
    sparse = "开板反转_5d"
    rows = {
        dense: _summary_row(0.04, 0.50),
        sparse: _summary_row(0.03, 0.40, pos_rate=0.60),
    }
    summary = pd.DataFrame(rows).T
    all_ic = {
        dense: _ic_series([0.04] * 60),
        sparse: _ic_series([0.06] * 36 + [-0.02] * 24),
    }
    sel = select_factors_multi_track(summary,
        all_ic=all_ic,
        raw_mode=True,
        corr_dedup=False,
        hold_period=5,
        enable_decay_gate=False,
        enable_emerging=False,
        enable_reversal_label=False,
        enable_sparse_track=True,
        use_fdr=False,
        t_threshold=2.0,
        nw_t_threshold=None,
        payoff_hits={sparse: 0.60},
        sparse_win_rate_min=0.56,
        sparse_payoff_min=0.55,
        min_long_share=0,
    )
    assert dense in sel.dense_kept
    assert sparse not in sel.dense_kept
    assert sparse in sel.sparse_kept
    assert sel.categories[dense] == CAT_DENSE
    assert sel.categories[sparse] == CAT_SPARSE


def test_multi_track_labels_decay_and_reversal():
    name = "动量_20d"
    # 前半正、近一季几乎全负 → 风格逆转；仍应留在 dense_kept
    vals = [0.08] * 40 + [-0.05] * 13
    summary = pd.DataFrame([_summary_row(0.03, 0.35)], index=[name])
    all_ic = {name: _ic_series(vals)}
    sel = select_factors_multi_track(summary,
        all_ic=all_ic,
        raw_mode=True,
        corr_dedup=False,
        hold_period=5,
        enable_decay_gate=True,
        decay_recent_months=3,
        decay_retention_min=0.40,
        decay_recent_icir_max=0.20,
        decay_recent_ic_max=0.015,
        enable_reversal_label=True,
        reversal_months=3,
        reversal_frac=0.75,
        reversal_abs_ic=0.015,
        enable_emerging=False,
        enable_sparse_track=False,
        use_fdr=False,
        t_threshold=2.0,
        nw_t_threshold=None,
        min_long_share=0,
    )
    assert name in sel.dense_kept
    assert name not in sel.exclusions
    assert CAT_REVERSAL in sel.labels.get(name, [])


def test_decay_label_weak_recent_icir():
    ic = _decayed_ic_series()
    decayed, stats = evaluate_decay_label(
        ic, recent_periods=20, retention_min=0.40,
        recent_icir_max=0.20, recent_ic_max=0.015,
    )
    assert decayed
    assert abs(stats["icir_recent"]) < 0.20
    assert abs(stats["ic_recent"]) < 0.015
    assert stats["retention"] < 0.40


def test_multi_track_emerging_label_only():
    """新兴只进 emerging_kept / categories，不进 dense_kept。"""
    name = "新兴测试因子"
    summary = pd.DataFrame([_summary_row(0.005, 0.10)], index=[name])
    # hold_period=20 → 6 个月 = 6 期；末尾强 + FDR/ICIR/lift
    all_ic = {name: _emerging_strong_recent_ic(40, 6)}
    sel = select_factors_multi_track(summary,
        all_ic=all_ic,
        raw_mode=True,
        corr_dedup=False,
        enable_decay_gate=False,
        enable_emerging=True,
        emerging_lookback=6,  # 日历月
        hold_period=20,
        emerging_recent_icir=0.3,
        emerging_fdr_alpha=0.05,
        emerging_lift_min=1.5,
        emerging_require_trend=False,
        enable_sparse_track=False,
        enable_reversal_label=False,
        use_fdr=False,
        t_threshold=2.0,
        nw_t_threshold=None,
        ic_threshold=0.02,
        icir_threshold=0.30,
        min_long_share=0,
    )
    assert name not in sel.dense_kept
    assert name in sel.emerging_kept
    assert sel.categories[name] == CAT_EMERGING


def test_emerging_uses_pure_not_raw_rescue():
    """barra 模式下：raw 近窗强、pure 近窗弱 → 不得标新兴。"""
    name = "新兴纯口径"
    summary = pd.DataFrame([_summary_row(0.005, 0.10)], index=[name])
    raw = _emerging_strong_recent_ic(40, 6)
    pure = _ic_series([0.0] * 46)  # pure 近窗≈0
    assert evaluate_emerging(
        name, pd.Series(_summary_row(0.005, 0.10)), raw,
        lookback=6, recent_icir_min=0.3,
        ic_threshold=0.02, icir_threshold=0.30,
        recent_fdr_sig=True, lift_min=1.5,
        require_trend=False,
    )
    sel = select_factors_multi_track(summary,
        all_ic={name: raw},
        pure_ic_means={name: 0.005},
        pure_ic_series={name: pure},
        raw_mode=True,
        corr_dedup=False,
        enable_decay_gate=False,
        enable_emerging=True,
        emerging_lookback=6,
        hold_period=20,
        emerging_recent_icir=0.3,
        emerging_fdr_alpha=0.05,
        emerging_lift_min=1.5,
        emerging_require_trend=False,
        enable_sparse_track=False,
        enable_reversal_label=False,
        use_fdr=False,
        t_threshold=2.0,
        nw_t_threshold=None,
        ic_threshold=0.02,
        icir_threshold=0.30,
        min_long_share=0,
    )
    assert name not in sel.emerging_kept
    assert name not in sel.dense_kept


def test_resolve_window_ic_prefer_pure_only():
    raw = _ic_series([0.05] * 20)
    pure = _ic_series([0.01] * 20)
    assert _resolve_window_ic(
        "a", {"a": raw}, {"a": pure}, prefer_pure_only=True,
    ).equals(pure)
    assert _resolve_window_ic(
        "a", {"a": raw}, {"b": pure}, prefer_pure_only=True,
    ) is None
    assert _resolve_window_ic(
        "a", {"a": raw}, None, prefer_pure_only=False,
    ).equals(raw)


def test_emerging_corr_dedup_vs_dense():
    """新兴相对主池相关去重：与 dense 高相关者不得进 emerging_kept。"""
    dense = "动量_20d"
    emerg = "市值克隆"
    idx = pd.date_range("2020-01-03", periods=60, freq="W-FRI")
    base = pd.Series(np.linspace(0.02, 0.05, 60), index=idx)
    noise = pd.Series(0.001 * np.arange(60), index=idx)
    all_ic = {dense: base, emerg: base + noise}
    summary = pd.DataFrame({
        dense: _summary_row(0.04, 0.50),
        emerg: _summary_row(0.005, 0.10),
    }).T
    # 末尾拉高 emerg 近窗，使其过 FDR/ICIR（lift 本测关闭，聚焦 corr-dedup）
    emerg_ic = all_ic[emerg].copy()
    emerg_ic.iloc[-6:] = np.tile([0.08, 0.05], 3)
    all_ic[emerg] = emerg_ic
    sel = select_factors_multi_track(summary,
        all_ic=all_ic,
        raw_mode=True,
        corr_dedup=True,
        corr_threshold=0.70,
        enable_decay_gate=False,
        enable_emerging=True,
        emerging_lookback=6,
        hold_period=20,
        emerging_recent_icir=0.3,
        emerging_fdr_alpha=0.05,
        emerging_lift_min=0.0,  # 本测只验 corr-dedup
        emerging_require_trend=False,
        enable_sparse_track=False,
        enable_reversal_label=False,
        use_fdr=False,
        t_threshold=2.0,
        nw_t_threshold=None,
        ic_threshold=0.02,
        icir_threshold=0.30,
        min_long_share=0,
    )
    assert dense in sel.dense_kept
    assert emerg not in sel.dense_kept
    # 与 dense 高度相关 → 新兴名单被去重掉
    assert emerg not in sel.emerging_kept
    assert "新兴观察相关去重" in sel.exclusions.get(emerg, "")


def test_dedup_emerging_helper():
    idx = pd.date_range("2020-01-03", periods=40, freq="W-FRI")
    a = pd.Series(np.linspace(0, 1, 40), index=idx)
    b = a * 0.99 + 0.01
    # 高频交替：与线性趋势低相关
    c = pd.Series(np.where(np.arange(40) % 2 == 0, 1.0, -1.0), index=idx)
    assert abs(float(a.corr(c))) < 0.70
    kept, ex = _dedup_emerging_by_ic_corr(
        ["b", "c"], ["a"], {"a": a, "b": b, "c": c},
        corr_threshold=0.70, score={"b": 0.5, "c": 0.4},
    )
    assert "b" not in kept
    assert "c" in kept
    assert "b" in ex


def test_sparse_corr_dedup_keeps_higher_icir():
    """高相关稀疏对：只保留 |ICIR| 更高者（IC 序列 fallback）。"""
    a, b = "龙虎榜上榜次数_20d", "龙虎榜净买额_20d"
    assert a in SPARSE_FACTOR_NAMES and b in SPARSE_FACTOR_NAMES
    idx = pd.date_range("2020-01-03", periods=40, freq="W-FRI")
    base = pd.Series(np.linspace(0.02, 0.06, 40), index=idx)
    all_ic = {a: base, b: base * 0.98 + 0.001}
    assert abs(float(all_ic[a].corr(all_ic[b]))) > 0.70
    summary = pd.DataFrame({
        a: _summary_row(0.04, 0.45, pos_rate=0.70),
        b: _summary_row(0.03, 0.25, pos_rate=0.65),
    }).T
    kept, ex = select_sparse_factors(
        summary, all_ic,
        win_rate_min=0.56,
        payoff_min=0.55,
        payoff_hits={a: 0.60, b: 0.58},
        corr_dedup=True,
        corr_threshold=0.70,
    )
    assert a in kept
    assert b not in kept
    assert "稀疏相关去重" in ex.get(b, "")


def test_dedup_sparse_helper_ic_fallback():
    """无面板时走 IC 序列相关；低相关第三方保留。"""
    strong, clone, weak = "涨停强度_20d", "涨跌停净强度_20d", "跌停弱势_20d"
    idx = pd.date_range("2020-01-03", periods=40, freq="W-FRI")
    a = pd.Series(np.linspace(0, 1, 40), index=idx)
    b = a * 0.99 + 0.01
    c = pd.Series(np.where(np.arange(40) % 2 == 0, 1.0, -1.0), index=idx)
    summary = pd.DataFrame({
        strong: _summary_row(0.04, 0.50),
        clone: _summary_row(0.03, 0.30),
        weak: _summary_row(0.02, 0.35),
    }).T
    kept, ex = _dedup_sparse_by_corr(
        [strong, clone, weak],
        summary,
        all_ic={strong: a, clone: b, weak: c},
        corr_threshold=0.70,
    )
    assert strong in kept
    assert clone not in kept
    assert weak in kept
    assert "稀疏相关去重" in ex[clone]


def test_sparse_corr_dedup_via_multi_track():
    """multi_track 稀疏轨：高相关对只留一个进 sparse_kept / categories。"""
    a, b = "龙虎榜连续上榜", "龙虎榜净买额_20d"
    idx = pd.date_range("2020-01-03", periods=50, freq="W-FRI")
    base = pd.Series(0.03 + 0.01 * np.sin(np.arange(50)), index=idx)
    all_ic = {a: base, b: base + 0.001}
    summary = pd.DataFrame({
        a: _summary_row(0.035, 0.40, pos_rate=0.70),
        b: _summary_row(0.030, 0.28, pos_rate=0.68),
    }).T
    sel = select_factors_multi_track(summary,
        all_ic=all_ic,
        raw_mode=True,
        corr_dedup=True,
        sparse_corr_threshold=0.70,
        enable_sparse_track=True,
        enable_emerging=False,
        enable_decay_gate=False,
        enable_reversal_label=False,
        payoff_hits={a: 0.60, b: 0.60},
        use_fdr=False,
        t_threshold=2.0,
        nw_t_threshold=None,
        min_long_share=0,
    )
    assert a in sel.sparse_kept
    assert b not in sel.sparse_kept
    assert sel.categories.get(a) == CAT_SPARSE
    assert b not in sel.categories or sel.categories.get(b) != CAT_SPARSE
    assert "稀疏相关去重" in sel.exclusions.get(b, "")


def test_recent_past_icir_retention():
    ic = _decayed_ic_series()
    stats = recent_past_icir_retention(ic, 20)
    assert np.isfinite(stats["retention"])
    assert stats["retention"] < 0.40


def test_variance_align_panel():
    dates = pd.date_range("2024-01-02", periods=50, freq="B")
    codes = ["a", "b", "c"]
    rng = np.random.default_rng(0)
    panel = pd.DataFrame(
        rng.normal(0, 0.1, size=(len(dates), len(codes))),
        index=dates, columns=codes,
    )
    aligned = variance_align_panel(panel, target_std=1.0, min_obs=30)
    finite = aligned.to_numpy().ravel()
    finite = finite[np.isfinite(finite)]
    assert abs(finite.std(ddof=0) - 1.0) < 0.05


def test_dynamic_denies_special_inject(monkeypatch):
    import strategies.ml as ml

    dates = pd.date_range("2024-01-02", periods=40, freq="B")
    codes = ["000001", "000002"]
    prices = pd.DataFrame(
        {c: np.linspace(10, 20, len(dates)) for c in codes},
        index=dates,
    )
    monkeypatch.setattr(
        ml, "_load_or_compute_registry",
        lambda *a, **k: {"动量_20d": prices.copy().astype("float32")},
    )
    injected = {"called": False}

    def _fake_inject(*a, **k):
        injected["called"] = True
        return []

    monkeypatch.setattr(ml, "inject_special_factors", _fake_inject)
    ds = ml.build_factor_dataset(
        prices, pd.DataFrame(),
        hold_period=5,
        special_factors="sparse,event",
        deny_special_inject=True,
        apply_tradable_filter=False,
        fwd_return_winsor=False,
        use_factor_cache=False,
        include_regime=False,
    )
    assert not injected["called"]
    assert "动量_20d" in ds.feature_names


def test_dense_gate_pure_icir_and_not_raw_rescue():
    """pure 弱但 summary/raw ICIR 强 → 不得仅靠 raw ICIR 入池（pure AND）。"""
    from research.ic.statistics import icir as _icir_fn

    name = "假强因子"
    # pure：低均值、低 ICIR（交替接近零均值）
    pure_vals = np.tile([0.012, -0.010], 50)
    pure = _ic_series(pure_vals)
    assert abs(float(pure.mean())) < 0.015
    assert abs(float(_icir_fn(pure))) < 0.30
    # summary 故意给很高的 raw ICIR（旧 bug 会靠它过 OR 门）
    summary = pd.DataFrame(
        [_summary_row(0.01, 0.90, t=5.0, nw_t=5.0)],
        index=[name],
    )
    raw_strong = _ic_series(np.tile([0.05, 0.04], 50))
    sel = select_factors_multi_track(summary,
        all_ic={name: raw_strong},
        pure_ic_means={name: float(pure.mean())},
        pure_ic_series={name: pure},
        raw_mode=True,
        corr_dedup=False,
        enable_decay_gate=False,
        enable_emerging=False,
        enable_sparse_track=False,
        enable_reversal_label=False,
        use_fdr=False,
        t_threshold=0.0,
        nw_t_threshold=None,
        ic_threshold=0.015,
        icir_threshold=0.30,
        min_long_share=0,
    )
    assert name not in sel.dense_kept
    reason = sel.exclusions.get(name, "")
    assert "纯IC" in reason or "纯ICIR" in reason or "需 |IC|∧|ICIR|" in reason


def test_dense_gate_requires_and_not_or():
    """|IC| 过线但 |ICIR| 不过 → 合取 AND 下不得入池。"""
    name = "高IC低ICIR"
    # 有偏交替：均值≈0.02，波动大 → ICIR≈0.25 < 0.30
    vals = np.tile([0.10, -0.06], 40)
    ic = _ic_series(vals)
    from research.ic.statistics import icir as _icir_fn
    assert abs(float(ic.mean())) >= 0.015
    assert abs(float(_icir_fn(ic))) < 0.30
    summary = pd.DataFrame(
        [_summary_row(float(ic.mean()), float(_icir_fn(ic)), t=5.0, nw_t=5.0)],
        index=[name],
    )
    sel = select_factors_multi_track(summary,
        all_ic={name: ic},
        raw_mode=True,
        corr_dedup=False,
        enable_decay_gate=False,
        enable_emerging=False,
        enable_sparse_track=False,
        enable_reversal_label=False,
        use_fdr=False,
        t_threshold=0.0,
        nw_t_threshold=None,
        ic_threshold=0.015,
        icir_threshold=0.30,
        min_long_share=0,
    )
    assert name not in sel.dense_kept


def test_decay_defaults_recent_icir_ic_max():
    """默认衰减近窗门槛：|ICIR_r|<0.20 且 |IC_r|<0.010。"""
    from config.settings import IC_DECAY_RECENT_IC_MAX, IC_DECAY_RECENT_ICIR_MAX

    assert IC_DECAY_RECENT_ICIR_MAX == 0.20
    assert IC_DECAY_RECENT_IC_MAX == 0.010
    # recent 交替 → |IC|≈0、|ICIR|≈0，R 小 → 衰减
    ic = _decayed_ic_series(26)
    decayed, stats = evaluate_decay_label(ic, recent_periods=26)
    assert decayed
    assert abs(stats["icir_recent"]) < 0.20
    assert abs(stats["ic_recent"]) < 0.010


def test_emerging_holdout_and_trend():
    """holdout 截断评效段；三季度增强可选。"""
    from research.ic.selection import _truncate_ic_for_emerging
    from research.ic.statistics import segment_metric_trend

    # 三段逐步增强 |ICIR|
    q1 = np.tile([0.01, -0.008], 7)[:13]
    q2 = np.tile([0.03, 0.01], 7)[:13]
    q3 = np.tile([0.08, 0.05], 7)[:13]
    # 末尾再加 6 期「评效段」噪声（应被 holdout 去掉）
    hold = np.tile([0.0, 0.0], 3)
    ic = _ic_series(np.concatenate([q1, q2, q3, hold]))
    ok, vals = segment_metric_trend(ic.iloc[:-6], 13, n_segments=3, eps=0.02)
    assert ok
    assert vals[-1] > vals[0]

    trunc = _truncate_ic_for_emerging(ic, holdout_periods=6)
    assert len(trunc) == len(ic) - 6
    asof_cut = _truncate_ic_for_emerging(
        ic, asof=ic.index[-7], holdout_periods=0,
    )
    assert asof_cut.index.max() <= ic.index[-7]

    # 衰减趋势（末段更弱）→ require_trend 拒绝新兴
    weak_end = _ic_series(np.concatenate([q3, q2, q1]))
    row = pd.Series(_summary_row(0.005, 0.10))
    assert not evaluate_emerging(
        "趋势弱", row, weak_end,
        lookback=13,
        recent_icir_min=0.01,
        recent_fdr_sig=True,
        lift_min=0.0,
        require_trend=True,
        trend_segment_periods=13,
        trend_segments=3,
        trend_eps=0.02,
    )
