"""
config/settings.py  —  全局配置，所有参数改这里，不要散落在各脚本
"""
from pathlib import Path
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

# ── 路径 ──────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent.parent
DATA_ROOT  = Path(os.getenv("DATA_ROOT", ROOT_DIR / "data"))

RAW_DIR        = DATA_ROOT / "raw"
PROCESSED_DIR  = DATA_ROOT / "processed"
UNIVERSE_DIR   = DATA_ROOT / "universe"

# ── API Token ─────────────────────────────────────────────────────────────────
TUSHARE_TOKEN  = os.getenv("TUSHARE_TOKEN", "")

# ── 回测参数 ──────────────────────────────────────────────────────────────────
BACKTEST_START = "2018-01-01"
BACKTEST_END   = pd.Timestamp.today().strftime("%Y-%m-%d")

INITIAL_CAPITAL     = 2_000_000   # 初始资金（元）
REBALANCE_FREQ      = "ME"        # 调仓频率：ME=月末, W-FRI=每周五
N_STOCKS            = 100         # Top-N 回测持仓数（截面技能度量，非实盘仓位）
COMMISSION_RATE     = 0.0001      # 手续费（双边各0.01%）
SLIPPAGE_RATE       = 0.0         # 手动小仓位操作，市场冲击可忽略
STAMP_DUTY          = 0.0005      # 印花税（卖出单边0.05%）
# bid-ask spread（单边，bp）：大盘股 ~2-5bp，小盘股 ~10-20bp，10bp 保守默认
# 用户资金量 ~200 万，market impact 可忽略，但 spread 是真实存在的隐性成本
BID_ASK_SPREAD_BPS  = 10.0

# 风险调整指标（backtest/risk_metrics.py）：A 股短期无风险利率简化为 0；
# 若要更严格，可设 0.02（10年期国债 ~2%）。
RISK_FREE_RATE      = float(os.getenv("RISK_FREE_RATE", "0.0"))

# amount/(volume×100×close) 量纲校验：中位数应≈1（amount=元, volume=手）。
# 偏离 [AMOUNT_UNIT_RATIO_LO, AMOUNT_UNIT_RATIO_HI] 时 WARNING；
# AMOUNT_UNIT_STRICT=1 时改为 raise。
AMOUNT_UNIT_RATIO_LO = float(os.getenv("AMOUNT_UNIT_RATIO_LO", "0.5"))
AMOUNT_UNIT_RATIO_HI = float(os.getenv("AMOUNT_UNIT_RATIO_HI", "2.0"))
AMOUNT_UNIT_STRICT = os.getenv("AMOUNT_UNIT_STRICT", "0").strip().lower() in (
    "1", "true", "yes",
)

# ── 并行 / 内存（32GB 机器建议保持默认）──────────────────────────────────────
# Walk-Forward 训练：多窗口×多模型同时 fit 时，每个任务都会复制一份训练矩阵，
# 并行度过高极易 OOM。默认串行跑各 (window, model) 组合，单模型用多核。
TRAIN_MAX_WORKERS   = int(os.getenv("TRAIN_MAX_WORKERS", "1"))   # 同时训练的任务数
TRAIN_N_JOBS        = int(os.getenv("TRAIN_N_JOBS", "10"))      # 单模型线程数（仅 TRAIN_MAX_WORKERS=1 时生效）
IC_MAX_WORKERS      = int(os.getenv("IC_MAX_WORKERS", "1"))     # ic_analysis 因子 IC 并行度（32GB 建议 1）
BARRA_IC_WORKERS    = int(os.getenv("BARRA_IC_WORKERS", "1"))   # Barra 纯 IC OLS 并行度
# DynamicFactorTrainer：按调仓日并行（单线程 ~4.5GB；线程池共享缓存但 >4 并行仍易 OOM）
# 32GB 且同时跑 ML 建议 1；仅 dynamic 时可设 4（默认上限）
DYNAMIC_MAX_WORKERS = int(os.getenv("DYNAMIC_MAX_WORKERS", "4"))

# ── 股票池过滤 ────────────────────────────────────────────────────────────────
MIN_MARKET_CAP      = 20e8        # 最小市值 20亿，过滤壳股
MIN_PRICE           = 2.0         # 最低股价，过滤仙股
EXCLUDE_ST          = True        # 排除ST股

# ── 小盘策略 universe（上限 = 小盘策略核心；下限剔壳股/微盘流动性陷阱）────────
# 注：流通市值（circ_mv）在 AKShare 降级路径下不可用，此时 build_small_cap_universe
# 会自动退化为用 total_mv 作近似（total_mv ≥ circ_mv，上界收紧，下界放宽，
# 整体偏向保守剔除小盘边缘股，避免误纳大盘）。Tushare 路径下走真实 circ_mv。
SMALL_CAP_UPPER   = 150e8   # 流通市值上限 150亿（小盘策略核心）
SHELL_CAP_LOWER   = 8e8     # 流通市值下限 8亿，剔壳股/微盘流动性陷阱
MIN_AMOUNT_20D    = 2000e4  # 20日均成交额 ≥ 2000万，剔僵尸股

# ── 市值带预设（cap-band 策略）──────────────────────────────────────────────────
CAP_BANDS = {
    "all":       (None, None),       # 全市场（默认，向后兼容，mask=None）
    "small":     (30e8, 150e8),      # 真小盘
    "small_mid": (50e8, 300e8),      # 小中盘（推荐默认带）
    "mid":       (100e8, 500e8),     # 中盘成长
    "micro":     (8e8, 30e8),        # 微盘（谨慎，左尾厚）
    "small_mid_wide": (20e8, 500e8), # 宽幅小至中盘（20-500亿，覆盖真小盘到中盘成长）
    # 无下限（含微盘）+ 流通市值 ≤100 亿；lower=0 表示不设壳股/微盘地板
    # （build_cap_band_mask 对 None 会回退 SHELL_CAP_LOWER，故显式用 0）
    "micro_small_100": (0.0, 100e8),
    # 更激进：无下限 + 流通市值 ≤30 亿（≠ 已有 micro=8~30 亿带壳股地板）
    # lower 必须用 0.0，勿写 None（否则回退 8 亿）
    "micro_30": (0.0, 30e8),
    "micro_lt30": (0.0, 30e8),  # alias of micro_30
}
CAP_BAND_DEFAULT = "all"

# ── 分位小市值宇宙（IC --universe small_mcap；面板仍全市场缓存，仅 mask）──
# 定义见 utils/universe.build_mcap_percentile_mask：调仓日截面流通市值升序分位 ≤ q
SMALL_MCAP_QUANTILE = float(os.getenv("SMALL_MCAP_QUANTILE", "0.30"))

# ── IC 分析 v2（research/ic_analysis_v2.py）──────────────────────────────────
MIN_IC_STOCKS       = int(os.getenv("MIN_IC_STOCKS", "30"))
IC_CLIP             = float(os.getenv("IC_CLIP", "0.3"))          # 绝对值截断；0=禁用
IC_WINSORIZE_PCT    = (0.01, 0.99)   # 分位 winsorize；None 禁用（与 IC_CLIP 二选一优先 clip）
IC_RANK_METHOD      = os.getenv("IC_RANK_METHOD", "average")    # average|dense|first
INDUSTRY_REFERENCE  = os.getenv("INDUSTRY_REFERENCE", "drop_first")  # Barra 行业哑变量参照
IC_CORR_METHOD      = os.getenv("IC_CORR_METHOD", "max")          # max|p95|mean 去冗余相关度
# 次新过滤：IC / ML / 回测共用同一阈值，禁止 252 vs 60 双标准
# 兼容旧环境变量 IC_MIN_LISTING_DAYS；优先读 MIN_LISTING_DAYS
MIN_LISTING_DAYS = int(
    os.getenv("MIN_LISTING_DAYS", os.getenv("IC_MIN_LISTING_DAYS", "252"))
)  # 剔除次新股高波动噪声（默认 1 年）
IC_MIN_LISTING_DAYS = MIN_LISTING_DAYS  # 向后兼容别名
IC_APPLY_TRADABLE   = True        # IC 截面是否应用可交易池 mask（ST/停牌/次新等）
# IC/ML 涨跌停三层语义（见 AGENTS.md）：
#   clean_ret — 因子侧仍屏蔽涨跌停日（不改）
#   research tradable — 信号日保留涨跌停样本（默认）
#   execution — 回测买/卖日拦截（不改）
# 独立开关（False=research 默认）；CLI --tradable-strict / --label-exec-mask 可恢复旧口径。
def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes")


TRADABLE_EXCLUDE_LIMIT_SIGNAL_DAY = _env_bool(
    "TRADABLE_EXCLUDE_LIMIT_SIGNAL_DAY", False
)
FWD_RETURN_EXEC_MASK = _env_bool("FWD_RETURN_EXEC_MASK", False)
# 兼容旧 env：TRADABLE_LIMIT_MODE=strict|research 覆盖上述两开关
TRADABLE_LIMIT_MODE = os.getenv("TRADABLE_LIMIT_MODE", "").strip().lower()
if TRADABLE_LIMIT_MODE == "strict":
    TRADABLE_EXCLUDE_LIMIT_SIGNAL_DAY = True
    FWD_RETURN_EXEC_MASK = True
elif TRADABLE_LIMIT_MODE == "research":
    TRADABLE_EXCLUDE_LIMIT_SIGNAL_DAY = False
    FWD_RETURN_EXEC_MASK = False


def normalize_tradable_limit_mode(mode: str | None = None) -> str:
    """Return validated tradable limit mode (``strict`` | ``research``)."""
    if mode is not None:
        m = mode.strip().lower()
        if m not in ("strict", "research"):
            raise ValueError(
                f"tradable_limit_mode 须为 strict|research，收到 {mode!r}"
            )
        return m
    if TRADABLE_EXCLUDE_LIMIT_SIGNAL_DAY and FWD_RETURN_EXEC_MASK:
        return "strict"
    return "research"


def resolve_exclude_limit_on_signal(
    exclude_limit_on_signal: bool | None = None,
    tradable_limit_mode: str | None = None,
) -> bool:
    if exclude_limit_on_signal is not None:
        return exclude_limit_on_signal
    if tradable_limit_mode is not None:
        return normalize_tradable_limit_mode(tradable_limit_mode) == "strict"
    return TRADABLE_EXCLUDE_LIMIT_SIGNAL_DAY


def resolve_apply_exec_mask(
    apply_exec_mask: bool | None = None,
    tradable_limit_mode: str | None = None,
) -> bool:
    if apply_exec_mask is not None:
        return apply_exec_mask
    if tradable_limit_mode is not None:
        return normalize_tradable_limit_mode(tradable_limit_mode) == "strict"
    return FWD_RETURN_EXEC_MASK


def apply_label_exec_mask_for_mode(mode: str | None = None) -> bool:
    """Whether IC/ML forward_return applies buy/sell-day limit execution masks."""
    return resolve_apply_exec_mask(tradable_limit_mode=mode)


def tradable_ckpt_tag(
    exclude_limit_on_signal: bool | None = None,
    apply_exec_mask: bool | None = None,
    tradable_limit_mode: str | None = None,
) -> str:
    """Checkpoint filename suffix; bump when tradable/limit semantics change."""
    ex_lim = resolve_exclude_limit_on_signal(
        exclude_limit_on_signal, tradable_limit_mode
    )
    ex_exec = resolve_apply_exec_mask(apply_exec_mask, tradable_limit_mode)
    if not ex_lim and not ex_exec:
        return "tmr_v2"
    if ex_lim and ex_exec:
        return "tmr_strict"
    return f"tmr_lim{int(ex_lim)}_exec{int(ex_exec)}"


def tradable_mode_metadata(
    exclude_limit_on_signal: bool | None = None,
    apply_exec_mask: bool | None = None,
    tradable_limit_mode: str | None = None,
) -> dict:
    ex_lim = resolve_exclude_limit_on_signal(
        exclude_limit_on_signal, tradable_limit_mode
    )
    ex_exec = resolve_apply_exec_mask(apply_exec_mask, tradable_limit_mode)
    mode = "strict" if ex_lim and ex_exec else (
        "research" if not ex_lim and not ex_exec else "mixed"
    )
    return {
        "tradable_mode": mode,
        "label_exec_mask": "on" if ex_exec else "off",
        "exclude_limit_signal_day": ex_lim,
    }


# ── IC 稠密基础门（research/ic/selection.py）────────────────────────────────
# 入池硬门（合取 AND，勿改 OR）：
#   |IC_eff| > IC_THRESHOLD  AND  |ICIR_eff| > ICIR_THRESHOLD
# --barra / 有 pure IC 序列时：IC_eff / ICIR_eff 均来自 **pure IC 序列**
# （禁止用 summary raw ICIR 救援）；无 barra 时退回 raw，仍为 AND。
IC_THRESHOLD = float(os.getenv("IC_THRESHOLD", "0.015"))
ICIR_THRESHOLD = float(os.getenv("ICIR_THRESHOLD", "0.30"))
# 稠密门追加：符号对齐后 long_share > 阈值（与 |IC|∧|ICIR| 合取；在 corr-dedup 之前）。
# 默认 0.4；设为 0 / 负值关闭（CLI: --min-long-share 0）。须用对齐后分位分解列。
IC_MIN_LONG_SHARE = float(os.getenv("IC_MIN_LONG_SHARE", "0.4"))

# ── IC 衰减标注 / 新兴 / 稀疏轨道（research/ic/selection.py）──────────────────
# 衰减 / 风格逆转 / 新兴：仅标注，不进 factors 主池剔除语义（衰减/逆转仍可在主池内警示）。
# 同持仓期 IC 序列（有 --barra 时用 pure IC 序列，与全样本纯因子同口径）：
#   R = |ICIR_recent| / |ICIR_past|
# 衰减 iff (R < retention_min ∧ |ICIR_recent| < recent_icir_max) ∧ |IC_recent| < recent_ic_max
# （合取 AND；勿改成 OR，否则易误标高 IC 因子）。
IC_DECAY_RECENT_MONTHS = int(os.getenv("IC_DECAY_RECENT_MONTHS", "6"))
IC_DECAY_RETENTION_MIN = float(os.getenv("IC_DECAY_RETENTION_MIN", "0.50"))
IC_DECAY_RECENT_ICIR_MAX = float(os.getenv("IC_DECAY_RECENT_ICIR_MAX", "0.20"))
IC_DECAY_RECENT_IC_MAX = float(os.getenv("IC_DECAY_RECENT_IC_MAX", "0.010"))
# 稀疏轨衰减略松：保留率门槛更低（更难被标为衰减）
IC_DECAY_RETENTION_MIN_SPARSE = float(os.getenv("IC_DECAY_RETENTION_MIN_SPARSE", "0.30"))
# 风格逆转：最近一季内与全样本符号相反且 |IC|>阈值 的占比 > frac（仅标注）
IC_REVERSAL_MONTHS = int(os.getenv("IC_REVERSAL_MONTHS", "3"))
IC_REVERSAL_FRAC = float(os.getenv("IC_REVERSAL_FRAC", "0.75"))
IC_REVERSAL_ABS_IC = float(os.getenv("IC_REVERSAL_ABS_IC", "0.015"))
# 旧 half-life / 长短窗 / 残差门：已弃用（CLI 保留为 no-op 警告）
IC_DECAY_HALF_LIFE_MIN = float(os.getenv("IC_DECAY_HALF_LIFE_MIN", "8.0"))
IC_DECAY_SHORT_LONG_MIN = float(os.getenv("IC_DECAY_SHORT_LONG_MIN", "0.35"))
IC_DECAY_RESIDUAL_ICIR = float(os.getenv("IC_DECAY_RESIDUAL_ICIR", "0.25"))
IC_DECAY_RESIDUAL_IC = float(os.getenv("IC_DECAY_RESIDUAL_IC", "0.015"))
# 新兴因子：全样本未过 IC∧ICIR 稠密门 + 近窗（默认 6 个月）pure 序列
#   BH-FDR(NW-t) ∧ |ICIR_recent| ∧ lift(|ICIR_r|/|ICIR_p|) [∧ 三季度增强]
#   → 仅标注（factors_emerging），**不**并入 dense_kept / factors。
# lookback 为**日历月数**（默认 6≈半年），经 months_to_rebalance_periods 换成 IC 期数
# （与衰减同口径：h5≈26 期，h20≈6 期）。勿写死 raw 期数。
# FDR 校正域 = 全体进入稠密筛选、且有近窗 IC 的因子（非仅新兴候选子集）。
# holdout：近窗统计截止到 end-holdout（默认 0=现状）；评 2026 时建议 6 或 --emerging-asof 2025-12-31。
# 三季度增强：近 3 段（每段默认 3 个月）|ICIR| 单调不降且末段>首段（默认 ON）。
IC_EMERGING_LOOKBACK = int(os.getenv("IC_EMERGING_LOOKBACK", "6"))
IC_EMERGING_RECENT_ICIR = float(os.getenv("IC_EMERGING_RECENT_ICIR", "0.35"))
IC_EMERGING_RECENT_IC = float(os.getenv("IC_EMERGING_RECENT_IC", "0.025"))  # deprecated no-op
IC_EMERGING_FDR_ALPHA = float(os.getenv("IC_EMERGING_FDR_ALPHA", "0.05"))
IC_EMERGING_LIFT_MIN = float(os.getenv("IC_EMERGING_LIFT_MIN", "1.5"))
IC_EMERGING_HOLDOUT_MONTHS = int(os.getenv("IC_EMERGING_HOLDOUT_MONTHS", "0"))
IC_EMERGING_REQUIRE_TREND = os.getenv("IC_EMERGING_REQUIRE_TREND", "1").strip().lower() not in (
    "0", "false", "no", "off",
)
IC_EMERGING_TREND_MONTHS = int(os.getenv("IC_EMERGING_TREND_MONTHS", "3"))  # 每段日历月
IC_EMERGING_TREND_SEGMENTS = int(os.getenv("IC_EMERGING_TREND_SEGMENTS", "3"))
IC_EMERGING_TREND_EPS = float(os.getenv("IC_EMERGING_TREND_EPS", "0.02"))  # 单调噪声容忍
# 稀疏轨道：主门槛 = 同向 IC 胜率 + 触发日相对截面均值胜率（payoff_hit）
# 均按 s=sign(mean_IC) 对齐（负 IC 因子触发侧为 f<0）；mean≈0 跳过稀疏门
# IC/ICIR 为软参考（默认不硬剔）；无 t / NW-t / FDR 要求
IC_SPARSE_IC_THRESHOLD = float(os.getenv("IC_SPARSE_IC_THRESHOLD", "0.015"))
IC_SPARSE_ICIR_THRESHOLD = float(os.getenv("IC_SPARSE_ICIR_THRESHOLD", "0.20"))
IC_SPARSE_T_THRESHOLD = float(os.getenv("IC_SPARSE_T_THRESHOLD", "2.0"))  # deprecated no-op
IC_SPARSE_WIN_RATE_MIN = float(os.getenv("IC_SPARSE_WIN_RATE_MIN", "0.56"))
# payoff_hit = mean_t 1{ mean(y|f*s>0) > mean(y) }，s=sign(mean_IC)
# 日期索引与普通 IC / 胜率相同：参与 IC 的全部有效交易日（非调仓日子集）
IC_SPARSE_PAYOFF_MIN = float(os.getenv("IC_SPARSE_PAYOFF_MIN", "0.55"))
# 稀疏轨相关去重阈值（默认与稠密 IC_CORR 同为 0.70；可独立覆盖）
IC_SPARSE_CORR_THRESHOLD = float(os.getenv("IC_SPARSE_CORR_THRESHOLD", "0.70"))
# Ridge 注入稀疏因子时的目标标准差（方差对齐）
SPARSE_VARIANCE_ALIGN_STD = float(os.getenv("SPARSE_VARIANCE_ALIGN_STD", "1.0"))

# ── IC 多头/空头贡献（Q1 vs Q5，挂在 --barra pure 路径）──────────────────────
# 默认：开启 --barra 时自动跑分位分解；CLI --quantile-decomp / --no-quantile-decomp 覆盖
IC_QUANTILE_DECOMP = os.getenv("IC_QUANTILE_DECOMP", "1").strip().lower() not in (
    "0", "false", "no", "off",
)
# 收益口径：residual=对 forward return 再做 Barra+行业残差（默认）；raw=pure IC 同款 y
IC_QUANTILE_Y_MODE = os.getenv("IC_QUANTILE_Y_MODE", "residual").strip().lower()
# long_share ∈ [lo, hi] → 双边；>hi 多头主导；<lo 空头主导
IC_QUANTILE_BILATERAL_LO = float(os.getenv("IC_QUANTILE_BILATERAL_LO", "0.35"))
IC_QUANTILE_BILATERAL_HI = float(os.getenv("IC_QUANTILE_BILATERAL_HI", "0.65"))

# ── forward_return 标签截面截尾（IC / ML 共用）────────────────────────────────
# 每个 date 对可交易样本做分位 winsorize，削弱妖股连板等极端持有收益对 IC/MSE 的污染。
# None 禁用；默认 (0.01, 0.99)。CLI ``--no-fwd-return-winsor`` 可关闭。
FWD_RETURN_WINSOR: tuple[float, float] | None = (0.01, 0.99)

# ── 因子权重（初始等权，之后用ML替换）────────────────────────────────────────
FACTOR_WEIGHTS = {
    "动量_20d":   0.20,
    "反转_5d":    0.15,   # 因子函数内已取反，正权重=反转得高分
    "价值_PB":    0.20,   # 因子函数内已取反（1/PB），正权重=低PB得高分
    "质量_ROE":   0.25,
    "规模":       0.20,   # 因子函数内已取反（-log），正权重=小市值得高分
}
