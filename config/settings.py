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
N_STOCKS            = 30          # 持仓股票数
COMMISSION_RATE     = 0.0001      # 手续费（双边各0.01%）
SLIPPAGE_RATE       = 0.0         # 手动小仓位操作，市场冲击可忽略
STAMP_DUTY          = 0.0005      # 印花税（卖出单边0.05%）
# bid-ask spread（单边，bp）：大盘股 ~2-5bp，小盘股 ~10-20bp，10bp 保守默认
# 用户资金量 ~200 万，market impact 可忽略，但 spread 是真实存在的隐性成本
BID_ASK_SPREAD_BPS  = 10.0

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

# ── IC 分析 v2（research/ic_analysis_v2.py）──────────────────────────────────
MIN_IC_STOCKS       = int(os.getenv("MIN_IC_STOCKS", "30"))
IC_CLIP             = float(os.getenv("IC_CLIP", "0.3"))          # 绝对值截断；0=禁用
IC_WINSORIZE_PCT    = (0.01, 0.99)   # 分位 winsorize；None 禁用（与 IC_CLIP 二选一优先 clip）
IC_RANK_METHOD      = os.getenv("IC_RANK_METHOD", "average")    # average|dense|first
INDUSTRY_REFERENCE  = os.getenv("INDUSTRY_REFERENCE", "drop_first")  # Barra 行业哑变量参照
IC_CORR_METHOD      = os.getenv("IC_CORR_METHOD", "max")          # max|p95|mean 去冗余相关度
IC_MIN_LISTING_DAYS = int(os.getenv("IC_MIN_LISTING_DAYS", "252"))  # 剔除次新股高波动噪声（1年）
IC_APPLY_TRADABLE   = True        # IC 截面是否应用可交易池 mask（ST/涨跌停/停牌）

# ── 因子权重（初始等权，之后用ML替换）────────────────────────────────────────
FACTOR_WEIGHTS = {
    "动量_20d":   0.20,
    "反转_5d":    0.15,   # 因子函数内已取反，正权重=反转得高分
    "价值_PB":    0.20,   # 因子函数内已取反（1/PB），正权重=低PB得高分
    "质量_ROE":   0.25,
    "规模":       0.20,   # 因子函数内已取反（-log），正权重=小市值得高分
}
