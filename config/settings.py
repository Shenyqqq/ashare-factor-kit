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

# ── 股票池过滤 ────────────────────────────────────────────────────────────────
MIN_MARKET_CAP      = 20e8        # 最小市值 20亿，过滤壳股
MIN_PRICE           = 2.0         # 最低股价，过滤仙股
EXCLUDE_ST          = True        # 排除ST股

# ── 因子权重（初始等权，之后用ML替换）────────────────────────────────────────
FACTOR_WEIGHTS = {
    "动量_20d":   0.20,
    "反转_5d":    0.15,   # 因子函数内已取反，正权重=反转得高分
    "价值_PB":    0.20,   # 因子函数内已取反（1/PB），正权重=低PB得高分
    "质量_ROE":   0.25,
    "规模":       0.20,   # 因子函数内已取反（-log），正权重=小市值得高分
}
