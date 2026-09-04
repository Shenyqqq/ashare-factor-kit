"""验证 fractional_diff 模块：合成数据 ADF 平稳性检验"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from utils.fractional_diff import frac_diff, frac_diff_ffd, frac_diff_weights

print("=== 1. 权重序列 ===")
w = frac_diff_weights(0.4, threshold=1e-5)
print(f"d=0.4, threshold=1e-5, L={len(w)}, w[:4]={w[:4]}")
w1 = frac_diff_weights(1.0, threshold=1e-5)
print(f"d=1.0 (等价 pct_change), L={len(w1)}, w[:4]={w1[:4]}")

print("\n=== 2. 合成数据 FFD ===")
np.random.seed(42)
# 用 2000 行保证 FFD 窗口 (L≈1458 for d=0.4) 后还有足够非 NaN 行做 ADF
s = pd.DataFrame(np.cumsum(np.random.randn(2000, 3), axis=0) + 100)
fd = frac_diff_ffd(s, d=0.4)
print(f"shape: {fd.shape}, NaN rows: {fd.isna().any(axis=1).sum()}, "
      f"有效 rows: {(~fd.isna().any(axis=1)).sum()}")

print("\n=== 3. ADF 平稳性检验 ===")
p_raw = adfuller(s.iloc[:, 0].dropna())[1]
p_fd = adfuller(fd.iloc[:, 0].dropna())[1]
print(f"ADF p-value: raw={p_raw:.4f}, frac_diff(d=0.4)={p_fd:.4f}")
print(f"raw 平稳? {'是' if p_raw < 0.05 else '否 (随机游走, 符合预期)'}")
print(f"frac_diff 平稳? {'是 (符合预期)' if p_fd < 0.05 else '否'}")

print("\n=== 4. expanding-window frac_diff 对比 ===")
fd_exp = frac_diff(s.iloc[:300], d=0.4, threshold=1e-3, max_lag=50)
print(f"expanding frac_diff shape={fd_exp.shape}, "
      f"NaN rows={fd_exp.isna().any(axis=1).sum()}")

print("\n=== 5. 因子注册表不破坏 ===")
from factors.factor import get_factor_registry, _PRICE_FACTOR_NAMES
print(f"分数差分动量_20d 在 _PRICE_FACTOR_NAMES: {'分数差分动量_20d' in _PRICE_FACTOR_NAMES}")
print("registry OK")
