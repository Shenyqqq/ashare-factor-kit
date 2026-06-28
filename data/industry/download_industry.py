"""
data/industry/download_industry.py  —  下载申万二级行业成分股映射

结果保存至 data/raw/industry_map.parquet
格式: columns=[code, sw_l1, sw_l2]  (申万一级名称, 申万二级名称)

用法:
    python -m data.industry.download_industry
"""
import sys
import time
from pathlib import Path

import akshare as ak
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import RAW_DIR

OUT_PATH = RAW_DIR / "industry_map.parquet"


def download_sw_industry() -> pd.DataFrame:
    """
    拉取申万二级行业成分股。
    思路：
      1. ak.sw_index_second_info() 获取所有申万二级行业列表（含一级名称）
      2. 对每个二级行业拉取成分股 ak.index_component_sw()
      3. 拼接并保存
    """
    logger.info("获取申万二级行业列表...")
    try:
        sw2_list = ak.sw_index_second_info()
    except Exception as e:
        logger.error(f"获取申万二级列表失败: {e}")
        raise

    logger.info(f"共 {len(sw2_list)} 个申万二级行业")

    records = []
    for i, row in sw2_list.iterrows():
        idx_code = str(row.iloc[0])   # 行业代码
        l2_name  = str(row.iloc[1])   # 二级名称
        l1_name  = str(row.iloc[2]) if len(row) > 2 else ""  # 一级名称

        try:
            cons = ak.index_component_sw(symbol=idx_code)
            if cons is None or len(cons) == 0:
                continue
            # 成分股代码列（通常第一列或名为"股票代码"/"code"）
            code_col = cons.columns[0]
            for code in cons[code_col].astype(str).str.zfill(6):
                records.append({
                    "code":  code,
                    "sw_l2": l2_name,
                    "sw_l1": l1_name,
                })
            logger.debug(f"  [{i+1}/{len(sw2_list)}] {l2_name}: {len(cons)}只")
        except Exception as e:
            logger.warning(f"  [{i+1}] {l2_name} 失败: {e}")

        time.sleep(0.3)

    if not records:
        raise RuntimeError("未获取到任何成分股，请检查AKShare版本")

    df = pd.DataFrame(records).drop_duplicates(subset="code")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    logger.info(f"完成: {len(df)}只股票，保存至 {OUT_PATH}")

    # 打印分布
    l2_counts = df.groupby(["sw_l1", "sw_l2"]).size().reset_index(name="count")
    l2_counts = l2_counts.sort_values(["sw_l1", "sw_l2"])
    logger.info("\n行业分布（申万一级→二级，股票数）:")
    for _, r in l2_counts.iterrows():
        logger.info(f"  {r['sw_l1']:<12} {r['sw_l2']:<16} {r['count']}只")

    return df


def load_industry_map() -> pd.DataFrame:
    """加载行业映射，code为索引"""
    if not OUT_PATH.exists():
        raise FileNotFoundError(
            f"行业映射文件不存在: {OUT_PATH}\n"
            "请先运行: python -m data.industry.download_industry"
        )
    return pd.read_parquet(OUT_PATH).set_index("code")


if __name__ == "__main__":
    download_sw_industry()
