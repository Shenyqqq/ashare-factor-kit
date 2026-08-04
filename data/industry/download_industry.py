"""
data/industry/download_industry.py  —  下载申万二级行业成分股映射（含历史变更记录）

接口：
    ak.stock_industry_clf_hist_sw()  获取全部股票的申万历史行业分类记录

输出：
    1) data/raw/industry_map.parquet         —— 当前静态映射（向后兼容）
       index=code(6位), columns=[industry_code, sw_l2(4位), sw_l1(2位)]
    2) data/raw/industry_map_panel.parquet   —— PIT 行业面板（新增，长表）
       columns=[code, effective_date, sw_l1, sw_l2, end_date]
       每行 = 该股票在某段时间内所属的申万二级行业。
       end_date = 下一段 effective_date - 1 day；最后一段 end_date = NaT。
       用 load_industry_as_of(panel, date) 取当期 code -> sw_l2。

注：
    AKShare 的 stock_industry_clf_hist_sw() 访问申万研究官网时存在 SSL 证书问题，
    需要在调用前 monkey-patch requests.get 以跳过验证。
    原申万二级成分股接口 index_component_sw() 在当前版本已损坏，不再使用。

    若 stock_industry_clf_hist_sw 接口不可用或失败，会 fallback 只产出
    industry_map.parquet（无 PIT 历史），并 loguru warning 说明无法构建 PIT 行业。

用法:
    python -m data.industry.download_industry
"""
import sys
import warnings
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import RAW_DIR

OUT_PATH = RAW_DIR / "industry_map.parquet"
PANEL_PATH = RAW_DIR / "industry_map_panel.parquet"


def _patch_ssl():
    """绕过申万研究所官网 SSL 证书验证失败问题（Windows 常见）"""
    import requests
    import urllib3
    urllib3.disable_warnings()
    warnings.filterwarnings("ignore")
    original_get = requests.get

    def patched_get(url, **kwargs):
        kwargs.setdefault("verify", False)
        return original_get(url, **kwargs)

    requests.get = patched_get


def _fetch_industry_hist() -> pd.DataFrame | None:
    """
    调用 ak.stock_industry_clf_hist_sw() 拿全部股票的历史行业变更记录。

    返回原始 DataFrame（含 symbol / start_date / industry_code / update_time 列），
    失败时返回 None。
    """
    _patch_ssl()
    try:
        import akshare as ak
    except Exception as e:  # pragma: no cover
        logger.warning(f"无法导入 akshare: {e}")
        return None

    try:
        df = ak.stock_industry_clf_hist_sw()
    except Exception as e:
        logger.warning(
            f"ak.stock_industry_clf_hist_sw() 调用失败: {e}\n"
            "将回退到只产出静态当前行业映射（无 PIT 历史）。"
        )
        return None

    if df is None or df.empty:
        logger.warning(
            "ak.stock_industry_clf_hist_sw() 返回空数据，"
            "无法构建 PIT 行业面板。"
        )
        return None

    required = {"symbol", "start_date", "industry_code"}
    if not required.issubset(set(df.columns)):
        logger.warning(
            f"stock_industry_clf_hist_sw 返回列缺失: {set(df.columns)} "
            f"不包含 {required}，无法构建 PIT 行业面板。"
        )
        return None

    return df


def build_industry_panel(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    将原始历史分类记录转为长表 PIT 面板。

    输入: raw_df 列 = [symbol, start_date, industry_code, ...]
    输出: columns = [code, effective_date, sw_l1, sw_l2, end_date]
          按 (code, effective_date) 升序排序。
          end_date = 同 code 下一段 effective_date - 1 day；最后一段 NaT。
    """
    df = raw_df[["symbol", "start_date", "industry_code"]].copy()
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    # P1-4: 未分类股票保留为 "UNKNOWN" 而非删除，避免幸存者偏差式行业缺失。
    # 下游 Barra 行业哑变量对 UNKNOWN 单独成一列或作为参照基线。
    # 仅丢弃 symbol / start_date 物理缺失的行，industry_code 缺失保留为 UNKNOWN。
    df = df.dropna(subset=["start_date", "symbol"])
    df["symbol"] = df["symbol"].astype(str)
    df["industry_code"] = df["industry_code"].astype(str)
    # 原始 industry_code 缺失（NaN→'nan' 字符串 / 空串）统一标记为 UNKNOWN
    df["industry_code"] = df["industry_code"].where(
        df["industry_code"].str.strip().str.len().ge(1) & (df["industry_code"] != "nan"),
        "UNKNOWN",
    )
    df = df.rename(columns={"symbol": "code", "start_date": "effective_date"})

    # 申万2021分类：6位代码，前4位=L2，前2位=L1
    # UNKNOWN / 非6位代码：sw_l1 / sw_l2 保持 "UNKNOWN"，下游哑变量单独成一列
    is_unknown = df["industry_code"] == "UNKNOWN"
    padded = df["industry_code"].str.zfill(6)
    df["industry_code"] = padded.where(~is_unknown, "UNKNOWN")
    df["sw_l2"] = padded.str[:4].where(~is_unknown, "UNKNOWN")
    df["sw_l1"] = padded.str[:2].where(~is_unknown, "UNKNOWN")

    # 同 code 同 effective_date 去重（保留最后一条，防止接口偶发重复）
    df = (
        df.sort_values(["code", "effective_date"])
        .drop_duplicates(subset=["code", "effective_date"], keep="last")
        .reset_index(drop=True)
    )

    # end_date = 下一段 effective_date - 1 day
    df["end_date"] = (
        df.groupby("code")["effective_date"].shift(-1) - pd.Timedelta(days=1)
    )

    return df[["code", "effective_date", "sw_l1", "sw_l2", "end_date"]]


def _snapshot_from_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """从 PIT 面板取每只股票最新一段，生成静态 industry_map.parquet 兼容格式。"""
    latest = (
        panel.sort_values("effective_date")
        .groupby("code", as_index=False)
        .last()
    )
    latest["industry_code"] = latest["sw_l2"].astype(str).str[:4]
    return (
        latest[["code", "industry_code", "sw_l2", "sw_l1"]]
        .set_index("code")
    )


def download_industry() -> pd.DataFrame:
    """
    下载申万行业分类。

    - 若 stock_industry_clf_hist_sw 可用：同时产出 industry_map.parquet（静态）
      和 industry_map_panel.parquet（PIT 长表），返回静态 snapshot DataFrame。
    - 若历史接口不可用：fallback 到 ak.stock_board_industry_* 等接口获取当前映射
      （保持向后兼容），仅产出 industry_map.parquet，并 warning 说明无 PIT 历史。

    返回静态 industry_map DataFrame（index=code, columns=[industry_code, sw_l2, sw_l1]）。
    """
    raw_df = _fetch_industry_hist()

    if raw_df is not None:
        panel = build_industry_panel(raw_df)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(PANEL_PATH, index=False)
        logger.info(
            f"PIT 行业面板已保存: {len(panel)} 条记录, "
            f"{panel['code'].nunique()} 只股票, "
            f"-> {PANEL_PATH}"
        )

        latest = _snapshot_from_panel(panel)
        latest.to_parquet(OUT_PATH)
        logger.info(
            f"完成: {len(latest)}只股票, "
            f"{latest['sw_l2'].nunique()}个申万L2行业, "
            f"{latest['sw_l1'].nunique()}个申万L1行业, "
            f"保存至 {OUT_PATH}"
        )
        return latest

    # Fallback：历史接口不可用，尝试拿当前快照
    logger.warning(
        "无法获取申万历史行业分类记录，回退到当前静态映射，"
        "下游 PIT 行业特性将不可用（industry_map_panel.parquet 不会生成）。"
    )
    return _fallback_download_current()


def _fallback_download_current() -> pd.DataFrame:
    """
    Fallback：用当前行业接口拿一份静态映射（无 PIT 时间维度）。
    尽量不依赖申万官网 SSL（已被历史接口走通的 patch 覆盖）。
    """
    _patch_ssl()
    try:
        import akshare as ak
    except Exception as e:
        raise RuntimeError(f"akshare 导入失败，无法 fallback 下载当前行业: {e}")

    # 优先尝试历史接口的"只取当前"子集；若整体失败，再用 board 接口
    last_err: Exception | None = None
    for fn_name, builder in (
        ("stock_industry_clf_hist_sw", _try_current_from_hist),
        ("stock_board_industry_cons_ths", _try_current_from_ths),
    ):
        try:
            df = builder(ak)
            if df is not None and not df.empty:
                OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(OUT_PATH)
                logger.info(
                    f"[fallback] 当前行业映射已保存: {len(df)} 只股票 -> {OUT_PATH}"
                )
                return df
        except Exception as e:
            last_err = e
            logger.warning(f"[fallback] {fn_name} 失败: {e}")

    raise RuntimeError(
        f"所有 fallback 当前行业下载方式均失败 (last error: {last_err})。"
        "请检查 akshare 版本或网络。"
    )


def _try_current_from_hist(ak) -> pd.DataFrame | None:
    df = ak.stock_industry_clf_hist_sw()
    if df is None or df.empty:
        return None
    panel = build_industry_panel(df)
    return _snapshot_from_panel(panel)


def _try_current_from_ths(ak) -> pd.DataFrame | None:
    """THS 同花顺行业概念，作为 last-resort。"""
    df = ak.stock_board_industry_cons_ths(symbol="行业")
    if df is None or df.empty:
        return None
    # 这里无法精确还原 sw_l2/sw_l1 代码，但保证有 industry_code 列
    df = df.rename(columns={"code": "code", "行业": "industry_code"})
    df["industry_code"] = df["industry_code"].astype(str)
    df["sw_l2"] = df["industry_code"].str[:4]
    df["sw_l1"] = df["industry_code"].str[:2]
    return df.set_index("code")[["industry_code", "sw_l2", "sw_l1"]]


def load_industry_map() -> pd.DataFrame:
    if not OUT_PATH.exists():
        raise FileNotFoundError(
            f"行业映射文件不存在: {OUT_PATH}\n"
            "请先运行: python -m data.industry.download_industry"
        )
    return pd.read_parquet(OUT_PATH)


def load_industry_panel() -> pd.DataFrame | None:
    """
    加载 PIT 行业面板。

    返回 DataFrame[code, effective_date, sw_l1, sw_l2, end_date]；
    若文件不存在（旧版下载产物或 fallback 路径），返回 None。
    """
    if not PANEL_PATH.exists():
        return None
    panel = pd.read_parquet(PANEL_PATH)
    # 兼容旧字段名
    if "effective_date" not in panel.columns and "start_date" in panel.columns:
        panel = panel.rename(columns={"start_date": "effective_date"})
    panel["effective_date"] = pd.to_datetime(panel["effective_date"])
    if "end_date" in panel.columns:
        panel["end_date"] = pd.to_datetime(panel["end_date"], errors="coerce")
    return panel


def load_industry_as_of(
    panel: pd.DataFrame,
    date,
    level: str = "sw_l2",
) -> pd.Series:
    """
    PIT 行业查询：给定日期 date，返回当期 code -> {level} Series。

    panel: 由 load_industry_panel() / build_industry_panel() 产出的长表。
    date: 任何 pd.Timestamp 兼容的日期。
    level: "sw_l2" 或 "sw_l1"。

    规则：取 effective_date <= date 且 (end_date 为空 或 end_date >= date)
    的记录；同一 code 取 effective_date 最晚的一条。
    """
    if panel is None or panel.empty:
        return pd.Series(dtype=object)
    if level not in panel.columns:
        raise ValueError(f"panel 不含列 {level}，已有: {list(panel.columns)}")

    panel = panel.copy()
    panel["effective_date"] = pd.to_datetime(panel["effective_date"], errors="coerce")
    if "end_date" in panel.columns:
        panel["end_date"] = pd.to_datetime(panel["end_date"], errors="coerce")
    panel = panel.dropna(subset=["effective_date"])

    ts = pd.Timestamp(date)
    mask = (panel["effective_date"] <= ts) & (
        panel["end_date"].isna() | (panel["end_date"] >= ts)
    )
    sub = (
        panel.loc[mask]
        .sort_values("effective_date")
        .drop_duplicates(subset=["code"], keep="last")
    )
    return sub.set_index("code")[level]


if __name__ == "__main__":
    download_industry()
