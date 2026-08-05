"""
简易图形界面：把常用参数映射到 ``python run.py ...`` 与
``python -m research.ic_analysis_v2 ...``，方便无代码基础的人改参试跑。

启动（仓库根目录）::

    streamlit run ui/app.py

不做券商下单、用户系统或 ``logs/driver.py`` 全量编排；全市场 / 全因子很慢，需自备数据。
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

# ── 路径 / 解释器 ─────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
OUTPUT_DIR = REPO_ROOT / "research" / "output"
RUN_PY = REPO_ROOT / "run.py"
LOG_TAIL_CHARS = 24_000

MODES = [
    "ridge",
    "cat",
    "lgbm",
    "xgb",
    "rf",
    "mlp",
    "ensemble",
    "linear",
    "dynamic",
    "industry",
]
HORIZONS = [5, 10, 20, 60]
IC_PERIODS = [5, 20]
CAP_BANDS = [
    "all",
    "micro_30",
    "micro_small_100",
    "micro",
    "small",
    "small_mid",
    "small_mid_wide",
    "mid",
]
LABEL_MODES = [
    "cs_zscore",
    "cs_rank",
    "raw",
    "top40_cs_zscore",
    "cs_rank_softlong",
    "barra_residual",
    "triple_barrier",
]


def _venv_python() -> Path:
    """优先用仓库 ``.venv`` 里的 Python，找不到则退回当前解释器。"""
    win = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    unix = REPO_ROOT / ".venv" / "bin" / "python"
    if win.is_file():
        return win
    if unix.is_file():
        return unix
    return Path(sys.executable)


def _list_factor_yamls() -> list[str]:
    if not CONFIG_DIR.is_dir():
        return []
    files = sorted(CONFIG_DIR.glob("*.yaml"), key=lambda p: p.name.lower())
    return [str(p.relative_to(REPO_ROOT)).replace("\\", "/") for p in files]


def _list_ic_factor_sources() -> list[str]:
    """可选短名单 / YAML / JSON / txt，供展开为 ``--factors``。"""
    found: list[Path] = []
    if CONFIG_DIR.is_dir():
        found.extend(CONFIG_DIR.glob("*.yaml"))
        found.extend(CONFIG_DIR.glob("*.yml"))
    if OUTPUT_DIR.is_dir():
        found.extend(OUTPUT_DIR.glob("selected_factors_*.json"))
        found.extend(OUTPUT_DIR.glob("shortlist*/shortlist_factors.json"))
        found.extend(OUTPUT_DIR.glob("shortlist*/shortlist_factors.txt"))
        found.extend(OUTPUT_DIR.glob("shortlist*/*.txt"))
    rels: list[str] = []
    seen: set[str] = set()
    for p in sorted(found, key=lambda x: str(x).lower()):
        if not p.is_file():
            continue
        rel = str(p.relative_to(REPO_ROOT)).replace("\\", "/")
        if rel in seen:
            continue
        seen.add(rel)
        rels.append(rel)
    return rels


def _load_factor_names_from_file(rel_path: str, period: int) -> list[str]:
    """从 yaml / json / txt 读因子名列表（UI 侧展开成 ``--factors``）。"""
    path = REPO_ROOT / rel_path
    if not path.is_file():
        raise FileNotFoundError(rel_path)
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        import yaml

        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        key = f"h{period}"
        block = cfg.get(key) if isinstance(cfg, dict) else None
        if not isinstance(block, dict) or not block.get("factors"):
            raise ValueError(f"{rel_path} 中无 {key}.factors")
        names = list(block["factors"])
    elif suffix == ".json":
        cfg = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(cfg, list):
            names = list(cfg)
        elif isinstance(cfg, dict):
            names = list(cfg.get("factors") or [])
        else:
            raise ValueError(f"{rel_path} JSON 格式无法识别")
        if not names:
            raise ValueError(f"{rel_path} 无 factors 列表")
    else:
        # txt / 其它：一行一个因子名
        names = [
            ln.strip()
            for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        if not names:
            raise ValueError(f"{rel_path} 为空")
    return [str(n).strip() for n in names if str(n).strip()]


def _quote(arg: str) -> str:
    """跨平台预览用引号（Windows PowerShell / bash 都尽量可读）。"""
    if os.name == "nt":
        if not arg or any(c in arg for c in ' \t\n\r"&|<>^'):
            return '"' + arg.replace('"', '\\"') + '"'
        return arg
    return shlex.quote(arg)


def build_run_argv(
    *,
    python: Path,
    mode: str,
    horizon: int,
    factor_config: str | None,
    cap_band: str,
    train_windows: str,
    val_window: int | None,
    feature_neutralize: bool,
    label_mode: str,
    objective: str,
    bid_ask_spread: float,
    sample: int,
    special_factors: str,
    output_dir: str,
    skip_download: bool,
) -> list[str]:
    argv = [str(python), str(RUN_PY), "--mode", mode, "--horizon", str(horizon)]

    if skip_download:
        argv.append("--skip-download")
    if sample and sample > 0:
        argv.extend(["--sample", str(sample)])
    if factor_config:
        argv.extend(["--factor-config", factor_config])
    if cap_band and cap_band != "all":
        argv.extend(["--cap-band", cap_band])
    tw = train_windows.strip()
    if tw:
        argv.extend(["--train-windows", tw])
    if val_window is not None:
        argv.extend(["--val-window", str(val_window)])
    if feature_neutralize:
        argv.append("--feature-neutralize")
    else:
        argv.append("--no-feature-neutralize")
    if label_mode and label_mode != "cs_zscore":
        argv.extend(["--label-mode", label_mode])
    if objective and objective != "regression":
        argv.extend(["--objective", objective])
    if bid_ask_spread is not None:
        argv.extend(["--bid-ask-spread", str(bid_ask_spread)])
    sf = special_factors.strip()
    if sf:
        argv.extend(["--special-factors", sf])
    od = output_dir.strip()
    if od:
        argv.extend(["--output-dir", od])
    return argv


def build_ic_argv(
    *,
    python: Path,
    period: int,
    barra: bool,
    save: bool,
    use_fdr: bool,
    min_long_share: float,
    cap_band: str,
    factors: str | None,
    resume: bool,
    fresh: bool,
    workers: int,
) -> list[str]:
    argv = [str(python), "-m", "research.ic_analysis_v2", "--period", str(period)]
    if barra:
        argv.append("--barra")
    if save:
        argv.append("--save")
    # CLI 默认 use_fdr=True；关时显式 --no-use-fdr，开时可不写（状态在 UI 展示）
    if not use_fdr:
        argv.append("--no-use-fdr")
    argv.extend(["--min-long-share", str(min_long_share)])
    if cap_band and cap_band != "all":
        argv.extend(["--cap-band", cap_band])
    if factors:
        argv.extend(["--factors", factors])
    if fresh:
        argv.append("--fresh")
    elif resume:
        argv.append("--resume")
    argv.extend(["--workers", str(max(1, int(workers)))])
    return argv


def format_cmd_preview(argv: list[str]) -> str:
    return " ".join(_quote(a) for a in argv)


def _init_session() -> None:
    defaults = {
        "proc": None,
        "log_text": "",
        "last_cmd": "",
        "run_started": None,
        "run_finished": None,
        "exit_code": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _poll_process() -> None:
    proc: subprocess.Popen | None = st.session_state.get("proc")
    if proc is None:
        return
    if proc.stdout is not None:
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            st.session_state.log_text += line
        if len(st.session_state.log_text) > LOG_TAIL_CHARS * 2:
            st.session_state.log_text = st.session_state.log_text[-LOG_TAIL_CHARS:]
    rc = proc.poll()
    if rc is not None:
        if proc.stdout is not None:
            rest = proc.stdout.read()
            if rest:
                st.session_state.log_text += rest
        st.session_state.exit_code = rc
        st.session_state.run_finished = time.time()
        st.session_state.proc = None


def _start_run(argv: list[str]) -> None:
    if st.session_state.proc is not None:
        st.warning("已有任务在跑，请先停止或等它结束。")
        return
    python = Path(argv[0])
    if not python.is_file():
        st.error(f"找不到 Python：{python}（请先创建 .venv 并 pip install -r requirements.txt）")
        return
    # run.py 路径校验仅在回测命令时需要
    if len(argv) >= 2 and Path(argv[1]).name == "run.py" and not RUN_PY.is_file():
        st.error(f"找不到 run.py：{RUN_PY}")
        return

    st.session_state.log_text = ""
    st.session_state.exit_code = None
    st.session_state.run_finished = None
    st.session_state.last_cmd = format_cmd_preview(argv)
    st.session_state.run_started = time.time()

    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    proc = subprocess.Popen(
        argv,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=creationflags,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    st.session_state.proc = proc


def _stop_run() -> None:
    proc: subprocess.Popen | None = st.session_state.get("proc")
    if proc is None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception as exc:  # noqa: BLE001 — UI 层展示即可
        st.error(f"停止失败: {exc}")
    finally:
        st.session_state.proc = None
        st.session_state.exit_code = -1
        st.session_state.run_finished = time.time()
        st.session_state.log_text += "\n\n[UI] 用户已请求停止进程。\n"


def _render_run_controls(argv: list[str], *, python: Path) -> None:
    """命令预览 + 启动/停止/日志（回测与 IC 共用）。"""
    preview = format_cmd_preview(argv)

    st.subheader("命令预览")
    st.code(preview, language="bash")
    st.caption(f"工作目录: `{REPO_ROOT}` · Python: `{python}`")

    c1, c2, c3, _ = st.columns([1, 1, 1, 3])
    with c1:
        run_clicked = st.button("启动运行", type="primary", use_container_width=True)
    with c2:
        stop_clicked = st.button("停止", use_container_width=True)
    with c3:
        refresh = st.button("刷新日志", use_container_width=True)

    if run_clicked:
        _start_run(argv)
        st.rerun()
    if stop_clicked:
        _stop_run()
        st.rerun()
    if refresh:
        st.rerun()

    running = st.session_state.proc is not None
    if running:
        st.success("任务运行中…（点「刷新日志」或稍等自动刷新）")
        time.sleep(1.5)
        st.rerun()
    elif st.session_state.exit_code is not None:
        code = st.session_state.exit_code
        if code == 0:
            st.success(f"已结束，退出码 {code}。请到输出目录查看结果。")
        elif code == -1:
            st.warning("进程已停止。")
        else:
            st.error(f"已结束，退出码 {code}。请查看下方日志。")

    if st.session_state.last_cmd:
        st.caption(f"最近一次命令: `{st.session_state.last_cmd}`")

    st.subheader("运行日志（尾部）")
    log = st.session_state.log_text or "（尚无日志）"
    if len(log) > LOG_TAIL_CHARS:
        log = "…\n" + log[-LOG_TAIL_CHARS:]
    st.text_area("log", value=log, height=360, label_visibility="collapsed")


def _render_backtest_tab(python: Path) -> None:
    st.caption(
        "把下面选项变成 ``run.py`` 命令并在本机跑起来。"
        " 全市场训练很慢；没有本机数据会直接失败。"
    )
    with st.expander("使用前请读（诚实说明）", expanded=True):
        st.markdown(
            """
- 仍需先安装 **Python**、创建虚拟环境、``pip install -r requirements.txt``，并**自行下载**行情/财务等数据（见 [docs/GETTING_STARTED.md](../docs/GETTING_STARTED.md)）。
- 本页只包装常用 CLI；批量编排请看 ``logs/driver.py``（本 UI **不做** driver）。
- 建议先用「仅前 N 只股票」冒烟；``sample=0`` 表示全市场，耗时长、吃内存。
- 输出默认写在 ``results/<tag>/``（或你指定的输出目录）。结果供人工二次筛选，**不是**自动交易。
            """
        )

    yamls = _list_factor_yamls()
    default_yaml = "config/factor_configs.yaml"
    if default_yaml not in yamls and yamls:
        default_yaml = yamls[0]

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("常用参数")
        mode = st.selectbox(
            "模式 mode（用什么模型打分）",
            MODES,
            index=MODES.index("ridge"),
            help="ridge=岭回归；cat/lgbm/xgb=树模型；ensemble=多模型集成。日常推荐先试 ridge。",
        )
        horizon = st.selectbox(
            "持仓期 horizon（交易日；5≈周 / 20≈月）",
            HORIZONS,
            index=HORIZONS.index(5),
            help="信号持有多久再调仓。5=约一周，20=约一月。",
        )
        factor_config = st.selectbox(
            "因子白名单 factor-config（用哪些因子）",
            options=yamls or ["（config/ 下没有 .yaml）"],
            index=(yamls.index(default_yaml) if default_yaml in yamls else 0),
            help="扫描 config/*.yaml。通常由 IC 筛选生成，也可手改。",
        )
        if not yamls:
            factor_config = None

        cap_band = st.selectbox(
            "市值带 cap-band（只在哪些市值股票里选）",
            CAP_BANDS,
            index=0,
            help=(
                "all=全市场；micro_30=流通市值≤30亿（无8亿地板）；"
                "micro_small_100=≤100亿。微盘流动性差、风险更大。"
            ),
        )
        train_windows = st.text_input(
            "训练窗口 train-windows（月数，逗号分隔）",
            value="6,12",
            help="例如 6,12：同时用近 6 个月与近 12 个月窗口训练再集成。留空则用程序默认。",
        )
        val_window = st.number_input(
            "验证窗口 val-window（月；0=不要独立验证集）",
            min_value=0,
            max_value=36,
            value=6,
            help="默认 6：多窗共用近期验证。0=训练贴到预测日（多窗须另设 wf-selection=average，本 UI 不暴露）。",
        )
        feature_neutralize = st.toggle(
            "特征中性化 feature-neutralize（去掉风格/行业敞口后再训练）",
            value=True,
            help="默认开：与 IC「纯因子」口径更一致。关掉会学到更多系统性风险。",
        )

    with right:
        st.subheader("标签 / 成本 / 试跑")
        label_mode = st.selectbox(
            "标签 label-mode（模型要预测的目标怎么标准化）",
            LABEL_MODES,
            index=0,
            help="默认 cs_zscore=截面 z 分数。barra_residual=先扣掉风格再当标签。",
        )
        objective = st.selectbox(
            "训练目标 objective（回归 / 排序）",
            ["regression", "rank"],
            index=0,
            help="日常请用 regression。rank=学习排序（CatBoost 侧接近 YetiRank 一类），实验向、更慢更挑参数。",
        )
        if objective == "rank":
            st.warning(
                "已选择 **rank**（排序学习）。CatBoost 会走 YetiRank 等 pairwise 损失；"
                "现网对照以 regression 为主，rank 曾有中止实验，非零基础推荐项。"
                " ridge / rf / mlp 不支持时会自动退回 regression。"
            )

        bid_ask = st.number_input(
            "买卖价差成本 bid-ask-spread（单边，单位 bp；10=0.1%）",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=1.0,
            help="回测里模拟买卖价差成本。小盘通常更贵；默认 10。",
        )
        sample = st.number_input(
            "快速试跑 sample（仅前 N 只股票；0=全市场）",
            min_value=0,
            max_value=5000,
            value=100,
            step=50,
            help="强烈建议先 100 冒烟。全市场可能数小时且需大内存。",
        )
        special_factors = st.text_input(
            "特殊因子 special-factors（可选，逗号分隔）",
            value="",
            placeholder="例如 event,size",
            help="白名单之外再注入的 pack，如 event / size。详见 docs/SPECIAL_FACTORS.md。",
        )
        output_dir = st.text_input(
            "输出目录 output-dir（可选）",
            value="",
            placeholder="留空 → results/<自动 tag>/",
            help="实验产物目录。勿提交 results/ 到 Git。",
        )
        skip_download = st.toggle(
            "跳过下载 skip-download（已有本地数据时请开）",
            value=True,
            help="关掉会尝试联网拉数据，慢且依赖外网接口。",
        )

    if sample == 0:
        st.info("sample=0：将跑全市场股票池。请确认本机数据齐全，并预留较长时间与内存。")

    argv = build_run_argv(
        python=python,
        mode=mode,
        horizon=int(horizon),
        factor_config=factor_config if yamls else None,
        cap_band=cap_band,
        train_windows=train_windows,
        val_window=int(val_window),
        feature_neutralize=feature_neutralize,
        label_mode=label_mode,
        objective=objective,
        bid_ask_spread=float(bid_ask),
        sample=int(sample),
        special_factors=special_factors,
        output_dir=output_dir,
        skip_download=skip_download,
    )
    _render_run_controls(argv, python=python)


def _render_ic_tab(python: Path) -> None:
    st.caption(
        "映射 ``python -m research.ic_analysis_v2`` 常用参数。"
        " **IC 全量很慢**（全 registry 可达数十分钟～数小时）；"
        "短名单 / ``--factors`` + ``--resume`` 较快。需自备本地数据。"
    )
    with st.expander("使用前请读（诚实说明）", expanded=True):
        st.markdown(
            f"""
- **全量 IC**：对注册表里大量因子算截面 IC +（可选）Barra 纯化 + FDR / corr-dedup 筛选，耗时长、吃内存；本机数据不全会直接失败。
- **较快路径**：用短名单 / YAML / JSON 限制 ``--factors``，或在已有 checkpoint 上开 ``--resume``（只改阈值时慎用 resume）。
- **输出（``--save``）**：主产物在 ``research/output/``：
  - ``selected_factors_h{{period}}.json``（及 cap-band 等后缀变体）
  - ``ic_summary_h{{period}}.csv`` / ``ic_barra_pure_*.csv`` 等
  - checkpoint：``research/output/_checkpoints/``
- **YAML 白名单**：本页**不**跑 ``logs/driver.py``；JSON → ``config/factor_configs.yaml`` 同步请另用 driver / 手改。回测页再选 YAML。
- 默认 **BH-FDR 已开启**（与 CLI 一致）；关掉会加 ``--no-use-fdr``。
- 默认 ``workers=1``（32GB 机器建议保持；勿与其它重任务叠高并发）。
            """
        )

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("筛选参数")
        period = st.selectbox(
            "持仓期 period / horizon（交易日）",
            IC_PERIODS,
            index=0,
            help="5≈周频 IC；20≈月频 IC。与回测 --horizon 对齐选用。",
        )
        barra = st.toggle(
            "Barra 纯 IC（--barra）",
            value=True,
            help="生产推荐：扣风格/行业后再算 IC。关掉则只看原始 IC。",
        )
        save = st.toggle(
            "写出结果（--save）",
            value=True,
            help="写出 selected_factors JSON 与 IC 汇总 CSV 到 research/output/。",
        )
        use_fdr = st.toggle(
            "BH-FDR 多重检验校正（默认已开）",
            value=True,
            help="CLI 默认 ON。关掉会传 --no-use-fdr。",
        )
        if use_fdr:
            st.caption("状态：FDR **开启**（命令里省略 ``--use-fdr``，与 CLI 默认一致）。")
        else:
            st.caption("状态：FDR **关闭**（将传 ``--no-use-fdr``）。")

        min_long_share = st.number_input(
            "稠密门 min-long-share（0=关闭）",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.05,
            help="分位多空分解的 long_share 下限；默认 0.4。设 0 关闭该门。",
        )
        cap_band = st.selectbox(
            "市值带 cap-band",
            CAP_BANDS,
            index=0,
            help="缩到指定流通市值带再算 IC。产物文件名可能带 cap_ 后缀。",
        )

    with right:
        st.subheader("因子范围 / 续跑")
        factor_src = st.radio(
            "计算哪些因子",
            ["全部（很慢）", "手动输入", "从文件选取"],
            index=0,
            help="全部=registry 全量；文件可选 shortlist / YAML / selected JSON。",
        )
        factors_arg: str | None = None
        if factor_src == "手动输入":
            factors_manual = st.text_input(
                "因子名（逗号分隔）→ --factors",
                value="",
                placeholder="例如 WQ_013,GTJA_099,涨跌停状态",
                help="名称须在 get_factor_registry() 中。",
            )
            factors_arg = factors_manual.strip() or None
        elif factor_src == "从文件选取":
            sources = _list_ic_factor_sources()
            pick = st.selectbox(
                "短名单 / YAML / JSON / txt",
                options=sources or ["（未找到可用文件）"],
                help="YAML 读 h{period}.factors；JSON 读 factors；txt 一行一名。",
            )
            if sources:
                try:
                    names = _load_factor_names_from_file(pick, int(period))
                    factors_arg = ",".join(names)
                    st.caption(f"已展开 **{len(names)}** 个因子 → ``--factors``。")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"读取失败: {exc}")
                    factors_arg = None
            else:
                st.warning("未找到 config/*.yaml 或 research/output 下的短名单/JSON。")
        else:
            st.info("将对注册表中的大量因子全量计算，请预留很长时间与内存。")

        workers = st.number_input(
            "并行 workers（默认 1）",
            min_value=1,
            max_value=8,
            value=1,
            help="IC 因子级并行。32GB 建议保持 1。",
        )

        st.markdown("**续跑 / 重算（危险项）**")
        resume = st.toggle(
            "从 checkpoint 续跑（--resume）",
            value=False,
            help="跳过已完成阶段。改阈值后勿盲目 resume；需全量重算用 --fresh。",
        )
        fresh = st.toggle(
            "清空 checkpoint 全量重算（--fresh）",
            value=False,
            help="删除本 period 相关 checkpoint 后重算。不可恢复，确认后再开。",
        )
        if fresh:
            st.error(
                "**危险**：``--fresh`` 会清空本 period 的 IC checkpoint 并全量重算。"
                "与 ``--resume`` / ``--factors`` 同时开时，CLI 以 fresh 优先。"
            )
        elif resume:
            st.warning(
                "``--resume`` 会复用旧 checkpoint。筛选阈值或 Barra 口径变了仍 resume，"
                "可能得到过期结论；不确定时用 ``--fresh``。"
            )

    if factor_src == "全部（很慢）":
        st.info("未限制 ``--factors``：全量 IC。若已有 checkpoint，可考虑开 ``--resume``。")

    argv = build_ic_argv(
        python=python,
        period=int(period),
        barra=barra,
        save=save,
        use_fdr=use_fdr,
        min_long_share=float(min_long_share),
        cap_band=cap_band,
        factors=factors_arg,
        resume=resume,
        fresh=fresh,
        workers=int(workers),
    )
    _render_run_controls(argv, python=python)


def main() -> None:
    st.set_page_config(
        page_title="量化选股 · 简易运行面板",
        page_icon="📈",
        layout="wide",
    )
    _init_session()
    _poll_process()

    st.title("量化选股 · 简易运行面板")
    st.caption(
        "本地包装常用 CLI：**回测**（``run.py``）与 **因子筛选**（``ic_analysis_v2``）。"
        " **不做**券商下单 / 账号系统 / ``logs/driver.py`` 编排 / 一键无数据复现。"
    )

    python = _venv_python()
    tab_bt, tab_ic = st.tabs(["回测", "因子筛选"])
    with tab_bt:
        _render_backtest_tab(python)
    with tab_ic:
        _render_ic_tab(python)

    st.divider()
    st.markdown(
        "更多说明：[docs/UI.md](../docs/UI.md) · "
        "[docs/CLI_QUICKSTART.md](../docs/CLI_QUICKSTART.md) · "
        "[docs/GETTING_STARTED.md](../docs/GETTING_STARTED.md)"
    )


if __name__ == "__main__":
    main()
