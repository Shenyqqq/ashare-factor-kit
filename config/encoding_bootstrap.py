"""UTF-8 console/pipe bootstrap — import before loguru on Windows."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def bootstrap_stdio_utf8() -> None:
    """Force UTF-8 stdio and console code page (Windows). Call at process entry."""
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass

    if sys.platform != "win32":
        return

    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        kernel32.SetConsoleOutputCP(65001)
        kernel32.SetConsoleCP(65001)
    except (AttributeError, OSError):
        pass


def utf8_subprocess_env() -> dict[str, str]:
    """Environment for child Python processes (driver / batch scripts)."""
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    if not (sys.stderr.isatty() and sys.stdout.isatty()):
        env.setdefault("LOGURU_COLORIZE", "0")
    return env


def configure_loguru() -> None:
    """Apply loguru settings after bootstrap_stdio_utf8()."""
    from loguru import logger

    piped = not (sys.stderr.isatty() and sys.stdout.isatty())
    if piped:
        os.environ.setdefault("LOGURU_COLORIZE", "0")

    if os.environ.get("LOGURU_COLORIZE") == "0":
        logger.remove()
        # stderr already UTF-8 via bootstrap_stdio_utf8(); keep console readable
        logger.add(sys.stderr, colorize=False)


def add_utf8_file_sink(path: str | Path, *, level: str = "DEBUG") -> int:
    """Add a loguru file sink that always writes UTF-8.

    Windows note: without encoding=\"utf-8\", open() may use the ANSI code page
    (often GBK). PowerShell Tee-Object / ``>`` often write UTF-16 LE and can also
    mis-decode UTF-8 pipes as GBK — prefer this sink over shell redirection.
    """
    from loguru import logger

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return logger.add(
        path,
        level=level,
        encoding="utf-8",  # Windows: must be explicit (default may be GBK)
        enqueue=True,
        colorize=False,
    )
