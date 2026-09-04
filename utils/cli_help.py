"""CLI help helpers: hide advanced / deprecated flags from default ``--help``.

Usage::

    from utils.cli_help import add_help_advanced, exit_if_help_advanced, help_text

    parser = argparse.ArgumentParser(...)
    add_help_advanced(parser)
    parser.add_argument("--foo", help=help_text("日常可见"))
    parser.add_argument("--bar", help=help_text("高级", advanced=True))
    parser.add_argument("--old", help=help_text("已弃用 no-op", deprecated=True))
    args = parser.parse_args()
    exit_if_help_advanced(parser, args)

When ``--help-advanced`` is present in ``sys.argv``, advanced/deprecated help
strings are registered (visible). After parse, ``exit_if_help_advanced`` prints
full help and exits so the pipeline does not run.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any


def show_advanced() -> bool:
    return "--help-advanced" in sys.argv


def _escape_percent(text: str) -> str:
    """argparse help uses ``%``-formatting; escape lone ``%`` as ``%%``."""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "%":
            if i + 1 < n and text[i + 1] == "%":
                out.append("%%")
                i += 2
            else:
                out.append("%%")
                i += 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def help_text(
    text: str,
    *,
    advanced: bool = False,
    deprecated: bool = False,
) -> Any:
    """Return help string, or ``argparse.SUPPRESS`` when hidden from default help."""
    if deprecated:
        if show_advanced():
            return _escape_percent(f"[deprecated] {text}")
        return argparse.SUPPRESS
    if advanced and not show_advanced():
        return argparse.SUPPRESS
    return _escape_percent(text)


def add_help_advanced(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--help-advanced",
        action="store_true",
        help="显示全部参数（含高级 / deprecated 隐藏项）后退出",
    )


def exit_if_help_advanced(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if getattr(args, "help_advanced", False):
        parser.print_help()
        raise SystemExit(0)
