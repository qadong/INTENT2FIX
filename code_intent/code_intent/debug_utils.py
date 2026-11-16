from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _get_edge_log_path() -> Optional[Path]:
    path = os.environ.get("EDGE_LOG_PATH")
    if not path:
        return None
    try:
        resolved = Path(path)
        if resolved.parent and not resolved.parent.exists():
            resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved
    except Exception:
        return None


def _edge_debug_enabled() -> bool:
    return os.environ.get("EDGE_DEBUG") == "1"


def log_edge_debug(message: str) -> None:
    log_path = _get_edge_log_path()
    if log_path:
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(message + "\n")
        except Exception:
            pass
    elif _edge_debug_enabled():
        print(message)


def _get_template_log_path() -> Optional[Path]:
    path = os.environ.get("TEMPLATE_LOG_PATH")
    if not path:
        return None
    try:
        resolved = Path(path)
        if resolved.parent and not resolved.parent.exists():
            resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved
    except Exception:
        return None


def _template_debug_enabled() -> bool:
    return os.environ.get("TEMPLATE_DEBUG") == "1"


def log_template_debug(message: str) -> None:
    log_path = _get_template_log_path()
    if log_path:
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(message + "\n")
        except Exception:
            pass
    elif _template_debug_enabled():
        print(message)


def _get_parse_log_path() -> Optional[Path]:
    path = os.environ.get("PARSE_LOG_PATH")
    if not path:
        return None
    try:
        resolved = Path(path)
        if resolved.parent and not resolved.parent.exists():
            resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved
    except Exception:
        return None


def _parse_debug_enabled() -> bool:
    return os.environ.get("PARSE_DEBUG") == "1"


def log_parse_debug(message: str) -> None:
    log_path = _get_parse_log_path()
    if log_path:
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(message + "\n")
        except Exception:
            pass
    elif _parse_debug_enabled():
        print(message)
