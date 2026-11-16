from __future__ import annotations

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def load_toenv(dotenv_path: Optional[str] = None, *, override: bool = False) -> Optional[Path]:
    """Load .env for code_intent. Tries repo root first, then code_intent/.env (legacy)."""
    if dotenv_path:
        candidate = Path(dotenv_path)
        if candidate.exists():
            load_dotenv(candidate, override=override)
            return candidate
        return None

    here = Path(__file__).resolve()
    for candidate in (
        here.parents[2] / ".env",  # …/INTENT2FIX/.env（仓库根）
        here.parents[1] / ".env",  # …/code_intent/.env
    ):
        if candidate.exists():
            load_dotenv(candidate, override=override)
            return candidate
    return None
