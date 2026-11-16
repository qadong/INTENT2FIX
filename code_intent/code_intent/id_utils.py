from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def to_posix_lower(path: str) -> str:
    return str(path).replace("\\", "/").lower()


def relpath_or_basename(path: str, project_root: str | Path) -> str:

    try:
        root = Path(project_root).resolve()
        p = Path(path).resolve()
        return p.relative_to(root).as_posix()
    except Exception:
        try:
            return Path(path).name
        except Exception:
            return str(path)


def stable_hash(text: str, length: int = 16) -> str:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return h[:length]


@dataclass(frozen=True)
class EntityId:

    kind: str
    value: str

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.kind}:{self.value}"


def make_entity_id(
    *,
    kind: str,
    usr: Optional[str],
    file_path: Optional[str],
    line: Optional[int],
    col: Optional[int] = None,
    project_root: str | Path = ".",
    extra: str = "",
) -> EntityId:
    if usr:
        return EntityId(kind=kind, value=usr)
    rel = relpath_or_basename(file_path or "unknown", project_root)
    loc = f"{rel}:{line or 0}:{col or 0}"
    payload = f"{kind}|{loc}|{extra}"
    return EntityId(kind=kind, value=f"hash_{stable_hash(payload)}")







