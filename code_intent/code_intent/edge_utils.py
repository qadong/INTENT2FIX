from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator


EXECUTABLE_EDGE_TYPES = frozenset({"call", "dependent_call"})


def iter_executable_edges(path: str | Path) -> Iterator[Dict[str, Any]]:
    """Yield call relations, accepting legacy rows that predate edge_type."""
    edge_path = Path(path)
    if not edge_path.exists():
        return

    with edge_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                edge = json.loads(line)
            except json.JSONDecodeError:
                continue
            if edge.get("edge_type", "call") not in EXECUTABLE_EDGE_TYPES:
                continue
            if edge.get("caller") and edge.get("callee"):
                yield edge
