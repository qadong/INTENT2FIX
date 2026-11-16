from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np


def _cache_root() -> Path:
    for key in ("INTENT_CACHE_DIR", "CODE_INTENT_CACHE_DIR"):
        raw = os.environ.get(key)
        if raw:
            return Path(raw)
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "code_intent" / "intent_cache"


def embedding_cache_writes_allowed() -> bool:
    """Do not persist a silent local fallback under a configured remote model."""
    return (
        os.getenv("CODE_INTENT_EMBED_ABORT_ON_FALLBACK") == "1"
        or not os.getenv("VOYAGE_EMBED_MODEL")
    )


class GlobalEmbeddingCache:
    """Persist normalized vectors by repository, model, dimension, and exact text."""

    def __init__(
        self,
        cache_type: str,
        *,
        repo_id: Optional[str],
        model: str,
        dimension: int,
    ) -> None:
        cache_dir = _cache_root() / (repo_id or "common")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = cache_dir / f"{cache_type}_embeddings.sqlite3"
        self.model = model
        self.dimension = dimension
        self._connection = sqlite3.connect(self.cache_file, timeout=60.0)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                cache_key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                vector BLOB NOT NULL
            )
            """
        )
        self._connection.commit()

    def _cache_key(self, text: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "dimension": self.dimension,
                "text": text,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        row = self._connection.execute(
            "SELECT vector FROM embeddings WHERE cache_key = ?",
            (self._cache_key(text),),
        ).fetchone()
        if row is None:
            return None
        vector = np.frombuffer(row[0], dtype="float32").copy()
        if vector.shape != (self.dimension,):
            return None
        return vector

    def update(self, records: Iterable[Tuple[str, np.ndarray]]) -> None:
        rows = []
        for text, vector in records:
            normalized = np.asarray(vector, dtype="float32")
            if normalized.shape != (self.dimension,):
                raise ValueError(
                    "cached embedding dimension mismatch: "
                    f"{normalized.shape} != ({self.dimension},)"
                )
            rows.append(
                (
                    self._cache_key(text),
                    self.model,
                    self.dimension,
                    normalized.tobytes(),
                )
            )
        if not rows:
            return
        with self._connection:
            self._connection.executemany(
                """
                INSERT OR IGNORE INTO embeddings
                    (cache_key, model, dimension, vector)
                VALUES (?, ?, ?, ?)
                """,
                rows,
            )

    def close(self) -> None:
        self._connection.close()
