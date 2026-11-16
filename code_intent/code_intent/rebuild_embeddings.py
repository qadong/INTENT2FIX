from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from tqdm import tqdm

from .embedding_manifest import load_embedding_manifest, write_embedding_manifest_entry
from .embedding_text import build_feature_embedding_text
from .intent_extractor import (
    EMBED_DIM,
    _build_embedding_text,
    _build_faiss_index,
    _normalize,
)
from . import llm_utils


def _load_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_index(index, destination: Path) -> None:
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    from faiss import write_index  # local import to avoid unused dep when not needed

    write_index(index, str(tmp_path))
    tmp_path.replace(destination)


def _print(message: str) -> None:
    print(f"[RebuildEmbeddings] {message}")


EMBED_RETRY_LIMIT = 5
EMBED_RETRY_BASE_DELAY = 0.5


def _ensure_embedding_client():
    client = getattr(llm_utils, "_EMBED_CLIENT", None)
    if not client or not client.config.enabled or getattr(client, "_client", None) is None:
        raise RuntimeError(
            "Voyage embedding client not configured or SDK unavailable, cannot rebuild vectors."
        )
    return client


def _strict_embed_batch(texts: List[str], label: str, client) -> np.ndarray:
    delay = EMBED_RETRY_BASE_DELAY
    for attempt in range(1, EMBED_RETRY_LIMIT + 1):
        try:
            return client.embed_texts(texts)
        except Exception as exc:  # pragma: no cover
            _print(
                f"Voyage embed failed ({label}) attempt {attempt}/{EMBED_RETRY_LIMIT}: {exc}"
            )
            if attempt >= EMBED_RETRY_LIMIT:
                raise RuntimeError(
                    f"Voyage embedding failed after retries ({label})"
                ) from exc
            time.sleep(delay)
            delay = min(delay * 2, 5.0)
    raise RuntimeError(f"Voyage embedding unknown error ({label})")


def _embed_batches(
    texts: List[str], labels: List[str], client, *, batch_size: int = 32
) -> List[np.ndarray]:
    embeddings: List[np.ndarray] = []
    for start in tqdm(
        range(0, len(texts), batch_size),
        desc="Embedding batches",
        unit="batch",
    ):
        batch = texts[start : start + batch_size]
        label = f"{labels[start]}..{labels[min(start + len(batch) - 1, len(labels) - 1)]}"
        embeddings.extend(_strict_embed_batch(batch, label, client))
    return embeddings


def _rebuild_function_embeddings(base: Path) -> None:
    intents_path = base / "function_intents.jsonl"
    meta_path = base / "func_meta.jsonl"
    index_path = base / "func_index.faiss"

    intents = _load_jsonl(intents_path)
    meta = _load_jsonl(meta_path)
    if not meta:
        _print(f"func_meta.jsonl empty, skip ({meta_path})")
        return
    lookup = {record["func_id"]: record for record in intents}
    missing: List[str] = []
    texts: List[str] = []
    labels: List[str] = []
    client = _ensure_embedding_client()

    for meta_record in tqdm(meta, desc="Function vectors", unit="func"):
        func_id = meta_record["func_id"]
        record = lookup.get(func_id)
        if not record:
            missing.append(func_id)
            continue
        text = _build_embedding_text(record, record)
        texts.append(text)
        labels.append(f"func:{func_id}")

    if missing:
        raise ValueError(
            f"Cannot rebuild function index: {len(missing)} metadata rows have no "
            f"intent record (first: {missing[0]})."
        )

    embeddings = [
        _normalize(vec).astype("float32")
        for vec in _embed_batches(texts, labels, client)
    ]
    matrix = (
        np.vstack(embeddings) if embeddings else np.zeros((0, EMBED_DIM), dtype="float32")
    )
    index = _build_faiss_index(matrix)
    _write_index(index, index_path)
    prior_entry = (
        load_embedding_manifest(base).get("indexes", {}).get("functions") or {}
    )
    preserved_metadata = {
        key: prior_entry[key]
        for key in (
            "builder",
            "intent_model",
            "intent_count",
            "embedded_count",
            "fallback_count",
            "input_signatures",
        )
        if key in prior_entry
    }
    write_embedding_manifest_entry(
        base,
        "functions",
        model=os.getenv("VOYAGE_EMBED_MODEL"),
        dimension=EMBED_DIM,
        metadata=preserved_metadata,
    )
    _print(f"Function vectors written to {index_path} ({len(embeddings)} rows)")


def _rebuild_feature_embeddings(base: Path) -> None:
    features_path = base / "features.jsonl"
    meta_path = base / "feature_meta.jsonl"
    index_path = base / "feature_index.faiss"

    features = _load_jsonl(features_path)
    meta = _load_jsonl(meta_path)
    if not meta:
        _print(f"feature_meta.jsonl empty, skip ({meta_path})")
        return
    lookup = {feature["feature_id"]: feature for feature in features}
    missing: List[str] = []
    texts: List[str] = []
    labels: List[str] = []
    client = _ensure_embedding_client()

    for meta_record in tqdm(meta, desc="Feature vectors", unit="feat"):
        feature_id = meta_record["feature_id"]
        feature = lookup.get(feature_id)
        if not feature:
            missing.append(feature_id)
            continue
        text = build_feature_embedding_text(feature)
        texts.append(text)
        labels.append(f"feature:{feature_id}")

    if missing:
        raise ValueError(
            f"Cannot rebuild feature index: {len(missing)} metadata rows have no "
            f"feature record (first: {missing[0]})."
        )

    embeddings = [
        _normalize(vec).astype("float32")
        for vec in _embed_batches(texts, labels, client)
    ]
    matrix = (
        np.vstack(embeddings) if embeddings else np.zeros((0, EMBED_DIM), dtype="float32")
    )
    index = _build_faiss_index(matrix)
    _write_index(index, index_path)
    write_embedding_manifest_entry(
        base,
        "features",
        model=os.getenv("VOYAGE_EMBED_MODEL"),
        dimension=EMBED_DIM,
    )
    _print(f"Feature vectors written to {index_path} ({len(embeddings)} rows)")


def _parse_targets(raw: str) -> Sequence[str]:
    parts = [item.strip().lower() for item in raw.split(",") if item.strip()]
    valid = {"functions", "features"}
    targets = [item for item in parts if item in valid]
    if not targets:
        return ["functions", "features"]
    return list(dict.fromkeys(targets))


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild function/feature vector index without regenerating intents."
    )
    parser.add_argument("--index-dir", default="index_output", help="Index directory")
    parser.add_argument(
        "--targets",
        default="functions,features",
        help="Comma-separated: functions/features, default all",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    base = Path(args.index_dir)
    targets = _parse_targets(args.targets)

    if "functions" in targets:
        _rebuild_function_embeddings(base)
    if "features" in targets:
        _rebuild_feature_embeddings(base)


if __name__ == "__main__":
    main()
