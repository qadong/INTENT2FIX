#!/usr/bin/env python3
"""Seed repository-scoped embedding and verified intent reuse caches."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable

import faiss
import numpy as np


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _normalized_vector(index: faiss.Index, position: int) -> np.ndarray:
    vector = np.asarray(index.reconstruct(position), dtype="float32")
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0 else vector / norm


def _compatible_entry(
    entry: dict[str, Any],
    *,
    builder: str | tuple[str, ...],
    model: str,
    dimension: int,
) -> bool:
    builders = (builder,) if isinstance(builder, str) else builder
    return (
        entry.get("builder") in builders
        and entry.get("model") == model
        and entry.get("dimension") == dimension
        and entry.get("metric") == "inner_product"
        and entry.get("normalized") is True
    )


def _seed_records(
    *,
    index_dir: Path,
    index_name: str,
    artifact_prefix: str,
    cache_type: str,
    repo_id: str,
    builder: str | tuple[str, ...],
    model: str,
    dimension: int,
    texts_for_metadata: Callable[[list[dict[str, Any]]], list[str]],
) -> int:
    metadata_path = index_dir / f"{artifact_prefix}_meta.jsonl"
    index_path = index_dir / f"{artifact_prefix}_index.faiss"
    if not metadata_path.exists() or not index_path.exists():
        return 0

    from code_intent.embedding_cache import GlobalEmbeddingCache
    from code_intent.embedding_manifest import load_embedding_manifest

    entry = (
        load_embedding_manifest(index_dir).get("indexes", {}).get(index_name)
        or {}
    )
    if not _compatible_entry(
        entry, builder=builder, model=model, dimension=dimension
    ):
        return 0

    metadata = _load_jsonl(metadata_path)
    index = faiss.read_index(str(index_path))
    if index.d != dimension or index.ntotal != len(metadata):
        raise ValueError(
            f"incoherent {cache_type} index in {index_dir}: "
            f"dimension={index.d}, vectors={index.ntotal}, metadata={len(metadata)}"
        )
    if any(meta.get("index") != position for position, meta in enumerate(metadata)):
        raise ValueError(f"unordered {cache_type} metadata in {index_dir}")

    texts = texts_for_metadata(metadata)
    if len(texts) != len(metadata):
        raise ValueError(f"incomplete {cache_type} text mapping in {index_dir}")
    records = [
        (text, _normalized_vector(index, position))
        for position, text in enumerate(texts)
    ]
    cache = GlobalEmbeddingCache(
        cache_type,
        repo_id=repo_id,
        model=model,
        dimension=dimension,
    )
    cache.update(records)
    cache.close()
    return len(records)


def _seed_instance(
    index_dir: Path, *, repo_id: str, model: str, dimension: int
) -> dict[str, int]:
    from code_intent.embedding_manifest import (
        FEATURE_INTENT_BUILDER,
        FUNCTION_INTENT_BUILDER,
        TEST_INTENT_BUILDER,
    )
    from code_intent.embedding_text import build_feature_embedding_text
    from code_intent.intent_extractor import (
        GlobalIntentCache,
        _build_embedding_text,
        _build_test_embedding_text,
        _function_context,
        _function_intent_payload,
        _function_intent_generation_reusable,
        _load_context_data,
        _should_embed_function,
    )
    from code_intent.llm_utils import (
        FUNCTION_INTENT_VALIDATION_CONTRACT,
        _legacy_test_intent_input_hash_v4,
        function_intent_input_hash,
        test_intent_input_hash,
    )

    functions = {
        record["id"]: record
        for record in _load_jsonl(index_dir / "functions.jsonl")
    }
    function_intents = {
        record["func_id"]: record
        for record in _load_jsonl(index_dir / "function_intents.jsonl")
    }

    function_context = _load_context_data(index_dir, list(functions.values()))
    reusable_function_intents: dict[str, dict[str, Any]] = {}
    runtime_model = os.getenv("LLM_MODEL") or "heuristic-fallback"
    for func_id, intent in function_intents.items():
        func = functions.get(func_id)
        if func is None or not _should_embed_function(func):
            continue
        input_hash = function_intent_input_hash(
            func, _function_context(func, function_context)
        )
        if not (
            intent.get("intent_input_hash") == input_hash
            and intent.get("intent_builder") == FUNCTION_INTENT_BUILDER
            and intent.get("intent_model") == runtime_model
            and intent.get("intent_validation_contract")
            == FUNCTION_INTENT_VALIDATION_CONTRACT
            and _function_intent_generation_reusable(intent)
        ):
            continue
        reusable_function_intents[input_hash] = {
            "content_hash": func.get("content_hash"),
            "intent_input_hash": input_hash,
            "intent_builder": FUNCTION_INTENT_BUILDER,
            "intent_model": runtime_model,
            "intent_validation_contract": FUNCTION_INTENT_VALIDATION_CONTRACT,
            **_function_intent_payload(intent),
        }

    function_intent_cache = GlobalIntentCache("function", repo_id=repo_id)
    function_intent_cache.update(list(reusable_function_intents.values()))

    def function_texts(metadata: list[dict[str, Any]]) -> list[str]:
        return [
            _build_embedding_text(
                functions[meta["func_id"]], function_intents[meta["func_id"]]
            )
            for meta in metadata
        ]

    test_cases_path = index_dir / "test_cases.jsonl"
    test_intents_path = index_dir / "test_intents.jsonl"
    if test_cases_path.exists() and test_intents_path.exists():
        test_cases = {
            (
                str(record.get("id") or ""),
                str(record.get("def_file") or ""),
                int(record.get("def_start_line") or 0),
            ): record
            for record in _load_jsonl(test_cases_path)
        }
        test_intents = _load_jsonl(test_intents_path)
    else:
        test_cases = {}
        test_intents = []

    semantic_intents: dict[str, dict[str, Any]] = {}
    intent_fields = (
        "short_intent",
        "action",
        "object",
        "conditions",
        "side_effects",
        "raw_summary",
        "usage_context",
        "oracle",
        "fixture",
        "covered_symbols",
        "_intent_generation",
    )
    for intent in test_intents:
        key = (
            str(intent.get("func_id") or ""),
            str(intent.get("file") or ""),
            int(intent.get("start_line") or 0),
        )
        case = test_cases.get(key)
        if case is None or intent.get("intent_model") != runtime_model:
            continue
        source_builder = intent.get("intent_builder")
        if source_builder == TEST_INTENT_BUILDER:
            source_hash = test_intent_input_hash(case)
        elif source_builder == "test-intent-v4":
            source_hash = _legacy_test_intent_input_hash_v4(case)
        else:
            continue
        if (
            intent.get("intent_input_hash") != source_hash
            or not _function_intent_generation_reusable(intent)
        ):
            continue
        semantic_hash = test_intent_input_hash(case)
        semantic_intents[semantic_hash] = {
            "content_hash": semantic_hash,
            "intent_input_hash": semantic_hash,
            "intent_builder": TEST_INTENT_BUILDER,
            "intent_model": runtime_model,
            **{
                field: intent[field]
                for field in intent_fields
                if field in intent
            },
        }

    intent_cache = GlobalIntentCache("test", repo_id=repo_id)
    intent_cache.update(list(semantic_intents.values()))

    def test_texts(metadata: list[dict[str, Any]]) -> list[str]:
        texts = []
        for meta in metadata:
            record_index = int(meta["record_index"])
            intent = test_intents[record_index]
            key = (
                str(meta.get("func_id") or ""),
                str(meta.get("file") or ""),
                int(meta.get("start_line") or 0),
            )
            case = test_cases[key]
            if intent.get("func_id") != meta.get("func_id"):
                raise ValueError(f"test metadata mismatch in {index_dir}")
            texts.append(_build_test_embedding_text(case, intent))
        return texts

    features_path = index_dir / "features.jsonl"
    features = (
        {
            record["feature_id"]: record
            for record in _load_jsonl(features_path)
        }
        if features_path.exists()
        else {}
    )

    def feature_texts(metadata: list[dict[str, Any]]) -> list[str]:
        return [
            build_feature_embedding_text(features[meta["feature_id"]])
            for meta in metadata
        ]

    return {
        "function": _seed_records(
            index_dir=index_dir,
            index_name="functions",
            artifact_prefix="func",
            cache_type="function",
            repo_id=repo_id,
            builder=FUNCTION_INTENT_BUILDER,
            model=model,
            dimension=dimension,
            texts_for_metadata=function_texts,
        ),
        "function_intent": len(reusable_function_intents),
        "test": _seed_records(
            index_dir=index_dir,
            index_name="tests",
            artifact_prefix="test",
            cache_type="test",
            repo_id=repo_id,
            builder=(TEST_INTENT_BUILDER, "test-intent-v4"),
            model=model,
            dimension=dimension,
            texts_for_metadata=test_texts,
        ),
        "test_intent": len(semantic_intents),
        "feature": _seed_records(
            index_dir=index_dir,
            index_name="features",
            artifact_prefix="feature",
            cache_type="feature",
            repo_id=repo_id,
            builder=FEATURE_INTENT_BUILDER,
            model=model,
            dimension=dimension,
            texts_for_metadata=feature_texts,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Seed exact-text embedding caches and verified intent reuse "
            "keys from compatible intent indexes."
        )
    )
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--model", default=os.getenv("VOYAGE_EMBED_MODEL") or "voyage-2"
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=int(os.getenv("CODE_INTENT_EMBED_DIM", "768")),
    )
    args = parser.parse_args()
    os.environ["INTENT_CACHE_DIR"] = str(args.cache_dir)

    totals = {
        "function": 0,
        "function_intent": 0,
        "test": 0,
        "test_intent": 0,
        "feature": 0,
    }
    instances = 0
    for instance_root in sorted(args.index_root.iterdir()):
        index_dir = instance_root / "index_output"
        if not (index_dir / "embedding_manifest.json").exists():
            continue
        repo_id = instance_root.name.rsplit("-", 1)[0]
        seeded = _seed_instance(
            index_dir,
            repo_id=repo_id,
            model=args.model,
            dimension=args.dimension,
        )
        instances += 1
        for cache_type, count in seeded.items():
            totals[cache_type] += count
        print(json.dumps({"instance_id": instance_root.name, **seeded}))
    print(json.dumps({"instances": instances, "seeded": totals}))


if __name__ == "__main__":
    main()
