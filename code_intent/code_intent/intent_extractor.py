from __future__ import annotations

import json
import asyncio
import hashlib
import re
import uuid
from collections import defaultdict
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import faiss
import numpy as np

from .embedding_manifest import (
    FUNCTION_INTENT_BUILDER,
    TEST_INTENT_BUILDER,
    load_embedding_manifest,
    write_embedding_manifest_entry,
)
from .embedding_cache import GlobalEmbeddingCache, embedding_cache_writes_allowed
from .edge_utils import iter_executable_edges
from .intent_params import TOKEN_BUDGET

from .llm_utils import (
    EMBED_DIM,
    FUNCTION_INTENT_VALIDATION_CONTRACT,
    close_llm_async_client,
    embed_texts,
    generate_intent_from_code_async,
    generate_test_intent_from_code_async,
    generate_class_intent_async,
    generate_stub_intent,
    function_intent_input_hash,
    test_intent_input_hash,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

logger = logging.getLogger(__name__)

INTERNAL_LINE_THRESHOLD = 6
HELPER_CALLER_THRESHOLD = int(os.getenv("HELPER_CALLER_THRESHOLD", "1"))
EMBED_HELPER_METHODS = os.getenv("EMBED_HELPER_METHODS", "0") == "1"
EMBED_CTORS_DTORS = os.getenv("EMBED_CTORS_DTORS", "0") == "1"
BATCH_SIZE = max(1, int(os.getenv("INTENT_BATCH_SIZE", "100")))
EMBED_BATCH_SIZE = max(1, int(os.getenv("INTENT_EMBED_BATCH_SIZE", "100")))
INTENT_CONCURRENCY = max(1, int(os.getenv("INTENT_CONCURRENCY", "8")))


def _global_intent_cache_dir() -> Path:
    for key in ("INTENT_CACHE_DIR", "CODE_INTENT_CACHE_DIR"):
        raw = os.environ.get(key)
        if raw:
            return Path(raw)
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "code_intent" / "intent_cache"


GLOBAL_INTENT_CACHE_DIR = _global_intent_cache_dir()
TEST_FILE_MARKERS = ("test/", "tests/", "test\\", "tests\\")
TEST_FILE_SUFFIXES = ("_test", "_tests", "-test", "-tests")
GTEST_METHOD_NAMES = {"testbody", "setuptestcase", "teardowntestcase"}


def _looks_like_test_record(func: Dict[str, Any]) -> bool:
    file_path = (func.get("def_file") or func.get("file") or "").lower()
    if "/tests/abi/include/" in file_path or "/test/abi/include/" in file_path:
        return False
    if any(marker in file_path for marker in TEST_FILE_MARKERS):
        return True
    if file_path:
        try:
            filename = Path(file_path).name.lower()
        except Exception:
            filename = file_path.rsplit("/", 1)[-1]
        for suffix in (".cc", ".cpp", ".cxx", ".c", ".mm", ".m", ".hpp", ".hh", ".h", ".ipp", ".inc"):
            if filename.endswith(suffix):
                stem = filename[: -len(suffix)]
                if stem.endswith(TEST_FILE_SUFFIXES):
                    return True
                break
    short_name = (func.get("short_name") or func.get("signature") or "").lower()
    classes = [str(cls).lower() for cls in func.get("classes") or []]
    if classes:
        for cls in classes:
            if any(token in cls for token in ("_test", "_tests", "_fixture")) or cls.endswith("test") or cls.endswith("tests"):
                return True
    if short_name in GTEST_METHOD_NAMES and classes:
        return True
    return False


def _tests_touching_prod(
    callgraph_path: Path, func_lookup: Dict[str, Dict[str, Any]]
) -> Optional[Set[str]]:
    if not callgraph_path.exists():
        logger.warning("callgraph.jsonl missing, skipping test intent filter.")
        return None

    test_ids = {fid for fid, meta in func_lookup.items() if meta.get("is_test")}
    if not test_ids:
        return set()

    eligible: Set[str] = set()
    for edge in iter_executable_edges(callgraph_path):
        caller = edge["caller"]
        callee = edge["callee"]
        if caller not in test_ids:
            continue
        callee_meta = func_lookup.get(callee)
        if not callee_meta or callee_meta.get("is_test"):
            continue
        eligible.add(caller)
        if len(eligible) == len(test_ids):
            break
    return eligible


def load_functions(path: str | Path) -> List[Dict[str, Any]]:
    funcs: List[Dict[str, Any]] = []
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            funcs.append(json.loads(line))
    return funcs


def load_classes(path: str | Path) -> List[Dict[str, Any]]:
    classes: List[Dict[str, Any]] = []
    file_path = Path(path)
    if not file_path.exists():
        return []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            classes.append(json.loads(line))
    return classes


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_jsonl_atomic(records: Iterable[Dict[str, Any]], destination: Path) -> None:
    _ensure_parent_dir(destination)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp_path.replace(destination)


def _load_jsonl_if_exists(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    return records


def intent_test_artifacts_ready(
    index_dir: str | Path,
    *,
    require_current_inputs: bool = False,
    require_runtime_embedding: bool = False,
    require_current_generation: bool = True,
) -> bool:
    base = Path(index_dir)
    cases_path = base / "test_cases.jsonl"
    intents_path = base / "test_intents.jsonl"
    meta_path = base / "test_meta.jsonl"
    index_path = base / "test_index.faiss"
    if not all(
        path.exists() for path in (cases_path, intents_path, meta_path, index_path)
    ):
        return False
    if index_path.stat().st_size == 0:
        return False
    manifest_entry = (
        load_embedding_manifest(base).get("indexes", {}).get("tests") or {}
    )
    if manifest_entry.get("builder") != TEST_INTENT_BUILDER:
        return False
    if require_runtime_embedding and not (
        manifest_entry.get("model") == os.getenv("VOYAGE_EMBED_MODEL")
        and manifest_entry.get("dimension") == EMBED_DIM
    ):
        return False
    if require_current_inputs and (
        manifest_entry.get("input_signatures")
        != _test_input_signatures(base)
    ):
        return False
    build_id = manifest_entry.get("build_id")
    if not isinstance(build_id, str) or not build_id:
        return False
    try:
        cases = _load_jsonl_if_exists(cases_path)
        intents = _load_jsonl_if_exists(intents_path)
        metadata = _load_jsonl_if_exists(meta_path)
        index = faiss.read_index(str(index_path))
        expected_count = int(manifest_entry.get("case_count"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError, RuntimeError):
        return False
    if not (
        len(cases)
        == len(intents)
        == len(metadata)
        == index.ntotal
        == expected_count
        and manifest_entry.get("dimension") == index.d
    ):
        return False
    case_lookup = {
        (
            str(record.get("id") or ""),
            str(record.get("def_file") or ""),
            int(record.get("def_start_line") or 0),
        ): record
        for record in cases
    }
    if any(
        record.get("_build_id") != build_id
        for records in (cases, intents, metadata)
        for record in records
    ):
        return False
    for record in intents:
        key = (
            str(record.get("func_id") or ""),
            str(record.get("file") or ""),
            int(record.get("start_line") or 0),
        )
        case = case_lookup.get(key)
        if case is None:
            return False
        if not require_current_generation:
            continue
        expected_hash = test_intent_input_hash(case)
        if not (
            record.get("intent_builder") == TEST_INTENT_BUILDER
            and record.get("intent_input_hash") == expected_hash
            and record.get("content_hash") == expected_hash
            and _function_intent_generation_reusable(record)
        ):
            return False
    return manifest_entry.get("fallback_count") == 0


def _file_signature(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"size": path.stat().st_size, "sha256": digest.hexdigest()}


def _function_input_signatures(index_dir: Path) -> Dict[str, Any]:
    return {
        "functions": _file_signature(index_dir / "functions.jsonl"),
        "callgraph": _file_signature(index_dir / "callgraph.jsonl"),
    }


def _function_intents_semantic_signature(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    count = 0
    for record in _load_jsonl_if_exists(path):
        stable = {
            key: value
            for key, value in record.items()
            if key not in {"features", "_feature_build_id"}
        }
        digest.update(
            json.dumps(
                stable,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
        count += 1
    return {"sha256": digest.hexdigest(), "count": count}


def _test_input_signatures(index_dir: Path) -> Dict[str, Any]:
    return {
        **_function_input_signatures(index_dir),
        "function_intents": _function_intents_semantic_signature(
            index_dir / "function_intents.jsonl"
        ),
    }


def intent_function_artifacts_ready(index_dir: str | Path) -> bool:
    base = Path(index_dir)
    functions_path = base / "functions.jsonl"
    intents_path = base / "function_intents.jsonl"
    meta_path = base / "func_meta.jsonl"
    index_path = base / "func_index.faiss"
    if not all(
        path.exists() for path in (functions_path, intents_path, meta_path, index_path)
    ):
        return False
    entry = (
        load_embedding_manifest(base).get("indexes", {}).get("functions") or {}
    )
    build_id = entry.get("build_id")
    if not (
        entry.get("builder") == FUNCTION_INTENT_BUILDER
        and entry.get("validation_contract")
        == FUNCTION_INTENT_VALIDATION_CONTRACT
        and entry.get("model") == os.getenv("VOYAGE_EMBED_MODEL")
        and entry.get("dimension") == EMBED_DIM
        and entry.get("input_signatures") == _function_input_signatures(base)
        and isinstance(build_id, str)
        and bool(build_id)
    ):
        return False
    try:
        functions = load_functions(functions_path)
        intents = _load_jsonl_if_exists(intents_path)
        metadata = _load_jsonl_if_exists(meta_path)
        index = faiss.read_index(str(index_path))
        expected_intents = int(entry.get("intent_count"))
        expected_embeddings = int(entry.get("embedded_count"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError, RuntimeError):
        return False
    candidate_functions = {
        func["id"]: func
        for func in functions
        if not func.get("is_test") and not _looks_like_test_record(func)
    }
    intent_lookup = {record.get("func_id"): record for record in intents}
    if not (
        len(candidate_functions) == len(intents) == expected_intents
        and len(metadata) == index.ntotal == expected_embeddings
        and index.d == EMBED_DIM
        and set(intent_lookup) == set(candidate_functions)
        and all(record.get("_build_id") == build_id for record in intents)
        and all(record.get("_build_id") == build_id for record in metadata)
    ):
        return False
    context_data = _load_context_data(base, functions)
    return all(
        record.get("intent_builder") == FUNCTION_INTENT_BUILDER
        and record.get("intent_validation_contract")
        == FUNCTION_INTENT_VALIDATION_CONTRACT
        and record.get("intent_input_hash")
        == function_intent_input_hash(
            candidate_functions[func_id],
            (
                _function_context(candidate_functions[func_id], context_data)
                if _should_embed_function(candidate_functions[func_id])
                else None
            ),
        )
        and record.get("content_hash") == candidate_functions[func_id].get("content_hash")
        and bool(record.get("is_internal"))
        == (not _should_embed_function(candidate_functions[func_id]))
        and (
            not _should_embed_function(candidate_functions[func_id])
            or _function_intent_generation_reusable(record)
        )
        for func_id, record in intent_lookup.items()
    ) and entry.get("fallback_count") == 0


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"\w+|[^\s\w]", text))


def _truncate_context_by_budget(
    callers: List[str], callees: List[str], tests: List[str], budget: int
) -> Tuple[List[str], List[str], List[str], int]:
    if budget <= 0:
        return [], [], [], 0
    remaining = budget
    out_callers: List[str] = []
    out_callees: List[str] = []
    out_tests: List[str] = []

    def _consume(items: List[str], output: List[str], remaining_tokens: int) -> int:
        for item in items:
            cost = max(1, _estimate_tokens(item))
            if remaining_tokens - cost < 0:
                break
            output.append(item)
            remaining_tokens -= cost
        return remaining_tokens

    remaining = _consume(callers, out_callers, remaining)
    remaining = _consume(callees, out_callees, remaining)
    remaining = _consume(tests, out_tests, remaining)
    return out_callers, out_callees, out_tests, remaining


def _build_embedding_text(func: Dict[str, Any], intent: Dict[str, Any]) -> str:
    namespace = "::".join(func.get("namespaces", []))
    cls = "::".join(func.get("classes", []))
    signature = func.get("fq_name") or func.get("signature") or func.get("id", "")
    header_parts = [signature]
    if namespace:
        header_parts.append(namespace)
    if cls:
        header_parts.append(cls)
    header = "\n".join(filter(None, header_parts))
    conditions = "; ".join(intent.get("conditions") or [])
    side_effects = "; ".join(intent.get("side_effects") or [])
    return "\n".join(
        filter(
            None,
            [
                f"Intent: {intent.get('short_intent', '')}",
                f"Action: {intent.get('action', '')}",
                f"Object: {intent.get('object', '')}",
                f"Conditions: {conditions}" if conditions else "",
                f"Side effects: {side_effects}" if side_effects else "",
                f"Usage: {intent.get('usage_context', '')}",
                f"Summary: {intent.get('raw_summary', '')}",
                f"Symbol:\n{header}",
            ],
        )
    ).strip()


def _build_test_embedding_text(func: Dict[str, Any], intent: Dict[str, Any]) -> str:
    """Describe what a test validates, while retaining its concrete API context."""
    signature = func.get("fq_name") or func.get("signature") or func.get("id", "")
    conditions = "; ".join(intent.get("conditions") or [])
    side_effects = "; ".join(intent.get("side_effects") or [])
    code = (func.get("code") or "").strip()
    focal_sections = []
    for focal in (func.get("focal_entities") or [])[:3]:
        focal_sections.append("\n".join(filter(None, [
            f"Production symbol: {focal.get('name', '')}",
            f"Binding evidence: {', '.join(focal.get('relations') or [])}",
            f"Production intent: {focal.get('short_intent', '')}",
            f"Production summary: {focal.get('raw_summary', '')}",
            f"Production source skeleton:\n{str(focal.get('source') or '')[:600]}",
        ])))
    return "\n".join(filter(None, [
        f"Test: {signature}",
        f"Framework: {func.get('framework', '')}",
        f"Parent test or section: {func.get('parent_name', '')}",
        f"Fixture: {intent.get('fixture', '')}",
        f"Validation intent: {intent.get('short_intent', '')}",
        f"Action: {intent.get('action', '')}",
        f"Object: {intent.get('object', '')}",
        f"Preconditions: {conditions}" if conditions else "",
        (
            "Inherited setup:\n"
            f"{str(func.get('inherited_setup') or '')[:1200]}"
            if func.get("inherited_setup")
            else ""
        ),
        f"Observable effects: {side_effects}" if side_effects else "",
        f"Oracle: {intent.get('oracle') or func.get('oracle_code') or ''}",
        f"Observed assertions: {func.get('oracle_code') or ''}",
        f"Usage context: {intent.get('usage_context', '')}",
        f"Summary: {intent.get('raw_summary', '')}",
        f"Test source skeleton:\n{code[:1600]}" if code else "",
        "\n\n".join(focal_sections),
    ])).strip()


def _test_intent_cache_key(func: Dict[str, Any]) -> str:
    return test_intent_input_hash(func)


def _build_class_name(func: Dict[str, Any]) -> str | None:
    classes = func.get("classes") or []
    if not classes:
        return None
    namespaces = func.get("namespaces") or []
    return "::".join([*namespaces, *classes])


def _line_span(func: Dict[str, Any]) -> int:
    start = func.get("def_start_line") or 0
    end = func.get("def_end_line") or start
    return max(0, end - start)


def _is_constructor(func: Dict[str, Any]) -> bool:
    class_name = func.get("classes", [])[-1] if func.get("classes") else None
    return bool(class_name) and func.get("short_name") == class_name


def _is_destructor(func: Dict[str, Any]) -> bool:
    short = func.get("short_name") or ""
    return short.startswith("~")


def _is_canonical_public_header_function(func: Dict[str, Any]) -> bool:
    """Keep short public C++ forwarding APIs eligible for semantic indexing."""
    path = str(func.get("def_file") or func.get("file") or "")
    path = path.replace("\\", "/").lstrip("./")
    return (
        path.startswith("include/")
        and not func.get("is_test")
        and not func.get("is_internal")
    )


def _is_helper_method(func: Dict[str, Any]) -> bool:
    if not func.get("is_method"):
        return False
    if _is_constructor(func) or _is_destructor(func):
        return not EMBED_CTORS_DTORS
    if _is_canonical_public_header_function(func):
        return False
    lines = _line_span(func)
    low_caller = func.get("caller_count", 0) <= HELPER_CALLER_THRESHOLD
    if lines <= INTERNAL_LINE_THRESHOLD and low_caller:
        return not EMBED_HELPER_METHODS
    return False


def _should_embed_function(func: Dict[str, Any]) -> bool:
    if func.get("is_test"):
        return False
    if _is_helper_method(func):
        return False
    return True


def _load_context_data(
    index_dir: Path,
    functions: List[Dict[str, Any]],
    _intent_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[
    Dict[str, List[str]],
    Dict[str, List[str]],
    Dict[str, List[str]],
    Dict[str, str],
    Set[str],
    Dict[str, Dict[str, Any]],
]:
    caller_sets: Dict[str, Set[str]] = {}
    callee_sets: Dict[str, Set[str]] = {}
    test_sets: Dict[str, Set[str]] = {}
    name_lookup: Dict[str, str] = {}
    summary_lookup: Dict[str, Dict[str, Any]] = {}
    test_ids = set()
    for func in functions:
        fid = func.get("id")
        if not fid:
            continue
        name_lookup[fid] = func.get("fq_name") or func.get("signature") or func.get("short_name") or fid
        if func.get("is_test") or _looks_like_test_record(func):
            test_ids.add(fid)

    callgraph_path = index_dir / "callgraph.jsonl"
    if not callgraph_path.exists():
        return {}, {}, {}, name_lookup, test_ids, summary_lookup

    for edge in iter_executable_edges(callgraph_path):
        caller = edge["caller"]
        callee = edge["callee"]
        if caller not in name_lookup or callee not in name_lookup:
            continue
        caller_sets.setdefault(callee, set()).add(caller)
        callee_sets.setdefault(caller, set()).add(callee)
        if caller in test_ids:
            test_sets.setdefault(callee, set()).add(caller)

    def _ordered(mapping: Dict[str, Set[str]]) -> Dict[str, List[str]]:
        return {
            key: sorted(values, key=lambda fid: (name_lookup.get(fid, fid), fid))
            for key, values in mapping.items()
        }

    return (
        _ordered(caller_sets),
        _ordered(callee_sets),
        _ordered(test_sets),
        name_lookup,
        test_ids,
        summary_lookup,
    )


def _function_context(
    func: Dict[str, Any],
    context_data: Optional[Tuple[Dict, Dict, Dict, Dict, Set, Dict]],
) -> Optional[Dict[str, List[str]]]:
    if not context_data or not func.get("id"):
        return None
    caller_map, callee_map, test_map, name_lookup, test_ids, _ = context_data
    func_id = func["id"]

    def _unique_names(ids: Iterable[str], limit: int) -> List[str]:
        names: List[str] = []
        seen: Set[str] = set()
        for related_id in ids:
            name = name_lookup.get(related_id)
            if not name:
                continue
            if name in seen:
                continue
            seen.add(name)
            names.append(name)
            if len(names) >= limit:
                break
        return names

    callers = _unique_names(
        (
            caller
            for caller in caller_map.get(func_id, [])
            if caller not in test_ids
        ),
        10,
    )
    callees = _unique_names(callee_map.get(func_id, []), 10)
    tests = _unique_names(test_map.get(func_id, []), 6)
    callers, callees, tests, _ = _truncate_context_by_budget(
        callers, callees, tests, TOKEN_BUDGET
    )
    if not callers and not callees and not tests:
        return None
    return {
        "callers": callers,
        "callees": callees,
        "related_tests": tests,
    }


async def _generate_intents_for_batch_async(
    batch: List[Dict[str, Any]],
    respect_internal: bool = True,
    context_data: Optional[Tuple[Dict, Dict, Dict, Dict, Set, Dict]] = None,
    on_generated: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
) -> List[Tuple[Dict[str, Any], Dict[str, Any], bool]]:
    results: List[Optional[Dict[str, Any]]] = [None] * len(batch)
    should_embed_flags: List[bool] = [False] * len(batch)
    tasks = []
    semaphore = asyncio.Semaphore(INTENT_CONCURRENCY)

    for idx, func in enumerate(batch):
        if respect_internal:
            should_embed = _should_embed_function(func)
            func["is_internal"] = not should_embed
        else:
            should_embed = True
            func["is_internal"] = False
        should_embed_flags[idx] = should_embed
        
        if should_embed:
            context = _function_context(func, context_data)

            async def run_task(i, f, ctx):
                async with semaphore:
                    try:
                        if f.get("is_test"):
                            results[i] = await generate_test_intent_from_code_async(f)
                        else:
                            results[i] = await generate_intent_from_code_async(f, ctx)
                    except Exception as exc:
                        if os.getenv("INTENT_ABORT_ON_FALLBACK") == "1":
                            raise
                        logger.warning("Intent generation failed for %s: %s", f.get("id"), exc)
                        results[i] = generate_stub_intent(f)
                    if on_generated is not None:
                        on_generated(f, results[i])

            tasks.append(asyncio.create_task(run_task(idx, func, context)))
        else:
            results[idx] = generate_stub_intent(func)

    if tasks:
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await close_llm_async_client()

    output: List[Tuple[Dict[str, Any], Dict[str, Any], bool]] = []
    for idx, func in enumerate(batch):
        intent = results[idx] or generate_stub_intent(func)
        output.append((func, intent, should_embed_flags[idx]))
    return output


def _generate_intents_for_batch(
    batch: List[Dict[str, Any]],
    respect_internal: bool = True,
    context_data: Optional[Tuple[Dict, Dict, Dict, Dict, Set, Dict]] = None,
    on_generated: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
) -> List[Tuple[Dict[str, Any], Dict[str, Any], bool]]:
    return asyncio.run(
        _generate_intents_for_batch_async(batch, respect_internal, context_data, on_generated)
    )


def _grounded_function_identifiers(
    func: Dict[str, Any], context: Optional[Dict[str, Any]] = None
) -> List[str]:
    values: List[str] = []
    for value in (
        func.get("fq_name") or func.get("signature") or func.get("id"),
        func.get("return_type"),
        *(func.get("parameter_types") or []),
        *(func.get("namespaces") or []),
        *(func.get("classes") or []),
    ):
        if value:
            values.append(str(value))
    for related in (context or {}).values():
        if isinstance(related, list):
            values.extend(str(value) for value in related if value)
    return list(dict.fromkeys(values))


def _combine_func_and_intent(
    func: Dict[str, Any],
    intent: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "func_id": func["id"],
        "signature": func.get("fq_name") or func.get("id"),
        "file": func.get("def_file"),
        "start_line": func.get("def_start_line"),
        "end_line": func.get("def_end_line"),
        "namespaces": func.get("namespaces", []),
        "classes": func.get("classes", []),
        "class_name": _build_class_name(func),
        "grounded_identifiers": _grounded_function_identifiers(func, context),
        "caller_count": func.get("caller_count", 0),
        "is_internal": func.get("is_internal", False),
        "is_test": func.get("is_test", False),
        **intent,
        "embedding_dim": EMBED_DIM,
    }


def _build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(vectors.shape[1])
    if vectors.size:
        index.add(vectors.astype("float32"))
    return index


class GlobalIntentCache:

    def __init__(self, cache_type: str = "function", repo_id: Optional[str] = None):
        self.cache_type = cache_type
        if repo_id:
            self.cache_dir = GLOBAL_INTENT_CACHE_DIR / repo_id
        else:
            self.cache_dir = GLOBAL_INTENT_CACHE_DIR / "common"
            
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / f"{cache_type}_intents.jsonl"
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        if not self.cache_file.exists():
            return
        try:
            with self.cache_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    data = json.loads(line)
                    key = self._record_key(data)
                    if key:
                        self._cache[key] = data
        except Exception as e:
            logger.warning("Failed to load global intent cache: %s", e)

    def _record_key(self, record: Dict[str, Any]) -> Optional[str]:
        key = (
            record.get("intent_input_hash")
            if self.cache_type == "function"
            else record.get("content_hash")
        )
        return str(key) if key else None

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        if not cache_key: return None
        return self._cache.get(cache_key)

    def update(self, new_records: List[Dict[str, Any]]):
        if not new_records: return
        to_write = []
        for rec in new_records:
            key = self._record_key(rec)
            if key and key not in self._cache:
                self._cache[key] = rec
                to_write.append(rec)
        
        if not to_write: return

        try:
            with self.cache_file.open("a", encoding="utf-8") as f:
                for rec in to_write:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("Failed to update global intent cache: %s", e)


def _build_class_embedding_text(cls: Dict[str, Any], intent: Dict[str, Any]) -> str:
    fq_name = cls.get("fq_name") or cls.get("id", "")
    namespaces = "::".join(cls.get("namespaces") or [])
    bases = ", ".join(cls.get("bases") or [])
    
    parts = [
        f"Class: {fq_name}",
        f"Namespaces: {namespaces}" if namespaces else "",
        f"Bases: {bases}" if bases else "",
        f"Intent: {intent.get('short_intent', '')}",
        f"Responsibilities: {intent.get('responsibility', '')}",
        f"Summary: {intent.get('raw_summary', '')}",
    ]
    return "\n".join(filter(None, parts))


def build_class_intents(index_dir: str | Path = "index_output", repo_id: Optional[str] = None) -> None:
    base = Path(index_dir)
    classes_path = base / "classes.jsonl"
    intents_path = base / "class_intents.jsonl"
    meta_path = base / "class_meta.jsonl"
    index_path = base / "class_index.faiss"

    classes = load_classes(classes_path)
    if not classes:
        _write_jsonl_atomic([], intents_path)
        _write_jsonl_atomic([], meta_path)
        empty_index = faiss.IndexFlatIP(EMBED_DIM)
        faiss.write_index(empty_index, str(index_path))
        write_embedding_manifest_entry(
            base,
            "classes",
            model=os.getenv("VOYAGE_EMBED_MODEL"),
            dimension=EMBED_DIM,
        )
        return
    global_cache = GlobalIntentCache("class", repo_id=repo_id)

    records: List[Dict[str, Any]] = []
    meta_records: List[Dict[str, Any]] = []
    embeddings: List[np.ndarray] = []
    
    new_intents_for_cache = []

    total = len(classes)
    progress = (
        tqdm(total=total, desc="Class intents", unit="class")
        if tqdm
        else None
    )

    for start in range(0, total, BATCH_SIZE):
        batch = classes[start : start + BATCH_SIZE]
        batch_to_generate = []
        batch_intents = [] # (cls, intent)
        for cls in batch:
            cached = global_cache.get(cls.get("content_hash"))
            if cached:
                intent_part = {k: v for k, v in cached.items() if k not in {"class_id", "fq_name", "file", "content_hash", "embedding_dim"}}
                batch_intents.append((cls, intent_part))
            else:
                batch_to_generate.append(cls)
        if batch_to_generate:
            async def generate_batch_intents(cls_list):
                tasks = [generate_class_intent_async(cls) for cls in cls_list]
                try:
                    return await asyncio.gather(*tasks)
                finally:
                    await close_llm_async_client()

            generated_intents = asyncio.run(generate_batch_intents(batch_to_generate))
            for cls, intent in zip(batch_to_generate, generated_intents):
                batch_intents.append((cls, intent))
                cache_rec = {
                    "content_hash": cls.get("content_hash"),
                    **intent
                }
                new_intents_for_cache.append(cache_rec)
        texts_to_embed = []
        for cls, intent in batch_intents:
            record = {
                "class_id": cls["id"],
                "fq_name": cls["fq_name"],
                "file": cls.get("def_file"),
                "content_hash": cls.get("content_hash"),
                **intent,
                "embedding_dim": EMBED_DIM,
            }
            records.append(record)
            meta_records.append({"index": len(meta_records), "class_id": cls["id"]})
            
            text_for_embedding = _build_class_embedding_text(cls, intent)
            texts_to_embed.append(text_for_embedding)
        if texts_to_embed:
            batch_embs = embed_texts(texts_to_embed)
            for emb in batch_embs:
                embeddings.append(_normalize(emb).astype("float32"))
            
        if progress:
            progress.update(len(batch))
                
    if progress:
        progress.close()
    global_cache.update(new_intents_for_cache)
    if records:
        matrix = np.vstack(embeddings)
        index = _build_faiss_index(matrix)
        _write_jsonl_atomic(records, intents_path)
        _write_jsonl_atomic(meta_records, meta_path)
        faiss.write_index(index, str(index_path))
        write_embedding_manifest_entry(
            base,
            "classes",
            model=os.getenv("VOYAGE_EMBED_MODEL"),
            dimension=EMBED_DIM,
        )


_FUNCTION_INTENT_METADATA_FIELDS = frozenset({
    "func_id",
    "signature",
    "file",
    "start_line",
    "end_line",
    "namespaces",
    "classes",
    "class_name",
    "grounded_identifiers",
    "caller_count",
    "is_internal",
    "is_test",
    "content_hash",
    "intent_input_hash",
    "intent_builder",
    "intent_model",
    "intent_validation_contract",
    "_build_id",
    "_feature_build_id",
    "embedding_dim",
    "features",
})


def _function_intent_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in _FUNCTION_INTENT_METADATA_FIELDS
    }


def _function_intent_generation_reusable(record: Dict[str, Any]) -> bool:
    expected = "llm" if os.getenv("LLM_MODEL") else "heuristic"
    return record.get("_intent_generation") == expected


def _function_embedding_reuse_allowed(index_dir: Path) -> bool:
    entry = (
        load_embedding_manifest(index_dir).get("indexes", {}).get("functions")
        or {}
    )
    return (
        entry.get("builder") == FUNCTION_INTENT_BUILDER
        and entry.get("model") == os.getenv("VOYAGE_EMBED_MODEL")
        and entry.get("dimension") == EMBED_DIM
        and entry.get("metric") == "inner_product"
        and entry.get("normalized") is True
    )


def build_function_intents(
    index_dir: str | Path = "index_output",
    max_functions: int | None = None,
    repo_id: Optional[str] = None,
) -> None:
    base = Path(index_dir)
    functions_path = base / "functions.jsonl"
    intents_path = base / "function_intents.jsonl"
    meta_path = base / "func_meta.jsonl"
    index_path = base / "func_index.faiss"
    input_signatures = _function_input_signatures(base)
    build_id = uuid.uuid4().hex

    functions = load_functions(functions_path)
    if not functions:
        _write_jsonl_atomic([], intents_path)
        _write_jsonl_atomic([], meta_path)
        empty_index = faiss.IndexFlatIP(EMBED_DIM)
        faiss.write_index(empty_index, str(index_path))
        write_embedding_manifest_entry(
            base,
            "functions",
            model=os.getenv("VOYAGE_EMBED_MODEL"),
            dimension=EMBED_DIM,
            metadata={
                "builder": FUNCTION_INTENT_BUILDER,
                "build_id": build_id,
                "intent_model": os.getenv("LLM_MODEL") or "heuristic-fallback",
                "intent_count": 0,
                "embedded_count": 0,
                "fallback_count": 0,
                "input_signatures": input_signatures,
            },
        )
        return

    existing_intents: Dict[str, Dict[str, Any]] = {}
    existing_embeddings: Dict[str, np.ndarray] = {}

    if intents_path.exists():
        print("Loading existing function_intents.jsonl for incremental build...")
        with intents_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    intent = json.loads(line)
                    func_id = intent.get("func_id")
                    if func_id:
                        existing_intents[func_id] = intent

        if (
            _function_embedding_reuse_allowed(base)
            and meta_path.exists()
            and index_path.exists()
        ):
            existing_meta_index_map: Dict[int, str] = {}
            with meta_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        meta = json.loads(line)
                        func_id = meta.get("func_id")
                        idx = meta.get("index")
                        if func_id and idx is not None:
                            existing_meta_index_map[int(idx)] = func_id

            try:
                index = faiss.read_index(str(index_path))
                if index.d != EMBED_DIM:
                    raise ValueError(
                        f"function index dimension {index.d} != {EMBED_DIM}"
                    )
                for idx, func_id in sorted(existing_meta_index_map.items()):
                    if idx < 0 or idx >= index.ntotal:
                        continue
                    try:
                        emb = index.reconstruct(idx)
                        existing_embeddings[func_id] = _normalize(emb).astype("float32")
                    except Exception:
                        pass
            except Exception as e:
                print(f"Error loading existing embeddings: {e}, will regenerate.")
        elif meta_path.exists() and index_path.exists():
            print(
                "Existing function vectors have incompatible or missing provenance; "
                "they will be regenerated."
            )
        print(f"Loaded {len(existing_intents)} existing intents.")
    global_cache = GlobalIntentCache("function", repo_id=repo_id)
    embedding_cache = GlobalEmbeddingCache(
        "function",
        repo_id=repo_id,
        model=os.getenv("VOYAGE_EMBED_MODEL") or "",
        dimension=EMBED_DIM,
    )

    candidate_funcs = [
        func
        for func in functions
        if not func.get("is_test") and not _looks_like_test_record(func)
    ]
    if max_functions is not None and max_functions > 0:
        candidate_funcs = candidate_funcs[:max_functions]

    context_data = _load_context_data(base, functions)
    prepared: List[Dict[str, Any]] = []
    funcs_to_generate: List[Dict[str, Any]] = []
    generation_func_by_hash: Dict[str, Dict[str, Any]] = {}
    prepared_by_hash: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    local_reuse_count = 0
    cache_reuse_count = 0

    for func in candidate_funcs:
        should_embed = _should_embed_function(func)
        func["is_internal"] = not should_embed
        context = _function_context(func, context_data) if should_embed else None
        input_hash = function_intent_input_hash(func, context)
        prior = existing_intents.get(func["id"])
        intent: Optional[Dict[str, Any]] = None
        source = "generated"
        if (
            prior
            and prior.get("intent_builder") == FUNCTION_INTENT_BUILDER
            and prior.get("intent_input_hash") == input_hash
            and bool(prior.get("is_internal")) == (not should_embed)
            and (
                not should_embed
                or _function_intent_generation_reusable(prior)
            )
        ):
            intent = _function_intent_payload(prior)
            source = "local"
            local_reuse_count += 1
        elif should_embed:
            cached = global_cache.get(input_hash)
            if cached and _function_intent_generation_reusable(cached):
                intent = _function_intent_payload(cached)
                source = "cache"
                cache_reuse_count += 1
            else:
                source = "pending"
                if input_hash not in generation_func_by_hash:
                    generation_func_by_hash[input_hash] = func
                    funcs_to_generate.append(func)
        else:
            intent = generate_stub_intent(func)
            source = "stub"
        item = {
            "func": func,
            "context": context,
            "input_hash": input_hash,
            "intent": intent,
            "source": source,
            "should_embed": should_embed,
        }
        prepared.append(item)
        if source == "pending":
            prepared_by_hash[input_hash].append(item)

    total = len(funcs_to_generate)
    def checkpoint_function(func: Dict[str, Any], intent: Dict[str, Any]) -> None:
        if not _function_intent_generation_reusable(intent):
            return
        global_cache.update([{
            "content_hash": func.get("content_hash"),
            "intent_input_hash": function_intent_input_hash(
                func, _function_context(func, context_data)
            ),
            "intent_builder": FUNCTION_INTENT_BUILDER,
            "intent_model": os.getenv("LLM_MODEL") or "heuristic-fallback",
            "intent_validation_contract": FUNCTION_INTENT_VALIDATION_CONTRACT,
            **intent,
        }])

    progress = (
        tqdm(total=total, desc="Function intents", unit="func")
        if tqdm
        else None
    )
    for start in range(0, total, BATCH_SIZE):
        batch = funcs_to_generate[start : start + BATCH_SIZE]
        generated_results = _generate_intents_for_batch(
            batch, respect_internal=True, context_data=context_data,
            on_generated=checkpoint_function,
        )
        batch_cache_records: List[Dict[str, Any]] = []
        for func, intent, should_embed in generated_results:
            context = _function_context(func, context_data) if should_embed else None
            input_hash = function_intent_input_hash(func, context)
            for item in prepared_by_hash[input_hash]:
                item["intent"] = intent
                item["should_embed"] = should_embed
            if should_embed and _function_intent_generation_reusable(intent):
                batch_cache_records.append({
                    "content_hash": func.get("content_hash"),
                    "intent_input_hash": input_hash,
                    "intent_builder": FUNCTION_INTENT_BUILDER,
                    "intent_model": os.getenv("LLM_MODEL") or "heuristic-fallback",
                    "intent_validation_contract": FUNCTION_INTENT_VALIDATION_CONTRACT,
                    **intent,
                })
        global_cache.update(batch_cache_records)
        if progress:
            progress.update(len(batch))
    if progress:
        progress.close()
    fallback_count = sum(
        item["should_embed"]
        and not _function_intent_generation_reusable(item["intent"] or {})
        for item in prepared
    )

    records: List[Dict[str, Any]] = []
    meta_records: List[Dict[str, Any]] = []
    embeddings: List[Optional[np.ndarray]] = []
    embed_requests: List[Tuple[int, str]] = []
    embed_request_indices: Dict[str, List[int]] = defaultdict(list)
    embedding_cache_hits = 0
    local_embedding_reuse_count = 0
    local_embedding_cache_records: List[Tuple[str, np.ndarray]] = []
    for item in prepared:
        func = item["func"]
        intent = item["intent"] or generate_stub_intent(func)
        record = _combine_func_and_intent(func, intent, item["context"])
        record.update({
            "_build_id": build_id,
            "content_hash": func.get("content_hash"),
            "intent_input_hash": item["input_hash"],
            "intent_builder": FUNCTION_INTENT_BUILDER,
            "intent_model": os.getenv("LLM_MODEL") or "heuristic-fallback",
            "intent_validation_contract": FUNCTION_INTENT_VALIDATION_CONTRACT,
            "features": [],
        })
        records.append(record)
        if not item["should_embed"]:
            continue
        vector_index = len(meta_records)
        meta_records.append({
            "_build_id": build_id,
            "index": vector_index,
            "func_id": func["id"],
        })
        embedding_text = _build_embedding_text(func, intent)
        reusable_vector = (
            existing_embeddings.get(func["id"])
            if item["source"] == "local"
            else None
        )
        if reusable_vector is not None:
            local_embedding_reuse_count += 1
            local_embedding_cache_records.append(
                (embedding_text, reusable_vector)
            )
        if reusable_vector is None:
            reusable_vector = embedding_cache.get(embedding_text)
            if reusable_vector is not None:
                embedding_cache_hits += 1
        if reusable_vector is not None and reusable_vector.shape[0] == EMBED_DIM:
            embeddings.append(reusable_vector)
        else:
            embeddings.append(None)
            if embedding_text not in embed_request_indices:
                embed_requests.append((vector_index, embedding_text))
            embed_request_indices[embedding_text].append(vector_index)

    if local_embedding_cache_records and embedding_cache_writes_allowed():
        embedding_cache.update(local_embedding_cache_records)
    for start in range(0, len(embed_requests), EMBED_BATCH_SIZE):
        batch = embed_requests[start : start + EMBED_BATCH_SIZE]
        batch_vectors = embed_texts([text for _, text in batch])
        cache_records: List[Tuple[str, np.ndarray]] = []
        for (_vector_index, embedding_text), vector in zip(batch, batch_vectors):
            normalized = _normalize(vector).astype("float32")
            for vector_index in embed_request_indices[embedding_text]:
                embeddings[vector_index] = normalized
            cache_records.append((embedding_text, normalized))
        if embedding_cache_writes_allowed():
            embedding_cache.update(cache_records)
    close_embedding_cache = getattr(embedding_cache, "close", None)
    if close_embedding_cache is not None:
        close_embedding_cache()

    if not records:
        _write_jsonl_atomic([], intents_path)
        _write_jsonl_atomic([], meta_path)
        empty_index = faiss.IndexFlatIP(EMBED_DIM)
        faiss.write_index(empty_index, str(index_path))
        write_embedding_manifest_entry(
            base,
            "functions",
            model=os.getenv("VOYAGE_EMBED_MODEL"),
            dimension=EMBED_DIM,
            metadata={
                "builder": FUNCTION_INTENT_BUILDER,
                "build_id": build_id,
                "intent_model": os.getenv("LLM_MODEL") or "heuristic-fallback",
                "validation_contract": FUNCTION_INTENT_VALIDATION_CONTRACT,
                "intent_count": 0,
                "embedded_count": 0,
                "fallback_count": 0,
                "input_signatures": input_signatures,
            },
        )
        return

    if any(vector is None for vector in embeddings):
        raise RuntimeError("function embedding generation returned fewer vectors than requested")
    matrix = (
        np.vstack(embeddings).astype("float32")
        if embeddings
        else np.zeros((0, EMBED_DIM), dtype="float32")
    )
    index = _build_faiss_index(matrix)

    if _function_input_signatures(base) != input_signatures:
        raise RuntimeError(
            "functions.jsonl or callgraph.jsonl changed during intent construction"
        )
    _write_jsonl_atomic(records, intents_path)
    _write_jsonl_atomic(meta_records, meta_path)

    tmp_index_path = index_path.with_suffix(index_path.suffix + ".tmp")
    faiss.write_index(index, str(tmp_index_path))
    tmp_index_path.replace(index_path)
    write_embedding_manifest_entry(
        base,
        "functions",
        model=os.getenv("VOYAGE_EMBED_MODEL"),
        dimension=EMBED_DIM,
        metadata={
            "builder": FUNCTION_INTENT_BUILDER,
            "build_id": build_id,
            "intent_model": os.getenv("LLM_MODEL") or "heuristic-fallback",
            "validation_contract": FUNCTION_INTENT_VALIDATION_CONTRACT,
            "intent_count": len(records),
            "embedded_count": len(meta_records),
            "local_embedding_reuse": local_embedding_reuse_count,
            "embedding_cache_hits": embedding_cache_hits,
            "embedding_requests": len(embed_requests),
            "embedding_duplicate_reuse": sum(
                len(indices) - 1 for indices in embed_request_indices.values()
            ),
            "intent_duplicate_reuse": sum(
                len(items) - 1 for items in prepared_by_hash.values()
            ),
            "fallback_count": fallback_count,
            "input_signatures": input_signatures,
        },
    )
    print(
        "Function intents: "
        f"local reuse {local_reuse_count}, global cache reuse {cache_reuse_count}, "
        f"generated {len(funcs_to_generate)}, embedded {len(embed_requests)}, "
        f"local embedding reuse {local_embedding_reuse_count}, "
        f"intent duplicate reuse {sum(len(items) - 1 for items in prepared_by_hash.values())}, "
        f"embedding duplicate reuse {sum(len(indices) - 1 for indices in embed_request_indices.values())}, "
        f"embedding cache reuse {embedding_cache_hits}, "
        f"fallback {fallback_count}."
    )


def build_test_intents(
    index_dir: str | Path = "index_output",
    repo_id: Optional[str] = None,
    repo_root: str | Path | None = None,
    reuse_cache: bool = True,
    fallback_allowed: bool = True,
) -> None:
    base = Path(index_dir)
    functions_path = base / "functions.jsonl"
    test_intents_path = base / "test_intents.jsonl"
    test_meta_path = base / "test_meta.jsonl"
    test_index_path = base / "test_index.faiss"
    input_signatures = _test_input_signatures(base)

    existing_intents: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for record in _load_jsonl_if_exists(test_intents_path):
        key = (
            str(record.get("func_id") or ""),
            str(record.get("file") or ""),
            int(record.get("start_line") or 0),
        )
        existing_intents[key] = record

    functions = load_functions(functions_path)
    if repo_root is not None:
        from .test_case_extractor import collect_test_cases

        tests = collect_test_cases(base, repo_root)
        logger.info("Collected %d source-level test cases from %s", len(tests), repo_root)
    else:
        tests = [func for func in functions if func.get("is_test")]
    global_cache = GlobalIntentCache("test", repo_id=repo_id) if reuse_cache else None
    embedding_cache = (
        GlobalEmbeddingCache(
            "test",
            repo_id=repo_id,
            model=os.getenv("VOYAGE_EMBED_MODEL") or "",
            dimension=EMBED_DIM,
        )
        if reuse_cache
        else None
    )

    records: List[Dict[str, Any]] = []
    meta_records: List[Dict[str, Any]] = []
    embeddings: List[Optional[np.ndarray]] = []
    embedding_request_indices: Dict[str, List[int]] = defaultdict(list)
    embedding_cache_hits = 0
    embedding_requests = 0
    embedding_duplicate_reuse = 0
    local_reuse_count = 0
    cache_reuse_count = 0
    generated_count = 0
    total = len(tests)
    progress = (
        tqdm(total=total, desc="Test intents", unit="test")
        if tqdm
        else None
    )
    
    build_id = uuid.uuid4().hex

    def checkpoint_test(func: Dict[str, Any], intent: Dict[str, Any]) -> None:
        if global_cache is not None and _function_intent_generation_reusable(intent):
            global_cache.update([{
                "content_hash": _test_intent_cache_key(func),
                "intent_input_hash": _test_intent_cache_key(func),
                "intent_builder": TEST_INTENT_BUILDER,
                "intent_model": os.getenv("LLM_MODEL") or "heuristic-fallback",
                **intent,
            }])

    for start in range(0, total, BATCH_SIZE):
        batch = tests[start : start + BATCH_SIZE]
        batch_to_generate = []
        pending_by_hash: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        batch_results = []
        for func in batch:
            cache_key = _test_intent_cache_key(func)
            existing = existing_intents.get((
                str(func.get("id") or ""),
                str(func.get("def_file") or ""),
                int(func.get("def_start_line") or 0),
            ))
            if (
                reuse_cache
                and existing
                and existing.get("intent_builder") == TEST_INTENT_BUILDER
                and existing.get("intent_input_hash") == cache_key
                and _function_intent_generation_reusable(existing)
            ):
                intent_part = {
                    key: existing.get(key)
                    for key in (
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
                    if key in existing
                }
                batch_results.append((func, intent_part, True))
                local_reuse_count += 1
                continue
            cached = global_cache.get(cache_key) if global_cache is not None else None
            if (
                cached
                and cached.get("intent_builder") == TEST_INTENT_BUILDER
                and cached.get("intent_input_hash") == cache_key
                and _function_intent_generation_reusable(cached)
            ):
                intent_part = {
                    k: v
                    for k, v in cached.items()
                    if k
                    not in {
                        "func_id",
                        "signature",
                        "file",
                        "content_hash",
                        "intent_input_hash",
                        "intent_builder",
                        "intent_model",
                        "embedding_dim",
                    }
                }
                batch_results.append((func, intent_part, True))
                cache_reuse_count += 1
            else:
                if cache_key not in pending_by_hash:
                    batch_to_generate.append(func)
                pending_by_hash[cache_key].append(func)

        if batch_to_generate:
            generated_count += len(batch_to_generate)
            generated_results = _generate_intents_for_batch(
                batch_to_generate, respect_internal=False, on_generated=checkpoint_test
            )
            batch_cache_records = []
            for func, intent, _ in generated_results:
                for duplicate in pending_by_hash[_test_intent_cache_key(func)]:
                    batch_results.append((duplicate, intent, True))
                cache_rec = {
                    "content_hash": _test_intent_cache_key(func),
                    "intent_input_hash": _test_intent_cache_key(func),
                    "intent_builder": TEST_INTENT_BUILDER,
                    "intent_model": os.getenv("LLM_MODEL") or "heuristic-fallback",
                    **intent,
                }
                batch_cache_records.append(cache_rec)
            if global_cache is not None:
                global_cache.update(batch_cache_records)

        texts_to_embed = []
        for func, intent, _ in batch_results:
            record = _combine_func_and_intent(func, intent)
            record["_build_id"] = build_id
            record["content_hash"] = _test_intent_cache_key(func)
            record["intent_input_hash"] = _test_intent_cache_key(func)
            record["intent_builder"] = TEST_INTENT_BUILDER
            record["intent_model"] = os.getenv("LLM_MODEL") or "heuristic-fallback"
            record["features"] = []
            for key in (
                "framework",
                "parent_name",
                "inherited_setup",
                "oracle_code",
                "source_func_ids",
                "focal_entities",
            ):
                record[key] = func.get(key, [] if key.endswith("s") else "")
            record_index = len(records)
            records.append(record)
            meta_records.append({
                "_build_id": build_id,
                "index": len(meta_records),
                "record_index": record_index,
                "func_id": func["id"],
                "test_case_id": func["id"],
                "file": func.get("def_file"),
                "start_line": func.get("def_start_line"),
                "end_line": func.get("def_end_line"),
            })
            texts_to_embed.append(_build_test_embedding_text(func, intent))
            if progress:
                progress.update(1)

        if texts_to_embed:
            for embedding_text in texts_to_embed:
                reusable = (
                    embedding_cache.get(embedding_text)
                    if embedding_cache is not None
                    else None
                )
                if reusable is not None and reusable.shape == (EMBED_DIM,):
                    embeddings.append(reusable)
                    embedding_cache_hits += 1
                else:
                    embedding_request_indices[embedding_text].append(len(embeddings))
                    embeddings.append(None)

    missing_texts = list(embedding_request_indices)
    for start in range(0, len(missing_texts), EMBED_BATCH_SIZE):
        batch = missing_texts[start : start + EMBED_BATCH_SIZE]
        generated_vectors = embed_texts(batch)
        if len(generated_vectors) != len(batch):
            raise RuntimeError(
                "test embedding generation returned fewer vectors than requested"
            )
        cache_records: List[Tuple[str, np.ndarray]] = []
        for embedding_text, vector in zip(batch, generated_vectors):
            normalized = _normalize(vector).astype("float32")
            for index in embedding_request_indices[embedding_text]:
                embeddings[index] = normalized
            cache_records.append((embedding_text, normalized))
        if embedding_cache is not None and embedding_cache_writes_allowed():
            embedding_cache.update(cache_records)
        embedding_requests += len(batch)
    embedding_duplicate_reuse = sum(
        len(indices) - 1 for indices in embedding_request_indices.values()
    )
    if any(vector is None for vector in embeddings):
        raise RuntimeError(
            "test embedding generation returned fewer vectors than requested"
        )
    
    if progress:
        progress.close()
    if embedding_cache is not None:
        embedding_cache.close()

    matrix = (
        np.vstack([vector for vector in embeddings if vector is not None])
        if embeddings
        else np.zeros((0, EMBED_DIM), dtype="float32")
    )
    index = _build_faiss_index(matrix)
    case_records = [{**test, "_build_id": build_id} for test in tests]
    fallback_count = sum(
        not _function_intent_generation_reusable(record) for record in records
    )
    if not fallback_allowed and fallback_count:
        raise RuntimeError(
            f"test intent construction produced {fallback_count} fallback summaries"
        )
    if _test_input_signatures(base) != input_signatures:
        raise RuntimeError(
            "functions.jsonl or callgraph.jsonl changed during test intent construction"
        )
    _write_jsonl_atomic(case_records, base / "test_cases.jsonl")
    _write_jsonl_atomic(records, test_intents_path)
    _write_jsonl_atomic(meta_records, test_meta_path)
    tmp_index_path = test_index_path.with_suffix(test_index_path.suffix + ".tmp")
    faiss.write_index(index, str(tmp_index_path))
    tmp_index_path.replace(test_index_path)
    write_embedding_manifest_entry(
        base,
        "tests",
        model=os.getenv("VOYAGE_EMBED_MODEL"),
        dimension=EMBED_DIM,
        metadata={
            "builder": TEST_INTENT_BUILDER,
            "build_id": build_id,
            "case_count": len(records),
            "fallback_allowed": fallback_allowed,
            "fallback_count": fallback_count,
            "embedding_cache_hits": embedding_cache_hits,
            "embedding_requests": embedding_requests,
            "embedding_duplicate_reuse": embedding_duplicate_reuse,
            "input_signatures": input_signatures,
        },
    )
    print(
        "Test intents: "
        f"local reuse {local_reuse_count}, global cache reuse {cache_reuse_count}, "
        f"generated {generated_count}, embedded {embedding_requests}, "
        f"embedding duplicate reuse {embedding_duplicate_reuse}, "
        f"embedding cache reuse {embedding_cache_hits}, fallback {fallback_count}."
    )
