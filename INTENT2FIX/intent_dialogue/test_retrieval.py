from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import faiss
import numpy as np

from code_intent.code_intent.embedding_manifest import TEST_INTENT_BUILDER
from code_intent.code_intent.test_case_extractor import (
    TEST_CASE_EXTRACTOR_VERSION,
    code_search_view,
    discover_test_macro_aliases,
    extract_test_cases,
    is_repository_test_source,
    is_vendored_or_framework_test,
    match_source_function_to_test_case,
)

from INTENT2FIX.env_paths import index_store_parent_optional
from INTENT2FIX.intent_dialogue.schema import (
    TestAnchorCandidate,
    TestAnchorRetrieval,
)
from INTENT2FIX.reproduce.intent_schema import IssueEvidenceFrame
from INTENT2FIX.repository.repository import Repository


_SEARCH_STOPWORDS = {
    "about",
    "after",
    "before",
    "behavior",
    "change",
    "could",
    "error",
    "expected",
    "from",
    "have",
    "into",
    "issue",
    "must",
    "should",
    "that",
    "their",
    "then",
    "this",
    "using",
    "when",
    "with",
    "would",
}
_GENERIC_RETRIEVAL_TERMS = {
    "documentation",
    "method",
    "methods",
    "test",
    "tests",
    "user",
    "users",
}
_RRF_K = 60
_CALLGRAPH_EAGER_SCAN_BYTES = 64 * 1024 * 1024
_RECALL_CANDIDATES = 5
_CHANNEL_WEIGHTS = {
    "lexical": 1.0,
    "test_intent": 1.0,
    "focal_entity": 0.1,
}
logger = logging.getLogger(__name__)


class IntentEmbeddingUnavailableError(RuntimeError):
    """The configured test-intent embedding path cannot be used safely."""


@dataclass
class _RankedHit:
    key: str
    path: str
    line: int
    end_line: int
    name: str
    parent_name: str
    ancestor_keys: tuple[str, ...]
    framework: str
    label: str
    claim_ids: set[str] = field(default_factory=set)


@dataclass
class _CandidateCluster:
    key: str
    path: str
    line: int
    end_line: int
    name: str
    parent_name: str
    ancestor_keys: tuple[str, ...]
    framework: str
    label: str
    claim_ids: set[str] = field(default_factory=set)
    ranks: dict[str, int] = field(default_factory=dict)


def _is_test_or_example_path(path: str) -> bool:
    return is_repository_test_source(path)


def _repository_test_paths(repository: Repository) -> list[str]:
    files = getattr(repository, "files", None)
    if isinstance(files, dict):
        paths = files
    elif hasattr(repository, "find_files"):
        paths = repository.find_files([
            "tests/**/*",
            "test/**/*",
            "testing/**/*",
            "examples/**/*",
            "example/**/*",
            "**/tests/**/*",
            "**/test/**/*",
            "**/testing/**/*",
            "**/examples/**/*",
            "**/example/**/*",
            "**/*test*.c",
            "**/*test*.cc",
            "**/*test*.cpp",
            "**/*test*.cxx",
            "**/*test*.h",
            "**/*test*.hpp",
        ])
    else:
        paths = []
    return sorted({path for path in paths if _is_test_or_example_path(path)})


def _repository_project_header_paths(repository: Repository) -> list[str]:
    files = getattr(repository, "files", None)
    if isinstance(files, dict):
        paths = files
    elif hasattr(repository, "find_files"):
        paths = repository.find_files([
            "**/*.h",
            "**/*.hh",
            "**/*.hpp",
            "**/*.hxx",
        ])
    else:
        paths = []
    return sorted({
        path
        for path in paths
        if Path(path).suffix.lower() in {".h", ".hh", ".hpp", ".hxx"}
        and not is_vendored_or_framework_test(path)
    })


def _code_terms(text: str) -> list[str]:
    explicit = re.findall(r"`([^`\n]{2,120})`", text)
    identifiers = re.findall(
        r"\b[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*\b",
        text,
    )
    terms = [*explicit]
    terms.extend(token for token in identifiers if _looks_code_like(token))
    return list(dict.fromkeys(term.strip() for term in terms if term.strip()))


def _looks_code_like(term: str) -> bool:
    value = term.strip()
    return bool(
        len(value) >= 3
        and value.lower() not in _SEARCH_STOPWORDS
        and (
            "::" in value
            or "_" in value
            or any(character.isdigit() for character in value)
            or value.isupper()
            or value.startswith("operator")
            or bool(re.search(r"[/.[\]#*()]", value))
        )
    )


def _add_weighted_term(
    weights: dict[str, tuple[str, int]],
    term: str,
    weight: int,
) -> None:
    value = " ".join(term.strip().split())
    if len(value) < 3 or len(value) > 160:
        return
    if value.lower() in _SEARCH_STOPWORDS:
        return
    if value.lower() in _GENERIC_RETRIEVAL_TERMS:
        return
    key = value.lower()
    previous = weights.get(key)
    if previous is None or weight > previous[1]:
        weights[key] = (value, weight)


def _weighted_search_terms(
    issue: str,
    issue_evidence: IssueEvidenceFrame,
) -> list[tuple[str, int]]:
    weights: dict[str, tuple[str, int]] = {}
    for term in issue_evidence.symbols:
        symbol_weight = 4 if term.endswith("/") or "*" in term else 12
        _add_weighted_term(weights, term, symbol_weight)
        for component in re.split(r"::|_+", term):
            if len(component) >= 4:
                _add_weighted_term(weights, component, 6)
    for term in issue_evidence.error_signatures:
        _add_weighted_term(weights, term, 12)
    for claim in issue_evidence.claims:
        for term in claim.search_terms:
            if _looks_code_like(term):
                weight = 10
            elif len(term.split()) > 1:
                weight = 5
            else:
                weight = 2
            _add_weighted_term(weights, term, weight)
            if " " not in term and term.lower().endswith("s"):
                _add_weighted_term(weights, term[:-1], weight)
        for term in _code_terms(claim.text):
            _add_weighted_term(weights, term, 8)
            for component in re.split(r"::|_+", term):
                if len(component) >= 4:
                    _add_weighted_term(weights, component, 4)
    for term in _code_terms(issue):
        _add_weighted_term(weights, term, 8)
        for component in re.split(r"::|_+", term):
            if len(component) >= 4:
                _add_weighted_term(weights, component, 4)
    ranked = sorted(
        weights.values(),
        key=lambda item: -item[1],
    )
    return ranked[:24]


def _claim_terms(issue_evidence: IssueEvidenceFrame) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for claim in issue_evidence.claims:
        terms = [*claim.search_terms, *_code_terms(claim.text)]
        result[claim.id] = list(dict.fromkeys(term for term in terms if term))[:12]
    return result


def _all_search_terms(
    issue: str,
    issue_evidence: IssueEvidenceFrame,
) -> list[str]:
    return [
        term
        for term, _weight in _weighted_search_terms(issue, issue_evidence)
    ]


def _validation_intent_query(
    issue: str,
    issue_evidence: IssueEvidenceFrame,
) -> str:
    sections = {
        "Requested outcome": [],
        "Trigger or precondition": [],
        "Observed mismatch": [],
        "Behavior to preserve": [],
    }
    section_by_kind = {
        "expected_behavior": "Requested outcome",
        "trigger": "Trigger or precondition",
        "example": "Trigger or precondition",
        "observed_behavior": "Observed mismatch",
        "preservation": "Behavior to preserve",
        "non_goal": "Behavior to preserve",
    }
    for claim in issue_evidence.claims:
        section = section_by_kind.get(claim.kind)
        if section:
            sections[section].append(claim.text)
    rendered = [
        f"Original issue: {issue[:800].strip()}",
    ]
    rendered.extend(
        f"{label}: {' '.join(values)}"
        for label, values in sections.items()
        if values
    )
    if issue_evidence.symbols:
        rendered.append("Public symbols: " + ", ".join(issue_evidence.symbols))
    return "\n".join(item for item in rendered if item).strip()[:1600]


def _matching_claim_ids(
    content: str,
    terms_by_claim: dict[str, list[str]],
) -> set[str]:
    lowered = content.lower()
    return {
        claim_id
        for claim_id, terms in terms_by_claim.items()
        if any(term.lower() in lowered for term in terms)
    }


def _repository_test_cases(
    repository: Repository,
    paths: list[str],
) -> list[dict]:
    sources = []
    for path in paths:
        content = repository.get_file_content(path) or ""
        if not content or len(content) > 1_500_000:
            continue
        sources.append((path, content))
    header_contents = []
    for path in _repository_project_header_paths(repository):
        content = repository.get_file_content(path) or ""
        if content and len(content) <= 1_500_000:
            header_contents.append(content)
    shared_aliases = discover_test_macro_aliases(
        header_contents
    )
    return [
        case
        for path, content in sources
        for case in extract_test_cases(path, content, shared_aliases)
    ]


def _indexed_test_cases(
    index_dir: Path,
    path_map: _RepositoryPathMap,
) -> list[dict]:
    cases = []
    for record in _iter_jsonl(index_dir / "test_cases.jsonl"):
        path = path_map.resolve(str(record.get("def_file") or ""))
        if path is None:
            continue
        case = dict(record)
        case["def_file"] = path
        case["_focals_precomputed"] = (
            case.get("extractor_version") == TEST_CASE_EXTRACTOR_VERSION
        )
        cases.append(case)
    return cases


def _lexical_hits(
    test_cases: list[dict],
    weighted_terms: list[tuple[str, int]],
    terms_by_claim: dict[str, list[str]],
) -> list[_RankedHit]:
    ranked: list[tuple[int, int, _RankedHit]] = []
    for case in test_cases:
        content = "\n".join(filter(None, [
            str(case.get("fq_name") or ""),
            str(case.get("parent_name") or ""),
            str(case.get("def_file") or ""),
            str(case.get("code") or ""),
            str(case.get("oracle_code") or ""),
        ]))
        lowered = content.lower()
        matched_terms = [
            (term, weight)
            for term, weight in weighted_terms
            if term.lower() in lowered
        ]
        if not matched_terms:
            continue
        matched_weight = sum(weight for _term, weight in matched_terms)
        frequency = sum(
            weight * min(lowered.count(term.lower()), 3)
            for term, weight in matched_terms
        )
        identity_score = sum(
            weight
            for term, weight in matched_terms
            if term.lower() in str(case.get("fq_name") or "").lower()
        )
        ranked.append((
            matched_weight,
            identity_score + frequency,
            _RankedHit(
                key=str(case.get("id") or ""),
                path=str(case.get("def_file") or ""),
                line=int(case.get("def_start_line") or 1),
                end_line=int(case.get("def_end_line") or case.get("def_start_line") or 1),
                name=str(case.get("fq_name") or ""),
                parent_name=str(case.get("parent_name") or ""),
                ancestor_keys=tuple(case.get("ancestor_test_ids") or []),
                framework=str(case.get("framework") or ""),
                label=(
                    f"test case {case.get('fq_name')}: lexical match: "
                    + ", ".join(term for term, _weight in matched_terms[:4])
                ),
                claim_ids=_matching_claim_ids(content, terms_by_claim),
            ),
        ))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2].path))
    return [item[2] for item in ranked[:30]]


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSONL record in %s", path)


def _read_jsonl(path: Path) -> list[dict]:
    return list(_iter_jsonl(path))


def _resolve_index_output(index_name: str | None) -> Path | None:
    root = index_store_parent_optional()
    if root is None or not index_name:
        return None
    names = [index_name]
    if not index_name.startswith("swe-bench_"):
        names.append(f"swe-bench_{index_name}")
    for name in names:
        candidate = root / name / "index_output"
        if candidate.exists():
            return candidate
    return None


class _RepositoryPathMap:
    def __init__(self, paths: list[str]):
        self.paths = paths
        self.path_set = set(paths)
        self.cache: dict[str, str | None] = {}

    def resolve(self, indexed_path: str) -> str | None:
        cached = self.cache.get(indexed_path)
        if indexed_path in self.cache:
            return cached
        normalized = indexed_path.replace("\\", "/").lstrip("./")
        if normalized in self.path_set:
            result = normalized
        else:
            matches = [
                path
                for path in self.paths
                if normalized.endswith(f"/{path}")
                or path.endswith(f"/{normalized}")
            ]
            result = min(matches, key=len) if matches else None
        self.cache[indexed_path] = result
        return result


def _semantic_query_embedding(query: str) -> np.ndarray | None:
    if not os.getenv("VOYAGE_API_KEY") or not os.getenv("VOYAGE_EMBED_MODEL"):
        if _strict_embedding_enabled():
            raise IntentEmbeddingUnavailableError(
                "Voyage query embedding is not configured"
            )
        return None
    from code_intent.code_intent.llm_utils import embed_query

    return embed_query(query)


def _strict_embedding_enabled() -> bool:
    return os.getenv("CODE_INTENT_EMBED_ABORT_ON_FALLBACK") == "1"


def _semantic_hits(
    index_dir: Path,
    path_map: _RepositoryPathMap,
    test_cases: list[dict],
    query: str,
    terms_by_claim: dict[str, list[str]],
    query_embedder: Callable[[str], np.ndarray | None],
) -> list[_RankedHit]:
    index_path = index_dir / "test_index.faiss"
    if not index_path.exists():
        return []
    try:
        from code_intent.code_intent.intent_extractor import (
            intent_test_artifacts_ready,
        )

        # Retrieval consumes a frozen offline-built index. The Scout process may
        # use a different dialogue model and does not need the builder's LLM
        # configuration merely to query those artifacts.
        if not intent_test_artifacts_ready(
            index_dir,
            require_current_generation=False,
        ):
            logger.warning("Ignoring incomplete or inconsistent test-intent index")
            return []
        meta_records = _read_jsonl(index_dir / "test_meta.jsonl")
        intent_records = _read_jsonl(index_dir / "test_intents.jsonl")
        if not meta_records or not intent_records:
            return []
        index = faiss.read_index(str(index_path))
        from code_intent.code_intent.embedding_manifest import (
            load_embedding_manifest,
        )

        manifest_entry = (
            load_embedding_manifest(index_dir).get("indexes", {}).get("tests")
            or {}
        )
        runtime_model = os.getenv("VOYAGE_EMBED_MODEL")
        index_model = manifest_entry.get("model")
        manifest_dimension = manifest_entry.get("dimension")
        if manifest_entry.get("builder") != TEST_INTENT_BUILDER:
            if _strict_embedding_enabled():
                raise IntentEmbeddingUnavailableError(
                    "Test-intent index uses an incompatible builder schema"
                )
            logger.warning("Ignoring legacy test-intent index without current schema")
            return []
        if index_model and runtime_model and index_model != runtime_model:
            if _strict_embedding_enabled():
                raise IntentEmbeddingUnavailableError(
                    f"Test-intent model mismatch: index={index_model} "
                    f"runtime={runtime_model}"
                )
            logger.warning(
                "Test-intent model mismatch: index=%s runtime=%s",
                index_model,
                runtime_model,
            )
            return []
        if manifest_dimension and manifest_dimension != index.d:
            if _strict_embedding_enabled():
                raise IntentEmbeddingUnavailableError(
                    "Test-intent manifest dimension mismatch: "
                    f"manifest={manifest_dimension} index={index.d}"
                )
            logger.warning(
                "Test-intent manifest mismatch: manifest=%s index=%d",
                manifest_dimension,
                index.d,
            )
            return []
        query_vector = query_embedder(query)
    except IntentEmbeddingUnavailableError:
        raise
    except Exception as exc:
        if _strict_embedding_enabled():
            raise IntentEmbeddingUnavailableError(
                "Test-intent semantic retrieval failed in strict embedding mode"
            ) from exc
        logger.warning("Test-intent semantic retrieval unavailable: %s", exc)
        return []
    if query_vector is None or index.ntotal == 0:
        return []
    query_array = np.asarray(query_vector, dtype="float32").reshape(1, -1)
    if query_array.shape[1] != index.d:
        if _strict_embedding_enabled():
            raise IntentEmbeddingUnavailableError(
                "Test-intent embedding dimension mismatch: "
                f"query={query_array.shape[1]} index={index.d}"
            )
        logger.warning(
            "Test-intent embedding mismatch: query=%d index=%d",
            query_array.shape[1],
            index.d,
        )
        return []
    query_norm = np.linalg.norm(query_array)
    if query_norm:
        query_array /= query_norm
    scores, indices = index.search(query_array, min(30, index.ntotal))
    hits = []
    for vector_index, score in zip(indices[0], scores[0]):
        if vector_index < 0 or vector_index >= len(meta_records):
            continue
        meta = meta_records[vector_index]
        record_index = meta.get("record_index", vector_index)
        if not isinstance(record_index, int) or not 0 <= record_index < len(intent_records):
            continue
        record = intent_records[record_index]
        path = path_map.resolve(str(record.get("file") or meta.get("file") or ""))
        if path is None:
            continue
        line = int(record.get("start_line") or meta.get("start_line") or 1)
        source_cases = [
            case for case in test_cases
            if str(case.get("def_file") or "") == path
        ]
        indexed_case_id = str(
            record.get("test_case_id") or meta.get("test_case_id") or ""
        )
        source_case = next((
            case for case in source_cases
            if indexed_case_id and str(case.get("id") or "") == indexed_case_id
        ), None)
        if source_case is None:
            source_case = match_source_function_to_test_case(source_cases, {
                "def_start_line": line,
                "def_start_column": (
                    record.get("start_column") or meta.get("start_column")
                ),
                "fq_name": record.get("signature") or record.get("func_id"),
            })
        intent_text = " ".join(str(record.get(key) or "") for key in (
            "signature",
            "short_intent",
            "action",
            "object",
            "oracle",
            "fixture",
            "covered_symbols",
            "raw_summary",
            "usage_context",
        ))
        intent_text += " " + " ".join(
            " ".join(filter(None, [
                str(focal.get("name") or ""),
                str(focal.get("short_intent") or ""),
                str(focal.get("raw_summary") or ""),
            ]))
            for focal in (record.get("focal_entities") or [])
        )
        hits.append(_RankedHit(
            key=str(
                (source_case or {}).get("id")
                or record.get("test_case_id")
                or record.get("func_id")
                or f"semantic:{path}:{line}"
            ),
            path=path,
            line=int((source_case or {}).get("def_start_line") or line),
            end_line=int(
                (source_case or {}).get("def_end_line")
                or record.get("end_line")
                or meta.get("end_line")
                or line
                or 1
            ),
            name=str((source_case or {}).get("fq_name") or record.get("signature") or ""),
            parent_name=str((source_case or {}).get("parent_name") or ""),
            ancestor_keys=tuple((source_case or {}).get("ancestor_test_ids") or []),
            framework=str(
                (source_case or {}).get("framework") or record.get("framework") or ""
            ),
            label=(
                f"test-intent similarity {float(score):.3f}: "
                f"{record.get('short_intent') or record.get('signature') or path}"
            )[:500],
            claim_ids=_matching_claim_ids(intent_text, terms_by_claim),
        ))
    return hits


def _focal_entity_hits(
    index_dir: Path,
    path_map: _RepositoryPathMap,
    test_cases: list[dict],
    weighted_terms: list[tuple[str, int]],
    terms_by_claim: dict[str, list[str]],
) -> list[_RankedHit]:
    production_candidates: dict[str, tuple[int, set[str], str]] = {}
    production_names_by_short: dict[str, set[str]] = {}
    test_functions: dict[str, dict] = {}
    test_function_code: dict[str, str] = {}
    for record in _iter_jsonl(index_dir / "functions.jsonl"):
        if record.get("is_test"):
            path = path_map.resolve(str(record.get("def_file") or ""))
            function_id = str(record.get("id") or "")
            if path is not None and function_id:
                test_functions[function_id] = {
                    **record,
                    "def_file": path,
                }
                test_function_code[function_id] = str(record.get("code") or "")
            continue
        entity_text = " ".join(str(record.get(key) or "") for key in (
            "fq_name",
            "short_name",
            "def_file",
        ))
        short_name = str(record.get("short_name") or "")
        entity_name = str(record.get("fq_name") or short_name)
        if len(short_name) >= 4:
            production_names_by_short.setdefault(short_name.lower(), set()).add(
                entity_name.lower()
            )
        lowered = entity_text.lower()
        matched_terms = {
            term.lower()
            for term, _weight in weighted_terms
            if term.lower() in lowered
        }
        if len(short_name) < 4 or not matched_terms:
            continue
        score = sum(
            weight
            for term, weight in weighted_terms
            if term.lower() in matched_terms
        )
        previous = production_candidates.get(entity_name)
        if previous is None or score > previous[0]:
            production_candidates[entity_name] = (score, matched_terms, short_name)
    if not test_cases:
        return []

    case_by_key = {str(case.get("id") or ""): case for case in test_cases}
    cases_by_path: dict[str, list[dict]] = {}
    for case in test_cases:
        cases_by_path.setdefault(str(case.get("def_file") or ""), []).append(case)

    case_key_by_function: dict[str, str] = {}
    for function_id, function in test_functions.items():
        path = str(function.get("def_file") or "")
        path_cases = cases_by_path.get(path, [])
        case = next((
            item for item in path_cases
            if function_id in (item.get("source_func_ids") or [])
        ), None)
        if case is None:
            case = match_source_function_to_test_case(path_cases, function)
        if case is not None:
            case_key_by_function[function_id] = str(case.get("id") or "")

    focal_by_case: dict[str, list[str]] = {}
    matched_terms_by_case: dict[str, set[str]] = {}
    precomputed_focals_complete = all(
        case.get("_focals_precomputed") is True for case in test_cases
    )
    for case_key, case in case_by_key.items():
        for focal in case.get("focal_entities") or []:
            focal_text = " ".join(filter(None, [
                str(focal.get("name") or ""),
                str(focal.get("file") or ""),
                str(focal.get("short_intent") or ""),
                str(focal.get("raw_summary") or ""),
            ]))
            lowered_focal = focal_text.lower()
            matched_terms = {
                term.lower()
                for term, _weight in weighted_terms
                if term.lower() in lowered_focal
            }
            if not matched_terms:
                continue
            focal_by_case.setdefault(case_key, []).append(
                str(focal.get("name") or focal_text)
            )
            matched_terms_by_case.setdefault(case_key, set()).update(matched_terms)

    indexed_code_by_case: dict[str, list[str]] = {}
    for function_id, case_key in case_key_by_function.items():
        code = test_function_code.get(function_id)
        if code:
            indexed_code_by_case.setdefault(case_key, []).append(code)
    ranked_production = sorted(
        production_candidates.items(),
        key=lambda item: (-item[1][0], -len(item[0]), item[0]),
    )[:100]
    for case_key, case in case_by_key.items():
        test_code = code_search_view("\n".join([
            str(case.get("code") or ""),
            *indexed_code_by_case.get(case_key, []),
        ])).lower()
        for name, (_score, matched_terms, short_name) in ranked_production:
            lowered_name = name.lower()
            qualified_match = "::" in lowered_name and re.search(
                rf"(?<!\w){re.escape(lowered_name)}(?!\w)",
                test_code,
            )
            unique_short = len(
                production_names_by_short.get(short_name.lower(), set())
            ) == 1
            short_match = unique_short and re.search(
                rf"\b{re.escape(short_name.lower())}\b",
                test_code,
            )
            if not qualified_match and not short_match:
                continue
            focal_by_case.setdefault(case_key, []).append(name)
            matched_terms_by_case.setdefault(case_key, set()).update(matched_terms)

    callgraph_path = index_dir / "callgraph.jsonl"
    scan_callgraph = (
        not precomputed_focals_complete
        and (
            not focal_by_case
            or not callgraph_path.exists()
            or callgraph_path.stat().st_size <= _CALLGRAPH_EAGER_SCAN_BYTES
        )
    )
    edge_records = _iter_jsonl(callgraph_path) if scan_callgraph else ()
    for edge in edge_records:
        caller = str(edge.get("caller") or "")
        case_key = case_key_by_function.get(caller)
        if not case_key:
            continue
        callee_path = str(edge.get("callee_file") or "")
        if _is_test_or_example_path(callee_path):
            continue
        focal = " ".join(filter(None, [
            str(edge.get("callee_name") or ""),
            callee_path,
        ]))
        lowered = focal.lower()
        matched_terms = {
            term.lower()
            for term, _weight in weighted_terms
            if term.lower() in lowered
        }
        if not matched_terms:
            continue
        focals = focal_by_case.setdefault(case_key, [])
        if focal not in focals and len(focals) < 6:
            focals.append(focal)
        matched_terms_by_case.setdefault(case_key, set()).update(matched_terms)

    ranked: list[tuple[int, _RankedHit]] = []
    weight_by_term = {term.lower(): weight for term, weight in weighted_terms}
    for case_key, case in case_by_key.items():
        focal_entities = focal_by_case.get(case_key, [])
        if not focal_entities:
            continue
        focal_text = " ".join(focal_entities)
        score = sum(
            weight_by_term[term]
            for term in matched_terms_by_case.get(case_key, set())
        ) + 2 * len(focal_entities)
        ranked.append((score, _RankedHit(
            key=case_key,
            path=str(case.get("def_file") or ""),
            line=int(case.get("def_start_line") or 1),
            end_line=int(case.get("def_end_line") or case.get("def_start_line") or 1),
            name=str(case.get("fq_name") or ""),
            parent_name=str(case.get("parent_name") or ""),
            ancestor_keys=tuple(case.get("ancestor_test_ids") or []),
            framework=str(case.get("framework") or ""),
            label=("focal entities: " + "; ".join(focal_entities[:3]))[:500],
            claim_ids=_matching_claim_ids(focal_text, terms_by_claim),
        )))
    ranked.sort(key=lambda item: (-item[0], item[1].path, item[1].line))
    return [hit for _score, hit in ranked[:30]]


def _find_cluster(
    clusters: list[_CandidateCluster],
    hit: _RankedHit,
) -> _CandidateCluster | None:
    return next(
        (cluster for cluster in clusters if cluster.key == hit.key),
        None,
    )


def _same_test_hierarchy(
    first: _CandidateCluster,
    second: _CandidateCluster,
) -> bool:
    return bool(
        first.key != second.key
        and (
            first.key in second.ancestor_keys
            or second.key in first.ancestor_keys
        )
    )


def _rrf_score(ranks: dict[str, int]) -> float:
    return sum(
        _CHANNEL_WEIGHTS.get(channel, 0.0) / (_RRF_K + rank)
        for channel, rank in ranks.items()
    )


def _snippet(
    repository: Repository,
    path: str,
    line: int,
    end_line: int,
) -> tuple[str, str]:
    content = repository.get_file_content(path) or ""
    lines = content.splitlines()
    if not lines:
        return f"{path}:1", ""
    start = min(max(line, 1), len(lines))
    bounded_end = min(max(end_line, start), len(lines))
    end = min(bounded_end, start + 79)
    rendered = "\n".join(
        f"{number:6d}  {lines[number - 1]}"
        for number in range(start, end + 1)
    )
    return f"{path}:{start}-{bounded_end}", rendered[:5000]


def _rank_test_clusters(
    *,
    index_dir: Path | None,
    path_map: _RepositoryPathMap,
    test_cases: list[dict],
    issue: str,
    issue_evidence: IssueEvidenceFrame,
    query_embedder: Callable[[str], np.ndarray | None],
    limit: int,
) -> tuple[str, list[str], list[_CandidateCluster]]:
    terms_by_claim = _claim_terms(issue_evidence)
    weighted_terms = _weighted_search_terms(issue, issue_evidence)
    terms = [term for term, _weight in weighted_terms]
    retrieval_query = _validation_intent_query(issue, issue_evidence)[:1600]
    rankings = {
        "lexical": _lexical_hits(
            test_cases,
            weighted_terms,
            terms_by_claim,
        ),
    }
    if index_dir is not None:
        rankings["test_intent"] = _semantic_hits(
            index_dir,
            path_map,
            test_cases,
            retrieval_query,
            terms_by_claim,
            query_embedder,
        )
        rankings["focal_entity"] = _focal_entity_hits(
            index_dir,
            path_map,
            test_cases,
            weighted_terms,
            terms_by_claim,
        )
    logger.info(
        "Test anchor retrieval channels: lexical=%d test_intent=%d "
        "focal_entity=%d index=%s",
        len(rankings.get("lexical", [])),
        len(rankings.get("test_intent", [])),
        len(rankings.get("focal_entity", [])),
        index_dir or "unavailable",
    )
    clusters: list[_CandidateCluster] = []
    for source, hits in rankings.items():
        seen: set[str] = set()
        for rank, hit in enumerate(hits, 1):
            if hit.key in seen:
                continue
            seen.add(hit.key)
            cluster = _find_cluster(clusters, hit)
            if cluster is None:
                cluster = _CandidateCluster(
                    key=hit.key,
                    path=hit.path,
                    line=hit.line,
                    end_line=hit.end_line,
                    name=hit.name,
                    parent_name=hit.parent_name,
                    ancestor_keys=hit.ancestor_keys,
                    framework=hit.framework,
                    label=hit.label,
                )
                clusters.append(cluster)
            cluster.claim_ids.update(hit.claim_ids)
            cluster.ranks[source] = min(
                rank,
                cluster.ranks.get(source, rank),
            )

    clusters.sort(key=lambda cluster: (
        -_rrf_score(cluster.ranks),
        cluster.path,
        cluster.line,
    ))
    selected: list[_CandidateCluster] = []
    for cluster in clusters:
        related = next((
            item for item in selected if _same_test_hierarchy(cluster, item)
        ), None)
        if related is not None:
            related.claim_ids.update(cluster.claim_ids)
            for channel, rank in cluster.ranks.items():
                related.ranks[channel] = min(
                    rank,
                    related.ranks.get(channel, rank),
                )
            if cluster.label and cluster.label not in related.label:
                related.label = (
                    related.label + "; related hierarchy: " + cluster.label
                )[:500]
            continue
        selected.append(cluster)
    selected.sort(key=lambda cluster: (
        -_rrf_score(cluster.ranks),
        cluster.path,
        cluster.line,
    ))
    return retrieval_query, terms, selected[:limit]


class TestAnchorRetriever:
    """Retrieve test/example anchors without deciding repair intent."""

    def __init__(
        self,
        repository: Repository,
        *,
        max_candidates: int = _RECALL_CANDIDATES,
        index_dir: str | Path | None = None,
        index_name: str | None = None,
        query_embedder: Callable[[str], np.ndarray | None] | None = None,
    ):
        self.repository = repository
        self.max_candidates = min(
            _RECALL_CANDIDATES,
            max(1, max_candidates),
        )
        self.index_dir = (
            Path(index_dir)
            if index_dir
            else _resolve_index_output(index_name)
        )
        self.query_embedder = query_embedder or _semantic_query_embedding

    def retrieve(
        self,
        issue: str,
        issue_evidence: IssueEvidenceFrame,
    ) -> TestAnchorRetrieval:
        paths = _repository_test_paths(self.repository)
        path_map = _RepositoryPathMap(paths)
        indexed_test_cases = (
            _indexed_test_cases(self.index_dir, path_map)
            if self.index_dir is not None
            else []
        )
        trusted_indexed_cases = bool(indexed_test_cases) and all(
            case.get("extractor_version") == TEST_CASE_EXTRACTOR_VERSION
            for case in indexed_test_cases
        )
        source_test_cases = (
            []
            if trusted_indexed_cases
            else _repository_test_cases(self.repository, paths)
        )
        test_cases = indexed_test_cases if trusted_indexed_cases else source_test_cases
        intent_units = list(issue_evidence.intent_units)
        intent_claim_ids = {
            unit.id: list(unit.source_claim_ids) for unit in intent_units
        }
        retrieval_inputs = []
        if intent_units:
            claim_by_id = {claim.id: claim for claim in issue_evidence.claims}
            for unit in intent_units:
                related_claims = [
                    claim_by_id[claim_id]
                    for claim_id in unit.source_claim_ids
                    if claim_id in claim_by_id
                ]
                retrieval_inputs.append((
                    unit.id,
                    unit.role,
                    unit.text,
                    issue_evidence.model_copy(update={
                        "claims": related_claims,
                        "intent_units": [unit],
                    }),
                ))
        else:
            retrieval_inputs.append(("", "target", issue, issue_evidence))

        aggregated: dict[str, dict] = {}
        group_keys: list[tuple[str, str, str, list[str]]] = []
        all_terms: list[str] = []
        all_queries: list[str] = []
        for intent_id, intent_role, intent_query, evidence_subset in retrieval_inputs:
            logger.info(
                "Retrieving test anchors for %s",
                intent_id or "legacy combined Issue query",
            )
            retrieval_query, terms, selected_clusters = _rank_test_clusters(
                index_dir=self.index_dir,
                path_map=path_map,
                test_cases=test_cases,
                issue=intent_query,
                issue_evidence=evidence_subset,
                query_embedder=self.query_embedder,
                limit=self.max_candidates,
            )
            all_terms.extend(terms)
            all_queries.append(
                f"{intent_id}: {retrieval_query}" if intent_id else retrieval_query
            )
            keys = []
            for cluster in selected_clusters:
                keys.append(cluster.key)
                record = aggregated.get(cluster.key)
                if record is None:
                    record = {
                        "cluster": cluster,
                        "intent_ids": set(),
                    }
                    aggregated[cluster.key] = record
                else:
                    existing = record["cluster"]
                    existing.claim_ids.update(cluster.claim_ids)
                    for channel, rank in cluster.ranks.items():
                        existing.ranks[channel] = min(
                            rank,
                            existing.ranks.get(channel, rank),
                        )
                if intent_id:
                    record["intent_ids"].add(intent_id)
            group_keys.append((intent_id, intent_role, retrieval_query, keys))

        candidates = []
        candidate_id_by_key = {}
        all_claim_ids = [claim.id for claim in issue_evidence.claims]
        for index, (key, record) in enumerate(aggregated.items(), 1):
            cluster = record["cluster"]
            locator, snippet = _snippet(
                self.repository,
                cluster.path,
                cluster.line,
                cluster.end_line,
            )
            candidate_id = f"TEST-CAND-{index}"
            candidate_id_by_key[key] = candidate_id
            candidates.append(TestAnchorCandidate(
                id=candidate_id,
                test_case_id=cluster.key,
                test_name=cluster.name,
                framework=cluster.framework,
                path=cluster.path,
                locator=locator,
                label=cluster.label,
                intent_ids=sorted(record["intent_ids"]),
                claim_ids=(
                    sorted(cluster.claim_ids)[:10]
                    or list(dict.fromkeys(
                        claim_id
                        for intent_id in sorted(record["intent_ids"])
                        for claim_id in intent_claim_ids.get(intent_id, [])
                    ))[:10]
                    or all_claim_ids[:10]
                ),
                retrieval_ranks=cluster.ranks,
                rrf_score=_rrf_score(cluster.ranks),
                snippet=snippet,
            ))
        intent_groups = [
            {
                "intent_id": intent_id,
                "role": intent_role,
                "query": retrieval_query,
                "candidate_ids": [
                    candidate_id_by_key[key]
                    for key in keys
                    if key in candidate_id_by_key
                ],
                "limitation": (
                    "No test or example candidate was recovered for this atomic intent."
                    if not keys
                    else ""
                ),
            }
            for intent_id, intent_role, retrieval_query, keys in group_keys
            if intent_id
        ]
        return TestAnchorRetrieval(
            query="\n\n".join(all_queries)[:1600],
            search_terms=list(dict.fromkeys(all_terms))[:24],
            candidates=candidates,
            intent_groups=intent_groups,
            status="candidates_found" if candidates else "no_candidates",
            limitation=(
                "No test or example candidate was recovered from Issue-derived "
                "local test retrieval."
                if not candidates
                else ""
            ),
        )
