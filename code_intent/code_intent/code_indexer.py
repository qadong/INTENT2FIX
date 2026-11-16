from __future__ import annotations

import os
import json
import logging
import re
import sys
import hashlib
import multiprocessing
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from .config import init_clang, load_compile_commands, get_clang_resource_dir
_MAIN_INDEX = None

from clang.cindex import (
    conf,
    Index,
    Cursor,
    CursorKind,
    TemplateArgumentKind,
    TranslationUnit,
    TypeKind,
)

from .debug_utils import log_edge_debug, log_template_debug, log_parse_debug
from .id_utils import make_entity_id
from .index_cache import (
    CODE_INDEX_CACHE_SCHEMA_VERSION,
    load_file_hash_cache,
    write_file_hash_cache,
)

CLASS_KINDS = (
    CursorKind.STRUCT_DECL,
    CursorKind.CLASS_DECL,
    CursorKind.CLASS_TEMPLATE,
    CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION,
)


def get_usr(cursor: Cursor) -> str:
    try:
        return cursor.get_usr() or ""
    except Exception:
        return ""


def _specialized_cursor_template(cursor: Cursor) -> Optional[Cursor]:
    candidate = None
    getter = getattr(cursor, "get_specialized_cursor_template", None)
    if callable(getter):
        try:
            candidate = getter()
        except Exception:
            candidate = None
    if not candidate:
        try:
            candidate = conf.lib.clang_getSpecializedCursorTemplate(cursor)
        except Exception:
            candidate = None
    try:
        if not candidate or candidate.kind.is_invalid():
            return None
    except Exception:
        return None
    return candidate


def get_namespace_chain(cursor: Cursor) -> List[str]:
    names: List[str] = []
    cur = cursor.semantic_parent
    while cur and cur.kind != CursorKind.TRANSLATION_UNIT:
        if cur.kind == CursorKind.NAMESPACE and cur.spelling:
            names.append(cur.spelling)
        cur = cur.semantic_parent
    return list(reversed(names))


def _is_third_party(path: str) -> bool:
    lowered = path.lower().replace("\\", "/")
    tp_dir_names = {
        "third_party", "thirdparty", "vendor", "extern", "external",
        "dependencies", "deps", "subprojects"
    }
    path_parts = lowered.split("/")
    if any(part in tp_dir_names for part in path_parts):
        return True
    
    return (
        "gtest" in lowered
        or "gmock" in lowered
        or "googletest" in lowered
        or "/usr/include" in lowered
        or "/usr/local/include" in lowered
    )

def get_class_chain(cursor: Cursor) -> List[str]:
    names: List[str] = []
    cur = cursor.semantic_parent
    while cur and cur.kind != CursorKind.TRANSLATION_UNIT:
        if cur.kind in CLASS_KINDS and cur.spelling:
            names.append(cur.spelling)
        cur = cur.semantic_parent
    return list(reversed(names))


def get_fully_qualified_name(cursor: Cursor) -> str:
    ns = get_namespace_chain(cursor)
    cls = get_class_chain(cursor)
    parts = ns + cls
    if cursor.spelling:
        parts.append(cursor.spelling)
    return "::".join(parts)


def get_location(cursor: Cursor) -> Optional[Dict[str, Any]]:
    try:
        loc = cursor.location
        if loc.file is None:
            return None
        return {
            "file": os.path.abspath(str(loc.file)),
            "line": loc.line,
            "column": loc.column,
        }
    except Exception:
        return None


# =============================================================================
# =============================================================================

@dataclass
class FunctionInfo:
    id: str
    fq_name: str
    short_name: str
    namespaces: List[str]
    classes: List[str]
    def_file: str
    def_start_line: int
    def_end_line: int
    def_start_column: int
    def_end_column: int
    is_method: bool
    is_static: bool
    is_template: bool
    return_type: str
    parameter_types: List[str]
    raw_comment: str

    macros_used: List[str]
    macro_guards: List[str]
    error_macros: List[str]
    config_macros: List[str]
    log_strings: List[str]
    called_usrs: List[str]
    caller_count: int = 0
    is_test: bool = False
    code: str = ""
    content_hash: str = ""
    owner_class_id: str = ""


@dataclass
class ClassInfo:
    id: str
    fq_name: str
    short_name: str
    namespaces: List[str]
    def_file: str
    def_line: int
    def_end_line: int
    raw_comment: str
    bases: List[str]
    is_test: bool
    kind: str  # "class", "struct", "template"
    code: str = ""
    member_summary: List[str] = None
    content_hash: str = ""


@dataclass
class MacroDefInfo:
    id: str
    name: str
    file: str
    line: int
    value_snippet: str
    raw_comment: str


@dataclass
class MacroUseInfo:
    name: str
    file: str
    line: int


_TEST_FILE_PATTERNS = ("_test.", "_tests.", "-test.", "-tests.", ".test.")


def _is_test_path(path: str | None) -> bool:
    if not path:
        return False
    normalized = str(path).replace("\\", "/").lower()
    if "/tests/abi/include/" in normalized or "/test/abi/include/" in normalized:
        return False
    try:
        parts = [part.lower() for part in Path(path).parts]
    except Exception:
        if "/test/" in normalized or "/tests/" in normalized:
            return True
        filename = normalized.rsplit("/", 1)[-1]
    else:
        if any(part in {"test", "tests"} for part in parts):
            return True
        filename = Path(path).name.lower()
    stem = filename
    for marker in _TEST_FILE_PATTERNS:
        if marker in filename:
            return True
    for suffix in (".cc", ".cpp", ".cxx", ".c", ".mm", ".m", ".hpp", ".hh", ".h", ".inc", ".ipp"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem.endswith(("_test", "_tests", "-test", "-tests"))


def _class_name_indicates_test(name: str) -> bool:
    lowered = name.lower()
    return (
        "_test" in lowered
        or lowered.endswith("tests")
        or lowered.endswith("fixture")
        or "_fixture" in lowered
        or name.endswith("Test")
        or name.endswith("Tests")
    )


def _looks_like_test_symbol(
    short_name: str, namespaces: List[str], classes: List[str]
) -> bool:
    lowered_short = (short_name or "").lower()
    if any(_class_name_indicates_test(cls) for cls in classes):
        return True
    if lowered_short in {"testbody", "setuptestcase", "teardowntestcase"} and classes:
        return True
    ns_lower = [ns.lower() for ns in namespaces]
    if any(ns in {"testing", "testing::internal", "gtest", "googletest"} for ns in ns_lower):
        if lowered_short.startswith("test") or lowered_short.endswith("test"):
            return True
    return False


def _get_source_lines(
    file_path: str,
    cache: Dict[str, List[str]],
) -> List[str]:
    if file_path in cache:
        return cache[file_path]
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        lines = []
    cache[file_path] = lines
    return lines


def _extract_code_snippet(
    file_path: str | None,
    start_line: int,
    end_line: int,
    cache: Dict[str, List[str]],
) -> str:
    if not file_path:
        return ""
    lines = _get_source_lines(file_path, cache)
    if not lines:
        return ""
    start = max(1, min(start_line, len(lines)))
    end = max(start, min(end_line, len(lines)))
    snippet = "".join(lines[start - 1 : end])
    return snippet

# =============================================================================
# =============================================================================

def is_function_like(kind: CursorKind) -> bool:
    return kind in (
        CursorKind.FUNCTION_DECL,
        CursorKind.CXX_METHOD,
        CursorKind.CONSTRUCTOR,
        CursorKind.DESTRUCTOR,
        CursorKind.FUNCTION_TEMPLATE,
        CursorKind.CONVERSION_FUNCTION,
    ) or "FUNCTION_TEMPLATE" in str(kind)


def build_function_info(node: Cursor) -> Optional[FunctionInfo]:
    loc = get_location(node)
    if loc is None:
        return None

    usr = get_usr(node)
    fq_name = get_fully_qualified_name(node)
    namespaces = get_namespace_chain(node)
    classes = get_class_chain(node)
    parent = node.semantic_parent
    owner_class_id = (
        get_usr(parent)
        if parent and parent.kind in CLASS_KINDS and parent.spelling
        else ""
    )

    result_type = ""
    try:
        if node.result_type.kind != TypeKind.INVALID:
            result_type = node.result_type.spelling
    except Exception:
        pass

    param_types: List[str] = []
    try:
        for arg in node.get_arguments():
            try:
                param_types.append(arg.type.spelling)
            except Exception:
                param_types.append("")
    except Exception:
        pass
    # Some libclang versions return no arguments for FUNCTION_TEMPLATE cursors
    # even though their direct PARM_DECL children are complete.
    if not param_types:
        try:
            for child in node.get_children():
                if child.kind != CursorKind.PARM_DECL:
                    continue
                try:
                    param_types.append(child.type.spelling)
                except Exception:
                    param_types.append("")
        except Exception:
            pass

    extent = node.extent
    start = extent.start
    end = extent.end
    if not start.file:
        return None

    def_file = os.path.abspath(str(start.file))
    def_start_line = start.line
    def_end_line = end.line
    def_start_column = start.column
    def_end_column = end.column
    raw_comment = node.raw_comment or ""

    is_method = node.kind in (
        CursorKind.CXX_METHOD,
        CursorKind.CONSTRUCTOR,
        CursorKind.DESTRUCTOR,
        CursorKind.CONVERSION_FUNCTION,
    ) or (node.kind == CursorKind.FUNCTION_TEMPLATE and bool(owner_class_id))
    is_static = False
    try:
        if hasattr(node, "is_static_method"):
            is_static = bool(node.is_static_method())
    except Exception:
        pass
    is_template = node.kind == CursorKind.FUNCTION_TEMPLATE

    return FunctionInfo(
        id=usr or fq_name,
        fq_name=fq_name,
        short_name=node.spelling or "",
        namespaces=namespaces,
        classes=classes,
        owner_class_id=owner_class_id,
        def_file=def_file,
        def_start_line=def_start_line,
        def_end_line=def_end_line,
        def_start_column=def_start_column,
        def_end_column=def_end_column,
        is_method=is_method,
        is_static=is_static,
        is_template=is_template,
        return_type=result_type,
        parameter_types=param_types,
        raw_comment=raw_comment,
        macros_used=[],
        macro_guards=[],
        error_macros=[],
        config_macros=[],
        log_strings=[],
        called_usrs=[],
        is_test=_is_test_path(def_file)
        or _looks_like_test_symbol(node.spelling or "", namespaces, classes),
    )


def build_class_info(node: Cursor) -> Optional[ClassInfo]:
    loc = get_location(node)
    if loc is None:
        return None

    usr = get_usr(node)
    fq_name = get_fully_qualified_name(node)
    namespaces = get_namespace_chain(node)
    raw_comment = node.raw_comment or ""
    
    bases: List[str] = []
    member_summary: List[str] = []
    
    kind_str = "class"
    if node.kind == CursorKind.STRUCT_DECL:
        kind_str = "struct"
    elif node.kind in (CursorKind.CLASS_TEMPLATE, CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION):
        kind_str = "template"

    for child in node.get_children():
        if child.kind == CursorKind.CXX_BASE_SPECIFIER:
            t = child.type
            if t and t.spelling:
                bases.append(t.spelling)
        elif child.kind in (CursorKind.FIELD_DECL, CursorKind.CXX_METHOD, CursorKind.FUNCTION_TEMPLATE):
            access = ""
            try:
                if child.access_specifier.name != "INVALID":
                    access = child.access_specifier.name.lower() + ": "
            except Exception: pass
            
            m_name = child.spelling or child.displayname
            m_extent = child.extent
            m_start = m_extent.start.line
            m_end = m_extent.end.line
            
            if child.kind == CursorKind.FIELD_DECL:
                member_summary.append(f"{access}field {m_name} ({m_start}-{m_end})")
            else:
                member_summary.append(f"{access}method {child.displayname} ({m_start}-{m_end})")

    extent = node.extent
    return ClassInfo(
        id=usr or fq_name,
        fq_name=fq_name,
        short_name=node.spelling or "",
        namespaces=namespaces,
        def_file=str(loc["file"]),
        def_line=extent.start.line,
        def_end_line=extent.end.line,
        raw_comment=raw_comment,
        bases=bases,
        is_test=_is_test_path(str(loc["file"])),
        kind=kind_str,
        member_summary=member_summary,
    )


def build_macro_def(node: Cursor) -> Optional[MacroDefInfo]:
    loc = get_location(node)
    if loc is None:
        return None
    try:
        tu = node.translation_unit
        extent = node.extent
        tokens = list(tu.get_tokens(extent=extent))
        value_tokens = []
        found_name = False
        for t in tokens:
            if found_name:
                value_tokens.append(t.spelling)
            elif t.spelling == node.spelling:
                found_name = True
        
        value_snippet = " ".join(value_tokens).strip()
        if not value_snippet:
            value_snippet = "(defined)"
    except Exception:
        value_snippet = ""
        
    return MacroDefInfo(
        id=f"macro:{node.spelling}@{loc['file']}:{loc['line']}",
        name=node.spelling,
        file=str(loc["file"]),
        line=loc["line"],
        value_snippet=value_snippet,
        raw_comment=node.raw_comment or "",
    )


def build_macro_use(node: Cursor) -> Optional[MacroUseInfo]:
    loc = get_location(node)
    if loc is None:
        return None
    return MacroUseInfo(
        name=node.spelling,
        file=str(loc["file"]),
        line=loc["line"],
    )


DEFAULT_ENABLED_EDGE_TYPES = {
    "call",
    "dependent_call",
    "template_ref",
    "decl_ref",
    "member_ref",
    "type_ref",
    "unexposed_ref",
    "instantiates",
}
CROSS_FILE_ONLY_EDGE_TYPES = {"type_ref"}


def _load_enabled_edge_types() -> Set[str]:
    env_value = os.environ.get("ENABLED_EDGE_TYPES", "")
    if env_value:
        parsed = {item.strip() for item in env_value.split(",") if item.strip()}
        if parsed:
            parsed.add("call")
            return parsed
    return set(DEFAULT_ENABLED_EDGE_TYPES)


ENABLED_EDGE_TYPES = _load_enabled_edge_types()
EDGE_ALLOW_TEST_TARGETS = os.environ.get("EDGE_ALLOW_TEST_TARGETS", "0") == "1"


def empty_result() -> Dict[str, Any]:
    return {
        "functions": [],
        "classes": [],
        "macro_defs": [],
        "macro_uses": [],
        "call_edges": [],
        "template_instantiations": [],
    }


def _require_nonempty_index(
    *,
    indexed_file_count: int,
    function_count: int,
    class_count: int,
    macro_count: int,
) -> None:
    """Do not report a macro-only shell after every AST parse failed."""
    if indexed_file_count and not (function_count or class_count):
        raise RuntimeError(
            "Structural index produced no entities with executable structure from "
            f"{indexed_file_count} input files; check parse errors, clang bindings, "
            f"and compile-command arguments (macros found: {macro_count})."
        )


def _deduplicate_edges(
    results: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Merge the same source relation observed through multiple compile units."""
    raw_counts: Counter[str] = Counter()
    merged: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for result in results:
        for edge in result.get("call_edges", []):
            edge_type = edge.get("edge_type", "call")
            raw_counts[edge_type] += 1
            key = (edge_type, edge.get("caller", ""), edge.get("callee", ""))
            existing = merged.get(key)
            if existing is None:
                merged[key] = dict(edge)
                continue
            existing["count"] = max(
                int(existing.get("count", 1)), int(edge.get("count", 1))
            )
            for field in (
                "caller_name",
                "callee_name",
                "caller_file",
                "callee_file",
                "resolution",
                "instantiation",
                "instantiated_type",
            ):
                if not existing.get(field) and edge.get(field):
                    existing[field] = edge[field]

    rows = [merged[key] for key in sorted(merged)]
    unique_counts = Counter(row.get("edge_type", "call") for row in rows)
    edge_types = sorted(set(raw_counts) | set(unique_counts))
    stats = {
        "raw_rows": sum(raw_counts.values()),
        "unique_edges": len(rows),
        "duplicates_removed": sum(raw_counts.values()) - len(rows),
        "by_type": {
            edge_type: {
                "raw_rows": raw_counts[edge_type],
                "unique_edges": unique_counts[edge_type],
                "duplicates_removed": (
                    raw_counts[edge_type] - unique_counts[edge_type]
                ),
            }
            for edge_type in edge_types
        },
    }
    return rows, stats


def _definition_path(path: str, project_root: str) -> str:
    if not path:
        return ""
    normalized = os.path.realpath(path) if os.path.isabs(path) else path
    if os.path.isabs(normalized):
        try:
            relative = os.path.relpath(normalized, project_root)
        except (OSError, ValueError):
            relative = normalized
        if not relative.startswith(".."):
            normalized = relative
    return normalized.replace("\\", "/")


def _function_definition_key(
    function: Dict[str, Any], project_root: str
) -> Tuple[str, int, int, str, str]:
    return (
        _definition_path(str(function.get("def_file", "")), project_root),
        int(function.get("def_start_line") or 0),
        int(function.get("def_end_line") or 0),
        str(function.get("content_hash") or ""),
        str(function.get("fq_name") or ""),
    )


def _qualified_function_id(
    original_id: str,
    site: Tuple[str, int, str],
) -> str:
    payload = "|".join((original_id, site[0], str(site[1]), site[2]))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{original_id}@definition:{digest}"


def _class_definition_key(
    cls: Dict[str, Any], project_root: str
) -> Tuple[str, int, int, str, str]:
    return (
        _definition_path(str(cls.get("def_file", "")), project_root),
        int(cls.get("def_line") or 0),
        int(cls.get("def_end_line") or cls.get("def_line") or 0),
        str(cls.get("content_hash") or ""),
        str(cls.get("fq_name") or ""),
    )


def _qualified_class_id(
    original_id: str,
    site: Tuple[str, int, str],
) -> str:
    payload = "|".join((original_id, site[0], str(site[1]), site[2]))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{original_id}@definition:{digest}"


def _canonical_definition_id(entity_id: str) -> str:
    return str(entity_id or "").split("@definition:", 1)[0]


def _reconcile_template_instantiation_targets(
    results: List[Dict[str, Any]], project_root: str
) -> Dict[str, Any]:
    """Bind template records to a unique indexed definition when available."""
    code_ids: Set[str] = set()
    identity_lookup: Dict[Tuple[str, str], Set[str]] = {}
    entity_spans: Dict[str, Tuple[int, int]] = {}
    for result in results:
        for entity in (*result.get("functions", []), *result.get("classes", [])):
            entity_id = str(entity.get("id") or "")
            file_name = _definition_path(
                str(entity.get("def_file") or ""), project_root
            )
            fq_name = str(entity.get("fq_name") or "")
            if not entity_id:
                continue
            code_ids.add(entity_id)
            if file_name and fq_name:
                identity_lookup.setdefault((file_name, fq_name), set()).add(
                    entity_id
                )
            start = int(
                entity.get("def_start_line") or entity.get("def_line") or 0
            )
            end = int(
                entity.get("def_end_line") or entity.get("def_line") or start
            )
            entity_spans[entity_id] = (start, end)

    stats: Counter[str] = Counter()
    for result in results:
        local_instantiations: Dict[str, List[Dict[str, Any]]] = {}
        for instantiation in result.get("template_instantiations", []):
            local_instantiations.setdefault(
                str(instantiation.get("id") or ""), []
            ).append(instantiation)
        edge_lookup: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for edge in result.get("call_edges", []):
            if edge.get("edge_type") != "instantiates":
                continue
            # Older caches used a location ID for empty-USR edges, while the
            # paired template row used its qualified name. Repair only a unique
            # local pairing with the same recorded definition coordinates.
            location_id = str(edge.get("callee") or "").rsplit(":", 2)
            if (
                len(location_id) == 3
                and location_id[1] == str(edge.get("callee_line") or "")
                and location_id[2] == str(edge.get("callee_column") or "")
                and _definition_path(location_id[0], project_root)
                == _definition_path(str(edge.get("callee_file") or ""), project_root)
            ):
                candidates = [
                    row
                    for row in local_instantiations.get(str(edge.get("caller") or ""), [])
                    if row.get("template_def_name") == edge.get("callee_name")
                    and _definition_path(str(row.get("template_def_file") or ""), project_root)
                    == _definition_path(str(edge.get("callee_file") or ""), project_root)
                    and row.get("template_def_line") == edge.get("callee_line")
                    and row.get("template_def_column") == edge.get("callee_column")
                ]
                if len(candidates) == 1:
                    edge["callee"] = str(candidates[0].get("template_def_id") or "")
                    stats["cached_edge_targets_aligned"] += 1
            edge_lookup.setdefault(
                (
                    str(edge.get("caller") or ""),
                    str(edge.get("callee") or ""),
                ),
                [],
            ).append(edge)

        for instantiation in result.get("template_instantiations", []):
            instantiation_id = str(instantiation.get("id") or "")
            observed_target = str(instantiation.get("template_def_id") or "")
            if observed_target in code_ids:
                stats["targets_already_valid"] += 1
                continue

            matching_edges = list(
                edge_lookup.get((instantiation_id, observed_target), [])
            )
            trigger_file = _definition_path(
                str(instantiation.get("trigger_file") or ""), project_root
            )
            instantiation_name = str(
                instantiation.get("fq_name")
                or instantiation.get("display_name")
                or ""
            )
            source_matches = [
                edge
                for edge in matching_edges
                if (
                    not trigger_file
                    or not edge.get("caller_file")
                    or _definition_path(
                        str(edge.get("caller_file") or ""), project_root
                    )
                    == trigger_file
                )
                and (
                    not instantiation_name
                    or not edge.get("caller_name")
                    or str(edge.get("caller_name") or "") == instantiation_name
                )
            ]
            if source_matches:
                matching_edges = source_matches

            identities: List[Tuple[str, str, int]] = []
            row_file = _definition_path(
                str(instantiation.get("template_def_file") or ""), project_root
            )
            row_name = str(instantiation.get("template_def_name") or "")
            row_line = int(instantiation.get("template_def_line") or 0)
            if row_file and row_name:
                identities.append((row_file, row_name, row_line))
            else:
                identities.extend(
                    (
                        _definition_path(
                            str(edge.get("callee_file") or ""), project_root
                        ),
                        str(edge.get("callee_name") or ""),
                        int(edge.get("callee_line") or 0),
                    )
                    for edge in matching_edges
                    if edge.get("callee_file") and edge.get("callee_name")
                )

            candidates: Set[str] = set()
            for file_name, fq_name, line in identities:
                identity_candidates = set(
                    identity_lookup.get((file_name, fq_name), set())
                )
                if line:
                    line_candidates = {
                        entity_id
                        for entity_id in identity_candidates
                        if entity_spans.get(entity_id, (0, 0))[0]
                        <= line
                        <= entity_spans.get(entity_id, (0, 0))[1]
                    }
                    if line_candidates:
                        identity_candidates = line_candidates
                candidates.update(identity_candidates)

            if len(candidates) == 1:
                canonical_target = next(iter(candidates))
                matched_identity = next(
                    (
                        identity
                        for identity in identities
                        if canonical_target
                        in identity_lookup.get((identity[0], identity[1]), set())
                    ),
                    ("", "", 0),
                )
                instantiation["canonical_template_def_usr"] = observed_target
                instantiation["template_def_id"] = canonical_target
                stats["targets_reconciled"] += 1
                for edge in matching_edges:
                    edge["callee"] = canonical_target
                    resolution_value = edge.get("resolution")
                    resolution = (
                        dict(resolution_value)
                        if isinstance(resolution_value, dict)
                        else {}
                    )
                    resolution["template_target_reconciliation"] = {
                        "method": "unique_source_entity_identity",
                        "observed_usr": observed_target,
                        "canonical_usr": canonical_target,
                        "file": matched_identity[0],
                        "fq_name": matched_identity[1],
                    }
                    edge["resolution"] = resolution
                    stats["edge_targets_reconciled"] += 1
            elif len(candidates) > 1:
                stats["ambiguous_targets"] += 1
            else:
                stats["unmatched_targets"] += 1

    unique_misses = 0
    for result in results:
        edge_lookup: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for edge in result.get("call_edges", []):
            if edge.get("edge_type") == "instantiates":
                edge_lookup.setdefault(
                    (
                        str(edge.get("caller") or ""),
                        str(edge.get("callee") or ""),
                    ),
                    [],
                ).append(edge)
        for instantiation in result.get("template_instantiations", []):
            target = str(instantiation.get("template_def_id") or "")
            if target in code_ids:
                continue
            identities = []
            row_file = _definition_path(
                str(instantiation.get("template_def_file") or ""), project_root
            )
            row_name = str(instantiation.get("template_def_name") or "")
            if row_file and row_name:
                identities.append((row_file, row_name))
            else:
                identities.extend(
                    (
                        _definition_path(
                            str(edge.get("callee_file") or ""), project_root
                        ),
                        str(edge.get("callee_name") or ""),
                    )
                    for edge in edge_lookup.get(
                        (str(instantiation.get("id") or ""), target), []
                    )
                    if edge.get("callee_file") and edge.get("callee_name")
                )
            candidates = {
                candidate
                for identity in identities
                for candidate in identity_lookup.get(identity, set())
            }
            if len(candidates) == 1:
                unique_misses += 1

    return {
        "targets_already_valid": stats["targets_already_valid"],
        "targets_reconciled": stats["targets_reconciled"],
        "edge_targets_reconciled": stats["edge_targets_reconciled"],
        "cached_edge_targets_aligned": stats["cached_edge_targets_aligned"],
        "ambiguous_targets": stats["ambiguous_targets"],
        "unmatched_targets": stats["unmatched_targets"],
        "unique_identity_misses_after_reconciliation": unique_misses,
    }


def _template_instantiation_key(
    instantiation: Dict[str, Any], project_root: str
) -> Tuple[str, str, int, str, str, str]:
    template_def_id = str(instantiation.get("template_def_id") or "")
    return (
        template_def_id,
        _definition_path(
            str(instantiation.get("trigger_file") or ""), project_root
        ),
        int(instantiation.get("trigger_line") or 0),
        str(
            instantiation.get("fq_name")
            or instantiation.get("display_name")
            or ""
        ),
        _definition_path(
            str(instantiation.get("template_def_file") or ""), project_root
        ),
        str(instantiation.get("template_def_name") or ""),
    )


def _qualified_template_instantiation_id(
    original_id: str, identity: Tuple[str, str, int, str, str, str]
) -> str:
    payload = "|".join((original_id, *map(str, identity)))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{original_id}@instantiation:{digest}"


def _qualify_multi_definition_template_instantiations(
    results: List[Dict[str, Any]], project_root: str
) -> Dict[str, Any]:
    """Keep one template-instantiation identity per recorded source trigger.

    Clang can reuse a specialization USR for macro-expanded architecture
    variants, even when those variants specialize different primary templates.
    The raw USR is therefore insufficient as a global table key. Qualify only
    colliding IDs, and rewrite each TU's emitted ``instantiates`` relation from
    the matching local record before global edge deduplication.
    """
    variants: Dict[
        str, Dict[Tuple[str, str, int, str, str, str], Dict[str, Any]]
    ] = {}
    for result in results:
        for instantiation in result.get("template_instantiations", []):
            instantiation_id = str(instantiation.get("id") or "")
            if not instantiation_id:
                continue
            identity = _template_instantiation_key(instantiation, project_root)
            variants.setdefault(instantiation_id, {}).setdefault(
                identity, instantiation
            )

    collisions = {
        instantiation_id: rows
        for instantiation_id, rows in variants.items()
        if len(rows) > 1
    }
    qualified_ids = {
        (instantiation_id, identity): _qualified_template_instantiation_id(
            instantiation_id, identity
        )
        for instantiation_id, rows in collisions.items()
        for identity in rows
    }

    rows_qualified = 0
    edge_rows_rewritten = 0
    edge_rows_expanded = 0
    edge_rows_not_disambiguated = 0
    for result in results:
        local_rows: Dict[
            str, List[Tuple[Tuple[str, str, int, str, str, str], Dict[str, Any]]]
        ] = {}
        for instantiation in result.get("template_instantiations", []):
            original_id = str(instantiation.get("id") or "")
            if original_id not in collisions:
                continue
            identity = _template_instantiation_key(instantiation, project_root)
            local_rows.setdefault(original_id, []).append((identity, instantiation))

        rewritten_edges: List[Dict[str, Any]] = []
        for edge in result.get("call_edges", []):
            original_id = str(edge.get("caller") or "")
            if (
                edge.get("edge_type") != "instantiates"
                or original_id not in collisions
            ):
                rewritten_edges.append(edge)
                continue

            candidates = list(local_rows.get(original_id, []))
            edge_target = str(edge.get("callee") or "")
            exact_target = [
                candidate
                for candidate in candidates
                if str(candidate[1].get("template_def_id") or "") == edge_target
            ]
            if exact_target:
                candidates = exact_target
            else:
                edge_canonical = _canonical_definition_id(edge_target)
                canonical_target = [
                    candidate
                    for candidate in candidates
                    if _canonical_definition_id(
                        str(candidate[1].get("template_def_id") or "")
                    )
                    == edge_canonical
                ]
                if canonical_target:
                    candidates = canonical_target

            filters = (
                (
                    "caller_file",
                    lambda row: _definition_path(
                        str(row.get("trigger_file") or ""), project_root
                    ),
                ),
                (
                    "caller_name",
                    lambda row: str(
                        row.get("fq_name") or row.get("display_name") or ""
                    ),
                ),
                (
                    "callee_file",
                    lambda row: _definition_path(
                        str(row.get("template_def_file") or ""), project_root
                    ),
                ),
                (
                    "callee_name",
                    lambda row: str(row.get("template_def_name") or ""),
                ),
            )
            for edge_field, row_value in filters:
                observed = str(edge.get(edge_field) or "")
                if edge_field.endswith("_file"):
                    observed = _definition_path(observed, project_root)
                if not observed:
                    continue
                matching = [
                    candidate
                    for candidate in candidates
                    if row_value(candidate[1]) == observed
                ]
                if matching:
                    candidates = matching

            if not candidates:
                edge_rows_not_disambiguated += 1
                rewritten_edges.append(edge)
                continue

            for index, (identity, instantiation) in enumerate(candidates):
                rewritten = dict(edge)
                rewritten["caller"] = qualified_ids[(original_id, identity)]
                # The table row is the authoritative pairing for this trigger.
                # This also adopts any definition-qualified target selected by
                # the preceding function/class collision passes.
                rewritten["callee"] = str(
                    instantiation.get("template_def_id") or edge_target
                )
                rewritten_edges.append(rewritten)
                edge_rows_rewritten += 1
                if index:
                    edge_rows_expanded += 1
        result["call_edges"] = rewritten_edges

        for original_id, rows in local_rows.items():
            for identity, instantiation in rows:
                instantiation["canonical_usr"] = original_id
                instantiation["definition_identity"] = {
                    "kind": "source_qualified",
                    "trigger_file": identity[1],
                    "trigger_line": identity[2],
                    "fq_name": identity[3],
                    "template_def_id": identity[0],
                    "template_def_file": identity[4],
                    "template_def_name": identity[5],
                }
                instantiation["id"] = qualified_ids[(original_id, identity)]
                rows_qualified += 1

    table_targets: Dict[str, Set[str]] = {}
    for result in results:
        for instantiation in result.get("template_instantiations", []):
            table_targets.setdefault(
                str(instantiation.get("id") or ""), set()
            ).add(str(instantiation.get("template_def_id") or ""))
    disagreements = 0
    for result in results:
        for edge in result.get("call_edges", []):
            if edge.get("edge_type") != "instantiates":
                continue
            targets = table_targets.get(str(edge.get("caller") or ""), set())
            if str(edge.get("callee") or "") not in targets:
                disagreements += 1

    return {
        "collision_usr_count": len(collisions),
        "instantiation_variant_count": sum(
            len(rows) for rows in collisions.values()
        ),
        "rows_qualified": rows_qualified,
        "edge_rows_rewritten": edge_rows_rewritten,
        "edge_rows_expanded": edge_rows_expanded,
        "edge_rows_not_disambiguated": edge_rows_not_disambiguated,
        "edge_target_disagreements_after_reconciliation": disagreements,
        "collisions": {
            instantiation_id: [
                {
                    "qualified_id": qualified_ids[(instantiation_id, identity)],
                    "template_def_id": identity[0],
                    "trigger_file": identity[1],
                    "trigger_line": identity[2],
                    "fq_name": identity[3],
                    "template_def_file": identity[4],
                    "template_def_name": identity[5],
                }
                for identity in sorted(rows)
            ]
            for instantiation_id, rows in sorted(collisions.items())
        },
    }


def _clear_unindexed_anonymous_owner_ids(results: List[Dict[str, Any]]) -> int:
    """Remove stale owner IDs for anonymous records that have no class node.

    Named semantic parents appear in a function's ``classes`` chain and are
    indexed as class entities. Anonymous local records are intentionally not
    indexed, so retaining their libclang USR would create a dangling owner.
    This merge-time normalization also repairs schema-9 parser caches written
    before owner extraction applied the same rule.
    """
    class_ids = {
        str(cls.get("id") or "")
        for result in results
        for cls in result.get("classes", [])
        if cls.get("id")
    }
    cleared = 0
    for result in results:
        for function in result.get("functions", []):
            owner_id = str(function.get("owner_class_id") or "")
            if owner_id and owner_id not in class_ids and not function.get("classes"):
                function["owner_class_id"] = ""
                cleared += 1
    return cleared


def _qualify_multi_definition_functions(
    results: List[Dict[str, Any]], project_root: str
) -> Dict[str, Any]:
    """Keep logical function definitions distinct across files and macro namespaces."""
    variants: Dict[
        str,
        Dict[
            Tuple[str, int, str],
            Dict[Tuple[str, int, int, str, str], Dict[str, Any]],
        ],
    ] = {}
    for result in results:
        for function in result.get("functions", []):
            function_id = str(function.get("id") or "")
            if not function_id:
                continue
            key = _function_definition_key(function, project_root)
            site = (key[0], key[1], key[4])
            variants.setdefault(function_id, {}).setdefault(site, {})[key] = function

    definitions = {
        function_id: {
            site: max(
                rows.values(),
                key=lambda row: (
                    int(row.get("def_end_line") or row.get("def_start_line") or 0),
                    str(row.get("content_hash") or ""),
                ),
            )
            for site, rows in sites.items()
        }
        for function_id, sites in variants.items()
    }
    collisions = {
        function_id: rows
        for function_id, rows in definitions.items()
        if len(rows) > 1
    }
    qualified_ids = {
        (function_id, site): _qualified_function_id(function_id, site)
        for function_id, rows in collisions.items()
        for site in rows
    }
    configuration_variants_merged = sum(
        len(rows) - 1
        for sites in variants.values()
        for rows in sites.values()
    )

    unresolved_edge_endpoints = 0
    edge_endpoints_disambiguated = 0
    template_def_ids_disambiguated = 0
    template_def_ids_not_disambiguated = 0
    for result in results:
        local_variants: Dict[str, Set[str]] = {}
        for function in result.get("functions", []):
            original_id = str(function.get("id") or "")
            key = _function_definition_key(function, project_root)
            site = (key[0], key[1], key[4])
            site_variants = variants.get(original_id, {}).get(site, {})
            if len(site_variants) > 1:
                function.update(definitions[original_id][site])
            if original_id not in collisions:
                continue
            qualified_id = qualified_ids[(original_id, site)]
            local_variants.setdefault(original_id, set()).add(qualified_id)
            function["canonical_usr"] = original_id
            function["definition_identity"] = {
                "kind": "source_qualified",
                "file": site[0],
                "start_line": site[1],
                "fq_name": site[2],
                "content_hash": str(function.get("content_hash") or ""),
            }
            function["id"] = qualified_id

        for edge in result.get("call_edges", []):
            for endpoint in ("caller", "callee"):
                original_id = str(edge.get(endpoint) or "")
                candidate_rows = collisions.get(original_id)
                if not candidate_rows:
                    continue
                edge_path = _definition_path(
                    str(edge.get(f"{endpoint}_file") or ""), project_root
                )
                edge_name = str(edge.get(f"{endpoint}_name") or "")
                edge_line = int(edge.get(f"{endpoint}_line") or 0)
                caller_definition_line = (
                    int(edge.get("caller_definition_line") or 0)
                    if endpoint == "caller"
                    else 0
                )
                candidates = {
                    qualified_ids[(original_id, site)]
                    for site in candidate_rows
                    if caller_definition_line
                    and edge_path == site[0]
                    and edge_name == site[2]
                    and caller_definition_line == site[1]
                }
                if len(candidates) != 1:
                    candidates = {
                        qualified_ids[(original_id, site)]
                        for site in candidate_rows
                        if edge_path
                        and site[0] == edge_path
                        and edge_name
                        and site[2] == edge_name
                    }
                if len(candidates) != 1 and edge_path and edge_line:
                    candidates = {
                        qualified_ids[(original_id, site)]
                        for site, row in candidate_rows.items()
                        if site[0] == edge_path
                        and int(row.get("def_start_line") or 0)
                        <= edge_line
                        <= int(row.get("def_end_line") or 0)
                    }
                if len(candidates) != 1 and edge_path:
                    path_candidates = {
                        qualified_ids[(original_id, site)]
                        for site in candidate_rows
                        if site[0] == edge_path
                    }
                    if len(path_candidates) == 1:
                        candidates = path_candidates
                if len(candidates) != 1:
                    named_local = {
                        candidate
                        for site, candidate in (
                            (site, qualified_ids[(original_id, site)])
                            for site in candidate_rows
                        )
                        if candidate in local_variants.get(original_id, set())
                        and edge_name
                        and site[2] == edge_name
                    }
                    candidates = (
                        named_local
                        if len(named_local) == 1
                        else set(local_variants.get(original_id, set()))
                    )
                if len(candidates) == 1:
                    edge[endpoint] = next(iter(candidates))
                    edge_endpoints_disambiguated += 1
                else:
                    unresolved_edge_endpoints += 1

        for instantiation in result.get("template_instantiations", []):
            template_id = str(instantiation.get("template_def_id") or "")
            candidate_rows = collisions.get(template_id)
            if not candidate_rows:
                continue
            template_path = _definition_path(
                str(instantiation.get("template_def_file") or ""), project_root
            )
            candidates = {
                qualified_ids[(template_id, site)]
                for site in candidate_rows
                if template_path and site[0] == template_path
            }
            if len(candidates) != 1:
                candidates = set(local_variants.get(template_id, set()))
            if len(candidates) == 1:
                instantiation["canonical_template_def_usr"] = template_id
                instantiation["template_def_id"] = next(iter(candidates))
                template_def_ids_disambiguated += 1
            else:
                template_def_ids_not_disambiguated += 1

        outgoing: Dict[str, Set[str]] = {}
        for edge in result.get("call_edges", []):
            if edge.get("edge_type") not in {"call", "dependent_call"}:
                continue
            outgoing.setdefault(str(edge.get("caller") or ""), set()).add(
                str(edge.get("callee") or "")
            )
        for function in result.get("functions", []):
            function["called_usrs"] = sorted(
                outgoing.get(str(function.get("id") or ""), set())
            )

    return {
        "collision_usr_count": len(collisions),
        "definition_variant_count": sum(len(rows) for rows in collisions.values()),
        "configuration_variants_merged": configuration_variants_merged,
        "edge_endpoints_disambiguated": edge_endpoints_disambiguated,
        "edge_endpoints_not_disambiguated": unresolved_edge_endpoints,
        "template_def_ids_disambiguated": template_def_ids_disambiguated,
        "template_def_ids_not_disambiguated": template_def_ids_not_disambiguated,
        "collisions": {
            function_id: [
                {
                    "qualified_id": qualified_ids[(function_id, site)],
                    "file": site[0],
                    "start_line": site[1],
                    "end_line": int(
                        row.get("def_end_line") or row.get("def_start_line") or 0
                    ),
                    "fq_name": site[2],
                    "content_hash": str(row.get("content_hash") or ""),
                }
                for site, row in sorted(rows.items())
            ]
            for function_id, rows in sorted(collisions.items())
        },
    }

def _qualify_multi_definition_classes(
    results: List[Dict[str, Any]], project_root: str
) -> Dict[str, Any]:
    """Keep same-USR class definitions and bind methods to the local variant."""
    variants: Dict[
        str,
        Dict[
            Tuple[str, int, str],
            Dict[Tuple[str, int, int, str, str], Dict[str, Any]],
        ],
    ] = {}
    for result in results:
        for cls in result.get("classes", []):
            class_id = str(cls.get("id") or "")
            if not class_id:
                continue
            key = _class_definition_key(cls, project_root)
            site = (key[0], key[1], key[4])
            variants.setdefault(class_id, {}).setdefault(site, {})[key] = cls

    definitions = {
        class_id: {
            site: max(
                rows.values(),
                key=lambda row: (
                    int(row.get("def_end_line") or row.get("def_line") or 0),
                    str(row.get("content_hash") or ""),
                ),
            )
            for site, rows in sites.items()
        }
        for class_id, sites in variants.items()
    }

    collisions = {
        class_id: rows
        for class_id, rows in definitions.items()
        if len(rows) > 1
    }
    qualified_ids = {
        (class_id, site): _qualified_class_id(class_id, site)
        for class_id, rows in collisions.items()
        for site in rows
    }
    configuration_variants_merged = sum(
        len(rows) - 1
        for sites in variants.values()
        for rows in sites.values()
    )

    owner_ids_disambiguated = 0
    owner_ids_not_disambiguated = 0
    edge_endpoints_disambiguated = 0
    edge_endpoints_not_disambiguated = 0
    template_def_ids_disambiguated = 0
    template_def_ids_not_disambiguated = 0
    for result in results:
        local_variants: Dict[str, Set[str]] = {}
        for cls in result.get("classes", []):
            original_id = str(cls.get("id") or "")
            key = _class_definition_key(cls, project_root)
            site = (key[0], key[1], key[4])
            site_variants = variants.get(original_id, {}).get(site, {})
            if len(site_variants) > 1:
                cls.update(definitions[original_id][site])
            if original_id not in collisions:
                continue
            qualified_id = qualified_ids[(original_id, site)]
            local_variants.setdefault(original_id, set()).add(qualified_id)
            cls["canonical_usr"] = original_id
            cls["definition_identity"] = {
                "kind": "file_qualified",
                "file": site[0],
                "start_line": site[1],
                "fq_name": site[2],
                "content_hash": str(cls.get("content_hash") or ""),
            }
            cls["id"] = qualified_id

        for edge in result.get("call_edges", []):
            for endpoint in ("caller", "callee"):
                original_id = str(edge.get(endpoint) or "")
                candidate_rows = collisions.get(original_id)
                if not candidate_rows:
                    continue
                edge_path = _definition_path(
                    str(edge.get(f"{endpoint}_file") or ""), project_root
                )
                edge_name = str(edge.get(f"{endpoint}_name") or "")
                candidates = {
                    qualified_ids[(original_id, site)]
                    for site in candidate_rows
                    if edge_path
                    and site[0] == edge_path
                    and edge_name
                    and site[2] == edge_name
                }
                if len(candidates) != 1 and edge_path:
                    path_candidates = {
                        qualified_ids[(original_id, site)]
                        for site in candidate_rows
                        if site[0] == edge_path
                    }
                    if len(path_candidates) == 1:
                        candidates = path_candidates
                if len(candidates) != 1:
                    named_local = {
                        qualified_ids[(original_id, site)]
                        for site in candidate_rows
                        if qualified_ids[(original_id, site)]
                        in local_variants.get(original_id, set())
                        and edge_name
                        and site[2] == edge_name
                    }
                    candidates = (
                        named_local
                        if len(named_local) == 1
                        else set(local_variants.get(original_id, set()))
                    )
                if len(candidates) == 1:
                    edge[endpoint] = next(iter(candidates))
                    edge_endpoints_disambiguated += 1
                else:
                    edge_endpoints_not_disambiguated += 1

        for function in result.get("functions", []):
            owner_id = str(function.get("owner_class_id") or "")
            candidate_rows = collisions.get(owner_id)
            if not candidate_rows:
                continue
            candidates = set(local_variants.get(owner_id, set()))
            function_path = _definition_path(
                str(function.get("def_file") or ""), project_root
            )
            owner_name = "::".join(
                [
                    *[str(value) for value in function.get("namespaces", [])],
                    *[str(value) for value in function.get("classes", [])],
                ]
            )
            named_candidates = {
                qualified_ids[(owner_id, site)]
                for site in candidate_rows
                if (
                    site[2] == owner_name
                    or (owner_name and site[2].endswith(f"::{owner_name}"))
                )
                and qualified_ids[(owner_id, site)] in candidates
            }
            if len(named_candidates) == 1:
                candidates = named_candidates
            if len(candidates) != 1:
                function_start = int(function.get("def_start_line") or 0)
                function_end = int(function.get("def_end_line") or function_start)
                containing_candidates = {
                    qualified_ids[(owner_id, site)]
                    for site, cls in candidate_rows.items()
                    if site[0] == function_path
                    and (
                        not owner_name
                        or site[2] == owner_name
                        or site[2].endswith(f"::{owner_name}")
                    )
                    and function_start
                    and int(cls.get("def_line") or 0)
                    <= function_start
                    <= function_end
                    <= int(cls.get("def_end_line") or cls.get("def_line") or 0)
                }
                same_file_candidates = {
                    qualified_ids[(owner_id, site)]
                    for site in candidate_rows
                    if site[0] == function_path
                    and (
                        not owner_name
                        or site[2] == owner_name
                        or site[2].endswith(f"::{owner_name}")
                    )
                }
                candidates = (
                    containing_candidates
                    if len(containing_candidates) == 1
                    else same_file_candidates
                )
            if len(candidates) == 1:
                function["owner_class_id"] = next(iter(candidates))
                owner_ids_disambiguated += 1
            else:
                owner_ids_not_disambiguated += 1

        for instantiation in result.get("template_instantiations", []):
            template_id = str(instantiation.get("template_def_id") or "")
            if template_id not in collisions:
                continue
            candidates = set(local_variants.get(template_id, set()))
            if len(candidates) == 1:
                instantiation["canonical_template_def_usr"] = template_id
                instantiation["template_def_id"] = next(iter(candidates))
                template_def_ids_disambiguated += 1
            else:
                template_def_ids_not_disambiguated += 1

    return {
        "collision_usr_count": len(collisions),
        "definition_variant_count": sum(len(rows) for rows in collisions.values()),
        "configuration_variants_merged": configuration_variants_merged,
        "owner_ids_disambiguated": owner_ids_disambiguated,
        "owner_ids_not_disambiguated": owner_ids_not_disambiguated,
        "edge_endpoints_disambiguated": edge_endpoints_disambiguated,
        "edge_endpoints_not_disambiguated": edge_endpoints_not_disambiguated,
        "template_def_ids_disambiguated": template_def_ids_disambiguated,
        "template_def_ids_not_disambiguated": template_def_ids_not_disambiguated,
        "collisions": {
            class_id: [
                {
                    "qualified_id": qualified_ids[(class_id, site)],
                    "file": site[0],
                    "start_line": site[1],
                    "end_line": int(
                        row.get("def_end_line") or row.get("def_line") or 0
                    ),
                    "fq_name": site[2],
                    "content_hash": str(row.get("content_hash") or ""),
                }
                for site, row in sorted(rows.items())
            ]
            for class_id, rows in sorted(collisions.items())
        },
    }


_DEPENDENCY_REFERENCE_EDGE_TYPES = {
    "template_ref",
    "decl_ref",
    "member_ref",
    "type_ref",
    "unexposed_ref",
}


def _reconcile_dependency_endpoints(
    results: List[Dict[str, Any]], project_root: str
) -> Dict[str, Any]:
    """Resolve specialization USRs to a unique source-backed function/class."""
    code_ids: Set[str] = set()
    identity_lookup: Dict[Tuple[str, str], Set[str]] = {}
    for result in results:
        for entity in (*result.get("functions", []), *result.get("classes", [])):
            entity_id = str(entity.get("id") or "")
            file_name = _definition_path(
                str(entity.get("def_file") or ""), project_root
            )
            fq_name = str(entity.get("fq_name") or "")
            if not entity_id:
                continue
            code_ids.add(entity_id)
            if file_name and fq_name:
                identity_lookup.setdefault((file_name, fq_name), set()).add(entity_id)

    stats: Counter[str] = Counter()
    by_edge_type: Dict[str, Counter[str]] = {}
    for result in results:
        for edge in result.get("call_edges", []):
            edge_type = str(edge.get("edge_type") or "call")
            if edge_type not in _DEPENDENCY_REFERENCE_EDGE_TYPES:
                continue
            edge_stats = by_edge_type.setdefault(edge_type, Counter())
            edge_stats["edges_observed"] += 1
            for endpoint in ("caller", "callee"):
                observed_id = str(edge.get(endpoint) or "")
                if observed_id in code_ids:
                    edge_stats["endpoints_already_valid"] += 1
                    stats["endpoints_already_valid"] += 1
                    continue
                file_name = _definition_path(
                    str(edge.get(f"{endpoint}_file") or ""), project_root
                )
                fq_name = str(edge.get(f"{endpoint}_name") or "")
                candidates = identity_lookup.get((file_name, fq_name), set())
                if len(candidates) == 1:
                    canonical_id = next(iter(candidates))
                    edge[endpoint] = canonical_id
                    resolution_value = edge.get("resolution")
                    resolution = (
                        dict(resolution_value)
                        if isinstance(resolution_value, dict)
                        else {}
                    )
                    reconciliations = dict(
                        resolution.get("dependency_endpoint_reconciliation") or {}
                    )
                    reconciliations[endpoint] = {
                        "method": "unique_source_entity_identity",
                        "observed_usr": observed_id,
                        "canonical_usr": canonical_id,
                        "file": file_name,
                        "fq_name": fq_name,
                    }
                    resolution["dependency_endpoint_reconciliation"] = reconciliations
                    edge["resolution"] = resolution
                    edge_stats["endpoints_reconciled"] += 1
                    stats["endpoints_reconciled"] += 1
                elif len(candidates) > 1:
                    edge_stats["ambiguous_endpoints"] += 1
                    stats["ambiguous_endpoints"] += 1
                else:
                    edge_stats["unmatched_endpoints"] += 1
                    stats["unmatched_endpoints"] += 1

    unique_misses = 0
    for result in results:
        for edge in result.get("call_edges", []):
            if edge.get("edge_type") not in _DEPENDENCY_REFERENCE_EDGE_TYPES:
                continue
            for endpoint in ("caller", "callee"):
                if str(edge.get(endpoint) or "") in code_ids:
                    continue
                identity = (
                    _definition_path(
                        str(edge.get(f"{endpoint}_file") or ""), project_root
                    ),
                    str(edge.get(f"{endpoint}_name") or ""),
                )
                if len(identity_lookup.get(identity, set())) == 1:
                    unique_misses += 1

    return {
        "edges_observed": sum(
            counters["edges_observed"] for counters in by_edge_type.values()
        ),
        "endpoints_already_valid": stats["endpoints_already_valid"],
        "endpoints_reconciled": stats["endpoints_reconciled"],
        "ambiguous_endpoints": stats["ambiguous_endpoints"],
        "unmatched_endpoints": stats["unmatched_endpoints"],
        "unique_identity_misses_after_reconciliation": unique_misses,
        "by_edge_type": {
            edge_type: dict(counters)
            for edge_type, counters in sorted(by_edge_type.items())
        },
    }


def _partition_graph_ready_edges(
    edges: List[Dict[str, Any]],
    function_ids: Set[str],
    source_definition_spans: Optional[
        Set[Tuple[str, str, int, int]]
    ] = None,
    source_definition_ids: Optional[
        Set[Tuple[str, str, int, int, str]]
    ] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Separate executable relations whose source-backed endpoints are absent."""
    retained: List[Dict[str, Any]] = []
    unresolved: List[Dict[str, Any]] = []
    by_reason: Counter[str] = Counter()
    span_lookup: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
    for file_name, fq_name, start, end in source_definition_spans or set():
        span_lookup.setdefault((file_name, fq_name), []).append((start, end))
    identity_lookup: Dict[Tuple[str, str], List[Tuple[int, int, str]]] = {}
    for file_name, fq_name, start, end, function_id in source_definition_ids or set():
        identity_lookup.setdefault((file_name, fq_name), []).append(
            (start, end, function_id)
        )
    reconciled = 0
    for edge in edges:
        if edge.get("edge_type", "call") not in {"call", "dependent_call"}:
            retained.append(edge)
            continue
        caller_present = edge.get("caller") in function_ids
        callee_present = edge.get("callee") in function_ids
        if caller_present and callee_present:
            retained.append(edge)
            continue
        edge_file = str(edge.get("callee_file") or "").replace("\\", "/")
        edge_name = str(edge.get("callee_name") or "")
        edge_line = int(edge.get("callee_line") or 0)
        identity_matches = {
            function_id
            for start, end, function_id in identity_lookup.get(
                (edge_file, edge_name), []
            )
            if edge_line and start <= edge_line <= end
        }
        if caller_present and len(identity_matches) == 1:
            row = dict(edge)
            observed_target = str(row.get("callee") or "")
            row["callee"] = next(iter(identity_matches))
            resolution = dict(row.get("resolution") or {})
            resolution["endpoint_reconciliation"] = {
                "method": "unique_source_definition_identity",
                "observed_target_usr": observed_target,
                "canonical_target_usr": row["callee"],
                "target_file": edge_file,
                "target_line": edge_line,
            }
            row["resolution"] = resolution
            retained.append(row)
            reconciled += 1
            continue
        row = dict(edge)
        source_span_present = any(
            start <= edge_line <= end
            for start, end in span_lookup.get((edge_file, edge_name), [])
        )
        explicitly_declaration_only = (
            edge.get("callee_is_definition") is False
            and edge.get("callee_has_body") is False
        )
        if not caller_present:
            reason = "missing_caller"
        elif source_span_present and not explicitly_declaration_only:
            reason = "missing_source_definition_callee"
        else:
            reason = "missing_declaration_or_implicit_callee"
        row["unresolved_reason"] = reason
        unresolved.append(row)
        by_reason[reason] += 1

    source_definition_orphans = by_reason["missing_source_definition_callee"]
    missing_callers = by_reason["missing_caller"]
    if source_definition_orphans or missing_callers:
        raise RuntimeError(
            "Executable edge endpoint validation failed: "
            f"missing_callers={missing_callers}, "
            f"missing_source_definition_callees={source_definition_orphans}."
        )
    executable_total = sum(
        edge.get("edge_type", "call") in {"call", "dependent_call"}
        for edge in edges
    )
    executable_retained = sum(
        edge.get("edge_type", "call") in {"call", "dependent_call"}
        for edge in retained
    )
    stats = {
        "executable_observed": executable_total,
        "executable_retained": executable_retained,
        "executable_unresolved": len(unresolved),
        "executable_retention_ratio": (
            executable_retained / executable_total if executable_total else 1.0
        ),
        "by_reason": dict(by_reason),
        "source_definition_endpoints_reconciled": reconciled,
    }
    return retained, unresolved, stats


_PREPROCESSOR_DIRECTIVE_RE = re.compile(
    r"^\s*#\s*(define|undef|include_next|include|if|ifdef|ifndef|elif|else|endif)\b(.*)$"
)


def _extract_preprocessor_directives(
    file_path: str, project_root: str
) -> List[Dict[str, Any]]:
    """Extract lexical directives, including includes, inactive branches and #undef."""
    try:
        lines = Path(file_path).read_text(
            encoding="utf-8", errors="ignore"
        ).splitlines()
    except OSError:
        return []
    relative = _definition_path(file_path, project_root)
    records: List[Dict[str, Any]] = []
    index = 0
    conditional_depth = 0
    while index < len(lines):
        start = index + 1
        logical = lines[index]
        while logical.rstrip().endswith("\\") and index + 1 < len(lines):
            index += 1
            logical += "\n" + lines[index]
        match = _PREPROCESSOR_DIRECTIVE_RE.match(logical)
        if match:
            directive = match.group(1)
            payload = match.group(2).strip()
            name = ""
            if directive in {"define", "undef", "ifdef", "ifndef"}:
                name_match = re.match(r"([A-Za-z_]\w*)", payload)
                name = name_match.group(1) if name_match else ""
            elif directive in {"include", "include_next"}:
                name = payload
            if directive == "endif":
                conditional_depth = max(0, conditional_depth - 1)
            identity = f"{relative}:{start}:{directive}:{name}:{payload}"
            records.append(
                {
                    "id": "pp:"
                    + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
                    "directive": directive,
                    "name": name,
                    "expression": payload,
                    "file": relative,
                    "line": start,
                    "end_line": index + 1,
                    "conditional_depth": conditional_depth,
                    "source": "lexical_scan",
                }
            )
            if directive in {"if", "ifdef", "ifndef"}:
                conditional_depth += 1
        index += 1
    return records


# 与 HEADER_SUFFIXES 一致；置于前向定义之前供 _task_executor 使用
_HEADER_SUFFIXES_FOR_CXX_RETRY = frozenset({".h", ".hpp", ".hh", ".hxx", ".h++"})


def _parse_result_is_effectively_empty(res: Optional[Dict[str, Any]]) -> bool:
    """广义「无产物」：用于合并/缓存等（含仅宏引用、边）。"""
    if not res:
        return True
    if res.get("functions") or res.get("classes") or res.get("macro_defs"):
        return False
    if res.get("call_edges") or res.get("template_instantiations"):
        return False
    if res.get("macro_uses"):
        return False
    return True


def _header_c_mode_missing_symbols(res: Optional[Dict[str, Any]]) -> bool:
    """
    是否应尝试 -xc++-header：仅看函数/类。

    C 模式解析 C++ 头时，仍可能从系统头带入 macro_uses / call_edges，
    若用 _parse_result_is_effectively_empty 会误判为「非空」从而不重试。
    """
    if not res:
        return True
    return not (res.get("functions") or res.get("classes"))


def _compile_args_for_cpp_header_retry(fargs: List[str]) -> Optional[List[str]]:
    """
    独立头文件从 -xc-header 改 -xc++-header 时，须去掉来自 .c 的 -std=c11 / -std=gnu99 等，
    否则 Clang 仍按 C 标准处理，C++ 类（如 contrib/pzstd/Logging.h 的 Logger）抽不出实体。
    """
    if "-xc-header" not in fargs:
        return None
    out: List[str] = []
    has_cpp_std = False
    for a in fargs:
        if a == "-xc-header":
            continue
        if a.startswith("-std="):
            low = a.lower()
            if "c++" in low or "gnu++" in low:
                has_cpp_std = True
                out.append(a)
            continue
        out.append(a)
    if not has_cpp_std:
        out.append("-std=c++17")
    out.append("-xc++-header")
    return out


def _compile_args_with_working_directory(
    fargs: List[str], cwd: Optional[str]
) -> List[str]:
    """Bind relative compiler arguments to this translation unit's directory."""
    if not cwd:
        return list(fargs)

    out: List[str] = []
    skip_next = False
    for arg in fargs:
        if skip_next:
            skip_next = False
            continue
        if arg == "-working-directory":
            skip_next = True
            continue
        if arg.startswith("-working-directory="):
            continue
        out.append(arg)
    out.append(f"-working-directory={os.path.realpath(cwd)}")
    return out


def get_file_hash(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return ""


def _task_executor(
    t: Tuple[str, List[str], Optional[str], str, Optional[str], Path]
) -> Dict[str, Any]:
    fpath, fargs, cwd, root, lib_path, c_file = t
    
    try:
        idx = Index.create()
        parse_args = _compile_args_with_working_directory(fargs, cwd)
        res = extract_translation_unit(idx, fpath, parse_args, root)

        # 独立 .h 使用 -xc-header 时若几乎无实体，改用 -xc++-header 重试一次（如 contrib/pzstd）
        suf = Path(fpath).suffix.lower()
        if (
            suf in _HEADER_SUFFIXES_FOR_CXX_RETRY
            and "-xc-header" in fargs
            and _header_c_mode_missing_symbols(res)
        ):
            cpp_args = _compile_args_for_cpp_header_retry(list(fargs))
            if cpp_args is not None:
                cpp_args = _compile_args_with_working_directory(cpp_args, cwd)
                res_cpp = extract_translation_unit(idx, fpath, cpp_args, root)
                if not _header_c_mode_missing_symbols(res_cpp):
                    log_parse_debug(f"[Indexer] Header re-parsed as C++: {fpath}")
                    res = res_cpp

        if res and c_file:
            rel_res = _convert_to_relative_paths(res, root)
            with c_file.open("w", encoding="utf-8") as f:
                json.dump(rel_res, f, ensure_ascii=False)
        return res
    except Exception as e:

        error_msg = f"[Task Error] Failed to process file: {fpath} - {type(e).__name__}: {e}"
        print(error_msg)
        log_parse_debug(f"[Task Executor Error] Failed to process file: {fpath}")
        log_parse_debug(f"   Error type: {type(e).__name__}")
        log_parse_debug(f"   Error detail: {e}")
        import traceback
        log_parse_debug(f"   Traceback:\n{traceback.format_exc()}")
        return empty_result()


def _init_process_worker(clang_library_path: Optional[str]) -> None:
    """Initialize libclang inside each freshly spawned parser process."""
    init_clang(clang_library_path)


def _process_task_executor(
    task: Tuple[str, List[str], Optional[str], str, Optional[str], Path]
) -> str:
    """Parse into the per-file cache without copying a large result over IPC."""
    _task_executor(task)
    cache_file = task[-1]
    return str(cache_file) if cache_file.is_file() else ""


def _convert_to_relative_paths(res: Dict[str, Any], project_root: str) -> Dict[str, Any]:
    def rel(path: str) -> str:
        if not path:
            return path
        if not os.path.isabs(path):
            return path
        try:
            return os.path.relpath(path, project_root)
        except Exception:
            return path

    new_res = res.copy()
    for f in new_res.get("functions", []):
        f["def_file"] = rel(f.get("def_file", ""))
    for c in new_res.get("classes", []):
        c["def_file"] = rel(c.get("def_file", ""))
    for m in new_res.get("macro_defs", []):
        m["file"] = rel(m.get("file", ""))
    for e in new_res.get("call_edges", []):
        e["caller_file"] = rel(e.get("caller_file", ""))
        e["callee_file"] = rel(e.get("callee_file", ""))
    for inst in new_res.get("template_instantiations", []):
        inst["trigger_file"] = rel(inst.get("trigger_file", ""))
        inst["template_def_file"] = rel(inst.get("template_def_file", ""))
    return new_res


def _restore_absolute_paths(res: Dict[str, Any], project_root: str) -> Dict[str, Any]:
    def abs_p(path: str) -> str:
        if not path:
            return path
        if os.path.isabs(path):
            return path
        return os.path.abspath(os.path.join(project_root, path))

    new_res = res.copy()
    for f in new_res.get("functions", []):
        f["def_file"] = abs_p(f.get("def_file", ""))
    for c in new_res.get("classes", []):
        c["def_file"] = abs_p(c.get("def_file", ""))
    for m in new_res.get("macro_defs", []):
        m["file"] = abs_p(m.get("file", ""))
    for e in new_res.get("call_edges", []):
        e["caller_file"] = abs_p(e.get("caller_file", ""))
        e["callee_file"] = abs_p(e.get("callee_file", ""))
    for inst in new_res.get("template_instantiations", []):
        inst["trigger_file"] = abs_p(inst.get("trigger_file", ""))
        inst["template_def_file"] = abs_p(inst.get("template_def_file", ""))
    return new_res

def extract_translation_unit(
    index: Index, file_path: str, args: List[str], project_root: str
) -> Dict[str, Any]:
    
    if not os.path.exists(file_path):
        log_parse_debug(f"[Skip] File not found: {file_path}")
        return empty_result()

    try:
        tu: TranslationUnit = index.parse(
            file_path,
            args=args,
            options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
        )
    except BaseException as e:
        error_msg = f"[Parse Error] Parse failed: {file_path} - {type(e).__name__}: {e}"
        print(error_msg)
        log_parse_debug(f"[Parse Error] Parse failed: {file_path}")
        log_parse_debug(f"   Error detail: {e}")
        import traceback
        log_parse_debug(f"   Traceback:\n{traceback.format_exc()}")
        return empty_result()
    if tu is None:
        log_parse_debug(f"[Parse Warn] TU is None: {file_path}")
        return empty_result()
    
    if tu.diagnostics:
        error_diags = [d for d in tu.diagnostics if "error" in str(d).lower()]
        if error_diags:
            log_parse_debug(f"[Parse Warn] {file_path} has parse errors:")
            for d in error_diags[:2]:
                log_parse_debug(f"    {d}")

    functions: List[FunctionInfo] = []
    classes: List[ClassInfo] = []
    macro_defs: List[MacroDefInfo] = []
    macro_uses: List[MacroUseInfo] = []
    template_instantiations: List[Dict[str, Any]] = []
    seen_template_instantiations: Set[
        Tuple[str, str, int, str, str, str]
    ] = set()
    edge_records: Dict[Tuple[str, str, str, str, str, str, str], Dict[str, Any]] = {}
    edge_stats: Counter[str] = Counter()

    seen_definition_ids: Set[Tuple[str, str, int, str, str]] = set()
    seen_macro_ids: Set[str] = set()
    context_meta: Dict[str, Dict[str, Any]] = {}
    func_cursors: List[Tuple[FunctionInfo, Cursor]] = []

    project_root = os.path.abspath(project_root)
    project_root_real = os.path.realpath(project_root)

    def is_project_file(fpath: str) -> bool:
        if not fpath:
            return False
        
        if not os.path.isabs(fpath):
            return True
        
        try:
            real_path = os.path.realpath(fpath)
            
            if real_path.startswith(project_root_real):
                return True
            
            try:
                rel_path = os.path.relpath(real_path, project_root_real)
                if not rel_path.startswith(".."):
                    return True
            except (ValueError, OSError):
                pass
            
            project_basename = os.path.basename(project_root_real)
            if project_basename:
                path_parts = real_path.split(os.sep)
                project_parts = project_root_real.split(os.sep)
                
                project_suffix_parts = project_parts[project_parts.index(project_basename)+1:] if project_basename in project_parts else []
                
                try:
                    for i in range(len(path_parts) - 1, -1, -1):
                        if path_parts[i] == project_basename:
                            file_suffix_parts = path_parts[i+1:]
                            
                            if not project_suffix_parts:
                                return True
                            
                            if len(file_suffix_parts) >= len(project_suffix_parts):
                                if file_suffix_parts[:len(project_suffix_parts)] == project_suffix_parts:
                                    return True
                            break
                except (ValueError, IndexError):
                    pass
            
            return False
        except (OSError, ValueError):
            return False

    def _resolve_template_target(cursor: Cursor) -> Optional[Cursor]:
        if cursor is None:
            return None
        ref = cursor
        parents = []
        while ref:
            parents.append(ref)
            ref = getattr(ref, "semantic_parent", None)

        target = cursor
        for parent in parents:
            if parent.kind in (
                CursorKind.CLASS_TEMPLATE,
                CursorKind.STRUCT_DECL,
                CursorKind.CLASS_DECL,
            ):
                return parent
        return cursor

    def append_edge(
        edge_type: str,
        source_id: Optional[str],
        source_name: Optional[str],
        target_cursor: Optional[Cursor],
        fallback_cursor: Optional[Cursor] = None,
        inst_data: Optional[Dict[str, Any]] = None,
        resolution_data: Optional[Dict[str, Any]] = None,
        source_file: Optional[str] = None,
        source_definition_line: int = 0,
    ) -> None:
        if edge_type not in ENABLED_EDGE_TYPES:
            return
        if not source_id or not source_name:
            log_edge_debug(f"[EDGE_SKIP] no_context {edge_type}")
            return
        try:
            target_cursor = target_cursor or fallback_cursor
            if not target_cursor:
                log_edge_debug(
                    f"[EDGE_SKIP] no_target {edge_type} from {source_name}"
                )
                return
            # Member/decl refs already point at the concrete function or field.
            # Collapsing them to a semantic parent erases forwarding call chains.
            if edge_type in {"template_ref", "type_ref"}:
                target_cursor = _resolve_template_target(target_cursor) or target_cursor
            loc_cursor = (
                target_cursor
                if target_cursor.location and target_cursor.location.file
                else fallback_cursor
            )
            loc = loc_cursor.location if loc_cursor else target_cursor.location
            callee_usr = get_usr(target_cursor)
            
            callee_path = None
            
            if edge_type == "call" and (not loc or not loc.file):
                if callee_usr and callee_usr in func_map:
                    callee_func = func_map[callee_usr]
                    callee_path = callee_func.def_file
                    if not is_project_file(callee_path):
                        log_edge_debug(
                            f"[EDGE_SKIP] outside_root {edge_type} -> {callee_path} (root: {project_root})"
                        )
                        return
                    if _is_third_party(callee_path):
                        log_edge_debug(
                            f"[EDGE_SKIP] third_party {edge_type} -> {callee_path}"
                        )
                        return
                    loc = type('obj', (object,), {
                        'file': callee_func.def_file,
                        'line': callee_func.def_start_line,
                        'column': 0
                    })()
                else:
                    log_edge_debug(
                        f"[EDGE_SKIP] no_path_no_usr {edge_type} -> {target_cursor.spelling or target_cursor.displayname or 'unknown'}"
                    )
                    return
            elif not loc or not loc.file:
                log_edge_debug(
                    f"[EDGE_SKIP] no_path {edge_type} -> {target_cursor.spelling or target_cursor.displayname or 'unknown'}"
                )
                return
            else:
                callee_path = os.path.realpath(str(loc.file))
                if not is_project_file(callee_path):
                    log_edge_debug(
                        f"[EDGE_SKIP] outside_root {edge_type} -> {callee_path} (root: {project_root})"
                    )
                    return
                if _is_third_party(callee_path):
                    log_edge_debug(
                        f"[EDGE_SKIP] third_party {edge_type} -> {callee_path}"
                    )
                    return
                if not EDGE_ALLOW_TEST_TARGETS and _is_test_path(callee_path):
                    log_edge_debug(
                        f"[EDGE_SKIP] target_is_test {edge_type} -> {callee_path}"
                    )
                    return
            
            source_info = context_meta.get(source_id)
            source_path = os.path.realpath(source_file) if source_file else None
            if not source_path and source_info and source_info.get("file"):
                source_path = os.path.realpath(source_info["file"])
            if (
                source_path
                and edge_type in CROSS_FILE_ONLY_EDGE_TYPES
                and callee_path == source_path
            ):
                log_edge_debug(
                    f"[EDGE_SKIP] same_file {edge_type} {source_name} -> {callee_path}"
                )
                return
            
            if not callee_path:
                if loc and loc.file:
                    callee_path = os.path.realpath(str(loc.file))
                elif callee_usr and callee_usr in func_map:
                    callee_func = func_map[callee_usr]
                    callee_path = callee_func.def_file
            
            if not callee_usr:
                callee_usr = (
                    get_fully_qualified_name(target_cursor)
                    if edge_type == "instantiates"
                    else ""
                ) or f"{callee_path}:{loc.line}:{loc.column}"
            callee_name = (
                get_fully_qualified_name(target_cursor)
                or target_cursor.displayname
                or callee_usr
                or "unknown"
            )
            # A macro-expanded header can expose the same Clang USR under
            # several namespaces in one translation unit.  Preserve those
            # relations until merge-time source qualification gives each
            # definition its final ID.
            edge_key = (
                edge_type,
                source_id,
                source_name,
                callee_usr,
                callee_name,
                source_path or "",
                callee_path or "",
                source_definition_line,
            )
            existing = edge_records.get(edge_key)
            if existing:
                existing["count"] += 1
                if not existing.get("caller_file") and source_path:
                    existing["caller_file"] = source_path
                if not existing.get("callee_file") and callee_path:
                    existing["callee_file"] = callee_path
                if not existing.get("resolution") and resolution_data:
                    existing["resolution"] = resolution_data
                return
            edge_stats[edge_type] += 1
            edge_payload = {
                "caller": source_id,
                "caller_name": source_name,
                "callee": callee_usr,
                "callee_name": callee_name,
                "edge_type": edge_type,
                "count": 1,
            }
            edge_payload["callee_line"] = int(getattr(loc, "line", 0) or 0)
            edge_payload["callee_column"] = int(
                getattr(loc, "column", 0) or 0
            )
            if edge_type in {"call", "dependent_call"}:
                try:
                    edge_payload["callee_kind"] = target_cursor.kind.name
                except Exception:
                    edge_payload["callee_kind"] = str(
                        getattr(target_cursor, "kind", "")
                    )
                try:
                    edge_payload["callee_is_definition"] = bool(
                        target_cursor.is_definition()
                    )
                except Exception:
                    edge_payload["callee_is_definition"] = False
                try:
                    edge_payload["callee_has_body"] = any(
                        child.kind == CursorKind.COMPOUND_STMT
                        for child in target_cursor.get_children()
                    )
                except Exception:
                    edge_payload["callee_has_body"] = False
            if source_path:
                edge_payload["caller_file"] = source_path
            if source_definition_line:
                edge_payload["caller_definition_line"] = source_definition_line
            if callee_path:
                edge_payload["callee_file"] = callee_path
            if inst_data:
                edge_payload["instantiation"] = inst_data
                if "type_spelling" in inst_data:
                    edge_payload["instantiated_type"] = inst_data.get("type_spelling")
            if resolution_data:
                edge_payload["resolution"] = resolution_data
            edge_records[edge_key] = edge_payload
            if edge_type != "call":
                log_edge_debug(
                    f"[EDGE_WRITE] {edge_type} {source_name} -> {callee_name} ({callee_path})"
                )
        except Exception as exc:
            log_edge_debug(
                f"[EDGE_ERROR] {edge_type} {source_name} target={getattr(target_cursor, 'spelling', '')}: {exc}"
            )

    def _extract_template_instantiation_info(cursor: Cursor) -> Optional[Dict[str, Any]]:
        candidates = []
        try:
            candidates.append(cursor.type)
        except Exception:
            pass
        try:
            if cursor.semantic_parent is not None:
                candidates.append(cursor.semantic_parent.type)
        except Exception:
            pass

        for ty in candidates:
            if not ty:
                continue
            try:
                ct = ty.get_canonical()
            except Exception:
                ct = ty
            num = -1
            try:
                num = ct.get_num_template_arguments()
            except Exception:
                num = -1
            if num is None or num <= 0:
                continue
            args: List[str] = []
            for i in range(num):
                try:
                    arg_t = ct.get_template_argument_type(i)
                    args.append(arg_t.spelling if arg_t else "")
                except Exception:
                    args.append("")
            return {"type_spelling": ct.spelling, "template_args": args}
        return None

    def record_dependency(
        cursor: Cursor,
        context_id: str,
        context_name: str,
        context_file: Optional[str],
    ) -> None:
        dep_type = None
        if cursor.kind == CursorKind.TEMPLATE_REF:
            dep_type = "template_ref"
        elif cursor.kind == CursorKind.DECL_REF_EXPR:
            dep_type = "decl_ref"
        elif cursor.kind == CursorKind.TYPE_REF:
            dep_type = "type_ref"
        elif cursor.kind == CursorKind.MEMBER_REF_EXPR:
            dep_type = "member_ref"
        elif cursor.kind in (
            CursorKind.UNEXPOSED_EXPR,
            CursorKind.UNEXPOSED_DECL,
            CursorKind.OVERLOADED_DECL_REF,
            CursorKind.UNEXPOSED_ATTR,
        ):
            dep_type = "unexposed_ref"
        if not dep_type or dep_type not in ENABLED_EDGE_TYPES:
            return
        if dep_type:
            loc = cursor.location
            loc_info = (
                f"{loc.file}:{loc.line}" if loc and loc.file else "unknown"
            )
            log_edge_debug(
                f"[EDGE_DEBUG] {dep_type} @ {loc_info} in context {context_name}"
            )
            target_cursor = None
            try:
                target_cursor = cursor.referenced
            except Exception as exc:
                log_edge_debug(
                    f"[EDGE_TRACE] referenced_error {dep_type} at {loc_info}: {exc}"
                )
            inst_data = None
            if dep_type == "template_ref":
                inst_data = _extract_template_instantiation_info(cursor)
            append_edge(
                dep_type,
                context_id,
                context_name,
                target_cursor,
                cursor,
                inst_data,
                source_file=context_file,
            )

    def _collect_template_arguments(cursor: Cursor) -> List[str]:
        args: List[str] = []
        if not hasattr(cursor, "get_template_argument_kind"):
            return args
        idx = 0
        while True:
            try:
                kind = cursor.get_template_argument_kind(idx)
            except Exception:
                break
            if kind == TemplateArgumentKind.INVALID:
                break
            if kind == TemplateArgumentKind.TYPE:
                try:
                    arg_type = cursor.get_template_argument_type(idx)
                except Exception:
                    args.append("type")
                else:
                    args.append(arg_type.spelling)
            else:
                args.append(str(kind))
            idx += 1
        return args

    def get_node_category(cursor: Cursor) -> str:
        kind = cursor.kind
        if kind == CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION:
            log_template_debug(
                f"[TplInst] Found Partial Spec: {cursor.spelling}"
            )
            return "Partial"
        if kind == CursorKind.CLASS_TEMPLATE:
            return "Primary"
        if kind in (CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL):
            get_tpl = getattr(cursor, "get_specialized_cursor_template", None)
            if callable(get_tpl):
                try:
                    tpl = get_tpl()
                    if tpl:
                        log_template_debug(
                            "[TplInst] Found Full Spec (via specialized): "
                            f"{cursor.spelling} -> {tpl.spelling}"
                        )
                        return "Full"
                except Exception:
                    pass
        return "Normal"

    def _collect_template_arguments(node: Cursor) -> List[str]:
        args = []
        try:
            for child in node.get_children():
                if child.kind == CursorKind.TEMPLATE_TYPE_PARAMETER:
                    args.append(child.spelling or child.displayname or "")
                elif child.kind == CursorKind.TEMPLATE_NON_TYPE_PARAMETER:
                    args.append(child.spelling or child.displayname or "")
        except Exception:
            pass
        return args

    def _register_template_instantiation(
        node: Cursor, display_name: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        
        try:
            name = node.spelling or node.displayname
        except Exception:
            name = "unknown"
            
        log_template_debug(f"[TEMPLATE_INST] START for {name} kind={node.kind}")

        template_cursor = None
        get_template = getattr(node, "get_specialized_cursor_template", None)
        if callable(get_template):
            try:
                template_cursor = get_template()
            except Exception:
                template_cursor = None
        
        if not template_cursor:
            try:
                from clang.cindex import conf  # type: ignore
                template_cursor = conf.lib.clang_getSpecializedCursorTemplate(node)
            except Exception as e:
                template_cursor = None
        
        is_invalid = True
        try:
            if template_cursor and not template_cursor.kind.is_invalid():
                is_invalid = False
        except Exception:
            pass
            
        if is_invalid:
            log_template_debug(f"[TEMPLATE_INST] SKIP - no template_cursor for {name}")
            return None
            
        loc = node.location or (node.extent.start if node.extent and node.extent.start else None)
        file_path = str(loc.file) if loc and getattr(loc, "file", None) else None
        
        if file_path and not is_project_file(os.path.abspath(file_path)):
            log_template_debug(f"[TEMPLATE_INST] SKIP - outside project: {file_path} for {name}")
            return None
        
        try:
            template_name = template_cursor.spelling or template_cursor.displayname or "unknown"
        except Exception:
            template_name = "unknown"
            
        log_template_debug(f"[TEMPLATE_INST] SUCCESS! name={name} from={template_name} file={file_path}")

        try:
            inst_id = str(
                make_entity_id(
                    kind="template_instantiation",
                    usr=get_usr(node),
                    file_path=file_path,
                    line=loc.line if loc else None,
                    project_root=project_root,
                    extra=display_name or node.spelling or "",
                )
            )
            log_template_debug(f"[TEMPLATE_INST] Generated ID: {inst_id}")
        except Exception as e:
            log_template_debug(f"[TEMPLATE_INST] ERROR in make_entity_id: {e} for {node.spelling or node.displayname}")
            if "MyTemplate" in (node.spelling or ""):
                 print(f"DEBUG: make_entity_id failed: {e}")
            return None
        if "MyTemplate" in (node.spelling or ""):
             print(f"DEBUG: Generated ID: {inst_id}")

        fq_name = get_fully_qualified_name(node) or display_name or name or ""
        template_def_name = get_fully_qualified_name(template_cursor) or template_name
        template_def_file = ""
        template_def_line = 0
        template_def_column = 0
        try:
            template_loc = template_cursor.location
            if template_loc and template_loc.file:
                candidate = os.path.realpath(str(template_loc.file))
                if is_project_file(candidate):
                    template_def_file = candidate
                    template_def_line = int(template_loc.line or 0)
                    template_def_column = int(template_loc.column or 0)
        except Exception:
            pass
        instantiation_key = (
            inst_id,
            os.path.realpath(file_path) if file_path else "",
            int(loc.line if loc else 0),
            fq_name,
            template_def_name,
            template_def_file,
        )
        if instantiation_key in seen_template_instantiations:
            log_template_debug(f"[TEMPLATE_INST] SKIP - duplicate source identity: {inst_id} for {node.spelling or node.displayname}")
            if "MyTemplate" in (node.spelling or ""):
                 print(f"DEBUG: Already seen ID: {inst_id}")
            return None
        seen_template_instantiations.add(instantiation_key)
        log_template_debug(f"[TEMPLATE_INST] Added source identity to seen set: {inst_id}")
        template_usr = get_usr(template_cursor)
        template_def_id = template_usr or get_fully_qualified_name(template_cursor)
        template_args = _collect_template_arguments(node)
        inst_record = {
            "id": inst_id,
            "display_name": display_name or node.displayname or node.spelling or "",
            "fq_name": fq_name,
            "template_def_id": template_def_id,
            "template_def_name": template_def_name,
            "template_def_file": template_def_file,
            "template_def_line": template_def_line,
            "template_def_column": template_def_column,
            "template_args": template_args,
            "trigger_file": file_path,
            "trigger_line": loc.line if loc else None,
            "category": get_node_category(node),
            "is_explicit_specialization": node.kind in (
                CursorKind.FUNCTION_TEMPLATE,
                *([CursorKind.CLASS_TEMPLATE_SPECIALIZATION] if hasattr(CursorKind, "CLASS_TEMPLATE_SPECIALIZATION") else []),
                CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION,
            ),
        }
        append_edge(
            "instantiates",
            inst_id,
            inst_record["fq_name"],
            template_cursor,
            node,
            source_file=file_path,
        )
        template_instantiations.append(inst_record)
        log_template_debug(f"[TEMPLATE_INST] APPENDED! list_len={len(template_instantiations)} id={inst_id} name={inst_record['display_name']}")
        return inst_record

    def visit(
        node: Cursor,
        context_id: Optional[str] = None,
        context_name: Optional[str] = None,
        context_file: Optional[str] = None,
    ):
        if node is None:
            return
        try:
            if node.kind.is_invalid():
                return
        except Exception:
            return

        try:
            kind = node.kind

            node_file = None
            if node.location.file:
                node_file = os.path.abspath(str(node.location.file))

            should_index = bool(node_file and is_project_file(node_file))

            tpl_kinds = {
                CursorKind.CLASS_TEMPLATE,
                CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION,
                CursorKind.FUNCTION_TEMPLATE,
            }
            if hasattr(CursorKind, "CLASS_TEMPLATE_SPECIALIZATION"):
                tpl_kinds.add(CursorKind.CLASS_TEMPLATE_SPECIALIZATION)
            
            if kind in tpl_kinds:
                log_template_debug(f"[TEMPLATE_NODE] kind={kind} name={node.spelling} file={node_file} is_def={node.is_definition()} should_index={should_index}")

            if should_index:
                is_def = node.is_definition()
                if not is_def and is_function_like(kind):
                    try:
                        # Parameters/template arguments are not a body. Recording
                        # a declaration here hides its later out-of-line definition.
                        if any(child.kind == CursorKind.COMPOUND_STMT for child in node.get_children()):
                            is_def = True
                    except Exception: pass

                if is_function_like(kind) and is_def:
                    fi = build_function_info(node)
                    if fi:
                        definition_key = (
                            "function",
                            fi.id,
                            fi.def_start_line,
                            os.path.realpath(fi.def_file),
                            fi.fq_name,
                        )
                        if not _is_third_party(fi.def_file):
                            if definition_key not in seen_definition_ids:
                                seen_definition_ids.add(definition_key)
                                functions.append(fi)
                                context_meta[fi.id] = {
                                    "file": fi.def_file,
                                    "is_test": fi.is_test,
                                }
                                func_cursors.append((fi, node))
                            context_id = fi.id
                            context_name = fi.fq_name
                            context_file = fi.def_file

                if kind in CLASS_KINDS:
                    if node.is_definition() and node.spelling:
                        ci = build_class_info(node)
                        if ci:
                            definition_key = (
                                "class",
                                ci.id,
                                ci.def_line,
                                os.path.realpath(ci.def_file),
                                ci.fq_name,
                            )
                            if not _is_third_party(ci.def_file):
                                if definition_key not in seen_definition_ids:
                                    seen_definition_ids.add(definition_key)
                                    classes.append(ci)
                                    context_meta[ci.id] = {
                                        "file": ci.def_file,
                                        "is_test": ci.is_test,
                                    }
                                context_id = ci.id
                                context_name = ci.fq_name
                                context_file = ci.def_file
                check_kinds = {
                    CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION,
                    CursorKind.FUNCTION_DECL,
                    CursorKind.CXX_METHOD,
                    CursorKind.CONSTRUCTOR,
                    CursorKind.STRUCT_DECL,
                    CursorKind.CLASS_DECL,
                }
                has_cls_tpl_spec = hasattr(CursorKind, "CLASS_TEMPLATE_SPECIALIZATION")
                if has_cls_tpl_spec:
                    check_kinds.add(CursorKind.CLASS_TEMPLATE_SPECIALIZATION)

                if kind in check_kinds:
                    is_def = node.is_definition()
                    try:
                        node_name = node.spelling or node.displayname
                    except Exception:
                        node_name = "unknown"
                    log_template_debug(f"[TEMPLATE_CHECK] kind={kind} in check_kinds, is_definition={is_def}, name={node_name}")
                    if is_def:
                        log_template_debug(f"[TEMPLATE_CHECK] CALLING _register_template_instantiation for {node_name}")
                        _register_template_instantiation(
                            node, node_name or context_name
                        )
                elif not has_cls_tpl_spec and node.is_definition():
                    try:
                        node_name = node.spelling or node.displayname
                    except Exception:
                        node_name = "unknown"
                    log_template_debug(f"[TEMPLATE_CHECK] FALLBACK: kind={kind}, is_definition=True, name={node_name}")
                    tmpl_cursor = None
                    get_template = getattr(node, "get_specialized_cursor_template", None)
                    if callable(get_template):
                        try:
                            tmpl_cursor = get_template()
                        except Exception:
                            tmpl_cursor = None
                    if tmpl_cursor:
                        log_template_debug(f"[TEMPLATE_CHECK] FALLBACK: Found template_cursor, CALLING _register_template_instantiation")
                        _register_template_instantiation(
                            node, node_name or context_name
                        )
                    else:
                        log_template_debug(f"[TEMPLATE_CHECK] FALLBACK: No template_cursor found")

                if kind == CursorKind.MACRO_DEFINITION:
                    mi = build_macro_def(node)
                    uniq_key = f"macro:{mi.name}:{mi.file}:{mi.line}"
                    if mi and uniq_key not in seen_macro_ids:
                        seen_macro_ids.add(uniq_key)
                        macro_defs.append(mi)

            if kind == CursorKind.MACRO_INSTANTIATION and should_index:
                mu = build_macro_use(node)
                if mu:
                    macro_uses.append(mu)

            if context_id and context_name:
                record_dependency(node, context_id, context_name, context_file)

            for child in node.get_children():
                visit(child, context_id, context_name, context_file)

        except Exception as e:
            print(f"ERROR in visit: {e}")
            import traceback
            traceback.print_exc()

    if "catch_version.cpp" in file_path:
        print(f"DEBUG: [CheckPoint] Before invoking visit for {file_path}")

    visit(tu.cursor)

    if "catch_version.cpp" in file_path:
        print(f"DEBUG: [CheckPoint] After invoking visit for {file_path}. functions count={len(functions)}")

    func_map: Dict[str, FunctionInfo] = {}
    for function in functions:
        func_map.setdefault(function.id, function)
    function_cursor_map: Dict[int, Cursor] = {
        id(function): cursor for function, cursor in func_cursors
    }
    class_map: Dict[str, ClassInfo] = {}
    for cls in classes:
        class_map.setdefault(cls.id, cls)

    # Out-of-line members of an implicit class specialization can point at a
    # synthetic specialization USR for which libclang emits no class cursor.
    # Preserve real partial/explicit specialization owners that were indexed,
    # and only walk the template chain for owners missing from this TU.
    for function, cursor in func_cursors:
        if not function.owner_class_id or function.owner_class_id in class_map:
            continue
        owner = cursor.semantic_parent if cursor else None
        seen_owner_usrs: Set[str] = set()
        for _ in range(8):
            if not owner or owner.kind not in CLASS_KINDS:
                break
            owner_usr = get_usr(owner)
            if owner_usr in seen_owner_usrs:
                break
            seen_owner_usrs.add(owner_usr)
            template_owner = _specialized_cursor_template(owner)
            if not template_owner or template_owner.kind not in CLASS_KINDS:
                break
            template_usr = get_usr(template_owner)
            if template_usr in class_map:
                function.owner_class_id = template_usr
                break
            owner = template_owner

    def cursor_parameters(cursor: Cursor) -> Optional[List[Cursor]]:
        """Return callable parameters, including FUNCTION_TEMPLATE children."""
        try:
            arguments = list(cursor.get_arguments())
            if arguments:
                return arguments
        except Exception:
            pass
        try:
            return [
                child
                for child in cursor.get_children()
                if child.kind == CursorKind.PARM_DECL
            ]
        except Exception:
            return None

    def cursor_parameter_count(cursor: Cursor) -> Optional[int]:
        """Return callable arity, including FUNCTION_TEMPLATE parameters."""
        parameters = cursor_parameters(cursor)
        return len(parameters) if parameters is not None else None

    def cursor_argument_bounds(cursor: Cursor) -> Optional[Tuple[int, int]]:
        """Return required/total argument counts from explicit default values."""
        parameters = cursor_parameters(cursor)
        if parameters is None:
            return None
        required = len(parameters)
        while required:
            try:
                has_default = any(
                    token.spelling == "="
                    for token in parameters[required - 1].get_tokens()
                )
            except Exception:
                has_default = False
            if not has_default:
                break
            required -= 1
        return required, len(parameters)

    def cursor_accepts_argument_count(cursor: Cursor, count: int) -> bool:
        bounds = cursor_argument_bounds(cursor)
        return bool(bounds and bounds[0] <= count <= bounds[1])

    def specialized_template_cursor(cursor: Cursor) -> Optional[Cursor]:
        return _specialized_cursor_template(cursor)

    def resolve_indexed_call_target(
        target: Cursor,
    ) -> Tuple[Cursor, Optional[Dict[str, Any]]]:
        """Map a specialized call target to its source-backed indexed definition."""
        original_usr = get_usr(target)
        if original_usr in func_map:
            return target, None

        template_targets: List[Cursor] = []
        seen_template_cursors: Set[Tuple[str, str, str, int, int]] = set()
        current_target = target
        for specialization_depth in range(1, 9):
            template_target = specialized_template_cursor(current_target)
            if not template_target:
                break
            try:
                location = template_target.location
                template_key = (
                    get_usr(template_target),
                    str(template_target.kind),
                    str(location.file) if location and location.file else "",
                    int(location.line if location else 0),
                    int(location.column if location else 0),
                )
            except Exception:
                break
            if template_key in seen_template_cursors:
                break
            seen_template_cursors.add(template_key)
            template_targets.append(template_target)
            template_usr = template_key[0]
            if template_usr in func_map:
                return template_target, {
                    "method": "clang_specialized_template",
                    "confidence": "high",
                    "observed_target_usr": original_usr,
                    "canonical_target_usr": template_usr,
                    "specialization_depth": specialization_depth,
                }
            current_target = template_target

        probes = [target]
        if template_targets:
            probes = list(reversed(template_targets)) + probes
        seen_probe_keys: Set[Tuple[str, int, str, Optional[int]]] = set()
        matches: Dict[
            Tuple[str, str, int, str], Tuple[FunctionInfo, Cursor]
        ] = {}
        for probe in probes:
            try:
                location = probe.location
                path = (
                    os.path.realpath(str(location.file))
                    if location and location.file
                    else ""
                )
                line = int(location.line if location else 0)
            except Exception:
                path = ""
                line = 0
            name = (
                get_fully_qualified_name(probe)
                or probe.spelling
                or probe.displayname
                or ""
            )
            arity = cursor_parameter_count(probe)
            probe_key = (path, line, name, arity)
            if not path or not line or not name or probe_key in seen_probe_keys:
                continue
            seen_probe_keys.add(probe_key)
            for candidate in functions:
                if (
                    os.path.realpath(candidate.def_file) != path
                    or candidate.fq_name != name
                    or not (
                        candidate.def_start_line
                        <= line
                        <= candidate.def_end_line
                    )
                ):
                    continue
                candidate_cursor = function_cursor_map.get(id(candidate))
                candidate_arity = (
                    cursor_parameter_count(candidate_cursor)
                    if candidate_cursor
                    else len(candidate.parameter_types)
                )
                if arity is not None and candidate_arity != arity:
                    continue
                matches[(
                    candidate.id,
                    candidate.def_file,
                    candidate.def_start_line,
                    candidate.fq_name,
                )] = (candidate, candidate_cursor)

        if len(matches) != 1:
            return target, None
        _, (candidate, canonical_cursor) = next(iter(matches.items()))
        canonical_id = candidate.id
        return canonical_cursor, {
            "method": "source_definition_identity",
            "confidence": "high",
            "observed_target_usr": original_usr,
            "canonical_target_usr": canonical_id,
            "target_file": candidate.def_file,
            "target_line": candidate.def_start_line,
            "target_arity": len(candidate.parameter_types),
        }

    def resolve_dependent_call(
        call_cursor: Cursor,
    ) -> Optional[Tuple[Cursor, Dict[str, Any]]]:
        """Resolve a dependent call only when Clang's overload set is unique."""
        try:
            argument_count = len(list(call_cursor.get_arguments()))
            call_children = list(call_cursor.get_children())
        except Exception:
            return None
        if not call_children:
            return None

        callee_expr = call_children[0]
        try:
            while callee_expr.kind == CursorKind.UNEXPOSED_EXPR:
                children = list(callee_expr.get_children())
                if len(children) != 1:
                    break
                callee_expr = children[0]
            if callee_expr.kind == CursorKind.MEMBER_REF_EXPR:
                # The first child is the receiver.  Calls inside that receiver
                # are not candidates for the outer member invocation.
                search_roots = list(callee_expr.get_children())[1:]
            else:
                search_roots = [callee_expr]
        except Exception:
            return None

        overload_refs: List[Cursor] = []
        stack = list(search_roots)
        while stack:
            cursor = stack.pop()
            try:
                kind = cursor.kind
            except Exception:
                continue
            if kind == CursorKind.OVERLOADED_DECL_REF:
                overload_refs.append(cursor)
            try:
                stack.extend(reversed(list(cursor.get_children())))
            except Exception:
                continue

        candidates: Dict[str, Cursor] = {}
        overload_name = ""
        for overload_ref in overload_refs:
            overload_name = overload_name or overload_ref.spelling or ""
            try:
                count = int(conf.lib.clang_getNumOverloadedDecls(overload_ref))
            except Exception:
                continue
            for index in range(count):
                try:
                    candidate = conf.lib.clang_getOverloadedDecl(
                        overload_ref, index
                    )
                    candidate_usr = get_usr(candidate)
                except Exception:
                    continue
                if (
                    not candidate_usr
                    or not cursor_accepts_argument_count(candidate, argument_count)
                ):
                    continue
                candidates[candidate_usr] = candidate

        log_edge_debug(
            "[EDGE_TRACE] dependent_call "
            f"name={overload_name or '<unknown>'} args={argument_count} "
            f"overload_refs={len(overload_refs)} matches={len(candidates)}"
        )
        if len(candidates) != 1:
            return None
        candidate_usr, candidate = next(iter(candidates.items()))
        location = call_cursor.location
        resolution = {
            "method": "libclang_overload_set",
            "confidence": "high",
            "overload_name": overload_name,
            "argument_count": argument_count,
            "unique_target_usr": candidate_usr,
        }
        if location and location.file:
            resolution.update({
                "evidence_file": os.path.abspath(str(location.file)),
                "evidence_line": location.line,
                "evidence_column": location.column,
            })
        return candidate, resolution

    def resolve_dependent_receiver_call(
        call_cursor: Cursor,
    ) -> List[Tuple[Cursor, Dict[str, Any]]]:
        """Recover a method family hidden behind a dependent receiver type.

        Some calls, such as ``underlying_.parse(ctx)``, expose neither a
        referenced declaration nor an overload set.  In that shape libclang
        still identifies the receiver type and its declaration.  This also
        applies to typed temporaries such as ``reader<T>(...).parse(...)``.
        Resolve only method families named by classes inside that type, and
        retain one representative source definition per family for fields.  A
        non-field receiver must have a unique arity-compatible target in the
        family.  The edge is explicitly medium-confidence because template
        instantiation may pick a narrower specialization at compile time.
        """
        try:
            argument_count = len(list(call_cursor.get_arguments()))
            call_children = list(call_cursor.get_children())
        except Exception:
            return []
        if not call_children:
            return []

        callee_expr = call_children[0]
        try:
            while callee_expr.kind == CursorKind.UNEXPOSED_EXPR:
                children = list(callee_expr.get_children())
                if len(children) != 1:
                    return []
                callee_expr = children[0]
            if callee_expr.kind != CursorKind.MEMBER_REF_EXPR:
                return []
            member_children = list(callee_expr.get_children())
        except Exception:
            return []
        if not member_children:
            return []

        # An overload set attached to the member name makes the target
        # ambiguous.  Overload sets inside the receiver expression (for
        # example ``std::move`` in a temporary's constructor arguments) do not
        # describe the outer member and must not suppress its recovery.
        member_target_stack = list(member_children[1:])
        while member_target_stack:
            cursor = member_target_stack.pop()
            try:
                if cursor.kind == CursorKind.OVERLOADED_DECL_REF:
                    return []
                member_target_stack.extend(cursor.get_children())
            except Exception:
                continue

        receiver_expr = member_children[0]
        try:
            while receiver_expr.kind == CursorKind.UNEXPOSED_EXPR:
                children = list(receiver_expr.get_children())
                if len(children) != 1:
                    return []
                receiver_expr = children[0]
            receiver_decl = receiver_expr.referenced
            receiver_type = receiver_expr.type
            receiver_type_decl = receiver_type.get_declaration()
        except Exception:
            return []

        alias_kinds = {
            CursorKind.TYPEDEF_DECL,
            CursorKind.TYPE_ALIAS_DECL,
        }
        alias_template_kind = getattr(CursorKind, "TYPE_ALIAS_TEMPLATE_DECL", None)
        if alias_template_kind is not None:
            alias_kinds.add(alias_template_kind)
        is_field_receiver = bool(
            receiver_decl and receiver_decl.kind == CursorKind.FIELD_DECL
        )
        if (
            not receiver_type_decl
            or (
                receiver_type_decl.kind not in alias_kinds
                and receiver_type_decl.kind not in CLASS_KINDS
            )
        ):
            return []

        try:
            member_tokens = [token.spelling for token in callee_expr.get_tokens()]
        except Exception:
            return []
        separators = [
            index
            for index, token in enumerate(member_tokens)
            if token in {".", "->"}
        ]
        if not separators:
            return []
        method_name = ""
        for token in member_tokens[separators[-1] + 1 :]:
            if token == "template":
                continue
            if re.fullmatch(r"[A-Za-z_]\w*", token):
                method_name = token
                break
        if not method_name:
            return []

        owner_families: Set[str] = set()
        alias_stack = [receiver_type_decl]
        visited = 0
        while alias_stack and visited < 256:
            cursor = alias_stack.pop()
            visited += 1
            try:
                target = cursor.referenced if cursor.kind == CursorKind.TEMPLATE_REF else cursor
                if target and target.kind in CLASS_KINDS:
                    family = get_fully_qualified_name(target)
                    if family:
                        owner_families.add(family)
                alias_stack.extend(cursor.get_children())
            except Exception:
                continue
        if not owner_families:
            return []

        grouped: Dict[str, List[FunctionInfo]] = {}
        for candidate in functions:
            if (
                not candidate.is_method
                or candidate.short_name != method_name
                or not candidate.owner_class_id
            ):
                continue
            owner_name = candidate.fq_name.rsplit("::", 1)[0]
            if owner_name not in owner_families:
                continue
            candidate_cursor = function_cursor_map.get(id(candidate))
            if (
                not candidate_cursor
                or not cursor_accepts_argument_count(
                    candidate_cursor, argument_count
                )
            ):
                continue
            grouped.setdefault(owner_name, []).append(candidate)

        def representative_score(candidate: FunctionInfo) -> Tuple[int, int, int, str]:
            owner = class_map.get(candidate.owner_class_id)
            owner_span = (
                max(0, owner.def_end_line - owner.def_line)
                if owner is not None
                else 0
            )
            body_span = max(0, candidate.def_end_line - candidate.def_start_line)
            return (owner_span, body_span, -candidate.def_start_line, candidate.id)

        resolved: List[Tuple[Cursor, Dict[str, Any]]] = []
        location = call_cursor.location
        for family, candidates in sorted(grouped.items()):
            if not is_field_receiver and len(candidates) != 1:
                continue
            representative = max(candidates, key=representative_score)
            target_cursor = function_cursor_map.get(id(representative))
            if not target_cursor:
                continue
            target_argument_bounds = cursor_argument_bounds(target_cursor)
            resolution: Dict[str, Any] = {
                "method": (
                    "receiver_type_method_family"
                    if is_field_receiver
                    else "typed_receiver_method_family"
                ),
                "confidence": "medium",
                "receiver": (
                    receiver_decl.spelling
                    if is_field_receiver
                    else receiver_type.spelling
                ) or "",
                "receiver_type": receiver_type.spelling or "",
                "method_name": method_name,
                "argument_count": argument_count,
                "target_family": family,
                "family_candidate_count": len(candidates),
                "representative_target_usr": representative.id,
            }
            if target_argument_bounds:
                resolution.update({
                    "target_required_argument_count": target_argument_bounds[0],
                    "target_parameter_count": target_argument_bounds[1],
                })
            try:
                canonical = receiver_type.get_canonical().spelling
            except Exception:
                canonical = ""
            if canonical:
                resolution["canonical_receiver_type"] = canonical
            if location and location.file:
                resolution.update({
                    "evidence_file": os.path.abspath(str(location.file)),
                    "evidence_line": location.line,
                    "evidence_column": location.column,
                })
            resolved.append((target_cursor, resolution))

        log_edge_debug(
            "[EDGE_TRACE] dependent_receiver_call "
            f"receiver={((receiver_decl.spelling if is_field_receiver else receiver_type.spelling) or '<unknown>')} "
            f"type={receiver_type.spelling or '<unknown>'} "
            f"method={method_name} args={argument_count} "
            f"families={len(owner_families)} matches={len(resolved)}"
        )
        return resolved

    
    cxx_construct_kind = getattr(CursorKind, "CXX_CONSTRUCT_EXPR", None)
    cxx_operator_call_kind = getattr(CursorKind, "CXX_OPERATOR_CALL_EXPR", None)

    def scan_function_body(info: FunctionInfo, root: Cursor):
        stack: List[Cursor] = [root]
        while stack:
            node = stack.pop()
            kind = None
            try:
                kind = node.kind
                is_call = False
                if kind == CursorKind.CALL_EXPR:
                    is_call = True
                elif cxx_construct_kind is not None and kind == cxx_construct_kind:
                    is_call = True
                elif (
                    cxx_operator_call_kind is not None
                    and kind == cxx_operator_call_kind
                ):
                    is_call = True

                if is_call and info:
                    try:
                        callee = node.referenced
                    except Exception:
                        callee = None
                    if callee:
                        callee, call_resolution = resolve_indexed_call_target(callee)
                        callee_usr = get_usr(callee)
                        callee_name = get_fully_qualified_name(callee) or callee.spelling or callee.displayname or "unknown"
                        should_add_edge = False
                        skip_reason = None
                        
                        if callee.location and callee.location.file:
                            callee_path = os.path.abspath(str(callee.location.file))
                            if is_project_file(callee_path):
                                should_add_edge = True
                            else:
                                skip_reason = f"outside_project ({callee_path})"
                        elif callee_usr:
                            if callee_usr in func_map:
                                should_add_edge = True
                            else:
                                skip_reason = f"usr_not_in_map ({callee_usr[:50]}...)"
                        else:
                            skip_reason = "no_location_no_usr"
                        
                        if should_add_edge:
                            if callee_usr and callee_usr not in info.called_usrs:
                                info.called_usrs.append(callee_usr)
                            loc = node.location
                            loc_info = (
                                f"{loc.file}:{loc.line}"
                                if loc and loc.file
                                else "unknown"
                            )
                            log_edge_debug(
                                f"[EDGE_DEBUG] call @ {loc_info} from {info.fq_name} -> {callee_name}"
                            )
                            append_edge(
                                "call",
                                info.id,
                                info.fq_name,
                                callee,
                                node,
                                resolution_data=call_resolution,
                                source_file=info.def_file,
                                source_definition_line=info.def_start_line,
                            )
                        elif skip_reason:
                            if os.environ.get("EDGE_DEBUG") == "1":
                                log_edge_debug(
                                    f"[EDGE_SKIP] call from {info.fq_name} -> {callee_name}: {skip_reason}"
                                )
                    else:
                        resolved = resolve_dependent_call(node)
                        resolutions = [resolved] if resolved else []
                        if not resolutions:
                            resolutions = resolve_dependent_receiver_call(node)
                        for target, resolution in resolutions:
                            target_usr = get_usr(target)
                            if target_usr and target_usr not in info.called_usrs:
                                info.called_usrs.append(target_usr)
                            append_edge(
                                "dependent_call",
                                info.id,
                                info.fq_name,
                                target,
                                node,
                                resolution_data=resolution,
                                source_file=info.def_file,
                                source_definition_line=info.def_start_line,
                            )

                dependency_type = None
                if kind == CursorKind.TEMPLATE_REF:
                    dependency_type = "template_ref"
                elif kind == CursorKind.DECL_REF_EXPR:
                    dependency_type = "decl_ref"
                elif kind == CursorKind.TYPE_REF:
                    dependency_type = "type_ref"
                elif kind == CursorKind.MEMBER_REF_EXPR:
                    dependency_type = "member_ref"
                elif kind in (
                    CursorKind.UNEXPOSED_EXPR,
                    CursorKind.OVERLOADED_DECL_REF,
                    CursorKind.UNEXPOSED_DECL,
                    CursorKind.UNEXPOSED_ATTR,
                ):
                    dependency_type = "unexposed_ref"

                if (
                    dependency_type
                    and dependency_type in ENABLED_EDGE_TYPES
                    and info
                ):
                    inst_data = None
                    if dependency_type == "template_ref":
                        inst_data = _extract_template_instantiation_info(node)
                    append_edge(
                        dependency_type,
                        info.id,
                        info.fq_name,
                        node.referenced,
                        node,
                        inst_data,
                        source_file=info.def_file,
                        source_definition_line=info.def_start_line,
                    )

                if kind == CursorKind.STRING_LITERAL and info:
                    try:
                        tokens = list(node.get_tokens())
                        text = (
                            "".join(t.spelling for t in tokens)
                            .strip('"')
                            .strip("'")
                        )
                        if len(text) > 5:
                            info.log_strings.append(text)
                    except Exception:
                        pass

                children = list(node.get_children())
                if kind in (
                    CursorKind.UNEXPOSED_EXPR,
                    CursorKind.OVERLOADED_DECL_REF,
                    CursorKind.UNEXPOSED_DECL,
                    CursorKind.UNEXPOSED_ATTR,
                ):
                    stack.extend(children)
                else:
                    stack.extend(reversed(children))
            except Exception as exc:
                log_edge_debug(
                    f"[EDGE_ERROR] scan_function_body {info.id} "
                    f"kind={kind}: {type(exc).__name__}: {exc}"
                )
                try:
                    children = list(node.get_children())
                except Exception:
                    continue
                if kind in (
                    CursorKind.UNEXPOSED_EXPR,
                    CursorKind.OVERLOADED_DECL_REF,
                    CursorKind.UNEXPOSED_DECL,
                    CursorKind.UNEXPOSED_ATTR,
                ):
                    stack.extend(children)
                else:
                    stack.extend(reversed(children))

    for function, cursor in func_cursors:
        for child in cursor.get_children():
            scan_function_body(function, child)

    func_regions: Dict[str, List[tuple[int, int, FunctionInfo]]] = {}
    for f in functions:
        func_regions.setdefault(f.def_file, []).append(
            (f.def_start_line, f.def_end_line, f)
        )

    def find_functions_by_loc(file: str, line: int) -> List[FunctionInfo]:
        return [
            function
            for start, end, function in func_regions.get(file, [])
            if start <= line <= end
        ]
        
    for mu in macro_uses:
        for info in find_functions_by_loc(mu.file, mu.line):
            if mu.name not in info.macros_used:
                info.macros_used.append(mu.name)

    call_edges = list(edge_records.values())
    incoming_counts = Counter(edge["callee"] for edge in call_edges)
    source_cache: Dict[str, List[str]] = {}
    
    for f in functions:
        f.caller_count = incoming_counts.get(f.id, 0)
        f.code = _extract_code_snippet(f.def_file, f.def_start_line, f.def_end_line, source_cache)
        if f.code:
            f.content_hash = hashlib.sha256(f.code.encode("utf-8")).hexdigest()

    for c in classes:
        c.code = _extract_code_snippet(c.def_file, c.def_line, c.def_end_line, source_cache)
        if c.code:
            c.content_hash = hashlib.sha256(c.code.encode("utf-8")).hexdigest()

    log_edge_debug(
        f"[EDGE_SUMMARY] {file_path} -> {dict(edge_stats)}"
    )
    
    log_template_debug(f"[TEMPLATE_INST] RETURN from {file_path}: count={len(template_instantiations)}")

    return {
        "functions": [asdict(f) for f in functions],
        "classes": [asdict(c) for c in classes],
        "macro_defs": [asdict(m) for m in macro_defs],
        "macro_uses": [asdict(m) for m in macro_uses],
        "call_edges": call_edges,
        "template_instantiations": template_instantiations,
    }



HEADER_SUFFIXES = {".h", ".hpp", ".hh", ".hxx", ".h++"}
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".c++", ".C"}
MODULE_SUFFIXES = {".cppm", ".ixx", ".mpp"}
TU_SUFFIXES = SOURCE_SUFFIXES | MODULE_SUFFIXES
_EXTRA_TU_STRESS_DIRECTORIES = {"compiletimeperftests"}
UNSUPPORTED_FLAGS = {
    "-Wno-stringop-overflow",
    "-Wno-format-truncation",
    "-Wno-alloca",
    "-Wcatch-value",
}
UNSUPPORTED_FLAG_PREFIXES = (
    "-Werror=catch-value",
    "-Wno-catch-value",
    "-ffile-prefix-map",  
)


def _collect_header_files(project_root: str, extra_dirs: Optional[List[str]] = None) -> List[Path]:
    """
    1) Use provided extra dirs.
    2) Default to include/, src/, tests/abi/include.
    3) Walk repo to find headers not under common third-party/test directories.
    """
    candidate_dirs: Set[Path] = set()
    extra = extra_dirs or []
    for d in extra:
        candidate_dirs.add(Path(d))

    defaults = [
        Path(project_root) / "include",
        Path(project_root) / "src",
        Path(project_root) / "source",
        Path(project_root) / "tests" / "abi" / "include",
    ]
    for d in defaults:
        candidate_dirs.add(d)

    headers: Set[Path] = set()
    def add_dir(dir_path: Path) -> None:
        if not dir_path.exists():
            return
        for path in dir_path.rglob("*"):
            if path.suffix.lower() in HEADER_SUFFIXES and path.is_file():
                headers.add(path)

    for candidate in candidate_dirs:
        add_dir(candidate)

    repo_root = Path(project_root)
    third_party_dirs = {"thirdparty", "third_party", "gtest", "googletest"}
    walk_root = str(repo_root.resolve())
    for dirpath, dirnames, filenames in os.walk(walk_root):
        _prune_lib_llvm_walk(dirpath, dirnames, walk_root)
        for fn in filenames:
            path = Path(dirpath) / fn
            if not path.is_file():
                continue
            if path.suffix.lower() not in HEADER_SUFFIXES:
                continue
            rel_parts = {p.lower() for p in path.relative_to(repo_root).parts}
            if rel_parts & third_party_dirs:
                continue
            if "tests" in rel_parts and "abi" in rel_parts and "include" in rel_parts:
                continue
            headers.add(path)
    pass

    return sorted(headers)


def _collect_extra_translation_units(
    project_root: str, known_files: set[str], extra_dirs: Optional[List[str]] = None, include_tests: bool = True
) -> List[Path]:
    """
    Collect extra translation units (source files) to index.
    
    Args:
        project_root: Project root directory
        known_files: Set of already known files (from compile_commands.json)
        extra_dirs: Optional list of additional directories to search
        include_tests: Whether to include test files (default: True)
    """
    search_dirs: List[Path] = []
    if extra_dirs:
        search_dirs = [Path(d) for d in extra_dirs]
    else:
        for name in ("src", "source", "modules", "tests", "test"):
            candidate = Path(project_root) / name
            if candidate.exists():
                search_dirs.append(candidate)
    if not search_dirs:
        search_dirs = [Path(project_root)]

    extra: List[Path] = []
    known_abs = {os.path.abspath(p) for p in known_files}

    root_prune = str(Path(project_root).resolve())
    for base in search_dirs:
        if not base.exists():
            continue
        base_s = str(base.resolve())
        for dirpath, dirnames, filenames in os.walk(base_s):
            _prune_lib_llvm_walk(dirpath, dirnames, root_prune)
            # These sources recursively include the real test suite 10x/100x to
            # benchmark compiler speed. They contain no unique implementation
            # and can turn one structural-index task into several CPU-hours.
            dirnames[:] = [
                dirname
                for dirname in dirnames
                if dirname.lower() not in _EXTRA_TU_STRESS_DIRECTORIES
            ]
            if any(
                part.lower() in _EXTRA_TU_STRESS_DIRECTORIES
                for part in Path(dirpath).parts
            ):
                continue
            for fn in filenames:
                path = Path(dirpath) / fn
                if not path.is_file():
                    continue
                if path.suffix.lower() not in TU_SUFFIXES:
                    continue
                abs_path = os.path.abspath(str(path))
                if abs_path in known_abs:
                    continue
                if not include_tests and _is_test_path(abs_path):
                    continue
                extra.append(path)
    return extra


def _is_generated_pch_include(path: str) -> bool:
    """Identify cotire force-includes whose compiler-specific .gch is unsafe on host."""
    candidate = Path(path)
    if candidate.suffix.lower() in {".gch", ".pch"}:
        return True
    normalized_parts = {part.lower() for part in candidate.parts}
    return (
        "cotire" in normalized_parts
        and "_prefix." in candidate.name.lower()
        and Path(str(candidate) + ".gch").is_file()
    )


def _sanitize_args(
    args: Optional[List[str]], *, input_file: Optional[str] = None
) -> List[str]:
    if not args:
        return []
    sanitized: List[str] = []
    skip_next = False
    normalized_input = os.path.normpath(input_file) if input_file else None
    compiler_names = {"c++", "cc", "g++", "gcc", "clang", "clang++"}
    for position, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg in ("-c", "-S"):
            continue
        if arg == "-Winvalid-pch":
            continue
        if (
            arg == "-include"
            and position + 1 < len(args)
            and _is_generated_pch_include(args[position + 1])
        ):
            skip_next = True
            continue
        if arg == "-o":
            skip_next = True
            continue
        if arg.startswith("-o"):
            continue
        if position == 0:
            compiler_name = Path(arg).name
            compiler_base = compiler_name.rsplit("-", 1)[0]
            if compiler_name in compiler_names or compiler_base in compiler_names:
                continue
        normalized_flag = arg.split("=", 1)[0]
        if normalized_flag in UNSUPPORTED_FLAGS:
            continue
        if any(normalized_flag.startswith(prefix) for prefix in UNSUPPORTED_FLAG_PREFIXES):
            continue
        if (
            normalized_input
            and not arg.startswith("-")
            and os.path.normpath(arg) == normalized_input
        ):
            continue
        if arg.endswith(tuple(TU_SUFFIXES)) and not arg.startswith("-"):
            continue
        if arg.endswith(tuple(SOURCE_SUFFIXES)) and not arg.startswith("-"):
            continue
        sanitized.append(arg)
    return sanitized


_CXX_SOURCE_SUFFIXES = frozenset({".cc", ".cpp", ".cxx", ".c++", ".C", ".cppm", ".ixx", ".mpp"})


def _append_indexer_isystem_flags(sanitized: List[str], res_dir: Optional[str]) -> None:
    """与 compile_commands 条目一致的系统头搜索路径（供 TU / 独立头文件解析）。"""
    if res_dir:
        sanitized.append(f"-isystem{res_dir}")
    for cpp_ver in ["12", "11", "10", "9"]:
        cpp_path = f"/usr/include/c++/{cpp_ver}"
        if os.path.exists(cpp_path):
            sanitized.append(f"-isystem{cpp_path}")
            break
    if os.path.exists("/usr/include"):
        sanitized.append("-isystem/usr/include")
    arch_path = "/usr/include/x86_64-linux-gnu"
    if os.path.exists(arch_path):
        sanitized.append(f"-isystem{arch_path}")


def _header_parse_mode_and_sample(
    comp_map: Dict[str, Tuple[str, List[str]]],
) -> Tuple[str, List[str], Optional[str], int, int]:
    """
    独立索引 .h 时 Clang 的 -x 模式与一条代表性 compile 命令。

    C 工程（如 zstd）若用 ``-xc++-header`` 解析 C 头文件，易出现 TranslationUnitLoadError。
    """
    n_c = sum(1 for s in comp_map if s.lower().endswith(".c"))
    n_cxx = sum(1 for s in comp_map if Path(s).suffix in _CXX_SOURCE_SUFFIXES)
    use_c = n_c >= n_cxx
    mode = "-xc-header" if use_c else "-xc++-header"
    for src, (cwd, args) in comp_map.items():
        is_c = src.lower().endswith(".c")
        is_cxx = Path(src).suffix in _CXX_SOURCE_SUFFIXES
        if use_c and is_c:
            return mode, list(args), cwd, n_c, n_cxx
        if not use_c and is_cxx:
            return mode, list(args), cwd, n_c, n_cxx
    first_src, (fcwd, fargs) = next(iter(comp_map.items()))
    return mode, list(fargs), fcwd, n_c, n_cxx


def _repo_host_parent_prefix() -> str:
    """``REPO_DIR`` 绝对路径 + 分隔符；用于从 compile_commands 路径反推宿主机实例根。"""
    try:
        from INTENT2FIX.env_paths import repo_instances_parent

        return str(repo_instances_parent().resolve()) + os.sep
    except Exception:
        return ""


def get_project_root(build_dir: str, comp_map: Dict[str, Tuple[str, List[str]]]) -> str:
    env_override = os.environ.get("PROJECT_ROOT_OVERRIDE")
    if env_override:
        return os.path.realpath(env_override)

    instance_id = os.environ.get("SWEBENCH_INSTANCE_NAME")
    host_pattern = _repo_host_parent_prefix()

    if instance_id and host_pattern:
        host_candidate = os.path.join(host_pattern.rstrip(os.sep), instance_id)
        if os.path.exists(host_candidate):
            return os.path.realpath(host_candidate)

    if comp_map and host_pattern:
        all_src_paths = [os.path.abspath(p) for p in comp_map.keys()]

        for src_path in all_src_paths:
            if host_pattern in src_path:
                idx = src_path.find(host_pattern)
                if idx != -1:
                    rest = src_path[idx + len(host_pattern) :]
                    if rest:
                        instance_part = rest.split(os.sep)[0]
                        if instance_part.startswith("swe-bench_"):
                            host_root = os.path.join(host_pattern.rstrip(os.sep), instance_part)
                            if os.path.exists(host_root):
                                return os.path.realpath(host_root)

        for src_path in all_src_paths:
            parts = src_path.split(os.sep)
            for i, part in enumerate(parts):
                if part.startswith("swe-bench_"):
                    potential_instance = part
                    host_root = os.path.join(host_pattern.rstrip(os.sep), potential_instance)
                    if os.path.exists(host_root):
                        return os.path.realpath(host_root)

    if os.path.exists("/home"):
        excludes = {"ubuntu", "coder", "node", "root", "swe-bench"}
        try:
            for item in os.listdir("/home"):
                path = os.path.join("/home", item)
                if os.path.isdir(path) and item not in excludes:
                    if (os.path.exists(os.path.join(path, ".git")) or 
                        os.path.exists(os.path.join(path, "CMakeLists.txt"))):
                        if instance_id and host_pattern:
                            host_candidate = os.path.join(host_pattern.rstrip(os.sep), instance_id)
                            if os.path.exists(host_candidate):
                                return os.path.realpath(host_candidate)
                        return os.path.realpath(path)
        except Exception:
            pass
            
    all_src_paths = [os.path.abspath(p) for p in comp_map.keys()] if comp_map else []
    project_root = os.path.commonpath(all_src_paths) if all_src_paths else os.path.abspath(build_dir)
    
    climb_names = {"src", "source", "sources", "test", "tests", "catch2"}
    while True:
        basename = os.path.basename(project_root).lower()
        if basename in climb_names:
            parent = os.path.dirname(project_root)
            if not parent or parent == project_root:
                break
            project_root = parent
            continue
        break
    return os.path.realpath(project_root)


def _parse_index_env_csv_prefixes(raw: Optional[str]) -> List[str]:
    """CODE_INTENT_INDEX_PREFIXES: comma-separated repo-relative prefixes, e.g. ``src,lib``."""
    if not raw or not raw.strip():
        return []
    out: List[str] = []
    for part in raw.split(","):
        p = part.strip().replace("\\", "/").strip("/")
        if p:
            out.append(p)
    return out


def _parse_index_env_csv_substrings(raw: Optional[str]) -> List[str]:
    """CODE_INTENT_INDEX_EXCLUDE_PATH_SUBSTR: comma-separated substrings matched against normalized paths."""
    if not raw or not raw.strip():
        return []
    return [x.strip().replace("\\", "/") for x in raw.split(",") if x.strip()]


def _norm_index_path(path: str) -> str:
    return os.path.normpath(path).replace("\\", "/")


def _prune_lib_llvm_walk(dirpath: str, dirnames: list[str], repo_root: str) -> None:
    """
    遍历时不要进入 ``lib/llvm``（LLVM 子模块），否则 ``rglob``/``os.walk`` 会扫上百万文件。
    通过 ``dirnames`` 原地修改剪枝。
    """
    try:
        rel = os.path.relpath(dirpath, repo_root)
    except ValueError:
        return
    parts = [p.lower() for p in rel.replace("\\", "/").split("/") if p and p not in (".",)]
    if len(parts) >= 2 and parts[0] == "lib" and parts[1] == "llvm":
        dirnames[:] = []
        return
    if len(parts) == 1 and parts[0] == "lib":
        dirnames[:] = [d for d in dirnames if d.lower() != "llvm"]


def _index_path_in_scope(
    project_root: str,
    abs_path: str,
    allow_prefixes: List[str],
    exclude_substrings: List[str],
) -> bool:
    """If ``allow_prefixes`` is non-empty, only paths under project_root whose relative path matches a prefix are kept."""
    try:
        norm = _norm_index_path(os.path.realpath(abs_path))
        root = _norm_index_path(os.path.realpath(project_root))
    except OSError:
        return False
    for sub in exclude_substrings:
        if sub in norm:
            return False
    if not allow_prefixes:
        return True
    try:
        rel = os.path.relpath(norm, root).replace("\\", "/")
    except ValueError:
        return False
    if rel.startswith("..") or rel == ".":
        return False
    for p in allow_prefixes:
        if rel == p or rel.startswith(p + "/"):
            return True
    return False


def _filter_comp_map_for_index(
    comp_map: Dict[str, Tuple[str, List[str]]],
    project_root: str,
    allow_prefixes: List[str],
    exclude_substrings: List[str],
) -> Dict[str, Tuple[str, List[str]]]:
    if not allow_prefixes and not exclude_substrings:
        return comp_map
    out: Dict[str, Tuple[str, List[str]]] = {}
    for src, v in comp_map.items():
        if _index_path_in_scope(project_root, src, allow_prefixes, exclude_substrings):
            out[src] = v
    if not out and comp_map:
        print(
            "[Indexer] WARN: path filter removed all compile_commands entries; "
            "using unfiltered compile_commands."
        )
        return comp_map
    if len(out) < len(comp_map):
        print(
            f"[Indexer] compile_commands: {len(comp_map)} -> {len(out)} entries "
            f"(prefixes={allow_prefixes or 'any'}, exclude_substr={exclude_substrings or 'none'})"
        )
    return out


def index_repo(
    build_dir: str,
    output_dir: str,
    clang_library_path: Optional[str] = None,
    extra_header_dirs: Optional[List[str]] = None,
    extra_tu_dirs: Optional[List[str]] = None,
    max_workers: Optional[int] = None,
) -> None:
    """
    Optional pruning via environment (after ``project_root`` is resolved):

    - ``CODE_INTENT_INDEX_PREFIXES``: comma-separated repo-relative prefixes; only those paths are indexed.
    - ``CODE_INTENT_INDEX_EXCLUDE_PATH_SUBSTR``: comma-separated substrings; paths containing any are dropped
      (e.g. ``.git/modules,/submodules/`` to skip git submodules).
    - ``CODE_INTENT_INDEX_NO_STANDALONE_HEADERS``: if truthy, skip standalone ``.h`` TU passes (saves a full repo header walk + parses).
    - ``CODE_INTENT_INDEX_NO_EXTRA_TU``: if truthy, skip ``_collect_extra_translation_units``.
    - ``CODE_INTENT_INDEX_EXTRA_TU_NO_TESTS``: if truthy, pass ``include_tests=False`` to extra-TU discovery.
    - ``CODE_INTENT_INDEX_MAX_WORKERS``: parser worker count when ``max_workers`` is omitted.
    - ``CODE_INTENT_INDEX_EXECUTOR``: ``process`` (default) or ``thread``. Process mode avoids
      serializing Python AST traversal behind one interpreter lock and contains libclang's
      process-wide working-directory side effect.
    """
    try:
        init_clang(clang_library_path)
        print(f"[Indexer] libclang initialized (path={clang_library_path})")
    except Exception as e:
        print(f"[Indexer] libclang init failed: {e}")
        return
    
    try:
        comp_map = load_compile_commands(build_dir)
    except Exception as e:
        print(f"Failed to load compile database: {e}")
        return

    project_root = get_project_root(build_dir, comp_map)
    print(f"[Indexer] Project root: {project_root}")

    env_prefixes = _parse_index_env_csv_prefixes(os.environ.get("CODE_INTENT_INDEX_PREFIXES"))
    env_exclude = _parse_index_env_csv_substrings(os.environ.get("CODE_INTENT_INDEX_EXCLUDE_PATH_SUBSTR"))
    no_standalone_h = os.environ.get("CODE_INTENT_INDEX_NO_STANDALONE_HEADERS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    no_extra_tu = os.environ.get("CODE_INTENT_INDEX_NO_EXTRA_TU", "").strip().lower() in ("1", "true", "yes")
    extra_tu_include_tests = os.environ.get("CODE_INTENT_INDEX_EXTRA_TU_NO_TESTS", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    )
    executor_mode = (os.environ.get("CODE_INTENT_INDEX_EXECUTOR") or "process").strip().lower()
    if executor_mode not in {"thread", "process"}:
        print(
            f"[Indexer] WARN: invalid CODE_INTENT_INDEX_EXECUTOR={executor_mode!r}; "
            "using 'process'"
        )
        executor_mode = "process"
    if max_workers is None:
        mw_env = (os.environ.get("CODE_INTENT_INDEX_MAX_WORKERS") or "").strip()
        if mw_env:
            try:
                max_workers = int(mw_env)
            except ValueError:
                print(f"[Indexer] WARN: invalid CODE_INTENT_INDEX_MAX_WORKERS={mw_env!r}, ignored")

    comp_map = _filter_comp_map_for_index(comp_map, project_root, env_prefixes, env_exclude)
    if no_standalone_h:
        print("[Indexer] CODE_INTENT_INDEX_NO_STANDALONE_HEADERS: standalone .h indexing disabled")
    if no_extra_tu:
        print("[Indexer] CODE_INTENT_INDEX_NO_EXTRA_TU: extra TU discovery disabled")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    entities_dir = out / "entities"
    edges_dir = out / "edges"
    entities_dir.mkdir(parents=True, exist_ok=True)
    edges_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out / f"cache_v{CODE_INDEX_CACHE_SCHEMA_VERSION}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    hash_path = out / "file_hashes.json"

    old_hashes = load_file_hash_cache(hash_path) if hash_path.exists() else {}
    new_hashes = {}

    tasks_to_run = []
    default_args: Optional[List[str]] = None
    default_src: Optional[str] = None
    default_cwd: Optional[str] = None
    for src, (cwd, args) in comp_map.items():
        default_src = src
        default_args = list(args)
        default_cwd = cwd
        break
    sanitized_default_args = _sanitize_args(default_args, input_file=default_src)
    if not sanitized_default_args:
        sanitized_default_args = ["-std=c++17"]

    res_dir = get_clang_resource_dir()
    if res_dir:
        print(f"[Indexer] Injected compiler resource dir: {res_dir}")
    _append_indexer_isystem_flags(sanitized_default_args, res_dir)

    all_files_to_index: List[Tuple[str, List[str], Optional[str]]] = []

    for src, (cwd, args) in comp_map.items():
        sanitized = _sanitize_args(args, input_file=src)
        _append_indexer_isystem_flags(sanitized, res_dir)
        all_files_to_index.append((src, sanitized, cwd))

    header_files: List[Path] = []
    if not no_standalone_h:
        header_files = _collect_header_files(project_root, extra_header_dirs)
        header_files = [
            h
            for h in header_files
            if _index_path_in_scope(project_root, str(h), env_prefixes, env_exclude)
        ]

    if header_files and sanitized_default_args:
        hf_mode, hf_raw, hf_cwd, n_c, n_cxx = _header_parse_mode_and_sample(comp_map)
        hdr = _sanitize_args(hf_raw)
        if not hdr:
            hdr = ["-std=c11"] if hf_mode == "-xc-header" else ["-std=c++17"]
        _append_indexer_isystem_flags(hdr, res_dir)
        hdr.append(hf_mode)
        print(
            f"[Indexer] Standalone .h parsing: {hf_mode} "
            f"(compile_commands: .c={n_c}, cxx={n_cxx}; cwd={hf_cwd or default_cwd})"
        )
        hcwd = hf_cwd or default_cwd or project_root
        for h in header_files:
            all_files_to_index.append((str(h), hdr, hcwd))
    
    known_sources = set(comp_map.keys())
    extra_tus: List[Path] = []
    if not no_extra_tu:
        extra_tus = _collect_extra_translation_units(
            project_root, known_sources, extra_tu_dirs, include_tests=extra_tu_include_tests
        )
        extra_tus = [
            tu
            for tu in extra_tus
            if _index_path_in_scope(project_root, str(tu), env_prefixes, env_exclude)
        ]

    if extra_tus and sanitized_default_args:
        for tu in extra_tus:
            all_files_to_index.append((str(tu), sanitized_default_args, default_cwd or project_root))

    results_to_merge = []
    for fpath, fargs, fcwd in all_files_to_index:
        f_hash = get_file_hash(fpath)
        new_hashes[fpath] = f_hash
        
        rel_fpath = os.path.relpath(fpath, project_root)
        path_id = hashlib.md5(rel_fpath.encode()).hexdigest()
        cache_file = cache_dir / f"{path_id}.json"
        
        if cache_file.exists() and old_hashes.get(fpath) == f_hash:
            try:
                with cache_file.open("r", encoding="utf-8") as f:
                    cached_res = json.load(f)
                    cached_res = _restore_absolute_paths(cached_res, project_root)
                    results_to_merge.append(cached_res)
            except Exception:
                tasks_to_run.append((fpath, fargs, fcwd, project_root, clang_library_path, cache_file))
        else:
            tasks_to_run.append((fpath, fargs, fcwd, project_root, clang_library_path, cache_file))

    # Write the hash inventory before launching workers. A cache blob is still
    # reused only when it exists and its source hash matches, so an interrupted
    # run can safely keep completed blobs without treating unfinished tasks as
    # cached.
    write_file_hash_cache(hash_path, new_hashes)

    if tasks_to_run:
        total_new = len(tasks_to_run)
        cpu_count = os.cpu_count() or 4
        if max_workers:
            workers = max_workers
        else:
            workers = max(cpu_count, int(cpu_count * 1.5))
        print(
            f"[Indexer] Incremental build: {total_new} files to parse "
            f"(total {len(all_files_to_index)}), workers: {workers}, "
            f"executor: {executor_mode} (CPU cores: {cpu_count})"
        )
        
        completed = 0
        if executor_mode == "process":
            executor = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_init_process_worker,
                initargs=(clang_library_path,),
            )
            task_executor = _process_task_executor
        else:
            executor = ThreadPoolExecutor(max_workers=workers)
            task_executor = _task_executor
        with executor:
            from concurrent.futures import as_completed
            futures = {executor.submit(task_executor, task): task for task in tasks_to_run}
            new_results = []
            for future in as_completed(futures):
                completed += 1
                if completed % 10 == 0 or completed == total_new:
                    print(f"[Indexer] Progress: {completed}/{total_new} ({completed*100//total_new}%)")
                try:
                    result = future.result()
                    if executor_mode == "process":
                        cache_file = Path(result) if result else None
                        if cache_file and cache_file.is_file():
                            with cache_file.open("r", encoding="utf-8") as handle:
                                result = _restore_absolute_paths(
                                    json.load(handle), project_root
                                )
                        else:
                            result = empty_result()
                    if result:
                        new_results.append(result)
                except Exception as e:
                    task_info = futures.get(future, "unknown")
                    file_path = task_info[0] if isinstance(task_info, tuple) and len(task_info) > 0 else "unknown"
                    print(f"[Task Error] Task failed: {file_path}")
                    print(f"   Error: {type(e).__name__}: {e}")
                    log_parse_debug(f"[Task Error] Task failed: {file_path}")
                    log_parse_debug(f"   Error type: {type(e).__name__}")
                    log_parse_debug(f"   Error detail: {e}")
                    import traceback
                    log_parse_debug(f"   Traceback:\n{traceback.format_exc()}")
                    new_results.append(empty_result())
            results_to_merge.extend(new_results)
    else:
        print("[Indexer] No file changes, loading from cache.")

    print("[Indexer] Merging results...")
    results_to_merge = [
        _convert_to_relative_paths(result, project_root)
        for result in results_to_merge
        if result
    ]
    global_seen_func = set()
    global_seen_cls = set()
    global_seen_macro = set()
    global_seen_template_inst = set()
    anonymous_owner_ids_cleared = _clear_unindexed_anonymous_owner_ids(
        results_to_merge
    )
    definition_collision_stats = _qualify_multi_definition_functions(
        results_to_merge, project_root
    )
    class_collision_stats = _qualify_multi_definition_classes(
        results_to_merge, project_root
    )
    if class_collision_stats["owner_ids_not_disambiguated"]:
        raise RuntimeError(
            "Class definition collision left ambiguous method owners: "
            f"{class_collision_stats['owner_ids_not_disambiguated']}"
    )
    definition_collision_stats["classes"] = class_collision_stats
    template_target_reconciliation_stats = (
        _reconcile_template_instantiation_targets(
            results_to_merge, project_root
        )
    )
    if template_target_reconciliation_stats[
        "unique_identity_misses_after_reconciliation"
    ]:
        raise RuntimeError(
            "Template-instantiation target reconciliation left uniquely "
            "resolvable definitions: "
            f"{template_target_reconciliation_stats['unique_identity_misses_after_reconciliation']}"
        )
    definition_collision_stats["template_target_reconciliation"] = (
        template_target_reconciliation_stats
    )
    template_instantiation_collision_stats = (
        _qualify_multi_definition_template_instantiations(
            results_to_merge, project_root
        )
    )
    if (
        template_instantiation_collision_stats[
            "edge_rows_not_disambiguated"
        ]
        or template_instantiation_collision_stats[
            "edge_target_disagreements_after_reconciliation"
        ]
    ):
        raise RuntimeError(
            "Template-instantiation collision reconciliation failed: "
            f"unresolved_edges={template_instantiation_collision_stats['edge_rows_not_disambiguated']}, "
            f"target_disagreements={template_instantiation_collision_stats['edge_target_disagreements_after_reconciliation']}."
        )
    definition_collision_stats["template_instantiations"] = (
        template_instantiation_collision_stats
    )
    definition_collision_stats["dependency_endpoint_reconciliation"] = (
        _reconcile_dependency_endpoints(results_to_merge, project_root)
    )
    definition_collision_stats["anonymous_owner_ids_cleared"] = (
        anonymous_owner_ids_cleared
    )
    merged_edges, edge_merge_stats = _deduplicate_edges(results_to_merge)
    all_function_ids = {
        str(function.get("id") or "")
        for result in results_to_merge
        for function in result.get("functions", [])
        if function.get("id")
    }
    source_definition_ids = {
        (
            str(function.get("def_file") or "").replace("\\", "/"),
            str(function.get("fq_name") or ""),
            int(function.get("def_start_line") or 0),
            int(function.get("def_end_line") or 0),
            str(function.get("id") or ""),
        )
        for result in results_to_merge
        for function in result.get("functions", [])
        if function.get("id")
        and function.get("def_file")
        and function.get("fq_name")
    }
    source_definition_spans = {row[:4] for row in source_definition_ids}
    merged_edges, unresolved_call_edges, edge_endpoint_stats = (
        _partition_graph_ready_edges(
            merged_edges,
            all_function_ids,
            source_definition_spans,
            source_definition_ids,
        )
    )
    outgoing_calls: Dict[str, Set[str]] = {}
    incoming_calls: Counter[str] = Counter()
    for edge in merged_edges:
        if edge.get("edge_type", "call") not in {"call", "dependent_call"}:
            continue
        caller = str(edge.get("caller") or "")
        callee = str(edge.get("callee") or "")
        outgoing_calls.setdefault(caller, set()).add(callee)
        incoming_calls[callee] += 1
    for result in results_to_merge:
        for function in result.get("functions", []):
            function_id = str(function.get("id") or "")
            function["called_usrs"] = sorted(outgoing_calls.get(function_id, set()))
            function["caller_count"] = incoming_calls.get(function_id, 0)
    edge_type_counts_final = Counter(
        edge.get("edge_type", "call") for edge in merged_edges
    )

    indexed_project_files = []
    for path in sorted(new_hashes):
        real_path = os.path.realpath(path)
        try:
            relative = os.path.relpath(real_path, project_root)
        except (OSError, ValueError):
            continue
        if relative.startswith("..") or not os.path.isfile(real_path):
            continue
        indexed_project_files.append((real_path, relative.replace("\\", "/")))

    with open(out / "functions.jsonl", "w", encoding="utf-8") as f_func_legacy, \
         open(out / "classes.jsonl", "w", encoding="utf-8") as f_cls_legacy, \
         open(out / "macros.jsonl", "w", encoding="utf-8") as f_macro_legacy, \
         open(out / "preprocessor_directives.jsonl", "w", encoding="utf-8") as f_pp_legacy, \
         open(out / "callgraph.jsonl", "w", encoding="utf-8") as f_call_legacy, \
         open(entities_dir / "files.jsonl", "w", encoding="utf-8") as f_files, \
         open(entities_dir / "functions.jsonl", "w", encoding="utf-8") as f_func, \
         open(entities_dir / "function_templates.jsonl", "w", encoding="utf-8") as f_func_tpl, \
         open(entities_dir / "classes.jsonl", "w", encoding="utf-8") as f_cls, \
         open(entities_dir / "class_templates.jsonl", "w", encoding="utf-8") as f_cls_tpl, \
         open(entities_dir / "template_instantiations.jsonl", "w", encoding="utf-8") as f_template_inst, \
         open(entities_dir / "macros.jsonl", "w", encoding="utf-8") as f_macro, \
         open(entities_dir / "preprocessor_directives.jsonl", "w", encoding="utf-8") as f_pp, \
         open(edges_dir / "edges_call.jsonl", "w", encoding="utf-8") as f_edges_call, \
         open(edges_dir / "edges_unresolved_call.jsonl", "w", encoding="utf-8") as f_edges_unresolved_call, \
         open(edges_dir / "edges_ref.jsonl", "w", encoding="utf-8") as f_edges_ref, \
         open(edges_dir / "edges_template.jsonl", "w", encoding="utf-8") as f_edges_tpl:
        
        seen_files = set()
        total_template_inst_count = 0

        for absolute_path, relative_path in indexed_project_files:
            seen_files.add(relative_path)
            f_files.write(
                json.dumps(
                    {
                        "file": relative_path,
                        "is_test": bool(_is_test_path(relative_path)),
                        "is_third_party": bool(_is_third_party(relative_path)),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            for directive in _extract_preprocessor_directives(
                absolute_path, project_root
            ):
                line = json.dumps(directive, ensure_ascii=False) + "\n"
                f_pp_legacy.write(line)
                f_pp.write(line)

        for res in results_to_merge:
            if not res:
                continue
            for f in res["functions"]:
                if f["id"] not in global_seen_func:
                    global_seen_func.add(f["id"])
                    f_func_legacy.write(json.dumps(f, ensure_ascii=False) + "\n")
                    if f.get("is_template"):
                        f_func_tpl.write(json.dumps(f, ensure_ascii=False) + "\n")
                    else:
                        f_func.write(json.dumps(f, ensure_ascii=False) + "\n")
                    df = f.get("def_file")
                    if df and df not in seen_files:
                        seen_files.add(df)
                        f_files.write(
                            json.dumps(
                                {"file": df, "is_test": bool(_is_test_path(df)), "is_third_party": bool(_is_third_party(df))},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
            for c in res["classes"]:
                if c["id"] not in global_seen_cls:
                    global_seen_cls.add(c["id"])
                    f_cls_legacy.write(json.dumps(c, ensure_ascii=False) + "\n")
                    if (c.get("kind") or "").lower() == "template":
                        f_cls_tpl.write(json.dumps(c, ensure_ascii=False) + "\n")
                    else:
                        f_cls.write(json.dumps(c, ensure_ascii=False) + "\n")
                    df = c.get("def_file")
                    if df and df not in seen_files:
                        seen_files.add(df)
                        f_files.write(
                            json.dumps(
                                {"file": df, "is_test": bool(_is_test_path(df)), "is_third_party": bool(_is_third_party(df))},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
            for m in res["macro_defs"]:
                m_key = f"{m['name']}@{m['file']}:{m['line']}"
                if m_key not in global_seen_macro:
                    global_seen_macro.add(m_key)
                    f_macro_legacy.write(json.dumps(m, ensure_ascii=False) + "\n")
                    f_macro.write(json.dumps(m, ensure_ascii=False) + "\n")
                    df = m.get("file")
                    if df and df not in seen_files:
                        seen_files.add(df)
                        f_files.write(
                            json.dumps(
                                {"file": df, "is_test": bool(_is_test_path(df)), "is_third_party": bool(_is_third_party(df))},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
            inst_list = res.get("template_instantiations", [])
            if inst_list:
                log_template_debug(f"[TEMPLATE_INST] WRITING {len(inst_list)} instantiations from result")
            for inst in inst_list:
                if inst.get("id") in global_seen_template_inst:
                    continue
                global_seen_template_inst.add(inst.get("id"))
                f_template_inst.write(json.dumps(inst, ensure_ascii=False) + "\n")
                total_template_inst_count += 1

        for edge in merged_edges:
            f_call_legacy.write(json.dumps(edge, ensure_ascii=False) + "\n")
            edge_type = edge.get("edge_type", "call")
            if edge_type in {"call", "dependent_call"}:
                f_edges_call.write(json.dumps(edge, ensure_ascii=False) + "\n")
            elif edge_type in {
                "template_ref",
                "decl_ref",
                "member_ref",
                "type_ref",
                "unexposed_ref",
            }:
                f_edges_ref.write(json.dumps(edge, ensure_ascii=False) + "\n")
            else:
                f_edges_tpl.write(json.dumps(edge, ensure_ascii=False) + "\n")
        for edge in unresolved_call_edges:
            f_edges_unresolved_call.write(
                json.dumps(edge, ensure_ascii=False) + "\n"
            )
    
    log_template_debug(f"[TEMPLATE_INST] FINAL SUMMARY: Total written to file: {total_template_inst_count}")
    print(f"📊 [Indexer] Template instantiations written: {total_template_inst_count}")

    _require_nonempty_index(
        indexed_file_count=len(all_files_to_index),
        function_count=len(global_seen_func),
        class_count=len(global_seen_cls),
        macro_count=len(global_seen_macro),
    )

    write_file_hash_cache(hash_path, new_hashes)

    edge_stat_path = out / "edge_type_counts.json"
    with edge_stat_path.open("w", encoding="utf-8") as f_stat:
        json.dump(dict(edge_type_counts_final), f_stat, ensure_ascii=False, indent=2)
    with (out / "edge_merge_stats.json").open("w", encoding="utf-8") as f_stat:
        json.dump(edge_merge_stats, f_stat, ensure_ascii=False, indent=2)
    with (out / "edge_endpoint_stats.json").open("w", encoding="utf-8") as f_stat:
        json.dump(edge_endpoint_stats, f_stat, ensure_ascii=False, indent=2)
    with (out / "definition_collision_stats.json").open(
        "w", encoding="utf-8"
    ) as f_stat:
        json.dump(definition_collision_stats, f_stat, ensure_ascii=False, indent=2)
    print(f"[Indexer] Build done. Edge stats: {dict(edge_type_counts_final)}")
