"""Agent Tools - Query entities from code index"""

from __future__ import annotations

import json
import os
import threading
from fnmatch import fnmatchcase
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional
import logging

from INTENT2FIX.env_paths import index_store_parent_optional

# Import RAG engine for semantic search
try:
    from .search_api import CodeIntentRAG
except ImportError:
    # Fallback for direct execution or testing
    try:
        from search_api import CodeIntentRAG
    except ImportError:
        CodeIntentRAG = None

logger = logging.getLogger(__name__)

_THREAD_CONTEXT = threading.local()

# Index root directory from environment variable INDEX_STORE_DIR
# This is set in .env file and loaded via load_dotenv()

# Entity type mapping: tool parameter -> actual file
ENTITY_TYPE_MAP = {
    "function": "functions",
    "class": "classes",
    "class_template": "class_templates",
    "function_template": "function_templates",
    "macro": "macros",
    "template_instantiation": "template_instantiations",
}

# Entity file mapping
ENTITY_FILES = {
    "functions": "functions.jsonl",
    "classes": "classes.jsonl",
    "class_templates": "class_templates.jsonl",
    "function_templates": "function_templates.jsonl",
    "macros": "macros.jsonl",
    "template_instantiations": "template_instantiations.jsonl",
}


def set_current_instance(instance_name: Optional[str]) -> None:
    """Bind entity/graph queries to an instance for the current thread."""
    if instance_name:
        _THREAD_CONTEXT.instance_name = instance_name


def set_current_index_store(index_store_dir: Optional[str]) -> None:
    """Bind graph queries to an index root for the current worker thread."""
    if index_store_dir:
        _THREAD_CONTEXT.index_store_dir = str(Path(index_store_dir).expanduser().resolve())


def _get_index_root() -> Optional[Path]:
    thread_root = getattr(_THREAD_CONTEXT, "index_store_dir", None)
    if thread_root:
        return Path(thread_root)
    return index_store_parent_optional()


def get_current_instance() -> str:
    """Get current instance name"""
    # 1. Prefer the thread-local instance set by EntityCodeIndex.
    instance = getattr(_THREAD_CONTEXT, "instance_name", None)
    if instance:
        return instance

    # 2. Get from environment variable
    instance = os.environ.get("SWEBENCH_INSTANCE_NAME")
    if instance:
        return instance
    
    # 3. Infer from index directory (find unique instance under index/)
    index_root = _get_index_root()
    if index_root is None:
        raise ValueError(
            "Cannot determine current instance name. Please set environment variable SWEBENCH_INSTANCE_NAME, "
            "or set INDEX_STORE_DIR in .env file."
        )
    index_root = Path(index_root)
    # Convert to absolute path if it's relative
    if not index_root.is_absolute():
        index_root = Path.cwd() / index_root
    
    if index_root.exists():
        instances = [d.name for d in index_root.iterdir() if d.is_dir()]
        if len(instances) == 1:
            return instances[0]
        elif len(instances) > 1:
            # If multiple instances, return the first one (or can raise exception)
            return instances[0]
    
    raise ValueError(
        "Cannot determine current instance name. Please set environment variable SWEBENCH_INSTANCE_NAME, "
        "or ensure there is only one instance directory under index/."
    )


def _get_entities_dir(instance_name: str) -> Path:
    """Get entity directory for the instance"""
    index_root = _get_index_root()
    if index_root is None:
        raise ValueError(
            "INDEX_STORE_DIR environment variable is not set. "
            "Please set it in .env file or as an environment variable."
        )
    index_root = Path(index_root)
    # Convert to absolute path if it's relative
    if not index_root.is_absolute():
        index_root = Path.cwd() / index_root
    
    # Try multiple folder name formats:
    # 1. swe-bench_{instance_name} (common format for SWEBench instances)
    # 2. {instance_name} (direct instance name)
    folder_names = [
        f"swe-bench_{instance_name}",
        instance_name,
    ]
    
    paths = []
    for folder_name in folder_names:
        paths.extend([
            index_root / folder_name / "index_output" / "entities",
            index_root / folder_name / "entities",
            index_root / folder_name / "intent_output" / folder_name / "index_output" / "entities",
            index_root / folder_name / "intent_output" / instance_name / "index_output" / "entities",
        ])
    
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Entity directory for instance '{instance_name}' does not exist. "
        f"INDEX_STORE_DIR: {index_root}, Tried paths: {[str(p) for p in paths]}"
    )


# RAG cache: (resolved index root, instance_name) -> CodeIntentRAG
_RAG_CACHE: Dict[tuple[str, str], Any] = {}


def _get_rag_instance() -> Optional[CodeIntentRAG]:
    """Get or initialize RAG instance for current instance"""
    if CodeIntentRAG is None:
        return None
        
    instance_name = get_current_instance()
    index_root = _get_index_root()
    cache_key = (str(index_root.resolve()) if index_root else "", instance_name)
    if cache_key in _RAG_CACHE:
        return _RAG_CACHE[cache_key]
    
    # Resolve index_output directory (parent of entities/)
    try:
        entities_dir = _get_entities_dir(instance_name)
        index_dir = entities_dir.parent
        rag = CodeIntentRAG(index_dir)
        _RAG_CACHE[cache_key] = rag
        return rag
    except Exception as e:
        logger.error(f"Failed to initialize CodeIntentRAG for {instance_name}: {e}")
        return None


def get_graph_rag() -> Optional[CodeIntentRAG]:
    """Compatibility helper for graph access."""
    return _get_rag_instance()


def _load_entities_from_file(file_path: Path) -> List[Dict[str, Any]]:
    """Load all entities from JSONL file"""
    entities = []
    if not file_path.exists():
        return entities
    
    try:
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entity = json.loads(line)
                    entities.append(entity)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        raise RuntimeError(f"Failed to read entity file: {str(e)}")
    
    return entities


def _matches_file_filter(file_path: str, file_filter: str) -> bool:
    """Match a normalized repo path against a literal path or a glob."""
    path = file_path.replace("\\", "/").lstrip("./").lower()
    pattern = file_filter.replace("\\", "/").lstrip("./").lower()
    if not path or not pattern:
        return False
    if not any(character in pattern for character in "*?["):
        return pattern in path

    patterns = {pattern}
    reduced = pattern
    while "**/" in reduced:
        reduced = reduced.replace("**/", "", 1)
        patterns.add(reduced)
    for candidate in patterns:
        if (
            fnmatchcase(path, candidate)
            or fnmatchcase(path, f"*/{candidate}")
            or PurePosixPath(path).match(candidate)
        ):
            return True
    return False


def _ordered_partial_subsequence(needle: List[str], haystack: List[str]) -> bool:
    position = 0
    for candidate in haystack:
        if position < len(needle) and needle[position] in candidate:
            position += 1
    return position == len(needle)


def _matches_namespace_filter(entity: Dict[str, Any], namespace: str) -> bool:
    """Match qualified namespaces across implementation/macro expansion layers."""
    requested = [part.strip().lower() for part in namespace.split("::") if part.strip()]
    if not requested:
        return True
    indexed = [str(part).lower() for part in entity.get("namespaces", []) if part]
    fq_parts = [
        part.lower()
        for part in str(entity.get("fq_name", "")).split("::")
        if part
    ]
    return _ordered_partial_subsequence(requested, indexed) or _ordered_partial_subsequence(
        requested, fq_parts
    )


def _entity_identity(entity: Dict[str, Any]) -> tuple[Any, ...]:
    file_path = entity.get("def_file") or entity.get("file") or entity.get("trigger_file")
    start = entity.get("def_start_line") or entity.get("def_line") or entity.get("line")
    end = entity.get("def_end_line") or entity.get("end_line") or start
    name = entity.get("fq_name") or entity.get("display_name") or entity.get("name")
    parameters = tuple(entity.get("parameter_types") or entity.get("template_args") or [])
    return (name, str(file_path or "").replace("\\", "/"), start, end, parameters)


def _entity_rank(entity: Dict[str, Any], name: Optional[str]) -> tuple[Any, ...]:
    requested = (name or "").lower()
    short_name = str(entity.get("short_name") or entity.get("name") or "").lower()
    fq_name = str(entity.get("fq_name") or entity.get("display_name") or "").lower()
    if requested and requested in {short_name, fq_name}:
        name_rank = 0
    elif requested and (short_name.startswith(requested) or fq_name.startswith(requested)):
        name_rank = 1
    else:
        name_rank = 2

    file_path = str(
        entity.get("def_file") or entity.get("file") or entity.get("trigger_file") or ""
    ).replace("\\", "/").lower()
    path_parts = set(PurePosixPath(file_path).parts)
    non_production = int(bool(path_parts & {"test", "tests", "testing", "benchmarks"}))
    generated = int("singleheader" in path_parts or "generated" in path_parts)
    start = entity.get("def_start_line") or entity.get("def_line") or entity.get("line") or 0
    end = entity.get("def_end_line") or entity.get("end_line") or start
    declaration_only = int(not isinstance(end, int) or not isinstance(start, int) or end <= start)
    return (name_rank, non_production, generated, declaration_only, file_path, fq_name, start)


def _entity_scope(entity: Dict[str, Any]) -> tuple[str, ...]:
    namespaces = tuple(str(part).lower() for part in entity.get("namespaces", []) if part)
    if namespaces:
        return ("namespace", *namespaces)
    file_path = str(
        entity.get("def_file") or entity.get("file") or entity.get("trigger_file") or ""
    ).replace("\\", "/")
    parts = tuple(part.lower() for part in PurePosixPath(file_path).parts[:2])
    return ("path", *parts)


def _rank_and_diversify_entities(
    entities: List[Dict[str, Any]], name: Optional[str]
) -> List[Dict[str, Any]]:
    """Deduplicate parser variants, then round-robin across independent scopes."""
    unique: Dict[tuple[Any, ...], Dict[str, Any]] = {}
    for entity in entities:
        unique.setdefault(_entity_identity(entity), entity)
    ordered = sorted(unique.values(), key=lambda entity: _entity_rank(entity, name))

    groups: Dict[tuple[str, ...], List[Dict[str, Any]]] = {}
    for entity in ordered:
        groups.setdefault(_entity_scope(entity), []).append(entity)
    diversified: List[Dict[str, Any]] = []
    while groups:
        for scope in list(groups):
            diversified.append(groups[scope].pop(0))
            if not groups[scope]:
                del groups[scope]
    return diversified


def _filter_entities(
    entities: List[Dict[str, Any]],
    entity_type: str,
    name: Optional[str] = None,
    name_exact_match: bool = False,
    file: Optional[str] = None,
    namespace: Optional[str] = None,
    class_name: Optional[str] = None,
    return_type: Optional[str] = None,
    parameter_types: Optional[List[str]] = None,
    base_class: Optional[str] = None,
    kind: Optional[str] = None,
    is_template: Optional[bool] = None,
    is_method: Optional[bool] = None,
    is_static: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Filter entity list"""
    filtered = []
    name_lower = name.lower() if name else None
    class_name_lower = class_name.lower() if class_name else None
    base_class_lower = base_class.lower() if base_class else None
    
    for entity in entities:
        # Filter by name
        if name_lower:
            fq_name = entity.get("fq_name", "").lower()
            short_name = entity.get("short_name", "").lower()
            display_name = entity.get("display_name", "").lower()
            entity_name = entity.get("name", "").lower()  # Macros use 'name' field
            
            if name_exact_match:
                # Exact match: must match exactly with fq_name, short_name, or entity_name
                if (name_lower != fq_name and 
                    name_lower != short_name and 
                    name_lower != entity_name):
                    continue
            else:
                # Partial match: substring match (default behavior)
                if (name_lower not in fq_name and 
                    name_lower not in short_name and 
                    name_lower not in display_name and
                    name_lower not in entity_name):
                    continue
        
        # Filter by file
        if file:
            paths = (
                entity.get("file", ""),
                entity.get("def_file", ""),
                entity.get("trigger_file", ""),
            )
            if not any(_matches_file_filter(str(path), file) for path in paths if path):
                continue
        
        # Filter by namespace (only for function/class)
        if namespace and entity_type in ("function", "class", "class_template", "function_template"):
            if not _matches_namespace_filter(entity, namespace):
                continue
        
        # Filter by class name (only for function)
        if class_name_lower and entity_type in ("function", "function_template"):
            classes = entity.get("classes", [])
            if not any(class_name_lower in cls.lower() for cls in classes):
                continue
        
        # Filter by return type (only for function)
        if return_type and entity_type in ("function", "function_template"):
            if entity.get("return_type") != return_type:
                continue
        
        # Filter by parameter types (only for function)
        if parameter_types and entity_type in ("function", "function_template"):
            entity_params = entity.get("parameter_types", [])
            if len(entity_params) != len(parameter_types):
                continue
            if not all(p1 == p2 for p1, p2 in zip(entity_params, parameter_types)):
                continue
        
        # Filter by base class (only for class)
        if base_class_lower and entity_type in ("class", "class_template"):
            bases = entity.get("bases", [])
            if not any(base_class_lower in base.lower() for base in bases):
                continue
        
        # Filter by kind (only for class)
        if kind and entity_type == "class":
            if entity.get("kind") != kind:
                continue
        
        # Filter by is_template
        if is_template is not None and entity_type in ("function", "class"):
            if entity.get("is_template") != is_template:
                continue
        
        # Filter by is_method (only for function)
        if is_method is not None and entity_type in ("function", "function_template"):
            if entity.get("is_method") != is_method:
                continue
        
        # Filter by is_static (only for function)
        if is_static is not None and entity_type in ("function", "function_template"):
            if entity.get("is_static") != is_static:
                continue
        
        filtered.append(entity)
    
    return filtered


def _get_relative_path(path: Optional[str]) -> Optional[str]:
    """Convert absolute path to relative path if possible"""
    if not path:
        return path

    # Remove container prefixes like /home/fmt/
    for prefix in ["/home/fmt/", "/home/simdjson/", "/home/json/", "/home/cpp_repo/"]:
        if path.startswith(prefix):
            path = path.replace(prefix, "")
            if not os.path.isabs(path):
                return path

    try:
        abs_path = Path(path).resolve()
        cwd = Path.cwd().resolve()
        
        # Try to find common prefix or check if path is under cwd
        if str(abs_path).startswith(str(cwd)):
            return str(abs_path.relative_to(cwd))
            
        # Fallback: if it contains the instance name, try to split from there
        instance = get_current_instance()
        if instance in str(abs_path):
            parts = str(abs_path).split(instance)
            return parts[-1].lstrip("/")
            
        return path
    except Exception:
        return path


def _add_line_numbers(code: str, start_line: int) -> str:
    """Add line numbers to code snippet in the format 'LINE| CONTENT'"""
    if not code:
        return code
    
    # Check if code already has line numbers (format: "   590 | ...")
    import re
    first_line = code.splitlines()[0] if code.splitlines() else ""
    if re.match(r'^\s*\d+\s*\|\s*', first_line):
        # Code already has line numbers, return as-is to avoid double numbering
        return code
    
    lines = code.splitlines()
    numbered_lines = []
    for i, line in enumerate(lines):
        line_num = start_line + i
        numbered_lines.append(f"{line_num:>6} | {line}")
    return "\n".join(numbered_lines)


def _function_skeleton(entity: Dict[str, Any]) -> str:
    """Build a declaration-only function outline from indexed metadata."""
    fq_name = entity.get("fq_name") or entity.get("short_name") or entity.get("name") or "unknown"
    short_name = entity.get("short_name") or fq_name.rsplit("::", 1)[-1]
    classes = entity.get("classes") or []
    fq_parts = fq_name.split("::")
    is_constructor = bool(classes and short_name == classes[-1]) or (
        len(fq_parts) >= 2 and fq_parts[-1] == fq_parts[-2]
    )
    is_destructor = short_name.startswith("~")
    return_type = entity.get("return_type")
    prefix = "" if is_constructor or is_destructor or not return_type else f"{return_type} "
    parameters = entity.get("parameter_types") or entity.get("parameters") or []
    return f"{prefix}{fq_name}({', '.join(parameters)});"


def _class_skeleton(entity: Dict[str, Any]) -> str:
    kind = entity.get("kind", "class")
    name = entity.get("fq_name") or entity.get("short_name") or entity.get("name") or "unknown"
    members = entity.get("member_summary") or []
    lines = [f"{kind} {name} {{"]
    lines.extend(f"    {member}" for member in members[:40])
    if len(members) > 40:
        lines.append(f"    ... and {len(members) - 40} more members")
    lines.append("};")
    return "\n".join(lines)


def _summarize_entity(entity: Dict[str, Any], entity_type: str) -> Dict[str, Any]:
    """Extract key fields for search results with optimized information density"""
    summary = {
        "id": entity.get("id"),
        "type": entity_type,
    }
    
    if entity_type in ("function", "function_template"):
        start_line = entity.get("def_start_line", 1)
        code = entity.get("code", "")
        numbered_code = _add_line_numbers(code, start_line)
        
        summary.update({
            "name": entity.get("short_name"),
            "fq_name": entity.get("fq_name"),
            "file": _get_relative_path(entity.get("def_file")),
            "line_range": [start_line, entity.get("def_end_line")],
            "return_type": entity.get("return_type"),
            "parameters": entity.get("parameter_types"),
            "is_method": entity.get("is_method"),
            "code": numbered_code,
            "skeleton": _function_skeleton(entity),
        })
    elif entity_type in ("class", "class_template"):
        # For classes, construct a skeleton from member summary
        members = entity.get("member_summary", [])
        start_line = entity.get("def_line", 1)
        end_line = entity.get("def_end_line", start_line)
        
        summary.update({
            "name": entity.get("short_name"),
            "fq_name": entity.get("fq_name"),
            "file": _get_relative_path(entity.get("def_file")),
            "line_range": [start_line, end_line],
            "kind": entity.get("kind"),
            "bases": entity.get("bases"),
            "skeleton": _class_skeleton(entity),
        })
    elif entity_type == "macro":
        value = entity.get("value_snippet") or ""
        summary.update({
            "name": entity.get("name"),
            "file": _get_relative_path(entity.get("file")),
            "line": entity.get("line"),
            "value": value,
            "skeleton": f"#define {entity.get('name', 'unknown')} {value}".rstrip(),
        })
    elif entity_type == "template_instantiation":
        display_name = entity.get("display_name") or "unknown"
        summary.update({
            "name": display_name,
            "template_def_id": entity.get("template_def_id"),
            "args": entity.get("template_args"),
            "file": _get_relative_path(entity.get("trigger_file")),
            "line": entity.get("trigger_line"),
            "category": entity.get("category"),
            "skeleton": f"template instantiation {display_name};",
        })
    
    # Remove None values
    return {k: v for k, v in summary.items() if v is not None}


def search_symbol(
    type: str,
    name: Optional[str] = None,
    name_exact_match: bool = False,
    file: Optional[str] = None,
    namespace: Optional[str] = None,
    class_name: Optional[str] = None,
    return_type: Optional[str] = None,
    parameter_types: Optional[List[str]] = None,
    base_class: Optional[str] = None,
    kind: Optional[str] = None,
    is_template: Optional[bool] = None,
    is_method: Optional[bool] = None,
    is_static: Optional[bool] = None,
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    """
    Search code symbols (functions, classes, macros, etc.)
    
    Args:
        type: Entity type, options: 'function', 'class', 'class_template', 'function_template', 'macro'
        name: Symbol name (partial match by default, supports fq_name, short_name)
        name_exact_match: If True, use exact match for name; if False, use partial match (default: False)
        file: File path (literal substring or glob)
        namespace: Qualified namespace; macro/implementation layers may occur between components
        class_name: Class name (partial match, only for function)
        return_type: Return type (exact match, only for function)
        parameter_types: Parameter type list (exact match, only for function)
        base_class: Base class name (partial match, only for class)
        kind: Type: 'class', 'struct', 'template' (only for class)
        is_template: Whether it's a template (only for function/class)
        is_method: Whether it's a method (only for function)
        is_static: Whether it's static (only for function)
        limit: Maximum number of results to return
        offset: Pagination offset
    
    Returns:
        Dictionary containing total, count, offset, limit, entities
    """
    if type not in ENTITY_TYPE_MAP:
        raise ValueError(
            f"Unsupported entity type: {type}. Supported types: {', '.join(ENTITY_TYPE_MAP.keys())}"
        )
    
    # Get current instance
    instance_name = get_current_instance()
    entities_dir = _get_entities_dir(instance_name)
    
    # Map to actual entity type
    entity_type = ENTITY_TYPE_MAP[type]
    file_path = entities_dir / ENTITY_FILES[entity_type]
    
    # Load all entities
    all_entities = _load_entities_from_file(file_path)
    
    # Filter entities
    filtered = _filter_entities(
        all_entities,
        type,
        name=name,
        name_exact_match=name_exact_match,
        file=file,
        namespace=namespace,
        class_name=class_name,
        return_type=return_type,
        parameter_types=parameter_types,
        base_class=base_class,
        kind=kind,
        is_template=is_template,
        is_method=is_method,
        is_static=is_static,
    )
    filtered = _rank_and_diversify_entities(filtered, name)
    
    # Pagination
    total = len(filtered)
    start = offset
    end = offset + limit
    entities = filtered[start:end]
    
    # Summarize entities to reduce token usage
    summarized = [_summarize_entity(e, type) for e in entities]
    
    return {
        "total": total,
        "count": len(summarized),
        "offset": offset,
        "limit": limit,
        "entities": summarized,
    }


def find_usages(
    type: str,
    name: str,
    namespace: Optional[str] = None,
    class_name: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """
    Find usages of a function or class in the code graph.
    """
    # 1. Find the target entity to get its ID
    search_types = [type]
    if type == "function":
        search_types.append("function_template")
    elif type == "class":
        search_types.append("class_template")
        
    all_targets = []
    for t in search_types:
        search_res = search_symbol(
            type=t,
            name=name,
            namespace=namespace,
            class_name=class_name,
            limit=5,
            name_exact_match=True,
        )
        all_targets.extend(search_res.get("entities", []))
    
    if not all_targets:
        for t in search_types:
            search_res = search_symbol(
                type=t,
                name=name,
                namespace=namespace,
                class_name=class_name,
                limit=5,
            )
            all_targets.extend(search_res.get("entities", []))
        
    if not all_targets:
        return {"error": f"Target {type} '{name}' not found", "usages": []}
    
    # Take the best match
    target = all_targets[0]
    target_id = target.get("id")
    target_file = target.get("file") or target.get("def_file")
    target_line = target.get("start_line") or target.get("def_start_line")
    
    # DEBUG
    # print(f"DEBUG_FIND_USAGES: target_id={target_id}, name={target.get('name')}, file={target_file}, line={target_line}")
    
    if not target_id:
        return {"error": f"Target entity has no ID", "usages": []}
        
    usage_targets = [target]
    if type == "function":
        classes = target.get("classes", [])
        fq_name_parts = (target.get("fq_name") or "").split("::")
        is_constructor = (
            classes and (target.get("short_name") or target.get("name")) == classes[-1]
        ) or (
            len(fq_name_parts) >= 2 and fq_name_parts[-1] == fq_name_parts[-2]
        )
        if is_constructor:
            owning_class = classes[-1] if classes else fq_name_parts[-2]
            owning_namespace = (
                "::".join(target.get("namespaces", []))
                if target.get("namespaces")
                else "::".join(fq_name_parts[:-2])
            )
            class_res = search_symbol(
                type="class",
                name=owning_class,
                namespace=owning_namespace or namespace,
                limit=5,
                name_exact_match=True,
            )
            usage_targets.extend(class_res.get("entities", []))

    # 2. Get incoming call/reference neighbors. Structural membership edges are
    # context, not usages.
    usage_edge_types = [
        "CALLS",
        "DEP_CALL",
        "DEP_TEMPLATE_REF",
        "DEP_DECL_REF",
        "DEP_MEMBER_REF",
        "DEP_TYPE_REF",
        "DEP_UNEXPOSED_REF",
    ]
    usages = []
    usage_ids = set()
    usage_target_ids = {
        item.get("id") for item in usage_targets if item.get("id")
    }
    for usage_target in usage_targets:
        usage_target_id = usage_target.get("id")
        if not usage_target_id:
            continue
        neighbors_res = get_entity_neighbors(
            entity_id=usage_target_id,
            edge_types=usage_edge_types,
            direction="in",
            max_depth=1,
            max_nodes=limit,
        )
        for usage in neighbors_res.get("neighbors", []):
            usage_id = usage.get("id")
            if usage_id in usage_ids:
                continue
            usage_ids.add(usage_id)
            usages.append(usage)
            if len(usages) >= limit:
                break
        if len(usages) >= limit:
            break
    
    # 3. Fallback: Search by file:line pattern if graph is sparse
    # Some edges use "file:line:col" as callee ID instead of USR
    if len(usages) < limit and target_file and target_line:
        from code_intent.code_intent.graph_builder import _read_jsonl

        instance_name = os.environ.get("SWEBENCH_INSTANCE_NAME")
        ir = _get_index_root()
        if instance_name and ir is not None:
            index_dir = ir / instance_name / "index_output"
            # Search both in 'edges' subdir and top-level index_output
            edge_paths = [
                index_dir / "edges/edges_call.jsonl",
                index_dir / "edges/edges_ref.jsonl",
                index_dir / "edges/edges_template.jsonl",
                index_dir / "callgraph.jsonl"
            ]
            
            # Pattern to match: "callee": "...file...:line:..."
            # Use multiple patterns to be safe
            pos_patterns = [f":{target_line}:", f":{target_line}\"", f":{target_line},"]
            
            for p in edge_paths:
                if p.exists():
                    try:
                        with p.open("r", encoding="utf-8") as f:
                            for line in f:
                                # Check if this line refers to our target by ID or by location
                                if any(pp in line for pp in pos_patterns) or any(
                                    item_id in line for item_id in usage_target_ids
                                ):
                                    edge = json.loads(line)
                                    # Ensure it's actually our target being called/referenced
                                    callee = edge.get("callee", "")
                                    if callee in usage_target_ids or (str(target_line) in callee and any(p_ in callee for p_ in [":", ","])):
                                        caller_id = edge.get("caller")
                                        if caller_id and caller_id not in usage_target_ids and caller_id not in usage_ids:
                                            # Resolve caller info
                                            rag = get_graph_rag()
                                            # Try finding in functions, then classes, then templates
                                            caller_info = (
                                                rag.graph_ctx.functions.get(caller_id) or 
                                                rag.graph_ctx.classes.get(caller_id)
                                            )
                                            
                                            if caller_info:
                                                l = (
                                                    caller_info.get("start_line") or 
                                                    caller_info.get("def_start_line") or
                                                    caller_info.get("def_line")
                                                )
                                                usages.append({
                                                    "id": caller_id,
                                                    "type": "FUNC" if "short_intent" in caller_info or "return_type" in caller_info else "CLASS",
                                                    "name": caller_info.get("fq_name") or caller_info.get("short_name"),
                                                    "file": _get_relative_path(caller_info.get("file") or caller_info.get("def_file")),
                                                    "line": l,
                                                    "def_line": l,
                                                })
                                                usage_ids.add(caller_id)
                                                if len(usages) >= limit:
                                                    break
                    except Exception:
                        continue
                if len(usages) >= limit:
                    break
        
    return {
        "target": target,
        "usages": usages,
        "count": len(usages),
    }


def find_template_instantiation(
    display_name: Optional[str] = None,
    template_def_id: Optional[str] = None,
    template_arg: Optional[str] = None,
    file: Optional[str] = None,
    category: Optional[str] = None,
    is_explicit_specialization: Optional[bool] = None,
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    """
    Find template instantiations
    
    Args:
        display_name: Display name (partial match)
        template_def_id: Template definition ID (exact match)
        template_arg: Template argument (partial match, matches any argument)
        file: Trigger file path (partial match)
        category: Category: 'Full', 'Partial', 'Explicit'
        is_explicit_specialization: Whether it's an explicit specialization
        limit: Maximum number of results to return
        offset: Pagination offset
    
    Returns:
        Dictionary containing total, count, offset, limit, entities
    """
    # Get current instance
    instance_name = get_current_instance()
    entities_dir = _get_entities_dir(instance_name)
    
    file_path = entities_dir / ENTITY_FILES["template_instantiations"]
    all_entities = _load_entities_from_file(file_path)
    
    # Filter
    filtered = []
    display_name_lower = display_name.lower() if display_name else None
    template_arg_lower = template_arg.lower() if template_arg else None
    file_lower = file.lower() if file else None
    
    for entity in all_entities:
        # Filter by display_name
        if display_name_lower:
            entity_display = entity.get("display_name", "").lower()
            if display_name_lower not in entity_display:
                continue
        
        # Filter by template_def_id (exact match)
        if template_def_id:
            if entity.get("template_def_id") != template_def_id:
                continue
        
        # Filter by template_arg (partial match any argument)
        if template_arg_lower:
            template_args = entity.get("template_args", [])
            if not any(template_arg_lower in str(arg).lower() for arg in template_args):
                continue
        
        # Filter by file
        if file_lower:
            trigger_file = entity.get("trigger_file", "").lower()
            if file_lower not in trigger_file:
                continue
        
        # Filter by category
        if category:
            if entity.get("category") != category:
                continue
        
        # Filter by is_explicit_specialization
        if is_explicit_specialization is not None:
            if entity.get("is_explicit_specialization") != is_explicit_specialization:
                continue
        
        filtered.append(entity)
    
    # Pagination
    total = len(filtered)
    start = offset
    end = offset + limit
    entities = filtered[start:end]
    
    # Summarize entities to reduce token usage
    summarized = [_summarize_entity(e, "template_instantiation") for e in entities]
    
    return {
        "total": total,
        "count": len(summarized),
        "offset": offset,
        "limit": limit,
        "entities": summarized,
    }


def get_entity_by_id(
    entity_id: str,
    entity_type: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get entity by ID (exact match). Returns full details with line numbers in code.
    
    Args:
        entity_id: Entity ID (exact match)
        entity_type: Entity type, if not provided will be auto-inferred
    
    Returns:
        Entity dictionary, or None if not found
    """
    # Get current instance
    instance_name = get_current_instance()
    entities_dir = _get_entities_dir(instance_name)
    
    found_entity = None
    actual_type = None

    # If type is specified, search directly
    if entity_type:
        if entity_type not in ENTITY_TYPE_MAP:
            raise ValueError(
                f"Unsupported entity type: {entity_type}. Supported types: {', '.join(ENTITY_TYPE_MAP.keys())}"
            )
        entity_type_key = ENTITY_TYPE_MAP[entity_type]
        file_path = entities_dir / ENTITY_FILES[entity_type_key]
        entities = _load_entities_from_file(file_path)
        for entity in entities:
            if entity.get("id") == entity_id:
                found_entity = entity
                actual_type = entity_type
                break
    else:
        # Auto-infer: iterate through all types
        for type_param, file_key in ENTITY_TYPE_MAP.items():
            file_path = entities_dir / ENTITY_FILES[file_key]
            entities = _load_entities_from_file(file_path)
            for entity in entities:
                if entity.get("id") == entity_id:
                    found_entity = entity
                    actual_type = type_param
                    break
            if found_entity:
                break
    
    if found_entity:
        # Add line numbers to code if it exists
        if "code" in found_entity:
            start_line = found_entity.get("def_start_line") or found_entity.get("def_line") or 1
            found_entity["code"] = _add_line_numbers(found_entity["code"], start_line)
        
        # Simplify path
        if "def_file" in found_entity:
            found_entity["def_file"] = _get_relative_path(found_entity["def_file"])
        if "file" in found_entity:
            found_entity["file"] = _get_relative_path(found_entity["file"])
            
        return found_entity
    
    return None


def search_by_intent(
    query: str,
    top_k: int = 10,
    *,
    rerank_first: bool = False,
    expand_structure: bool = False,
    structural_reranker=None,
) -> Dict[str, Any]:
    """
    Semantic search for code using natural language descriptions.
    
    Args:
        query: Natural language description of what you are looking for
        top_k: Number of results to return
        rerank_first: Use reranker order directly instead of fusing with retrieval rank
        expand_structure: Supplement final seeds with traced structural companions
        
    Returns:
        List of matching functions with intent and code snippets
    """
    rag = _get_rag_instance()
    if not rag:
        return {"error": "RAG engine not available. Please check if VOYAGE_API_KEY is set or if indices are built.", "results": []}
    
    try:
        results = rag.search_code(query, top_k_funcs=top_k, rerank_first=rerank_first,
                                  expand_structure=expand_structure, structural_reranker=structural_reranker)
        
        # Enrich results with full entity summary (including code/skeleton)
        enriched_results = []
        for res in results:
            try:
                func_id = res.get("func_id")
                if not func_id:
                    enriched_results.append(res)
                    continue
                    
                # Get full entity details
                entity = get_entity_by_id(func_id, entity_type="function")
                if entity:
                    # Add the RAG score to the summary
                    summary = _summarize_entity(entity, "function")
                    # Keep the semantic/feature provenance when adding source
                    # locators; callers need it to follow cross-layer results.
                    summary.update({key: res[key] for key in (
                        "score", "short_intent", "feature_id", "feature_name",
                        "content_hash", "retrieval_role", "graph_paths",
                        "structural_candidates", "structural_status",
                    ) if res.get(key) is not None})
                    enriched_results.append(summary)
                else:
                    # Fallback to RAG result if entity not found in detail search
                    enriched_results.append(res)
            except Exception as e:
                print(f"DEBUG: Failed to enrich result {res.get('func_id')}: {e}")
                enriched_results.append(res)
                
        return {
            "query": query,
            "count": len(enriched_results),
            "results": enriched_results
        }
    except Exception as e:
        logger.error(f"Search failed: {str(e)}", exc_info=True)
        return {"error": f"Search failed: {str(e)}", "results": []}


def list_features(
    query: Optional[str] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    List or search high-level code features (functional clusters).
    Helpful for understanding the macro structure of the codebase.
    
    Args:
        query: Optional search keyword to filter features
        limit: Maximum number of features to return
        
    Returns:
        List of features with descriptions and member functions
    """
    rag = _get_rag_instance()
    if not rag:
        return {"error": "RAG engine not available.", "features": []}
    
    try:
        if query:
            features = rag.search_features(query, top_k=limit)
        else:
            # If no query, list most important features from graph context
            # (Note: search_features with empty query might not work well, 
            # so we use a default if available or just return empty list with error)
            features = rag.search_features("", top_k=limit)
            
        # Clean paths in members
        for feat in features:
            for member in feat.get("members", []):
                if "file" in member:
                    member["file"] = _get_relative_path(member["file"])
            if "files" in feat:
                feat["files"] = [_get_relative_path(f) for f in feat["files"]]
                
        return {
            "query": query,
            "count": len(features),
            "features": features
        }
    except Exception as e:
        return {"error": f"Failed to list features: {str(e)}", "features": []}


def get_macro_usages(
    name: str,
) -> Dict[str, Any]:
    """
    Find where a macro is defined and all functions that use it.
    
    Args:
        name: Name of the macro
        
    Returns:
        Macro definition and list of usage locations
    """
    rag = _get_rag_instance()
    if not rag:
        return {"error": "RAG engine not available."}
    
    try:
        details = rag.get_macro_details(name)
        
        # Clean paths
        for defn in details.get("definitions", []):
            if "file" in defn:
                defn["file"] = _get_relative_path(defn["file"])
        
        for usage in details.get("usages", []):
            if "file" in usage:
                usage["file"] = _get_relative_path(usage["file"])
                
        return details
    except Exception as e:
        return {"error": f"Failed to get macro details: {str(e)}"}


def get_entity_neighbors(
    entity_id: str,
    edge_types: Optional[List[str]] = None,
    direction: str = "both",
    max_depth: int = 1,
    max_nodes: int = 50,
) -> Dict[str, Any]:
    """
    Get neighbors of an entity in the code graph. This is a core GRAPHRAG function
    that enables graph traversal and relationship discovery.
    
    Args:
        entity_id: Entity ID (function, class, etc.)
        edge_types: List of edge types to filter (e.g., ["CALLS", "IN_CLASS"]).
                   If None, returns neighbors via all edge types.
        direction: "out" (outgoing edges), "in" (incoming edges), or "both"
        max_depth: Maximum depth for traversal (1 = direct neighbors only)
        max_nodes: Maximum number of nodes to return
        
    Returns:
        Dictionary with neighbors and their relationships
    """
    rag = _get_rag_instance()
    if not rag:
        return {"error": "RAG engine not available.", "neighbors": []}
    
    try:
        graph = rag.graph_ctx.graph
        
        # Check if entity exists in graph
        if entity_id not in graph.nodes:
            return {"error": f"Entity {entity_id} not found in graph", "neighbors": []}
        
        visited = set()
        current_level = {entity_id}
        all_neighbors = []
        depth = 0
        
        # BFS traversal
        while depth < max_depth and len(all_neighbors) < max_nodes:
            next_level = set()
            
            for node_id in current_level:
                if node_id in visited:
                    continue
                visited.add(node_id)
                
                # Get neighbors based on edge types and direction
                if edge_types:
                    neighbors_set = set()
                    for edge_type in edge_types:
                        neighbors_set.update(graph.neighbors(node_id, edge_type, direction))
                    neighbors = neighbors_set
                else:
                    neighbors = graph.neighbors(node_id, None, direction)
                
                def neighbor_sort_key(neighbor_id: str):
                    neighbor = graph.nodes.get(neighbor_id, {})
                    file_path = (
                        neighbor.get("file")
                        or neighbor.get("def_file")
                        or neighbor.get("path")
                        or ""
                    )
                    normalized_path = file_path.lower().replace("\\", "/")
                    is_test = bool(neighbor.get("is_test")) or any(
                        part in normalized_path
                        for part in ("/test/", "/tests/", "test_", "_test.", ".tests.")
                    )
                    name = neighbor.get("fq_name") or neighbor.get("name") or neighbor_id
                    return (is_test, normalized_path, name)

                for neighbor_id in sorted(neighbors, key=neighbor_sort_key):
                    if neighbor_id in visited or neighbor_id == entity_id:
                        continue
                    
                    if len(all_neighbors) >= max_nodes:
                        break
                    
                    neighbor_node = graph.nodes.get(neighbor_id)
                    if not neighbor_node:
                        continue
                    
                    # Get edge types connecting these nodes
                    edge_buckets = graph.edge_buckets(node_id, direction)
                    connecting_edges = []
                    for edge_type, targets in edge_buckets.items():
                        if edge_types and edge_type not in edge_types:
                            continue
                        if neighbor_id in targets:
                            connecting_edges.append(edge_type)
                    
                    # Get entity details based on node type
                    neighbor_info = {
                        "id": neighbor_id,
                        "type": neighbor_node.get("type"),
                        "edge_types": connecting_edges,
                        "depth": depth + 1,
                    }
                    
                    # Add entity-specific information
                    node_type = neighbor_node.get("type")
                    if node_type == "FUNC":
                        func_data = rag.graph_ctx.functions.get(neighbor_id)
                        if func_data:
                            # Use various possible line fields
                            line = (
                                func_data.get("start_line") or 
                                func_data.get("def_start_line") or 
                                func_data.get("def_line") or 
                                func_data.get("line")
                            )
                            neighbor_info.update({
                                "name": func_data.get("fq_name"),
                                "fq_name": func_data.get("fq_name"),
                                "file": _get_relative_path(func_data.get("file") or func_data.get("def_file")),
                                "def_file": _get_relative_path(func_data.get("file") or func_data.get("def_file")),
                                "line": line,
                                "def_line": line,
                                "def_start_line": line,
                                "start_line": line,
                                "short_intent": func_data.get("short_intent"),
                                "skeleton": _function_skeleton(func_data),
                            })
                    elif node_type == "CLASS":
                        cls_data = rag.graph_ctx.classes.get(neighbor_id)
                        if cls_data:
                            line = (
                                cls_data.get("start_line") or
                                cls_data.get("def_start_line") or 
                                cls_data.get("def_line") or 
                                cls_data.get("line")
                            )
                            neighbor_info.update({
                                "name": cls_data.get("fq_name"),
                                "fq_name": cls_data.get("fq_name"),
                                "file": _get_relative_path(cls_data.get("file") or cls_data.get("def_file")),
                                "def_file": _get_relative_path(cls_data.get("file") or cls_data.get("def_file")),
                                "line": line,
                                "def_line": line,
                                "def_start_line": line,
                                "start_line": line,
                                "skeleton": _class_skeleton(cls_data),
                            })
                    elif node_type == "INTENT_FEATURE":
                        feat_data = rag.graph_ctx.features.get(neighbor_id)
                        if feat_data:
                            neighbor_info.update({
                                "name": feat_data.get("feature_name"),
                                "description": feat_data.get("feature_description"),
                            })
                    elif node_type == "FILE":
                        neighbor_info.update({
                            "path": neighbor_node.get("path"),
                        })
                    
                    all_neighbors.append(neighbor_info)
                    if depth + 1 < max_depth:
                        next_level.add(neighbor_id)
            
            if not next_level:
                break
            current_level = next_level
            depth += 1
        
        return {
            "entity_id": entity_id,
            "count": len(all_neighbors),
            "max_depth": max_depth,
            "neighbors": all_neighbors[:max_nodes],
        }
    except Exception as e:
        return {"error": f"Failed to get neighbors: {str(e)}", "neighbors": []}


def get_entity_subgraph(
    entity_id: str,
    radius: int = 2,
    edge_types: Optional[List[str]] = None,
    include_features: bool = True,
    max_nodes: int = 100,
) -> Dict[str, Any]:
    """
    Extract a subgraph around an entity. This is useful for understanding
    the context and relationships of a specific code element.
    
    Args:
        entity_id: Entity ID to center the subgraph around
        radius: Subgraph radius (number of hops)
        edge_types: Filter by specific edge types (None = all types)
        include_features: Whether to include Feature nodes in the subgraph
        max_nodes: Maximum number of nodes in the subgraph
        
    Returns:
        Subgraph structure with nodes and edges
    """
    rag = _get_rag_instance()
    if not rag:
        return {"error": "RAG engine not available.", "nodes": [], "edges": []}
    
    try:
        graph = rag.graph_ctx.graph
        
        if entity_id not in graph.nodes:
            return {"error": f"Entity {entity_id} not found in graph", "nodes": [], "edges": []}
        
        # Collect nodes using BFS
        visited = set()
        nodes = {}
        edges = []
        current_level = {entity_id}
        
        for depth in range(radius + 1):
            next_level = set()
            
            for node_id in current_level:
                if node_id in visited or len(nodes) >= max_nodes:
                    continue
                visited.add(node_id)
                
                node = graph.nodes.get(node_id)
                if not node:
                    continue
                
                # Filter by node type if needed
                if not include_features and node.get("type") == "INTENT_FEATURE":
                    continue
                
                nodes[node_id] = {
                    "id": node_id,
                    "type": node.get("type"),
                    **{k: v for k, v in node.items() if k not in ("id", "type")}
                }
                
                # Get outgoing edges
                edge_buckets = graph.edge_buckets(node_id, "out")
                for edge_type, targets in edge_buckets.items():
                    if edge_types and edge_type not in edge_types:
                        continue
                    
                    for target_id in targets:
                        if target_id not in visited and len(nodes) < max_nodes:
                            if depth < radius:
                                next_level.add(target_id)
                            
                            # Add edge
                            edges.append({
                                "source": node_id,
                                "target": target_id,
                                "type": edge_type,
                            })
            
            if not next_level or len(nodes) >= max_nodes:
                break
            current_level = next_level
        
        return {
            "center_entity_id": entity_id,
            "radius": radius,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": list(nodes.values()),
            "edges": edges,
        }
    except Exception as e:
        return {"error": f"Failed to extract subgraph: {str(e)}", "nodes": [], "edges": []}


def find_related_entities(
    entity_id: str,
    relation_types: Optional[List[str]] = None,
    max_hops: int = 2,
    entity_types: Optional[List[str]] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    Find entities related to the given entity through various graph relationships.
    This is a key GRAPHRAG function for discovering code relationships.
    
    Args:
        entity_id: Starting entity ID
        relation_types: Types of relationships to follow (e.g., ["CALLS", "IN_CLASS"])
        max_hops: Maximum number of hops to traverse
        entity_types: Filter by entity types (e.g., ["function", "class"])
        limit: Maximum number of results to return
        
    Returns:
        List of related entities with relationship paths
    """
    rag = _get_rag_instance()
    if not rag:
        return {"error": "RAG engine not available.", "related": []}
    
    try:
        graph = rag.graph_ctx.graph
        
        if entity_id not in graph.nodes:
            return {"error": f"Entity {entity_id} not found in graph", "related": []}
        
        # Map entity types to node types
        type_map = {
            "function": "FUNC",
            "class": "CLASS",
            "file": "FILE",
            "feature": "INTENT_FEATURE",
            "macro": "MACRO",
        }
        
        visited = set()
        related = []
        current_level = [(entity_id, [])]  # (node_id, path)
        
        for hop in range(max_hops):
            next_level = []
            
            for node_id, path in current_level:
                if node_id in visited:
                    continue
                visited.add(node_id)
                
                # Get neighbors
                neighbors = graph.neighbors(node_id, None, "both")
                
                for neighbor_id in neighbors:
                    if neighbor_id == entity_id or neighbor_id in visited:
                        continue
                    
                    if len(related) >= limit:
                        break
                    
                    neighbor_node = graph.nodes.get(neighbor_id)
                    if not neighbor_node:
                        continue
                    
                    # Filter by entity type
                    node_type = neighbor_node.get("type")
                    if entity_types:
                        if node_type not in [type_map.get(et) for et in entity_types]:
                            continue
                    
                    # Get edge types connecting these nodes
                    edge_buckets_out = graph.edge_buckets(node_id, "out")
                    edge_buckets_in = graph.edge_buckets(node_id, "in")
                    connecting_edges = []
                    
                    # Check outgoing edges
                    for edge_type, targets in edge_buckets_out.items():
                        if neighbor_id in targets:
                            if not relation_types or edge_type in relation_types:
                                connecting_edges.append(edge_type)
                    
                    # Check incoming edges
                    for edge_type, sources in edge_buckets_in.items():
                        if neighbor_id in sources:
                            if not relation_types or edge_type in relation_types:
                                connecting_edges.append(edge_type)
                    
                    if not connecting_edges:
                        continue
                    
                    # Build entity info
                    entity_info = {
                        "id": neighbor_id,
                        "type": node_type,
                        "hops": hop + 1,
                        "path": path + [{"from": node_id, "to": neighbor_id, "via": connecting_edges}],
                        "edge_types": connecting_edges,
                    }
                    
                    # Add entity-specific details
                    if node_type == "FUNC":
                        func_data = rag.graph_ctx.functions.get(neighbor_id)
                        if func_data:
                            entity_info.update({
                                "name": func_data.get("fq_name"),
                                "file": _get_relative_path(func_data.get("file")),
                            })
                    elif node_type == "CLASS":
                        cls_data = rag.graph_ctx.classes.get(neighbor_id)
                        if cls_data:
                            entity_info.update({
                                "name": cls_data.get("fq_name"),
                                "file": _get_relative_path(cls_data.get("file")),
                            })
                    
                    related.append(entity_info)
                    if hop + 1 < max_hops:
                        next_level.append((neighbor_id, entity_info["path"]))
            
            if not next_level or len(related) >= limit:
                break
            current_level = next_level
        
        return {
            "entity_id": entity_id,
            "count": len(related),
            "related": related,
        }
    except Exception as e:
        return {"error": f"Failed to find related entities: {str(e)}", "related": []}


# =============================================================================
# OpenAI Function Calling format tool definitions
# =============================================================================

def get_tools_definition() -> List[Dict[str, Any]]:
    """Get tool definitions in OpenAI Function Calling format"""
    return [
        {
            "type": "function",
            "function": {
                "name": "search_symbol",
                "description": "Search code symbols. Returns full 'code' for functions, and a 'skeleton' (outline) for classes. Use get_entity_by_id for complete class implementation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["function", "class", "class_template", "function_template", "macro"],
                            "description": "Entity type: 'function'=regular function, 'class'=regular class, 'class_template'=class template, 'function_template'=function template, 'macro'=macro definition"
                        },
                        "name": {
                            "type": "string",
                            "description": "Symbol name (partial match by default, supports fq_name, short_name)"
                        },
                        "name_exact_match": {
                            "type": "boolean",
                            "default": False,
                            "description": "If true, use exact match for name; if false, use partial match (substring match)"
                        },
                        "file": {
                            "type": "string",
                            "description": "File path (partial match)"
                        },
                        "namespace": {
                            "type": "string",
                            "description": "Namespace (partial match, only for function/class)"
                        },
                        "class_name": {
                            "type": "string",
                            "description": "Class name (partial match, only for function)"
                        },
                        "return_type": {
                            "type": "string",
                            "description": "Return type (exact match, only for function)"
                        },
                        "parameter_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Parameter type list (exact match, only for function)"
                        },
                        "base_class": {
                            "type": "string",
                            "description": "Base class name (partial match, only for class)"
                        },
                        "kind": {
                            "type": "string",
                            "enum": ["class", "struct", "template"],
                            "description": "Type: 'class', 'struct', 'template' (only for class)"
                        },
                        "is_template": {
                            "type": "boolean",
                            "description": "Whether it's a template (only for function/class)"
                        },
                        "is_method": {
                            "type": "boolean",
                            "description": "Whether it's a method (only for function)"
                        },
                        "is_static": {
                            "type": "boolean",
                            "description": "Whether it's static (only for function)"
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 1000,
                            "default": 50,
                            "description": "Maximum number of results to return"
                        },
                        "offset": {
                            "type": "integer",
                            "minimum": 0,
                            "default": 0,
                            "description": "Pagination offset"
                        }
                    },
                    "required": ["type"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "find_template_instantiation",
                "description": "Find template instantiations. Returns summarized results (id, name, template_def_id, args, file, line, category). Use get_entity_by_id to get full details.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "display_name": {
                            "type": "string",
                            "description": "Display name (partial match)"
                        },
                        "template_def_id": {
                            "type": "string",
                            "description": "Template definition ID (exact match)"
                        },
                        "template_arg": {
                            "type": "string",
                            "description": "Template argument (partial match, matches any argument)"
                        },
                        "file": {
                            "type": "string",
                            "description": "Trigger file path (partial match)"
                        },
                        "category": {
                            "type": "string",
                            "enum": ["Full", "Partial", "Explicit"],
                            "description": "Category: 'Full'=full instantiation, 'Partial'=partial specialization, 'Explicit'=explicit specialization"
                        },
                        "is_explicit_specialization": {
                            "type": "boolean",
                            "description": "Whether it's an explicit specialization"
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 1000,
                            "default": 50,
                            "description": "Maximum number of results to return"
                        },
                        "offset": {
                            "type": "integer",
                            "minimum": 0,
                            "default": 0,
                            "description": "Pagination offset"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_entity_by_id",
                "description": "Get entity by ID (exact match). Returns COMPLETE information including code snippet, comments, and members summary.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "Entity ID (exact match)"
                        },
                        "entity_type": {
                            "type": "string",
                            "enum": ["function", "class", "class_template", "function_template", "macro", "template_instantiation"],
                            "description": "Entity type, if not provided will be auto-inferred"
                        }
                    },
                    "required": ["entity_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_by_intent",
                "description": "Semantic search for code using natural language descriptions. Use this when you are not sure about the exact symbol name but know what the code does.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language description of the code's behavior or intent (e.g., 'how is memory allocated', 'exception handling for network')"
                        },
                        "top_k": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 50,
                            "default": 10,
                            "description": "Number of results to return"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_features",
                "description": "List or search high-level functional features (clusters of related functions). Helps in understanding the macro architecture and finding entry points for specific modules.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Optional search keyword to filter features by name or description"
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100,
                            "default": 20,
                            "description": "Maximum number of features to return"
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_macro_usages",
                "description": "Find all functions and locations where a specific macro is used. Essential for understanding the impact of changing a configuration macro or finding cross-cutting concerns.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Exact name of the macro"
                        }
                    },
                    "required": ["name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_entity_neighbors",
                "description": "Get neighbors of an entity in the code graph. Core GRAPHRAG function for graph traversal and relationship discovery. Returns entities connected via various edge types (CALLS, IN_CLASS, etc.).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "Entity ID (function, class, etc.)"
                        },
                        "edge_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of edge types to filter (e.g., ['CALLS', 'IN_CLASS']). If not provided, returns neighbors via all edge types."
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["out", "in", "both"],
                            "default": "both",
                            "description": "Direction of edges: 'out'=outgoing, 'in'=incoming, 'both'=both directions"
                        },
                        "max_depth": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 5,
                            "default": 1,
                            "description": "Maximum depth for traversal (1=direct neighbors only, 2=neighbors of neighbors)"
                        },
                        "max_nodes": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 200,
                            "default": 50,
                            "description": "Maximum number of nodes to return"
                        }
                    },
                    "required": ["entity_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_entity_subgraph",
                "description": "Extract a subgraph around an entity. Useful for understanding the context and relationships of a specific code element. Returns nodes and edges within a specified radius.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "Entity ID to center the subgraph around"
                        },
                        "radius": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 5,
                            "default": 2,
                            "description": "Subgraph radius (number of hops from center)"
                        },
                        "edge_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by specific edge types. If not provided, includes all edge types."
                        },
                        "include_features": {
                            "type": "boolean",
                            "default": True,
                            "description": "Whether to include Feature nodes in the subgraph"
                        },
                        "max_nodes": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 500,
                            "default": 100,
                            "description": "Maximum number of nodes in the subgraph"
                        }
                    },
                    "required": ["entity_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "find_related_entities",
                "description": "Find entities related to the given entity through various graph relationships. Key GRAPHRAG function for discovering code relationships and dependencies.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "Starting entity ID"
                        },
                        "relation_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Types of relationships to follow (e.g., ['CALLS', 'IN_CLASS']). If not provided, follows all relationship types."
                        },
                        "max_hops": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 5,
                            "default": 2,
                            "description": "Maximum number of hops to traverse"
                        },
                        "entity_types": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["function", "class", "file", "feature", "macro"]},
                            "description": "Filter by entity types. If not provided, includes all types."
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100,
                            "default": 20,
                            "description": "Maximum number of results to return"
                        }
                    },
                    "required": ["entity_id"]
                }
            }
        }
    ]


def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute tool function
    
    Args:
        tool_name: Tool name
        arguments: Tool arguments
    
    Returns:
        Tool execution result
    """
    if tool_name == "search_symbol":
        return search_symbol(**arguments)
    elif tool_name == "find_template_instantiation":
        return find_template_instantiation(**arguments)
    elif tool_name == "get_entity_by_id":
        return get_entity_by_id(**arguments)
    elif tool_name == "search_by_intent":
        return search_by_intent(**arguments)
    elif tool_name == "list_features":
        return list_features(**arguments)
    elif tool_name == "get_macro_usages":
        return get_macro_usages(**arguments)
    elif tool_name == "get_entity_neighbors":
        return get_entity_neighbors(**arguments)
    elif tool_name == "get_entity_subgraph":
        return get_entity_subgraph(**arguments)
    elif tool_name == "find_related_entities":
        return find_related_entities(**arguments)
    else:
        raise ValueError(f"Unknown tool name: {tool_name}")
