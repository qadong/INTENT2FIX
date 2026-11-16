from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _is_test_path(path: str | None) -> bool:
    if not path:
        return False
    try:
        parts = [part.lower() for part in Path(path).parts]
    except Exception:
        normalized = path.replace("\\", "/").lower()
        return "/test/" in normalized or "/tests/" in normalized
    return any(part in {"test", "tests"} for part in parts)


def _read_file_lines(file_path: str, cache: Dict[str, List[str]]) -> List[str]:
    if file_path in cache:
        return cache[file_path]
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        lines = []
    cache[file_path] = lines
    return lines


def _extract_code(file_path: str, start_line: int, end_line: int, cache: Dict[str, List[str]]) -> str:
    lines = _read_file_lines(file_path, cache)
    if not lines:
        return ""
    start = max(1, min(start_line, len(lines)))
    end = max(start, min(end_line, len(lines)))
    return "".join(lines[start - 1 : end])


def _parse_symbol(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    sym = record.get("Symbol")
    if not sym:
        return None
    kind = sym.get("SymInfo", {}).get("Kind")
    # 2 -> Function, 3 -> Method (clangd enum)
    if kind not in (2, 3):
        return None
    loc = sym.get("Definition") or sym.get("CanonicalDeclaration")
    if not loc:
        return None
    file_path = loc.get("File") or loc.get("FilePath")
    if not file_path or _is_test_path(file_path):
        return None
    start = loc.get("Start", {})
    end = loc.get("End", {})
    start_line = start.get("Line")
    end_line = end.get("Line", start_line)
    if not isinstance(start_line, int):
        return None
    scope = sym.get("Scope", "")
    fq_name = f"{scope}{sym.get('Name', '')}"
    namespaces = [seg for seg in scope.split("::") if seg]
    result = {
        "id": sym.get("ID") or f"clangd::{fq_name}@{file_path}:{start_line}",
        "fq_name": fq_name,
        "short_name": sym.get("Name", ""),
        "namespaces": namespaces,
        "classes": [],
        "def_file": os.path.abspath(file_path),
        "def_start_line": start_line,
        "def_end_line": end_line,
        "is_method": kind == 3,
        "is_static": False,
        "is_template": False,
        "return_type": "",
        "parameter_types": [],
        "raw_comment": sym.get("Documentation", ""),
        "macros_used": [],
        "macro_guards": [],
        "error_macros": [],
        "config_macros": [],
        "log_strings": [],
        "called_usrs": [],
    }
    return result


def merge_clangd_symbols(index_dir: str | Path, clangd_jsonl: str | Path) -> None:
    base = Path(index_dir)
    symbols_path = Path(clangd_jsonl)
    if not symbols_path.exists():
        print(f"clangd symbols file not found: {symbols_path}")
        return

    functions_path = base / "functions.jsonl"
    if not functions_path.exists():
        print(f"{functions_path} not found, skipping clangd merge")
        return

    existing: Dict[str, Dict[str, Any]] = {}
    with functions_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                existing[rec["id"]] = rec

    source_cache: Dict[str, List[str]] = {}
    added = 0

    for record in _load_jsonl(symbols_path):
        parsed = _parse_symbol(record)
        if not parsed:
            continue
        if parsed["id"] in existing:
            continue
        code = _extract_code(
            parsed["def_file"], parsed["def_start_line"], parsed["def_end_line"], source_cache
        )
        parsed["code"] = code
        existing[parsed["id"]] = parsed
        added += 1

    if added == 0:
        print("clangd provided no new function symbols.")
        return

    with functions_path.open("w", encoding="utf-8") as f:
        for rec in existing.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Imported {added} extra functions from clangd.")
