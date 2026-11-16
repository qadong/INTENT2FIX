from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}
TEST_PATH_PARTS = {"test", "tests", "testing", "unittest", "unittests"}
EXAMPLE_PATH_PARTS = {"example", "examples"}
VENDORED_PATH_PARTS = {
    "dependencies",
    "deps",
    "external",
    "extern",
    "subprojects",
    "third_party",
    "thirdparty",
    "vendor",
    "vendors",
}
GENERATED_PATH_PARTS = {".cache", "_build", "_deps", "build"}
FRAMEWORK_TEST_PREFIXES = (
    "gmock/test/",
    "googlemock/test/",
    "googletest/test/",
    "gtest/test/",
)
FRAMEWORK_SOURCE_NAMES = {
    "gmock-all.cc",
    "gmock-gtest-all.cc",
    "gtest-all.cc",
}
FRAMEWORK_SOURCE_SUFFIXES = ("gmock/gmock.h", "gtest/gtest.h")

OUTER_MACROS = {
    "SCENARIO",
    "TEMPLATE_LIST_TEST_CASE",
    "TEMPLATE_PRODUCT_TEST_CASE",
    "TEMPLATE_TEST_CASE",
    "TEST",
    "TEST_CASE",
    "TEST_CASE_METHOD",
    "TEST_F",
    "TEST_P",
    "TYPED_TEST",
    "TYPED_TEST_P",
}
GTEST_MACROS = {"TEST", "TEST_F", "TEST_P", "TYPED_TEST", "TYPED_TEST_P"}
SECTION_MACROS = {
    "AND_GIVEN",
    "AND_THEN",
    "AND_WHEN",
    "GIVEN",
    "SECTION",
    "SUBCASE",
    "SUBTEST",
    "THEN",
    "WHEN",
}
ALL_TEST_MACROS = OUTER_MACROS | SECTION_MACROS
BOOL_TEST_RE = re.compile(
    r"(?m)^\s*(?:static\s+)?bool\s+(?P<name>[A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{"
)
MAIN_RE = re.compile(r"(?m)^\s*int\s+main\s*\([^;{}]*\)\s*\{")
QUOTED_RE = re.compile(r'"(?P<value>(?:\\.|[^"\\])*)"')
ASSERTION_RE = re.compile(
    r"\b(?:ASSERT\w*|CHECK\w*|EXPECT\w*|FAIL|REQUIRE\w*|STATIC_REQUIRE|VERIFY)\s*\("
)
CALL_NAME_RE = re.compile(r"\b(?P<name>[A-Za-z_]\w*)\s*\(")
QUALIFIED_NAME_RE = re.compile(
    r"\b[A-Za-z_]\w*(?:::[A-Za-z_]\w*)+"
)
TEST_CASE_EXTRACTOR_VERSION = 4


@dataclass(frozen=True)
class _MacroRecord:
    start: int
    end: int
    block_start: int
    canonical_macro: str
    name: str
    code: str


def normalize_repo_path(path: str, repo_root: str | Path | None = None) -> str:
    normalized = path.replace("\\", "/")
    if repo_root:
        root = str(Path(repo_root).resolve()).replace("\\", "/").rstrip("/")
        if normalized == root:
            return ""
        if normalized.startswith(root + "/"):
            normalized = normalized[len(root) + 1 :]
    return normalized.lstrip("./")


def is_vendored_or_framework_test(path: str) -> bool:
    pure = PurePosixPath(path.replace("\\", "/").lstrip("./"))
    parts = {part.lower() for part in pure.parts[:-1]}
    name = pure.name.lower()
    normalized = pure.as_posix().lower()
    return bool(
        parts & (VENDORED_PATH_PARTS | GENERATED_PATH_PARTS)
        or any(part.startswith("cmake-build-") for part in parts)
        or any(
            normalized.startswith(prefix) or f"/{prefix}" in normalized
            for prefix in FRAMEWORK_TEST_PREFIXES
        )
        or name in FRAMEWORK_SOURCE_NAMES
        or normalized.endswith(FRAMEWORK_SOURCE_SUFFIXES)
    )


def is_repository_test_source(path: str) -> bool:
    normalized = path.replace("\\", "/").lstrip("./")
    pure = PurePosixPath(normalized)
    if pure.suffix.lower() not in SOURCE_SUFFIXES:
        return False
    if is_vendored_or_framework_test(normalized):
        return False
    parts = {part.lower() for part in pure.parts[:-1]}
    if parts & (TEST_PATH_PARTS | EXAMPLE_PATH_PARTS):
        return True
    stem = pure.stem.lower()
    return bool(re.fullmatch(
        r"(?:tests?|examples?)|(?:tests?|examples?)[_.-].*|"
        r".*[_.-](?:tests?|examples?)",
        stem,
    ))


def _line_number(content: str, offset: int) -> int:
    return content.count("\n", 0, offset) + 1


def _column_number(content: str, offset: int) -> int:
    return offset - content.rfind("\n", 0, offset)


def _skip_quoted_or_comment(content: str, index: int) -> int | None:
    if content.startswith("//", index):
        newline = content.find("\n", index + 2)
        return len(content) if newline < 0 else newline + 1
    if content.startswith("/*", index):
        end = content.find("*/", index + 2)
        return len(content) if end < 0 else end + 2
    if content.startswith('R"', index):
        delimiter_end = content.find("(", index + 2, index + 20)
        if delimiter_end >= 0:
            delimiter = content[index + 2 : delimiter_end]
            terminator = ")" + delimiter + '"'
            end = content.find(terminator, delimiter_end + 1)
            return len(content) if end < 0 else end + len(terminator)
    if content[index] not in {'"', "'"}:
        return None
    quote = content[index]
    index += 1
    while index < len(content):
        if content[index] == "\\":
            index += 2
            continue
        if content[index] == quote:
            return index + 1
        index += 1
    return len(content)


def _mask_range(buffer: list[str], start: int, end: int) -> None:
    for index in range(start, min(end, len(buffer))):
        if buffer[index] != "\n":
            buffer[index] = " "


_PP_TOKEN_RE = re.compile(
    r"\s*(?P<token>&&|\|\||!|\(|\)|"
    r"0[xX][0-9A-Fa-f]+[uUlL]*|\d+[uUlL]*|[A-Za-z_]\w*)"
)


def _preprocessor_boolean(expression: str) -> bool | None:
    """Evaluate boolean-only preprocessor expressions, preserving unknowns."""
    tokens: list[str] = []
    offset = 0
    while offset < len(expression):
        match = _PP_TOKEN_RE.match(expression, offset)
        if match is None:
            if expression[offset:].strip():
                return None
            break
        tokens.append(match.group("token"))
        offset = match.end()

    position = 0

    def parse_primary() -> tuple[bool, bool | None]:
        nonlocal position
        if position >= len(tokens):
            return False, None
        token = tokens[position]
        if token == "(":
            position += 1
            valid, value = parse_or()
            if not valid or position >= len(tokens) or tokens[position] != ")":
                return False, None
            position += 1
            return True, value
        if re.fullmatch(r"(?:0[xX][0-9A-Fa-f]+|\d+)[uUlL]*", token):
            position += 1
            number = re.sub(r"[uUlL]+$", "", token)
            try:
                base = 16 if number.lower().startswith("0x") else 10
                if len(number) > 1 and number.startswith("0") and base == 10:
                    base = 8
                return True, int(number, base) != 0
            except ValueError:
                return True, None
        if re.fullmatch(r"[A-Za-z_]\w*", token):
            position += 1
            if token == "defined" and position < len(tokens):
                if tokens[position] == "(":
                    position += 1
                    if position >= len(tokens) or not re.fullmatch(
                        r"[A-Za-z_]\w*", tokens[position]
                    ):
                        return False, None
                    position += 1
                    if position >= len(tokens) or tokens[position] != ")":
                        return False, None
                    position += 1
                elif re.fullmatch(r"[A-Za-z_]\w*", tokens[position]):
                    position += 1
            return True, None
        return False, None

    def parse_unary() -> tuple[bool, bool | None]:
        nonlocal position
        if position < len(tokens) and tokens[position] == "!":
            position += 1
            valid, value = parse_unary()
            return valid, None if value is None else not value
        return parse_primary()

    def parse_and() -> tuple[bool, bool | None]:
        nonlocal position
        valid, value = parse_unary()
        while valid and position < len(tokens) and tokens[position] == "&&":
            position += 1
            rhs_valid, rhs = parse_unary()
            valid = rhs_valid
            if value is False or rhs is False:
                value = False
            elif value is True and rhs is True:
                value = True
            else:
                value = None
        return valid, value

    def parse_or() -> tuple[bool, bool | None]:
        nonlocal position
        valid, value = parse_and()
        while valid and position < len(tokens) and tokens[position] == "||":
            position += 1
            rhs_valid, rhs = parse_and()
            valid = rhs_valid
            if value is True or rhs is True:
                value = True
            elif value is False and rhs is False:
                value = False
            else:
                value = None
        return valid, value

    valid, value = parse_or()
    return value if valid and position == len(tokens) else None


def _logical_source_lines(content: str) -> Iterable[tuple[int, int, str]]:
    lines = content.splitlines(keepends=True)
    offset = 0
    index = 0
    while index < len(lines):
        start = offset
        logical = lines[index]
        offset += len(lines[index])
        index += 1
        while logical.rstrip("\r\n").endswith("\\") and index < len(lines):
            logical += lines[index]
            offset += len(lines[index])
            index += 1
        yield start, offset, logical


def _source_analysis(content: str) -> tuple[str, list[tuple[int, str, str]]]:
    buffer = list(content)
    index = 0
    while index < len(content):
        if content.startswith("//", index):
            end = content.find("\n", index + 2)
            end = len(content) if end < 0 else end
            _mask_range(buffer, index, end)
            index = end
            continue
        if content.startswith("/*", index):
            end = content.find("*/", index + 2)
            end = len(content) if end < 0 else end + 2
            _mask_range(buffer, index, end)
            index = end
            continue
        if content.startswith('R"', index):
            delimiter_end = content.find("(", index + 2, index + 20)
            if delimiter_end >= 0:
                delimiter = content[index + 2 : delimiter_end]
                terminator = ")" + delimiter + '"'
                end = content.find(terminator, delimiter_end + 1)
                end = len(content) if end < 0 else end + len(terminator)
                _mask_range(buffer, index, end)
                index = end
                continue
        if content[index] in {'"', "'"}:
            end = _skip_quoted_or_comment(content, index) or len(content)
            _mask_range(buffer, index, end)
            index = end
            continue
        index += 1

    active_directives: list[tuple[int, str, str]] = []
    conditional_stack: list[list[bool]] = []
    for start, end, logical_line in _logical_source_lines(content):
        directive_view = "".join(buffer[start:end])
        stripped = directive_view.lstrip()
        directive = re.match(
            r"#\s*(?P<name>\w+)(?P<body>.*)",
            stripped,
            flags=re.DOTALL,
        )
        mask_line = any(not branch[0] for branch in conditional_stack)
        if directive:
            name = directive.group("name").lower()
            body = re.sub(r"\\\r?\n", " ", directive.group("body")).strip()
            if all(branch[0] for branch in conditional_stack):
                active_directives.append((start, name, body))
            condition = _preprocessor_boolean(body)
            if name == "if":
                possible = condition is not False
                definite = condition is True
                conditional_stack.append([possible, definite, possible, definite])
            elif name in {"ifdef", "ifndef"}:
                conditional_stack.append([True, False, True, False])
            elif name == "else" and conditional_stack:
                _possible, _definite, may_be_taken, must_be_taken = conditional_stack[-1]
                conditional_stack[-1] = [
                    not must_be_taken,
                    not may_be_taken,
                    True,
                    True,
                ]
            elif name == "elif" and conditional_stack:
                _possible, _definite, may_be_taken, must_be_taken = conditional_stack[-1]
                condition_possible = condition is not False
                condition_definite = condition is True
                conditional_stack[-1] = [
                    not must_be_taken and condition_possible,
                    not may_be_taken and condition_definite,
                    may_be_taken or condition_possible,
                    must_be_taken or condition_definite,
                ]
            elif name == "endif" and conditional_stack:
                conditional_stack.pop()
            mask_line = True
        if mask_line:
            _mask_range(buffer, start, end)
    return "".join(buffer), active_directives


def code_search_view(content: str) -> str:
    """Mask non-code and unreachable regions while preserving source offsets."""
    return _source_analysis(content)[0]


_ALIAS_DEFINE_RE = re.compile(
    r"^(?P<alias>[A-Za-z_]\w*)(?:\s*\([^)]*\))?"
    r"\s+(?P<target>[A-Za-z_]\w*)\b"
)
_ALIAS_UNDEF_RE = re.compile(r"^(?P<alias>[A-Za-z_]\w*)\b")


def _macro_alias_events(
    directives: list[tuple[int, str, str]],
) -> list[tuple[int, str, str | None]]:
    events = []
    for offset, name, body in directives:
        if name == "define":
            match = _ALIAS_DEFINE_RE.match(body)
            if match:
                events.append((offset, match.group("alias"), match.group("target")))
        elif name == "undef":
            match = _ALIAS_UNDEF_RE.match(body)
            if match:
                events.append((offset, match.group("alias"), None))
    return events


def _resolve_test_macro(name: str, definitions: dict[str, str]) -> str | None:
    seen = set()
    while name not in ALL_TEST_MACROS:
        if name in seen or name not in definitions:
            return None
        seen.add(name)
        name = definitions[name]
    return name


def discover_test_macro_aliases(contents: Iterable[str]) -> dict[str, str]:
    definitions: dict[str, str] = {}
    for content in contents:
        _view, directives = _source_analysis(content)
        for _offset, alias, target in _macro_alias_events(directives):
            if target is None:
                definitions.pop(alias, None)
            else:
                definitions[alias] = target
    return {
        alias: canonical
        for alias in definitions
        if (canonical := _resolve_test_macro(alias, definitions)) is not None
    }


def _matching_delimiter(content: str, start: int, opening: str, closing: str) -> int | None:
    depth = 0
    index = start
    while index < len(content):
        skipped = _skip_quoted_or_comment(content, index)
        if skipped is not None:
            index = skipped
            continue
        character = content[index]
        if character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return None


def _following_block(content: str, start: int, max_gap: int = 1200) -> tuple[int, int] | None:
    index = start
    limit = min(len(content), start + max_gap)
    while index < limit:
        skipped = _skip_quoted_or_comment(content, index)
        if skipped is not None:
            index = skipped
            continue
        if content[index] == "{":
            end = _matching_delimiter(content, index, "{", "}")
            return (index, end) if end is not None else None
        if content[index] == ";":
            return None
        index += 1
    return None


def _macro_name(macro: str, invocation: str) -> str:
    quoted = [match.group("value") for match in QUOTED_RE.finditer(invocation)]
    if macro.startswith("TEST") and macro not in {
        "TEST_CASE",
        "TEST_CASE_METHOD",
        "TEMPLATE_TEST_CASE",
    }:
        inner = invocation[invocation.find("(") + 1 : invocation.rfind(")")]
        fields = [field.strip() for field in inner.split(",", 2)]
        if len(fields) >= 2:
            return f"{fields[0]}.{fields[1]}"
    if quoted:
        return quoted[0]
    return macro.lower()


def _setup_before_child(
    content: str,
    parent: _MacroRecord,
    child: _MacroRecord,
    direct_children: list[_MacroRecord],
) -> str:
    pieces = []
    cursor = parent.block_start + 1
    for sibling in direct_children:
        if sibling.start >= child.start:
            break
        pieces.append(content[cursor:sibling.start])
        cursor = max(cursor, sibling.end)
    pieces.append(content[cursor:child.start])
    return "".join(pieces).strip()


def _oracle_excerpt(code: str, max_lines: int = 16) -> str:
    lines = code.splitlines()
    selected = [line.strip() for line in lines if ASSERTION_RE.search(line)]
    return "\n".join(selected[:max_lines])[:2400]


def _stable_test_id(path: str, line: int, name: str, framework: str) -> str:
    payload = f"{path}:{line}:{framework}:{name}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"testcase:{digest}"


def _record(
    *,
    path: str,
    start_line: int,
    end_line: int,
    framework: str,
    name: str,
    code: str,
    start_column: int = 1,
    end_column: int = 1,
    parent_name: str = "",
    parent_test_id: str = "",
    ancestor_test_ids: list[str] | None = None,
    inherited_setup: str = "",
) -> dict[str, Any]:
    normalized_code = code.strip()
    return {
        "id": _stable_test_id(path, start_line, name, framework),
        "fq_name": name,
        "short_name": name,
        "signature": name,
        "def_file": path,
        "def_start_line": start_line,
        "def_end_line": end_line,
        "def_start_column": start_column,
        "def_end_column": end_column,
        "framework": framework,
        "extractor_version": TEST_CASE_EXTRACTOR_VERSION,
        "parent_name": parent_name,
        "parent_test_id": parent_test_id,
        "ancestor_test_ids": list(ancestor_test_ids or []),
        "inherited_setup": inherited_setup.strip()[:4000],
        "is_test": True,
        "code": normalized_code[:16000],
        "oracle_code": _oracle_excerpt(normalized_code),
        "source_func_ids": [],
        "focal_entities": [],
        "content_hash": hashlib.sha256(normalized_code.encode("utf-8")).hexdigest(),
    }


def extract_test_cases(
    path: str,
    content: str,
    macro_aliases: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    if not content or not is_repository_test_source(path):
        return []

    search_view, directives = _source_analysis(content)
    alias_events = _macro_alias_events(directives)
    macro_names = (
        ALL_TEST_MACROS
        | set(macro_aliases or {})
        | {alias for _offset, alias, _target in alias_events}
    )
    macro_re = re.compile(
        r"\b(?P<macro>"
        + "|".join(re.escape(name) for name in sorted(macro_names, key=len, reverse=True))
        + r")\s*\("
    )
    macro_records: list[_MacroRecord] = []
    definitions = dict(macro_aliases or {})
    event_index = 0
    for match in macro_re.finditer(search_view):
        while event_index < len(alias_events) and alias_events[event_index][0] < match.start():
            _offset, alias, target = alias_events[event_index]
            if target is None:
                definitions.pop(alias, None)
            else:
                definitions[alias] = target
            event_index += 1
        macro = match.group("macro")
        canonical_macro = _resolve_test_macro(macro, definitions)
        if canonical_macro is None:
            continue
        open_paren = content.find("(", match.start(), match.end() + 1)
        if open_paren < 0:
            continue
        close_paren = _matching_delimiter(content, open_paren, "(", ")")
        if close_paren is None:
            continue
        block = _following_block(content, close_paren + 1)
        if block is None:
            continue
        block_start, block_end = block
        invocation = content[match.start() : close_paren + 1]
        macro_records.append(_MacroRecord(
            start=match.start(),
            end=block_end + 1,
            block_start=block_start,
            canonical_macro=canonical_macro,
            name=_macro_name(canonical_macro, invocation),
            code=content[match.start() : block_end + 1],
        ))

    records: list[dict[str, Any]] = []
    parent_by_start: dict[int, _MacroRecord | None] = {}
    children_by_parent: dict[int, list[_MacroRecord]] = {}
    for record in macro_records:
        parents = [
            candidate
            for candidate in macro_records
            if candidate.start < record.start and record.end <= candidate.end
        ]
        parent = min(parents, key=lambda item: item.end - item.start) if parents else None
        parent_by_start[record.start] = parent
        if parent is not None:
            children_by_parent.setdefault(parent.start, []).append(record)

    display_name_by_start: dict[int, str] = {}
    test_id_by_start: dict[int, str] = {}
    ancestor_ids_by_start: dict[int, list[str]] = {}
    inherited_setup_by_start: dict[int, str] = {}
    for record in macro_records:
        parent_name = ""
        inherited_setup = ""
        parent = parent_by_start[record.start]
        if record.canonical_macro in SECTION_MACROS and parent is not None:
            parent_name = display_name_by_start.get(parent.start, parent.name)
            inherited_setup = "\n".join(filter(None, [
                inherited_setup_by_start.get(parent.start, ""),
                _setup_before_child(
                    content,
                    parent,
                    record,
                    children_by_parent.get(parent.start, []),
                ),
            ]))[-4000:]
        display_name = f"{parent_name} / {record.name}" if parent_name else record.name
        display_name_by_start[record.start] = display_name
        inherited_setup_by_start[record.start] = inherited_setup
        ancestor_test_ids = (
            [
                *ancestor_ids_by_start.get(parent.start, []),
                test_id_by_start[parent.start],
            ]
            if parent is not None and parent_name and parent.start in test_id_by_start
            else []
        )
        case = _record(
            path=path,
            start_line=_line_number(content, record.start),
            end_line=_line_number(content, record.end - 1),
            start_column=_column_number(content, record.start),
            end_column=_column_number(content, record.end - 1),
            framework=(
                "gtest"
                if record.canonical_macro in GTEST_MACROS
                else record.canonical_macro.lower()
            ),
            name=display_name,
            code=record.code,
            parent_name=parent_name,
            parent_test_id=(
                test_id_by_start.get(parent.start, "")
                if parent is not None and parent_name
                else ""
            ),
            ancestor_test_ids=ancestor_test_ids,
            inherited_setup=inherited_setup,
        )
        records.append(case)
        test_id_by_start[record.start] = str(case["id"])
        ancestor_ids_by_start[record.start] = ancestor_test_ids

    occupied = [(item.start, item.end) for item in macro_records]
    for match in BOOL_TEST_RE.finditer(search_view):
        if any(start <= match.start() < end for start, end in occupied):
            continue
        block_start = content.find("{", match.start(), match.end())
        block_end = _matching_delimiter(content, block_start, "{", "}")
        if block_start < 0 or block_end is None:
            continue
        records.append(_record(
            path=path,
            start_line=_line_number(content, match.start()),
            end_line=_line_number(content, block_end),
            framework="bool-function",
            name=match.group("name"),
            code=content[match.start() : block_end + 1],
        ))

    main_match = MAIN_RE.search(search_view)
    if main_match is not None and not records:
        block_start = content.find("{", main_match.start(), main_match.end())
        block_end = _matching_delimiter(content, block_start, "{", "}")
        if block_start >= 0 and block_end is not None:
            records.append(_record(
                path=path,
                start_line=_line_number(content, main_match.start()),
                end_line=_line_number(content, block_end),
                framework="standalone",
                name=PurePosixPath(path).stem,
                code=content[main_match.start() : block_end + 1],
            ))

    unique: dict[tuple[int, str], dict[str, Any]] = {}
    for item in records:
        unique[(item["def_start_line"], item["fq_name"])] = item
    return sorted(
        unique.values(),
        key=lambda item: (item["def_start_line"], item["def_end_line"], item["fq_name"]),
    )


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(path))


def _submodule_roots(repo_root: Path) -> tuple[str, ...]:
    gitmodules = repo_root / ".gitmodules"
    if not gitmodules.exists():
        return ()
    roots = []
    for line in gitmodules.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = re.match(r"\s*path\s*=\s*(?P<path>.+?)\s*$", line)
        if match:
            roots.append(match.group("path").strip().strip("/") + "/")
    return tuple(roots)


def _repo_source_paths(repo_root: Path) -> Iterable[Path]:
    submodule_roots = _submodule_roots(repo_root)
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(repo_root).as_posix()
        if any(relative.startswith(root) for root in submodule_roots):
            continue
        if is_repository_test_source(relative):
            yield path


def _repo_project_header_paths(repo_root: Path) -> Iterable[Path]:
    submodule_roots = _submodule_roots(repo_root)
    for path in repo_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".h", ".hh", ".hpp", ".hxx"}:
            continue
        relative = path.relative_to(repo_root).as_posix()
        if (
            any(relative.startswith(root) for root in submodule_roots)
            or is_vendored_or_framework_test(relative)
            or ".git" in PurePosixPath(relative).parts
        ):
            continue
        yield path


def match_source_function_to_test_case(
    cases: list[dict[str, Any]],
    function: dict[str, Any],
) -> dict[str, Any] | None:
    line = int(function.get("def_start_line") or 0)
    containing = [
        case for case in cases
        if case["def_start_line"] <= line <= case["def_end_line"]
    ]
    if not containing:
        return None
    smallest_span = min(
        case["def_end_line"] - case["def_start_line"]
        for case in containing
    )
    candidates = [
        case for case in containing
        if case["def_end_line"] - case["def_start_line"] == smallest_span
    ]
    if len(candidates) == 1:
        return candidates[0]

    column = int(function.get("def_start_column") or 0)
    if column:
        column_matches = [
            case for case in candidates
            if (
                case["def_start_line"] < line
                or int(case.get("def_start_column") or 1) <= column
            )
            and (
                line < case["def_end_line"]
                or column <= int(case.get("def_end_column") or column)
            )
        ]
        if len(column_matches) == 1:
            return column_matches[0]

    return None


def _attach_source_functions(
    test_cases: list[dict[str, Any]],
    functions: list[dict[str, Any]],
    repo_root: Path,
) -> None:
    by_path: dict[str, list[dict[str, Any]]] = {}
    for case in test_cases:
        by_path.setdefault(case["def_file"], []).append(case)
    for function in functions:
        if not function.get("is_test") or not function.get("id"):
            continue
        path = normalize_repo_path(str(function.get("def_file") or ""), repo_root)
        if path not in by_path:
            continue
        case = match_source_function_to_test_case(by_path[path], function)
        if case is None:
            continue
        function_id = str(function.get("id") or "")
        if function_id and function_id not in case["source_func_ids"]:
            case["source_func_ids"].append(function_id)


def _append_ast_test_functions(
    test_cases: list[dict[str, Any]],
    functions: list[dict[str, Any]],
    repo_root: Path,
) -> None:
    by_path: dict[str, list[dict[str, Any]]] = {}
    for case in test_cases:
        by_path.setdefault(case["def_file"], []).append(case)
    for function in functions:
        if not function.get("is_test") or not function.get("id"):
            continue
        path = normalize_repo_path(str(function.get("def_file") or ""), repo_root)
        if not is_repository_test_source(path):
            continue
        start_line = int(function.get("def_start_line") or 0)
        end_line = int(function.get("def_end_line") or start_line)
        if not start_line or any(
            case["def_start_line"] <= start_line <= case["def_end_line"]
            for case in by_path.get(path, [])
        ):
            continue
        name = str(
            function.get("fq_name") or function.get("short_name") or function.get("id") or ""
        )
        code = str(function.get("code") or "")
        if name.rsplit("::", 1)[-1] == "main":
            continue
        if not _oracle_excerpt(code) and "test" not in name.lower():
            continue
        case = _record(
            path=path,
            start_line=start_line,
            end_line=end_line,
            framework="libclang-function",
            name=name,
            code=code,
        )
        case["source_func_ids"] = [str(function["id"])]
        test_cases.append(case)
        by_path.setdefault(path, []).append(case)


def _production_focals(
    test_cases: list[dict[str, Any]],
    functions: list[dict[str, Any]],
    callgraph: Iterable[dict[str, Any]],
    function_intents: list[dict[str, Any]],
) -> None:
    function_by_id = {
        str(function.get("id")): function
        for function in functions
        if function.get("id")
    }
    intent_by_id = {
        str(intent.get("func_id")): intent
        for intent in function_intents
        if intent.get("func_id")
    }
    edges_by_caller: dict[str, list[dict[str, Any]]] = {}
    test_function_ids = {
        source_id
        for case in test_cases
        for source_id in (case.get("source_func_ids") or [])
    }
    for edge in callgraph:
        caller = str(edge.get("caller") or "")
        if caller in test_function_ids:
            edges_by_caller.setdefault(caller, []).append(edge)

    production = [
        function for function in functions
        if not function.get("is_test") and function.get("id")
    ]
    production_by_short: dict[str, dict[str, dict[str, Any]]] = {}
    production_by_qualified: dict[str, dict[str, dict[str, Any]]] = {}
    for target in production:
        target_id = str(target["id"])
        short_name = str(target.get("short_name") or "").lower()
        if len(short_name) >= 4:
            production_by_short.setdefault(short_name, {})[target_id] = target
        components = str(target.get("fq_name") or "").lower().split("::")
        if len(components) >= 2:
            production_by_qualified.setdefault(
                "::".join(components), {}
            )[target_id] = target
            production_by_qualified.setdefault(
                "::".join(components[-2:]), {}
            )[target_id] = target

    for case in test_cases:
        ranked: dict[str, tuple[int, set[str]]] = {}
        for source_id in case.get("source_func_ids") or []:
            for edge in edges_by_caller.get(source_id, []):
                callee_id = str(edge.get("callee") or "")
                target = function_by_id.get(callee_id)
                if not target or target.get("is_test"):
                    continue
                edge_type = str(edge.get("edge_type") or "reference")
                priority = 30 if edge_type == "call" else 20
                previous = ranked.get(callee_id)
                if previous is None or priority > previous[0]:
                    ranked[callee_id] = (priority, {edge_type})
                else:
                    previous[1].add(edge_type)

        code = code_search_view(str(case.get("code") or ""))
        source_candidates: set[str] = set()
        for match in CALL_NAME_RE.finditer(code):
            targets = production_by_short.get(match.group("name").lower(), {})
            if len(targets) == 1:
                source_candidates.update(targets)
        for match in QUALIFIED_NAME_RE.finditer(code):
            targets = production_by_qualified.get(match.group(0).lower(), {})
            if len(targets) == 1:
                source_candidates.update(targets)
        for target_id in source_candidates:
            ranked.setdefault(target_id, (10, {"source_reference"}))

        focal_entities = []
        for target_id, (priority, relations) in sorted(
            ranked.items(),
            key=lambda item: (-item[1][0], str(function_by_id[item[0]].get("fq_name") or item[0])),
        )[:3]:
            target = function_by_id[target_id]
            intent = intent_by_id.get(target_id) or {}
            focal_entities.append({
                "func_id": target_id,
                "name": target.get("fq_name") or target.get("short_name") or target_id,
                "file": target.get("def_file"),
                "start_line": target.get("def_start_line"),
                "relations": sorted(relations),
                "short_intent": intent.get("short_intent") or "",
                "raw_summary": intent.get("raw_summary") or "",
                "source": str(target.get("code") or "")[:1200],
            })
        case["focal_entities"] = focal_entities
        cache_payload = {
            "extractor_version": TEST_CASE_EXTRACTOR_VERSION,
            "identity": case.get("fq_name") or "",
            "code": case.get("code") or "",
            "inherited_setup": case.get("inherited_setup") or "",
            "oracle": case.get("oracle_code") or "",
            "focal_entities": focal_entities,
        }
        case["content_hash"] = hashlib.sha256(
            json.dumps(cache_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()


def collect_test_cases(index_dir: str | Path, repo_root: str | Path) -> list[dict[str, Any]]:
    index_path = Path(index_dir)
    root = Path(repo_root).resolve()
    functions = _load_jsonl(index_path / "functions.jsonl")
    sources: list[tuple[str, str]] = []
    for source_path in _repo_source_paths(root):
        relative = source_path.relative_to(root).as_posix()
        content = source_path.read_text(encoding="utf-8", errors="ignore")
        if len(content) > 1_500_000:
            continue
        sources.append((relative, content))
    header_contents = []
    for header_path in _repo_project_header_paths(root):
        if header_path.stat().st_size > 1_500_000:
            continue
        header_contents.append(
            header_path.read_text(encoding="utf-8", errors="ignore")
        )
    shared_aliases = discover_test_macro_aliases(header_contents)
    test_cases = [
        case
        for relative, content in sources
        for case in extract_test_cases(relative, content, shared_aliases)
    ]

    _append_ast_test_functions(test_cases, functions, root)
    _attach_source_functions(test_cases, functions, root)
    _production_focals(
        test_cases,
        functions,
        _iter_jsonl(index_path / "callgraph.jsonl"),
        _load_jsonl(index_path / "function_intents.jsonl"),
    )
    return test_cases
