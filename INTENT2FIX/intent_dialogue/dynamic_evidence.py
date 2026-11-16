from __future__ import annotations

import hashlib
import json
import re
import shlex
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, ClassVar, Protocol, Type

from pydantic import ConfigDict, Field, PrivateAttr

from INTENT2FIX.actions.action import Action
from INTENT2FIX.actions.schema import ActionArguments, Observation
from INTENT2FIX.runtime.dockerenv import DockerEnv


_COMPILER_NAME = re.compile(
    r"^(?:c\+\+|cc|clang\+\+|clang|g\+\+|gcc)(?:-\d+)?$"
)
_CPP_SUFFIXES = {".cc", ".cpp", ".cxx", ".c++", ".hpp", ".hh", ".hxx"}
_LLVM_VERSIONS = tuple(range(21, 6, -1))


class InvestigationSessionLike(Protocol):
    actions: list[dict]
    evidence: list[Any]
    investigation_files: dict[str, str]

    def record(
        self,
        command: str,
        output: str,
        exit_code: int,
        *,
        prefix: str = "SEARCH",
        path: str = "<container>",
        action: str = "ReadOnlySearch",
        extra: dict | None = None,
    ) -> Any: ...


@dataclass(frozen=True)
class ProbeRun:
    run_id: str
    compile_run_id: str
    execution_run_id: str
    command: str
    compile_tokens: list[str]
    run_tokens: list[str]
    files: dict[str, str]
    output_path: str


class DynamicEvidenceUnavailable(RuntimeError):
    pass


def available_dynamic_evidence_tools(docker_env: DockerEnv) -> set[str]:
    """Return supported contrast collectors without changing the container."""
    try:
        compiler = _discover_tool(docker_env, "clang++")
    except DynamicEvidenceUnavailable:
        return set()
    compiler_major = _clang_major(docker_env, compiler)
    available: set[str] = set()
    try:
        _discover_tool(docker_env, "llvm-profdata", compiler_major)
        _discover_tool(docker_env, "llvm-cov", compiler_major)
    except DynamicEvidenceUnavailable:
        pass
    else:
        available.add("coverage")
    try:
        _discover_tool(docker_env, "llvm-xray", compiler_major)
    except DynamicEvidenceUnavailable:
        pass
    else:
        available.add("xray")
    return available


def _command_segments(command: str) -> list[list[str]]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars="|&;<>()")
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        tokens = list(lexer)
    except ValueError as exc:
        raise DynamicEvidenceUnavailable(
            "The source RUN has invalid shell quoting."
        ) from exc

    segments: list[list[str]] = [[]]
    for token in tokens:
        if token == "&&":
            if not segments[-1]:
                raise DynamicEvidenceUnavailable(
                    "The source RUN contains an empty command segment."
                )
            segments.append([])
            continue
        if token and all(char in "|&;<>()" for char in token):
            raise DynamicEvidenceUnavailable(
                "Dynamic evidence requires a standalone compile followed by "
                "the compiled probe; pipelines and shell control operators "
                "cannot be rebound safely."
            )
        segments[-1].append(token)
    if not segments[-1] or len(segments) > 2:
        raise DynamicEvidenceUnavailable(
            "Dynamic evidence accepts a standalone compile or probe execution, "
            "or one compile followed by its execution joined with `&&`."
        )
    return segments


def _output_path(tokens: list[str]) -> str:
    for index, token in enumerate(tokens):
        if token == "-o" and index + 1 < len(tokens):
            return tokens[index + 1]
        if token.startswith("-o") and len(token) > 2:
            return token[2:]
    return ""


def _is_compiler_command(tokens: list[str]) -> bool:
    if not tokens:
        return False
    compiler = tokens[0].rsplit("/", 1)[-1]
    return bool(_COMPILER_NAME.fullmatch(compiler))


def _same_executable(output_path: str, executed_path: str) -> bool:
    if not output_path or not executed_path:
        return False
    if output_path.startswith("/") or executed_path.startswith("/"):
        return output_path == executed_path
    return output_path.removeprefix("./") == executed_path.removeprefix("./")


def _single_segment(action: dict) -> list[str] | None:
    try:
        segments = _command_segments(str(action.get("command", "")))
    except DynamicEvidenceUnavailable:
        return None
    return segments[0] if len(segments) == 1 else None


def _referenced_probe_files(action: dict, compile_tokens: list[str]) -> dict[str, str]:
    snapshot = dict(action.get("frozen_probe_files") or {})
    normalized_tokens = {token.removeprefix("./") for token in compile_tokens}
    return {
        path: content
        for path, content in snapshot.items()
        if path in normalized_tokens
        or any(token.endswith("/" + path) for token in normalized_tokens)
    }


def _resolve_probe_run(
    session: InvestigationSessionLike,
    run_id: str,
) -> ProbeRun:
    run_actions = [
        (index, item)
        for index, item in enumerate(session.actions)
        if item.get("action") == "RunInvestigationCommand"
    ]
    selected = next(
        (
            (index, item)
            for index, item in run_actions
            if item.get("evidence_id") == run_id
        ),
        None,
    )
    action = selected[1] if selected else None
    if action is None:
        raise DynamicEvidenceUnavailable(
            f"{run_id} is not a RunInvestigationCommand observation."
        )
    selected_index = selected[0]
    command = str(action.get("command", ""))
    segments = _command_segments(command)
    compile_action = action
    execution_action = action

    if len(segments) == 2:
        compile_tokens, run_tokens = segments
    elif _is_compiler_command(segments[0]):
        compile_tokens = segments[0]
        output_path = _output_path(compile_tokens)
        match = next(
            (
                candidate
                for index, candidate in run_actions
                if index > selected_index
                and (tokens := _single_segment(candidate)) is not None
                and not _is_compiler_command(tokens)
                and _same_executable(output_path, tokens[0])
            ),
            None,
        )
        if match is None:
            raise DynamicEvidenceUnavailable(
                f"{run_id} compiles a probe but no later RUN-N executes its binary."
            )
        execution_action = match
        run_tokens = _single_segment(match) or []
    else:
        run_tokens = segments[0]
        match = next(
            (
                candidate
                for index, candidate in reversed(run_actions)
                if index < selected_index
                and (tokens := _single_segment(candidate)) is not None
                and _is_compiler_command(tokens)
                and _same_executable(_output_path(tokens), run_tokens[0])
            ),
            None,
        )
        if match is None:
            raise DynamicEvidenceUnavailable(
                f"{run_id} executes a probe but no earlier RUN-N compiled its binary."
            )
        compile_action = match
        compile_tokens = _single_segment(match) or []

    if not _is_compiler_command(compile_tokens):
        raise DynamicEvidenceUnavailable(
            f"{run_id} is not bound to a supported C/C++ compiler command."
        )
    output_path = _output_path(compile_tokens)
    if not output_path or not run_tokens or not _same_executable(
        output_path, run_tokens[0]
    ):
        raise DynamicEvidenceUnavailable(
            f"{run_id} must execute the binary produced by its compile command."
        )

    referenced_files = _referenced_probe_files(compile_action, compile_tokens)
    if not referenced_files:
        raise DynamicEvidenceUnavailable(
            f"{run_id} has no frozen WriteInvestigationFile source in its "
            "compile command."
        )
    compile_run_id = str(compile_action.get("evidence_id", run_id))
    execution_run_id = str(execution_action.get("evidence_id", run_id))
    return ProbeRun(
        run_id=run_id,
        compile_run_id=compile_run_id,
        execution_run_id=execution_run_id,
        command=(
            shlex.join(compile_tokens) + " && " + shlex.join(run_tokens)
        ),
        compile_tokens=compile_tokens,
        run_tokens=run_tokens,
        files=referenced_files,
        output_path=output_path,
    )


def _tool_candidates(name: str, preferred_version: int | None = None) -> str:
    mounted = f"/opt/intent2fix-llvm/bin/{name}"
    versioned = [f"{name}-{version}" for version in _LLVM_VERSIONS]
    if preferred_version in _LLVM_VERSIONS:
        preferred = f"{name}-{preferred_version}"
        versioned = [preferred, *(item for item in versioned if item != preferred)]
    candidates = [
        mounted,
        name,
        *versioned,
    ]
    return " ".join(shlex.quote(candidate) for candidate in candidates)


def _discover_tool(
    docker_env: DockerEnv,
    name: str,
    preferred_version: int | None = None,
) -> str:
    command = (
        f"for tool in {_tool_candidates(name, preferred_version)}; do "
        "if command -v \"$tool\" >/dev/null 2>&1; then "
        "command -v \"$tool\"; exit 0; fi; done; exit 1"
    )
    output, exit_code = docker_env.run_bash(command, timeout=30)
    if exit_code != 0 or not output.strip():
        raise DynamicEvidenceUnavailable(
            f"{name} is unavailable in this benchmark container. Continue "
            "with source and ordinary execution evidence."
        )
    return output.strip().splitlines()[-1]


def _clang_major(docker_env: DockerEnv, compiler: str) -> int | None:
    output, exit_code = docker_env.run_bash(
        f"{shlex.quote(compiler)} --version",
        timeout=30,
    )
    if exit_code != 0:
        return None
    match = re.search(r"\bclang version (\d+)", output, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _discover_clang(docker_env: DockerEnv, runs: list[ProbeRun]) -> str:
    is_cpp = any(
        PurePosixPath(path).suffix.lower() in _CPP_SUFFIXES
        for run in runs
        for path in run.files
    ) or any("++" in run.compile_tokens[0] for run in runs)
    return _discover_tool(docker_env, "clang++" if is_cpp else "clang")


def _instrumented_compile_tokens(
    run: ProbeRun,
    compiler: str,
    output_path: str,
    flags: list[str],
) -> list[str]:
    tokens = [compiler]
    skip_next = False
    for token in run.compile_tokens[1:]:
        if skip_next:
            skip_next = False
            continue
        if token == "-o":
            skip_next = True
            continue
        if token.startswith("-o") and len(token) > 2:
            continue
        if token.startswith(
            (
                "-fprofile-instr-generate",
                "-fcoverage-mapping",
                "-fxray-instrument",
                "-fxray-instruction-threshold",
            )
        ):
            continue
        tokens.append(token)
    return [*tokens, *flags, "-o", output_path]


def _instrumented_run_tokens(run: ProbeRun, output_path: str) -> list[str]:
    return [output_path, *run.run_tokens[1:]]


def _fingerprint(runs: list[ProbeRun], kind: str) -> str:
    digest = hashlib.sha256(kind.encode())
    for run in runs:
        digest.update(run.run_id.encode())
        digest.update(run.command.encode())
        for path, content in sorted(run.files.items()):
            digest.update(path.encode())
            digest.update(content.encode())
    return digest.hexdigest()[:12]


def _write_snapshot(docker_env: DockerEnv, run: ProbeRun) -> None:
    root = docker_env.work_dir.rstrip("/")
    for path, content in run.files.items():
        docker_env.write_file(f"{root}/{path}", content)


def _restore_current_files(
    docker_env: DockerEnv,
    session: InvestigationSessionLike,
    paths: set[str],
) -> None:
    root = docker_env.work_dir.rstrip("/")
    for path in paths:
        content = session.investigation_files.get(path)
        if content is not None:
            docker_env.write_file(f"{root}/{path}", content)


def _compile_instrumented(
    docker_env: DockerEnv,
    session: InvestigationSessionLike,
    run: ProbeRun,
    compiler: str,
    output_path: str,
    flags: list[str],
    timeout: int,
) -> None:
    _write_snapshot(docker_env, run)
    command = shlex.join(
        _instrumented_compile_tokens(
            run,
            compiler,
            output_path,
            flags,
        )
    )
    try:
        output, exit_code = docker_env.run_bash(command, timeout=timeout)
    finally:
        _restore_current_files(docker_env, session, set(run.files))
    if exit_code != 0:
        excerpt = output.strip()[-2000:]
        raise DynamicEvidenceUnavailable(
            f"{run.run_id} cannot be instrumented because its probe does not "
            f"compile with Clang. This is usually a compile-time contrast, so "
            f"use its ordinary diagnostic instead.\n{excerpt}"
        )


def _repo_path(path: str, work_dir: str) -> str | None:
    root = work_dir.rstrip("/")
    if path.startswith(root + "/"):
        return path[len(root) + 1 :]
    if path.startswith("/") or path.startswith("../"):
        return None
    return path.removeprefix("./")


def _parse_coverage_export(
    payload: str,
    work_dir: str,
    probe_paths: set[str],
) -> dict[str, Any]:
    try:
        document = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise DynamicEvidenceUnavailable(
            "llvm-cov returned malformed JSON output."
        ) from exc
    data = document.get("data") or []
    if not data:
        raise DynamicEvidenceUnavailable("llvm-cov returned no coverage data.")

    lines: set[tuple[str, int]] = set()
    functions: set[str] = set()
    for file_data in data[0].get("files", []):
        path = _repo_path(str(file_data.get("filename", "")), work_dir)
        if not path or path in probe_paths:
            continue
        for segment in file_data.get("segments", []):
            if len(segment) < 4:
                continue
            line, count, has_count = segment[0], segment[2], segment[3]
            if has_count and count > 0:
                lines.add((path, int(line)))
    for function in data[0].get("functions", []):
        if function.get("count", 0) <= 0:
            continue
        filenames = {
            _repo_path(str(path), work_dir)
            for path in function.get("filenames", [])
        }
        if not any(path and path not in probe_paths for path in filenames):
            continue
        name = str(function.get("name", "")).strip()
        if name:
            functions.add(name)
    return {"lines": lines, "functions": functions}


def _compact_line_ranges(
    locations: set[tuple[str, int]],
    *,
    limit: int = 60,
) -> list[str]:
    by_path: dict[str, list[int]] = {}
    for path, line in locations:
        by_path.setdefault(path, []).append(line)
    ranges: list[str] = []
    for path in sorted(by_path):
        numbers = sorted(set(by_path[path]))
        if not numbers:
            continue
        start = previous = numbers[0]
        for number in numbers[1:] + [numbers[-1] + 2]:
            if number == previous + 1:
                previous = number
                continue
            ranges.append(
                f"{path}:{start}" if start == previous else f"{path}:{start}-{previous}"
            )
            if len(ranges) >= limit:
                return ranges
            start = previous = number
    return ranges


def _coverage_difference(
    anchor: dict[str, Any],
    target: dict[str, Any],
) -> dict[str, Any]:
    anchor_lines = anchor["lines"]
    target_lines = target["lines"]
    anchor_functions = anchor["functions"]
    target_functions = target["functions"]
    return {
        "common_locations": _compact_line_ranges(anchor_lines & target_lines),
        "anchor_only_locations": _compact_line_ranges(
            anchor_lines - target_lines
        ),
        "target_only_locations": _compact_line_ranges(
            target_lines - anchor_lines
        ),
        "common_functions": sorted(anchor_functions & target_functions)[:40],
        "anchor_only_functions": sorted(
            anchor_functions - target_functions
        )[:40],
        "target_only_functions": sorted(
            target_functions - anchor_functions
        )[:40],
    }


def _collect_coverage_contrast(
    docker_env: DockerEnv,
    session: InvestigationSessionLike,
    anchor: ProbeRun,
    target: ProbeRun,
    timeout: int,
) -> dict[str, Any]:
    runs = [anchor, target]
    compiler = _discover_clang(docker_env, runs)
    compiler_major = _clang_major(docker_env, compiler)
    llvm_profdata = _discover_tool(
        docker_env,
        "llvm-profdata",
        compiler_major,
    )
    llvm_cov = _discover_tool(docker_env, "llvm-cov", compiler_major)
    fingerprint = _fingerprint(runs, "coverage")
    instrumented: dict[str, dict[str, Any]] = {}

    for role, run in (("anchor", anchor), ("target", target)):
        binary = f"/tmp/intent2fix-cov-{fingerprint}-{role}"
        raw_profile = f"/tmp/intent2fix-cov-{fingerprint}-{role}.profraw"
        profile = f"/tmp/intent2fix-cov-{fingerprint}-{role}.profdata"
        _compile_instrumented(
            docker_env,
            session,
            run,
            compiler,
            binary,
            ["-fprofile-instr-generate", "-fcoverage-mapping"],
            timeout,
        )
        run_command = shlex.join(_instrumented_run_tokens(run, binary))
        output, exit_code = docker_env.run_bash(
            f"rm -f {shlex.quote(raw_profile)} && "
            f"LLVM_PROFILE_FILE={shlex.quote(raw_profile)} {run_command}",
            timeout=timeout,
        )
        _, exists_code = docker_env.run_bash(
            f"test -s {shlex.quote(raw_profile)}",
            timeout=30,
        )
        if exists_code != 0:
            raise DynamicEvidenceUnavailable(
                f"{run.run_id} produced no coverage profile. The probe may "
                "terminate before LLVM can flush coverage data."
            )
        merge_output, merge_code = docker_env.run_bash(
            shlex.join(
                [
                    llvm_profdata,
                    "merge",
                    "-sparse",
                    raw_profile,
                    "-o",
                    profile,
                ]
            ),
            timeout=timeout,
        )
        if merge_code != 0:
            raise DynamicEvidenceUnavailable(
                "llvm-profdata could not merge the profile:\n"
                + merge_output.strip()[-2000:]
            )
        export_output, export_code = docker_env.run_bash(
            shlex.join(
                [
                    llvm_cov,
                    "export",
                    binary,
                    f"-instr-profile={profile}",
                    "-ignore-filename-regex=(^/usr/|^/opt/|^/lib/|^/tmp/)",
                ]
            ),
            timeout=timeout,
        )
        if export_code != 0:
            raise DynamicEvidenceUnavailable(
                "llvm-cov could not export the profile:\n"
                + export_output.strip()[-2000:]
            )
        instrumented[role] = {
            "exit_code": exit_code,
            "output_excerpt": output.strip()[-1200:],
            "coverage": _parse_coverage_export(
                export_output,
                docker_env.work_dir,
                set(run.files),
            ),
        }

    if instrumented["anchor"]["exit_code"] != 0:
        raise DynamicEvidenceUnavailable(
            "The instrumented anchor no longer passes, so its coverage cannot "
            "serve as a controlled baseline."
        )
    if instrumented["target"]["exit_code"] == 0:
        raise DynamicEvidenceUnavailable(
            "The instrumented target no longer fails, so the dynamic contrast "
            "does not preserve the original witness."
        )
    return {
        "kind": "llvm_coverage_contrast",
        "source_run_ids": [anchor.run_id, target.run_id],
        "source_run_bindings": {
            "anchor": {
                "requested": anchor.run_id,
                "compile": anchor.compile_run_id,
                "execution": anchor.execution_run_id,
            },
            "target": {
                "requested": target.run_id,
                "compile": target.compile_run_id,
                "execution": target.execution_run_id,
            },
        },
        "source_fingerprint": fingerprint,
        "instrumented_exit_codes": {
            role: value["exit_code"] for role, value in instrumented.items()
        },
        **_coverage_difference(
            instrumented["anchor"]["coverage"],
            instrumented["target"]["coverage"],
        ),
        "interpretation_boundary": (
            "Coverage includes only translation units rebuilt by this tool; "
            "prebuilt linked objects are not instrumented. Observed locations "
            "do not prove repair scope or product intent."
        ),
    }


def _parse_xray_trace(payload: str) -> list[str]:
    try:
        document = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise DynamicEvidenceUnavailable(
            "llvm-xray returned malformed trace-event JSON."
        ) from exc
    events = (
        document
        if isinstance(document, list)
        else document.get("traceEvents", [])
    )
    sequence: list[str] = []
    def timestamp(event: dict) -> float:
        try:
            return float(event.get("ts", 0))
        except (TypeError, ValueError):
            return 0.0

    for event in sorted(events, key=timestamp):
        if event.get("ph") not in {"B", "X"}:
            continue
        name = str(event.get("name", "")).strip()
        if not name or name == "[unknown]":
            continue
        if not sequence or sequence[-1] != name:
            sequence.append(name)
        if len(sequence) >= 2000:
            break
    if not sequence:
        raise DynamicEvidenceUnavailable(
            "llvm-xray produced no symbolized function-entry events."
        )
    return sequence


def _trace_difference(anchor: list[str], target: list[str]) -> dict[str, Any]:
    common_length = 0
    for left, right in zip(anchor, target):
        if left != right:
            break
        common_length += 1
    return {
        "common_prefix": anchor[max(0, common_length - 40) : common_length],
        "common_prefix_length": common_length,
        "first_divergence": {
            "anchor": (
                anchor[common_length] if common_length < len(anchor) else "<ended>"
            ),
            "target": (
                target[common_length] if common_length < len(target) else "<ended>"
            ),
        },
        "anchor_after_divergence": anchor[common_length : common_length + 40],
        "target_after_divergence": target[common_length : common_length + 40],
    }


def _collect_xray_contrast(
    docker_env: DockerEnv,
    session: InvestigationSessionLike,
    anchor: ProbeRun,
    target: ProbeRun,
    timeout: int,
) -> dict[str, Any]:
    runs = [anchor, target]
    compiler = _discover_clang(docker_env, runs)
    llvm_xray = _discover_tool(
        docker_env,
        "llvm-xray",
        _clang_major(docker_env, compiler),
    )
    fingerprint = _fingerprint(runs, "xray")
    instrumented: dict[str, dict[str, Any]] = {}

    for role, run in (("anchor", anchor), ("target", target)):
        binary = f"/tmp/intent2fix-xray-{fingerprint}-{role}"
        log_prefix = f"/tmp/intent2fix-xray-{fingerprint}-{role}-"
        _compile_instrumented(
            docker_env,
            session,
            run,
            compiler,
            binary,
            ["-g", "-fxray-instrument", "-fxray-instruction-threshold=1"],
            timeout,
        )
        run_command = shlex.join(_instrumented_run_tokens(run, binary))
        output, exit_code = docker_env.run_bash(
            f"rm -f {shlex.quote(log_prefix)}* && "
            "XRAY_BASIC_OPTIONS=func_duration_threshold_us=0 "
            "XRAY_OPTIONS="
            + shlex.quote(
                "patch_premain=true xray_mode=xray-basic "
                f"xray_logfile_base={log_prefix}"
            )
            + f" {run_command}",
            timeout=timeout,
        )
        log_output, log_code = docker_env.run_bash(
            f"ls -1t {shlex.quote(log_prefix)}* 2>/dev/null | head -1",
            timeout=30,
        )
        log_path = log_output.strip().splitlines()[0] if log_output.strip() else ""
        if log_code != 0 or not log_path:
            raise DynamicEvidenceUnavailable(
                f"{run.run_id} produced no XRay log. The container runtime or "
                "selected target may not support XRay instrumentation."
            )
        trace_output, trace_code = docker_env.run_bash(
            shlex.join(
                [
                    llvm_xray,
                    "convert",
                    "--symbolize",
                    f"--instr_map={binary}",
                    "--output-format=trace_event",
                    log_path,
                ]
            ),
            timeout=timeout,
        )
        if trace_code != 0:
            raise DynamicEvidenceUnavailable(
                "llvm-xray could not symbolize the trace:\n"
                + trace_output.strip()[-2000:]
            )
        instrumented[role] = {
            "exit_code": exit_code,
            "output_excerpt": output.strip()[-1200:],
            "sequence": _parse_xray_trace(trace_output),
        }

    if instrumented["anchor"]["exit_code"] != 0:
        raise DynamicEvidenceUnavailable(
            "The instrumented anchor no longer passes, so its trace cannot "
            "serve as a controlled baseline."
        )
    if instrumented["target"]["exit_code"] == 0:
        raise DynamicEvidenceUnavailable(
            "The instrumented target no longer fails, so the dynamic contrast "
            "does not preserve the original witness."
        )
    return {
        "kind": "llvm_xray_trace_contrast",
        "source_run_ids": [anchor.run_id, target.run_id],
        "source_run_bindings": {
            "anchor": {
                "requested": anchor.run_id,
                "compile": anchor.compile_run_id,
                "execution": anchor.execution_run_id,
            },
            "target": {
                "requested": target.run_id,
                "compile": target.compile_run_id,
                "execution": target.execution_run_id,
            },
        },
        "source_fingerprint": fingerprint,
        "instrumented_exit_codes": {
            role: value["exit_code"] for role, value in instrumented.items()
        },
        **_trace_difference(
            instrumented["anchor"]["sequence"],
            instrumented["target"]["sequence"],
        ),
        "interpretation_boundary": (
            "XRay includes only translation units rebuilt by this tool; "
            "prebuilt linked objects are not instrumented. The observed "
            "call-order divergence does not prove the only valid repair "
            "location or product intent."
        ),
    }


class CollectCoverageContrastArgs(ActionArguments):
    anchor_run_id: str = Field(
        ...,
        pattern=r"^RUN-\d+$",
        description=(
            "RUN-N for the passing anchor. It may be a combined `compile && "
            "run` record, or either half of adjacent compile and execution "
            "records for the same probe binary."
        ),
    )
    target_run_id: str = Field(
        ...,
        pattern=r"^RUN-\d+$",
        description=(
            "RUN-N for the failing Issue-mutated probe. It may be a combined "
            "record, or either half of adjacent compile and execution records "
            "for the same probe binary."
        ),
    )
    timeout: int = Field(default=300, ge=30, le=600)

    model_config = ConfigDict(title="CollectCoverageContrast")


class CollectXRayTraceContrastArgs(ActionArguments):
    anchor_run_id: str = Field(
        ...,
        pattern=r"^RUN-\d+$",
        description=(
            "RUN-N for the passing anchor, either combined or bound to its "
            "separate compile/execution RUN-N counterpart."
        ),
    )
    target_run_id: str = Field(
        ...,
        pattern=r"^RUN-\d+$",
        description=(
            "RUN-N for the failing Issue-mutated probe, either combined or "
            "bound to its separate compile/execution RUN-N counterpart."
        ),
    )
    timeout: int = Field(default=300, ge=30, le=600)

    model_config = ConfigDict(title="CollectXRayTraceContrast")


class _DynamicContrastAction(Action):
    _docker_env: DockerEnv = PrivateAttr()
    _session: InvestigationSessionLike = PrivateAttr()

    evidence_prefix: ClassVar[str]
    evidence_action: ClassVar[str]

    def __init__(
        self,
        docker_env: DockerEnv,
        session: InvestigationSessionLike,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._docker_env = docker_env
        self._session = session

    def _collect(
        self,
        anchor: ProbeRun,
        target: ProbeRun,
        timeout: int,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def execute(self, args: ActionArguments, **kwargs) -> Observation:
        anchor_run_id = getattr(args, "anchor_run_id")
        target_run_id = getattr(args, "target_run_id")
        if anchor_run_id == target_run_id:
            return Observation(
                message="Anchor and target must cite different RUN-N records.",
                expect_correction=True,
            )
        try:
            anchor = _resolve_probe_run(self._session, anchor_run_id)
            target = _resolve_probe_run(self._session, target_run_id)
            result = self._collect(anchor, target, getattr(args, "timeout"))
        except DynamicEvidenceUnavailable as exc:
            return Observation(
                message=f"Dynamic contrast unavailable: {exc}",
                properties={"available": False},
            )

        rendered = json.dumps(result, ensure_ascii=False, indent=2)
        evidence = self._session.record(
            f"{self.evidence_action}({anchor_run_id}, {target_run_id})",
            rendered,
            0,
            prefix=self.evidence_prefix,
            path="<dynamic-contrast>",
            action=self.evidence_action,
            extra={
                "source_run_ids": [anchor_run_id, target_run_id],
                "source_fingerprint": result["source_fingerprint"],
                "result": result,
            },
        )
        return Observation(
            message=f"[{evidence.id}]\n{rendered}",
            properties={
                "available": True,
                "evidence_id": evidence.id,
                "source_run_ids": [anchor_run_id, target_run_id],
                "result": result,
            },
        )


class CollectCoverageContrast(_DynamicContrastAction):
    args_schema: ClassVar[Type[ActionArguments]] = CollectCoverageContrastArgs
    evidence_prefix = "COVERAGE"
    evidence_action = "CollectCoverageContrast"

    def _collect(
        self,
        anchor: ProbeRun,
        target: ProbeRun,
        timeout: int,
    ) -> dict[str, Any]:
        return _collect_coverage_contrast(
            self._docker_env,
            self._session,
            anchor,
            target,
            timeout,
        )


class CollectXRayTraceContrast(_DynamicContrastAction):
    args_schema: ClassVar[Type[ActionArguments]] = CollectXRayTraceContrastArgs
    evidence_prefix = "XRAY"
    evidence_action = "CollectXRayTraceContrast"

    def _collect(
        self,
        anchor: ProbeRun,
        target: ProbeRun,
        timeout: int,
    ) -> dict[str, Any]:
        return _collect_xray_contrast(
            self._docker_env,
            self._session,
            anchor,
            target,
            timeout,
        )
