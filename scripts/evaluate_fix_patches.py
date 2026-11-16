#!/usr/bin/env python3
"""Evaluate model patches with each Multi-SWE-bench image's fix-run.sh."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_REGISTRY = "6pqg5gclf6yaz1ch6g.xuanyuan.run"
INSTANCE_PATTERN = re.compile(r"^(?P<org>.+?)__(?P<repo>.+)-(?P<number>\d+)$")
FAILED_COUNT_PATTERN = re.compile(
    r"(?P<count>\d+)\s+tests?\s+failed(?:\s+out\s+of\s+\d+)?",
    re.IGNORECASE,
)
FAILED_TEST_PATTERNS = (
    re.compile(r"Test\s+#\d+:\s+(.+?)\s+\.+\s*\*+Failed", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*-\s+(.+?)\s+\((?:Failed|Timeout|Not Run)\)\s*$", re.MULTILINE),
)
EXCLUDED_PATH_PARTS = {"test", "tests", "testing", "tmp", "temp", "home"}
TEST_FILE_STEM_PATTERN = re.compile(
    r"(?:^tests?(?:[._-]|$)|(?:^|[._-])tests?$)",
    re.IGNORECASE,
)
REPRODUCTION_PATH_TOKEN_PATTERN = re.compile(
    r"(?:^|[._-])(?:repro(?:ducer|duction)?|poc)(?:[._-]|$)",
    re.IGNORECASE,
)


def load_patches(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        rows = data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]

    patches = []
    for row in rows:
        instance_id = str(row.get("instance_id", "")).strip()
        model_patch = row.get("model_patch", row.get("patch", ""))
        if not instance_id or not isinstance(model_patch, str):
            raise ValueError("Each row requires instance_id and model_patch")
        patches.append({"instance_id": instance_id, "model_patch": model_patch})
    return patches


def image_for_instance(instance_id: str, registry: str) -> str:
    match = INSTANCE_PATTERN.fullmatch(instance_id)
    if match is None:
        raise ValueError(f"Invalid instance_id: {instance_id}")
    image = (
        f"mswebench/{match['org']}_m_{match['repo']}:pr-{match['number']}"
    ).lower()
    return f"{registry.rstrip('/')}/{image}" if registry else image


def failed_tests_from_log(log: str) -> list[str]:
    failed = []
    for pattern in FAILED_TEST_PATTERNS:
        failed.extend(match.group(1).strip() for match in pattern.finditer(log))
    return list(dict.fromkeys(failed))


def log_has_failed_tests(log: str) -> bool:
    if failed_tests_from_log(log):
        return True
    return any(
        int(match.group("count")) > 0
        for match in FAILED_COUNT_PATTERN.finditer(log)
    )


def patch_file_chunks(patch: str) -> list[tuple[str, str]]:
    chunks: list[list[str]] = []
    current: list[str] = []
    has_old_header = False

    for line in patch.splitlines(keepends=True):
        if line.startswith("diff --git "):
            if current:
                chunks.append(current)
            current = [line]
            has_old_header = False
            continue
        if line.startswith("--- ") and has_old_header:
            chunks.append(current)
            current = [line]
            has_old_header = True
            continue
        if line.startswith("--- "):
            has_old_header = True
        current.append(line)
    if current:
        chunks.append(current)

    parsed = []
    for chunk_lines in chunks:
        old_path = ""
        new_path = ""
        for line in chunk_lines:
            if line.startswith("--- "):
                old_path = line[4:].strip().split("\t", 1)[0]
            elif line.startswith("+++ "):
                new_path = line[4:].strip().split("\t", 1)[0]
                break
        path = new_path if new_path != "/dev/null" else old_path
        if path.startswith(("a/", "b/")):
            path = path[2:]
        if path:
            parsed.append((path, "".join(chunk_lines)))
    return parsed


def test_named_patch_path(path: str) -> bool:
    """Recognize test conventions without matching product names containing test."""
    parts = Path(path.lower()).parts
    if not parts:
        return False

    directories = parts[:-1]
    if any(
        part == "selftest"
        or part.endswith("tests")
        or part.startswith(("test_", "tests_", "test-", "tests-"))
        for part in directories
    ):
        return True

    filename_stem = Path(parts[-1]).stem
    return TEST_FILE_STEM_PATTERN.search(filename_stem) is not None


def excluded_patch_path(path: str) -> bool:
    lowered = path.lower()
    parts = set(Path(lowered).parts)
    return (
        bool(parts & EXCLUDED_PATH_PARTS)
        or test_named_patch_path(path)
        or any(
            REPRODUCTION_PATH_TOKEN_PATTERN.search(part)
            for part in Path(lowered).parts
        )
    )


def filter_model_patch(patch: str) -> tuple[str, list[str], list[str]]:
    kept_chunks = []
    kept_files = []
    excluded_files = []
    for path, chunk in patch_file_chunks(patch):
        if excluded_patch_path(path):
            excluded_files.append(path)
        else:
            kept_files.append(path)
            kept_chunks.append(chunk)
    return "".join(kept_chunks), kept_files, excluded_files


def _patch_hunks(chunk: str) -> tuple[str, list[str]]:
    lines = chunk.splitlines(keepends=True)
    starts = [
        index for index, line in enumerate(lines) if line.startswith("@@ ")
    ]
    if not starts:
        return chunk, []
    preamble = "".join(lines[: starts[0]])
    hunks = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        hunks.append("".join(lines[start:end]))
    return preamble, hunks


def _hunk_signature(hunk: str) -> tuple[int, int, tuple[str, ...]]:
    lines = hunk.splitlines(keepends=True)
    if not lines:
        return 0, 0, ()
    location = re.match(
        r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@",
        lines[0],
    )
    source_start = int(location.group(1)) if location else 0
    target_start = int(location.group(2)) if location else 0
    changes = tuple(
        line
        for line in lines[1:]
        if line.startswith(("+", "-"))
    )
    return source_start, target_start, changes


def deduplicate_test_patch_hunks(
    model_patch: str,
    test_patch: str,
) -> tuple[str, int]:
    """Drop same-location model edits already present in the benchmark tests."""
    test_hunks = {
        (path, _hunk_signature(hunk))
        for path, chunk in patch_file_chunks(test_patch)
        for hunk in _patch_hunks(chunk)[1]
    }
    if not test_hunks:
        return model_patch, 0

    output = []
    removed = 0
    for path, chunk in patch_file_chunks(model_patch):
        preamble, hunks = _patch_hunks(chunk)
        if not hunks:
            output.append(chunk)
            continue
        kept_hunks = []
        for hunk in hunks:
            if (path, _hunk_signature(hunk)) in test_hunks:
                removed += 1
            else:
                kept_hunks.append(hunk)
        if kept_hunks:
            output.append(preamble + "".join(kept_hunks))
    return "".join(output), removed


def remove_model_paths_from_test_patch(
    model_patch: str,
    test_patch: str,
) -> tuple[str, int, list[str]]:
    """Keep the candidate authoritative on every production path it changes."""
    model_paths = {path for path, _ in patch_file_chunks(model_patch)}
    output = []
    removed_hunks = 0
    removed_files = []
    for path, chunk in patch_file_chunks(test_patch):
        if path not in model_paths:
            output.append(chunk)
            continue
        removed_files.append(path)
        hunks = _patch_hunks(chunk)[1]
        removed_hunks += len(hunks) if hunks else 1
    return "".join(output), removed_hunks, removed_files


def dependency_preflight_command(instance_id: str) -> list[str] | None:
    if instance_id.startswith("simdjson__"):
        return [
            "/bin/mkdir",
            "-p",
            "/home/simdjson/dependencies/.cache",
        ]
    return None


def evaluate_patch(
    row: dict[str, str],
    *,
    output_dir: Path,
    registry: str,
    timeout: int,
) -> dict[str, Any]:
    instance_id = row["instance_id"]
    instance_dir = output_dir / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)
    model_patch_path = instance_dir / "model.patch"
    patch_path = instance_dir / "fix.patch"
    log_path = instance_dir / "eval.log"
    result_path = instance_dir / "result.json"
    model_patch_path.write_text(row["model_patch"], encoding="utf-8")
    filtered_patch, kept_files, excluded_files = filter_model_patch(
        row["model_patch"]
    )
    deduplicated_hunks = 0
    removed_test_patch_files: list[str] = []
    dependency_cache_initialized = False

    image = image_for_instance(instance_id, registry)
    safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "-", instance_id)
    container = f"fix-eval-{safe_id}-{os.getpid()}-{time.time_ns()}"
    exit_code = 1
    log = ""
    started = time.monotonic()

    if not filtered_patch.strip():
        log = "No product-source changes remain after patch filtering.\n"
    else:
        try:
            subprocess.run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--network",
                    "host",
                    "--name",
                    container,
                    "--entrypoint",
                    "sleep",
                    image,
                    "infinity",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            original_test_patch_path = (
                instance_dir / "benchmark_test.original.patch"
            )
            benchmark_test_patch_path = instance_dir / "benchmark_test.patch"
            copied_test_patch = subprocess.run(
                [
                    "docker",
                    "cp",
                    f"{container}:/home/test.patch",
                    str(original_test_patch_path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=60,
            )
            if copied_test_patch.returncode == 0:
                sanitized_test_patch, deduplicated_hunks, removed_test_patch_files = (
                    remove_model_paths_from_test_patch(
                        filtered_patch,
                        original_test_patch_path.read_text(encoding="utf-8"),
                    )
                )
                benchmark_test_patch_path.write_text(
                    sanitized_test_patch,
                    encoding="utf-8",
                )
                subprocess.run(
                    [
                        "docker",
                        "cp",
                        str(benchmark_test_patch_path),
                        f"{container}:/home/test.patch",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
            patch_path.write_text(filtered_patch, encoding="utf-8")
            subprocess.run(
                ["docker", "cp", str(patch_path), f"{container}:/home/fix.patch"],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            dependency_command = dependency_preflight_command(instance_id)
            if dependency_command is not None:
                subprocess.run(
                    ["docker", "exec", container, *dependency_command],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                dependency_cache_initialized = True
            completed = subprocess.run(
                ["docker", "exec", container, "/bin/bash", "/home/fix-run.sh"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
            )
            exit_code = completed.returncode
            log = completed.stdout
        except subprocess.TimeoutExpired as exc:
            partial = exc.stdout or ""
            if isinstance(partial, bytes):
                partial = partial.decode(errors="replace")
            log = f"{partial}\nEVALUATION TIMEOUT after {timeout}s\n"
            exit_code = 124
        except subprocess.CalledProcessError as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            log = f"{stdout}\n{stderr}\n"
            exit_code = exc.returncode
        finally:
            subprocess.run(
                ["docker", "rm", "-f", container],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )

    log_path.write_text(log, encoding="utf-8")
    failed_tests = failed_tests_from_log(log)
    has_failed_tests = log_has_failed_tests(log)
    resolved = exit_code == 0 and not has_failed_tests
    result = {
        "instance_id": instance_id,
        "image": image,
        "resolved": resolved,
        "exit_code": exit_code,
        "has_failed_tests": has_failed_tests,
        "failed_tests": failed_tests,
        "kept_files": kept_files,
        "excluded_files": excluded_files,
        "deduplicated_test_patch_hunks": deduplicated_hunks,
        "removed_overlapping_test_patch_files": removed_test_patch_files,
        "dependency_cache_initialized": dependency_cache_initialized,
        "duration_seconds": round(time.monotonic() - started, 2),
        "log_path": str(log_path),
    }
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Multi-SWE-bench model patches with fix-run.sh."
    )
    parser.add_argument("submission", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("fix_patch_eval"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument(
        "--registry",
        default=os.getenv("COTEST_DOCKER_REGISTRY", DEFAULT_REGISTRY),
    )
    args = parser.parse_args()

    rows = load_patches(args.submission)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, args.workers)
    ) as executor:
        futures = {
            executor.submit(
                evaluate_patch,
                row,
                output_dir=args.output_dir,
                registry=args.registry,
                timeout=args.timeout,
            ): row["instance_id"]
            for row in rows
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            status = "RESOLVED" if result["resolved"] else "UNRESOLVED"
            print(
                f"{result['instance_id']}: {status} "
                f"(exit={result['exit_code']}, "
                f"failed_tests={len(result['failed_tests'])}, "
                f"{result['duration_seconds']}s)",
                flush=True,
            )

    order = {row["instance_id"]: index for index, row in enumerate(rows)}
    results.sort(key=lambda result: order[result["instance_id"]])
    summary = {
        "submission": str(args.submission),
        "total": len(results),
        "resolved": sum(result["resolved"] for result in results),
        "unresolved": sum(not result["resolved"] for result in results),
        "results": results,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Summary: {summary_path}")
    return 0 if summary["unresolved"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
