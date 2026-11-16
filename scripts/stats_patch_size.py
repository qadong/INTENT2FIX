#!/usr/bin/env python3
"""
Compute patch size statistics from a JSONL file.

Default input format matches files like:
  patch/gpt5_2_cotest_patch.jsonl
where each line may contain a patch in `fix_patch`.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Set, Tuple


DIFF_GIT_RE = re.compile(r"^diff --git a/(.*?) b/(.*?)$")


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e


def parse_patch_stats(patch_text: str) -> Tuple[int, int, int]:
    """
    Returns:
      (files_changed, lines_added, lines_deleted)
    """
    files: Set[str] = set()
    added = 0
    deleted = 0

    in_hunk = False
    for raw_line in patch_text.splitlines():
        line = raw_line.rstrip("\n")

        m = DIFF_GIT_RE.match(line)
        if m:
            in_hunk = False
            # Prefer target path (b/...)
            files.add(m.group(2))
            continue

        if line.startswith("@@"):
            in_hunk = True
            continue

        if line.startswith("--- ") or line.startswith("+++ "):
            # fallback file extraction for patches without "diff --git"
            if line.startswith("+++ b/"):
                files.add(line[6:])
            in_hunk = False
            continue

        if not in_hunk:
            continue

        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            deleted += 1

    return len(files), added, deleted


def summarize_patch_list(patches: List[str]) -> Dict[str, Any]:
    file_counts: List[int] = []
    add_counts: List[int] = []
    del_counts: List[int] = []
    total_changed_counts: List[int] = []

    for patch_text in patches:
        files_changed, lines_added, lines_deleted = parse_patch_stats(patch_text)
        file_counts.append(files_changed)
        add_counts.append(lines_added)
        del_counts.append(lines_deleted)
        total_changed_counts.append(lines_added + lines_deleted)

    valid = len(file_counts)
    return {
        "valid_patch_rows": valid,
        "files_changed": {
            "avg": mean(file_counts) if valid else 0.0,
            "median": median(file_counts) if valid else 0.0,
        },
        "lines_added": {
            "avg": mean(add_counts) if valid else 0.0,
            "median": median(add_counts) if valid else 0.0,
        },
        "lines_deleted": {
            "avg": mean(del_counts) if valid else 0.0,
            "median": median(del_counts) if valid else 0.0,
        },
        "lines_changed_total": {
            "avg": mean(total_changed_counts) if valid else 0.0,
            "median": median(total_changed_counts) if valid else 0.0,
        },
    }


def build_benchmark_map(benchmark_jsonl: Path) -> Dict[str, Dict[str, Any]]:
    bench_map: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(benchmark_jsonl):
        repo = row.get("repo")
        pull_number = row.get("pull_number")
        if not repo or pull_number is None:
            continue
        key = f"{repo}#{pull_number}"
        bench_map[key] = row
    return bench_map


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch size statistics")
    parser.add_argument(
        "--input-jsonl",
        default="patch/gpt5_2_cotest_patch.jsonl",
        help="Path to JSONL containing generated patches",
    )
    parser.add_argument(
        "--patch-field",
        default="fix_patch",
        help="Field name that stores patch text (e.g., fix_patch, solution_patch, patch)",
    )
    parser.add_argument(
        "--output-json",
        default="patch_size_summary.json",
        help="Path to save summary JSON",
    )
    parser.add_argument(
        "--benchmark-jsonl",
        default="",
        help="Optional benchmark JSONL to also compute golden patch stats (e.g., INTENT2FIX/benchmark/multi_swebench_cpp_change.jsonl)",
    )
    parser.add_argument(
        "--golden-patch-field",
        default="patch",
        help="Golden patch field name in benchmark JSONL",
    )
    args = parser.parse_args()

    rows = list(read_jsonl(Path(args.input_jsonl)))
    generated_patches: List[str] = []

    missing_patch = 0
    for row in rows:
        patch_text = row.get(args.patch_field)
        if not patch_text or not isinstance(patch_text, str):
            missing_patch += 1
            continue
        generated_patches.append(patch_text)

    summary = {
        "input_jsonl": args.input_jsonl,
        "patch_field": args.patch_field,
        "total_rows": len(rows),
        "missing_or_empty_patch_rows": missing_patch,
        "generated_patch_stats": summarize_patch_list(generated_patches),
    }

    if args.benchmark_jsonl:
        benchmark_path = Path(args.benchmark_jsonl)
        bench_map = build_benchmark_map(benchmark_path)

        golden_patches: List[str] = []
        unmatched_rows = 0

        for row in rows:
            org = row.get("org")
            repo_name = row.get("repo")
            number = row.get("number")
            if not org or not repo_name or number is None:
                continue

            key = f"{org}/{repo_name}#{number}"
            bench_row = bench_map.get(key)
            if not bench_row:
                unmatched_rows += 1
                continue

            gp = bench_row.get(args.golden_patch_field)
            if isinstance(gp, str) and gp:
                golden_patches.append(gp)

        summary["golden_source"] = {
            "benchmark_jsonl": args.benchmark_jsonl,
            "golden_patch_field": args.golden_patch_field,
            "matched_instances": len(golden_patches),
            "unmatched_instances": unmatched_rows,
        }
        summary["golden_patch_stats"] = summarize_patch_list(golden_patches)

    out_path = Path(args.output_json)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[OK] Wrote summary to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
