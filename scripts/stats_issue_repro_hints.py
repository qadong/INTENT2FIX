#!/usr/bin/env python3
"""
Statistics for reproduction hints in issue descriptions.

Default input:
  INTENT2FIX/benchmark/multi_swebench_cpp_change.jsonl

Expected text field:
  problem_statement

Outputs:
  - <output_dir>/per_instance.jsonl
  - <output_dir>/summary.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


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


PATTERNS = {
    # Explicit section hints
    "has_repro_steps_section": re.compile(r"\b(reproduction steps|steps to reproduce|how to reproduce)\b", re.I),
    "has_expected_behavior": re.compile(r"\b(expected behavior|expected result)\b", re.I),
    "has_actual_behavior": re.compile(r"\b(actual behavior|observed behavior)\b", re.I),
    # Stacktrace / diagnostic traces
    "has_stacktrace_keyword": re.compile(r"\b(stacktrace|stack trace|backtrace|traceback|call stack)\b", re.I),
    "has_file_line_trace": re.compile(r"[A-Za-z0-9_./\\-]+\.(c|cc|cpp|cxx|h|hpp|py|java|js|ts):\d+", re.I),
    # Commands and outputs
    "has_shell_command": re.compile(r"(^|\n)\s*(\$|#)?\s*(\./|cmake|ctest|make|ninja|pytest|python|cargo|mvn|gradle|bazel)\b", re.I),
    "has_code_block": re.compile(r"```"),
    "has_output_keywords": re.compile(r"\b(output|stderr|stdout|log|trace|dump)\b", re.I),
    # Errors / exceptions
    "has_error_terms": re.compile(r"\b(error|failed|failure|exception|assert|segmentation fault|abort|crash)\b", re.I),
    # Environment / platform
    "has_platform_info": re.compile(r"\b(platform|os|ubuntu|debian|windows|macos|compiler|clang|gcc|version)\b", re.I),
}


def extract_hints(text: str) -> Dict[str, bool]:
    flags = {name: bool(pattern.search(text)) for name, pattern in PATTERNS.items()}
    # "detail output" proxy: output section + code block, or explicit stderr/stdout
    flags["has_detailed_output"] = flags["has_output_keywords"] and flags["has_code_block"]
    # Strong reproducibility signal: commands + errors + steps/code block
    flags["has_strong_repro_signal"] = (
        (flags["has_shell_command"] or flags["has_repro_steps_section"])
        and flags["has_error_terms"]
        and (flags["has_code_block"] or flags["has_file_line_trace"])
    )
    return flags


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue reproduction-hint statistics")
    parser.add_argument(
        "--input-jsonl",
        default="INTENT2FIX/benchmark/multi_swebench_cpp_change.jsonl",
        help="Dataset JSONL path",
    )
    parser.add_argument(
        "--text-field",
        default="problem_statement",
        help="Issue description field name",
    )
    parser.add_argument(
        "--output-dir",
        default="eval_issue_hints",
        help="Output directory",
    )
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = list(read_jsonl(input_path))
    per_instance_path = output_dir / "per_instance.jsonl"

    hint_counter = Counter()
    nonempty_count = 0
    per_rows: List[Dict[str, Any]] = []

    for row in rows:
        instance_id = row.get("instance_id", "unknown")
        text = row.get(args.text_field) or ""
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        has_text = bool(text)
        if has_text:
            nonempty_count += 1

        flags = extract_hints(text) if has_text else {k: False for k in PATTERNS.keys()} | {
            "has_detailed_output": False,
            "has_strong_repro_signal": False,
        }
        for k, v in flags.items():
            if v:
                hint_counter[k] += 1

        per_row = {
            "instance_id": instance_id,
            "has_issue_text": has_text,
            "text_length": len(text),
            **flags,
        }
        per_rows.append(per_row)

    with per_instance_path.open("w", encoding="utf-8") as f:
        for r in per_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    denom = nonempty_count if nonempty_count > 0 else 1
    summary = {
        "input_jsonl": str(input_path),
        "text_field": args.text_field,
        "total_instances": len(rows),
        "instances_with_nonempty_issue_text": nonempty_count,
        "hint_counts": dict(hint_counter),
        "hint_ratios_over_nonempty_issue_text": {
            k: hint_counter.get(k, 0) / denom for k in sorted(set(list(PATTERNS.keys()) + ["has_detailed_output", "has_strong_repro_signal"]))
        },
        "notes": {
            "has_detailed_output": "Proxy heuristic: output-related keywords + fenced code block.",
            "has_strong_repro_signal": "Proxy heuristic: command/steps + error terms + code block/file:line trace.",
        },
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] per-instance: {per_instance_path}")
    print(f"[DONE] summary: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
