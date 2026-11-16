"""Offline audit of baseline golden-edit lines shown in saved Scout tool outputs.

This measures source exposure, not causal understanding or patch correctness.
Golden data is read only by this offline script, never by an agent or API.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shlex
import subprocess
from pathlib import Path

from unidiff import PatchSet


ROOT = Path(__file__).resolve().parents[1]
DATASET = Path("/data/dqa/multi-swe-bench/dataset/multi_swebench_cpp.jsonl")
BASELINE_REPOS = Path("/data/dqa/auto_swebench_cxx/data")
RUNS = (
    "hypothesis_v4_downstream10_unified_four_graph_compare_low_20260925/graph_on",
    "fullflow_graph_scope_v5_paired10_low_20260926/graph_on",
    "fullflow_graph_scope_v5_round3_paired10_low_20260926/graph_on",
    "fullflow_graph_scope_v5_round4_paired10_low_20260926/graph_on",
    "fullflow_graph_scope_v5_remaining88_paired_low_20260927/graph_on",
    "fullflow_verification_v7_remaining34_paired_low_20260928/graph_on",
    "graph_pruning_retry4_scout_hypothesis_repair_low_20260926",
    "fullflow_verification_v7_retry_hypothesis3_low_20260928/graph_on",
    "fullflow_v112_regression10_graph_on_low_20260928/graph_on",
)
PAIRED_ON_RUNS = tuple(run for run in RUNS if not run.startswith("graph_pruning_retry4_"))
PAIRED_OFF_RUNS = tuple(
    run.replace("fullflow_v112_regression10_graph_on_low", "fullflow_v112_regression10_graph_off_low")
    .replace("/graph_on", "/graph_off")
    for run in PAIRED_ON_RUNS
)
COHORT_RUNS = {
    "on-latest": RUNS,
    "on-paired": PAIRED_ON_RUNS,
    "off-paired": PAIRED_OFF_RUNS,
}
OMITTED = re.compile(r"^\.\.\. \[(\d+) lines omitted\] \.\.\.$")
NUMBERED = re.compile(r"^(\d+):(.*)$")
PATH_NUMBERED = re.compile(r"^(.+?):(\d+):(.*)$")
EDIT_LINE = tuple[str, int, str]
CPP_SUFFIXES = (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".hxx", ".ipp", ".inl", ".re2c", ".h.in")
NONPRODUCTION_DIRS = {"test", "tests", "doc", "docs", "example", "examples", "benchmark", "benchmarks"}


def normalized_code(value: str) -> str:
    return value.rstrip("\r\n").rstrip()


def is_production_cpp(path: str) -> bool:
    return path.endswith(CPP_SUFFIXES) and not (set(Path(path).parts) & NONPRODUCTION_DIRS)


def baseline_hunks(patch: str) -> tuple[list[dict], int]:
    """Return one old-code target set per hunk; count unmappable hunks."""
    hunks = []
    unmappable = 0
    for changed_file in PatchSet(io.StringIO(patch)):
        if changed_file.is_added_file:
            unmappable += len(changed_file)
            continue
        path = changed_file.source_file.removeprefix("a/")
        for hunk in changed_file:
            removed = {
                (path, line.source_line_no, normalized_code(line.value))
                for line in hunk
                if line.is_removed and line.source_line_no and line.value.strip()
            }
            if removed:
                targets = removed
                kind = "preimage"
            else:
                # Insertion-only edits have no deleted line. Use the nearest
                # unchanged old-code line on either side of each added block.
                targets = set()
                lines = list(hunk)
                for position, line in enumerate(lines):
                    if not line.is_added or (position and lines[position - 1].is_added):
                        continue
                    end = position
                    while end + 1 < len(lines) and lines[end + 1].is_added:
                        end += 1
                    for neighbor in (position - 1, end + 1):
                        if 0 <= neighbor < len(lines):
                            adjacent = lines[neighbor]
                            if adjacent.is_context and adjacent.source_line_no:
                                targets.add((
                                    path,
                                    adjacent.source_line_no,
                                    normalized_code(adjacent.value),
                                ))
                kind = "insertion_anchor"
            if not targets:
                unmappable += 1
                continue
            hunks.append({
                "path": path,
                "old_start": hunk.source_start,
                "kind": kind,
                "old_lines": [
                    normalized_code(line.value)
                    for line in hunk if line.is_context or line.is_removed
                ],
                "targets": targets,
            })
    return hunks, unmappable


def aligned_hunk_start(hunk: dict, baseline_lines: list[str]) -> int | None:
    """Find the patch's old-code sequence even when diff line offsets drift."""
    old_lines = hunk["old_lines"]
    if not old_lines:
        return None
    preferred = hunk["old_start"] - 1
    if baseline_lines[preferred:preferred + len(old_lines)] == old_lines:
        return preferred
    matches = (
        index for index, line in enumerate(baseline_lines)
        if line == old_lines[0]
        and baseline_lines[index:index + len(old_lines)] == old_lines
    )
    return min(matches, key=lambda index: abs(index - preferred), default=None)


def validated_targets(
    hunk: dict, baseline_lines: list[str], *, aligned_start: int | None = None
) -> set[EDIT_LINE]:
    matched_start = (
        aligned_start if aligned_start is not None
        else aligned_hunk_start(hunk, baseline_lines)
    )
    if matched_start is None:
        return set()
    offset = matched_start - (hunk["old_start"] - 1)
    return {
        (path, line_number + offset, text)
        for path, line_number, text in hunk["targets"]
        if 0 < line_number + offset <= len(baseline_lines)
        and baseline_lines[line_number + offset - 1] == text
    }


def validate_against_baseline(item: dict, hunks: list[dict]) -> tuple[list[dict], dict]:
    """Exclude targets not present in the case's actual pre-fix commit."""
    repo = BASELINE_REPOS / item["repo"]
    commit = item["base"]["sha"]
    exists = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "-e", f"{commit}^{{commit}}"],
        capture_output=True,
    )
    if exists.returncode:
        raise ValueError(f"Baseline commit unavailable: {item['instance_id']} {commit}")
    files = {}
    for path in {hunk["path"] for hunk in hunks}:
        shown = subprocess.run(
            ["git", "-C", str(repo), "show", f"{commit}:{path}"],
            capture_output=True, text=True, errors="replace",
        )
        files[path] = (
            [normalized_code(line) for line in shown.stdout.splitlines()]
            if shown.returncode == 0 else None
        )
    validated = []
    missing_file_hunks = 0
    mismatched_hunks = 0
    for hunk in hunks:
        baseline_lines = files[hunk["path"]]
        if baseline_lines is None:
            missing_file_hunks += 1
            continue
        matched_start = aligned_hunk_start(hunk, baseline_lines)
        targets = validated_targets(hunk, baseline_lines, aligned_start=matched_start)
        if not targets:
            mismatched_hunks += 1
            continue
        validated.append({
            **hunk, "targets": targets, "baseline_start": matched_start + 1,
        })
    return validated, {
        "missing_baseline_files": sorted(path for path, lines in files.items() if lines is None),
        "missing_file_hunks": missing_file_hunks,
        "mismatched_preimage_hunks": mismatched_hunks,
    }


def resolve_path(raw: str, reference_paths: set[str]) -> str | None:
    raw = raw.strip("\"' ").removeprefix("./").removeprefix("a/")
    if raw in reference_paths:
        return raw
    matches = [path for path in reference_paths if raw.endswith("/" + path)]
    return matches[0] if len(matches) == 1 else None


def graph_lines(
    observation: str, reference_paths: set[str], *, mode: str
) -> set[EDIT_LINE]:
    try:
        payload = json.loads(observation)
    except (TypeError, ValueError):
        return set()
    exposed = set()

    def add_excerpt(path: str | None, lines: list[str]) -> None:
        if not path:
            return
        for row in lines:
            match = re.match(r"^(\d+): (.*)$", row)
            if match:
                exposed.add((path, int(match[1]), normalized_code(match[2])))

    if mode in {"function", "trace"}:
        # These modes show a compact relation-frontier view to the Agent,
        # not the full raw JSON saved in repository_evidence[].observation.
        limit = 5 if mode == "trace" else 12
        for candidate in payload if isinstance(payload, list) else []:
            if not isinstance(candidate, dict):
                continue
            path = resolve_path(str(candidate.get("file", "")), reference_paths)
            if candidate.get("source_status") == "verified":
                add_excerpt(path, str(candidate.get("source_excerpt", "")).splitlines()[:limit])
            for peer in candidate.get("scope_frontier") or []:
                if isinstance(peer, dict) and peer.get("source_status") == "verified":
                    peer_path = resolve_path(str(peer.get("file", "")), reference_paths)
                    add_excerpt(peer_path, peer.get("source_preview") or [])
        return exposed

    def visit(value: object) -> None:
        if isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, dict):
            path = resolve_path(str(value.get("file", "")), reference_paths)
            excerpt = value.get("source_excerpt")
            if path and isinstance(excerpt, str) and value.get("source_status") == "verified":
                add_excerpt(path, excerpt.splitlines())
            for child in value.values():
                if isinstance(child, (dict, list)):
                    visit(child)

    visit(payload)
    return exposed


def _tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return []


def _single_command_path(tokens: list[str], reference_paths: set[str]) -> str | None:
    paths = {resolved for token in tokens if (resolved := resolve_path(token, reference_paths))}
    return next(iter(paths)) if len(paths) == 1 else None


def source_lines(command: str, observation: str, reference_paths: set[str]) -> set[EDIT_LINE]:
    """Parse only line-bearing returned source, not bare query/path mentions."""
    tokens = _tokens(command)
    path = _single_command_path(tokens, reference_paths)
    exposed = set()
    rows = observation.splitlines()
    for row in rows:
        match = PATH_NUMBERED.match(row)
        if match:
            resolved = resolve_path(match[1], reference_paths)
            if resolved:
                exposed.add((resolved, int(match[2]), normalized_code(match[3])))
                continue
        if path and tokens and tokens[0] in {"grep", "rg"}:
            match = NUMBERED.match(row)
            if match:
                exposed.add((path, int(match[1]), normalized_code(match[2])))

    if not tokens:
        return exposed
    requested_lines = []
    if tokens[0] == "sed":
        segments = []
        segment = []
        for token in tokens + ["&&"]:
            if token in {"&&", ";"}:
                if segment:
                    segments.append(segment)
                segment = []
            else:
                segment.append(token)
        if all(part and part[0] == "sed" and "-n" in part for part in segments):
            for part in segments:
                script_index = part.index("-n") + 1
                if script_index >= len(part):
                    return exposed
                segment_path = _single_command_path(part, reference_paths)
                for instruction in part[script_index].split(";"):
                    match = re.fullmatch(r"(\d+)(?:,(\d+))?p", instruction.strip())
                    if match:
                        requested_lines.extend(
                            (segment_path, number)
                            for number in range(int(match[1]), int(match[2] or match[1]) + 1)
                        )
    elif path and tokens[0] == "head" and "-n" in tokens:
        count_index = tokens.index("-n") + 1
        if count_index < len(tokens) and tokens[count_index].isdigit():
            requested_lines = [(path, number) for number in range(1, int(tokens[count_index]) + 1)]
    elif path and tokens[0] == "cat" and len(tokens) == 2:
        requested_lines = [(path, number) for number in range(1, len(rows) + 1)]
    if not requested_lines:
        return exposed

    requested = iter(requested_lines)
    for row in rows:
        omitted = OMITTED.match(row)
        if omitted:
            for _ in range(int(omitted[1])):
                next(requested, None)
            continue
        if "[output truncated]" in row:
            break  # Tail positions cannot be reconstructed from a character cut.
        position = next(requested, None)
        if position is None:
            break
        source_path, line_number = position
        if source_path:
            exposed.add((source_path, line_number, normalized_code(row)))
    return exposed


def scout_exposure(
    artifact: Path, reference_paths: set[str]
) -> tuple[dict[EDIT_LINE, set[str]], dict[EDIT_LINE, set[str]], dict]:
    result = json.loads(artifact.read_text())
    investigation = result.get("investigation") or {}
    actions = {
        action.get("evidence_id"): action
        for action in investigation.get("actions", [])
        if action.get("evidence_id")
    }
    direct, graph = {}, {}
    unparsed_searches = 0
    nonempty_searches = 0
    for evidence in investigation.get("repository_evidence", []):
        action = actions.get(evidence.get("id"), {})
        output = evidence.get("observation", "")
        kind = action.get("action")
        if kind == "RepositoryGraphQuery":
            for line in graph_lines(
                output, reference_paths, mode=action.get("graph_mode", "")
            ):
                graph.setdefault(line, set()).add(evidence["id"])
        elif kind in {"ReadOnlySearch", "RunInvestigationCommand"} and output:
            parsed = source_lines(action.get("command", ""), output, reference_paths)
            for line in parsed:
                direct.setdefault(line, set()).add(evidence["id"])
            if kind == "ReadOnlySearch":
                nonempty_searches += 1
                if not parsed:
                    unparsed_searches += 1
    return direct, graph, {
        "nonempty_source_searches": nonempty_searches,
        "searches_without_reference_line_extraction": unparsed_searches,
    }


def selected_artifacts(runs: tuple[str, ...]) -> dict[str, Path]:
    selected = {}
    for run in runs:  # Later runs supersede earlier case artifacts.
        case_dir = ROOT / "results" / run / "cases_glm-5.3-flash_cotest"
        for artifact in sorted(case_dir.glob("*/intent_grounding_result.json")):
            selected[artifact.parent.name] = artifact
    return selected


def audit(dataset: Path, runs: tuple[str, ...] = RUNS) -> dict:
    instances = {}
    for row in dataset.read_text().splitlines():
        item = json.loads(row)
        if not item["instance_id"].startswith("yhirose__cpp-httplib-"):
            instances[item["instance_id"]] = item
    if len(instances) != 128:
        raise ValueError(f"Expected 128 non-httplib cases, got {len(instances)}")
    artifacts = selected_artifacts(runs)
    if artifacts.keys() - instances.keys():
        raise ValueError("Saved Scout cases are not in the reference dataset")

    rows = []
    for case_id, item in sorted(instances.items()):
        parsed_hunks, unmappable = baseline_hunks(item["fix_patch"])
        hunks, validation = validate_against_baseline(item, parsed_hunks)
        unmappable += validation["missing_file_hunks"] + validation["mismatched_preimage_hunks"]
        paths = {hunk["path"] for hunk in hunks}
        artifact = artifacts.get(case_id)
        direct, graph, parse_stats = (
            scout_exposure(artifact, paths) if artifact else ({}, {}, {})
        )
        source_seen_files = {line[0] for line in direct.keys() | graph.keys()}
        row_hunks = []
        for hunk in hunks:
            targets = hunk["targets"]
            direct_hits = targets & direct.keys()
            graph_hits = targets & graph.keys()
            row_hunks.append({
                "path": hunk["path"],
                "baseline_start": hunk["baseline_start"],
                "kind": hunk["kind"],
                "production_cpp": is_production_cpp(hunk["path"]),
                "direct": bool(direct_hits),
                "graph": bool(graph_hits),
                "file_source_seen": hunk["path"] in source_seen_files,
                "direct_evidence_ids": sorted({
                    evidence_id for line in direct_hits for evidence_id in direct[line]
                }),
                "graph_evidence_ids": sorted({
                    evidence_id for line in graph_hits for evidence_id in graph[line]
                }),
            })
        cpp_hunks = [h for h in row_hunks if h["production_cpp"]]
        rows.append({
            "case_id": case_id,
            "scout_artifact": str(artifact) if artifact else None,
            "scout_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest() if artifact else None,
            "mappable_hunks": len(hunks),
            "unmappable_hunks": unmappable,
            "baseline_validation": validation,
            "direct_hunks": sum(h["direct"] for h in row_hunks),
            "graph_hunks": sum(h["graph"] for h in row_hunks),
            "union_hunks": sum(h["direct"] or h["graph"] for h in row_hunks),
            "graph_only_hunks": sum(h["graph"] and not h["direct"] for h in row_hunks),
            "missed_hunks_in_seen_file": sum(
                h["file_source_seen"] and not (h["direct"] or h["graph"])
                for h in row_hunks
            ),
            "missed_hunks_in_unseen_file": sum(
                not h["file_source_seen"] for h in row_hunks
            ),
            "production_cpp_hunks": len(cpp_hunks),
            "production_cpp_exposed": sum(h["direct"] or h["graph"] for h in cpp_hunks),
            "parse_stats": parse_stats,
            "hunks": row_hunks,
        })
    eligible = [row for row in rows if row["mappable_hunks"]]
    cpp_eligible = [row for row in rows if row["production_cpp_hunks"]]
    return {
        "dataset": str(dataset),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "run_precedence": list(runs),
        "summary": {
            "cases": len(rows),
            "scout_artifacts": sum(row["scout_artifact"] is not None for row in rows),
            "eligible_cases": len(eligible),
            "unmappable_only_cases": [row["case_id"] for row in rows if not row["mappable_hunks"]],
            "mappable_hunks": sum(row["mappable_hunks"] for row in rows),
            "unmappable_hunks": sum(row["unmappable_hunks"] for row in rows),
            "missing_baseline_files": sum(
                len(row["baseline_validation"]["missing_baseline_files"]) for row in rows
            ),
            "missing_file_hunks": sum(
                row["baseline_validation"]["missing_file_hunks"] for row in rows
            ),
            "mismatched_preimage_hunks": sum(
                row["baseline_validation"]["mismatched_preimage_hunks"] for row in rows
            ),
            "macro_union_hunk_recall": sum(
                row["union_hunks"] / row["mappable_hunks"] for row in eligible
            ) / len(eligible),
            "macro_direct_hunk_recall": sum(
                row["direct_hunks"] / row["mappable_hunks"] for row in eligible
            ) / len(eligible),
            "all_mappable_hunks_exposed": sum(
                row["union_hunks"] == row["mappable_hunks"] for row in eligible
            ),
            "production_cpp_eligible_cases": len(cpp_eligible),
            "production_cpp_hunks": sum(row["production_cpp_hunks"] for row in rows),
            "macro_production_cpp_hunk_recall": sum(
                row["production_cpp_exposed"] / row["production_cpp_hunks"]
                for row in cpp_eligible
            ) / len(cpp_eligible),
            "all_production_cpp_hunks_exposed": sum(
                row["production_cpp_exposed"] == row["production_cpp_hunks"]
                for row in cpp_eligible
            ),
            "graph_only_hunks": sum(row["graph_only_hunks"] for row in rows),
            "missed_hunks_in_seen_file": sum(
                row["missed_hunks_in_seen_file"] for row in rows
            ),
            "missed_hunks_in_unseen_file": sum(
                row["missed_hunks_in_unseen_file"] for row in rows
            ),
            "searches_without_reference_line_extraction": sum(
                row["parse_stats"].get("searches_without_reference_line_extraction", 0)
                for row in rows
            ),
            "nonempty_source_searches": sum(
                row["parse_stats"].get("nonempty_source_searches", 0) for row in rows
            ),
        },
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--cohort", choices=tuple(COHORT_RUNS), default="on-latest")
    parser.add_argument(
        "--output", type=Path,
        default=None,
    )
    args = parser.parse_args()
    output = args.output or (
        ROOT / "results" / "rq3_scout_edit_exposure_20261001" /
        ("audit.json" if args.cohort == "on-latest" else f"audit_{args.cohort.replace('-', '_')}.json")
    )
    report = audit(args.dataset, COHORT_RUNS[args.cohort])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
