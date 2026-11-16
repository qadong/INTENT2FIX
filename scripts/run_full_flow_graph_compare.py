#!/usr/bin/env python3
"""Run a frozen, paired full-flow graph-on/off comparison."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_full_pipeline_unified_repair as baseline  # noqa: E402
from scripts.run_repair_only_ablation import unified_repair_system_prompt  # noqa: E402


PRIOR_CASES = frozenset({
    "catchorg__Catch2-1422", "catchorg__Catch2-1616",
    "fmtlib__fmt-2317", "fmtlib__fmt-4286", "fmtlib__fmt-3555",
    "nlohmann__json-2866", "nlohmann__json-708", "nlohmann__json-2297",
    "simdjson__simdjson-1615", "simdjson__simdjson-1896",
})
QUOTAS = {"catchorg/Catch2": 2, "fmtlib/fmt": 3,
          "nlohmann/json": 3, "simdjson/simdjson": 2}
SELECTION_SEED = "full_flow_graph_scope_v5_20260926"
SOURCE_FILES = (
    "scripts/run_full_flow_graph_compare.py",
    "scripts/run_no_memory_cotest.py",
    "scripts/run_repair_only_ablation.py",
    "INTENT2FIX/agent/code_agent.py",
    "INTENT2FIX/actions/unified_repair.py",
    "INTENT2FIX/intent_dialogue/developer_agent.py",
    "INTENT2FIX/intent_dialogue/dialogue.py",
    "INTENT2FIX/intent_dialogue/prompts.py",
    "INTENT2FIX/intent_dialogue/schema.py",
    "INTENT2FIX/intent_portfolio.py",
    "INTENT2FIX/prompt_guidance.py",
    "INTENT2FIX/scout_probe_handoff.py",
    "INTENT2FIX/prompts/repair_evidence.md",
    "INTENT2FIX/prompts/hypothesis_evidence.md",
)
ARMS = ("graph_on", "graph_off")


def ranked_case_id(row: dict) -> str:
    return hashlib.sha256(
        f"{SELECTION_SEED}:{row['instance_id']}".encode("utf-8")
    ).hexdigest()


def select_cases(excluded_cases: frozenset[str] = PRIOR_CASES, *,
                 all_remaining: bool = False) -> list[dict]:
    rows = baseline.read_jsonl(baseline.CANONICAL_DATASET)
    selected = []
    for repo, count in QUOTAS.items():
        available = sorted(
            (row for row in rows if row["repo"] == repo
             and row["instance_id"] not in excluded_cases),
            key=ranked_case_id,
        )
        if not all_remaining and len(available) < count:
            raise ValueError(f"Not enough cases for {repo}: {len(available)}")
        selected.extend(available if all_remaining else available[:count])
    assert selected and len({row["instance_id"] for row in selected}) == len(selected)
    if not all_remaining:
        assert len(selected) == 10
    public = [
        {key: row[key] for key in baseline.PUBLIC_KEYS if key in row}
        for row in selected
    ]
    assert all("problem_statement" in row and "base_commit" in row for row in public)
    assert all(not ({"patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS"} & set(row))
               for row in public)
    return baseline.round_robin(public) if all_remaining else public


def prepare(root: Path, workers_per_arm: int, *,
            exclude_manifests: list[Path] | None = None,
            all_remaining: bool = False,
            stop_after_repair: bool = False,
            evaluate_all_candidates: bool = False,
            case_ids: set[str] | None = None,
            allow_prior_cases: bool = False,
            arms: tuple[str, ...] = ARMS) -> None:
    if root.exists():
        raise FileExistsError(f"Comparison directory already exists: {root}")
    if allow_prior_cases and not case_ids:
        raise ValueError("Prior-case reruns require explicit case IDs")
    if not arms or set(arms) - set(ARMS) or len(set(arms)) != len(arms):
        raise ValueError(f"Invalid comparison arms: {arms}")
    excluded = set() if allow_prior_cases else set(PRIOR_CASES)
    exclusion_hashes = {}
    for path in exclude_manifests or []:
        path = path.resolve()
        excluded.update(json.loads(path.read_text(encoding="utf-8"))["cases"])
        exclusion_hashes[str(path)] = baseline.sha256(path)
    if case_ids:
        if all_remaining:
            raise ValueError("Explicit cases and all_remaining cannot be combined")
        available = select_cases(frozenset(excluded), all_remaining=True)
        rows = [row for row in available if row["instance_id"] in case_ids]
        if {row["instance_id"] for row in rows} != case_ids:
            raise ValueError("Explicit cases must be eligible, non-excluded canonical cases")
    else:
        rows = select_cases(frozenset(excluded), all_remaining=all_remaining)
    preflight = baseline.preflight(rows)
    base_env = json.loads(baseline.A_ENV.read_text(encoding="utf-8"))
    for key in ("COTEST_GLM_REASONING_EFFORT",
                "COTEST_INTENT_SCOUT_GLM_REASONING_EFFORT",
                "COTEST_INTENT_EVIDENCE_REVIEW_GLM_REASONING_EFFORT",
                "COTEST_PATCH_INTENT_ADMISSION_REASONING_EFFORT"):
        assert base_env[key] == "low", (key, base_env[key])
    for key in ("COTEST_REPAIR_MODEL", "COTEST_REPRO_MODEL",
                "COTEST_EXTRACTION_MODEL", "COTEST_INTENT_DIALOGUE_MODEL"):
        assert base_env[key] == "glm-5.3-flash", (key, base_env[key])
    assert base_env["COTEST_PORTFOLIO_REPAIR_STEPS"] == "60"

    root.mkdir(parents=True)
    source_files = SOURCE_FILES + (
        ("scripts/evaluate_pair_candidate_patches.py", "scripts/evaluate_fix_patches.py")
        if evaluate_all_candidates else ()
    )
    source_hashes = {name: baseline.sha256(ROOT / name) for name in source_files}
    for arm in arms:
        arm_dir = root / arm
        arm_dir.mkdir()
        graph = arm == "graph_on"
        instances = arm_dir / "instances.jsonl"
        instances.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        prompt = arm_dir / "repair_system_prompt.txt"
        prompt.write_text(unified_repair_system_prompt(graph).strip() + "\n", encoding="utf-8")
        config = {**base_env, **{
            "COTEST_MAX_WORKERS": str(workers_per_arm),
            "COTEST_RESULTS_DIR_PREFIX": str(arm_dir / "cases"),
            "COTEST_JSONL": str(instances),
            "COTEST_SUBMISSION_FILE": str(arm_dir / "submission.json"),
            "INDEX_STORE_DIR": str(baseline.INDEX_STORE),
            "REPO_DIR": str(baseline.REPO_STORE),
            "COTEST_ABLATE_DUAL_LAYER_GRAPH": "0",
            "COTEST_SCOUT_GRAPH_SEARCH": "1" if graph else "0",
            "COTEST_REPAIR_GRAPH_SEARCH": "1" if graph else "0",
            "COTEST_ENABLE_INTENT_PORTFOLIO": "1",
            "COTEST_INTENT_DIALOGUE_MODE": "poc",
            "COTEST_INTENT_TEST_ANCHOR_RETRIEVAL": "0",
            "COTEST_PORTFOLIO_SCOUT_ONLY": "0",
            "COTEST_PORTFOLIO_REPAIR_ONLY": "0",
            "COTEST_PORTFOLIO_STOP_AFTER_REPAIR": "1" if stop_after_repair else "0",
            "COTEST_PORTFOLIO_SELECTOR_BACKEND": "legacy",
            "COTEST_INTENT_PORTFOLIO_CACHE_DIR": "",
            "COTEST_INTENT_SCOUT_CACHE_DIR": "",
            "COTEST_BRANCH_CONTEXT": "off",
            "COTEST_ENABLE_PATCH_INTENT_AUDIT": "0",
            "COTEST_ENABLE_TRAJECTORY_AUDIT_LEADS": "0",
            "COTEST_CODING_TOOL_PROFILE": "standard",
            "COTEST_REPAIR_TOOL_PROFILE": "unified_four",
            "COTEST_REPAIR_SYSTEM_PROMPT_FILE": str(prompt),
            "COTEST_REPAIR_OMIT_REPOSITORY_OUTLINE": "1",
            "COTEST_REPAIR_EVIDENCE_GUIDANCE": os.environ.get("COTEST_REPAIR_EVIDENCE_GUIDANCE", "1"),
            "COTEST_HYPOTHESIS_EVIDENCE_GUIDANCE": os.environ.get("COTEST_HYPOTHESIS_EVIDENCE_GUIDANCE", "1"),
        }}
        baseline.write_json(arm_dir / "common_env.json", config)
        baseline.write_json(arm_dir / "manifest.json", {
            "status": "PREPARED", "arm": arm,
            "prepared_at": datetime.now().astimezone().isoformat(),
            "stages": ["Scout", "Hypothesis", "Repair"] + (
                [] if stop_after_repair else ["Selector"]),
            "cases": [row["instance_id"] for row in rows], "workers": workers_per_arm,
            "model": "glm-5.3-flash", "reasoning_effort_all_stages": "low",
            "repair_steps_per_hypothesis": 60,
            "repair_visible_tools": ["Bash", "Editor", "CodeGraph", "Finish"] if graph
                                    else ["Bash", "Editor", "Finish"],
            "selector_backend": None if stop_after_repair else "legacy",
            "graph_enabled": graph,
            "public_only_input": True,
            "instances_sha256": baseline.sha256(instances),
            "config_sha256": baseline.sha256(arm_dir / "common_env.json"),
            "prompt_sha256": baseline.sha256(prompt),
            "source_sha256": source_hashes,
            "preflight": preflight,
        })
    baseline.write_json(root / "manifest.json", {
        "status": "PREPARED", "prepared_at": datetime.now().astimezone().isoformat(),
        "arms": arms, "cases": [row["instance_id"] for row in rows],
        "selection_seed": SELECTION_SEED,
        "selection_mode": "explicit_cases" if case_ids else "all_remaining" if all_remaining else "repository_quotas",
        "selection_quotas": None if all_remaining else QUOTAS,
        "excluded_prior_cases": sorted(excluded),
        "exclusion_manifest_sha256": exclusion_hashes,
        "workers_per_arm": workers_per_arm,
        "evaluate_all_candidates": evaluate_all_candidates,
        "canonical_dataset_sha256": baseline.sha256(baseline.CANONICAL_DATASET),
    })


def run_arm(arm_dir: Path) -> int:
    manifest_path = arm_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "PREPARED":
        raise RuntimeError(f"Arm is not prepared: {arm_dir}")
    for name, expected in manifest["source_sha256"].items():
        assert baseline.sha256(ROOT / name) == expected, f"Source drift: {name}"
    for name, expected in (("instances.jsonl", "instances_sha256"),
                           ("common_env.json", "config_sha256"),
                           ("repair_system_prompt.txt", "prompt_sha256")):
        assert baseline.sha256(arm_dir / name) == manifest[expected], name
    config = json.loads((arm_dir / "common_env.json").read_text(encoding="utf-8"))
    manifest.update(status="RUNNING", started_at=datetime.now().astimezone().isoformat(),
                    pipeline_pid=os.getpid())
    baseline.write_json(manifest_path, manifest)
    command = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/run_no_memory_cotest.py"),
               "--jsonl", str(arm_dir / "instances.jsonl"),
               "--workers", str(manifest["workers"]),
               "--submission", str(arm_dir / "submission.json")]
    with (arm_dir / "launcher.log").open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, env={**os.environ, **config},
                                stdout=stream, stderr=subprocess.STDOUT, check=False)
    manifest.update(status="COMPLETED" if result.returncode == 0 else "FAILED",
                    exit_code=result.returncode,
                    finished_at=datetime.now().astimezone().isoformat())
    baseline.write_json(manifest_path, manifest)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers-per-arm", type=int, default=5)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--all-remaining", action="store_true")
    parser.add_argument("--case-ids", default="", help="Comma-separated explicit public cases, including a smoke subset.")
    parser.add_argument("--stop-after-repair", action="store_true")
    parser.add_argument("--evaluate-all-candidates", action="store_true")
    parser.add_argument("--run-prepared", action="store_true")
    args = parser.parse_args()
    root = args.output_dir.resolve()
    if args.run_prepared:
        if not (root / "manifest.json").is_file():
            raise FileNotFoundError(root / "manifest.json")
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] != "PREPARED":
            raise RuntimeError(f"Comparison is not prepared: {root}")
        manifest.update(status="RUNNING", started_at=datetime.now().astimezone().isoformat(),
                        launcher_pid=os.getpid())
        baseline.write_json(root / "manifest.json", manifest)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(run_arm, root / arm) for arm in ARMS]
            codes = [future.result() for future in futures]
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        manifest.update(status="COMPLETED" if all(code == 0 for code in codes) else "FAILED",
                        exit_codes=codes, finished_at=datetime.now().astimezone().isoformat())
        baseline.write_json(root / "manifest.json", manifest)
        if all(code == 0 for code in codes) and manifest.get("evaluate_all_candidates"):
            command = [str(ROOT / ".venv/bin/python"),
                       str(ROOT / "scripts/evaluate_pair_candidate_patches.py"),
                       "--source-dir", str(root),
                       "--output-dir", str(root / "official_eval_all_candidates"),
                       "--workers", str(manifest["workers_per_arm"])]
            with (root / "official_eval_launcher.log").open("w", encoding="utf-8") as stream:
                result = subprocess.run(command, cwd=ROOT, stdout=stream,
                                        stderr=subprocess.STDOUT, check=False)
            manifest["official_evaluation_exit_code"] = result.returncode
            baseline.write_json(root / "manifest.json", manifest)
            return result.returncode
        return 0 if all(code == 0 for code in codes) else 1
    if args.workers_per_arm < 1:
        parser.error("workers-per-arm must be positive")
    prepare(root, args.workers_per_arm, exclude_manifests=args.exclude_manifest,
            all_remaining=args.all_remaining,
            stop_after_repair=args.stop_after_repair,
            evaluate_all_candidates=args.evaluate_all_candidates,
            case_ids={item.strip() for item in args.case_ids.split(",") if item.strip()} or None)
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
