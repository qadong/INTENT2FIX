#!/usr/bin/env python3
"""Run A's missing cases through the full pipeline with unified-four Repair."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_repair_only_ablation import unified_repair_system_prompt  # noqa: E402


CANONICAL_DATASET = ROOT / "INTENT2FIX/benchmark/multi_swebench_cpp_change.jsonl"
A_SUBMISSION = (
    ROOT
    / "results/scout_v112_other115_low_finalists101_official_eval_20260917/submission.json"
)
A_ENV = ROOT / "results/scout_v112_other115_full_glm_low_20260917/common_env.json"
INDEX_STORE = ROOT / "results/intent_canonical128_20260921/index_store"
REPO_STORE = Path("/data/dqa/cache/repos")
PUBLIC_KEYS = ("repo", "instance_id", "base_commit", "problem_statement", "created_at")
SOURCE_FILES = (
    "INTENT2FIX/actions/unified_repair.py",
    "INTENT2FIX/agent/code_agent.py",
    "INTENT2FIX/intent_dialogue/developer_agent.py",
    "INTENT2FIX/intent_dialogue/dialogue.py",
    "INTENT2FIX/intent_dialogue/prompts.py",
    "INTENT2FIX/intent_dialogue/schema.py",
    "INTENT2FIX/intent_portfolio.py",
    "scripts/run_no_memory_cotest.py",
    "scripts/run_repair_only_ablation.py",
    "scripts/run_full_pipeline_unified_repair.py",
    "INTENT2FIX/prompt_guidance.py",
    "INTENT2FIX/scout_probe_handoff.py",
    "INTENT2FIX/prompts/repair_evidence.md",
    "INTENT2FIX/prompts/hypothesis_evidence.md",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def round_robin(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row["repo"]), []).append(row)
    for group in groups.values():
        group.sort(key=lambda row: str(row["instance_id"]))
    ordered: list[dict] = []
    while any(groups.values()):
        for repo in sorted(groups):
            if groups[repo]:
                ordered.append(groups[repo].pop(0))
    return ordered


def missing_a_rows() -> list[dict]:
    rows = [
        row
        for row in read_jsonl(CANONICAL_DATASET)
        if not str(row.get("instance_id", "")).startswith("yhirose__")
    ]
    submitted = json.loads(A_SUBMISSION.read_text(encoding="utf-8"))
    submitted_ids = {str(item["instance_id"]) for item in submitted}
    assert len(rows) == 128, len(rows)
    assert len(submitted_ids) == 101, len(submitted_ids)
    canonical_ids = {str(row["instance_id"]) for row in rows}
    assert submitted_ids <= canonical_ids
    missing = [row for row in rows if row["instance_id"] not in submitted_ids]
    assert len(missing) == 27, len(missing)
    public = [
        {key: row[key] for key in PUBLIC_KEYS if key in row}
        for row in missing
    ]
    assert all(
        not ({"patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS"} & set(row))
        for row in public
    )
    return round_robin(public)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def preflight(rows: list[dict]) -> dict[str, dict[str, str]]:
    runtime_manifest = INDEX_STORE / "runtime_store_manifest.json"
    assert runtime_manifest.is_file(), runtime_manifest
    report: dict[str, dict[str, str]] = {}
    for row in rows:
        instance_id = str(row["instance_id"])
        repo = REPO_STORE / instance_id
        assert (repo / ".git").is_dir(), repo
        assert git(repo, "rev-parse", "HEAD") == row["base_commit"], instance_id
        assert not git(repo, "status", "--porcelain"), f"Dirty repository: {repo}"
        index_dir = INDEX_STORE / f"swe-bench_{instance_id}" / "index_output"
        for required in ("embedding_manifest.json", "file_hashes.json"):
            assert (index_dir / required).is_file(), index_dir / required
        report[instance_id] = {
            "repository": str(repo),
            "index_dir": str(index_dir),
            "embedding_manifest_sha256": sha256(index_dir / "embedding_manifest.json"),
            "file_hashes_sha256": sha256(index_dir / "file_hashes.json"),
        }
    return report


def prepare(
    run_dir: Path,
    workers: int,
    *,
    case_ids: set[str] | None = None,
    poc_scout: bool = False,
) -> dict:
    if (run_dir / "manifest.json").exists():
        raise RuntimeError(f"Run already prepared: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = missing_a_rows()
    if case_ids is not None:
        available_ids = {str(row["instance_id"]) for row in rows}
        if not case_ids or not case_ids <= available_ids:
            raise ValueError(f"Unknown or empty case subset: {sorted(case_ids - available_ids)}")
        rows = [row for row in rows if row["instance_id"] in case_ids]
    preflight_report = preflight(rows)
    instances_path = run_dir / "instances.jsonl"
    instances_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    prompt_path = run_dir / "repair_system_prompt.txt"
    prompt_path.write_text(unified_repair_system_prompt(True).strip() + "\n", encoding="utf-8")

    config = json.loads(A_ENV.read_text(encoding="utf-8"))
    config.update(
        {
            "COTEST_MAX_WORKERS": str(workers),
            "COTEST_RESULTS_DIR_PREFIX": str(run_dir / "cases"),
            "COTEST_JSONL": str(instances_path),
            "COTEST_SUBMISSION_FILE": str(run_dir / "submission.json"),
            "INDEX_STORE_DIR": str(INDEX_STORE),
            "REPO_DIR": str(REPO_STORE),
            "COTEST_ABLATE_DUAL_LAYER_GRAPH": "0",
            "COTEST_PORTFOLIO_SCOUT_ONLY": "0",
            "COTEST_PORTFOLIO_REPAIR_ONLY": "0",
            "COTEST_PORTFOLIO_STOP_AFTER_REPAIR": "0",
            "COTEST_PORTFOLIO_SELECTOR_BACKEND": "legacy",
            "COTEST_BRANCH_CONTEXT": "off",
            "COTEST_ENABLE_PATCH_INTENT_AUDIT": "0",
            "COTEST_ENABLE_TRAJECTORY_AUDIT_LEADS": "0",
            "COTEST_CODING_TOOL_PROFILE": "standard",
            "COTEST_REPAIR_TOOL_PROFILE": "unified_four",
            "COTEST_REPAIR_SYSTEM_PROMPT_FILE": str(prompt_path),
            "COTEST_REPAIR_OMIT_REPOSITORY_OUTLINE": "1",
        }
    )
    if poc_scout:
        config.update({
            "COTEST_INTENT_DIALOGUE_MODE": "poc",
            "COTEST_INTENT_TEST_ANCHOR_RETRIEVAL": "0",
        })
    for name in (
        "COTEST_GLM_REASONING_EFFORT",
        "COTEST_INTENT_SCOUT_GLM_REASONING_EFFORT",
        "COTEST_INTENT_EVIDENCE_REVIEW_GLM_REASONING_EFFORT",
        "COTEST_PATCH_INTENT_ADMISSION_REASONING_EFFORT",
    ):
        assert config[name] == "low", (name, config[name])
    for name in (
        "COTEST_REPAIR_MODEL",
        "COTEST_REPRO_MODEL",
        "COTEST_EXTRACTION_MODEL",
        "COTEST_INTENT_DIALOGUE_MODEL",
    ):
        assert config[name] == "glm-5.3-flash", (name, config[name])
    assert config["COTEST_PORTFOLIO_REPAIR_STEPS"] == "60"
    write_json(run_dir / "common_env.json", config)

    manifest = {
        "status": "PREPARED",
        "prepared_at": datetime.now().astimezone().isoformat(),
        "stage": "Scout -> Hypothesis -> unified-four Repair -> Selector",
        "case_source": (
            "frozen selected subset of A missing27"
            if case_ids is not None
            else "canonical128 minus frozen V112 A submission101"
        ),
        "scout_mode": "poc" if poc_scout else "contrastive",
        "cases": [row["instance_id"] for row in rows],
        "case_count": len(rows),
        "concurrency": workers,
        "model": "glm-5.3-flash",
        "reasoning_effort_all_stages": "low",
        "repair_steps_per_hypothesis": 60,
        "repair_tool_profile": "unified_four",
        "repair_visible_tools": ["Bash", "Editor", "CodeGraph", "Finish"],
        "selector_backend": "legacy",
        "graph_enabled": True,
        "index_store": str(INDEX_STORE),
        "public_only_input": True,
        "input_jsonl_sha256": sha256(instances_path),
        "canonical_dataset_sha256": sha256(CANONICAL_DATASET),
        "a_submission_sha256": sha256(A_SUBMISSION),
        "a_environment_sha256": sha256(A_ENV),
        "common_env_sha256": sha256(run_dir / "common_env.json"),
        "repair_prompt_sha256": sha256(prompt_path),
        "runtime_index_manifest_sha256": sha256(
            INDEX_STORE / "runtime_store_manifest.json"
        ),
        "source_sha256": {name: sha256(ROOT / name) for name in SOURCE_FILES},
        "preflight": preflight_report,
    }
    write_json(run_dir / "manifest.json", manifest)
    return manifest


def verify_frozen(run_dir: Path) -> tuple[dict, dict]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "PREPARED":
        raise RuntimeError(f"Run is not PREPARED: {manifest.get('status')}")
    config = json.loads((run_dir / "common_env.json").read_text(encoding="utf-8"))
    assert sha256(run_dir / "instances.jsonl") == manifest["input_jsonl_sha256"]
    assert sha256(run_dir / "common_env.json") == manifest["common_env_sha256"]
    assert sha256(run_dir / "repair_system_prompt.txt") == manifest["repair_prompt_sha256"]
    for name, expected in manifest["source_sha256"].items():
        assert sha256(ROOT / name) == expected, f"Source drift: {name}"
    assert config["COTEST_REPAIR_TOOL_PROFILE"] == "unified_four"
    assert config["COTEST_CODING_TOOL_PROFILE"] == "standard"
    assert config["COTEST_GLM_REASONING_EFFORT"] == "low"
    assert config["COTEST_INTENT_DIALOGUE_MODE"] == manifest["scout_mode"]
    return manifest, config


def run(run_dir: Path) -> int:
    manifest, config = verify_frozen(run_dir)
    manifest.update(
        status="RUNNING",
        pipeline_pid=os.getpid(),
        started_at=datetime.now().astimezone().isoformat(),
    )
    write_json(run_dir / "manifest.json", manifest)
    environment = os.environ.copy()
    environment.update(config)
    command = [
        str(ROOT / ".venv/bin/python"),
        str(ROOT / "scripts/run_no_memory_cotest.py"),
        "--jsonl",
        str(run_dir / "instances.jsonl"),
        "--workers",
        str(manifest["concurrency"]),
        "--submission",
        str(run_dir / "submission.json"),
    ]
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(
        status="COMPLETED" if completed.returncode == 0 else "FAILED",
        exit_code=completed.returncode,
        finished_at=datetime.now().astimezone().isoformat(),
    )
    write_json(run_dir / "manifest.json", manifest)
    return completed.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--case-ids", default="",
        help="Comma-separated frozen subset of A's 27 missing cases.",
    )
    parser.add_argument("--poc-scout", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    run_dir = Path(args.output_dir)
    if not run_dir.is_absolute():
        run_dir = (ROOT / run_dir).resolve()
    if not (run_dir / "manifest.json").exists():
        case_ids = (
            {item.strip() for item in args.case_ids.split(",") if item.strip()}
            if args.case_ids else None
        )
        prepare(
            run_dir, args.workers,
            case_ids=case_ids, poc_scout=args.poc_scout,
        )
    if args.prepare_only:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        print(json.dumps({"status": manifest["status"], "cases": manifest["case_count"]}))
        return 0
    return run(run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
