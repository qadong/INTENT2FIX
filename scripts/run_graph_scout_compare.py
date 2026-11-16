#!/usr/bin/env python3
"""Run a frozen, public-input Scout/Hypothesis comparison arm."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "results/graph_scout_handoff_compare_20260923"
DATASET = ROOT / "INTENT2FIX/benchmark/multi_swebench_cpp_change.jsonl"
BASE_ENV = RUN_ROOT / "hypothesis_v4_remaining122_low_20260925/common_env.json"
INDEX_STORE = ROOT / "results/intent_canonical128_20260921/index_store"
CASE_IDS = (
    "fmtlib__fmt-3819",
    "fmtlib__fmt-3824",
    "nlohmann__json-3073",
    "nlohmann__json-3601",
    "nlohmann__json-4536",
    "simdjson__simdjson-954",
)
SOURCE_FILES = (
    "INTENT2FIX/intent_dialogue/developer_agent.py",
    "INTENT2FIX/intent_dialogue/prompts.py",
    "INTENT2FIX/intent_dialogue/schema.py",
    "INTENT2FIX/intent_portfolio.py",
    "INTENT2FIX/hypothesis_source.py",
    "scripts/run_no_memory_cotest.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prepare(arm: str, case_ids: tuple[str, ...], workers: int) -> Path:
    arm_dir = RUN_ROOT / arm
    if arm_dir.exists():
        raise RuntimeError(f"Comparison arm already exists: {arm_dir}")
    arm_dir.mkdir(parents=True)
    rows = [json.loads(line) for line in DATASET.read_text().splitlines() if line]
    by_id = {row["instance_id"]: row for row in rows}
    public_keys = ("repo", "instance_id", "base_commit", "problem_statement", "created_at")
    public_rows = [{key: by_id[case_id][key] for key in public_keys if key in by_id[case_id]}
                   for case_id in case_ids]
    for case_id in case_ids:
        index = INDEX_STORE / f"swe-bench_{case_id}/index_output"
        if not (index / "embedding_manifest.json").is_file():
            raise RuntimeError(f"Missing graph index: {case_id}")
    instances = arm_dir / "instances.jsonl"
    instances.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in public_rows))

    config = json.loads(BASE_ENV.read_text())
    config.update({
        "COTEST_INTENT_DIALOGUE_MODE": "poc",
        "COTEST_INTENT_TEST_ANCHOR_RETRIEVAL": "0",
        "COTEST_PORTFOLIO_SCOUT_ONLY": "1",
        "COTEST_MAX_WORKERS": str(workers),
        "COTEST_RESULTS_DIR_PREFIX": str(arm_dir / "cases"),
        "COTEST_JSONL": str(instances),
        "COTEST_SUBMISSION_FILE": str(arm_dir / "submission.json"),
        "INDEX_STORE_DIR": str(INDEX_STORE),
    })
    if arm.endswith("no_graph"):
        config["COTEST_SCOUT_GRAPH_SEARCH"] = "0"
    for key in (
        "COTEST_GLM_REASONING_EFFORT",
        "COTEST_INTENT_SCOUT_GLM_REASONING_EFFORT",
        "COTEST_INTENT_EVIDENCE_REVIEW_GLM_REASONING_EFFORT",
        "COTEST_PATCH_INTENT_ADMISSION_REASONING_EFFORT",
    ):
        assert config[key] == "low", (key, config[key])
    write_json(arm_dir / "common_env.json", config)
    manifest = {
        "status": "PREPARED",
        "prepared_at": datetime.now().astimezone().isoformat(),
        "arm": arm,
        "cases": list(case_ids),
        "input": "public Issue and base commit only; no golden patch or hidden tests",
        "mode": "POC Scout -> Hypothesis/Audit; stop before Repair",
        "model": "glm-5.3-flash",
        "reasoning_effort": "low",
        "workers": workers,
        "scout_graph_enabled": not arm.endswith("no_graph"),
        "instances_sha256": sha256(instances),
        "config_sha256": sha256(arm_dir / "common_env.json"),
        "index_manifest_sha256": sha256(INDEX_STORE / "runtime_store_manifest.json"),
        "source_sha256": {name: sha256(ROOT / name) for name in SOURCE_FILES},
    }
    write_json(arm_dir / "manifest.json", manifest)
    return arm_dir


def run(arm_dir: Path) -> int:
    manifest_path = arm_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "PREPARED":
        raise RuntimeError(f"Arm is not prepared: {manifest['status']}")
    config = json.loads((arm_dir / "common_env.json").read_text())
    assert sha256(arm_dir / "instances.jsonl") == manifest["instances_sha256"]
    assert sha256(arm_dir / "common_env.json") == manifest["config_sha256"]
    assert sha256(INDEX_STORE / "runtime_store_manifest.json") == manifest["index_manifest_sha256"]
    for name, expected in manifest["source_sha256"].items():
        assert sha256(ROOT / name) == expected, f"Source drift: {name}"
    manifest.update(status="RUNNING", started_at=datetime.now().astimezone().isoformat())
    write_json(manifest_path, manifest)
    env = os.environ.copy()
    env.update(config)
    command = [str(ROOT / ".venv/bin/python"), "-m", "scripts.run_no_memory_cotest",
               "--jsonl", str(arm_dir / "instances.jsonl"),
               "--workers", str(manifest["workers"]),
               "--portfolio-scout-only", "--submission", str(arm_dir / "submission.json")]
    completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
    manifest.update(status="COMPLETED" if completed.returncode == 0 else "FAILED",
                    exit_code=completed.returncode,
                    finished_at=datetime.now().astimezone().isoformat())
    write_json(manifest_path, manifest)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm")
    parser.add_argument("--cases", nargs="+", default=CASE_IDS)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-prepared", action="store_true")
    args = parser.parse_args()
    if args.run_prepared:
        if args.prepare_only:
            parser.error("--run-prepared and --prepare-only are mutually exclusive")
        return run(RUN_ROOT / args.arm)
    if args.workers < 1 or len(set(args.cases)) != len(args.cases):
        parser.error("workers must be positive and cases must be unique")
    arm_dir = prepare(args.arm, tuple(args.cases), args.workers)
    return 0 if args.prepare_only else run(arm_dir)


if __name__ == "__main__":
    raise SystemExit(main())
