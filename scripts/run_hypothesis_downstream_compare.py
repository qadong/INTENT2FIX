#!/usr/bin/env python3
"""Resume paired graph-on/off Hypothesis checkpoints through Repair only."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from INTENT2FIX.intent_portfolio import (  # noqa: E402
    HypothesisPoolAudit,
    IntentHypothesisPool,
    select_hypothesis_pool,
)
from scripts import run_no_memory_cotest as pipeline  # noqa: E402
from scripts.run_repair_only_ablation import unified_repair_system_prompt  # noqa: E402


PRIOR = ROOT / "results/graph_scout_handoff_compare_20260923"
GRAPH_ON = PRIOR / "hypothesis_v4_remaining122_low_20260925"
GRAPH_OFF = PRIOR / "hypothesis_v4_focus10_low_20260925_scout_no_graph"
ON_CASE_DIRS = (
    GRAPH_ON / "cases_glm-5.3-flash_cotest",
    GRAPH_ON / "resume113_low6/cases_glm-5.3-flash_cotest",
    GRAPH_ON / "resume45_low6/cases_glm-5.3-flash_cotest",
    GRAPH_ON / "retry_fmt3913_low1/cases_glm-5.3-flash_cotest",
)
OFF_CASE_DIR = GRAPH_OFF / "cases_glm-5.3-flash_cotest"
CASE_IDS = (
    "catchorg__Catch2-1422", "catchorg__Catch2-1616",
    "fmtlib__fmt-2317", "fmtlib__fmt-4286", "fmtlib__fmt-3555",
    "nlohmann__json-2866", "nlohmann__json-708", "nlohmann__json-2297",
    "simdjson__simdjson-1615", "simdjson__simdjson-1896",
)
CHECKPOINT_FILES = (
    "intent_scout_checkpoint.json", "intent_grounding_result.json",
    "intent_portfolio.json", "intent_hypothesis_execution.json",
    "hypothesis_source_evidence.json", "hypothesis_audit_evidence.json",
)
SOURCE_FILES = (
    "INTENT2FIX/actions/unified_repair.py", "INTENT2FIX/agent/code_agent.py",
    "INTENT2FIX/intent_dialogue/developer_agent.py", "INTENT2FIX/intent_portfolio.py",
    "scripts/run_no_memory_cotest.py", "scripts/run_repair_only_ablation.py",
    "scripts/run_hypothesis_downstream_compare.py",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_case(arm: str, case_id: str) -> Path:
    if arm == "graph_off":
        candidates = [OFF_CASE_DIR / case_id]
    else:
        candidates = [parent / case_id for parent in ON_CASE_DIRS]
    found = [path for path in candidates if (path / "intent_portfolio.json").is_file()]
    if len(found) != 1:
        raise ValueError(f"Expected one {arm} Hypothesis checkpoint for {case_id}: {found}")
    return found[0]


def prepare(root: Path, workers_per_arm: int) -> None:
    if root.exists():
        raise FileExistsError(f"Comparison output already exists: {root}")
    rows = [json.loads(line) for line in (GRAPH_OFF / "instances.jsonl").read_text().splitlines() if line]
    by_id = {row["instance_id"]: row for row in rows}
    if len(rows) != len(CASE_IDS) or set(by_id) != set(CASE_IDS):
        raise ValueError("The paired 10-case public input is incomplete")
    public_keys = {"repo", "instance_id", "base_commit", "problem_statement", "created_at"}
    if any(set(row) - public_keys for row in rows):
        raise ValueError("Paired input contains fields beyond public Issue/base metadata")
    base_env = json.loads((GRAPH_OFF / "common_env.json").read_text())
    if any(base_env[key] != "low" for key in (
        "COTEST_GLM_REASONING_EFFORT", "COTEST_INTENT_SCOUT_GLM_REASONING_EFFORT",
        "COTEST_INTENT_EVIDENCE_REVIEW_GLM_REASONING_EFFORT",
        "COTEST_PATCH_INTENT_ADMISSION_REASONING_EFFORT",
    )) or base_env["COTEST_PORTFOLIO_REPAIR_STEPS"] != "60":
        raise ValueError("Expected the frozen all-low, 60-step configuration")

    root.mkdir(parents=True)
    source_hashes = {name: digest(ROOT / name) for name in SOURCE_FILES}
    index_store = Path(base_env["INDEX_STORE_DIR"])
    repo_store = Path(base_env["REPO_DIR"])
    for arm in ("graph_on", "graph_off"):
        arm_dir = root / arm
        cache = arm_dir / "checkpoints"
        cache.mkdir(parents=True)
        instances = arm_dir / "instances.jsonl"
        instances.write_text(
            "".join(json.dumps(by_id[case_id], ensure_ascii=False) + "\n" for case_id in CASE_IDS),
            encoding="utf-8",
        )
        prompt = arm_dir / "repair_system_prompt.txt"
        prompt.write_text(unified_repair_system_prompt(arm == "graph_on").strip() + "\n")
        config = {**base_env, **{
            "COTEST_PORTFOLIO_SCOUT_ONLY": "0",
            "COTEST_PORTFOLIO_REPAIR_ONLY": "1",
            "COTEST_PORTFOLIO_STOP_AFTER_REPAIR": "1",
            "COTEST_ABLATE_DUAL_LAYER_GRAPH": "0",
            "COTEST_SCOUT_GRAPH_SEARCH": "1" if arm == "graph_on" else "0",
            "COTEST_REPAIR_GRAPH_SEARCH": "1" if arm == "graph_on" else "0",
            "COTEST_REPAIR_TOOL_PROFILE": "unified_four",
            "COTEST_CODING_TOOL_PROFILE": "standard",
            "COTEST_REPAIR_SYSTEM_PROMPT_FILE": str(prompt),
            "COTEST_REPAIR_OMIT_REPOSITORY_OUTLINE": "1",
            "COTEST_BRANCH_CONTEXT": "off",
            "COTEST_ENABLE_PATCH_INTENT_AUDIT": "0",
            "COTEST_ENABLE_TRAJECTORY_AUDIT_LEADS": "0",
            "COTEST_MAX_WORKERS": str(workers_per_arm),
            "COTEST_RESULTS_DIR_PREFIX": str(arm_dir / "cases"),
            "COTEST_JSONL": str(instances),
            "COTEST_SUBMISSION_FILE": str(arm_dir / "submission.json"),
            "COTEST_INTENT_PORTFOLIO_CACHE_DIR": str(cache),
        }}
        write_json(arm_dir / "common_env.json", config)
        checkpoints = {}
        for case_id in CASE_IDS:
            source = source_case(arm, case_id)
            destination = cache / case_id
            destination.mkdir()
            hashes = {}
            files = CHECKPOINT_FILES + (
                ("intent_hypothesis_rerank.json",)
                if (source / "intent_hypothesis_rerank.json").is_file() else ()
            )
            for name in files:
                shutil.copy2(source / name, destination / name)
                hashes[name] = digest(destination / name)
            write_json(destination / "repair_checkpoint.json", {"artifact_sha256": hashes})
            scout = json.loads((destination / "intent_scout_checkpoint.json").read_text())
            issue_hash = hashlib.sha256(by_id[case_id]["problem_statement"].encode()).hexdigest()
            if (
                scout["base_commit"] != by_id[case_id]["base_commit"]
                or scout["problem_sha256"] != issue_hash
                or scout["policy_version"] != pipeline._INTENT_SCOUT_POLICY_VERSION
                or scout["investigation_mode"] != "poc"
                or scout["scout_reasoning_effort"] != "low"
            ):
                raise ValueError(f"Checkpoint does not match public Issue/base: {arm}/{case_id}")
            pool = IntentHypothesisPool.model_validate_json((destination / "intent_portfolio.json").read_text())
            if "intent_hypothesis_rerank.json" in hashes:
                audit = HypothesisPoolAudit.model_validate_json((destination / "intent_hypothesis_rerank.json").read_text())
                if select_hypothesis_pool(pool, audit).selected_hypothesis_ids != pool.selected_hypothesis_ids:
                    raise ValueError(f"Checkpoint Audit selection mismatch: {arm}/{case_id}")
            else:
                execution = json.loads((destination / "intent_hypothesis_execution.json").read_text())
                if (execution.get("audit_available") is not False
                        or execution.get("selection_mode") != "generated_pool_fallback"
                        or execution.get("selected_hypothesis_ids") != pool.selected_hypothesis_ids):
                    raise ValueError(f"Invalid fallback Hypothesis checkpoint: {arm}/{case_id}")
            repo = repo_store / case_id
            if subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip() != by_id[case_id]["base_commit"]:
                raise ValueError(f"Base checkout mismatch: {case_id}")
            if arm == "graph_on":
                index_dir = index_store / f"swe-bench_{case_id}" / "index_output"
                if not all((index_dir / name).is_file() for name in ("embedding_manifest.json", "file_hashes.json")):
                    raise ValueError(f"Graph index is incomplete: {case_id}")
            checkpoints[case_id] = {"source": str(source), "artifact_sha256": hashes}
        write_json(arm_dir / "manifest.json", {
            "status": "PREPARED", "arm": arm, "prepared_at": datetime.now().astimezone().isoformat(),
            "cases": list(CASE_IDS), "workers": workers_per_arm,
            "stages": ["Repair"], "selector_ran": False, "scout_hypothesis_reused": True,
            "repair_visible_tools": ["Bash", "Editor", "CodeGraph", "Finish"] if arm == "graph_on" else ["Bash", "Editor", "Finish"],
            "model": "glm-5.3-flash", "reasoning_effort": "low", "repair_steps_per_hypothesis": 60,
            "comparison_metric": "case has at least one official-resolved Repair candidate",
            "public_only_input": True,
            "instances_sha256": digest(instances), "config_sha256": digest(arm_dir / "common_env.json"),
            "prompt_sha256": digest(prompt), "source_sha256": source_hashes,
            "checkpoints": checkpoints,
        })
    write_json(root / "manifest.json", {"status": "PREPARED", "arms": ["graph_on", "graph_off"],
                                         "cases": list(CASE_IDS), "workers_total": workers_per_arm * 2})


def run_arm(arm_dir: Path) -> int:
    manifest_path = arm_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "PREPARED":
        raise ValueError(f"Arm is not PREPARED: {arm_dir}")
    config = json.loads((arm_dir / "common_env.json").read_text())
    for name, expected in manifest["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Source changed after preparation: {name}")
    for case_id, record in manifest["checkpoints"].items():
        for name, expected in record["artifact_sha256"].items():
            if digest(arm_dir / "checkpoints" / case_id / name) != expected:
                raise ValueError(f"Checkpoint changed after preparation: {arm_dir.name}/{case_id}/{name}")
    manifest.update(status="RUNNING", started_at=datetime.now().astimezone().isoformat())
    write_json(manifest_path, manifest)
    command = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/run_no_memory_cotest.py"),
               "--jsonl", str(arm_dir / "instances.jsonl"), "--workers", str(manifest["workers"]),
               "--submission", str(arm_dir / "submission.json")]
    with (arm_dir / "launcher.log").open("w") as stream:
        result = subprocess.run(command, cwd=ROOT, env={**os.environ, **config},
                                stdout=stream, stderr=subprocess.STDOUT, check=False)
    manifest.update(status="COMPLETED" if result.returncode == 0 else "FAILED",
                    exit_code=result.returncode, finished_at=datetime.now().astimezone().isoformat())
    write_json(manifest_path, manifest)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers-per-arm", type=int, default=5)
    parser.add_argument("--run-prepared", action="store_true")
    args = parser.parse_args()
    root = Path(args.output_dir).resolve()
    if args.run_prepared:
        if not (root / "manifest.json").is_file():
            raise FileNotFoundError(root / "manifest.json")
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(run_arm, root / arm) for arm in ("graph_on", "graph_off")]
            codes = [future.result() for future in futures]
        write_json(root / "manifest.json", {"status": "COMPLETED" if all(code == 0 for code in codes) else "FAILED",
                                             "arms": ["graph_on", "graph_off"], "exit_codes": codes,
                                             "finished_at": datetime.now().astimezone().isoformat()})
        return 0 if all(code == 0 for code in codes) else 1
    if args.workers_per_arm < 1:
        parser.error("workers-per-arm must be positive")
    prepare(root, args.workers_per_arm)
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
