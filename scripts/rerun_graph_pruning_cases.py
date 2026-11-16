#!/usr/bin/env python3
"""Rerun the four graph-pruning failures through fresh Scout/Hypothesis/Repair."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_full_flow_graph_compare as fullflow  # noqa: E402
from scripts import run_full_pipeline_unified_repair as baseline  # noqa: E402


CASE_IDS = frozenset({
    "catchorg__Catch2-2187", "fmtlib__fmt-3732",
    "nlohmann__json-3415", "nlohmann__json-3523",
})


def prepare(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    prior = json.loads((source / "manifest.json").read_text())
    config = json.loads((source / "common_env.json").read_text())
    if prior["status"] != "COMPLETED" or prior["exit_code"] != 0:
        raise ValueError("Source arm has not completed successfully")
    rows = [row for row in baseline.read_jsonl(source / "instances.jsonl")
            if row["instance_id"] in CASE_IDS]
    if len(rows) != len(CASE_IDS) or {row["instance_id"] for row in rows} != CASE_IDS:
        raise ValueError("Source arm does not contain the four affected cases")
    if any(set(row) - set(baseline.PUBLIC_KEYS) for row in rows):
        raise ValueError("Source inputs contain non-public fields")
    expected = {
        "COTEST_GLM_REASONING_EFFORT": "low",
        "COTEST_INTENT_SCOUT_GLM_REASONING_EFFORT": "low",
        "COTEST_INTENT_EVIDENCE_REVIEW_GLM_REASONING_EFFORT": "low",
        "COTEST_PATCH_INTENT_ADMISSION_REASONING_EFFORT": "low",
        "COTEST_REPAIR_MODEL": "glm-5.3-flash",
        "COTEST_INTENT_DIALOGUE_MODEL": "glm-5.3-flash",
        "COTEST_INTENT_DIALOGUE_MODE": "poc",
        "COTEST_INTENT_EVIDENCE_STEPS": "60",
        "COTEST_PORTFOLIO_REPAIR_STEPS": "60",
        "COTEST_REPAIR_TOOL_PROFILE": "unified_four",
        "COTEST_SCOUT_GRAPH_SEARCH": "1",
        "COTEST_REPAIR_GRAPH_SEARCH": "1",
        "COTEST_ABLATE_DUAL_LAYER_GRAPH": "0",
        "INDEX_STORE_DIR": str(baseline.INDEX_STORE),
        "REPO_DIR": str(baseline.REPO_STORE),
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Unexpected source configuration: {key}")
    # Only the intended graph-pruning retry may differ from the original runtime.
    for name, expected_hash in prior["source_sha256"].items():
        if name != "INTENT2FIX/intent_dialogue/developer_agent.py":
            if baseline.sha256(ROOT / name) != expected_hash:
                raise ValueError(f"Unrelated source drift since original run: {name}")
    preflight = baseline.preflight(rows)

    output.mkdir(parents=True)
    instances = output / "instances.jsonl"
    instances.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    prompt = output / "repair_system_prompt.txt"
    shutil.copy2(source / "repair_system_prompt.txt", prompt)
    if baseline.sha256(prompt) != prior["prompt_sha256"]:
        raise ValueError("Original Repair prompt changed")
    config.update({
        "COTEST_MAX_WORKERS": "4",
        "COTEST_JSONL": str(instances),
        "COTEST_RESULTS_DIR_PREFIX": str(output / "cases"),
        "COTEST_SUBMISSION_FILE": str(output / "submission.json"),
        "COTEST_PORTFOLIO_SCOUT_ONLY": "0",
        "COTEST_PORTFOLIO_REPAIR_ONLY": "0",
        "COTEST_PORTFOLIO_STOP_AFTER_REPAIR": "1",
        "COTEST_INTENT_PORTFOLIO_CACHE_DIR": "",
        "COTEST_INTENT_SCOUT_CACHE_DIR": "",
        "COTEST_REPAIR_SYSTEM_PROMPT_FILE": str(prompt),
    })
    baseline.write_json(output / "common_env.json", config)
    source_files = (*baseline.SOURCE_FILES, "scripts/run_full_flow_graph_compare.py",
                    "scripts/rerun_graph_pruning_cases.py")
    baseline.write_json(output / "manifest.json", {
        "status": "PREPARED", "source_run": str(source), "arm": "graph_on",
        "prepared_at": datetime.now().astimezone().isoformat(),
        "cases": [row["instance_id"] for row in rows], "workers": 4,
        "stages": ["Scout", "Hypothesis", "Repair"], "selector_ran": False,
        "scout_hypothesis_reused": False, "graph_enabled": True,
        "repair_visible_tools": ["Bash", "Editor", "CodeGraph", "Finish"],
        "model": "glm-5.3-flash", "reasoning_effort_all_stages": "low",
        "scout_steps": 60, "repair_steps_per_hypothesis": 60,
        "public_only_input": True,
        "instances_sha256": baseline.sha256(instances),
        "config_sha256": baseline.sha256(output / "common_env.json"),
        "prompt_sha256": baseline.sha256(prompt),
        "source_sha256": {name: baseline.sha256(ROOT / name) for name in source_files},
        "preflight": preflight,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-arm", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-prepared", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.run_prepared:
        return fullflow.run_arm(output)
    if args.source_arm is None:
        parser.error("Preparation requires --source-arm")
    prepare(args.source_arm.resolve(), output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
