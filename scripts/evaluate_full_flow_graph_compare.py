#!/usr/bin/env python3
"""Wait for a paired full-flow run and officially evaluate its finalists."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("graph_on", "graph_off")


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def wait_for_pipeline(run_dir: Path) -> list[str]:
    cases = read_json(run_dir / "manifest.json")["cases"]
    while True:
        manifests = {arm: read_json(run_dir / arm / "manifest.json") for arm in ARMS}
        if any(item["status"] == "FAILED" for item in manifests.values()):
            raise RuntimeError(f"Pipeline failed: {[(arm, item['status']) for arm, item in manifests.items()]}")
        if all(item["status"] == "COMPLETED" for item in manifests.values()):
            if any(item["exit_code"] != 0 or item["cases"] != cases
                   for item in manifests.values()):
                raise RuntimeError("Paired pipelines have incompatible or unsuccessful manifests")
            return cases
        time.sleep(30)


def evaluate(run_dir: Path, cases: list[str], workers: int) -> dict:
    output = run_dir / "official_eval"
    output.mkdir(exist_ok=True)
    summaries = {}
    for arm in ARMS:
        submission = run_dir / arm / "submission.json"
        rows = read_json(submission)
        submitted = [row["instance_id"] for row in rows]
        if len(submitted) != len(set(submitted)) or not set(submitted) <= set(cases):
            raise ValueError(f"Invalid finalist submission for {arm}")
        arm_dir = output / arm
        with (output / f"{arm}.log").open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/evaluate_fix_patches.py"),
                 str(submission), "--output-dir", str(arm_dir), "--workers", str(workers)],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False,
            )
        summary_path = arm_dir / "summary.json"
        if not summary_path.is_file():
            raise RuntimeError(f"Official evaluation did not produce {summary_path}; exit={result.returncode}")
        summaries[arm] = read_json(summary_path)
        print(f"{arm}: {summaries[arm]['resolved']}/{len(cases)} resolved", flush=True)

    per_case = {}
    for instance_id in cases:
        states = {}
        for arm in ARMS:
            result = next((row for row in summaries[arm]["results"]
                           if row["instance_id"] == instance_id), None)
            states[arm] = (
                "no_finalist" if result is None else
                "resolved" if result["resolved"] else
                "evaluation_error" if result.get("evaluation_error") or
                result.get("exit_code") in (124, 125, 126, 127) else
                "unresolved"
            )
        per_case[instance_id] = states
    comparison = {
        "completed_at": datetime.now().astimezone().isoformat(),
        "cases": per_case,
        "resolved": {arm: sum(states[arm] == "resolved" for states in per_case.values())
                     for arm in ARMS},
        "graph_on_only": [case for case, states in per_case.items()
                          if states["graph_on"] == "resolved" and states["graph_off"] != "resolved"],
        "graph_off_only": [case for case, states in per_case.items()
                           if states["graph_off"] == "resolved" and states["graph_on"] != "resolved"],
    }
    write_json(output / "comparison.json", comparison)
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    run_dir = args.run_dir.resolve()
    print(f"Waiting for both pipeline arms: {run_dir}", flush=True)
    cases = wait_for_pipeline(run_dir)
    print(f"Pipelines completed; evaluating finalists for {len(cases)} cases", flush=True)
    comparison = evaluate(run_dir, cases, args.workers)
    print(json.dumps(comparison["resolved"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
