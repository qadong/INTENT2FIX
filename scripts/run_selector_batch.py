#!/usr/bin/env python3
"""Run a frozen Selector-only case list with bounded parallelism and checkpoints."""

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from INTENT2FIX.portfolio_selector import patch_hash

SELECTOR_SOURCES = (
    "INTENT2FIX/portfolio_selector.py",
    "INTENT2FIX/selector_execution.py",
    "scripts/run_portfolio_selector.py",
    "scripts/run_selector_batch.py",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prepare(args):
    frozen_manifest = json.loads((args.frozen / "manifest.json").read_text())
    inputs_path = args.frozen / "inputs.jsonl"
    if digest(inputs_path) != frozen_manifest["input_sha256"]:
        raise ValueError("Frozen Selector input hash mismatch")
    inputs = {row["instance_id"]: row for line in inputs_path.read_text().splitlines()
              if (row := json.loads(line))}
    summary = json.loads(args.case_summary.read_text())
    ids = [row["instance_id"] for row in summary["results"]]
    if len(ids) != args.expect_cases or len(set(ids)) != len(ids):
        raise ValueError("Case summary has the wrong count or duplicate IDs")
    if set(ids) - set(inputs):
        raise ValueError("Case summary includes IDs absent from frozen inputs")
    manifest = {
        "case_summary": str(args.case_summary.resolve()),
        "case_summary_sha256": digest(args.case_summary),
        "input_sha256": digest(inputs_path),
        "source_sha256": {name: digest(ROOT / name) for name in SELECTOR_SOURCES},
        "case_ids": ids,
        "candidate_count": sum(len(inputs[iid]["candidates"]) for iid in ids),
        "model": args.model,
        "reasoning_effort": "low",
        "execute_probes": not args.no_execute_probes,
        "workers": args.workers,
        "official_labels_in_model_input": False,
    }
    if args.resume:
        if json.loads((args.output / "manifest.json").read_text()) != manifest:
            raise ValueError("Resume manifest differs from the frozen run")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        save(args.output / "manifest.json", manifest)
    (args.output / "cases").mkdir(exist_ok=True)
    (args.output / "logs").mkdir(exist_ok=True)
    return manifest


def existing_result(output, iid, expected_input_sha):
    directory = output / "cases" / iid
    if not directory.exists():
        return None
    result_path = directory / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["input_sha256"] != expected_input_sha:
            raise ValueError(f"Stale Selector input for {iid}")
        selected = result.get("selected_patch_sha256")
        if selected and patch_hash((directory / "selected.patch").read_text()) != selected:
            raise ValueError(f"Selected patch hash mismatch for {iid}")
        return {"status": result["status"], "selected_patch_sha256": selected}
    if (directory / "error.json").exists():
        return {"status": "pipeline_error", "selected_patch_sha256": None}
    raise ValueError(f"Interrupted case directory requires inspection before resume: {directory}")


def run_one(args, iid):
    command = [
        sys.executable, str(ROOT / "scripts/run_portfolio_selector.py"),
        "--inputs", str(args.frozen / "inputs.jsonl"),
        "--output", str(args.output / "cases"),
        "--case", iid,
        "--model", args.model,
        "--reasoning-effort", "low",
    ]
    if not args.no_execute_probes:
        command.append("--execute-probes")
    start = time.monotonic()
    with (args.output / "logs" / f"{iid}.log").open("w") as log:
        process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    directory = args.output / "cases" / iid
    result_path = directory / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        status = result["status"]
        selected = result.get("selected_patch_sha256")
    else:
        status, selected = "pipeline_error", None
        directory.mkdir(exist_ok=True)
        if not (directory / "input.json").exists():
            for line in (args.frozen / "inputs.jsonl").read_text().splitlines():
                case = json.loads(line)
                if case["instance_id"] == iid:
                    save(directory / "input.json", case)
                    break
        if not (directory / "error.json").exists():
            save(directory / "error.json", {"instance_id": iid,
                                           "error_type": "SelectorSubprocessFailure"})
    return {"status": status, "selected_patch_sha256": selected,
            "exit_code": process.returncode, "seconds": round(time.monotonic() - start, 2)}


def run(args, manifest):
    frozen_manifest = json.loads((args.frozen / "manifest.json").read_text())
    rows = {}
    pending = []
    for iid in manifest["case_ids"]:
        previous = existing_result(args.output, iid,
                                   frozen_manifest["case_input_sha256"][iid])
        if previous is None:
            pending.append(iid)
        else:
            rows[iid] = previous

    def progress(message):
        status = {"updated_at": datetime.now(timezone.utc).isoformat(),
                  "total": len(manifest["case_ids"]), "completed": len(rows),
                  "selected": sum(row["status"] == "selected" for row in rows.values()),
                  "pipeline_errors": sum(row["status"] == "pipeline_error" for row in rows.values()),
                  "cases": rows}
        save(args.output / "progress.json", status)
        print(message, flush=True)
        with (args.output / "driver.log").open("a") as log:
            log.write(message + "\n")

    progress(f"START {len(pending)} pending / {len(manifest['case_ids'])} total, workers={args.workers}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_one, args, iid): iid for iid in pending}
        for future in concurrent.futures.as_completed(futures):
            iid = futures[future]
            try:
                rows[iid] = future.result()
            except Exception as exc:
                rows[iid] = {"status": "pipeline_error", "selected_patch_sha256": None,
                             "error_type": type(exc).__name__}
            progress(f"DONE {len(rows)}/{len(manifest['case_ids'])} {iid} {rows[iid]['status']}")
    progress("COMPLETE")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--case-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-cases", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", default="glm-5.3-flash")
    parser.add_argument("--no-execute-probes", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    args.frozen = args.frozen.resolve()
    args.case_summary = args.case_summary.resolve()
    args.output = args.output.resolve()
    manifest = prepare(args)
    if args.prepare_only:
        print(json.dumps({"cases": len(manifest["case_ids"]),
                          "candidates": manifest["candidate_count"],
                          "workers": args.workers, "output": str(args.output)}, indent=2))
        return
    run(args, manifest)


if __name__ == "__main__":
    main()
