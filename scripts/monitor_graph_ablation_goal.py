#!/usr/bin/env python3
"""Half-hour monitor and phase-1 -> official-evaluation handoff."""

from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def service_state(unit: str) -> dict[str, str]:
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "-p",
            "LoadState",
            "-p",
            "ActiveState",
            "-p",
            "SubState",
            "-p",
            "MainPID",
            "-p",
            "Result",
            "-p",
            "ExecMainStatus",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    values: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    if result.returncode and not values:
        values["error"] = result.stderr.strip() or f"exit={result.returncode}"
    return values


def phase1_snapshot(batch_dir: Path, id_manifest: Path) -> dict[str, Any]:
    expected = {
        json.loads(line)["instance_id"]
        for line in id_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    case_dirs = {
        path.name
        for path in batch_dir.iterdir()
        if path.is_dir() and path.name in expected
    }
    status_paths = {
        path.parent.name: path
        for path in batch_dir.glob("*/portfolio_repair_status.json")
    }
    pool_paths = {
        path.parent.name: path
        for path in batch_dir.glob("*/portfolio_candidate_pool.json")
    }
    candidate_counts = {
        instance_id: len(read_json(path))
        for instance_id, path in pool_paths.items()
    }
    inconsistencies = []
    for instance_id, status_path in status_paths.items():
        status = read_json(status_path)
        pool_count = candidate_counts.get(instance_id)
        if (
            status.get("status") != "completed"
            or status.get("candidate_count") != pool_count
            or status.get("selector_ran") is not False
        ):
            inconsistencies.append(instance_id)
    return {
        "scope": len(expected),
        "started": len(case_dirs),
        "completed": len(status_paths),
        "active_estimate": len(case_dirs - set(status_paths)),
        "queued": len(expected - case_dirs),
        "candidate_cases": sum(count > 0 for count in candidate_counts.values()),
        "total_candidates": sum(candidate_counts.values()),
        "cases_without_candidates": sorted(
            instance_id
            for instance_id, count in candidate_counts.items()
            if count == 0
        ),
        "inconsistencies": sorted(inconsistencies),
        "missing_status_ids": sorted(expected - set(status_paths)),
    }


def start_official_evaluation(unit: str, run_script: Path) -> str:
    command = [
        "systemd-run",
        "--user",
        f"--unit={unit}",
        "--collect",
        f"--working-directory={ROOT}",
        "--property=KillMode=control-group",
        "--property=TimeoutStartSec=infinity",
        sys.executable,
        str(run_script),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(
            "Failed to launch official evaluation: "
            + (result.stderr.strip() or result.stdout.strip())
        )
    return result.stdout.strip()


def official_eval_progress(eval_dir: Path) -> dict[str, Any]:
    rows = []
    invalid = []
    for path in (eval_dir / "candidate_eval").rglob("result.json"):
        try:
            rows.append(read_json(path))
        except Exception as error:
            invalid.append(f"{path}: {type(error).__name__}: {error}")
    resolved = [row for row in rows if row.get("resolved") is True]
    resolved_cases = sorted(
        {
            str(row["instance_id"])
            for row in resolved
            if row.get("instance_id")
        }
    )
    return {
        "result_file_count": len(rows),
        "resolved_patches_so_far": len(resolved),
        "resolved_cases_so_far": len(resolved_cases),
        "invalid_result_files": invalid,
    }


def evaluate_transition(args: argparse.Namespace, snapshot: dict[str, Any]) -> None:
    phase1 = snapshot["phase1"]
    if phase1["completed"] != phase1["scope"]:
        snapshot["phase"] = "graph_ablation"
        return
    if phase1["inconsistencies"]:
        snapshot["phase"] = "phase1_invalid"
        snapshot["action_required"] = "repair inconsistent completed artifacts"
        return

    eval_dir = args.eval_prepare.parent
    manifest_path = eval_dir / "manifest.json"
    if not manifest_path.exists():
        prepared = subprocess.run(
            [sys.executable, str(args.eval_prepare)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        snapshot["prepare"] = {
            "exit_code": prepared.returncode,
            "stdout": prepared.stdout.strip(),
            "stderr": prepared.stderr.strip(),
        }
        if prepared.returncode:
            snapshot["phase"] = "official_eval_prepare_failed"
            return

    manifest = read_json(manifest_path)
    snapshot["official_eval"] = {
        key: manifest.get(key)
        for key in (
            "status",
            "candidate_count",
            "candidate_case_count",
            "completed_candidates",
            "resolved_candidates",
            "resolved_case_count",
            "cases_without_candidates",
            "summary",
        )
    }
    snapshot["official_eval"].update(official_eval_progress(eval_dir))
    eval_service = service_state(args.eval_service)
    snapshot["official_eval_service"] = eval_service
    if manifest["status"] == "COMPLETED":
        snapshot["phase"] = "official_eval_completed"
        return
    if eval_service.get("ActiveState") == "active":
        snapshot["phase"] = "official_eval_running"
        return
    if manifest["status"] == "PREPARED":
        snapshot["launch"] = start_official_evaluation(
            args.eval_service, args.eval_run
        )
        snapshot["official_eval_service"] = service_state(args.eval_service)
        snapshot["phase"] = "official_eval_started"
        return
    snapshot["phase"] = "official_eval_not_running"
    snapshot["action_required"] = (
        f"manifest status is {manifest['status']} while service is inactive"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--id-manifest", type=Path, required=True)
    parser.add_argument("--phase1-service", required=True)
    parser.add_argument("--eval-prepare", type=Path, required=True)
    parser.add_argument("--eval-run", type=Path, required=True)
    parser.add_argument("--eval-service", required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    with (args.state_dir / "monitor.lock").open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot = {
            "polled_at": now(),
            "phase1_service": service_state(args.phase1_service),
            "phase1": phase1_snapshot(args.batch_dir, args.id_manifest),
        }
        try:
            evaluate_transition(args, snapshot)
        except Exception as error:
            snapshot["phase"] = "monitor_error"
            snapshot["error"] = f"{type(error).__name__}: {error}"
        write_json(args.state_dir / "latest.json", snapshot)
        with (args.state_dir / "history.jsonl").open(
            "a", encoding="utf-8"
        ) as history:
            history.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
        print(json.dumps(snapshot, ensure_ascii=False, indent=2), flush=True)
        return 1 if snapshot["phase"].endswith(("failed", "error")) else 0


if __name__ == "__main__":
    raise SystemExit(main())
