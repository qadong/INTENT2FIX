#!/usr/bin/env python3
"""Officially evaluate all candidates from a completed single or paired run."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import threading


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "scripts/evaluate_fix_patches.py"
ARMS = ("graph_on", "graph_off")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_evaluator():
    spec = importlib.util.spec_from_file_location("official_evaluator", EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(source: Path, output: Path, workers: int, timeout: int) -> dict:
    if output.exists():
        raise FileExistsError(output)
    paired = (source / "graph_on" / "manifest.json").is_file()
    arms = ARMS if paired else ("run",)
    directories = {arm: source / arm if paired else source for arm in arms}
    manifests = {arm: read_json(directory / "manifest.json")
                 for arm, directory in directories.items()}
    cases = manifests[arms[0]]["cases"]
    if not cases or len(set(cases)) != len(cases):
        raise ValueError("Expected a nonempty set of distinct cases")
    if any(item["status"] != "COMPLETED" or item["exit_code"] != 0 or
           item["cases"] != cases for item in manifests.values()):
        raise ValueError("Source runs must have completed the same cases")

    source_hashes = {}
    jobs = {}
    raw_counts = {arm: 0 for arm in arms}
    distinct_counts = {arm: 0 for arm in arms}
    for arm, directory in directories.items():
        manifest_path = directory / "manifest.json"
        source_hashes[str(manifest_path.relative_to(ROOT))] = sha256(manifest_path.read_bytes())
        for instance_id in cases:
            pool = directory / "cases_glm-5.3-flash_cotest" / instance_id / "portfolio_candidate_pool.json"
            source_hashes[str(pool.relative_to(ROOT))] = sha256(pool.read_bytes())
            seen = set()
            for candidate in read_json(pool):
                patch = candidate["patch"]
                if not isinstance(patch, str) or not patch.strip():
                    raise ValueError(f"Empty candidate patch: {arm} {instance_id}")
                patch_hash = sha256(patch.encode("utf-8"))
                key = (instance_id, patch_hash)
                raw_counts[arm] += 1
                if key not in seen:
                    distinct_counts[arm] += 1
                    seen.add(key)
                job = jobs.setdefault(key, {
                    "instance_id": instance_id, "patch_sha256": patch_hash,
                    "model_patch": patch, "sources": [],
                })
                job["sources"].append({
                    "arm": arm, "hypothesis_id": candidate.get("hypothesis_id"),
                    "variant_id": candidate.get("variant_id"),
                    "reported_sha256": candidate.get("patch_sha256"),
                })

    ordered = [job for case in cases for (instance_id, _), job in jobs.items()
               if instance_id == case]
    for number, job in enumerate(ordered, 1):
        job["candidate_key"] = f"{job['instance_id']}:{job['patch_sha256']}"
        job["job_id"] = f"{number:04d}_{job['patch_sha256'][:12]}"
    output.mkdir(parents=True)
    lines = "".join(json.dumps(job, ensure_ascii=False) + "\n" for job in ordered)
    (output / "candidates.jsonl").write_text(lines, encoding="utf-8")
    evaluator = load_evaluator()
    manifest = {
        "status": "PREPARED", "prepared_at": datetime.now().astimezone().isoformat(),
        "source": str(source), "cases": cases, "arms": arms,
        "raw_candidate_counts": raw_counts, "distinct_candidate_counts": distinct_counts,
        "unique_docker_jobs": len(ordered), "workers": workers,
        "timeout_seconds": timeout, "registry": evaluator.DEFAULT_REGISTRY,
        "candidates_sha256": sha256(lines.encode("utf-8")),
        "evaluator_sha256": sha256(EVALUATOR.read_bytes()),
        "source_sha256": source_hashes,
    }
    save_json(output / "manifest.json", manifest)
    return manifest


def load_prepared(output: Path) -> tuple[dict, list[dict]]:
    manifest = read_json(output / "manifest.json")
    if manifest["status"] not in {"PREPARED", "RUNNING", "COMPLETED"}:
        raise ValueError("Evaluation is not resumable in its current state")
    candidate_file = output / "candidates.jsonl"
    if sha256(candidate_file.read_bytes()) != manifest["candidates_sha256"]:
        raise ValueError("Candidate manifest changed")
    if sha256(EVALUATOR.read_bytes()) != manifest["evaluator_sha256"]:
        raise ValueError("Official evaluator changed")
    for name, expected in manifest["source_sha256"].items():
        if sha256((ROOT / name).read_bytes()) != expected:
            raise ValueError(f"Source changed: {name}")
    jobs = [json.loads(line) for line in candidate_file.read_text(encoding="utf-8").splitlines()]
    if len(jobs) != manifest["unique_docker_jobs"]:
        raise ValueError("Candidate count changed")
    return manifest, jobs


def run(output: Path, manifest: dict, jobs: list[dict]) -> None:
    subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                   check=True, capture_output=True, text=True, timeout=30)
    evaluator = load_evaluator()
    results = {}
    for job in jobs:
        result_file = output / "candidates" / job["job_id"] / job["instance_id"] / "result.json"
        if result_file.is_file():
            result = read_json(result_file)
            if result.get("candidate_key") == job["candidate_key"]:
                results[job["candidate_key"]] = result
    lock = threading.Lock()
    active = set()

    def summarize() -> str:
        arms = {}
        for arm in manifest["arms"]:
            cases = []
            for instance_id in manifest["cases"]:
                selected = [job for job in jobs if job["instance_id"] == instance_id
                            and any(source["arm"] == arm for source in job["sources"])]
                done = [results[job["candidate_key"]] for job in selected
                        if job["candidate_key"] in results]
                passing = [row["patch_sha256"] for row in done if row["resolved"]]
                uncertain = any(row.get("evaluation_error") or
                                row.get("exit_code") in (124, 125, 126, 127) for row in done)
                state = ("resolved" if passing else "pending" if len(done) < len(selected)
                         else "no_candidate" if not selected else
                         "evaluation_incomplete" if uncertain else "no_passing_candidate")
                cases.append({
                    "instance_id": instance_id, "candidate_count": len(selected),
                    "evaluated": len(done), "passing_patch_sha256": passing, "state": state,
                })
            arms[arm] = {
                "resolved_cases": sum(case["state"] == "resolved" for case in cases),
                "passing_candidates": sum(len(case["passing_patch_sha256"]) for case in cases),
                "cases": cases,
            }
        status = "COMPLETED" if len(results) == len(jobs) else "RUNNING"
        save_json(output / "summary.json", {
            "status": status, "updated_at": datetime.now().astimezone().isoformat(),
            "total": len(jobs), "evaluated": len(results), "running": len(active),
            "queued": len(jobs) - len(results) - len(active), "arms": arms,
        })
        return status

    def evaluate(job: dict) -> None:
        key = job["candidate_key"]
        with lock:
            active.add(key)
            summarize()
            print(f"START {job['job_id']} {job['instance_id']}", flush=True)
        job_dir = output / "candidates" / job["job_id"]
        try:
            result = evaluator.evaluate_patch(
                job, output_dir=job_dir, registry=manifest["registry"],
                timeout=manifest["timeout_seconds"],
            )
        except Exception as exc:
            result = {
                "instance_id": job["instance_id"], "resolved": False,
                "exit_code": 125, "evaluation_error": f"{type(exc).__name__}: {exc}",
            }
        result.update({
            "candidate_key": key, "job_id": job["job_id"],
            "patch_sha256": job["patch_sha256"], "sources": job["sources"],
        })
        result_file = job_dir / job["instance_id"] / "result.json"
        result_file.parent.mkdir(parents=True, exist_ok=True)
        save_json(result_file, result)
        with lock:
            active.remove(key)
            results[key] = result
            summarize()
            print(f"DONE {job['job_id']} {job['instance_id']} "
                  f"resolved={result['resolved']} exit={result['exit_code']}", flush=True)

    manifest["status"] = "RUNNING"
    save_json(output / "manifest.json", manifest)
    with lock:
        summarize()
    pending = [job for job in jobs if job["candidate_key"] not in results]
    with ThreadPoolExecutor(max_workers=manifest["workers"]) as executor:
        list(executor.map(evaluate, pending))
    with lock:
        manifest["status"] = summarize()
        save_json(output / "manifest.json", manifest)
    print("EVALUATION_COMPLETE", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-prepared", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.timeout < 1:
        parser.error("workers and timeout must be positive")
    output = args.output_dir.resolve()
    if args.run_prepared:
        manifest, jobs = load_prepared(output)
        run(output, manifest, jobs)
        return 0
    if args.source_dir is None:
        parser.error("source-dir is required when preparing")
    manifest = prepare(args.source_dir.resolve(), output, args.workers, args.timeout)
    print(f"PREPARED {manifest['unique_docker_jobs']} unique jobs from "
          f"{manifest['raw_candidate_counts']}", flush=True)
    if not args.prepare_only:
        loaded, jobs = load_prepared(output)
        run(output, loaded, jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
