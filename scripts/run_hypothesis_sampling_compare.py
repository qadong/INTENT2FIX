#!/usr/bin/env python3
"""Repeat the frozen V2/V4 Hypothesis-only comparison three times per arm."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/hypothesis_sampling_compare3_20260925"
PRIOR = ROOT / "results/graph_scout_handoff_compare_20260923"
V2 = PRIOR / "hypothesis_uncertainty_source6_low_20260924"
V4 = PRIOR / "hypothesis_typeflow_contract3_low_20260924"
CASES = ("fmtlib__fmt-3819", "fmtlib__fmt-3824", "simdjson__simdjson-954")
SOURCE_FILES = (
    "INTENT2FIX/intent_dialogue/developer_agent.py",
    "INTENT2FIX/intent_dialogue/prompts.py",
    "INTENT2FIX/intent_dialogue/schema.py",
    "INTENT2FIX/intent_portfolio.py",
    "INTENT2FIX/hypothesis_source.py",
    "scripts/run_no_memory_cotest.py",
)
OLD_MEMBERS = (
    (
        ROOT / "results/checkpoints/hypothesis_uncertainty_branch_source_20260924/source_after_v2.tar.gz",
        "INTENT2FIX/intent_portfolio.py",
    ),
    (
        ROOT / "results/checkpoints/hypothesis_uncertainty_branch_source_20260924/contract_boundary_before_v3.tar.gz",
        "scripts/run_no_memory_cotest.py",
    ),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prepare_source(version: str) -> Path:
    source = OUT / "source" / version
    if not source.exists():
        source.mkdir(parents=True)
        for name in ("INTENT2FIX", "scripts"):
            shutil.copytree(ROOT / name, source / name, symlinks=True)
        (source / ".env").symlink_to(ROOT / ".env")
        if version == "v2":
            for archive, name in OLD_MEMBERS:
                with tarfile.open(archive, "r:gz") as bundle:
                    member = bundle.extractfile(name)
                    if member is None:
                        raise RuntimeError(f"Missing checkpoint member: {name}")
                    (source / name).write_bytes(member.read())
    expected = json.loads((V2 if version == "v2" else V4).joinpath("manifest.json").read_text())[
        "source_sha256"
    ]
    for name in SOURCE_FILES:
        actual = digest(source / name)
        if actual != expected[name]:
            raise RuntimeError(f"{version} source mismatch: {name}: {actual} != {expected[name]}")
    return source


def prepare_arm(version: str, repetition: int) -> Path:
    arm = OUT / f"{version}_r{repetition}"
    if arm.exists():
        raise RuntimeError(f"Refusing to overwrite existing arm: {arm}")
    arm.mkdir(parents=True)
    prior_rows = [json.loads(line) for line in (V2 / "instances.jsonl").read_text().splitlines()]
    by_id = {row["instance_id"]: row for row in prior_rows}
    (arm / "instances.jsonl").write_text(
        "".join(json.dumps(by_id[case], ensure_ascii=False) + "\n" for case in CASES)
    )
    config = json.loads((V2 / "common_env.json").read_text())
    config.update({
        "COTEST_JSONL": str(arm / "instances.jsonl"),
        "COTEST_RESULTS_DIR_PREFIX": str(arm / "cases"),
        "COTEST_SUBMISSION_FILE": str(arm / "submission.json"),
        "COTEST_INTENT_SCOUT_CACHE_DIR": str(V2 / "cases_glm-5.3-flash_cotest"),
        "COTEST_MAX_WORKERS": "3",
    })
    for key in (
        "COTEST_GLM_REASONING_EFFORT",
        "COTEST_INTENT_SCOUT_GLM_REASONING_EFFORT",
        "COTEST_INTENT_EVIDENCE_REVIEW_GLM_REASONING_EFFORT",
    ):
        if config[key] != "low":
            raise RuntimeError(f"Unexpected reasoning level: {key}={config[key]}")
    if config["COTEST_PORTFOLIO_SCOUT_ONLY"] != "1":
        raise RuntimeError("Repair must remain disabled")
    save(arm / "common_env.json", config)
    save(arm / "manifest.json", {
        "status": "PREPARED",
        "version": version,
        "repetition": repetition,
        "cases": CASES,
        "mode": "cached Scout -> Hypothesis/Audit; no Repair",
        "model": "glm-5.3-flash",
        "reasoning_effort": "low",
        "workers": 3,
        "scout_cache_dir": config["COTEST_INTENT_SCOUT_CACHE_DIR"],
        "source_sha256": {name: digest(OUT / "source" / version / name) for name in SOURCE_FILES},
        "instances_sha256": digest(arm / "instances.jsonl"),
        "config_sha256": digest(arm / "common_env.json"),
    })
    return arm


def run_arm(source: Path, arm: Path) -> int:
    manifest = json.loads((arm / "manifest.json").read_text())
    config = json.loads((arm / "common_env.json").read_text())
    for name, expected in manifest["source_sha256"].items():
        if digest(source / name) != expected:
            raise RuntimeError(f"Source drift before {arm.name}: {name}")
    if digest(arm / "instances.jsonl") != manifest["instances_sha256"]:
        raise RuntimeError(f"Instances drift before {arm.name}")
    if digest(arm / "common_env.json") != manifest["config_sha256"]:
        raise RuntimeError(f"Config drift before {arm.name}")
    manifest.update(status="RUNNING", started_at=datetime.now().astimezone().isoformat())
    save(arm / "manifest.json", manifest)
    command = [
        str(ROOT / ".venv/bin/python"), "-m", "scripts.run_no_memory_cotest",
        "--jsonl", str(arm / "instances.jsonl"), "--workers", "3",
        "--portfolio-scout-only", "--submission", str(arm / "submission.json"),
    ]
    with (arm / "launcher.log").open("w") as log:
        completed = subprocess.run(command, cwd=source, env={**os.environ, **config},
                                   stdout=log, stderr=subprocess.STDOUT, check=False)
    manifest.update(status="FINISHED" if completed.returncode == 0 else "FAILED",
                    exit_code=completed.returncode,
                    finished_at=datetime.now().astimezone().isoformat())
    save(arm / "manifest.json", manifest)
    return completed.returncode


def main() -> int:
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite existing experiment: {OUT}")
    OUT.mkdir(parents=True)
    sources = {version: prepare_source(version) for version in ("v2", "v4")}
    arms = [(version, prepare_arm(version, repetition))
            for repetition in range(1, 4) for version in ("v2", "v4")]
    for version, arm in arms:
        print(f"Starting {arm.name}", flush=True)
        code = run_arm(sources[version], arm)
        print(f"Finished {arm.name}: exit={code}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
