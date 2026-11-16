#!/usr/bin/env python3
"""Rerun the ten V112-only successes with the current graph-on full flow."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_full_flow_graph_compare as flow  # noqa: E402
from scripts import run_full_pipeline_unified_repair as baseline  # noqa: E402


RUN = ROOT / "results/fullflow_v112_regression10_graph_on_low_20260928"
PYTHON = ROOT / ".venv/bin/python"
EVALUATOR = ROOT / "scripts/evaluate_pair_candidate_patches.py"
CASE_IDS = frozenset({
    "fmtlib__fmt-2610", "fmtlib__fmt-3260", "fmtlib__fmt-3555",
    "nlohmann__json-1343", "nlohmann__json-2576", "nlohmann__json-2963",
    "nlohmann__json-3543", "nlohmann__json-4512",
    "simdjson__simdjson-1667", "simdjson__simdjson-524",
})
EXTRA_SOURCES = (
    "scripts/rerun_v112_regressions_graph_on.py",
    "scripts/evaluate_pair_candidate_patches.py",
    "scripts/evaluate_fix_patches.py",
    "INTENT2FIX/actions/run_command.py",
    "INTENT2FIX/actions/verification_sync.py",
    "INTENT2FIX/file_context.py",
    "INTENT2FIX/patch_review/sanitizer.py",
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict) -> None:
    baseline.write_json(path, value)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def validate() -> None:
    manifest = read(RUN / "manifest.json")
    arm = read(RUN / "graph_on/manifest.json")
    assert manifest["status"] == arm["status"] == "PREPARED"
    assert set(manifest["cases"]) == CASE_IDS and manifest["arms"] == ["graph_on"]
    assert arm["stages"] == ["Scout", "Hypothesis", "Repair"]
    assert arm["selector_backend"] is None and arm["graph_enabled"] is True
    assert arm["reasoning_effort_all_stages"] == "low"
    assert arm["repair_steps_per_hypothesis"] == 60
    assert arm["source_sha256"] == {
        name: baseline.sha256(ROOT / name) for name in arm["source_sha256"]
    }
    for name, expected in (
        ("instances.jsonl", arm["instances_sha256"]),
        ("common_env.json", arm["config_sha256"]),
        ("repair_system_prompt.txt", arm["prompt_sha256"]),
    ):
        assert baseline.sha256(RUN / "graph_on" / name) == expected, name
    config = read(RUN / "graph_on/common_env.json")
    assert config["COTEST_PORTFOLIO_STOP_AFTER_REPAIR"] == "1"
    assert config["COTEST_SCOUT_GRAPH_SEARCH"] == "1"
    assert config["COTEST_REPAIR_GRAPH_SEARCH"] == "1"
    assert config["COTEST_PORTFOLIO_REPAIR_STEPS"] == "60"
    assert config["COTEST_INTENT_SCOUT_CACHE_DIR"] == ""
    assert config["COTEST_INTENT_PORTFOLIO_CACHE_DIR"] == ""
    assert all(value == "low" for key, value in config.items()
               if "REASONING_EFFORT" in key)
    assert all(config[key] == "glm-5.3-flash" for key in (
        "COTEST_REPAIR_MODEL", "COTEST_REPRO_MODEL",
        "COTEST_EXTRACTION_MODEL", "COTEST_INTENT_DIALOGUE_MODEL",
    ))
    rows = baseline.read_jsonl(RUN / "graph_on/instances.jsonl")
    assert len(rows) == 10 and {row["instance_id"] for row in rows} == CASE_IDS
    assert all(set(row) <= set(baseline.PUBLIC_KEYS) for row in rows)


def prepare() -> None:
    flow.prepare(RUN, 5, case_ids=set(CASE_IDS), allow_prior_cases=True,
                 arms=("graph_on",), stop_after_repair=True)
    path = RUN / "graph_on/manifest.json"
    arm = read(path)
    arm["source_sha256"].update({
        name: baseline.sha256(ROOT / name) for name in EXTRA_SOURCES
    })
    write(path, arm)
    manifest_path = RUN / "manifest.json"
    manifest = read(manifest_path)
    manifest.update(
        experiment="V112-only graph-on regression recovery",
        requested_scope="ten V112-passing/current-graph-on-failing cases",
        selector_skipped=True,
        candidate_metric="any candidate passes official tests",
        model_inputs="public Issue and baseline repository only",
    )
    write(manifest_path, manifest)
    validate()
    print(f"PREPARED {RUN}: ten public cases, graph-on, five workers, all low", flush=True)


def evaluation_source() -> tuple[Path | None, list[str]]:
    arm = read(RUN / "graph_on/manifest.json")
    source = RUN / "official_source_graph_on"
    cases_dir = source / "cases_glm-5.3-flash_cotest"
    available = []
    missing = []
    for case in arm["cases"]:
        directory = RUN / "graph_on/cases_glm-5.3-flash_cotest" / case
        status = directory / "portfolio_repair_status.json"
        pool = directory / "portfolio_candidate_pool.json"
        if (status.is_file() and pool.is_file()
                and read(status).get("status") == "completed"
                and isinstance(json.loads(pool.read_text(encoding="utf-8")), list)):
            available.append((case, directory))
        else:
            missing.append(case)
    if not available:
        return None, missing
    cases_dir.mkdir(parents=True)
    for case, directory in available:
        (cases_dir / case).symlink_to(directory, target_is_directory=True)
    write(source / "manifest.json", {
        "status": "COMPLETED", "exit_code": 0,
        "cases": [case for case, _ in available],
        "source_snapshot": True,
    })
    write(source / "source_provenance.json", {
        "prepared_at": now(), "original_run": str(RUN),
        "available_cases": [case for case, _ in available],
        "missing_generation": missing,
        "missing_generation_is_not_official_failure": True,
        "pool_sha256": {
            case: baseline.sha256(directory / "portfolio_candidate_pool.json")
            for case, directory in available
        },
    })
    return source, missing


def run() -> int:
    validate()
    manifest_path = RUN / "manifest.json"
    manifest = read(manifest_path)
    manifest.update(status="RUNNING", started_at=now(), controller_pid=os.getpid())
    write(manifest_path, manifest)
    repair_exit = flow.run_arm(RUN / "graph_on")
    manifest = read(manifest_path)
    manifest["repair_exit_code"] = repair_exit
    if repair_exit != 0:
        manifest.update(status="FAILED", finished_at=now())
        write(manifest_path, manifest)
        return repair_exit
    source, missing = evaluation_source()
    manifest["missing_generation"] = missing
    if source is None:
        manifest.update(status="PARTIAL", finished_at=now(), evaluation="NO_POOLS")
        write(manifest_path, manifest)
        return 1
    manifest.update(status="EVALUATING", official_source=str(source))
    write(manifest_path, manifest)
    command = [str(PYTHON), str(EVALUATOR), "--source-dir", str(source),
               "--output-dir", str(RUN / "official_eval_graph_on"),
               "--workers", "5", "--timeout", "3600"]
    with (RUN / "official_eval_graph_on_launcher.log").open("x", encoding="utf-8") as log:
        evaluation = subprocess.run(command, cwd=ROOT, stdout=log,
                                    stderr=subprocess.STDOUT, check=False)
    manifest = read(manifest_path)
    manifest.update(status=("PARTIAL" if missing or evaluation.returncode else "COMPLETED"),
                    evaluation_exit_code=evaluation.returncode, finished_at=now())
    write(manifest_path, manifest)
    return evaluation.returncode


def launch() -> None:
    validate()
    subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                   check=True, capture_output=True, text=True, timeout=30)
    command = ["nohup", "setsid", "--wait", str(PYTHON), str(Path(__file__).resolve()), "run"]
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("COTEST_")}
    environment["PYTHONUNBUFFERED"] = "1"
    log = RUN / "launcher.log"
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                   stdin=subprocess.DEVNULL, stdout=stream,
                                   stderr=subprocess.STDOUT)
    write(RUN / "background_launch.json", {
        "started_at": now(), "pid": process.pid,
        "log": str(log), "detachment": "nohup + setsid",
        "workers": 5, "official_evaluation_after_repair": True,
    })
    print(f"LAUNCHED pid={process.pid} log={log}", flush=True)


if __name__ == "__main__":
    commands = {"prepare": prepare, "launch": launch, "run": run}
    if sys.argv[1:] not in (["prepare"], ["launch"], ["run"]):
        raise SystemExit("Use prepare, launch, or run")
    raise SystemExit(commands[sys.argv[1]]() or 0)
