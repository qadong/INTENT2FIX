#!/usr/bin/env python3
"""Run the V112-regression ten without graph tools, then evaluate every patch."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import rerun_v112_regressions_graph_on as paired  # noqa: E402
from scripts import run_full_flow_graph_compare as flow  # noqa: E402
from scripts import run_full_pipeline_unified_repair as baseline  # noqa: E402


RUN = ROOT / "results/fullflow_v112_regression10_graph_off_low_20260928"
ARM = RUN / "graph_off"
ON_ARM = paired.RUN / "graph_on"
PYTHON = ROOT / ".venv/bin/python"
EXTRA_SOURCES = tuple(dict.fromkeys((
    *paired.EXTRA_SOURCES,
    "scripts/rerun_v112_regressions_graph_off.py",
)))


def validate() -> None:
    root = paired.read(RUN / "manifest.json")
    arm = paired.read(ARM / "manifest.json")
    assert root["status"] == arm["status"] == "PREPARED"
    assert set(root["cases"]) == paired.CASE_IDS and root["arms"] == ["graph_off"]
    assert arm["stages"] == ["Scout", "Hypothesis", "Repair"]
    assert arm["selector_backend"] is None and arm["graph_enabled"] is False
    assert arm["repair_visible_tools"] == ["Bash", "Editor", "Finish"]
    assert arm["reasoning_effort_all_stages"] == "low"
    assert arm["repair_steps_per_hypothesis"] == 60
    assert arm["source_sha256"] == {
        name: baseline.sha256(ROOT / name) for name in arm["source_sha256"]
    }
    previous = paired.read(ON_ARM / "manifest.json")
    assert previous["source_sha256"] == {
        name: baseline.sha256(ROOT / name) for name in previous["source_sha256"]
    }
    for name, expected in (
        ("instances.jsonl", arm["instances_sha256"]),
        ("common_env.json", arm["config_sha256"]),
        ("repair_system_prompt.txt", arm["prompt_sha256"]),
    ):
        assert baseline.sha256(ARM / name) == expected, name
    config = paired.read(ARM / "common_env.json")
    assert config["COTEST_SCOUT_GRAPH_SEARCH"] == "0"
    assert config["COTEST_REPAIR_GRAPH_SEARCH"] == "0"
    assert config["COTEST_BRANCH_CONTEXT"] == "off"
    assert config["COTEST_INTENT_TEST_ANCHOR_RETRIEVAL"] == "0"
    assert config["COTEST_PORTFOLIO_STOP_AFTER_REPAIR"] == "1"
    assert config["COTEST_PORTFOLIO_REPAIR_STEPS"] == "60"
    assert config["COTEST_INTENT_SCOUT_CACHE_DIR"] == ""
    assert config["COTEST_INTENT_PORTFOLIO_CACHE_DIR"] == ""
    assert all(value == "low" for key, value in config.items()
               if "REASONING_EFFORT" in key)
    assert all(config[key] == "glm-5.3-flash" for key in (
        "COTEST_REPAIR_MODEL", "COTEST_REPRO_MODEL",
        "COTEST_EXTRACTION_MODEL", "COTEST_INTENT_DIALOGUE_MODEL",
    ))
    assert baseline.read_jsonl(ARM / "instances.jsonl") == baseline.read_jsonl(
        ON_ARM / "instances.jsonl"
    )


def prepare() -> None:
    flow.prepare(RUN, 5, case_ids=set(paired.CASE_IDS), allow_prior_cases=True,
                 arms=("graph_off",), stop_after_repair=True)
    arm = paired.read(ARM / "manifest.json")
    arm["source_sha256"].update({
        name: baseline.sha256(ROOT / name) for name in EXTRA_SOURCES
    })
    paired.write(ARM / "manifest.json", arm)
    root = paired.read(RUN / "manifest.json")
    root.update(
        experiment="V112-only graph-off paired control",
        paired_graph_on=str(paired.RUN),
        paired_graph_on_manifest_sha256=baseline.sha256(ON_ARM / "manifest.json"),
        selector_skipped=True,
        candidate_metric="any candidate passes official tests",
        model_inputs="public Issue and baseline repository only",
    )
    paired.write(RUN / "manifest.json", root)
    validate()
    print(f"PREPARED {RUN}: same ten public cases, graph-off, five workers, all low", flush=True)


def evaluation_source() -> tuple[Path | None, list[str]]:
    source = RUN / "official_source_graph_off"
    cases_dir = source / "cases_glm-5.3-flash_cotest"
    available = []
    missing = []
    for case in paired.read(ARM / "manifest.json")["cases"]:
        directory = ARM / "cases_glm-5.3-flash_cotest" / case
        status = directory / "portfolio_repair_status.json"
        pool = directory / "portfolio_candidate_pool.json"
        if (status.is_file() and pool.is_file()
                and paired.read(status).get("status") == "completed"
                and isinstance(json.loads(pool.read_text(encoding="utf-8")), list)):
            available.append((case, directory))
        else:
            missing.append(case)
    if not available:
        return None, missing
    cases_dir.mkdir(parents=True)
    for case, directory in available:
        (cases_dir / case).symlink_to(directory, target_is_directory=True)
    paired.write(source / "manifest.json", {
        "status": "COMPLETED", "exit_code": 0,
        "cases": [case for case, _ in available], "source_snapshot": True,
    })
    paired.write(source / "source_provenance.json", {
        "prepared_at": paired.now(), "original_run": str(RUN),
        "paired_graph_on": str(paired.RUN),
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
    root = paired.read(RUN / "manifest.json")
    root.update(status="RUNNING", started_at=paired.now(), controller_pid=os.getpid())
    paired.write(RUN / "manifest.json", root)
    repair_exit = flow.run_arm(ARM)
    root = paired.read(RUN / "manifest.json")
    root["repair_exit_code"] = repair_exit
    if repair_exit != 0:
        root.update(status="FAILED", finished_at=paired.now())
        paired.write(RUN / "manifest.json", root)
        return repair_exit
    source, missing = evaluation_source()
    root["missing_generation"] = missing
    if source is None:
        root.update(status="PARTIAL", finished_at=paired.now(), evaluation="NO_POOLS")
        paired.write(RUN / "manifest.json", root)
        return 1
    root.update(status="EVALUATING", official_source=str(source))
    paired.write(RUN / "manifest.json", root)
    command = [str(PYTHON), str(paired.EVALUATOR), "--source-dir", str(source),
               "--output-dir", str(RUN / "official_eval_graph_off"),
               "--workers", "5", "--timeout", "3600"]
    with (RUN / "official_eval_graph_off_launcher.log").open("x", encoding="utf-8") as log:
        evaluation = subprocess.run(command, cwd=ROOT, stdout=log,
                                    stderr=subprocess.STDOUT, check=False)
    root = paired.read(RUN / "manifest.json")
    root.update(status=("PARTIAL" if missing or evaluation.returncode else "COMPLETED"),
                evaluation_exit_code=evaluation.returncode, finished_at=paired.now())
    paired.write(RUN / "manifest.json", root)
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
    paired.write(RUN / "background_launch.json", {
        "started_at": paired.now(), "pid": process.pid,
        "log": str(log), "detachment": "nohup + setsid",
        "workers": 5, "official_evaluation_after_repair": True,
    })
    print(f"LAUNCHED pid={process.pid} log={log}", flush=True)


if __name__ == "__main__":
    commands = {"prepare": prepare, "launch": launch, "run": run}
    if sys.argv[1:] not in (["prepare"], ["launch"], ["run"]):
        raise SystemExit("Use prepare, launch, or run")
    raise SystemExit(commands[sys.argv[1]]() or 0)
