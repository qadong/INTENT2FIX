import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts/evaluate_pair_candidate_patches.py"
SPEC = importlib.util.spec_from_file_location("candidate_evaluator", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_run(directory, cases, candidates, status="COMPLETED"):
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps({
        "status": status, "exit_code": 0, "cases": cases,
    }))
    for instance_id in cases:
        case_dir = directory / "cases_glm-5.3-flash_cotest" / instance_id
        case_dir.mkdir(parents=True)
        (case_dir / "portfolio_candidate_pool.json").write_text(json.dumps(candidates))


def test_single_run_deduplicates_and_runs_its_own_group(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    cases = ["fmtlib__fmt-3732"]
    candidates = [
        {"patch": "patch A", "hypothesis_id": "H0", "variant_id": "initial"},
        {"patch": "patch A", "hypothesis_id": "H1", "variant_id": "initial"},
        {"patch": "patch B", "hypothesis_id": "H1", "variant_id": "continuation_1"},
    ]
    source, output = tmp_path / "source", tmp_path / "evaluation"
    write_run(source, cases, candidates)
    manifest = MODULE.prepare(source, output, workers=2, timeout=3600)
    assert manifest["arms"] == ("run",)
    assert manifest["raw_candidate_counts"] == {"run": 3}
    assert manifest["distinct_candidate_counts"] == {"run": 2}
    assert manifest["unique_docker_jobs"] == 2
    manifest, jobs = MODULE.load_prepared(output)
    assert len(jobs[0]["sources"]) == 2

    monkeypatch.setattr(MODULE.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(MODULE, "load_evaluator", lambda: SimpleNamespace(
        evaluate_patch=lambda job, **kwargs: {"resolved": True, "exit_code": 0},
    ))
    MODULE.run(output, manifest, jobs)
    summary = MODULE.read_json(output / "summary.json")
    assert summary["status"] == "COMPLETED"
    assert summary["evaluated"] == 2
    assert summary["arms"]["run"]["resolved_cases"] == 1
    assert summary["arms"]["run"]["passing_candidates"] == 2


@pytest.mark.parametrize("count", [1, 10, 20])
def test_paired_preparation_retains_existing_layout(tmp_path, monkeypatch, count):
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    cases = [f"fmtlib__fmt-{number}" for number in range(count)]
    source = tmp_path / "source"
    for arm in MODULE.ARMS:
        write_run(source / arm, cases, [{"patch": "same patch"}])
    manifest = MODULE.prepare(source, tmp_path / "evaluation", workers=5, timeout=3600)
    assert manifest["arms"] == MODULE.ARMS
    assert manifest["raw_candidate_counts"] == {"graph_on": count, "graph_off": count}
    assert manifest["unique_docker_jobs"] == count


@pytest.mark.parametrize("cases", [[], ["fmtlib__fmt-1", "fmtlib__fmt-1"]])
def test_paired_empty_or_duplicate_cases_are_rejected(tmp_path, monkeypatch, cases):
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    source = tmp_path / "source"
    for arm in MODULE.ARMS:
        write_run(source / arm, list(dict.fromkeys(cases)), [{"patch": "patch"}])
        manifest = source / arm / "manifest.json"
        payload = json.loads(manifest.read_text())
        payload["cases"] = cases
        manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="nonempty set of distinct"):
        MODULE.prepare(source, tmp_path / "evaluation", workers=1, timeout=3600)


def test_unfinished_single_run_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    source, output = tmp_path / "source", tmp_path / "evaluation"
    write_run(source, ["fmtlib__fmt-3732"], [{"patch": "patch"}], status="RUNNING")
    with pytest.raises(ValueError, match="must have completed"):
        MODULE.prepare(source, output, workers=5, timeout=3600)
    assert not output.exists()
