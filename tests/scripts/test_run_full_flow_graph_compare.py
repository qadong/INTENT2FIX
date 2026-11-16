import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts/run_full_flow_graph_compare.py"
SPEC = importlib.util.spec_from_file_location("graph_comparison", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    rows = [{
        "repo": repo, "instance_id": f"{repo.replace('/', '__')}-{number}",
        "base_commit": "base", "problem_statement": "Public issue",
        "patch": "golden", "test_patch": "hidden", "FAIL_TO_PASS": ["hidden"],
    } for repo in MODULE.QUOTAS for number in range(12)]
    path = tmp_path / "instances.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(MODULE.baseline, "CANONICAL_DATASET", path)
    return rows


def test_selection_is_deterministic_held_out_and_public_only(dataset):
    previous = MODULE.select_cases()
    excluded = MODULE.PRIOR_CASES | frozenset(row["instance_id"] for row in previous)
    selected = MODULE.select_cases(excluded)
    assert selected == MODULE.select_cases(excluded)
    assert len(selected) == 10
    assert not excluded & {row["instance_id"] for row in selected}
    assert all(set(row) <= set(MODULE.baseline.PUBLIC_KEYS) for row in selected)
    for repo, count in MODULE.QUOTAS.items():
        assert sum(row["repo"] == repo for row in selected) == count


def test_all_remaining_selection_excludes_prior_and_unsupported_repository(dataset):
    previous = MODULE.select_cases()
    excluded = frozenset(row["instance_id"] for row in previous)
    rows = MODULE.baseline.read_jsonl(MODULE.baseline.CANONICAL_DATASET)
    rows.append({"repo": "yhirose/cpp-httplib", "instance_id": "yhirose__cpp-httplib-1765",
                 "base_commit": "base", "problem_statement": "Public issue"})
    MODULE.baseline.CANONICAL_DATASET.write_text(
        "".join(json.dumps(row) + "\n" for row in rows))
    selected = MODULE.select_cases(excluded, all_remaining=True)
    assert {row["instance_id"] for row in selected} == {
        row["instance_id"] for row in dataset if row["instance_id"] not in excluded}
    assert len(selected) == 38
    assert selected == MODULE.select_cases(excluded, all_remaining=True)
    assert all(set(row) <= set(MODULE.baseline.PUBLIC_KEYS) for row in selected)


def test_explicit_smoke_subset_remains_public_only(dataset, tmp_path, monkeypatch):
    base_config = {key: "low" for key in (
        "COTEST_GLM_REASONING_EFFORT", "COTEST_INTENT_SCOUT_GLM_REASONING_EFFORT",
        "COTEST_INTENT_EVIDENCE_REVIEW_GLM_REASONING_EFFORT", "COTEST_PATCH_INTENT_ADMISSION_REASONING_EFFORT",
    )}
    base_config.update({key: "glm-5.3-flash" for key in (
        "COTEST_REPAIR_MODEL", "COTEST_REPRO_MODEL", "COTEST_EXTRACTION_MODEL", "COTEST_INTENT_DIALOGUE_MODEL",
    )})
    base_config["COTEST_PORTFOLIO_REPAIR_STEPS"] = "60"
    config_path = tmp_path / "env.json"
    config_path.write_text(json.dumps(base_config))
    monkeypatch.setattr(MODULE.baseline, "A_ENV", config_path)
    monkeypatch.setattr(MODULE.baseline, "preflight", lambda rows: [])
    ids = {dataset[0]["instance_id"]}
    output = tmp_path / "smoke"
    MODULE.prepare(output, 1, case_ids=ids, stop_after_repair=True)
    manifest = json.loads((output / "manifest.json").read_text())
    assert set(manifest["cases"]) == ids
    assert manifest["selection_mode"] == "explicit_cases"
    for arm in MODULE.ARMS:
        row = json.loads((output / arm / "instances.jsonl").read_text())
        assert set(row) <= set(MODULE.baseline.PUBLIC_KEYS)


def test_explicit_prior_case_can_prepare_graph_on_rerun(dataset, tmp_path, monkeypatch):
    base_config = {key: "low" for key in (
        "COTEST_GLM_REASONING_EFFORT", "COTEST_INTENT_SCOUT_GLM_REASONING_EFFORT",
        "COTEST_INTENT_EVIDENCE_REVIEW_GLM_REASONING_EFFORT",
        "COTEST_PATCH_INTENT_ADMISSION_REASONING_EFFORT",
    )}
    base_config.update({key: "glm-5.3-flash" for key in (
        "COTEST_REPAIR_MODEL", "COTEST_REPRO_MODEL", "COTEST_EXTRACTION_MODEL",
        "COTEST_INTENT_DIALOGUE_MODEL",
    )})
    base_config["COTEST_PORTFOLIO_REPAIR_STEPS"] = "60"
    env_file = tmp_path / "env.json"
    env_file.write_text(json.dumps(base_config))
    monkeypatch.setattr(MODULE.baseline, "A_ENV", env_file)
    monkeypatch.setattr(MODULE.baseline, "preflight", lambda rows: [])
    prior = dataset[0]["instance_id"]
    monkeypatch.setattr(MODULE, "PRIOR_CASES", MODULE.PRIOR_CASES | {prior})
    output = tmp_path / "rerun"
    MODULE.prepare(output, 2, case_ids={prior}, allow_prior_cases=True,
                   arms=("graph_on",), stop_after_repair=True)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["cases"] == [prior]
    assert manifest["arms"] == ["graph_on"]
    assert (output / "graph_on/manifest.json").is_file()
    assert not (output / "graph_off").exists()
    row = json.loads((output / "graph_on/instances.jsonl").read_text())
    assert set(row) <= set(MODULE.baseline.PUBLIC_KEYS)


@pytest.mark.parametrize("all_remaining", [False, True])
def test_prepare_freezes_paired_low_config_and_skips_selector(
        dataset, tmp_path, monkeypatch, all_remaining):
    previous = MODULE.select_cases()
    exclusion = tmp_path / "previous.json"
    exclusion.write_text(json.dumps({"cases": [row["instance_id"] for row in previous]}))
    base_config = {key: "low" for key in (
        "COTEST_GLM_REASONING_EFFORT", "COTEST_INTENT_SCOUT_GLM_REASONING_EFFORT",
        "COTEST_INTENT_EVIDENCE_REVIEW_GLM_REASONING_EFFORT",
        "COTEST_PATCH_INTENT_ADMISSION_REASONING_EFFORT",
    )}
    base_config.update({key: "glm-5.3-flash" for key in (
        "COTEST_REPAIR_MODEL", "COTEST_REPRO_MODEL", "COTEST_EXTRACTION_MODEL",
        "COTEST_INTENT_DIALOGUE_MODEL",
    )})
    base_config["COTEST_PORTFOLIO_REPAIR_STEPS"] = "60"
    env_file = tmp_path / "base_env.json"
    env_file.write_text(json.dumps(base_config))
    monkeypatch.setattr(MODULE.baseline, "A_ENV", env_file)
    monkeypatch.setattr(MODULE.baseline, "preflight", lambda rows: [])
    output = tmp_path / "comparison"
    MODULE.prepare(output, 5, exclude_manifests=[exclusion],
                   all_remaining=all_remaining,
                   stop_after_repair=True, evaluate_all_candidates=True)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["evaluate_all_candidates"] is True
    assert manifest["workers_per_arm"] == 5
    assert len(manifest["cases"]) == (38 if all_remaining else 10)
    assert manifest["selection_mode"] == (
        "all_remaining" if all_remaining else "repository_quotas")
    assert not set(manifest["cases"]) & {row["instance_id"] for row in previous}
    assert manifest["exclusion_manifest_sha256"][str(exclusion)] == MODULE.baseline.sha256(exclusion)
    inputs = []
    for arm in MODULE.ARMS:
        directory = output / arm
        inputs.append((directory / "instances.jsonl").read_bytes())
        config = json.loads((directory / "common_env.json").read_text())
        frozen = json.loads((directory / "manifest.json").read_text())
        graph = arm == "graph_on"
        assert config["COTEST_SCOUT_GRAPH_SEARCH"] == str(int(graph))
        assert config["COTEST_REPAIR_GRAPH_SEARCH"] == str(int(graph))
        assert config["COTEST_PORTFOLIO_STOP_AFTER_REPAIR"] == "1"
        assert config["COTEST_INTENT_SCOUT_CACHE_DIR"] == ""
        assert config["COTEST_INTENT_PORTFOLIO_CACHE_DIR"] == ""
        assert config["COTEST_REPAIR_EVIDENCE_GUIDANCE"] == "1"
        assert config["COTEST_HYPOTHESIS_EVIDENCE_GUIDANCE"] == "1"
        assert all(config[key] == value for key, value in base_config.items())
        assert frozen["stages"] == ["Scout", "Hypothesis", "Repair"]
        assert frozen["selector_backend"] is None
        assert ("CodeGraph" in frozen["repair_visible_tools"]) is graph
        assert "scripts/evaluate_pair_candidate_patches.py" in frozen["source_sha256"]
        assert "INTENT2FIX/prompts/repair_evidence.md" in frozen["source_sha256"]
        assert "INTENT2FIX/prompts/hypothesis_evidence.md" in frozen["source_sha256"]
    assert inputs[0] == inputs[1]
    with pytest.raises(FileExistsError):
        MODULE.prepare(output, 5)


@pytest.mark.parametrize("arm_exit", [0, 1])
def test_all_candidates_evaluation_only_follows_successful_arms(tmp_path, monkeypatch, arm_exit):
    output = tmp_path / "comparison"
    output.mkdir()
    MODULE.baseline.write_json(output / "manifest.json", {
        "status": "PREPARED", "workers_per_arm": 5, "evaluate_all_candidates": True,
    })
    monkeypatch.setattr(MODULE.sys, "argv", [str(SCRIPT), "--output-dir", str(output),
                                          "--run-prepared"])
    monkeypatch.setattr(MODULE, "run_arm", lambda directory: arm_exit)
    commands = []
    monkeypatch.setattr(MODULE.subprocess, "run", lambda command, **kwargs:
                        commands.append(command) or SimpleNamespace(returncode=0))
    assert MODULE.main() == arm_exit
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == ("COMPLETED" if arm_exit == 0 else "FAILED")
    assert bool(commands) is (arm_exit == 0)
    if commands:
        assert commands[0][1] == str(MODULE.ROOT / "scripts/evaluate_pair_candidate_patches.py")
        assert commands[0][commands[0].index("--source-dir") + 1] == str(output)
        assert commands[0][commands[0].index("--workers") + 1] == "5"
