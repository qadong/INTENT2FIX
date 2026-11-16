import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from INTENT2FIX.portfolio_selector import (
    Assessment, Citation, Clause, Comparison, Contract, Decision, Judgment, Probe, base_sources,
    check_apply, check_contract, check_decision, check_fixture_refinement, freeze_candidates, ground_contract, ground_decision,
    ground_probe_route, indistinguishable_groups, issue_contract, patch_hash, public_header_route, rank_payload, resolve,
    search_base_sources,
)
from INTENT2FIX.selector_execution import BINARY, PROBE_PATH, compile_candidate, execution_rows, probe_commands, refinement_args


def script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sample():
    candidates = freeze_candidates([{"patch": "patch A"}, {"patch": "patch B"}])
    for candidate in candidates:
        candidate["application"] = {"status": "passed"}
    case = {"instance_id": "org__repo-1", "repo": "org/repo", "base_commit": "base",
            "sources": [{"source_id": "ISSUE", "text": "preserve sign of zero"}],
            "candidates": candidates, "evidence": []}
    contract = Contract(clauses=[Clause(clause_id="K1", kind="target", requirement="preserve sign of zero",
                                       citations=[Citation(source_id="ISSUE", quote="sign of zero")])], unresolved=[])
    decision = Decision(ranking=["C1", "C2"], assessments=[
        Assessment(candidate_id=c["candidate_id"], patch_sha256=c["patch_sha256"],
                   judgments=[Judgment(clause_id="K1", status="unknown", basis="unknown", evidence_ids=[],
                                       explanation="No execution evidence yet")], unsupported_scope=[], unresolved=[])
        for c in candidates], why_top_over_runner_up="Tentative static preference; unknown execution", unresolved=[])
    return case, contract, decision


def compiled(case, _image):
    return {c["candidate_id"]: {"candidate_id": c["candidate_id"],
                                "patch_sha256": c["patch_sha256"],
                                "status": "passed" if c["application"]["status"] != "failed" else "failed"}
            for c in case["candidates"]}


def test_input_whitelist_keeps_hypothesis_proposals_but_hides_verdicts():
    forbidden = {"hypothesis_id": "H3", "variant_id": "latest", "resolved": True,
                 "branch_compliance": "compliant", "repair_contract": {"hard_obligations": ["invented"]},
                 "basic_tests_passed": True, "test_patch": "GOLD", "patch": "actual patch"}
    hypotheses = [{"id": "H3", "title": "Proposed cause", "root_cause": "Unverified mechanism",
                   "repair_scope": ["Affected helper"], "branch_assumption": "The helper is reached"}]
    frozen = freeze_candidates([forbidden, {**forbidden, "variant_id": "initial"}], hypotheses)
    assert len(frozen) == 1
    assert set(frozen[0]) == {"candidate_id", "patch_sha256", "patch", "hypotheses"}
    assert {origin["variant_id"] for origin in frozen[0]["hypotheses"]} == {"initial", "latest"}
    assert all(origin["proposed_cause"] == "Unverified mechanism" for origin in frozen[0]["hypotheses"])
    assert "GOLD" not in json.dumps(frozen) and "compliant" not in json.dumps(frozen)
    assert freeze_candidates([{"patch": "B"}, {"patch": "A"}]) == freeze_candidates([{"patch": "A"}, {"patch": "B"}])


def test_hypothesis_origins_follow_patch_hash_after_sorting_and_deduplication():
    hypotheses = [
        {"id": "H0", "title": "First proposed mechanism", "root_cause": "Cause A", "repair_scope": ["path A"]},
        {"id": "H1", "title": "Second proposed mechanism", "root_cause": "Cause B", "repair_scope": ["path B"]},
    ]
    raw = [
        {"patch": "patch B", "hypothesis_id": "H1", "variant_id": "refined"},
        {"patch": "patch A", "hypothesis_id": "H0", "variant_id": "initial"},
        {"patch": "patch A", "hypothesis_id": "H1", "variant_id": "initial"},
    ]
    frozen = freeze_candidates(raw, hypotheses)
    assert frozen == freeze_candidates(list(reversed(raw)), hypotheses)
    by_patch = {candidate["patch"]: candidate["hypotheses"] for candidate in frozen}
    assert [(h["hypothesis_id"], h["variant_id"]) for h in by_patch["patch A"]] == [
        ("H0", "initial"), ("H1", "initial")]
    assert by_patch["patch B"][0]["proposed_cause"] == "Cause B"


def test_export_unlabeled_uses_exact_selected_patch(tmp_path):
    benchmark = script("selector_benchmark")
    patch = "diff --git a/a.cc b/a.cc\n"
    case_dir = tmp_path / "results" / "org__repo-1"
    case_dir.mkdir(parents=True)
    (case_dir / "selected.patch").write_text(patch)
    score = tmp_path / "score.json"
    score.write_text(json.dumps({"cases": [
        {"instance_id": "org__repo-1", "outcome_basis": "unknown_label",
         "selected_patch_sha256": patch_hash(patch)},
        {"instance_id": "org__repo-2", "outcome_basis": "official_label",
         "selected_patch_sha256": "not exported"},
    ]}))
    output = tmp_path / "submission.json"

    benchmark.export_unlabeled(SimpleNamespace(
        score=score, results=tmp_path / "results", output=output, case=[],
    ))

    assert json.loads(output.read_text()) == [
        {"instance_id": "org__repo-1", "model_patch": patch}
    ]


def test_patch_identity_does_not_strip_blank_context_or_mutate_checkout(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True)
    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    (repo / "a.cc").write_text('#include <lib/fwd.hpp>\none\ntwo\n\n')
    (repo / "include/lib").mkdir(parents=True)
    (repo / "include/lib/fwd.hpp").write_text("template<class Elem> class box;\n")
    (repo / "CMakeLists.txt").write_text('option(MULTIPLE_HEADERS "Public build route" ON)\n')
    git("add", "a.cc", "CMakeLists.txt", "include/lib/fwd.hpp")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD").strip()
    (repo / "a.cc").write_text("one\nchanged\n\n")
    patch = git("diff")
    assert patch.endswith(" \n")
    frozen = freeze_candidates([{"patch": patch}])
    assert frozen[0]["patch"] == patch
    assert check_apply(repo, base, frozen[0]["patch"])["status"] == "passed"
    assert check_apply(repo, base, patch.strip() + "\n")["status"] == "failed"
    assert git("diff") == patch
    sources = base_sources(repo, base, "Issue", frozen)
    assert "changed" not in sources[1]["text"]
    assert "two" in sources[1]["text"]
    assert any(s["source_id"] == "BUILD1" and 'MULTIPLE_HEADERS "Public build route" ON' in s["text"] for s in sources)
    assert any(s["source_id"].startswith("DECL") and "template<class Elem> class box" in s["text"] for s in sources)


def test_base_sources_keep_changed_code_without_unrelated_preamble_or_build_file(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True)
    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    (repo / "src.cc").write_text("// unrelated license line\n" * 250 + "int target() { return 0; }\n")
    (repo / "CMakeLists.txt").write_text("project(public_route)\n")
    (repo / "Makefile").write_text("unused route:\n\techo not selected\n")
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD").strip()
    (repo / "src.cc").write_text("// unrelated license line\n" * 250 + "int target() { return 1; }\n")
    sources = base_sources(repo, base, "Issue", freeze_candidates([{"patch": git("diff")}]))
    code = next(source["text"] for source in sources if source["source_id"] == "BASE1")
    assert "251: int target() { return 0; }" in code
    assert not any(line.startswith("1: ") for line in code.splitlines())
    assert [source["path"].split(":")[-1] for source in sources if source["source_id"].startswith("BUILD")] == ["CMakeLists.txt"]


def test_selector_search_reads_base_commit_not_dirty_worktree(sample, tmp_path, monkeypatch):
    runner = script("run_portfolio_selector")
    case, _, decision = sample
    for assessment in decision.assessments:
        assessment.judgments[0].clause_id = "T1"
    repo = tmp_path / "repo"
    repo.mkdir()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    (repo / "src.cc").write_text("int target_call(int x) { return x; }\n")
    git("add", "src.cc")
    git("commit", "-qm", "base")
    case["base_commit"] = git("rev-parse", "HEAD")
    case["repo_path"] = str(repo)
    (repo / "src.cc").write_text("int FUTURE_ONLY(int x) { return x; }\n")
    stages = []
    class Model:
        def __init__(self, *args):
            pass
        def ask(self, stage, _instruction, payload, schema, validate):
            stages.append(stage)
            if stage == "intent_search":
                result = schema(queries=["target_call"], reason="Find the base public call path")
            else:
                assert stage == "rank"
                assert any("target_call" in source["text"] for source in payload["sources"])
                assert "FUTURE_ONLY" not in json.dumps(payload)
                result = decision
            validate(result)
            return result
    monkeypatch.setattr(runner, "Model", Model)
    result = runner.run_case(case, tmp_path / "selection", compile_runner=compiled)
    assert result["selected_candidate_id"] == "C1"
    assert stages == ["intent_search", "rank"]
    frozen = json.loads((tmp_path / "selection" / "input.json").read_text())
    assert not any(s["source_id"].startswith("SEARCH") for s in frozen["sources"])
    assert "target_call" in (tmp_path / "selection" / "source_search.json").read_text()


def test_contract_requires_real_quote_and_closure_counterfactual(sample):
    case, contract, _ = sample
    check_contract(contract, case["sources"])
    contract.clauses[0].citations[0].quote = "invented contract"
    with pytest.raises(ValueError, match="quote"):
        check_contract(contract, case["sources"])


def test_unclassified_old_execution_is_archived_but_not_judge_input(sample):
    case, contract, _ = sample
    legacy = {"phase": "legacy_unclassified", "output": "MISLEADING_OLD_TEST_PASS"}
    fresh = {"phase": "run", "output": "fresh observed value"}
    payload = rank_payload(case, contract, [legacy, fresh])
    assert payload["evidence"] == [fresh]
    assert "intent_review" not in payload
    assert payload["candidates"] == case["candidates"]


def test_model_input_uses_build_gate_without_truncated_logs(sample):
    case, contract, _ = sample
    case["candidates"][0]["application"]["output"] = "TRUNCATED_APPLY_LOG"
    builds = compiled(case, "image")
    for row in builds.values():
        row.update(phase="build", build_route="cmake", command="cmake --build public",
                   exit_code=0, output="TRUNCATED_BUILD_LOG", image="private-image")
    payload = rank_payload(case, contract, [], compilation=builds, final=True,
                           execution_disabled=True)
    assert payload["compilation"]["C1"] == {
        "patch_sha256": case["candidates"][0]["patch_sha256"], "status": "passed",
        "phase": "build", "build_route": "cmake", "command": "cmake --build public",
        "exit_code": 0,
    }
    assert payload["candidates"][0]["application"] == {"status": "passed"}
    assert "TRUNCATED" not in json.dumps(payload)
    assert "private-image" not in json.dumps(payload)
    assert "probe_build_policy" not in payload and "previous_decision" not in payload
    assert builds["C1"]["output"] == "TRUNCATED_BUILD_LOG"
    assert case["candidates"][0]["application"]["output"] == "TRUNCATED_APPLY_LOG"


def test_model_input_keeps_scout_scope_without_unverified_authority_labels(sample):
    case, contract, _ = sample
    obligation = {"id": "FIX-1", "kind": "fix", "statement": "Restore the public behavior",
                  "authority": "inferred", "strength": "hard", "evidence_basis": ["RUN-9"]}
    case["repair_intent"] = {"common_obligations": [obligation], "open_questions": []}
    payload = rank_payload(case, contract, [])
    assert payload["repair_intent"]["common_obligations"] == [
        {"id": "FIX-1", "kind": "fix", "statement": "Restore the public behavior"}]
    assert case["repair_intent"]["common_obligations"] == [obligation]


def test_ungrounded_extra_clause_stays_unresolved_without_invented_authority(sample):
    case, contract, _ = sample
    contract.clauses.append(Clause(clause_id="BAD", kind="target", requirement="An invented extra requirement",
                                   citations=[Citation(source_id="ISSUE", quote="not in this issue")]))
    ground_contract(contract, case["sources"])
    assert [c.clause_id for c in contract.clauses] == ["K1"]
    assert "UNVERIFIED BAD" in contract.unresolved[0]
    contract.clauses[0].citations[0].quote = "not in this issue"
    with pytest.raises(ValueError, match="No independently grounded"):
        ground_contract(contract, case["sources"])


def test_numbered_source_quote_normalization_keeps_code_exact(sample):
    case, contract, _ = sample
    case["sources"].append({"source_id": "BASE1", "path": "base:a.cc", "partial_excerpt": True,
                            "text": "10: if (x) {\n11:     return true;\n12: }"})
    clause = Clause(clause_id="K2", kind="preservation", requirement="Keep documented existing behavior",
                    citations=[Citation(source_id="BASE1", quote="if (x) {\n    return true;", path="base:a.cc")])
    contract.clauses.append(clause)
    check_contract(contract, case["sources"])
    clause.citations[0].quote = "if (x) {\r\n    return true;"
    check_contract(contract, case["sources"])
    clause.citations[0].quote = "if (x) {\n    return false;"
    with pytest.raises(ValueError, match="quote"):
        check_contract(contract, case["sources"])


@pytest.mark.parametrize("fault", ["stale_assessment", "missing_candidate", "missing_clause", "evidence_free_violation", "unknown_evidence", "other_patch"])
def test_rejects_unbound_or_incomplete_decisions(sample, fault):
    case, contract, decision = sample
    check_decision(decision, case, contract, [], allow_probes=False)
    a = decision.assessments[0]
    j = a.judgments[0]
    if fault == "stale_assessment":
        a.patch_sha256 = "old"
    elif fault == "missing_candidate":
        decision.ranking.pop()
    elif fault == "missing_clause":
        a.judgments.clear()
    elif fault == "evidence_free_violation":
        j.status = "violated"
    elif fault == "unknown_evidence":
        j.evidence_ids = ["made_up"]
    else:
        j.evidence_ids = ["PATCH:C999"]
    with pytest.raises(ValueError):
        check_decision(decision, case, contract, [], allow_probes=False)


def test_legacy_failure_is_not_execution_refutation(sample):
    case, contract, decision = sample
    j = decision.assessments[0].judgments[0]
    j.status, j.basis, j.evidence_ids = "violated", "execution", ["OLD"]
    evidence = [{"evidence_id": "OLD", "candidate_id": "C1", "patch_sha256": case["candidates"][0]["patch_sha256"],
                 "status": "failed", "adjudicative": False, "clause_id": "K1", "base_commit": "base"}]
    with pytest.raises(ValueError, match="No valid"):
        check_decision(decision, case, contract, evidence, allow_probes=False)
    evidence[0]["adjudicative"] = True
    check_decision(decision, case, contract, evidence, allow_probes=False)
    evidence[0]["patch_sha256"] = "stale"
    with pytest.raises(ValueError, match="stale"):
        check_decision(decision, case, contract, evidence, allow_probes=False)


def test_peer_comparison_is_allowed_but_cannot_replace_own_patch_proof(sample):
    case, contract, decision = sample
    j = decision.assessments[0].judgments[0]
    j.status, j.basis = "satisfied", "static"
    j.evidence_ids = ["PATCH:C1", "PATCH:C2"]
    check_decision(decision, case, contract, [], allow_probes=False)
    j.evidence_ids = ["PATCH:C2"]
    with pytest.raises(ValueError, match="actual candidate patch"):
        check_decision(decision, case, contract, [], allow_probes=False)


def test_nonempty_unknown_pool_selects_and_proven_defect_moves_back(sample):
    case, _, decision = sample
    assert resolve(decision, case)["selected_candidate_id"] == "C1"
    decision.assessments[0].judgments[0].status = "violated"
    assert resolve(decision, case)["selected_candidate_id"] == "C2"
    decision.assessments[1].judgments[0].status = "violated"
    assert resolve(decision, case)["all_submittable_refuted"] is True
    case["candidates"][0]["application"]["status"] = "failed"
    assert resolve(decision, case)["selected_candidate_id"] is None


def test_intent_coverage_outranks_shorter_underfix_and_requires_compile(sample):
    case, _, decision = sample
    decision.assessments[0].judgments[0].status = "violated"
    builds = compiled(case, "image")
    assert resolve(decision, case, compilation=builds)["selected_candidate_id"] == "C2"
    builds["C2"]["status"] = "failed"
    result = resolve(decision, case, compilation=builds)
    assert result["selected_candidate_id"] is None
    assert result["all_submittable_refuted"] is True


def test_rank_accepts_bound_patch_and_common_obligation_evidence(sample):
    case, contract, decision = sample
    case["repair_intent"] = {"common_obligations": [{"id": "FIX-1"}]}
    decision.assessments[0].judgments[0].evidence_ids = ["PATCH:C1", "FIX-1"]
    check_decision(decision, case, contract, [], allow_probes=False)
    decision.assessments[0].judgments[0].evidence_ids.append("FIX-99")
    with pytest.raises(ValueError, match="Unknown evidence"):
        check_decision(decision, case, contract, [], allow_probes=False)


@pytest.mark.parametrize("mode,expected", [
    ("passed", ("passed", "build")),
    ("build_failed", ("failed", "build")),
    ("timeout", ("unknown", "build")),
    ("no_target", ("unknown", "build")),
    ("configure_failed", ("unknown", "configure")),
    ("apply_failed", ("failed", "apply")),
    ("no_route", ("unknown", "route")),
    ("public_option", ("passed", "build")),
])
def test_public_build_gate_requires_actual_successful_compilation(mode, expected):
    commands = []
    class Env:
        def __init__(self, **kwargs):
            self.work_dir = kwargs["work_dir"]
            self.container = self
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            pass
        def put_archive(self, _path, _data):
            return True
        def run_bash(self, command, timeout):
            commands.append(command)
            if command.startswith("git apply"):
                return ("invalid patch", 1) if mode == "apply_failed" else ("", 0)
            if command.startswith("if test -f CMakeLists.txt"):
                return ("unavailable", 0) if mode == "no_route" else ("cmake", 0)
            if command.startswith("if test -f /home/build_run.sh"):
                return ("#!/bin/sh\ncd build\ncmake -DSIMDJSON_DEVELOPER_MODE=ON ..\n"
                        "cmake --build .\nctest\n", 0) if mode == "public_option" else ("", 0)
            if command.startswith("cmake -S"):
                return ("missing dependency", 1) if mode == "configure_failed" else ("", 0)
            if command.startswith("cmake --build"):
                return ("source: error", 1) if mode == "build_failed" else (
                    "timed out", 124) if mode == "timeout" else ("built", 0)
            if command.startswith("grep -q"):
                return ("", 1) if mode == "no_target" else ("", 0)
            raise AssertionError(command)
    row = compile_candidate({"candidate_id": "C1", "patch_sha256": "sha", "patch": "patch"},
                            image="image", work_dir="/home/repo", base_commit="base", env_factory=Env)
    assert (row["status"], row["phase"]) == expected
    assert row["patch_sha256"] == "sha" and row["base_commit"] == "base"
    assert not any("ctest" in command for command in commands)
    if mode == "public_option":
        assert any("-DSIMDJSON_DEVELOPER_MODE=ON" in command for command in commands)


def test_no_compiling_candidate_never_reaches_model(sample, tmp_path, monkeypatch):
    runner = script("run_portfolio_selector")
    case, _, _ = sample
    monkeypatch.setattr(runner, "Model", lambda *_args: pytest.fail("Model must not be called"))
    def no_builds(case, _image):
        rows = compiled(case, _image)
        for row in rows.values():
            row["status"] = "unknown"
        return rows
    result = runner.run_case(case, tmp_path / "selection", compile_runner=no_builds)
    assert result["status"] == "no_compiling_patch"
    assert result["selected_patch_sha256"] is None
    assert not (tmp_path / "selection" / "selected.patch").exists()


def make_probe(**kwargs):
    return Probe(probe_id="P1", purpose="Check sign", deciding_between=["C1", "C2"], clause_id="K1",
                 oracle_reason="Issue explicitly requires signed zero", source="int main() { return 0; }",
                 compile_args=["-std=c++11", "-Iinclude"], expected_exit_code=0,
                 required_output_regex=None, **kwargs)


@pytest.mark.parametrize("arg", ["/tmp/stale.cc", "@flags", "-o", "-c", "../future.cc", "-fplugin=evil.so"])
def test_probe_rejects_hidden_artifacts_and_nonexecuting_build(arg):
    p = make_probe()
    p.compile_args.append(arg)
    with pytest.raises(ValueError):
        probe_commands(p)


def test_probe_budget_and_unanchored_oracle(sample):
    case, contract, decision = sample
    decision.probes = [make_probe()]
    with pytest.raises(ValueError, match="budget"):
        check_decision(decision, case, contract, [], allow_probes=False)
    decision.probes[0].clause_id = None
    with pytest.raises(ValueError, match="Unanchored"):
        check_decision(decision, case, contract, [], allow_probes=True)


def test_clean_matrix_compile_failures_stay_unknown(sample):
    case, _, _ = sample
    created = []
    class Env:
        def __init__(self, **kwargs):
            self.work_dir = kwargs["work_dir"]
            self.container = self
            self.ordinal = len(created)
            created.append(self)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def put_archive(self, path, data):
            return True
        def run_bash(self, command, timeout):
            if command.startswith("g++"):
                return ("invalid fixture API", 1) if self.ordinal == 1 else ("", 0)
            if command.startswith("/tmp/"):
                return "observed", 0
            return "", 0
    rows = execution_rows(case, make_probe(), image="test", env_factory=Env)
    assert len(created) == 3
    c1 = [r for r in rows if r["candidate_id"] == "C1"]
    assert c1[-1]["phase"] == "compile" and c1[-1]["status"] == "unknown"
    assert not any(r["adjudicative"] for r in c1)
    assert {r["candidate_id"] for r in rows if r["adjudicative"]} == {"BASELINE", "C2"}


@pytest.mark.parametrize("truncated_phase", ["run", "integrity"])
def test_truncated_probe_output_or_snapshot_cannot_establish_pass(sample, truncated_phase):
    case, _, _ = sample

    class Env:
        def __init__(self, **kwargs):
            self.work_dir = kwargs["work_dir"]
            self.container = self
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def put_archive(self, path, data):
            return True
        def run_bash(self, command, timeout):
            if (truncated_phase == "run" and command.startswith("/tmp/")) or (
                truncated_phase == "integrity" and command.startswith("git add")
            ):
                return "retained output\n[command output capture truncated]", 0
            return "", 0

    rows = execution_rows(case, make_probe(), image="test", env_factory=Env)
    assertions = [row for row in rows if row["phase"] == "assert"]
    assert len(assertions) == 3
    assert all(row["status"] == "unknown" and not row["adjudicative"] for row in assertions)


def test_official_scoring_keeps_unknown_and_reports_regressions(sample):
    benchmark = script("selector_benchmark")
    case, _, _ = sample
    c1, c2 = case["candidates"]
    ledger = {"baseline": {case["instance_id"]: {"resolved": True}}, "labels": {
        case["instance_id"] + ":" + c1["patch_sha256"]: {"resolved": True}}}
    selected = {"instance_id": case["instance_id"], "selected_patch_sha256": c2["patch_sha256"]}
    report = benchmark.score_results([case], ledger, [selected])
    assert report["unknown_selected_labels"] == 1 and report["regressed"] == []
    assert report["net_change_bounds"] == [-1, 0]
    ledger["labels"][case["instance_id"] + ":" + c2["patch_sha256"]] = {"resolved": False}
    assert benchmark.score_results([case], ledger, [selected])["regressed"] == [case["instance_id"]]
    selected["selected_patch_sha256"] = "unseen_patch"
    with pytest.raises(ValueError, match="absent"):
        benchmark.score_results([case], ledger, [selected])


def test_conflicting_labels_are_rejected(tmp_path):
    benchmark = script("selector_benchmark")
    (tmp_path / "model.patch").write_text("exact submitted patch")
    (tmp_path / "fix.patch").write_text("exact submitted patch")
    row = {"instance_id": "org__repo-1", "log_path": str(tmp_path / "eval.log"), "resolved": True}
    one, two = tmp_path / "one.json", tmp_path / "two.json"
    one.write_text(json.dumps({"results": [row]}))
    two.write_text(json.dumps({"results": [{**row, "resolved": False}]}))
    with pytest.raises(ValueError, match="Conflicting"):
        benchmark.label_ledger([one, two])


def test_terminal_pipeline_failure_gets_zero_credit_without_a_fake_patch_label(sample):
    benchmark = script("selector_benchmark")
    case, _, _ = sample
    ledger = {"baseline": {case["instance_id"]: {"resolved": True}}, "labels": {}}
    report = benchmark.score_results([case], ledger, [
        {"instance_id": case["instance_id"], "selected_patch_sha256": None, "status": "pipeline_error"}])
    assert report["scored_cases"] == 1 and report["unknown_selected_labels"] == 0
    assert report["regressed"] == [case["instance_id"]]
    assert report["cases"][0]["outcome_basis"] == "no_submission"


def test_static_evidence_is_bound_to_declared_owner_without_duplicate_fields(sample):
    case, contract, decision = sample
    j = decision.assessments[0].judgments[0]
    j.status, j.basis, j.evidence_ids = "violated", "static", ["ISSUE"]
    j.explanation = "For input -0.0 this candidate returns +0.0, contrary to the Issue requirement"
    ground_decision(decision, case, contract, [])
    assert j.evidence_ids == ["ISSUE", "PATCH:C1"]
    check_decision(decision, case, contract, [], allow_probes=False)
    decision.assessments[0].patch_sha256 = "stale"
    with pytest.raises(ValueError, match="stale"):
        ground_decision(decision, case, contract, [])


def test_omitted_hash_binds_to_frozen_candidate_but_conflicting_hash_is_rejected(sample):
    case, contract, decision = sample
    a = decision.assessments[0]
    a.patch_sha256 = ""
    ground_decision(decision, case, contract, [])
    assert a.patch_sha256 == case["candidates"][0]["patch_sha256"]
    check_decision(decision, case, contract, [], allow_probes=False)
    a.patch_sha256 = case["candidates"][1]["patch_sha256"]
    with pytest.raises(ValueError, match="stale"):
        ground_decision(decision, case, contract, [])


@pytest.mark.parametrize("default,root", [("ON", "headers"), ("OFF", "amalgamated"), ("${DYNAMIC}", None)])
def test_public_cmake_default_controls_header_route_without_case_or_label_rules(default, root):
    cmake = '''option(MANY_HEADERS "Use split headers" DEFAULT)
if(UNRELATED)
message(STATUS "x")
endif()
if(MANY_HEADERS)
set(PUBLIC_HEADERS "${PROJECT_SOURCE_DIR}/headers/")
else()
set(PUBLIC_HEADERS "${PROJECT_SOURCE_DIR}/amalgamated/")
endif()
target_include_directories(lib INTERFACE $<BUILD_INTERFACE:${PUBLIC_HEADERS}>)
'''.replace("DEFAULT", default)
    route = public_header_route([{"source_id": "BUILD1", "path": "base:CMakeLists.txt", "text": cmake}])
    if root is None:
        assert route is None
        return
    assert route["include_root"] == root
    probe = make_probe()
    probe.compile_args = ["-std=c++11", "-Iheaders/", "-I", "amalgamated", "-Iother", "-DREQUIRED=1"]
    ground_probe_route(probe, route)
    assert probe.compile_args == ["-I" + root, "-std=c++11", "-Iother", "-DREQUIRED=1"]
    assert public_header_route([{"source_id": "BUILD1", "path": "base:CMakeLists.txt",
                                "text": cmake.replace("$<BUILD_INTERFACE:${PUBLIC_HEADERS}>", "") }]) is None


def test_build_refinement_removes_only_known_harness_argv():
    probe = make_probe()
    probe.compile_args = refinement_args(["-std=c++11", "src/lib.cc", PROBE_PATH, "-o", BINARY])
    build, _ = probe_commands(probe)
    assert build == ["g++", "-std=c++11", "src/lib.cc", PROBE_PATH, "-o", BINARY]
    probe.compile_args = refinement_args(["-o", "/tmp/unrelated-output"])
    with pytest.raises(ValueError):
        probe_commands(probe)


def test_issue_contract_does_not_invent_preservation_or_mirror_requirements(sample):
    case, _, _ = sample
    contract = issue_contract(case["sources"])
    assert len(contract.clauses) == 1 and contract.clauses[0].clause_id == "T1"
    check_contract(contract, case["sources"])


def test_partial_probe_output_can_be_interpreted_without_claiming_assertion_pass(sample):
    case, contract, decision = sample
    common = {"candidate_id": "C1", "patch_sha256": case["candidates"][0]["patch_sha256"],
              "base_commit": "base", "probe_sha256": "probe", "build_route": ["g++"]}
    rows = [{**common, "evidence_id": "RUN", "phase": "run", "status": "observed",
             "output": "-0.0\ninvalid auxiliary fixture\n", "exit_code": 0, "adjudicative": False},
            {**common, "evidence_id": "INTACT", "phase": "integrity", "status": "passed"},
            {**common, "evidence_id": "ASSERT", "phase": "assert", "status": "failed", "clause_id": "K1", "adjudicative": True}]
    j = decision.assessments[0].judgments[0]
    j.status, j.basis, j.evidence_ids = "satisfied", "execution", ["RUN", "ASSERT", "ISSUE"]
    j.explanation = "Observed -0.0 satisfies the Issue; the auxiliary fixture has no valid oracle"
    ground_decision(decision, case, contract, rows)
    assert j.basis == "observation" and j.observed_output == "-0.0"
    check_decision(decision, case, contract, rows, allow_probes=False)
    j.observed_output = "invented output"
    with pytest.raises(ValueError, match="exact own output"):
        check_decision(decision, case, contract, rows, allow_probes=False)
    j.observed_output = "-0.0"
    rows[1]["probe_sha256"] = "different fixture"
    with pytest.raises(ValueError, match="source-integrity"):
        check_decision(decision, case, contract, rows, allow_probes=False)


def test_every_challenger_is_compared_and_low_ranked_candidate_can_be_probed(sample):
    case, contract, decision = sample
    third = {**case["candidates"][1], "candidate_id": "C3", "patch_sha256": "third"}
    case["candidates"].append(third)
    decision.ranking.append("C3")
    decision.assessments.append(decision.assessments[1].model_copy(update={"candidate_id": "C3", "patch_sha256": "third"}))
    decision.comparisons = [Comparison(candidate_id=c, peer_id="C1", difference="Different sign handling for zero",
                                      consequence="Could change signed zero output", evidence_ids=["PATCH:C1", f"PATCH:{c}"],
                                      verification_needed=True) for c in ("C2", "C3")]
    probe = make_probe()
    probe.deciding_between = ["C1", "C3"]
    decision.probes = [probe]
    check_decision(decision, case, contract, [], allow_probes=True, require_comparisons=True)
    decision.comparisons.pop()
    with pytest.raises(ValueError, match="every non-top"):
        check_decision(decision, case, contract, [], allow_probes=True, require_comparisons=True)
    probe.deciding_between = ["C1", "C1"]
    with pytest.raises(ValueError, match="two distinct"):
        check_decision(decision, case, contract, [], allow_probes=True)


def test_comparison_orientation_and_implicit_patch_references_are_normalized(sample):
    case, contract, decision = sample
    decision.comparisons = [Comparison(candidate_id="C1", peer_id="C2", difference="C1 keeps the sign; C2 drops it",
        consequence="The output could differ for signed zero", evidence_ids=["C1", "C2"], verification_needed=True)]
    ranking = list(decision.ranking)
    ground_decision(decision, case, contract, [])
    assert decision.ranking == ranking
    assert decision.comparisons[0].candidate_id == "C2"
    assert decision.comparisons[0].peer_id == "C1"
    assert set(decision.comparisons[0].evidence_ids) == {"PATCH:C1", "PATCH:C2"}
    check_decision(decision, case, contract, [], allow_probes=False, require_comparisons=True)


def test_unassessed_clauses_and_comparisons_remain_explicitly_unknown(sample):
    case, contract, decision = sample
    j = decision.assessments[0].judgments[0]
    j.clause_id = "unbound_extra"
    j.status = "satisfied"
    ground_decision(decision, case, contract, [])
    assert decision.assessments[0].judgments[0].status == "unknown"
    assert "unbound_extra" in decision.assessments[0].unresolved[0]
    assert decision.comparisons[0].verification_needed
    assert "not provided" in decision.unresolved[-1]
    check_decision(decision, case, contract, [], allow_probes=False, require_comparisons=True)


def test_repeated_target_claims_stay_unknown_without_changing_ranking(sample):
    case, contract, decision = sample
    first = decision.assessments[0].judgments[0]
    first.status, first.basis, first.evidence_ids = "satisfied", "static", ["ISSUE"]
    decision.assessments[0].judgments.append(first.model_copy(update={"status": "violated"}))
    ground_decision(decision, case, contract, [])
    assert decision.ranking == ["C1", "C2"]
    assert len(decision.assessments[0].judgments) == 1
    assert decision.assessments[0].judgments[0].status == "unknown"
    assert len(decision.assessments[0].unresolved) == 2
    check_decision(decision, case, contract, [], allow_probes=False)


@pytest.mark.parametrize("basis", ["execution", "observation"])
def test_compile_only_claims_are_not_verdicts_or_submission_failures(sample, basis):
    case, contract, decision = sample
    j = decision.assessments[0].judgments[0]
    j.status, j.basis, j.evidence_ids = "violated", basis, ["COMPILE"]
    rows = [{"evidence_id": "COMPILE", "phase": "compile", "candidate_id": "C1",
             "patch_sha256": case["candidates"][0]["patch_sha256"], "status": "unknown",
             "adjudicative": False, "output": "Missing fixture dependency"}]
    ground_decision(decision, case, contract, rows)
    assert j.status == j.basis == "unknown"
    assert decision.assessments[0].unresolved
    check_decision(decision, case, contract, rows, allow_probes=False)
    assert resolve(decision, case)["selected_candidate_id"] == "C1"
    rows[0]["patch_sha256"] = "stale"
    with pytest.raises(ValueError, match="stale"):
        check_decision(decision, case, contract, rows, allow_probes=False)


def successful_rows(case, probe):
    rows = []
    for c in case["candidates"]:
        common = {"candidate_id": c["candidate_id"], "patch_sha256": c["patch_sha256"],
                  "base_commit": case["base_commit"], "probe_id": probe.probe_id,
                  "probe_sha256": probe.probe_id + "fixture", "build_route": ["g++"], "adjudicative": False}
        rows.append({**common, "evidence_id": probe.probe_id + c["candidate_id"], "phase": "run",
                     "status": "observed", "exit_code": 0, "output": "same"})
        rows.append({**common, "evidence_id": probe.probe_id + c["candidate_id"] + "integrity",
                     "phase": "integrity", "status": "passed"})
    return rows


def test_extra_probe_requires_identical_successful_candidate_runs(sample):
    case, _, _ = sample
    probe = make_probe()
    rows = successful_rows(case, probe)
    assert indistinguishable_groups(case, rows, [probe]) == [["C1", "C2"]]
    assert not indistinguishable_groups(case, rows[:-1], [probe])
    assert not indistinguishable_groups(case, rows, [])
    rows[0]["output"] = "different"
    assert not indistinguishable_groups(case, rows, [probe])
    rows[0]["output"], rows[0]["exit_code"] = "same", 134
    assert not indistinguishable_groups(case, rows, [probe])


def test_extra_probe_targets_remaining_tie_despite_other_failures(sample):
    case, contract, _ = sample
    case["candidates"].append({**case["candidates"][0], "candidate_id": "C3", "patch_sha256": "third"})
    probe = make_probe()
    probe.deciding_between = ["C1", "C2", "C3"]
    runner = script("run_portfolio_selector")
    runner.check_probes([probe], case, contract)
    rows = successful_rows(case, probe)
    rows[-2]["exit_code"] = 134
    assert indistinguishable_groups(case, rows, [probe]) == [["C1", "C2"]]


@pytest.mark.parametrize("unavailable", ["runtime_failure", "compile_failure"])
def test_unavailable_auxiliary_probe_allows_more_evidence_but_not_a_final_tie(sample, unavailable):
    case, _, decision = sample
    probe, auxiliary = make_probe(), make_probe()
    auxiliary.probe_id = "P2"
    rows = successful_rows(case, probe) + successful_rows(case, auxiliary)
    for row in rows:
        if row["probe_id"] == "P2" and row["phase"] == "run":
            row["exit_code"] = 134
            if unavailable == "compile_failure":
                row["phase"], row["status"] = "compile", "unknown"
    assert indistinguishable_groups(case, rows, [probe, auxiliary]) == [["C1", "C2"]]
    assert resolve(decision, case, evidence=rows, probes=[probe, auxiliary])["unverified_top_tie"] == []
    # A successful auxiliary execution for just one side prevents grouping them.
    rows[-2].update(phase="run", status="observed", exit_code=0)
    assert not indistinguishable_groups(case, rows, [probe, auxiliary])


def test_selector_reasoning_budget_is_stage_local():
    model = object.__new__(script("run_portfolio_selector").Model)
    model.kwargs = {"extra_body": {"reasoning_effort": "low", "thinking": {"type": "enabled"}},
                    "max_tokens": 16000, "timeout": 240}
    original = json.loads(json.dumps(model.kwargs))
    for stage in ("probe_plan", "oracle_review", "discriminator_plan",
                  "discriminator_oracle_review", "intent_search", "rank"):
        assert model.stage_kwargs(stage)["extra_body"]["reasoning_effort"] == "high"
        assert model.stage_kwargs(stage)["max_tokens"] == 32768
    for stage in ("P1_build_refine", "P3_fixture_refine"):
        assert model.stage_kwargs(stage) == original
    assert model.kwargs == original


def test_selector_low_reasoning_applies_to_every_stage():
    model = object.__new__(script("run_portfolio_selector").Model)
    model.kwargs = {"extra_body": {"reasoning_effort": "low", "thinking": {"type": "enabled"}},
                    "max_tokens": 16000, "timeout": 240}
    model.reasoning_effort = "low"
    for stage in ("probe_plan", "oracle_review", "discriminator_plan",
                  "discriminator_oracle_review", "intent_search", "rank", "P1_build_refine"):
        assert model.stage_kwargs(stage) == model.kwargs


def test_selector_entry_points_default_to_low_reasoning():
    runner = script("run_portfolio_selector")
    assert inspect.signature(runner.Model).parameters["reasoning_effort"].default == "low"
    assert inspect.signature(runner.run_case).parameters["reasoning_effort"].default == "low"


@pytest.mark.parametrize("interrupt_first", [False, True])
def test_streamed_json_keeps_usage_closes_connections_and_never_saves_reasoning(tmp_path, monkeypatch, interrupt_first):
    runner = script("run_portfolio_selector")
    model = object.__new__(runner.Model)
    model.kwargs, model.directory, model.usage = {}, tmp_path, []
    content = json.dumps({"compiler": "g++", "compile_args": ["-Iinclude"], "reason": "Use the public header route"})
    streams = []
    class Stream:
        closed = False
        def __iter__(self):
            def chunk(text=None, reasoning=None, finish=None):
                return SimpleNamespace(usage=None, choices=[SimpleNamespace(
                    delta=SimpleNamespace(content=text, reasoning_content=reasoning), finish_reason=finish)])
            yield chunk(reasoning="PRIVATE_REASONING_SENTINEL")
            yield chunk(content[:20])
            if interrupt_first and len(streams) == 1:
                raise ConnectionError("fixture interruption")
            yield chunk(content[20:], finish="stop")
            yield SimpleNamespace(choices=[], usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 123}))
        def close(self):
            self.closed = True
    def completion(**kwargs):
        assert kwargs["stream"] and kwargs["stream_options"] == {"include_usage": True}
        streams.append(Stream())
        return streams[-1]
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))
    result = model.ask("P1_build_refine", "Return the build arguments", {}, runner.BuildRefinement, lambda r: None)
    assert result.compile_args == ["-Iinclude"]
    assert len(streams) == 1 + interrupt_first and all(s.closed for s in streams)
    assert model.usage[-1]["usage"] == {"total_tokens": 123}
    assert model.usage[-1]["reasoning_chars_received"] == len("PRIVATE_REASONING_SENTINEL")
    assert model.usage[-1]["finish_reason"] == "stop"
    assert all("PRIVATE_REASONING_SENTINEL" not in p.read_text() for p in tmp_path.iterdir())
    if interrupt_first:
        assert model.usage[0]["error_type"] == "ConnectionError"
        assert not (tmp_path / "P1_build_refine.response1.txt").exists()


def test_smaller_diff_does_not_override_intent_ranking_on_unverified_tie(sample):
    case, _, decision = sample
    for c, a, patch in zip(case["candidates"], decision.assessments, ["+many\n+extra\n+changes", "+one"]):
        c["patch"], c["patch_sha256"] = patch, patch_hash(patch)
        a.patch_sha256 = c["patch_sha256"]
    decision.comparisons = [Comparison(candidate_id="C2", peer_id="C1", difference="The top adds more optional methods",
        consequence="Their necessity remains unverified after the bounded tests", evidence_ids=["PATCH:C1", "PATCH:C2"],
        verification_needed=True)]
    probe = make_probe()
    rows = successful_rows(case, probe)
    result = resolve(decision, case, evidence=rows, probes=[probe])
    assert result["selected_candidate_id"] == "C1" and result["unverified_top_tie"] == []
    assert result["model_ranking"] == ["C1", "C2"] and not result["policy_reordered"]
    assert resolve(decision, case)["selected_candidate_id"] == "C1"
    rows[2]["output"] = "different supported output"
    assert resolve(decision, case, evidence=rows, probes=[probe])["selected_candidate_id"] == "C1"
    rows[2]["output"] = "same"
    decision.comparisons[0].verification_needed = False
    assert resolve(decision, case, evidence=rows, probes=[probe])["selected_candidate_id"] == "C1"
    decision.comparisons[0].verification_needed = True
    decision.assessments[1].judgments[0].status = "violated"
    assert resolve(decision, case, evidence=rows, probes=[probe])["selected_candidate_id"] == "C1"


def test_single_compiling_candidate_still_gets_independent_rank(sample, tmp_path, monkeypatch):
    runner = script("run_portfolio_selector")
    case, _, decision = sample
    case["candidates"][0]["application"]["status"] = "failed"
    for assessment in decision.assessments:
        assessment.judgments[0].clause_id = "T1"
    calls = []
    class Model:
        def __init__(self, *args):
            pass
        def ask(self, stage, _instruction, _payload, schema, validate):
            calls.append(stage)
            assert stage == "rank"
            result = decision
            validate(result)
            return result
    monkeypatch.setattr(runner, "Model", Model)
    out = tmp_path / "selection"
    result = runner.run_case(case, out, execute=True, compile_runner=compiled)
    assert result["selected_candidate_id"] == "C2"
    assert (out / "selected.patch").read_text() == case["candidates"][1]["patch"]
    assert calls == ["rank"]


def test_fixture_retry_has_distinct_evidence_identity(sample):
    case, _, _ = sample
    class Unavailable:
        def __init__(self, **kwargs):
            raise RuntimeError("No test environment")
    first = execution_rows(case, make_probe(), image="test", env_factory=Unavailable)
    second = execution_rows(case, make_probe(), image="test", env_factory=Unavailable, attempt="build_retry")
    assert not {r["evidence_id"] for r in first} & {r["evidence_id"] for r in second}
    assert all(r["adjudicative"] is False for r in first + second)


def test_rank_receives_exact_probe_source_and_oracle_without_treating_them_as_a_pass(sample):
    case, contract, decision = sample
    probe = make_probe()
    class Unavailable:
        def __init__(self, **kwargs):
            raise RuntimeError("No test environment")
    rows = execution_rows(case, probe, image="test", env_factory=Unavailable)
    definition = rank_payload(case, contract, rows)["evidence"][0]
    assert definition["phase"] == "probe_definition" and definition["probe"] == probe.model_dump()
    assert definition["adjudicative"] is False and definition["status"] == "frozen"
    j = decision.assessments[0].judgments[0]
    j.status, j.basis, j.evidence_ids = "satisfied", "execution", [definition["evidence_id"]]
    with pytest.raises(ValueError, match="No valid clause-bound"):
        check_decision(decision, case, contract, rows, allow_probes=False)


@pytest.mark.parametrize("replacement", [
    "int main(){ Container value; for(auto entry : input) value.emplace(entry); return 0; }",
    "int main(){ auto value = product::other_api<Container>(input); return 0; }",
    "int main(){ auto value = product::convert<Container>(input); return 1; }",
])
def test_fixture_refinement_cannot_remove_or_replace_the_tested_operation(replacement):
    original = "int main(){ auto value = product::convert<Container>(input); return 0; }"
    with pytest.raises(ValueError, match="preserve every original code token"):
        check_fixture_refinement(original, replacement)


def test_fixture_refinement_allows_declarations_and_formatting_but_preserves_literals():
    original = 'int main(){ Input input; target(input); puts("OK"); } // old comment'
    refined = '#include <cstdio>\nstruct Input {};\nvoid target(Input);\nint main() { Input input;\n target (input); puts("OK"); }'
    check_fixture_refinement(original, refined)
    with pytest.raises(ValueError):
        check_fixture_refinement(original, refined.replace('"OK"', '"different"'))


def test_shared_fixture_compile_correction_is_bounded_and_preserves_external_oracle(sample, tmp_path, monkeypatch):
    runner = script("run_portfolio_selector")
    case, _, decision = sample
    for a in decision.assessments:
        a.judgments[0].clause_id = "T1"
    probe = make_probe()
    probe.clause_id, probe.source = "T1", "struct Input { explicit Input(int); }; int main(){ Input input; target(input); }"
    fixed_source = probe.source.replace("explicit Input(int);", "Input() = default; explicit Input(int);")
    expected = probe.model_dump()
    calls, executed = [], []
    class Model:
        def __init__(self, *args):
            pass
        def ask(self, stage, instruction, payload, schema, validate):
            calls.append(stage)
            if stage == "probe_plan":
                result = schema(probes=[probe], reason="Faithfully check the public target")
            elif stage == "oracle_review":
                assert "candidates" not in payload
                result = schema(reviews=[{"probe_id": "P1", "keep_expected_exit_code": True,
                                          "keep_required_output_regex": False,
                                          "reason": "Issue requires the target invocation to finish"}])
            elif stage == "P1_build_refine":
                result = schema(compiler=probe.compiler, compile_args=probe.compile_args,
                                reason="Arguments are already correct; API compatibility is the problem")
            elif stage == "P1_fixture_refine":
                assert payload["probe"] == expected
                result = schema(source=fixed_source,
                                reason="Supply the missing default constructor without changing the tested call")
            else:
                assert stage == "rank"
                definitions = [r for r in payload["evidence"] if r["phase"] == "probe_definition"]
                assert [r["probe"]["source"] for r in definitions] == [expected["source"], fixed_source]
                result = decision
            validate(result)
            return result
    def execute(case, value, **kwargs):
        executed.append(value.model_dump())
        for field in ("expected_exit_code", "required_output_regex", "compile_args", "clause_id"):
            assert value.model_dump()[field] == expected[field]
        rows = [{"evidence_id": "definition:" + kwargs.get("attempt", ""), "candidate_id": "BASELINE",
                 "patch_sha256": None, "phase": "probe_definition", "adjudicative": False,
                 "probe": value.model_dump()}]
        if value.source == expected["source"]:
            rows.extend({"phase": "compile", "candidate_id": cid, "evidence_id": "compile:" + cid,
                         "patch_sha256": sha, "status": "unknown", "adjudicative": False,
                         "output": PROBE_PATH + ": error: no default constructor for Input"}
                        for cid, sha in [("BASELINE", None)] + [(c["candidate_id"], c["patch_sha256"]) for c in case["candidates"]])
        else:
            observed = successful_rows(case, value)
            observed[2]["output"] = "different observation"
            rows.extend(observed)
        return rows
    monkeypatch.setattr(runner, "Model", Model)
    monkeypatch.setattr(runner, "execution_rows", execute)
    runner.run_case(case, tmp_path / "selection", execute=True, compile_runner=compiled)
    assert calls == ["probe_plan", "oracle_review", "P1_build_refine", "P1_fixture_refine", "rank"]
    assert len(executed) == 2


def test_probe_execution_precedes_the_only_ranking_call(sample, tmp_path, monkeypatch):
    runner = script("run_portfolio_selector")
    case, _, decision = sample
    for assessment in decision.assessments:
        assessment.judgments[0].clause_id = "T1"
    events = []
    class Model:
        def __init__(self, *args):
            pass
        def ask(self, stage, instruction, payload, schema, validate):
            events.append(stage)
            assert payload.get("previous_decision") is None
            if stage == "probe_plan":
                probe = make_probe()
                probe.clause_id = "T1"
                result = schema(probes=[probe], reason="Check the original public reproduction")
            elif stage == "oracle_review":
                assert "candidates" not in payload
                result = schema(reviews=[{"probe_id": "P1", "keep_expected_exit_code": True,
                                          "keep_required_output_regex": False,
                                          "reason": "Issue requires a successful target call"}])
            else:
                assert stage == "rank" and payload["final_round"]
                assert payload["evidence"][0]["evidence_id"] == "OBSERVED"
                result = decision
            validate(result)
            return result
    def execute(*args, **kwargs):
        events.append("execute")
        return [{"evidence_id": "OBSERVED", "phase": "fixture", "candidate_id": "BASELINE",
                 "patch_sha256": None, "status": "unknown", "adjudicative": False}]
    monkeypatch.setattr(runner, "Model", Model)
    monkeypatch.setattr(runner, "execution_rows", execute)
    result = runner.run_case(case, tmp_path / "selection", execute=True, compile_runner=compiled)
    assert events == ["probe_plan", "oracle_review", "execute", "rank"]
    assert result["selected_candidate_id"] == "C1" and result["probe_count"] == 1


def test_oracle_review_drops_speculative_output_without_dropping_probe(sample, tmp_path, monkeypatch):
    runner = script("run_portfolio_selector")
    case, _, decision = sample
    for assessment in decision.assessments:
        assessment.judgments[0].clause_id = "T1"
    executed = []

    class Model:
        def __init__(self, *args):
            pass

        def ask(self, stage, instruction, payload, schema, validate):
            if stage == "probe_plan":
                probe = make_probe()
                probe.clause_id = "T1"
                probe.required_output_regex = "guessed exact output"
                result = schema(probes=[probe], reason="Run the public example for observation")
            elif stage == "oracle_review":
                assert "candidates" not in payload and "patch" not in json.dumps(payload)
                result = schema(reviews=[{
                    "probe_id": "P1", "keep_expected_exit_code": True,
                    "keep_required_output_regex": None,
                    "reason": "The Issue requires success but supplies no exact output",
                }])
            else:
                assert stage == "rank"
                result = decision
            validate(result)
            return result

    def execute(_case, probe, **_kwargs):
        executed.append(probe.model_dump())
        return []

    monkeypatch.setattr(runner, "Model", Model)
    monkeypatch.setattr(runner, "execution_rows", execute)
    runner.run_case(case, tmp_path / "selection", execute=True, compile_runner=compiled)

    assert executed[0]["expected_exit_code"] == 0
    assert executed[0]["required_output_regex"] is None
    assert executed[0]["clause_id"] == "T1"
    assert json.loads((tmp_path / "selection" / "oracle_review_applied.json").read_text())[0] == executed[0]


def test_identical_observations_get_one_extra_probe_before_ranking(sample, tmp_path, monkeypatch):
    runner = script("run_portfolio_selector")
    case, _, decision = sample
    for assessment in decision.assessments:
        assessment.judgments[0].clause_id = "T1"
    events = []
    class Model:
        def __init__(self, *args):
            pass
        def ask(self, stage, instruction, payload, schema, validate):
            events.append(stage)
            assert payload.get("previous_decision") is None
            if stage in ("probe_plan", "discriminator_plan"):
                probe = make_probe()
                probe.clause_id = "T1"
                probe.probe_id = "P1" if stage == "probe_plan" else "P3"
                if stage == "discriminator_plan":
                    assert payload["earlier_probes"][0]["probe_id"] == "P1"
                result = schema(probes=[probe], reason="Investigate the surviving owner state")
            elif stage in ("oracle_review", "discriminator_oracle_review"):
                assert "candidates" not in payload
                result = schema(reviews=[{"probe_id": payload["probes"][0]["probe_id"],
                                          "keep_expected_exit_code": True,
                                          "keep_required_output_regex": False,
                                          "reason": "Issue requires a successful target call"}])
            else:
                assert stage == "rank" and payload["final_round"]
                result = decision
            validate(result)
            return result
    def execute(case, probe, **kwargs):
        events.append(probe.probe_id)
        return successful_rows(case, probe)
    monkeypatch.setattr(runner, "Model", Model)
    monkeypatch.setattr(runner, "execution_rows", execute)
    result = runner.run_case(case, tmp_path / "selection", execute=True, compile_runner=compiled)
    assert events == ["probe_plan", "oracle_review", "P1", "discriminator_plan",
                      "discriminator_oracle_review", "P3", "rank"]
    assert result["probe_count"] == 2
