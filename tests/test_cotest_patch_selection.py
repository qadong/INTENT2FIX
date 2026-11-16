import importlib.util
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ConfigDict, Field

from INTENT2FIX.reproduce.intent_schema import IssueEvidenceFrame
from INTENT2FIX.intent_dialogue.schema import CandidateRepairBoundary
from INTENT2FIX.patch_intent_audit.trajectory_leads import (
    AuditLead,
    EvidenceQuote,
)
from INTENT2FIX.intent_portfolio import (
    BranchComplianceDecision,
    CrossExecutionResult,
    HypothesisMergeAdjudication,
    IntentHypothesis,
    IntentObligation,
    IntentPortfolio,
    MergeBoundaryAlternative,
    MergeBoundaryDecision,
    PortfolioCandidate,
    PortfolioCoverageAudit,
    RepairProbeEvidence,
    UnresolvedDimension,
)
from INTENT2FIX.memory.memory import Memory
from INTENT2FIX.file_context import FileContext
from INTENT2FIX.patch_review.sanitizer import sanitize_candidate_patch
from INTENT2FIX.repository.repository import InMemRepository


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "run_no_memory_cotest.py"
SPEC = importlib.util.spec_from_file_location("run_no_memory_cotest", SCRIPT_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _stamp_probe_execution_fixture(trajectory, candidate):
    for node in trajectory.get_all_nodes():
        for step in node.action_steps:
            action = step.action
            if action.name != "RunSingleTest":
                continue
            step.observation.properties["execution_snapshot"] = {
                "production_patch_sha256": RUNNER._portfolio_patch_sha256(candidate.patch),
                "command": action.command,
                "expected_exit_code": getattr(action, "expected_exit_code", 0),
                "required_output_regex": getattr(action, "required_output_regex", ""),
                "probe_files": {
                    path: node.file_context.get_file(path).content
                    for path in node.file_context.verification.get("ephemeral_paths", [])
                    if path in action.command and node.file_context.get_file(path) is not None
                },
            }


class RequiredStructuredField(BaseModel):
    value: str


class BoundedStructuredField(BaseModel):
    value: str = Field(..., max_length=10)
    evidence_ids: list[str] = Field(default_factory=list, max_length=2)

    model_config = ConfigDict(extra="forbid")


def test_sanitizer_retains_excluded_test_diff_as_audit_evidence():
    context = FileContext(
        repo=InMemRepository(
            files={
                "src/fix.cpp": "int value = 1;\n",
                "tests/fix.cpp": "int expected = 1;\n",
            }
        )
    )
    context.add_file("src/fix.cpp").apply_changes("int value = 2;\n")
    context.add_file("tests/fix.cpp").apply_changes("int expected = 2;\n")

    candidate = sanitize_candidate_patch(context)

    assert "src/fix.cpp" in candidate.patch
    assert "tests/fix.cpp" not in candidate.patch
    assert "tests/fix.cpp" in candidate.excluded_test_patch
    assert "src/fix.cpp" not in candidate.excluded_test_patch


def test_compliance_regression_evidence_keeps_both_soft_views():
    context = FileContext(
        repo=InMemRepository(),
        change_seq=4,
        verification={
            "regression": {
                "change_seq": 4,
                "passed": True,
                "exit_code": 0,
                "summary": "80 passed",
                "output": "large current output",
            },
            "production_regression": {
                "change_seq": 4,
                "passed": False,
                "exit_code": 8,
                "summary": "old message pin failed",
                "output": "large production-only output",
                "failed_test_names": ["legacy_exact_message"],
                "test_view": "original_tests_before_test_edits",
                "test_view_stale": True,
            },
        },
    )

    evidence = RUNNER._portfolio_compliance_regression_evidence(context)

    assert evidence["working_tree_with_verification_changes"]["passed"] is True
    assert evidence["production_only_with_original_tests"]["passed"] is False
    assert evidence["production_only_with_original_tests"][
        "failed_test_names"
    ] == ["legacy_exact_message"]
    assert evidence["production_only_with_original_tests"][
        "test_view_stale"
    ] is True
    assert evidence["production_only_with_original_tests"]["output"] == "large production-only output"


def test_repair_verification_status_marks_unfinished_patch_unverified(tmp_path):
    RUNNER._save_repair_verification_status(
        str(tmp_path),
        verified_finish=False,
        basic_tests_passed=True,
        patch_review_status="skipped_without_verified_finish",
    )

    payload = json.loads(
        (tmp_path / "repair_verification_status.json").read_text()
    )
    assert payload == {
        "verified_finish": False,
        "basic_tests_passed": True,
        "patch_review_status": "skipped_without_verified_finish",
        "submission_status": "unverified",
    }


def test_portfolio_candidate_status_keeps_basic_failure_as_soft_evidence(
    tmp_path,
):
    candidate = SimpleNamespace(
        patch="diff --git a/src/fix.cc b/src/fix.cc\n",
        kept_paths=["src/fix.cc"],
        excluded_paths={"tests/repro.cc": "test_or_fixture"},
    )

    RUNNER._save_portfolio_candidate_status(
        str(tmp_path),
        finish_succeeded=True,
        basic_tests_passed=False,
        candidate=candidate,
        branch_compliance="compliant",
    )

    payload = json.loads(
        (tmp_path / "portfolio_candidate_status.json").read_text()
    )
    assert payload["admitted"] is True
    assert payload["basic_tests_passed"] is False
    assert payload["regression_passed"] is False
    assert payload["focused_oracle_satisfied"] is False
    assert payload["basic_tests_role"] == "soft_evidence"


def test_portfolio_regression_status_is_independent_of_focused_oracle():
    context = FileContext(
        repo=InMemRepository(),
        change_seq=4,
        verification={
            "regression": {
                "change_seq": 4, "passed": True,
                "verification_view": "production",
                "verification_isolation": RUNNER.PRODUCTION_ISOLATION,
                "production_patch_sha256": RUNNER._portfolio_patch_sha256(""),
            },
            "single_test": {
                "change_seq": 4,
                "probe_change_seq": 0,
                "passed": False,
            },
        },
    )

    assert RUNNER._portfolio_regression_passed(context) is True
    assert RUNNER._basic_tests_passed(context) is False


@pytest.mark.parametrize('late_path_evidence', [False, True, 'uncited_premise', 'crowded_source', 'uncited_observations', 'observation_budget'])
def test_dialogue_memory_retains_bounded_repository_surface_evidence(late_path_evidence, monkeypatch):
    monkeypatch.setenv('COTEST_PORTFOLIO_RERANK', '0')
    report = SimpleNamespace(
        input_path_check=None,
        repository_evidence_ids=["SEARCH-1"],
        preserve_evidence_ids=[],
        repository_facts=["A public seed forwards into the operator family."],
        candidate_boundaries=[],
        open_decisions=["The implementation mechanism is open."],
        contrastive_witness=None,
    )
    result = SimpleNamespace(
        render_for_repair=lambda: "grounded contract",
        intent_contract=RUNNER.RepairIntentContract(),
        investigation=SimpleNamespace(
            report=report,
            repository_evidence=[
                SimpleNamespace(
                    id="SEARCH-1",
                    path="include/widget.hpp",
                    locator="Widget::seed",
                    observation="x" * 7000,
                ),
                SimpleNamespace(
                    id="SEARCH-2",
                    path="src/unrelated.cpp",
                    locator="",
                    observation="unrelated",
                ),
            ],
        ),
        contrastive_decision=None,
        pair_status="",
        stop_reason="",
    )

    expanded_observations = late_path_evidence in {'uncited_observations', 'observation_budget'}
    if expanded_observations:
        monkeypatch.setenv('COTEST_PORTFOLIO_RERANK', '1')
        result.investigation.repository_evidence[1].observation = 'Uncited public ingress forwards into the same parser.'
        if late_path_evidence == 'observation_budget':
            result.investigation.repository_evidence.extend(
                SimpleNamespace(id=f'RUN-{i}', path='probe', locator='run', observation='x' * 7000)
                for i in range(20)
            )
    elif late_path_evidence:
        from INTENT2FIX.intent_dialogue.schema import InvestigationInputPathCheck
        report.input_path_check = InvestigationInputPathCheck(
            actual_input='object', consumer_input='object', condition_evidence_id='SEARCH-20',
            condition_quote='gate', evaluation='gate=false', matches_input='no',
            alternative_path='fallback', verification_run_id='RUN-1', verification_quote='gate=false',
        )
        result.investigation.repository_evidence = [
            SimpleNamespace(id=f'SEARCH-{i}', path='header', locator='gate', observation='gate')
            for i in range(1, 21)
        ] + [SimpleNamespace(id='RUN-1', path='probe', locator='run', observation='gate=false')]
        report.repository_evidence_ids = [f'SEARCH-{i}' for i in range(1, 21)]
        if late_path_evidence == 'uncited_premise':
            report.input_path_check = None
            result.investigation.actions = [{'action': 'RunInvestigationCommand',
                'evidence_id': 'RUN-1', 'evidence_role': 'premise', 'exit_code': 1}]
        if late_path_evidence == 'crowded_source':
            report.input_path_check = report.input_path_check.model_copy(update={
                'condition_evidence_id': 'SEARCH-1',
            })
            result.investigation.repository_evidence = [
                SimpleNamespace(id=f'SEARCH-{i}', path='header', locator='source',
                                observation='upstream predicate' if i == 4 else 'caller')
                for i in range(1, 5)
            ] + [SimpleNamespace(id=f'RUN-{i}', path='probe', locator='run',
                                 observation='diagnostic') for i in range(1, 9)]
            report.repository_evidence_ids = ['SEARCH-1', 'SEARCH-2', 'SEARCH-3', 'SEARCH-4']
            result.investigation.actions = [
                {'action': 'RunInvestigationCommand', 'evidence_id': f'RUN-{i}',
                 'evidence_role': 'premise', 'exit_code': 1} for i in range(1, 9)
            ]

    memory = RUNNER._memory_from_intent_dialogue_result(
        instance_name="owner__repo-1",
        problem_statement="Fix public seed selection.",
        issue_evidence=IssueEvidenceFrame(),
        result=result,
    )

    bundle = memory.repair_intent_bundle
    if expanded_observations:
        evidence = bundle['cited_repository_evidence']
        assert [item['id'] for item in evidence[:2]] == ['SEARCH-1', 'SEARCH-2']
        assert evidence[1]['observation'].startswith('Uncited public ingress')
        assert sum(len(item['observation']) for item in evidence) <= 60000
        assert all(len(item['observation']) <= 6000 for item in evidence)
        assert len({item['id'] for item in evidence}) == len(evidence)
        if late_path_evidence == 'observation_budget':
            assert len(evidence) < len(result.investigation.repository_evidence)
        return
    if late_path_evidence:
        evidence = bundle['cited_repository_evidence']
        assert len(evidence) == len(result.investigation.repository_evidence)
        if late_path_evidence == 'crowded_source':
            assert {item['id'] for item in evidence[:2]} == {'SEARCH-1', 'RUN-1'}
            assert {'SEARCH-1', 'SEARCH-2', 'SEARCH-3', 'SEARCH-4'} <= {item['id'] for item in evidence}
        elif late_path_evidence == 'uncited_premise':
            assert evidence[0]['id'] == 'RUN-1'
            assert evidence[0]['observation'] == 'gate=false'
        else:
            assert [item['id'] for item in evidence[:2]] == ['SEARCH-20', 'RUN-1']
        return
    assert bundle["repository_facts"] == [
        "A public seed forwards into the operator family."
    ]
    assert [item["id"] for item in bundle["cited_repository_evidence"]] == [
        "SEARCH-1"
    ]
    assert len(bundle["cited_repository_evidence"][0]["observation"]) == 6000
    assert bundle["repair_surface_leads"] == [
        {
            "id": "SURFACE-1",
            "kind": "repository_fact",
            "statement": "A public seed forwards into the operator family.",
        }
    ]


def test_dialogue_memory_preserves_every_focused_intent_report():
    from INTENT2FIX.intent_dialogue.schema import (
        FocusedInvestigationReport,
        InvestigationReport,
    )

    first_report = InvestigationReport(
        status="partial",
        answer="The input-selection cause was inspected.",
        repository_facts=[
            "[SEARCH-1] The public input reaches the selection predicate."
        ],
        repository_evidence_ids=["SEARCH-1"],
        open_decisions=["The selected fallback remains an intent branch."],
    )
    second_report = InvestigationReport(
        status="partial",
        answer="The independent state-preservation path was inspected.",
        repository_facts=[
            "[SEARCH-2] The failure path mutates the public cursor state."
        ],
        repository_evidence_ids=["SEARCH-2"],
        open_decisions=["The cursor rollback boundary remains open."],
    )
    result = SimpleNamespace(
        render_for_repair=lambda: "grounded contract",
        intent_contract=RUNNER.RepairIntentContract(),
        investigation=SimpleNamespace(
            report=second_report,
            report_history=[
                FocusedInvestigationReport(
                    focus_intent_ids=["INTENT-1"], report=first_report,
                ),
                FocusedInvestigationReport(
                    focus_intent_ids=["INTENT-2"], report=second_report,
                ),
            ],
            repository_evidence=[
                SimpleNamespace(
                    id="SEARCH-1", path="src/select.cpp", locator="predicate",
                    observation="The public input reaches the selection predicate.",
                ),
                SimpleNamespace(
                    id="SEARCH-2", path="src/cursor.cpp", locator="failure",
                    observation="The failure path mutates the public cursor state.",
                ),
            ],
            actions=[],
        ),
        contrastive_decision=None,
        pair_status="",
        stop_reason="",
    )

    memory = RUNNER._memory_from_intent_dialogue_result(
        instance_name="org__repo-1",
        problem_statement="Fix selection and preserve state.",
        issue_evidence=IssueEvidenceFrame(),
        result=result,
    )

    bundle = memory.repair_intent_bundle
    assert bundle["repository_facts"] == [
        "[SEARCH-1] The public input reaches the selection predicate.",
        "[SEARCH-2] The failure path mutates the public cursor state.",
    ]
    assert [item["focus_intent_ids"] for item in bundle["intent_reports"]] == [
        ["INTENT-1"], ["INTENT-2"],
    ]
    assert bundle["intent_evidence_index"] == [
        {
            "focus_intent_ids": ["INTENT-1"],
            "repository_evidence_ids": ["SEARCH-1"],
        },
        {
            "focus_intent_ids": ["INTENT-2"],
            "repository_evidence_ids": ["SEARCH-2"],
        },
    ]
    assert [
        item["id"] for item in bundle["cited_repository_evidence"]
    ] == ["SEARCH-1", "SEARCH-2"]
    assert bundle["open_decisions"] == [
        "The selected fallback remains an intent branch.",
        "The cursor rollback boundary remains open.",
    ]


@pytest.mark.parametrize("only_in_contract", [False, True])
def test_suspected_sibling_boundary_reaches_hypothesis_as_non_normative_lead(
    monkeypatch, only_in_contract,
):
    monkeypatch.setenv("COTEST_PORTFOLIO_RERANK", "1")
    boundary = CandidateRepairBoundary(
        id="BOUNDARY-1",
        anchor_operation="Widget::encode",
        candidate_public_forms=["Widget::encode", "Widget::decode"],
        evidence_ids=["SEARCH-1"],
        rationale=(
            "Both use the custom representation; decode's participation "
            "remains unverified until its full body is inspected."
        ),
    )
    report = SimpleNamespace(
        input_path_check=None,
        investigation_paths=[],
        repository_evidence_ids=["SEARCH-1"],
        preserve_evidence_ids=[],
        repository_facts=[],
        candidate_boundaries=[] if only_in_contract else [boundary],
        open_decisions=[],
        contrastive_witness=None,
    )
    result = SimpleNamespace(
        render_for_repair=lambda: "source-backed investigation",
        intent_contract=RUNNER.RepairIntentContract(
            candidate_boundaries=[boundary] if only_in_contract else []
        ),
        investigation=SimpleNamespace(
            report=report,
            actions=[],
            repository_evidence=[SimpleNamespace(
                id="SEARCH-1", path="include/widget.hpp",
                locator="Widget::decode", observation="decode consumes custom_text",
            )],
        ),
        contrastive_decision=None,
        pair_status="",
        stop_reason="",
    )

    memory = RUNNER._memory_from_intent_dialogue_result(
        instance_name="owner__repo-1",
        problem_statement="Custom text should work across the widget API.",
        issue_evidence=IssueEvidenceFrame(),
        result=result,
    )

    bundle = memory.repair_intent_bundle
    assert bundle["candidate_boundaries"][0]["rationale"] == boundary.rationale
    assert bundle["repair_surface_leads"] == [{
        "id": "SURFACE-1",
        "kind": "candidate_boundary",
        "statement": "Widget::encode: Widget::encode; Widget::decode",
        "evidence_ids": ["SEARCH-1"],
    }]
    assert bundle["cited_repository_evidence"][0]["id"] == "SEARCH-1"


def test_dialogue_memory_prioritizes_source_referenced_by_final_fact(monkeypatch):
    monkeypatch.setenv("COTEST_PORTFOLIO_RERANK", "0")
    evidence = [
        SimpleNamespace(
            id=f"SEARCH-{index}",
            path="include/simdjson/generic/ondemand/object-inl.h",
            locator=f"read-{index}",
            observation=(
                "return value(iter.child());"
                if index == 12 else f"older source record {index}"
            ),
        )
        for index in range(1, 13)
    ]
    report = SimpleNamespace(
        input_path_check=None,
        investigation_paths=[],
        repository_evidence_ids=[item.id for item in evidence],
        preserve_evidence_ids=[],
        repository_facts=[
            "Object lookup advances through value(iter.child()) (SEARCH-12)."
        ],
        candidate_boundaries=[],
        open_decisions=[],
        contrastive_witness=None,
    )
    result = SimpleNamespace(
        render_for_repair=lambda: "grounded contract",
        intent_contract=RUNNER.RepairIntentContract(),
        investigation=SimpleNamespace(
            report=report,
            actions=[],
            repository_evidence=evidence,
        ),
        contrastive_decision=None,
        pair_status="",
        stop_reason="",
    )

    memory = RUNNER._memory_from_intent_dialogue_result(
        instance_name="simdjson__simdjson-1615",
        problem_statement="Implement JSON Pointer on top of On Demand.",
        issue_evidence=IssueEvidenceFrame(),
        result=result,
    )

    bundle = memory.repair_intent_bundle
    assert bundle["cited_repository_evidence"][0]["id"] == "SEARCH-12"
    assert bundle["cited_repository_evidence"][0]["observation"] == (
        "return value(iter.child());"
    )
    assert bundle["repair_surface_leads"] == [{
        "id": "SURFACE-1",
        "kind": "repository_fact",
        "statement": "Object lookup advances through value(iter.child()) (SEARCH-12).",
        "evidence_ids": ["SEARCH-12"],
    }]
    assert bundle["omitted_fact_evidence_ids"] == []


def test_dialogue_memory_preserves_the_latest_valid_attempt_witness():
    accepted_decision = SimpleNamespace(
        witness_valid=True,
        boundary_resolved=False,
    )
    accepted_witness = SimpleNamespace(
        observed_trace_delta="The accepted target fails on the baseline."
    )
    report = SimpleNamespace(
        input_path_check=None,
        investigation_paths=[],
        repository_evidence_ids=[],
        preserve_evidence_ids=[],
        repository_facts=[],
        candidate_boundaries=[],
        open_decisions=[],
        contrastive_witness=None,
    )
    result = SimpleNamespace(
        render_for_repair=lambda: "grounded contract",
        intent_contract=RUNNER.RepairIntentContract(),
        investigation=SimpleNamespace(
            report=report,
            actions=[],
            repository_evidence=[],
        ),
        contrastive_attempts=[
            SimpleNamespace(
                witness=accepted_witness,
                decision=accepted_decision,
            ),
            SimpleNamespace(
                witness=None,
                decision=SimpleNamespace(witness_valid=False),
            ),
        ],
        contrastive_decision=accepted_decision,
        pair_status="validated",
        stop_reason="contrastive_witness_only",
    )

    memory = RUNNER._memory_from_intent_dialogue_result(
        instance_name="org__repo-1",
        problem_statement="Fix the public behavior.",
        issue_evidence=IssueEvidenceFrame(),
        result=result,
    )

    assert memory.reproduction_result == accepted_witness.observed_trace_delta


def test_dialogue_memory_does_not_reserve_failed_graph_over_successful_graph(
    monkeypatch,
):
    monkeypatch.setenv("COTEST_PORTFOLIO_RERANK", "0")
    source = [
        SimpleNamespace(
            id=f"SEARCH-{index}", path="source.h", locator="read",
            observation=f"source {index}",
        )
        for index in range(1, 11)
    ]
    graphs = [
        SimpleNamespace(
            id="GRAPH-1", path="<code-graph>", locator="query",
            observation='[{"source_status":"verified"}]',
        ),
        SimpleNamespace(
            id="GRAPH-2", path="<code-graph>", locator="query",
            observation="No real one-hop relation matched.",
        ),
    ]
    report = SimpleNamespace(
        input_path_check=None, investigation_paths=[],
        repository_evidence_ids=[item.id for item in source],
        preserve_evidence_ids=[], repository_facts=[], candidate_boundaries=[],
        open_decisions=[], contrastive_witness=None,
    )
    result = SimpleNamespace(
        render_for_repair=lambda: "", intent_contract=RUNNER.RepairIntentContract(),
        investigation=SimpleNamespace(
            report=report,
            repository_evidence=[*source, *graphs],
            actions=[
                {"action": "RepositoryGraphQuery", "evidence_id": "GRAPH-1", "exit_code": 0},
                {"action": "RepositoryGraphQuery", "evidence_id": "GRAPH-2", "exit_code": 1},
            ],
        ),
        contrastive_decision=None, pair_status="", stop_reason="",
    )

    memory = RUNNER._memory_from_intent_dialogue_result(
        instance_name="org__repo-1", problem_statement="Issue",
        issue_evidence=IssueEvidenceFrame(), result=result,
    )

    evidence_ids = [
        item["id"]
        for item in memory.repair_intent_bundle["cited_repository_evidence"]
    ]
    assert "GRAPH-1" in evidence_ids
    assert "GRAPH-2" not in evidence_ids

    portfolio = IntentPortfolio(
        obligations=[
            IntentObligation(
                id="FIX-1",
                kind="fix",
                statement="Fix public seed selection.",
                authority="issue_explicit",
                strength="hard",
            )
        ],
        hypotheses=[
            IntentHypothesis(
                id="H0",
                title="repair selection",
                branch_assumption="none",
                covered_obligation_ids=["FIX-1"],
                hard_obligation_coverage={"FIX-1": "Repair the public seed."},
            )
        ],
    )
    assert not hasattr(RUNNER, "_portfolio_surface_disposition_gaps")


def test_repair_surface_facts_are_split_into_atomic_decisions():
    assert RUNNER._atomic_repair_surface_statements(
        [
            "The build uses a packaged header; the public macro expands to a "
            "seed operation, and the consumer only accepts library results, "
            "so a foreign result breaks later."
        ]
    ) == [
        "The build uses a packaged header",
        "the public macro expands to a seed operation",
        "the consumer only accepts library results",
        "a foreign result breaks later.",
    ]


def test_repair_discriminator_requires_exact_pass_marker_in_output():
    marker = RUNNER._portfolio_discriminator_marker("H1")
    probe = RepairProbeEvidence(
        probe_id="REPAIR-H1-1",
        source_hypothesis_id="H1",
        kind="discriminator",
        command=f"./probe && echo {marker}",
        observed_status="passed",
        observed_output=marker,
        production_patch_sha256="abc",
    )

    assert RUNNER._repair_discriminator_verified([probe], "H1") is True
    assert RUNNER._repair_discriminator_verified(
        [probe.model_copy(update={"observed_output": ""})], "H1"
    ) is False
    assert RUNNER._repair_discriminator_verified(
        [probe.model_copy(update={"kind": "focused"})], "H1"
    ) is False


def test_final_patch_probe_uses_marker_from_executed_output(monkeypatch):
    marker = RUNNER._portfolio_discriminator_marker("H1")
    candidate = SimpleNamespace(patch="diff --git a/src/fix.cc b/src/fix.cc\n")
    file_context = SimpleNamespace(
        verification={"ephemeral_paths": []},
        get_file=lambda _path: None,
    )
    step = SimpleNamespace(
        action=SimpleNamespace(
            name="RunSingleTest",
            command="g++ probe.cc -o /tmp/probe && /tmp/probe",
        ),
        observation=SimpleNamespace(
            properties={
                "exit_code": 0,
                "test_summary": {"raw_output": marker},
            },
            message=marker,
        ),
    )
    node = SimpleNamespace(file_context=file_context, action_steps=[step])
    trajectory = SimpleNamespace(get_all_nodes=lambda: [node])
    monkeypatch.setattr(
        RUNNER,
        "sanitize_candidate_patch",
        lambda _context: candidate,
    )

    _stamp_probe_execution_fixture(trajectory, candidate)
    probes = RUNNER._collect_final_patch_repair_probes(
        final_trajectory=trajectory,
        candidate=candidate,
        hypothesis_context={
            "branch": {
                "id": "H1",
                "discriminator": "conflicting providers select the new path",
                "predicted_outcome": "new path wins",
            }
        },
    )

    assert len(probes) == 1
    assert probes[0].kind == "discriminator"
    assert RUNNER._repair_discriminator_verified(probes, "H1")


def test_final_patch_probe_preserves_controlled_nonzero_oracle(monkeypatch):
    candidate = SimpleNamespace(
        patch="diff --git a/src/fix.cc b/src/fix.cc\n"
    )
    file_context = SimpleNamespace(
        verification={"ephemeral_paths": []},
        get_file=lambda _path: None,
    )
    output = '<OverallResult success="false" skips="0"/>'
    step = SimpleNamespace(
        action=SimpleNamespace(
            name="RunSingleTest",
            command="./expected_failure",
            expected_exit_code=1,
            required_output_regex=r'OverallResult success="false"',
        ),
        observation=SimpleNamespace(
            properties={
                "exit_code": 1,
                "oracle_satisfied": True,
                "test_summary": {"raw_output": output},
            },
            message=output,
        ),
    )
    node = SimpleNamespace(file_context=file_context, action_steps=[step])
    trajectory = SimpleNamespace(get_all_nodes=lambda: [node])
    monkeypatch.setattr(
        RUNNER,
        "sanitize_candidate_patch",
        lambda _context: candidate,
    )

    _stamp_probe_execution_fixture(trajectory, candidate)
    probes = RUNNER._collect_final_patch_repair_probes(
        final_trajectory=trajectory,
        candidate=candidate,
        hypothesis_context={"branch": {"id": "H0"}},
    )

    assert len(probes) == 1
    assert probes[0].observed_status == "passed"
    assert probes[0].expected_exit_code == 1
    assert probes[0].required_output_regex == (
        r'OverallResult success="false"'
    )


def test_final_patch_probe_retains_failed_focused_evidence(monkeypatch):
    candidate = SimpleNamespace(
        patch="diff --git a/src/fix.cc b/src/fix.cc\n"
    )
    file_context = SimpleNamespace(
        verification={"ephemeral_paths": []},
        get_file=lambda _path: None,
    )

    def step(command, exit_code, output):
        return SimpleNamespace(
            action=SimpleNamespace(name="RunSingleTest", command=command),
            observation=SimpleNamespace(
                properties={
                    "exit_code": exit_code,
                    "test_summary": {"raw_output": output},
                },
                message=output,
            ),
        )

    node = SimpleNamespace(
        file_context=file_context,
        action_steps=[
            step("c++ full_context.cc -o /tmp/full && /tmp/full", 1, "ambiguous"),
            step("c++ reduced.cc -o /tmp/reduced && /tmp/reduced", 0, "ok"),
        ],
    )
    trajectory = SimpleNamespace(get_all_nodes=lambda: [node])
    monkeypatch.setattr(
        RUNNER,
        "sanitize_candidate_patch",
        lambda _context: candidate,
    )

    _stamp_probe_execution_fixture(trajectory, candidate)
    probes = RUNNER._collect_final_patch_repair_probes(
        final_trajectory=trajectory,
        candidate=candidate,
        hypothesis_context={"branch": {"id": "H0"}},
    )

    assert [probe.observed_status for probe in probes] == ["failed", "passed"]
    assert RUNNER._portfolio_has_unresolved_focused_failure(
        probes, discriminator_required=False
    )
    assert RUNNER._portfolio_has_unresolved_focused_failure(
        probes, discriminator_required=True
    )

    marker_named_failure = probes[0].model_copy(
        update={"kind": "discriminator"}
    )
    assert RUNNER._portfolio_has_unresolved_focused_failure(
        [marker_named_failure], discriminator_required=False
    )
    assert "ambiguous" in RUNNER._portfolio_focused_failure_diagnostic(
        [marker_named_failure]
    )
    assert not RUNNER._portfolio_has_unresolved_focused_failure(
        [marker_named_failure], discriminator_required=True
    )
    soft_regression = probes[0].model_copy(
        update={"kind": "soft_regression"}
    )
    assert not RUNNER._portfolio_has_unresolved_focused_failure(
        [soft_regression], discriminator_required=False
    )


def test_final_patch_probe_drops_probe_wiring_failures(monkeypatch):
    candidate = SimpleNamespace(
        patch="diff --git a/src/fix.cc b/src/fix.cc\n"
    )
    file_context = SimpleNamespace(
        verification={"ephemeral_paths": []},
        get_file=lambda _path: None,
    )

    def step(command, output):
        return SimpleNamespace(
            action=SimpleNamespace(name="RunSingleTest", command=command),
            observation=SimpleNamespace(
                properties={
                    "exit_code": 1,
                    "test_summary": {"raw_output": output},
                },
                message=output,
            ),
        )

    node = SimpleNamespace(
        file_context=file_context,
        action_steps=[
            step("c++ repro.cc", "undefined reference to `main'"),
            step("sh repro.sh", 'sh: 1: Syntax error: "(" unexpected'),
            step("c++ real.cc", "existing caller has no member named get"),
        ],
    )
    trajectory = SimpleNamespace(get_all_nodes=lambda: [node])
    monkeypatch.setattr(
        RUNNER,
        "sanitize_candidate_patch",
        lambda _context: candidate,
    )

    _stamp_probe_execution_fixture(trajectory, candidate)
    probes = RUNNER._collect_final_patch_repair_probes(
        final_trajectory=trajectory,
        candidate=candidate,
        hypothesis_context={"branch": {"id": "H0"}},
    )

    assert [probe.command for probe in probes] == ["c++ real.cc"]


def test_final_patch_probe_drops_superseded_link_setup_failure(monkeypatch):
    candidate = SimpleNamespace(
        patch="diff --git a/src/fix.cc b/src/fix.cc\n"
    )
    context_file = SimpleNamespace(content="int main() { return 1; }\n")
    file_context = SimpleNamespace(
        verification={"ephemeral_paths": ["repro.cpp"]},
        get_file=lambda path: context_file if path == "repro.cpp" else None,
    )

    def step(command, exit_code, output, *, oracle_satisfied=None):
        properties = {
            "exit_code": exit_code,
            "test_summary": {"raw_output": output},
        }
        if oracle_satisfied is not None:
            properties["oracle_satisfied"] = oracle_satisfied
        return SimpleNamespace(
            action=SimpleNamespace(
                name="RunSingleTest",
                command=command,
                expected_exit_code=1 if oracle_satisfied else 0,
                required_output_regex=(
                    "OverallResult" if oracle_satisfied else ""
                ),
            ),
            observation=SimpleNamespace(
                properties=properties,
                message=output,
            ),
        )

    node = SimpleNamespace(
        file_context=file_context,
        action_steps=[
            step(
                "c++ repro.cpp -lMain -lCore -o /tmp/repro",
                1,
                "undefined reference to `Session::run()`\n"
                "collect2: error: ld returned 1 exit status",
            ),
            step(
                "c++ repro.cpp -lCore -lMain -o /tmp/repro && /tmp/repro",
                1,
                '<OverallResult success="false"/>',
                oracle_satisfied=True,
            ),
        ],
    )
    trajectory = SimpleNamespace(get_all_nodes=lambda: [node])
    monkeypatch.setattr(
        RUNNER,
        "sanitize_candidate_patch",
        lambda _context: candidate,
    )

    _stamp_probe_execution_fixture(trajectory, candidate)
    probes = RUNNER._collect_final_patch_repair_probes(
        final_trajectory=trajectory,
        candidate=candidate,
        hypothesis_context={"branch": {"id": "H0"}},
    )

    assert len(probes) == 1
    assert probes[0].observed_status == "passed"
    assert probes[0].command.endswith("&& /tmp/repro")


def test_final_patch_probe_keeps_all_current_evidence_without_two_record_cap(monkeypatch):
    candidate = SimpleNamespace(
        patch="diff --git a/src/fix.cc b/src/fix.cc\n"
    )
    file_context = SimpleNamespace(
        verification={"ephemeral_paths": []},
        get_file=lambda _path: None,
    )

    def step(command, output):
        return SimpleNamespace(
            action=SimpleNamespace(name="RunSingleTest", command=command),
            observation=SimpleNamespace(
                properties={
                    "exit_code": 1,
                    "test_summary": {"raw_output": output},
                },
                message=output,
            ),
        )

    node = SimpleNamespace(
        file_context=file_context,
        action_steps=[
            step("compile bad_setup.cc", "missing setup input"),
            step("compile full_issue.cc", "real competing candidates"),
            step("compile reduced_1.cc", "toy failure one"),
            step("compile reduced_2.cc", "toy failure two"),
        ],
    )
    trajectory = SimpleNamespace(get_all_nodes=lambda: [node])
    monkeypatch.setattr(
        RUNNER,
        "sanitize_candidate_patch",
        lambda _context: candidate,
    )

    _stamp_probe_execution_fixture(trajectory, candidate)
    probes = RUNNER._collect_final_patch_repair_probes(
        final_trajectory=trajectory,
        candidate=candidate,
        hypothesis_context={"branch": {"id": "H0"}},
    )

    assert [probe.command for probe in probes] == [
        "compile bad_setup.cc",
        "compile full_issue.cc",
        "compile reduced_1.cc",
        "compile reduced_2.cc",
    ]
    diagnostic = RUNNER._portfolio_focused_failure_diagnostic(probes)
    assert "real competing candidates" in diagnostic
    assert "toy failure" in diagnostic


def test_final_patch_probe_retains_real_issue_pass_after_setup_failures(monkeypatch):
    monkeypatch.setenv("COTEST_PORTFOLIO_REPAIR_PROBES_PER_BRANCH", "2")
    candidate = SimpleNamespace(patch="diff --git a/src/fix.cc b/src/fix.cc\n")
    context = SimpleNamespace(verification={}, get_file=lambda _: None)
    source = "#include <library.hpp>\nint main(){assert(issue_result()==42);}\n"
    steps = []
    for command, code, output, files in [
        ("c++ native_one.cpp", 1, "fatal error: helper.hpp: No such file or directory", {}),
        ("c++ native_two.cpp", 1, "undefined reference to `TestHarness::run()'", {}),
        ("c++ /tmp/issue.cpp && ./a.out", 0, "", {"/tmp/issue.cpp": source}),
    ]:
        steps.append(SimpleNamespace(
            action=SimpleNamespace(name="Bash", verification=True, command=command),
            observation=SimpleNamespace(message=output, properties={
                "exit_code": code, "output": output,
                "execution_snapshot": {
                    "production_patch_sha256": RUNNER._portfolio_patch_sha256(candidate.patch),
                    "command": command, "expected_exit_code": 0, "required_output_regex": "",
                    "probe_files": files,
                },
            }),
        ))
    trajectory = SimpleNamespace(get_all_nodes=lambda: [
        SimpleNamespace(file_context=context, action_steps=steps)
    ])
    probes = RUNNER._collect_final_patch_repair_probes(
        final_trajectory=trajectory, candidate=candidate, hypothesis_context=None,
    )
    assert len(probes) == 3
    assert probes[-1].observed_status == "passed"
    assert probes[-1].files == {"/tmp/issue.cpp": source}


def test_final_patch_probe_deduplicates_execution_with_same_frozen_source_only():
    candidate = SimpleNamespace(patch="diff --git a/src/fix.cc b/src/fix.cc\n")
    context = SimpleNamespace(verification={}, get_file=lambda _: None)
    command = "c++ /tmp/p.cpp && ./a.out"
    original, placeholder = "int main(){assert(check());}\n", "int main(){return 0;}\n"
    nodes = []
    for source, code in [(original, 1), (placeholder, 0), (placeholder, 0)]:
        nodes.append(SimpleNamespace(file_context=context, action_steps=[SimpleNamespace(
            action=SimpleNamespace(name="Bash", verification=True, command=command),
            observation=SimpleNamespace(message="", properties={
                "exit_code": code, "output": "assertion failed" if code else "",
                "execution_snapshot": {
                    "production_patch_sha256": RUNNER._portfolio_patch_sha256(candidate.patch),
                    "command": command, "expected_exit_code": 0, "required_output_regex": "",
                    "probe_files": {"/tmp/p.cpp": source},
                },
            }),
        )]))
    probes = RUNNER._collect_final_patch_repair_probes(
        final_trajectory=SimpleNamespace(get_all_nodes=lambda: nodes),
        candidate=candidate, hypothesis_context=None,
    )
    assert [p.files["/tmp/p.cpp"] for p in probes] == [original, placeholder]
    assert [p.observed_status for p in probes] == ["failed", "passed"]


def test_portfolio_candidate_status_keeps_branch_contradiction_for_selector(
    tmp_path,
):
    candidate = SimpleNamespace(
        patch="diff --git a/src/fix.cc b/src/fix.cc\n",
        kept_paths=["src/fix.cc"],
        excluded_paths={},
    )

    RUNNER._save_portfolio_candidate_status(
        str(tmp_path),
        finish_succeeded=True,
        basic_tests_passed=True,
        candidate=candidate,
        branch_compliance="noncompliant",
        discriminator_required=True,
        discriminator_verified=False,
    )

    payload = json.loads(
        (tmp_path / "portfolio_candidate_status.json").read_text()
    )
    assert payload["admitted"] is True
    assert payload["branch_compliance"] == "noncompliant"
    assert payload["branch_compliance_role"] == "selector_evidence"
    assert payload["discriminator_role"] == "selector_evidence"
    assert payload["admission_policy"] == (
        "finished_nonempty_production_patch"
    )


def test_portfolio_candidate_status_keeps_compile_regression_as_selector_evidence(
    tmp_path,
):
    candidate = SimpleNamespace(
        patch="diff --git a/src/fix.cc b/src/fix.cc\n",
        kept_paths=["src/fix.cc"],
        excluded_paths={},
    )
    regression = {
        "passed": False,
        "log_type": "compilation_error",
        "summary": "build failed",
    }

    RUNNER._save_portfolio_candidate_status(
        str(tmp_path),
        finish_succeeded=True,
        basic_tests_passed=False,
        candidate=candidate,
        blocking_compile_regression=True,
        regression_evidence=regression,
    )

    payload = json.loads(
        (tmp_path / "portfolio_candidate_status.json").read_text()
    )
    assert payload["admitted"] is True
    assert payload["blocking_compile_regression"] is True
    assert payload["regression_evidence"] == regression
    assert payload["regression_role"] == "soft_evidence"


def test_portfolio_candidate_status_keeps_focused_failure_as_evidence(tmp_path):
    candidate = SimpleNamespace(
        patch="diff --git a/src/fix.cc b/src/fix.cc\n",
        kept_paths=["src/fix.cc"],
        excluded_paths={},
    )

    RUNNER._save_portfolio_candidate_status(
        str(tmp_path),
        finish_succeeded=True,
        basic_tests_passed=True,
        candidate=candidate,
        branch_compliance="compliant",
        discriminator_verified=True,
        unresolved_focused_failure=True,
    )

    payload = json.loads(
        (tmp_path / "portfolio_candidate_status.json").read_text()
    )
    assert payload["admitted"] is True
    assert payload["unresolved_focused_failure"] is True
    assert payload["focused_failure_role"] == "selector_evidence"


def test_portfolio_candidate_status_retains_precontinuation_variant(tmp_path):
    candidate = SimpleNamespace(
        patch="diff --git a/src/final.cc b/src/final.cc\n",
        kept_paths=["src/final.cc"],
        excluded_paths={},
    )
    initial = {
        "variant_id": "initial",
        "patch": "diff --git a/src/initial.cc b/src/initial.cc\n",
        "admitted": True,
        "finish_succeeded": True,
        "kept_paths": ["src/initial.cc"],
        "excluded_paths": {},
        "repair_probes": [],
    }
    failed_continuation = {
        "variant_id": "continuation_1",
        "patch": candidate.patch,
        "admitted": False,
        "finish_succeeded": False,
        "kept_paths": candidate.kept_paths,
        "excluded_paths": {},
        "repair_probes": [],
    }

    RUNNER._save_portfolio_candidate_status(
        str(tmp_path),
        finish_succeeded=False,
        basic_tests_passed=False,
        candidate=candidate,
        candidate_variants=[initial, failed_continuation],
    )

    payload = json.loads(
        (tmp_path / "portfolio_candidate_status.json").read_text()
    )
    assert payload["admitted"] is True
    assert [item["variant_id"] for item in payload["candidate_variants"]] == [
        "initial",
        "continuation_1",
    ]


def test_failed_continuation_does_not_replace_same_finished_patch():
    variants = {}
    initial = {
        "variant_id": "initial",
        "patch": "same-patch",
        "admitted": True,
    }
    failed_continuation = {
        "variant_id": "continuation_1",
        "patch": "same-patch",
        "admitted": False,
    }

    RUNNER._retain_portfolio_candidate_variant(variants, initial)
    RUNNER._retain_portfolio_candidate_variant(variants, failed_continuation)

    assert list(variants.values()) == [initial]


def test_portfolio_recovers_persisted_initial_candidate_after_repair_error(
    monkeypatch, tmp_path
):
    from INTENT2FIX import selector_execution
    portfolio = IntentPortfolio(
        obligations=[
            IntentObligation(
                id="FIX-1",
                kind="fix",
                statement="Fix the public behavior.",
                authority="issue_explicit",
                strength="hard",
            )
        ],
        hypotheses=[
            IntentHypothesis(
                id="H0",
                title="bounded repair",
                branch_assumption="none",
                covered_obligation_ids=["FIX-1"],
                hard_obligation_coverage={"FIX-1": "Fix it."},
            )
        ],
    )
    initial_patch = "diff --git a/src/fix.cc b/src/fix.cc\n"

    def fake_repair_step(*_args, output_dir, **_kwargs):
        status = {
            "admitted": True,
            "candidate_variants": [
                {
                    "variant_id": "initial",
                    "patch": initial_patch,
                    "admitted": True,
                    "finish_succeeded": True,
                    "basic_tests_passed": False,
                    "branch_compliance": "noncompliant",
                    "discriminator_verified": False,
                    "kept_paths": ["src/fix.cc"],
                    "excluded_paths": {},
                    "repair_probes": [],
                }
            ],
        }
        Path(output_dir, "portfolio_candidate_status.json").write_text(
            json.dumps(status), encoding="utf-8"
        )
        raise RuntimeError("continuation failed after initial persistence")

    class FakeDockerEnv:
        def __init__(self, **_kwargs):
            pass

    monkeypatch.setattr(
        RUNNER, "_generate_intent_portfolio", lambda *_args, **_kwargs: portfolio
    )
    monkeypatch.setattr(RUNNER, "run_repair_step", fake_repair_step)
    monkeypatch.setattr(RUNNER, "DockerEnv", FakeDockerEnv)
    monkeypatch.setattr(selector_execution, "compile_candidate", lambda candidate, **_kwargs: {
        "candidate_id": candidate["candidate_id"], "patch_sha256": candidate["patch_sha256"],
        "status": "passed", "phase": "build", "exit_code": 0,
    })
    monkeypatch.setattr(
        RUNNER,
        "_cross_execute_portfolio_candidate",
        lambda **kwargs: CrossExecutionResult(
            status="passed" if kwargs.get("patch") else "unavailable",
            patch_applied=True if kwargs.get("patch") else None,
        ),
    )
    monkeypatch.setattr(
        RUNNER,
        "_run_structured_extraction",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("selector unavailable")
        ),
    )

    result = RUNNER.run_intent_portfolio_repair_step(
        instance_name="org__repo-1",
        model_name="glm-5.3-flash",
        image_name="image",
        work_dir="/repo",
        base_commit="abc",
        memory=Memory(
            instance_id="org__repo-1",
            problem_statement="Fix the public behavior.",
        ),
        output_dir=str(tmp_path),
        container_suffix="run",
    )

    assert result is None
    pool = json.loads((tmp_path / "portfolio_candidate_pool.json").read_text())
    assert pool[0]["variant_id"] == "initial"
    assert json.loads((tmp_path / "portfolio_compile_gate.json").read_text())[0]["status"] == "passed"
    assert not (tmp_path / "solution.patch").exists()


def test_independent_selector_sees_public_issue_and_uses_low_reasoning(
    monkeypatch, tmp_path
):
    from scripts import run_portfolio_selector

    candidates = [
        PortfolioCandidate.create(
            hypothesis_id=f"H{index}",
            branch_assumption="private hypothesis assumption",
            patch=f"patch-{index}",
            kept_paths=["src/fix.cc"],
            excluded_paths={},
            finish_succeeded=True,
            basic_tests_passed=False,
            cross_execution=CrossExecutionResult(status="unavailable"),
        )
        for index in range(2)
    ]
    monkeypatch.setattr(
        RUNNER, "get_benchmark_instance",
        lambda _iid: {
            "instance_id": "org__repo-1",
            "repo": "org/repo",
            "base_commit": "abc",
            "problem_statement": "Public issue text",
            "patch": "hidden golden patch",
            "test_patch": "hidden official test",
        },
    )

    def fake_prepare(public, raw, repo, repair_intent=None, hypotheses=None):
        assert set(public) == {
            "instance_id", "repo", "base_commit", "problem_statement"
        }
        assert len(raw) == 2
        assert repo == Path(tmp_path) / "org__repo-1"
        assert repair_intent["scout"]["target_behavior"] == ["repair public behavior"]
        assert repair_intent["common_obligations"][0]["id"] == "FIX-1"
        assert "hypotheses" not in repair_intent
        assert [h["id"] for h in hypotheses] == ["H0"]
        assert "hard_obligation_coverage" in hypotheses[0]
        return {"public": public, "candidates": raw}

    def fake_run(case, directory, **kwargs):
        assert case["public"]["problem_statement"] == "Public issue text"
        assert directory == tmp_path / "output" / "portfolio_independent_selector"
        assert kwargs["execute"] is True
        assert kwargs["reasoning_effort"] == "low"
        assert kwargs["model_alias"] == "glm-5.3-flash"
        return {"selected_patch_sha256": candidates[1].patch_sha256}

    monkeypatch.setattr(run_portfolio_selector, "prepare_case", fake_prepare)
    monkeypatch.setattr(run_portfolio_selector, "run_case", fake_run)
    monkeypatch.setenv("REPO_DIR", str(tmp_path))

    selected = RUNNER._select_portfolio_candidate_independently(
        instance_name="org__repo-1", base_commit="abc",
        model_name="glm-5.3-flash", output_dir=str(tmp_path / "output"),
        candidates=candidates,
        memory=Memory(instance_id="org__repo-1", problem_statement="Public issue text",
                      repair_intent_contract={"target_behavior": ["repair public behavior"]}),
        portfolio=IntentPortfolio(
            obligations=[IntentObligation(id="FIX-1", kind="fix", statement="Repair public behavior",
                                          authority="issue_explicit", strength="hard")],
            hypotheses=[IntentHypothesis(id="H0", title="branch", branch_assumption="none",
                                         covered_obligation_ids=["FIX-1"],
                                         hard_obligation_coverage={"FIX-1": "Repair public behavior"})],
        ),
    )

    assert selected is candidates[1]


def test_independent_backend_skips_hypothesis_probe_matrix(monkeypatch, tmp_path):
    portfolio = IntentPortfolio(
        obligations=[IntentObligation(
            id="FIX-1", kind="fix", statement="Fix it.",
            authority="issue_explicit", strength="hard",
        )],
        hypotheses=[IntentHypothesis(
            id="H0", title="repair", branch_assumption="hypothesis only",
            covered_obligation_ids=["FIX-1"],
            hard_obligation_coverage={"FIX-1": "Fix it."},
        )],
    )
    patches = [f"patch-{index}" for index in range(2)]

    def fake_repair(*_args, output_dir, **_kwargs):
        Path(output_dir, "portfolio_candidate_status.json").write_text(
            json.dumps({"admitted": True, "candidate_variants": [
                {"variant_id": f"v{index}", "patch": patch, "admitted": True,
                 "kept_paths": ["src/fix.cc"], "excluded_paths": {},
                 "repair_probes": []}
                for index, patch in enumerate(patches)
            ]})
        )
        return patches[0]

    monkeypatch.setenv("COTEST_PORTFOLIO_SELECTOR_BACKEND", "independent")
    monkeypatch.setattr(RUNNER, "DockerEnv", lambda **_kwargs: None)
    monkeypatch.setattr(RUNNER, "_build_portfolio_branch_memory", lambda *a: a[0])
    monkeypatch.setattr(RUNNER, "run_repair_step", fake_repair)
    monkeypatch.setattr(
        RUNNER, "_cross_execute_portfolio_candidate",
        lambda **_kwargs: pytest.fail("H-authored probes must not be cross-executed"),
    )
    monkeypatch.setattr(
        RUNNER, "_select_portfolio_candidate_independently",
        lambda **kwargs: kwargs["candidates"][1],
    )

    result = RUNNER.run_intent_portfolio_repair_step(
        instance_name="org__repo-1", model_name="glm-5.3-flash",
        image_name="image", work_dir="/repo", base_commit="abc",
        memory=Memory(instance_id="org__repo-1", problem_statement="Fix it."),
        output_dir=str(tmp_path), container_suffix="run", portfolio=portfolio,
    )

    assert result == patches[1]
    assert (tmp_path / "solution.patch").read_text() == patches[1]
    assert len(json.loads((tmp_path / "portfolio_candidate_pool.json").read_text())) == 2
    assert not (tmp_path / "portfolio_baseline_execution.json").exists()


def test_legacy_selector_rejects_candidate_without_public_compilation(monkeypatch, tmp_path):
    from INTENT2FIX import selector_execution

    portfolio = IntentPortfolio(
        obligations=[IntentObligation(id="FIX-1", kind="fix", statement="Fix the public behavior",
                                      authority="issue_explicit", strength="hard")],
        hypotheses=[IntentHypothesis(id="H0", title="repair", branch_assumption="none",
                                     covered_obligation_ids=["FIX-1"],
                                     hard_obligation_coverage={"FIX-1": "Fix the public behavior"})],
    )
    patch = "diff --git a/src/fix.cc b/src/fix.cc\n"
    def fake_repair(*_args, output_dir, **_kwargs):
        Path(output_dir, "portfolio_candidate_status.json").write_text(json.dumps({
            "admitted": True, "candidate_variants": [{"variant_id": "final", "patch": patch,
                "admitted": True, "finish_succeeded": True, "kept_paths": ["src/fix.cc"],
                "excluded_paths": {}, "repair_probes": []}]}))
        return patch
    monkeypatch.setenv("COTEST_PORTFOLIO_SELECTOR_BACKEND", "legacy")
    monkeypatch.setattr(RUNNER, "DockerEnv", lambda **_kwargs: None)
    monkeypatch.setattr(RUNNER, "_build_portfolio_branch_memory", lambda *a: a[0])
    monkeypatch.setattr(RUNNER, "run_repair_step", fake_repair)
    monkeypatch.setattr(selector_execution, "compile_candidate", lambda candidate, **_kwargs: {
        "candidate_id": candidate["candidate_id"], "patch_sha256": candidate["patch_sha256"],
        "status": "failed", "phase": "build", "exit_code": 1,
    })
    monkeypatch.setattr(RUNNER, "_cross_execute_portfolio_candidate",
                        lambda **_kwargs: pytest.fail("No uncompiled candidate may reach cross execution"))

    result = RUNNER.run_intent_portfolio_repair_step(
        instance_name="org__repo-1", model_name="glm-5.3-flash", image_name="image",
        work_dir="/repo", base_commit="abc",
        memory=Memory(instance_id="org__repo-1", problem_statement="Fix the public behavior"),
        output_dir=str(tmp_path), container_suffix="run", portfolio=portfolio,
    )

    assert result is None
    assert json.loads((tmp_path / "portfolio_compile_gate.json").read_text())[0]["status"] == "failed"
    assert len(json.loads((tmp_path / "portfolio_candidate_pool.json").read_text())) == 1
    assert not (tmp_path / "solution.patch").exists()


def test_independent_selector_failure_cannot_fallback_around_compile_gate(monkeypatch, tmp_path):
    portfolio = IntentPortfolio(
        obligations=[IntentObligation(id="FIX-1", kind="fix", statement="Fix the public behavior",
                                      authority="issue_explicit", strength="hard")],
        hypotheses=[IntentHypothesis(id="H0", title="repair", branch_assumption="none",
                                     covered_obligation_ids=["FIX-1"],
                                     hard_obligation_coverage={"FIX-1": "Fix the public behavior"})],
    )
    patch = "diff --git a/src/fix.cc b/src/fix.cc\n"
    def fake_repair(*_args, output_dir, **_kwargs):
        Path(output_dir, "portfolio_candidate_status.json").write_text(json.dumps({
            "admitted": True, "candidate_variants": [{"variant_id": "final", "patch": patch,
                "admitted": True, "finish_succeeded": True, "kept_paths": ["src/fix.cc"],
                "excluded_paths": {}, "repair_probes": []}]}))
        return patch
    monkeypatch.setenv("COTEST_PORTFOLIO_SELECTOR_BACKEND", "independent")
    monkeypatch.setattr(RUNNER, "DockerEnv", lambda **_kwargs: None)
    monkeypatch.setattr(RUNNER, "_build_portfolio_branch_memory", lambda *a: a[0])
    monkeypatch.setattr(RUNNER, "run_repair_step", fake_repair)
    monkeypatch.setattr(RUNNER, "_select_portfolio_candidate_independently",
                        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("build unavailable")))

    result = RUNNER.run_intent_portfolio_repair_step(
        instance_name="org__repo-1", model_name="glm-5.3-flash", image_name="image",
        work_dir="/repo", base_commit="abc",
        memory=Memory(instance_id="org__repo-1", problem_statement="Fix the public behavior"),
        output_dir=str(tmp_path), container_suffix="run", portfolio=portfolio,
    )

    assert result is None
    assert json.loads((tmp_path / "portfolio_independent_selector_failure.json").read_text())["fallback"] == "none"
    assert not (tmp_path / "solution.patch").exists()


def test_only_current_compilation_failure_blocks_portfolio_candidate():
    context = FileContext(
        repo=InMemRepository(),
        change_seq=3,
        verification={
            "regression": {
                "passed": False,
                "log_type": "test_results",
                "change_seq": 3,
                "verification_view": "production",
                "verification_isolation": RUNNER.PRODUCTION_ISOLATION,
                "production_patch_sha256": RUNNER._portfolio_patch_sha256(""),
            }
        },
    )

    assert not RUNNER._portfolio_compile_regression_failure(context)

    context.verification["regression"]["log_type"] = "compilation_error"
    assert RUNNER._portfolio_compile_regression_failure(context)

    context.verification["regression"].update({
        "output": "Could NOT find an optional package",
        "summary": "src/f.cpp:12:3: error: cannot convert value",
    })
    assert RUNNER._portfolio_compile_regression_failure(context)

    context.verification["regression"]["change_seq"] = 2
    assert not RUNNER._portfolio_compile_regression_failure(context)


def test_current_production_regression_takes_precedence_over_probe_view():
    context = FileContext(
        repo=InMemRepository(),
        change_seq=4,
        verification={
            "regression": {
                "passed": False,
                "log_type": "compilation_error",
                "change_seq": 4,
            },
            "production_regression": {
                "passed": True,
                "log_type": "test_results",
                "change_seq": 4,
                "verification_view": "production",
                "verification_isolation": RUNNER.PRODUCTION_ISOLATION,
                "production_patch_sha256": RUNNER._portfolio_patch_sha256(""),
            },
        },
    )

    assert RUNNER._fresh_portfolio_regression(context)["passed"] is True
    assert not RUNNER._portfolio_compile_regression_failure(context)


def test_portfolio_repair_has_no_verification_budget_gate():
    assert not hasattr(RUNNER, "_install_portfolio_verification_budget")


def test_portfolio_repair_has_no_diagnostic_convergence_gate():
    assert not hasattr(
        RUNNER, "_install_portfolio_diagnostic_convergence_feedback"
    )


@pytest.mark.parametrize("finish_type", (RUNNER.VerifiedFinish, RUNNER.Finish))
def test_portfolio_finish_tail_allows_only_finish_without_changing_tool_schema(
    monkeypatch, finish_type
):
    finish = finish_type()
    search = SimpleNamespace(name="Search")

    class FakeAgent:
        def __init__(self):
            self.actions = [search, finish]
            self.seen = []

        def set_actions(self, _actions):
            raise AssertionError("initialized tool schema must not change")

        def set_execution_action_allowlist(self, action_types):
            self.seen.append(action_types)

    class FakeRoot:
        def __init__(self):
            self.nodes = [object(), object()]

        def get_all_nodes(self):
            return self.nodes

    class FakeLoop:
        def __init__(self):
            self.agent = FakeAgent()
            self.root = FakeRoot()
            self.max_iterations = 0
            self.feedback = ""

        def append_user_feedback(self, message):
            self.feedback = message
            self.root.nodes.append(object())

        def run(self):
            assert self.agent.actions == [search, finish]
            assert self.agent.seen[-1] == {finish.args_schema}
            assert self.max_iterations == 5
            return "finished"

    trajectory = SimpleNamespace(
        is_finished=lambda: False,
        file_context=FileContext(repo=InMemRepository()),
    )
    loop = FakeLoop()
    monkeypatch.setattr(
        RUNNER, "_generate_submission_patch", lambda _context: "production patch"
    )

    result = RUNNER._run_portfolio_finish_tail(
        loop, trajectory, max_finish_attempts=2
    )

    assert result == "finished"
    assert "do not run another test" in loop.feedback
    assert loop.agent.actions == [search, finish]
    assert loop.agent.seen == [{finish.args_schema}, None]


def test_repair_prompt_file_override_is_scoped_and_requires_absolute_path(
    monkeypatch, tmp_path
):
    prompt_path = tmp_path / "repair_prompt.txt"
    prompt_path.write_text("four-tool repair prompt\n", encoding="utf-8")
    monkeypatch.setenv("COTEST_REPAIR_SYSTEM_PROMPT_FILE", str(prompt_path))

    assert RUNNER._repair_agent_system_prompt(None) == "four-tool repair prompt"
    assert RUNNER._repair_agent_system_prompt("explicit prompt") == "explicit prompt"

    monkeypatch.setenv("COTEST_REPAIR_SYSTEM_PROMPT_FILE", "relative.txt")
    with pytest.raises(ValueError, match="must be absolute"):
        RUNNER._repair_agent_system_prompt(None)


def test_repair_outline_override_is_repair_scoped(monkeypatch):
    monkeypatch.setenv("COTEST_REPAIR_OMIT_REPOSITORY_OUTLINE", "1")
    assert RUNNER._env_bool("COTEST_REPAIR_OMIT_REPOSITORY_OUTLINE") is True


def test_candidate_introduced_missing_header_is_not_probe_setup_failure():
    context = SimpleNamespace(
        verification={"ephemeral_paths": ["focused_probe.cpp"]}
    )

    assert not RUNNER._portfolio_probe_setup_failure(
        "g++ focused_probe.cpp -o /tmp/probe",
        (
            "In file included from focused_probe.cpp:1:\n"
            "src/changed.hpp:2:10: fatal error: new/dependency.hpp: "
            "No such file or directory"
        ),
        file_context=context,
        production_patch=(
            "diff --git a/src/changed.hpp b/src/changed.hpp\n"
            "+#include <new/dependency.hpp>\n"
        ),
    )


def test_cross_execution_applies_sanitized_patch_in_fresh_container(monkeypatch):
    instances = []

    class FakeDockerEnv:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.work_dir = kwargs["work_dir"]
            self.calls = []
            self.writes = []
            instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def write_file(self, path, content):
            self.writes.append((path, content))

        def run_bash(self, command, timeout):
            self.calls.append((command, timeout))
            if command.startswith("git apply --check"):
                return "applied", 0
            return "target passed", 0

    monkeypatch.setattr(RUNNER, "DockerEnv", FakeDockerEnv)
    memory = SimpleNamespace(
        test_command="./run-target.sh",
        test_files=[{"path": "repro_case.cc", "content": "int main() {}"}],
    )

    result = RUNNER._cross_execute_portfolio_candidate(
        instance_name="org__repo-1",
        image_name="image",
        work_dir="/home/repo",
        base_commit="abc",
        container_suffix="run",
        label="H0",
        memory=memory,
        patch="diff --git a/src/a.cc b/src/a.cc\n",
    )

    assert result.status == "passed"
    assert result.patch_applied is True
    assert len(instances) == 1
    assert instances[0].kwargs["container_name"].startswith(
        "portfolio-check-H0-"
    )
    assert instances[0].calls[0][0].startswith("git apply --check")
    assert instances[0].calls[1][0] == "./run-target.sh"
    assert any(path.endswith("repro_case.cc") for path, _ in instances[0].writes)


def test_repair_probe_specs_distinguish_same_hypothesis_variants():
    candidates = []
    for variant_id in ("initial", "continuation_1"):
        candidates.append(
            PortfolioCandidate.create(
                hypothesis_id="H1",
                variant_id=variant_id,
                branch_assumption="assigned behavior",
                patch=f"patch-{variant_id}",
                kept_paths=["src/fix.cc"],
                excluded_paths={},
                finish_succeeded=True,
                basic_tests_passed=False,
                repair_probes=[
                    RepairProbeEvidence(
                        probe_id="REPAIR-H1-1",
                        source_hypothesis_id="H1",
                        command="./probe",
                        expected_outcome=f"prints {variant_id}",
                        required_output_regex=variant_id,
                        observed_status="passed",
                        production_patch_sha256="abc",
                    )
                ],
                cross_execution=CrossExecutionResult(status="unavailable"),
            )
        )

    specs = RUNNER._portfolio_repair_probe_specs(candidates)

    assert [item[0] for item in specs] == [
        "REPAIR-H1-1@initial",
        "REPAIR-H1-1@continuation_1",
    ]
    assert [item[1]["source_hypothesis_id"] for item in specs] == [
        "H1@initial",
        "H1@continuation_1",
    ]


def test_cross_execution_replays_required_frozen_acceptance_probes(monkeypatch):
    instances = []

    class FakeDockerEnv:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.calls = []
            self.writes = []
            instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def write_file(self, path, content):
            self.writes.append((path, content))

        def run_bash(self, command, timeout):
            self.calls.append((command, timeout))
            return "ok", 0

    monkeypatch.setattr(RUNNER, "DockerEnv", FakeDockerEnv)
    memory = SimpleNamespace(
        test_command="",
        test_files=[],
        repair_intent_contract={
            "acceptance_ids": ["ACCEPT-2"],
            "probes": {
                "ACCEPT-2": {
                    "authority": "normative",
                    "kind": "f2p",
                    "description": "Issue target",
                    "baseline_exit_code": 1,
                    "command": "c++ probe.cc -o /tmp/probe",
                    "files": {"probe.cc": "int main() {}"},
                }
            },
        },
    )

    result = RUNNER._cross_execute_portfolio_candidate(
        instance_name="org__repo-1",
        image_name="image",
        work_dir="/home/repo",
        base_commit="abc",
        container_suffix="run",
        label="H0",
        memory=memory,
        patch="diff --git a/src/a.cc b/src/a.cc\n",
    )

    assert result.status == "passed"
    assert result.patch_applied is True
    assert [item.check_id for item in result.probe_results] == ["ACCEPT-2"]
    assert result.probe_results[0].baseline_expected_exit_code == 1
    assert instances[0].calls[1][0] == "c++ probe.cc -o /tmp/probe"
    assert ("/home/repo/probe.cc", "int main() {}") in instances[0].writes


def test_cross_execution_rejects_non_replayable_scout_probe(monkeypatch):
    calls = []

    class FakeDockerEnv:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def write_file(self, _path, _content):
            return None

        def run_bash(self, command, timeout):
            calls.append((command, timeout))
            return "must not execute", 0

    monkeypatch.setattr(RUNNER, "DockerEnv", FakeDockerEnv)
    memory = SimpleNamespace(
        test_command="",
        test_files=[],
        repair_intent_contract={
            "acceptance_ids": ["ACCEPT-1"],
            "probes": {
                "ACCEPT-1": {
                    "authority": "normative",
                    "kind": "f2p",
                    "command": (
                        "c++ probe.cc /tmp/build/libtarget.a "
                        "-o /tmp/probe && /tmp/probe"
                    ),
                    "files": {"probe.cc": "int main() {}"},
                }
            },
        },
    )

    result = RUNNER._cross_execute_portfolio_candidate(
        instance_name="org__repo-1",
        image_name="image",
        work_dir="/home/repo",
        base_commit="abc",
        container_suffix="run",
        label="baseline",
        memory=memory,
        patch=None,
    )

    assert result.status == "infrastructure_error"
    assert result.probe_results[0].status == "infrastructure_error"
    assert "cannot depend on /tmp" in result.probe_results[0].output
    assert calls == []


def test_cross_execution_replays_illustrative_probe_as_soft_evidence(monkeypatch):
    class FakeDockerEnv:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def write_file(self, _path, _content):
            return None

        def run_bash(self, _command, timeout):
            assert timeout > 0
            return "target still fails", 1

    monkeypatch.setattr(RUNNER, "DockerEnv", FakeDockerEnv)
    memory = SimpleNamespace(
        test_command="",
        test_files=[],
        repair_intent_contract={
            "acceptance_ids": [],
            "diagnostic_ids": ["ACCEPT-1"],
            "probes": {
                "ACCEPT-1": {
                    "authority": "illustrative",
                    "kind": "diagnostic",
                    "description": "Issue example",
                    "baseline_exit_code": 1,
                    "command": "c++ probe.cc -o /tmp/probe",
                    "files": {"probe.cc": "int main() {}"},
                }
            },
        },
    )

    result = RUNNER._cross_execute_portfolio_candidate(
        instance_name="org__repo-1",
        image_name="image",
        work_dir="/home/repo",
        base_commit="abc",
        container_suffix="run",
        label="baseline",
        memory=memory,
        patch=None,
    )

    assert result.status == "unavailable"
    assert result.probe_results[0].check_id == "ACCEPT-1"
    assert result.probe_results[0].authority == "illustrative"
    assert result.probe_results[0].status == "failed"


def test_cross_execution_detects_nonzero_status_hidden_by_shell_echo(monkeypatch):
    class FakeDockerEnv:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def write_file(self, _path, _content):
            return None

        def run_bash(self, command, timeout):
            assert timeout > 0
            if command.startswith("git apply --check"):
                return "applied", 0
            return "R1=0\nR2=5\n", 0

    monkeypatch.setattr(RUNNER, "DockerEnv", FakeDockerEnv)
    memory = SimpleNamespace(
        test_command="",
        test_files=[],
        repair_intent_contract={},
    )
    result = RUNNER._cross_execute_portfolio_candidate(
        instance_name="org__repo-1",
        image_name="image",
        work_dir="/home/repo",
        base_commit="abc",
        container_suffix="run",
        label="H0",
        memory=memory,
        patch="diff --git a/src/a.cc b/src/a.cc\n",
        extra_probes=[
            (
                "REPAIR-H1-1",
                {
                    "authority": "hypothesis",
                    "kind": "discriminator",
                    "origin": "repair",
                    "source_hypothesis_id": "H1",
                    "command": "./probe; echo R2=$?",
                    "files": {},
                },
            )
        ],
    )

    assert result.status == "unavailable"
    assert result.probe_results[0].status == "failed"


def test_executed_probe_status_rejects_missing_binary_masked_by_echo():
    output = "sh: 1: /tmp/ranges_test: not found\nexit=0\n"

    assert RUNNER._executed_probe_status(output, 0) == "failed"


def test_verification_status_rejects_lowercase_masked_nonzero_exit():
    step = SimpleNamespace(
        observation=SimpleNamespace(
            message="compile finished\nexit=1\n",
            properties={
                "exit_code": 0,
                "test_summary": {"raw_output": "compile finished\nexit=1\n"},
            },
        )
    )

    assert RUNNER._verification_observation_status(step)[0] == "failed"


def test_executed_probe_status_rejects_sanitizer_failure_masked_by_success():
    output = "AddressSanitizer:DEADLYSIGNAL\nINTENT2FIX_DISCRIMINATOR_H0_PASS\n"

    assert RUNNER._executed_probe_status(output, 0) == "failed"


def test_executed_probe_status_supports_controlled_nonzero_oracle():
    output = '<OverallResult success="false" skips="0"/>'

    assert RUNNER._executed_probe_status(
        output,
        1,
        expected_exit_code=1,
        required_output_regex=r'OverallResult success="false"',
    ) == "passed"
    assert RUNNER._executed_probe_status(
        "unrelated failure",
        1,
        expected_exit_code=1,
        required_output_regex=r'OverallResult success="false"',
    ) == "failed"


def test_portfolio_generation_uses_explicit_repair_model(monkeypatch):
    calls = []
    portfolio = IntentPortfolio(
        obligations=[
            IntentObligation(
                id="FIX-1",
                kind="fix",
                statement="Fix it",
                authority="issue_explicit",
                strength="hard",
            )
        ],
        hypotheses=[
            IntentHypothesis(
                id="H0",
                title="literal",
                branch_assumption="none",
                covered_obligation_ids=["FIX-1"],
                hard_obligation_coverage={
                    "FIX-1": "Implement and verify the requested behavior."
                },
            )
        ]
    )

    audit = PortfolioCoverageAudit(
        decision="ACCEPT",
        reason="The portfolio covers the supplied contract.",
    )

    def fake_extract(prompt, schema, **kwargs):
        calls.append((prompt, schema, kwargs))
        return audit if schema is PortfolioCoverageAudit else portfolio

    monkeypatch.setattr(RUNNER, "_run_structured_extraction", fake_extract)
    memory = Memory(
        instance_id="org__repo-1",
        problem_statement="Fix it",
        repair_intent_bundle={"hypothesis_ready": False},
    )

    assert RUNNER._generate_intent_portfolio(
        memory, "glm-5.3-flash"
    ) == portfolio
    assert [call[1] for call in calls] == [
        IntentPortfolio,
        PortfolioCoverageAudit,
    ]
    assert all(
        call[2]["model_override"] == "glm-5.3-flash" for call in calls
    )
    assert "Readiness is diagnostic only, never a stop instruction" in calls[0][0]


def test_portfolio_generation_adjudicates_grounded_open_decision(
    monkeypatch, tmp_path
):
    adjudication = HypothesisMergeAdjudication(
        decisions=[
            MergeBoundaryDecision(
                id="MERGE-1",
                question="Which provider wins when member and ADL coexist?",
                verdict="keep_separate",
                alternatives=[
                    MergeBoundaryAlternative(
                        label="member-first",
                        selection_rule="Prefer a member provider.",
                        contract_viability="plausible",
                        viability_reason="It satisfies the common target.",
                        predicted_outcomes={"coexistence": "member wins"},
                    ),
                    MergeBoundaryAlternative(
                        label="ADL-first",
                        selection_rule="Use unqualified lookup.",
                        contract_viability="plausible",
                        viability_reason="It satisfies the common target.",
                        predicted_outcomes={"coexistence": "ADL wins"},
                    ),
                ],
                boundary_scenarios=["coexistence"],
                reason="The selected iterator differs.",
            )
        ]
    )
    portfolio = IntentPortfolio(
        obligations=[
            IntentObligation(
                id="FIX-1",
                kind="fix",
                statement="Use ADL.",
                authority="issue_explicit",
                strength="hard",
            )
        ],
        hypotheses=[
            IntentHypothesis(
                id="H0",
                title="member first",
                branch_assumption="member wins",
                covered_obligation_ids=["FIX-1"],
                hard_obligation_coverage={"FIX-1": "Use member-first lookup."},
            ),
            IntentHypothesis(
                id="H1",
                title="ADL first",
                branch_assumption="ADL wins",
                covered_obligation_ids=["FIX-1"],
                hard_obligation_coverage={"FIX-1": "Use unqualified lookup."},
            ),
        ],
    )
    calls = []
    audit = PortfolioCoverageAudit(
        decision="ACCEPT",
        reason="The alternatives remain distinct and complete.",
    )

    def fake_extract(prompt, schema, **kwargs):
        calls.append((prompt, schema, kwargs))
        if schema is HypothesisMergeAdjudication:
            return adjudication
        if schema is PortfolioCoverageAudit:
            return audit
        return portfolio

    monkeypatch.setattr(RUNNER, "_run_structured_extraction", fake_extract)
    memory = Memory(
        instance_id="fmtlib__fmt-3824",
        problem_statement="join should use ADL",
        repair_intent_contract={
            "open_decisions": [
                "member-first versus ADL-first when both providers exist"
            ]
        },
    )

    result = RUNNER._generate_intent_portfolio(
        memory,
        "glm-5.3-flash",
        output_dir=str(tmp_path),
    )

    assert result == portfolio
    assert [call[1] for call in calls] == [
        HypothesisMergeAdjudication,
        IntentPortfolio,
        PortfolioCoverageAudit,
    ]
    assert all(
        call[2]["model_override"] == "glm-5.3-flash" for call in calls
    )
    assert "member-first" in calls[1][0]
    assert "keep_separate" in calls[1][0]
    assert (
        tmp_path / "intent_hypothesis_merge_adjudication.json"
    ).exists()


def test_portfolio_generation_can_reuse_explicit_schema_valid_cache(
    monkeypatch, tmp_path
):
    portfolio = IntentPortfolio(
        obligations=[
            IntentObligation(
                id="FIX-1",
                kind="fix",
                statement="Fix it",
                authority="issue_explicit",
                strength="hard",
            )
        ],
        hypotheses=[
            IntentHypothesis(
                id="H0",
                title="literal",
                branch_assumption="none",
                covered_obligation_ids=["FIX-1"],
                hard_obligation_coverage={"FIX-1": "Implement the fix."},
            )
        ],
    )
    cache_path = tmp_path / "org__repo-1" / "intent_portfolio.json"
    cache_path.parent.mkdir()
    cache_path.write_text(portfolio.model_dump_json())
    monkeypatch.setenv("COTEST_INTENT_PORTFOLIO_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(
        RUNNER,
        "_run_structured_extraction",
        lambda *_args, **_kwargs: pytest.fail("cache should avoid extraction"),
    )

    memory = Memory(instance_id="org__repo-1", problem_statement="Fix it")

    assert RUNNER._generate_intent_portfolio(
        memory, "glm-5.3-flash"
    ) == portfolio


def test_portfolio_generation_falls_back_when_extraction_returns_none(
    monkeypatch,
):
    monkeypatch.setattr(
        RUNNER,
        "_run_structured_extraction",
        lambda *_args, **_kwargs: None,
    )
    memory = Memory(instance_id="org__repo-1", problem_statement="Fix it")

    portfolio = RUNNER._generate_intent_portfolio(
        memory, "glm-5.3-flash"
    )

    assert portfolio.hypotheses[0].covered_obligation_ids == ["FIX-1"]
    assert portfolio.obligations[0].statement == "Fix it"


def test_portfolio_fallback_preserves_grounded_scout_scope(monkeypatch):
    monkeypatch.setattr(
        RUNNER,
        "_run_structured_extraction",
        lambda *_args, **_kwargs: None,
    )
    memory = Memory(
        instance_id="org__repo-1",
        problem_statement="Repair update().",
        repair_intent_contract={
            "intent_ledger": [
                {
                    "supported_delta": (
                        "Both public update overloads must maintain parent "
                        "pointers under diagnostics."
                    ),
                    "evidence_ids": ["RUN-1", "RUN-2"],
                }
            ],
            "target_behavior": ["No parent invariant assertion fires."],
            "preserve_behavior": ["Value semantics remain unchanged."],
            "local_behavior_witnesses": [
                "Both overloads abort on the same parent invariant."
            ],
        },
    )

    portfolio = RUNNER._generate_intent_portfolio(
        memory, "glm-5.3-flash"
    )

    hypothesis = portfolio.hypotheses[0]
    assert "Both public update overloads" in portfolio.obligations[0].statement
    assert "No parent invariant assertion fires." in hypothesis.expected_behavior
    assert hypothesis.preservation_behavior == [
        "Value semantics remain unchanged."
    ]
    assert "RUN-2" in hypothesis.evidence_basis


def test_portfolio_generation_keeps_initial_when_coverage_audit_fails(
    monkeypatch, tmp_path
):
    portfolio = IntentPortfolio(
        obligations=[
            IntentObligation(
                id="FIX-1",
                kind="fix",
                statement="Fix it",
                authority="issue_explicit",
                strength="hard",
            )
        ],
        hypotheses=[
            IntentHypothesis(
                id="H0",
                title="literal",
                branch_assumption="none",
                covered_obligation_ids=["FIX-1"],
                hard_obligation_coverage={
                    "FIX-1": "Implement and verify the requested behavior."
                },
            )
        ],
    )
    calls = 0

    def fake_extract(_prompt, _schema, **_kwargs):
        nonlocal calls
        calls += 1
        return portfolio if calls == 1 else None

    monkeypatch.setattr(RUNNER, "_run_structured_extraction", fake_extract)
    memory = Memory(instance_id="org__repo-1", problem_statement="Fix it")

    result = RUNNER._generate_intent_portfolio(
        memory, "glm-5.3-flash", output_dir=str(tmp_path)
    )

    assert result == portfolio
    assert calls == 2
    assert (tmp_path / "intent_portfolio_coverage.json").exists()
    assert not (tmp_path / "intent_portfolio_coverage_audit.json").exists()


def test_portfolio_generation_revises_once_on_coverage_gap(
    monkeypatch, tmp_path
):
    initial = IntentPortfolio(
        obligations=[
            IntentObligation(
                id="FIX-1",
                kind="fix",
                statement="Fix both public entry points",
                authority="issue_explicit",
                strength="hard",
            )
        ],
        hypotheses=[
            IntentHypothesis(
                id="H0",
                title="first entry only",
                branch_assumption="none",
                repair_scope=["Repair the first entry point."],
                covered_obligation_ids=["FIX-1"],
                hard_obligation_coverage={
                    "FIX-1": "Repair the first entry point."
                },
            )
        ],
    )
    audit = PortfolioCoverageAudit(
        decision="REVISE",
        uncovered_issue_obligations=[
            "H0 claims FIX-1 but omits the second Issue-named entry point."
        ],
        reason="The claimed hard coverage is incomplete.",
    )
    revised = initial.model_copy(deep=True)
    revised.hypotheses[0].title = "both public entries"
    revised.hypotheses[0].repair_scope.append(
        "Repair the second entry point."
    )
    revised.hypotheses[0].hard_obligation_coverage["FIX-1"] = (
        "Repair both Issue-named public entry points."
    )
    calls = []

    def fake_extract(prompt, schema, **kwargs):
        calls.append((prompt, schema, kwargs))
        if schema is PortfolioCoverageAudit:
            return audit
        return initial if len(calls) == 1 else revised

    monkeypatch.setattr(RUNNER, "_run_structured_extraction", fake_extract)
    memory = Memory(
        instance_id="org__repo-1",
        problem_statement="Fix both public entry points",
    )

    result = RUNNER._generate_intent_portfolio(
        memory, "glm-5.3-flash", output_dir=str(tmp_path)
    )

    assert result == revised
    assert [call[1] for call in calls] == [
        IntentPortfolio,
        PortfolioCoverageAudit,
        IntentPortfolio,
    ]
    assert "one bounded revision" in calls[2][0]
    assert "omits the second" in calls[2][0]
    assert json.loads(
        (tmp_path / "intent_portfolio_coverage_audit.json").read_text()
    )["decision"] == "REVISE"


def test_per_branch_compliance_audit_persists_validated_decision(
    monkeypatch, tmp_path
):
    decision = BranchComplianceDecision(
        status="noncompliant",
        observed_behavior="The competing path wins.",
        reason="The assigned shared discriminator contradicts H1.",
    )
    captured = {}

    def fake_extract(prompt, schema, **kwargs):
        captured["prompt"] = prompt
        captured["schema"] = schema
        captured["kwargs"] = kwargs
        return decision

    monkeypatch.setattr(RUNNER, "_run_structured_extraction", fake_extract)
    candidate = SimpleNamespace(
        patch="diff --git a/src/fix.cc b/src/fix.cc\n"
    )
    context = FileContext(repo=InMemRepository(), change_seq=1, verification={})

    result = RUNNER._audit_portfolio_branch_compliance(
        hypothesis_context={
            "branch": {"id": "H1"},
            "assigned_dimensions": [
                {"assigned_outcome": "new path wins"}
            ],
        },
        candidate=candidate,
        repair_probes=[],
        file_context=context,
        model_name="glm-5.3-flash",
        output_dir=str(tmp_path),
        label="initial",
    )

    assert result is decision
    assert captured["schema"] is BranchComplianceDecision
    assert "new path wins" in captured["prompt"]
    assert json.loads(
        (tmp_path / "branch_compliance_initial.json").read_text()
    )["status"] == "noncompliant"


def test_missing_discriminator_is_selector_evidence_not_candidate_gate(tmp_path):
    candidate = SimpleNamespace(
        patch="diff --git a/src/fix.cc b/src/fix.cc\n",
        kept_paths=["src/fix.cc"],
        excluded_paths={},
    )

    RUNNER._save_portfolio_candidate_status(
        str(tmp_path),
        finish_succeeded=True,
        basic_tests_passed=False,
        candidate=candidate,
        branch_compliance="compliant",
        discriminator_required=True,
        discriminator_verified=False,
    )

    payload = json.loads(
        (tmp_path / "portfolio_candidate_status.json").read_text()
    )
    assert payload["admitted"] is True
    assert payload["discriminator_required"] is True
    assert payload["discriminator_role"] == "selector_evidence"


def test_branch_correction_feedback_names_actionable_hypothesis_mismatch():
    feedback = RUNNER._render_portfolio_branch_correction_feedback(
        continuation_round=1,
        continuation_rounds=2,
        hypothesis_context={"branch": {"id": "H1"}},
        compliance=BranchComplianceDecision(
            status="noncompliant",
            observed_behavior="The old path still wins.",
            reason="DIM-1 selects the competing outcome.",
            actionable_correction=True,
            violated_clause_ids=["DIM-1"],
            suggested_correction="Change only the upstream selection point.",
        ),
    )

    assert "Continuation 1 of 2" in feedback
    assert "DIM-1" in feedback
    assert "Change only the upstream selection point" in feedback
    assert "true Issue intent" in feedback
    assert "call Finish again" in feedback


def test_branch_correction_requires_actionable_hypothesis_mismatch():
    inconclusive = BranchComplianceDecision(
        status="inconclusive",
        observed_behavior="No final-revision discriminator was executed.",
        reason="Evidence is missing.",
    )
    non_actionable = BranchComplianceDecision(
        status="noncompliant",
        observed_behavior="The competing behavior is present.",
        reason="The assigned hypothesis does not match the patch.",
    )
    actionable = non_actionable.model_copy(
        update={
            "actionable_correction": True,
            "violated_clause_ids": ["DIM-1"],
            "suggested_correction": "Change the selection point.",
        }
    )

    assert not RUNNER._portfolio_branch_requires_correction(
        compliance=inconclusive,
    )
    assert not RUNNER._portfolio_branch_requires_correction(
        compliance=non_actionable,
    )
    assert RUNNER._portfolio_branch_requires_correction(
        compliance=actionable,
    )


def test_intent_scout_checkpoint_reuses_only_matching_issue_evidence(
    monkeypatch, tmp_path
):
    instance_id = "org__repo-1"
    instance = {
        "problem_statement": "Fix public_call",
        "base_commit": "abc123",
    }
    evidence = IssueEvidenceFrame(symbols=["public_call"])
    cache_dir = tmp_path / "cache"
    cache_instance_dir = cache_dir / instance_id
    cache_instance_dir.mkdir(parents=True)
    cached = RUNNER.IntentDialogueResult(issue_evidence=evidence)
    (cache_instance_dir / "intent_grounding_result.json").write_text(
        cached.model_dump_json(), encoding="utf-8"
    )
    (cache_instance_dir / "intent_scout_checkpoint.json").write_text(
        json.dumps(
            RUNNER._intent_scout_checkpoint_manifest(instance_id, instance)
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("COTEST_INTENT_SCOUT_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(
        RUNNER,
        "get_benchmark_instance",
        lambda _instance_id: instance,
    )
    monkeypatch.setattr(
        RUNNER,
        "_extract_issue_evidence",
        lambda _issue: pytest.fail("valid cache must skip Issue extraction"),
    )
    monkeypatch.setattr(
        RUNNER,
        "create_repository",
        lambda _instance: pytest.fail("cache hit must skip repository setup"),
    )

    memory = RUNNER.run_test_derived_intent_step(
        instance_name=instance_id,
        model_name="glm-5.3-flash",
        docker_env=SimpleNamespace(),
        output_dir=str(tmp_path / "output"),
    )

    assert memory.instance_id == instance_id
    assert memory.issue_evidence == evidence.model_dump()
    assert (tmp_path / "output" / "intent_grounding_result.json").exists()


def test_intent_scout_checkpoint_manifest_tracks_semantic_policy():
    manifest = RUNNER._intent_scout_checkpoint_manifest(
        "org__repo-1",
        {"problem_statement": "Fix public_call", "base_commit": "abc123"},
    )

    assert manifest["policy_version"] == RUNNER._INTENT_SCOUT_POLICY_VERSION


def test_poc_mode_keeps_intent_pipeline_and_disables_anchor_retrieval(monkeypatch, tmp_path):
    monkeypatch.setenv("COTEST_ENABLE_INTENT_GROUNDING", "1")
    monkeypatch.setenv("COTEST_INTENT_DIALOGUE_MODE", "poc")
    monkeypatch.setenv("COTEST_INTENT_TEST_ANCHOR_RETRIEVAL", "1")
    captured = {}
    monkeypatch.setattr(
        RUNNER.ContainerRepositoryInvestigator, "create",
        lambda **kwargs: captured.setdefault("investigator", kwargs),
    )

    class FakeDialogue:
        def __init__(self, **kwargs):
            captured["dialogue"] = kwargs

        def run(self, **_kwargs):
            raise RuntimeError("stop before model execution")

    monkeypatch.setattr(RUNNER, "IntentDialogue", FakeDialogue)
    assert RUNNER._test_derived_intent_enabled()
    with pytest.raises(RuntimeError, match="stop before model execution"):
        RUNNER._run_intent_grounding(
            instance={"problem_statement": "Fix public_call"},
            repository=InMemRepository(), docker_env=SimpleNamespace(),
            code_index=None, issue_evidence=IssueEvidenceFrame(),
            reproduction_context={}, output_dir=str(tmp_path),
            fallback_model="glm-5.3-flash",
        )

    assert captured["investigator"]["mode"] == "poc"
    assert captured["dialogue"]["enable_test_anchor_retrieval"] is False
    assert RUNNER._intent_scout_checkpoint_manifest(
        "org__repo-1", {"problem_statement": "Fix public_call"}
    )["investigation_mode"] == "poc"


def test_scout_reasoning_override_isolated_from_repair_and_cache(monkeypatch):
    monkeypatch.setenv("COTEST_GLM_REASONING_EFFORT", "low")
    monkeypatch.delenv("COTEST_INTENT_SCOUT_GLM_REASONING_EFFORT", raising=False)
    monkeypatch.delenv(
        "COTEST_INTENT_EVIDENCE_REVIEW_GLM_REASONING_EFFORT",
        raising=False,
    )
    instance = {"problem_statement": "Fix public_call", "base_commit": "abc123"}
    before = RUNNER._intent_scout_checkpoint_manifest("org__repo-1", instance)
    monkeypatch.setenv("COTEST_INTENT_SCOUT_GLM_REASONING_EFFORT", "max")
    monkeypatch.setenv(
        "COTEST_INTENT_EVIDENCE_REVIEW_GLM_REASONING_EFFORT", "xhigh"
    )
    scout = RUNNER._model_kwargs(
        "glm-5.3-flash", "COTEST_INTENT_DIALOGUE_GENERATION_MAX_TOKENS",
        glm_reasoning_effort=RUNNER._intent_scout_reasoning_effort(),
    )
    repair = RUNNER._model_kwargs("glm-5.3-flash", "COTEST_REPAIR_MAX_TOKENS")
    after = RUNNER._intent_scout_checkpoint_manifest("org__repo-1", instance)
    assert scout["extra_body"]["reasoning_effort"] == "max"
    assert repair["extra_body"]["reasoning_effort"] == "low"
    assert before != after
    assert before["scout_reasoning_effort"] == "high"
    assert before["evidence_review_reasoning_effort"] == "high"
    assert after["scout_reasoning_effort"] == "max"
    assert after["evidence_review_reasoning_effort"] == "xhigh"


def test_intent_scout_cache_rejects_non_replayable_frozen_probe():
    cached = RUNNER.IntentDialogueResult(
        intent_contract=RUNNER.RepairIntentContract(
            acceptance_checks=[
                {
                    "description": "Compile the exact Issue scenario.",
                    "command": (
                        "c++ probe.cc /tmp/build/libtarget.a "
                        "-o /tmp/probe && /tmp/probe"
                    ),
                    "files": {"probe.cc": "int main() {}"},
                    "source_evidence_id": "RUN-1",
                    "baseline_exit_code": 1,
                    "kind": "f2p",
                    "authority": "normative",
                }
            ]
        )
    )

    rejection = RUNNER._cached_intent_scout_probe_rejection(cached)

    assert rejection.startswith("ACCEPT-1:")
    assert "cannot depend on /tmp" in rejection


def test_repair_verification_status_marks_unresolved_review_unverified(tmp_path):
    RUNNER._save_repair_verification_status(
        str(tmp_path),
        verified_finish=True,
        basic_tests_passed=True,
        patch_review_status="rejected_refine_no_persisted_change",
    )

    payload = json.loads(
        (tmp_path / "repair_verification_status.json").read_text()
    )
    assert payload["submission_status"] == "unverified"


def test_repair_verification_status_marks_unresolved_intent_audit_unverified(
    tmp_path,
):
    RUNNER._save_repair_verification_status(
        str(tmp_path),
        verified_finish=True,
        basic_tests_passed=True,
        patch_review_status="disabled",
        patch_intent_audit_status="underfit_persists_after_refine",
    )

    payload = json.loads(
        (tmp_path / "repair_verification_status.json").read_text()
    )
    assert payload["patch_intent_audit_status"] == (
        "underfit_persists_after_refine"
    )
    assert payload["submission_status"] == "unverified"


def test_admission_unavailable_marks_intent_audit_unresolved():
    assert RUNNER._patch_intent_audit_unresolved(
        "inconclusive_admission_unavailable"
    )


def test_audit_infrastructure_failure_marks_intent_audit_unresolved():
    assert RUNNER._patch_intent_audit_unresolved(
        "inconclusive_infrastructure"
    )


def test_patch_intent_audit_oracle_authority_excludes_open_decisions():
    memory = SimpleNamespace(
        repair_summary="free-form investigator statement",
        repair_intent_contract={
            "target_behavior": ["The public parse call accepts the new input."],
            "open_decisions": ["Whether the behavior should be opt-in."],
        },
    )

    review_context, normative = RUNNER._patch_intent_audit_contracts(memory)

    assert "Whether the behavior should be opt-in" in review_context
    assert "The public parse call accepts the new input" in normative
    assert "Whether the behavior should be opt-in" not in normative
    assert "free-form investigator statement" not in normative


def test_trajectory_audit_leads_are_opt_in(monkeypatch):
    monkeypatch.delenv("COTEST_ENABLE_TRAJECTORY_AUDIT_LEADS", raising=False)
    assert not RUNNER._trajectory_audit_leads_enabled()
    monkeypatch.setenv("COTEST_ENABLE_TRAJECTORY_AUDIT_LEADS", "true")
    assert RUNNER._trajectory_audit_leads_enabled()


def test_legacy_reproducer_remains_enabled_by_default(monkeypatch):
    monkeypatch.delenv("COTEST_ENABLE_LEGACY_REPRODUCER", raising=False)
    assert RUNNER._legacy_reproducer_enabled()


def test_legacy_reproducer_can_be_disabled(monkeypatch):
    monkeypatch.setenv("COTEST_ENABLE_LEGACY_REPRODUCER", "false")
    assert not RUNNER._legacy_reproducer_enabled()


def test_issue_only_memory_preserves_issue_without_reproduction_evidence():
    memory = RUNNER._build_issue_only_memory(
        "org__repo-17",
        {"problem_statement": "The public API chooses the wrong overload."},
    )

    assert memory.instance_id == "org__repo-17"
    assert memory.problem_statement == (
        "The public API chooses the wrong overload."
    )
    assert memory.reproduction_result == ""
    assert memory.expected_reproduction_result == ""
    assert memory.test_files == []


def test_trajectory_audit_lead_handoff_is_non_authoritative_json():
    lead = AuditLead(
        evidence_quotes=[
            EvidenceQuote(evidence_id="EV-7", quote="The focused probe failed.")
        ],
        observed_fact="The selected probe did not establish the target behavior.",
        repair_assumption="The Repair may assume the helper provides ADL.",
        unverified_public_behavior="ADL selection on the changed public path.",
        suggested_investigation="Inspect the helper and run an overlap probe.",
    )

    rendered = RUNNER._render_trajectory_audit_leads([lead])

    payload = json.loads(rendered)
    assert payload[0]["evidence_quotes"][0]["evidence_id"] == "EV-7"
    assert payload[0]["unverified_public_behavior"].startswith("ADL")


def test_patch_intent_refinement_feedback_contains_complete_frozen_pair():
    probe = SimpleNamespace(
        oracle_reference="The public call must succeed.",
        oracle="The public call must succeed for the overlap input.",
        oracle_mode="runtime_behavior",
        purpose="distinguish public dispatch",
        distinguishing_input="overlap_input",
        candidate_prediction="returns an error",
        alternative_prediction="returns the formatted value",
        control_compile_exit_code=0,
        control_run_exit_code=0,
        target_compile_exit_code=0,
        target_run_exit_code=3,
        fixture_content="struct overlap_input {};",
        control_content="int main() { return public_call(control_input); }",
        target_content="int main() { return public_call(overlap_input{}); }",
        control_compile_command="c++ control.cpp -o control",
        control_run_command="./control",
        target_compile_command="c++ target.cpp -o target",
        target_run_command="./target",
        output="control_run=0 target_run=3",
    )

    feedback = RUNNER._render_patch_intent_counterexample_feedback(
        SimpleNamespace(counterexample=probe)
    )

    assert "Shared fixture:" in feedback
    assert "Passing control:" in feedback
    assert "Failing target:" in feedback
    assert probe.control_content in feedback
    assert probe.target_content in feedback
    assert probe.output in feedback
    assert "Oracle mode: runtime_behavior" in feedback
    assert "target_run=3" in feedback


def test_patch_intent_feedback_omits_refuted_mechanism_predictions():
    probe = SimpleNamespace(
        oracle_reference="The public call must succeed.",
        oracle="The public call must succeed for the overlap input.",
        oracle_mode="runtime_behavior",
        purpose="Only the observed behavioral underfit is supported.",
        distinguishing_input="overlap_input",
        candidate_prediction="incorrect candidate mechanism prediction",
        alternative_prediction="unconfirmed alternative mechanism prediction",
        control_compile_exit_code=0,
        control_run_exit_code=0,
        target_compile_exit_code=0,
        target_run_exit_code=3,
        fixture_content="struct overlap_input {};",
        control_content="int main() { return public_call(control_input); }",
        target_content="int main() { return public_call(overlap_input{}); }",
        control_compile_command="c++ control.cpp -o control",
        control_run_command="./control",
        target_compile_command="c++ target.cpp -o target",
        target_run_command="./target",
        output="control_run=0 target_run=3",
    )
    audit_result = SimpleNamespace(
        counterexample=probe,
        semantic_admission=SimpleNamespace(
            candidate_prediction_matches_observation=False
        ),
    )

    feedback = RUNNER._render_patch_intent_counterexample_feedback(audit_result)

    assert "Mechanism attribution: unconfirmed" in feedback
    assert probe.candidate_prediction not in feedback
    assert probe.alternative_prediction not in feedback
    assert probe.target_content in feedback
    assert probe.output in feedback


def test_unresolved_discriminator_feedback_is_labeled_non_oracle():
    probe = SimpleNamespace(
        oracle_reference="The Issue names this dispatch dimension.",
        oracle="The Issue names this dispatch dimension.",
        oracle_mode="runtime_behavior",
        purpose="observe competing public dispatch outcomes",
        distinguishing_input="overlap_input",
        candidate_prediction="selects member path",
        alternative_prediction="selects customization path",
        control_compile_exit_code=0,
        control_run_exit_code=0,
        target_compile_exit_code=0,
        target_run_exit_code=0,
        fixture_content="struct overlap_input {};",
        control_content="int main() { return 0; }",
        target_content="int main() { return observe(overlap_input{}); }",
        control_compile_command="c++ control.cpp -o control",
        control_run_command="./control",
        target_compile_command="c++ target.cpp -o target",
        target_run_command="./target",
        output="control_run=0 target_run=0",
    )
    audit_result = SimpleNamespace(
        unresolved_discriminator=probe,
        semantic_admission=SimpleNamespace(
            candidate_prediction_matches_observation=True
        ),
    )

    feedback = RUNNER._render_patch_intent_unresolved_discriminator_feedback(
        audit_result
    )

    assert "<unresolved_mechanism_discriminator>" in feedback
    assert probe.candidate_prediction in feedback
    assert probe.alternative_prediction in feedback


def test_patch_equivalence_ignores_only_edge_whitespace():
    patch = "diff --git a/a b/a\n+fixed\n"

    assert RUNNER._patches_equivalent(patch, "\n" + patch + "\n")
    assert not RUNNER._patches_equivalent(patch, patch + "+mirror\n")


def test_structured_extraction_retry_reports_schema_failure(monkeypatch):
    prompts = []
    token_budgets = []

    class FakeLLM:
        def __init__(self, **kwargs):
            token_budgets.append(kwargs["max_tokens"])

        def generate(self, prompt):
            prompts.append(prompt)
            message = '{}' if len(prompts) == 1 else '{"value": "fixed"}'
            return SimpleNamespace(message=message)

    monkeypatch.setattr(RUNNER, "LiteLLM", FakeLLM)

    result = RUNNER._run_structured_extraction(
        "Return the object.",
        RequiredStructuredField,
        max_tokens=4000,
    )

    assert result == RequiredStructuredField(value="fixed")
    assert len(prompts) == 2
    assert "Validation failure:" in prompts[1]
    assert "value" in prompts[1]
    assert "Field required" in prompts[1]
    assert token_budgets == [4000, 8000]


def test_intent_structured_generation_budget_is_separate_from_report_fields(
    monkeypatch,
):
    token_budgets = []

    class FakeLLM:
        def __init__(self, **kwargs):
            token_budgets.append(kwargs["max_tokens"])

        def generate(self, prompt):
            return SimpleNamespace(message='{"value": "ok"}')

    monkeypatch.setenv("COTEST_EXTRACTION_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv(
        "COTEST_INTENT_STRUCTURED_GENERATION_MAX_TOKENS",
        "64000",
    )
    monkeypatch.setenv("COTEST_INTENT_DIALOGUE_MAX_TOKENS", "600")
    monkeypatch.setattr(RUNNER, "LiteLLM", FakeLLM)

    result = RUNNER._run_intent_dialogue_extraction(
        "Return the object.",
        RequiredStructuredField,
    )

    assert result == RequiredStructuredField(value="ok")
    assert token_budgets == [64000]


def test_intent_evidence_reviewer_uses_dedicated_high_reasoning(monkeypatch):
    calls = []

    class FakeLLM:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def generate(self, prompt):
            return SimpleNamespace(message='{"value": "ok"}')

    monkeypatch.setenv("COTEST_EXTRACTION_MODEL", "glm-5.3-flash")
    monkeypatch.setenv("COTEST_GLM_REASONING_EFFORT", "low")
    monkeypatch.delenv(
        "COTEST_INTENT_EVIDENCE_REVIEW_GLM_REASONING_EFFORT",
        raising=False,
    )
    monkeypatch.setattr(RUNNER, "LiteLLM", FakeLLM)

    result = RUNNER._run_intent_evidence_review(
        "Return the object.",
        RequiredStructuredField,
    )

    assert result == RequiredStructuredField(value="ok")
    assert calls[0]["extra_body"]["reasoning_effort"] == "high"


def test_structured_extraction_resolves_project_model_alias(monkeypatch):
    models = []

    class FakeLLM:
        def __init__(self, **kwargs):
            models.append(kwargs["model"])

        def generate(self, prompt):
            return SimpleNamespace(message='{"value": "ok"}')

    monkeypatch.setenv("COTEST_EXTRACTION_MODEL", "deepseek-v4-flash")
    monkeypatch.setattr(RUNNER, "LiteLLM", FakeLLM)

    result = RUNNER._run_structured_extraction(
        "Return the object.",
        RequiredStructuredField,
    )

    assert result == RequiredStructuredField(value="ok")
    assert models == ["deepseek/deepseek-v4-flash"]


def test_structured_extraction_preserves_volcengine_provider_config(monkeypatch):
    calls = []

    class FakeLLM:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def generate(self, prompt):
            return SimpleNamespace(message='{"value": "ok"}')

    monkeypatch.setenv(
        "COTEST_EXTRACTION_MODEL",
        "volcengine-deepseek-v4-flash",
    )
    monkeypatch.setenv("ARK_API_KEY", "test-ark-key")
    monkeypatch.setattr(RUNNER, "LiteLLM", FakeLLM)

    result = RUNNER._run_structured_extraction(
        "Return the object.",
        RequiredStructuredField,
    )

    assert result == RequiredStructuredField(value="ok")
    assert calls[0]["model"] == "openai/deepseek-v4-flash-260425"
    assert calls[0]["api_base"] == "https://ark.cn-beijing.volces.com/api/v3"
    assert calls[0]["api_key"] == "test-ark-key"
    assert calls[0]["extra_body"] == {"thinking": {"type": "enabled"}}


def test_structured_extraction_preserves_glm_provider_config(monkeypatch):
    calls = []

    class FakeLLM:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def generate(self, prompt):
            return SimpleNamespace(message='{"value": "ok"}')

    monkeypatch.setenv("COTEST_EXTRACTION_MODEL", "glm-5.3-flash")
    monkeypatch.setenv("ZAI_API_KEY", "test-zai-key")
    monkeypatch.setattr(RUNNER, "LiteLLM", FakeLLM)

    result = RUNNER._run_structured_extraction(
        "Return the object.",
        RequiredStructuredField,
    )

    assert result == RequiredStructuredField(value="ok")
    assert calls[0]["model"] == "openai/glm-5.3-flash"
    assert calls[0]["api_base"] == "https://api.z.ai/api/paas/v4"
    assert calls[0]["api_key"] == "test-zai-key"
    assert calls[0]["max_tokens"] == 32768
    assert calls[0]["extra_body"] == {
        "thinking": {"type": "enabled", "clear_thinking": False},
        "reasoning_effort": "high",
    }


def test_glm_structured_extraction_supports_bounded_reasoning_override(
    monkeypatch,
):
    calls = []

    class FakeLLM:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def generate(self, prompt):
            return SimpleNamespace(message='{"value": "ok"}')

    monkeypatch.setenv("ZAI_API_KEY", "test-zai-key")
    monkeypatch.setattr(RUNNER, "LiteLLM", FakeLLM)

    result = RUNNER._run_structured_extraction(
        "Return the object.",
        RequiredStructuredField,
        model_override="glm-5.3-flash",
        max_tokens=4096,
        glm_reasoning_effort="low",
    )

    assert result == RequiredStructuredField(value="ok")
    assert calls[0]["max_tokens"] == 4096
    assert calls[0]["extra_body"] == {
        "thinking": {"type": "enabled", "clear_thinking": False},
        "reasoning_effort": "low",
    }


def test_glm_structured_extraction_compacts_bounded_transport_fields(monkeypatch):
    prompts = []
    calls = []

    class FakeLLM:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def generate(self, prompt):
            prompts.append(prompt)
            if "Single-Turn Structured Payload Compactor" in prompt:
                return SimpleNamespace(message=(
                    '{"value":"concise",'
                    '"evidence_ids":["SEARCH-1 SEARCH-2","SEARCH-3"]}'
                ))
            return SimpleNamespace(message=(
                '{"value":"This explanation is much too long",'
                '"evidence_ids":["SEARCH-1","SEARCH-2","SEARCH-3"],'
                '"unlisted":"redundant metadata"}'
            ))

    monkeypatch.setenv("COTEST_EXTRACTION_MODEL", "glm-5.3-flash")
    monkeypatch.setattr(RUNNER, "LiteLLM", FakeLLM)

    result = RUNNER._run_structured_extraction(
        "Return the object.",
        BoundedStructuredField,
    )

    assert result.value == "concise"
    assert result.evidence_ids == ["SEARCH-1 SEARCH-2", "SEARCH-3"]
    assert len(prompts) == 2
    assert "compact transport artifact" in prompts[0]
    assert "Single-Turn Structured Payload Compactor" in prompts[1]
    assert calls[1]["max_retries"] == 0
    assert calls[1]["extra_body"] == {
        "thinking": {"type": "enabled", "clear_thinking": False},
        "reasoning_effort": "low",
    }


def test_glm_runner_uses_native_tools_and_large_generation_budget(monkeypatch):
    monkeypatch.delenv("COTEST_REPAIR_MAX_TOKENS", raising=False)
    monkeypatch.delenv("COTEST_GLM_MAX_TOKENS", raising=False)
    monkeypatch.delenv("COTEST_GLM_REASONING_EFFORT", raising=False)

    assert RUNNER._completion_protocol_kwargs("glm-5.3-flash") == {
        "response_format": RUNNER.LLMResponseFormat.TOOLS,
        "message_history_type": RUNNER.MessageHistoryType.MESSAGES,
    }
    assert RUNNER._model_kwargs(
        "glm-5.3-flash", "COTEST_REPAIR_MAX_TOKENS"
    ) == {
        "temperature": 1.0,
        "max_tokens": 32768,
        "extra_body": {
            "thinking": {"type": "enabled", "clear_thinking": False},
            "reasoning_effort": "high",
        },
    }

    monkeypatch.setenv("COTEST_GLM_THINKING", "disabled")
    assert RUNNER._model_kwargs(
        "glm-5.3-flash", "COTEST_REPAIR_MAX_TOKENS"
    )["extra_body"] == {
        "thinking": {"type": "disabled", "clear_thinking": False},
        "reasoning_effort": "high",
    }

    monkeypatch.setenv("COTEST_GLM_REASONING_EFFORT", "max")
    assert RUNNER._model_kwargs(
        "glm-5.3-flash", "COTEST_REPAIR_MAX_TOKENS"
    )["extra_body"]["reasoning_effort"] == "max"


def test_qwen_runner_uses_native_tools_and_preserved_thinking(monkeypatch):
    monkeypatch.delenv("COTEST_REPAIR_MAX_TOKENS", raising=False)
    monkeypatch.delenv("COTEST_QWEN_THINKING", raising=False)
    monkeypatch.delenv("COTEST_QWEN_REASONING_EFFORT", raising=False)

    assert RUNNER._completion_protocol_kwargs("qwen3.8-flash") == {
        "response_format": RUNNER.LLMResponseFormat.TOOLS,
        "message_history_type": RUNNER.MessageHistoryType.MESSAGES,
    }
    assert RUNNER._model_kwargs(
        "qwen3.8-flash", "COTEST_REPAIR_MAX_TOKENS"
    ) == {
        "temperature": 1.0,
        "max_tokens": 131072,
        "extra_body": {
            "enable_thinking": True,
            "preserve_thinking": True,
            "reasoning_effort": "xhigh",
        },
    }

    monkeypatch.setenv("COTEST_QWEN_REASONING_EFFORT", "medium")
    assert RUNNER._model_kwargs(
        "qwen3.8-flash", "COTEST_REPAIR_MAX_TOKENS"
    )["extra_body"]["reasoning_effort"] == "medium"

    monkeypatch.setenv("COTEST_QWEN_THINKING", "false")
    assert RUNNER._model_kwargs(
        "qwen3.8-flash", "COTEST_REPAIR_MAX_TOKENS"
    )["extra_body"] == {
        "enable_thinking": False,
        "preserve_thinking": False,
    }


def test_relative_artifact_paths_are_grouped_under_repository_roots():
    assert RUNNER.results_path("results_experiment") == os.path.join(
        RUNNER.PROJECT_ROOT,
        "results",
        "results_experiment",
    )
    assert RUNNER.results_path("results/nested/experiment") == os.path.join(
        RUNNER.PROJECT_ROOT,
        "results",
        "nested",
        "experiment",
    )
    assert RUNNER.submission_path("submission_run.json") == os.path.join(
        RUNNER.PROJECT_ROOT,
        "submission",
        "submission_run.json",
    )


def test_absolute_artifact_paths_remain_explicit(tmp_path):
    external = tmp_path / "submission.json"

    assert RUNNER.submission_path(str(external)) == str(external)


def test_relative_artifact_path_cannot_escape_group_root():
    with pytest.raises(ValueError, match="escapes results"):
        RUNNER.results_path("../outside")


def test_issue_evidence_retries_when_first_pass_has_no_grounded_target(
    monkeypatch,
):
    issue = """Colored print improvements
There are two things lacking which would be great:
* colored print to stderr as well as stdout
* text style, i.e. print bold
"""
    first = IssueEvidenceFrame.model_validate({
        "claims": [{
            "id": "OBS-1",
            "kind": "observed_behavior",
            "text": "Colored output is incomplete.",
            "issue_span": "not an exact issue span",
        }]
    })
    recovered = IssueEvidenceFrame.model_validate({
        "claims": [
            {
                "id": "EXP-1",
                "kind": "expected_behavior",
                "text": "Colored print should support stderr as well as stdout.",
                "issue_span": "* colored print to stderr as well as stdout",
            },
            {
                "id": "EXP-2",
                "kind": "expected_behavior",
                "text": "Text style should support bold output.",
                "issue_span": "* text style, i.e. print bold",
            },
        ]
    })
    prompts = []

    def extract(prompt, schema, **kwargs):
        prompts.append(prompt)
        assert schema is IssueEvidenceFrame
        return first if len(prompts) == 1 else recovered

    monkeypatch.setattr(RUNNER, "_run_structured_extraction", extract)

    frame = RUNNER._extract_issue_evidence(issue)

    assert [claim.text for claim in frame.claims] == [
        "Colored print should support stderr as well as stdout.",
        "Text style should support bold output.",
    ]
    assert len(prompts) == 2
    assert "requested-feature bullet lists" in prompts[1]


def test_registers_structured_intent_contract_for_repair_verification():
    file_context = SimpleNamespace(verification={})
    memory = SimpleNamespace(
        repair_intent_bundle={},
        repair_intent_contract={
            "target_behavior": ["Target works."],
            "preserve_behavior": ["Existing path remains valid."],
            "acceptance_checks": [
                {
                    "description": "Exercise target and contrast.",
                    "command": (
                        "c++ intent_probe.cpp -o /tmp/intent_probe && "
                        "/tmp/intent_probe"
                    ),
                    "files": {
                        "intent_probe.cpp": "int main() { return 0; }"
                    },
                    "source_evidence_id": "RUN-1",
                    "baseline_exit_code": 1,
                }
            ],
            "open_decisions": [],
        }
    )

    RUNNER._register_intent_verification_contract(file_context, memory)

    assert file_context.verification["intent_contract"] == {
        "checks": {
            "TARGET-1": "Target works.",
            "PRESERVE-1": "Existing path remains valid.",
            "ACCEPT-1": "Exercise target and contrast.",
        },
        "target_ids": ["TARGET-1"],
        "preserve_ids": ["PRESERVE-1"],
        "acceptance_ids": ["ACCEPT-1"],
        "diagnostic_ids": [],
        "probes": {
            "ACCEPT-1": {
                "description": "Exercise target and contrast.",
                "command": (
                    "c++ intent_probe.cpp -o /tmp/intent_probe && "
                    "/tmp/intent_probe"
                ),
                "files": {
                    "intent_probe.cpp": "int main() { return 0; }"
                },
                "source_evidence_id": "RUN-1",
                "baseline_exit_code": 1,
                "kind": "diagnostic",
                "authority": "normative",
            }
        },
    }


def test_reproduction_output_compaction_preserves_both_ends():
    output = "FIRST_ERROR\n" + "x" * 10000 + "\nTARGET_CANDIDATE"

    compact = RUNNER._compact_reproduction_output(output, limit=100)

    assert compact.startswith("FIRST_ERROR")
    assert compact.endswith("TARGET_CANDIDATE")
    assert "compacted" in compact


def test_contrastive_intent_prefers_witness_and_enables_reviewer(
    monkeypatch,
):
    monkeypatch.setenv("COTEST_ENABLE_INTENT_GROUNDING", "true")
    monkeypatch.setenv("COTEST_INTENT_DIALOGUE_MODE", "contrastive")
    monkeypatch.delenv("COTEST_ENABLE_PATCH_REVIEWER", raising=False)

    assert RUNNER._test_derived_intent_enabled() is True
    assert RUNNER._patch_reviewer_enabled() is True


def test_reviewer_can_be_disabled_for_test_derived_ablation(monkeypatch):
    monkeypatch.setenv("COTEST_ENABLE_INTENT_GROUNDING", "true")
    monkeypatch.setenv("COTEST_INTENT_DIALOGUE_MODE", "contrastive")
    monkeypatch.setenv("COTEST_ENABLE_PATCH_REVIEWER", "false")

    assert RUNNER._test_derived_intent_enabled() is True
    assert RUNNER._patch_reviewer_enabled() is False


def test_legacy_intent_mode_keeps_reproducer_and_reviewer_default_off(
    monkeypatch,
):
    monkeypatch.setenv("COTEST_ENABLE_INTENT_GROUNDING", "true")
    monkeypatch.setenv("COTEST_INTENT_DIALOGUE_MODE", "legacy")
    monkeypatch.delenv("COTEST_ENABLE_PATCH_REVIEWER", raising=False)

    assert RUNNER._test_derived_intent_enabled() is False
    assert RUNNER._patch_reviewer_enabled() is False


def test_test_derived_intent_keeps_reproducer_fallback_opt_in(monkeypatch):
    monkeypatch.delenv("COTEST_ENABLE_REPRODUCER_FALLBACK", raising=False)
    assert RUNNER._reproducer_fallback_enabled() is False

    monkeypatch.setenv("COTEST_ENABLE_REPRODUCER_FALLBACK", "true")
    assert RUNNER._reproducer_fallback_enabled() is True


def test_intent_evidence_budget_prefers_current_name(monkeypatch):
    monkeypatch.setenv("COTEST_INTENT_EVIDENCE_STEPS", "37")
    monkeypatch.setenv("COTEST_INTENT_DIALOGUE_TOTAL_STEPS", "60")

    assert RUNNER._intent_evidence_steps() == 37


def test_repository_only_intent_memory_has_no_contrastive_witness():
    repository_only = RUNNER.Memory(
        instance_id="repo__name-1",
        problem_statement="Clarify the public behavior.",
        repair_summary="<repair_intent_contract>repo evidence</repair_intent_contract>",
        repair_intent_contract={"target_behavior": ["Apply documented behavior."]},
    )
    witnessed = repository_only.model_copy(
        update={"reproduction_result": "TARGET-1 fails on the baseline."}
    )

    assert not RUNNER._has_valid_contrastive_witness(repository_only)
    assert RUNNER._has_valid_contrastive_witness(witnessed)


def test_post_adjudication_rejection_does_not_count_as_valid_witness():
    decision = SimpleNamespace(witness_valid=True)
    accepted = SimpleNamespace(
        contrastive_decision=decision,
        pair_status="validated",
        stop_reason="contrastive_witness_only",
    )
    invalid_claim_binding = SimpleNamespace(
        contrastive_decision=decision,
        pair_status="validated",
        stop_reason="contrastive_invalid_claim_coverage",
    )

    assert RUNNER._dialogue_accepted_contrastive_witness(accepted)
    assert not RUNNER._dialogue_accepted_contrastive_witness(
        invalid_claim_binding
    )


def test_reproducer_fallback_retains_intent_contract_and_failure_evidence():
    intent_memory = RUNNER.Memory(
        instance_id="repo__name-1",
        problem_statement="Clarify the public behavior.",
        repair_summary="<repair_intent_contract>repo evidence</repair_intent_contract>",
        issue_evidence={"claims": [{"id": "EXP-1"}]},
        repair_intent_contract={"target_behavior": ["Apply documented behavior."]},
    )
    reproduction_memory = RUNNER.Memory(
        instance_id="repo__name-1",
        problem_statement="Clarify the public behavior.",
        reproduction_summary="Reproduced the reported failure.",
        test_command="c++ repro.cpp && ./a.out",
        test_files=[{"path": "repro.cpp", "content": "int main(){}"}],
        reproduction_result="baseline failure",
    )

    merged = RUNNER._merge_intent_and_reproduction_memory(
        intent_memory,
        reproduction_memory,
    )

    assert merged.reproduction_result == "baseline failure"
    assert merged.test_command == reproduction_memory.test_command
    assert merged.repair_summary == intent_memory.repair_summary
    assert merged.repair_intent_contract == intent_memory.repair_intent_contract
    assert merged.issue_evidence == intent_memory.issue_evidence


def test_successful_test_enhancement_does_not_replace_repair_patch():
    repair_patch = "diff --git a/src/fix.cpp b/src/fix.cpp\n"
    diagnostic_patch = "diff --git a/tests/repro.cpp b/tests/repro.cpp\n"

    assert RUNNER._submission_patch_after_test_enhancement(
        repair_patch, diagnostic_patch
    ) == repair_patch


def test_submission_patch_excludes_dedicated_reproduction_artifacts():
    class FakeContext:
        def __init__(self):
            self.paths = [
                "src/fix.cpp",
                "include/CMakeLists.txt",
                "tests/test_fix.cpp",
                "home/json/repro_bug.cpp",
                "compile_probe.cpp",
                "scratch.cpp",
            ]
            self.verification = {"ephemeral_paths": ["scratch.cpp"]}

        @property
        def files(self):
            return [
                SimpleNamespace(
                    file_path=path,
                    is_new=path in {
                        "compile_probe.cpp",
                        "scratch.cpp",
                        "home/json/repro_bug.cpp",
                    },
                )
                for path in self.paths
            ]

        def clone(self):
            clone = FakeContext()
            clone.paths = list(self.paths)
            clone.verification = dict(self.verification)
            return clone

        def get_patched_files(self):
            return self.paths

        def remove_file(self, path):
            self.paths.remove(path)

        def generate_git_patch(self):
            return "\n".join(self.paths)

        def get_file(self, path):
            return SimpleNamespace(content="int value;\n")

    assert RUNNER._generate_submission_patch(FakeContext()) == (
        "src/fix.cpp\ninclude/CMakeLists.txt"
    )


def test_reproduction_artifact_cleanup_removes_only_dedicated_files(tmp_path):
    repro_file = tmp_path / "repro_bug.cpp"
    source_file = tmp_path / "src" / "fix.cpp"
    repro_file.write_text("repro")
    source_file.parent.mkdir()
    source_file.write_text("source")

    RUNNER._remove_host_reproduction_artifacts(
        str(tmp_path), ["repro_bug.cpp", "src/fix.cpp", "../outside.cpp"]
    )

    assert not repro_file.exists()
    assert source_file.exists()


@pytest.mark.parametrize("final_revise", [False, True])
@pytest.mark.parametrize("verified_evidence", [False, True])
@pytest.mark.parametrize("provisional_without_ranking", [False, True])
def test_expanded_pool_reaudits_revision_before_scheduling(monkeypatch, tmp_path, final_revise, verified_evidence, provisional_without_ranking):
    from INTENT2FIX.intent_portfolio import IntentHypothesisPool, HypothesisPoolAudit

    def pool(last_id):
        return IntentHypothesisPool(
            obligations=[IntentObligation(
                id="FIX-1", kind="fix", statement="Format the value",
                authority="issue_explicit", strength="hard",
            )],
            hypotheses=[IntentHypothesis(
                id=h, title=h, branch_assumption=f"policy {h}",
                covered_obligation_ids=["FIX-1"],
                hard_obligation_coverage={"FIX-1": f"Format using policy {h}."},
            ) for h in ["H0", "H1", "H2", last_id]],
        )

    def audit(last_id, revise=False):
        return HypothesisPoolAudit(
            decision="REVISE" if revise else "ACCEPT", reason="Review complete",
            selected_hypothesis_ids=[last_id, "H0", "H1"],
            selection_reasons={h: "Additional supported behavior" for h in [last_id, "H0", "H1"]},
            unselected_reasons={"H2": "Budget deferred"},
            hypothesis_reviews={h: {
                "premise_status": "supported", "assigned_outcome_status": "consistent",
            "premise_check": "Input and receiver match the supported path.",
                "hard_contract_check": "Target handled.",
                "assigned_outcome_check": "Policy agrees with outcomes.",
                "implementation_feasibility_check": "Supported implementation.",
                "conflicts": [],
            } for h in ["H0", "H1", "H2", last_id]},
            discriminator_reviews={},
            joint_policy_losses={"H2": "Complete H2 policy is not executed."},
            causal_coverage_gaps=["H3 misses a supported path"] if revise else [],
        )

    initial_audit = audit("H3", True)
    if provisional_without_ranking:
        initial_audit.selected_hypothesis_ids = []
        initial_audit.selection_reasons = {}
        initial_audit.unselected_reasons = {}
        initial_audit.joint_policy_losses = {}
    responses = iter([pool("H3"), initial_audit, pool("H4"), audit("H4", final_revise)])
    calls = []
    audit_tools = SimpleNamespace(evidence=[{
        'id': 'AUDIT-RUN-1', 'observation': 'CONCRETE_COMPILER_COUNTEREXAMPLE',
    }]) if verified_evidence else None

    def extract(prompt, schema, **kwargs):
        calls.append(schema)
        if schema is HypothesisPoolAudit:
            assert kwargs['audit_tools'] is audit_tools
        if verified_evidence and len(calls) >= 3:
            assert 'CONCRETE_COMPILER_COUNTEREXAMPLE' in prompt
        response = next(responses)
        if schema is HypothesisPoolAudit:
            response.validate_pool(kwargs['audit_pool'])
        return response

    monkeypatch.setenv("COTEST_PORTFOLIO_RERANK", "1")
    monkeypatch.delenv("COTEST_INTENT_PORTFOLIO_CACHE_DIR", raising=False)
    monkeypatch.setattr(RUNNER, "_run_structured_extraction", extract)
    memory = Memory(
        instance_id="org__repo-1", problem_statement="Format the value",
        repair_intent_bundle={"open_decisions": ["Scout's tentative implementation choice"]},
    )
    result = RUNNER._generate_intent_portfolio(
        memory,
        "glm-5.3-flash", str(tmp_path), audit_tools=audit_tools,
    )
    assert calls == [IntentHypothesisPool, HypothesisPoolAudit,
                     IntentHypothesisPool, HypothesisPoolAudit]
    assert [h.id for h in result.repair_hypotheses] == ["H4", "H0", "H1"]
    saved = json.loads((tmp_path / "intent_hypothesis_pool.json").read_text())
    assert len(saved["hypotheses"]) == 4
    assert saved["selected_hypothesis_ids"] == []
    assert json.loads((tmp_path / "intent_hypothesis_rerank.json").read_text())[
        "selected_hypothesis_ids"
    ] == ["H4", "H0", "H1"]
    execution = json.loads(
        (tmp_path / "intent_hypothesis_execution.json").read_text()
    )
    assert execution["audit_decision"] == (
        "REVISE" if final_revise else "ACCEPT"
    )
    assert execution["selected_hypothesis_ids"] == ["H4", "H0", "H1"]


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('fails', [False, True])
def test_hypothesis_verification_container_lifetime(monkeypatch, tmp_path, enabled, fails):
    events = []
    tools = object()

    class Container:
        def __init__(self, **kwargs):
            assert kwargs['benchmark_base_commit'] == 'base'
            events.append('create')

        def __enter__(self):
            events.append('enter')
            return self

        def __exit__(self, *args):
            events.append('close')

    def generate(memory, model, output_dir, audit_tools=None):
        assert audit_tools is (tools if enabled else None)
        events.append('generate')
        if fails:
            raise ValueError('audit failed')
        return 'portfolio'

    monkeypatch.setenv('COTEST_PORTFOLIO_RERANK', '1')
    monkeypatch.setenv('COTEST_PORTFOLIO_VERIFY_PREMISES', '1' if enabled else '0')
    monkeypatch.setattr(RUNNER, 'DockerEnv', Container)
    monkeypatch.setattr(RUNNER, 'HypothesisEvidenceTools', lambda *args, **kwargs: tools)
    monkeypatch.setattr(RUNNER, '_generate_intent_portfolio', generate)
    args = dict(memory=Memory(instance_id='org__repo-1', problem_statement='Issue'),
                model_name='glm-5.3-flash', output_dir=str(tmp_path),
                image_name='public-image', work_dir='/repo', base_commit='base',
                container_suffix='test')
    if fails:
        with pytest.raises(ValueError, match='audit failed'):
            RUNNER._generate_intent_portfolio_with_verification(**args)
    else:
        assert RUNNER._generate_intent_portfolio_with_verification(**args) == 'portfolio'
    assert events == (['create', 'enter', 'generate', 'close'] if enabled else ['generate'])


def test_expanded_pool_schedules_first_three_when_audit_is_unavailable(monkeypatch, tmp_path):
    from INTENT2FIX.intent_portfolio import IntentHypothesisPool

    pool = IntentHypothesisPool(
        obligations=[IntentObligation(
            id="FIX-1", kind="fix", statement="Fix the target",
            authority="issue_explicit", strength="hard",
        )],
        hypotheses=[IntentHypothesis(
            id=f"H{i}", title=f"policy {i}", branch_assumption=f"policy {i}",
            covered_obligation_ids=["FIX-1"],
            hard_obligation_coverage={"FIX-1": f"Repair using policy {i}."},
        ) for i in range(4)],
    )
    responses = iter([pool, None])
    monkeypatch.setenv("COTEST_PORTFOLIO_RERANK", "1")
    monkeypatch.delenv("COTEST_INTENT_PORTFOLIO_CACHE_DIR", raising=False)
    monkeypatch.setattr(RUNNER, "_run_structured_extraction", lambda *_a, **_k: next(responses))
    result = RUNNER._generate_intent_portfolio(
        Memory(instance_id="org__repo-1", problem_statement="Fix the target"),
        "glm-5.3-flash", str(tmp_path),
    )
    assert result.selected_hypothesis_ids == ["H0", "H1", "H2"]
    assert (tmp_path / "intent_hypothesis_pool.json").exists()
    assert not (tmp_path / "intent_hypothesis_rerank.json").exists()
    execution = json.loads(
        (tmp_path / "intent_hypothesis_execution.json").read_text()
    )
    assert execution["audit_available"] is False
    assert execution["selection_mode"] == "generated_pool_fallback"


@pytest.mark.parametrize("model,json_mode", [
    ("openai/glm-5.3-flash", False),
    ("other/model", True),
])
def test_structured_extraction_preserves_identifiers_without_glm_json_mode(monkeypatch, model, json_mode):
    captured = []

    class Reply(BaseModel):
        symbols: list[str]

    symbols = ["from_json", "nlohmann::json", "simdjson.h", "raw_json_string"]

    class FakeLLM:
        def __init__(self, **kwargs):
            captured.append(kwargs)

        def generate(self, prompt):
            return SimpleNamespace(message="```json\n" + json.dumps({"symbols": symbols}) + "\n```")

    monkeypatch.setattr(RUNNER, "get_model_config", lambda _m: {"model": model})
    monkeypatch.setattr(RUNNER, "LiteLLM", FakeLLM)
    result = RUNNER._run_structured_extraction("Return exact identifiers as JSON.", Reply, model_override=model)
    assert result.symbols == symbols
    assert ("response_format" in captured[0]) == json_mode


def test_expanded_pool_cache_requires_current_complete_audit(monkeypatch, tmp_path):
    from INTENT2FIX.intent_portfolio import IntentHypothesisPool
    cached = IntentHypothesisPool(
        obligations=[IntentObligation(id='PRESERVE-1', kind='preserve',
                                     statement='Keep supported inputs',
                                     authority='inferred', strength='soft')],
        hypotheses=[IntentHypothesis(id='H0', title='Policy', branch_assumption='Policy')],
        selected_hypothesis_ids=['H0'],
    )
    case_dir = tmp_path / 'org__repo-1'
    case_dir.mkdir()
    (case_dir / 'intent_portfolio.json').write_text(cached.model_dump_json())
    # Old selections contain no per-branch review and cannot bypass the new gate.
    (case_dir / 'intent_hypothesis_rerank.json').write_text(json.dumps({
        'decision': 'ACCEPT', 'reason': 'Legacy audit',
        'selected_hypothesis_ids': ['H0'], 'selection_reasons': {'H0': 'Chosen'},
        'unselected_reasons': {},
    }))
    monkeypatch.setenv('COTEST_PORTFOLIO_RERANK', '1')
    monkeypatch.setenv('COTEST_INTENT_PORTFOLIO_CACHE_DIR', str(tmp_path))
    def regenerate(*args, **kwargs):
        raise RuntimeError('new generation reached')
    monkeypatch.setattr(RUNNER, '_run_structured_extraction', regenerate)
    with pytest.raises(RuntimeError, match='new generation reached'):
        RUNNER._generate_intent_portfolio(
            Memory(instance_id='org__repo-1', problem_statement='Target'),
            'glm-5.3-flash', str(tmp_path / 'new'),
        )


@pytest.mark.parametrize('transport', ['tool', 'retry_missing_tool', 'text', 'invalid_text'])
def test_glm_pool_audit_validates_native_and_text_with_same_schema(monkeypatch, transport):
    from types import SimpleNamespace
    from INTENT2FIX.intent_portfolio import HypothesisPoolAudit
    payload = {
        'decision': 'ACCEPT', 'reason': 'Review complete',
        'hypothesis_reviews': {'H0': {
            'premise_status': 'supported', 'assigned_outcome_status': 'consistent',
            'premise_check': 'from_json input path checked',
            'hard_contract_check': 'Target satisfied',
            'assigned_outcome_check': 'Consistent',
            'implementation_feasibility_check': 'Valid signature', 'conflicts': [],
        }},
        'discriminator_reviews': {}, 'selected_hypothesis_ids': ['H0'],
        'selection_reasons': {'H0': 'Only complete policy'},
        'unselected_reasons': {}, 'joint_policy_losses': {},
    }
    calls = []
    def complete(**kwargs):
        calls.append(kwargs)
        tool = SimpleNamespace(function=SimpleNamespace(
            name='SubmitHypothesisAudit', arguments=json.dumps(payload)))
        tools = [] if transport in {'text', 'invalid_text'} or (
            transport == 'retry_missing_tool' and len(calls) == 1) else [tool]
        content = json.dumps(payload) if transport == 'text' else 'Not a valid audit'
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=content, tool_calls=tools))])
    monkeypatch.setattr('litellm.completion', complete)
    monkeypatch.setattr(RUNNER, '_load_env', lambda: None)
    from INTENT2FIX.intent_portfolio import IntentHypothesisPool
    pool = IntentHypothesisPool(
        obligations=[IntentObligation(id='PRESERVE-1', kind='preserve',
                                     statement='Keep supported inputs', authority='inferred', strength='soft')],
        hypotheses=[IntentHypothesis(id='H0', title='Policy', branch_assumption='Policy')],
    )
    result = RUNNER._run_structured_extraction(
        'Audit the pool', HypothesisPoolAudit, model_override='glm-5.3-flash',
        max_tokens=16384, glm_reasoning_effort='high', audit_pool=pool,
    )
    if transport == 'invalid_text':
        assert result is None
        assert len(calls) == 2
        return
    assert result.hypothesis_reviews['H0'].premise_check.startswith('from_json')
    assert len(calls) == (2 if transport == 'retry_missing_tool' else 1)
    assert all('response_format' not in c for c in calls)
    parameters = calls[0]['tools'][0]['function']['parameters']
    assert parameters['properties']['hypothesis_reviews']['required'] == ['H0']
    assert parameters['properties']['hypothesis_reviews']['additionalProperties'] is False
    assert parameters['properties']['discriminator_reviews']['properties'] == {}
    assert 'discriminator_reviews' not in parameters['required']
    assert 'joint_policy_losses' not in parameters['required']
    assert calls[0]['extra_body']['reasoning_effort'] == 'high'


def test_native_audit_schema_requires_all_nonempty_review_tables(monkeypatch):
    from types import SimpleNamespace
    from INTENT2FIX.intent_portfolio import HypothesisPoolAudit, IntentHypothesisPool

    pool = IntentHypothesisPool.model_validate({
        'obligations': [{'id': 'PRESERVE-1', 'kind': 'preserve',
                         'statement': 'Keep supported inputs',
                         'authority': 'inferred', 'strength': 'soft'}],
        'hypotheses': [{'id': f'H{i}', 'title': f'Policy {i}',
                        'branch_assumption': f'Policy {i}'} for i in range(4)],
        'unresolved_dimensions': [{
            'id': 'DIM-1', 'statement': 'Key policy', 'impact': 'high',
            'disposition': 'branched', 'hypothesis_ids': ['H0', 'H1'],
            'shared_discriminator': 'Read the same escaped key',
            'outcomes': [{'hypothesis_id': f'H{i}', 'predicted_outcome': f'outcome {i}'}
                         for i in range(2)], 'reason': 'Unresolved key semantics',
        }],
    })
    calls = []
    def complete(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content='No audit', tool_calls=[]))])
    monkeypatch.setattr('litellm.completion', complete)
    monkeypatch.setattr(RUNNER, '_load_env', lambda: None)
    assert RUNNER._run_structured_extraction(
        'Audit the pool', HypothesisPoolAudit, model_override='glm-5.3-flash',
        audit_pool=pool,
    ) is None
    for call in calls:
        schema = call['tools'][0]['function']['parameters']
        assert {'hypothesis_reviews', 'discriminator_reviews'} <= set(schema['required'])
        assert 'joint_policy_losses' not in schema['required']  # Optional on REVISE.
        accept = schema['allOf'][0]
        assert accept['if']['properties']['decision'] == {'const': 'ACCEPT'}
        assert {'selected_hypothesis_ids', 'selection_reasons', 'joint_policy_losses'} <= set(accept['then']['required'])
        assert accept['then']['properties']['selected_hypothesis_ids']['minItems'] == 1
        assert schema['properties']['discriminator_reviews']['required'] == ['DIM-1']
        assert schema['properties']['hypothesis_reviews']['required'] == ['H0', 'H1', 'H2', 'H3']


@pytest.mark.parametrize('defect', ['selection_reason', 'premise_check'])
@pytest.mark.parametrize('with_evidence_tools', [False, True])
def test_audit_retry_reuses_failed_payload_and_validates_assignment(monkeypatch, defect, with_evidence_tools):
    from copy import deepcopy
    from INTENT2FIX.intent_portfolio import HypothesisPoolAudit, IntentHypothesisPool

    pool = IntentHypothesisPool(
        obligations=[IntentObligation(id='PRESERVE-1', kind='preserve',
                                     statement='Preserve supported inputs',
                                     authority='inferred', strength='soft')],
        hypotheses=[IntentHypothesis(
            id=f'H{i}', title=f'Policy {i}', branch_assumption=f'Policy {i}') for i in range(3)])
    valid = {
        'decision': 'ACCEPT', 'reason': 'All policies reviewed',
        'hypothesis_reviews': {h.id: {
            'premise_status': 'supported', 'assigned_outcome_status': 'consistent',
            'premise_check': f'{h.id} actual input gate evaluated',
            'hard_contract_check': 'ORIGINAL_REVIEW_EVIDENCE_DO_NOT_DISCARD',
            'assigned_outcome_check': 'Input produces the promised output',
            'implementation_feasibility_check': 'Public route remains callable',
        } for h in pool.hypotheses},
        'selected_hypothesis_ids': ['H0', 'H1', 'H2'],
        'selection_reasons': {h.id: 'Supported interpretation' for h in pool.hypotheses},
    }
    invalid = deepcopy(valid)
    if defect == 'selection_reason':
        invalid['selection_reasons'].pop('H1')
    else:
        invalid['hypothesis_reviews']['H1'].pop('premise_check')
    prompts = []
    def submit(prompt):
        prompts.append(prompt)
        return json.dumps(invalid if len(prompts) == 1 else valid)
    def complete(**kwargs):
        payload = submit(kwargs['messages'][0]['content'])
        call = SimpleNamespace(function=SimpleNamespace(name='SubmitHypothesisAudit', arguments=payload))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='', tool_calls=[call]))])
    audit_tools = SimpleNamespace(complete=lambda kwargs, prompt, schema: submit(prompt)) if with_evidence_tools else None
    monkeypatch.setattr('litellm.completion', complete)
    monkeypatch.setattr(RUNNER, '_load_env', lambda: None)
    result = RUNNER._run_structured_extraction(
        'Review the unchanged pool', HypothesisPoolAudit, model_override='glm-5.3-flash',
        audit_pool=pool, audit_tools=audit_tools)
    assert len(prompts) == 2
    assert 'ORIGINAL_REVIEW_EVIDENCE_DO_NOT_DISCARD' in prompts[1]
    assert json.dumps(invalid) in prompts[1]
    assert ('premise_check' if defect == 'premise_check' else 'select') in prompts[1]
    assert result.selected_hypothesis_ids == ['H0', 'H1', 'H2']

@pytest.mark.parametrize('fails', [False, True])
def test_source_expansion_reaches_generator_without_mutating_scout(monkeypatch, tmp_path, fails):
    from INTENT2FIX import hypothesis_source
    events = []
    memory = Memory(instance_id='org__repo-1', problem_statement='Issue',
                    repair_intent_bundle={'cited_repository_evidence': []})

    class Container:
        def __init__(self, **kwargs):
            assert kwargs['benchmark_base_commit'] == 'base'
        def __enter__(self):
            return self
        def __exit__(self, *args):
            events.append('closed')

    def expand(*args):
        return {'cited_repository_evidence': [{'id': 'SOURCE-1', 'observation': 'actual trait'}]}

    def generate(enriched, model, output_dir, audit_tools=None):
        assert enriched is not memory
        assert audit_tools is None
        assert enriched.repair_intent_bundle['cited_repository_evidence'][0]['id'] == 'SOURCE-1'
        if fails:
            raise ValueError('generation failed')
        return 'pool'

    monkeypatch.setenv('COTEST_PORTFOLIO_RERANK', '1')
    monkeypatch.setenv('COTEST_PORTFOLIO_EXPAND_SOURCE', '1')
    monkeypatch.setenv('COTEST_PORTFOLIO_VERIFY_PREMISES', '0')
    monkeypatch.setattr(RUNNER, 'DockerEnv', Container)
    monkeypatch.setattr(hypothesis_source, 'expand_source_evidence', expand)
    monkeypatch.setattr(RUNNER, '_generate_intent_portfolio', generate)
    args = dict(memory=memory, model_name='glm-5.3-flash', output_dir=str(tmp_path),
                image_name='public-image', work_dir='/repo', base_commit='base', container_suffix='test')
    if fails:
        with pytest.raises(ValueError, match='generation failed'):
            RUNNER._generate_intent_portfolio_with_verification(**args)
    else:
        assert RUNNER._generate_intent_portfolio_with_verification(**args) == 'pool'
    assert events == ['closed']
    assert memory.repair_intent_bundle == {'cited_repository_evidence': []}

@pytest.mark.parametrize('verification_id', ['', 'RUN-1'])
def test_memory_hands_off_exact_scout_probe_version(verification_id):
    from INTENT2FIX.intent_dialogue.schema import InvestigationInputPathCheck
    path_check = InvestigationInputPathCheck(
        actual_input='original input', consumer_input='original input',
        condition_evidence_id='RUN-1', condition_quote='old', evaluation='observed',
        matches_input='yes', alternative_path='none', verification_run_id=verification_id,
    )
    report = SimpleNamespace(
        input_path_check=path_check, repository_evidence_ids=['RUN-1', 'RUN-2'],
        preserve_evidence_ids=[], repository_facts=[], candidate_boundaries=[],
        open_decisions=[], contrastive_witness=None,
    )
    actions = [dict(action='RunInvestigationCommand', evidence_id=f'RUN-{i}',
                    command=f'command-{i}', frozen_probe_files={'probe.cpp': f'version-{i}'})
               for i in [1, 2]]
    result = SimpleNamespace(
        render_for_repair=lambda: 'summary', intent_contract=RUNNER.RepairIntentContract(),
        investigation=SimpleNamespace(report=report, actions=actions, repository_evidence=[
            SimpleNamespace(id=f'RUN-{i}', path='probe.cpp', locator=f'command-{i}', observation='old')
            for i in [1, 2]]),
        contrastive_decision=None, pair_status='', stop_reason='',
    )
    memory = RUNNER._memory_from_intent_dialogue_result(
        instance_name='org__repo-1', problem_statement='Issue',
        issue_evidence=IssueEvidenceFrame(), result=result,
    )
    chosen = 1 if verification_id else 2
    assert memory.repair_intent_bundle['scout_probe_snapshot'] == {
        'evidence_id': f'RUN-{chosen}', 'command': f'command-{chosen}',
        'files': {'probe.cpp': f'version-{chosen}'},
    }
    assert actions[0]['frozen_probe_files']['probe.cpp'] == 'version-1'


def test_scout_relation_source_reaches_policy_generation_before_uncited_noise(monkeypatch):
    from INTENT2FIX.intent_dialogue.schema import (
        EvidenceSufficiencyFollowup,
        EvidenceSufficiencyReview,
        InvestigationPath,
    )
    from INTENT2FIX.intent_portfolio import build_hypothesis_prompt
    monkeypatch.setenv('COTEST_PORTFOLIO_RERANK', '0')
    path = InvestigationPath(question='Where is state lost?', finding='Wrapper forwards to the consumer.',
        status='supported', evidence_ids=['SEARCH-20'], source_evidence_id='SEARCH-20',
        source_quote='return consume(state);')
    report = SimpleNamespace(input_path_check=None, investigation_paths=[path],
        repository_evidence_ids=[f'SEARCH-{i}' for i in range(1, 20)],
        preserve_evidence_ids=[], repository_facts=['UNVERIFIED_SCOUT_SUMMARY'],
        candidate_boundaries=[], open_decisions=[], contrastive_witness=None)
    review = EvidenceSufficiencyReview(
        decision="continue",
        reason="The upstream state producer remains unknown.",
        preserved_alternatives=["Producer-side repair", "Consumer-side repair"],
        next_investigation=EvidenceSufficiencyFollowup(
            question="Which producer first creates the invalid state?",
            why_material="The answer changes the causal repair locus.",
            known_anchor="SEARCH-20 consumer call",
        ),
    )
    result = SimpleNamespace(render_for_repair=lambda: 'UNVERIFIED_SCOUT_SUMMARY',
        intent_contract=RUNNER.RepairIntentContract(), investigation=SimpleNamespace(report=report,
        repository_evidence=[SimpleNamespace(id=f'SEARCH-{i}', path='impl.h', locator='read',
            observation='return consume(state);' if i == 20 else 'unrelated source') for i in range(1, 21)]),
        contrastive_decision=None, pair_status='', stop_reason='', hypothesis_ready=False,
        evidence_sufficiency_review=review,
        evidence_sufficiency_followup_performed=True)
    memory = RUNNER._memory_from_intent_dialogue_result(instance_name='org__repo-1',
        problem_statement='Fix public state handling.', issue_evidence=IssueEvidenceFrame(), result=result)
    bundle = memory.repair_intent_bundle
    assert bundle['investigation_paths'] == [{**path.model_dump(), 'omitted_evidence_ids': []}]
    assert bundle['cited_repository_evidence'][0]['id'] == 'SEARCH-20'
    assert bundle['hypothesis_ready'] is False
    assert bundle['evidence_sufficiency_review'] == review.model_dump()
    assert bundle['evidence_sufficiency_followup_performed'] is True
    prompt = build_hypothesis_prompt(issue=memory.problem_statement, issue_evidence={},
        repair_intent_bundle=bundle, repair_intent_contract={}, scout_summary='UNVERIFIED_SCOUT_SUMMARY',
        expanded_pool=True, policy_prompt=True)
    assert path.question in prompt and path.source_quote in prompt
    assert review.next_investigation.question in prompt
    assert "explicitly incomplete" in prompt
    assert "Readiness is diagnostic only, never a stop instruction" in prompt
    assert "proceed to Repair" in prompt
    assert 'UNVERIFIED_SCOUT_SUMMARY' not in prompt


def test_scout_handoff_merges_live_and_refined_path_history():
    from INTENT2FIX.intent_dialogue.schema import (
        InvestigationPath,
        InvestigationPathHop,
    )
    early = InvestigationPath(
        path_id="PATH-1", role="state_producer", question="Where is state changed?",
        finding="The graph trace is awaiting source verification.", status="unresolved",
        evidence_ids=["SEARCH-1"], missing_evidence="Read the producer body.",
        causal_hops=[InvestigationPathHop(
            cause="public entry",
            relation="creates",
            effect="state",
            evidence_ids=["SEARCH-1"],
        )],
        root_cause_candidate="The producer initializes the wrong state.",
        scope_consequences=["The wrapper forwards the produced state."],
    )
    late = InvestigationPath(
        path_id="PATH-2", role="consumer", question="Where is state checked?",
        finding="The consumer checks the state.", status="supported",
        evidence_ids=["SEARCH-2"], source_evidence_id="SEARCH-2",
        source_quote="check(state);",
    )
    report = SimpleNamespace(
        input_path_check=None, investigation_paths=[late],
        repository_evidence_ids=["SEARCH-1", "SEARCH-2"], preserve_evidence_ids=[],
        repository_facts=[], candidate_boundaries=[], open_decisions=[],
        contrastive_witness=None,
    )
    live = {**early.model_dump(), "checkpoint_decision": "traced"}
    refined_early = InvestigationPath(
        path_id="PATH-1",
        question="Where is state changed?",
        finding="The producer source was also inspected.",
        status="unresolved",
        evidence_ids=["SEARCH-3"],
        missing_evidence="Read the producer body.",
    )
    result = SimpleNamespace(
        render_for_repair=lambda: "summary", intent_contract=RUNNER.RepairIntentContract(),
        investigation=SimpleNamespace(
            report=report,
            actions=[
                {"action": "RepositoryGraphQuery", "live_path": live},
                {"action": "SubmitInvestigationReport", "report": {
                    "investigation_paths": [refined_early.model_dump()] }},
            ],
            repository_evidence=[
                SimpleNamespace(id="SEARCH-1", path="producer.cc", locator="read",
                                observation="produce(state);"),
                SimpleNamespace(id="SEARCH-2", path="consumer.cc", locator="read",
                                observation="check(state);"),
                SimpleNamespace(id="SEARCH-3", path="producer.cc", locator="read",
                                observation="state = next_state;"),
            ],
        ),
        contrastive_decision=None, pair_status="", stop_reason="",
    )
    memory = RUNNER._memory_from_intent_dialogue_result(
        instance_name="org__repo-1", problem_statement="Fix state handling.",
        issue_evidence=IssueEvidenceFrame(), result=result,
    )
    paths = memory.repair_intent_bundle["investigation_paths"]
    assert [path["path_id"] for path in paths] == ["PATH-1", "PATH-2"]
    assert paths[0]["evidence_ids"] == ["SEARCH-1", "SEARCH-3"]
    assert paths[0]["role"] == "state_producer"
    assert paths[0]["causal_hops"] == [
        item.model_dump() for item in early.causal_hops
    ]
    assert paths[0]["root_cause_candidate"] == early.root_cause_candidate
    assert paths[0]["scope_consequences"] == early.scope_consequences
    assert all(path["omitted_evidence_ids"] == [] for path in paths)
    workspace = memory.repair_intent_bundle["causal_workspace_markdown"]
    assert "- Evidence: SEARCH-1, SEARCH-3" in workspace
    assert "The producer initializes the wrong state." in workspace


@pytest.mark.parametrize("mode", ["artifact", "compile", "compile_persists", "legacy", "positive_probe"])
def test_repair_loop_mechanically_continues_without_semantic_audit_permission(mode, monkeypatch, tmp_path):
    """Exercise run_repair_step, including its early returns and the shared continuation budget."""
    from unittest.mock import MagicMock
    from INTENT2FIX.repository.file import FileRepository
    from INTENT2FIX.actions.create_file import CreateFile, CreateFileArgs

    repository = FileRepository(repo_path=str(tmp_path))
    (tmp_path / "base.txt").write_text("base\n")
    monkeypatch.setenv("INDEX_STORE_DIR", str(tmp_path))
    monkeypatch.setenv("COTEST_PORTFOLIO_CONTINUATION_ROUNDS", "2")
    monkeypatch.setattr(RUNNER, "_load_env", lambda: None)
    monkeypatch.setattr(RUNNER, "get_benchmark_instance", lambda _: {})
    monkeypatch.setattr(RUNNER, "create_repository", lambda _: repository)
    monkeypatch.setattr(RUNNER.CodeIndex, "from_index_name", lambda **_: None)
    monkeypatch.setattr(RUNNER.CodingAgent, "create", lambda **_: SimpleNamespace(actions=[]))
    monkeypatch.setattr(RUNNER, "_repo_structure_for_model", lambda *_: "base.txt")
    monkeypatch.setattr(RUNNER, "_collect_final_patch_repair_probes", lambda **_: [])
    monkeypatch.setattr(RUNNER, "_run_structured_extraction", lambda *_args, **_kwargs: BranchComplianceDecision(
        status="inconclusive", observed_behavior="Insufficient semantic evidence.", reason="No authorized semantic correction."
    ))
    monkeypatch.setattr(RUNNER, "_model_kwargs", lambda *_: {})
    monkeypatch.setattr(RUNNER, "_completion_protocol_kwargs", lambda *_: {})
    monkeypatch.setattr(RUNNER, "_patch_reviewer_enabled", lambda: False)
    monkeypatch.setattr(RUNNER, "_patch_intent_audit_enabled", lambda: False)

    class FakeLoop:
        def __init__(self, file_context, **kwargs):
            self.context = file_context
            self.calls = 0
            self.feedback = []
            self.agent = SimpleNamespace(actions=[])
            self.root = SimpleNamespace(get_all_nodes=lambda: [None] * self.calls)

        def append_user_feedback(self, feedback):
            self.feedback.append(feedback)

        def run(self):
            self.calls += 1
            path = "include/helper.hpp"
            if self.calls == 1:
                self.context.add_file(path).apply_changes("int value;\n")
                if mode == "artifact":
                    self.context.verification["ephemeral_paths"] = [path]
            elif mode == "artifact":
                CreateFile().execute(CreateFileArgs(path=path, file_text="int value;\n",
                                                   artifact_kind="production"), self.context)
            else:
                self.context.get_file(path).apply_changes(f"int value{self.calls};\n")
                self.context.change_seq += 1
            return SimpleNamespace(file_context=self.context,
                                   is_finished=lambda: mode != "artifact" or self.calls > 1)

    if mode == "positive_probe":
        def current_probes(**kwargs):
            return [RepairProbeEvidence(
                probe_id="RP-1", source_hypothesis_id="H0", command="g++ repro.cpp",
                expected_exit_code=0, observed_exit_code=1, observed_status="failed",
                observed_output="include/helper.hpp:1: error: cannot convert value",
                production_patch_sha256=RUNNER._portfolio_patch_sha256(
                    sanitize_candidate_patch(loops[0].context).patch
                ),
            )] if loops[0].calls == 1 else []
        monkeypatch.setattr(RUNNER, "_collect_final_patch_repair_probes", current_probes)

    loops = []
    def create_loop(**kwargs):
        loop = FakeLoop(**kwargs)
        loops.append(loop)
        return loop
    monkeypatch.setattr(RUNNER.AgenticLoop, "create", create_loop)

    def regression(context, runtime):
        compile_failure = mode == "compile_persists" or (mode == "compile" and loops[0].calls == 1)
        record = {
            "change_seq": context.change_seq, "verification_view": "production",
            "verification_isolation": RUNNER.PRODUCTION_ISOLATION,
            "production_patch_sha256": hashlib.sha256((sanitize_candidate_patch(context).patch.strip() + "\n").encode()).hexdigest(),
            "passed": not compile_failure and mode != "legacy", "exit_code": 1 if compile_failure else 0,
            "log_type": "compilation_error" if compile_failure else "test_results",
            "output": "include/helper.hpp:1: error: cannot convert value" if compile_failure else "legacy assertion failed",
        }
        context.verification["production_regression"] = record
        return record
    monkeypatch.setattr(RUNNER, "_ensure_portfolio_regression", regression)
    runtime = MagicMock(spec=RUNNER.DockerEnv)
    runtime.container_name = "offline-loop-test"
    result = RUNNER.run_repair_step(
        instance_name="nlohmann__json-3463", model_name="offline", docker_env=runtime,
        memory=Memory(instance_id="nlohmann__json-3463", problem_statement="Frozen assigned behavior."),
        output_dir=str(tmp_path / "output"), candidate_mode=True,
        candidate_hypothesis_context={"branch": {"id": "H0", "expected_behavior": ["Keep assigned behavior."]}},
    )
    expected_calls = 1 if mode == "legacy" else 3 if mode == "compile_persists" else 2
    assert loops[0].calls == expected_calls
    assert result and "include/helper.hpp" in result
    for feedback in loops[0].feedback:
        assert "SAME assigned hypothesis" in feedback
        assert "No semantic contract correction is authorized" in feedback
    status = json.loads((tmp_path / "output/portfolio_candidate_status.json").read_text())
    assert status["admitted"]
    if mode == "artifact":
        raw = (tmp_path / "output/solution_raw_portfolio_candidate.patch").read_text()
        assert "int value;" in raw
        assert not status.get("artifact_errors")
    elif mode.startswith("compile"):
        assert any(v["variant_id"] == "initial" for v in status["candidate_variants"])


@pytest.mark.parametrize("fault", ["stale_hash", "working_tree", "infrastructure", "negative_probe"])
def test_nonproduction_or_unavailable_compile_evidence_does_not_trigger_correction(fault):
    context = FileContext(repo=InMemRepository())
    context.add_file("src/fix.cpp").apply_changes("int value;\n")
    candidate = sanitize_candidate_patch(context)
    record = {
        "change_seq": 0, "passed": False, "log_type": "compilation_error", "exit_code": 1,
        "production_patch_sha256": RUNNER._portfolio_patch_sha256(candidate.patch),
        "verification_view": "production",
        "verification_isolation": RUNNER.PRODUCTION_ISOLATION, "output": "src/fix.cpp:1: error: cannot convert value",
    }
    if fault == "stale_hash":
        record["production_patch_sha256"] = "earlier-patch"
    if fault == "working_tree":
        record["verification_view"] = "working_tree"
    if fault == "infrastructure":
        record["output"] = "c++: command not found"
    context.verification["single_test" if fault == "negative_probe" else "regression"] = record
    assert not RUNNER._portfolio_engineering_diagnostic(context, candidate)


def test_sanitizer_reports_excluded_literal_dependency_without_matching_unrelated_basename():
    context = FileContext(repo=InMemRepository({"api/use.hpp": "int value;\n"}))
    context.add_file("api/use.hpp").apply_changes('#include "helper.hpp"\nint value;\n')
    context.add_file("diagnostics/helper.hpp").apply_changes("// unrelated\n")
    context.verification["ephemeral_paths"] = ["diagnostics/helper.hpp"]
    assert not sanitize_candidate_patch(context).artifact_errors
    context.add_file("api/helper.hpp").apply_changes("// required\n")
    context.verification["ephemeral_paths"].append("api/helper.hpp")
    candidate = sanitize_candidate_patch(context)
    assert any("api/helper.hpp" in error for error in candidate.artifact_errors)


@pytest.mark.parametrize("source_size", [300, 2400])
def test_graph_handoff_preserves_all_entities_paths_and_failed_command(source_size, monkeypatch):
    monkeypatch.setenv("COTEST_PORTFOLIO_RERANK", "1")
    ids = [f"cpp-owner-specialization-{i}-" + "T" * 250 for i in range(5)]
    graph = [{
        "func_id": key, "fq_name": f"Widget::member{i}", "file": "src/widget.h",
        "source_status": "verified", "source_excerpt": "x" * source_size + f"\nreturn state_{i};",
        "source_truncated": False,
        "graph_paths": [{"seed_id": ids[0], "edges": [
            {"source": ids[0], "edge": "IN_CLASS", "target": "owner-" + "T" * 250},
            {"source": key, "edge": "IN_CLASS", "target": "owner-" + "T" * 250},
        ]}],
    } for i, key in enumerate(ids)]
    raw = json.dumps(graph, indent=2)
    assert len(raw) > 6000
    evidence = [SimpleNamespace(id=f"RUN-{i}", path="probe", locator="command", observation="x" * 7000)
                for i in range(1, 21)]
    evidence.extend([
        SimpleNamespace(id="GRAPH-1", path="<graph>", locator="query", observation=raw),
        SimpleNamespace(id="RUN-34", path="probe", locator="./assert_target", observation=""),
    ])
    result = SimpleNamespace(
        render_for_repair=lambda: "", intent_contract=RUNNER.RepairIntentContract(),
        investigation=SimpleNamespace(
            report=SimpleNamespace(input_path_check=None, repository_evidence_ids=[e.id for e in evidence[:10]],
                preserve_evidence_ids=[], repository_facts=[], candidate_boundaries=[], open_decisions=[], contrastive_witness=None),
            repository_evidence=evidence,
            actions=[{"action": "RunInvestigationCommand", "evidence_id": "RUN-34", "exit_code": 1}],
        ), contrastive_decision=None, pair_status="", stop_reason="",
    )
    memory = RUNNER._memory_from_intent_dialogue_result(
        instance_name="test", problem_statement="Fix state propagation", issue_evidence=IssueEvidenceFrame(), result=result)
    records = memory.repair_intent_bundle["cited_repository_evidence"]
    assert sum(len(e["observation"]) for e in records) <= 60000
    packed = json.loads(next(e["observation"] for e in records if e["id"] == "GRAPH-1"))

    def restore(value, key=""):
        if isinstance(value, dict):
            return {k: restore(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [restore(v, key) for v in value]
        if key in {"id", "func_id", "seed_id", "source", "target"}:
            return packed["entity_ids"][value]
        return value

    assert restore(packed["entities"]) == graph
    failure = next(e for e in records if e["id"] == "RUN-34")
    assert failure["exit_code"] == 1 and failure["observation"] == ""
    prompt = RUNNER.build_hypothesis_prompt(
        issue=memory.problem_statement, issue_evidence={}, repair_intent_bundle=memory.repair_intent_bundle,
        repair_intent_contract={}, scout_summary="", expanded_pool=True, policy_prompt=True)
    assert '"exit_code": 1' in prompt
    assert "return state_4;" in prompt
    assert "Constructibility does not imply implicit convertibility" in prompt


def test_handoff_uses_spare_budget_for_issue_symbol_graph(monkeypatch):
    monkeypatch.delenv("COTEST_PORTFOLIO_RERANK", raising=False)
    evidence = [
        SimpleNamespace(
            id=f"SEARCH-{index}", path="impl.h", locator="read",
            observation=f"source fact {index}",
        )
        for index in range(1, 11)
    ]
    evidence.extend([
        SimpleNamespace(
            id="GRAPH-1", path="<graph>", locator="issue bootstrap",
            observation=json.dumps([{
                "fq_name": "lib::range_formatter::parse",
                "source_status": "verified",
                "source_excerpt": "return underlying_.parse(ctx);",
            }]),
        ),
        SimpleNamespace(
            id="GRAPH-2", path="<graph>", locator="broad bootstrap",
            observation=json.dumps([{
                "fq_name": "lib::join_view::begin",
                "source_status": "verified",
                "source_excerpt": "return iterator();",
            }]),
        ),
    ])
    facts = [
        " ".join(f"SEARCH-{index}" for index in range(1, 11))
    ]
    report = SimpleNamespace(
        input_path_check=None, investigation_paths=[],
        repository_evidence_ids=[item.id for item in evidence[:10]],
        preserve_evidence_ids=[], repository_facts=facts,
        candidate_boundaries=[], open_decisions=[], contrastive_witness=None,
    )
    result = SimpleNamespace(
        render_for_repair=lambda: "", intent_contract=RUNNER.RepairIntentContract(),
        investigation=SimpleNamespace(
            report=report, repository_evidence=evidence,
            actions=[
                {"action": "RepositoryGraphQuery", "evidence_id": "GRAPH-1", "exit_code": 0},
                {"action": "RepositoryGraphQuery", "evidence_id": "GRAPH-2", "exit_code": 0},
            ],
        ),
        contrastive_decision=None, pair_status="", stop_reason="",
    )
    memory = RUNNER._memory_from_intent_dialogue_result(
        instance_name="test", problem_statement="Fix range_formatter::parse.",
        issue_evidence=IssueEvidenceFrame(symbols=["range_formatter::parse"]),
        result=result,
    )
    bundle = memory.repair_intent_bundle
    selected = [item["id"] for item in bundle["cited_repository_evidence"]]
    assert selected[:10] == [f"SEARCH-{index}" for index in range(1, 11)]
    assert selected[10:] == ["GRAPH-1"]
    assert bundle["supplemental_graph_evidence_ids"] == ["GRAPH-1"]
    assert "range_formatter::parse" in bundle["cited_repository_evidence"][-1]["observation"]


def test_graph_ablation_scout_ignores_cache_and_never_loads_index(monkeypatch, tmp_path):
    instance_id = "org__repo-1"
    instance = {"problem_statement": "Fix public_call", "base_commit": "abc123"}
    cache_dir = tmp_path / "cache" / instance_id
    cache_dir.mkdir(parents=True)
    (cache_dir / "intent_grounding_result.json").write_text(
        RUNNER.IntentDialogueResult(issue_evidence=IssueEvidenceFrame()).model_dump_json()
    )
    monkeypatch.setenv("COTEST_ABLATE_DUAL_LAYER_GRAPH", "1")
    monkeypatch.setenv("COTEST_INTENT_SCOUT_CACHE_DIR", str(cache_dir.parent))
    monkeypatch.setenv("COTEST_INTENT_SCOUT_CACHE_TRUST_LEGACY", "1")
    monkeypatch.delenv("INDEX_STORE_DIR", raising=False)
    monkeypatch.setattr(RUNNER, "get_benchmark_instance", lambda _id: instance)
    monkeypatch.setattr(RUNNER, "create_repository", lambda _instance: InMemRepository())
    monkeypatch.setattr(RUNNER, "_extract_issue_evidence", lambda _issue: IssueEvidenceFrame())
    monkeypatch.setattr(
        RUNNER.CodeIndex, "from_index_name",
        lambda **_kwargs: pytest.fail("ablation must not load the dual-layer index"),
    )

    class FakeDocker:
        container_name = "scout-ablation"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    captured = {}
    def fake_grounding(**kwargs):
        captured.update(kwargs)
        return None
    monkeypatch.setattr(RUNNER, "_run_intent_grounding", fake_grounding)

    RUNNER.run_test_derived_intent_step(
        instance_name=instance_id, model_name="glm-5.3-flash",
        docker_env=FakeDocker(), output_dir=str(tmp_path / "output"),
    )

    assert captured["code_index"] is None
    assert (tmp_path / "output" / "issue_evidence.json").exists()
    assert not (tmp_path / "output" / "intent_grounding_result.json").exists()


def test_graph_ablation_disables_scout_graph_and_test_anchor_tools(monkeypatch, tmp_path):
    monkeypatch.setenv("COTEST_ABLATE_DUAL_LAYER_GRAPH", "1")
    monkeypatch.setenv("COTEST_SCOUT_GRAPH_SEARCH", "1")
    monkeypatch.setenv("COTEST_INTENT_TEST_ANCHOR_RETRIEVAL", "1")
    captured = {}
    monkeypatch.setattr(
        RUNNER.ContainerRepositoryInvestigator, "create",
        lambda **kwargs: captured.setdefault("investigator", kwargs),
    )

    class FakeDialogue:
        def __init__(self, **kwargs):
            captured["dialogue"] = kwargs

        def run(self, **_kwargs):
            raise RuntimeError("stop before model execution")

    monkeypatch.setattr(RUNNER, "IntentDialogue", FakeDialogue)
    with pytest.raises(RuntimeError, match="stop before model execution"):
        RUNNER._run_intent_grounding(
            instance={"problem_statement": "Fix public_call"},
            repository=InMemRepository(), docker_env=SimpleNamespace(),
            code_index=None, issue_evidence=IssueEvidenceFrame(),
            reproduction_context={}, output_dir=str(tmp_path),
            fallback_model="glm-5.3-flash",
        )

    assert captured["investigator"]["enable_graph_search"] is False
    assert captured["dialogue"]["enable_test_anchor_retrieval"] is False


def test_graph_ablation_saves_repair_candidates_without_selector(monkeypatch, tmp_path):
    portfolio = IntentPortfolio(
        obligations=[IntentObligation(
            id="FIX-1", kind="fix", statement="Fix public behavior",
            authority="issue_explicit", strength="hard",
        )],
        hypotheses=[IntentHypothesis(
            id="H0", title="repair", branch_assumption="none",
            covered_obligation_ids=["FIX-1"],
            hard_obligation_coverage={"FIX-1": "Fix public behavior"},
        )],
    )
    patch = "diff --git a/src/fix.cc b/src/fix.cc\n"
    def fake_repair(*_args, output_dir, **_kwargs):
        Path(output_dir, "portfolio_candidate_status.json").write_text(json.dumps({
            "admitted": True,
            "candidate_variants": [{
                "variant_id": "initial", "patch": patch, "admitted": True,
                "finish_succeeded": True, "kept_paths": ["src/fix.cc"],
                "excluded_paths": {}, "repair_probes": [],
            }],
        }))
        return patch

    monkeypatch.setenv("COTEST_ABLATE_DUAL_LAYER_GRAPH", "1")
    monkeypatch.setenv("COTEST_PORTFOLIO_STOP_AFTER_REPAIR", "1")
    monkeypatch.setenv("COTEST_PORTFOLIO_SELECTOR_BACKEND", "independent")
    monkeypatch.setattr(RUNNER, "DockerEnv", lambda **_kwargs: None)
    monkeypatch.setattr(RUNNER, "_build_portfolio_branch_memory", lambda memory, *_args: memory)
    monkeypatch.setattr(RUNNER, "run_repair_step", fake_repair)
    monkeypatch.setattr(
        RUNNER, "_select_portfolio_candidate_independently",
        lambda **_kwargs: pytest.fail("Selector must not run in the ablation"),
    )

    result = RUNNER.run_intent_portfolio_repair_step(
        instance_name="org__repo-1", model_name="glm-5.3-flash",
        image_name="image", work_dir="/repo", base_commit="abc",
        memory=Memory(instance_id="org__repo-1", problem_statement="Fix public behavior"),
        output_dir=str(tmp_path), container_suffix="run", portfolio=portfolio,
    )

    assert result is None
    pool = json.loads((tmp_path / "portfolio_candidate_pool.json").read_text())
    status = json.loads((tmp_path / "portfolio_repair_status.json").read_text())
    assert len(pool) == 1 and pool[0]["variant_id"] == "initial"
    assert status["selector_ran"] is False
    assert status["dual_layer_graph_ablation"] is True
    assert not (tmp_path / "solution.patch").exists()


def test_graph_ablation_hypothesis_ignores_cached_graph_pool(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache" / "org__repo-1"
    cache_dir.mkdir(parents=True)
    cached = IntentPortfolio(
        obligations=[IntentObligation(
            id="FIX-1", kind="fix", statement="Cached graph-guided fix",
            authority="issue_explicit", strength="hard",
        )],
        hypotheses=[IntentHypothesis(
            id="H9", title="cached graph branch", branch_assumption="none",
            covered_obligation_ids=["FIX-1"],
            hard_obligation_coverage={"FIX-1": "Cached graph-guided fix"},
        )],
    )
    (cache_dir / "intent_portfolio.json").write_text(cached.model_dump_json())
    monkeypatch.setenv("COTEST_ABLATE_DUAL_LAYER_GRAPH", "1")
    monkeypatch.setenv("COTEST_INTENT_PORTFOLIO_CACHE_DIR", str(cache_dir.parent))
    monkeypatch.setenv("COTEST_PORTFOLIO_RERANK", "0")
    calls = []
    def fake_extract(*_args, **_kwargs):
        calls.append(True)
        raise RuntimeError("no model in unit test")
    monkeypatch.setattr(RUNNER, "_run_structured_extraction", fake_extract)

    portfolio = RUNNER._generate_intent_portfolio(
        Memory(instance_id="org__repo-1", problem_statement="Fix the Issue"),
        "glm-5.3-flash", str(tmp_path / "output"),
    )

    assert calls
    assert portfolio.hypotheses[0].id == "H0"


def test_graph_ablation_repair_never_loads_index_or_graph_branch_context(monkeypatch, tmp_path):
    from INTENT2FIX import branch_context

    monkeypatch.setenv("COTEST_ABLATE_DUAL_LAYER_GRAPH", "1")
    monkeypatch.setenv("COTEST_BRANCH_CONTEXT", "graph")
    monkeypatch.delenv("INDEX_STORE_DIR", raising=False)
    monkeypatch.setattr(
        RUNNER, "get_benchmark_instance",
        lambda _id: {"base_commit": "abc", "problem_statement": "Fix it"},
    )
    monkeypatch.setattr(RUNNER, "create_repository", lambda _instance: InMemRepository())
    monkeypatch.setattr(
        RUNNER.CodeIndex, "from_index_name",
        lambda **_kwargs: pytest.fail("Repair must not load the dual-layer index"),
    )
    captured = {}
    def fake_context(memory, _context, _repository, _base, _output, mode, **kwargs):
        captured["mode"] = mode
        captured["code_index"] = kwargs["code_index"]
        return memory, "/tmp/unused-ablation-archive.json", {"contract_sha256": "x"}
    monkeypatch.setattr(branch_context, "prepare_branch_context", fake_context)

    class StopBeforeRepairContainer:
        def __enter__(self):
            raise RuntimeError("stop before Repair actions")

        def __exit__(self, *_args):
            return False

    with pytest.raises(RuntimeError, match="stop before Repair actions"):
        RUNNER.run_repair_step(
            instance_name="org__repo-1", model_name="glm-5.3-flash",
            docker_env=StopBeforeRepairContainer(),
            memory=Memory(instance_id="org__repo-1", problem_statement="Fix it"),
            output_dir=str(tmp_path), candidate_mode=True,
            candidate_hypothesis_context={},
        )

    assert captured == {"mode": "text", "code_index": None}


def test_graph_ablation_cli_selects_fresh_three_stage_mode(monkeypatch, tmp_path):
    import sys

    for key in (
        "COTEST_ABLATE_DUAL_LAYER_GRAPH", "COTEST_ENABLE_INTENT_GROUNDING",
        "COTEST_INTENT_DIALOGUE_MODE", "COTEST_ENABLE_INTENT_PORTFOLIO",
        "COTEST_ENABLE_LEGACY_REPRODUCER", "COTEST_ENABLE_REPRODUCER_FALLBACK",
        "COTEST_PORTFOLIO_REPAIR_ONLY", "COTEST_PORTFOLIO_SCOUT_ONLY",
        "COTEST_PORTFOLIO_STOP_AFTER_REPAIR", "COTEST_RESULTS_DIR_PREFIX",
    ):
        monkeypatch.setenv(key, os.getenv(key, ""))
    empty_jsonl = tmp_path / "empty.jsonl"
    empty_jsonl.write_text("")
    monkeypatch.setattr(sys, "argv", [
        "run_no_memory_cotest.py", "--jsonl", str(empty_jsonl),
        "--ablate-dual-layer-graph", "--ablation-results-prefix",
        "results/test_graph_ablation",
    ])
    monkeypatch.setattr(RUNNER, "extract_instance_ids_from_jsonl", lambda **_kwargs: [])

    RUNNER.main()

    assert os.environ["COTEST_ABLATE_DUAL_LAYER_GRAPH"] == "1"
    assert os.environ["COTEST_ENABLE_INTENT_GROUNDING"] == "1"
    assert os.environ["COTEST_ENABLE_INTENT_PORTFOLIO"] == "1"
    assert os.environ["COTEST_ENABLE_LEGACY_REPRODUCER"] == "0"
    assert os.environ["COTEST_ENABLE_REPRODUCER_FALLBACK"] == "0"
    assert os.environ["COTEST_PORTFOLIO_STOP_AFTER_REPAIR"] == "1"
    assert os.environ["COTEST_RESULTS_DIR_PREFIX"] == "results/test_graph_ablation"
