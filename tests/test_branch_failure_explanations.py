"""A failed execution needs explicit branch authority before Audit calls it compliant."""
import copy

import pytest

from INTENT2FIX.intent_portfolio import (
    BranchComplianceDecision,
    RepairProbeEvidence,
    build_branch_compliance_prompt,
    validate_branch_correction,
)
from INTENT2FIX.utils.patch import production_patch_sha256


def test_compliance_reads_frozen_test_source_not_just_success_labels():
    patch = "frozen production patch\n"
    probe = RepairProbeEvidence(
        probe_id="RP-1", source_hypothesis_id="H0", command="c++ empty.cpp && ./a.out",
        files={"empty.cpp": "int main(){return 0;}\n"}, observed_status="passed",
        observed_exit_code=0, production_patch_sha256=production_patch_sha256(patch),
    )
    prompt = build_branch_compliance_prompt(
        hypothesis_context={}, patch=patch, repair_probes=[probe], regression_evidence={},
    )
    assert "int main(){return 0;}" in prompt
    assert "inspect its frozen files" in prompt
    assert "does not establish Issue behavior" in " ".join(prompt.split())


def failure_fixture():
    patch = "frozen production patch\n"
    digest = production_patch_sha256(patch)
    clause = "Formatting a function pointer must fail to compile."
    context = {
        "hard_obligations": [
            {"id": "FIX-1", "strength": "hard", "statement": clause},
            {"id": "PRESERVE-1", "strength": "hard", "statement": "Actual bool output is unchanged."},
        ],
        "soft_evidence": [{"id": "PRESERVE-3", "statement": "Monitor other bool-convertible types."}],
        "branch": {"expected_behavior": [clause], "repair_scope": ["Constrain the bool mapping."],
                   "preservation_behavior": ["Other bool-convertible types may be rejected."]},
    }
    record = {
        "passed": False, "exit_code": 2, "verification_view": "production",
        "production_patch_sha256": digest,
        "output": "old_function_pointer_test: error: Formatting of non-void pointers is disallowed.",
    }
    evidence = {"production_only_with_original_tests": record}
    explanation = {
        "failure_id": "production_only_with_original_tests",
        "clause_id": "BRANCH-EXPECTED-1", "clause_quote": clause,
        "scenario": "Compile old_function_pointer_test formatting a function pointer.",
        "expected_old_behavior": "The old test expects compilation to succeed.",
        "observed_behavior": "Compilation rejects the function pointer, as assigned.",
        "evidence_quote": record["output"],
    }
    decision = dict(status="compliant", observed_behavior="Function pointers are rejected.",
                    reason="An explicit branch change conflicts with the old test.",
                    production_patch_sha256=digest, failure_explanations=[explanation])
    return context, patch, evidence, decision


@pytest.mark.parametrize("fault", [
    "missing", "soft", "scope", "preserve", "stale_patch", "quote", "invented_evidence",
    "uncovered_probe", "duplicate", "wrong_failure", "empty_scenario", "abbreviated_quote",
])
def test_compliance_requires_complete_current_failure_justifications(fault):
    context, patch, evidence, data = failure_fixture()
    explanations = data["failure_explanations"]
    probes = []
    if fault == "missing":
        data["failure_explanations"] = []
    elif fault in {"soft", "scope", "preserve"}:
        cid, quote = {
            "soft": ("PRESERVE-3", context["soft_evidence"][0]["statement"]),
            "scope": ("BRANCH-SCOPE-1", context["branch"]["repair_scope"][0]),
            "preserve": ("PRESERVE-1", context["hard_obligations"][1]["statement"]),
        }[fault]
        explanations[0].update(clause_id=cid, clause_quote=quote)
    elif fault == "stale_patch":
        data["production_patch_sha256"] = "old-patch"
    elif fault == "quote":
        explanations[0]["clause_quote"] = "Any bool-convertible output may change."
    elif fault == "invented_evidence":
        explanations[0]["evidence_quote"] = "Compilation succeeded and printed true."
    elif fault == "abbreviated_quote":
        explanations[0]["evidence_quote"] = "old_function_pointer_test: ... non-void pointers is disallowed."
    elif fault == "uncovered_probe":
        probes.append(RepairProbeEvidence(
            probe_id="RP-1", source_hypothesis_id="H0", command="probe",
            expected_exit_code=0, observed_exit_code=1, observed_status="failed",
            observed_output="Expected true, got 1.",
            production_patch_sha256=production_patch_sha256(patch)))
    elif fault == "duplicate":
        explanations.append(copy.deepcopy(explanations[0]))
    elif fault == "wrong_failure":
        explanations[0]["failure_id"] = "RP-other"
    else:
        explanations[0]["scenario"] = ""
    raw = BranchComplianceDecision.model_validate(data)
    checked = validate_branch_correction(raw, hypothesis_context=context, patch=patch,
                                        repair_probes=probes, regression_evidence=evidence)
    assert checked.status == "inconclusive"
    assert not checked.actionable_correction
    assert checked.suggested_correction == ""
    assert raw.status == "compliant"


@pytest.mark.parametrize("clause_id", ["FIX-1", "BRANCH-EXPECTED-1", "DIM-1"])
def test_exact_assigned_behavior_can_explain_old_test_failure(clause_id):
    context, patch, evidence, data = failure_fixture()
    context["assigned_dimensions"] = [{"dimension_id": "DIM-1", "assigned_outcome": context["branch"]["expected_behavior"][0]}]
    data["failure_explanations"][0]["clause_id"] = clause_id
    raw = BranchComplianceDecision.model_validate(data)
    checked = validate_branch_correction(raw, hypothesis_context=context, patch=patch,
                                        repair_probes=[], regression_evidence=evidence)
    assert checked.status == "compliant"
    assert not checked.actionable_correction


@pytest.mark.parametrize("state", ["passed", "stale", "unproven"])
def test_only_current_production_failures_require_explanations(state):
    context, patch, evidence, data = failure_fixture()
    data["failure_explanations"] = []
    record = evidence["production_only_with_original_tests"]
    if state == "passed":
        record.update(passed=True, exit_code=0)
    elif state == "stale":
        record["production_patch_sha256"] = "old-patch"
    else:
        record["verification_view"] = "unproven"
    raw = BranchComplianceDecision.model_validate(data)
    checked = validate_branch_correction(raw, hypothesis_context=context, patch=patch,
                                        repair_probes=[], regression_evidence=evidence)
    assert checked.status == "compliant"


def test_unexplained_failure_does_not_instruct_repair_to_change_hypothesis():
    from tests.test_cotest_patch_selection import RUNNER
    context, patch, evidence, data = failure_fixture()
    data["failure_explanations"] = []
    checked = validate_branch_correction(BranchComplianceDecision.model_validate(data),
                                        hypothesis_context=context, patch=patch,
                                        repair_probes=[], regression_evidence=evidence)
    assert checked.status == "inconclusive"
    assert not RUNNER._portfolio_branch_requires_correction(compliance=checked)
