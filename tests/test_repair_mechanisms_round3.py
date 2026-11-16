"""Boundaries around correction evidence refinement and focused compilation feedback."""
import json
from types import SimpleNamespace

import litellm
import pytest

from INTENT2FIX.file_context import FileContext
from INTENT2FIX.repository.repository import InMemRepository
from INTENT2FIX.intent_portfolio import BranchComplianceDecision, BranchCorrectionRefinement, RepairProbeEvidence
from INTENT2FIX.patch_review.sanitizer import sanitize_candidate_patch
from tests.test_cotest_patch_selection import RUNNER


@pytest.mark.parametrize("fault", ["none", "empty", "quote", "stale_evidence", "changed_clause", "exception", "stale_patch", "soft_clause"])
def test_refine_only_fills_same_correction_evidence_once(monkeypatch, tmp_path, fault):
    context = FileContext(repo=InMemRepository({"src/fix.cpp": "int value;\n"}))
    context.add_file("src/fix.cpp").apply_changes("int value = 2;\n")
    candidate = sanitize_candidate_patch(context)
    hypothesis = {"branch": {"id": "H0", "expected_behavior": ["Return 1."]}}
    raw = BranchComplianceDecision(
        status="noncompliant", observed_behavior="Returns 2.", reason="Value mismatch.",
        actionable_correction=True, violated_clause_ids=["BRANCH-EXPECTED-1"],
        suggested_correction="Return 1 within H0.",
        production_patch_sha256=RUNNER._portfolio_patch_sha256(candidate.patch))
    if fault == "stale_patch":
        raw.production_patch_sha256 = "earlier-patch"
    if fault == "soft_clause":
        raw.violated_clause_ids = ["PRESERVE-1"]
    violation = dict(clause_id="BRANCH-EXPECTED-1", clause_quote="Return 1.",
                     scenario="Read value.", expected_behavior="1", observed_behavior="2", evidence_ids=["PATCH"])
    if fault == "quote":
        violation["clause_quote"] = "Return 3."
    if fault == "stale_evidence":
        violation["evidence_ids"] = ["RP-old"]
    if fault == "changed_clause":
        violation["clause_id"] = "BRANCH-EXPECTED-2"
    calls = []
    def extract(prompt, schema, **kwargs):
        calls.append(schema)
        if schema is BranchComplianceDecision:
            return raw
        assert schema is BranchCorrectionRefinement
        assert kwargs["max_attempts"] == 1 and kwargs["glm_reasoning_effort"] == "low"
        assert "Return 1 within H0." in prompt and candidate.patch.splitlines()[0] in prompt
        if fault == "exception":
            raise ValueError("bad JSON")
        return BranchCorrectionRefinement(violations=[] if fault == "empty" else [violation])
    monkeypatch.setattr(RUNNER, "_run_structured_extraction", extract)
    decision = RUNNER._audit_portfolio_branch_compliance(
        hypothesis_context=hypothesis, candidate=candidate, repair_probes=[], file_context=context,
        model_name="offline", output_dir=str(tmp_path), label="initial")
    assert len(calls) == (1 if fault in {"soft_clause", "stale_patch"} else 2)
    assert decision.actionable_correction == (fault == "none")
    assert decision.violated_clause_ids == raw.violated_clause_ids
    assert not raw.violations  # Original audit was not mutated.
    if fault == "none":
        assert decision.suggested_correction == raw.suggested_correction
        assert decision.production_patch_sha256 == raw.production_patch_sha256
    assert json.loads((tmp_path / "branch_compliance_initial_raw.json").read_text()) == raw.model_dump()


@pytest.mark.parametrize("fault", ["none", "negative", "stale", "timeout", "missing_tool", "probe_source", "passed", "soft"])
def test_positive_production_compile_probe_only(fault):
    context = FileContext(repo=InMemRepository())
    context.add_file("include/api.hpp").apply_changes("int value;\n")
    candidate = sanitize_candidate_patch(context)
    probe = RepairProbeEvidence(
        probe_id="RP-1", source_hypothesis_id="H0", command="g++ repro.cpp",
        expected_exit_code=0, observed_exit_code=1, observed_status="failed",
        observed_output="In file included from repro.cpp:1:\ninclude/api.hpp:215:82: error: always_inline function not inlinable\n",
        production_patch_sha256=RUNNER._portfolio_patch_sha256(candidate.patch))
    if fault == "negative":
        probe.expected_exit_code = 1
    if fault == "stale":
        probe.production_patch_sha256 = "earlier-patch"
    if fault == "timeout":
        probe.observed_exit_code = 124
    if fault == "missing_tool":
        probe.observed_output = "g++: command not found"
    if fault == "probe_source":
        probe.observed_output = "repro.cpp:215:82: error: no matching function"
    if fault == "passed":
        probe.observed_status = "passed"
    if fault == "soft":
        probe.kind = "soft_regression"
    assert bool(RUNNER._portfolio_engineering_diagnostic(context, candidate, [probe])) == (fault == "none")


def test_refine_sees_existing_tests_without_shipping_them(monkeypatch, tmp_path):
    context = FileContext(repo=InMemRepository())
    context.add_file("src/fix.cpp").apply_changes("int value = 2;\n")
    context.add_file("tests/fix.cpp").apply_changes("CHECK(value == 2);\n")
    candidate = sanitize_candidate_patch(context)
    assert "tests/fix.cpp" not in candidate.patch
    hypothesis = {"branch": {"id": "H0", "repair_scope": ["Write a test for value."]}}
    raw = BranchComplianceDecision(
        status="noncompliant", observed_behavior="No tests in production patch.", reason="Missing test.",
        actionable_correction=True, violated_clause_ids=["BRANCH-SCOPE-1"],
        suggested_correction="Write a test.", production_patch_sha256=RUNNER._portfolio_patch_sha256(candidate.patch))
    def extract(prompt, schema, **kwargs):
        assert "CHECK(value == 2)" in prompt
        if schema is BranchComplianceDecision:
            assert "Their contents alone do not prove execution" in prompt
            return raw
        assert "Never ask to ship test files as production code" in prompt
        return BranchCorrectionRefinement(violations=[])
    monkeypatch.setattr(RUNNER, "_run_structured_extraction", extract)
    result = RUNNER._audit_portfolio_branch_compliance(
        hypothesis_context=hypothesis, candidate=candidate, repair_probes=[], file_context=context,
        model_name="offline", output_dir=str(tmp_path), label="initial")
    assert not result.actionable_correction


def test_refine_transport_and_structure_attempts_are_bounded(monkeypatch):
    calls = []
    class InvalidResponse:
        def __init__(self, **kwargs):
            calls.append(kwargs)
        def generate(self, prompt):
            return type("Response", (), {"message": "invalid JSON"})()
    monkeypatch.setattr(RUNNER, "LiteLLM", InvalidResponse)
    monkeypatch.setattr(RUNNER, "_load_env", lambda: None)
    result = RUNNER._run_structured_extraction("Fill violations.", BranchCorrectionRefinement,
                                             model_override="offline", max_attempts=1)
    assert result is None
    assert len(calls) == 1
    assert calls[0]["max_retries"] == 0 and calls[0]["timeout"] == 180.0


@pytest.mark.parametrize("rate_limits,outcome", [
    (2, "valid"), (4, "valid"), (5, "exhausted"),
    (0, "invalid_json"), (1, "invalid_json"), (0, "bad_request"), (0, "auth"),
])
def test_refine_retries_only_429_without_spending_structure_attempts(monkeypatch, rate_limits, outcome):
    requests, settings, sleeps = [], [], []

    class Provider:
        def __init__(self, **kwargs):
            settings.append(kwargs)

        def generate(self, prompt):
            requests.append(prompt)
            if len(requests) <= rate_limits:
                raise litellm.RateLimitError(
                    message="1305: The service may be temporarily overloaded, please try again later",
                    llm_provider="openai", model="offline")
            if outcome in {"bad_request", "auth"}:
                error = litellm.BadRequestError if outcome == "bad_request" else litellm.AuthenticationError
                raise error(message="Rejected request", llm_provider="openai", model="offline")
            return SimpleNamespace(message='{"violations": []}' if outcome == "valid" else "invalid JSON")

    monkeypatch.setattr(RUNNER, "LiteLLM", Provider)
    monkeypatch.setattr(RUNNER, "_load_env", lambda: None)
    monkeypatch.setattr(RUNNER, "get_model_config", lambda _: {"model": "offline"})
    monkeypatch.setattr(RUNNER.time, "sleep", sleeps.append)
    monkeypatch.setattr("tenacity.wait.random.uniform", lambda low, high: high / 2)
    prompt = "Fill violations for the unchanged patch and hypothesis."
    result = RUNNER._run_structured_extraction(
        prompt, BranchCorrectionRefinement, model_override="offline", max_attempts=1)

    assert (isinstance(result, BranchCorrectionRefinement)) == (outcome == "valid")
    assert requests == [prompt] * min(rate_limits + 1, 5)
    assert sleeps == [5.5, 10.5, 20.5, 40.5][:min(rate_limits, 4)]
    assert len(settings) == 1
    assert settings[0]["max_retries"] == 0  # Do not nest SDK retries inside our backoff.
