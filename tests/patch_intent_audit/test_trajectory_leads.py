from __future__ import annotations

from unittest.mock import patch

from INTENT2FIX.patch_intent_audit.trajectory_leads import (
    AuditLead,
    AuditLeadBatch,
    EvidenceQuote,
    extract_trajectory_leads,
    slice_trajectory,
    validate_audit_leads,
)


def _step(
    action_type: str,
    message: str,
    *,
    action: dict | None = None,
    exit_code: int | None = None,
    correction: bool = False,
) -> dict:
    return {
        "reasoning_content": "provider reasoning that must never be retained",
        "action": {
            "thoughts": "private reasoning that must never be retained",
            "action_args_class": f"pkg.{action_type}Args",
            **(action or {}),
        },
        "observation": {
            "message": message,
            "expect_correction": correction,
            "properties": {} if exit_code is None else {"exit_code": exit_code},
        },
    }


def _trajectory(
    *steps: dict,
    issue: str = "Formatting must preserve the requested base.",
) -> dict:
    return {
        "metadata": {
            "instance_id": "project__repo-1",
            "problem_statement": issue,
        },
        "nodes": [
            {
                "assistant_message": "must not leak",
                "assistant_reasoning_content": "must not leak",
                "completions": {"raw": "must not leak"},
                "action_steps": list(steps),
            }
        ],
    }


def _finished(
    *steps: dict, issue: str = "Formatting must preserve the requested base."
) -> dict:
    return _trajectory(*steps, _step("VerifiedFinish", "Done."), issue=issue)


def test_slicer_excludes_reasoning_and_uses_allowlisted_action_fields() -> None:
    sliced = slice_trajectory(
        _finished(
            _step(
                "RunCommand",
                "Command executed successfully with exit code 0.\nbase path found",
                action={
                    "command": "grep -n base include/api.h",
                    "untrusted_payload": "must not leak",
                },
                exit_code=0,
            )
        ),
        "diff --git a/include/api.h b/include/api.h\n+base path",
    )
    rendered = sliced.model_dump_json()
    assert "private reasoning" not in rendered
    assert "provider reasoning" not in rendered
    assert "assistant_message" not in rendered
    assert "untrusted_payload" not in rendered
    assert "grep -n base include/api.h" in rendered


def test_first_finish_cuts_off_same_and_later_nodes() -> None:
    trajectory = _trajectory(
        _step("RunCommand", "Pre-finish observation.", exit_code=0),
        _step("VerifiedFinish", "Complete."),
        _step("RunRegression", "Post-finish regression passed.", exit_code=0),
    )
    trajectory["nodes"].append(
        {"action_steps": [_step("RunCommand", "Reviewer refinement evidence.")]}
    )
    sliced = slice_trajectory(trajectory, "diff --git a/a b/a\n+change")
    rendered = sliced.model_dump_json()
    assert sliced.repair_finished is True
    assert sliced.finish_ordinal == 2
    assert sliced.stats.post_finish_steps_skipped == 2
    assert "Post-finish" not in rendered
    assert "Reviewer refinement" not in rendered


def test_corrections_and_infrastructure_failures_are_only_operational_stats() -> None:
    sliced = slice_trajectory(
        _finished(
            _step(
                "RunCommand",
                "Use one direct compiler invocation.",
                correction=True,
            ),
            _step(
                "RunRegression",
                "API connection error while starting the test container",
                exit_code=1,
            ),
            _step("RunCommand", "Public overload found.", exit_code=0),
        ),
        "diff --git a/include/a.hpp b/include/a.hpp\n+overload",
    )
    assert sliced.stats.corrections_skipped == 1
    assert sliced.stats.infrastructure_failures_skipped == 1
    assert len(sliced.events) == 1
    assert "Public overload" in sliced.events[0].observation_excerpt


def test_equivalent_events_aggregate_within_a_phase_but_not_after_edit() -> None:
    first = _step(
        "RunSingleTest",
        "a.hpp:10:2: error: no matching function for alpha<int> after 1.2 seconds",
        action={"command": "c++ repro.cpp"},
        exit_code=1,
    )
    equivalent = _step(
        "RunSingleTest",
        "a.hpp:99:8: error: no matching function for alpha<int> after 2 seconds",
        action={"command": "c++ repro.cpp"},
        exit_code=1,
    )
    sliced = slice_trajectory(
        _finished(
            first,
            equivalent,
            _step(
                "StringReplace",
                "The file include/a.hpp has been edited.",
                action={"path": "include/a.hpp"},
            ),
            equivalent,
        ),
        "diff --git a/include/a.hpp b/include/a.hpp\n+alpha",
    )
    diagnostics = [event for event in sliced.events if event.diagnostic_key]
    assert len(diagnostics) == 2
    assert diagnostics[0].occurrences == 2
    assert diagnostics[0].phase_id != diagnostics[1].phase_id


def test_masked_pipeline_exit_does_not_hide_explicit_compilation_error() -> None:
    sliced = slice_trajectory(
        _finished(
            _step(
                "RunSingleTest",
                "Single test finished with exit code 0.\n"
                "Log Type: compilation_error\nerror: no viable overload",
                action={"command": "c++ repro.cpp 2>&1 | head -40"},
                exit_code=0,
            )
        ),
        "diff --git a/include/api.hpp b/include/api.hpp\n+overload",
    )
    assert sliced.events[0].outcome == "failed"


def test_temporary_probe_edit_does_not_start_a_production_phase() -> None:
    sliced = slice_trajectory(
        _finished(
            _step(
                "CreateFile",
                "Created probe.cpp.",
                action={"path": "probe.cpp", "artifact_kind": "temporary_probe"},
            ),
            _step("RunSingleTest", "Probe failed.", exit_code=1),
        ),
        "diff --git a/include/api.hpp b/include/api.hpp\n+change",
    )
    assert all(event.phase_id == 0 for event in sliced.events)
    assert all(event.action_type != "CreateFile" for event in sliced.events)


def test_production_edits_define_phases_but_are_not_evidence() -> None:
    sliced = slice_trajectory(
        _finished(
            _step("RunSingleTest", "Before edit failure.", exit_code=1),
            _step(
                "StringReplace",
                "Candidate code copied here must not become evidence.",
                action={"path": "include/api.hpp"},
            ),
            _step("RunSingleTest", "After edit result.", exit_code=0),
        ),
        "diff --git a/include/api.hpp b/include/api.hpp\n+change",
    )
    assert [event.phase_id for event in sliced.events] == [0, 1]
    assert all(event.action_type != "StringReplace" for event in sliced.events)
    assert sliced.stats.edit_observations_skipped == 1


def test_lead_validation_requires_exact_unique_quotes_and_issue_span() -> None:
    sliced = slice_trajectory(
        _finished(
            _step(
                "RunCommand",
                "The decimal path groups digits; the hexadecimal path bypasses it.",
                action={"command": "grep grouping include/api.h"},
                exit_code=0,
            )
        ),
        "diff --git a/include/api.h b/include/api.h\n+group_digits",
    )
    event = sliced.events[0]
    valid = AuditLead(
        evidence_quotes=[
            EvidenceQuote(
                evidence_id=event.id,
                quote="the hexadecimal path bypasses it",
            )
        ],
        observed_fact="The observed hexadecimal path bypasses grouping.",
        repair_assumption="The Repair may assume decimal coverage is sufficient.",
        unverified_public_behavior="Grouping for non-decimal presentations.",
        suggested_investigation="Compare decimal and hexadecimal formatting.",
        issue_claim_span="preserve the requested base",
    )
    invalid_quote = valid.model_copy(
        update={
            "evidence_quotes": [
                EvidenceQuote(evidence_id=event.id, quote="invented quote")
            ]
        }
    )
    duplicate_quote = valid.model_copy(
        deep=True,
        update={"evidence_quotes": valid.evidence_quotes * 2},
    )
    leads, errors = validate_audit_leads(
        AuditLeadBatch(leads=[valid, invalid_quote, duplicate_quote]), sliced
    )
    assert leads == [valid]
    assert len(errors) == 2


def test_quote_whitespace_is_canonicalized_and_distinct_same_event_quotes_are_valid() -> (
    None
):
    sliced = slice_trajectory(
        _finished(
            _step("RunCommand", "first line\n    second line; third fact", exit_code=0)
        ),
        "diff --git a/a b/a\n+change",
    )
    event = sliced.events[0]
    lead = AuditLead(
        evidence_quotes=[
            EvidenceQuote(evidence_id=event.id, quote="first line second line"),
            EvidenceQuote(evidence_id=event.id, quote="third fact"),
        ],
        observed_fact="Two facts were observed.",
        repair_assumption="The Repair may rely on both facts.",
        unverified_public_behavior="A public boundary remains unverified.",
        suggested_investigation="Run a focused public probe in the current checkout.",
    )
    leads, errors = validate_audit_leads(AuditLeadBatch(leads=[lead]), sliced)
    assert errors == []
    assert leads[0].evidence_quotes[0].quote == "first line\n    second line"


def test_future_patch_investigation_is_rejected() -> None:
    sliced = slice_trajectory(
        _finished(_step("RunCommand", "A public fact.", exit_code=0)),
        "diff --git a/a b/a\n+change",
    )
    event = sliced.events[0]
    lead = AuditLead(
        evidence_quotes=[EvidenceQuote(evidence_id=event.id, quote="A public fact.")],
        observed_fact="A fact was observed.",
        repair_assumption="The Repair may rely on it.",
        unverified_public_behavior="A boundary remains unverified.",
        suggested_investigation="Compare the upstream fix in a pull request.",
    )
    leads, errors = validate_audit_leads(AuditLeadBatch(leads=[lead]), sliced)
    assert leads == []
    assert "forbidden future evidence" in errors[0]


def test_evidence_quote_accepts_provider_aliases_but_serializes_canonical_names() -> (
    None
):
    quote = EvidenceQuote.model_validate({"id": "EV-3", "excerpt": "exact text"})
    assert quote.evidence_id == "EV-3"
    assert quote.quote == "exact text"
    assert quote.model_dump() == {"evidence_id": "EV-3", "quote": "exact text"}


def test_issue_slice_covers_distinct_issue_units() -> None:
    sliced = slice_trajectory(
        _finished(
            _step(
                "RunCommand",
                "Parsing accepts lowercase names through the public API.",
                action={"command": "grep lowercase include/parser.hpp"},
                exit_code=0,
            ),
            _step(
                "RunCommand",
                "The parser preserves previous state after invalid input.",
                action={"command": "grep previous_state include/parser.hpp"},
                exit_code=0,
            ),
            issue=(
                "Accept lowercase names through the public API. "
                "Preserve the previous state after invalid input."
            ),
        ),
        "diff --git a/include/parser.hpp b/include/parser.hpp\n+lowercase\n+previous_state",
        issue_budget=2,
    )
    assert len(sliced.selected_by_slice["issue"]) == 2


def test_unfinished_trajectory_never_calls_llm() -> None:
    sliced = slice_trajectory(
        _trajectory(_step("RunCommand", "Public path found.", exit_code=0)),
        "diff --git a/include/a.hpp b/include/a.hpp\n+change",
    )
    with patch(
        "INTENT2FIX.patch_intent_audit.trajectory_leads._one_completion"
    ) as completion:
        result = extract_trajectory_leads(sliced, "unused-model")
    completion.assert_not_called()
    assert result.extraction_status == "empty"
    assert "no Finish" in result.validation_errors[0]


def test_missing_candidate_patch_never_calls_llm() -> None:
    sliced = slice_trajectory(
        _finished(_step("RunCommand", "Public path found.", exit_code=0)), ""
    )
    with patch(
        "INTENT2FIX.patch_intent_audit.trajectory_leads._one_completion"
    ) as completion:
        result = extract_trajectory_leads(sliced, "unused-model")
    completion.assert_not_called()
    assert result.extraction_status == "empty"
    assert result.validation_errors == ["No candidate patch is available to audit."]


def test_malformed_nodes_are_skipped_without_failing_the_trajectory() -> None:
    trajectory = _finished(_step("RunCommand", "Public symbol found.", exit_code=0))
    trajectory["nodes"].insert(0, {"action_steps": "legacy-shape"})
    sliced = slice_trajectory(trajectory, "diff --git a/a b/a\n+symbol")
    assert sliced.stats.skipped_unknown_shape == 1
    assert sliced.repair_finished is True
