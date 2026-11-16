import inspect

import INTENT2FIX.patch_intent_audit.auditor as auditor_module
from INTENT2FIX.file_context import FileContext
from INTENT2FIX.patch_intent_audit.auditor import (
    PatchIntentAuditSession,
    PatchIntentAuditor,
    RunContrastiveIntentProbe,
    RunContrastiveIntentProbeArgs,
    SubmitPatchIntentAudit,
    SubmitPatchIntentAuditArgs,
    WriteContrastiveIntentProbe,
    WriteContrastiveIntentProbeArgs,
    _candidate_patch_change_view,
    _probe_command_binding,
    _protected_worktree_fingerprint,
    _single_command_rejection,
    _stage_output,
    _validated_probe_path,
)
from INTENT2FIX.patch_intent_audit.schema import (
    FrozenPatchIntentProbe,
    PatchIntentAuditResult,
    PatchIntentSemanticAdmission,
)


ISSUE = "Calling the public formatter with emphasis::bold must succeed."


class _FakeAuditRuntime:
    work_dir = "/repo"

    def __init__(
        self,
        *,
        control_compile_exit=0,
        control_run_exit=0,
        target_compile_exit=0,
        target_run_exit=3,
    ):
        self.writes = {}
        self.control_compile_exit = control_compile_exit
        self.control_run_exit = control_run_exit
        self.target_compile_exit = target_compile_exit
        self.target_run_exit = target_run_exit

    def write_file(self, path, content):
        self.writes[path] = content

    def run_bash(self, command, work_dir=None, timeout=None):
        if "git diff --no-ext-diff" in command:
            return "same-fingerprint  -\n", 0
        if command == "git ls-files --others --exclude-standard -z":
            return "", 0
        if command.startswith("c++ "):
            if "probe_target.cpp" in command:
                return "target compile", self.target_compile_exit
            return "control compile", self.control_compile_exit
        if command == "/tmp/intent2fix-audit/test/probe_control":
            return "control result", self.control_run_exit
        if command == "/tmp/intent2fix-audit/test/probe_target":
            return "target result", self.target_run_exit
        raise AssertionError(f"unexpected command: {command}")


def _admit(_prompt, schema):
    assert schema is PatchIntentSemanticAdmission
    return PatchIntentSemanticAdmission(
        admitted=True,
        authority_entails_target_expectation=True,
        authority_identifies_disputed_dimension=True,
        pair_exposes_observable_choice=True,
        target_distinguishes_predictions=True,
        pair_is_comparable=True,
        target_audits_modified_mechanism=True,
        candidate_prediction_matches_observation=True,
        reason="isolated pair",
    )


def _reject_admission(_prompt, schema):
    assert schema is PatchIntentSemanticAdmission
    return PatchIntentSemanticAdmission(
        admitted=False,
        authority_entails_target_expectation=False,
        authority_identifies_disputed_dimension=False,
        pair_exposes_observable_choice=False,
        target_distinguishes_predictions=False,
        pair_is_comparable=False,
        target_audits_modified_mechanism=False,
        candidate_prediction_matches_observation=False,
        reason="control does not follow from the Issue",
    )


def _unavailable_admission(_prompt, schema):
    assert schema is PatchIntentSemanticAdmission
    return PatchIntentSemanticAdmission(
        admitted=False,
        available=False,
        authority_entails_target_expectation=False,
        authority_identifies_disputed_dimension=False,
        pair_exposes_observable_choice=False,
        target_distinguishes_predictions=False,
        pair_is_comparable=False,
        target_audits_modified_mechanism=False,
        candidate_prediction_matches_observation=False,
        reason="provider unavailable",
    )


def _admit_behavior_without_mechanism(_prompt, schema):
    assert schema is PatchIntentSemanticAdmission
    return PatchIntentSemanticAdmission(
        admitted=False,
        authority_entails_target_expectation=True,
        authority_identifies_disputed_dimension=True,
        pair_exposes_observable_choice=True,
        target_distinguishes_predictions=True,
        pair_is_comparable=True,
        target_audits_modified_mechanism=True,
        candidate_prediction_matches_observation=False,
        reason=(
            "The target proves a behavioral violation, but execution refutes "
            "the proposed mechanism explanation."
        ),
    )


def _unresolved_mechanism_admission():
    return PatchIntentSemanticAdmission(
        admitted=False,
        authority_entails_target_expectation=False,
        authority_identifies_disputed_dimension=True,
        pair_exposes_observable_choice=True,
        target_distinguishes_predictions=False,
        pair_is_comparable=True,
        target_audits_modified_mechanism=True,
        candidate_prediction_matches_observation=True,
        reason=(
            "The Issue requires this dispatch dimension but does not select "
            "one precedence outcome."
        ),
    )


def test_semantic_admission_cannot_override_failed_consistency_checks():
    admission = PatchIntentSemanticAdmission(
        admitted=True,
        authority_entails_target_expectation=True,
        authority_identifies_disputed_dimension=True,
        pair_exposes_observable_choice=True,
        target_distinguishes_predictions=False,
        pair_is_comparable=True,
        target_audits_modified_mechanism=True,
        candidate_prediction_matches_observation=True,
        reason="The prose claims that the pair is decisive.",
    )

    assert not admission.admitted
    assert "target_distinguishes_predictions" in admission.reason


def test_semantic_admission_requires_each_atomic_judgment():
    try:
        PatchIntentSemanticAdmission.model_validate(
            {"admitted": True, "reason": "pair accepted"}
        )
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("missing semantic judgments must not be defaulted")

    assert "authority_entails_target_expectation" in message
    assert "authority_identifies_disputed_dimension" in message
    assert "pair_exposes_observable_choice" in message
    assert "target_distinguishes_predictions" in message
    assert "pair_is_comparable" in message
    assert "target_audits_modified_mechanism" in message
    assert "candidate_prediction_matches_observation" in message


def test_semantic_admission_separates_behavior_from_mechanism_attribution():
    admission = PatchIntentSemanticAdmission(
        admitted=True,
        authority_entails_target_expectation=True,
        authority_identifies_disputed_dimension=True,
        pair_exposes_observable_choice=True,
        target_distinguishes_predictions=True,
        pair_is_comparable=True,
        target_audits_modified_mechanism=True,
        candidate_prediction_matches_observation=False,
        reason="The target passed although the candidate predicted rejection.",
    )

    assert admission.admitted
    assert not admission.candidate_prediction_matches_observation


def test_semantic_admission_rejects_structural_sibling_counterexample():
    admission = PatchIntentSemanticAdmission(
        admitted=True,
        authority_entails_target_expectation=True,
        authority_identifies_disputed_dimension=True,
        pair_exposes_observable_choice=True,
        target_distinguishes_predictions=True,
        pair_is_comparable=True,
        target_audits_modified_mechanism=False,
        candidate_prediction_matches_observation=True,
        reason=(
            "The target only reaches an unmodified sibling overload; it does "
            "not challenge the mechanism chosen in the candidate hunk."
        ),
    )

    assert not admission.admitted
    assert "target_audits_modified_mechanism" in admission.reason


def test_semantic_admission_accepts_public_entry_using_changed_shared_mechanism():
    admission = PatchIntentSemanticAdmission(
        admitted=True,
        authority_entails_target_expectation=True,
        authority_identifies_disputed_dimension=True,
        pair_exposes_observable_choice=True,
        target_distinguishes_predictions=True,
        pair_is_comparable=True,
        target_audits_modified_mechanism=True,
        candidate_prediction_matches_observation=True,
        reason=(
            "The public entry is unchanged, but execution reaches the shared "
            "dispatch mechanism modified by the candidate patch."
        ),
    )

    assert admission.admitted


def test_unresolved_dimension_does_not_admit_blocking_counterexample():
    admission = _unresolved_mechanism_admission()

    assert not admission.admitted
    assert admission.authority_identifies_disputed_dimension
    assert admission.pair_exposes_observable_choice
    assert not admission.authority_entails_target_expectation
    assert not admission.target_distinguishes_predictions


def test_candidate_patch_change_view_preserves_middle_hunk():
    patch = "\n".join(
        [
            "--- a/include/api.h",
            "+++ b/include/api.h",
            "@@ -1 +1 @@",
            "-" + "a" * 12000,
            "+FIRST_TOKEN",
            "@@ -2 +2 @@",
            "-" + "b" * 12000,
            "+MIDDLE_TOKEN",
            "@@ -3 +3 @@",
            "-" + "c" * 12000,
            "+LAST_TOKEN",
        ]
    )

    view = _candidate_patch_change_view(patch, max_chars=1200)

    assert "@@ -1 +1 @@" in view
    assert "@@ -2 +2 @@" in view
    assert "@@ -3 +3 @@" in view
    assert "+FIRST_TOKEN" in view
    assert "+MIDDLE_TOKEN" in view
    assert "+LAST_TOKEN" in view


def test_candidate_patch_change_view_does_not_drop_late_hunks():
    patch = "\n".join(
        ["--- a/f", "+++ b/f"]
        + [
            line
            for index in range(500)
            for line in (
                f"@@ -{index} +{index} @@",
                f"-OLD_{index}",
                f"+TOKEN_{index}",
            )
        ]
    )

    view = _candidate_patch_change_view(patch, max_chars=32000)

    assert "@@ -0 +0 @@" in view
    assert "+TOKEN_0" in view
    assert "@@ -499 +499 @@" in view
    assert "+TOKEN_499" in view


def _execute_probe_with_admission(
    semantic_adjudicator,
    *,
    issue: str = ISSUE,
    contract: str = "",
    oracle_source: str = "issue_explicit",
    purpose: str = "distinguish public formatter acceptance",
    oracle_reference: str | None = None,
    runtime: _FakeAuditRuntime | None = None,
    oracle_mode: str = "runtime_behavior",
    target_content: str = "int main() { return 3; }",
    candidate_prediction: str = "returns an error",
    alternative_prediction: str = "formats the value",
):
    runtime = runtime or _FakeAuditRuntime()
    session = PatchIntentAuditSession(audit_dir="/tmp/intent2fix-audit/test")
    WriteContrastiveIntentProbe(
        docker_env=runtime,
        session=session,
    ).execute(
        WriteContrastiveIntentProbeArgs(
            thoughts="",
            probe_name="probe",
            fixture_content="struct input {};",
            control_content="int main() { return 0; }",
            target_content=target_content,
            purpose=purpose,
        )
    )
    run = RunContrastiveIntentProbe(
        docker_env=runtime,
        session=session,
        included_paths=set(),
        issue=issue,
        contract=contract,
        semantic_adjudicator=semantic_adjudicator,
    )
    authority = issue if oracle_source == "issue_explicit" else contract
    args = RunContrastiveIntentProbeArgs(
        thoughts="",
        probe_name="probe",
        control_compile_command=(
            "c++ /tmp/intent2fix-audit/test/probe_control.cpp "
            "-o /tmp/intent2fix-audit/test/probe_control"
        ),
        control_run_command="/tmp/intent2fix-audit/test/probe_control",
        target_compile_command=(
            "c++ /tmp/intent2fix-audit/test/probe_target.cpp "
            "-o /tmp/intent2fix-audit/test/probe_target"
        ),
        target_run_command="/tmp/intent2fix-audit/test/probe_target",
        oracle=authority,
        oracle_source=oracle_source,
        oracle_reference=oracle_reference or authority,
        distinguishing_input="public formatter with emphasis::bold",
        candidate_prediction=candidate_prediction,
        alternative_prediction=alternative_prediction,
        oracle_mode=oracle_mode,
        purpose=purpose,
    )
    return session, run, args, run.execute(args, file_context=object())


def _session_with_run(
    *,
    control_compile_exit: int = 0,
    control_run_exit: int = 0,
    target_compile_exit: int = 0,
    target_run_exit: int = 3,
):
    session = PatchIntentAuditSession(audit_dir="/tmp/intent2fix-audit/test")
    evidence = session.record(
        "contrastive pair",
        "assertion failed",
        target_run_exit,
        prefix="RUN",
        action="RunContrastiveIntentProbe",
        extra={
            "probe_name": "probe",
            "fixture_path": "/tmp/intent2fix-audit/test/probe_fixture.hpp",
            "fixture_content": "struct input {};",
            "control_path": "/tmp/intent2fix-audit/test/probe_control.cpp",
            "control_content": (
                '#include "probe_fixture.hpp"\nint main() { return 0; }'
            ),
            "target_path": "/tmp/intent2fix-audit/test/probe_target.cpp",
            "target_content": (
                '#include "probe_fixture.hpp"\nint main() { return 3; }'
            ),
            "control_compile_command": (
                "c++ /tmp/intent2fix-audit/test/probe_control.cpp -o "
                "/tmp/intent2fix-audit/test/probe_control"
            ),
            "control_run_command": "/tmp/intent2fix-audit/test/probe_control",
            "target_compile_command": (
                "c++ /tmp/intent2fix-audit/test/probe_target.cpp -o "
                "/tmp/intent2fix-audit/test/probe_target"
            ),
            "target_run_command": "/tmp/intent2fix-audit/test/probe_target",
            "control_compile_exit_code": control_compile_exit,
            "control_run_exit_code": (
                control_run_exit if control_compile_exit == 0 else None
            ),
            "target_compile_exit_code": target_compile_exit,
            "target_run_exit_code": (
                target_run_exit if target_compile_exit == 0 else None
            ),
            "oracle": ISSUE,
            "oracle_source": "issue_explicit",
            "oracle_reference": ISSUE,
            "oracle_mode": "runtime_behavior",
            "distinguishing_input": "public formatter with emphasis::bold",
            "candidate_prediction": "returns an error",
            "alternative_prediction": "formats the value",
            "purpose": (
                "The candidate rejects emphasis::bold while the "
                "Issue-compatible formatter accepts it."
            ),
        },
    )
    return session, evidence.id


def test_underfit_requires_failing_compiled_probe_and_exact_issue_oracle():
    session, run_id = _session_with_run()
    submit = SubmitPatchIntentAudit(
        session=session,
        issue=ISSUE,
        contract="",
        semantic_adjudicator=_admit,
    )

    observation = submit.execute(
        SubmitPatchIntentAuditArgs(
            thoughts="",
            verdict="underfit",
            report="The candidate still rejects the explicitly required call.",
            modified_behavior="Public formatter overload participation",
            evidence_ids=[run_id],
            run_id=run_id,
            oracle_source="issue_explicit",
            oracle_reference=(
                "Calling the public formatter with emphasis::bold must succeed."
            ),
        )
    )

    result = PatchIntentAuditResult.model_validate(
        observation.properties["patch_intent_audit"]
    )
    assert observation.terminal
    assert result.blocking
    assert result.counterexample.run_id == run_id
    assert result.semantic_admission.admitted


def test_underfit_keeps_behavioral_counterexample_when_mechanism_is_wrong():
    session, run_id = _session_with_run()
    observation = SubmitPatchIntentAudit(
        session=session,
        issue=ISSUE,
        contract="",
        semantic_adjudicator=_admit_behavior_without_mechanism,
    ).execute(
        SubmitPatchIntentAuditArgs(
            thoughts="",
            verdict="underfit",
            report="The original mechanism explanation was too specific.",
            evidence_ids=[run_id],
            run_id=run_id,
            oracle_source="issue_explicit",
            oracle_reference=ISSUE,
        )
    )

    result = PatchIntentAuditResult.model_validate(
        observation.properties["patch_intent_audit"]
    )
    assert observation.terminal
    assert result.blocking
    assert "supports only the observed behavioral underfit" in result.report
    assert result.counterexample is not None
    assert result.counterexample.purpose == result.report
    assert not result.semantic_admission.candidate_prediction_matches_observation


def test_underfit_rejects_pair_with_failing_control():
    session, run_id = _session_with_run(control_run_exit=4)
    observation = SubmitPatchIntentAudit(
        session=session,
        issue=ISSUE,
        contract="",
        semantic_adjudicator=_admit,
    ).execute(
        SubmitPatchIntentAuditArgs(
            thoughts="",
            verdict="underfit",
            report="The target failed after an invalid control.",
            evidence_ids=[run_id],
            run_id=run_id,
            oracle_source="issue_explicit",
            oracle_reference=ISSUE,
        )
    )

    assert observation.expect_correction
    assert "control must compile and pass" in observation.message


def test_semantic_admission_rejection_allows_pair_correction():
    session, run_id = _session_with_run()
    session.candidate_patch = "diff --git a/a.hpp b/a.hpp\n"
    observation = SubmitPatchIntentAudit(
        session=session,
        issue=ISSUE,
        contract="",
        semantic_adjudicator=_reject_admission,
    ).execute(
        SubmitPatchIntentAuditArgs(
            thoughts="",
            verdict="underfit",
            report="The target fails while the control passes.",
            evidence_ids=[run_id],
            run_id=run_id,
            oracle_source="issue_explicit",
            oracle_reference=ISSUE,
        )
    )

    assert observation.expect_correction
    assert not observation.terminal
    assert "semantic admission rejected" in observation.message.lower()
    assert "Replace the control/target pair" in observation.message


def test_inconclusive_preserves_issue_relevant_unresolved_discriminator():
    session, run_id = _session_with_run(target_run_exit=0)
    session.semantic_admissions[run_id] = _unresolved_mechanism_admission()
    observation = SubmitPatchIntentAudit(
        session=session,
        issue=ISSUE,
        contract="",
        semantic_adjudicator=lambda _prompt, _schema: None,
    ).execute(
        SubmitPatchIntentAuditArgs(
            thoughts="",
            verdict="inconclusive",
            report="The Issue leaves the observed precedence choice open.",
            evidence_ids=[run_id],
            run_id=run_id,
        )
    )

    result = PatchIntentAuditResult.model_validate(
        observation.properties["patch_intent_audit"]
    )
    assert observation.terminal
    assert not result.blocking
    assert result.counterexample is None
    assert result.unresolved_discriminator is not None
    assert result.unresolved_discriminator.run_id == run_id
    assert "does not choose either outcome" in result.report


def test_semantic_admission_unavailable_is_distinct_terminal_result():
    session, run_id = _session_with_run()
    observation = SubmitPatchIntentAudit(
        session=session,
        issue=ISSUE,
        contract="",
        semantic_adjudicator=_unavailable_admission,
    ).execute(
        SubmitPatchIntentAuditArgs(
            thoughts="",
            verdict="underfit",
            report="The target fails while the control passes.",
            evidence_ids=[run_id],
            run_id=run_id,
            oracle_source="issue_explicit",
            oracle_reference=ISSUE,
        )
    )

    result = PatchIntentAuditResult.model_validate(
        observation.properties["patch_intent_audit"]
    )
    assert observation.terminal
    assert result.verdict == "inconclusive"
    assert not result.blocking
    assert not result.semantic_admission.available
    assert "admission was unavailable" in result.report.lower()


def test_semantic_admission_receives_cited_repository_evidence():
    session, run_id = _session_with_run()
    search = session.record(
        "grep -RIn -- 'formatter' test include",
        "test/format-test.cc:42: formatter(value)",
        0,
        prefix="SEARCH",
        action="ReadOnlySearch",
    )
    prompts = []

    def admit(prompt, schema):
        prompts.append(prompt)
        return _admit(prompt, schema)

    observation = SubmitPatchIntentAudit(
        session=session,
        issue=ISSUE,
        contract="",
        semantic_adjudicator=admit,
    ).execute(
        SubmitPatchIntentAuditArgs(
            thoughts="",
            verdict="underfit",
            report="The target fails while the control passes.",
            evidence_ids=[search.id, run_id],
            run_id=run_id,
            oracle_source="issue_explicit",
            oracle_reference=ISSUE,
        )
    )

    assert observation.terminal
    assert search.id in prompts[0]
    assert "formatter(value)" in prompts[0]


def test_semantic_admission_excludes_prior_run_transcripts():
    session, prior_run_id = _session_with_run()
    session.evidence[0].observation = "PRIOR_RUN_POISON"
    prior_action = session.actions[0]
    run_fields = {
        key: value
        for key, value in prior_action.items()
        if key not in {"action", "command", "exit_code", "evidence_id"}
    }
    current = session.record(
        "current contrastive pair",
        "CURRENT_RUN_TRANSCRIPT",
        3,
        prefix="RUN",
        action="RunContrastiveIntentProbe",
        extra=run_fields,
    )
    search = session.record(
        "grep -RIn -- 'formatter' test include",
        "NORMATIVE_SEARCH_EVIDENCE",
        0,
        prefix="SEARCH",
        action="ReadOnlySearch",
    )
    prompts = []

    def admit(prompt, schema):
        prompts.append(prompt)
        return _admit(prompt, schema)

    observation = SubmitPatchIntentAudit(
        session=session,
        issue=ISSUE,
        contract="",
        semantic_adjudicator=admit,
    ).execute(
        SubmitPatchIntentAuditArgs(
            thoughts="",
            verdict="underfit",
            report="The current target fails while the control passes.",
            evidence_ids=[prior_run_id, search.id, current.id],
            run_id=current.id,
            oracle_source="issue_explicit",
            oracle_reference=ISSUE,
        )
    )

    assert observation.terminal
    assert len(prompts) == 1
    assert '"current_run_id": "RUN-2"' in prompts[0]
    assert "CURRENT_RUN_TRANSCRIPT" in prompts[0]
    assert "NORMATIVE_SEARCH_EVIDENCE" in prompts[0]
    assert "PRIOR_RUN_POISON" not in prompts[0]


def test_underfit_rejects_oracle_invented_from_graph_proximity():
    session, run_id = _session_with_run()
    observation = SubmitPatchIntentAudit(
        session=session,
        issue=ISSUE,
        contract="",
        semantic_adjudicator=_admit,
    ).execute(
        SubmitPatchIntentAuditArgs(
            thoughts="",
            verdict="underfit",
            report="Nearby wrapper also fails.",
            evidence_ids=[run_id],
            run_id=run_id,
            oracle_source="issue_explicit",
            oracle_reference="Every nearby wrapper must support the operation.",
        )
    )

    assert observation.expect_correction
    assert "exact span" in observation.message


def test_runtime_compile_failure_is_inconclusive_not_underfit():
    session, run_id = _session_with_run(target_compile_exit=1)
    observation = SubmitPatchIntentAudit(
        session=session,
        issue=ISSUE,
        contract="",
        semantic_adjudicator=_admit,
    ).execute(
        SubmitPatchIntentAuditArgs(
            thoughts="",
            verdict="underfit",
            report="Probe did not compile.",
            evidence_ids=[run_id],
            run_id=run_id,
            oracle_source="issue_explicit",
            oracle_reference=ISSUE,
        )
    )

    assert observation.expect_correction
    assert "runtime_behavior target must compile" in observation.message
    assert "WriteContrastiveIntentProbe" in observation.message
    assert "empty run_id" in observation.message


def test_explicit_compile_acceptance_failure_can_prove_underfit():
    session, run_id = _session_with_run(target_compile_exit=1)
    session.actions[-1]["oracle_mode"] = "compile_acceptance"
    observation = SubmitPatchIntentAudit(
        session=session,
        issue=ISSUE,
        contract="",
        semantic_adjudicator=_admit,
    ).execute(
        SubmitPatchIntentAuditArgs(
            thoughts="",
            verdict="underfit",
            report=(
                "The Issue-required public expression fails overload resolution."
            ),
            evidence_ids=[run_id],
            run_id=run_id,
            oracle_source="issue_explicit",
            oracle_reference=ISSUE,
        )
    )

    result = PatchIntentAuditResult.model_validate(
        observation.properties["patch_intent_audit"]
    )
    assert observation.terminal
    assert result.blocking
    assert result.counterexample.oracle_mode == "compile_acceptance"
    assert result.counterexample.target_run_exit_code is None


def test_compile_acceptance_cannot_block_when_target_compiles():
    session, run_id = _session_with_run(target_run_exit=0)
    session.actions[-1]["oracle_mode"] = "compile_acceptance"
    observation = SubmitPatchIntentAudit(
        session=session,
        issue=ISSUE,
        contract="",
        semantic_adjudicator=_admit,
    ).execute(
        SubmitPatchIntentAuditArgs(
            thoughts="",
            verdict="underfit",
            report="The expression should not be accepted as a counterexample.",
            evidence_ids=[run_id],
            run_id=run_id,
            oracle_source="issue_explicit",
            oracle_reference=ISSUE,
        )
    )

    assert observation.expect_correction
    assert "requires" in observation.message


def test_no_counterexample_found_is_explicitly_nonblocking():
    result = PatchIntentAuditResult(
        verdict="no_counterexample_found",
        blocking=False,
        report="The inspected path passed; this is not a global proof.",
    )

    assert result.blocking is False


def test_submit_no_counterexample_requires_passing_bound_run_and_oracle():
    session, run_id = _session_with_run(target_run_exit=0)
    submit = SubmitPatchIntentAudit(
        session=session,
        issue=ISSUE,
        contract="",
        semantic_adjudicator=_admit,
    )

    missing_run = submit.execute(
        SubmitPatchIntentAuditArgs(
            thoughts="",
            verdict="no_counterexample_found",
            report="The discriminator passed.",
            evidence_ids=[run_id],
            oracle_source="issue_explicit",
            oracle_reference=ISSUE,
        )
    )
    assert missing_run.expect_correction
    assert "requires a cited discriminating" in missing_run.message

    observation = submit.execute(
        SubmitPatchIntentAuditArgs(
            thoughts="",
            verdict="no_counterexample_found",
            report="The discriminator passed.",
            evidence_ids=[run_id],
            run_id=run_id,
            oracle_source="issue_explicit",
            oracle_reference=ISSUE,
        )
    )
    result = PatchIntentAuditResult.model_validate(
        observation.properties["patch_intent_audit"]
    )
    assert observation.terminal
    assert not result.blocking
    assert result.verdict == "inconclusive"
    assert "passing local probe" in result.report


def test_no_counterexample_pair_also_requires_semantic_admission():
    session, run_id = _session_with_run(target_run_exit=0)
    observation = SubmitPatchIntentAudit(
        session=session,
        issue=ISSUE,
        contract="",
        semantic_adjudicator=_reject_admission,
    ).execute(
        SubmitPatchIntentAuditArgs(
            thoughts="",
            verdict="no_counterexample_found",
            report="Both sides pass the attempted discriminator.",
            evidence_ids=[run_id],
            run_id=run_id,
            oracle_source="issue_explicit",
            oracle_reference=ISSUE,
        )
    )

    assert observation.expect_correction
    assert "semantic admission rejected" in observation.message.lower()


def test_auditor_prompt_separates_behavior_from_structural_review():
    source = " ".join(inspect.getsource(PatchIntentAuditor).split())

    assert "Graph adjacency is navigation only" in source
    assert "plausible Issue-compatible alternative" in source
    assert "Normative behavioral wording in the selected Issue or" in source
    assert "The alternative is not the rejected pre-patch behavior" in source
    assert "must independently satisfy the Issue on the motivating witness" in source
    assert "valid overlap case where both are" in source
    assert "Both paths must be complete and independently viable" in source
    assert "remove one member of a multi-operation protocol" in source
    assert "observable result of the specifically required mechanism" in source
    assert "does not donate its other rules" in source
    assert "cannot silently add an unstated fallback or precedence policy" in source
    assert "direct realization of the required customization" in source
    assert "candidate patch visibly attempts the same Issue-named capability" in source
    assert "An isolated question, analogy, or author hypothesis" in source
    assert "viable candidates reach that mechanism's normal selection process" in source
    assert "This does not borrow precedence from an analogy" in source
    assert "control must directly compile or execute that exact construct" in source
    assert "hand-written approximation" in source
    assert "probe's self-authored assertion or a descriptive" in source
    assert "Submit `no_counterexample_found` only after executing a pair" in source
    assert "A comparison against rejected pre-patch" in source
    assert "submit `inconclusive`" in source
    assert "compile_acceptance" in source
    assert "no clean-baseline execution channel" in source
    assert "Patch Surface Reviewer owns those checks" in source
    assert "Repair-trajectory AuditLeads" in source
    assert "Never cite a lead or its EV-N identifier as oracle" in source
    assert "non-authoritative investigation leads" in source
    assert "Investigation order is mandatory" in source
    assert "public path actually changed by the patch" in source
    assert "Do not use an untouched" in source
    assert "unchanged public entry is still a valid" in source
    assert "candidate-participation and selection" in source
    assert "only as a fallback is not automatically equivalent" in source
    assert "derive precedence only from a cited normative public contract" in source
    assert "mentioned only as an author hypothesis" in source
    assert "preserve that pair's fixture" in source
    assert "This is an oracle correction" in source
    assert "Change the input only if" in source
    assert "pending oracle correction must be resolved" in source
    assert "Submit the admitted correction" in source
    assert "One step remains while a correctable semantic" in source


def test_failed_control_does_not_falsify_precedence_hypothesis():
    _, _, _, observation = _execute_probe_with_admission(
        _admit,
        runtime=_FakeAuditRuntime(control_compile_exit=1),
    )

    assert "does not falsify the behavioral hypothesis" in observation.message
    assert "both complete competing mechanisms valid at once" in observation.message


def test_mechanism_mismatch_requests_complete_precedence_overlap():
    _, _, _, observation = _execute_probe_with_admission(
        _admit_behavior_without_mechanism,
        runtime=_FakeAuditRuntime(target_run_exit=0),
    )

    assert observation.expect_correction
    assert "keep each complete mechanism independently viable" in observation.message
    assert "Removing only part of one mechanism" in observation.message


def test_oracle_correction_exposes_bounded_selected_authority():
    session, run_id = _session_with_run()
    observation = SubmitPatchIntentAudit(
        session=session,
        issue=ISSUE,
        contract="",
        semantic_adjudicator=_admit,
    ).execute(
        SubmitPatchIntentAuditArgs(
            thoughts="",
            verdict="underfit",
            report="The candidate rejects the public call.",
            evidence_ids=[run_id],
            run_id=run_id,
            oracle_source="issue_explicit",
            oracle_reference="invented requirement",
        )
    )

    assert observation.expect_correction
    assert "<selected_authority>" in observation.message
    assert ISSUE in observation.message


def test_probe_commands_cannot_mask_compile_or_run_exit_status():
    assert _single_command_rejection("c++ probe.cpp -o probe; echo $?")
    assert _single_command_rejection("./probe || true")
    assert not _single_command_rejection("c++ /tmp/probe.cpp -o /tmp/probe")


def test_probe_commands_bind_compiler_source_output_and_executed_binary():
    audit_dir = "/tmp/intent2fix-audit/test"
    source = audit_dir + "/probe.cpp"
    output, error = _probe_command_binding(
        compile_command=f"c++ {source} -o {audit_dir}/probe",
        run_command=f"{audit_dir}/probe",
        source_path=source,
        audit_dir=audit_dir,
    )

    assert not error
    assert output == audit_dir + "/probe"
    _, fake_compile = _probe_command_binding(
        compile_command=f"/bin/true {source} -o {audit_dir}/probe",
        run_command=f"{audit_dir}/probe",
        source_path=source,
        audit_dir=audit_dir,
    )
    assert "C/C++ compiler" in fake_compile
    _, wrong_binary = _probe_command_binding(
        compile_command=f"c++ {source} -o {audit_dir}/probe",
        run_command=f"{audit_dir}/other",
        source_path=source,
        audit_dir=audit_dir,
    )
    assert "binary produced" in wrong_binary


def test_probe_path_rejects_shell_metacharacters():
    _, error = _validated_probe_path("probe;touch-owned.cpp")

    assert error


def test_probe_writer_rejects_nested_probe_name():
    observation = WriteContrastiveIntentProbe(
        docker_env=_FakeAuditRuntime(),
        session=PatchIntentAuditSession(audit_dir="/tmp/intent2fix-audit/test"),
    ).execute(
        WriteContrastiveIntentProbeArgs(
            thoughts="",
            probe_name="nested/probe",
            fixture_content="struct input {};",
            control_content="int main() { return 0; }",
            target_content="int main() { return 1; }",
        )
    )

    assert observation.expect_correction
    assert "without directories" in observation.message


def test_worktree_fingerprint_detects_new_untracked_paths():
    class Runtime:
        def __init__(self):
            self.untracked = ""

        def run_bash(self, command, **kwargs):
            if command.startswith("tmp=/tmp/intent2fix-tracked-diff"):
                return "base-fingerprint  -\n", 0
            if command == "git ls-files --others --exclude-standard -z":
                return self.untracked, 0
            raise AssertionError(command)

    runtime = Runtime()
    before, _, error = _protected_worktree_fingerprint(runtime, set())
    assert not error

    runtime.untracked = "audit-leak.txt\0"
    after, paths, error = _protected_worktree_fingerprint(runtime, set())

    assert not error
    assert paths == {"audit-leak.txt"}
    assert after != before


def test_auditor_metrics_distinguish_steps_tools_and_corrections():
    auditor = PatchIntentAuditor(
        agent=object(),
        session=PatchIntentAuditSession(),
        repository=object(),
        docker_env=object(),
        included_paths=set(),
        max_steps=40,
    )
    auditor.steps_used = 11
    auditor.tool_calls_attempted = 14
    auditor.rejected_tool_calls = 2
    auditor.no_action_retries = 0
    auditor.termination_reason = "submitted"

    assert auditor.metrics == {
        "completion_steps": 11,
        "tool_calls_attempted": 14,
        "rejected_tool_calls": 2,
        "no_action_retries": 0,
        "max_steps": 40,
        "termination_reason": "submitted",
    }


def test_probe_actions_execute_only_the_bound_isolated_binary():
    runtime = _FakeAuditRuntime()
    session = PatchIntentAuditSession(
        audit_dir="/tmp/intent2fix-audit/test"
    )
    write = WriteContrastiveIntentProbe(docker_env=runtime, session=session)
    written = write.execute(
        WriteContrastiveIntentProbeArgs(
            thoughts="",
            probe_name="probe",
            fixture_content="struct input {};",
            control_content="int main() { return 0; }",
            target_content="int main() { return 3; }",
            purpose="exercise the Issue-required call",
        )
    )
    assert not written.expect_correction
    assert set(runtime.writes) == {
        "/tmp/intent2fix-audit/test/probe_fixture.hpp",
        "/tmp/intent2fix-audit/test/probe_control.cpp",
        "/tmp/intent2fix-audit/test/probe_target.cpp",
    }

    run = RunContrastiveIntentProbe(
        docker_env=runtime,
        session=session,
        included_paths=set(),
        issue=ISSUE,
    )
    run_args = RunContrastiveIntentProbeArgs(
        thoughts="",
        probe_name="probe",
        control_compile_command=(
            "c++ /tmp/intent2fix-audit/test/probe_control.cpp "
            "-o /tmp/intent2fix-audit/test/probe_control"
        ),
        control_run_command="/tmp/intent2fix-audit/test/probe_control",
        target_compile_command=(
            "c++ /tmp/intent2fix-audit/test/probe_target.cpp "
            "-o /tmp/intent2fix-audit/test/probe_target"
        ),
        target_run_command="/tmp/intent2fix-audit/test/probe_target",
        oracle=ISSUE,
        oracle_source="issue_explicit",
        oracle_reference=ISSUE,
        oracle_mode="runtime_behavior",
        distinguishing_input="public formatter with emphasis::bold",
        candidate_prediction="returns an error",
        alternative_prediction="formats the value",
        purpose="distinguish public formatter acceptance",
    )
    observation = run.execute(
        run_args,
        file_context=object(),
    )

    assert observation.properties["control_run_exit_code"] == 0
    assert observation.properties["target_run_exit_code"] == 3
    assert session.actions[-1]["action"] == "RunContrastiveIntentProbe"

    duplicate = run.execute(
        run_args.model_copy(
            update={
                "candidate_prediction": "fails with status 3",
                "alternative_prediction": "returns status 0",
                "purpose": "reworded conclusion for the same executable pair",
            }
        ),
        file_context=object(),
    )

    assert duplicate.expect_correction
    assert "already ran as RUN-1" in duplicate.message
    assert "WriteContrastiveIntentProbe" in duplicate.message
    assert len(session.evidence) == 1

    WriteContrastiveIntentProbe(
        docker_env=runtime,
        session=session,
    ).execute(
        WriteContrastiveIntentProbeArgs(
            thoughts="",
            probe_name="renamed",
            fixture_content="struct input {};",
            control_content="int main() { return 0; }",
            target_content="int main() { return 3; }",
        )
    )
    renamed = run.execute(
        run_args.model_copy(
            update={
                "probe_name": "renamed",
                "control_compile_command": (
                    "c++ /tmp/intent2fix-audit/test/renamed_control.cpp "
                    "-o /tmp/intent2fix-audit/test/renamed_control"
                ),
                "control_run_command": (
                    "/tmp/intent2fix-audit/test/renamed_control"
                ),
                "target_compile_command": (
                    "c++ /tmp/intent2fix-audit/test/renamed_target.cpp "
                    "-o /tmp/intent2fix-audit/test/renamed_target"
                ),
                "target_run_command": "/tmp/intent2fix-audit/test/renamed_target",
                "purpose": "reworded renamed clone",
            }
        ),
        file_context=object(),
    )

    assert renamed.expect_correction
    assert "already ran as RUN-1" in renamed.message
    assert len(session.evidence) == 1


def test_probe_rejects_nonverbatim_oracle_before_execution():
    runtime = _FakeAuditRuntime()
    session = PatchIntentAuditSession(audit_dir="/tmp/intent2fix-audit/test")
    WriteContrastiveIntentProbe(
        docker_env=runtime,
        session=session,
    ).execute(
        WriteContrastiveIntentProbeArgs(
            thoughts="",
            probe_name="probe",
            fixture_content="struct input {};",
            control_content="int main() { return 0; }",
            target_content="int main() { return 3; }",
        )
    )

    observation = RunContrastiveIntentProbe(
        docker_env=runtime,
        session=session,
        included_paths=set(),
        issue=ISSUE,
    ).execute(
        RunContrastiveIntentProbeArgs(
            thoughts="",
            probe_name="probe",
            control_compile_command=(
                "c++ /tmp/intent2fix-audit/test/probe_control.cpp "
                "-o /tmp/intent2fix-audit/test/probe_control"
            ),
            control_run_command="/tmp/intent2fix-audit/test/probe_control",
            target_compile_command=(
                "c++ /tmp/intent2fix-audit/test/probe_target.cpp "
                "-o /tmp/intent2fix-audit/test/probe_target"
            ),
            target_run_command="/tmp/intent2fix-audit/test/probe_target",
            oracle="The formatter should work.",
            oracle_source="issue_explicit",
            oracle_reference="The formatter should work.",
            distinguishing_input="emphasis::bold",
            candidate_prediction="rejects",
            alternative_prediction="formats",
            purpose="distinguish public formatter acceptance",
        ),
        file_context=object(),
    )

    assert observation.expect_correction
    assert "before this pair can run" in observation.message
    assert not session.evidence


def test_complete_run_performs_and_caches_semantic_admission():
    runtime = _FakeAuditRuntime()
    session = PatchIntentAuditSession(audit_dir="/tmp/intent2fix-audit/test")
    WriteContrastiveIntentProbe(
        docker_env=runtime,
        session=session,
    ).execute(
        WriteContrastiveIntentProbeArgs(
            thoughts="",
            probe_name="probe",
            fixture_content="struct input {};",
            control_content="int main() { return 0; }",
            target_content="int main() { return 3; }",
            purpose="distinguish public formatter acceptance",
        )
    )
    prompts = []

    def admit(prompt, schema):
        prompts.append(prompt)
        return _admit(prompt, schema)

    run = RunContrastiveIntentProbe(
        docker_env=runtime,
        session=session,
        included_paths=set(),
        issue=ISSUE,
        semantic_adjudicator=admit,
    )
    observation = run.execute(
        RunContrastiveIntentProbeArgs(
            thoughts="",
            probe_name="probe",
            control_compile_command=(
                "c++ /tmp/intent2fix-audit/test/probe_control.cpp "
                "-o /tmp/intent2fix-audit/test/probe_control"
            ),
            control_run_command="/tmp/intent2fix-audit/test/probe_control",
            target_compile_command=(
                "c++ /tmp/intent2fix-audit/test/probe_target.cpp "
                "-o /tmp/intent2fix-audit/test/probe_target"
            ),
            target_run_command="/tmp/intent2fix-audit/test/probe_target",
            oracle=ISSUE,
            oracle_source="issue_explicit",
            oracle_reference=ISSUE,
            distinguishing_input="public formatter with emphasis::bold",
            candidate_prediction="returns an error",
            alternative_prediction="formats the value",
            purpose="distinguish public formatter acceptance",
        ),
        file_context=object(),
    )

    assert not observation.expect_correction
    assert "Semantic admission accepted" in observation.message
    assert len(prompts) == 1
    assert "top-level schema fields" in prompts[0]
    assert "deterministic language-standard semantics" in prompts[0]
    assert "narrow normative requirement" in prompts[0]
    assert "does not import an unstated fallback or precedence rule" in prompts[0]
    assert "leaves precedence open" in prompts[0]
    assert "bug-report wording at the requested-capability level" in prompts[0]
    assert "candidate patch visibly attempts that exact Issue-named capability" in prompts[0]
    assert "cannot invent a capability or observable outcome absent" in prompts[0]
    assert "An isolated question, analogy, or author hypothesis" in prompts[0]
    assert "admitting its viable candidates" in prompts[0]
    assert "suppresses those candidates before selection" in prompts[0]
    assert "sole normative source" in prompts[0]
    assert "unselected Issue or contract" in prompts[0]
    assert "possible implementation or author hypothesis" in prompts[0]
    assert "generic familiarity with a project hook cannot invent" in prompts[0]
    assert "weaker fallback approximation" in prompts[0]
    assert "alternative must itself satisfy the selected authority" in prompts[0]
    assert "rejected pre-patch behavior" in prompts[0]
    assert '"oracle_source": "issue_explicit"' in prompts[0]
    run_id = observation.properties["evidence_id"]
    submitted = SubmitPatchIntentAudit(
        session=session,
        issue=ISSUE,
        contract="",
        semantic_adjudicator=admit,
    ).execute(
        SubmitPatchIntentAuditArgs(
            thoughts="",
            verdict="underfit",
            report="The target fails while the control passes.",
            evidence_ids=[run_id],
            run_id=run_id,
            oracle_source="issue_explicit",
            oracle_reference=ISSUE,
        )
    )

    assert submitted.terminal
    assert len(prompts) == 1
    result = PatchIntentAuditResult.model_validate(
        submitted.properties["patch_intent_audit"]
    )
    assert result.report == "distinguish public formatter acceptance"


def test_complete_run_preserves_unresolved_discriminator_without_retry():
    def unresolved(_prompt, schema):
        assert schema is PatchIntentSemanticAdmission
        return _unresolved_mechanism_admission()

    session, _, _, observation = _execute_probe_with_admission(
        unresolved,
        runtime=_FakeAuditRuntime(target_run_exit=0),
    )

    assert not observation.expect_correction
    assert "valid Issue-relevant mechanism discriminator" in observation.message
    run_id = observation.properties["evidence_id"]
    assert session.semantic_admissions[run_id].authority_identifies_disputed_dimension


def test_semantic_admission_receives_selected_accepted_contract_authority():
    prompts = []
    contract = "The public operation must preserve insertion order."

    def admit(prompt, schema):
        prompts.append(prompt)
        return _admit(prompt, schema)

    _execute_probe_with_admission(
        admit,
        issue="The ordering behavior is unclear.",
        contract=contract,
        oracle_source="accepted_contract",
        oracle_reference=contract,
    )

    assert len(prompts) == 1
    assert '"oracle_source": "accepted_contract"' in prompts[0]
    assert f'"selected_authority": "{contract}"' in prompts[0]
    assert f'"oracle_reference": "{contract}"' in prompts[0]
    assert "The ordering behavior is unclear." not in prompts[0]


def test_complete_run_reports_semantic_rejection_immediately():
    session, _, _, observation = _execute_probe_with_admission(
        _reject_admission
    )

    assert observation.expect_correction
    assert "Semantic admission rejected" in observation.message
    run_id = observation.properties["evidence_id"]
    assert not session.semantic_admissions[run_id].admitted


def test_complete_run_rejects_internally_inconsistent_admission():
    def inconsistent_admission(_prompt, schema):
        assert schema is PatchIntentSemanticAdmission
        return PatchIntentSemanticAdmission(
            admitted=True,
            authority_entails_target_expectation=True,
            authority_identifies_disputed_dimension=True,
            pair_exposes_observable_choice=True,
            target_distinguishes_predictions=False,
            pair_is_comparable=True,
            target_audits_modified_mechanism=True,
            candidate_prediction_matches_observation=True,
            reason="The prose describes two different outcomes.",
        )

    session, _, _, observation = _execute_probe_with_admission(
        inconsistent_admission
    )

    assert observation.expect_correction
    assert "Semantic admission rejected" in observation.message
    run_id = observation.properties["evidence_id"]
    assert not session.semantic_admissions[run_id].admitted
    assert (
        "target_distinguishes_predictions"
        in session.semantic_admissions[run_id].reason
    )


def test_passing_target_cannot_admit_candidate_accepting_oracle():
    def inconsistent_admission(_prompt, schema):
        assert schema is PatchIntentSemanticAdmission
        return PatchIntentSemanticAdmission(
            admitted=True,
            authority_entails_target_expectation=True,
            authority_identifies_disputed_dimension=True,
            pair_exposes_observable_choice=True,
            target_distinguishes_predictions=False,
            pair_is_comparable=True,
            target_audits_modified_mechanism=True,
            candidate_prediction_matches_observation=True,
            reason="The target passes, but it encodes the candidate outcome.",
        )

    session, run, args, observation = _execute_probe_with_admission(
        inconsistent_admission,
        runtime=_FakeAuditRuntime(target_run_exit=0),
        target_content="int main() { return 0; }",
        candidate_prediction="formats the value",
        alternative_prediction="returns an error",
    )

    assert observation.expect_correction
    assert "Semantic admission rejected" in observation.message
    assert "keep the same shared fixture" in observation.message
    assert "Compare the frozen target check" in observation.message
    assert "Correct whichever is inconsistent" in observation.message
    assert "Do not replace an overlap discriminator" in observation.message
    run_id = observation.properties["evidence_id"]
    assert f"corrects_run_id set to {run_id}" in observation.message
    assert not session.semantic_admissions[run_id].admitted
    assert session.pending_correction_run_id == run_id
    assert session.latest_correction_run_id == ""
    stale_correction = session.record(
        "older correction",
        "provider unavailable",
        0,
        prefix="RUN",
        action="RunContrastiveIntentProbe",
        extra={"corrects_run_id": run_id},
    )
    session.semantic_admissions[stale_correction.id] = _unavailable_admission(
        "",
        PatchIntentSemanticAdmission,
    )

    premature = SubmitPatchIntentAudit(
        session=session,
        issue=ISSUE,
        contract="",
        semantic_adjudicator=inconsistent_admission,
    ).execute(
        SubmitPatchIntentAuditArgs(
            thoughts="",
            verdict="inconclusive",
            report="No discriminator was found.",
            evidence_ids=[run_id],
        )
    )

    assert premature.expect_correction
    assert not premature.terminal
    assert run_id in premature.message
    assert "correctable semantic rejection remains pending" in premature.message

    for verdict in ("underfit", "no_counterexample_found"):
        bypass = SubmitPatchIntentAudit(
            session=session,
            issue=ISSUE,
            contract="",
            semantic_adjudicator=_admit,
        ).execute(
            SubmitPatchIntentAuditArgs(
                thoughts="",
                verdict=verdict,
                report="Reuse an older RUN instead of correcting the pair.",
                evidence_ids=[run_id],
                run_id=run_id,
                oracle_source="issue_explicit",
                oracle_reference=ISSUE,
            )
        )

        assert bypass.expect_correction
        assert not bypass.terminal
        assert "before submitting any verdict" in bypass.message

    WriteContrastiveIntentProbe(
        docker_env=run._docker_env,
        session=session,
    ).execute(
        WriteContrastiveIntentProbeArgs(
            thoughts="",
            probe_name="unrelated",
            fixture_content="struct input {};",
            control_content="int main() { return 0; }",
            target_content="int main() { return 4; }",
        )
    )
    unrelated_args = args.model_copy(
        update={
            "probe_name": "unrelated",
            "control_compile_command": (
                "c++ /tmp/intent2fix-audit/test/unrelated_control.cpp "
                "-o /tmp/intent2fix-audit/test/unrelated_control"
            ),
            "control_run_command": (
                "/tmp/intent2fix-audit/test/unrelated_control"
            ),
            "target_compile_command": (
                "c++ /tmp/intent2fix-audit/test/unrelated_target.cpp "
                "-o /tmp/intent2fix-audit/test/unrelated_target"
            ),
            "target_run_command": "/tmp/intent2fix-audit/test/unrelated_target",
        }
    )
    unrelated = run.execute(unrelated_args, file_context=object())

    assert unrelated.expect_correction
    assert f"corrects_run_id={run_id}" in unrelated.message
    assert session.pending_correction_run_id == run_id

    def reject_claimed_lineage(prompt, schema):
        assert schema is PatchIntentSemanticAdmission
        assert '"corrects_run_id": "RUN-1"' in prompt
        assert '"correction_of": {' in prompt
        assert '"fixture": "struct input {};"' in prompt
        return PatchIntentSemanticAdmission(
            admitted=False,
            authority_entails_target_expectation=True,
            authority_identifies_disputed_dimension=True,
            pair_exposes_observable_choice=True,
            target_distinguishes_predictions=True,
            pair_is_comparable=False,
            target_audits_modified_mechanism=True,
            candidate_prediction_matches_observation=True,
            reason="The claimed correction replaced the original fixture.",
        )

    WriteContrastiveIntentProbe(
        docker_env=run._docker_env,
        session=session,
    ).execute(
        WriteContrastiveIntentProbeArgs(
            thoughts="",
            probe_name="probe",
            fixture_content="struct different_input {};",
            control_content="int main() { return 0; }",
            target_content="int main() { return 4; }",
        )
    )
    corrected_args = args.model_copy(
        update={
            "corrects_run_id": run_id,
            "purpose": "correct the frozen target/prediction mismatch",
        }
    )
    false_lineage = RunContrastiveIntentProbe(
        docker_env=run._docker_env,
        session=session,
        included_paths=set(),
        issue=ISSUE,
        semantic_adjudicator=reject_claimed_lineage,
    ).execute(corrected_args, file_context=object())

    assert false_lineage.expect_correction
    assert session.pending_correction_run_id == run_id
    assert session.latest_correction_run_id == false_lineage.properties["evidence_id"]

    WriteContrastiveIntentProbe(
        docker_env=run._docker_env,
        session=session,
    ).execute(
        WriteContrastiveIntentProbeArgs(
            thoughts="",
            probe_name="probe",
            fixture_content="struct input {};",
            control_content="int main() { return 0; }",
            target_content="int main() { return 4; }",
        )
    )
    corrected = RunContrastiveIntentProbe(
        docker_env=run._docker_env,
        session=session,
        included_paths=set(),
        issue=ISSUE,
        semantic_adjudicator=_admit,
    ).execute(corrected_args, file_context=object())

    assert not corrected.expect_correction
    assert "Semantic admission accepted" in corrected.message
    corrected_run_id = corrected.properties["evidence_id"]
    assert session.pending_correction_run_id == run_id
    assert session.latest_correction_run_id == corrected_run_id
    assert session.resolved_correction_run_id == corrected_run_id

    submitted = SubmitPatchIntentAudit(
        session=session,
        issue=ISSUE,
        contract="",
        semantic_adjudicator=_admit,
    ).execute(
        SubmitPatchIntentAuditArgs(
            thoughts="",
            verdict="no_counterexample_found",
            report="The corrected discriminator passes.",
            evidence_ids=[corrected_run_id],
            run_id=corrected_run_id,
            oracle_source="issue_explicit",
            oracle_reference=ISSUE,
        )
    )

    assert submitted.terminal

    stale_submission = SubmitPatchIntentAudit(
        session=session,
        issue=ISSUE,
        contract="",
        semantic_adjudicator=_admit,
    ).execute(
        SubmitPatchIntentAuditArgs(
            thoughts="",
            verdict="inconclusive",
            report="Use an outage that predates the admitted correction.",
            evidence_ids=[stale_correction.id],
            run_id=stale_correction.id,
        )
    )

    assert stale_submission.expect_correction
    assert not stale_submission.terminal
    assert corrected_run_id in stale_submission.message


def test_compile_failure_cannot_admit_candidate_accepting_oracle():
    def inconsistent_admission(_prompt, schema):
        assert schema is PatchIntentSemanticAdmission
        return PatchIntentSemanticAdmission(
            admitted=True,
            authority_entails_target_expectation=True,
            authority_identifies_disputed_dimension=True,
            pair_exposes_observable_choice=True,
            target_distinguishes_predictions=False,
            pair_is_comparable=True,
            target_audits_modified_mechanism=True,
            candidate_prediction_matches_observation=True,
            reason="The compile failure matches the candidate, not the alternative.",
        )

    session, _, _, observation = _execute_probe_with_admission(
        inconsistent_admission,
        runtime=_FakeAuditRuntime(target_compile_exit=1),
        oracle_mode="compile_acceptance",
        target_content="int main() { missing_public_expression(); }",
        candidate_prediction="fails to compile",
        alternative_prediction="compiles successfully",
    )

    assert observation.expect_correction
    assert "Semantic admission rejected" in observation.message
    run_id = observation.properties["evidence_id"]
    assert not session.semantic_admissions[run_id].admitted


def test_unavailable_run_admission_cannot_be_bypassed_as_inconclusive():
    session, _, _, observation = _execute_probe_with_admission(
        _unavailable_admission
    )
    run_id = observation.properties["evidence_id"]

    submitted = SubmitPatchIntentAudit(
        session=session,
        issue=ISSUE,
        contract="",
        semantic_adjudicator=_admit,
    ).execute(
        SubmitPatchIntentAuditArgs(
            thoughts="",
            verdict="inconclusive",
            report="No decisive pair was found.",
            evidence_ids=[run_id],
        )
    )

    result = PatchIntentAuditResult.model_validate(
        submitted.properties["patch_intent_audit"]
    )
    assert submitted.terminal
    assert result.verdict == "inconclusive"
    assert not result.semantic_admission.available
    assert "admission was unavailable" in result.report.lower()


def test_current_correction_admission_outage_can_end_pending_audit():
    runtime = _FakeAuditRuntime()
    session = PatchIntentAuditSession(audit_dir="/tmp/intent2fix-audit/test")
    original = session.record(
        "original pair",
        "target accepted candidate",
        0,
        prefix="RUN",
        action="RunContrastiveIntentProbe",
    )
    session.pending_correction_run_id = original.id
    WriteContrastiveIntentProbe(
        docker_env=runtime,
        session=session,
    ).execute(
        WriteContrastiveIntentProbeArgs(
            thoughts="",
            probe_name="probe",
            fixture_content="struct input {};",
            control_content="int main() { return 0; }",
            target_content="int main() { return 3; }",
            purpose="correct the target expectation",
        )
    )
    observation = RunContrastiveIntentProbe(
        docker_env=runtime,
        session=session,
        included_paths=set(),
        issue=ISSUE,
        semantic_adjudicator=_unavailable_admission,
    ).execute(
        RunContrastiveIntentProbeArgs(
            thoughts="",
            probe_name="probe",
            control_compile_command=(
                "c++ /tmp/intent2fix-audit/test/probe_control.cpp "
                "-o /tmp/intent2fix-audit/test/probe_control"
            ),
            control_run_command="/tmp/intent2fix-audit/test/probe_control",
            target_compile_command=(
                "c++ /tmp/intent2fix-audit/test/probe_target.cpp "
                "-o /tmp/intent2fix-audit/test/probe_target"
            ),
            target_run_command="/tmp/intent2fix-audit/test/probe_target",
            oracle=ISSUE,
            oracle_source="issue_explicit",
            oracle_reference=ISSUE,
            distinguishing_input="public formatter with emphasis::bold",
            candidate_prediction="returns an error",
            alternative_prediction="formats the value",
            purpose="correct the target expectation",
            corrects_run_id=original.id,
        ),
        file_context=object(),
    )
    correction_run_id = observation.properties["evidence_id"]
    outage = session.semantic_admissions[correction_run_id]

    assert "Submit inconclusive" in observation.message
    assert f"run_id={correction_run_id}" in observation.message

    missing_provenance = SubmitPatchIntentAudit(
        session=session,
        issue=ISSUE,
        contract="",
        semantic_adjudicator=_unavailable_admission,
    ).execute(
        SubmitPatchIntentAuditArgs(
            thoughts="",
            verdict="inconclusive",
            report="The correction could not be adjudicated.",
            evidence_ids=[original.id],
            run_id=correction_run_id,
        )
    )

    assert missing_provenance.expect_correction
    assert not missing_provenance.terminal
    assert "include it in evidence_ids" in missing_provenance.message

    submitted = SubmitPatchIntentAudit(
        session=session,
        issue=ISSUE,
        contract="",
        semantic_adjudicator=_unavailable_admission,
    ).execute(
        SubmitPatchIntentAuditArgs(
            thoughts="",
            verdict="inconclusive",
            report="The correction could not be adjudicated.",
            evidence_ids=[correction_run_id],
            run_id=correction_run_id,
        )
    )

    assert submitted.terminal
    result = PatchIntentAuditResult.model_validate(
        submitted.properties["patch_intent_audit"]
    )
    assert result.semantic_admission == outage
    assert "Semantic admission was unavailable" in result.report


def test_superseded_correction_outage_cannot_end_pending_audit():
    session = PatchIntentAuditSession(audit_dir="/tmp/intent2fix-audit/test")
    original = session.record(
        "original pair",
        "target accepted candidate",
        0,
        prefix="RUN",
        action="RunContrastiveIntentProbe",
    )
    session.pending_correction_run_id = original.id
    stale = session.record(
        "first correction",
        "provider unavailable",
        0,
        prefix="RUN",
        action="RunContrastiveIntentProbe",
        extra={"corrects_run_id": original.id},
    )
    session.semantic_admissions[stale.id] = _unavailable_admission(
        "",
        PatchIntentSemanticAdmission,
    )
    latest = session.record(
        "newer correction",
        "available rejection",
        0,
        prefix="RUN",
        action="RunContrastiveIntentProbe",
        extra={"corrects_run_id": original.id},
    )
    session.semantic_admissions[latest.id] = _reject_admission(
        "",
        PatchIntentSemanticAdmission,
    )
    session.latest_correction_run_id = latest.id

    submitted = SubmitPatchIntentAudit(
        session=session,
        issue=ISSUE,
        contract="",
        semantic_adjudicator=_admit,
    ).execute(
        SubmitPatchIntentAuditArgs(
            thoughts="",
            verdict="inconclusive",
            report="Use the stale outage.",
            evidence_ids=[stale.id],
            run_id=stale.id,
        )
    )

    assert submitted.expect_correction
    assert not submitted.terminal
    assert "correctable semantic rejection remains pending" in submitted.message


def test_unrelated_admission_outage_does_not_override_selected_valid_run():
    session, _, _, observation = _execute_probe_with_admission(_admit)
    run_id = observation.properties["evidence_id"]
    session.semantic_admissions["RUN-2"] = _unavailable_admission(
        "",
        PatchIntentSemanticAdmission,
    )

    submitted = SubmitPatchIntentAudit(
        session=session,
        issue=ISSUE,
        contract="",
        semantic_adjudicator=_admit,
    ).execute(
        SubmitPatchIntentAuditArgs(
            thoughts="",
            verdict="underfit",
            report="The selected pair remains decisive.",
            evidence_ids=[run_id],
            run_id=run_id,
            oracle_source="issue_explicit",
            oracle_reference=ISSUE,
        )
    )

    result = PatchIntentAuditResult.model_validate(
        submitted.properties["patch_intent_audit"]
    )
    assert submitted.terminal
    assert result.verdict == "underfit"
    assert result.semantic_admission.available


def test_step_budget_fallback_preserves_semantic_admission_outage():
    session = PatchIntentAuditSession(audit_dir="/tmp/intent2fix-audit/test")

    class OutageAgent:
        def run(self, _node):
            if session.semantic_admissions:
                return
            evidence = session.record(
                "contrastive pair",
                "provider unavailable",
                3,
                prefix="RUN",
                action="RunContrastiveIntentProbe",
            )
            session.semantic_admissions[evidence.id] = _unavailable_admission(
                "",
                PatchIntentSemanticAdmission,
            )

    audit = PatchIntentAuditor(
        agent=OutageAgent(),
        session=session,
        repository=object(),
        docker_env=object(),
        included_paths=set(),
        max_steps=2,
    )
    result = audit.audit(
        issue=ISSUE,
        repair_intent_contract="",
        patch="diff --git a/a.hpp b/a.hpp\n",
        file_context=FileContext(repo=None),
    )

    assert result.verdict == "inconclusive"
    assert not result.semantic_admission.available
    assert "step budget" in result.report


def test_step_budget_pending_correction_ignores_unrelated_outage():
    session = PatchIntentAuditSession(audit_dir="/tmp/intent2fix-audit/test")

    class PendingAgent:
        def run(self, _node):
            if session.actions:
                return
            original = session.record(
                "rejected pair",
                "target encoded candidate outcome",
                0,
                prefix="RUN",
                action="RunContrastiveIntentProbe",
            )
            session.pending_correction_run_id = original.id
            unrelated = session.record(
                "independent pair",
                "provider unavailable",
                0,
                prefix="RUN",
                action="RunContrastiveIntentProbe",
            )
            session.semantic_admissions[unrelated.id] = _unavailable_admission(
                "",
                PatchIntentSemanticAdmission,
            )

    result = PatchIntentAuditor(
        agent=PendingAgent(),
        session=session,
        repository=object(),
        docker_env=object(),
        included_paths=set(),
        max_steps=2,
    ).audit(
        issue=ISSUE,
        repair_intent_contract="",
        patch="diff --git a/a.hpp b/a.hpp\n",
        file_context=FileContext(repo=None),
    )

    assert result.verdict == "inconclusive"
    assert result.semantic_admission is None
    assert "unresolved semantic correction from RUN-1" in result.report


def test_step_budget_pending_correction_uses_only_bound_outage():
    session = PatchIntentAuditSession(audit_dir="/tmp/intent2fix-audit/test")

    class CorrectionOutageAgent:
        def run(self, _node):
            if session.actions:
                return
            original = session.record(
                "rejected pair",
                "target encoded candidate outcome",
                0,
                prefix="RUN",
                action="RunContrastiveIntentProbe",
            )
            session.pending_correction_run_id = original.id
            unrelated = session.record(
                "independent pair",
                "historical outage",
                0,
                prefix="RUN",
                action="RunContrastiveIntentProbe",
            )
            session.semantic_admissions[unrelated.id] = _unavailable_admission(
                "",
                PatchIntentSemanticAdmission,
            ).model_copy(update={"reason": "historical outage"})
            correction = session.record(
                "corrected pair",
                "current outage",
                0,
                prefix="RUN",
                action="RunContrastiveIntentProbe",
                extra={"corrects_run_id": original.id},
            )
            session.semantic_admissions[correction.id] = _unavailable_admission(
                "",
                PatchIntentSemanticAdmission,
            ).model_copy(update={"reason": "current correction outage"})
            session.latest_correction_run_id = correction.id

    result = PatchIntentAuditor(
        agent=CorrectionOutageAgent(),
        session=session,
        repository=object(),
        docker_env=object(),
        included_paths=set(),
        max_steps=2,
    ).audit(
        issue=ISSUE,
        repair_intent_contract="",
        patch="diff --git a/a.hpp b/a.hpp\n",
        file_context=FileContext(repo=None),
    )

    assert result.verdict == "inconclusive"
    assert result.semantic_admission.reason == "current correction outage"
    assert "current correction outage" in result.report


def test_step_budget_finalizes_admitted_correction_without_extra_submit_step():
    template_session, _ = _session_with_run()
    template_action = template_session.actions[0]
    template_extra = {
        key: value
        for key, value in template_action.items()
        if key not in {"action", "command", "exit_code", "evidence_id"}
    }
    session = PatchIntentAuditSession(audit_dir="/tmp/intent2fix-audit/test")
    submit = SubmitPatchIntentAudit(
        session=session,
        issue=ISSUE,
        contract="",
        semantic_adjudicator=_admit,
    )

    class ResolvedOnLastStepAgent:
        actions = [submit]

        def run(self, _node):
            if session.actions:
                return
            original = session.record(
                "rejected pair",
                "target encoded candidate outcome",
                0,
                prefix="RUN",
                action="RunContrastiveIntentProbe",
            )
            session.pending_correction_run_id = original.id
            resolved = session.record(
                template_action["command"],
                "assertion failed",
                3,
                prefix="RUN",
                action="RunContrastiveIntentProbe",
                extra={**template_extra, "corrects_run_id": original.id},
            )
            session.semantic_admissions[resolved.id] = _admit(
                "",
                PatchIntentSemanticAdmission,
            )
            session.latest_correction_run_id = resolved.id
            session.resolved_correction_run_id = resolved.id

    audit = PatchIntentAuditor(
        agent=ResolvedOnLastStepAgent(),
        session=session,
        repository=object(),
        docker_env=object(),
        included_paths=set(),
        max_steps=2,
    )
    result = audit.audit(
        issue=ISSUE,
        repair_intent_contract="",
        patch="diff --git a/a.hpp b/a.hpp\n",
        file_context=FileContext(repo=None),
    )

    assert result.verdict == "underfit"
    assert result.blocking
    assert result.counterexample.run_id == "RUN-2"
    assert audit.termination_reason == "resolved_correction_fallback"


def test_short_exact_issue_span_is_valid_probe_authority():
    short_issue = "Use ADL."
    session, _, _, observation = _execute_probe_with_admission(
        _admit,
        issue=short_issue,
        purpose="Use the ADL-selected public result for the overlap input.",
    )

    assert not observation.expect_correction
    assert observation.properties["evidence_id"] == "RUN-1"
    assert session.actions[-1]["oracle_reference"] == short_issue


def test_oracle_reference_whitespace_is_canonicalized_at_run_time():
    session, _, _, observation = _execute_probe_with_admission(
        _admit,
        issue="Use ADL.",
        oracle_reference="  Use ADL.  ",
        purpose="Use the ADL-selected result for the overlap input.",
    )
    run_id = observation.properties["evidence_id"]

    submitted = SubmitPatchIntentAudit(
        session=session,
        issue="Use ADL.",
        contract="",
        semantic_adjudicator=_admit,
    ).execute(
        SubmitPatchIntentAuditArgs(
            thoughts="",
            verdict="underfit",
            report="This broader wording must not replace the frozen purpose.",
            evidence_ids=[run_id],
            run_id=run_id,
            oracle_source="issue_explicit",
            oracle_reference="Use ADL.",
        )
    )

    assert submitted.terminal
    assert session.actions[-1]["oracle_reference"] == "Use ADL."


def test_contrastive_probe_rejects_different_compile_environment():
    runtime = _FakeAuditRuntime()
    session = PatchIntentAuditSession(audit_dir="/tmp/intent2fix-audit/test")
    WriteContrastiveIntentProbe(
        docker_env=runtime,
        session=session,
    ).execute(
        WriteContrastiveIntentProbeArgs(
            thoughts="",
            probe_name="probe",
            fixture_content="struct input {};",
            control_content="int main() { return 0; }",
            target_content="int main() { return 1; }",
        )
    )

    observation = RunContrastiveIntentProbe(
        docker_env=runtime,
        session=session,
        included_paths=set(),
        issue=ISSUE,
    ).execute(
        RunContrastiveIntentProbeArgs(
            thoughts="",
            probe_name="probe",
            control_compile_command=(
                "c++ -DFEATURE=1 /tmp/intent2fix-audit/test/probe_control.cpp "
                "-o /tmp/intent2fix-audit/test/probe_control"
            ),
            control_run_command="/tmp/intent2fix-audit/test/probe_control",
            target_compile_command=(
                "c++ /tmp/intent2fix-audit/test/probe_target.cpp "
                "-o /tmp/intent2fix-audit/test/probe_target"
            ),
            target_run_command="/tmp/intent2fix-audit/test/probe_target",
            oracle=ISSUE,
            oracle_source="issue_explicit",
            oracle_reference=ISSUE,
            distinguishing_input="overlap input",
            candidate_prediction="rejects",
            alternative_prediction="accepts",
            purpose="distinguish public formatter acceptance",
        ),
        file_context=object(),
    )

    assert observation.expect_correction
    assert "identical compiler" in observation.message
    assert not session.evidence


def test_control_cannot_rewrite_frozen_target_before_target_compile():
    class TamperingRuntime(_FakeAuditRuntime):
        def __init__(self):
            super().__init__()
            self.target_restored = False

        def run_bash(self, command, work_dir=None, timeout=None):
            target_path = "/tmp/intent2fix-audit/test/probe_target.cpp"
            if command == "/tmp/intent2fix-audit/test/probe_control":
                self.writes[target_path] = "tampered"
                return "control passed", 0
            if command.startswith("c++ ") and target_path in command:
                self.target_restored = "return 3" in self.writes[target_path]
            return super().run_bash(command, work_dir=work_dir, timeout=timeout)

    runtime = TamperingRuntime()
    session = PatchIntentAuditSession(audit_dir="/tmp/intent2fix-audit/test")
    WriteContrastiveIntentProbe(
        docker_env=runtime,
        session=session,
    ).execute(
        WriteContrastiveIntentProbeArgs(
            thoughts="",
            probe_name="probe",
            fixture_content="struct input {};",
            control_content="int main() { return 0; }",
            target_content="int main() { return 3; }",
        )
    )

    observation = RunContrastiveIntentProbe(
        docker_env=runtime,
        session=session,
        included_paths=set(),
        issue=ISSUE,
    ).execute(
        RunContrastiveIntentProbeArgs(
            thoughts="",
            probe_name="probe",
            control_compile_command=(
                "c++ /tmp/intent2fix-audit/test/probe_control.cpp "
                "-o /tmp/intent2fix-audit/test/probe_control"
            ),
            control_run_command="/tmp/intent2fix-audit/test/probe_control",
            target_compile_command=(
                "c++ /tmp/intent2fix-audit/test/probe_target.cpp "
                "-o /tmp/intent2fix-audit/test/probe_target"
            ),
            target_run_command="/tmp/intent2fix-audit/test/probe_target",
            oracle=ISSUE,
            oracle_source="issue_explicit",
            oracle_reference=ISSUE,
            distinguishing_input="overlap input",
            candidate_prediction="rejects",
            alternative_prediction="accepts",
            purpose="distinguish public formatter acceptance",
        ),
        file_context=object(),
    )

    assert not observation.expect_correction
    assert runtime.target_restored


def test_stage_output_distinguishes_empty_output_from_not_run():
    assert _stage_output("", 0) == "(no output)"
    assert _stage_output("", None) == "(not run)"


def test_shared_witness_is_rejected_as_non_discriminating():
    runtime = _FakeAuditRuntime()
    session = PatchIntentAuditSession(
        audit_dir="/tmp/intent2fix-audit/test"
    )
    WriteContrastiveIntentProbe(
        docker_env=runtime,
        session=session,
    ).execute(
        WriteContrastiveIntentProbeArgs(
            thoughts="",
            probe_name="probe",
            fixture_content="struct input {};",
            control_content="int main() { return 0; }",
            target_content="int main() { return 0; }",
        )
    )

    observation = RunContrastiveIntentProbe(
        docker_env=runtime,
        session=session,
        included_paths=set(),
        issue=ISSUE,
    ).execute(
        RunContrastiveIntentProbeArgs(
            thoughts="",
            probe_name="probe",
            control_compile_command=(
                "c++ /tmp/intent2fix-audit/test/probe_control.cpp "
                "-o /tmp/intent2fix-audit/test/probe_control"
            ),
            control_run_command="/tmp/intent2fix-audit/test/probe_control",
            target_compile_command=(
                "c++ /tmp/intent2fix-audit/test/probe_target.cpp "
                "-o /tmp/intent2fix-audit/test/probe_target"
            ),
            target_run_command="/tmp/intent2fix-audit/test/probe_target",
            oracle=ISSUE,
            oracle_source="issue_explicit",
            oracle_reference=ISSUE,
            distinguishing_input="ADL-only range",
            candidate_prediction="formats 1,2,3",
            alternative_prediction="formats 1,2,3",
            purpose="distinguish public formatter acceptance",
        ),
        file_context=object(),
    )

    assert observation.expect_correction
    assert "not a discriminator" in observation.message
    assert not session.evidence


def _frozen_probe(
    oracle_mode: str = "runtime_behavior",
) -> FrozenPatchIntentProbe:
    return FrozenPatchIntentProbe(
        run_id="RUN-1",
        probe_name="probe",
        fixture_path="/tmp/intent2fix-audit/test/probe_fixture.hpp",
        fixture_content="struct input {};",
        control_path="/tmp/intent2fix-audit/test/probe_control.cpp",
        control_content='#include "probe_fixture.hpp"\nint main() { return 0; }',
        target_path="/tmp/intent2fix-audit/test/probe_target.cpp",
        target_content='#include "probe_fixture.hpp"\nint main() { return 3; }',
        control_compile_command=(
            "c++ /tmp/intent2fix-audit/test/probe_control.cpp -o "
            "/tmp/intent2fix-audit/test/probe_control"
        ),
        control_run_command="/tmp/intent2fix-audit/test/probe_control",
        target_compile_command=(
            "c++ /tmp/intent2fix-audit/test/probe_target.cpp -o "
            "/tmp/intent2fix-audit/test/probe_target"
        ),
        target_run_command="/tmp/intent2fix-audit/test/probe_target",
        oracle=ISSUE,
        oracle_mode=oracle_mode,
        distinguishing_input="public formatter with emphasis::bold",
        candidate_prediction="returns an error",
        alternative_prediction="formats the value",
        purpose="distinguish overload participation",
        oracle_source="issue_explicit",
        oracle_reference=ISSUE,
        control_compile_exit_code=0,
        control_run_exit_code=0,
        target_compile_exit_code=0,
        target_run_exit_code=3,
    )


class _RecheckRuntime:
    work_dir = "/repo"

    def __init__(self, *, control_exit=0, target_exit=0):
        self.control_exit = control_exit
        self.target_exit = target_exit
        self.commands = []

    def write_file(self, path, content):
        pass

    def run_bash(self, command, work_dir=None, timeout=None):
        self.commands.append(command)
        if command.startswith("tmp=/tmp/intent2fix-tracked-diff"):
            return "same-fingerprint  -\n", 0
        if command == "git ls-files --others --exclude-standard -z":
            return "", 0
        if command.startswith("c++ "):
            return "", 0
        if command.endswith("probe_control"):
            return "control", self.control_exit
        if command.endswith("probe_target"):
            return "target", self.target_exit
        raise AssertionError(command)


def test_frozen_pair_recheck_stops_when_control_no_longer_passes(monkeypatch):
    monkeypatch.setattr(
        auditor_module,
        "sync_verification_files",
        lambda *args, **kwargs: "",
    )
    runtime = _RecheckRuntime(control_exit=4)
    auditor = PatchIntentAuditor(
        agent=object(),
        session=PatchIntentAuditSession(
            audit_dir="/tmp/intent2fix-audit/test"
        ),
        repository=object(),
        docker_env=runtime,
        included_paths=set(),
    )

    result = auditor.recheck(
        _frozen_probe(),
        file_context=object(),
        included_paths=set(),
    )

    assert not result.conclusive
    assert not result.passed
    assert result.target_compile_exit_code is None
    assert (
        "/tmp/intent2fix-audit/test/probe_target"
        not in runtime.commands
    )


def test_frozen_pair_recheck_requires_both_programs_to_pass(monkeypatch):
    monkeypatch.setattr(
        auditor_module,
        "sync_verification_files",
        lambda *args, **kwargs: "",
    )
    runtime = _RecheckRuntime(control_exit=0, target_exit=0)
    auditor = PatchIntentAuditor(
        agent=object(),
        session=PatchIntentAuditSession(
            audit_dir="/tmp/intent2fix-audit/test"
        ),
        repository=object(),
        docker_env=runtime,
        included_paths=set(),
    )

    result = auditor.recheck(
        _frozen_probe(),
        file_context=object(),
        included_paths=set(),
    )

    assert result.conclusive
    assert result.passed
    assert result.control_run_exit_code == 0
    assert result.target_run_exit_code == 0


def test_compile_acceptance_recheck_stops_after_target_compiles(monkeypatch):
    monkeypatch.setattr(
        auditor_module,
        "sync_verification_files",
        lambda *args, **kwargs: "",
    )
    runtime = _RecheckRuntime(control_exit=0, target_exit=9)
    auditor = PatchIntentAuditor(
        agent=object(),
        session=PatchIntentAuditSession(
            audit_dir="/tmp/intent2fix-audit/test"
        ),
        repository=object(),
        docker_env=runtime,
        included_paths=set(),
    )

    result = auditor.recheck(
        _frozen_probe("compile_acceptance"),
        file_context=object(),
        included_paths=set(),
    )

    assert result.conclusive
    assert result.passed
    assert result.target_compile_exit_code == 0
    assert result.target_run_exit_code is None
    assert (
        "/tmp/intent2fix-audit/test/probe_target"
        not in runtime.commands
    )
