from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from INTENT2FIX.actions.entity_intent_search import EntityIntentSearch
from INTENT2FIX.actions.schema import Observation
from INTENT2FIX.actions.search_base import SearchBaseAction
from INTENT2FIX.index.entity_code_index import EntityCodeIndex
from INTENT2FIX.node import ActionStep, Node
from INTENT2FIX.repository.repository import InMemRepository
from INTENT2FIX.reproduce.actions.create_file import CreateFile, CreateFileArgs
from INTENT2FIX.reproduce.actions.finish import Finish, FinishArgs
from INTENT2FIX.reproduce.actions.run_command import RunCommand, RunCommandArgs
from INTENT2FIX.reproduce.reproduction_agent import ReproductionAgent
from INTENT2FIX.reproduce.intent_pipeline import (
    build_intent_adjudication_prompt,
    build_issue_evidence_prompt,
    fallback_issue_evidence,
    ground_issue_evidence,
    issue_evidence_is_grounded,
    merge_issue_evidence_with_fallback,
    parse_structured_response,
    unresolved_bundle,
)
from INTENT2FIX.reproduce.intent_schema import (
    ExecutionRecord,
    IssueEvidenceFrame,
    RepairIntentBundle,
)
from INTENT2FIX.reproduce.state import ReproductionSession
from INTENT2FIX.reproduction_loop import ReproductionLoop
from INTENT2FIX.runtime.dockerenv import DockerEnv


def test_issue_evidence_prompt_extracts_anchors_without_declaring_intent():
    prompt = build_issue_evidence_prompt("Comparison should compile.")

    assert "search anchors" in prompt
    assert "repository-informed final repair contract" in prompt
    assert "root cause" in prompt
    assert "Statements marked unrelated, speculative, or uncertain" in prompt
    assert "not affected scope" in prompt
    assert "NONGOAL" in prompt
    assert "Coverage is mandatory" in prompt
    assert "atomic, independently observable" in prompt
    assert "imperative issue titles" in prompt
    assert '"is by design"' in prompt
    assert "does not by itself create an" in prompt
    assert "model-inferred repair policies" in prompt
    assert "authorizes only the exact observable relation" in prompt
    assert "must not add X's other ordering" in prompt
    assert "extract the applicability wording as a" in prompt
    assert "does not need its own Expected Behavior" in prompt


def test_negative_behavior_title_recovers_expected_intent_unit():
    issue = (
        "join does not use ADL on begin/end of its ranges\n"
        "Is the current behavior by design?"
    )
    structured = IssueEvidenceFrame.model_validate({
        "claims": [
            {
                "id": "OBS-1",
                "kind": "observed_behavior",
                "text": "The current join path does not use ADL.",
                "issue_span": "join does not use ADL on begin/end of its ranges",
            }
        ]
    })

    grounded = merge_issue_evidence_with_fallback(issue, structured)

    assert any(
        claim.kind == "expected_behavior"
        and claim.issue_span == "join does not use ADL on begin/end of its ranges"
        for claim in grounded.claims
    )
    assert [unit.text for unit in grounded.intent_units] == [
        "join should use ADL on begin/end of its ranges."
    ]


def test_explicit_non_goal_keeps_exact_issue_provenance():
    issue = "Do not change the default parser behavior."
    frame = IssueEvidenceFrame.model_validate({
        "claims": [
            {
                "id": "NONGOAL-1",
                "kind": "non_goal",
                "text": issue,
                "issue_span": issue,
            }
        ]
    })

    grounded = ground_issue_evidence(issue, frame)

    assert grounded is not None
    assert grounded.claims[0].kind == "non_goal"


def test_issue_intent_ledger_preserves_every_normative_claim():
    issue = "Parser accepts string input.\nFailed parses preserve cursor state."
    frame = IssueEvidenceFrame.model_validate({
        "claims": [
            {
                "id": "EXP-1",
                "kind": "expected_behavior",
                "text": "Parser accepts string input.",
                "issue_span": "Parser accepts string input.",
            },
            {
                "id": "PRES-1",
                "kind": "preservation",
                "text": "Failed parses preserve cursor state.",
                "issue_span": "Failed parses preserve cursor state.",
            },
        ],
        "intent_units": [
            {
                "id": "INTENT-7",
                "role": "target",
                "text": "Parser accepts string input.",
                "issue_span": "Parser accepts string input.",
                "source_claim_ids": ["EXP-1"],
            }
        ],
    })

    grounded = ground_issue_evidence(issue, frame)

    assert grounded is not None
    assert [item.id for item in grounded.intent_units] == ["INTENT-1", "INTENT-2"]
    assert grounded.intent_units[1].role == "preserve"
    assert grounded.intent_units[1].source_claim_ids == ["PRES-1"]


def test_issue_intent_text_excludes_linked_example_semantics():
    issue = (
        "join should use ADL on begin/end of its ranges.\n"
        "For example, range-for does ADL on associated namespaces."
    )
    frame = IssueEvidenceFrame.model_validate({
        "claims": [
            {
                "id": "EXP-1",
                "kind": "expected_behavior",
                "text": "join should use ADL on begin/end of its ranges.",
                "issue_span": "join should use ADL on begin/end of its ranges.",
            },
            {
                "id": "EX-1",
                "kind": "example",
                "text": "range-for also performs ADL.",
                "issue_span": (
                    "For example, range-for does ADL on associated namespaces."
                ),
            },
        ],
        "intent_units": [
            {
                "id": "INTENT-1",
                "role": "target",
                "text": (
                    "join should use ADL, with range-for member precedence."
                ),
                "issue_span": "join should use ADL on begin/end of its ranges.",
                "source_claim_ids": ["EXP-1", "EX-1"],
            }
        ],
    })

    grounded = ground_issue_evidence(issue, frame)

    assert grounded is not None
    assert grounded.intent_units[0].text == (
        "join should use ADL on begin/end of its ranges."
    )
    assert grounded.intent_units[0].source_claim_ids == ["EXP-1", "EX-1"]


def test_structured_response_and_unresolved_fallback():
    raw = """```json
{"claims": [], "symbols": ["ExprLhs"], "error_signatures": [],
 "implementation_suggestions": [], "ambiguities": [], "extraction_status": "structured"}
```"""
    frame = parse_structured_response(raw, IssueEvidenceFrame)
    bundle = unresolved_bundle("adjudication unavailable")

    assert frame.symbols == ["ExprLhs"]
    assert bundle.status == "UNRESOLVED"
    assert bundle.unresolved == ["adjudication unavailable"]
    assert bundle.compact_dump()["reproduction_status"] == "UNRESOLVED"
    assert "execution_records" not in bundle.compact_dump()


def test_structured_response_extracts_json_from_provider_wrapper_text():
    raw = """I will return the requested object.\n```json
{"claims": [], "symbols": ["json_pointer"], "error_signatures": [],
 "implementation_suggestions": [], "ambiguities": [], "extraction_status": "structured"}
```\nDone."""

    frame = parse_structured_response(raw, IssueEvidenceFrame)

    assert frame.symbols == ["json_pointer"]


def test_structured_response_recovers_relaxed_json():
    raw = """{'claims': [], 'symbols': ['ExprLhs'], 'error_signatures': [],
    'implementation_suggestions': [], 'ambiguities': [],
    'extraction_status': 'structured',}"""

    frame = parse_structured_response(raw, IssueEvidenceFrame)

    assert frame.symbols == ["ExprLhs"]


def test_structured_response_recovers_provider_properties_wrapper():
    raw = """{"properties": {"claims": [], "symbols": ["ExprLhs"],
    "error_signatures": [], "implementation_suggestions": [],
    "ambiguities": [], "extraction_status": "structured"}}"""

    frame = parse_structured_response(raw, IssueEvidenceFrame)

    assert frame.symbols == ["ExprLhs"]


def test_structured_response_does_not_accept_literal_json_schema():
    raw = '{"properties": {"claims": {"type": "array"}}}'

    with pytest.raises(ValidationError):
        parse_structured_response(raw, IssueEvidenceFrame)


def test_issue_fallback_recovers_explicit_expected_and_preservation_sections():
    issue = """**Expected behavior**
The test should compile.

**Fix**
However, this broke a test for bitfields.
"""

    frame = fallback_issue_evidence(issue)

    assert [(claim.id, claim.kind) for claim in frame.claims] == [
        ("OBS-FALLBACK-1", "observed_behavior"),
        ("EXP-FALLBACK-1", "expected_behavior"),
        ("PRES-FALLBACK-1", "preservation"),
    ]
    assert all(claim.issue_span in issue for claim in frame.claims)


def test_issue_fallback_recovers_imperative_title_as_expected_behavior():
    issue = """Make CHECKED_IF and CHECKED_ELSE ok to fail

Entering the else branch currently fails the test.
"""

    frame = fallback_issue_evidence(issue)

    expected = [
        claim for claim in frame.claims if claim.kind == "expected_behavior"
    ]
    assert len(expected) == 1
    assert expected[0].id == "EXP-FALLBACK-TITLE"
    assert expected[0].text == "Make CHECKED_IF and CHECKED_ELSE ok to fail"
    assert expected[0].issue_span in issue


def test_issue_fallback_recovers_negative_capability_title():
    issue = """truncating_iterator is no std::output_iterator

The iterator fails the concept check in format_to_n.
"""

    frame = fallback_issue_evidence(issue)

    expected = [
        claim for claim in frame.claims if claim.kind == "expected_behavior"
    ]
    assert len(expected) == 1
    assert expected[0].text == (
        "truncating_iterator should satisfy std::output_iterator."
    )
    assert expected[0].issue_span == (
        "truncating_iterator is no std::output_iterator"
    )


def test_merge_adds_imperative_title_when_structured_extraction_misses_target():
    issue = "Make CHECKED_IF and CHECKED_ELSE ok to fail\nCurrent behavior fails."
    frame = IssueEvidenceFrame.model_validate({
        "claims": [
            {
                "id": "OBS-1",
                "kind": "observed_behavior",
                "text": "Current behavior fails.",
                "issue_span": "Current behavior fails.",
            }
        ]
    })

    merged = merge_issue_evidence_with_fallback(issue, frame)

    assert any(
        claim.kind == "expected_behavior"
        and claim.text == "Make CHECKED_IF and CHECKED_ELSE ok to fail"
        for claim in merged.claims
    )


def test_issue_evidence_does_not_promote_uncertain_unrelated_note():
    issue = """**Expected behavior**
The test should compile.

**Fix**
The operator== member function at line 187 is the source of the issue.
However, this broke a test for bitfields.

**Unrelated note**
I don't think const reference here prolongs the lifetime of rhs, because it's not local but stored in a class.
"""
    frame = IssueEvidenceFrame.model_validate({
        "claims": [
            {
                "id": "EXP-1",
                "kind": "expected_behavior",
                "text": "The test should compile.",
                "issue_span": "The test should compile.",
            },
            {
                "id": "SCOPE-1",
                "kind": "affected_scope",
                "text": "The operator== member function at line 187 is the source of the issue.",
                "issue_span": "The operator== member function at line 187 is the source of the issue.",
            },
            {
                "id": "PRES-1",
                "kind": "preservation",
                "text": "Const references do not create a lifetime problem.",
                "issue_span": "I don't think const reference here prolongs the lifetime of rhs, because it's not local but stored in a class.",
            },
        ],
        "ambiguities": [
            "The exact implementation mechanism remains unclear."
        ],
    })

    merged = merge_issue_evidence_with_fallback(issue, frame)

    assert [claim.id for claim in merged.claims] == [
        "EXP-1",
        "PRES-FALLBACK-1",
    ]
    assert merged.claims[-1].issue_span == "However, this broke a test for bitfields."
    assert any("Non-normative preservation statement PRES-1" in item for item in merged.ambiguities)
    assert any("SCOPE-1" in item for item in merged.implementation_suggestions)
    assert "The exact implementation mechanism remains unclear." in merged.ambiguities
    assert not any("exact mechanism" in item.lower() for item in merged.ambiguities)


def test_issue_evidence_rejects_claims_without_exact_provenance():
    invalid_frame = IssueEvidenceFrame.model_validate({
        "claims": [{
            "id": "EXP-1",
            "kind": "expected_behavior",
            "text": "comparison compiles",
            "issue_span": "not present in the issue",
        }]
    })

    assert issue_evidence_is_grounded("The comparison should compile.", invalid_frame) is False


def test_issue_evidence_keeps_valid_claims_when_one_span_is_normalized():
    frame = IssueEvidenceFrame.model_validate({
        "claims": [
            {
                "id": "EXP-1",
                "kind": "expected_behavior",
                "text": "comparison compiles",
                "issue_span": "comparison should compile",
            },
            {
                "id": "EX-1",
                "kind": "example",
                "text": "normalized code block",
                "issue_span": "int main() {}",
            },
        ]
    })

    grounded = ground_issue_evidence("The comparison should compile.", frame)

    assert [claim.id for claim in grounded.claims] == ["EXP-1"]
    assert "Discarded 1 claim" in grounded.ambiguities[-1]


def test_adjudication_sees_test_source_and_compact_execution_output():
    prompt = build_intent_adjudication_prompt(
        issue="compile failure",
        issue_evidence=IssueEvidenceFrame(),
        repository_evidence=[],
        execution_records=[
            ExecutionRecord(
                run_id="RUN-1",
                command="c++ repro.cpp",
                exit_code=1,
                stderr="x" * 3000,
            )
        ],
        tests=[],
        test_files=[{"path": "repro.cpp", "content": "int main() { return 1; }"}],
        graph_bindings=[],
    )

    assert "int main() { return 1; }" in prompt
    assert "x" * 2000 in prompt
    assert "x" * 2001 not in prompt
    assert "MISMATCHED" in prompt
    assert "observable behavior contracts, not patch operations" in prompt
    assert "Every issue-explicit operand form" in prompt
    assert "does not by itself define conversion direction" in prompt
    assert "Unknown implementation details do not make" in prompt
    assert "Leave `execution_records` and `tests` empty" in prompt


def test_deepseek_reproducer_uses_native_grounding_actions():
    docker_env = Mock(spec=DockerEnv)
    docker_env.work_dir = "/repo"
    agent = ReproductionAgent.create(
        repository=InMemRepository(),
        code_index=EntityCodeIndex(),
        docker_env=docker_env,
        model="deepseek-v4-flash",
    )

    action_names = {action.name for action in agent.actions}
    assert {"EntityIntentSearch", "EntitySearchSymbol", "ViewCode", "RunCommand"} <= action_names
    assert "# Intent-Validating Reproduction Policy" in agent.system_prompt
    assert "Answer Format" not in agent.system_prompt
    assert agent.message_generator.max_tokens == 850000
    assert all(
        action.progressive_disclosure
        for action in agent.actions
        if isinstance(action, SearchBaseAction)
    )
    assert next(
        action for action in agent.actions if isinstance(action, EntityIntentSearch)
    ).progressive_disclosure


def _trajectory(evidence_run_id: str) -> Node:
    root = Node(node_id=0)
    root.action_steps = [
        ActionStep(
            action=CreateFileArgs(path="repro.cpp", file_text="int main() {}"),
            observation=Observation(message="created"),
        ),
        ActionStep(
            action=RunCommandArgs(command="c++ repro.cpp"),
            observation=Observation(
                message="compile failed",
                properties={
                    "run_id": "RUN-1",
                    "command": "c++ repro.cpp",
                    "stdout": "",
                    "stderr": "target overload resolution failure",
                    "exit_code": 1,
                },
            ),
        ),
    ]

    finish_args = FinishArgs(
        determinant=True,
        evidence_run_id=evidence_run_id,
        failure_class="TARGET_COMPILE_FAILURE",
        expected_behavior_predicate="compilation reaches the target overload",
        matched_claim_ids=["OBS-1"],
        test_artifacts=[
            {"path": "repro.cpp", "obligation_ids": ["OBS-1"], "purpose": "symptom"}
        ],
        graph_bindings=[
            {
                "claim_ids": ["OBS-1"],
                "entity": "Catch::ExprLhs::operator==",
                "file": "src/catch_decomposer.hpp",
                "relation": "LIKELY_REALIZES",
                "evidence": "symbol search result",
            }
        ],
    )
    finish_observation = Finish().execute(finish_args)
    terminal = Node(
        node_id=1,
        terminal=True,
        action_steps=[ActionStep(action=finish_args, observation=finish_observation)],
    )
    root.add_child(terminal)
    return root


def test_reproduction_result_binds_finish_to_real_execution_and_test():
    result = ReproductionLoop._create_result_from_trajectory(None, _trajectory("RUN-1"))

    assert result.is_reproduced is True
    assert result.test_command == "c++ repro.cpp"
    assert result.test_files == [{"path": "repro.cpp", "content": "int main() {}"}]
    assert result.reproduction_output == "target overload resolution failure"
    assert result.graph_bindings[0].entity == "Catch::ExprLhs::operator=="


def test_reproduction_result_rejects_unverifiable_finish_evidence():
    result = ReproductionLoop._create_result_from_trajectory(None, _trajectory("RUN-404"))

    assert result.is_reproduced is False
    assert any("valid target-failure command" in item for item in result.unresolved)


def test_reproduction_result_normalizes_provider_run_id_format():
    result = ReproductionLoop._create_result_from_trajectory(None, _trajectory("run_1"))

    assert result.is_reproduced is True
    assert result.test_command == "c++ repro.cpp"


@pytest.mark.parametrize("path", ["/home/repro.cpp", "../repro.cpp", "tests/../../repro.cpp"])
def test_reproduction_create_file_requires_repository_relative_path(path):
    with pytest.raises(ValidationError):
        CreateFileArgs(path=path, file_text="int main() {}")


def test_reproduction_create_file_rejects_existing_repository_file():
    repository = InMemRepository(
        files={"tests/repro_existing.cpp": "original"}
    )
    action = CreateFile(
        repository=repository,
        docker_env=Mock(spec=DockerEnv),
        session=ReproductionSession(),
    )

    observation = action.execute(
        CreateFileArgs(
            path="tests/repro_existing.cpp",
            file_text="replacement",
        ),
        None,
    )

    assert observation.expect_correction is True
    assert "cannot overwrite" in observation.message
    action.docker_env.copy_anything_to_container.assert_not_called()


def test_reproduction_create_file_rejects_non_repro_artifact_name():
    repository = InMemRepository()
    action = CreateFile(
        repository=repository,
        docker_env=Mock(spec=DockerEnv),
        session=ReproductionSession(),
    )

    observation = action.execute(
        CreateFileArgs(path="tests/new_test.cpp", file_text="int main() {}"),
        None,
    )

    assert observation.expect_correction is True
    assert "dedicated C/C++ reproduction file" in observation.message
    action.docker_env.copy_anything_to_container.assert_not_called()


def _finish_args(evidence_run_id: str = "RUN-1") -> FinishArgs:
    return FinishArgs(
        determinant=True,
        evidence_run_id=evidence_run_id,
        failure_class="TARGET_COMPILE_FAILURE",
        expected_behavior_predicate="the target overload does not compile",
        matched_claim_ids=["OBS-1"],
        test_artifacts=[
            {"path": "repro.cpp", "obligation_ids": ["OBS-1"], "purpose": "symptom"}
        ],
    )


def test_finish_rebinds_wrong_id_to_only_artifact_failure_record():
    session = ReproductionSession()
    session.register_artifact("repro.cpp", "int main() {}")
    session.add_execution(ExecutionRecord(
        run_id="RUN-1", command="ls", exit_code=0
    ))
    session.add_execution(ExecutionRecord(
        run_id="RUN-2", command="c++ repro.cpp", exit_code=1, stderr="target error"
    ))

    observation = Finish(session=session).execute(_finish_args("run_1"))

    assert observation.terminal is True
    assert observation.properties["evidence_run_id"] == "RUN-2"
    assert "only eligible" in observation.message


def test_finish_rejects_successful_observation_as_reproduction_evidence():
    session = ReproductionSession()
    session.register_artifact("repro.cpp", "int main() {}")
    session.add_execution(ExecutionRecord(
        run_id="RUN-1", command="c++ repro.cpp && ./a.out", exit_code=0
    ))

    observation = Finish(session=session).execute(_finish_args())

    assert observation.terminal is False
    assert observation.expect_correction is True
    assert observation.properties["candidate_run_ids"] == []


def test_finish_rejects_nonzero_inspection_command_as_reproduction_evidence():
    session = ReproductionSession()
    session.register_artifact("repro.cpp", "int main() {}")
    session.add_execution(ExecutionRecord(
        run_id="RUN-1", command="ls missing repro.cpp", exit_code=2
    ))

    observation = Finish(session=session).execute(_finish_args())

    assert observation.terminal is False
    assert observation.properties["candidate_run_ids"] == []


def test_run_command_marks_artifact_failure_as_finish_candidate(monkeypatch):
    session = ReproductionSession()
    session.register_artifact("repro.cpp", "int main() {}")
    monkeypatch.setattr(
        RunCommand,
        "_run_bash_detailed",
        lambda self, command: ("", "target compile error", 1),
    )

    observation = RunCommand(
        docker_env=Mock(spec=DockerEnv), session=session
    ).execute(RunCommandArgs(command="c++ repro.cpp"))

    assert observation.properties["candidate_reproduction_evidence"] is True
    assert observation.properties["candidate_artifacts"] == ["repro.cpp"]
    assert "call Finish next with RUN-1" in observation.message


def test_candidate_evidence_blocks_inspection_until_finish_or_revision(monkeypatch):
    session = ReproductionSession()
    session.register_artifact("repro.cpp", "int main() {}")
    monkeypatch.setattr(
        RunCommand,
        "_run_bash_detailed",
        lambda self, command: ("", "target compile error", 1),
    )
    action = RunCommand(docker_env=Mock(spec=DockerEnv), session=session)

    action.execute(RunCommandArgs(command="c++ repro.cpp"))
    observation = action.execute(RunCommandArgs(command="grep -n operator include/a.hpp"))

    assert observation.expect_correction is True
    assert "Call Finish with RUN-1" in observation.message
    assert len(session.execution_records) == 1


def test_one_revision_and_one_revised_execution_force_finish(monkeypatch):
    session = ReproductionSession()
    session.register_artifact("repro.cpp", "int main() {}")
    session.add_execution(ExecutionRecord(
        run_id="RUN-1", command="c++ repro.cpp", exit_code=1
    ))
    session.note_post_candidate_revision()
    calls = []

    def run(self, command):
        calls.append(command)
        return "", "broader target error", 1

    monkeypatch.setattr(RunCommand, "_run_bash_detailed", run)
    action = RunCommand(docker_env=Mock(spec=DockerEnv), session=session)
    action._run_counter = 1

    revised = action.execute(RunCommandArgs(command="c++ repro.cpp"))
    rejected = action.execute(RunCommandArgs(command="c++ repro.cpp"))

    assert revised.properties["candidate_reproduction_evidence"] is True
    assert session.candidate_run_ids == ["RUN-1", "RUN-2"]
    assert rejected.expect_correction is True
    assert "Call Finish now" in rejected.message
    assert calls == ["c++ repro.cpp"]


def test_candidate_session_blocks_search_action_after_second_candidate():
    session = ReproductionSession()
    session.register_artifact("repro.cpp", "int main() {}")
    session.add_execution(ExecutionRecord(
        run_id="RUN-1", command="c++ repro.cpp", exit_code=1
    ))
    session.note_post_candidate_revision()
    session.note_revised_execution()
    session.add_execution(ExecutionRecord(
        run_id="RUN-2", command="c++ repro.cpp", exit_code=1
    ))

    rejection = session.convergence_rejection("EntitySearchSymbol")

    assert "Call Finish now with RUN-2" in rejection


def test_run_command_rejects_pipeline_that_masks_compile_status(monkeypatch):
    docker_env = Mock(spec=DockerEnv)
    session = ReproductionSession()
    session.register_artifact("repro.cpp", "int main() {}")
    run = Mock()
    monkeypatch.setattr(RunCommand, "_run_bash_detailed", run)

    observation = RunCommand(
        docker_env=docker_env, session=session
    ).execute(RunCommandArgs(command="c++ repro.cpp 2>&1 | head"))

    assert observation.expect_correction is True
    assert "mask" in observation.message
    run.assert_not_called()


def test_run_command_rejects_pipeline_that_masks_cmake_build_status(monkeypatch):
    docker_env = Mock(spec=DockerEnv)
    run = Mock()
    monkeypatch.setattr(RunCommand, "_run_bash_detailed", run)

    observation = RunCommand(docker_env=docker_env).execute(
        RunCommandArgs(command="cmake --build build --parallel 4 2>&1 | tail -20")
    )

    assert observation.expect_correction is True
    assert "mask" in observation.message
    run.assert_not_called()


def test_reproduction_run_command_rejects_future_data_access(monkeypatch):
    def unexpected_run(*args, **kwargs):
        raise AssertionError("network command must not execute")

    monkeypatch.setattr(RunCommand, "_run_bash_detailed", unexpected_run)
    observation = RunCommand(docker_env=Mock(spec=DockerEnv)).execute(
        RunCommandArgs(command="git clone https://github.com/org/repo.git /tmp/upstream")
    )

    assert observation.expect_correction is True
    assert "Do not retrieve or reference future fixes" in observation.message


def test_finish_accepts_runtime_failure_from_executable_derived_from_artifact():
    session = ReproductionSession()
    session.register_artifact("repro.cpp", "int main() { return 1; }")
    session.add_execution(ExecutionRecord(
        run_id="RUN-1", command="c++ repro.cpp -o repro", exit_code=0
    ))
    session.add_execution(ExecutionRecord(
        run_id="RUN-2", command="./repro", exit_code=1, stderr="target assertion"
    ))
    args = _finish_args("RUN-2")
    args.failure_class = "TARGET_RUNTIME_FAILURE"

    observation = Finish(session=session).execute(args)

    assert observation.terminal is True
    assert observation.properties["evidence_run_id"] == "RUN-2"


def test_finish_rejects_artifact_missing_from_container():
    session = ReproductionSession()
    session.register_artifact("repro.cpp", "int main() {}")
    session.add_execution(ExecutionRecord(
        run_id="RUN-1", command="c++ repro.cpp", exit_code=1
    ))
    docker_env = object.__new__(DockerEnv)
    docker_env.work_dir = "/repo"
    docker_env.container = Mock()
    docker_env.container.exec_run.return_value = (1, b"")

    observation = Finish(session=session, docker_env=docker_env).execute(_finish_args())

    assert observation.terminal is False
    assert observation.properties["missing_artifacts"] == ["repro.cpp"]


@pytest.mark.parametrize(
    "command",
    [
        "cat > repro.cpp <<'EOF'\nint main() {}\nEOF",
        "printf 'int main() {}' > repro.cpp",
        "rm -f repro.cpp",
    ],
)
def test_run_command_rejects_shell_writes_to_reproduction_artifacts(command):
    docker_env = Mock(spec=DockerEnv)
    session = ReproductionSession()
    session.register_artifact("repro.cpp", "int main() {}")
    action = RunCommand(docker_env=docker_env, session=session)

    observation = action.execute(RunCommandArgs(command=command))

    assert observation.expect_correction is True
    assert session.execution_records == []


def test_posthoc_gate_rejects_artifact_deleted_after_evidence_run():
    root = _trajectory("RUN-1")
    terminal = root.children[0]
    root.children = []
    cleanup = Node(node_id=1)
    cleanup.action_steps = [
        ActionStep(
            action=RunCommandArgs(command="rm -f repro.cpp"),
            observation=Observation(
                message="removed",
                properties={
                    "run_id": "RUN-2",
                    "command": "rm -f repro.cpp",
                    "stdout": "",
                    "stderr": "",
                    "exit_code": 0,
                },
            ),
        )
    ]
    root.add_child(cleanup)
    cleanup.add_child(terminal)

    result = ReproductionLoop._create_result_from_trajectory(None, root)

    assert result.is_reproduced is False
    assert any("No declared reproduction test artifact" in item for item in result.unresolved)
