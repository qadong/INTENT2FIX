import json
from typing import List
from types import SimpleNamespace
from unittest.mock import Mock

from INTENT2FIX.actions.find_class import FindClass
from INTENT2FIX.actions.find_code_snippet import FindCodeSnippet
from INTENT2FIX.actions.find_function import FindFunction
from INTENT2FIX.actions.finish import Finish
from INTENT2FIX.actions.reject import Reject
from INTENT2FIX.actions.run_regression import RunRegression
from INTENT2FIX.actions.run_single_test import RunSingleTest
from INTENT2FIX.actions.run_tests import RunTests
from INTENT2FIX.actions.search_base import SearchBaseAction
from INTENT2FIX.actions.entity_intent_search import EntityIntentSearch
from INTENT2FIX.actions.semantic_search import SemanticSearch
from INTENT2FIX.actions.string_replace import StringReplace
from INTENT2FIX.actions.view_code import ViewCode, ViewCodeArgs
from INTENT2FIX.agent.code_agent import CodingAgent
from INTENT2FIX.agent.code_prompts import generate_compact_native_prompt
from INTENT2FIX.completion import LLMResponseFormat
from INTENT2FIX.completion.base import BaseCompletionModel
from INTENT2FIX.index.code_index import CodeIndex
from INTENT2FIX.repository.repository import InMemRepository
from INTENT2FIX.runtime.runtime import RuntimeEnvironment
from INTENT2FIX.runtime.runtime import TestResult


class MockCodeIndex(CodeIndex):
    def __init__(self):
        pass


class MockRuntimeEnvironment(RuntimeEnvironment):
    def run_tests(self, test_files: List[str] | None = None) -> list[TestResult]:
        return []



def test_dump_and_load_coding_agent():
    repository = InMemRepository()
    code_index = MockCodeIndex()
    runtime = MockRuntimeEnvironment()

    completion_model = BaseCompletionModel.create(
        response_format=LLMResponseFormat.TOOLS,
        model="gpt-3.5-turbo",
        max_tokens=1000,
        temperature=0.7,
        model_api_key="dummy_key",
    )

    actions = [
        FindClass(repository=repository, code_index=code_index, completion_model=completion_model.clone()),
        FindFunction(repository=repository, code_index=code_index, completion_model=completion_model.clone()),
        FindCodeSnippet(repository=repository, code_index=code_index, completion_model=completion_model.clone()),
        SemanticSearch(repository=repository, code_index=code_index, completion_model=completion_model.clone()),
        ViewCode(repository=repository, completion_model=completion_model.clone()),
        StringReplace(repository=repository, completion_model=completion_model.clone()),
        RunTests(repository=repository, code_index=code_index, runtime=runtime),
        Finish(),
        Reject(),
    ]

    original_agent = CodingAgent(actions=actions, completion=completion_model, system_prompt="test")

    dumped_agent = json.dumps(original_agent.model_dump(), indent=2)
    print(dumped_agent)

    # Load the agent from JSON
    loaded_agent_data = json.loads(dumped_agent)
    loaded_agent = CodingAgent.model_validate(loaded_agent_data, repository=repository, code_index=code_index, runtime=runtime)

    # Manually set the dependencies after loading
    for action in loaded_agent.actions:
        if isinstance(action, SearchBaseAction):
            action._repository = repository
            action._code_index = code_index
        if hasattr(action, "runtime"):
            action.runtime = runtime

    # Assert that the number of actions is the same
    assert len(original_agent.actions) == len(loaded_agent.actions)

    # Assert that all action types are preserved
    original_action_types = set(type(action) for action in original_agent.actions)
    loaded_action_types = set(type(action) for action in loaded_agent.actions)
    assert original_action_types == loaded_action_types

    # Assert that the _action_map is correctly populated
    assert set(original_agent._action_map.keys()) == set(
        loaded_agent._action_map.keys()
    )

    # Assert that the completion model is preserved
    assert isinstance(loaded_agent.completion, BaseCompletionModel)
    assert loaded_agent.completion.model == original_agent.completion.model
    assert loaded_agent.completion.max_tokens == original_agent.completion.max_tokens
    assert loaded_agent.completion.temperature == original_agent.completion.temperature
    assert not loaded_agent.completion.model_api_key

    # Check if the dependencies are correctly set for each action
    for action in loaded_agent.actions:
        if hasattr(action, "_repository"):
            assert action._repository == repository
        if hasattr(action, "_code_index"):
            assert action._code_index == code_index
        if hasattr(action, "_runtime"):
            assert action._runtime == runtime

    for original_action, loaded_action in zip(
        original_agent.actions, loaded_agent.actions
    ):
        assert type(original_action) == type(loaded_action)
        assert original_action.name == loaded_action.name


def test_deepseek_v4_uses_compact_native_prompt_and_progressive_search():
    agent = CodingAgent.create(
        repository=InMemRepository(),
        code_index=MockCodeIndex(),
        runtime=MockRuntimeEnvironment(),
        model="deepseek-v4-flash",
    )

    assert "# Repair Policy" in agent.system_prompt
    assert "# Workflow Overview" not in agent.system_prompt
    assert "STRICT Single Action" not in agent.system_prompt
    assert agent.use_few_shots is False
    assert agent.disable_thoughts is True
    assert agent.message_generator.max_tokens == 850000

    search_actions = [
        action
        for action in agent.actions
        if isinstance(action, (SearchBaseAction, EntityIntentSearch))
    ]
    assert search_actions
    assert all(action.progressive_disclosure for action in search_actions)
    assert "For overloaded or templated APIs" in agent.system_prompt
    assert "language branches" in agent.system_prompt
    assert "conversions, and exceptions" in agent.system_prompt


def test_explicit_system_prompt_override_keeps_tools_without_production_policy():
    prompt = "Standalone repair prompt."
    agent = CodingAgent.create(
        repository=InMemRepository(),
        code_index=MockCodeIndex(),
        runtime=MockRuntimeEnvironment(),
        model="deepseek-v4-flash",
        system_prompt_override=prompt,
    )

    assert agent.system_prompt == prompt
    assert agent.use_few_shots is False
    names = {action.__class__.__name__ for action in agent.actions}
    assert {"EntityIntentSearch", "EntitySearchSymbol", "RunCommand",
            "StringReplace", "RunSingleTest", "VerifiedFinish"} <= names


def test_graph_ablation_repair_keeps_source_and_verification_tools_only():
    agent = CodingAgent.create(
        repository=InMemRepository(),
        code_index=None,
        runtime=MockRuntimeEnvironment(),
        model="deepseek-v4-flash",
        soft_verification_finish=True,
        enable_indexed_tools=False,
    )

    names = {action.__class__.__name__ for action in agent.actions}
    assert {"ViewCode", "RunCommand", "RunSingleTest", "RunRegression",
            "RunIntentCheck", "StringReplace", "VerifiedFinish"} <= names
    assert not names.intersection({
        "EntityIntentSearch", "EntityFunctionSemanticSearch", "EntitySearchSymbol",
        "FindEntityUsage", "FindTemplateInstantiation", "GetMacroUsages",
    })
    assert "EntityIntentSearch" not in agent.system_prompt
    assert "FindEntityUsage" not in agent.system_prompt
    assert "focused `grep`" in agent.system_prompt
    assert "candidate pool records" in agent.system_prompt


def test_compact_repair_prompt_forbids_test_evasion():
    prompt = generate_compact_native_prompt(has_runtime=True)
    no_runtime_prompt = generate_compact_native_prompt(has_runtime=False)

    assert "Do not modify repository tests" in prompt
    assert "otherwise fix production code" in prompt
    assert "The Issue is authoritative" in prompt
    assert "completion obligations" in prompt
    assert "including all `[F2P]` and `[P2P]` checks" in prompt
    assert "Every required check must pass" in prompt
    assert "do\n   not call `Finish`" in prompt
    assert "unrun, stale, unavailable, or failed" in prompt
    assert "rerun every required ACCEPT-N" in prompt
    assert "Diagnostic-only ACCEPT-N" in prompt
    assert "optional and non-binding" in prompt
    assert "If no required ACCEPT-N is listed" in prompt
    assert "not a hard finish gate" not in prompt
    assert "do not broaden the repair merely to force a diagnostic-only" in prompt
    assert "`rg` is unavailable" in prompt
    assert "both checks" not in prompt
    assert "Preserve public signatures, overload ranking, template deduction" in prompt
    assert (
        "Every additional production file or module must be justified by concrete evidence"
        in prompt
    )
    assert "revert or narrow that edit rather than stacking" in prompt
    assert "Every newly selected runtime helper must have a reachable definition" in prompt
    assert "not a reason to create another probe" in prompt
    assert "Do not edit CMake, source lists, or build flags" in prompt
    assert "run a focused public-API compile probe" in prompt
    assert "actual selection point" in prompt
    assert "which path wins when old and requested mechanisms are both viable" in prompt
    assert "do not invent one or spend steps on competing-path probes" in prompt
    assert "call `Finish` immediately" in prompt
    assert "numbered probe variants" in prompt
    assert "not an\n     equivalent implementation" in prompt
    assert "On conflict, follow the Issue" in prompt
    assert "Do not reduce" in prompt
    assert "minimum semantic impact only as a" in prompt
    assert "tie-breaker and state the assumption" in prompt
    normalized_prompt = " ".join(prompt.split())
    assert "Do not repeat anchor-test discovery" in normalized_prompt
    assert "form and test the smallest Issue-consistent candidate" in normalized_prompt
    assert "Patch Reviewer handles structural closure" not in normalized_prompt
    assert "You are responsible for the completeness of the production change" in normalized_prompt
    assert "BOUNDARY-N" not in prompt
    assert "create one repository-relative `repro*.cpp`" in prompt
    assert "Frozen ACCEPT witnesses are separate contract checks" in prompt
    assert "do not copy an" in prompt
    assert "required behavior to remove ambiguity" in prompt
    assert "Only independent read/search calls may be batched" in prompt
    assert "Route commands by effect" in prompt
    assert "RunSingleTest` only for a focused compile/test command" in prompt
    assert "capture_generated_changes=true" in prompt
    assert "pre-existing" in prompt
    assert "Do not create a temporary transformation script" in prompt
    assert "generated artifact is not an authoritative edit surface" in prompt
    assert "later repository build will overwrite" in prompt
    assert "compiler/linker error, observed call chain" in prompt
    assert "If an edit introduces unrelated errors" in prompt
    finish_condition = (
        "Call `Finish` only when the smallest complete repair satisfying the Issue"
    )
    assert finish_condition in prompt
    assert finish_condition in no_runtime_prompt
    assert "5. Required intent checks are completion obligations" in prompt
    assert "\n5. Required intent checks are completion obligations" in prompt
    assert "\n6. Call `Finish`" in prompt
    assert "Required intent checks are completion obligations" not in no_runtime_prompt
    assert "5. Call `Finish`" in no_runtime_prompt


def test_portfolio_compact_prompt_makes_verification_soft_but_honest():
    prompt = generate_compact_native_prompt(
        has_runtime=True,
        soft_verification_finish=True,
    )

    assert "Verification is evidence, not a completion gate" in prompt
    assert "Report every" in prompt
    assert "failed or missing checks do not by themselves prohibit Finish" in prompt
    assert "Every required check must pass" not in prompt
    assert "Do not modify repository tests" in prompt
    assert "fallback-only" in prompt
    assert "compile/link failure in an existing" in prompt
    assert "candidate's own assertion" in prompt


def test_portfolio_action_set_includes_soft_regression():
    agent = CodingAgent.create(
        repository=InMemRepository(),
        code_index=MockCodeIndex(),
        runtime=MockRuntimeEnvironment(),
        model="deepseek-v4-flash",
        soft_verification_finish=True,
        allow_regression=True,
    )

    assert any(isinstance(action, RunRegression) for action in agent.actions)
    assert any(isinstance(action, RunSingleTest) for action in agent.actions)
    assert "run\n   `RunRegression`" in agent.system_prompt


def test_execution_allowlist_blocks_action_without_reinitializing_tool_schema():
    agent = CodingAgent.create(
        repository=InMemRepository(),
        code_index=MockCodeIndex(),
        runtime=MockRuntimeEnvironment(),
        model="deepseek-v4-flash",
    )
    original_schema = agent.completion.response_schema
    action_step = SimpleNamespace(
        action=ViewCodeArgs(thoughts="Inspect code", files=[]),
        observation=None,
    )
    node = SimpleNamespace(terminal=False)

    agent.set_execution_action_allowlist({Finish.args_schema})
    agent._execute(node, action_step)

    assert agent.completion.response_schema == original_schema
    assert action_step.observation.expect_correction is True
    assert (
        action_step.observation.properties["fail_reason"]
        == "action_not_allowed_in_current_phase"
    )
    assert node.terminal is False
