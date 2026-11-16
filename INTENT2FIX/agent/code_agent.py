import json
import logging
import os
from typing import List

from INTENT2FIX.actions import (
    ViewCode,
)
from INTENT2FIX.actions.entity_search_symbol import EntitySearchSymbol
from INTENT2FIX.actions.entity_function_semantic_search import EntityFunctionSemanticSearch
from INTENT2FIX.actions.entity_intent_search import EntityIntentSearch
from INTENT2FIX.actions.get_macro_usages import GetMacroUsages
from INTENT2FIX.actions.find_template_instantiation import FindTemplateInstantiation
from INTENT2FIX.actions.find_entity_usage import FindEntityUsage
from INTENT2FIX.actions.list_features import ListFeatures
from INTENT2FIX.actions.action import Action
from INTENT2FIX.actions.append_string import AppendString
from INTENT2FIX.actions.claude_text_editor import ClaudeEditTool
from INTENT2FIX.actions.create_file import CreateFile
from INTENT2FIX.actions.finish import Finish
from INTENT2FIX.actions.list_files import ListFiles
from INTENT2FIX.actions.run_command import RunCommand
from INTENT2FIX.actions.read_source import ReadSource
from INTENT2FIX.actions.run_regression import RunRegression
from INTENT2FIX.actions.run_single_test import RunSingleTest
from INTENT2FIX.actions.run_intent_check import RunIntentCheck
from INTENT2FIX.actions.string_replace import StringReplace
from INTENT2FIX.actions.verified_finish import VerifiedFinish
from INTENT2FIX.actions.unified_repair import (
    UnifiedBash,
    UnifiedCodeGraph,
    UnifiedEditor,
)
from INTENT2FIX.agent.agent import ActionAgent
from INTENT2FIX.agent.code_prompts import (
    AGENT_ROLE,
    generate_react_guidelines,
    REACT_CORE_OPERATION_RULES,
    ADDITIONAL_NOTES,
    generate_workflow_prompt,
    REACT_CORE_OPERATION_RULES_NO_THOUGHTS,
    generate_compact_native_prompt,
    generate_guideline_prompt,
)
from INTENT2FIX.completion.base import (
    LLMResponseFormat,
    BaseCompletionModel,
)
from INTENT2FIX.index import CodeIndex
from INTENT2FIX.message_history import MessageHistoryGenerator
from INTENT2FIX.model_config import get_model_config
from INTENT2FIX.repository.repository import Repository
from INTENT2FIX.runtime.runtime import RuntimeEnvironment
from INTENT2FIX.schema import MessageHistoryType

logger = logging.getLogger(__name__)


class CodingAgent(ActionAgent):
    @classmethod
    def create(
        cls,
        repository: Repository,
        completion_model: BaseCompletionModel | None = None,
        model: str | None = None,
        code_index: CodeIndex | None = None,
        runtime: RuntimeEnvironment | None = None,
        message_history_type: MessageHistoryType | None = None,
        thoughts_in_action: bool | None = None,
        disable_thoughts: bool | None = None,
        few_shot_examples: bool | None = None,
        compact_native_prompt: bool | None = None,
        soft_verification_finish: bool = False,
        allow_regression: bool = True,
        message_history_max_tokens: int | None = None,
        evidence_archive_path: str | None = None,
        evidence_archive_sha256: str | None = None,
        enable_indexed_tools: bool = True,
        system_prompt_override: str | None = None,
        tool_profile: str | None = None,
        graph_issue_text: str = "",
        **kwargs,
    ):
        if bool(evidence_archive_path) != bool(evidence_archive_sha256):
            raise ValueError("Evidence reader requires both archive path and hash")
        if completion_model is None:
            if model is None:
                raise ValueError("Either completion_model or model name must be provided")

            # Get default config for the model from model_config
            model_config = get_model_config(model).copy()

            configured_compact_prompt = model_config.pop("compact_native_prompt", False)
            configured_history_max_tokens = model_config.pop("message_history_max_tokens", None)
            if compact_native_prompt is None:
                compact_native_prompt = configured_compact_prompt
            if message_history_max_tokens is None:
                message_history_max_tokens = configured_history_max_tokens

            # Set instance variables from model config if not explicitly provided
            if thoughts_in_action is None:
                thoughts_in_action = model_config.get("thoughts_in_action", False)
            if disable_thoughts is None:
                disable_thoughts = model_config.get("disable_thoughts", False)
            if few_shot_examples is None:
                few_shot_examples = model_config.get("few_shot_examples", True)

            # Override with any provided kwargs
            model_config.update(kwargs)

            # Create completion model
            completion_model = BaseCompletionModel.create(**model_config)
        else:
            # Clone the completion model to ensure we have our own instance
            completion_model = completion_model.clone()

            # Set instance variables from completion model if not explicitly provided
            if thoughts_in_action is None:
                thoughts_in_action = completion_model.thoughts_in_action
            if disable_thoughts is None:
                disable_thoughts = completion_model.disable_thoughts

        if compact_native_prompt is None:
            compact_native_prompt = False

        if message_history_type is None:
            if completion_model.response_format == LLMResponseFormat.TOOLS:
                message_history_type = MessageHistoryType.MESSAGES
            else:
                message_history_type = MessageHistoryType.REACT

        if few_shot_examples is None:
            few_shot_examples = message_history_type == MessageHistoryType.REACT

        action_completion_format = completion_model.response_format
        if action_completion_format != LLMResponseFormat.TOOLS:
            logger.info("Default to JSON as Response format for action completion model")
            action_completion_format = LLMResponseFormat.JSON

        # Create action completion model by cloning the input model with JSON response format
        action_completion_model = completion_model.clone(response_format=action_completion_format)
        # Inherit message_cache setting from the main completion model
        action_completion_model.message_cache = completion_model.message_cache

        if tool_profile is None:
            tool_profile = os.environ.get(
                "COTEST_CODING_TOOL_PROFILE",
                "standard",
            )

        supports_anthropic_computer_use = completion_model.model.startswith("claude-3-5-sonnet")

        if tool_profile == "unified_four":
            if runtime is None:
                raise ValueError("unified_four requires a runtime")
            if evidence_archive_path:
                raise ValueError(
                    "unified_four does not expose the separate evidence reader"
                )
            actions = create_unified_repair_actions(
                repository=repository,
                code_index=code_index,
                completion_model=action_completion_model,
                runtime=runtime,
                enable_indexed_tools=enable_indexed_tools,
                issue_text=graph_issue_text,
            )
            action_type = "unified four-tool repair actions"
        elif tool_profile != "standard":
            raise ValueError(f"Unknown coding tool profile: {tool_profile}")
        elif supports_anthropic_computer_use:
            actions = create_claude_coding_actions(
                repository=repository,
                code_index=code_index,
                completion_model=action_completion_model,
                runtime=runtime,
                enable_indexed_tools=enable_indexed_tools,
            )

            action_type = "Claude actions with computer use capability"
        else:
            actions = create_edit_code_actions(
                repository=repository,
                code_index=code_index,
                completion_model=action_completion_model,
                runtime=runtime,
                progressive_disclosure=compact_native_prompt,
                allow_regression=allow_regression,
                enable_indexed_tools=enable_indexed_tools,
            )
            action_type = "standard edit code actions"

        if evidence_archive_path:
            actions.append(ReadSource(archive_path=evidence_archive_path, archive_sha256=evidence_archive_sha256))

        # A caller may provide a deliberately small prompt for an isolated
        # experiment. Production callers keep the existing generated policy.
        if system_prompt_override is not None:
            system_prompt = system_prompt_override
            few_shot_examples = False
        else:
            system_prompt = AGENT_ROLE
            if compact_native_prompt:
                if completion_model.response_format != LLMResponseFormat.TOOLS:
                    raise ValueError("compact_native_prompt requires native tool calling")
                system_prompt += generate_compact_native_prompt(
                    runtime is not None,
                    soft_verification_finish=soft_verification_finish,
                    source_only_search=not enable_indexed_tools,
                )
                few_shot_examples = False
            elif completion_model.response_format == LLMResponseFormat.REACT:
                system_prompt += "\n"

                if disable_thoughts:
                    system_prompt += REACT_CORE_OPERATION_RULES_NO_THOUGHTS
                else:
                    system_prompt += REACT_CORE_OPERATION_RULES

                system_prompt += "\n**IMPORTANT:** You MUST STOP generating immediately after outputting the JSON for your ONE action. Do not predict thoughts or actions for the next turn."

            elif completion_model.response_format == LLMResponseFormat.TOOLS:
                system_prompt += generate_react_guidelines(disable_thoughts)
            else:
                raise ValueError(f"Unsupported response format: {completion_model.response_format}")

            if not compact_native_prompt:
                workflow_prompt = generate_workflow_prompt(
                    actions, runtime is not None,
                    source_only_search=not enable_indexed_tools,
                )
                system_prompt += (
                    "\n"
                    + workflow_prompt
                    + "\n"
                    + generate_guideline_prompt(runtime is not None, thoughts_in_action)
                    + "\n"
                    + ADDITIONAL_NOTES
                )

        if evidence_archive_path:
            system_prompt += (
                "\nDiagnostic summaries index frozen pre-repair evidence; they are not contracts."
                " Use read_source(source_id=...) to inspect original errors, tested operands,"
                " execution conditions, conflicting evidence or omitted source bodies before"
                " relying on those details. Continue from next_offset for the remaining text."
                " Use ViewCode for current repository code.\n"
            )

        message_generator_kwargs = {
            "message_history_type": message_history_type,
            "include_file_context": True,
            "thoughts_in_action": thoughts_in_action,
        }
        if message_history_max_tokens is not None:
            message_generator_kwargs["max_tokens"] = message_history_max_tokens
        message_generator = MessageHistoryGenerator.create(
            **message_generator_kwargs,
        )

        config = {
            "completion_model": completion_model.__class__.__name__,
            "code_index_enabled": code_index is not None,
            "runtime_enabled": runtime is not None,
            "action_type": action_type,
            "actions": [a.__class__.__name__ for a in actions],
            "message_history_type": message_history_type.value,
            "thoughts_in_action": thoughts_in_action,
            "file_context_enabled": True,
            "system_prompt_override": system_prompt_override is not None,
            "tool_profile": tool_profile,
        }

        logger.info(f"Created CodingAgent with configuration: {json.dumps(config, indent=2)}")

        return cls(
            completion=completion_model,
            actions=actions,
            system_prompt=system_prompt,
            message_generator=message_generator,
            use_few_shots=few_shot_examples,
            disable_thoughts=disable_thoughts,
            thoughts_in_action=thoughts_in_action,
            code_index=code_index,
            **kwargs,
        )


def create_base_actions(
    repository: Repository,
    code_index: CodeIndex | None = None,
    completion_model: BaseCompletionModel | None = None,
    progressive_disclosure: bool = False,
    enable_indexed_tools: bool = True,
) -> List[Action]:
    """Create the common base actions using entity_tools."""
    if not enable_indexed_tools:
        return [ViewCode(repository=repository, completion_model=completion_model.clone())]
    return [
        EntityFunctionSemanticSearch(
            code_index=code_index,
            repository=repository,
            completion_model=completion_model.clone(),
            progressive_disclosure=progressive_disclosure,
        ),
        EntityIntentSearch(
            code_index=code_index,
            repository=repository,
            completion_model=completion_model.clone(),
            progressive_disclosure=progressive_disclosure,
        ),
        EntitySearchSymbol(
            code_index=code_index,
            repository=repository,
            completion_model=completion_model.clone(),
            progressive_disclosure=progressive_disclosure,
        ),
        GetMacroUsages(
            code_index=code_index,
            repository=repository,
            completion_model=completion_model.clone(),
            progressive_disclosure=progressive_disclosure,
        ),
        FindTemplateInstantiation(
            code_index=code_index,
            repository=repository,
            completion_model=completion_model.clone(),
            progressive_disclosure=progressive_disclosure,
        ),
        FindEntityUsage(
            code_index=code_index,
            repository=repository,
            completion_model=completion_model.clone(),
            progressive_disclosure=progressive_disclosure,
        ),
        ViewCode(repository=repository, completion_model=completion_model.clone()),
    ]


def create_edit_code_actions(
    repository: Repository,
    code_index: CodeIndex | None = None,
    completion_model: BaseCompletionModel | None = None,
    runtime: RuntimeEnvironment | None = None,
    progressive_disclosure: bool = False,
    allow_regression: bool = True,
    enable_indexed_tools: bool = True,
) -> List[Action]:
    """Create a list of simple code modification actions."""
    actions = create_base_actions(
        repository,
        code_index,
        completion_model.clone(),
        progressive_disclosure=progressive_disclosure,
        enable_indexed_tools=enable_indexed_tools,
    )

    edit_actions = [
        StringReplace(repository=repository, code_index=code_index),
        # InsertLine(repository=repository,  code_index=code_index),
        CreateFile(repository=repository, code_index=code_index),
        AppendString(repository=repository, code_index=code_index),
    ]

    if runtime:
        edit_actions.append(RunCommand(docker_env=runtime))

    if runtime and allow_regression:
        edit_actions.append(RunRegression(runtime=runtime))
    if runtime:
        edit_actions.append(RunSingleTest(runtime=runtime))
        edit_actions.append(RunIntentCheck(runtime=runtime))

    actions.extend(edit_actions)
    actions.extend([VerifiedFinish()])
    return actions


def create_unified_repair_actions(
    repository: Repository,
    code_index: CodeIndex | None,
    completion_model: BaseCompletionModel,
    runtime: RuntimeEnvironment,
    enable_indexed_tools: bool = True,
    issue_text: str = "",
) -> List[Action]:
    """Expose one shell, one editor, one optional graph, and Finish."""
    actions: List[Action] = [
        UnifiedBash(runtime=runtime),
        UnifiedEditor(
            repository=repository,
            code_index=code_index,
            completion_model=completion_model.clone(),
            runtime=runtime,
        ),
    ]
    if enable_indexed_tools:
        if code_index is None:
            raise ValueError("CodeGraph requires an initialized code index")
        actions.append(
            UnifiedCodeGraph(
                code_index=code_index,
                completion_model=completion_model.clone(),
                issue_text=issue_text,
            )
        )
    actions.append(Finish())
    return actions


def create_claude_coding_actions(
    repository: Repository,
    code_index: CodeIndex | None = None,
    completion_model: BaseCompletionModel | None = None,
    runtime: RuntimeEnvironment | None = None,
    enable_indexed_tools: bool = True,
) -> List[Action]:
    actions = create_base_actions(
        repository, code_index, completion_model.clone(),
        enable_indexed_tools=enable_indexed_tools,
    )
    actions.append(
        ClaudeEditTool(
            code_index=code_index,
            repository=repository,
            completion_model=completion_model.clone(),
        ),
    )
    actions.append(ListFiles())
    if runtime:
        actions.append(RunRegression(runtime=runtime))
        actions.append(RunSingleTest(runtime=runtime))
    actions.extend([Finish()])
    return actions


def create_all_actions(
    repository: Repository,
    code_index: CodeIndex | None = None,
    completion_model: BaseCompletionModel | None = None,
) -> List[Action]:
    print("Creating all actions")
    actions = create_base_actions(repository, code_index, completion_model.clone())
    actions.extend(create_edit_code_actions(repository, code_index, completion_model.clone()))
    actions.append(
        ClaudeEditTool(code_index=code_index, repository=repository, completion_model=completion_model.clone())
    )
    return actions
