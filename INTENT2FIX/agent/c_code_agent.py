import json
import logging
from typing import List

from INTENT2FIX.actions import (
    ViewCode,
)
from INTENT2FIX.actions.action import Action
from INTENT2FIX.actions.append_string import AppendString
from INTENT2FIX.actions.claude_text_editor import ClaudeEditTool
from INTENT2FIX.actions.create_file import CreateFile
from INTENT2FIX.actions.entity_search_symbol import EntitySearchSymbol
from INTENT2FIX.actions.finish import Finish
from INTENT2FIX.actions.get_macro_usages import GetMacroUsages
from INTENT2FIX.actions.list_files import ListFiles
from INTENT2FIX.actions.run_command import RunCommand
from INTENT2FIX.actions.run_regression import RunRegression
from INTENT2FIX.actions.run_single_test import RunSingleTest
from INTENT2FIX.actions.string_replace import StringReplace
from INTENT2FIX.actions.verified_finish import VerifiedFinish
from INTENT2FIX.agent.agent import ActionAgent
from INTENT2FIX.agent.c_code_prompts import (
    ADDITIONAL_NOTES,
    AGENT_ROLE,
    REACT_CORE_OPERATION_RULES,
    REACT_CORE_OPERATION_RULES_NO_THOUGHTS,
    generate_guideline_prompt,
    generate_react_guidelines,
    generate_workflow_prompt,
)
from INTENT2FIX.completion.schema import ChatCompletionAssistantMessage, ChatCompletionUserMessage, AllMessageValues
from INTENT2FIX.completion.base import (
    BaseCompletionModel,
    LLMResponseFormat,
)
from INTENT2FIX.index import CodeIndex
from INTENT2FIX.message_history import MessageHistoryGenerator
from INTENT2FIX.message_history.react import ReactMessageHistoryGenerator
from INTENT2FIX.model_config import get_model_config
from INTENT2FIX.repository.repository import Repository
from INTENT2FIX.runtime.runtime import RuntimeEnvironment
from INTENT2FIX.schema import MessageHistoryType
from INTENT2FIX.utils.tokenizer import count_tokens

logger = logging.getLogger(__name__)


def _action_display_name(action_args) -> str:
    class_name = action_args.__class__.__name__
    if class_name.endswith("Args"):
        return class_name[:-4]
    return class_name


class CReactMessageHistoryGenerator(ReactMessageHistoryGenerator):
    """ReAct history formatter that keeps action names distinct from symbol names."""

    def generate_messages(self, node) -> List[AllMessageValues]:
        node_messages = self.get_node_messages(node)
        logger.info(f"Node messages: {len(node_messages)}")

        messages = []

        for node_message in node_messages:
            if node_message.user_message:
                messages.append(ChatCompletionUserMessage(role="user", content=node_message.user_message))

            if node_message.assistant_message:
                messages.append(ChatCompletionAssistantMessage(role="assistant", content=node_message.assistant_message))

            if node_message.action:
                if self.disable_thoughts:
                    thought = ""
                else:
                    thought = (
                        f"Thought: {node_message.action.thoughts}"
                        if hasattr(node_message.action, "thoughts")
                        else ""
                    )

                action_str = f"Action: {_action_display_name(node_message.action)}"
                action_input = node_message.action.format_args_for_llm()

                assistant_content = f"{thought}\n{action_str}" if thought else action_str
                if action_input:
                    assistant_content += f"\n{action_input}"

                messages.append(ChatCompletionAssistantMessage(role="assistant", content=assistant_content))
                if node_message.observation:
                    messages.append(ChatCompletionUserMessage(role="user", content=f"Observation: {node_message.observation}"))

        tokens = count_tokens("".join([m["content"] for m in messages if m.get("content")]))
        logger.info(f"Generated {len(messages)} messages with {tokens} tokens")
        return messages


class CCodingAgent(ActionAgent):
    """C-specific CodingAgent copy with a C-oriented toolset and prompt."""

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
        **kwargs,
    ):
        if completion_model is None:
            if model is None:
                raise ValueError("Either completion_model or model name must be provided")

            model_config = get_model_config(model)

            if thoughts_in_action is None:
                thoughts_in_action = model_config.get("thoughts_in_action", False)
            if disable_thoughts is None:
                disable_thoughts = model_config.get("disable_thoughts", False)
            if few_shot_examples is None:
                few_shot_examples = True

            model_config.update(kwargs)
            completion_model = BaseCompletionModel.create(**model_config)
        else:
            completion_model = completion_model.clone()

            if thoughts_in_action is None:
                thoughts_in_action = completion_model.thoughts_in_action
            if disable_thoughts is None:
                disable_thoughts = completion_model.disable_thoughts

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

        action_completion_model = completion_model.clone(response_format=action_completion_format)
        action_completion_model.message_cache = completion_model.message_cache

        supports_anthropic_computer_use = completion_model.model.startswith("claude-3-5-sonnet")

        if supports_anthropic_computer_use:
            actions = create_c_claude_coding_actions(
                repository=repository,
                code_index=code_index,
                completion_model=action_completion_model,
                runtime=runtime,
            )
            action_type = "C Claude actions"
        else:
            actions = create_c_edit_code_actions(
                repository=repository,
                code_index=code_index,
                completion_model=action_completion_model,
                runtime=runtime,
            )
            action_type = "C edit actions"

        workflow_prompt = generate_workflow_prompt(actions, runtime is not None)
        print(f'+++++++++++++++++++++++++++++++++++++++++\n{workflow_prompt}')

        system_prompt = AGENT_ROLE
        if completion_model.response_format == LLMResponseFormat.REACT:
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

        system_prompt += (
            "\n"
            + workflow_prompt
            + "\n"
            + generate_guideline_prompt(runtime is not None, thoughts_in_action)
            + "\n"
            + ADDITIONAL_NOTES
        )

        if message_history_type == MessageHistoryType.REACT:
            message_generator = CReactMessageHistoryGenerator(
                message_history_type=message_history_type,
                include_file_context=True,
                thoughts_in_action=thoughts_in_action,
                disable_thoughts=disable_thoughts,
            )
        else:
            message_generator = MessageHistoryGenerator.create(
                message_history_type=message_history_type,
                include_file_context=True,
                thoughts_in_action=thoughts_in_action,
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
        }

        logger.info(f"Created CCodingAgent with configuration: {json.dumps(config, indent=2)}")

        return cls(
            completion=completion_model,
            actions=actions,
            system_prompt=system_prompt,
            message_generator=message_generator,
            use_few_shots=few_shot_examples,
            thoughts_in_action=thoughts_in_action,
            code_index=code_index,
            **kwargs,
        )

    def generate_few_shots(self) -> str:
        return """

# Examples
Here are C-specific examples of how to use the available actions. Always put the tool class name after `Action:`.

**Example 1**
Task: Find the builtin function named f_match.
Thought: I have a likely C symbol name, so I should use symbol search with the function type.
Action: EntitySearchSymbol
{"name": "f_match", "file_pattern": "src/*.c", "type": "function", "name_exact_match": false, "limit": 20}

**Example 2**
Task: Find where the ONIG_OPTION_FIND_LONGEST macro is used.
Thought: This is a macro, so I should use the macro usage tool instead of a function search.
Action: GetMacroUsages
{"name": "ONIG_OPTION_FIND_LONGEST", "file_pattern": "src/*"}

**Example 3**
Task: Symbol search did not find a builtin name from the issue.
Thought: The user-facing builtin name may not match a C function name, so I should use text search to locate references.
Action: RunCommand
{"command": "rg -n \"capture|match\" src/ tests/"}

**Example 4**
Task: Inspect a likely implementation region after finding a file and line range.
Thought: I found the relevant code in src/builtin.c and need to read the surrounding implementation before editing.
Action: ViewCode
{"files": [{"file_path": "src/builtin.c", "start_line": 980, "end_line": 1100, "span_ids": []}]}
"""


def create_c_base_actions(
    repository: Repository,
    code_index: CodeIndex | None = None,
    completion_model: BaseCompletionModel | None = None,
) -> List[Action]:
    """Create C runner search/view actions."""
    return [
        EntitySearchSymbol(
            code_index=code_index,
            repository=repository,
            completion_model=completion_model.clone(),
        ),
        GetMacroUsages(
            code_index=code_index,
            repository=repository,
            completion_model=completion_model.clone(),
        ),
        ViewCode(repository=repository, completion_model=completion_model.clone()),
    ]


def create_c_edit_code_actions(
    repository: Repository,
    code_index: CodeIndex | None = None,
    completion_model: BaseCompletionModel | None = None,
    runtime: RuntimeEnvironment | None = None,
) -> List[Action]:
    actions = create_c_base_actions(repository, code_index, completion_model.clone())

    actions.extend(
        [
            StringReplace(repository=repository, code_index=code_index),
            CreateFile(repository=repository, code_index=code_index),
            AppendString(repository=repository, code_index=code_index),
        ]
    )

    if runtime:
        actions.append(RunCommand(docker_env=runtime))
        actions.append(RunRegression(runtime=runtime))
        actions.append(RunSingleTest(runtime=runtime))

    actions.append(VerifiedFinish())
    return actions


def create_c_claude_coding_actions(
    repository: Repository,
    code_index: CodeIndex | None = None,
    completion_model: BaseCompletionModel | None = None,
    runtime: RuntimeEnvironment | None = None,
) -> List[Action]:
    actions = create_c_base_actions(repository, code_index, completion_model.clone())
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
    actions.append(Finish())
    return actions
