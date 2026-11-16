import json
import logging
from typing import List
from INTENT2FIX.actions.schema import Observation
from INTENT2FIX.node import ActionStep, Node
from INTENT2FIX.runtime.dockerenv import DockerEnv
from INTENT2FIX.reproduce.actions import (
    CreateFile,
    RunCommand,
    Finish,
    Reject

)
from INTENT2FIX.actions.action import Action
from INTENT2FIX.actions.append_string import AppendString
from INTENT2FIX.actions.claude_text_editor import ClaudeEditTool

from INTENT2FIX.actions.list_files import ListFiles
from INTENT2FIX.actions.reject import Reject
from INTENT2FIX.actions.run_tests import RunTests
from INTENT2FIX.actions.string_replace import StringReplace
from INTENT2FIX.actions.verified_finish import VerifiedFinish
from INTENT2FIX.reproduce.state import ReproductionSession
from INTENT2FIX.agent.agent import ActionAgent
from INTENT2FIX.agent.code_prompts import (
    generate_react_guidelines,
    REACT_CORE_OPERATION_RULES,
    REACT_CORE_OPERATION_RULES_NO_THOUGHTS,
)

from INTENT2FIX.reproduce.reproduction_prompts import (
    AGENT_ROLE,
    generate_workflow_prompt,
    generate_compact_native_prompt,
    CORE_RULES,
    GUIDELINES
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

logger = logging.getLogger(__name__)

class ReproductionAgent(ActionAgent):
    """
    An agent specialized in reproducing bugs.

    This agent is structurally identical to the CodingAgent but is configured
    with a different set of actions and a system prompt tailored for bug reproduction.
    """

    def _execute(self, node: Node, action_step: ActionStep):
        session = next(
            (
                action.session
                for action in self.actions
                if getattr(action, "session", None) is not None
            ),
            None,
        )
        if session:
            action_name = action_step.action.__class__.__name__.removesuffix("Args")
            rejection_reason = session.convergence_rejection(action_name)
            if rejection_reason:
                action_step.observation = Observation(
                    message=rejection_reason,
                    expect_correction=True,
                    properties={"rejected_action": action_name},
                )
                logger.info(
                    "Rejected reproduction action %s after candidate evidence: %s",
                    action_name,
                    rejection_reason,
                )
                return
        super()._execute(node, action_step)

    @classmethod
    def create(
        cls,
        repository: Repository,
        completion_model: BaseCompletionModel | None = None,
        model: str | None = None,
        code_index: CodeIndex | None = None,
        docker_env: DockerEnv | None = None,
        message_history_type: MessageHistoryType | None = None,
        thoughts_in_action: bool | None = None,
        disable_thoughts: bool | None = None,
        few_shot_examples: bool | None = None,
        compact_native_prompt: bool | None = None,
        message_history_max_tokens: int | None = None,
        intent_validating: bool = True,
        **kwargs,
    ):
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

        supports_anthropic_computer_use = completion_model.model.startswith("claude-3-5-sonnet")

        if supports_anthropic_computer_use:
            # actions = create_claude_coding_actions(
            #     repository=repository,
            #     code_index=code_index,
            #     completion_model=action_completion_model,
            #     runtime=runtime,
            # )

            # action_type = "Claude actions with computer use capability"
            pass
        else:
            actions = create_reproduction_actions(
                repository=repository,
                code_index=code_index,
                docker_env=docker_env,
                completion_model=action_completion_model,
                progressive_disclosure=bool(compact_native_prompt),
            )
            action_type = (
                "intent-grounded reproduction actions"
                if intent_validating
                else "issue-grounded reproduction actions"
            )

        if compact_native_prompt:
            if completion_model.response_format != LLMResponseFormat.TOOLS:
                raise ValueError("compact_native_prompt requires native tool calling")
            system_prompt = AGENT_ROLE + generate_compact_native_prompt(
                docker_env.work_dir,
                intent_validating=intent_validating,
            )
            few_shot_examples = False
        elif completion_model.response_format == LLMResponseFormat.REACT:
            system_prompt = AGENT_ROLE + "\n"
            system_prompt += "\n"

            if disable_thoughts:
                system_prompt += REACT_CORE_OPERATION_RULES_NO_THOUGHTS
            else:
                system_prompt += REACT_CORE_OPERATION_RULES

        elif completion_model.response_format == LLMResponseFormat.TOOLS:
            system_prompt = AGENT_ROLE
            system_prompt += generate_react_guidelines(disable_thoughts)
        else:
            raise ValueError(f"Unsupported response format: {completion_model.response_format}")

        if not compact_native_prompt:
            system_prompt += (
                f"\nThe project is located in the `{docker_env.work_dir}` directory.\n"
                + CORE_RULES
                + "\n"
                + generate_workflow_prompt(actions)
                + "\n"
                + GUIDELINES
            )
        message_generator_kwargs = {
            "message_history_type": message_history_type,
            "include_file_context": True,
            "thoughts_in_action": thoughts_in_action,
        }
        if message_history_max_tokens is not None:
            message_generator_kwargs["max_tokens"] = message_history_max_tokens
        message_generator = MessageHistoryGenerator.create(**message_generator_kwargs)

        config = {
            "completion_model": completion_model.__class__.__name__,
            "code_index_enabled": code_index is not None,
            "runtime_enabled": docker_env is not None,
            "action_type": action_type,
            "actions": [a.__class__.__name__ for a in actions],
            "message_history_type": message_history_type.value,
            "thoughts_in_action": thoughts_in_action,
            "file_context_enabled": True,
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

def create_reproduction_actions(
    repository: Repository,
    docker_env: DockerEnv,
    code_index: CodeIndex | None = None,
    completion_model: BaseCompletionModel | None = None,
    progressive_disclosure: bool = False,
) -> List[Action]:
    """
    Creates the specific list of actions available to the ReproductionAgent.
    """
    from INTENT2FIX.agent.code_agent import create_base_actions

    actions = create_base_actions(
        repository=repository,
        code_index=code_index,
        completion_model=completion_model,
        progressive_disclosure=progressive_disclosure,
    )
    session = ReproductionSession()
    actions.extend([
        CreateFile(repository=repository, docker_env=docker_env, code_index=code_index, session=session),
        RunCommand(docker_env=docker_env, session=session),
        Finish(session=session, docker_env=docker_env),
        Reject(),
    ])
    return actions
