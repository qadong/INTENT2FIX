import json
import logging
from typing import List

from INTENT2FIX.actions import (
    ViewCode,
)
from INTENT2FIX.actions.entity_intent_search import EntityIntentSearch
from INTENT2FIX.actions.action import Action
from INTENT2FIX.actions.append_string import AppendString
from INTENT2FIX.actions.create_file import CreateFile
from INTENT2FIX.actions.finish import Finish
from INTENT2FIX.actions.list_files import ListFiles
from INTENT2FIX.actions.run_command import RunCommand
from INTENT2FIX.actions.run_regression import RunRegression
from INTENT2FIX.actions.run_single_test import RunSingleTest
from INTENT2FIX.actions.string_replace import StringReplace
from INTENT2FIX.actions.verified_finish import VerifiedFinish
from INTENT2FIX.agent.agent import ActionAgent
from INTENT2FIX.agent.ablation_no_structural_prompts import (
    AGENT_ROLE,
    generate_react_guidelines,
    REACT_CORE_OPERATION_RULES,
    ADDITIONAL_NOTES,
    generate_workflow_prompt,
    REACT_CORE_OPERATION_RULES_NO_THOUGHTS,
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


class NoStructuralCodingAgent(ActionAgent):
    """
    Ablated version of CodingAgent that excludes Symbol/Structural Search.
    """
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
                few_shot_examples = model_config.get("few_shot_examples", True)
            model_config.update(kwargs)
            completion_model = BaseCompletionModel.create(**model_config)
        else:
            completion_model = completion_model.clone()
            if thoughts_in_action is None:
                thoughts_in_action = completion_model.thoughts_in_action
            if disable_thoughts is None:
                disable_thoughts = completion_model.disable_thoughts

        if message_history_type is None:
            message_history_type = MessageHistoryType.REACT

        if few_shot_examples is None:
            few_shot_examples = message_history_type == MessageHistoryType.REACT

        action_completion_format = completion_model.response_format
        if action_completion_format != LLMResponseFormat.TOOLS:
            action_completion_format = LLMResponseFormat.JSON

        action_completion_model = completion_model.clone(response_format=action_completion_format)
        # Inherit message_cache setting from the main completion model
        action_completion_model.message_cache = completion_model.message_cache

        # ACTIONS: Exclude Symbol/Semantic/Usages
        actions = create_no_structural_edit_code_actions(
            repository=repository,
            code_index=code_index,
            completion_model=action_completion_model,
            runtime=runtime,
        )
        action_type = "ablation (no-structural) edit code actions"

        workflow_prompt = generate_workflow_prompt(actions, runtime is not None)
        
        system_prompt = AGENT_ROLE
        if completion_model.response_format == LLMResponseFormat.REACT:
            system_prompt += "\n"
            if disable_thoughts:
                system_prompt += REACT_CORE_OPERATION_RULES_NO_THOUGHTS
            else:
                system_prompt += REACT_CORE_OPERATION_RULES
        elif completion_model.response_format == LLMResponseFormat.TOOLS:
            system_prompt += generate_react_guidelines(disable_thoughts)

        system_prompt += (
            "\n" + workflow_prompt + "\n" +
            generate_guideline_prompt(runtime is not None, thoughts_in_action) + "\n" +
            ADDITIONAL_NOTES
        )

        message_generator = MessageHistoryGenerator.create(
            message_history_type=message_history_type,
            include_file_context=True,
            thoughts_in_action=thoughts_in_action,
        )

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


def create_no_structural_base_actions(
    repository: Repository,
    code_index: CodeIndex | None = None,
    completion_model: BaseCompletionModel | None = None,
) -> List[Action]:
    return [
        EntityIntentSearch(
            code_index=code_index,
            repository=repository,
            completion_model=completion_model.clone(),
        ),
        ViewCode(repository=repository, completion_model=completion_model.clone()),
    ]


def create_no_structural_edit_code_actions(
    repository: Repository,
    code_index: CodeIndex | None = None,
    completion_model: BaseCompletionModel | None = None,
    runtime: RuntimeEnvironment | None = None,
) -> List[Action]:
    actions = create_no_structural_base_actions(repository, code_index, completion_model.clone())
    edit_actions = [
        StringReplace(repository=repository, code_index=code_index),
        CreateFile(repository=repository, code_index=code_index),
        AppendString(repository=repository, code_index=code_index),
        ListFiles(),
    ]
    if runtime:
        edit_actions.append(RunCommand(docker_env=runtime))
        edit_actions.append(RunRegression(runtime=runtime))
        edit_actions.append(RunSingleTest(runtime=runtime))
    actions.extend(edit_actions)
    actions.extend([VerifiedFinish()])
    return actions
