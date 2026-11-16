import json
import logging
from typing import List, Dict, Any
from pydantic import BaseModel

from INTENT2FIX.actions import ViewCode
from INTENT2FIX.actions.entity_search_symbol import EntitySearchSymbol
from INTENT2FIX.actions.get_macro_usages import GetMacroUsages
from INTENT2FIX.actions.find_template_instantiation import FindTemplateInstantiation
from INTENT2FIX.actions.action import Action
from INTENT2FIX.actions.append_string import AppendString
from INTENT2FIX.actions.create_file import CreateFile
from INTENT2FIX.actions.string_replace import StringReplace
from INTENT2FIX.actions.run_command import RunCommand
from INTENT2FIX.actions.run_regression import RunRegression
from INTENT2FIX.actions.run_single_test import RunSingleTest
from INTENT2FIX.agent.agent import ActionAgent
from INTENT2FIX.agent.code_prompts import (
    generate_react_guidelines,
    REACT_CORE_OPERATION_RULES,
    REACT_CORE_OPERATION_RULES_NO_THOUGHTS,
)
from INTENT2FIX.agent.no_intent_test_prompts import (
    AGENT_ROLE,
    generate_test_enhancement_workflow,
    CORE_RULES,
    GUIDELINES,
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

class NoIntentTestAgent(ActionAgent):
    """
    An agent specialized in generating adversarial and boundary test cases,
    with No Intent Search ablation (EntityIntentSearch and EntityFunctionSemanticSearch removed).
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
            if completion_model.response_format == LLMResponseFormat.TOOLS:
                message_history_type = MessageHistoryType.MESSAGES
            else:
                message_history_type = MessageHistoryType.REACT

        if few_shot_examples is None:
            few_shot_examples = message_history_type == MessageHistoryType.REACT

        action_completion_format = completion_model.response_format
        if action_completion_format != LLMResponseFormat.TOOLS:
            action_completion_format = LLMResponseFormat.JSON

        action_completion_model = completion_model.clone(response_format=action_completion_format)
        # Inherit message_cache setting from the main completion model
        action_completion_model.message_cache = completion_model.message_cache

        # Create no-intent test enhancement actions
        actions = create_no_intent_test_enhancement_actions(
            repository=repository,
            code_index=code_index,
            completion_model=action_completion_model,
            runtime=runtime,
        )
        action_type = "no-intent test enhancement actions"

        # Generate workflow prompt based on available actions
        workflow_prompt = generate_test_enhancement_workflow(actions)
        
        react_rules = ""
        if completion_model.response_format == LLMResponseFormat.REACT:
            if disable_thoughts:
                react_rules = REACT_CORE_OPERATION_RULES_NO_THOUGHTS
            else:
                react_rules = REACT_CORE_OPERATION_RULES
            
            react_rules += "\n**IMPORTANT:** You MUST STOP generating immediately after outputting the JSON for your ONE action. Do not predict thoughts or actions for the next turn."

        elif completion_model.response_format == LLMResponseFormat.TOOLS:
            react_rules = generate_react_guidelines(disable_thoughts)
        else:
            raise ValueError(f"Unsupported response format: {completion_model.response_format}")

        if runtime:
            work_dir = getattr(runtime, 'work_dir', '/home')
            system_prompt = (
                AGENT_ROLE
                + "\n"
                + f"The project is located in the `{work_dir}` directory. All commands should be executed from this directory unless otherwise specified.\n"
                + react_rules
                + "\n"
                + CORE_RULES
                + "\n"
                + workflow_prompt
                + "\n"
                + GUIDELINES
            )
        else:
            system_prompt = (
                AGENT_ROLE
                + "\n"
                + react_rules
                + "\n"
                + CORE_RULES
                + "\n"
                + workflow_prompt
                + "\n"
                + GUIDELINES
            )
        
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

        logger.info(f"Created NoIntentTestAgent with configuration: {json.dumps(config, indent=2)}")

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


def create_no_intent_test_enhancement_actions(
    repository: Repository,
    code_index: CodeIndex | None = None,
    completion_model: BaseCompletionModel | None = None,
    runtime: RuntimeEnvironment | None = None,
) -> List[Action]:
    """
    Creates actions available to the NoIntentTestAgent.
    Excludes EntityIntentSearch and EntityFunctionSemanticSearch.
    """
    # Restricted search actions
    actions = [
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
        FindTemplateInstantiation(
            code_index=code_index,
            repository=repository,
            completion_model=completion_model.clone(),
        ),
        ViewCode(repository=repository, completion_model=completion_model.clone()),
    ]
    
    # Test file modification actions
    test_actions = [
        StringReplace(repository=repository, code_index=code_index),
        CreateFile(repository=repository, code_index=code_index),
        AppendString(repository=repository, code_index=code_index),
    ]
    
    # Runtime actions
    if runtime:
        test_actions.append(RunCommand(docker_env=runtime))
        test_actions.append(RunRegression(runtime=runtime))
        test_actions.append(RunSingleTest(runtime=runtime))
    
    # Finish action
    from INTENT2FIX.actions.test_enhancement_passed import TestEnhancementPassed
    from INTENT2FIX.actions.test_enhancement_need_repair import TestEnhancementNeedRepair
    test_actions.append(TestEnhancementPassed())
    test_actions.append(TestEnhancementNeedRepair())
    
    actions.extend(test_actions)
    return actions
