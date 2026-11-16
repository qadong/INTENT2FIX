import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from INTENT2FIX.actions.run_tests import RunTestsArgs
from INTENT2FIX.actions.schema import ActionArguments
from INTENT2FIX.actions.view_code import ViewCodeArgs, CodeSpan
from INTENT2FIX.actions.view_diff import ViewDiffArgs
from INTENT2FIX.completion.schema import (
    ChatCompletionAssistantMessage,
    ChatCompletionToolMessage,
    ChatCompletionUserMessage,
    AllMessageValues,
)
from INTENT2FIX.message_history.message_history import MessageHistoryGenerator
from INTENT2FIX.node import Node
from INTENT2FIX.runtime.testresult_parser import LogType, TestStatus, analyze_log
from INTENT2FIX.utils.tokenizer import count_tokens

logger = logging.getLogger(__name__)


class NodeMessage(BaseModel):
    user_message: Optional[str] = Field(default=None, description="The user message")
    assistant_message: Optional[str] = Field(default=None, description="The assistant message")
    assistant_reasoning_content: Optional[str] = Field(
        default=None,
        description="Provider-specific reasoning content for assistant message replay",
    )
    action: Optional[ActionArguments] = Field(default=None, description="The action")
    assistant_tool_call: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Raw provider tool call for this action, if available",
    )
    observation: Optional[str] = Field(default=None, description="The observation")


class CompactMessageHistoryGenerator(MessageHistoryGenerator):
    message_cache: bool = Field(default=False, description="Cache the message history if the LLM supports it")

    def generate_messages(self, node: Node) -> List[AllMessageValues]:
        messages = []
        # print(f'!!!!!!!!!!!!length{len(node)}')
        # logging.DEBUG(f'!!!!!!!!!!!!length{len(node)}')
        node_messages = self.get_node_messages(node)

        tool_idx = 0
        tokens = 0

        for msg in node_messages:
            # 1. Handle user message
            if msg.user_message:
                messages.append(ChatCompletionUserMessage(role="user", content=msg.user_message))
                tokens += count_tokens(msg.user_message)

            # 2. Handle assistant message (without tool calls)
            if msg.assistant_message and not msg.action:
                messages.append(ChatCompletionAssistantMessage(role="assistant", content=msg.assistant_message))
                tokens += count_tokens(msg.assistant_message)

            # 3. Handle action and observation (tool calls)
            if msg.action:
                tool_calls = []
                tool_idx += 1
                tool_call_id = f"tool_{tool_idx}"

                exclude = None
                content = msg.assistant_message # Use assistant_message if present

                if not self.thoughts_in_action:
                    exclude = {"thoughts"}
                    if not content and hasattr(msg.action, "thoughts"):
                        content = msg.action.thoughts

                if msg.assistant_tool_call:
                    tool_call = msg.assistant_tool_call
                    tool_call_id = tool_call.get("id") or tool_call_id
                else:
                    tool_call = {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": msg.action.name,
                            "arguments": msg.action.model_dump_json(exclude=exclude),
                        },
                    }

                tool_calls.append(tool_call)
                tokens += count_tokens(msg.action.model_dump_json(exclude=exclude))

                assistant_message = ChatCompletionAssistantMessage(
                    role="assistant",
                    tool_calls=tool_calls,
                    content=content or "",
                )
                if msg.assistant_reasoning_content:
                    assistant_message["reasoning_content"] = msg.assistant_reasoning_content
                messages.append(assistant_message)
                
                if msg.observation:
                    observation_content = msg.observation
                    # Avoid double prefixing
                    if not observation_content.startswith("Observation:"):
                        observation_content = f"Observation: {observation_content}"
                        
                    messages.append(
                        ChatCompletionToolMessage(
                            role="tool",
                            tool_call_id=tool_call_id,
                            content=observation_content,
                        )
                    )
                    tokens += count_tokens(observation_content)

        logger.info(f"Generated {len(messages)} messages with {tokens} tokens")

        return messages

    def get_node_messages(self, node: "Node") -> List[NodeMessage]:
        """
        Creates a list of (action, observation) tuples from the node's trajectory,
        respecting token limits by prioritizing the initial task and recent interactions.
        """
        previous_nodes = node.get_trajectory()
        logger.info(f"Generating message history from {len(previous_nodes)} previous nodes.")

        # 1. Initialize with the root message (core task)
        root_message_content = node.get_root().message
        root_message = NodeMessage(user_message=root_message_content)
        
        # The initial token count starts with the root message.
        # DO NOT include file_context.context_size() here, as it's added to the prompt separately.
        total_tokens = count_tokens(root_message_content)

        # 2. Process history from most recent to oldest to fill the remaining token budget
        # We skip the first node because it's the root message, which is already added.
        reversed_history = list(reversed(previous_nodes[1:]))
        
        # This set will only contain files present in the final, truncated context
        shown_files = set() 
        
        # This will hold the messages that fit, in chronological order
        recent_history: List[NodeMessage] = []

        for i, previous_node in enumerate(reversed_history):
            current_turn_messages: List[NodeMessage] = []
            turn_tokens = 0

            # A. Handle user/assistant messages for the turn
            user_message = previous_node.user_message
            assistant_message = previous_node.assistant_message

            if previous_node.feedback_data and not user_message:
                user_message = previous_node.feedback_data.feedback

            if user_message or assistant_message:
                msg = NodeMessage(
                    user_message=user_message,
                    assistant_message=assistant_message,
                    assistant_reasoning_content=previous_node.assistant_reasoning_content,
                )
                current_turn_messages.append(msg)
                turn_tokens += count_tokens(user_message) if user_message else 0
                turn_tokens += count_tokens(assistant_message) if assistant_message else 0

            # B. Handle action steps for the turn
            if previous_node.action_steps:
                raw_tool_calls = previous_node.assistant_tool_calls or []
                for action_index, action_step in enumerate(previous_node.action_steps):
                    action = action_step.action
                    
                    # Always prefer message over summary if it's a search or view action
                    # to ensure the agent sees the actual code snippets.
                    # We check if the action class is a search or view action, or if the name matches.
                    action_name = action.name if action and action.name else ""
                    is_search_or_view = (
                        action_name in ["ViewCode", "EntitySearchSymbol", "EntityFunctionSemanticSearch", "EntityIntentSearch", "GetMacroUsages", "FindTemplateInstantiation"]
                        or "Search" in action_name 
                        or "Find" in action_name
                        or "View" in action_name
                    )

                    if i == 0 or is_search_or_view:
                        observation_str = action_step.observation.message if action_step.observation else "No output found."
                    else:
                        observation_str = (
                            action_step.observation.summary
                            if self.include_file_context and hasattr(action_step.observation, "summary") and action_step.observation.summary
                            else action_step.observation.message if action_step.observation else "No output found."
                        )

                    action_tokens = count_tokens(action.model_dump_json())
                    observation_tokens = count_tokens(observation_str)
                    
                    assistant_tool_call = (
                        raw_tool_calls[action_index]
                        if action_index < len(raw_tool_calls)
                        else None
                    )
                    current_turn_messages.append(
                        NodeMessage(
                            action=action,
                            assistant_reasoning_content=previous_node.assistant_reasoning_content,
                            assistant_tool_call=assistant_tool_call,
                            observation=observation_str,
                        )
                    )
                    turn_tokens += action_tokens + observation_tokens

            # C. Check if the entire turn can fit in the remaining budget
            if total_tokens + turn_tokens <= self.max_tokens:
                total_tokens += turn_tokens
                # Prepend the turn's messages to the front of our recent history list
                recent_history = current_turn_messages + recent_history

                # Now that we know this turn is included, update shown_files
                for msg in current_turn_messages:
                    if msg.action and msg.action.name == "ViewCode":
                        file_path = msg.action.files[0].file_path
                        shown_files.add(file_path)
            else:
                logger.warning(
                    f"Token limit reached. Truncating history before this turn. "
                    f"Total tokens would be {total_tokens + turn_tokens} > {self.max_tokens}."
                )
                break  # Stop processing older nodes

        # 3. Final Assembly
        final_messages = [root_message] + recent_history

        logger.info(f"Generated final message history with {total_tokens} tokens and {len(final_messages)} messages.")
        return final_messages
