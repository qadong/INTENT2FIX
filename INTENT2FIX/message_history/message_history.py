import logging
from typing import List

from pydantic import BaseModel, Field, model_serializer

from INTENT2FIX.completion.schema import ChatCompletionTextObject, ChatCompletionUserMessage, AllMessageValues
from INTENT2FIX.node import Node
from INTENT2FIX.schema import MessageHistoryType
from INTENT2FIX.utils.tokenizer import count_tokens

logger = logging.getLogger(__name__)


class MessageHistoryGenerator(BaseModel):
    message_history_type: MessageHistoryType = Field(
        default=MessageHistoryType.MESSAGES,
        description="Type of message history to generate",
    )
    max_tokens: int = Field(default=150000, description="Maximum number of tokens allowed in message history")
    max_observation_tokens: int = Field(
        default=8000,
        description="Maximum tokens retained from one tool observation.",
    )
    include_file_context: bool = Field(default=True, description="Whether to include file context in messages")
    include_git_patch: bool = Field(default=True, description="Whether to include git patch in messages")
    thoughts_in_action: bool = Field(
        default=False,
        description="Whether to include thoughts in the action or in the message",
    )

    @model_serializer
    def serialize_model(self) -> dict:
        return {
            "message_history_type": self.message_history_type.value,
            "max_tokens": self.max_tokens,
            "max_observation_tokens": self.max_observation_tokens,
            "include_file_context": self.include_file_context,
            "include_git_patch": self.include_git_patch,
            "thoughts_in_action": self.thoughts_in_action,
        }

    def generate_messages(self, node: Node) -> List[AllMessageValues]:  # type: ignore
        previous_nodes = node.get_trajectory()

        turns = []
        tool_idx = 0

        for i, previous_node in enumerate(previous_nodes):
            turn_messages = []
            turn_tokens = 0
            # Handle user message
            message_content = []
            if previous_node.user_message:
                message_content.append(ChatCompletionTextObject(type="text", text=previous_node.user_message))

            logger.info(f"Artifact changes: {previous_node.artifact_changes}")

            if previous_node.artifact_changes:
                for change in previous_node.artifact_changes:
                    artifact = previous_node.workspace.get_artifact_by_id(change.artifact_id)
                    if artifact:
                        message_content.append(
                            ChatCompletionTextObject(
                                type="text",
                                text=f"The artifact {artifact.id} with type {artifact.type} was {change.change_type} by the user",
                            )
                        )
                        message_content.append(artifact.to_prompt_message_content())

            if message_content:
                turn_messages.append(ChatCompletionUserMessage(role="user", content=message_content))
                turn_tokens += count_tokens(str(message_content))

            tool_calls = []
            tool_responses = []

            if not previous_node.assistant_message and not previous_node.action_steps:
                if turn_messages:
                    turns.append((turn_messages, turn_tokens))
                continue

            raw_tool_calls = previous_node.assistant_tool_calls or []

            for action_index, action_step in enumerate(previous_node.action_steps):
                tool_idx += 1

                exclude = None
                if not self.thoughts_in_action:
                    exclude = {"thoughts"}

                if action_index < len(raw_tool_calls):
                    tool_call = raw_tool_calls[action_index]
                    tool_call_id = tool_call.get("id") or f"tool_{tool_idx}"
                    tool_calls.append(tool_call)
                else:
                    tool_call_id = f"tool_{tool_idx}"
                    tool_calls.append(
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": action_step.action.name,
                                "arguments": action_step.action.model_dump_json(exclude=exclude),
                            },
                        }
                    )

                turn_tokens += count_tokens(action_step.action.model_dump_json(exclude=exclude))

                observation_content = self._truncate_observation(
                    action_step.observation.message
                    if action_step.observation and action_step.observation.message
                    else "No output found."
                )

                tool_responses.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": observation_content,
                    }
                )
                
                turn_tokens += count_tokens(observation_content)

            assistant_message = {"role": "assistant"}

            if tool_calls:
                assistant_message["tool_calls"] = tool_calls

            if previous_node.assistant_reasoning_content:
                assistant_message["reasoning_content"] = previous_node.assistant_reasoning_content
                turn_tokens += count_tokens(previous_node.assistant_reasoning_content)

            if previous_node.assistant_message:
                assistant_message["content"] = previous_node.assistant_message
                turn_tokens += count_tokens(previous_node.assistant_message)
            elif tool_calls:
                assistant_message["content"] = ""

            turn_messages.append(assistant_message)
            turn_messages.extend(tool_responses)
            turns.append((turn_messages, turn_tokens))

        if not turns:
            return []

        root_messages, root_tokens = turns[0]
        selected_turns = []
        tokens = root_tokens
        for turn_messages, turn_tokens in reversed(turns[1:]):
            if tokens + turn_tokens > self.max_tokens:
                logger.warning(
                    "Message history budget reached: %d + %d > %d tokens.",
                    tokens,
                    turn_tokens,
                    self.max_tokens,
                )
                break
            selected_turns.insert(0, turn_messages)
            tokens += turn_tokens

        messages = list(root_messages)
        for turn_messages in selected_turns:
            messages.extend(turn_messages)

        logger.info(f"Generated {len(messages)} messages with {tokens} tokens")

        return messages

    def _truncate_observation(self, content: str) -> str:
        tokens = count_tokens(content)
        if tokens <= self.max_observation_tokens:
            return content

        target_chars = max(
            1000,
            int(len(content) * self.max_observation_tokens / max(tokens, 1)),
        )
        marker = (
            f"\n\n... [observation truncated from {tokens} tokens] ...\n\n"
        )
        head_chars = target_chars // 2
        tail_chars = target_chars - head_chars
        return content[:head_chars] + marker + content[-tail_chars:]

    @classmethod
    def create(cls, message_history_type: MessageHistoryType, **obj):
        obj["message_history_type"] = message_history_type

        if message_history_type == MessageHistoryType.REACT:
            from INTENT2FIX.message_history.react import ReactMessageHistoryGenerator

            return ReactMessageHistoryGenerator(**obj)

        elif message_history_type == MessageHistoryType.MESSAGES_COMPACT:
            from INTENT2FIX.message_history.compact import CompactMessageHistoryGenerator

            return CompactMessageHistoryGenerator(**obj)

        elif message_history_type == MessageHistoryType.SUMMARY:
            from INTENT2FIX.message_history.summary import SummaryMessageHistoryGenerator

            return SummaryMessageHistoryGenerator(**obj)

        elif message_history_type == MessageHistoryType.MESSAGES:
            return cls(**obj)
        else:
            raise ValueError(f"Invalid message_history_type: {message_history_type}")

    @classmethod
    def model_validate(cls, obj):
        if isinstance(obj, dict) and "message_history_type" in obj:
            obj["message_history_type"] = MessageHistoryType(obj["message_history_type"])
            return cls.create(**obj)

        return obj
