import json
import logging
from typing import List, Dict, Any, Type, Union, Optional

from pydantic import PrivateAttr, ValidationError

from INTENT2FIX.completion import BaseCompletionModel
from INTENT2FIX.completion.base import CompletionRetryError
from INTENT2FIX.completion.schema import (
    ChatCompletionToolMessage,
    ResponseSchema,
)
from INTENT2FIX.completion.structured_compactor import (
    StructuredCompactionError,
    compact_structured_payload_once,
    is_compactable_validation_error,
)

logger = logging.getLogger(__name__)


class ToolCallCompletionModel(BaseCompletionModel):
    """Tool call-specific implementation of the completion model.

    This class handles:
    1. Converting response schemas into OpenAI function/tool schemas
    2. Configuring the LLM to use tool calls
    3. Validating and parsing tool call responses
    4. Managing thought inclusion in tool calls
    """

    _no_tool_retry_used: bool = PrivateAttr(default=False)
    _structured_compaction_used: bool = PrivateAttr(default=False)

    def create_completion(self, messages: List[dict]):
        self._no_tool_retry_used = False
        self._structured_compaction_used = False
        return super().create_completion(messages)

    def _get_completion_params(self, schema: List[Type[ResponseSchema]]) -> Dict[str, Union[str, Dict, List]]:
        """Get tool call-specific completion parameters.

        This method:
        1. Converts schemas to OpenAI tool format
        2. Configures tool choice behavior
        3. Handles thought inclusion settings

        Args:
            schema: List of prepared tool schemas

        Returns:
            Parameters for tool-based completion including:
            - tools: List of available tools/functions
            - tool_choice: How tools should be selected
        """
        params = {
            "tools": [s.tool_schema(thoughts_in_action=self.thoughts_in_action) for s in schema],
        }
        if self.tool_choice is not None and not self._thinking_mode_disallows_tool_choice():
            params["tool_choice"] = self.tool_choice
        return params

    def _prepare_system_prompt(
        self,
        system_prompt: str,
        response_schema: Union[List[type[ResponseSchema]], type[ResponseSchema]],
    ) -> str:
        prompt = super()._prepare_system_prompt(system_prompt, response_schema)
        if not self.compact_structured_output:
            return prompt
        return prompt + """

# Compact Structured-Output Protocol
Reason in the provider's separate reasoning channel. Do not repeat that analysis
inside tool arguments. Operational tool arguments must remain exact. For a
Submit* tool, return only the decision, exact evidence/object IDs, and the
shortest reason needed to interpret them. Keep every bounded string materially
below its schema maximum, rank list items by importance, obey list limits, and
never add fields that are absent from the tool schema.
"""

    def _thinking_mode_disallows_tool_choice(self) -> bool:
        if "deepseek-v4" not in self.model:
            return False
        thinking = (self.extra_body or {}).get("thinking")
        if not isinstance(thinking, dict):
            return False
        return thinking.get("type") != "disabled"

    def _create_retry_message(self, tool_call: Any, error: str):
        # Be explicit here: some models will otherwise keep inventing tool names.
        return ChatCompletionToolMessage(
            role="tool",
            tool_call_id=tool_call.id,
            content=error,
        )

    def _execute_structured_compaction(self, prompt: str) -> str:
        """Run one isolated JSON-only compaction turn without agent history."""
        import litellm

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "max_tokens": min(max(self.max_tokens, 2048), 8192),
            "timeout": self.timeout,
            "max_retries": 0,
            "metadata": {
                **dict(self.metadata or {}),
                "stage": "structured_payload_compaction",
            },
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.model_base_url:
            kwargs["api_base"] = self.model_base_url
        if self.model_api_key:
            kwargs["api_key"] = self.model_api_key
        if "glm-5.3-flash" in self.model:
            # This provider's JSON mode strips `json` inside identifiers,
            # including facts rewritten by the compactor. Parse JSON locally.
            kwargs.pop("response_format")
            kwargs["extra_body"] = {
                "thinking": {
                    "type": "enabled",
                    "clear_thinking": False,
                },
                "reasoning_effort": "low",
            }

        response = litellm.completion(**kwargs)
        return response.choices[0].message.content or ""

    def _normalize_tool_name(self, raw: str | None) -> str:
        """Normalize tool name coming back from the model.

        Some models (notably Gemini) may wrap tool names in quotes/backticks or
        use the action name (e.g. CreateFile) instead of the args-schema name
        (e.g. CreateFileArgs). We only apply aliasing when the raw name is not
        directly found in the available tool schemas.
        """
        name = (raw or "").strip()
        if name.startswith("`") and name.endswith("`") and len(name) >= 2:
            name = name[1:-1].strip()
        if name.startswith('"') and name.endswith('"') and len(name) >= 2:
            name = name[1:-1].strip()
        if name.startswith("'") and name.endswith("'") and len(name) >= 2:
            name = name[1:-1].strip()
        return name

    def _resolve_tool_alias(self, tool_name: str, valid_names: List[str]) -> str:
        """Resolve common tool-name aliases when the model uses action names.

        This is a conservative fallback: if tool_name is already valid, we do nothing.
        """
        if tool_name in valid_names:
            return tool_name

        aliases = {
            "CreateFile": "CreateFileArgs",
            "RunCommand": "RunCommandArgs",
        }
        mapped = aliases.get(tool_name)
        if mapped and mapped in valid_names:
            return mapped

        return tool_name

    def _validate_completion(
        self,
        completion_response: Any,
    ) -> tuple[List[ResponseSchema], Optional[str], List[str]]:
        """Validate tool call completion response.

        Args:
            completion_response: Raw response from the LLM

        Returns:
            Tuple of:
            - List of validated structured outputs
            - Optional text response
            - List of flags

        Raises:
            CompletionRejectError: If validation fails
        """
        message = completion_response.choices[0].message
        content = message.content or ""

        # Thinking-mode providers cannot always enforce tool_choice="required".
        if not hasattr(message, "tool_calls") or not message.tool_calls:
            logged_content = content
            if len(logged_content) > 2000:
                logged_content = logged_content[:2000] + "... [truncated]"
            logger.warning(
                "Tool-call response contained no tool call. content=%r",
                logged_content,
            )
            if not self._no_tool_retry_used:
                self._no_tool_retry_used = True
                raise CompletionRetryError(
                    "Response contained no tool call; retrying once.",
                    retry_message={
                        "role": "user",
                        "content": (
                            "Your previous response did not call a tool. Call "
                            "exactly one available tool now. If the task is "
                            "complete, call the available submit or finish tool. "
                            "Do not answer in plain text."
                        ),
                    },
                )
            return [], content, []

        # Track seen arguments to detect duplicates
        seen_arguments = set()
        flags = []
        structured_outputs = []
        valid_names = [s.name for s in self.response_schema]
        invalid_function_names = []

        retry_messages = []
        retry = False

        for tool_call in message.tool_calls:
            tool_name = self._normalize_tool_name(getattr(tool_call.function, "name", None))
            tool_name = self._resolve_tool_alias(tool_name, valid_names)

            if tool_name not in valid_names:
                strict_error = (
                    f"Tool {tool_name} not found.\n"
                    f"YOU MUST call exactly one of these tools (name must match exactly): {valid_names}\n"
                    f"Do NOT invent new tool names. Do NOT wrap the tool name in backticks or quotes."
                )
                retry_message = self._create_retry_message(
                    tool_call, strict_error
                )
                retry_messages.append(retry_message)
                retry = True
                continue

            # Check for duplicate arguments
            if tool_call.function.arguments in seen_arguments:
                logger.warning(f"Duplicate tool call arguments found for {tool_call.function.name}")
                flags.append("duplicate_tool_call")
                continue

            seen_arguments.add(tool_call.function.arguments)

            # Parse and validate tool arguments
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                retry_message = self._create_retry_message(tool_call, f"Invalid JSON in tool args: {e}")
                retry_messages.append(retry_message)
                retry = True
                continue

            # Find matching schema for tool
            schema = None
            for s in self.response_schema:
                if s.name == tool_name:
                    schema = s
                    break

            if not schema:
                retry_message = self._create_retry_message(tool_call, f"Tool {tool_name} not found.")
                retry_messages.append(retry_message)
                retry = True
                continue

            try:
                validated = schema.model_validate(args)
                structured_outputs.append(validated)
            except ValidationError as e:
                if (
                    self.compact_structured_output
                    and tool_name.startswith("Submit")
                    and is_compactable_validation_error(e)
                    and not self._structured_compaction_used
                ):
                    self._structured_compaction_used = True
                    try:
                        validated, audit = compact_structured_payload_once(
                            args,
                            schema,
                            self._execute_structured_compaction,
                        )
                    except (StructuredCompactionError, ValidationError, ValueError) as compact_error:
                        logger.warning(
                            "Single-turn compaction failed for %s; returning the "
                            "schema error to the agent: %s",
                            tool_name,
                            compact_error,
                        )
                    except Exception as compact_error:
                        logger.warning(
                            "Single-turn compaction request failed for %s; returning "
                            "the schema error to the agent: %s",
                            tool_name,
                            compact_error,
                        )
                    else:
                        strategy = audit.strategy
                        logger.info(
                            "Compacted %s structured arguments: strategy=%s "
                            "paths=%s preserved_ids=%d",
                            tool_name,
                            strategy,
                            ",".join(audit.mutable_paths),
                            len(audit.reference_ids),
                        )
                        flags.append(
                            "llm_compacted_tool_arguments"
                            if strategy == "llm"
                            else "deterministically_compacted_tool_arguments"
                        )
                        structured_outputs.append(validated)
                        continue
                retry_message = self._create_retry_message(
                    tool_call,
                    f"Tool '{tool_name}' arguments failed schema validation. "
                    f"Correct this tool's arguments and call it again. Error: {e}",
                )
                retry_messages.append(retry_message)
                retry = True
                continue

            # Just in case we need to retry
            retry_message = self._create_retry_message(
                tool_call, f"Tool {tool_name} could not be executed as other tools failed."
            )
            retry_messages.append(retry_message)

        if retry:
            raise CompletionRetryError(
                message=f"Tool calls failed. Retry messages: {retry_messages}",
                retry_messages=retry_messages,
            )

        return structured_outputs, content, flags

    def _get_response_model(self, tool_name: str) -> Type[ResponseSchema]:
        """Get the response model for a tool name.

        Args:
            tool_name: Name of the tool to find schema for

        Returns:
            Matching ResponseSchema class or None if not found
        """
        for r in self.response_schema:
            if r.name == tool_name:
                return r
