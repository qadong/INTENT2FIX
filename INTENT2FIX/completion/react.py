import json
import logging
from textwrap import dedent
from typing import List, Dict, Any, Type, Optional

from INTENT2FIX.completion import BaseCompletionModel
from INTENT2FIX.completion.base import CompletionRetryError
from INTENT2FIX.completion.schema import ResponseSchema, ChatCompletionUserMessage, extract_json_from_message
from pydantic import ValidationError

logger = logging.getLogger(__name__)


class ReActCompletionModel(BaseCompletionModel):
    """ReAct-specific implementation of the completion model.

    This class handles:
    1. Converting response schemas into ReAct format instructions
    2. Parsing and validating ReAct format responses
    3. Managing thought inclusion in actions
    4. Validating action sequence format
    """

    def _prepare_system_prompt(self, system_prompt: str, response_schema: List[Type[ResponseSchema]]) -> str:
        """Add ReAct format instructions to system prompt.

        This method appends the ReAct format instructions and available
        actions with their schemas to the base system prompt.

        Args:
            system_prompt: Base system prompt
            response_schema: List of response schemas

        Returns:
            System prompt with ReAct format instructions
        """
        action_input_schemas = []
        for action in response_schema:
            action_input_schemas.append(f" * {action.name} {action.format_schema_for_llm()}")

        input_schemas = "\n\n".join(action_input_schemas)

        system_prompt += dedent(f"""\n# Response format

Use the following format:
{'''Thought: You should always think about what to do
''' if not self.disable_thoughts else ''}Action: The action to take followed by the input arguments based on the schema below

Use one of the following actions and provide input arguments matching the schema.

{input_schemas}

Important: Do not include multiple{' ' if self.disable_thoughts else ' Thought-'}Action blocks. Do not include code blocks or additional text outside of this format.
""")
        return system_prompt

    def _get_completion_params(self, schema: ResponseSchema) -> Dict[str, str | Dict | List]:
        params = super()._get_completion_params(schema)
        params["stop"] = ["Observation:"]
        return params

    def _validate_completion(
        self,
        completion_response: Any,
    ) -> tuple[List[ResponseSchema], Optional[str], List[str]]:
        """Validate and parse ReAct format responses.

        This method:
        1. Validates ReAct format structure
        2. Extracts thoughts and actions
        3. Parses action parameters
        4. Validates against schema

        Args:
            completion_response: Raw response from the LLM

        Returns:
            Tuple of:
            - List of validated ResponseSchema instances
            - Optional text response string
            - List of flags indicating any special conditions

        Raises:
            CompletionRejectError: For invalid format that should be retried
            CompletionRuntimeError: For fundamentally invalid responses
        """
        try:
            response_text = completion_response.choices[0].message.content

            self._validate_react_format(response_text)

            thought, action_input = self._extract_thought_action(response_text)
            
            # Try normal parsing first (for backward compatibility)
            try:
                action_name, action_input = self._parse_action(action_input)
            except ValueError as e:
                # If normal parsing fails and input starts with code block, try code block parsing
                if action_input.strip().startswith("```"):
                    try:
                        action_name, action_input = self._parse_action_from_code_block(action_input)
                    except Exception:
                        # Re-raise original error if code block parsing also fails
                        raise e
                else:
                    # Re-raise original error if not a code block
                    raise e
            
            action_name = self._resolve_action_name(action_name)
            action_class = self._get_action_class(action_name)

            if action_input.strip().startswith("<") or action_input.strip().startswith("```xml"):
                try:
                    action_request = action_class.model_validate_xml(action_input)
                except Exception as e:
                    format_example = (
                        action_class.format_schema_for_llm() if hasattr(action_class, "format_schema_for_llm") else ""
                    )
                    raise ValueError(
                        f"Invalid XML format for {action_name}. Error: {e}\n\n" f"Expected format:\n{format_example}"
                    )
            else:
                try:
                    action_request = action_class.model_validate_json(action_input)
                except Exception as e:
                    # Gemini (and some other models) sometimes returns "Path: ... / Content: ..."
                    # or wraps JSON in extra prose. Only attempt coercion on failure to avoid
                    # impacting well-formed outputs from other models.
                    coerced = None
                    try:
                        coerced = self._maybe_coerce_action_input(action_name, action_input)
                        if coerced is not None:
                            action_request = action_class.model_validate_json(coerced)
                        else:
                            raise e
                    except Exception:
                        raise ValueError(
                            f"Invalid format for {action_name}. Error: {e}\n\n"
                            f"Expected schema:\n{action_class.format_schema_for_llm()}"
                        )

            action_request.thoughts = thought
            return [action_request], None, []

        except Exception as e:
            logger.warning(f"ReAct parsing failed: {e}. Response: {response_text}")
            retry_message = ChatCompletionUserMessage(role="user", content=str(e))
            raise CompletionRetryError(
                message=str(e),
                retry_messages=[retry_message],
            ) from e

    def _validate_react_format(self, response_text: str) -> None:
        """Validate the ReAct format structure.

        Args:
            response_text: Raw response to validate

        Raises:
            ValueError: If format is invalid
        """
        # Split into lines and remove empty ones
        lines = [line.strip() for line in response_text.split("\n") if line.strip()]

        # Count occurrences of each section
        thought_count = sum(1 for line in lines if line.startswith("Thought:"))
        action_count = sum(1 for line in lines if line.startswith("Action:"))

        # Check for multiple action blocks
        if thought_count > 1 or action_count > 1:
            logger.warning(f"Multiple Thought or Action sections found in response: {response_text}")

        # Check if all sections exist
        if action_count < 1:
            raise ValueError("Response must have one 'Action:' section")

        if not self.disable_thoughts:
            # Find the starting lines for each section
            thought_line = next((i for i, line in enumerate(lines) if line.startswith("Thought:")), -1)
            action_line = next((i for i, line in enumerate(lines) if line.startswith("Action:")), -1)

            # Check if sections are in correct order (if Thought exists)
            if thought_line != -1 and not (thought_line < action_line):
                raise ValueError(
                    "Your response is incorrect. The Thought section must come before the Action section. Please try the same request again with the correct format."
                )

    def _extract_thought_action(self, response_text: str) -> tuple[str, str]:
        """Extract thought and action from response text.

        Args:
            response_text: Raw response text

        Returns:
            Tuple of (thought, action_input)

        Raises:
            ValueError: If sections can't be extracted
        """
        thought = ""
        action_start = response_text.find("Action:")
        if action_start == -1:
            raise ValueError("Missing Action section")

        if not self.disable_thoughts:
            thought_start = response_text.find("Thought:")

            if thought_start != -1 and thought_start < action_start:
                thought = response_text[thought_start + 8 : action_start].strip()

            action_input = response_text[action_start + 7 :].strip()
        else:
            action_input = response_text[action_start + 7 :].strip()

        # If the model accidentally includes another Thought/Action block after the first Action,
        # keep only the first block. This is conservative: it triggers only on explicit newlines.
        cut_points = []
        for marker in ("\nThought:", "\nAction:"):
            idx = action_input.find(marker)
            if idx != -1:
                cut_points.append(idx)
        if cut_points:
            action_input = action_input[: min(cut_points)].strip()

        return thought, action_input

    def _parse_action(self, action_input: str) -> tuple[str, str]:
        """Parse action name and input from action text.

        Args:
            action_input: Raw action text

        Returns:
            Tuple of (action_name, action_parameters)

        Raises:
            ValueError: If action format is invalid
        """
        action_lines = action_input.split("\n", 1)
        if len(action_lines) < 2:
            raise ValueError("Missing action name and input")

        return self._strip_action_name(action_lines[0]), action_lines[1].strip()

    def _parse_action_from_code_block(self, action_input: str) -> tuple[str, str]:
        """Parse action name and input from code block format (e.g., ```json).

        This method handles cases where the model outputs code blocks instead of
        the expected format. It extracts JSON from the code block and tries to
        match it against available action schemas.

        Args:
            action_input: Raw action text starting with code block

        Returns:
            Tuple of (action_name, action_parameters)

        Raises:
            ValueError: If action format is invalid or no matching action found
        """
        # Extract JSON from code block
        _, all_jsons = extract_json_from_message(action_input)
        if not all_jsons:
            raise ValueError("No valid JSON found in code block")
        
        json_data = all_jsons[0]
        json_str = json.dumps(json_data)
        
        # Try to match against all available action schemas
        if not isinstance(self.response_schema, list):
            schemas = [self.response_schema]
        else:
            schemas = self.response_schema
        
        # Try each schema to find a match
        for schema_class in schemas:
            try:
                # Try to validate the JSON against this schema
                schema_class.model_validate(json_data)
                # If validation succeeds, return the action name and JSON string
                return schema_class.name, json_str
            except ValidationError:
                # This schema doesn't match, try the next one
                continue
        
        # If no schema matched, raise an error
        action_names = [s.name for s in schemas]
        raise ValueError(
            f"Could not match JSON to any action. JSON: {json_str}. Available actions: {', '.join(action_names)}"
        )

    def _strip_action_name(self, raw: str) -> str:
        """Normalize action name token (handles backticks/quotes)."""
        name = (raw or "").strip()
        # Common Gemini formatting: Action: `CreateFile`
        if name.startswith("`") and name.endswith("`") and len(name) >= 2:
            name = name[1:-1].strip()
        if name.startswith('"') and name.endswith('"') and len(name) >= 2:
            name = name[1:-1].strip()
        if name.startswith("'") and name.endswith("'") and len(name) >= 2:
            name = name[1:-1].strip()
        return name

    def _resolve_action_name(self, action_name: str) -> str:
        """Resolve common action-name mismatches without affecting other models.

        We only apply aliases if the provided name does not match any available action schema.
        """
        cleaned = self._strip_action_name(action_name)

        if not isinstance(self.response_schema, list):
            schemas = [self.response_schema]
        else:
            schemas = self.response_schema
        valid_names = {s.name for s in schemas}

        if cleaned in valid_names:
            return cleaned

        aliases = {
            # Reproduce agents often expose *Args schemas, while models output the action name.
            "CreateFile": "CreateFileArgs",
            "RunCommand": "RunCommandArgs",
            "Finish": "Finish",
            "Reject": "Reject",
        }
        mapped = aliases.get(cleaned)
        if mapped and mapped in valid_names:
            return mapped

        # No change: let the normal error path handle it
        return cleaned

    def _maybe_coerce_action_input(self, action_name: str, action_input: str) -> str | None:
        """Coerce common non-JSON action argument formats into JSON.

        This is only called after normal JSON validation fails, to avoid impacting
        other models that already follow the schema.
        """
        text = (action_input or "").strip()
        if not text:
            return None

        # If it already looks like JSON, don't touch it.
        if text.startswith("{") or text.startswith("["):
            return None

        # If it looks like XML, don't touch it.
        if text.startswith("<") or text.startswith("```xml"):
            return None

        # Helper: try to extract first fenced code block content (any language).
        def extract_first_fenced_block(s: str) -> str | None:
            start = s.find("```")
            if start == -1:
                return None
            header_end = s.find("\n", start)
            if header_end == -1:
                return None
            end = s.find("```", header_end + 1)
            if end == -1:
                return None
            return s[header_end + 1 : end].strip("\n")

        # Helper: parse simple "Key: Value" lines.
        kv: dict[str, str] = {}
        for line in text.splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k = k.strip().lower()
            v = v.strip()
            if v.startswith("`") and v.endswith("`") and len(v) >= 2:
                v = v[1:-1]
            kv[k] = v.strip()

        if action_name == "CreateFileArgs":
            path = kv.get("path")
            # Prefer an explicit fenced code block as file_text, otherwise fall back to "content"
            file_text = extract_first_fenced_block(text)
            if not file_text:
                file_text = kv.get("file_text") or kv.get("content")
                if file_text:
                    # Some models append "Thought:" inline after Content; trim it.
                    if "Thought:" in file_text:
                        file_text = file_text.split("Thought:", 1)[0].rstrip()
            if not path or not file_text:
                return None
            return json.dumps({"path": path, "file_text": file_text})

        if action_name == "RunCommandArgs":
            # Try to find embedded JSON anywhere in the text first
            selected, all_jsons = extract_json_from_message(text)
            if all_jsons:
                for obj in all_jsons:
                    if isinstance(obj, dict) and "command" in obj:
                        return json.dumps({"command": obj["command"]})

            command = kv.get("command")
            if not command:
                command = kv.get("cmd")
            if not command:
                return None
            return json.dumps({"command": command})

        # No supported coercion for this action
        return None

    def _get_action_class(self, action_name: str) -> Optional[Type[ResponseSchema]]:
        """Get the action class for an action name.

        Args:
            action_name: Name of the action

        Returns:
            Matching ResponseSchema class

        Raises:
            ValueError: If action name is invalid
        """
        if not isinstance(self.response_schema, list):
            schemas = [self.response_schema]
        else:
            schemas = self.response_schema

        action_class = next((a for a in schemas if a.name == action_name), None)
        if not action_class:
            action_names = [a.name for a in schemas]
            raise ValueError(f"Unknown action: {action_name}. Available actions: {', '.join(action_names)}")

        return action_class
