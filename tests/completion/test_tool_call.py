import json
from unittest.mock import patch

import pytest
from litellm.types.utils import ModelResponse, Usage, Message, Function, ChatCompletionMessageToolCall
from pydantic import ConfigDict, Field

from INTENT2FIX.actions.schema import ActionArguments
from INTENT2FIX.completion.base import (
    CompletionRetryError,
    LLMResponseFormat,
    _empty_completion_retry_message,
    _is_transient_provider_error,
)
from INTENT2FIX.completion.tool_call import ToolCallCompletionModel
from INTENT2FIX.exceptions import CompletionRejectError, CompletionRuntimeError


class TestActionArgs(ActionArguments):
    command: str = Field(..., description="The command to execute")
    args: list[str] = Field(default_factory=list, description="Command arguments")

    class Config:
        title = "test_action"


class SubmitCompactReportArgs(ActionArguments):
    summary: str = Field(..., max_length=20)
    facts: list[str] = Field(default_factory=list, max_length=2)

    model_config = ConfigDict(title="SubmitCompactReport", extra="forbid")


def test_length_exhaustion_gets_concise_tool_call_retry_instruction():
    response = ModelResponse(
        choices=[{
            "message": Message(content="", role="assistant"),
            "finish_reason": "length",
            "index": 0,
        }]
    )

    message = _empty_completion_retry_message(response)

    assert "emit the next tool call early" in message


def test_non_length_empty_response_does_not_get_length_instruction():
    response = ModelResponse(
        choices=[{
            "message": Message(content="", role="assistant"),
            "finish_reason": "stop",
            "index": 0,
        }]
    )

    assert _empty_completion_retry_message(response) == ""


def test_deepseek_connection_api_error_is_transient():
    import litellm

    error = litellm.APIError(
        status_code=500,
        message="DeepseekException - Connection error.",
        llm_provider="deepseek",
        model="deepseek-v4-flash",
    )

    assert _is_transient_provider_error(error)


def test_generic_provider_5xx_api_error_is_transient():
    import litellm

    error = litellm.APIError(
        status_code=500,
        message="Internal network failure, please try again later.",
        llm_provider="openai",
        model="glm-5.3-flash",
    )

    assert _is_transient_provider_error(error)


def test_non_connection_api_error_is_not_transient():
    import litellm

    error = litellm.APIError(
        status_code=400,
        message="Invalid request payload",
        llm_provider="deepseek",
        model="deepseek-v4-flash",
    )

    assert not _is_transient_provider_error(error)


@pytest.fixture
def test_schema():
    return [TestActionArgs]


@pytest.fixture
def test_messages():
    return [{"role": "user", "content": "Run the test command"}]


@pytest.fixture
def mock_litellm_tool_response():
    """Mock LiteLLM response with tool calls"""
    def _create_mock(tool_calls=None, usage=None):
        # Create tool calls
        if tool_calls:
            message_tool_calls = []
            for tc in tool_calls:
                function = Function(name=tc["name"], arguments=tc["args"])
                tool_call = ChatCompletionMessageToolCall(function=function)
                message_tool_calls.append(tool_call)
        else:
            message_tool_calls = None

        # Create message
        message = Message(
            content="Test response",
            tool_calls=message_tool_calls,
            role="assistant"
        )

        # Create usage
        if usage:
            usage_obj = Usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
            )
        else:
            usage_obj = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

        # Create ModelResponse
        return ModelResponse(
            id="test_id",
            created=1234567890,
            model="test",
            choices=[{
                "message": message,
                "finish_reason": "stop",
                "index": 0
            }],
            usage=usage_obj
        )

    return _create_mock


def test_prepare_schema_single(test_schema):
    """Test preparing single tool schema"""
    model = ToolCallCompletionModel(
        model="test",
        response_format=LLMResponseFormat.TOOLS,
    )
    model.initialize(response_schema=test_schema, system_prompt="Test prompt")
    assert len(model.response_schema) == 1
    assert model.response_schema[0] == test_schema[0]


def test_prepare_schema_multiple(test_schema):
    """Test preparing multiple tool schemas"""
    class AnotherAction(TestActionArgs):
        class Config:
            title = "another_action"
    
    model = ToolCallCompletionModel(
        model="test",
        response_format=LLMResponseFormat.TOOLS,
    )
    model.initialize(response_schema=[test_schema[0], AnotherAction], system_prompt="Test prompt")
    assert len(model.response_schema) == 2
    assert model.response_schema[0] == test_schema[0]
    assert model.response_schema[1] == AnotherAction


def test_prepare_schema_invalid():
    """Test preparing invalid schema"""
    class InvalidSchema:
        pass

    model = ToolCallCompletionModel(
        model="test",
        response_format=LLMResponseFormat.TOOLS,
    )
    with pytest.raises(CompletionRuntimeError):
        model.initialize(response_schema=[InvalidSchema], system_prompt="Test prompt")


def test_get_completion_params(test_schema):
    """Test getting tool completion parameters"""
    model = ToolCallCompletionModel(
        model="test",
        response_format=LLMResponseFormat.TOOLS,
    )
    model.initialize(response_schema=test_schema, system_prompt="Test prompt")
    
    assert "tools" in model._completion_params
    assert len(model._completion_params["tools"]) == 1
    assert model._completion_params["tools"][0]["type"] == "function"
    assert model._completion_params["tool_choice"] == "required"


def test_get_completion_params_with_thoughts(test_schema):
    """Test getting tool parameters with thoughts enabled"""
    model = ToolCallCompletionModel(
        model="test",
        response_format=LLMResponseFormat.TOOLS,
        thoughts_in_action=True,
    )
    model.initialize(response_schema=test_schema, system_prompt="Test prompt")
    
    assert model._completion_params["tool_choice"] == "required"
    assert "thoughts" in model._completion_params["tools"][0]["function"]["parameters"]["properties"]


def test_deepseek_v4_thinking_omits_tool_choice(test_schema):
    """DeepSeek V4 thinking mode rejects explicit tool_choice."""
    model = ToolCallCompletionModel(
        model="deepseek/deepseek-v4-flash",
        response_format=LLMResponseFormat.TOOLS,
        extra_body={"thinking": {"type": "enabled"}},
    )
    model.initialize(response_schema=test_schema, system_prompt="Test prompt")

    assert "tools" in model._completion_params
    assert "tool_choice" not in model._completion_params


def test_provider_can_use_auto_tool_choice(test_schema):
    model = ToolCallCompletionModel(
        model="openai/glm-5.3-flash",
        response_format=LLMResponseFormat.TOOLS,
        tool_choice="auto",
        extra_body={
            "thinking": {"type": "enabled", "clear_thinking": False}
        },
    )
    model.initialize(response_schema=test_schema, system_prompt="Test prompt")

    assert model._completion_params["tool_choice"] == "auto"


def test_compact_structured_output_adds_protocol_instruction():
    model = ToolCallCompletionModel(
        model="openai/glm-5.3-flash",
        response_format=LLMResponseFormat.TOOLS,
        compact_structured_output=True,
    )
    model.initialize(
        response_schema=[SubmitCompactReportArgs],
        system_prompt="Investigate",
    )

    assert "Compact Structured-Output Protocol" in model.system_prompt
    assert "decision, exact evidence/object IDs" in model.system_prompt


def test_glm_compacts_bounded_submit_payload_in_one_isolated_turn(
    mock_litellm_tool_response,
):
    model = ToolCallCompletionModel(
        model="openai/glm-5.3-flash",
        response_format=LLMResponseFormat.TOOLS,
        compact_structured_output=True,
    )
    model.initialize(
        response_schema=[SubmitCompactReportArgs],
        system_prompt="Investigate",
    )
    response = mock_litellm_tool_response(tool_calls=[{
        "name": "SubmitCompactReport",
        "args": json.dumps({
            "summary": "A concise decision that is still much too verbose.",
            "facts": ["SEARCH-1", "SEARCH-2", "SEARCH-3"],
            "unlisted": "remove me",
            "thoughts": "private analysis",
        }),
    }])

    compacted = json.dumps({
        "summary": "Concise decision",
        "facts": ["SEARCH-1 SEARCH-2", "SEARCH-3"],
    })
    with patch.object(
        ToolCallCompletionModel,
        "_execute_structured_compaction",
        return_value=compacted,
    ) as compact:
        outputs, _, flags = model._validate_completion(response)

    assert len(outputs[0].summary) <= 20
    assert outputs[0].facts == ["SEARCH-1 SEARCH-2", "SEARCH-3"]
    assert "llm_compacted_tool_arguments" in flags
    compact.assert_called_once()


@patch("litellm.completion")
def test_glm_compactor_preserves_text_mode_with_low_reasoning(
    mock_completion,
    mock_litellm_tool_response,
):
    model = ToolCallCompletionModel(
        model="openai/glm-5.3-flash",
        response_format=LLMResponseFormat.TOOLS,
        compact_structured_output=True,
    )
    mock_completion.return_value = mock_litellm_tool_response()

    assert model._execute_structured_compaction("Compact this") == "Test response"
    kwargs = mock_completion.call_args.kwargs
    assert "response_format" not in kwargs
    assert kwargs["max_retries"] == 0
    assert "tools" not in kwargs
    assert kwargs["extra_body"] == {
        "thinking": {"type": "enabled", "clear_thinking": False},
        "reasoning_effort": "low",
    }


def test_glm_does_not_repair_missing_submit_fields(
    mock_litellm_tool_response,
):
    model = ToolCallCompletionModel(
        model="openai/glm-5.3-flash",
        response_format=LLMResponseFormat.TOOLS,
        compact_structured_output=True,
    )
    model.initialize(
        response_schema=[SubmitCompactReportArgs],
        system_prompt="Investigate",
    )
    response = mock_litellm_tool_response(tool_calls=[{
        "name": "SubmitCompactReport",
        "args": json.dumps({"facts": [], "thoughts": "analysis"}),
    }])

    with patch.object(
        ToolCallCompletionModel,
        "_execute_structured_compaction",
    ) as compact, pytest.raises(CompletionRetryError):
        model._validate_completion(response)
    compact.assert_not_called()


def test_missing_tool_call_logs_content_and_retries_once(
    caplog,
    mock_litellm_tool_response,
    test_schema,
):
    model = ToolCallCompletionModel(
        model="deepseek/deepseek-v4-flash",
        response_format=LLMResponseFormat.TOOLS,
        extra_body={"thinking": {"type": "enabled"}},
    )
    model.initialize(response_schema=test_schema, system_prompt="Test prompt")
    response = mock_litellm_tool_response()

    with caplog.at_level("WARNING"), pytest.raises(
        CompletionRetryError
    ) as exc_info:
        model._validate_completion(response)

    assert "Test response" in caplog.text
    assert len(exc_info.value.retry_messages) == 1
    assert "call a tool" in exc_info.value.retry_messages[0]["content"]

    outputs, content, flags = model._validate_completion(response)
    assert outputs == []
    assert content == "Test response"
    assert flags == []


@patch("litellm.completion")
def test_validate_completion_valid_tool(mock_completion, mock_litellm_tool_response, test_schema, test_messages):
    """Test validating valid tool call response"""
    model = ToolCallCompletionModel(
        model="test",
        response_format=LLMResponseFormat.TOOLS,
    )
    model.initialize(response_schema=test_schema, system_prompt="Test prompt")
    
    # Mock valid tool call
    valid_args = {
        "command": "test",
        "args": ["--flag"],
        "thoughts": "Testing command execution"
    }
    mock_response = mock_litellm_tool_response(
        tool_calls=[
            {"name": "test_action", "args": json.dumps(valid_args)}
        ]
    )
    
    structured_outputs, text_response, flags = model._validate_completion(
        completion_response=mock_response
    )
    
    assert structured_outputs
    assert len(structured_outputs) == 1
    assert structured_outputs[0].command == "test"
    assert structured_outputs[0].args == ["--flag"]
    assert structured_outputs[0].thoughts == "Testing command execution"


@patch("litellm.completion")
def test_completion_forwards_timeout_and_provider_retries(
    mock_completion, mock_litellm_tool_response, test_schema, test_messages
):
    model = ToolCallCompletionModel(
        model="test",
        response_format=LLMResponseFormat.TOOLS,
        timeout=600.0,
        max_retries=0,
    )
    model.initialize(response_schema=test_schema, system_prompt="Execute the command")
    mock_completion.return_value = mock_litellm_tool_response(
        tool_calls=[
            {
                "name": "test_action",
                "args": json.dumps(
                    {
                        "command": "test",
                        "args": [],
                        "thoughts": "Verify provider request settings",
                    }
                ),
            }
        ]
    )

    model.create_completion(messages=test_messages)

    assert mock_completion.call_args.kwargs["timeout"] == 600.0
    assert mock_completion.call_args.kwargs["max_retries"] == 0


@patch("litellm.completion")
def test_validate_completion_invalid_tool_name(mock_completion, mock_litellm_tool_response, test_schema, test_messages):
    """Test validating invalid tool name"""
    model = ToolCallCompletionModel(
        model="test",
        response_format=LLMResponseFormat.TOOLS,
    )
    model.initialize(response_schema=test_schema, system_prompt="Test prompt")
    
    # Mock invalid tool name
    valid_args = {
        "command": "test",
        "args": ["--flag"],
        "thoughts": "Testing invalid tool name"
    }
    mock_response = mock_litellm_tool_response(
        tool_calls=[
            {"name": "invalid_action", "args": json.dumps(valid_args)}
        ]
    )
    
    with pytest.raises(CompletionRetryError):
        model._validate_completion(
            completion_response=mock_response
        )


@patch("litellm.completion")
def test_validate_completion_invalid_args(mock_completion, mock_litellm_tool_response, test_schema, test_messages):
    """Test validating invalid tool arguments"""
    model = ToolCallCompletionModel(
        model="test",
        response_format=LLMResponseFormat.TOOLS,
    )
    model.initialize(response_schema=test_schema, system_prompt="Test prompt")
    
    # Mock invalid arguments
    invalid_args = {
        "invalid_field": "test",  # Missing required 'command'
        "thoughts": "Testing invalid arguments"
    }
    mock_response = mock_litellm_tool_response(
        tool_calls=[
            {"name": "test_action", "args": json.dumps(invalid_args)}
        ]
    )
    
    with pytest.raises(CompletionRetryError) as exc_info:
        model._validate_completion(
            completion_response=mock_response
        )

    retry_content = exc_info.value.retry_messages[0]["content"]
    assert "test_action" in retry_content
    assert "schema validation" in retry_content
    assert "call it again" in retry_content


@patch("litellm.completion")
def test_validate_completion_duplicate_calls(mock_completion, mock_litellm_tool_response, test_schema, test_messages):
    """Test handling duplicate tool calls"""
    model = ToolCallCompletionModel(
        model="test",
        response_format=LLMResponseFormat.TOOLS,
    )
    model.initialize(response_schema=test_schema, system_prompt="Test prompt")
    
    # Mock duplicate tool calls with same arguments
    valid_args = {
        "command": "test",
        "args": ["--flag"],
        "thoughts": "Testing duplicate calls"
    }
    mock_response = mock_litellm_tool_response(
        tool_calls=[
            {"name": "test_action", "args": json.dumps(valid_args)},
            {"name": "test_action", "args": json.dumps(valid_args)}
        ]
    )
    
    structured_outputs, text_response, flags = model._validate_completion(
        completion_response=mock_response
    )
    
    assert "duplicate_tool_call" in flags
    assert len(structured_outputs) == 1  # Only first call should be included


@patch("litellm.completion")
def test_end_to_end_completion(mock_completion, mock_litellm_tool_response, test_schema, test_messages):
    """Test complete tool call completion flow"""
    model = ToolCallCompletionModel(
        model="test",
        response_format=LLMResponseFormat.TOOLS,
    )
    model.initialize(response_schema=test_schema, system_prompt="Execute the command")
    
    valid_args = {
        "command": "test",
        "args": ["--flag"],
        "thoughts": "Testing end to end completion"
    }
    mock_response = mock_litellm_tool_response(
        tool_calls=[
            {"name": "test_action", "args": json.dumps(valid_args)}
        ]
    )
    mock_completion.return_value = mock_response
    
    result = model.create_completion(messages=test_messages)
    
    assert result.structured_outputs
    assert len(result.structured_outputs) == 1
    assert result.structured_outputs[0].command == "test"
    assert result.structured_outputs[0].args == ["--flag"]
    assert result.structured_outputs[0].thoughts == "Testing end to end completion"
    
    # Verify correct parameters were passed
    mock_completion.assert_called_once()
    call_kwargs = mock_completion.call_args.kwargs
    assert "tools" in call_kwargs
    assert call_kwargs["tool_choice"] == "required"
    assert len(call_kwargs["messages"]) == len(test_messages) + 1  # +1 for system prompt 


@patch("litellm.completion")
def test_retry_on_transient_connection_error(
    mock_completion,
    mock_litellm_tool_response,
    test_schema,
    test_messages,
    monkeypatch,
):
    import litellm

    monkeypatch.setenv("COTEST_PROVIDER_RETRY_ATTEMPTS", "3")
    monkeypatch.setenv("COTEST_PROVIDER_RETRY_MIN_SECONDS", "0")
    monkeypatch.setenv("COTEST_PROVIDER_RETRY_MAX_SECONDS", "0")
    connection_error = litellm.APIError(
        status_code=500,
        message="DeepseekException - Connection error.",
        llm_provider="deepseek",
        model="deepseek-v4-flash",
    )
    valid_response = mock_litellm_tool_response(
        tool_calls=[
            {
                "name": "test_action",
                "args": json.dumps(
                    {"command": "test", "args": [], "thoughts": "retry worked"}
                ),
            }
        ]
    )
    mock_completion.side_effect = [connection_error, valid_response]
    model = ToolCallCompletionModel(
        model="test",
        response_format=LLMResponseFormat.TOOLS,
    )
    model.initialize(response_schema=test_schema, system_prompt="Execute")

    result = model.create_completion(messages=test_messages)

    assert mock_completion.call_count == 2
    assert result.structured_outputs[0].command == "test"


@patch("litellm.completion")
def test_retry_on_validation_error(mock_completion, mock_litellm_tool_response, test_schema, test_messages):
    """Test that validation errors trigger a retry of the entire completion process"""
    model = ToolCallCompletionModel(
        model="test",
        response_format=LLMResponseFormat.TOOLS,
    )
    model.initialize(response_schema=test_schema, system_prompt="Execute the command")
    
    # First response has invalid args to trigger validation error
    invalid_args = {
        "invalid_field": "test",  # Missing required 'command'
        "thoughts": "Testing invalid arguments"
    }
    invalid_response = mock_litellm_tool_response(
        tool_calls=[
            {"name": "test_action", "args": json.dumps(invalid_args)}
        ]
    )
    
    # Second response is valid
    valid_args = {
        "command": "test",
        "args": ["--flag"],
        "thoughts": "Testing retry success"
    }
    valid_response = mock_litellm_tool_response(
        tool_calls=[
            {"name": "test_action", "args": json.dumps(valid_args)}
        ]
    )
    
    # Mock completion to return invalid response first, then valid response
    mock_completion.side_effect = [invalid_response, valid_response]
    
    result = model.create_completion(messages=test_messages)
    
    # Verify completion was called twice
    assert mock_completion.call_count == 2
    
    # Verify both calls used the same messages
    first_call_messages = mock_completion.call_args_list[0].kwargs["messages"]
    second_call_messages = mock_completion.call_args_list[1].kwargs["messages"]
    assert first_call_messages == second_call_messages
    
    # Verify final result is from the valid response
    assert result.structured_outputs
    assert len(result.structured_outputs) == 1
    assert result.structured_outputs[0].command == "test"
    assert result.structured_outputs[0].args == ["--flag"]
    assert result.structured_outputs[0].thoughts == "Testing retry success" 


@patch("litellm.completion")
def test_retry_max_attempts_exceeded(mock_completion, mock_litellm_tool_response, test_schema, test_messages):
    """Test that validation errors stop retrying after max attempts and raise the error"""
    model = ToolCallCompletionModel(
        model="test",
        response_format=LLMResponseFormat.TOOLS,
    )
    model.initialize(response_schema=test_schema, system_prompt="Execute the command")
    
    # Create invalid response that will always fail validation
    invalid_args = {
        "invalid_field": "test",  # Missing required 'command'
        "thoughts": "Testing invalid arguments"
    }
    invalid_response = mock_litellm_tool_response(
        tool_calls=[
            {"name": "test_action", "args": json.dumps(invalid_args)}
        ]
    )
    
    # Mock completion to always return invalid response
    mock_completion.return_value = invalid_response
    
    # Should retry 10 times then fail
    with pytest.raises(CompletionRejectError) as exc_info:
        model.create_completion(messages=test_messages)
    
    # Verify completion was called exactly 10 times
    assert mock_completion.call_count == 10
    
    # Verify all calls used the same messages
    messages_used = [
        call.kwargs["messages"] 
        for call in mock_completion.call_args_list
    ]
    assert all(m == messages_used[0] for m in messages_used)
    
    # Verify the error message contains the validation error
    assert "test_action" in str(exc_info.value)
    assert "schema validation" in str(exc_info.value)
    assert "command" in str(exc_info.value)  # The missing required field 


@patch("litellm.completion")
def test_usage_accumulation_on_retries(mock_completion, mock_litellm_tool_response, test_schema, test_messages):
    """Test that usage is accumulated correctly across retries"""
    model = ToolCallCompletionModel(
        model="test",
        response_format=LLMResponseFormat.TOOLS,
    )
    model.initialize(response_schema=test_schema, system_prompt="Execute the command")
    
    # First response has invalid args to trigger validation error
    invalid_args = {
        "invalid_field": "test",  # Missing required 'command'
        "thoughts": "Testing invalid arguments"
    }
    invalid_response = mock_litellm_tool_response(
        tool_calls=[
            {"name": "test_action", "args": json.dumps(invalid_args)}
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 5}
    )
    
    # Second response is valid
    valid_args = {
        "command": "test",
        "args": ["--flag"],
        "thoughts": "Testing retry success"
    }
    valid_response = mock_litellm_tool_response(
        tool_calls=[
            {"name": "test_action", "args": json.dumps(valid_args)}
        ],
        usage={"prompt_tokens": 15, "completion_tokens": 8}
    )
    
    # Mock completion to return invalid response first, then valid response
    mock_completion.side_effect = [invalid_response, valid_response]
    
    result = model.create_completion(messages=test_messages)
    
    # Verify completion was called twice
    assert mock_completion.call_count == 2
    
    # Verify usage was accumulated
    assert result.completion.usage is not None
    assert result.completion.usage.prompt_tokens == 25  # 10 + 15
    assert result.completion.usage.completion_tokens == 13  # 5 + 8


@patch("litellm.completion")
def test_usage_accumulation_on_max_retries(mock_completion, mock_litellm_tool_response, test_schema, test_messages):
    """Test that usage is accumulated correctly when max retries is exceeded"""
    model = ToolCallCompletionModel(
        model="test",
        response_format=LLMResponseFormat.TOOLS,
    )
    model.initialize(response_schema=test_schema, system_prompt="Execute the command")
    
    # Create invalid response that will always fail validation
    invalid_args = {
        "invalid_field": "test",  # Missing required 'command'
        "thoughts": "Testing invalid arguments"
    }
    
    # Set different usage values for each attempt
    def get_response_with_usage(attempt):
        return mock_litellm_tool_response(
            tool_calls=[
                {"name": "test_action", "args": json.dumps(invalid_args)}
            ],
            usage={
                "prompt_tokens": 10 * (attempt + 1),
                "completion_tokens": 5 * (attempt + 1)
            }
        )
    
    # Mock completion to return responses with different usage values
    mock_completion.side_effect = [
        get_response_with_usage(i) for i in range(10)
    ]
    
    # Should retry 10 times then fail
    with pytest.raises(CompletionRejectError) as exc_info:
        model.create_completion(messages=test_messages)
    
    # Verify completion was called exactly 10 times
    assert mock_completion.call_count == 10
    
    # Verify the accumulated usage in the error
    assert exc_info.value.accumulated_usage is not None
    assert exc_info.value.accumulated_usage.prompt_tokens == 550
    assert exc_info.value.accumulated_usage.completion_tokens == 275
