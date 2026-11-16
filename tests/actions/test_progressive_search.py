from INTENT2FIX.actions.search_base import SearchBaseAction
from INTENT2FIX.actions.entity_intent_search import EntityIntentSearch, EntityIntentSearchArgs
from INTENT2FIX.actions.run_command import RunCommandArgs
from INTENT2FIX.actions.schema import Observation
from INTENT2FIX.completion import LLMResponseFormat
from INTENT2FIX.completion.base import BaseCompletionModel
from INTENT2FIX.index.types import SearchCodeHit, SearchCodeResponse, SpanHit
from INTENT2FIX.index.entity_code_index import EntityCodeIndex
from INTENT2FIX.message_history.message_history import MessageHistoryGenerator
from INTENT2FIX.node import Node


def _completion_model():
    return BaseCompletionModel.create(
        response_format=LLMResponseFormat.TOOLS,
        model="gpt-3.5-turbo",
        max_tokens=1000,
        temperature=0,
        model_api_key="dummy",
    )


def test_progressive_search_returns_ten_declaration_skeletons_without_bodies():
    action = SearchBaseAction(
        completion_model=_completion_model(),
        progressive_disclosure=True,
    )
    response = SearchCodeResponse(
        message="Found matching symbols.",
        hits=[
            SearchCodeHit(
                file_path=f"src/file_{index}.cpp",
                spans=[
                    SpanHit(
                        span_id=f"symbol_{index}@{index + 1}-{index + 2}",
                        label=f"ns::symbol_{index}",
                        start_line=index + 1,
                        end_line=index + 2,
                        tokens=20,
                        skeleton=f"void ns::symbol_{index}();",
                    )
                ],
            )
            for index in range(12)
        ],
    )

    candidates = action._progressive_candidates(response)
    message = action._format_progressive_candidates(response, candidates, False)

    assert len(candidates) == 10
    assert candidates[0]["label"] == "ns::symbol_0"
    assert "Showing declaration skeletons for 10 of 12 candidate locations" in message
    assert "src/file_0.cpp:1-2" in message
    assert "void ns::symbol_0();" in message
    assert "function body" not in message
    assert "Refine the symbol" in message
    assert "```" not in message


def test_progressive_intent_search_keeps_ten_features_and_three_entry_points(monkeypatch):
    requested_limits = []

    def feature_search(self, query, max_results):
        requested_limits.append(max_results)
        return {
            "features": [
                {
                    "feature_name": f"feature-{index}",
                    "feature_description": "feature outline",
                    "members": [{"func_id": f"entry-{member}"} for member in range(5)],
                }
                for index in range(max_results)
            ]
        }

    monkeypatch.setattr(EntityCodeIndex, "feature_search", feature_search)
    action = EntityIntentSearch(
        code_index=EntityCodeIndex(),
        progressive_disclosure=True,
    )

    result = action._execute(EntityIntentSearchArgs(query="formatting behavior"))

    assert requested_limits == [10]
    assert result.count("FEATURE:") == 10
    assert "entry-2" in result
    assert "entry-3" not in result


def test_native_history_truncates_large_observations():
    generator = MessageHistoryGenerator(max_observation_tokens=100)
    content = "compiler output line\n" * 1000

    truncated = generator._truncate_observation(content)

    assert "observation truncated" in truncated
    assert len(truncated) < len(content)


def test_native_history_keeps_tool_call_and_result_atomic():
    root = Node(node_id=0, user_message="Fix the bug")
    action_node = Node(node_id=1)
    action_node.action = RunCommandArgs(thoughts="", command="rg symbol src")
    action_node.observation = Observation(message="match\n" * 1000)
    action_node.assistant_tool_calls = [
        {
            "id": "call_search",
            "type": "function",
            "function": {
                "name": "RunCommand",
                "arguments": '{"command":"rg symbol src"}',
            },
        }
    ]
    root.add_child(action_node)
    current = Node(node_id=2)
    action_node.add_child(current)

    generator = MessageHistoryGenerator(
        max_tokens=1000,
        max_observation_tokens=100,
    )
    messages = generator.generate_messages(current)

    assert [message["role"] for message in messages] == ["user", "assistant", "tool"]
    assert messages[1]["tool_calls"][0]["id"] == "call_search"
    assert messages[2]["tool_call_id"] == "call_search"
    assert "observation truncated" in messages[2]["content"]
