import json

import pytest
from pydantic import ValidationError

from INTENT2FIX.actions.action import Action
from INTENT2FIX.actions.read_source import ReadSource, ReadSourceArgs
from INTENT2FIX.diagnostic_evidence import record_digest


@pytest.fixture
def reader(tmp_path):
    archive = {"base_commit": "base", "records": {
        "RUN-14": {"id": "RUN-14", "locator": "c++ -std=c++20 probe.cpp", "exit_code": 1,
                   "observation": "中文错误\n" * 2500},
        "SOURCE-1": {"id": "SOURCE-1", "observation": "original legacy record"},
        "CODE-1": {"id": "CODE-1", "observation": "verified body"}},
        "aliases": {"SOURCE-1": "CODE-1", "SOURCE-2": "CODE-1"}}
    path = tmp_path / "archive.json"
    path.write_text(json.dumps(archive, ensure_ascii=False))
    return ReadSource(archive_path=str(path), archive_sha256=record_digest(archive)), archive


def test_reader_pagination_roundtrip_and_persisted_configuration(reader):
    tool, archive = reader
    restored = Action.model_validate(tool.model_dump())
    offset, pages = 0, []
    while offset is not None:
        obs = restored.execute(ReadSourceArgs(source_id="RUN-14", offset=offset, limit=1000))
        data = json.loads(obs.message)
        pages.append(data["content"])
        offset = data["next_offset"]
        assert data["base_commit"] == "base" and not data["original_truncated"]
    assert len(pages) > 10
    assert json.loads("".join(pages)) == archive["records"]["RUN-14"]


def test_original_id_takes_precedence_over_alias(reader):
    tool, _ = reader
    original = json.loads(tool.execute(ReadSourceArgs(source_id="SOURCE-1")).message)
    aliased = json.loads(tool.execute(ReadSourceArgs(source_id="SOURCE-2")).message)
    assert original["source_id"] == "SOURCE-1" and "original legacy record" in original["content"]
    assert aliased["source_id"] == "CODE-1"


def test_reader_rejects_arbitrary_paths_offsets_and_archive_tampering(reader):
    tool, _ = reader
    assert tool.execute(ReadSourceArgs(source_id="../../etc/passwd")).properties["fail_reason"] == "unknown_source_id"
    assert tool.execute(ReadSourceArgs(source_id="RUN-14", offset=1000000)).expect_correction
    for args in ({"offset": -1}, {"limit": 12001}, {"archive_path": "/etc/passwd"}):
        with pytest.raises(ValidationError):
            ReadSourceArgs(source_id="RUN-14", **args)
    from pathlib import Path
    Path(tool.archive_path).write_text('{"records": {}}')
    assert tool.execute(ReadSourceArgs(source_id="RUN-14")).properties["fail_reason"] == "archive_changed"


def test_native_tool_dispatch_survives_agent_reload_and_enters_history(reader):
    from litellm.types.utils import ModelResponse
    from INTENT2FIX.agent.code_agent import CodingAgent
    from INTENT2FIX.completion import LLMResponseFormat
    from INTENT2FIX.completion.base import BaseCompletionModel
    from INTENT2FIX.node import ActionStep, Node
    from INTENT2FIX.repository.repository import InMemRepository

    tool, _ = reader
    model = BaseCompletionModel.create(model="gpt-3.5-turbo", response_format=LLMResponseFormat.TOOLS,
                                       model_api_key="dummy")
    repo = InMemRepository()
    regular = CodingAgent.create(repository=repo, completion_model=model, compact_native_prompt=True)
    agent = CodingAgent.create(repository=repo, completion_model=model, compact_native_prompt=True,
                               evidence_archive_path=tool.archive_path, evidence_archive_sha256=tool.archive_sha256)
    assert [type(a) for a in agent.actions] == [type(a) for a in regular.actions] + [ReadSource]
    assert "read_source" not in regular.system_prompt
    agent = CodingAgent.model_validate(agent.model_dump(), repository=repo)
    native = ReadSourceArgs.openai_schema()["function"]
    assert native["name"] == "read_source"
    assert set(native["parameters"]["properties"]) == {"source_id", "offset", "limit"}
    response = ModelResponse(choices=[{"index": 0, "finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None, "tool_calls": [{"id": "read-1", "type": "function", "function": {
            "name": "read_source", "arguments": json.dumps({"source_id": "RUN-14", "limit": 500})}}]}}])
    args = agent.completion._validate_completion(response)[0][0]
    assert isinstance(args, ReadSourceArgs)
    node = Node(node_id=1, user_message="Inspect RUN-14")
    step = ActionStep(action=args)
    node.action_steps = [step]
    agent._execute(node, step)
    history = agent.message_generator.generate_messages(Node(node_id=2, parent=node))
    messages = [m for m in history if m["role"] == "tool"]
    assert len(messages) == 1 and messages[0]["content"] == step.observation.message
    assert json.loads(messages[0]["content"])["next_offset"] == 500
