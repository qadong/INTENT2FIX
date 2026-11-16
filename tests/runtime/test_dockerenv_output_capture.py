import logging
import tracemalloc
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from INTENT2FIX.actions.run_command import RunCommand  # noqa: F401
from INTENT2FIX.actions.run_regression import _compute_passed as regression_passed
from INTENT2FIX.actions.run_single_test import _compute_passed as focused_passed
from INTENT2FIX.actions.test_parser import parse_test_output
from INTENT2FIX.runtime.dockerenv import DockerEnv


MARKER = "[command output capture truncated]"


def _env(chunks, exit_code=0):
    api = Mock()
    api.exec_start.side_effect = lambda _, stream=False: chunks if stream else b"".join(chunks)
    api.exec_inspect.return_value = {"ExitCode": exit_code}
    env = object.__new__(DockerEnv)
    env.container = SimpleNamespace(id="capture-test")
    env.container_name = "capture-test"
    env.work_dir = "/home/repo"
    env.client = SimpleNamespace(api=api)
    return env


@pytest.mark.parametrize("size_mib", [1, 64])
def test_noisy_command_uses_bounded_memory_and_drains_to_real_exit(size_mib):
    seen = []

    def chunks():
        yield b"HEAD\n"
        for _ in range(size_mib * 16):
            yield b"x" * 65536
        yield b"\nTAIL"
        seen.append("drained")

    env = _env(chunks(), exit_code=7)
    tracemalloc.start()
    try:
        output, code = env.run_bash("noisy-command")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 20 * 1024 * 1024
    assert len(output.encode()) <= 4 * 1024 * 1024 + 512
    assert "\nTAIL" in output
    assert ("HEAD\n" in output) == (size_mib <= 4)
    assert (MARKER in output) == (size_mib > 4)
    assert code == 7 and seen == ["drained"]
    assert env.client.api.exec_start.call_args.kwargs == {"stream": True}


@pytest.mark.parametrize("exit_code", [0, 7, 124])
def test_capture_preserves_utf8_split_across_frames_and_exit_codes(exit_code):
    raw = "stdout: 你好\nstderr: é\n".encode()
    env = _env((raw[i:i + 1] for i in range(len(raw))), exit_code)

    output, code = env.run_bash("printf test", timeout=2)

    assert output.startswith(raw.decode())
    assert code == exit_code
    assert ("[Command timed out after 2 seconds]" in output) == (exit_code == 124)


def test_capture_closes_stream_on_read_error():
    closed = []

    def chunks():
        try:
            yield b"partial"
            raise OSError("stream disconnected")
        finally:
            closed.append(True)

    output, code = _env(chunks()).run_bash("true")
    assert code == 1 and "stream disconnected" in output
    assert closed == [True]


def test_empty_command_output_is_preserved():
    output, code = _env((b"" for _ in range(0))).run_bash("true")
    assert output == "" and code == 0


def test_bash_tool_default_has_30minute_timeout_and_forced_kill():
    from INTENT2FIX.actions.unified_repair import UnifiedBash, BashArgs
    from INTENT2FIX.file_context import FileContext
    from INTENT2FIX.repository.repository import InMemRepository

    env = _env((b"ok" for _ in range(1)))
    result = UnifiedBash(runtime=env).execute(BashArgs(command="printf ok"),
        file_context=FileContext(repo=InMemRepository()))
    command = env.client.api.exec_create.call_args.args[1]
    assert command.startswith("timeout -k 5s 1800s sh -c ")
    assert result.properties["exit_code"] == 0


def test_incomplete_output_cannot_be_successful_verification():
    output = "[       OK ] retained_test\n" + MARKER
    summary = parse_test_output(output)
    assert not regression_passed(0, summary)
    assert not focused_passed(0, summary, output=output)
    assert not focused_passed(7, summary, expected_exit_code=7,
                              required_output_regex="retained_test", output=output)
    from scripts.run_no_memory_cotest import _executed_probe_status
    assert _executed_probe_status(output, 0) == "unknown"


def test_capture_warning_remains_visible_after_model_display_limit():
    from INTENT2FIX.actions.output_sanitizer import sanitize_action_output

    output, code = _env((b"x\n" * 32768 for _ in range(100))).run_bash("noisy")
    displayed = sanitize_action_output(output, max_chars=12000, max_lines=200, keep="head_tail")
    assert code == 0 and len(displayed.text) <= 12000
    assert MARKER in displayed.text


def test_command_is_logged_before_action_which_never_returns(caplog):
    from INTENT2FIX.actions.unified_repair import BashArgs
    from INTENT2FIX.agent.code_agent import CodingAgent
    from INTENT2FIX.repository.repository import InMemRepository
    from INTENT2FIX.completion.base import BaseCompletionModel
    from INTENT2FIX.completion import LLMResponseFormat

    completion = BaseCompletionModel.create(model="glm-5.3-flash", model_api_key="dummy",
                                            response_format=LLMResponseFormat.TOOLS)
    agent = CodingAgent(completion=completion, system_prompt="Test", actions=[],
                        repository=InMemRepository())
    args = BashArgs(command="while true; do printf noisy; done")

    def execute(*args, **kwargs):
        assert "Node40" in caplog.text
        assert "while true; do printf noisy; done" in caplog.text
        raise KeyboardInterrupt

    agent._action_map[type(args)] = SimpleNamespace(execute=execute)
    with caplog.at_level(logging.INFO), pytest.raises(KeyboardInterrupt):
        agent._execute(SimpleNamespace(node_id=40, file_context=None, workspace=None),
                       SimpleNamespace(action=args))
