from unittest.mock import Mock, call

from INTENT2FIX.actions.run_intent_check import (
    RunIntentCheck,
    RunIntentCheckArgs,
)
from INTENT2FIX.runtime.dockerenv import DockerEnv


def _file_context():
    context = Mock()
    context.get_patched_files.return_value = []
    context.verification = {
        "intent_contract": {
            "checks": {"ACCEPT-1": "Exercise target and contrast."},
            "acceptance_ids": ["ACCEPT-1"],
            "probes": {
                "ACCEPT-1": {
                    "description": "Exercise target and contrast.",
                    "command": (
                        "c++ intent_probe.cpp -o /tmp/intent_probe && "
                        "/tmp/intent_probe"
                    ),
                    "files": {
                        "intent_probe.cpp": "int main() { return 0; }"
                    },
                    "source_evidence_id": "RUN-2",
                    "baseline_exit_code": 1,
                    "authority": "normative",
                }
            },
        }
    }
    context.change_seq = 7
    return context


def test_run_intent_check_rejects_unknown_id_without_running_command():
    runtime = Mock(spec=DockerEnv)
    context = _file_context()

    observation = RunIntentCheck(runtime=runtime).execute(
        RunIntentCheckArgs(thoughts="verify", check_id="ACCEPT-9"),
        context,
    )

    assert observation.expect_correction
    assert observation.properties["fail_reason"] == "unknown_intent_check_id"
    runtime.run_bash.assert_not_called()


def test_run_intent_check_redeploys_and_executes_frozen_probe():
    runtime = Mock(spec=DockerEnv)
    runtime.work_dir = "/repo"
    runtime.run_bash.return_value = ("probe passed", 0)
    context = _file_context()

    action = RunIntentCheck(runtime=runtime)
    observation = action.execute(
        RunIntentCheckArgs(thoughts="verify", check_id="ACCEPT-1"),
        context,
    )
    action.execute(
        RunIntentCheckArgs(thoughts="verify again", check_id="ACCEPT-1"),
        context,
    )

    assert runtime.write_file.call_args_list == [
        call("/repo/intent_probe.cpp", "int main() { return 0; }"),
        call("/repo/intent_probe.cpp", "int main() { return 0; }"),
    ]
    assert runtime.run_bash.call_args_list == [
        call(
            "c++ intent_probe.cpp -o /tmp/intent_probe && /tmp/intent_probe",
            timeout=600,
        ),
        call(
            "c++ intent_probe.cpp -o /tmp/intent_probe && /tmp/intent_probe",
            timeout=600,
        ),
    ]
    result = context.verification["intent_check_results"]["ACCEPT-1"]
    assert result["passed"] is True
    assert result["change_seq"] == 7
    assert result["probe_sha256"] == observation.properties["probe_sha256"]


def test_run_intent_check_rebinds_tmp_source_to_materialized_file():
    runtime = Mock(spec=DockerEnv)
    runtime.work_dir = "/repo"
    runtime.run_bash.return_value = ("probe passed", 0)
    context = _file_context()
    probe = context.verification["intent_contract"]["probes"]["ACCEPT-1"]
    probe["command"] = (
        "c++ /tmp/intent_probe.cpp -o /tmp/intent_probe && "
        "/tmp/intent_probe"
    )

    observation = RunIntentCheck(runtime=runtime).execute(
        RunIntentCheckArgs(thoughts="verify", check_id="ACCEPT-1"),
        context,
    )

    runtime.write_file.assert_called_once_with(
        "/repo/intent_probe.cpp",
        "int main() { return 0; }",
    )
    runtime.run_bash.assert_called_once_with(
        "c++ intent_probe.cpp -o /tmp/intent_probe && /tmp/intent_probe",
        timeout=600,
    )
    assert observation.properties["executed_command"] == (
        "c++ intent_probe.cpp -o /tmp/intent_probe && /tmp/intent_probe"
    )


def test_run_intent_check_prepares_nested_tmp_output_directory():
    runtime = Mock(spec=DockerEnv)
    runtime.work_dir = "/repo"
    runtime.run_bash.return_value = ("probe passed", 0)
    context = _file_context()
    probe = context.verification["intent_contract"]["probes"]["ACCEPT-1"]
    probe["command"] = (
        "c++ intent_probe.cpp -o /tmp/probe/intent_probe && "
        "/tmp/probe/intent_probe"
    )

    observation = RunIntentCheck(runtime=runtime).execute(
        RunIntentCheckArgs(thoughts="verify", check_id="ACCEPT-1"),
        context,
    )

    expected = (
        "mkdir -p /tmp/probe && c++ intent_probe.cpp "
        "-o /tmp/probe/intent_probe && /tmp/probe/intent_probe"
    )
    runtime.run_bash.assert_called_once_with(expected, timeout=600)
    assert observation.properties["executed_command"] == expected


def test_failed_frozen_probe_does_not_satisfy_gate():
    runtime = Mock(spec=DockerEnv)
    runtime.work_dir = "/repo"
    runtime.run_bash.return_value = ("compile error", 1)
    context = _file_context()

    observation = RunIntentCheck(runtime=runtime).execute(
        RunIntentCheckArgs(thoughts="verify", check_id="ACCEPT-1"),
        context,
    )

    assert observation.expect_correction
    assert context.verification["intent_check_results"]["ACCEPT-1"][
        "passed"
    ] is False


def test_failed_illustrative_probe_is_diagnostic_only():
    runtime = Mock(spec=DockerEnv)
    runtime.work_dir = "/repo"
    runtime.run_bash.return_value = ("compile error", 1)
    context = _file_context()
    context.verification["intent_contract"]["acceptance_ids"] = []
    context.verification["intent_contract"]["diagnostic_ids"] = ["ACCEPT-1"]
    context.verification["intent_contract"]["probes"]["ACCEPT-1"][
        "authority"
    ] = "illustrative"

    observation = RunIntentCheck(runtime=runtime).execute(
        RunIntentCheckArgs(thoughts="inspect local witness", check_id="ACCEPT-1"),
        context,
    )

    assert not observation.expect_correction
    assert observation.properties["required"] is False
    assert observation.properties["authority"] == "illustrative"
    assert "still diverges" in observation.summary
