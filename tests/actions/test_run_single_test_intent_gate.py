from unittest.mock import Mock

from INTENT2FIX.actions.run_single_test import (
    RunSingleTest,
    RunSingleTestArgs,
    _compute_passed,
    _compilation_failure_signature,
)
from INTENT2FIX.actions.test_parser import LogType
from INTENT2FIX.actions.verification_sync import sync_verification_files
from INTENT2FIX.file_context import FileContext
from INTENT2FIX.repository.repository import InMemRepository
from INTENT2FIX.runtime.dockerenv import DockerEnv


def test_compilation_failure_signature_ignores_diagnostic_location():
    first = "a.cpp:24:15: error: ambiguous overload for 'operator=='"
    second = "b.cpp:99:3: error: ambiguous overload for 'operator=='"

    assert _compilation_failure_signature(first) == _compilation_failure_signature(second)


def test_compilation_error_never_passes_when_shell_exit_is_zero():
    summary = Mock(log_type=LogType.COMPILATION_ERROR)

    assert _compute_passed(0, summary) is False


def test_memory_leak_never_passes_when_shell_exit_is_zero():
    summary = Mock(log_type=LogType.MEMORY_LEAK)

    assert _compute_passed(0, summary) is False


def test_controlled_nonzero_exit_passes_only_with_matching_output_oracle():
    summary = Mock(log_type=LogType.TEST_RESULTS, results=[])
    output = '<OverallResult success="false" skips="0"/>'

    assert _compute_passed(
        1,
        summary,
        expected_exit_code=1,
        output=output,
        required_output_regex=r'OverallResult success="false"',
    ) is True
    assert _compute_passed(
        1,
        summary,
        expected_exit_code=1,
        output="unrelated failure",
        required_output_regex=r'OverallResult success="false"',
    ) is False


def test_expected_compilation_rejection_satisfies_its_explicit_oracle():
    summary = Mock(log_type=LogType.COMPILATION_ERROR)

    assert _compute_passed(
        1,
        summary,
        expected_exit_code=1,
        output="repro.cpp:1: error: Formatting of non-void pointers is disallowed",
        required_output_regex="Formatting of non-void pointers is disallowed",
    ) is True


def test_single_test_records_controlled_nonzero_oracle_as_satisfied():
    runtime = Mock(spec=DockerEnv)
    output = '<OverallResult success="false" skips="0"/>'
    runtime.run_bash.return_value = (output, 1)
    file_context = FileContext(repo=InMemRepository({}))

    observation = RunSingleTest(runtime=runtime).execute(
        RunSingleTestArgs(
            thoughts="The repaired runner must report this case as failed.",
            command="false",
            expected_exit_code=1,
            required_output_regex=r'OverallResult success="false"',
        ),
        file_context,
    )

    assert observation.properties["oracle_satisfied"] is True
    assert file_context.verification["single_test"]["passed"] is True
    assert file_context.verification["single_test"]["expected_exit_code"] == 1


def test_nonzero_oracle_requires_an_output_signature():
    runtime = Mock(spec=DockerEnv)
    file_context = FileContext(repo=InMemRepository({}))

    observation = RunSingleTest(runtime=runtime).execute(
        RunSingleTestArgs(
            thoughts="Expect a controlled failure.",
            command="false",
            expected_exit_code=1,
        ),
        file_context,
    )

    assert observation.expect_correction is True
    assert observation.properties["fail_reason"] == (
        "missing_expected_output_regex"
    )
    runtime.run_bash.assert_not_called()


def test_focused_verification_rejects_pipeline_that_masks_test_failure():
    runtime = Mock(spec=DockerEnv)
    file_context = FileContext(repo=InMemRepository({}))

    observation = RunSingleTest(runtime=runtime).execute(
        RunSingleTestArgs(
            thoughts="show only the failing assertion",
            command="ctest -R target --output-on-failure 2>&1 | grep FAILED | head",
        ),
        file_context,
    )

    assert observation.expect_correction is True
    assert observation.properties["fail_reason"] == (
        "masked_verification_exit_status"
    )
    assert "real exit status" in observation.message
    runtime.run_bash.assert_not_called()


def test_focused_verification_allows_fail_fast_and_chain():
    runtime = Mock(spec=DockerEnv)
    runtime.run_bash.return_value = ("passed\n", 0)
    file_context = FileContext(repo=InMemRepository({}))

    observation = RunSingleTest(runtime=runtime).execute(
        RunSingleTestArgs(
            thoughts="compile and run the exact witness",
            command="c++ probe.cpp -o /tmp/probe && /tmp/probe",
        ),
        file_context,
    )

    assert observation.expect_correction is False
    assert observation.properties["exit_code"] == 0
    assert any(
        call.args == ("c++ probe.cpp -o /tmp/probe && /tmp/probe",)
        for call in runtime.run_bash.call_args_list
    )


def test_repository_writer_is_rejected_from_single_test_after_observed_write():
    repository = InMemRepository({"dist/public.hpp": "int old_value;\n"})
    file_context = FileContext(repo=repository)
    runtime = Mock(spec=DockerEnv)
    runtime.run_bash.side_effect = [
        ("before  -\n", 0),
        ("updated\n", 0),
        ("after  -\n", 0),
        ("dist/public.hpp\0", 0),
        ("", 0),
    ]
    action = RunSingleTest(runtime=runtime)

    observation = action.execute(
        RunSingleTestArgs(
            thoughts="refresh the generated mirror",
            command="python3 tools/update_release.py",
        ),
        file_context,
    )

    assert observation.expect_correction is True
    assert observation.properties["fail_reason"] == (
        "verification_modified_tracked_files"
    )
    assert "capture_generated_changes=true" in observation.message
    assert runtime.run_bash.call_count == 5


def test_repository_test_command_without_tracked_writes_is_allowed():
    file_context = FileContext(repo=InMemRepository({}))
    runtime = Mock(spec=DockerEnv)
    runtime.run_bash.side_effect = [
        ("same  -\n", 0),
        ("all tests passed\n", 0),
        ("same  -\n", 0),
    ]

    observation = RunSingleTest(runtime=runtime).execute(
        RunSingleTestArgs(thoughts="run focused target", command="make test-widget"),
        file_context,
    )

    assert observation.expect_correction is False
    assert observation.properties["exit_code"] == 0


def test_single_test_suppresses_binary_output_after_parsing_result():
    file_context = FileContext(repo=InMemRepository({}))
    runtime = Mock(spec=DockerEnv)
    runtime.run_bash.return_value = ("\x7fELF\x00\x0ebinary", 1)

    observation = RunSingleTest(runtime=runtime).execute(
        RunSingleTestArgs(
            thoughts="inspect failure",
            command="cat /tmp/compiled-probe",
        ),
        file_context,
    )

    assert "binary output suppressed" in observation.message
    assert "\x00" not in observation.message
    assert "\x0e" not in observation.message
    assert observation.properties["output_metadata"]["binary_suppressed"] is True


def test_repeated_compilation_failure_gives_generic_guidance():
    runtime = Mock(spec=DockerEnv)
    runtime.run_bash.return_value = (
        "repro.cpp:24:15: error: ambiguous overload for 'operator=='\n",
        1,
    )
    file_context = FileContext(repo=InMemRepository())
    file_context.change_seq = 1
    action = RunSingleTest(runtime=runtime)
    first_args = RunSingleTestArgs(thoughts="verify", command="c++ repro.cpp")
    second_args = RunSingleTestArgs(thoughts="verify differently", command="c++ repro2.cpp")

    first = action.execute(first_args, file_context)
    second = action.execute(second_args, file_context)

    assert "Repeated compilation failure" not in first.message
    assert "Repeated compilation failure" in second.message
    assert "hidden friend" not in second.message
    assert runtime.run_bash.call_count == 2


def test_failed_verification_can_rerun_after_code_change():
    runtime = Mock(spec=DockerEnv)
    runtime.run_bash.return_value = ("repro.cpp:1: error: failure\n", 1)
    file_context = FileContext(repo=InMemRepository())
    file_context.change_seq = 1
    action = RunSingleTest(runtime=runtime)
    args = RunSingleTestArgs(thoughts="verify", command="c++ repro.cpp")

    action.execute(args, file_context)
    file_context.change_seq = 2
    second = action.execute(args, file_context)

    assert second.properties["exit_code"] == 1
    assert runtime.run_bash.call_count == 2


def test_failed_verification_can_be_rerun_on_unchanged_code():
    runtime = Mock(spec=DockerEnv)
    runtime.run_bash.return_value = ("failed\n", 1)
    file_context = FileContext(repo=InMemRepository({}))
    action = RunSingleTest(runtime=runtime)
    args = RunSingleTestArgs(thoughts="confirm failure", command="c++ repro.cpp")

    first = action.execute(args, file_context)
    second = action.execute(args, file_context)

    assert first.properties["exit_code"] == 1
    assert second.properties["exit_code"] == 1
    assert runtime.run_bash.call_count == 2


def test_third_identical_verification_is_blocked_on_unchanged_state():
    runtime = Mock(spec=DockerEnv)
    runtime.run_bash.return_value = ("same behavioral failure\n", 1)
    file_context = FileContext(repo=InMemRepository({}))
    action = RunSingleTest(runtime=runtime)
    args = RunSingleTestArgs(thoughts="confirm failure", command="c++ repro.cpp")

    action.execute(args, file_context)
    action.execute(args, file_context)
    third = action.execute(args, file_context)

    assert third.expect_correction is True
    assert third.properties["fail_reason"] == "unchanged_duplicate_verification"
    assert "call Finish" in third.message
    assert runtime.run_bash.call_count == 2


def test_stale_portfolio_diagnostic_metadata_does_not_block_verification():
    runtime = Mock(spec=DockerEnv)
    runtime.run_bash.return_value = ("ok\n", 0)
    file_context = FileContext(repo=InMemRepository({}))
    file_context.change_seq = 3
    file_context.verification["portfolio_diagnostic_policy"] = {
        "anchor_command": "c++ full.cpp && ./full",
        "required_action": "production_revision",
        "production_change_seq": 3,
    }
    action = RunSingleTest(runtime=runtime)

    before_edit = action.execute(
        RunSingleTestArgs(thoughts="branch again", command="c++ toy.cpp"),
        file_context,
    )
    file_context.change_seq = 4
    after_edit = action.execute(
        RunSingleTestArgs(thoughts="branch after edit", command="c++ toy.cpp"),
        file_context,
    )
    assert runtime.run_bash.call_count == 2
    anchor = action.execute(
        RunSingleTestArgs(
            thoughts="rerun retained witness",
            command="c++ full.cpp && ./full",
        ),
        file_context,
    )

    assert before_edit.properties["exit_code"] == 0
    assert after_edit.properties["exit_code"] == 0
    assert anchor.properties["exit_code"] == 0
    assert any(
        call.args == ("c++ full.cpp && ./full",)
        for call in runtime.run_bash.call_args_list
    )


def test_new_focused_test_parent_is_created_before_container_copy():
    runtime = Mock(spec=DockerEnv)
    runtime.work_dir = "/repo"
    runtime.run_bash.side_effect = [("", 0), ("", 0)]
    runtime.copy_anything_to_container.return_value = True
    file_context = FileContext(repo=InMemRepository())
    file_context.add_file("tmp/repro.cpp").apply_changes("int main() {}")
    file_context.change_seq = 1
    action = RunSingleTest(runtime=runtime)

    observation = action.execute(
        RunSingleTestArgs(thoughts="verify", command="c++ tmp/repro.cpp"),
        file_context,
    )

    assert runtime.run_bash.call_args_list[0].args == ("mkdir -p /repo/tmp",)
    assert runtime.run_bash.call_args_list[0].kwargs == {"work_dir": "/"}
    runtime.copy_anything_to_container.assert_called_once()
    assert observation.properties["exit_code"] == 0


def test_verification_sync_restores_file_after_patch_is_reverted():
    runtime = Mock(spec=DockerEnv)
    runtime.work_dir = "/repo"
    runtime.run_bash.return_value = ("", 0)
    runtime.copy_anything_to_container.return_value = True
    context_file = Mock()
    context_file.content = "patched"
    context_file.is_new = False
    file_context = Mock()
    file_context.verification = {}
    file_context.get_patched_files.return_value = ["src/f.cpp"]
    file_context.get_file.return_value = context_file

    assert sync_verification_files(file_context, runtime) == ""
    context_file.content = "baseline"
    file_context.get_patched_files.return_value = []
    assert sync_verification_files(file_context, runtime) == ""

    assert runtime.copy_anything_to_container.call_count == 2
    assert file_context.verification["_container_synced_patched_files"] == []


def test_verification_sync_restores_patch_excluded_from_selected_view():
    runtime = Mock(spec=DockerEnv)
    runtime.work_dir = "/repo"
    runtime.run_bash.return_value = ("", 0)
    copied_contents = []

    def capture_copy(source, _destination):
        with open(source, encoding="utf-8") as copied:
            copied_contents.append(copied.read())
        return True

    runtime.copy_anything_to_container.side_effect = capture_copy
    context_file = Mock()
    context_file.content = "modified test"
    context_file.is_new = False
    context_file.get_base_content.return_value = "original test"
    file_context = Mock()
    file_context.verification = {"_container_synced_patched_files": ["tests/t.cpp"]}
    file_context.get_patched_files.return_value = ["tests/t.cpp"]
    file_context.get_file.return_value = context_file

    assert sync_verification_files(
        file_context,
        runtime,
        included_paths=set(),
    ) == ""

    assert copied_contents == ["original test"]
    assert file_context.verification["_container_synced_patched_files"] == []
