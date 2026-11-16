from unittest.mock import Mock
import subprocess
import pytest

from INTENT2FIX.actions.run_regression import (
    RunRegression,
    RunRegressionArgs,
    _compute_passed,
)
from INTENT2FIX.actions.test_parser import LogType
from INTENT2FIX.file_context import FileContext
from INTENT2FIX.repository.repository import InMemRepository
from INTENT2FIX.runtime.dockerenv import DockerEnv


@pytest.fixture(autouse=True)
def mock_isolated_suite(monkeypatch):
    # These action/cache tests mock the executor. Fresh-image isolation is tested
    # separately with independent filesystem fixtures and actual Docker images.
    monkeypatch.setattr(
        "INTENT2FIX.actions.run_regression.run_isolated_production_command",
        lambda runtime, command, paths, context: runtime.run_bash(command),
    )


def test_memory_leak_never_passes_regression_when_shell_exit_is_zero():
    summary = Mock(log_type=LogType.MEMORY_LEAK)

    assert _compute_passed(0, summary) is False


def test_regression_records_failed_test_names():
    context = FileContext(repo=InMemRepository({}))
    runtime = Mock(spec=DockerEnv)
    runtime.work_dir = "/repo"
    runtime.run_bash.return_value = (
        "1/1 Test #33: legacy_exact_message .....***Failed\n",
        8,
    )
    runtime.copy_anything_to_container.return_value = True

    RunRegression(runtime=runtime).execute(
        RunRegressionArgs(thoughts="record soft regression failures"),
        context,
    )

    assert context.verification["regression"]["failed_test_names"] == [
        "legacy_exact_message"
    ]


@pytest.mark.parametrize("script_name", ["build_run.sh", "run.sh"])
@pytest.mark.parametrize("pipeline_exit", [0, 7])
def test_regression_scripts_without_shebang_use_bash_and_keep_pipeline_failures(
    tmp_path, script_name, pipeline_exit
):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "CMakeLists.txt").write_text("project(smoke)\n")
    (tmp_path / script_name).write_text(
        "set -euo pipefail\n"
        "test -f CMakeLists.txt\n"
        "printf 'bash-script-started\\n'\n"
        f"sh -c 'exit {pipeline_exit}' | cat\n"
        "printf '1/1 Test #1: smoke ...... Passed\\n'\n"
    )
    context = FileContext(repo=InMemRepository({}))
    runtime = Mock(spec=DockerEnv)
    runtime.work_dir = str(repo_dir)

    def run(command, **kwargs):
        # Match DockerEnv's sh wrapper, with the container's /home mapped locally.
        process = subprocess.run(
            ["sh", "-c", command.replace("/home", str(tmp_path))],
            cwd=repo_dir, capture_output=True, text=True, timeout=10,
        )
        return process.stdout + process.stderr, process.returncode

    runtime.run_bash.side_effect = run
    observation = RunRegression(runtime=runtime).execute(
        RunRegressionArgs(thoughts="verify script interpreter and pipeline status"),
        context,
    )

    assert "bash-script-started" in observation.message
    assert "Illegal option -o pipefail" not in observation.message
    assert observation.properties["exit_code"] == pipeline_exit
    assert context.verification["production_regression"]["passed"] is (
        pipeline_exit == 0
    )


@pytest.mark.parametrize("cached_change_seq", [0, -1])
def test_regression_retries_cached_pipefail_startup_failure(cached_change_seq):
    from INTENT2FIX.actions.verification_sync import PRODUCTION_ISOLATION
    from INTENT2FIX.utils.patch import production_patch_sha256

    context = FileContext(repo=InMemRepository({}))
    cached = {
        "passed": False,
        "exit_code": 2,
        "log_type": "unknown",
        "output": "./build_run.sh: 1: set: Illegal option -o pipefail\n",
        "production_patch_sha256": production_patch_sha256(""),
        "change_seq": cached_change_seq,
        "verification_view": "production",
        "verification_isolation": PRODUCTION_ISOLATION,
    }
    context.verification["regression"] = dict(cached)
    context.verification["production_regression"] = dict(cached)
    runtime = Mock(spec=DockerEnv)
    runtime.work_dir = "/repo"
    runtime.run_bash.return_value = ("1/1 Test #1: smoke ...... Passed\n", 0)

    observation = RunRegression(runtime=runtime).execute(
        RunRegressionArgs(thoughts="retry after the public script launch fix"),
        context,
    )

    assert not observation.properties.get("reused", False)
    assert runtime.run_bash.called
    assert context.verification["production_regression"]["passed"] is True


def test_regression_runs_production_only_view_after_test_edit():
    repository = InMemRepository(
        {
            "src/fix.cpp": "int value = 1;\n",
            "tests/fix.cpp": "int expected = 1;\n",
        }
    )
    context = FileContext(repo=repository)
    context.add_file("src/fix.cpp").apply_changes("int value = 2;\n")
    context.add_file("tests/fix.cpp").apply_changes("int expected = 2;\n")
    context.change_seq = 4

    runtime = Mock(spec=DockerEnv)
    runtime.work_dir = "/repo"
    runtime.run_bash.return_value = ("", 0)
    runtime.copy_anything_to_container.return_value = True

    observation = RunRegression(runtime=runtime).execute(
        RunRegressionArgs(thoughts="verify current and submission views"),
        context,
    )

    suite_calls = [
        call
        for call in runtime.run_bash.call_args_list
        if "/home/build_run.sh" in call.args[0]
    ]
    assert len(suite_calls) == 2
    assert context.verification["regression"]["passed"] is True
    production_result = context.verification["production_regression"]
    assert production_result["passed"] is True
    assert production_result["exit_code"] == 0
    assert production_result["log_type"] == "unknown"
    assert production_result["output"] == ""
    assert production_result["change_seq"] == 4
    assert production_result["summary"]
    assert observation.properties["production_regression"]["passed"] is True
    assert "Production-only regression" in observation.message
    assert context.verification["_container_synced_patched_files"] == [
        "src/fix.cpp"
    ]


def test_regression_reuses_result_after_only_temporary_probe_changes():
    repository = InMemRepository(
        {
            "src/fix.cpp": "int value = 1;\n",
            "focused_probe.cpp": "int main() { return 0; }\n",
        }
    )
    context = FileContext(repo=repository)
    context.add_file("src/fix.cpp").apply_changes("int value = 2;\n")
    probe = context.add_file("focused_probe.cpp")
    context.verification["ephemeral_paths"] = ["focused_probe.cpp"]
    context.change_seq = 1

    runtime = Mock(spec=DockerEnv)
    runtime.work_dir = "/repo"
    runtime.run_bash.return_value = ("", 0)
    runtime.copy_anything_to_container.return_value = True
    action = RunRegression(runtime=runtime)

    action.execute(RunRegressionArgs(thoughts="verify production"), context)
    probe.apply_changes("int main() { return 1; }\n")
    context.verification["probe_change_seq"] = 1
    observation = action.execute(
        RunRegressionArgs(thoughts="verify after probe correction"), context
    )

    suite_calls = [
        call
        for call in runtime.run_bash.call_args_list
        if "/home/build_run.sh" in call.args[0]
    ]
    assert len(suite_calls) == 1
    assert observation.properties["reused"] is True
    assert "production/test revision is unchanged" in observation.message


def test_regression_reuses_production_view_after_only_test_changes():
    repository = InMemRepository(
        {
            "src/fix.cpp": "int value = 1;\n",
            "tests/fix.cpp": "int expected = 1;\n",
        }
    )
    context = FileContext(repo=repository)
    context.add_file("src/fix.cpp").apply_changes("int value = 2;\n")
    context.change_seq = 1

    runtime = Mock(spec=DockerEnv)
    runtime.work_dir = "/repo"
    runtime.run_bash.return_value = ("", 0)
    runtime.copy_anything_to_container.return_value = True
    action = RunRegression(runtime=runtime)

    action.execute(RunRegressionArgs(thoughts="verify production"), context)
    context.add_file("tests/fix.cpp").apply_changes("int expected = 2;\n")
    context.change_seq = 2
    observation = action.execute(
        RunRegressionArgs(thoughts="verify after changing a soft test pin"),
        context,
    )

    suite_calls = [
        call
        for call in runtime.run_bash.call_args_list
        if "/home/build_run.sh" in call.args[0]
    ]
    assert len(suite_calls) == 1
    assert observation.properties["reused"] is True
    assert observation.properties["test_view_stale"] is True
    assert "Use RunSingleTest" in observation.message
    assert context.verification["regression"]["change_seq"] == 1
    production_result = context.verification["production_regression"]
    assert production_result["change_seq"] == 2
    assert production_result["test_view"] == "original_tests_before_test_edits"


def test_regression_reruns_after_production_revision_changes():
    repository = InMemRepository({"src/fix.cpp": "int value = 1;\n"})
    context = FileContext(repo=repository)
    source = context.add_file("src/fix.cpp")
    source.apply_changes("int value = 2;\n")
    context.change_seq = 1

    runtime = Mock(spec=DockerEnv)
    runtime.work_dir = "/repo"
    runtime.run_bash.return_value = ("", 0)
    runtime.copy_anything_to_container.return_value = True
    action = RunRegression(runtime=runtime)

    action.execute(RunRegressionArgs(thoughts="verify first revision"), context)
    source.apply_changes("int value = 3;\n")
    context.change_seq = 2
    observation = action.execute(
        RunRegressionArgs(thoughts="verify second revision"), context
    )

    suite_calls = [
        call
        for call in runtime.run_bash.call_args_list
        if "/home/build_run.sh" in call.args[0]
    ]
    assert len(suite_calls) == 2
    assert not observation.properties.get("reused", False)


def test_submitted_view_removes_excluded_header_and_promotion_requires_rebuild(tmp_path):
    """Actually compile the delivered files: a leftover empty header must not mask omission."""
    import hashlib
    import shutil
    import subprocess
    from pathlib import Path
    import pytest
    from INTENT2FIX.actions.create_file import CreateFile, CreateFileArgs
    from INTENT2FIX.patch_review.sanitizer import sanitize_candidate_patch

    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    context = FileContext(repo=InMemRepository({"src/fix.cpp": "int value;\n"}))
    context.add_file("src/fix.cpp").apply_changes('#include "helper.hpp"\nint value;\n')
    context.add_file("include/helper.hpp").apply_changes("// required header\n")
    context.verification["ephemeral_paths"] = ["include/helper.hpp"]
    context.change_seq = 1
    (tmp_path / "include").mkdir()
    (tmp_path / "include/helper.hpp").write_text("// left by a prior probe\n")
    context.verification["regression"] = {
        "passed": True, "change_seq": 1,
        "production_patch_sha256": hashlib.sha256((sanitize_candidate_patch(context).patch.strip() + "\n").encode()).hexdigest(),
        # Legacy records mislabelled a working-tree result with the production hash.
    }
    runtime = Mock(spec=DockerEnv)
    runtime.work_dir = str(tmp_path)
    builds = []

    def run(command, **kwargs):
        if "/home/build_run.sh" in command:
            builds.append(command)
            proc = subprocess.run([compiler, "-fsyntax-only", "-Iinclude", "src/fix.cpp"],
                                  cwd=tmp_path, capture_output=True, text=True)
        else:
            proc = subprocess.run(["bash", "-c", command], cwd=tmp_path, capture_output=True, text=True)
        return proc.stdout + proc.stderr, proc.returncode

    def copy(source, target):
        Path(target).write_bytes(Path(source).read_bytes())
        return True

    runtime.run_bash.side_effect = run
    runtime.copy_anything_to_container.side_effect = copy
    action = RunRegression(runtime=runtime)
    action.execute(RunRegressionArgs(thoughts="check actual submission"), context)
    first = context.verification["production_regression"]
    assert not first["passed"]
    assert first["log_type"] == "compilation_error"
    assert not (tmp_path / "include/helper.hpp").exists()
    assert first["verified_paths"] == ["src/fix.cpp"]
    CreateFile().execute(CreateFileArgs(path="include/helper.hpp", file_text="// required header\n",
                                        artifact_kind="production"), context)
    action.execute(RunRegressionArgs(thoughts="verify corrected ownership"), context)
    assert len(builds) == 2
    final = context.verification["production_regression"]
    assert final["passed"]
    assert final["verification_view"] == "production"
    assert final["production_patch_sha256"] != first["production_patch_sha256"]
    assert final["verified_paths"] == ["include/helper.hpp", "src/fix.cpp"]


def test_sync_removes_previously_synced_new_probe_after_context_removal():
    from INTENT2FIX.actions.verification_sync import sync_verification_files
    context = FileContext(repo=InMemRepository())
    context.verification["_container_synced_patched_files"] = ["removed_probe.cpp"]
    runtime = Mock(spec=DockerEnv)
    runtime.work_dir = "/repo"
    runtime.run_bash.return_value = ("", 0)
    assert not sync_verification_files(context, runtime, included_paths=set())
    runtime.run_bash.assert_called_once_with("rm -f -- /repo/removed_probe.cpp", work_dir="/")
    runtime.copy_anything_to_container.assert_not_called()
