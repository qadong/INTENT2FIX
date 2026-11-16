import base64
from unittest.mock import Mock

import pytest

from INTENT2FIX.actions.run_command import RunCommand, RunCommandArgs
from INTENT2FIX.actions.command_policy import may_execute_repository_code
from INTENT2FIX.file_context import FileContext
from INTENT2FIX.repository.repository import InMemRepository
from INTENT2FIX.runtime.dockerenv import DockerEnv


@pytest.mark.parametrize(
    "command",
    [
        "cat >> src/fix.cpp <<'EOF'\nnew code\nEOF",
        "printf 'new code' > src/fix.cpp",
        "generate | tee src/fix.cpp",
        "sed -i 's/old/new/' src/fix.cpp",
        "sed --in-place=.bak 's/old/new/' src/fix.cpp",
        "perl -pi -e 's/old/new/' src/fix.cpp",
    ],
)
def test_repair_run_command_rejects_file_writes(command):
    docker_env = Mock(spec=DockerEnv)

    observation = RunCommand(docker_env=docker_env).execute(
        RunCommandArgs(command=command)
    )

    assert observation.expect_correction is True
    docker_env.run_bash.assert_not_called()


@pytest.mark.parametrize(
    "command",
    [
        "g++ -std=c++17 repro.cpp 2>&1 | head -30",
        "cmake --build build 2>&1 | grep error",
        "ctest --test-dir build | tail -20",
    ],
)
def test_repair_run_command_rejects_build_or_test_commands(command):
    docker_env = Mock(spec=DockerEnv)

    observation = RunCommand(docker_env=docker_env).execute(
        RunCommandArgs(command=command)
    )

    assert observation.expect_correction is True
    assert "must use RunSingleTest" in observation.message
    docker_env.run_bash.assert_not_called()


@pytest.mark.parametrize(
    "command",
    [
        "g++ -std=c++17 repro.cpp -o repro",
        "cmake --build build --parallel 4",
        "ninja -C build target",
    ],
)
def test_repair_run_command_rejects_unpiped_verification(command):
    docker_env = Mock(spec=DockerEnv)

    observation = RunCommand(docker_env=docker_env).execute(
        RunCommandArgs(command=command)
    )

    assert observation.expect_correction is True
    assert "current patch sequence" in observation.message
    docker_env.run_bash.assert_not_called()


def test_repair_run_command_allows_piped_source_search():
    docker_env = Mock(spec=DockerEnv)
    docker_env.run_bash.return_value = ("match\n", 0)

    observation = RunCommand(docker_env=docker_env).execute(
        RunCommandArgs(command="rg operator src | head -20")
    )

    assert observation.properties["exit_code"] == 0
    docker_env.run_bash.assert_called_once()


def test_repair_run_command_syncs_current_patch_before_source_search():
    repository = InMemRepository({"include/widget.hpp": "int old_value;\n"})
    file_context = FileContext(repo=repository)
    context_file = file_context.add_file("include/widget.hpp", add_extra=False)
    context_file.apply_changes("int new_value;\n")
    docker_env = Mock(spec=DockerEnv)
    docker_env.work_dir = "/repo"
    docker_env.copy_anything_to_container.return_value = True
    docker_env.run_bash.side_effect = [
        ("", 0),
        ("before  -\n", 0),
        ("1:int new_value;\n", 0),
        ("before  -\n", 0),
    ]

    observation = RunCommand(docker_env=docker_env).execute(
        RunCommandArgs(command="grep -n new_value include/widget.hpp"),
        file_context=file_context,
    )

    assert observation.properties["exit_code"] == 0
    assert "new_value" in observation.message
    assert docker_env.copy_anything_to_container.call_count == 1
    assert docker_env.copy_anything_to_container.call_args.args[1] == (
        "/repo/include/widget.hpp"
    )
    assert docker_env.run_bash.call_args_list[-2].args[0] == (
        "grep -n new_value include/widget.hpp"
    )


@pytest.mark.parametrize(
    "command",
    [
        "rm -f probe.cpp",
        "find . -maxdepth 1 -name probe.cpp -delete",
        r"find . -maxdepth 1 -name probe.cpp -exec rm {} \;",
    ],
)
def test_repair_run_command_rejects_removing_registered_temporary_probe(command):
    file_context = FileContext(repo=InMemRepository())
    file_context.verification["ephemeral_paths"] = ["probe.cpp"]
    docker_env = Mock(spec=DockerEnv)

    observation = RunCommand(docker_env=docker_env).execute(
        RunCommandArgs(command=command),
        file_context=file_context,
    )

    assert observation.expect_correction is True
    assert "cannot remove or move" in observation.message.lower()
    docker_env.run_bash.assert_not_called()


def test_repair_run_command_allows_compiler_version_inspection():
    docker_env = Mock(spec=DockerEnv)
    docker_env.run_bash.return_value = ("g++ 12", 0)

    observation = RunCommand(docker_env=docker_env).execute(
        RunCommandArgs(command="g++ --version")
    )

    assert observation.properties["exit_code"] == 0


def test_repair_run_command_suppresses_binary_output():
    docker_env = Mock(spec=DockerEnv)
    docker_env.run_bash.return_value = (
        "\x7fELF\x00\x0e" + "binary payload" * 100,
        0,
    )

    observation = RunCommand(docker_env=docker_env).execute(
        RunCommandArgs(command="cat /tmp/compiled-probe")
    )

    assert "binary output suppressed" in observation.message
    assert "\x00" not in observation.message
    assert "\x0e" not in observation.message
    assert observation.properties["output_metadata"]["binary_suppressed"] is True


def test_repair_run_command_strips_terminal_controls_and_caps_one_line_output():
    docker_env = Mock(spec=DockerEnv)
    docker_env.run_bash.return_value = (
        "\x1b[31merror:\x1b[0m " + "x" * 20_000 + "\x0e",
        1,
    )

    observation = RunCommand(docker_env=docker_env).execute(
        RunCommandArgs(command="grep error build.log")
    )

    assert "error:" in observation.message
    assert "\x1b" not in observation.message
    assert "\x0e" not in observation.message
    assert len(observation.properties["output"]) <= 8000
    assert observation.properties["output_metadata"]["truncated"] is True


@pytest.mark.parametrize(
    "command",
    [
        "curl -L https://github.com/org/repo/commit/fix.patch",
        "/usr/bin/wget https://example.com/later-release.tar.gz",
        "timeout 20 curl https://raw.githubusercontent.com/org/repo/main/file.cpp",
        "git clone https://github.com/org/repo.git /tmp/upstream",
        "git -C /tmp/upstream fetch --unshallow",
        "command git pull origin main",
        "git ls-remote https://github.com/org/repo.git",
    ],
)
def test_repair_run_command_rejects_future_data_access(command):
    docker_env = Mock(spec=DockerEnv)

    observation = RunCommand(docker_env=docker_env).execute(
        RunCommandArgs(command=command)
    )

    assert observation.expect_correction is True
    assert "Do not retrieve or reference future fixes" in observation.message
    docker_env.run_bash.assert_not_called()


def test_repair_run_command_allows_local_git_inspection():
    docker_env = Mock(spec=DockerEnv)
    docker_env.run_bash.return_value = ("clean\n", 0)

    observation = RunCommand(docker_env=docker_env).execute(
        RunCommandArgs(command="git status --short")
    )

    assert observation.properties["exit_code"] == 0
    docker_env.run_bash.assert_called_once()


@pytest.mark.parametrize(
    "command",
    [
        "git checkout -- src/fix.cpp",
        "git restore src/fix.cpp && grep -n needle src/fix.cpp",
        "git -C . reset --hard HEAD",
        "command git clean -fd",
    ],
)
def test_repair_run_command_rejects_local_git_mutation(command):
    docker_env = Mock(spec=DockerEnv)

    observation = RunCommand(docker_env=docker_env).execute(
        RunCommandArgs(command=command)
    )

    assert observation.expect_correction is True
    assert "file editing actions" in observation.message
    docker_env.run_bash.assert_not_called()


def test_repository_generator_changes_are_imported_into_file_context():
    repository = InMemRepository({
        "include/widget.hpp": "int old_value;\n",
        "single_include/widget.hpp": "int old_value;\n",
    })
    file_context = FileContext(repo=repository)
    source_file = file_context.add_file("include/widget.hpp", add_extra=False)
    source_file.apply_changes("int new_value;\n")
    generated_content = "int new_value;\n"
    encoded_content = base64.b64encode(generated_content.encode()).decode()
    docker_env = Mock(spec=DockerEnv)
    docker_env.work_dir = "/repo"
    docker_env.copy_anything_to_container.return_value = True
    docker_env.run_bash.side_effect = [
        ("", 0),
        ("generator completed\n", 0),
        (
            "include/widget.hpp\0single_include/widget.hpp\0",
            0,
        ),
        (
            "__INTENT2FIX_GENERATED_FILE__\n" + encoded_content + "\n",
            0,
        ),
        (
            "__INTENT2FIX_GENERATED_FILE__\n" + encoded_content + "\n",
            0,
        ),
    ]

    observation = RunCommand(docker_env=docker_env).execute(
        RunCommandArgs(
            command="python3 tools/update_release.py",
            capture_generated_changes=True,
        ),
        file_context=file_context,
    )

    assert observation.expect_correction is False
    assert observation.properties["captured_generated_paths"] == [
        "single_include/widget.hpp",
    ]
    assert file_context.get_file("single_include/widget.hpp").content == (
        generated_content
    )
    assert set(file_context.get_patched_files()) == {
        "include/widget.hpp",
        "single_include/widget.hpp",
    }


def test_build_driver_generator_is_allowed_with_capture():
    repository = InMemRepository({
        "single_include/widget.hpp": "int old_value;\n",
    })
    file_context = FileContext(repo=repository)
    encoded_content = base64.b64encode(b"int new_value;\n").decode()
    docker_env = Mock(spec=DockerEnv)
    docker_env.work_dir = "/repo"
    docker_env.run_bash.side_effect = [
        ("generator completed\n", 0),
        ("single_include/widget.hpp\0", 0),
        ("__INTENT2FIX_GENERATED_FILE__\n" + encoded_content + "\n", 0),
    ]

    observation = RunCommand(docker_env=docker_env).execute(
        RunCommandArgs(
            command="make refresh-public-api",
            capture_generated_changes=True,
        ),
        file_context=file_context,
    )

    assert observation.expect_correction is False
    assert observation.properties["captured_generated_paths"] == [
        "single_include/widget.hpp",
    ]
    assert "int new_value;" in file_context.generate_git_patch()


def test_repository_writer_requires_capture_after_observed_write():
    repository = InMemRepository({"dist/public.hpp": "int old_value;\n"})
    file_context = FileContext(repo=repository)
    docker_env = Mock(spec=DockerEnv)
    docker_env.run_bash.side_effect = [
        ("before  -\n", 0),
        ("updated\n", 0),
        ("after  -\n", 0),
        ("dist/public.hpp\0", 0),
        ("", 0),
    ]

    observation = RunCommand(docker_env=docker_env).execute(
        RunCommandArgs(
            command="python3 tools/update_release.py",
        ),
        file_context=file_context,
    )

    assert observation.expect_correction is True
    assert observation.properties["fail_reason"] == "repository_writes_not_captured"
    assert "capture_generated_changes=true" in observation.message
    assert docker_env.run_bash.call_count == 5


def test_unclassified_writer_is_discarded_by_worktree_fingerprint():
    file_context = FileContext(
        repo=InMemRepository({"dist/public.hpp": "int old_value;\n"})
    )
    docker_env = Mock(spec=DockerEnv)
    docker_env.run_bash.side_effect = [
        ("before  -\n", 0),
        ("", 0),
        ("after  -\n", 0),
        ("dist/public.hpp\0", 0),
        ("", 0),
    ]
    command = "cp /tmp/replacement dist/public.hpp"
    assert not may_execute_repository_code(command)

    observation = RunCommand(docker_env=docker_env).execute(
        RunCommandArgs(command=command),
        file_context=file_context,
    )

    assert observation.expect_correction is True
    assert observation.properties["fail_reason"] == (
        "repository_writes_not_captured"
    )
    assert "side effects were discarded" in observation.message
    assert "file editing action" in observation.message


def test_generator_search_is_not_misclassified_as_generator_execution():
    docker_env = Mock(spec=DockerEnv)
    docker_env.run_bash.return_value = ("Makefile: amalgamate\n", 0)

    observation = RunCommand(docker_env=docker_env).execute(
        RunCommandArgs(command="grep -n 'amalgamate' Makefile")
    )

    assert observation.expect_correction is False
    assert observation.properties["exit_code"] == 0


@pytest.mark.parametrize(
    "command",
    [
        "timeout 30 python3 tools/update_release.py",
        "env MODE=release ./scripts/refresh_headers",
        "grep -n refresh Makefile; make refresh-public-api",
    ],
)
def test_repository_command_detection_is_name_agnostic(command):
    assert may_execute_repository_code(command)


def test_capture_flag_does_not_bypass_direct_compiler_policy():
    docker_env = Mock(spec=DockerEnv)

    observation = RunCommand(docker_env=docker_env).execute(
        RunCommandArgs(
            command="g++ -std=c++17 repro.cpp -o repro",
            capture_generated_changes=True,
        )
    )

    assert observation.expect_correction is True
    assert "RunSingleTest" in observation.message
    docker_env.run_bash.assert_not_called()


def test_repository_generator_capture_requires_file_context():
    docker_env = Mock(spec=DockerEnv)

    observation = RunCommand(docker_env=docker_env).execute(
        RunCommandArgs(
            command="python3 tools/update_release.py",
            capture_generated_changes=True,
        )
    )

    assert observation.expect_correction is True
    assert "current patch context" in observation.message
    docker_env.run_bash.assert_not_called()


def test_failed_repository_generator_discards_partial_tracked_writes():
    file_context = FileContext(
        repo=InMemRepository({"dist/public.hpp": "int old_value;\n"})
    )
    docker_env = Mock(spec=DockerEnv)
    docker_env.run_bash.side_effect = [
        ("failed halfway\n", 2),
        ("dist/public.hpp\0", 0),
        ("", 0),
    ]

    observation = RunCommand(docker_env=docker_env).execute(
        RunCommandArgs(
            command="python3 tools/update_release.py",
            capture_generated_changes=True,
        ),
        file_context=file_context,
    )

    assert observation.expect_correction is True
    assert observation.properties["fail_reason"] == "repository_generator_failed"
    assert "side effects were discarded" in observation.message
