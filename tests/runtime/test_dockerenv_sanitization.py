from unittest.mock import Mock

import docker
import pytest

from INTENT2FIX.actions.run_command import RunCommand  # noqa: F401
from INTENT2FIX.exceptions import InfrastructureRuntimeError
from INTENT2FIX.runtime.dockerenv import DockerEnv


def _docker_env(base_commit: str = "abc123") -> DockerEnv:
    env = object.__new__(DockerEnv)
    env.benchmark_base_commit = base_commit
    env.work_dir = "/home/repo"
    env.run_bash = Mock(return_value=("", 0))
    return env


def test_sanitize_benchmark_workspace_removes_future_git_data_and_patches():
    env = _docker_env()

    env._sanitize_benchmark_workspace()

    command = env.run_bash.call_args.args[0]
    assert "actual_commit=$(git rev-parse HEAD)" in command
    assert "git remote remove" in command
    assert "git update-ref -d" in command
    assert "git merge-base --is-ancestor" in command
    assert "git reflog expire --expire=now --all" in command
    assert "git gc --prune=now" in command
    assert "-name '*.patch' -delete" in command
    env.run_bash.assert_called_once_with(command, timeout=600)


def test_sanitize_benchmark_workspace_fails_closed():
    env = _docker_env()
    env.run_bash.return_value = ("Benchmark HEAD mismatch", 86)

    with pytest.raises(RuntimeError, match="Failed to sanitize benchmark container"):
        env._sanitize_benchmark_workspace()


def test_init_container_mounts_shared_llvm_toolchain(monkeypatch, tmp_path):
    client = Mock()
    client.containers.get.side_effect = docker.errors.NotFound("missing")
    container = Mock(status="running", id="container-id")
    client.containers.run.return_value = container
    monkeypatch.setattr(docker, "from_env", lambda: client)
    monkeypatch.setenv("COTEST_LLVM_TOOLCHAIN_DIR", str(tmp_path))

    env = DockerEnv(
        image_name="benchmark-image",
        container_name="intent-container",
    )

    assert env.init_container()
    volumes = client.containers.run.call_args.kwargs["volumes"]
    assert volumes[str(tmp_path)] == {
        "bind": "/opt/intent2fix-llvm",
        "mode": "ro",
    }


def test_run_bash_raises_when_active_container_was_removed():
    env = object.__new__(DockerEnv)
    env.container_name = "repair-instance"
    env.container = Mock(id="missing-container")
    env.client = Mock()
    env.work_dir = "/home/repo"
    env.client.api.exec_create.side_effect = docker.errors.NotFound(
        "No such container: missing-container"
    )

    with pytest.raises(
        InfrastructureRuntimeError,
        match="repair-instance no longer exists",
    ):
        env.run_bash("true")
