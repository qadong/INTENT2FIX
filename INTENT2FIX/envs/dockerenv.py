from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import random
import re
import shlex
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from functools import lru_cache
import docker
import docker.errors
import docker.models.containers
from simple_parsing.helpers.serialization.serializable import FrozenSerializable
from git import Repo
import gymnasium as gym

from INTENT2FIX.envs.utils import read_with_timeout, get_container

LONG_TIMEOUT = float(500)

@dataclass(frozen=True)
class EnvironmentArguments(FrozenSerializable):
    """Configure data sources and setup instructions for the environment in which we solve the tasks."""

    language:str | str = 'python',
    # Source of issue statement/problem statement. To run over a batch of issues: Path to a data file
    # (`json`, `jsonl`) or directory. To run over single issue: github issue url or path to markdown file
    # with problem statement or problem statement as text prefixed with `text://`.
    data_path: str | str = 'None',
    # Name of the docker image to use for the environment. Defaults to sweagent/swe-agent:latest
    image_name: str = "sweagent/swe-agent:latest"
    # When running over SWE-bench issues: Specify the split to use.
    split: str = "dev"
    # Specify a branch name or a commit hash to checkout before running the task.
    # Only used when running over a single problem statement/issue.
    base_commit: str | None = None
    # Use a persistent container with this name. After every task, the container will be paused, but not removed.
    # This is useful for speedup when running multiple tasks from the same repositories in a row, as the repositories
    # will have already been cloned and the conda environments will have been installed.
    container_name: str | None = None
    # Try to install the environment before running the task.
    install_environment: bool = True
    # No effect, kept for backwards compatibility.
    timeout: int | None = None
    # Enable environment logger.
    verbose: bool = False
    # Do not use attempt to use a repository mirror from https://github.com/swe-bench.
    no_mirror: bool = False
    # Cache task images to speed up task initialization. This means that the environment will be saved as a
    # docker image for every repository, base commit, and setup combination. This uses quite a bit of disk space
    # but speeds up task initialization significantly when running over multiple issues from the same repository
    # (or using different models for the same issues).
    cache_task_images: bool = False
    # Custom environment setup. Currently only used when data_path points to a single issue.
    # This needs to be either a string pointing to a yaml file (with yaml, yml file extension)
    # or a shell script (with sh extension).
    # See https://princeton-nlp.github.io/SWE-agent/usage/cl_tutorial#environment-setup
    environment_setup: str | None = None
    # Only used when running on single issue. Path to local repository or github repository.
    repo_path: str = ""
    persistent_volume: str | None = None

    def __post_init__(self):
        if self.timeout is not None:
            # default_logger.warning("The 'timeout' argument is deprecated and has no effect.")
            pass
        if self.cache_task_images and self.container_name:
            msg = (
                "Not allowed to use persistent container with caching task images "
                "(probably doesn't make sense and takes excessive space)."
            )
            raise ValueError(msg)
        if self.container_name is not None and self.container_name.strip() == "":
            msg = "Set container_name to None if you don't want to use a persistent container."
            raise ValueError(msg)


class DockerEnv(gym.Env):
    name = "swe_main"
    # This prefix will be prepended to the image name when caching task images
    cached_image_prefix = "swe-agent-task-env-"

    def __init__(self, args: EnvironmentArguments):
        super().__init__()
        t0 = time.perf_counter()
        self.args = args
        self.language = args.language
        # self.base_commit: str | None = None
        # self.communicate_output: str | None = None
        self.container_name: str | None = args.container_name
        self.install_environment = args.install_environment
        # self.logger = get_logger("SWEEnv")
        self.persistent = args.container_name is not None

        self.logger = logging.getLogger("dockerenv")
        # self.returncode: None | int = None
        # if not self.args.verbose:
        #     # fixme: This creates problems if we have multiple instances of this class
        #     self.logger.disabled = True

        # #: The commit hash of the swe-agent repository
        # self.commit_sha = None
        # try:
        #     repo = Repo(REPO_ROOT, search_parent_directories=True)
        #     self.commit_sha = repo.head.object.hexsha
        # except KeyboardInterrupt:
        #     raise
        # except Exception as e:
        #     self.logger.exception("Failed to get commit hash for this repo: %s", str(e))

        # self._github_token: str = keys_config.get("GITHUB_TOKEN", "")  # type: ignore

        # # Load Task Instances
        # self.data_path = self.args.data_path
        # self.data = get_instances(
        #     self.data_path,
        #     self.args.base_commit,
        #     self.args.split,
        #     token=self._github_token,
        #     repo_path=self.args.repo_path,
        # )
        #: Instance we're currently processing. Gets set in self.reset.
        # self.record: dict[str, Any] | None = None
        # self.logger.info(f"💽 Loaded dataset from {self.data_path}")

        # Establish connection with execution container
        self.image_name = args.image_name
        self.container_obj: docker.models.containers.Container | None = None
        self.container: subprocess.Popen | None = None
        self._init_container()

        self.idx = 0
        self.clean_multi_line_functions = lambda x: x

        self.logger.debug("Environment initialization took %.2f seconds", time.perf_counter() - t0)
    
    def _get_container_name(image_name: str) -> str:
        """Return name of container"""
        process_id = str(os.getpid())
        current_time = str(datetime.datetime.now())
        unique_string = current_time + process_id
        hash_object = hashlib.sha256(unique_string.encode())
        image_name_sanitized = image_name.replace("/", "-")
        image_name_sanitized = image_name_sanitized.replace(":", "-")
        return f"{image_name_sanitized}-{hash_object.hexdigest()[:10]}"



    def _init_container(self, cached_image: str | None = None) -> None:
        """
        Handles container initialization. Defines container name and creates it.
        If cached_image is provided, it will use that image name instead of the default.
        """
        image_name = self.image_name
        if cached_image is not None:
            image_name = cached_image
            self.logger.info(f"Using cached image: {image_name}")
        if self.persistent:
            assert self.container_name is not None
        else:
            # Make sure that we get a new container name just in case removing didn't work.
            # Might be a fix for https://github.com/princeton-nlp/SWE-agent/issues/451
            # self.container_name = self._get_container_name(image_name)
            
            self.container_name = 'test_container'
        volume_mount = f"{self.args.persistent_volume}:/root/persistent_data" if self.args.persistent_volume else None
        self.container, self.parent_pids = get_container(self.container_name, image_name, persistent=self.persistent, volume_mount=volume_mount)
        
        self.logger.info(f"🐳 Initializing container {self.container_name} {self.parent_pids}")
        try:
            client = docker.from_env(timeout=600)
        except docker.errors.DockerException as e:
            if "Error while fetching server API version" in str(e):
                msg = "Docker is not running. Please start Docker and try again."
            else:
                msg = "Unknown docker exception occurred. Are you sure docker is running?"
            raise RuntimeError(msg) from e
        t0 = time.time()
        self.container_obj = None
        while time.time() - t0 < 60:
            try:
                self.container_obj = client.containers.get(self.container_name)
            except docker.errors.NotFound:
                self.logger.debug("Couldn't find container. Let's wait and retry.")
                time.sleep(1)
            else:
                break
        else:
            print(f"{self.persistent=}")
            available_containers = client.containers.list(all=True)
            available_containers_info = json.dumps([str(c.attrs) for c in available_containers], indent=2)
            print(available_containers_info)
            msg = "Failed to get container object."
            raise RuntimeError(msg)
        self.logger.info("🌱 Environment Initialized")

    def _check_syntax(self, input: str) -> tuple[str, bool]:
            """
            Check syntax of command.

            Returns:
                output: Output of the command
                success: whether the exit code was 0
            """
            output = self._communicate(f"/bin/bash -n <<'EOF'\n{input}\nEOF\n")
            return output, self.returncode == 0
            
    def get_pids(self, all_pids: bool = False) -> list[str]:
            """
            Gets list of processes running inside docker container

            Args:
                all_pids: whether to return all pids, or whether to exclude ps
                    and parent PIDs

            Returns:
                list of PIDs
            """
            pids = self.container_obj.exec_run("ps -eo pid,comm --no-headers").output.decode().split("\n")
            pids = [x.split() for x in pids if x]
            if not all_pids:
                pids = [x for x in pids if x[1] != "ps" and x[0] not in self.parent_pids]
            return pids

    
    def _communicate(
        self,
        input: str,
        timeout_duration: int | float = 25,
    ) -> str:
        """Runs command in container and returns output

        Args:
            input: command to run in container
            timeout_duration: duration to wait for output
        """
        assert self.container is not None
        # communicate_method = keys_config.get(
        #     "SWE_AGENT_COMMUNICATE_METHOD", default="end-marker", choices=["end-marker", "processes"]
        # )
        # if communicate_method == "end-marker":
        #     return self._communicate_experimental(input, timeout_duration)
        try:
            self.returncode = None
            cmd = input if input.endswith("\n") else input + "\n"
            os.write(self.container.stdin.fileno(), cmd.encode())
            time.sleep(0.1)
            self.container.stdin.flush()
        except BrokenPipeError:
            traceback.print_exc()
            self.logger.error("Failed to communicate with container. Check docker logs for more information.")
            msg = "Failed to communicate with container"
            raise RuntimeError(msg)
        try:
            buffer = read_with_timeout(self.container, self.get_pids, timeout_duration)
            self.container.stdin.write("echo $?\n")
            time.sleep(0.1)
            self.container.stdin.flush()
            exit_code = read_with_timeout(self.container, self.get_pids, 5).strip()
        except Exception as e:
            self.logger.error(f"Read with timeout failed on input:\n---\n{input}\n---")
            raise e
        if not exit_code.isdigit():
            msg = f"Failed to get exit code. Output:\n---\n{buffer}\n---"
            raise RuntimeError(msg)
        self.returncode = int(exit_code)
        return buffer


    def communicate(self, input: str, timeout_duration: int | float = 25, *, set_last_action: bool = False) -> str:
        """
        Sends input to container and returns output

        Args:
            input: input to send to container
            timeout_duration: duration to wait for output
            set_last_action: whether to set the LAST_ACTION environment variable

        Returns:
            output: output from container
        """
        if input.strip() != "exit":
            self.logger.log(logging.DEBUG, "Input:\n%s", input)  # type: ignore
            output, valid = self._check_syntax(input)
            if not valid:
                return output  # shows syntax errors
            output = self._communicate(
                input,
                timeout_duration=timeout_duration,
            )
            self.logger.log(logging.DEBUG, "Output:\n%s", output)  # type: ignore
            self.communicate_output = output
            if set_last_action:
                # Cannot merge this with last command, because of multiline command
                # handling.
                last_action_string = shlex.quote(input.strip())
                input = f"export LAST_ACTION={last_action_string}"
                self._communicate(input, timeout_duration=60)
            return output
        else:
            self.container.terminate()
            self.returncode = 0
            self.communicate_output = ""
            return ""
            
    def communicate_with_handling(self, input: str, error_msg: str, timeout_duration: int | float = 25) -> str:
        """
        Wrapper for communicate function that raises error if return code is non-zero

        Args:
            input: input to send to container
            error_msg: error message to raise if return code is non-zero
            timeout_duration: duration to wait for output

        Returns:
            output: output from container
        """
        logs = self.communicate(input, timeout_duration=timeout_duration)
        if self.returncode != 0:
            self.logger.error(f"{error_msg}: {logs}")
            self.close()
            msg = f"{error_msg}: {logs}"
            raise RuntimeError(msg)
        return logs


    @lru_cache(maxsize=32)
    def _file_exists_in_repo(self, filename: str) -> bool:
        """
        
        """
        # 直接检查常见构建系统文件（快速路径）
        common_files = {
            "Makefile": f"{self.repo_container_path}/Makefile",
            "CMakeLists.txt": f"{self.repo_container_path}/CMakeLists.txt",
            "conanfile.txt": f"{self.repo_container_path}/conanfile.txt"
        }
        
        if filename in common_files:
            exit_code, _ = self.communicate(
                f"test -f {common_files[filename]}",
                verbose=False
            )
            return exit_code == 0
        
        # 通用查找逻辑
        return self._find_file_in_repo(filename)

    def _find_file_in_repo(self, filename: str) -> bool:
        """

        """
        repo_path = f"{self.repo_container_path}/"
        exit_code, output = self.communicate(
            f"find {repo_path} -path '*/{filename}' -type f -print -quit",
            verbose=False
        )
        return exit_code == 0 and bool(output.strip())


    def install_env(self) -> None:
        assert self.container is not None
        if self.language == "python":
            pass
            # install_env_py()
        elif self.language == "c":
            self.install_env_c()
        elif self.language == "cpp":
            self.install_env_cpp()
        elif self.language == "java":
            self.install_env_java()


    def install_env_c(self) -> None:
        """
        Installs C compiler and dependencies for C projects
        """
        t0 = time.perf_counter()
        # for hook in self.hooks:
        #     hook.on_install_env_started()
        
        # install_configs = self._get_install_configs()
        # if not install_configs:
        #     return
        
        # if "shell_script_path" in install_configs:
        #     assert len(install_configs) == 1
        #     self.run_shell_script(Path(install_configs["shell_script_path"]), location="container")
        #     return
        
        # assert self.record is not None
        
        # Install system dependencies
        self.logger.info("Installing C build essentials...")
        self.communicate_with_handling(
            "apt-get update && apt-get install -y build-essential gcc make cmake",
            error_msg="Failed to install C compiler toolchain",
            timeout_duration=LONG_TIMEOUT,
        )
        
        # Handle custom dependencies
        # if "packages" in install_configs:
        #     packages = install_configs["packages"]
        #     if packages.strip():
        #         self.communicate_with_handling(
        #             f"apt-get install -y {packages}",
        #             error_msg="Failed to install system packages",
        #             timeout_duration=LONG_TIMEOUT,
        #         )
        
        # Handle makefile if exists
        if self._file_exists_in_repo("Makefile"):
            self.logger.info("Found Makefile, running make...")
            self.communicate_with_handling(
                "make",
                error_msg="Make failed",
                timeout_duration=LONG_TIMEOUT,
            )
        
        # Handle CMake if exists
        if self._file_exists_in_repo("CMakeLists.txt"):
            self.logger.info("Found CMakeLists.txt, building with CMake...")
            self.communicate_with_handling(
                "mkdir -p build && cd build && cmake .. && make",
                error_msg="CMake build failed",
                timeout_duration=LONG_TIMEOUT,
            )
        
        self.logger.info(f"C environment setup completed in {time.perf_counter()-t0:.2f}s")

    def install_env_cpp(self) -> None:
        """
        Installs C++ compiler and dependencies for C++ projects
        """
        t0 = time.perf_counter()
        # for hook in self.hooks:
        #     hook.on_install_env_started()
        
        # install_configs = self._get_install_configs()
        # if not install_configs:
        #     return
        
        # if "shell_script_path" in install_configs:
        #     assert len(install_configs) == 1
        #     self.run_shell_script(Path(install_configs["shell_script_path"]), location="container")
        #     return
        
        # assert self.record is not None
        
        # Install system dependencies
        self.logger.info("Installing C++ build essentials...")
        self.communicate_with_handling(
            "apt-get update && apt-get install -y build-essential g++ make cmake",
            error_msg="Failed to install C++ compiler toolchain",
            timeout_duration=LONG_TIMEOUT,
        )
        cpp_standard = 14
        # Handle C++ standard if specified
        # cpp_standard = install_configs.get("cpp_standard", "14")
        # if "packages" in install_configs:
        #     packages = install_configs["packages"]
        #     if packages.strip():
        #         self.communicate_with_handling(
        #             f"apt-get install -y {packages}",
        #             error_msg="Failed to install system packages",
        #             timeout_duration=LONG_TIMEOUT,
        #         )
        
        # Handle makefile if exists
        if self._file_exists_in_repo("Makefile"):
            self.logger.info("Found Makefile, running make with CXXFLAGS...")
            self.communicate_with_handling(
                f"make CXXFLAGS='-std=c++{cpp_standard}'",
                error_msg="Make failed",
                timeout_duration=LONG_TIMEOUT,
            )
        
        # Handle CMake if exists
        if self._file_exists_in_repo("CMakeLists.txt"):
            self.logger.info("Found CMakeLists.txt, building with CMake...")
            self.communicate_with_handling(
                f"mkdir -p build && cd build && cmake -DCMAKE_CXX_STANDARD={cpp_standard} .. && make",
                error_msg="CMake build failed",
                timeout_duration=LONG_TIMEOUT,
            )
        
        # Handle conan if conanfile exists
        if self._file_exists_in_repo("conanfile.txt"):
            self.logger.info("Found conanfile.txt, installing dependencies...")
            self.communicate_with_handling(
                "apt-get install -y python3-pip && pip install conan",
                error_msg="Failed to install conan",
                timeout_duration=LONG_TIMEOUT,
            )
            self.communicate_with_handling(
                "conan profile detect --force",
                error_msg="Conan profile detection failed",
            )
            self.communicate_with_handling(
                "conan install . --build=missing",
                error_msg="Conan dependency installation failed",
                timeout_duration=LONG_TIMEOUT,
            )
        
        self.logger.info(f"C++ environment setup completed in {time.perf_counter()-t0:.2f}s")

    def install_env_java(self) -> None:
        """
        Installs java compiler and dependencies for C++ projects
        """
        pass           

    def close_container(self) -> None:
        """
        Closes the container and removes it if not persistent
        """
        if self.container is not None:
            self.container.terminate()
            self.container = None
        if self.container_obj is not None and not self.persistent:
            self.container_obj.remove(force=True)
            self.container_obj = None
            
    def copy_anything_to_container(self, host_path: str, container_path: str) -> None:
            """Copy files or directories from host to container

            Note: Will need to set ownership on the copied files in the container.
            """
            if not Path(host_path).exists():
                msg = f"Path {host_path} does not exist, cannot copy it to container."
                raise FileNotFoundError(msg)
            cmd = ["docker", "cp", host_path, f"{self.container_name}:{container_path}"]
            self.logger.debug(f"Copying {host_path} to container at {container_path} with command: {shlex.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                msg = f"Error copying {host_path} to container at {container_path}: {e}"
                raise RuntimeError(msg) from e