import subprocess
from collections import deque
from INTENT2FIX.runtime.runtime import RuntimeEnvironment
from INTENT2FIX.actions.test_parser import parse_test_output, TestResult
from typing import Tuple
import atexit
import os
import shlex
import tempfile
import time
from typing import Optional, List
import logging
logger = logging.getLogger(__name__)
import docker

from INTENT2FIX.exceptions import InfrastructureRuntimeError
from INTENT2FIX.actions.output_sanitizer import OUTPUT_CAPTURE_TRUNCATION_MARKER


_MAX_COMMAND_OUTPUT_BYTES = 4 * 1024 * 1024


class DockerEnv(RuntimeEnvironment):
    def __init__(
        self,
        image_name: str,
        container_name: str,
        work_dir: str = "/app",
        host_work_dir: str = None,
        reuse_existing: bool = False,
        benchmark_base_commit: str | None = None,
    ):
        self.image_name = image_name
        self.container_name = container_name
        self.work_dir = work_dir
        self.host_work_dir = host_work_dir
        self.reuse_existing = reuse_existing
        self.benchmark_base_commit = benchmark_base_commit
        self.client = docker.from_env()
        self.container = None
        self._enter_count = 0

    def __enter__(self):
        if not self.init_container():
            raise ConnectionError("Failed to initialize Docker container.")
        try:
            if self.benchmark_base_commit:
                self._sanitize_benchmark_workspace()
        except Exception:
            self.cleanup()
            raise
        self._enter_count += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._enter_count = max(0, self._enter_count - 1)
        if self._enter_count == 0:
            self.cleanup()
        
    def init_container(self) -> bool:
        try:
            # Clean up any existing container with the same name
            try:
                existing_container = self.client.containers.get(self.container_name)
                existing_container.reload()
                if self.reuse_existing and existing_container.status == 'running':
                    logger.info(f"Reusing existing container: {self.container_name}")
                    self.container = existing_container
                    return True
                logger.warning(f"Removing existing container: {self.container_name}")
                existing_container.remove(force=True)
            except docker.errors.NotFound:
                pass  # No existing container, which is fine

            volumes = {}
            if self.host_work_dir:
                volumes[self.host_work_dir] = {'bind': self.work_dir, 'mode': 'rw'}
                logger.info(f"Mounting host directory '{self.host_work_dir}' to container directory '{self.work_dir}'")
            llvm_toolchain_dir = os.getenv("COTEST_LLVM_TOOLCHAIN_DIR", "").strip()
            if llvm_toolchain_dir:
                llvm_toolchain_dir = os.path.abspath(
                    os.path.expanduser(llvm_toolchain_dir)
                )
                if not os.path.isdir(llvm_toolchain_dir):
                    raise FileNotFoundError(
                        "COTEST_LLVM_TOOLCHAIN_DIR does not exist: "
                        f"{llvm_toolchain_dir}"
                    )
                volumes[llvm_toolchain_dir] = {
                    'bind': '/opt/intent2fix-llvm',
                    'mode': 'ro',
                }
                logger.info(
                    "Mounting shared LLVM toolchain '%s' read-only",
                    llvm_toolchain_dir,
                )

            logger.info(f"Creating container {self.container_name} from image {self.image_name}")
            self.container = self.client.containers.run(
                self.image_name,
                name=self.container_name,
                detach=True,
                tty=True,
                working_dir=self.work_dir,
                stdin_open=True,
                security_opt=["seccomp=unconfined"],
                network_mode="host",
                volumes=volumes
            )
            
            # Verify the container is running
            self.container.reload()
            if self.container.status != 'running':
                logger.error(f"Container {self.container_name} failed to start. Status: {self.container.status}")
                return False

            logger.info(f"Container {self.container_name} started successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to start container: {e}", exc_info=True)
            return False

    def _sanitize_benchmark_workspace(self) -> None:
        expected_commit = shlex.quote(self.benchmark_base_commit)
        command = rf"""
set -eu
expected_commit={expected_commit}
actual_commit=$(git rev-parse HEAD)
if [ "$actual_commit" != "$expected_commit" ]; then
    echo "Benchmark HEAD mismatch: expected $expected_commit, found $actual_commit" >&2
    exit 86
fi
git checkout --detach "$expected_commit" >/dev/null 2>&1
git remote | while IFS= read -r remote; do
    [ -z "$remote" ] || git remote remove "$remote"
done
git for-each-ref --format='%(refname)' | while IFS= read -r ref; do
    [ -z "$ref" ] && continue
    case "$ref" in
        refs/tags/*)
            tag_commit=$(git rev-parse "$ref^{{commit}}" 2>/dev/null || true)
            if [ -n "$tag_commit" ] && git merge-base --is-ancestor "$tag_commit" "$expected_commit"; then
                continue
            fi
            ;;
    esac
    git update-ref -d "$ref"
done
git reflog expire --expire=now --all
git gc --prune=now >/dev/null 2>&1
find /home -maxdepth 1 \( -type f -o -type l \) -name '*.patch' -delete
test -z "$(git remote)"
test "$(git rev-parse HEAD)" = "$expected_commit"
"""
        output, exit_code = self.run_bash(command, timeout=600)
        if exit_code != 0:
            raise RuntimeError(
                "Failed to sanitize benchmark container before agent execution: "
                f"{output.strip()}"
            )
        logger.info(
            "Sanitized benchmark workspace at %s: retained HEAD %s, removed "
            "future refs/remotes and /home/*.patch",
            self.work_dir,
            self.benchmark_base_commit,
        )

    def cleanup(self):
        if self.container:
            try:
                logger.info(f"Stopping and removing container {self.container_name}...")
                # First, try to stop the container if it's running
                try:
                    self.container.reload()
                    if self.container.status == 'running':
                        logger.info(f"Stopping container {self.container_name}...")
                        self.container.stop(timeout=10)  # Stop with 10 second timeout
                        logger.info(f"Container {self.container_name} stopped.")
                except docker.errors.NotFound:
                    logger.warning(f"Container {self.container_name} not found, may have been removed already.")
                except Exception as e:
                    logger.warning(f"Error stopping container {self.container_name}: {e}. Will try to remove anyway.")
                
                # Then remove the container
                try:
                    self.container.remove(force=True)
                    logger.info(f"Container {self.container_name} removed.")
                except docker.errors.NotFound:
                    logger.warning(f"Container {self.container_name} not found during removal, likely already removed.")
            except docker.errors.NotFound:
                logger.warning(f"Container {self.container_name} not found during cleanup, likely already removed.")
            except Exception as e:
                logger.error(f"An error occurred during container cleanup: {e}", exc_info=True)
                # Try to force remove even if there was an error
                try:
                    if self.container:
                        self.container.remove(force=True)
                        logger.info(f"Force removed container {self.container_name} after error.")
                except:
                    pass  # Ignore errors during force cleanup
            finally:
                self.container = None

    def cleanup_image(self, force: bool = False):
        """
        Removes the Docker image associated with this environment.
        """
        try:
            logger.info(f"Removing image: {self.image_name}")
            self.client.images.remove(image=self.image_name, force=force)
            logger.info(f"Successfully removed image: {self.image_name}")
        except docker.errors.ImageNotFound:
            logger.warning(f"Image {self.image_name} not found during cleanup.")
        except Exception as e:
            logger.error(f"An error occurred during image cleanup: {e}")
			
    def copy_anything_to_container(self, local_path: str, container_path: str) -> bool:
        if not self.container:
            logger.error("Container not initialized")
            return False
            
        try:
            safe_local = f"'{local_path}'" if ' ' in local_path else local_path
            safe_container = f"'{container_path}'" if ' ' in container_path else container_path
            
            copy_cmd = f"docker cp {safe_local} {self.container_name}:{safe_container}"
            subprocess.run(copy_cmd, shell=True, check=True, capture_output=True)
            
            chmod_cmd = f"docker exec -u root {self.container_name} chmod -R 777 {safe_container}"
            subprocess.run(chmod_cmd, shell=True, check=True, capture_output=True)
            logger.info(f"Copied and set permissions: {local_path} -> {container_path}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"File copy/chmod failed: {e.stderr.decode().strip()}")
            return False
        except Exception as e:
            logger.error(f"Error during file copy: {str(e)}")
            return False
            
    def run_bash(self, command: str, work_dir: Optional[str] = None, timeout: int = 1800) -> tuple[str, int]:
        if not self.container:
            logger.error("Container not initialized")
            return "Container not initialized", 1
            
        work_dir = work_dir or self.work_dir
        try:
            # Properly escape the command for sh -c
            # Escape single quotes by replacing ' with '\''
            escaped_command = command.replace("'", "'\\''")
            
            # Use timeout command inside the container
            # Wrap the entire command in parentheses or use sh -c to ensure timeout executes the full command sequence
            if timeout:
                # Use sh -c to wrap the command so timeout can execute the entire command sequence
                # Format: timeout <seconds> sh -c '<command>'
                full_command = f"timeout -k 5s {timeout}s sh -c 'cd {work_dir} && {escaped_command}'"
            else:
                full_command = f"sh -c 'cd {work_dir} && {escaped_command}'"
            
            exec_id = self.client.api.exec_create(
                self.container.id,
                full_command,
                stdout=True,
                stderr=True
            )
            stream = self.client.api.exec_start(exec_id, stream=True)
            tail = deque()
            tail_bytes = 0
            total_bytes = 0
            try:
                for chunk in stream:
                    total_bytes += len(chunk)
                    if chunk:
                        chunk = chunk[-_MAX_COMMAND_OUTPUT_BYTES:]
                        tail.append(chunk)
                        tail_bytes += len(chunk)
                        while tail_bytes > _MAX_COMMAND_OUTPUT_BYTES:
                            first = tail.popleft()
                            excess = tail_bytes - _MAX_COMMAND_OUTPUT_BYTES
                            if len(first) > excess:
                                tail.appendleft(first[excess:])
                                tail_bytes = _MAX_COMMAND_OUTPUT_BYTES
                            else:
                                tail_bytes -= len(first)
            finally:
                stream.close()
            output = b"".join(tail).decode('utf-8', errors='replace')
            if total_bytes > _MAX_COMMAND_OUTPUT_BYTES:
                marker = OUTPUT_CAPTURE_TRUNCATION_MARKER
                output = (
                    f"{marker} {total_bytes} bytes; retaining last {_MAX_COMMAND_OUTPUT_BYTES} bytes.\n"
                    + output
                    + f"\n{marker}"
                )
            
            exit_code = self.client.api.exec_inspect(exec_id)['ExitCode']
            
            if exit_code == 124:
                logger.warning(f"Command timed out after {timeout} seconds: {command}")
                output += f"\n\n[Command timed out after {timeout} seconds]"
            
            return output, exit_code
        except docker.errors.NotFound as e:
            logger.error(f"Repair container is no longer available: {e}")
            raise InfrastructureRuntimeError(
                f"Docker container {self.container_name} no longer exists."
            ) from e
        except docker.errors.APIError as e:
            status_code = getattr(
                getattr(e, "response", None),
                "status_code",
                None,
            )
            if status_code == 404 or "No such container" in str(e):
                logger.error(f"Repair container is no longer available: {e}")
                raise InfrastructureRuntimeError(
                    f"Docker container {self.container_name} no longer exists."
                ) from e
            logger.error(f"Failed to execute command in container: {e}")
            return str(e), 1
        except Exception as e:
            logger.error(f"Failed to execute command in container: {e}")
            return str(e), 1

    def write_file(self, file_path: str, content: str):
        if not self.container:
            logger.error("Container not initialized. Cannot write file.")
            return

        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding='utf-8') as tmp:
            tmp.write(content)
            temp_host_path = tmp.name
        
        try:
            parent_dir = os.path.dirname(file_path)
            if parent_dir:
                self.run_bash(f"mkdir -p {parent_dir}")
            self.copy_anything_to_container(temp_host_path, file_path)
        finally:
            os.remove(temp_host_path)
# from INTENT2FIX.actions.test_parser import parse_test_output, TestResult

# Other methods (run_compile, etc.) remain unchanged
    def run_compile(self, workspace: Optional[str] = None) -> Tuple[bool, str]:
        if not self.container:
            logger.error("Container not initialized")
            return False, "Container not running"
        
        try:
            work_dir = workspace if workspace is not None else self.work_dir
            # Try build_run.sh first if it's a combined build/run script, then fallback to build.sh
            compile_command = r'''
                if [ -f "./build_run.sh" ]; then
                    ./build_run.sh --build-only
                elif [ -f "./build.sh" ]; then
                    ./build.sh
                else
                    make -j4
                fi
            '''
            output, exit_code = self.run_bash(compile_command, work_dir=work_dir)
            if exit_code == 0:
                return True, "Compilation successful"
            
            return False, f"Build failed (exit code {exit_code}):\n{output}"
        except Exception as e:
            logger.exception("Unexpected error during compilation")
            return False, f"Unexpected error: {str(e)}"

    def run_test(self, test_command: str = None, language: str = "python", work_dir: str=None) -> List[TestResult]:
        """
        Run tests in the container based on language.
        
        Args:
            test_command: Custom test command to run (optional)
            language: Language of the tests (python, cpp, etc.)
            
        Returns:
            List[TestResult]: Results of the test execution
        """
        if not test_command:
            if language.lower() == "python":
                test_command = "pytest"
            elif language.lower() in ["cpp", "c++"]:
                test_command = "cd build && ctest --output-on-failure"
            else:
                logger.error(f"Unsupported language: {language}")
                return []
        
        output, exit_code = self.run_bash(test_command, work_dir=work_dir)
        
        logger.info(f"Test output: {output}")
        
        test_summary = parse_test_output(output)
        return test_summary.results



    def run_tests(self, patch: str | None = None, test_files: List[str] | None = None) -> List[TestResult]:
        """
        Implementation of abstract method from RuntimeEnvironment.
        Runs tests in the Docker container.
        
        Args:
            patch: Optional patch to apply before running tests
            test_files: Optional list of specific test files to run
            
        Returns:
            List[TestResult]: Results of the test execution
        """
        # If patch is provided, apply it (implementation depends on your needs)
        if patch:
            logger.warning("Patch application not implemented in DockerEnv")
            
        # Determine test command based on files
        test_command = "pytest"
        if test_files:
            test_command += " " + " ".join(test_files)
            
        return self.run_test(test_command=test_command)

    def is_test_failed(self,test_summary: str | tuple) -> bool:
        """
        判断测试是否失败，支持两种输入格式：
        - 字符串: "Test output..."
        - 元组: ("Test output...", exit_code)
        
        Args:
            test_summary: 测试输出，可以是字符串或元组
            
        Returns:
            bool: 如果测试失败返回True
        """
        # 如果是元组，提取字符串部分和退出码
        if isinstance(test_summary, tuple):
            output_str, exit_code = test_summary
            # 如果退出码非零，直接判定失败
            if exit_code != 0:
                return True
        else:
            output_str = test_summary
        
        # 检查输出中是否包含失败关键词
        return ("failed" in output_str.lower()) or ("error" in output_str.lower())




    def _parse_ctest_output(self, output: str | tuple) -> List[TestResult]:
        """
        增强的 ctest 输出解析器，现在可以处理：
        - 标准 ctest 输出格式
        - 详细输出 (-V 标志)
        - 错误消息
        - 测试计时信息
        - 元组格式的输出 (输出字符串, 退出码)

        Args:
            output: 原始 ctest 输出，可以是字符串或元组

        Returns:
            List[TestResult]: 包含状态和消息的详细测试结果
        """
        # 处理元组输入
        if isinstance(output, tuple):
            output_str = output[0]  # 提取字符串部分
            exit_code = output[1]   # 提取退出码
        else:
            output_str = output
            exit_code = 0           # 假设字符串输入表示成功

        results = []
        current_test = None
        test_output = []
        failed_tests = set()  # 存储失败测试的名称
        
        # 解析失败测试列表（从元组输出的最后部分）
        if isinstance(output, tuple) and "The following tests FAILED:" in output_str:
            failed_section = output_str.split("The following tests FAILED:")[1]
            for line in failed_section.split("\n"):
                if " - " in line:
                    failed_test = line.split(" - ")[1].split(" ")[0]
                    failed_tests.add(failed_test.strip())

        for line in output_str.split("\n"):
            # 检测新测试开始
            if line.startswith(("Test ", "    Start ")):
                if current_test:  # 保存上一个测试
                    results.append(current_test)
                
                # 解析测试名称和状态
                if line.startswith("Test "):
                    parts = line.split()
                    if len(parts) >= 5:
                        test_name = parts[2].strip("'")
                        status = TestStatus.PASSED if "Passed" in line else TestStatus.FAILED
                elif line.startswith("    Start "):
                    test_name = line.split(":")[1].strip()
                    # 初始状态设为PASSED，后续会根据失败列表更新
                    status = TestStatus.PASSED
                
                # 如果这个测试在失败列表中，更新状态
                if test_name in failed_tests:
                    status = TestStatus.FAILED
                    
                current_test = TestResult(
                    name=test_name,
                    status=status,
                    message=line
                )
                test_output = [line]
            
            # 收集当前测试的详细输出
            elif current_test:
                test_output.append(line)
                current_test.message = "\n".join(test_output)
                
                # 检查错误指示器
                if any(err in line.lower() for err in ["error", "fail", "exception"]):
                    current_test.status = TestStatus.FAILED
        
        # 添加最后一个测试
        if current_test:
            results.append(current_test)
        
        # 如果退出码非零但未捕获到失败，标记为整体失败
        if exit_code != 0 and not any(r.status == TestStatus.FAILED for r in results):
            if results:
                results[-1].status = TestStatus.FAILED
                results[-1].message += "\nCTest exited with code {}".format(exit_code)
        
        return results
