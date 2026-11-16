import logging
import re
from pydantic import ConfigDict, Field, PrivateAttr
from INTENT2FIX.actions.action import Action
from INTENT2FIX.actions.command_policy import future_data_command_rejection
from INTENT2FIX.actions.schema import ActionArguments, Observation
from INTENT2FIX.runtime.dockerenv import DockerEnv
from INTENT2FIX.reproduce.intent_schema import ExecutionRecord
from INTENT2FIX.reproduce.state import (
    ReproductionSession,
    command_references_artifact,
    is_build_or_test_command,
    is_reproduction_command,
    shell_file_write_reason,
)

logger = logging.getLogger(__name__)

class RunCommandArgs(ActionArguments):
    """Arguments for the RunCommand action."""
    command: str = Field(
        ...,
        description=(
            "The compile or execution command to run in the repository. Do not create, "
            "overwrite, move, or delete files; use CreateFile for reproduction source files."
        ),
    )

    model_config = ConfigDict(title="RunCommand")

class RunCommand(Action):
    """
    An action to execute a shell command inside the Docker container.
    This is critical for compiling and running the code to reproduce the bug.
    """
    args_schema: type[ActionArguments] = RunCommandArgs
    docker_env: DockerEnv
    session: ReproductionSession | None = None
    _run_counter: int = PrivateAttr(default=0)

    def _run_bash_detailed(self, command: str) -> tuple[str, str, int]:
        """
        A self-contained method to execute a bash command in the container 
        and return separated stdout, stderr, and exit code.
        This uses the high-level exec_run() method for stability.
        """
        if not self.docker_env.container:
            logger.error("Container not initialized")
            return "", "Container not initialized", 1

        try:
            # Use the high-level exec_run, which handles streams automatically
            # Properly escape the command for sh -c
            escaped_command = command.replace("'", "'\\''")
            exit_code, (stdout_bytes, stderr_bytes) = self.docker_env.container.exec_run(
                cmd=f"sh -c '{escaped_command}'",
                workdir=self.docker_env.work_dir,
                demux=True  # Demux separates stdout and stderr
            )

            stdout = stdout_bytes.decode('utf-8', errors='replace') if stdout_bytes else ""
            stderr = stderr_bytes.decode('utf-8', errors='replace') if stderr_bytes else ""

            return stdout, stderr, exit_code
        except Exception as e:
            logger.error(f"Failed to execute command in container: {e}")
            return "", str(e), 1

    def execute(self, args: RunCommandArgs, **kwargs) -> Observation:
        """
        Executes the given command in the Docker container and returns the output.
        """
        artifact_paths = self.session.created_artifacts if self.session else ()
        rejection_reason = future_data_command_rejection(args.command)
        if rejection_reason:
            return Observation(
                message=rejection_reason,
                expect_correction=True,
                properties={"rejected_command": args.command},
            )
        if self.session:
            rejection_reason = self.session.convergence_rejection("RunCommand")
            if rejection_reason:
                return Observation(
                    message=rejection_reason,
                    expect_correction=True,
                    properties={"rejected_command": args.command},
                )
            if self.session.revised_after_candidate and (
                not is_reproduction_command(args.command)
                or not any(
                    command_references_artifact(args.command, path)
                    for path in artifact_paths
                )
            ):
                return Observation(
                    message=(
                        "After revising a candidate reproducer, RunCommand must directly "
                        "compile or run a registered repro artifact."
                    ),
                    expect_correction=True,
                    properties={"rejected_command": args.command},
                )
        rejection_reason = shell_file_write_reason(args.command, artifact_paths)
        if not rejection_reason and (
            is_reproduction_command(args.command)
            or is_build_or_test_command(args.command)
        ) and re.search(r"(?<!\|)\|(?!\|)", args.command):
            rejection_reason = (
                "Compile and reproduction commands cannot use pipelines because they mask "
                "the tested command's exit status. Run the command directly."
            )
        if rejection_reason:
            return Observation(
                message=rejection_reason,
                expect_correction=True,
                properties={"rejected_command": args.command},
            )

        if self.session and self.session.revised_after_candidate:
            self.session.note_revised_execution()

        # Use the new detailed method
        self._run_counter += 1
        run_id = f"RUN-{self._run_counter}"
        stdout, stderr, exit_code = self._run_bash_detailed(args.command)

        # Truncate stdout and stderr to the last 100 lines for the message
        stdout_lines = stdout.splitlines()
        stderr_lines = stderr.splitlines()

        logger.info(f"Command executed ({run_id}): {args.command}")
        logger.info(f"Exit code: {exit_code}")

        stdout_truncated = "\n".join(stdout_lines[-100:])
        stderr_truncated = "\n".join(stderr_lines[-100:])

        # Combine truncated stdout and stderr into a single output string for the LLM
        output_str = ""
        if stdout_truncated:
            output_str += f"--- stdout (last 100 lines) ---\n{stdout_truncated}\n"

        if stderr_truncated:
            output_str += f"--- stderr (last 100 lines) ---\n{stderr_truncated}\n"

        if exit_code == 0:
            message = f"[{run_id}] Command executed successfully with exit code 0."
            if not output_str.strip():
                message += " No output."
            else:
                message += f"\n{output_str}"
        else:
            message = f"[{run_id}] Command failed with exit code {exit_code}.\n{output_str}"

        candidate_artifacts = []
        if self.session:
            record = ExecutionRecord(
                run_id=run_id,
                command=args.command,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
            )
            self.session.add_execution(record)
            if any(
                candidate.run_id == run_id
                for candidate in self.session.eligible_failure_records(artifact_paths)
            ):
                candidate_artifacts = [
                    path
                    for path in artifact_paths
                    if command_references_artifact(args.command, path)
                ]
                message += (
                    f"\n[{run_id}] is candidate reproduction evidence tied to "
                    f"{', '.join(candidate_artifacts)}. Assess whether this is the issue's "
                    f"target failure. If yes, call Finish next with {run_id}; otherwise "
                    "revise the reproducer."
                )

        # The full output is returned in the observation properties for the agent to analyze.
        return Observation(
            message=message,
            properties={
                "run_id": run_id,
                "command": args.command,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code,
                "candidate_reproduction_evidence": bool(candidate_artifacts),
                "candidate_artifacts": candidate_artifacts,
            }
        )
