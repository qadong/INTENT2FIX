import base64
import shlex
from typing import ClassVar, Type

from pydantic import Field, PrivateAttr, ConfigDict
from INTENT2FIX.actions.action import Action
from INTENT2FIX.actions.command_policy import (
    future_data_command_rejection,
    local_git_mutation_rejection,
    may_execute_repository_code,
)
from INTENT2FIX.actions.repository_write_guard import (
    restore_file_context_worktree,
    tracked_worktree_fingerprint,
    tracked_worktree_paths,
)
from INTENT2FIX.actions.output_sanitizer import sanitize_action_output
from INTENT2FIX.actions.schema import ActionArguments, Observation
from INTENT2FIX.actions.verification_sync import (
    mark_file_change,
    sync_verification_files,
)
from INTENT2FIX.file_context import FileContext
from INTENT2FIX.runtime.dockerenv import DockerEnv
from INTENT2FIX.reproduce.state import is_build_or_test_command, shell_file_write_reason


def _verification_command_reason(command: str) -> str:
    if not is_build_or_test_command(command):
        return ""
    return (
        "RunCommand is only for search and environment inspection. Compiler, build, and "
        "test commands must use RunSingleTest for focused verification or RunRegression "
        "for the repository suite so failures and the current patch sequence are tracked."
    )

class RunCommandArgs(ActionArguments):
    """Run a read-only repository search or environment-inspection command.

    The command starts in the repository directory. Do not use cd, file writes,
    build/test commands, network access, or Git history. Use grep/find because rg
    is unavailable; use RunSingleTest or RunRegression for verification. Do not
    cat executables or other binary files; inspect them with file, strings, readelf,
    or objdump. The only write-capable exception is a repository-owned artifact
    generator (a script or build target) invoked with
    capture_generated_changes=true; its tracked text outputs are imported into the
    patch.
    """
    command: str = Field(
        ...,
        description=(
            "One shell command in the current repository directory. It must be "
            "read-only unless it invokes a repository-owned generator together "
            "with capture_generated_changes=true."
        ),
    )
    capture_generated_changes: bool = Field(
        default=False,
        description=(
            "Set true only when running a repository-owned artifact generator. "
            "Do not use a temporary or candidate-created transformation script. "
            "The current patch is synced first and changes to repository-tracked "
            "generated text files are imported into the patch. Keep false for "
            "searches and environment inspection."
        ),
    )

    model_config = ConfigDict(title="RunCommand")

class RunCommand(Action):
    """
    An action to execute a shell command inside the Docker container.
    """
    args_schema: ClassVar[Type[ActionArguments]] = RunCommandArgs
    _docker_env: DockerEnv = PrivateAttr()

    def __init__(self, docker_env: DockerEnv, **kwargs):
        super().__init__(**kwargs)
        self._docker_env = docker_env

    def _worktree_paths(self) -> tuple[list[str], str]:
        return tracked_worktree_paths(self._docker_env)

    def _read_generated_text(self, path: str) -> tuple[str | None, str]:
        quoted = shlex.quote(path)
        marker = "__INTENT2FIX_GENERATED_FILE__"
        deleted = "__INTENT2FIX_GENERATED_DELETED__"
        output, exit_code = self._docker_env.run_bash(
            f"if [ ! -e {quoted} ]; then printf '{deleted}'; "
            f"elif [ -f {quoted} ]; then printf '{marker}\\n'; base64 -- {quoted}; "
            "else exit 65; fi"
        )
        if exit_code != 0:
            return None, f"Generated path is not a readable text file: {path}"
        if output == deleted:
            return None, ""
        prefix = marker + "\n"
        if not output.startswith(prefix):
            return None, f"Could not decode generated repository file: {path}"
        try:
            encoded = "".join(output[len(prefix):].splitlines())
            content = base64.b64decode(encoded, validate=True).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            return None, f"Generated file must be UTF-8 text ({path}): {exc}"
        return content, ""

    def _capture_generated_changes(
        self,
        file_context: FileContext,
    ) -> tuple[list[str], str]:
        paths, error = self._worktree_paths()
        if error:
            return [], error
        # Editor-created files are untracked in the container, but already belong
        # to the candidate. Import their shell edits/deletions as well as git diff.
        paths = list(dict.fromkeys([
            *paths,
            *(file.file_path for file in file_context.files if file.is_new),
        ]))
        if len(paths) > 100:
            return [], (
                "Generator changed more than 100 repository files; use a more "
                "focused repository generation target."
            )

        pending = []
        total_bytes = 0
        for path in paths:
            context_file = file_context.get_context_file(path, add_extra=False)
            current_content = context_file.content if context_file else ""
            generated_content, read_error = self._read_generated_text(path)
            if read_error:
                return [], read_error
            if generated_content == current_content:
                continue
            total_bytes += len((generated_content or "").encode("utf-8"))
            if total_bytes > 25 * 1024 * 1024:
                return [], (
                    "Generator produced more than 25 MiB of changed text; use a "
                    "more focused repository generation target."
                )
            pending.append((path, context_file, generated_content))

        imported = []
        for path, context_file, generated_content in pending:
            if generated_content is None and context_file is not None and context_file.is_new:
                mark_file_change(file_context, path)
                file_context.remove_file(path)
                for key in ("production_paths", "ephemeral_paths"):
                    if key in file_context.verification:
                        file_context.verification[key] = [
                            item for item in file_context.verification[key] if item != path
                        ]
                imported.append(path)
                continue
            if context_file is None:
                context_file = file_context.add_file(
                    path,
                    content="",
                    add_extra=False,
                )
            context_file.apply_changes(generated_content or "")
            mark_file_change(file_context, path)
            imported.append(path)
        return imported, ""

    def execute(
        self,
        args: RunCommandArgs,
        file_context: FileContext | None = None,
        **kwargs,
    ) -> Observation:
        """
        Executes the given command in the Docker container and returns the output.
        """
        rejection_reason = future_data_command_rejection(args.command)
        if rejection_reason:
            return Observation(
                message=rejection_reason,
                expect_correction=True,
                properties={"rejected_command": args.command},
            )

        rejection_reason = local_git_mutation_rejection(args.command)
        if rejection_reason:
            return Observation(
                message=rejection_reason,
                expect_correction=True,
                properties={"rejected_command": args.command},
            )

        artifact_paths = (
            file_context.verification.get("ephemeral_paths", [])
            if file_context is not None
            else ()
        )
        rejection_reason = shell_file_write_reason(args.command, artifact_paths)
        if rejection_reason:
            return Observation(
                message=rejection_reason,
                expect_correction=True,
                properties={"rejected_command": args.command},
            )

        repository_command = may_execute_repository_code(args.command)
        verification_reason = _verification_command_reason(args.command)
        if verification_reason and not (
            args.capture_generated_changes and repository_command
        ):
            return Observation(
                message=verification_reason,
                expect_correction=True,
                properties={"rejected_command": args.command},
            )

        if args.capture_generated_changes:
            if file_context is None:
                return Observation(
                    message=(
                        "Generated changes cannot be captured without the current "
                        "patch context."
                    ),
                    expect_correction=True,
                    properties={"rejected_command": args.command},
                )
            sync_failure = sync_verification_files(
                file_context,
                self._docker_env,
            )
            if sync_failure:
                return Observation(
                    message=sync_failure,
                    expect_correction=True,
                    properties={"rejected_command": args.command},
                )
        elif file_context is not None:
            sync_failure = sync_verification_files(
                file_context,
                self._docker_env,
            )
            if sync_failure:
                return Observation(
                    message=sync_failure,
                    expect_correction=True,
                    properties={"rejected_command": args.command},
                )

        write_guard_fingerprint = ""
        if (
            not args.capture_generated_changes
            and file_context is not None
        ):
            write_guard_fingerprint, fingerprint_error = (
                tracked_worktree_fingerprint(self._docker_env)
            )
            if fingerprint_error:
                return Observation(
                    message=fingerprint_error,
                    expect_correction=True,
                    properties={"rejected_command": args.command},
                )

        output, exit_code = self._docker_env.run_bash(args.command)
        safe_output = sanitize_action_output(
            output,
            max_chars=8000,
            max_lines=100,
            keep="head_tail",
        )
        if write_guard_fingerprint:
            after_fingerprint, fingerprint_error = tracked_worktree_fingerprint(
                self._docker_env
            )
            if fingerprint_error:
                return Observation(
                    message=fingerprint_error,
                    expect_correction=True,
                    properties={"rejected_command": args.command},
                )
            if after_fingerprint != write_guard_fingerprint:
                restore_error = restore_file_context_worktree(
                    self._docker_env,
                    file_context,
                )
                message = (
                    "This RunCommand modified tracked repository files, so its "
                    "side effects were discarded. Use a file editing action for "
                    "direct edits. Only a repository-owned artifact generator "
                    "may be re-run with capture_generated_changes=true to import "
                    "its generated outputs into the patch."
                )
                if restore_error:
                    message += f" Worktree restoration failed: {restore_error}"
                return Observation(
                    message=message,
                    expect_correction=True,
                    properties={
                        "fail_reason": "repository_writes_not_captured",
                        "rejected_command": args.command,
                        "exit_code": exit_code,
                    },
                )

        imported_paths = []
        if args.capture_generated_changes and exit_code != 0:
            restore_error = restore_file_context_worktree(
                self._docker_env,
                file_context,
            )
            message = (
                f"Repository generator failed with exit code {exit_code}; partial "
                "tracked-file side effects were discarded."
            )
            if safe_output.text:
                message += (
                    "\n--- output (sanitized and bounded) ---\n"
                    + safe_output.text
                )
            if restore_error:
                message += f" Worktree restoration failed: {restore_error}"
            return Observation(
                message=message,
                expect_correction=True,
                properties={
                    "command": args.command,
                    "output": safe_output.text,
                    "output_metadata": safe_output.metadata,
                    "exit_code": exit_code,
                    "fail_reason": "repository_generator_failed",
                },
            )
        if args.capture_generated_changes and exit_code == 0:
            imported_paths, capture_error = self._capture_generated_changes(
                file_context,
            )
            if capture_error:
                restore_error = restore_file_context_worktree(
                    self._docker_env,
                    file_context,
                )
                if restore_error:
                    capture_error += f" Worktree restoration failed: {restore_error}"
                return Observation(
                    message=capture_error,
                    expect_correction=True,
                    properties={
                        "command": args.command,
                        "output": safe_output.text,
                        "output_metadata": safe_output.metadata,
                        "exit_code": exit_code,
                    },
                )

        if exit_code == 0:
            message = "Command executed successfully with exit code 0."
        else:
            message = f"Command failed with exit code {exit_code}."

        if safe_output.text:
            message += (
                "\n--- output (sanitized and bounded) ---\n"
                + safe_output.text
            )
            
        if not safe_output.text:
            message += " No output."
        if args.capture_generated_changes:
            if imported_paths:
                message += (
                    "\nTracked generated changes: "
                    + ", ".join(imported_paths)
                )
            elif exit_code == 0:
                message += "\nThe generator produced no new tracked text changes."

        return Observation(
            message=message,
            properties={
                "command": args.command,
                "output": safe_output.text,
                "output_metadata": safe_output.metadata,
                "exit_code": exit_code,
                "captured_generated_paths": imported_paths,
            }
        )
