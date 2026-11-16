import hashlib
import logging
import re
from pydantic import Field, PrivateAttr, ConfigDict
from typing import List
from INTENT2FIX.actions.action import Action
from INTENT2FIX.actions.command_policy import (
    may_execute_repository_code,
    verification_exit_status_rejection,
)
from INTENT2FIX.actions.output_sanitizer import sanitize_action_output
from INTENT2FIX.actions.repository_write_guard import (
    restore_file_context_worktree,
    tracked_worktree_fingerprint,
)
from INTENT2FIX.actions.schema import ActionArguments, Observation
from INTENT2FIX.actions.schema import ActionArguments, Observation, FewShotExample
from INTENT2FIX.file_context import FileContext
from INTENT2FIX.patch_review.sanitizer import sanitize_candidate_patch
from INTENT2FIX.runtime.dockerenv import DockerEnv
from INTENT2FIX.utils.patch import production_patch_sha256
from .test_parser import parse_test_output, LogType
from .verification_sync import (
    current_probe_change_seq,
    sync_verification_files,
)

logger = logging.getLogger(__name__)

def _compute_passed(
    exit_code: int,
    test_summary,
    *,
    expected_exit_code: int = 0,
    output: str = "",
    required_output_regex: str = "",
) -> bool:
    """
    Match the declared process outcome; parsed partial success cannot override it.
    """
    if exit_code != expected_exit_code:
        return False
    if expected_exit_code != 0 and not required_output_regex:
        return False
    if expected_exit_code != 0 and re.search(
        r"command not found|No such file or directory|No CMAKE_CXX_COMPILER|"
        r"Cannot connect to the Docker daemon|(?:sh|bash):[^\n]*Syntax error",
        output, re.I,
    ):
        return False
    try:
        log_type = getattr(test_summary, "log_type", None)
        if log_type in {LogType.MEMORY_LEAK, LogType.NO_TESTS, LogType.OUTPUT_TRUNCATED}:
            return False
        if log_type == LogType.COMPILATION_ERROR and expected_exit_code == 0:
            return False
        if expected_exit_code == 0 and log_type == LogType.TEST_RESULTS:
            results = getattr(test_summary, "results", []) or []
            failed = sum(1 for r in results if getattr(r, "status", None) == "failed")
            exit_matches = failed == 0
        else:
            exit_matches = exit_code == expected_exit_code
    except Exception:
        # Fall back to exit code below
        exit_matches = exit_code == expected_exit_code
    if not exit_matches:
        return False
    if required_output_regex:
        try:
            return re.search(required_output_regex, output, re.MULTILINE) is not None
        except re.error:
            return False
    return True


def _compilation_failure_signature(output: str) -> str:
    for line in output.splitlines():
        match = re.search(r"\b(?:fatal\s+)?error:\s*(.+)", line, re.IGNORECASE)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()
    return ""


class RunSingleTestArgs(ActionArguments):
    """Compile or run one focused reproducer, probe, or targeted test.

    The latest patch is synchronized first and the command starts in the repository
    directory. Preserve the real exit status: do not pipe into head or append an
    echo that masks failure. Use RunRegression for the configured full suite. Do
    not run repository artifact generators here; use RunCommand with
    capture_generated_changes=true so derived files enter the tracked patch.
    """

    thoughts: str = Field(..., description="What behavior this focused command distinguishes.")
    command: str = Field(
        ...,
        description=(
            "Direct repository-relative compile/run command whose exit status "
            "represents success or failure. It must be verification-only: it "
            "must not generate, synchronize, or edit tracked repository files."
        ),
    )
    expected_exit_code: int = Field(
        default=0,
        ge=0,
        le=255,
        description=(
            "Exit code that means this focused behavioral oracle succeeded. "
            "Use a nonzero value only when the intended fixed behavior itself "
            "is a controlled failure, and pair it with required_output_regex."
        ),
    )
    required_output_regex: str = Field(
        default="",
        max_length=1000,
        description=(
            "Optional Python regular expression that the command output must "
            "match. Required when expected_exit_code is nonzero so an arbitrary "
            "failure cannot be counted as successful verification."
        ),
    )
    model_config = ConfigDict(title="RunSingleTest")

    @property
    def log_name(self):
        return f"RunSingleTest(command='{self.command}')"


class RunSingleTest(Action):
    args_schema = RunSingleTestArgs

    _runtime: DockerEnv = PrivateAttr()
    _last_compilation_failure: str = PrivateAttr(default="")
    _compilation_failure_repeats: int = PrivateAttr(default=0)
    _last_execution_key: tuple | None = PrivateAttr(default=None)
    _last_execution_result: tuple | None = PrivateAttr(default=None)
    _identical_execution_repeats: int = PrivateAttr(default=0)

    def __init__(
        self,
        runtime: DockerEnv | None = None,
        **data,
    ):
        super().__init__(**data)
        self._runtime = runtime

    def execute(
        self,
        args: RunSingleTestArgs,
        file_context: FileContext,
        **kwargs,
    ) -> Observation:
        """
        Run a single test in the Docker environment after applying a patch.
        """
        if not self._runtime:
            raise ValueError("Runtime environment must be provided to execute the run single test action.")

        status_rejection = verification_exit_status_rejection(args.command)
        if status_rejection:
            return Observation(
                message=status_rejection,
                summary="Focused verification command can mask test failure.",
                expect_correction=True,
                properties={"fail_reason": "masked_verification_exit_status"},
            )

        if args.expected_exit_code != 0 and not args.required_output_regex:
            return Observation(
                message=(
                    "A nonzero expected_exit_code needs required_output_regex "
                    "to identify the intended controlled failure."
                ),
                summary="Focused nonzero oracle is underspecified.",
                expect_correction=True,
                properties={"fail_reason": "missing_expected_output_regex"},
            )
        if args.required_output_regex:
            try:
                re.compile(args.required_output_regex)
            except re.error as exc:
                return Observation(
                    message=f"Invalid required_output_regex: {exc}",
                    summary="Focused output oracle regex is invalid.",
                    expect_correction=True,
                    properties={"fail_reason": "invalid_expected_output_regex"},
                )

        execution_key = (
            args.command,
            args.expected_exit_code,
            args.required_output_regex,
            getattr(file_context, "change_seq", 0),
            current_probe_change_seq(file_context),
        )
        if (
            execution_key == self._last_execution_key
            and self._identical_execution_repeats >= 2
        ):
            return Observation(
                message=(
                    "This exact focused verification already produced the same "
                    "result twice with unchanged production and probe state. "
                    "Do not run it again unchanged. Revise production code or "
                    "the existing probe only when the observation identifies a "
                    "specific defect; otherwise report the unresolved evidence "
                    "and call Finish."
                ),
                summary="Unchanged duplicate focused verification blocked.",
                expect_correction=True,
                properties={
                    "fail_reason": "unchanged_duplicate_verification",
                    "identical_execution_repeats": (
                        self._identical_execution_repeats
                    ),
                },
            )

        sync_failure = sync_verification_files(file_context, self._runtime)
        if sync_failure:
            return Observation(message=sync_failure, summary="File synchronization failed.")

        write_guard_fingerprint = ""
        if may_execute_repository_code(args.command):
            write_guard_fingerprint, fingerprint_error = (
                tracked_worktree_fingerprint(self._runtime)
            )
            if fingerprint_error:
                return Observation(
                    message=fingerprint_error,
                    summary="Could not guard focused verification writes.",
                    expect_correction=True,
                )

        candidate = sanitize_candidate_patch(file_context)
        execution_snapshot = {
            "production_patch_sha256": production_patch_sha256(candidate.patch),
            "command": args.command,
            "expected_exit_code": args.expected_exit_code,
            "required_output_regex": args.required_output_regex,
            "change_seq": file_context.change_seq,
            "probe_change_seq": current_probe_change_seq(file_context),
            "artifact_errors": list(candidate.artifact_errors),
            "probe_files": {
                path: file_context.get_file(path).content
                for path in file_context.verification.get("ephemeral_paths", [])
                if path in args.command and file_context.get_file(path) is not None
            },
        }
        output, exit_code = self._runtime.run_bash(args.command)
        safe_output = sanitize_action_output(
            output,
            max_chars=2000,
            keep="head_tail",
        )
        if write_guard_fingerprint:
            after_fingerprint, fingerprint_error = tracked_worktree_fingerprint(
                self._runtime
            )
            if fingerprint_error:
                return Observation(
                    message=fingerprint_error,
                    summary="Could not guard focused verification writes.",
                    expect_correction=True,
                )
            if after_fingerprint != write_guard_fingerprint:
                restore_error = restore_file_context_worktree(
                    self._runtime,
                    file_context,
                )
                message = (
                    "Focused verification modified tracked repository files. Its "
                    "side effects were discarded and cannot count as verification. "
                    "If this is a repository generator, invoke it with "
                    "RunCommand(capture_generated_changes=true), then run the test "
                    "separately."
                )
                if restore_error:
                    message += f" Worktree restoration failed: {restore_error}"
                file_context.verification["single_test"] = {
                    "passed": False,
                    "oracle_satisfied": False,
                    "exit_code": exit_code,
                    "expected_exit_code": args.expected_exit_code,
                    "required_output_regex": args.required_output_regex,
                    "summary": "verification command modified tracked files",
                    "command": args.command,
                    "change_seq": getattr(file_context, "change_seq", 0),
                    "probe_change_seq": current_probe_change_seq(file_context),
                    "execution_snapshot": execution_snapshot,
                }
                return Observation(
                    message=message,
                    summary="Focused verification writes were discarded.",
                    expect_correction=True,
                    properties={
                        "fail_reason": "verification_modified_tracked_files",
                        "exit_code": exit_code,
                        "expected_exit_code": args.expected_exit_code,
                        "required_output_regex": args.required_output_regex,
                        "oracle_satisfied": False,
                        "execution_snapshot": execution_snapshot,
                    },
                )
        
        test_summary = parse_test_output(output)
        passed = _compute_passed(
            exit_code,
            test_summary,
            expected_exit_code=args.expected_exit_code,
            output=output,
            required_output_regex=args.required_output_regex,
        )
        execution_result = (
            exit_code,
            passed,
            hashlib.sha256(output.encode("utf-8", errors="replace")).hexdigest(),
        )
        if (
            execution_key == self._last_execution_key
            and execution_result == self._last_execution_result
        ):
            self._identical_execution_repeats += 1
        else:
            self._identical_execution_repeats = 1
        self._last_execution_key = execution_key
        self._last_execution_result = execution_result
        safe_summary_line = sanitize_action_output(
            test_summary.summary_line,
            max_chars=2000,
            keep="head_tail",
        ).text
        safe_results = [
            result.model_copy(
                update={
                    "name": sanitize_action_output(
                        result.name,
                        max_chars=500,
                        keep="head_tail",
                    ).text,
                    "details": sanitize_action_output(
                        result.details,
                        max_chars=2000,
                        keep="head_tail",
                    ).text,
                }
            )
            for result in test_summary.results
        ]
        safe_test_summary = test_summary.model_copy(
            update={
                "summary_line": safe_summary_line,
                "raw_output": safe_output.text,
                "results": safe_results,
            }
        )

        repeated_failure_guidance = ""
        if test_summary.log_type == LogType.COMPILATION_ERROR and not passed:
            signature = _compilation_failure_signature(output)
            if signature and signature == self._last_compilation_failure:
                self._compilation_failure_repeats += 1
            else:
                self._last_compilation_failure = signature
                self._compilation_failure_repeats = 1 if signature else 0
            if self._compilation_failure_repeats >= 2:
                repeated_failure_guidance = (
                    "\n\nRepeated compilation failure: the same compiler error remains after "
                    f"{self._compilation_failure_repeats} focused verification attempts: "
                    f"{signature}. Compare the diagnostic with the current production diff, "
                    "correct the implementation within the assigned hypothesis "
                    "before repeating this verification."
                )
        else:
            self._last_compilation_failure = ""
            self._compilation_failure_repeats = 0

        message = f"Single test execution finished with exit code {exit_code}.\n"
        message += (
            f"Behavioral oracle: {'satisfied' if passed else 'not satisfied'} "
            f"(expected exit {args.expected_exit_code}"
        )
        if args.required_output_regex:
            output_matched = bool(
                re.search(args.required_output_regex, output, re.MULTILINE)
            )
            message += (
                ", required output pattern "
                + ("matched" if output_matched else "did not match")
            )
        message += ").\n"
        message += f"Log Type: {test_summary.log_type.value}\n"
        message += f"Summary: {safe_summary_line}\n\n"
        
        message += f"Output (sanitized and bounded):\n{safe_output.text}"
        message += repeated_failure_guidance

        # Record verification state on the file_context so VerifiedFinish can enforce the pipeline.
        try:
            file_context.verification["single_test"] = {
                "passed": passed,
                "oracle_satisfied": passed,
                "exit_code": exit_code,
                "expected_exit_code": args.expected_exit_code,
                "required_output_regex": args.required_output_regex,
                "summary": safe_summary_line,
                "command": args.command,
                "execution_snapshot": execution_snapshot,
                "change_seq": getattr(file_context, "change_seq", 0),
                "probe_change_seq": current_probe_change_seq(file_context),
            }
        except Exception:
            # Never fail the action due to bookkeeping
            pass

        return Observation(
            message=message,
            properties={
                "test_summary": safe_test_summary.model_dump(),
                "exit_code": exit_code,
                "expected_exit_code": args.expected_exit_code,
                "required_output_regex": args.required_output_regex,
                "oracle_satisfied": passed,
                "execution_snapshot": execution_snapshot,
                "output_metadata": safe_output.metadata,
                "compilation_failure_signature": self._last_compilation_failure,
                "compilation_failure_repeats": self._compilation_failure_repeats,
                "identical_execution_repeats": self._identical_execution_repeats,
            },
            summary=f"Single test completed: {safe_summary_line}",
        )

    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return [
            FewShotExample(
                user_input="Run a specific test command to debug a failure.",
                thoughts="I need to debug a specific test failure. I will run a targeted command to compile and execute only the relevant test file.",
                action=RunSingleTestArgs(
                    thoughts="Compile and run a specific test file to debug a failure.",
                    command="cd build && make my_specific_test && ./my_specific_test"
                ),
                observation=Observation(
                    message="Single test execution finished with exit code 1.\nLog Type: test_results\nSummary: 0 passed, 1 failed, 0 skipped.\n\nOutput:\n...",
                    summary="Single test completed: 0 passed, 1 failed, 0 skipped.",
                    properties={"exit_code": 1}
                )
            )
        ]
