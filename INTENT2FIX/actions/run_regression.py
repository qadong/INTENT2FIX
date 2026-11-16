import logging
from pydantic import Field, PrivateAttr, ConfigDict
from typing import List
from INTENT2FIX.actions.action import Action
from INTENT2FIX.actions.schema import ActionArguments, Observation
from INTENT2FIX.actions.schema import ActionArguments, Observation, FewShotExample
from INTENT2FIX.file_context import FileContext
from INTENT2FIX.patch_review.sanitizer import sanitize_candidate_patch
from INTENT2FIX.runtime.dockerenv import DockerEnv
from INTENT2FIX.utils.patch import production_patch_sha256 as patch_sha256
from .test_parser import parse_test_output, LogType
from .verification_sync import (
    PRODUCTION_ISOLATION,
    run_isolated_production_command,
    sync_verification_files,
)

logger = logging.getLogger(__name__)


def _bounded_output(output: str, max_chars: int = 2000) -> str:
    if len(output) <= max_chars:
        return output
    half = max_chars // 2
    return (
        output[:half]
        + f"\n\n... [Output truncated to {max_chars} characters, "
        f"total {len(output)}] ...\n\n"
        + output[-half:]
    )

def _compute_passed(exit_code: int, test_summary) -> bool:
    """
    Positive regression requires both a successful process and consistent results.
    """
    if exit_code != 0:
        return False
    try:
        log_type = getattr(test_summary, "log_type", None)
        if log_type in {LogType.COMPILATION_ERROR, LogType.MEMORY_LEAK, LogType.NO_TESTS, LogType.OUTPUT_TRUNCATED}:
            return False
        if log_type == LogType.TEST_RESULTS:
            results = getattr(test_summary, "results", []) or []
            failed = sum(1 for r in results if getattr(r, "status", None) == "failed")
            return failed == 0
    except Exception:
        pass
    return exit_code == 0


def _failed_test_names(test_summary) -> list[str]:
    return [
        str(result.name)
        for result in (getattr(test_summary, "results", []) or [])
        if getattr(getattr(result, "status", None), "value", None)
        == "failed"
    ]


class RunRegressionArgs(ActionArguments):
    """Run the configured regression suite and report production-patch verification.

    Production verification uses original tests without temporary probes or excluded
    edits. If repository tests were edited, their working-tree results are reported
    separately. This can rebuild the project and is more expensive than
    RunSingleTest; unchanged production results may be reused.
    Run it after the final production edit or for a suspected regression. It does
    not run official hidden tests or establish complete Issue coverage.
    """

    thoughts: str = Field(..., description="Why a full regression run is useful now.")

    model_config = ConfigDict(title="RunRegression")

    @property
    def log_name(self):
        return "RunRegression"


class RunRegression(Action):
    args_schema = RunRegressionArgs

    _runtime: DockerEnv = PrivateAttr()

    def __init__(
        self,
        runtime: DockerEnv | None = None,
        **data,
    ):
        super().__init__(**data)
        self._runtime = runtime

    def execute(
        self,
        args: RunRegressionArgs,
        file_context: FileContext,
        **kwargs,
    ) -> Observation:
        """
        Run regression tests in the Docker environment after applying a patch.
        """
        if not self._runtime:
            raise ValueError("Runtime environment must be provided to execute the run regression action.")

        current_change_seq = getattr(file_context, "change_seq", 0)
        candidate = sanitize_candidate_patch(file_context)
        production_patch_sha256 = patch_sha256(candidate.patch)
        cached_result = file_context.verification.get("regression") or {}
        if (
            cached_result.get("change_seq") == current_change_seq
            and cached_result.get("verification_view") == "production"
            and cached_result.get("production_patch_sha256") == production_patch_sha256
            and cached_result.get("verification_isolation") == PRODUCTION_ISOLATION
            and "Illegal option -o pipefail" not in str(
                cached_result.get("output") or ""
            )
        ):
            production_result = file_context.verification.get(
                "production_regression"
            )
            summary = str(cached_result.get("summary") or "result available")
            return Observation(
                message=(
                    "Reused the existing regression result because the "
                    "production/test revision is unchanged. Temporary probe "
                    "changes do not invalidate the repository regression.\n"
                    f"Summary: {summary}"
                ),
                properties={
                    "exit_code": cached_result.get("exit_code"),
                    "production_regression": production_result,
                    "reused": True,
                },
                summary=f"Regression result reused: {summary}",
            )
        cached_production = file_context.verification.get(
            "production_regression"
        ) or {}
        reusable_production = next(
            (
                record
                for record in (cached_production, cached_result)
                if record.get("production_patch_sha256")
                == production_patch_sha256
                and record.get("verification_view") == "production"
                and record.get("verification_isolation") == PRODUCTION_ISOLATION
                and "Illegal option -o pipefail" not in str(record.get("output") or "")
            ),
            None,
        )
        if reusable_production is not None:
            has_test_edits = any(
                reason == "test_or_fixture"
                for reason in candidate.excluded_paths.values()
            )
            production_result = {
                **reusable_production,
                "change_seq": current_change_seq,
                "test_view": (
                    "original_tests_before_test_edits"
                    if has_test_edits
                    else "unchanged_tests"
                ),
                "test_view_stale": has_test_edits,
            }
            file_context.verification["production_regression"] = (
                production_result
            )
            summary = str(
                reusable_production.get("summary") or "result available"
            )
            return Observation(
                message=(
                    "Reused the existing production regression because the "
                    "sanitized production patch is unchanged. Only excluded "
                    "test/verification files changed, so the altered "
                    + (
                        "test view was not rerun as a full suite. Use "
                        "RunSingleTest to validate the changed test "
                        "expectation, then report both that focused result "
                        "and this original-test regression as soft evidence.\n"
                        if has_test_edits
                        else
                        "verification view did not invalidate the unchanged "
                        "test suite.\n"
                    )
                    +
                    f"Original-test summary: {summary}"
                ),
                properties={
                    "exit_code": reusable_production.get("exit_code"),
                    "production_regression": production_result,
                    "reused": True,
                    "test_view_stale": has_test_edits,
                },
                summary=f"Production regression result reused: {summary}",
            )

        has_test_edits = any(
            reason == "test_or_fixture"
            for reason in candidate.excluded_paths.values()
        )
        sync_failure = sync_verification_files(
            file_context, self._runtime,
            included_paths=None if has_test_edits else set(candidate.kept_paths),
        )
        if sync_failure:
            return Observation(message=sync_failure, summary="File synchronization failed.")

        test_command = r'''
            if [ -d "build" ]; then
                find build -mindepth 1 -delete
            fi

            if [ -f "/home/build_run.sh" ]; then
                chmod 755 /home/build_run.sh && \
                sed -i "s|ctest|ctest --output-on-failure|g" /home/build_run.sh && \
                sed -i "s|\bmake\b|make -j4|g" /home/build_run.sh && \
                bash /home/build_run.sh
            elif [ -f "/home/run.sh" ]; then
                chmod 755 /home/run.sh && \
                sed -i "s|ctest|ctest --output-on-failure|g" /home/run.sh && \
                sed -i "s|\bmake\b|make -j4|g" /home/run.sh && \
                bash /home/run.sh
            else
                echo "Neither build_run.sh nor run.sh found in /home"
                exit 1
            fi
        '''

        if has_test_edits:
            output, exit_code = self._runtime.run_bash(test_command)
        else:
            output, exit_code = run_isolated_production_command(
                self._runtime, test_command, candidate.kept_paths, file_context
            )
        
        logger.info(f"Raw test output:\n---\n{output}\n---")
        
        test_summary = parse_test_output(output)
        passed = _compute_passed(exit_code, test_summary)

        message = f"Regression test execution finished with exit code {exit_code}.\n"
        message += f"Log Type: {test_summary.log_type.value}\n"
        message += f"Summary: {test_summary.summary_line}\n\n"
        
        # Keep the actionable compiler/test tail without persisting massive logs.
        MAX_CHARS = 2000
        bounded_output = _bounded_output(output, MAX_CHARS)
        output_label = "Output (truncated)" if bounded_output != output else "Output"
        message += f"{output_label}:\n{bounded_output}"

        production_result = None
        if has_test_edits:
            production_sync_failure = sync_verification_files(
                file_context,
                self._runtime,
                included_paths=set(candidate.kept_paths),
            )
            if production_sync_failure:
                production_result = {
                    "passed": False,
                    "exit_code": None,
                    "log_type": LogType.UNKNOWN.value,
                    "summary": production_sync_failure,
                    "output": production_sync_failure,
                    "production_patch_sha256": production_patch_sha256,
                    "change_seq": getattr(file_context, "change_seq", 0),
                    "verification_view": "unavailable",
                }
                message += (
                    "\n\nProduction-only regression could not start: "
                    + production_sync_failure
                )
            else:
                production_output, production_exit_code = run_isolated_production_command(
                    self._runtime, test_command, candidate.kept_paths, file_context
                )
                logger.info(
                    "Raw production-only regression output:\n---\n%s\n---",
                    production_output,
                )
                production_summary = parse_test_output(production_output)
                production_passed = _compute_passed(
                    production_exit_code,
                    production_summary,
                )
                production_result = {
                    "passed": production_passed,
                    "exit_code": production_exit_code,
                    "log_type": production_summary.log_type.value,
                    "summary": production_summary.summary_line,
                    "output": _bounded_output(production_output, MAX_CHARS),
                    "failed_test_names": _failed_test_names(
                        production_summary
                    ),
                    "production_patch_sha256": production_patch_sha256,
                    "change_seq": getattr(file_context, "change_seq", 0),
                    "verification_view": "production",
                    "verification_isolation": PRODUCTION_ISOLATION,
                    "verified_paths": sorted(candidate.kept_paths),
                }
                production_display = _bounded_output(
                    production_output, MAX_CHARS
                )
                message += (
                    "\n\nProduction-only regression (sanitized production patch "
                    "with original tests):\n"
                    f"Exit code: {production_exit_code}\n"
                    f"Summary: {production_summary.summary_line}\n"
                    f"Output:\n{production_display}"
                )

        # Record verification state on the file_context so VerifiedFinish can enforce the pipeline.
        try:
            file_context.verification["regression"] = {
                "passed": passed,
                "exit_code": exit_code,
                "log_type": test_summary.log_type.value,
                "summary": test_summary.summary_line,
                "output": bounded_output,
                "failed_test_names": _failed_test_names(test_summary),
                "production_patch_sha256": production_patch_sha256,
                "change_seq": getattr(file_context, "change_seq", 0),
                "verification_view": "working_tree" if has_test_edits else "production",
                "verification_isolation": None if has_test_edits else PRODUCTION_ISOLATION,
                "verified_paths": sorted(
                    file_context.get_patched_files() if has_test_edits else candidate.kept_paths
                ),
            }
            if not has_test_edits:
                production_result = dict(file_context.verification["regression"])
            if production_result is not None:
                file_context.verification["production_regression"] = (
                    production_result
                )
        except Exception:
            # Never fail the action due to bookkeeping
            pass

        return Observation(
            message=message,
            properties={
                "test_summary": test_summary.model_dump(),
                "exit_code": exit_code,
                "production_regression": production_result,
            },
            summary=f"Regression tests completed: {test_summary.summary_line}",
        )

    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return [
            FewShotExample(
                user_input="Run regression tests after refactoring a function.",
                thoughts="I have just refactored the `calculate_total` function to improve readability. Now I need to run the full regression test suite to ensure that my changes haven't introduced any unexpected bugs or broken existing functionality.",
                action=RunRegressionArgs(
                    thoughts="Run regression tests to verify the refactoring of `calculate_total`."
                ),
                observation=Observation(
                    message="Regression test execution finished with exit code 0.\nLog Type: test_results\nSummary: 15 passed, 0 failed, 0 skipped.\n\nOutput:\n...",
                    summary="Regression tests completed: 15 passed, 0 failed, 0 skipped.",
                    properties={"exit_code": 0}
                )
            )
        ]
