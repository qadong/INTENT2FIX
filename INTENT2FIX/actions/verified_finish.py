from typing import ClassVar, List, Type

from pydantic import Field, ConfigDict

from INTENT2FIX.actions.action import Action
from INTENT2FIX.actions.schema import (
    ActionArguments,
    Observation,
    RewardScaleEntry,
    FewShotExample,
)
from INTENT2FIX.file_context import FileContext
from INTENT2FIX.workspace import Workspace
from INTENT2FIX.actions.verification_sync import current_probe_change_seq
from INTENT2FIX.patch_review.sanitizer import sanitize_candidate_patch


def _missing_intent_checks(verification: dict, change_seq: int) -> List[str]:
    contract = verification.get("intent_contract") or {}
    results = verification.get("intent_check_results") or {}

    def passed(check_id: str) -> bool:
        result = results.get(check_id) or {}
        return bool(result.get("passed")) and result.get("change_seq") == change_seq

    missing: List[str] = []
    acceptance_ids = contract.get("acceptance_ids") or []
    checks = contract.get("checks") or {}

    for check_id in acceptance_ids:
        if not passed(check_id):
            missing.append(
                f"{check_id}: {checks.get(check_id, 'intent check')}"
            )
    return missing


class VerifiedFinishArgs(ActionArguments):
    """Conclude the repair and report available verification evidence.

    This does not run tests. Follow the Repair Policy for the current mode and
    report failures and missing evidence honestly. Portfolio mode permits failed
    checks; submission artifact classification errors still require correction.
    Finishing does not certify that the patch passes official tests.
    """

    thoughts: str = Field(
        ...,
        description="Concise reasoning that the original issue and repair intent contract are satisfied.",
    )
    finish_reason: str = Field(
        ...,
        description="Explain how the final production-code change fully addresses the original issue and required repair clauses.",
    )
    test_verification: str = Field(
        ...,
        description="List attempted RunSingleTest, RunRegression, and applicable RunIntentCheck results, including failures or limitations. Do not claim test-file changes that were not made.",
    )

    model_config = ConfigDict(title="Finish")

    def to_prompt(self):
        return f"Finish with reason: {self.finish_reason}\n" f"Test verification: {self.test_verification}\n"

    def equals(self, other: "ActionArguments") -> bool:
        return isinstance(other, VerifiedFinishArgs)


class VerifiedFinish(Action):
    args_schema: ClassVar[Type[ActionArguments]] = VerifiedFinishArgs

    def execute(
        self,
        args: VerifiedFinishArgs,
        file_context: FileContext | None = None,
        workspace: Workspace | None = None,
    ):
        warnings: List[str] = []
        if file_context is not None and file_context.has_patch(ignore_tests=False):
            candidate = sanitize_candidate_patch(file_context)
            if candidate.artifact_errors:
                return Observation(
                    message="Submission file classification must be corrected before Finish:\n"
                    + "\n".join(candidate.artifact_errors),
                    terminal=False,
                    expect_correction=True,
                    properties={"fail_reason": "submission_artifact_error",
                                "artifact_errors": list(candidate.artifact_errors)},
                )
            change_seq = getattr(file_context, "change_seq", 0)
            verification = getattr(file_context, "verification", {}) or {}

            reg = verification.get("regression") or {}
            single = verification.get("single_test") or {}
            probe_change_seq = current_probe_change_seq(file_context)

            reg_ok = bool(reg.get("passed")) and reg.get("change_seq") == change_seq
            single_ok = (
                bool(single.get("passed"))
                and single.get("change_seq") == change_seq
                and single.get("probe_change_seq", probe_change_seq)
                == probe_change_seq
            )
            missing_intent = _missing_intent_checks(verification, change_seq)

            if not reg_ok:
                warnings.append("RunRegression did not pass after the latest code change")
            if not single_ok:
                warnings.append("RunSingleTest did not pass after the latest code change")
            if file_context.has_test_patch():
                production_regression = (
                    verification.get("production_regression") or {}
                )
                production_current = (
                    production_regression.get("change_seq") == change_seq
                )
                if not production_current:
                    warnings.append(
                        "Production-only regression was not run after the latest "
                        "test-file edit"
                    )
                elif not production_regression.get("passed"):
                    warnings.append(
                        "Production-only regression with the original tests did "
                        "not pass; determine whether the old expectation is "
                        "intentionally obsolete before claiming verification"
                    )
            if missing_intent:
                if verification.get("finish_policy") == "portfolio_soft_evidence":
                    warnings.append(
                        "Intent acceptance evidence is missing, stale, or failed: "
                        + "; ".join(missing_intent)
                    )
                else:
                    return Observation(
                        message=(
                            "Required intent acceptance checks are missing, stale, "
                            "or failed after the latest production-code change:\n- "
                            + "\n- ".join(missing_intent)
                            + "\nRun every required ACCEPT-N check successfully "
                            "before finishing."
                        ),
                        terminal=False,
                        expect_correction=True,
                        properties={
                            "fail_reason": "required_intent_checks_incomplete",
                            "missing_intent_checks": missing_intent,
                        },
                    )

        message = args.finish_reason
        properties = {}
        if warnings:
            message += "\n\nNon-blocking verification warnings:\n- " + "\n- ".join(warnings)
            properties = {"verification_warnings": warnings}

        return Observation(message=message, terminal=True, properties=properties)

    @classmethod
    def get_evaluation_criteria(cls, trajectory_length: int) -> List[str]:
        return [
            "**Full Trajectory Review:** Evaluate the complete sequence of actions taken by the agent leading to this finish action. Assess whether the trajectory represents an efficient and logical path to the solution.",
            "**Solution Correctness and Quality:** Verify that all changes made throughout the trajectory logically address the problem statement. Ensure the changes fit contextually within the existing codebase without introducing new issues.",
            "**Verification Evidence:**",
            " * **Current State:** Evaluate the latest RunSingleTest, RunRegression, and applicable RunIntentCheck outcomes without treating a missing or failed check as an automatic rejection.",
            " * **Behavior Coverage:** Evaluate whether the focused checks cover the requested behavior and relevant preservation constraints.",
            " * **Evidence Accuracy:** Check that test_verification accurately reports the checks actually run and their outcomes.",
            "**Assessment of Complete Trajectory:** Evaluate if the sequence of actions taken represents the most optimal path to the solution.",
            "**Verification of Task Completion:** Confirm that all aspects of the original issue have been addressed and properly tested.",
        ]

    @classmethod
    def get_reward_scale(cls, trajectory_length) -> List[RewardScaleEntry]:
        return cls.generate_reward_scale_entries(
            [
                (
                    90,
                    100,
                    "The trajectory fully resolves the original issue with a minimal complete change and strong focused, regression, and intent-check evidence for the final code state.",
                ),
                (
                    75,
                    89,
                    "The trajectory resolves the issue with persuasive final-state evidence and only minor gaps in efficiency or verification explanation.",
                ),
                (
                    50,
                    74,
                    "The trajectory resolves the core issue but the verification evidence has gaps or does not clearly cover all requested behavior.",
                ),
                (
                    25,
                    49,
                    "The trajectory partially resolves the issue, but verification is incomplete or does not verify the final code state.",
                ),
                (
                    0,
                    24,
                    "The trajectory shows some progress but fails to resolve the issue or provides ineffective verification.",
                ),
                (
                    -49,
                    -1,
                    "The trajectory has major gaps in implementation or required verification.",
                ),
                (
                    -100,
                    -50,
                    "The trajectory is incorrect and lacks required verification; the finish action is premature.",
                ),
            ]
        )

    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Add input validation to the process_order function",
                action=VerifiedFinishArgs(
                    thoughts="The validation requirement is implemented and the final code state passed focused and regression verification",
                    finish_reason="Added robust input validation to process_order function with proper error handling",
                    test_verification=(
                        "RunSingleTest passed for valid, missing-field, and invalid-value inputs after the final edit.\n"
                        "RunRegression passed for the same code state.\n"
                        "All required ACCEPT checks passed."
                    ),
                ),
            ),
            FewShotExample.create(
                user_input="Fix the bug in the date parsing logic",
                action=VerifiedFinishArgs(
                    thoughts="The date parsing contract is implemented and the available final-state checks support the repair",
                    finish_reason="Fixed date parsing bug that was incorrectly handling timezone conversions",
                    test_verification=(
                        "RunSingleTest passed for timezone conversion, DST boundaries, and invalid formats.\n"
                        "RunRegression passed after the final edit.\n"
                        "All required ACCEPT checks passed for that same state."
                    ),
                ),
            ),
        ]
