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


class TestEnhancementFinishArgs(ActionArguments):
    """Indicate that the test enhancement task is complete."""

    thoughts: str = Field(
        ...,
        description="Your reasoning about why the test enhancement task is complete.",
    )
    finish_reason: str = Field(..., description="Explain why the test enhancement is complete.")
    new_test_files: List[str] = Field(
        default_factory=list,
        description="List of new test files created or modified during test enhancement.",
    )
    test_summary: str = Field(
        ...,
        description="Summary of the test cases added, including what scenarios they cover (adversarial, boundary values, etc.).",
    )
    test_results: str = Field(
        ...,
        description="Results of running the tests: which tests passed, which failed (if any), and exit codes.",
    )
    failure_analysis: str = Field(
        default="",
        description=(
            "Detailed analysis of any UNRESOLVED test failures that indicate a bug in the project's source code (the patch). "
            "Leave this EMPTY if all tests pass. Do NOT include analysis of fixed 'Test Authoring Errors' or transient issues."
        ),
    )

    model_config = ConfigDict(title="TestEnhancementFinish")

    def to_prompt(self):
        return (
            f"Finish test enhancement with reason: {self.finish_reason}\n"
            f"New test files: {', '.join(self.new_test_files)}\n"
            f"Test summary: {self.test_summary}\n"
            f"Test results: {self.test_results}\n"
            + (f"Failure analysis: {self.failure_analysis}\n" if self.failure_analysis else "")
        )

    def equals(self, other: "ActionArguments") -> bool:
        return isinstance(other, TestEnhancementFinishArgs)


class TestEnhancementFinish(Action):
    args_schema: ClassVar[Type[ActionArguments]] = TestEnhancementFinishArgs

    def execute(
        self,
        args: TestEnhancementFinishArgs,
        file_context: FileContext | None = None,
        workspace: Workspace | None = None,
    ):
        # TestEnhancementFinish is always terminal
        # It reports the results of test enhancement, including any failures
        message = f"Test enhancement complete: {args.finish_reason}\n\n"
        message += f"New test files: {', '.join(args.new_test_files) if args.new_test_files else 'None'}\n"
        message += f"Test summary: {args.test_summary}\n"
        message += f"Test results: {args.test_results}\n"
        
        if args.failure_analysis:
            message += f"\nFailure analysis:\n{args.failure_analysis}"
        
        return Observation(message=message, terminal=True)

    @classmethod
    def get_evaluation_criteria(cls, trajectory_length: int) -> List[str]:
        return [
            "**Test Enhancement Quality:** Evaluate the quality and comprehensiveness of the generated test cases.",
            "**Adversarial Test Coverage:** Assess whether adversarial test cases were generated to find weaknesses in the patch.",
            "**Boundary Value Coverage:** Evaluate whether boundary value test cases were generated (extreme values, edge cases, invalid inputs, etc.).",
            "**Test Execution:** Verify that all generated tests were executed and results were properly reported.",
            "**Failure Analysis:** If tests failed, evaluate the quality of the failure analysis provided.",
            "**Test File Organization:** Assess whether test files were properly organized and follow project conventions.",
        ]

    @classmethod
    def get_reward_scale(cls, trajectory_length) -> List[RewardScaleEntry]:
        return cls.generate_reward_scale_entries(
            [
                (
                    90,
                    100,
                    "Comprehensive test enhancement with excellent adversarial and boundary value coverage. All tests pass and failure analysis (if any) is thorough.",
                ),
                (
                    75,
                    89,
                    "Good test enhancement with adequate adversarial and boundary value coverage. Most tests pass.",
                ),
                (
                    50,
                    74,
                    "Adequate test enhancement but missing some adversarial or boundary value test cases. Some tests may have failed.",
                ),
                (
                    25,
                    49,
                    "Minimal test enhancement with limited coverage. Many important test cases are missing.",
                ),
                (
                    0,
                    24,
                    "Poor test enhancement with very limited or no meaningful test cases generated.",
                ),
            ]
        )