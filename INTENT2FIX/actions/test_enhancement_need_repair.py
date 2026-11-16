from typing import ClassVar, List, Type
from pydantic import Field, ConfigDict
from INTENT2FIX.actions.action import Action
from INTENT2FIX.actions.schema import ActionArguments, Observation

class TestEnhancementNeedRepairArgs(ActionArguments):
    """Indicate that the test enhancement found a bug in the patch that needs fixing."""
    
    thoughts: str = Field(..., description="Your reasoning about why the patch needs repair.")
    failure_analysis: str = Field(..., description="Detailed analysis of the failures indicating a bug in the source code.")
    new_test_files: List[str] = Field(default_factory=list, description="The test files that contain the failing regression guards.")
    test_results: str = Field(..., description="Details of the failing test runs.")

    model_config = ConfigDict(title="TestEnhancementNeedRepair")

class TestEnhancementNeedRepair(Action):
    args_schema: ClassVar[Type[ActionArguments]] = TestEnhancementNeedRepairArgs

    def execute(self, args: TestEnhancementNeedRepairArgs, file_context=None, workspace=None):
        message = "Test enhancement found bugs: Repair required.\n"
        message += f"Failure Analysis: {args.failure_analysis}\n"
        message += f"Failing Tests: {', '.join(args.new_test_files)}"
        return Observation(message=message, terminal=True)
