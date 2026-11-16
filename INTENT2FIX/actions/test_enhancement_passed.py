from typing import ClassVar, List, Type
from pydantic import Field, ConfigDict
from INTENT2FIX.actions.action import Action
from INTENT2FIX.actions.schema import ActionArguments, Observation

class TestEnhancementPassedArgs(ActionArguments):
    """Indicate that the test enhancement task is complete and all tests pass."""
    
    thoughts: str = Field(..., description="Your reasoning about why the test enhancement is complete.")
    new_test_files: List[str] = Field(default_factory=list, description="List of new test files created or modified.")
    test_summary: str = Field(..., description="Summary of the adversarial and boundary test cases added.")
    test_results: str = Field(..., description="Summary of test results showing all tests passed.")

    model_config = ConfigDict(title="TestEnhancementPassed")

class TestEnhancementPassed(Action):
    args_schema: ClassVar[Type[ActionArguments]] = TestEnhancementPassedArgs

    def execute(self, args: TestEnhancementPassedArgs, file_context=None, workspace=None):
        message = "Test enhancement successful: All tests passed.\n"
        message += f"New tests: {', '.join(args.new_test_files)}\n"
        message += f"Summary: {args.test_summary}"
        return Observation(message=message, terminal=True)
