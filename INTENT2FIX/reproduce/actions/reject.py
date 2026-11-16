from pydantic import Field
from INTENT2FIX.actions.action import Action
from INTENT2FIX.actions.schema import ActionArguments, Observation

class RejectArgs(ActionArguments):
    """Arguments for the Reject action."""
    reason: str = Field(..., description="A clear explanation of why the task cannot be completed.")

class Reject(Action):
    """
    An action to reject the task if it cannot be completed.
    For instance, if the issue description is insufficient to create a reproduction case.
    """
    args_schema: type[ActionArguments] = RejectArgs

    def execute(self, args: RejectArgs, **kwargs) -> Observation:
        """
        Sets the terminal flag and returns the rejection reason.
        """
        return Observation(
            message=f"Task rejected: {args.reason}",
            terminal=True
        )
