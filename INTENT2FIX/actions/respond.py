from pydantic import Field, ConfigDict

from INTENT2FIX.actions.action import Action
from INTENT2FIX.actions.schema import ActionArguments, Observation
from INTENT2FIX.file_context import FileContext
from INTENT2FIX.workspace import Workspace


class MessageArgs(ActionArguments):
    """Respond with a message to the user."""

    message: str = Field(..., description="The message to send to the user.")

    model_config = ConfigDict(title="SendMessage")

    def to_prompt(self):
        return f"Message: {self.message}"

    def equals(self, other: "ActionArguments") -> bool:
        return other.message == self.message


class MessageAction(Action):
    """
    Action to just send a response to the user.
    """

    args_schema = MessageArgs

    def execute(
        self,
        args: MessageArgs,
        file_context: FileContext | None = None,
        workspace: Workspace | None = None,
    ) -> Observation:
        return Observation()
