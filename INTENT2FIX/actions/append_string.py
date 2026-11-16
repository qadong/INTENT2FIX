import re
from typing import List

from pydantic import Field, ConfigDict

from INTENT2FIX.actions.action import Action, FewShotExample
from INTENT2FIX.actions.code_action_value_mixin import CodeActionValueMixin
from INTENT2FIX.actions.code_modification_mixin import CodeModificationMixin
from INTENT2FIX.actions.schema import ActionArguments, Observation
from INTENT2FIX.file_context import FileContext
from INTENT2FIX.index import CodeIndex
from INTENT2FIX.repository.file import do_diff
from INTENT2FIX.repository.repository import Repository
from INTENT2FIX.runtime.runtime import RuntimeEnvironment
from INTENT2FIX.workspace import Workspace
from INTENT2FIX.actions.verification_sync import mark_file_change


class AppendStringArgs(ActionArguments):
    """Append text strictly to the end of an existing file.

    Use StringReplace to modify or insert within existing content, and CreateFile
    when the target path does not exist.
    """

    model_config = ConfigDict(title="AppendString")

    path: str = Field(..., description="Repository-relative existing file path.")
    new_str: str = Field(..., description="Exact text to add at end of file.")

    def format_args_for_llm(self) -> str:
        return f"""<path>{self.path}</path>
<new_str>
{self.new_str}
</new_str>"""

    @classmethod
    def format_schema_for_llm(cls) -> str:
        return (
            "Append only to the end of an existing file. Use StringReplace for "
            "in-file insertion and CreateFile for a new path.\n\n"
            + cls.format_xml_schema(
                {"path": "file/path.py", "new_str": "\ncontent to append at end of file\n"}
            )
        )

    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Add a new helper function at the end of the utilities file",
                action=AppendStringArgs(
                    thoughts="Adding a new utility function for date formatting",
                    path="utils/formatters.py",
                    new_str="""

def format_timestamp(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()""",
                ),
            ),
        ]


class AppendString(Action, CodeActionValueMixin, CodeModificationMixin):
    """
    Action to append text content strictly to the end of a file.
    This action only adds content at the file's end and cannot modify existing content.
    """

    args_schema = AppendStringArgs

    def __init__(
        self,
        runtime: RuntimeEnvironment | None = None,
        code_index: CodeIndex | None = None,
        repository: Repository | None = None,
        **data,
    ):
        super().__init__(**data)
        # Initialize mixin attributes directly
        object.__setattr__(self, "_runtime", runtime)
        object.__setattr__(self, "_code_index", code_index)
        object.__setattr__(self, "_repository", repository)

    def execute(
        self,
        args: AppendStringArgs,
        file_context: FileContext | None = None,
        workspace: Workspace | None = None,
    ) -> Observation:
        path_str = self.normalize_path(args.path)
        path, error = self.validate_file_access(path_str, file_context)
        if error:
            return error

        immutable_paths = file_context.verification.get(
            "immutable_reproduction_paths", []
        )
        if str(path) in immutable_paths:
            return Observation(
                message=(
                    f"Edit rejected: {path} is immutable reproduction evidence. "
                    "Fix production code and use the existing PoC unchanged for verification."
                ),
                properties={"fail_reason": "immutable_reproduction_artifact"},
                expect_correction=True,
            )

        context_file = file_context.get_context_file(str(path))
        if not context_file:
            return Observation(
                message=f"Could not get context for file: {path}",
                properties={"fail_reason": "context_error"},
            )

        file_text = context_file.content.expandtabs()
        new_str = args.new_str.expandtabs()

        # Check if this looks like top-of-file content (imports, etc)
        looks_like_import = bool(re.match(r"^(import|from)\s+\w+", new_str.lstrip()))

        if looks_like_import:
            return Observation(
                message=(
                    "It looks like you're trying to add imports or other top-of-file content. "
                    "Please use StringReplace action to add content at the beginning of files."
                ),
                properties={"fail_reason": "wrong_action_for_imports"},
                expect_correction=True,
            )

        # Ensure there's a newline between old content and the new string
        if file_text and not file_text.endswith('\n'):
            new_file_text = file_text + '\n' + new_str
        else:
            new_file_text = file_text + new_str

        diff = do_diff(str(path), file_text, new_file_text)
        context_file.apply_changes(new_file_text)
        mark_file_change(file_context, str(path))

        message = (
            f"The file {path} has been edited. "
            "Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). "
            "Edit the file again if necessary."
        )

        observation = Observation(
            message=message,
            properties={"diff": diff, "success": True},
        )

        return observation
