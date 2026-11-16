import logging
import re
from typing import List, Literal

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
from INTENT2FIX.actions.verification_sync import (
    mark_file_change,
)
from INTENT2FIX.patch_review.sanitizer import needs_production_classification

logger = logging.getLogger(__name__)


class CreateFileArgs(ActionArguments):
    """Create a new repository file with complete content.

    The path must be repository-relative after an optional /repo/ prefix is
    removed and must not contain ``..``. Existing repository and production
    files cannot be overwritten. Reusing the path of a registered temporary
    probe replaces that probe's complete content. Use artifact_kind=
    temporary_probe for a disposable compile or diagnostic probe so it is
    excluded from review and submission; use production only for an intended
    repository deliverable. To correct an old temporary classification, use
    production with the same path and unchanged content; then reverify.
    """

    path: str = Field(
        ...,
        description=(
            "Repository-relative file path. It may name an existing registered "
            "temporary probe when replacing that probe."
        ),
    )
    file_text: str = Field(..., description="Complete replacement file content.")
    artifact_kind: Literal["production", "temporary_probe"] | None = Field(
        default=None,
        description=(
            "Required for execution: explicitly select production for a repository "
            "deliverable or temporary_probe for a disposable diagnostic."
        ),
    )

    model_config = ConfigDict(title="CreateFile")

    @classmethod
    def model_json_schema(cls, **kwargs):
        # Older saved action arguments remain readable; new tool calls must
        # supply a classification, and execute also rejects omitted/null kinds.
        schema = super().model_json_schema(**kwargs)
        schema.setdefault("required", []).append("artifact_kind")
        schema["properties"]["artifact_kind"] = {
            "type": "string", "enum": ["production", "temporary_probe"],
            "description": cls.model_fields["artifact_kind"].description,
        }
        return schema

    def format_args_for_llm(self) -> str:
        return f"""<path>{self.path}</path>
<artifact_kind>{self.artifact_kind}</artifact_kind>
<file_text>
{self.file_text}
</file_text>"""

    @classmethod
    def format_schema_for_llm(cls) -> str:
        return cls.format_xml_schema({
            "path": "file/path.py",
            "artifact_kind": "production",
            "file_text": "\ncomplete file content\n",
        })


class CreateFile(Action, CodeActionValueMixin, CodeModificationMixin):
    """
    Action to create a new file with specified content.
    """

    args_schema = CreateFileArgs

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
        args: CreateFileArgs,
        file_context: FileContext | None = None,
        workspace: Workspace | None = None,
    ) -> Observation:
        path, path_error = self.validate_repository_path(args.path)
        if path_error:
            return path_error

        if args.artifact_kind is None or (
            args.artifact_kind == "temporary_probe"
            and needs_production_classification(str(path))
            and (
                not file_context.file_exists(str(path))
                or str(path) in file_context.verification.get("ephemeral_paths", [])
            )
        ):
            return Observation(
                message=(
                    f"Choose the file classification explicitly for {path}. "
                    "New production-directory deliverables require artifact_kind=production. "
                    "Place disposable diagnostics in a probe/test path and use "
                    "temporary_probe. No file was changed."
                ),
                expect_correction=True,
                properties={"fail_reason": "artifact_classification_required", "success": False},
            )

        if (
            args.artifact_kind == "temporary_probe"
            and re.search(
                r"INTENT2FIX_DISCRIMINATOR_H[0-9]+_PASS",
                args.file_text,
            )
        ):
            marker = re.compile(
                r"INTENT2FIX_DISCRIMINATOR_H[0-9]+_PASS"
            )
            for raw_path in file_context.verification.get(
                "ephemeral_paths", []
            ):
                existing_path = str(raw_path)
                if existing_path == str(path):
                    continue
                existing_file = file_context.get_file(existing_path)
                if (
                    existing_file is not None
                    and marker.search(existing_file.content)
                ):
                    return Observation(
                        message=(
                            "A portfolio branch discriminator already exists "
                            f"at {existing_path}; edit and rerun "
                            f"{existing_path} instead of creating another "
                            "discriminator."
                        ),
                        expect_correction=True,
                        properties={
                            "fail_reason": (
                                "duplicate_portfolio_discriminator"
                            ),
                            "success": False,
                        },
                    )

        if file_context.file_exists(str(path)):
            context_file = file_context.get_file(str(path))
            ephemeral_paths = set(
                file_context.verification.get("ephemeral_paths", [])
            )
            if (
                args.artifact_kind == "production"
                and str(path) in ephemeral_paths
                and context_file is not None
                and context_file.is_new
                and context_file.content.rstrip("\n") == args.file_text.rstrip("\n")
            ):
                file_context.verification["ephemeral_paths"] = sorted(
                    ephemeral_paths - {str(path)}
                )
                file_context.verification.setdefault("production_paths", []).append(str(path))
                mark_file_change(file_context, str(path))
                return Observation(
                    message=f"Reclassified {path} as production; content unchanged. Reverify the submitted patch.",
                    properties={"success": True, "reclassified": True},
                )
            can_replace_probe = (
                args.artifact_kind == "temporary_probe"
                and str(path) in ephemeral_paths
                and context_file is not None
                and context_file.is_new
            )
            if not can_replace_probe:
                return Observation(
                    message=(
                        f"File already exists at: {path}. Only a registered "
                        "temporary probe can be replaced using CreateFile."
                    ),
                    properties={"fail_reason": "file_exists", "success": False},
                )

            old_content = context_file.content
            context_file.apply_changes(args.file_text)
            mark_file_change(file_context, str(path))
            return Observation(
                message=(
                    "Temporary probe replaced successfully at "
                    f"repository-relative path: {path}."
                ),
                properties={
                    "diff": do_diff(str(path), old_content, args.file_text),
                    "success": True,
                    "replaced": True,
                },
            )

        context_file = file_context.add_file(str(path), show_all_spans=True)
        context_file.apply_changes(args.file_text)
        if args.artifact_kind == "temporary_probe":
            ephemeral_paths = file_context.verification.setdefault(
                "ephemeral_paths", []
            )
            if str(path) not in ephemeral_paths:
                ephemeral_paths.append(str(path))
        else:
            file_context.verification.setdefault("production_paths", []).append(str(path))
        mark_file_change(file_context, str(path))

        diff = do_diff(str(path), "", args.file_text)

        observation = Observation(
            message=(
                f"File created successfully at repository-relative path: {path}. "
                f"Reference this exact path in test commands; do not rewrite it as /{path}."
            ),
            properties={"diff": diff, "success": True},
        )

        # test_summary = self.run_tests(
        #     file_path=str(path),
        #     file_context=file_context,
        # )

        # if test_summary:
        #     observation.message += f"\n\n{test_summary}"

        return observation

    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Create a new Python file for handling user authentication",
                action=CreateFileArgs(
                    thoughts="Creating a new authentication module with basic user authentication functionality",
                    path="auth/user_auth.py",
                    artifact_kind="production",
                    file_text="""import logging
from typing import Optional

logger = logging.getLogger(__name__)

class UserAuth:
    def __init__(self):
        self._users = {}

    def authenticate(self, username: str, password: str) -> bool:
        if username not in self._users:
            logger.warning(f"Authentication failed: User {username} not found")
            return False

        return self._users[username] == password

    def register(self, username: str, password: str) -> bool:
        if username in self._users:
            logger.error(f"Registration failed: User {username} already exists")
            return False

        self._users[username] = password
        logger.info(f"User {username} registered successfully")
        return True""",
                ),
            )
        ]
