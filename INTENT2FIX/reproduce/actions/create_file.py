from pydantic import ConfigDict, Field, field_validator
import os
import logging
import json
from typing import Any, Type

from INTENT2FIX.actions.action import Action
from INTENT2FIX.actions.schema import ActionArguments, Observation
from INTENT2FIX.file_context import FileContext
from INTENT2FIX.repository.repository import Repository
from INTENT2FIX.runtime.dockerenv import DockerEnv
from INTENT2FIX.reproduce.state import (
    ReproductionSession,
    is_dedicated_reproduction_path,
    normalize_artifact_path,
)

logger = logging.getLogger(__name__)


class CreateFileArgs(ActionArguments):
    """Arguments for the CreateFile action."""

    path: str = Field(
        ...,
        description=(
            "A new repository-relative C/C++ reproduction file whose basename starts "
            "with 'repro', for example repro_bug.cpp or tests/repro_bug.cpp."
        ),
    )
    file_text: str = Field(..., description="The content to write into the file.")

    model_config = ConfigDict(title="CreateFile")

    @field_validator("path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        normalized = normalize_artifact_path(value)
        if normalized is None:
            raise ValueError("path must be a normalized repository-relative path")
        return normalized

    @classmethod
    def model_validate_json(
        cls: Type["CreateFileArgs"],
        json_data: str | bytes | bytearray,
        *,
        strict: bool | None = None,
        context: dict[str, Any] | None = None,
    ) -> "CreateFileArgs":
        """
        Cleans markdown fences from the JSON string before validation.
        """
        if isinstance(json_data, str):
            # Find the start of the JSON content
            start_brace = json_data.find("{")
            start_bracket = json_data.find("[")

            if start_brace == -1 and start_bracket == -1:
                # Let the parent validator handle the error
                pass
            else:
                if start_brace == -1:
                    start_pos = start_bracket
                elif start_bracket == -1:
                    start_pos = start_brace
                else:
                    start_pos = min(start_brace, start_bracket)

                # Find the end of the JSON content
                end_brace = json_data.rfind("}")
                end_bracket = json_data.rfind("]")

                if end_brace != -1 or end_bracket != -1:
                    end_pos = max(end_brace, end_bracket)
                    json_data = json_data[start_pos : end_pos + 1]

        # Call the parent's validation method
        return super(CreateFileArgs, cls).model_validate_json(
            json_data, strict=strict, context=context
        )


class CreateFile(Action):
    """
    Creates a dedicated reproduction artifact without modifying repository files.
    """

    args_schema: type[ActionArguments] = CreateFileArgs
    repository: Repository
    docker_env: DockerEnv
    session: ReproductionSession | None = None

    def execute(
        self, args: CreateFileArgs, file_context: FileContext, **kwargs
    ) -> Observation:
        """
        Writes the file content to a local temporary path and then copies it
        into the specified location within the Docker container.
        """
        registered_artifacts = self.session.created_artifacts if self.session else {}
        if self.session:
            rejection_reason = self.session.convergence_rejection("CreateFile")
            if rejection_reason:
                return Observation(
                    message=rejection_reason,
                    expect_correction=True,
                    properties={"rejected_path": args.path},
                )
        if not is_dedicated_reproduction_path(args.path):
            return Observation(
                message=(
                    "CreateFile requires a dedicated C/C++ reproduction file whose "
                    "basename starts with 'repro' (for example repro_bug.cpp)."
                ),
                expect_correction=True,
                properties={"rejected_path": args.path},
            )
        if args.path not in registered_artifacts and self.repository.file_exists(args.path):
            return Observation(
                message=(
                    f"CreateFile cannot overwrite existing repository file '{args.path}'. "
                    "Choose a new dedicated repro*.cpp path."
                ),
                expect_correction=True,
                properties={"rejected_path": args.path},
            )

        # The local repository path acts as a staging area.
        local_file_path = os.path.join(self.repository.repo_path, args.path)

        try:
            # Ensure the local directory exists before writing.
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            # Write the file to the local staging path first.
            with open(local_file_path, "w") as f:
                f.write(args.file_text)
            logger.info(f"Staged file '{args.path}' locally at '{local_file_path}'.")

            # Determine the destination path inside the container.
            container_path = os.path.join(self.docker_env.work_dir, args.path)

            # Use the docker_env to copy the file into the container.
            if self.docker_env.copy_anything_to_container(
                local_file_path, container_path
            ):
                if file_context:
                    file_context.add_file(args.path)
                if self.session:
                    self.session.register_artifact(args.path, args.file_text)
                    if self.session.candidate_run_ids:
                        self.session.note_post_candidate_revision()

                message = (
                    f"Successfully created and synced '{args.path}' to the container."
                )
                return Observation(message=message, files=[args.path])
            else:
                message = f"Failed to copy '{args.path}' to the container."
                logger.error(message)
                return Observation(message=message)

        except Exception as e:
            logger.exception(
                f"An unexpected error occurred in CreateFile for '{args.path}'."
            )
            return Observation(message=f"Error creating file {args.path}: {e}")
