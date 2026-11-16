from typing import Optional, Dict, Any, List
from docker import DockerClient
from docker.errors import DockerException

import logging
from abc import ABC
from typing import List, Optional, Type, Any, ClassVar, Tuple

from pydantic import Field, PrivateAttr, BaseModel, field_validator, model_validator

from INTENT2FIX.actions.action import Action
from INTENT2FIX.actions.schema import ActionArguments, Observation, RewardScaleEntry
from INTENT2FIX.completion import BaseCompletionModel
from INTENT2FIX.completion.model import Completion
from INTENT2FIX.completion.schema import (
    ChatCompletionAssistantMessage,
    ChatCompletionUserMessage,
)
from INTENT2FIX.completion.schema import ResponseSchema
from INTENT2FIX.exceptions import CompletionRejectError
from INTENT2FIX.file_context import FileContext
from INTENT2FIX.index import CodeIndex
from INTENT2FIX.index.types import SearchCodeResponse
from INTENT2FIX.repository.repository import Repository
from INTENT2FIX.workspace import Workspace


class DockerEnvArgs(ActionArguments):
    docker_image: str = Field(
        ...,
        description="The Docker image address to use for the task.",
    )
    command: Optional[str] = Field(
        default=None,
        description="The command to run inside the Docker container.",
    )
    environment_vars: Optional[Dict[str, str]] = Field(
        default=None,
        description="Environment variables to set inside the Docker container.",
    )
    volumes: Optional[Dict[str, Dict[str, str]]] = Field(
        default=None,
        description="Volumes to mount inside the Docker container.",
    )
    ports: Optional[Dict[str, int]] = Field(
        default=None,
        description="Ports to expose from the Docker container.",
    )

    @field_validator("docker_image")
    @classmethod
    def validate_docker_image(cls, v):
        if not v:
            raise ValueError("Docker image address must be provided.")
        return v

class DockerEnvObservation(Observation):
    container_id: Optional[str] = Field(
        default=None,
        description="The ID of the Docker container where the task was executed.",
    )
    logs: Optional[str] = Field(
        default=None,
        description="The logs output from the Docker container.",
    )
    exit_code: Optional[int] = Field(
        default=None,
        description="The exit code of the command executed in the Docker container.",
    )

class DockerEnv(Action):
    args_schema: ClassVar[Type[ActionArguments]] = DockerEnvArgs

    _docker_client: DockerClient = PrivateAttr()

    def __init__(
        self,
        docker_client: DockerClient = None,
        **data,
    ):
        super().__init__(**data)
        self._docker_client = docker_client or DockerClient.from_env()

    def _run_container(self, args: DockerEnvArgs) -> DockerEnvObservation:
        try:
            # Pull the Docker image
            self._docker_client.images.pull(args.docker_image)

            # Run the Docker container with the specified configuration
            container = self._docker_client.containers.run(
                args.docker_image,
                command=args.command,
                environment=args.environment_vars,
                volumes=args.volumes,
                ports=args.ports,
                detach=True,
            )

            # Wait for the container to finish execution
            container.wait()

            # Get the logs from the container
            logs = container.logs().decode("utf-8")

            # Get the exit code of the container
            exit_code = container.attrs["State"]["ExitCode"]

            # Return the observation with the container ID, logs, and exit code
            return DockerEnvObservation(
                container_id=container.id,
                logs=logs,
                exit_code=exit_code,
                message="Docker container execution completed successfully.",
                summary=f"Docker container execution completed with exit code {exit_code}.",
            )

        except DockerException as e:
            return DockerEnvObservation(
                message=f"Failed to execute Docker container: {str(e)}",
                summary="Docker container execution failed.",
                properties={"error": str(e)},
            )

    def execute(
        self,
        args: DockerEnvArgs,
        file_context: FileContext | None = None,
        workspace: Workspace | None = None,
    ) -> DockerEnvObservation:
        raise NotImplementedError("Subclasses must implement this method.")

    @classmethod
    def get_evaluation_criteria(cls, trajectory_length) -> List[str]:
        evaluation_criteria = super().get_evaluation_criteria(trajectory_length)
        evaluation_criteria.extend(
            [
                "Docker Image Validity: Ensure the Docker image address is valid and accessible.",
                "Command Execution: Evaluate if the command executed inside the container was successful.",
                "Logs Relevance: Assess the relevance and usefulness of the logs generated by the container.",
                "Exit Code Analysis: Check if the exit code indicates a successful or failed execution.",
            ]
        )
        return evaluation_criteria

    @classmethod
    def get_reward_scale(cls, trajectory_length) -> List[RewardScaleEntry]:
        return cls.generate_reward_scale_entries(
            [
                (
                    90,
                    100,
                    "The Docker container execution was successful with relevant logs and a successful exit code.",
                ),
                (
                    75,
                    89,
                    "The Docker container execution had minor issues, but the logs and exit code are still useful.",
                ),
                (
                    50,
                    74,
                    "The Docker container execution had some issues, but the logs and exit code provide some useful information.",
                ),
                (
                    25,
                    49,
                    "The Docker container execution failed, but some logs were generated that could be useful for debugging.",
                ),
                (
                    0,
                    24,
                    "The Docker container execution failed entirely, with no useful logs or exit code information.",
                ),
            ]
        )