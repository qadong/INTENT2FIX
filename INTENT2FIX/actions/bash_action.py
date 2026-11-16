import logging
import subprocess
from pathlib import Path
from typing import List, Optional

from pydantic import Field, ConfigDict

from INTENT2FIX.actions.action import Action, FewShotExample
from INTENT2FIX.actions.schema import ActionArguments, Observation
from INTENT2FIX.file_context import FileContext
from INTENT2FIX.repository.repository import Repository
from INTENT2FIX.runtime.runtime import RuntimeEnvironment
from INTENT2FIX.workspace import Workspace

from INTENT2FIX.envs.dockerenv import DockerEnv

logger = logging.getLogger(__name__)


class BashCommandArgs(ActionArguments):
    """
    Execute a bash command in the target environment.

    Notes:
    * Commands should be properly escaped for shell execution
    * Avoid interactive commands that require user input
    * Commands with sensitive data should use environment variables
    """

    command: str = Field(..., description="Bash command to execute")
    timeout: int = Field(
        30,
        description="Maximum execution time in seconds",
        gt=0,
        le=300,
    )
    working_dir: Optional[str] = Field(
        None,
        description="Working directory for command execution (default: repository root)",
    )

    model_config = ConfigDict(title="BashCommand")

    def format_args_for_llm(self) -> str:
        return f"""<command>{self.command}</command>
<timeout>{self.timeout}</timeout>
<working_dir>{self.working_dir or 'default'}</working_dir>"""

    @classmethod
    def format_schema_for_llm(cls) -> str:
        return cls.format_xml_schema({
            "command": "ls -la /path/to/dir",
            "timeout": 30,
            "working_dir": "/optional/path"
        })


class BashAction(Action):
    """
    Action to execute bash commands in Docker environment.
    """

    args_schema = BashCommandArgs

    def __init__(
        self,
        runtime: RuntimeEnvironment | None = None,
        repository: Repository | None = None,
        docker_env: DockerEnv | None = None,  
        **data,
    ):
        super().__init__(**data)
        self._docker_env = docker_env  
    def execute(
        self,
        args: BashCommandArgs,
        file_context: FileContext | None = None,
        workspace: Workspace | None = None,
    ) -> Observation:
        if not self._docker_env:
            return Observation(
                message="Docker environment not initialized",
                properties={"success": False, "fail_reason": "no_docker_env"}
            )

        try:
            output = self._docker_env.communicate(
                input=args.command,
                timeout_duration=args.timeout
            )
            
            return Observation(
                message="Command executed",
                properties={
                    "success": True,
                    "command": args.command,
                    "output": output,
                }
            )

        except Exception as e:  
            return Observation(
                message=f"Command failed: {str(e)}",
                properties={
                    "success": False,
                    "command": args.command,
                    "fail_reason": type(e).__name__
                }
            )

    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Run the project's test suite",
                action=BashCommandArgs(
                    thoughts="Execute pytest with coverage reporting",
                    command="python -m pytest tests/ --cov=src",
                    timeout=120,
                    working_dir="/repo",
                ),
            ),
        ]