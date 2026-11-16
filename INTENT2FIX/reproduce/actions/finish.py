import re
from typing import List, Literal, Optional

from pydantic import Field, ConfigDict
from INTENT2FIX.actions.action import Action
from INTENT2FIX.actions.schema import ActionArguments, Observation
from INTENT2FIX.reproduce.intent_schema import GraphBinding, TestArtifact
from INTENT2FIX.reproduce.state import ReproductionSession, normalize_artifact_path
from INTENT2FIX.runtime.dockerenv import DockerEnv

class FinishArgs(ActionArguments):
    """Arguments for the Finish action."""
    determinant: bool = Field(..., description="A boolean indicating if the issue is reproduced.")
    evidence_run_id: Optional[str] = Field(
        default=None,
        description=(
            "The exact RUN-N shown by the failing compile or execution observation. "
            "Required when determinant is true; successful and inspection commands are invalid."
        ),
    )
    failure_class: Literal[
        "TARGET_COMPILE_FAILURE",
        "TARGET_RUNTIME_FAILURE",
        "TARGET_ASSERTION_FAILURE",
        "TARGET_CRASH_OR_SANITIZER",
        "UNRELATED_FAILURE",
        "SETUP_ERROR",
        "NOT_REPRODUCED",
    ] = "NOT_REPRODUCED"
    expected_behavior_predicate: str = Field(
        default="",
        description="A stable predicate for recognizing the reported behavior, not an exact copied error string.",
    )
    matched_claim_ids: List[str] = Field(default_factory=list)
    test_artifacts: List[TestArtifact] = Field(default_factory=list)
    graph_bindings: List[GraphBinding] = Field(default_factory=list)
    unresolved: List[str] = Field(default_factory=list)

    model_config = ConfigDict(title="Finish")

class Finish(Action):
    """
    An action to terminate the reproduction task.
    The agent uses this to signal that it has either succeeded or failed to reproduce the bug.
    """
    args_schema: type[ActionArguments] = FinishArgs
    session: ReproductionSession | None = None
    docker_env: DockerEnv | None = None

    @staticmethod
    def _canonical_run_id(run_id: Optional[str]) -> Optional[str]:
        if not run_id:
            return None
        match = re.fullmatch(r"run[-_ ]?(\d+)", run_id, re.IGNORECASE)
        return f"RUN-{match.group(1)}" if match else run_id

    def _missing_container_artifacts(self, paths: List[str]) -> List[str]:
        if not self.docker_env or not self.docker_env.container:
            return []
        missing = []
        for path in paths:
            exit_code, _ = self.docker_env.container.exec_run(
                cmd=["test", "-f", path],
                workdir=self.docker_env.work_dir,
            )
            if exit_code != 0:
                missing.append(path)
        return missing

    def execute(self, args: FinishArgs, **kwargs) -> Observation:
        """
        Sets the terminal flag and returns the final summary.
        """
        properties = args.model_dump(exclude={"thoughts"})
        if args.determinant and self.session:
            declared_paths = []
            invalid_paths = []
            for artifact in args.test_artifacts:
                normalized = normalize_artifact_path(artifact.path)
                if normalized is None or normalized not in self.session.created_artifacts:
                    invalid_paths.append(artifact.path)
                else:
                    declared_paths.append(normalized)

            if invalid_paths or not declared_paths:
                return Observation(
                    message=(
                        "Finish rejected: every test artifact must be a repository-relative "
                        "file successfully created by CreateFile and kept until Finish. "
                        f"Invalid artifacts: {invalid_paths or ['<none declared>']}"
                    ),
                    expect_correction=True,
                    properties={"invalid_artifacts": invalid_paths},
                )

            missing_paths = self._missing_container_artifacts(declared_paths)
            if missing_paths:
                return Observation(
                    message=(
                        "Finish rejected: declared reproduction artifacts no longer exist "
                        f"in the container: {missing_paths}"
                    ),
                    expect_correction=True,
                    properties={"missing_artifacts": missing_paths},
                )

            requested_run_id = self._canonical_run_id(args.evidence_run_id)
            candidates = self.session.eligible_failure_records(declared_paths)
            selected = self.session.get_execution(requested_run_id) if requested_run_id else None
            selected_is_eligible = selected in candidates

            if not selected_is_eligible:
                if len(candidates) == 1:
                    selected = candidates[0]
                    properties["evidence_run_id"] = selected.run_id
                    message = (
                        f"Task finished: submitted evidence {args.evidence_run_id} was replaced "
                        f"by the only eligible artifact-failure record {selected.run_id}."
                    )
                else:
                    return Observation(
                        message=(
                            "Finish rejected: evidence_run_id must name a non-zero RunCommand "
                            "that directly compiles or runs a declared CreateFile artifact. "
                            f"Eligible run IDs: {[record.run_id for record in candidates]}"
                        ),
                        expect_correction=True,
                        properties={
                            "requested_run_id": requested_run_id,
                            "candidate_run_ids": [record.run_id for record in candidates],
                        },
                    )
            else:
                properties["evidence_run_id"] = selected.run_id
                message = f"Task finished: reproduction evidence submitted as {selected.run_id}."
        elif args.determinant:
            message = f"Task finished: reproduction evidence submitted as {args.evidence_run_id}."
        else:
            message = "Task finished: bug was not reliably reproduced."
            
        return Observation(
            message=message, 
            terminal=True, 
            properties={
                **properties,
            }
        )
