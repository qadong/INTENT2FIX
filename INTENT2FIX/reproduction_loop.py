import logging
import re
from typing import Dict
from INTENT2FIX.agent.agent import ActionAgent
from INTENT2FIX.file_context import FileContext
from INTENT2FIX.repository.repository import Repository
from INTENT2FIX.runtime.dockerenv import DockerEnv
from INTENT2FIX.node import Node
from INTENT2FIX.reproduce.result import ReproductionResult
from INTENT2FIX.reproduce.intent_schema import ExecutionRecord, GraphBinding, TestArtifact
from INTENT2FIX.reproduce.state import (
    command_references_artifact,
    command_removes_artifact,
    is_reproduction_command,
    normalize_artifact_path,
)

logger = logging.getLogger(__name__)

class ReproductionLoop:
    """
    A dedicated loop to drive the ReproductionAgent.
    This loop manages the lifecycle of the Docker container and the agent's execution,
    producing a structured result.
    """
    def __init__(
        self,
        agent: ActionAgent,
        docker_env: DockerEnv,
        repository: Repository,
        repo_path: str,
        max_turns: int = 7,
        require_intent_evidence: bool = True,
    ):
        self.agent = agent
        self.docker_env = docker_env
        self.repository = repository
        self.repo_path = repo_path
        self.max_turns = max_turns
        self.require_intent_evidence = require_intent_evidence

    def run(self, issue: str) -> ReproductionResult:
        """
        Runs the agent loop to reproduce an issue.

        Note: This method assumes the Docker environment is already running.
        Lifecycle management (init, close) should be handled by the caller.
        """
        logger.info(f"Syncing repository from {self.repo_path} to container...")
        if not self.docker_env.copy_anything_to_container(f"{self.repo_path}/.", self.docker_env.work_dir):
            raise ConnectionError("Failed to sync repository to container.")

        file_context = FileContext(repo=self.repository, runtime=self.docker_env)
        root_node = Node.stub(user_message=issue, file_context=file_context)
        current_node = root_node

        for i in range(self.max_turns):
            logger.info(f"--- Reproduction Turn {i + 1}/{self.max_turns} ---")
            self.agent.run(current_node)

            if current_node.is_terminal():
                logger.info(f"Agent has finished the task.")
                break
            
            new_node = Node.stub(parent=current_node, file_context=current_node.file_context)
            current_node.add_child(new_node)
            current_node = new_node
        
        if not current_node.is_terminal():
            logger.warning("Max turns reached, but the agent has not finished the task.")
        
        return self._create_result_from_trajectory(root_node)

    def _create_result_from_trajectory(self, root_node: Node) -> ReproductionResult:
        """
        Processes the trajectory to create a structured ReproductionResult.
        """
        if not root_node:
            return ReproductionResult(is_reproduced=False, full_trajectory=None)

        all_nodes = root_node.get_all_nodes()
        terminal_node = all_nodes[-1]
        
        is_reproduced = False
        reproduction_output = None
        expected_output = None
        failure_class = None
        matched_claim_ids = []
        graph_bindings = []
        test_artifacts = []
        unresolved = []
        evidence_run_id = None

        latest_test_files: Dict[str, str] = {}
        removed_test_files = set()
        execution_records = []
        repository_evidence = []
        grounding_actions = {
            "EntityFunctionSemanticSearch",
            "EntityIntentSearch",
            "EntitySearchSymbol",
            "GetMacroUsages",
            "FindTemplateInstantiation",
            "FindEntityUsage",
            "ViewCode",
        }

        for node in all_nodes:
            for action_step in node.action_steps:
                action_name = action_step.action.name
                if action_name == "CreateFile":
                    action_args = action_step.action.model_dump(exclude={"thoughts"})
                    path = action_args.get("path")
                    content = action_args.get("file_text")
                    if path and content is not None:
                        latest_test_files[path] = content
                        removed_test_files.discard(path)
                if action_name == "RunCommand" and action_step.observation:
                    props = action_step.observation.properties or {}
                    if props.get("run_id") and props.get("command") is not None:
                        execution_records.append(ExecutionRecord(
                            run_id=props["run_id"],
                            command=props["command"],
                            exit_code=int(props.get("exit_code", -1)),
                            stdout=props.get("stdout", ""),
                            stderr=props.get("stderr", ""),
                        ))
                        removed_test_files.update(
                            path
                            for path in latest_test_files
                            if command_removes_artifact(props["command"], [path])
                        )
                if action_name in grounding_actions and action_step.observation:
                    repository_evidence.append({
                        "action": action_name,
                        "arguments": action_step.action.model_dump(exclude={"thoughts"}),
                        "observation": (action_step.observation.message or "")[:6000],
                    })

        if terminal_node.is_terminal() and terminal_node.observation and terminal_node.observation.properties:
            props = terminal_node.observation.properties
            is_reproduced = props.get("determinant", False)
            evidence_run_id = props.get("evidence_run_id")
            if evidence_run_id:
                run_id_match = re.fullmatch(r"run[-_ ]?(\d+)", evidence_run_id, re.IGNORECASE)
                if run_id_match:
                    evidence_run_id = f"RUN-{run_id_match.group(1)}"
            expected_output = props.get("expected_behavior_predicate")
            failure_class = props.get("failure_class")
            matched_claim_ids = props.get("matched_claim_ids", [])
            graph_bindings = [GraphBinding.model_validate(item) for item in props.get("graph_bindings", [])]
            test_artifacts = [TestArtifact.model_validate(item) for item in props.get("test_artifacts", [])]
            unresolved = props.get("unresolved", [])

        normalized_declared_paths = {
            normalized
            for artifact in test_artifacts
            if (normalized := normalize_artifact_path(artifact.path)) is not None
        }
        selected_record = next(
            (record for record in execution_records if record.run_id == evidence_run_id),
            None,
        )
        allowed_failure_classes = {
            "TARGET_COMPILE_FAILURE",
            "TARGET_RUNTIME_FAILURE",
            "TARGET_ASSERTION_FAILURE",
            "TARGET_CRASH_OR_SANITIZER",
        }
        if is_reproduced and (
            selected_record is None
            or selected_record.exit_code == 0
            or not is_reproduction_command(selected_record.command)
            or not any(
                command_references_artifact(selected_record.command, path)
                for path in normalized_declared_paths
            )
            or failure_class not in allowed_failure_classes
            or (getattr(self, "require_intent_evidence", True) and not matched_claim_ids)
        ):
            is_reproduced = False
            unresolved.append("Finish evidence did not reference a valid target-failure command and issue claim.")

        test_files = [
            {"path": path, "content": content}
            for path, content in latest_test_files.items()
            if path in normalized_declared_paths and path not in removed_test_files
        ]
        if is_reproduced and not test_files:
            is_reproduced = False
            unresolved.append("No declared reproduction test artifact matched a created file.")

        if selected_record:
            reproduction_output = selected_record.output

        # Extract test script from CreateFile actions
        test_script_parts = []
        for node in all_nodes:
            for action_step in node.action_steps:
                if action_step.action.name == 'CreateFile':
                    file_text = action_step.action.model_dump().get('file_text')
                    if file_text:
                        test_script_parts.append(file_text)
        
        test_script = "\n--- (New File) ---\n".join(test_script_parts)

        return ReproductionResult(
            is_reproduced=is_reproduced,
            test_script=test_script,
            test_command=selected_record.command if selected_record else None,
            test_files=test_files,
            reproduction_output=reproduction_output,
            expected_output=expected_output,
            failure_class=failure_class,
            matched_claim_ids=matched_claim_ids,
            test_artifacts=test_artifacts,
            graph_bindings=graph_bindings,
            execution_records=execution_records,
            repository_evidence=repository_evidence,
            unresolved=unresolved,
            full_trajectory=root_node
        )
