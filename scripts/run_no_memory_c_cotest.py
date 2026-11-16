import argparse
import concurrent.futures
import json
import logging
import os
import re
import signal
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from INTENT2FIX.benchmark.swebench import create_repository
from INTENT2FIX.artifact_paths import results_path, submission_path
from INTENT2FIX.runtime.dockerenv import DockerEnv
from INTENT2FIX.agent.c_code_agent import CCodingAgent
from INTENT2FIX.index import CodeIndex
from INTENT2FIX.loop import AgenticLoop
from INTENT2FIX.file_context import FileContext
from INTENT2FIX.completion.base import LLMResponseFormat
from INTENT2FIX.schema import MessageHistoryType
from INTENT2FIX.repo_structure import CodeParser, DirectoryAnalyzer, show_code_structure
from INTENT2FIX.memory.memory import Memory
from INTENT2FIX.reproduction_loop import ReproductionLoop
from INTENT2FIX.reproduce.reproduction_agent import ReproductionAgent
from INTENT2FIX.completion.lite_llm import LiteLLM

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DOTENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_CONFIG: Dict[str, Any] = {
    "jsonl": "INTENT2FIX/benchmark/multi_swebench_c_change.jsonl",
    "workers": 8,
    "submission_file": "submission_c.json",
    "results_dir_prefix": "results_c",
    "repair_model": "claude-3.7-sonnet",
    "reproduction_model": None,
    "extraction_model": "openrouter/deepseek/deepseek-v3.2",
    "reproduction_temperature": 0.2,
    "repair_temperature": 0.2,
    "reproduction_max_tokens": 2000,
    "repair_max_tokens": 2000,
    "extraction_max_tokens": None,
    "reproduction_steps": 20,
    "repair_steps": 60,
    "message_cache": True,
    "instance_ids": None,
    "docker_registry": "6pqg5gclf6yaz1ch6g.xuanyuan.run",
}

ENV_CONFIG_MAP = {
    "jsonl": ("C_COTEST_JSONL", str),
    "workers": ("C_COTEST_MAX_WORKERS", int),
    "submission_file": ("C_COTEST_SUBMISSION_FILE", str),
    "results_dir_prefix": ("C_COTEST_RESULTS_DIR_PREFIX", str),
    "repair_model": ("C_COTEST_REPAIR_MODEL", str),
    "reproduction_model": ("C_COTEST_REPRO_MODEL", str),
    "extraction_model": ("C_COTEST_EXTRACTION_MODEL", str),
    "reproduction_temperature": ("C_COTEST_REPRO_TEMPERATURE", float),
    "repair_temperature": ("C_COTEST_REPAIR_TEMPERATURE", float),
    "reproduction_max_tokens": ("C_COTEST_REPRO_MAX_TOKENS", int),
    "repair_max_tokens": ("C_COTEST_REPAIR_MAX_TOKENS", int),
    "extraction_max_tokens": ("C_COTEST_EXTRACTION_MAX_TOKENS", int),
    "reproduction_steps": ("C_COTEST_REPRO_STEPS", int),
    "repair_steps": ("C_COTEST_REPAIR_STEPS", int),
    "message_cache": ("C_COTEST_MESSAGE_CACHE", "bool"),
    "instance_ids": ("C_COTEST_INSTANCE_IDS", "list"),
    "docker_registry": ("C_COTEST_DOCKER_REGISTRY", str),
}


def _load_env() -> None:
    load_dotenv(DOTENV_PATH, override=True)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _coerce_config_value(value: Any, value_type: Any) -> Any:
    if value is None:
        return None
    if value_type == "bool":
        return _parse_bool(value)
    if value_type == "list":
        if isinstance(value, list):
            return value
        return [item.strip() for item in str(value).split(",") if item.strip()]
    return value_type(value)


def _resolve_config_path(path: str | None) -> str | None:
    if not path:
        return None
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(PROJECT_ROOT, path))


def _load_json_config(config_path: str | None) -> Dict[str, Any]:
    resolved_path = _resolve_config_path(config_path)
    if not resolved_path:
        return {}
    with open(resolved_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a JSON object: {resolved_path}")
    return data.get("c_cotest", data)


def _apply_env_config(config: Dict[str, Any]) -> None:
    for key, (env_key, value_type) in ENV_CONFIG_MAP.items():
        value = os.getenv(env_key)
        if value is None or not str(value).strip():
            continue
        try:
            config[key] = _coerce_config_value(value, value_type)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid value for {env_key}: {value}") from e


def _normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    aliases = {
        "max_workers": "workers",
        "submission": "submission_file",
        "result_prefix": "results_dir_prefix",
        "repro_model": "reproduction_model",
        "repro_steps": "reproduction_steps",
        "repair_max_iterations": "repair_steps",
        "reproduction_max_turns": "reproduction_steps",
    }
    for old_key, new_key in aliases.items():
        if old_key in config and new_key not in config:
            config[new_key] = config[old_key]

    if not config.get("reproduction_model"):
        config["reproduction_model"] = config["repair_model"]

    int_keys = [
        "workers",
        "reproduction_max_tokens",
        "repair_max_tokens",
        "reproduction_steps",
        "repair_steps",
    ]
    optional_int_keys = ["extraction_max_tokens"]
    float_keys = ["reproduction_temperature", "repair_temperature"]

    for key in int_keys:
        config[key] = int(config[key])
    for key in optional_int_keys:
        if config.get(key) is not None:
            config[key] = int(config[key])
    for key in float_keys:
        config[key] = float(config[key])
    config["message_cache"] = _parse_bool(config["message_cache"])
    if config.get("instance_ids") in (None, ""):
        config["instance_ids"] = None
    else:
        config["instance_ids"] = _coerce_config_value(config["instance_ids"], "list")
    return config


def load_runner_config(config_path: str | None = None) -> Dict[str, Any]:
    _load_env()
    config = dict(DEFAULT_CONFIG)
    _apply_env_config(config)
    config.update(_load_json_config(config_path))
    return _normalize_config(config)


def _parse_skip_instances() -> set[str]:
    raw = os.getenv("COTEST_SKIP_INSTANCE_IDS", "")
    if not raw.strip():
        return set()
    return {x.strip() for x in raw.split(",") if x.strip()}


def _resolve_jsonl_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(PROJECT_ROOT, path))


def load_instances_from_jsonl(file_path: str) -> Dict[str, Dict[str, Any]]:
    instances: Dict[str, Dict[str, Any]] = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    instance = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping line due to JSON decode error: {line.strip()[:100]}...")
                    continue

                instance_id = instance.get("instance_id")
                if instance_id:
                    instances[instance_id] = instance
    except FileNotFoundError:
        logger.error(f"Error: File not found at {file_path}")
    return instances


def get_c_benchmark_instance(instance_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    jsonl_file = _resolve_jsonl_path(config["jsonl"])
    instances = load_instances_from_jsonl(jsonl_file)
    instance = instances.get(instance_id)
    if not instance:
        raise ValueError(f"Instance {instance_id} not found in C benchmark JSONL: {jsonl_file}")
    return dict(instance)


def parse_instance_string(repo_str: str):
    pattern = r'^([a-zA-Z0-9_.-]+)__([a-zA-Z0-9_.-]+)[-_](\d+)$'
    match = re.match(pattern, repo_str)
    if not match:
        return None
    return {"org": match.group(1), "repo": match.group(2), "number": match.group(3)}
    
def get_repo_structure(repo_path: str) -> str:
    parser = CodeParser()
    analyzer = DirectoryAnalyzer(parser)
    repo_structure = analyzer.analyze(repo_path)
    return show_code_structure(repo_structure)

def _extract_reproduction_details_with_agent(
    problem_statement: str,
    trajectory: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Tuple[str, str, List[Dict[str, str]]]:
    """
    Uses an LLM to act as an "Extractor Agent" to summarize the trajectory,
    pull out the definitive test files, and identify the test command.
    """
    prompt = (
        "You are an expert software engineer reviewing a bug reproduction attempt. "
        "Based on the following problem description and the step-by-step action/observation trajectory, "
        "your task is to produce a JSON object with three keys:\n"
        "1. `reproduction_summary`: A concise, one-paragraph summary of the process of reproducing the bug. "
        "Focus on the actions taken, the commands run, and the final outcome that confirmed the bug's presence.\n"
        "2. `test_command`: A string containing the compelte command(containing compile && excute) that was run to trigger the bug. "
        "A good example: gcc -I . test_issue.c -o test_issue && ./test_issue"
        "This should be the command that, when executed, produced the expected error or failure. "
        "Focus on commands that compile and execute C tests, like 'gcc -I . test_issue.c -o test_issue && ./test_issue', `make test`, `ctest`, or `./a.out`. "
        "Exclude setup commands like `ls`, `cd`, or `pip install`.\n"
        "3. `test_files`: A JSON array of objects, where each object has a `path` and `content` key. "
        "This array should ONLY contain the final, correct versions of the files that are essential to reproduce the bug. "
        "Do not include files that were created but later found to be incorrect or unnecessary.\n\n"
        "Analyze the trajectory carefully. The `test_command` should be the one whose observation contains the bug's signature (e.g., an error message, a failed assertion). "
        "The `test_files` should be the files created or modified by `CreateFile` or `EditFile` actions that are necessary for the `test_command` to run and fail as expected."
    )
    
    trajectory_str = "\n".join(
        f"Action: {json.dumps(item.get('action', {}))}\nObservation: {item.get('observation', '')}"
        for item in trajectory
    )
    
    extraction_kwargs = {"model": config["extraction_model"], "response_format": {"type": "json_object"}}
    if config.get("extraction_max_tokens") is not None:
        extraction_kwargs["max_tokens"] = config["extraction_max_tokens"]
    llm = LiteLLM(**extraction_kwargs)
    full_prompt = f"{prompt}\n\nProblem: {problem_statement}\n\nTrajectory:\n{trajectory_str}"
    
    try:
        response = llm.generate(full_prompt)
                
        # Read the response message ONCE and cache it in a local variable
        raw_response_message = response.message if response else ""
        
        # Log the actual response message for debugging
        logger.info(f"--- Raw LLM response for extraction ---\n{raw_response_message}\n------------------------------------")

        if not raw_response_message or not raw_response_message.strip():
             raise json.JSONDecodeError("LLM returned an empty response.", raw_response_message, 0)

        # Extract JSON from markdown code block if present
        json_match = re.search(r"```json\n(.*?)\n```", raw_response_message, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Fallback for cases where the response is pure JSON
            json_str = raw_response_message

        data = json.loads(json_str)
        summary = data.get("reproduction_summary", "")
        test_command = data.get("test_command", "")
        test_files = data.get("test_files", [])
        
        if not summary or not isinstance(test_files, list):
            raise ValueError("LLM response did not contain the expected keys or format.")
            
        return summary, test_command, test_files    
    except json.JSONDecodeError as e:
        logger.error(f"Failed to generate and parse details from LLM: {e}")
        return None, None, None
    except Exception as e:
        logger.error(f"An unexpected error occurred during LLM response processing: {e}")
        return None, None, None

def run_reproduction_step(
    instance_name: str,
    model_name: str,
    docker_env: DockerEnv,
    output_dir: str,
    config: Dict[str, Any],
) -> Memory | None:
    """
    Runs the reproduction step and returns a Memory object if successful.
    """
    logger.info(f"--- Starting Reproduction Step for {instance_name} ---")
    
    instance = get_c_benchmark_instance(instance_name, config)
    repository = create_repository(instance)

    with docker_env:
        logger.info(f"Docker container '{docker_env.container_name}' is running for reproduction.")
        
        agent = ReproductionAgent.create(
            repository=repository,
            model=model_name,
            temperature=config["reproduction_temperature"],
            max_tokens=config["reproduction_max_tokens"],
            docker_env=docker_env,
            message_cache=config["message_cache"],
        )

        loop = ReproductionLoop(
            agent=agent,
            docker_env=docker_env,
            repository=repository,
            repo_path=repository.repo_path,
            max_turns=config["reproduction_steps"]
        )

        repo_structure = f"\nRepo_structure'''\n{get_repo_structure(repository.repo_path)}"
        try:
            repro_result = loop.run(issue=instance["problem_statement"] + repo_structure)
        except Exception as e:
            # Catch any exception from loop.run() to prevent process crash
            logger.error(f"Reproduction loop failed with exception: {e}", exc_info=True)
            from INTENT2FIX.exceptions import CompletionRuntimeError
            if isinstance(e, CompletionRuntimeError):
                logger.error(f"CompletionRuntimeError detected during reproduction. This may be due to incorrect model name or API issues.")
            return None

    # --- Process and Save results ---
    if repro_result.full_trajectory:
        # Save trajectory to the specified output directory
        persist_path = os.path.join(output_dir, f"repro_trajectory_{instance_name}.json")
        os.makedirs(os.path.dirname(persist_path), exist_ok=True)
        logger.info(f"Saving full reproduction trajectory to {persist_path}")
        with open(persist_path, "w") as f:
            json.dump(repro_result.full_trajectory.model_dump(exclude_none=True), f, indent=2)
    else:
        logger.error("Reproduction task failed to produce a trajectory.")
        return None

    if repro_result.is_reproduced:
        logger.info("Reproduction successful! Generating memory...")
        
        trajectory_data = [
            {"action": step.action.model_dump(), "observation": step.observation.message if step.observation else None}
            for node in repro_result.full_trajectory.get_all_nodes()
            for step in node.action_steps
        ]
        
        summary, test_command, test_files = _extract_reproduction_details_with_agent(
            instance["problem_statement"],
            trajectory_data,
            config,
        )

        if not summary or not test_files:
            logger.error("Could not extract reproduction details from trajectory. Aborting memory creation.")
            return None

        new_memory = Memory(
            instance_id=instance_name,
            problem_statement=instance["problem_statement"],
            reproduction_summary=summary,
            golden_patch=instance.get("patch", ""),
            test_command=test_command,
            test_files=test_files,
            reproduction_result=repro_result.reproduction_output or "",
            expected_reproduction_result=repro_result.expected_output or ""
        )
        logger.info(f"Successfully created memory for instance {instance_name}.")
        return new_memory
    else:
        logger.warning("Reproduction was unsuccessful. No memory will be created.")
        return None

def run_repair_step(
    instance_name: str,
    model_name: str,
    docker_env: DockerEnv,
    memory: Memory,
    output_dir: str,
    config: Dict[str, Any],
) -> str | None:
    """
    Runs the bug repair step using the provided memory from the reproduction step.
    Note: This version does NOT use MemoryModule for searching past experiences.
    """

    logger.info(f"--- Starting Repair Step for {instance_name} ---")
    _load_env()
    
    instance = get_c_benchmark_instance(instance_name, config)
    repository = create_repository(instance)
    
    # Index is already processed, just load it
    # Set environment variables for entity_tools
    # These should be loaded from .env file via load_dotenv()
    index_store_dir = os.getenv("INDEX_STORE_DIR")
    if not index_store_dir:
        raise ValueError("INDEX_STORE_DIR environment variable is not set. Please set it in .env file.")
    
    # Set SWEBENCH_INSTANCE_NAME for entity_tools
    os.environ["SWEBENCH_INSTANCE_NAME"] = instance_name
    
    logger.info(f"Loading index for instance: {instance_name}")
    logger.info(f"INDEX_STORE_DIR: {index_store_dir}")
    logger.info(f"SWEBENCH_INSTANCE_NAME: {os.environ.get('SWEBENCH_INSTANCE_NAME')}")
    
    # Load the index using EntityCodeIndex (which is now aliased as CodeIndex)
    code_index = CodeIndex.from_index_name(
        index_name=instance_name,
        file_repo=repository,
        index_store_dir=index_store_dir
    )

    with docker_env:
        logger.info(f"Docker container '{docker_env.container_name}' is running for repair.")

        file_context = FileContext(repo=repository, runtime=docker_env)

        if memory.test_files:
            for test_file in memory.test_files:
                # Ensure the path from memory is treated as a relative path
                relative_path = test_file['path'].lstrip('/')
                
                # Construct the full path inside the container's workspace
                full_container_path = os.path.join(docker_env.work_dir, relative_path)
                
                # Write the file to the correct workspace location
                docker_env.write_file(full_container_path, test_file['content'])
                
                # ALSO write the file to the host repository path
                # This ensures the FileRepository can find it and it survives serialization
                full_host_path = repository.get_full_path(relative_path)
                os.makedirs(os.path.dirname(full_host_path), exist_ok=True)
                with open(full_host_path, "w") as f:
                    f.write(test_file['content'])
                
                # Add the file to the context using its relative path
                file_context.add_file(relative_path, content=test_file['content'])
                logger.info(f"Created test file in container workspace and host repo at {relative_path} and added to context.")
        else:
            logger.info("No test files from reproduction step. Proceeding with repair.")

        agent = CCodingAgent.create(
            repository=repository,
            code_index=code_index,
            model=model_name,
            temperature=config["repair_temperature"],
            max_tokens=config["repair_max_tokens"],
            response_format=LLMResponseFormat.REACT,
            message_history_type=MessageHistoryType.REACT,
            runtime=docker_env,
            message_cache=config["message_cache"],
        )

        repo_structure = show_code_structure(DirectoryAnalyzer(CodeParser()).analyze(repository.repo_path))

        message = (
            f"<task>\nProblem Statement:\n{memory.problem_statement}\n<task>\n"
        )

        if memory.test_files:
            poc_test_paths = [f"  - {test_file['path']}" for test_file in memory.test_files]
            poc_block = "<Poc_test>\n" + "\n".join(poc_test_paths) + "\n</Poc_test>"
            message += (
                f"The following Proof-of-Concept (PoC) test file(s) have been created to demonstrate the bug. "
                f"You should run these tests when appropriate to understand the failure and verify your fix.\n"
                f"{poc_block}\n\n"
            )
        if memory.test_command:
            command_block = f"<Poc_command>\n{memory.test_command}\n</Poc_command>"
            logger.info(f'Poc_command: f{command_block}')
            message += (
                f"Use the following command to run the PoC test:\n"
                f"{command_block}\n\n"
            )
        if memory.reproduction_summary:
            message += f"Reproduction Summary:\n{memory.reproduction_summary}\n\n"
        if memory.reproduction_result:
            message += f"Reproduction Output:\n{memory.reproduction_result}\n\n"
        
        message += f"Repo Structure:\n{repo_structure}\n\nPlease provide a patch to fix the issue."

        persist_path = os.path.join(output_dir, f"repair_trajectory_{instance_name}.json")
        os.makedirs(os.path.dirname(persist_path), exist_ok=True)
        metadata = {
            "instance_id": instance_name,
            "problem_statement": memory.problem_statement,
            "reproduction_summary": memory.reproduction_summary,
            "reproduction_result": memory.reproduction_result,
            "expected_reproduction_result": memory.expected_reproduction_result,
            "result_persist_path": output_dir, # Use the dynamic output directory
        }

        # Create loop WITHOUT memory_module by manually creating and disabling it
        loop = AgenticLoop.create(
            agent=agent,
            file_context=file_context,
            message=message,
            repository=repository,
            runtime=docker_env,
            code_index=code_index,
            metadata=metadata,
            persist_path=persist_path,
            max_iterations=config["repair_steps"]
        )
        
        # Disable memory module - set it to None after creation
        loop.memory_module = None

        # Run the loop WITHOUT search_weights (since we don't have memory_module)
        logger.info("Running repair loop without memory module...")
        try:
            final_trajectory = loop.run()
        except Exception as e:
            # Catch any exception from loop.run() to prevent process crash
            logger.error(f"Repair loop failed with exception: {e}", exc_info=True)
            # Check if it's a CompletionRuntimeError due to model name issues
            from INTENT2FIX.exceptions import CompletionRuntimeError
            if isinstance(e, CompletionRuntimeError):
                logger.error(f"CompletionRuntimeError detected. This may be due to incorrect model name or API issues.")
            return None
        
        try:
            solution_patch = final_trajectory.file_context.generate_git_patch()
            # Check if there are any patches (non-empty string after stripping)
            # Empty string from generate_git_patch() means no patches, which is falsy
            has_patches = solution_patch and solution_patch.strip()
            if final_trajectory and has_patches:
                logger.info("Repair successful! Solution patch generated.")
                # Save solution patch to the dynamic output directory
                patch_path = os.path.join(output_dir, "solution.patch")
                with open(patch_path, "w") as f:
                    f.write(solution_patch)
                logger.info(f"Solution patch saved to {patch_path}")
                logger.info("Test enhancement is disabled for the C runner. Returning repair patch directly.")
                return solution_patch
            else:
                # Check if there were any file modifications at all
                has_any_patches = final_trajectory.file_context.has_patch(ignore_tests=False)
                if has_any_patches:
                    logger.warning(f"Repair completed but generate_git_patch() returned empty string. "
                                 f"FileContext has patches but they may have been reverted or only test files were modified.")
                else:
                    logger.warning("Repair was unsuccessful. No solution patch found and no file modifications detected.")
                return None
        except Exception as e:
            logger.error(f"Error generating or saving solution patch: {e}", exc_info=True)
            return None


def extract_instance_ids_from_jsonl(file_path: str) -> List[str]:
    instance_ids = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    json_object = json.loads(line.strip())
                    if 'instance_id' in json_object:
                        instance_ids.append(json_object['instance_id'])
                except json.JSONDecodeError:
                    logger.warning(f"Skipping line due to JSON decode error: {line.strip()[:100]}...")
    except FileNotFoundError:
        logger.error(f"Error: File not found at {file_path}")
    return instance_ids

def _signal_handler(signum, frame):
    """Handle system signals that might terminate the process"""
    raise SystemExit(f"Process terminated by signal {signum}")

def process_instance(instance_name: str, config: Dict[str, Any]) -> Dict[str, str] | None:
    """
    Processes a single instance, running repair step directly (reproduction skipped).
    Configures logging to a file specific to the instance.
    """
    # Register signal handlers to catch system-level terminations
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    try:
        # Ensure worker process sees the same env as main, and override any exported defaults.
        _load_env()
        os.environ["SWEBENCH_INSTANCE_NAME"] = instance_name
        repair_model_name = config["repair_model"]
        repro_model_name = config["reproduction_model"]
        sanitized_model_name = repair_model_name.replace("/", "_")
        results_prefix = results_path(config["results_dir_prefix"])
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = f"{results_prefix}_{sanitized_model_name}_cotest/{instance_name}"
        os.makedirs(result_dir, exist_ok=True)
        log_file_path = os.path.join(result_dir, f'repair_run_{time_stamp}.log')

        # Get the root logger for this process
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        # Remove any existing handlers to prevent conflicts or duplicates from parent process
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()

        formatter = logging.Formatter('%(asctime)s - %(process)d - %(levelname)s - %(message)s')
        
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Optionally, add a console handler for this process's output
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        # --- End of per-instance logging setup ---

        logger.info(f"===== Starting repair process for instance: {instance_name} =====")
        logger.info(f"Logging for this instance will be saved to: {log_file_path}")
        logger.info(f"All results will be saved in: {result_dir}")
        logger.info("Stage 1: reproduction; Stage 2: repair only (test enhancement disabled).")
        
        instance_str = parse_instance_string(instance_name)
        if not instance_str:
            logger.error(f"Invalid instance name format: {instance_name}")
            return None

        image_name = f'mswebench/{instance_str["org"]}_m_{instance_str["repo"]}:pr-{instance_str["number"]}'.lower()
        docker_registry = str(config.get("docker_registry", "")).strip().rstrip("/")
        if docker_registry:
            image_name = f"{docker_registry}/{image_name}"
        
        # --- Stage 1: Reproduction ---
        repro_docker_env = None
        memory_instance = None
        logger.info(f"===== Starting Stage 1: Reproduction for {instance_name} =====")
        try:
            repro_docker_env = DockerEnv(
                image_name=image_name,
                container_name=f'repro-{instance_name}-{time_stamp}',
                work_dir=f'/home/{instance_str["repo"]}',
            )
            memory_instance = run_reproduction_step(
                instance_name=instance_name,
                model_name=repro_model_name,
                docker_env=repro_docker_env,
                output_dir=result_dir,
                config=config,
            )
        except Exception as e:
            logger.error(f"An error occurred during the reproduction stage for '{instance_name}': {e}", exc_info=True)
        # Note: No need to cleanup here - run_reproduction_step uses 'with docker_env:' which auto-cleans up

        if not memory_instance:
            logger.warning(f"Reproduction failed or skipped for {instance_name}. Using mock memory for repair.")
            instance = get_c_benchmark_instance(instance_name, config)
            memory_instance = Memory(
                instance_id=instance_name,
                problem_statement=instance["problem_statement"],
                reproduction_summary="",
                repair_summary="",
                solution_patch="",
                golden_patch=instance.get("patch", ""),
                test_command="",
                test_files=[],
                reproduction_result="",
                expected_reproduction_result=""
            )

        # --- Stage 2: Repair (test enhancement disabled) ---
        repair_docker_env = None
        solution_patch = None
        logger.info(f"Passing memory_instance to run_repair_step: {memory_instance}")
        try:
            repair_docker_env = DockerEnv(
                image_name=image_name,
                container_name=f'repair-{instance_name}-{time_stamp}',
                work_dir=f'/home/{instance_str["repo"]}',
            )
            logger.info(f"===== Starting Stage 2: Repair for {instance_name} =====")
            solution_patch = run_repair_step(
                instance_name=instance_name, 
                model_name=repair_model_name, 
                docker_env=repair_docker_env, 
                memory=memory_instance,
                output_dir=result_dir,
                config=config,
            )
            logger.info(f"===== Successfully finished Repair for instance: {instance_name} ====")
        except Exception as e:
            logger.error(f"An error occurred during the repair stage for '{instance_name}': {e}", exc_info=True)
        # Note: No need to cleanup here - run_repair_step uses 'with docker_env:' which auto-cleans up
        
        if solution_patch:
            submission = {
                "instance_id": instance_name,
                "model_patch": solution_patch
            }
            return submission
        
        logger.info(f"===== End of processing for instance: {instance_name} ====")
        return None
    except Exception as e:
        # Global exception handler to prevent process crashes
        # Try to log to file if logger is set up, otherwise use print
        try:
            logger = logging.getLogger()
            logger.error(f"Unhandled exception in process_instance for '{instance_name}': {e}", exc_info=True)
        except Exception:
            print(f"CRITICAL: Unhandled exception in process_instance for '{instance_name}': {e}", file=sys.stderr)
        return None

def main():
    _load_env()
    if os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]

    parser = argparse.ArgumentParser(description="INTENT2FIX C co-test (no memory module).")
    parser.add_argument(
        "--config",
        default=os.getenv("C_COTEST_CONFIG"),
        help="Optional JSON config path. Env defaults use C_COTEST_* keys.",
    )
    parser.add_argument(
        "--jsonl",
        default=None,
        help="Benchmark JSONL (relative paths resolve from repo root). Overrides config/env.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker count. Overrides config/env.",
    )
    parser.add_argument(
        "--submission",
        default=None,
        help="Output submission JSON path. Overrides config/env.",
    )
    parser.add_argument(
        "--skip",
        default=None,
        help="Comma-separated instance_id values to skip (merged with COTEST_SKIP_INSTANCE_IDS).",
    )
    args = parser.parse_args()

    config = load_runner_config(args.config)
    if args.jsonl:
        config["jsonl"] = args.jsonl
    if args.workers is not None:
        config["workers"] = args.workers
    if args.submission:
        config["submission_file"] = args.submission
    config = _normalize_config(config)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(process)d - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    jsonl_file = _resolve_jsonl_path(config["jsonl"])
    logging.info("Loading instance IDs from: %s", jsonl_file)

    test_instance_names = extract_instance_ids_from_jsonl(file_path=jsonl_file)
    if not test_instance_names:
        logging.warning("No instance names extracted. Exiting.")
        return

    configured_instance_ids = config.get("instance_ids")
    if configured_instance_ids:
        available_instance_ids = set(test_instance_names)
        missing_instance_ids = [instance_id for instance_id in configured_instance_ids if instance_id not in available_instance_ids]
        if missing_instance_ids:
            logging.warning("Configured instance IDs not found in benchmark JSONL: %s", ", ".join(missing_instance_ids))
        test_instance_names = [instance_id for instance_id in configured_instance_ids if instance_id in available_instance_ids]
        if not test_instance_names:
            logging.warning("No configured instance IDs were found in the benchmark JSONL. Exiting.")
            return

    skip_ids = _parse_skip_instances()
    if args.skip:
        skip_ids |= {x.strip() for x in args.skip.split(",") if x.strip()}
    valid_instance_names = [n for n in test_instance_names if n not in skip_ids]

    logging.info("Extracted %d instance IDs (%d after skip).", len(test_instance_names), len(valid_instance_names))

    max_workers = config["workers"]
    logging.info("Effective C runner config: %s", json.dumps(config, indent=2, sort_keys=True))
    logging.info(
        "Starting concurrent processing for %d instances with up to %d workers.",
        len(valid_instance_names),
        max_workers,
    )

    submissions = []
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
    try:
        future_to_instance = {executor.submit(process_instance, name, config): name for name in valid_instance_names}

        for future in concurrent.futures.as_completed(future_to_instance):
            instance_name = future_to_instance[future]
            try:
                result = future.result()
                if result:
                    submissions.append(result)
                    logging.info("Successfully processed and got result for instance: %s", instance_name)
                else:
                    logging.warning("Processing for instance %s did not return a result.", instance_name)
            except concurrent.futures.process.BrokenProcessPool as bpp_exc:
                logging.error(
                    "Process pool broken for instance %s. A worker process crashed: %s",
                    instance_name,
                    bpp_exc,
                    exc_info=True,
                )
                logging.warning(
                    "BrokenProcessPool detected. Shutting down executor and continuing with remaining futures."
                )
                break
            except Exception as exc:
                logging.error(
                    "Instance %s generated an exception in the worker process: %s",
                    instance_name,
                    exc,
                    exc_info=True,
                )
    except KeyboardInterrupt:
        logging.warning("\nCaught Ctrl+C! Shutting down workers immediately.")
    finally:
        executor.shutdown(wait=False)
        logging.info("Executor shutdown initiated. Main process exiting.")

    logging.info("All instances have been processed.")

    submission_file_path = submission_path(config["submission_file"])
    os.makedirs(os.path.dirname(submission_file_path), exist_ok=True)
    with open(submission_file_path, "w", encoding="utf-8") as f:
        json.dump(submissions, f, indent=2)
    logging.info("All submissions have been saved to %s", submission_file_path)

if __name__ == "__main__":
    main()
