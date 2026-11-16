import logging
import os
import re
import json
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from datetime import datetime
import concurrent.futures
import traceback
import sys
import signal
import resource

# Adjust the python path to include the root of the project
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from INTENT2FIX.benchmark.utils import get_INTENT2FIX_instance
from INTENT2FIX.benchmark.swebench import create_repository
from INTENT2FIX.artifact_paths import results_path, submission_path
from INTENT2FIX.runtime.dockerenv import DockerEnv
from INTENT2FIX.agent.ablation_code_agent import AblationCodingAgent
from INTENT2FIX.agent.ablation_test_agent import AblationTestAgent
from INTENT2FIX.agent.test_agent import TestEnhancementResult, RepairResult
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


def _load_env() -> None:
    load_dotenv(DOTENV_PATH, override=True)


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

def _extract_reproduction_details_with_agent(problem_statement: str, trajectory: List[Dict[str, Any]]) -> Tuple[str, str, List[Dict[str, str]]]:
    prompt = (
        "You are an expert software engineer reviewing a bug reproduction attempt. "
        "Based on the following problem description and the step-by-step action/observation trajectory, "
        "your task is to produce a JSON object with three keys:\n"
        "1. `reproduction_summary`: A concise, one-paragraph summary of the process of reproducing the bug. "
        "Focus on the actions taken, the commands run, and the final outcome that confirmed the bug's presence.\n"
        "2. `test_command`: A string containing the compelte command(containing compile && excute) that was run to trigger the bug. "
        "A good example: g++ -I . test_issue.cpp -o test_issue && ./test_issue"
        "This should be the command that, when executed, produced the expected error or failure. "
        "Focus on commands that execute tests or run the compiled code, like 'g++ -I . test_issue.cpp -o test_issue && ./test_issue', `ctest`, or `./a.out`. "
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
    
    llm = LiteLLM(model="openrouter/deepseek/deepseek-v3.2", response_format={"type": "json_object"})
    full_prompt = f"{prompt}\n\nProblem: {problem_statement}\n\nTrajectory:\n{trajectory_str}"
    
    try:
        response = llm.generate(full_prompt)
        raw_response_message = response.message if response else ""
        
        json_match = re.search(r"```json\n(.*?)\n```", raw_response_message, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = raw_response_message

        data = json.loads(json_str)
        return data.get("reproduction_summary", ""), data.get("test_command", ""), data.get("test_files", [])
    except Exception as e:
        logger.error(f"Failed to extract reproduction details: {e}")
        return None, None, None

def run_reproduction_step(instance_name: str, model_name: str, docker_env: DockerEnv, output_dir: str) -> Memory | None:
    logger.info(f"--- Starting Reproduction Step for {instance_name} ---")
    instance = get_INTENT2FIX_instance(instance_name)
    repository = create_repository(instance)

    with docker_env:
        agent = ReproductionAgent.create(
            repository=repository,
            model=model_name,
            temperature=0.2,
            max_tokens=4000,
            response_format=LLMResponseFormat.REACT,
            docker_env=docker_env,
        )

        loop = ReproductionLoop(
            agent=agent,
            docker_env=docker_env,
            repository=repository,
            repo_path=repository.repo_path,
            max_turns=20
        )

        repo_structure = f"\nRepo_structure'''\n{get_repo_structure(repository.repo_path)}"
        try:
            repro_result = loop.run(issue=instance["problem_statement"] + repo_structure)
        except Exception as e:
            logger.error(f"Reproduction loop failed: {e}")
            return None

    if repro_result.full_trajectory:
        persist_path = os.path.join(output_dir, f"repro_trajectory_{instance_name}.json")
        os.makedirs(os.path.dirname(persist_path), exist_ok=True)
        with open(persist_path, "w") as f:
            json.dump(repro_result.full_trajectory.model_dump(exclude_none=True), f, indent=2)

    if repro_result.is_reproduced:
        trajectory_data = [
            {"action": step.action.model_dump(), "observation": step.observation.message if step.observation else None}
            for node in repro_result.full_trajectory.get_all_nodes()
            for step in node.action_steps
        ]
        
        summary, test_command, test_files = _extract_reproduction_details_with_agent(instance["problem_statement"], trajectory_data)
        if not summary or not test_files:
            return None

        return Memory(
            instance_id=instance_name,
            problem_statement=instance["problem_statement"],
            reproduction_summary=summary,
            golden_patch=instance.get("patch", ""),
            test_command=test_command,
            test_files=test_files,
            reproduction_result=repro_result.reproduction_output or "",
            expected_reproduction_result=repro_result.expected_output or ""
        )
    return None

def run_repair_step(instance_name: str, model_name: str, docker_env: DockerEnv, memory: Memory, output_dir: str) -> str | None:
    """
    Runs the bug repair step using the provided memory from the reproduction step.
    Note: This version uses AblationCodingAgent and includes the iterative loop.
    """

    logger.info(f"--- Starting Ablation Repair Step for {instance_name} ---")
    _load_env()
    
    instance = get_INTENT2FIX_instance(instance_name)
    repository = create_repository(instance)
    index_store_dir = os.getenv("INDEX_STORE_DIR")
    os.environ["SWEBENCH_INSTANCE_NAME"] = instance_name
    
    code_index = CodeIndex.from_index_name(
        index_name=instance_name,
        file_repo=repository,
        index_store_dir=index_store_dir
    )

    with docker_env:
        file_context = FileContext(repo=repository, runtime=docker_env)

        if memory.test_files:
            for test_file in memory.test_files:
                relative_path = test_file['path'].lstrip('/')
                full_container_path = os.path.join(docker_env.work_dir, relative_path)
                docker_env.write_file(full_container_path, test_file['content'])
                full_host_path = repository.get_full_path(relative_path)
                os.makedirs(os.path.dirname(full_host_path), exist_ok=True)
                with open(full_host_path, "w") as f:
                    f.write(test_file['content'])
                file_context.add_file(relative_path, content=test_file['content'])

        # --- USE ABLATION AGENT ---
        agent = AblationCodingAgent.create(
            repository=repository,
            code_index=code_index,
            model=model_name,
            temperature=0.2,
            max_tokens=50000,
            response_format=LLMResponseFormat.REACT,
            message_history_type=MessageHistoryType.REACT,
            runtime=docker_env,
        )

        repo_structure = show_code_structure(DirectoryAnalyzer(CodeParser()).analyze(repository.repo_path))
        message = f"<task>\nProblem Statement:\n{memory.problem_statement}\n<task>\n"
        
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
            message += (
                f"Use the following command to run the PoC test:\n"
                f"{command_block}\n\n"
            )
        
        message += f"Repo Structure:\n{repo_structure}\n\nPlease provide a patch to fix the issue."

        persist_path = os.path.join(output_dir, f"repair_trajectory_{instance_name}.json")
        loop = AgenticLoop.create(
            agent=agent,
            file_context=file_context,
            message=message,
            repository=repository,
            runtime=docker_env,
            code_index=code_index,
            persist_path=persist_path,
            max_iterations=60
        )
        loop.memory_module = None

        logger.info("Running ablated repair loop (grep-only search)...")
        try:
            final_trajectory = loop.run()
        except Exception as e:
            logger.error(f"Repair loop failed with exception: {e}", exc_info=True)
            return None
        
        try:
            solution_patch = final_trajectory.file_context.generate_git_patch()
            has_patches = solution_patch and solution_patch.strip()
            if final_trajectory and has_patches:
                logger.info("Repair successful! Solution patch generated.")
                patch_path = os.path.join(output_dir, "solution.patch")
                with open(patch_path, "w") as f:
                    f.write(solution_patch)
                
                # Check if basic tests passed, then trigger TestAgent for test enhancement
                if _basic_tests_passed(final_trajectory.file_context):
                    logger.info("Basic tests passed. Triggering TestAgent for test enhancement...")
                    
                    solution_before_iter_path = os.path.join(output_dir, "solution_before_iteration.patch")
                    with open(solution_before_iter_path, "w") as f:
                        f.write(solution_patch)
                    
                    max_iterations = 1
                    current_patch = solution_patch
                    current_file_context = final_trajectory.file_context
                    test_model_name = model_name
                    
                    for iteration in range(max_iterations):
                        logger.info(f"Test Enhancement iteration {iteration + 1}/{max_iterations}")
                        
                        docker_env.reuse_existing = True
                        enhancement_result = run_test_enhancement_step(
                            instance_name=instance_name,
                            model_name=test_model_name,
                            docker_env=docker_env,
                            file_context=current_file_context,
                            memory=memory,
                            output_dir=output_dir,
                            iteration=iteration + 1
                        )
                        
                        if enhancement_result.success:
                            logger.info("Test enhancement successful! All tests passed.")
                            solution_patch = enhancement_result.enhanced_patch
                            enhanced_patch_path = os.path.join(output_dir, f"enhanced_solution_iter{iteration + 1}.patch")
                            with open(enhanced_patch_path, "w") as f:
                                f.write(solution_patch)
                            with open(patch_path, "w") as f:
                                f.write(solution_patch)
                            break
                        else:
                            logger.warning(f"Test enhancement iteration {iteration + 1} failed. Attempting repair...")
                            
                            if not enhancement_result.failed_tests and not enhancement_result.failure_analysis:
                                break
                            
                            docker_env.reuse_existing = True
                            repair_result = run_iterative_repair_step(
                                instance_name=instance_name,
                                model_name=model_name,
                                docker_env=docker_env,
                                file_context=current_file_context,
                                memory=memory,
                                test_failures=enhancement_result.failed_tests,
                                failure_analysis=enhancement_result.failure_analysis or "",
                                output_dir=output_dir,
                                iteration=iteration + 1
                            )
                            
                            if repair_result.success:
                                current_patch = repair_result.enhanced_patch
                                current_file_context = repair_result.file_context
                                solution_patch = current_patch
                                repaired_patch_path = os.path.join(output_dir, f"repaired_solution_iter{iteration + 1}.patch")
                                with open(repaired_patch_path, "w") as f:
                                    f.write(current_patch)
                                with open(patch_path, "w") as f:
                                    f.write(current_patch)
                            else:
                                break
                return solution_patch
            return None
        except Exception as e:
            logger.error(f"Error processing solution patch: {e}", exc_info=True)
            return None

def _basic_tests_passed(file_context: FileContext) -> bool:
    try:
        verification = getattr(file_context, "verification", {}) or {}
        change_seq = getattr(file_context, "change_seq", 0)
        reg = verification.get("regression") or {}
        single = verification.get("single_test") or {}
        reg_ok = bool(reg.get("passed")) and reg.get("change_seq") == change_seq
        single_ok = bool(single.get("passed")) and single.get("change_seq") == change_seq
        return reg_ok and single_ok
    except Exception as e:
        logger.warning(f"Error checking basic tests passed: {e}")
        return False

def _format_test_failures(failed_tests: List[Dict[str, str]]) -> str:
    if not failed_tests:
        return ""
    formatted = "The following tests failed:\n\n"
    for i, test in enumerate(failed_tests, 1):
        formatted += f"Test {i}:\n"
        formatted += f"  File: {test.get('file', 'Unknown')}\n"
        formatted += f"  Test Name: {test.get('test_name', 'Unknown')}\n"
        formatted += f"  Error: {test.get('error', 'Unknown error')}\n"
        if test.get('expected'):
            formatted += f"  Expected: {test.get('expected')}\n"
        if test.get('actual'):
            formatted += f"  Actual: {test.get('actual')}\n"
        formatted += "\n"
    return formatted

def run_test_enhancement_step(
    instance_name: str,
    model_name: str,
    docker_env: DockerEnv,
    file_context: FileContext,
    memory: Memory,
    output_dir: str,
    iteration: int = 1
) -> TestEnhancementResult:
    logger.info(f"--- Starting Ablation Test Enhancement Step (iteration {iteration}) for {instance_name} ---")
    _load_env()
    instance = get_INTENT2FIX_instance(instance_name)
    repository = create_repository(instance)
    os.environ["SWEBENCH_INSTANCE_NAME"] = instance_name
    index_store_dir = os.getenv("INDEX_STORE_DIR")
    code_index = CodeIndex.from_index_name(instance_name, repository, index_store_dir)
    
    with docker_env:
        test_file_context = FileContext(repo=repository, runtime=docker_env)
        for file_path, context_file in file_context._files.items():
            try:
                file_content = context_file.content
                base_content = getattr(context_file, "_cached_base_content", None)
                if base_content is None and context_file.patch:
                    base_content = context_file.get_base_content()
                test_context_file = test_file_context.add_file(file_path, content=file_content)
                if context_file.patch:
                    test_context_file.patch = context_file.patch
                    if base_content is not None:
                        test_context_file._cached_base_content = base_content
                full_container_path = os.path.join(docker_env.work_dir, file_path)
                docker_env.write_file(full_container_path, file_content)
            except Exception as e:
                logger.warning(f"Failed to copy {file_path}: {e}")

        # Use AblationTestAgent
        agent = AblationTestAgent.create(
            repository=repository,
            code_index=code_index,
            model=model_name,
            temperature=0.2,
            max_tokens=50000,
            response_format=LLMResponseFormat.REACT,
            message_history_type=MessageHistoryType.REACT,
            runtime=docker_env,
        )
        
        current_patch = file_context.generate_git_patch()
        message = (
            f"<task>\nTest Enhancement Task (Iteration {iteration})\n\n"
            f"Problem Statement:\n{memory.problem_statement}\n\n"
        )
        if current_patch:
            message += f"<Current_Patch>\n{current_patch}\n</Current_Patch>\n\n"
        
        message += "Please generate comprehensive test cases to enhance the robustness of the patch.\n</task>"
        
        persist_path = os.path.join(output_dir, f"test_enhancement_trajectory_{instance_name}_iter{iteration}.json")
        loop = AgenticLoop.create(
            agent=agent,
            file_context=test_file_context,
            message=message,
            repository=repository,
            runtime=docker_env,
            code_index=code_index,
            persist_path=persist_path,
            max_iterations=40
        )
        loop.memory_module = None
        final_trajectory = loop.run()
        
        enhanced_patch = final_trajectory.file_context.generate_git_patch()
        failed_tests = []
        failure_analysis = None
        new_test_files = []
        
        if final_trajectory and final_trajectory.action_steps:
            last_action = final_trajectory.action_steps[-1].action
            if hasattr(last_action, 'failure_analysis'):
                failure_analysis = last_action.failure_analysis
                if "failed" in failure_analysis.lower():
                    failed_tests.append({"file": "Unknown", "test_name": "See analysis", "error": failure_analysis})
        
        verification = getattr(final_trajectory.file_context, "verification", {}) or {}
        success = bool(verification.get("regression", {}).get("passed")) and \
                  bool(verification.get("single_test", {}).get("passed")) and not failure_analysis
                  
        return TestEnhancementResult(
            success=success,
            enhanced_patch=enhanced_patch,
            failed_tests=failed_tests,
            failure_analysis=failure_analysis,
            new_test_files=[],
            file_context=final_trajectory.file_context
        )

def run_iterative_repair_step(
    instance_name: str,
    model_name: str,
    docker_env: DockerEnv,
    file_context: FileContext,
    memory: Memory,
    test_failures: List[Dict[str, str]],
    failure_analysis: str,
    output_dir: str,
    iteration: int = 1
) -> RepairResult:
    logger.info(f"--- Starting Ablation Iterative Repair Step (iteration {iteration}) for {instance_name} ---")
    _load_env()
    instance = get_INTENT2FIX_instance(instance_name)
    repository = create_repository(instance)
    os.environ["SWEBENCH_INSTANCE_NAME"] = instance_name
    index_store_dir = os.getenv("INDEX_STORE_DIR")
    code_index = CodeIndex.from_index_name(instance_name, repository, index_store_dir)
    
    with docker_env:
        repair_file_context = FileContext(repo=repository, runtime=docker_env)
        for file_path, context_file in file_context._files.items():
            try:
                file_content = context_file.content
                base_content = getattr(context_file, "_cached_base_content", None)
                if base_content is None and context_file.patch:
                    base_content = context_file.get_base_content()
                repair_context_file = repair_file_context.add_file(file_path, content=file_content)
                if context_file.patch:
                    repair_context_file.patch = context_file.patch
                    if base_content is not None:
                        repair_context_file._cached_base_content = base_content
                full_container_path = os.path.join(docker_env.work_dir, file_path)
                docker_env.write_file(full_container_path, file_content)
            except Exception as e:
                logger.warning(f"Failed to copy {file_path}: {e}")

        # Use AblationCodingAgent for iterative repair too
        agent = AblationCodingAgent.create(
            repository=repository,
            code_index=code_index,
            model=model_name,
            temperature=0.2,
            max_tokens=50000,
            response_format=LLMResponseFormat.REACT,
            message_history_type=MessageHistoryType.REACT,
            runtime=docker_env,
        )
        
        current_patch = file_context.generate_git_patch()
        formatted_failures = _format_test_failures(test_failures)
        message = (
            f"<task>\nIterative Repair Task (Iteration {iteration})\n\n"
            f"Problem Statement:\n{memory.problem_statement}\n\n"
        )
        if current_patch:
            message += f"<Current_Patch>\n{current_patch}\n</Current_Patch>\n\n"
        message += f"<Failed_Tests>\n{formatted_failures}\n</Failed_Tests>\n\n"
        if failure_analysis:
            message += f"<Failure_Analysis>\n{failure_analysis}\n</Failure_Analysis>\n\n"
        message += "Please fix the code to make the failing tests pass.\n</task>"
        
        persist_path = os.path.join(output_dir, f"iterative_repair_trajectory_{instance_name}_iter{iteration}.json")
        loop = AgenticLoop.create(
            agent=agent,
            file_context=repair_file_context,
            message=message,
            repository=repository,
            runtime=docker_env,
            code_index=code_index,
            persist_path=persist_path,
            max_iterations=40
        )
        loop.memory_module = None
        final_trajectory = loop.run()
        
        enhanced_patch = final_trajectory.file_context.generate_git_patch()
        verification = getattr(final_trajectory.file_context, "verification", {}) or {}
        success = bool(verification.get("regression", {}).get("passed")) and \
                  bool(verification.get("single_test", {}).get("passed"))
                  
        return RepairResult(
            success=success,
            enhanced_patch=enhanced_patch,
            file_context=final_trajectory.file_context
        )

def _signal_handler(signum, frame):
    raise SystemExit(f"Process terminated by signal {signum}")

def process_instance(instance_name: str) -> Dict[str, str] | None:
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    
    try:
        _load_env()
        os.environ["SWEBENCH_INSTANCE_NAME"] = instance_name
        repair_model_name = 'deepseek-v3-0324'
        repro_model_name = 'deepseek-v3-0324'
        sanitized_model_name = repair_model_name.replace('/', '_')
        
        # --- Output directory for ablation ---
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(
            results_path(f"results_ablation_grep_only_{sanitized_model_name}_V3"),
            instance_name,
        )
        os.makedirs(result_dir, exist_ok=True)
        log_file_path = os.path.join(result_dir, f'repair_run_{time_stamp}.log')

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()

        formatter = logging.Formatter('%(asctime)s - %(process)d - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"===== Starting Ablation (Grep-Only) repair for: {instance_name} =====")
        instance_str = parse_instance_string(instance_name)
        image_name = f'mswebench/{instance_str["org"]}_m_{instance_str["repo"]}:pr-{instance_str["number"]}'.lower()
        
        # Stage 1: Reproduction (Already Grep-only)
        repro_docker_env = DockerEnv(
            image_name=image_name,
            container_name=f'repro-{instance_name}-{time_stamp}',
            work_dir=f'/home/{instance_str["repo"]}',
        )
        memory_instance = run_reproduction_step(instance_name, repro_model_name, repro_docker_env, result_dir)

        if not memory_instance:
            instance = get_INTENT2FIX_instance(instance_name)
            memory_instance = Memory(
                instance_id=instance_name,
                problem_statement=instance["problem_statement"],
                reproduction_summary="",
                test_files=[]
            )

        # Stage 2: Repair (Ablated)
        repair_docker_env = DockerEnv(
            image_name=image_name,
            container_name=f'repair-{instance_name}-{time_stamp}',
            work_dir=f'/home/{instance_str["repo"]}',
        )
        solution_patch = run_repair_step(instance_name, repair_model_name, repair_docker_env, memory_instance, result_dir)
        
        if solution_patch:
            return {"instance_id": instance_name, "model_patch": solution_patch}
        return None
    except Exception as e:
        logger.error(f"Error in process_instance {instance_name}: {e}", exc_info=True)
        return None

def main():
    _load_env()
    if os.environ.get('OPENROUTER_API_KEY') and not os.environ.get('OPENAI_API_KEY'):
        os.environ['OPENAI_API_KEY'] = os.environ['OPENROUTER_API_KEY']
    
    jsonl_file = 'INTENT2FIX/benchmark/multi_swebench_cpp_change.jsonl'
    instance_ids = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            instance_ids.append(json.loads(line)['instance_id'])
    

    # Full instances
    valid_instances = instance_ids
    # Selection for experiment
    # target_instances = [
    #     "simdjson__simdjson-1896",
    #     "simdjson__simdjson-1624",
    #     "simdjson__simdjson-524",
    #     "nlohmann__json-4537",
    #     "nlohmann__json-4525",
    #     "nlohmann__json-3601",
    #     "nlohmann__json-3109",
    #     "nlohmann__json-3073",
    #     "nlohmann__json-2332",
    #     "nlohmann__json-1769",
    #     "nlohmann__json-708",
    #     "nlohmann__json-807",
    #     "nlohmann__json-545",
    #     "nlohmann__json-18",
    #     "nlohmann__json-508",
    #     "fmtlib__fmt-4310",
    #     "fmtlib__fmt-4055",
    #     "fmtlib__fmt-4057",
    #     "fmtlib__fmt-3732",
    #     "fmtlib__fmt-3729",
    #     "fmtlib__fmt-3713",
    #     "fmtlib__fmt-3555",
    #     "fmtlib__fmt-3569",
    #     "fmtlib__fmt-3248",
    #     "fmtlib__fmt-2940",
    #     "fmtlib__fmt-2317",
    #     "fmtlib__fmt-2394",
    #     "fmtlib__fmt-2070",
    #     "fmtlib__fmt-2158",
    #     "fmtlib__fmt-2292",
    #     "fmtlib__fmt-493",
    #     "catchorg__Catch2-2288"
    # ] 
    # valid_instances = [name for name in instance_ids if name not in target_instances]

    max_workers = 12
    submissions = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_instance = {executor.submit(process_instance, name): name for name in valid_instances}
        for future in concurrent.futures.as_completed(future_to_instance):
            res = future.result()
            if res:
                submissions.append(res)

    submission_file_path = submission_path("submission_ablation.json")
    os.makedirs(os.path.dirname(submission_file_path), exist_ok=True)
    with open(submission_file_path, "w") as f:
        json.dump(submissions, f, indent=2)

if __name__ == "__main__":
    main()
