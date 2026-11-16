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
from INTENT2FIX.agent.ablation_no_intent_agent import NoIntentCodingAgent
from INTENT2FIX.agent.test_agent import TestAgent, TestEnhancementResult, RepairResult
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
    # Force .env to override any pre-existing exported environment variables.
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

def run_reproduction_step(instance_name: str, model_name: str, docker_env: DockerEnv, output_dir: str) -> Memory | None:
    """
    Runs the reproduction step and returns a Memory object if successful.
    """
    logger.info(f"--- Starting Reproduction Step for {instance_name} ---")
    
    instance = get_INTENT2FIX_instance(instance_name)
    repository = create_repository(instance)

    with docker_env:
        logger.info(f"Docker container '{docker_env.container_name}' is running for reproduction.")
        
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
        
        summary, test_command, test_files = _extract_reproduction_details_with_agent(instance["problem_statement"], trajectory_data)

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

def run_repair_step(instance_name: str, model_name: str, docker_env: DockerEnv, memory: Memory, output_dir: str) -> str | None:
    """
    Runs the bug repair step using the provided memory from the reproduction step.
    Note: This version does NOT use MemoryModule for searching past experiences.
    """

    logger.info(f"--- Starting NoIntent Repair Step for {instance_name} ---")
    _load_env()
    
    instance = get_INTENT2FIX_instance(instance_name)
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
        instance_str = parse_instance_string(instance_name)

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

        # --- USE NO-INTENT ABLATION AGENT ---
        agent = NoIntentCodingAgent.create(
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
            max_iterations=60
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
                
                # Check if basic tests passed, then trigger TestAgent for test enhancement
                if _basic_tests_passed(final_trajectory.file_context):
                    logger.info("Basic tests passed. Triggering TestAgent for test enhancement...")
                    
                    # Save the patch before iteration (iter0)
                    solution_before_iter_path = os.path.join(output_dir, "solution_before_iteration.patch")
                    with open(solution_before_iter_path, "w") as f:
                        f.write(solution_patch)
                    logger.info(f"Solution patch before iteration saved to {solution_before_iter_path}")
                    
                    # Iterative repair loop
                    max_iterations = 1  # Maximum number of iterations
                    current_patch = solution_patch
                    current_file_context = final_trajectory.file_context
                    test_model_name = model_name  # Can use different model for test enhancement
                    
                    for iteration in range(max_iterations):
                        logger.info(f"Test Enhancement iteration {iteration + 1}/{max_iterations}")
                        
                        # Reuse the existing repair container for test enhancement
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
                            # Save enhanced patch
                            enhanced_patch_path = os.path.join(output_dir, f"enhanced_solution_iter{iteration + 1}.patch")
                            with open(enhanced_patch_path, "w") as f:
                                f.write(solution_patch)
                            logger.info(f"Enhanced solution patch saved to {enhanced_patch_path}")
                            # Update solution.patch with the enhanced version
                            with open(patch_path, "w") as f:
                                f.write(solution_patch)
                            logger.info(f"Updated solution.patch with enhanced version")
                            break
                        else:
                            logger.warning(f"Test enhancement iteration {iteration + 1} failed. Attempting repair...")
                            
                            if not enhancement_result.failed_tests and not enhancement_result.failure_analysis:
                                logger.warning("No failure information available. Stopping iteration.")
                                break
                            
                            # Trigger CodeAgent for iterative repair
                            # Reuse the existing repair container for iterative repair
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
                                solution_patch = current_patch  # Update solution_patch for return value
                                # Save repaired patch
                                repaired_patch_path = os.path.join(output_dir, f"repaired_solution_iter{iteration + 1}.patch")
                                with open(repaired_patch_path, "w") as f:
                                    f.write(current_patch)
                                logger.info(f"Repaired solution patch saved to {repaired_patch_path}")
                                # Update solution.patch with the repaired version
                                with open(patch_path, "w") as f:
                                    f.write(current_patch)
                                logger.info(f"Updated solution.patch with repaired version")
                            else:
                                logger.error("Iterative repair failed. Stopping.")
                                break
                else:
                    logger.warning("Basic tests did not pass. Skipping test enhancement.")
                
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

def _basic_tests_passed(file_context: FileContext) -> bool:
    """
    检查基本测试是否通过（RunRegression 和 RunSingleTest）
    """
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
    """
    格式化失败的测试信息，供 CodeAgent 使用
    """
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
        if test.get('location'):
            formatted += f"  Location: {test.get('location')}\n"
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
    """
    运行测试增强步骤，生成对抗测试和边界值测试用例
    """
    logger.info(f"--- Starting Test Enhancement Step (iteration {iteration}) for {instance_name} ---")
    _load_env()
    
    instance = get_INTENT2FIX_instance(instance_name)
    repository = create_repository(instance)
    
    # Set SWEBENCH_INSTANCE_NAME for entity_tools
    os.environ["SWEBENCH_INSTANCE_NAME"] = instance_name
    
    index_store_dir = os.getenv("INDEX_STORE_DIR")
    if not index_store_dir:
        raise ValueError("INDEX_STORE_DIR environment variable is not set. Please set it in .env file.")
    
    # Load the index
    code_index = CodeIndex.from_index_name(
        index_name=instance_name,
        file_repo=repository,
        index_store_dir=index_store_dir
    )
    
    # Note: docker_env is already in a 'with' context from run_repair_step, so we don't need another one
    logger.info(f"Docker container '{docker_env.container_name}' is running for test enhancement.")
    
    # Create a new file context from the existing one (copy state)
    test_file_context = FileContext(repo=repository, runtime=docker_env)
    
    for file_path, context_file in file_context._files.items():
        try:
            # Get content with patch applied
            file_content = context_file.content
            
            # Preserve original base content
            base_content = None
            if context_file.patch:
                base_content = getattr(context_file, "_cached_base_content", None)
                if base_content is None:
                    try:
                        base_content = context_file.get_base_content()
                    except Exception as e:
                        logger.warning(f"Failed to get base content from context_file for {file_path}: {e}")
                        if repository.file_exists(file_path):
                            base_content = repository.get_file_content(file_path)
            
            # Add to test_file_context
            test_context_file = test_file_context.add_file(file_path, content=file_content)
            
            # Preserve the patch
            if context_file.patch:
                test_context_file.patch = context_file.patch
                if base_content is not None:
                    test_context_file._cached_base_content = base_content
                    test_context_file._initial_patch = None
            
            # Write to container
            full_container_path = os.path.join(docker_env.work_dir, file_path)
            docker_env.write_file(full_container_path, file_content)
            
            logger.debug(f"Copied patched file {file_path} to test_file_context (with patch preserved) and container")
        except Exception as e:
            logger.warning(f"Failed to copy {file_path} to test_file_context: {e}")

    # Also copy test files from memory
    if memory.test_files:
        for test_file in memory.test_files:
            relative_path = test_file['path'].lstrip('/')
            full_container_path = os.path.join(docker_env.work_dir, relative_path)
            docker_env.write_file(full_container_path, test_file['content'])
            full_host_path = repository.get_full_path(relative_path)
            os.makedirs(os.path.dirname(full_host_path), exist_ok=True)
            with open(full_host_path, "w") as f:
                f.write(test_file['content'])
            if relative_path not in test_file_context._files:
                test_file_context.add_file(relative_path, content=test_file['content'])
    
    # Create TestAgent
    agent = TestAgent.create(
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
    current_patch = file_context.generate_git_patch()
    
    # Build test enhancement message
    message = (
        f"<task>\n"
        f"Test Enhancement Task (Iteration {iteration})\n\n"
        f"The following code patch has been applied and has passed basic PoC and regression tests. "
        f"Your task is to generate comprehensive test cases to enhance the robustness of this patch.\n\n"
        f"Problem Statement:\n{memory.problem_statement}\n\n"
    )
    
    if current_patch:
        message += f"<Current_Patch>\n{current_patch}\n</Current_Patch>\n\n"
    
    if memory.test_files:
        poc_test_paths = [f"  - {test_file['path']}" for test_file in memory.test_files]
        poc_block = "<Poc_test>\n" + "\n".join(poc_test_paths) + "\n</Poc_test>"
        message += (
            f"The following PoC test file(s) exist:\n"
            f"{poc_block}\n\n"
        )
    
    if memory.test_command:
        command_block = f"<Poc_command>\n{memory.test_command}\n</Poc_command>"
        message += (
            f"Use the following command to run the PoC test:\n"
            f"{command_block}\n\n"
        )
    
    message += (
        f"<Requirements>\n"
        f"1. Generate adversarial test cases that try to find weaknesses in the patch\n"
        f"2. Generate boundary value test cases (extreme values, edge cases, invalid inputs, error paths, overflow/underflow, empty/null values)\n"
        f"3. Ensure at least some test cases would fail on the buggy version and pass only because of the fix\n"
        f"4. Run all new tests and verify they pass\n"
        f"5. If any tests fail, provide detailed failure analysis\n"
        f"</Requirements>\n\n"
        f"Repo Structure:\n{repo_structure}\n\n"
        f"Please generate comprehensive test cases to enhance the robustness of the patch.\n"
        f"</task>\n"
    )
    
    persist_path = os.path.join(output_dir, f"test_enhancement_trajectory_{instance_name}_iter{iteration}.json")
    os.makedirs(os.path.dirname(persist_path), exist_ok=True)
    metadata = {
        "instance_id": instance_name,
        "problem_statement": memory.problem_statement,
        "iteration": iteration,
        "result_persist_path": output_dir,
    }
    
    # Create loop
    loop = AgenticLoop.create(
        agent=agent,
        file_context=test_file_context,
        message=message,
        repository=repository,
        runtime=docker_env,
        code_index=code_index,
        metadata=metadata,
        persist_path=persist_path,
        max_iterations=40
    )
    
    loop.memory_module = None
    
    logger.info("Running test enhancement loop...")
    try:
        final_trajectory = loop.run()
    except Exception as e:
        logger.error(f"Test enhancement loop failed with exception: {e}", exc_info=True)
        return TestEnhancementResult(
            success=False,
            enhanced_patch=None,
            failed_tests=[],
            failure_analysis=f"Test enhancement loop failed: {str(e)}",
            file_context=test_file_context
        )
    
    try:
        enhanced_patch = final_trajectory.file_context.generate_git_patch()
        
        # Check if tests passed
        failed_tests = []
        failure_analysis = None
        
        final_node = final_trajectory
        if final_node and final_node.action_steps:
            last_action = final_node.action_steps[-1].action
            action_name = last_action.name
            
            if action_name == "TestEnhancementNeedRepair":
                success = False
                failure_analysis = getattr(last_action, 'failure_analysis', "No analysis provided")
                failed_tests.append({"error": failure_analysis})
            elif action_name == "TestEnhancementPassed":
                success = True
            else:
                # Fallback
                if hasattr(last_action, 'failure_analysis') and last_action.failure_analysis:
                    failure_analysis = last_action.failure_analysis
                    failed_tests.append({"error": failure_analysis})
                verification = getattr(final_trajectory.file_context, "verification", {}) or {}
                single_test = verification.get("single_test") or {}
                success = bool(single_test.get("passed", True)) and not failure_analysis

        if final_trajectory and enhanced_patch:
            return TestEnhancementResult(
                success=success,
                enhanced_patch=enhanced_patch,
                failed_tests=failed_tests,
                failure_analysis=failure_analysis,
                file_context=final_trajectory.file_context
            )
        else:
            return TestEnhancementResult(
                success=False,
                enhanced_patch=None,
                failed_tests=[],
                failure_analysis="No enhanced patch generated",
                file_context=test_file_context
            )
    except Exception as e:
        logger.error(f"Error processing test enhancement result: {e}", exc_info=True)
        return TestEnhancementResult(
            success=False,
            enhanced_patch=None,
            failed_tests=[],
            failure_analysis=f"Error processing result: {str(e)}",
            file_context=test_file_context
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
    """
    运行迭代修复步骤，根据测试失败信息修复代码
    """
    logger.info(f"--- Starting NoIntent Iterative Repair Step (iteration {iteration}) for {instance_name} ---")
    _load_env()
    
    instance = get_INTENT2FIX_instance(instance_name)
    repository = create_repository(instance)
    
    os.environ["SWEBENCH_INSTANCE_NAME"] = instance_name
    index_store_dir = os.getenv("INDEX_STORE_DIR")
    code_index = CodeIndex.from_index_name(
        index_name=instance_name,
        file_repo=repository,
        index_store_dir=index_store_dir
    )
    
    repair_file_context = FileContext(repo=repository, runtime=docker_env)
    
    for file_path, context_file in file_context._files.items():
        try:
            file_content = context_file.content
            base_content = None
            if context_file.patch:
                base_content = getattr(context_file, "_cached_base_content", None)
                if base_content is None:
                    try:
                        base_content = context_file.get_base_content()
                    except:
                        if repository.file_exists(file_path):
                            base_content = repository.get_file_content(file_path)
            
            repair_context_file = repair_file_context.add_file(file_path, content=file_content)
            if context_file.patch:
                repair_context_file.patch = context_file.patch
                if base_content is not None:
                    repair_context_file._cached_base_content = base_content
                    repair_context_file._initial_patch = None
            
            full_container_path = os.path.join(docker_env.work_dir, file_path)
            docker_env.write_file(full_container_path, file_content)
        except Exception as e:
            logger.warning(f"Failed to copy {file_path} to repair_file_context: {e}")
    
    if memory.test_files:
        for test_file in memory.test_files:
            relative_path = test_file['path'].lstrip('/')
            full_container_path = os.path.join(docker_env.work_dir, relative_path)
            docker_env.write_file(full_container_path, test_file['content'])
            full_host_path = repository.get_full_path(relative_path)
            os.makedirs(os.path.dirname(full_host_path), exist_ok=True)
            with open(full_host_path, "w") as f:
                f.write(test_file['content'])
            repair_file_context.add_file(relative_path, content=test_file['content'])
    
    # --- USE NO-INTENT ABLATION AGENT ---
    agent = NoIntentCodingAgent.create(
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
    current_patch = file_context.generate_git_patch()
    formatted_failures = _format_test_failures(test_failures)
    
    message = (
        f"<task>\n"
        f"Iterative Repair Task (Iteration {iteration})\n\n"
        f"The following test cases failed after test enhancement. "
        f"Your task is to fix the code to make these tests pass.\n\n"
        f"Problem Statement:\n{memory.problem_statement}\n\n"
    )
    
    if current_patch:
        message += f"<Current_Patch>\n{current_patch}\n</Current_Patch>\n\n"
    
    message += f"<Failed_Tests>\n{formatted_failures}\n</Failed_Tests>\n\n"
    
    if failure_analysis:
        message += f"<Failure_Analysis>\n{failure_analysis}\n</Failure_Analysis>\n\n"
    
    message += (
        f"<Requirements>\n"
        f"1. Analyze why the tests failed\n"
        f"2. Fix the code (NOT the tests) to make the failing tests pass\n"
        f"3. Ensure your fix does not break existing functionality\n"
        f"4. Run RunRegression to verify no regressions\n"
        f"5. Run RunSingleTest to verify the failing tests now pass\n"
        f"</Requirements>\n\n"
        f"Repo Structure:\n{repo_structure}\n\n"
        f"Please fix the code to make the failing tests pass.\n"
        f"</task>\n"
    )
    
    persist_path = os.path.join(output_dir, f"iterative_repair_trajectory_{instance_name}_iter{iteration}.json")
    os.makedirs(os.path.dirname(persist_path), exist_ok=True)
    metadata = {
        "instance_id": instance_name,
        "problem_statement": memory.problem_statement,
        "iteration": iteration,
        "result_persist_path": output_dir,
    }
    
    loop = AgenticLoop.create(
        agent=agent,
        file_context=repair_file_context,
        message=message,
        repository=repository,
        runtime=docker_env,
        code_index=code_index,
        metadata=metadata,
        persist_path=persist_path,
        max_iterations=40
    )
    
    loop.memory_module = None
    
    logger.info("Running iterative repair loop...")
    try:
        final_trajectory = loop.run()
    except Exception as e:
        logger.error(f"Iterative repair loop failed with exception: {e}", exc_info=True)
        return RepairResult(success=False, enhanced_patch=None, file_context=repair_file_context)
    
    try:
        enhanced_patch = final_trajectory.file_context.generate_git_patch()
        verification = getattr(final_trajectory.file_context, "verification", {}) or {}
        reg = verification.get("regression") or {}
        single = verification.get("single_test") or {}
        success = bool(reg.get("passed")) and bool(single.get("passed"))
        
        return RepairResult(
            success=success,
            enhanced_patch=enhanced_patch,
            file_context=final_trajectory.file_context
        )
    except Exception as e:
        logger.error(f"Error processing iterative repair result: {e}", exc_info=True)
        return RepairResult(success=False, enhanced_patch=None, file_context=repair_file_context)

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
    # #region agent log
    try:
        with open('./INTENT2FIX/.cursor/debug.log', 'a') as f:
            import json, time
            f.write(json.dumps({"sessionId":"debug-session","runId":"pre-fix","hypothesisId":"E","location":"run_ablation_no_intent.py:361","message":"signal received","data":{"signal":signum,"pid":os.getpid(),"frame":str(frame)},"timestamp":int(time.time()*1000)}) + '\n')
    except: pass
    # #endregion
    raise SystemExit(f"Process terminated by signal {signum}")

def process_instance(instance_name: str) -> Dict[str, str] | None:
    """
    Processes a single instance, running repair step directly (reproduction skipped).
    Configures logging to a file specific to the instance.
    """
    # Register signal handlers to catch system-level terminations
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    # #region agent log
    try:
        with open('./INTENT2FIX/.cursor/debug.log', 'a') as f:
            import json, time
            mem_info = resource.getrusage(resource.RUSAGE_SELF)
            f.write(json.dumps({"sessionId":"debug-session","runId":"pre-fix","hypothesisId":"B","location":"run_ablation_no_intent.py:375","message":"process_instance entry","data":{"instance_name":instance_name,"pid":os.getpid(),"mem_mb":mem_info.ru_maxrss/1024},"timestamp":int(time.time()*1000)}) + '\n')
    except: pass
    # #endregion
    try:
        _load_env()
        # Set SWEBENCH_INSTANCE_NAME early so it's available for ClangdParser and other components
        os.environ["SWEBENCH_INSTANCE_NAME"] = instance_name
        model_name = 'deepseek-v3-0324'
        repro_model_name = 'deepseek-v3-0324'
        sanitized_model_name = model_name.replace('/', '_')
        
        # --- Per-instance logging and result directory setup ---
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(
            results_path(f"results_ablation_no_intent_{sanitized_model_name}"),
            instance_name,
        )
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

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        # --- End of per-instance logging setup ---

        logger.info(f"===== Starting NoIntent repair process for instance: {instance_name} =====")
        logger.info(f"Logging for this instance will be saved to: {log_file_path}")
        logger.info(f"All results will be saved in: {result_dir}")
        
        instance_str = parse_instance_string(instance_name)
        if not instance_str:
            logger.error(f"Invalid instance name format: {instance_name}")
            return None

        image_name = f'mswebench/{instance_str["org"]}_m_{instance_str["repo"]}:pr-{instance_str["number"]}'.lower()
        
        # --- Stage 1: Reproduction ---
        repro_docker_env = None
        memory_instance = None
        logger.info(f"===== Starting Stage 1: Reproduction for {instance_name} =====")
        try:
            repro_docker_env = DockerEnv(
                image_name=image_name,
                container_name=f'repro-nointent-{instance_name}-{time_stamp}',
                work_dir=f'/home/{instance_str["repo"]}',
            )
            memory_instance = run_reproduction_step(
                instance_name=instance_name,
                model_name=repro_model_name,
                docker_env=repro_docker_env,
                output_dir=result_dir
            )
        except Exception as e:
            logger.error(f"An error occurred during the reproduction stage for '{instance_name}': {e}", exc_info=True)

        if not memory_instance:
            logger.warning(f"Reproduction failed or skipped for {instance_name}. Using mock memory for repair.")
            instance = get_INTENT2FIX_instance(instance_name)
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

        # --- Stage 2: Repair (includes Test Enhancement and Iterative Repair) ---
        repair_docker_env = None
        solution_patch = None
        try:
            repair_docker_env = DockerEnv(
                image_name=image_name,
                container_name=f'repair-nointent-{instance_name}-{time_stamp}',
                work_dir=f'/home/{instance_str["repo"]}',
            )
            logger.info(f"===== Starting Stage 2: Repair (includes Test Enhancement) for {instance_name} =====")
            solution_patch = run_repair_step(
                instance_name=instance_name, 
                model_name=model_name, 
                docker_env=repair_docker_env, 
                memory=memory_instance,
                output_dir=result_dir
            )
            logger.info(f"===== Successfully finished Repair and Test Enhancement for instance: {instance_name} ====")
        except Exception as e:
            logger.error(f"An error occurred during the repair stage for '{instance_name}': {e}", exc_info=True)
        
        if solution_patch:
            submission = {
                "instance_id": instance_name,
                "model_patch": solution_patch
            }
            return submission
        
        logger.info(f"===== End of processing for instance: {instance_name} ====")
        return None
    except Exception as e:
        try:
            logger = logging.getLogger()
            logger.error(f"Unhandled exception in process_instance for '{instance_name}': {e}", exc_info=True)
        except:
            import sys
            print(f"CRITICAL: Unhandled exception in process_instance for '{instance_name}': {e}", file=sys.stderr)
        return None

def main():
    _load_env()
    
    if os.environ.get('OPENROUTER_API_KEY') and not os.environ.get('OPENAI_API_KEY'):
        os.environ['OPENAI_API_KEY'] = os.environ['OPENROUTER_API_KEY']
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(process)d - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

    jsonl_file = 'INTENT2FIX/benchmark/multi_swebench_cpp_change.jsonl'
    logging.info(f"Loading instance IDs from: {jsonl_file}")

    test_instance_names = extract_instance_ids_from_jsonl(file_path=jsonl_file)
    
    if not test_instance_names:
        logging.warning("No instance names extracted. Exiting.")
        return
        
    logging.info(f"Extracted {len(test_instance_names)} instance IDs.")

    # Filter out instances that already have solution.patch
    results_dir = results_path("results_ablation_no_intent_deepseek-v3-0324")
    completed_instances = set()
    if os.path.exists(results_dir):
        for instance_name in test_instance_names:
            instance_result_dir = os.path.join(results_dir, instance_name)
            solution_patch_path = os.path.join(instance_result_dir, "solution.patch")
            if os.path.exists(solution_patch_path) and os.path.getsize(solution_patch_path) > 0:
                completed_instances.add(instance_name)
        
        if completed_instances:
            logging.info(f"Found {len(completed_instances)} instances with existing solution.patch. Skipping these instances.")
            logging.info(f"Completed instances: {sorted(completed_instances)[:10]}{'...' if len(completed_instances) > 10 else ''}")
    else:
        logging.info(f"Results directory {results_dir} does not exist. Will process all instances.")
    
    # Filter out completed instances
    valid_instance_names = [name for name in test_instance_names if name not in completed_instances]
    logging.info(f"Will process {len(valid_instance_names)} instances (skipped {len(completed_instances)} completed instances).")

    max_workers = 8
    logging.info(f"Starting concurrent processing for {len(valid_instance_names)} instances with up to {max_workers} workers.")

    submissions = []
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
    try:
        future_to_instance = {executor.submit(process_instance, name): name for name in valid_instance_names}
        
        for future in concurrent.futures.as_completed(future_to_instance):
            instance_name = future_to_instance[future]
            try:
                result = future.result()
                if result:
                    submissions.append(result)
                    logging.info(f"Successfully processed and got result for instance: {instance_name}")
                else:
                    logging.warning(f"Processing for instance {instance_name} did not return a result.")
            except concurrent.futures.process.BrokenProcessPool as bpp_exc:
                logging.error(f"Process pool broken for instance {instance_name}. A worker process crashed: {bpp_exc}", exc_info=True)
                break
            except Exception as exc:
                logging.error(f"Instance {instance_name} generated an exception in the worker process: {exc}", exc_info=True)
    except KeyboardInterrupt:
        logging.warning("\nCaught Ctrl+C! Shutting down workers immediately.")
    finally:
        executor.shutdown(wait=False)
        logging.info("Executor shutdown initiated. Main process exiting.")

    logging.info("All instances have been processed.")
    
    submission_file_path = submission_path("submission_ablation_no_intent.json")
    os.makedirs(os.path.dirname(submission_file_path), exist_ok=True)
    with open(submission_file_path, "w") as f:
        json.dump(submissions, f, indent=2)
    logging.info(f"All submissions have been saved to {submission_file_path}")

if __name__ == "__main__":
    main()
