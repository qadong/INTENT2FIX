

import glob
import json
import logging
import os
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from INTENT2FIX.api.trajectory_utils import load_trajectory_from_file
from INTENT2FIX.memory.memory import Memory
from INTENT2FIX.memory.memory_module import MemoryModule
from INTENT2FIX.completion.lite_llm import LiteLLM
from INTENT2FIX.actions.finish import Finish 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _summarize_trajectory(problem_statement: str, trajectory: List[Dict[str, Any]]) -> str:
    """Generates a concise summary of the problem-solving trajectory."""
    summary_prompt = (
        "You are an expert software engineer. Based on the following problem description and the "
        "step-by-step action/observation trajectory, write a concise, one-paragraph summary of the "
        "problem-solving process. Highlight the key discoveries, critical code modifications, "
        "and the final logic that led to the solution."
    )

    # TODO: Handle long trajectories that might exceed context window
    trajectory_str = ""
    for item in trajectory:
        action_str = json.dumps(item.get('action', {}))
        observation_str = item.get('observation', '')
        trajectory_str += f"Action: {action_str}\nObservation: {observation_str}\n"


    llm = LiteLLM(model="deepseek/deepseek-chat")
    
    full_prompt = f"{summary_prompt}\n\nProblem: {problem_statement}\n\nTrajectory:\n{trajectory_str}"
    
    try:
        response = llm.generate(full_prompt)
        return response.message
    except Exception as e:
        logging.error(f"Failed to generate summary: {e}")
        return ""

def generate_memories_from_trajectories():
    """
    Loads successful trajectories, generates summaries, and stores them as memories.
    """
    # Load golden patches
    golden_patches = {}
    problem_statements = {}
    with open(os.path.join(project_root, 'INTENT2FIX/benchmark/multi_swebench_cpp_change.jsonl'), 'r') as f:
        for line in f:
            data = json.loads(line)
            golden_patches[data['instance_id']] = data['patch']
            problem_statements[data['instance_id']] = data['problem_statement']
    memory_module = MemoryModule(file_path="INTENT2FIX/memory/memory.json")
    trajectory_pattern = os.path.join(project_root, '*deepseek-v3_V2.json')
    # trajectory_pattern= "nlohmann__json_3601_trajectory_deepseek-v3_V2.json"
    logging.info(f"Searching for trajectories in: {trajectory_pattern}")
    
    trajectory_files = glob.glob(trajectory_pattern)
    
    if not trajectory_files:
        logging.warning("No trajectory files found matching the pattern.")
        return

    logging.info(f"Found {len(trajectory_files)} trajectory files.")

    for file_path in trajectory_files:
        try:
            logging.info(f"Processing {file_path}...")
            trajectory_dto = load_trajectory_from_file(file_path)
            
            if not trajectory_dto.nodes:
                logging.warning(f"Skipping empty trajectory: {file_path}")
                continue
            final_node = trajectory_dto.nodes[-1]
            if not final_node.actionSteps[-1].action:
                logging.warning(f"Skipping trajectory with no final action: {file_path}")
                continue
            
            # Check if the last action is FinishAction
            if final_node.actionSteps[-1].action.name == 'Finish':
                logging.info(f"Found successful trajectory in {file_path}.")

                problem_statement = problem_statements.get(instance_id, "")
                
                # Reconstruct trajectory for summarization
                trajectory_data = []
                for node in trajectory_dto.nodes:
                    for step in node.actionSteps:
                        trajectory_data.append({
                            "action": step.action.model_dump(),
                            "observation": step.observation.message if step.observation else None
                        })

                summary = _summarize_trajectory(problem_statement, trajectory_data)

                if not summary:
                    logging.error(f"Could not generate summary for {file_path}. Skipping.")
                    continue

                solution_patch = final_node.fileContext.patch
                
                # Extract instance_id from filename
                instance_id = os.path.basename(file_path).replace('_trajectory_deepseek-v3_V2.json', '')

                new_memory = Memory(
                    instance_id=instance_id,
                    problem_statement=problem_statement,
                    trajectory_summary=summary,
                    solution_patch=solution_patch,
                    golden_patch=golden_patches.get(instance_id, ""),
                    test_script="",  # Populate with empty string
                    reproduction_result="",  # Populate with empty string
                    expected_reproduction_result="" # Populate with empty string
                )

                memory_module.add(new_memory)
                logging.info(f"Successfully created and saved memory for {instance_id}.")

            else:
                logging.info(f"Skipping unsuccessful trajectory (final action was {final_node.actionSteps[-1].action.name}): {file_path}")

        except Exception as e:
            logging.error(f"Failed to process {file_path}: {e}", exc_info=True)

def update_problem_statements():
    """
    Updates the problem_statement field for all existing memories from the source JSONL file,
    preserving all other fields.
    """
    logging.info("Starting to update problem statements for existing memories...")

    # 1. Load the authoritative problem statements from the JSONL file
    problem_statements = {}
    jsonl_path = os.path.join(project_root, 'INTENT2FIX/benchmark/multi_swebench_cpp_change.jsonl')
    try:
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                problem_statements[data['instance_id']] = data['problem_statement']
        logging.info(f"Loaded {len(problem_statements)} problem statements from {jsonl_path}.")
    except FileNotFoundError:
        logging.error(f"Authoritative problem statement file not found at: {jsonl_path}")
        return

    # 2. Initialize MemoryModule to load existing memories
    memory_module = MemoryModule(file_path="INTENT2FIX/memory/memory.json")
    if not memory_module.memories:
        logging.warning("No memories found to update. Exiting.")
        return
    
    logging.info(f"Found {len(memory_module.memories)} existing memories to check.")

    # 3. Iterate through memories and update the problem_statement if necessary
    updated_count = 0
    for memory in memory_module.memories:
        instance_id = memory.instance_id
        if instance_id == "unknown":
            logging.warning(f"Skipping memory with instance_id 'unknown'.")
            continue
            
        if instance_id in problem_statements:
            new_statement = problem_statements[instance_id]
            # Check if the statement is different before updating
            if memory.problem_statement != new_statement:
                logging.info(f"Updating problem statement for instance_id: {instance_id}")
                memory.problem_statement = new_statement
                updated_count += 1
        else:
            logging.warning(f"No matching problem statement found for instance_id: {instance_id}")

    # 4. Save the changes back to the file if any updates were made
    if updated_count > 0:
        try:
            memory_module.save()
            logging.info(f"Successfully saved {updated_count} updated memories to {memory_module.file_path}.")
        except Exception as e:
            logging.error(f"Failed to save updated memories: {e}")
    else:
        logging.info("No problem statements required an update.")


if __name__ == "__main__":
    # You can choose which function to run.
    # To generate new memories from trajectories:
    # generate_memories_from_trajectories()
    
    # To update problem statements for existing memories:
    update_problem_statements()
    
    logging.info("Memory script process completed.")

