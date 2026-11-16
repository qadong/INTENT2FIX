from trajectory_utils import load_trajectory_from_file, create_trajectory_dto



trajectory_path = "nlohmann__json_3601_trajectory_deepseek-v3_V2.json"


trajectory_dto = load_trajectory_from_file(trajectory_path)

action_steps = trajectory_dto.nodes[-1].actionSteps

print(action_steps)