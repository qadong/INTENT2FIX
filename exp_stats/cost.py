import glob
from pathlib import Path



def read_json_file(file_path):
    import json
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# 自定义费率 (每百万 tokens 的美元价格)
INPUT_RATE = 0.03  # 输入 tokens 费率
OUTPUT_RATE = 0.14# 输出 tokens 费率
CACHE_RATE = 0.00 # 缓存命中 tokens 费率 (如果有)

def calculate_custom_cost(usage):
    if not usage:
        return 0.0
    
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    cache_read_tokens = usage.get("cache_read_tokens", 0) # 某些结构可能在这里
    
    # 检查更深层的缓存信息 (兼容不同格式)
    if not cache_read_tokens and "prompt_tokens_details" in usage:
        cache_read_tokens = usage["prompt_tokens_details"].get("cached_tokens", 0)

    non_cached_prompt_tokens = prompt_tokens - cache_read_tokens
    
    cost = (non_cached_prompt_tokens * INPUT_RATE / 1_000_000) + \
           (completion_tokens * OUTPUT_RATE / 1_000_000) + \
           (cache_read_tokens * CACHE_RATE / 1_000_000)
    return cost

def get_flat_cost(trajectory):
    total_cost = 0
    for node in trajectory.get('nodes', []):
        completions = node.get('completions', {})
        if 'build_action' in completions:
            usage = completions['build_action'].get('response', {}).get('usage', {})
            total_cost += calculate_custom_cost(usage)
    return total_cost

def get_tree_cost(node):
    total_cost = 0
    completions = node.get('completions', {})
    if 'build_action' in completions:
        usage = completions['build_action'].get('response', {}).get('usage', {})
        total_cost += calculate_custom_cost(usage)
    
    for child in node.get('children', []):
        total_cost += get_tree_cost(child)
    return total_cost

def get_repro_cost(trajectory):
    if 'nodes' in trajectory:
        return get_flat_cost(trajectory)
    return get_tree_cost(trajectory)

def get_repair_cost(trajectory):
    if 'nodes' in trajectory:
        return get_flat_cost(trajectory)
    return get_tree_cost(trajectory)

if __name__ == "__main__":
    results_dir = Path("results_deepseek-v3-0324_cotest_V7")
    repro_costs = []
    repair_costs = []
    for problem in results_dir.iterdir() :
        if problem.is_dir() :
            for traj_file in glob.glob(str(problem / "*.json")):
                traj_path = Path(traj_file)
                if traj_path.name.startswith('repair') :
                    print(traj_path)
                    trajectory = read_json_file(traj_path)
                    repair_cost = get_repair_cost(trajectory)
                    repair_costs.append(repair_cost)
                else :
                    print(traj_path)
                    trajectory = read_json_file(traj_path)
                    repro_cost = get_repro_cost(trajectory)
                    repro_costs.append(repro_cost)

    avg_repro_cost = sum(repro_costs) / len(repro_costs) if repro_costs else 0
    avg_repair_cost = sum(repair_costs) / len(repair_costs) if repair_costs else 0
    print('========Summary========')
    print(f"Problem: {len(repro_costs)}")
    print(f"Total Repro Cost: {sum(repro_costs)}\nTotal Repair Cost: {sum(repair_costs)}")
    print(f"Total Cost: {sum(repro_costs) + sum(repair_costs)}")
    print(f'Avg Repro Cost: {avg_repro_cost}\nAvg Repair Cost: {avg_repair_cost}')
    print(f'Total Avg cost: {(sum(repro_costs) + sum(repair_costs)) / len(repro_costs)}')