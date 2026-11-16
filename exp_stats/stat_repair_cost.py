#!/usr/bin/env python3
"""
统计所有 repair*.json 文件中的 cost
从 usage.cost 字段提取费用信息
"""

import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

def extract_cost_from_node(node: Dict) -> float:
    """
    从节点中提取 cost 信息
    只统计 response.usage.cost（第一次调用的 cost）
    排除 retry 的 cost（不统计顶层的 usage.completion_cost，因为它可能包含 retry）
    """
    total_cost = 0.0
    
    # 检查节点的 completions 字段
    completions = node.get("completions", {})
    if not completions:
        return 0.0
    
    # 遍历所有 completion
    for completion_key, completion_data in completions.items():
        if not isinstance(completion_data, dict):
            continue
        
        # 只统计 response.usage.cost（第一次调用的 cost）
        # 不统计顶层的 usage.completion_cost，因为它可能包含 retry 的总 cost
        response = completion_data.get("response")
        if response and isinstance(response, dict):
            response_usage = response.get("usage")
            if response_usage and isinstance(response_usage, dict):
                # 只提取 response.usage.cost（第一次调用，不包含 retry）
                cost = response_usage.get("cost", 0.0)
                if cost > 0:
                    total_cost += cost
        
        # 明确排除顶层的 usage（可能包含 retry）
        # 不统计 completion_data.get("usage")，因为它可能包含 retry 的总 cost
    
    return total_cost

def process_trajectory_file(json_path: Path) -> Optional[Dict[str, float]]:
    """
    处理单个轨迹文件，提取所有 cost
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return None
    
    total_cost = 0.0
    cost_count = 0
    
    # 处理 nodes 列表（扁平结构）
    if 'nodes' in data:
        nodes = data['nodes']
        if isinstance(nodes, list):
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                cost = extract_cost_from_node(node)
                if cost > 0:
                    total_cost += cost
                    cost_count += 1
        elif isinstance(nodes, dict):
            # 如果是树形结构，递归处理
            def process_tree_node(node):
                nonlocal total_cost, cost_count
                cost = extract_cost_from_node(node)
                if cost > 0:
                    total_cost += cost
                    cost_count += 1
                # 处理子节点
                children = node.get('children', [])
                for child in children:
                    process_tree_node(child)
            process_tree_node(nodes)
    
    return {
        "total_cost": total_cost,
        "cost_count": cost_count,
        "file": str(json_path)
    }

def find_repair_files(root_dir: str) -> List[Path]:
    """
    查找所有 repair*.json 文件
    """
    repair_files = []
    
    # 支持多种模式
    patterns = [
        "repair*.json",
        "repair*.traj"  # 虽然用户说的是 .traj，但实际可能是 JSON 格式
    ]
    
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"Directory {root_dir} does not exist.")
        return repair_files
    
    for pattern in patterns:
        # 递归查找
        matches = list(root_path.rglob(pattern))
        repair_files.extend(matches)
    
    # 去重并排序
    repair_files = sorted(set(repair_files))
    return repair_files

def main():
    import sys
    
    # 默认搜索当前目录
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = "."
    
    print(f"正在搜索目录: {root_dir}")
    print("查找 repair*.json 文件...")
    
    repair_files = find_repair_files(root_dir)
    
    if not repair_files:
        print("未找到 repair*.json 文件")
        return
    
    print(f"找到 {len(repair_files)} 个文件\n")
    
    all_costs = []
    file_stats = []
    total_cost_all = 0.0
    total_cost_count = 0
    
    for traj_file in repair_files:
        stats = process_trajectory_file(traj_file)
        if stats and stats['total_cost'] > 0:
            file_stats.append(stats)
            all_costs.append(stats['total_cost'])
            total_cost_all += stats['total_cost']
            total_cost_count += stats['cost_count']
            print(f"{traj_file.name:60s} | Cost: ${stats['total_cost']:10.6f} | Calls: {stats['cost_count']:4d}")
    
    # 按 instance（目录）分组统计
    instance_stats = defaultdict(lambda: {"files": [], "cost": 0.0, "calls": 0})
    for stat in file_stats:
        file_path = Path(stat['file'])
        instance_name = file_path.parent.name  # instance 目录名
        instance_stats[instance_name]["files"].append(stat)
        instance_stats[instance_name]["cost"] += stat['total_cost']
        instance_stats[instance_name]["calls"] += stat['cost_count']
    
    print("\n" + "="*80)
    print("按 Instance 统计平均值:")
    print("="*80)
    
    instance_avg_costs = []
    instance_avg_calls = []
    
    for instance_name in sorted(instance_stats.keys()):
        stats = instance_stats[instance_name]
        file_count = len(stats["files"])
        total_cost = stats["cost"]
        total_calls = stats["calls"]
        
        if file_count > 0:
            avg_cost = total_cost / file_count
            avg_calls = total_calls / file_count if file_count > 0 else 0
            avg_cost_per_call = total_cost / total_calls if total_calls > 0 else 0
            
            instance_avg_costs.append(avg_cost)
            instance_avg_calls.append(avg_calls)
            
            print(f"{instance_name:60s} | Files: {file_count:3d} | "
                  f"Avg Cost: ${avg_cost:10.6f} | "
                  f"Avg Calls: {avg_calls:6.2f} | "
                  f"Avg Cost/Call: ${avg_cost_per_call:10.6f} | "
                  f"Total: ${total_cost:10.6f}")
    
    # 全局平均
    if instance_avg_costs:
        print("\n" + "="*80)
        print("全局平均（所有 Instance 的平均值）:")
        print("="*80)
        global_avg_cost = sum(instance_avg_costs) / len(instance_avg_costs)
        global_avg_calls = sum(instance_avg_calls) / len(instance_avg_calls) if instance_avg_calls else 0
        global_avg_cost_per_call = total_cost_all / total_cost_count if total_cost_count > 0 else 0
        
        print(f"平均每个 Instance 的费用: ${global_avg_cost:.6f}")
        print(f"平均每个 Instance 的 API 调用次数: {global_avg_calls:.2f}")
        print(f"平均每次 API 调用费用: ${global_avg_cost_per_call:.6f}")
        print(f"\n总计:")
        print(f"  Instance 数量: {len(instance_stats)}")
        print(f"  总文件数: {len(file_stats)}")
        print(f"  总 API 调用次数: {total_cost_count}")
        print(f"  总费用: ${total_cost_all:.6f}")

if __name__ == "__main__":
    main()
