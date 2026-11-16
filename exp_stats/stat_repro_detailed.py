import json
import os
from pathlib import Path
from collections import defaultdict

def get_reproduction_stats(results_dir):
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Directory {results_dir} does not exist.")
        return

    stats = defaultdict(lambda: {"total": 0, "reproduced": 0, "failed": 0, "no_trajectory": 0})
    overall = {"total": 0, "reproduced": 0, "failed": 0, "no_trajectory": 0}

    for sub_dir in results_path.iterdir():
        if not sub_dir.is_dir():
            continue
        
        instance_id = sub_dir.name
        project = instance_id.split("__")[0]
        
        overall["total"] += 1
        stats[project]["total"] += 1
        
        repro_file = sub_dir / f"repro_trajectory_{instance_id}.json"
        if not repro_file.exists():
            overall["no_trajectory"] += 1
            stats[project]["no_trajectory"] += 1
            continue
            
        try:
            with open(repro_file, "r") as f:
                data = json.load(f)
            
            # Find the last node or Finish action
            reproduced = False
            
            # Check for determinant in the last node's observation or recursively
            def check_reproduced(node):
                if "action_steps" in node:
                    for step in node["action_steps"]:
                        if step.get("action", {}).get("name") == "Finish":
                            if step.get("action", {}).get("args", {}).get("determinant") is True:
                                return True
                            # Some formats have determinant in properties
                        if "observation" in step and step["observation"] and "properties" in step["observation"]:
                            if step["observation"]["properties"].get("determinant") is True:
                                return True
                
                if "children" in node:
                    for child in node["children"]:
                        if check_reproduced(child):
                            return True
                return False

            if "nodes" in data:
                # Format with list of nodes
                for node in data["nodes"]:
                    obs = node.get("observation", {})
                    if obs and obs.get("properties", {}).get("determinant") is True:
                        reproduced = True
                        break
            else:
                # Tree format
                reproduced = check_reproduced(data)
            
            if reproduced:
                overall["reproduced"] += 1
                stats[project]["reproduced"] += 1
            else:
                overall["failed"] += 1
                stats[project]["failed"] += 1
                
        except Exception as e:
            print(f"Error processing {repro_file}: {e}")
            overall["failed"] += 1
            stats[project]["failed"] += 1

    print(f"{'Project':<25} {'Total':<10} {'Reproduced':<12} {'Failed':<10} {'No Traj':<10} {'Success Rate'}")
    print("-" * 80)
    for project, p_stats in sorted(stats.items()):
        rate = (p_stats["reproduced"] / p_stats["total"]) * 100 if p_stats["total"] > 0 else 0
        print(f"{project:<25} {p_stats['total']:<10} {p_stats['reproduced']:<12} {p_stats['failed']:<10} {p_stats['no_trajectory']:<10} {rate:.2f}%")
    
    print("-" * 80)
    overall_rate = (overall["reproduced"] / overall["total"]) * 100 if overall["total"] > 0 else 0
    print(f"{'OVERALL':<25} {overall['total']:<10} {overall['reproduced']:<12} {overall['failed']:<10} {overall['no_trajectory']:<10} {overall_rate:.2f}%")

if __name__ == "__main__":
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results_claude-3.7-sonnet_cotest"
    get_reproduction_stats(results_dir)
