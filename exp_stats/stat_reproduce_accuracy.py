#!/usr/bin/env python3
"""
统计复现结果的准确率（默认 determinant=True 视为复现成功）。

支持两种常见格式：
1) .traj 文件（metadata.reproduction_traj 内含 Finish 动作的 determinant）
2) 轨迹 JSON（nodes[*].observation.properties.determinant）

用法示例：
  python stat_reproduce_accuracy.py --results_dir results_gpt5.2_cotest
  python stat_reproduce_accuracy.py --results_dir results_gpt5.2_cotest --include_missing_as_fail
  python stat_reproduce_accuracy.py --results_dir results_gpt5.2_cotest --stage repro
  python stat_reproduce_accuracy.py --results_dir results_gpt5.2_cotest --stage repair
  python stat_reproduce_accuracy.py --results_dir results_gpt5.2_cotest --require_evidence
  python stat_reproduce_accuracy.py --results_dir results_gpt5.2_cotest --require_expected
  python stat_reproduce_accuracy.py --results_dir results_gpt5.2_cotest --success_by repro_output --stage repair
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _safe_json_load(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _maybe_parse_json_string(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _extract_finish_info_from_repro_traj(traj: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    last_payload: Optional[Dict[str, Any]] = None
    for step in traj:
        content = step.get("content")
        payload = _maybe_parse_json_string(content)
        if isinstance(payload, dict) and "determinant" in payload:
            if isinstance(payload.get("determinant"), bool):
                last_payload = payload
    return last_payload


def _extract_finish_info_from_nodes(nodes: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    last_payload: Optional[Dict[str, Any]] = None
    for node in nodes:
        obs = node.get("observation") or {}
        props = obs.get("properties") or {}
        if "determinant" in props and isinstance(props["determinant"], bool):
            last_payload = props
    return last_payload


def _search_determinant_recursive(obj: Any) -> Optional[bool]:
    last_value: Optional[bool] = None
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "determinant" and isinstance(v, bool):
                last_value = v
            else:
                nested = _search_determinant_recursive(v)
                if isinstance(nested, bool):
                    last_value = nested
    elif isinstance(obj, list):
        for item in obj:
            nested = _search_determinant_recursive(item)
            if isinstance(nested, bool):
                last_value = nested
    return last_value


def extract_finish_info(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # 1) .traj 格式：metadata.reproduction_traj
    metadata = data.get("metadata") if isinstance(data, dict) else None
    if isinstance(metadata, dict):
        traj = metadata.get("reproduction_traj")
        if isinstance(traj, list):
            payload = _extract_finish_info_from_repro_traj(traj)
            if isinstance(payload, dict) and isinstance(payload.get("determinant"), bool):
                return payload

    # 2) 轨迹 JSON：nodes[*].observation.properties.determinant
    nodes = data.get("nodes")
    if isinstance(nodes, list):
        payload = _extract_finish_info_from_nodes(nodes)
        if isinstance(payload, dict) and isinstance(payload.get("determinant"), bool):
            return payload

    # 3) 回退：递归搜索 determinant
    det = _search_determinant_recursive(data)
    if isinstance(det, bool):
        return {"determinant": det}
    return None


def find_result_files(results_dir: Path) -> List[Path]:
    if results_dir.is_file():
        return [results_dir]

    traj_files = sorted(results_dir.rglob("*.traj"))
    if traj_files:
        return traj_files

    json_files = sorted(results_dir.rglob("*.json"))
    return json_files


def get_instance_id(data: Dict[str, Any], path: Path) -> str:
    metadata = data.get("metadata") if isinstance(data, dict) else None
    if isinstance(metadata, dict) and metadata.get("instance_id"):
        return str(metadata["instance_id"])
    stem = path.stem
    for prefix in ("repro_trajectory_", "repair_trajectory_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def detect_stage(path: Path) -> Optional[str]:
    lower_path = str(path).lower()
    if "test_enhancement_trajectory" in lower_path:
        return "test_enhancement"
    if "iterative_repair_trajectory" in lower_path:
        return "repair"
    if "repair_trajectory" in lower_path:
        return "repair"
    if "repro_trajectory" in lower_path:
        return "repro"
    return None


def extract_repro_outputs(data: Dict[str, Any]) -> Tuple[str, str]:
    metadata = data.get("metadata") if isinstance(data, dict) else None
    if isinstance(metadata, dict):
        repro = metadata.get("reproduction_result") or ""
        expected = metadata.get("expected_reproduction_result") or ""
        return str(repro), str(expected)
    return "", ""


def main() -> None:
    parser = argparse.ArgumentParser(description="统计复现准确率")
    parser.add_argument("--results_dir", required=True, help="结果目录或单个文件路径")
    parser.add_argument(
        "--stage",
        choices=["repro", "repair", "all"],
        default="repro",
        help="选择统计阶段（默认repro）",
    )
    parser.add_argument(
        "--include_missing_as_fail",
        action="store_true",
        help="将缺失determinant的样本计入失败",
    )
    parser.add_argument(
        "--require_evidence",
        action="store_true",
        help="仅当determinant为True且evidence非空才算成功",
    )
    parser.add_argument(
        "--require_expected",
        action="store_true",
        help="仅当determinant为True且expected_output非空才算成功",
    )
    parser.add_argument(
        "--success_by",
        choices=["determinant", "repro_output"],
        default="determinant",
        help="成功判定依据：determinant 或 reproduction_result",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"错误：路径不存在：{results_dir}")
        return

    files = find_result_files(results_dir)
    if not files:
        print(f"错误：未在 {results_dir} 下找到 .traj 或 .json 文件")
        return

    # 仅保留每个实例最新的一个文件（限定阶段）
    latest_by_instance: Dict[str, Tuple[float, Path, Dict[str, Any], Optional[str]]] = {}
    for path in files:
        stage = detect_stage(path)
        if args.stage != "all" and stage != args.stage:
            continue
        data = _safe_json_load(path)
        if not isinstance(data, dict):
            continue
        inst_id = get_instance_id(data, path)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        prev = latest_by_instance.get(inst_id)
        if (prev is None) or (mtime > prev[0]):
            latest_by_instance[inst_id] = (mtime, path, data, stage)

    selected_items = list(latest_by_instance.values())

    total = 0
    with_det = 0
    success = 0
    fail = 0
    missing: List[str] = []
    with_repro_output = 0
    with_expected_output = 0

    for _, path, data, _ in selected_items:
        total += 1
        finish_info = extract_finish_info(data)
        inst_id = get_instance_id(data, path)
        repro_out, expected_repro = extract_repro_outputs(data)
        if repro_out.strip():
            with_repro_output += 1
        if expected_repro.strip():
            with_expected_output += 1

        if args.success_by == "repro_output":
            if repro_out.strip():
                success += 1
            else:
                fail += 1
            continue

        if isinstance(finish_info, dict) and isinstance(finish_info.get("determinant"), bool):
            with_det += 1
            det = finish_info.get("determinant")
            evidence = finish_info.get("evidence")
            expected_output = finish_info.get("expected_output")
            has_evidence = isinstance(evidence, str) and bool(evidence.strip())
            has_expected = isinstance(expected_output, str) and bool(expected_output.strip())
            success_ok = bool(det)
            if args.require_evidence:
                success_ok = success_ok and has_evidence
            if args.require_expected:
                success_ok = success_ok and has_expected
            if success_ok:
                success += 1
            else:
                fail += 1
        else:
            missing.append(inst_id)

    if args.success_by == "repro_output":
        denom = total
    else:
        denom = (with_det + len(missing)) if args.include_missing_as_fail else with_det
    acc = (success / denom) if denom > 0 else 0.0

    print("=" * 80)
    print(f"结果目录: {results_dir}")
    print(f"统计阶段: {args.stage}")
    print(f"原始文件数: {len(files)}")
    print(f"实例数(去重后): {len(selected_items)}")
    print(f"样本总数: {total}")
    if args.success_by == "determinant":
        print(f"有determinant: {with_det}")
    print(f"有reproduction_result: {with_repro_output}")
    print(f"有expected_reproduction_result: {with_expected_output}")
    print(f"成功: {success}")
    print(f"失败: {fail}")
    print(f"缺失determinant: {len(missing)}")
    print("-" * 80)
    if args.include_missing_as_fail:
        print("准确率(缺失计入失败): {:.2f}%".format(acc * 100))
    else:
        print("准确率(仅统计有determinant样本): {:.2f}%".format(acc * 100))

    if missing:
        print("-" * 80)
        print("缺失determinant的实例(前50条):")
        for inst_id in missing[:50]:
            print(f"  - {inst_id}")


if __name__ == "__main__":
    main()
