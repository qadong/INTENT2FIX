#!/usr/bin/env python3
"""
论文用统计：1) 补丁规模（来自 patch JSONL）；2) CoTest 各阶段 agent 轮次（来自 results_* 目录下 trajectory JSON）。

轮次定义：每个 trajectory 中每个 node 的 action_steps 长度之和（与 run_no_memory_cotest 中展平 trajectory 的方式一致）。
repro 轨迹为树形（含 children），需递归遍历所有节点。
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# --- patch parsing (aligned with scripts/stats_patch_size.py) ---
DIFF_GIT_RE = re.compile(r"^diff --git a/(.*?) b/(.*?)$")
HUNK_HEADER_RE = re.compile(r"^@@\s*-[0-9,]+\s+\+[0-9,]+\s*@@\s*(.*)$")


def _normalize_function_hint(s: str) -> str:
    s = s.strip()
    if not s:
        return ""
    # Remove trailing inline comments to reduce duplicate variants.
    s = re.sub(r"\s*//.*$", "", s).strip()
    # Collapse spaces for stable dedup.
    s = re.sub(r"\s+", " ", s)
    return s


def _looks_like_function_signature(line: str) -> bool:
    """
    Heuristic for C/C++ function-like lines in diff context.
    """
    s = line.strip()
    if not s:
        return False
    if "(" not in s or ")" not in s:
        return False
    # Exclude common control-flow statements.
    if re.match(r"^(if|for|while|switch|catch)\s*\(", s):
        return False
    # Exclude macro-like invocation lines.
    if re.match(r"^[A-Z0-9_]+\s*\(", s):
        return False
    return True


def _is_test_like_path(path: str) -> bool:
    p = path.lower()
    parts = [x for x in p.split("/") if x]
    if any(x in {"test", "tests", "testing", "unittests", "unit_test", "unit_tests"} for x in parts):
        return True
    name = parts[-1] if parts else p
    if name.startswith("test_") or name.endswith("_test.c") or name.endswith("_test.cc") or name.endswith("_test.cpp"):
        return True
    if name.endswith("_test.h") or name.endswith("_test.hpp") or name.endswith(".test.c") or name.endswith(".test.cpp"):
        return True
    return False


def parse_patch_extended(
    patch_text: str,
    allowed_exts: Optional[Set[str]] = None,
    exclude_tests: bool = False,
) -> Tuple[int, int, int, int, int]:
    """
    Returns:
      (files_changed, lines_added, lines_deleted, hunk_count, functions_touched_est)
    """
    files: set[str] = set()
    added = 0
    deleted = 0
    hunks = 0
    functions_touched: set[str] = set()
    cur_hunk_has_function = False
    current_file = ""
    current_file_included = False

    def file_in_scope(path: str) -> bool:
        if not path:
            return False
        if exclude_tests and _is_test_like_path(path):
            return False
        if allowed_exts is not None:
            suffix = Path(path).suffix.lower()
            if suffix not in allowed_exts:
                return False
        return True

    in_hunk = False
    for raw_line in patch_text.splitlines():
        line = raw_line.rstrip("\n")

        m = DIFF_GIT_RE.match(line)
        if m:
            in_hunk = False
            current_file = m.group(2)
            current_file_included = file_in_scope(current_file)
            if current_file_included:
                files.add(current_file)
            continue

        if line.startswith("@@"):
            in_hunk = True
            cur_hunk_has_function = False
            if current_file_included:
                hunks += 1
            m_h = HUNK_HEADER_RE.match(line)
            if current_file_included and m_h:
                hint = _normalize_function_hint(m_h.group(1))
                if hint:
                    functions_touched.add(hint)
                    cur_hunk_has_function = True
            continue

        if line.startswith("--- ") or line.startswith("+++ "):
            if line.startswith("+++ b/"):
                current_file = line[6:]
                current_file_included = file_in_scope(current_file)
                if current_file_included:
                    files.add(current_file)
            in_hunk = False
            continue

        if not in_hunk or not current_file_included:
            continue

        # Fallback for hunks without function header context:
        # infer from first context/signature-looking line inside hunk.
        if not cur_hunk_has_function and line.startswith(" "):
            maybe = line[1:]
            if _looks_like_function_signature(maybe):
                hint = _normalize_function_hint(maybe)
                if hint:
                    functions_touched.add(hint)
                    cur_hunk_has_function = True

        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            deleted += 1

    return len(files), added, deleted, hunks, len(functions_touched)


def patch_metrics_from_text(pt: str) -> Dict[str, int]:
    fc, ad, dl, hk, fn = parse_patch_extended(pt)
    return {
        "files_changed": fc,
        "diff_hunks": hk,
        "functions_touched_estimated": fn,
        "lines_added": ad,
        "lines_deleted": dl,
        "lines_changed_total": ad + dl,
        "net_line_change_abs": abs(ad - dl),
    }


def summarize_patch_metrics(rows: List[Dict[str, int]]) -> Dict[str, Any]:
    if not rows:
        return {
            "valid_patch_rows": 0,
            "files_changed": summarize_ints([]),
            "diff_hunks": summarize_ints([]),
            "functions_touched_estimated": summarize_ints([]),
            "lines_added": summarize_ints([]),
            "lines_deleted": summarize_ints([]),
            "lines_changed_total": summarize_ints([]),
            "net_line_change_abs": summarize_ints([]),
        }
    keys = [
        "files_changed",
        "diff_hunks",
        "functions_touched_estimated",
        "lines_added",
        "lines_deleted",
        "lines_changed_total",
        "net_line_change_abs",
    ]
    out: Dict[str, Any] = {"valid_patch_rows": len(rows)}
    for k in keys:
        out[k] = summarize_ints([r[k] for r in rows])
    return out


def build_benchmark_map(benchmark_jsonl: Path) -> Dict[str, Dict[str, Any]]:
    bench_map: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(benchmark_jsonl):
        repo = row.get("repo")
        pull_number = row.get("pull_number")
        if not repo or pull_number is None:
            continue
        bench_map[f"{repo}#{pull_number}"] = row
    return bench_map


def instance_id_from_patch_row(row: Dict[str, Any]) -> Optional[str]:
    org = row.get("org")
    repo_name = row.get("repo")
    number = row.get("number")
    if not org or not repo_name or number is None:
        return None
    return f"{org}__{repo_name}-{number}"


def load_only_instance_ids(args: Any) -> Tuple[Optional[Set[str]], List[str]]:
    ids: Set[str] = set()
    fp = getattr(args, "only_instances_file", None)
    if fp and str(fp).strip():
        p = Path(fp)
        if p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    ids.add(line)
    s = getattr(args, "only_instances", None)
    if s and str(s).strip():
        for x in str(s).split(","):
            x = x.strip()
            if x:
                ids.add(x)
    if not ids:
        return None, []
    return ids, sorted(ids)


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def iter_trajectory_nodes(obj: Any) -> List[Dict[str, Any]]:
    """收集 trajectory JSON 中所有带 action_steps 的节点（repair: nodes 列表；repro: 根 + children 树）。"""
    out: List[Dict[str, Any]] = []

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        if "action_steps" in node:
            out.append(node)
        ch = node.get("children")
        if isinstance(ch, list):
            for c in ch:
                walk(c)

    if isinstance(obj, dict):
        if "nodes" in obj and isinstance(obj["nodes"], list):
            for n in obj["nodes"]:
                walk(n)
        else:
            walk(obj)
    return out


def action_class(step: Dict[str, Any]) -> str:
    act = step.get("action") or {}
    return (act.get("action_args_class") or "").lower()


def classify_repair_action(ac: str) -> str:
    """将 repair agent 的单步动作分为 localization / edit / verify / other。"""
    if any(
        x in ac
        for x in (
            "entity_intent_search",
            "entity_function_semantic_search",
            "entity_search_symbol",
            "get_macro_usages",
            "find_template_instantiation",
            "view_code",
            "run_command",
        )
    ):
        return "localization"
    if any(x in ac for x in ("string_replace", "create_file", "append_string")):
        return "edit"
    if any(x in ac for x in ("run_single_test", "run_regression", "verified_finish", "finish")):
        return "verify"
    return "other"


def count_steps_by_category(nodes: List[Dict[str, Any]], category_fn) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for node in nodes:
        for step in node.get("action_steps") or []:
            if not isinstance(step, dict):
                continue
            cat = category_fn(step)
            counts[cat] += 1
    return dict(counts)


def summarize_ints(xs: List[int]) -> Dict[str, float]:
    if not xs:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0, "max": 0}
    return {
        "count": len(xs),
        "mean": float(mean(xs)),
        "median": float(median(xs)),
        "min": min(xs),
        "max": max(xs),
    }


def aggregate_trajectory_results(
    result_dir: Path,
    only_instances: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    每个子目录（实例）聚合：
    - repro: test generation 总轮次
    - repair: localization / edit / verify 轮次及总轮次
    - test_enhancement: 测试增强轮次（可多文件 iter1, iter2...）
    - iterative_repair: 迭代修复轮次
    """
    by_instance: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "test_generation_rounds": 0,
            "repair_total_rounds": 0,
            "repair_localization_rounds": 0,
            "repair_edit_rounds": 0,
            "repair_verify_rounds": 0,
            "repair_other_rounds": 0,
            "test_augmentation_rounds": 0,
            "iterative_fix_rounds": 0,
            "files_seen": [],
        }
    )

    json_files = sorted(result_dir.rglob("*.json"))
    for jpath in json_files:
        name = jpath.name
        parent = jpath.parent.name
        if parent == result_dir.name:
            continue
        if only_instances is not None and parent not in only_instances:
            continue
        try:
            data = json.loads(jpath.read_text(encoding="utf-8"))
        except Exception as e:
            continue

        nodes = iter_trajectory_nodes(data)
        total = sum(len(n.get("action_steps") or []) for n in nodes)

        if name.startswith("repro_trajectory_"):
            by_instance[parent]["test_generation_rounds"] += total
        elif name.startswith("repair_trajectory_") and "iterative" not in name:
            by_instance[parent]["repair_total_rounds"] += total
            cats = count_steps_by_category(nodes, lambda s: classify_repair_action(action_class(s)))
            by_instance[parent]["repair_localization_rounds"] += cats.get("localization", 0)
            by_instance[parent]["repair_edit_rounds"] += cats.get("edit", 0)
            by_instance[parent]["repair_verify_rounds"] += cats.get("verify", 0)
            by_instance[parent]["repair_other_rounds"] += cats.get("other", 0)
        elif name.startswith("test_enhancement_trajectory_"):
            by_instance[parent]["test_augmentation_rounds"] += total
        elif name.startswith("iterative_repair_trajectory_"):
            by_instance[parent]["iterative_fix_rounds"] += total

        by_instance[parent]["files_seen"].append(name)

    instances = sorted(by_instance.keys())
    rows = [by_instance[k] for k in instances]

    def col(key: str) -> List[int]:
        return [r[key] for r in rows]

    out: Dict[str, Any] = {
        "result_dir": str(result_dir),
        "num_instances": len(instances),
        "per_instance": {inst: {k: v for k, v in by_instance[inst].items() if k != "files_seen"} for inst in instances},
        "aggregate": {
            "test_generation_rounds": summarize_ints(col("test_generation_rounds")),
            "repair_total_rounds": summarize_ints(col("repair_total_rounds")),
            "repair_localization_rounds": summarize_ints(col("repair_localization_rounds")),
            "repair_edit_rounds": summarize_ints(col("repair_edit_rounds")),
            "repair_verify_rounds": summarize_ints(col("repair_verify_rounds")),
            "repair_other_rounds": summarize_ints(col("repair_other_rounds")),
            "test_augmentation_rounds": summarize_ints(col("test_augmentation_rounds")),
            "iterative_fix_rounds": summarize_ints(col("iterative_fix_rounds")),
            "final_fix_pipeline_rounds": summarize_ints(
                [r["repair_edit_rounds"] + r["repair_verify_rounds"] + r["repair_other_rounds"] for r in rows]
            ),
        },
        "notes": {
            "test_generation": "repro_trajectory_*.json 中所有节点 action_steps 之和（含树形 children）",
            "fault_localization": "repair_trajectory 中 action 归类为 Entity*Search / ViewCode / RunCommand 等",
            "test_augmentation": "test_enhancement_trajectory_* 的 action_steps 之和",
            "final_fix_and_verify": "repair 中的 edit(StringReplace/CreateFile/AppendString) + verify(Run* / Finish) + other；不含 localization。iterative_fix 单独统计。",
        },
    }
    if only_instances is not None:
        out["instance_filter"] = {
            "only_instances_count": len(only_instances),
            "only_instances": sorted(only_instances),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="CoTest 论文统计：补丁规模 + agent 轮次")
    parser.add_argument(
        "--patch-jsonl",
        default="patch/gpt5_2_cotest_patch.jsonl",
        help="含 fix_patch 的 JSONL",
    )
    parser.add_argument(
        "--results-dir",
        default="results_gpt5.2_cotest",
        help="CoTest 结果根目录（每个实例一个子文件夹）",
    )
    parser.add_argument(
        "--output-json",
        default="cotest_paper_stats.json",
        help="输出汇总 JSON 路径",
    )
    parser.add_argument(
        "--benchmark-jsonl",
        default="INTENT2FIX/benchmark/multi_swebench_cpp_change.jsonl",
        help="含 golden patch 的 benchmark JSONL（字段 patch）；空字符串则跳过 golden 对比",
    )
    parser.add_argument(
        "--golden-patch-field",
        default="patch",
        help="benchmark 中 golden patch 字段名",
    )
    parser.add_argument(
        "--allowed-exts",
        default=".txt,.c,.cc,.cpp,.h,.hh,.hpp,.hxx,.cxx",
        help="仅统计这些后缀（逗号分隔，含点）；传空字符串表示不过滤后缀",
    )
    parser.add_argument(
        "--exclude-test-files",
        action="store_true",
        help="排除测试相关路径（test/tests 目录及常见 *_test.* 命名）",
    )
    parser.add_argument(
        "--only-instances-file",
        default="",
        help="每行一个 instance_id（如 fmtlib__fmt-493）；与 --only-instances 合并",
    )
    parser.add_argument(
        "--only-instances",
        default="",
        help="逗号分隔的 instance_id，仅统计这些实例",
    )
    args = parser.parse_args()

    allowed_exts: Optional[Set[str]] = None
    if args.allowed_exts.strip():
        allowed_exts = {x.strip().lower() for x in args.allowed_exts.split(",") if x.strip()}

    only_ids, only_ids_sorted = load_only_instance_ids(args)

    patch_path = Path(args.patch_jsonl)
    gen_metrics_rows: List[Dict[str, int]] = []
    missing = 0
    skipped_instance_filter = 0
    seen_instance_ids: Set[str] = set()

    for row in read_jsonl(patch_path):
        iid = instance_id_from_patch_row(row)
        if only_ids is not None:
            if iid is None or iid not in only_ids:
                skipped_instance_filter += 1
                continue
        if iid is not None:
            seen_instance_ids.add(iid)
        pt = row.get("fix_patch")
        if not pt or not isinstance(pt, str):
            missing += 1
            continue
        fc, ad, dl, hk, fn = parse_patch_extended(pt, allowed_exts=allowed_exts, exclude_tests=args.exclude_test_files)
        gen_metrics_rows.append(
            {
                "files_changed": fc,
                "diff_hunks": hk,
                "functions_touched_estimated": fn,
                "lines_added": ad,
                "lines_deleted": dl,
                "lines_changed_total": ad + dl,
                "net_line_change_abs": abs(ad - dl),
            }
        )

    n = len(gen_metrics_rows)
    patch_summary: Dict[str, Any] = {
        "input_jsonl": str(patch_path),
        "total_rows": n + missing,
        "missing_or_empty_patch_rows": missing,
        "generated_fix_patch": summarize_patch_metrics(gen_metrics_rows),
        "note": (
            "functions_touched_estimated 为基于 diff hunk 的函数级启发式统计："
            "优先使用 @@ 头部上下文，缺失时从 hunk 内上下文行推断函数签名。"
            "该指标是近似值（对匿名代码块/宏展开不完全准确）。"
        ),
    }
    if only_ids is not None:
        patch_summary["instance_filter"] = {
            "only_instances_count": len(only_ids),
            "only_instances": only_ids_sorted,
            "skipped_patch_jsonl_rows_not_in_filter": skipped_instance_filter,
            "only_instances_missing_from_patch_jsonl": sorted(only_ids - seen_instance_ids),
        }

    golden_vs_generated: Dict[str, Any] = {}
    if args.benchmark_jsonl:
        bench_path = Path(args.benchmark_jsonl)
        if bench_path.is_file():
            bench_map = build_benchmark_map(bench_path)
            golden_all: List[Dict[str, int]] = []
            for brow in read_jsonl(bench_path):
                biid = brow.get("instance_id")
                if only_ids is not None:
                    if not biid or biid not in only_ids:
                        continue
                gp = brow.get(args.golden_patch_field)
                if isinstance(gp, str) and gp.strip():
                    fc, ad, dl, hk, fn = parse_patch_extended(
                        gp, allowed_exts=allowed_exts, exclude_tests=args.exclude_test_files
                    )
                    golden_all.append(
                        {
                            "files_changed": fc,
                            "diff_hunks": hk,
                            "functions_touched_estimated": fn,
                            "lines_added": ad,
                            "lines_deleted": dl,
                            "lines_changed_total": ad + dl,
                            "net_line_change_abs": abs(ad - dl),
                        }
                    )

            paired_gen: List[Dict[str, int]] = []
            paired_gold: List[Dict[str, int]] = []
            paired_instance_ids: List[str] = []
            no_bench_match = 0
            for row in read_jsonl(patch_path):
                iid = instance_id_from_patch_row(row)
                if only_ids is not None:
                    if iid is None or iid not in only_ids:
                        continue
                pt = row.get("fix_patch")
                if not pt or not isinstance(pt, str):
                    continue
                org = row.get("org")
                repo_name = row.get("repo")
                number = row.get("number")
                if not org or not repo_name or number is None:
                    no_bench_match += 1
                    continue
                key = f"{org}/{repo_name}#{number}"
                br = bench_map.get(key)
                if not br:
                    no_bench_match += 1
                    continue
                gp = br.get(args.golden_patch_field)
                if not isinstance(gp, str) or not gp.strip():
                    no_bench_match += 1
                    continue
                g_fc, g_ad, g_dl, g_hk, g_fn = parse_patch_extended(
                    pt, allowed_exts=allowed_exts, exclude_tests=args.exclude_test_files
                )
                o_fc, o_ad, o_dl, o_hk, o_fn = parse_patch_extended(
                    gp, allowed_exts=allowed_exts, exclude_tests=args.exclude_test_files
                )
                paired_gen.append(
                    {
                        "files_changed": g_fc,
                        "diff_hunks": g_hk,
                        "functions_touched_estimated": g_fn,
                        "lines_added": g_ad,
                        "lines_deleted": g_dl,
                        "lines_changed_total": g_ad + g_dl,
                        "net_line_change_abs": abs(g_ad - g_dl),
                    }
                )
                paired_gold.append(
                    {
                        "files_changed": o_fc,
                        "diff_hunks": o_hk,
                        "functions_touched_estimated": o_fn,
                        "lines_added": o_ad,
                        "lines_deleted": o_dl,
                        "lines_changed_total": o_ad + o_dl,
                        "net_line_change_abs": abs(o_ad - o_dl),
                    }
                )
                if iid:
                    paired_instance_ids.append(iid)

            metric_keys = [
                "files_changed",
                "diff_hunks",
                "functions_touched_estimated",
                "lines_added",
                "lines_deleted",
                "lines_changed_total",
                "net_line_change_abs",
            ]
            delta_summary: Dict[str, Any] = {}
            if paired_gen and paired_gold:
                for mk in metric_keys:
                    deltas = [g[mk] - o[mk] for g, o in zip(paired_gen, paired_gold)]
                    delta_summary[f"{mk}_generated_minus_golden"] = summarize_ints(deltas)

            golden_vs_generated = {
                "benchmark_jsonl": str(bench_path),
                "golden_patch_field": args.golden_patch_field,
                "filters": {
                    "allowed_exts": sorted(allowed_exts) if allowed_exts is not None else "ALL",
                    "exclude_test_files": bool(args.exclude_test_files),
                },
                "golden_all_rows_in_benchmark": summarize_patch_metrics(golden_all),
                "paired_with_generated": {
                    "matched_instances": len(paired_instance_ids),
                    "generated_rows_without_benchmark_golden": no_bench_match,
                    "paired_instance_ids": paired_instance_ids,
                    "generated": summarize_patch_metrics(paired_gen),
                    "golden": summarize_patch_metrics(paired_gold),
                    "delta_generated_minus_golden": delta_summary,
                },
            }
            if only_ids is not None:
                golden_vs_generated["instance_filter"] = {
                    "only_instances_count": len(only_ids),
                    "only_instances": only_ids_sorted,
                }
        else:
            golden_vs_generated = {
                "error": f"benchmark file not found: {bench_path}",
            }

    traj_summary = aggregate_trajectory_results(Path(args.results_dir), only_ids)

    out = {
        "patch_statistics": patch_summary,
        "golden_vs_generated_patch_statistics": golden_vs_generated,
        "agent_round_statistics": traj_summary,
    }

    out_path = Path(args.output_json)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n[OK] Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
