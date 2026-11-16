#!/usr/bin/env python3
import os
import subprocess
import argparse
import json
import re
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def _bootstrap_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    base = Path(__file__).resolve().parent
    for p in (base.parent / ".env", base / ".env"):
        if p.exists():
            load_dotenv(p, override=False)
            return


_bootstrap_dotenv()

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from INTENT2FIX.env_paths import index_store_parent, repo_instances_parent  # noqa: E402

_BASE = Path(__file__).resolve().parent
# --- Configuration（路径见 INTENT2FIX/env_paths.py：REPO_DIR / INDEX_STORE_DIR）---
BASE_REPO_DIR = str(repo_instances_parent())
BASE_OUTPUT_DIR = str(index_store_parent())
EXAMPLE_SCRIPT = os.environ.get("BATCH_INTENT_EXAMPLE_SCRIPT", str(_BASE / "example.py"))
BATCH_INTENT_LOG = os.environ.get(
    "BATCH_INTENT_SUMMARY_LOG", str(_BASE / "batch_intent_summary.log")
)

def is_valid_instance(instance_id: str) -> bool:
    """
    Checks if the directory name matches the expected SweBench instance format.
    """
    return bool(re.match(r"swe-bench_([^_]+)__([^-]+)-(\d+)", instance_id))

def run_intent_generation(
    instance_id: str,
    repo_path: str,
    output_base: str,
    from_stage: str,
    request_concurrency: int = 1,
):
    """
    Runs the example.py script to generate intents for a specific instance.
    """
    cmd = [
        sys.executable,
        EXAMPLE_SCRIPT,
        "--instance-id", instance_id,
        "--file-path", repo_path,
        "--output-path", output_base,
        "--from-stage", from_stage
    ]
    
    print(f"🚀 [Batch-Intent] Starting intent generation for: {instance_id}")
    print(f"📂 [Batch-Intent] Repo path: {repo_path}")
    print(f"📤 [Batch-Intent] Output base: {output_base}")
    print(f"🎬 [Batch-Intent] From stage: {from_stage}")
    print(f"💻 [Batch-Intent] Running command: {' '.join(cmd)}")
    
    # This will raise CalledProcessError if the command fails
    child_env = os.environ.copy()
    child_env["INTENT_CONCURRENCY"] = str(max(1, request_concurrency))
    child_env.setdefault(
        "INTENT_CACHE_DIR", str(Path(output_base) / ".intent_cache")
    )
    subprocess.check_call(cmd, env=child_env)

def get_pr_number(instance_id: str) -> int:
    """提取 instance_id 中的 PR 号码并转为整数"""
    match = re.search(r"-(\d+)$", instance_id)
    return int(match.group(1)) if match else 0


def _project_key(instance_id: str) -> str:
    return instance_id.rsplit("-", 1)[0]


def _sort_instances(instance_ids: list[str], order: str) -> list[str]:
    grouped = defaultdict(list)
    for instance_id in instance_ids:
        grouped[_project_key(instance_id)].append(instance_id)
    newest_first = order == "newest-first"
    ordered = []
    for project in sorted(grouped):
        ordered.extend(
            sorted(
                grouped[project],
                key=get_pr_number,
                reverse=newest_first,
            )
        )
    return ordered


def _intent_artifacts_ready(index_dir: Path) -> bool:
    from code_intent.feature_builder import intent_feature_artifacts_ready
    from code_intent.intent_extractor import (
        intent_function_artifacts_ready,
        intent_test_artifacts_ready,
    )

    return (
        intent_function_artifacts_ready(index_dir)
        and intent_test_artifacts_ready(
            index_dir,
            require_current_inputs=True,
            require_runtime_embedding=True,
        )
        and intent_feature_artifacts_ready(index_dir)
    )


def _canonical_instance_id(instance_id: str) -> str:
    value = instance_id.strip()
    if not value or value.startswith("swe-bench_"):
        return value
    return f"swe-bench_{value}"


def _repository_path(repo_base: Path, instance_id: str) -> Path:
    prefixed = repo_base / instance_id
    if prefixed.exists():
        return prefixed
    prefix = "swe-bench_"
    if instance_id.startswith(prefix):
        return repo_base / instance_id[len(prefix):]
    return prefixed


def _load_instance_ids(path: Path) -> set[str]:
    selected: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    value = json.loads(line).get("instance_id", "")
                except (AttributeError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"Invalid JSONL instance record at {path}:{line_number}"
                    ) from exc
            else:
                value = line
            instance_id = _canonical_instance_id(str(value))
            if not is_valid_instance(instance_id):
                raise ValueError(
                    f"Invalid instance ID at {path}:{line_number}: {value!r}"
                )
            selected.add(instance_id)
    return selected


def _allocate_request_slots(
    worker_budget: int,
    lane_count: int,
    max_per_lane: int = 0,
) -> list[int]:
    if lane_count <= 0:
        return []
    budget = max(1, worker_budget)
    if max_per_lane > 0:
        budget = min(budget, max_per_lane * lane_count)
    if budget < lane_count:
        return [1] * lane_count
    base, remainder = divmod(budget, lane_count)
    return [base + (index < remainder) for index in range(lane_count)]


def write_summary(summary_path, start_time, duration, indexed_count, skipped_count, processed_count, success_list, failure_list):
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Batch Intent Generation Summary Report\n")
        f.write("====================================\n")
        f.write(f"Start Time: {time.ctime(start_time)}\n")
        f.write(f"Last Updated: {time.ctime()}\n")
        f.write(f"Duration so far: {duration:.2f} seconds\n")
        f.write(f"Total indexed instances found: {indexed_count}\n")
        f.write(f"Instances skipped: {skipped_count}\n")
        f.write(f"Instances processed: {processed_count}\n")
        f.write(f"Successes: {len(success_list)}\n")
        f.write(f"Failures: {len(failure_list)}\n\n")
        
        if failure_list:
            f.write("Failed Instances:\n")
            for inst_id, err in failure_list:
                f.write(f" - {inst_id}: {err}\n")
            f.write("\n")
            
        if success_list:
            f.write("Successful Instances:\n")
            for inst_id in success_list:
                f.write(f" - {inst_id}\n")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip(), 10)
    except ValueError:
        return default


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def main():
    parser = argparse.ArgumentParser(
        description="Batch generate intents for indexed repositories. 路径默认读 REPO_DIR / INDEX_STORE_DIR；其余 BATCH_INTENT_* 见仓库根 .env。"
    )
    parser.add_argument(
        "--repo-dir",
        default=BASE_REPO_DIR,
        help="实例源码父目录；默认读 REPO_DIR",
    )
    parser.add_argument(
        "--output-dir",
        default=BASE_OUTPUT_DIR,
        help="含各实例 index_output 的根；默认读 INDEX_STORE_DIR",
    )
    parser.add_argument(
        "--from-stage",
        choices=("auto", "functions", "tests", "features"),
        default=os.environ.get("BATCH_INTENT_FROM_STAGE", "auto").strip() or "auto",
        help="或 BATCH_INTENT_FROM_STAGE",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=_env_int("BATCH_INTENT_LIMIT", 0),
        help="最多处理数量；或 BATCH_INTENT_LIMIT",
    )
    parser.add_argument(
        "--instances-file",
        type=Path,
        help="只处理该 JSONL（instance_id 字段）或逐行实例 ID 文件中的 case。",
    )
    parser.add_argument(
        "--exclude-instance",
        action="append",
        default=[],
        help="排除一个实例；可重复指定，接受带或不带 swe-bench_ 前缀的 ID。",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("BATCH_INTENT_SKIP_EXISTING"),
        help="跳过已有 function_intents；或 BATCH_INTENT_SKIP_EXISTING=1",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=_env_int("BATCH_INTENT_WORKERS", 4),
        help="全局 intent 请求并发预算；或 BATCH_INTENT_WORKERS",
    )
    parser.add_argument(
        "--max-requests-per-case",
        type=int,
        default=_env_int("BATCH_INTENT_MAX_REQUESTS_PER_CASE", 0),
        help="单个 case 的请求并发上限；0 表示不限制",
    )
    parser.add_argument(
        "--order",
        choices=("oldest-first", "newest-first"),
        default=os.environ.get("BATCH_INTENT_ORDER", "oldest-first").strip()
        or "oldest-first",
        help="每个仓库内的构建顺序；或 BATCH_INTENT_ORDER",
    )
    
    args = parser.parse_args()
    
    repo_base = Path(args.repo_dir).expanduser()
    output_base = Path(args.output_dir).expanduser()
    summary_path = Path(BATCH_INTENT_LOG)
    
    if not output_base.exists():
        print(f"❌ Base output directory does not exist: {output_base}. Please run batch_index.py first.")
        return

    # 1. 寻找已经构建过索引的实例
    indexed_instances = []
    for d in output_base.iterdir():
        if d.is_dir() and is_valid_instance(d.name):
            idx_out = d / "index_output"
            if (idx_out / "functions.jsonl").exists():
                indexed_instances.append(d.name)
    
    indexed_instances = _sort_instances(indexed_instances, args.order)

    if args.instances_file is not None:
        requested = _load_instance_ids(args.instances_file)
        indexed_set = set(indexed_instances)
        missing = sorted(requested - indexed_set)
        if missing:
            print(
                "⚠️ [Batch-Intent] Requested instances without a base index: "
                + ", ".join(missing)
            )
        indexed_instances = [
            instance_id
            for instance_id in indexed_instances
            if instance_id in requested
        ]
    excluded = {
        _canonical_instance_id(instance_id)
        for instance_id in args.exclude_instance
    }
    indexed_instances = [
        instance_id
        for instance_id in indexed_instances
        if instance_id not in excluded
    ]

    print(f"🔍 Found {len(indexed_instances)} instances with existing index in {output_base}")
    print(f"   🎯 Will process: {len(indexed_instances)} instances")
    
    if not indexed_instances:
        print("⚠️ No indexed instances found. Ensure batch_index.py has completed successfully.")
        return

    success_list = []
    failure_list = []
    skipped_count = 0
    
    start_time = time.time()
    processed_count = 0

    to_process = []
    for instance_id in indexed_instances:
        instance_repo_path = _repository_path(repo_base, instance_id)
        if not instance_repo_path.exists():
            print(f"⚠️ [Batch-Intent] Skipping {instance_id}: Repo source not found at {instance_repo_path}")
            skipped_count += 1
            continue

        index_dir = output_base / instance_id / "index_output"
        if args.skip_existing and _intent_artifacts_ready(index_dir):
            print(
                f"⏭️ [Batch-Intent] Skipping {instance_id} "
                "(all current intent artifacts are ready)."
            )
            skipped_count += 1
            continue

        to_process.append((instance_id, instance_repo_path))

    if args.limit > 0:
        to_process = to_process[:args.limit]

    project_lanes = defaultdict(list)
    for instance_id, instance_repo_path in to_process:
        project_lanes[_project_key(instance_id)].append(
            (instance_id, instance_repo_path)
        )

    sorted_lanes = sorted(project_lanes.items())
    request_slots = _allocate_request_slots(
        args.workers,
        len(sorted_lanes),
        args.max_requests_per_case,
    )
    lane_workers = min(max(1, args.workers), max(1, len(sorted_lanes)))
    print(
        f"⚙️ [Batch-Intent] Concurrency: {lane_workers} project lanes; "
        f"request slots {request_slots} (global budget {max(1, args.workers)}, "
        f"per-case cap {args.max_requests_per_case or 'unlimited'})"
    )
    print(f"🚦 [Batch-Intent] Ready to process: {len(to_process)} instances")

    result_lock = threading.Lock()

    def run_project_lane(entries, request_concurrency):
        nonlocal processed_count
        for instance_id, instance_repo_path in entries:
            failure = None
            try:
                run_intent_generation(
                    instance_id,
                    str(instance_repo_path.resolve()),
                    str(output_base.resolve()),
                    args.from_stage,
                    request_concurrency,
                )
            except subprocess.CalledProcessError as exc:
                failure = f"Exit code {exc.returncode}"
            except Exception as exc:  # pragma: no cover - subprocess boundary
                failure = str(exc)

            with result_lock:
                if failure is None:
                    print(f"✅ [Batch-Intent] Successfully generated intents for: {instance_id}")
                    success_list.append(instance_id)
                else:
                    print(f"❌ [Batch-Intent] Failed for {instance_id}: {failure}")
                    failure_list.append((instance_id, failure))
                processed_count += 1
                write_summary(
                    summary_path,
                    start_time,
                    time.time() - start_time,
                    len(indexed_instances),
                    skipped_count,
                    processed_count,
                    success_list,
                    failure_list,
                )

    with ThreadPoolExecutor(max_workers=lane_workers) as executor:
        futures = [
            executor.submit(run_project_lane, entries, request_slots[index])
            for index, (_, entries) in enumerate(sorted_lanes)
        ]
        for future in as_completed(futures):
            future.result()

    print("\n🏁 [Batch-Intent] Completed batch intent generation.")
    print(f"📊 Summary: {len(success_list)} success, {len(failure_list)} failure, {skipped_count} skipped.")
    print(f"📝 Detailed report written to: {BATCH_INTENT_LOG}")

if __name__ == "__main__":
    main()
