from __future__ import annotations

import argparse
import json
from pathlib import Path

from code_intent.env_utils import load_toenv
from code_intent.intent_extractor import (
    build_function_intents,
    build_test_intents,
    intent_function_artifacts_ready,
    intent_test_artifacts_ready,
)

STAGES = ("functions", "tests", "features")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Code Intent 构建示例程序")
    parser.add_argument(
        "--instance-id",
        required=True,
        help="SweBench instance_id（每个实例唯一）",
    )
    parser.add_argument(
        "--file-path",
        required=True,
        help="实例源码根目录（用于产出元数据）",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="实例输出根目录（每个实例的 intent graph 会写入此处）",
    )
    parser.add_argument(
        "--from-stage",
        choices=("auto", *STAGES),
        default="auto",
        help=(
            "从指定阶段开始强制重建（functions/tests/features）。"
            "默认 auto 为按需构建（若文件已存在则跳过）。"
        ),
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=0,
        help="构建完 feature 后预览前 N 条记录（默认不预览）",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="仅生成并预览 feature（不继续执行后续流程）",
    )
    parser.add_argument(
        "--function-intent-limit",
        type=int,
        default=0,
        help="仅构建前 N 个函数 intent，0 表示全部",
    )
    parser.add_argument(
        "--functions-only",
        action="store_true",
        help="只构建函数 intent，完成后立即退出",
    )
    parser.add_argument(
        "--tests-only",
        action="store_true",
        help="只增量构建测试 intent 及其向量索引",
    )
    return parser.parse_args()


def _stage_plan(from_stage: str) -> tuple[dict[str, bool], dict[str, bool]]:
    if from_stage == "auto":
        enabled = {stage: True for stage in STAGES}
        force = {stage: False for stage in STAGES}
        return enabled, force
    start_idx = STAGES.index(from_stage)
    enabled = {stage: idx >= start_idx for idx, stage in enumerate(STAGES)}
    force = enabled.copy()
    return enabled, force

def _preview_features(features_path: Path, limit: int) -> None:
    if limit <= 0 or not features_path.exists():
        return
    print(f"\n👀 预览前 {limit} 条 feature：")
    shown = 0
    with features_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            feature = json.loads(line)
            shown += 1
            print(
                json.dumps(
                    {
                        "feature_id": feature.get("feature_id"),
                        "feature_name": feature.get("feature_name"),
                        "feature_type": feature.get("feature_type"),
                        "tags": feature.get("tags", [])[:5],
                    },
                    ensure_ascii=False,
                )
            )
            if shown >= limit:
                break
    if shown == 0:
        print("(features.jsonl 为空)")


def _prepare_instance_dirs(instance_id: str, output_path: str) -> tuple[Path, Path]:
    base = Path(output_path).expanduser()
    base.mkdir(parents=True, exist_ok=True)
    instance_root = base / instance_id
    instance_root.mkdir(parents=True, exist_ok=True)
    index_dir = instance_root / "index_output"
    index_dir.mkdir(parents=True, exist_ok=True)
    return instance_root, index_dir


def main() -> None:
    args = _parse_args()
    enabled, force = _stage_plan(args.from_stage)
    if args.functions_only and args.tests_only:
        raise ValueError("--functions-only and --tests-only cannot be combined")
    if args.tests_only:
        enabled = {"functions": False, "tests": True, "features": False}
        force = {"functions": False, "tests": True, "features": False}

    load_toenv()
    instance_root, index_dir = _prepare_instance_dirs(args.instance_id, args.output_path)
    cache_namespace = args.instance_id.rsplit("-", 1)[0]
    print(f"📁 当前实例: {args.instance_id} | 输出目录: {instance_root}")
    functions_file = index_dir / "functions.jsonl"
    if not functions_file.exists():
        raise FileNotFoundError(
            f"{functions_file} 不存在，请先运行 code_indexer 生成基础 JSONL。"
        )

    function_intents_file = index_dir / "function_intents.jsonl"
    test_intents_file = index_dir / "test_intents.jsonl"

    # 函数级 intent
    if enabled["functions"]:
        if force["functions"] or not intent_function_artifacts_ready(index_dir):
            print("🚀 构建函数级 intent ...")
            func_limit = (
                args.function_intent_limit if args.function_intent_limit > 0 else None
            )
            build_function_intents(
                index_dir,
                max_functions=func_limit,
                repo_id=cache_namespace,
            )
        else:
            print("✅ 检测到完整且输入一致的函数 intent 索引，跳过构建。")
        if args.functions_only:
            return
    else:
        print("⏭️ 跳过函数级 intent（from-stage 设置）")

    # 测试 intent
    if enabled["tests"]:
        if force["tests"] or not intent_test_artifacts_ready(
            index_dir,
            require_current_inputs=True,
            require_runtime_embedding=True,
        ):
            print("🚀 构建测试 intent ...")
            build_test_intents(
                index_dir,
                repo_id=cache_namespace,
                repo_root=args.file_path,
            )
        else:
            print("✅ 检测到完整测试 intent 索引，跳过测试 intent 构建。")
        if args.tests_only:
            return
    else:
        print("⏭️ 跳过测试 intent（from-stage 设置）")

    # feature-level intent
    if enabled["features"]:
        from code_intent.feature_builder import (
            build_feature_intents,
            intent_feature_artifacts_ready,
        )

        if force["features"] or not intent_feature_artifacts_ready(index_dir):
            print("🚀 构建 feature-level intent ...")
            build_feature_intents(index_dir, repo_id=cache_namespace)
        else:
            print("✅ 检测到完整且输入一致的 feature intent 索引，跳过构建。")
        _preview_features(index_dir / "features.jsonl", args.preview_count)
        if args.preview_only:
            return
    else:
        print("⏭️ 跳过 feature-level intent（from-stage 设置）")

    print("🔎 初始化 RAG 检索器 ...")
    from code_intent.search_api import CodeIntentRAG

    rag = CodeIntentRAG(index_dir)

    query = "login request fails when JWT token is expired"
    print(f"\n🔍 测试查询: {query}")
    results = rag.search_code(query, top_k_funcs=10)
    for item in results:
        print(
            json.dumps(
                {
                    "score": round(item["score"], 4),
                    "func_id": item["func_id"],
                    "file": item.get("file"),
                    "feature": item.get("feature_name"),
                },
                ensure_ascii=False,
            )
        )

    manifest = {
        "instance_id": args.instance_id,
        "source_path": str(Path(args.file_path).expanduser()),
        "index_dir": str(index_dir),
        "function_intents": str(function_intents_file)
        if function_intents_file.exists()
        else None,
        "test_intents": str(test_intents_file) if test_intents_file.exists() else None,
        "features": str(index_dir / "features.jsonl")
        if (index_dir / "features.jsonl").exists()
        else None,
    }
    manifest_path = instance_root / "intent_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"📝 已更新清单: {manifest_path}")


if __name__ == "__main__":
    main()
