from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_BIN = sys.executable


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
from INTENT2FIX.env_paths import build_intent_repo_cache_default  # noqa: E402

CACHE_DIR = build_intent_repo_cache_default()


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip(), 10)
    except ValueError:
        return default


def _env_cmake_args() -> Optional[List[str]]:
    raw = (os.environ.get("BUILD_INTENT_CMAKE_ARGS") or "").strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.replace("|", ";").split(";") if p.strip()]
    return parts or None

def _run_step(name: str, cmd: Iterable[str]) -> None:
    cmd_display = shlex.join(cmd)
    print(f"\n▶️ {name}: {cmd_display}")
    try:
        subprocess.check_call(list(cmd), cwd=PROJECT_ROOT)
    except subprocess.CalledProcessError as exc:
        print(f"❌ {name} 失败，退出码 {exc.returncode}")
        raise


def build_intent(
    instance_id: str,
    file_path: str,
    output_path: str,
    *,
    build_dir: str | None = None,
    cmake_args: Optional[List[str]] = None,
    skip_index: bool = False,
    skip_intents: bool = False,
    from_stage: str = "auto",
    function_intent_limit: int = 0,
    functions_only: bool = False,
    preview_count: int = 0,
    preview_only: bool = False,
    max_workers: int = 0,
    shell_after_index: bool = False,
) -> None:
    demo_cmd = [
        PYTHON_BIN,
        str(PROJECT_ROOT / "demo.py"),
        "--instance-id",
        instance_id,
        "--file-path",
        file_path,
        "--output-path",
        output_path,
    ]
    if max_workers > 0:
        demo_cmd.extend(["--max-workers", str(max_workers)])
    if build_dir:
        demo_cmd.extend(["--build-dir", build_dir])
    if cmake_args:
        for extra in cmake_args:
            demo_cmd.append(f"--cmake-arg={extra}")
    include_dir = Path(file_path) / "include"
    if include_dir.exists():
        demo_cmd.extend(["--extra-header-dir", str(include_dir)])
    abi_include_dir = Path(file_path) / "tests" / "abi" / "include"
    if abi_include_dir.exists():
        demo_cmd.extend(["--extra-header-dir", str(abi_include_dir)])
    if shell_after_index:
        demo_cmd.append("--shell-after-index")

    example_cmd = [
        PYTHON_BIN,
        str(PROJECT_ROOT / "example.py"),
        "--instance-id",
        instance_id,
        "--file-path",
        file_path,
        "--output-path",
        output_path,
        "--from-stage",
        from_stage,
        "--function-intent-limit",
        str(function_intent_limit),
        "--preview-count",
        str(preview_count),
    ]

    if functions_only:
        example_cmd.append("--functions-only")
    if preview_only:
        example_cmd.append("--preview-only")

    if not skip_index:
        _run_step("构建索引 (demo.py)", demo_cmd)
    else:
        print("⏭️ 跳过索引阶段（用户指定）")

    if not skip_intents:
        _run_step("构建 intents (example.py)", example_cmd)
    else:
        print("⏭️ 跳过 intent 阶段（用户指定）")

    print("\n✅ build_intent.py 执行完成。")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="一键构建 SweBench 实例的索引与 Intent Graph；路径与 INDEX_STORE_DIR / REPO_DIR 对齐，其余见 BUILD_INTENT_*。",
    )
    parser.add_argument(
        "--instance-id",
        default=(os.environ.get("BUILD_INTENT_INSTANCE_ID") or "").strip(),
        help="SweBench instance_id（或 BUILD_INTENT_INSTANCE_ID）。无默认值。",
    )
    parser.add_argument(
        "--file-path",
        default=(os.environ.get("BUILD_INTENT_FILE_PATH") or "").strip(),
        help=(
            "实例源码仓库根目录的绝对路径（.git/顶层工程文件所在层；或 BUILD_INTENT_FILE_PATH）。"
            "无默认值，须与 .env 或 CLI 二选一提供。"
        ),
    )
    parser.add_argument(
        "--output-path",
        default=(os.environ.get("INDEX_STORE_DIR") or "").strip(),
        help="索引与 intent 输出根目录；默认读 INDEX_STORE_DIR。",
    )
    parser.add_argument(
        "--build-dir",
        default=(os.environ.get("BUILD_INTENT_BUILD_DIR") or "").strip(),
        help="含 compile_commands.json 的构建目录，可选（或 BUILD_INTENT_BUILD_DIR）；默认空。",
    )
    parser.add_argument(
        "--skip-index",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("BUILD_INTENT_SKIP_INDEX"),
        help="跳过索引；默认可由 BUILD_INTENT_SKIP_INDEX=1 打开，命令行可用 --no-skip-index 关闭",
    )
    parser.add_argument(
        "--skip-intents",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("BUILD_INTENT_SKIP_INTENTS"),
        help="跳过 intent；或 BUILD_INTENT_SKIP_INTENTS=1",
    )
    parser.add_argument(
        "--index-only",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("BUILD_INTENT_INDEX_ONLY"),
        help="同只跑索引；或 BUILD_INTENT_INDEX_ONLY=1（与 --skip-intents 任一即可）",
    )
    parser.add_argument(
        "--from-stage",
        choices=("auto", "functions", "tests", "features"),
        default=os.environ.get("BUILD_INTENT_FROM_STAGE", "auto").strip() or "auto",
    )
    parser.add_argument(
        "--function-intent-limit",
        type=int,
        default=_env_int("BUILD_INTENT_FUNCTION_INTENT_LIMIT", 0),
    )
    parser.add_argument(
        "--functions-only",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("BUILD_INTENT_FUNCTIONS_ONLY"),
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=_env_int("BUILD_INTENT_PREVIEW_COUNT", 0),
    )
    parser.add_argument(
        "--preview-only",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("BUILD_INTENT_PREVIEW_ONLY"),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=_env_int("BUILD_INTENT_MAX_WORKERS", 5),
        help="并行索引进程数，0 自动；或 BUILD_INTENT_MAX_WORKERS",
    )
    parser.add_argument(
        "--shell-after-index",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("BUILD_INTENT_SHELL_AFTER_INDEX"),
        help="索引后进 shell；或 BUILD_INTENT_SHELL_AFTER_INDEX=1",
    )
    args = parser.parse_args()
    if not args.instance_id:
        parser.error("缺少 --instance-id 或环境变量 BUILD_INTENT_INSTANCE_ID")
    if not args.file_path:
        parser.error("缺少 --file-path 或环境变量 BUILD_INTENT_FILE_PATH")
    if not args.output_path:
        parser.error("缺少 --output-path 或环境变量 INDEX_STORE_DIR")
    return args


def _cli_main() -> None:
    args = _parse_args()
    build_dir = args.build_dir or None
    os.environ["PROJECT_ROOT_OVERRIDE"] = args.file_path
    build_intent(
        instance_id=args.instance_id,
        file_path=args.file_path,
        output_path=args.output_path,
        build_dir=build_dir,
        cmake_args=_env_cmake_args(),
        skip_index=args.skip_index,
        skip_intents=bool(args.skip_intents or args.index_only),
        from_stage=args.from_stage,
        function_intent_limit=args.function_intent_limit,
        functions_only=args.functions_only,
        preview_count=args.preview_count,
        preview_only=args.preview_only,
        max_workers=args.max_workers,
        shell_after_index=args.shell_after_index,
    )


def main() -> None:
    """无参数时的本地示例（Catch2）；日常请用命令行参数或 _cli_main。"""
    instance_id = "swe-bench_catchorg__Catch2-1422"
    file_path = f"{CACHE_DIR}/{instance_id}"
    output_path = f"./index/{instance_id}/intent_output"

    os.environ["PROJECT_ROOT_OVERRIDE"] = file_path

    build_dir = f"/home/Catch2/build"
    build_intent(
        instance_id=instance_id,
        file_path=file_path,
        output_path=output_path,
        build_dir=build_dir,
        cmake_args=["-DCATCH_DEVELOPMENT_BUILD=ON"],
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        _cli_main()
    else:
        main()

