#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _add_boolean_optional_argument(
    parser: argparse.ArgumentParser,
    name: str,
    default: bool,
    help_text: str,
) -> None:
    """
    等价于 ``action=argparse.BooleanOptionalAction``（``--opt`` / ``--no-opt``），兼容 Python 3.8。
    ``name`` 为长选项主体，如 ``skip-existing``（不要前缀 ``--``）。
    """
    dest = name.replace("-", "_")
    mutex = parser.add_mutually_exclusive_group()
    mutex.add_argument(
        f"--{name}",
        dest=dest,
        action="store_true",
        help=help_text,
    )
    mutex.add_argument(
        f"--no-{name}",
        dest=dest,
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(**{dest: default})


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
from INTENT2FIX.env_paths import (  # noqa: E402
    build_instances_parent,
    index_store_parent,
    repo_instances_parent,
)

_BASE = Path(__file__).resolve().parent
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))
from ponyc_build_from_run_sh import ponyc_make_build_shell_for_instance  # noqa: E402

# --- Configuration（路径见 INTENT2FIX/env_paths.py：REPO_DIR / INDEX_STORE_DIR / INTENT2FIX_BUILD_DIR）---
BASE_REPO_DIR = str(repo_instances_parent())
BASE_OUTPUT_DIR = str(index_store_parent())
BASE_BUILD_DIR = str(build_instances_parent())
DEMO_SCRIPT = os.environ.get("BATCH_INDEX_DEMO_SCRIPT", str(_BASE / "demo.py"))
BATCH_SUMMARY_LOG = os.environ.get(
    "BATCH_INDEX_SUMMARY_LOG", str(_BASE / "batch_index_summary.log")
)

_summary_lock = threading.Lock()


def _append_summary_line(path: Path, line: str) -> None:
    """多线程写同一份 summary 时加锁。"""
    with _summary_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)


def _repo_name_from_instance_id(instance_id: str) -> str:
    """
    从 instance_id 解析仓库目录名（镜像内通常为 /home/<repo>）。

    例如 ``facebook__zstd-637`` → ``zstd``；
    ``swe-bench_catchorg__Catch2-2849`` → ``Catch2``（最后一段 ``-<PR>`` 为 PR 号）。
    """
    m = re.match(r"^swe-bench_[^_]+__(.+)-(\d+)$", instance_id)
    if m:
        return m.group(1)
    m = re.match(r"^[^_]+__(.+)-(\d+)$", instance_id)
    if m:
        return m.group(1)
    if "__" in instance_id:
        return instance_id.split("__", 1)[-1]
    return instance_id


def _docker_repo_path(instance_id: str) -> str:
    """容器内源码路径：``/home/<repo_name>``，``repo_name`` 由 ``_repo_name_from_instance_id`` 解析。"""
    return f"/home/{_repo_name_from_instance_id(instance_id)}"


# 排除列表：这些实例将被跳过
EXCLUDED_INSTANCES = [
    "swe-bench_fmtlib__fmt-4055",
    "swe-bench_nlohmann__json-4537",
    "swe-bench_simdjson__simdjson-2178",
    "swe-bench_nlohmann__json-18",
]
# ---------------------

def is_valid_instance(instance_id: str) -> bool:
    """
    Checks if the directory name matches the expected SweBench instance format.
    Format: swe-bench_<org>__<repo>-<pr_number>
    Example: swe-bench_catchorg__Catch2-2849
    Example: swe-bench_yhirose__cpp-httplib-1765 (repo name can contain hyphens)
    
    Only instances with PR numbers are considered valid.
    Instances without PR numbers (e.g., swe-bench_catchorg) are excluded.
    """
    # 要求格式: swe-bench_<org>__<repo>-<pr_number>
    # repo 名称可以包含连字符，但最后必须是 -数字 结尾（PR number）
    # 使用负向前瞻确保最后一部分是数字
    pattern = r"swe-bench_([^_]+)__(.+)-(\d+)$"
    match = re.match(pattern, instance_id)
    if not match:
        return False
    # 确保 repo 部分不为空，且最后一部分确实是 PR number（数字）
    repo_part = match.group(2)
    pr_number = match.group(3)
    return bool(repo_part and pr_number.isdigit())

def get_repo_config(instance_id: str):
    """
    Returns specific configuration for certain repositories.
    """
    config = {
        "cmake_args": [],
        "build_dir": None,
        "build_targets": None,  # None means build all targets
        "build_backend": None,  # None：不传参，由 demo 的 auto / 环境变量决定
        "make_build_shell": None,
    }

    # ponyc：``make_build_shell`` 优先由 ``INTENT2FIX/benchmark/ponyc_image_run_sh.json`` 推导
    # （镜像 ``run.sh`` 去掉末尾 ``make … test`` 等）；无 JSON 时回退 ``make``。
    if "ponylang__ponyc" in instance_id:
        config["build_backend"] = "make-bear"
        config["make_build_shell"] = ponyc_make_build_shell_for_instance(instance_id) or "make"

    # jq：固定 bear 录制命令（与 code_intent/demo.py 中 JQLANG_JQ_MAKE_BUILD_SHELL 保持一致）。
    if "jqlang__jq" in instance_id:
        config["build_backend"] = "make-bear"
        config["make_build_shell"] = (
            "autoreconf -i && ./configure --with-oniguruma=builtin && make clean && make -j8"
        )

    # Special config for Catch2
    if "catchorg__Catch2" in instance_id:
        config["cmake_args"] = ["-DCATCH_DEVELOPMENT_BUILD=ON"]
    
    # Structural indexing only needs the library compile command. Developer mode
    # configures optional benchmark dependencies and can download them, which
    # makes an otherwise local compile-database build fail offline.
    if "simdjson__simdjson" in instance_id:
        config["cmake_args"] = [
            "-DSIMDJSON_DEVELOPER_MODE=OFF",
            "-DSIMDJSON_JUST_LIBRARY=ON",
            "-DSIMDJSON_GOOGLE_BENCHMARKS=OFF",
            "-DSIMDJSON_COMPETITION=OFF",
        ]
        config["build_targets"] = ["simdjson"]
        
    return config


def _simdjson_supports_library_only(repo_path: Path) -> bool:
    """Whether this checkout can configure without benchmark submodules."""
    for candidate in (repo_path / "simdjson-flags.cmake", repo_path / "CMakeLists.txt"):
        try:
            if "SIMDJSON_JUST_LIBRARY" in candidate.read_text(errors="ignore"):
                return True
        except OSError:
            continue
    return False


def _write_legacy_simdjson_compile_database(repo_path: Path, build_dir: Path) -> Path:
    """Create a production-only compile DB for releases without JUST_LIBRARY.

    Those releases configure benchmark dependencies unconditionally and try to
    clone submodules. Structural indexing only needs a faithful command for
    the aggregate production translation unit.
    """
    source = repo_path / "src" / "simdjson.cpp"
    if not source.is_file():
        raise FileNotFoundError(f"legacy simdjson aggregate source is missing: {source}")
    build_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "/usr/bin/c++",
        "-DSIMDJSON_THREADS_ENABLED=1",
        f"-I{repo_path / 'include'}",
        f"-I{repo_path / 'src'}",
        "-std=c++17",
        "-c",
        str(source),
        "-o",
        str(build_dir / "simdjson.cpp.o"),
    ]
    output = build_dir / "compile_commands.json"
    output.write_text(
        json.dumps(
            [{"directory": str(build_dir), "arguments": command, "file": str(source)}],
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(f"✅ [Batch] legacy simdjson production compile DB: {output}")
    return output

def clear_build_cache(build_dir: str):
    """
    Clears build cache by removing the entire build directory.
    Since we're rebuilding the index anyway, it's simpler to just delete everything.

    若 ``BATCH_INDEX_SKIP_BUILD_CACHE_CLEAR=1``：不删除目录（可与容器中
    ``CODE_INTENT_SKIP_BEAR_IF_COMPILE_COMMANDS_EXISTS=1`` 联用，在同一 checkout 下复用已有
    ``compile_commands.json``；跨 commit / 跨实例复用会得到错误索引，请勿滥用）。
    
    Args:
        build_dir: Path to the build directory (e.g. ``<INTENT2FIX_BUILD_DIR>/<instance_id>``)
    """
    if _env_bool("BATCH_INDEX_SKIP_BUILD_CACHE_CLEAR"):
        print(
            f"⏭️ [Batch] 跳过清空 build 目录（BATCH_INDEX_SKIP_BUILD_CACHE_CLEAR=1）：{build_dir}"
        )
        return
    build_path = Path(build_dir)
    if not build_path.exists():
        print(f"🧹 [Batch] Build directory does not exist: {build_dir}")
        return
    
    try:
        shutil.rmtree(build_path)
        print(f"🧹 [Batch] Removed build directory: {build_dir}")
    except Exception as e:
        print(f"⚠️ [Batch] Failed to remove build directory {build_dir}: {e}")

def clear_index_cache(instance_output_dir: Path):
    """
    Clears index cache by removing file_hashes.json and cache directory.
    This forces a complete re-indexing of the instance.
    
    Args:
        instance_output_dir: Path to the instance's index_output directory
    """
    if not instance_output_dir.exists():
        return
    
    # Remove file_hashes.json
    file_hashes_path = instance_output_dir / "file_hashes.json"
    if file_hashes_path.exists():
        try:
            file_hashes_path.unlink()
            print(f"🧹 [Batch] Removed file_hashes.json: {file_hashes_path}")
        except Exception as e:
            print(f"⚠️ [Batch] Failed to remove file_hashes.json {file_hashes_path}: {e}")
    
    # Remove cache directory
    cache_dir = instance_output_dir / "cache"
    if cache_dir.exists():
        try:
            shutil.rmtree(cache_dir)
            print(f"🧹 [Batch] Removed cache directory: {cache_dir}")
        except Exception as e:
            print(f"⚠️ [Batch] Failed to remove cache directory {cache_dir}: {e}")


def _run_batch_host_index_phase(
    instance_id: str,
    repo_path: str,
    output_base: str,
    build_dir: str,
    docker_repo: str,
    repo_cfg: Dict[str, Any],
) -> None:
    """
    两阶段流水线第 2 步：宿主机 ``demo.py --host-index``（在容器 ``--compile-db-only`` 成功后调用）。
    """
    host_repo = str(Path(repo_path).resolve())
    bd = str(Path(build_dir).resolve())
    path_map = f"{docker_repo}|{host_repo}"
    host_cmd: List[str] = [
        sys.executable,
        str(DEMO_SCRIPT),
        "--host-index",
        "--instance-id",
        instance_id,
        "--file-path",
        host_repo,
        "--output-path",
        output_base,
        "--build-dir",
        bd,
        "--path-map",
        path_map,
    ]
    for arg in repo_cfg.get("cmake_args") or []:
        host_cmd.append(f"--cmake-arg={arg}")
    if repo_cfg.get("build_backend"):
        host_cmd.extend(["--build-backend", str(repo_cfg["build_backend"])])
    if repo_cfg.get("make_build_shell"):
        host_cmd.extend(["--make-build-shell", str(repo_cfg["make_build_shell"])])

    print(f"🖥️ [Batch] 阶段 2/2：宿主机 libclang 索引 → {instance_id}")
    print(f"💻 [Batch] {' '.join(host_cmd)}")
    host_env = os.environ.copy()
    host_env["CODE_INTENT_HOST_REPO_PATH_FOR_META"] = host_repo
    # ponyc：compile_commands 常含 LLVM；排除 lib/llvm/ 并避免 header 全仓 rglob 扫进子模块
    if instance_id.startswith("ponylang__ponyc"):
        ex_key = "CODE_INTENT_INDEX_EXCLUDE_PATH_SUBSTR"
        cur = (host_env.get(ex_key) or "").strip()
        parts = [x.strip() for x in cur.split(",") if x.strip()]
        llvm_tok = "lib/llvm/"
        if not any(llvm_tok in x.replace("\\", "/").lower() for x in parts):
            parts.append(llvm_tok)
            host_env[ex_key] = ",".join(parts)
            print(f"🔧 [Batch] ponyc 索引默认排除 LLVM 路径段: {ex_key}={host_env[ex_key]!r}")
    # 避免宿主 demo 仍把 COMPILE_DB_ONLY 当成「仅录库」语义干扰
    host_env.pop("CODE_INTENT_COMPILE_DB_ONLY", None)
    subprocess.check_call(host_cmd, cwd=str(_REPO_ROOT), env=host_env)


def load_instance_ids_from_summary_log(path: Path) -> List[str]:
    """
    从 ``batch_index_summary.log`` 的 ``Successful Instances:`` 小节解析实例 id（每行 `` - <id>``）。
    """
    out: List[str] = []
    in_section = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip() == "Successful Instances:":
            in_section = True
            continue
        if not in_section:
            continue
        if line.strip().startswith("End Time:"):
            break
        if line.startswith(" - "):
            out.append(line[3:].strip())
        elif out and line.strip() != "" and not line.startswith(" "):
            break
    return out


def load_instance_ids_from_batch_file(path: Path) -> List[str]:
    """每行一个 instance_id，``#`` 后为注释。"""
    out: List[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.split("#", 1)[0].strip()
        if s:
            out.append(s)
    return out


def _dedupe_preserve_order(ids: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in ids:
        x = x.strip()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def run_host_index_batch(
    instance_ids: List[str],
    repo_base: Path,
    output_base: Path,
    build_base: Path,
) -> None:
    """
    批量阶段 2：逐条跑 ``_run_batch_host_index_phase``，实时打印成功/失败计数，最后汇总。
    """
    ids = _dedupe_preserve_order(instance_ids)
    if not ids:
        print("❌ [HostBatch] 实例列表为空")
        return
    output_base.mkdir(parents=True, exist_ok=True)
    build_base.mkdir(parents=True, exist_ok=True)
    ok: List[str] = []
    fail: List[Tuple[str, str]] = []
    total = len(ids)
    print(f"📋 [HostBatch] 共 {total} 个实例；REPO={repo_base} BUILD={build_base} OUT={output_base}")
    for n, iid in enumerate(ids, 1):
        repo_path = str(repo_base / iid)
        if not Path(repo_path).is_dir():
            msg = f"仓库目录不存在: {repo_path}"
            fail.append((iid, msg))
            print(f"❌ [HostBatch {n}/{total}] {iid} — {msg} | 成功={len(ok)} 失败={len(fail)}")
            continue
        repo_cfg = get_repo_config(iid)
        build_dir = repo_cfg["build_dir"] or str(build_base / iid)
        cc = Path(build_dir) / "compile_commands.json"
        if not cc.is_file():
            msg = f"无 compile_commands.json: {cc}"
            fail.append((iid, msg))
            print(f"❌ [HostBatch {n}/{total}] {iid} — {msg} | 成功={len(ok)} 失败={len(fail)}")
            continue
        docker_repo = _docker_repo_path(iid)
        try:
            _run_batch_host_index_phase(
                iid, repo_path, str(output_base), build_dir, docker_repo, repo_cfg
            )
            ok.append(iid)
            print(f"✅ [HostBatch {n}/{total}] {iid} | 成功={len(ok)} 失败={len(fail)}")
        except subprocess.CalledProcessError as e:
            fail.append((iid, f"exit {e.returncode}"))
            print(f"❌ [HostBatch {n}/{total}] {iid} — exit {e.returncode} | 成功={len(ok)} 失败={len(fail)}")
        except Exception as e:
            fail.append((iid, str(e)))
            print(f"❌ [HostBatch {n}/{total}] {iid} — {e} | 成功={len(ok)} 失败={len(fail)}")

    print("\n" + "=" * 60)
    print(f"🏁 [HostBatch] 结束：成功 {len(ok)}，失败 {len(fail)}，合计 {total}")
    if fail:
        print("失败列表:")
        for iid, err in fail:
            print(f"  - {iid}: {err}")
    if ok:
        print("成功列表:")
        for iid in ok:
            print(f"  - {iid}")


def run_indexing(instance_id: str, repo_path: str, output_base: str, build_base: str) -> None:
    """
    Runs the demo.py script for a specific instance.

    ``repo_path``：宿主机克隆路径，仅用于 ``git reset``/``clean``；传给 demo 的源码路径为容器内
    ``/home/<repo_name>``（由 instance_id 解析出 repo，如 ``facebook__zstd-637`` → ``/home/zstd``）。
    构建目录为 ``os.path.join(build_base, instance_id)``，其中 ``build_base`` 来自环境变量
    ``INTENT2FIX_BUILD_DIR``（或 CLI ``--build-dir``），与 ``get_repo_config`` 中的 ``build_dir`` 覆盖一致。

    未设置 ``BATCH_INDEX_COMPILE_DB_ONLY`` 时：单次 ``demo.py`` 在容器内完成构建与索引（单阶段）。

    设置 ``BATCH_INDEX_COMPILE_DB_ONLY=1`` 时：阶段 1 容器内生成 ``compile_commands.json`` 后，**自动**
    在本机再执行 ``demo.py --host-index``（阶段 2），写入 ``output_base/<instance_id>/index_output``。

    若阶段 1 已跑完：单实例 ``--host-index-only --instance-id <id>``；或
    ``--host-index-from-summary-log <batch_index_summary.log>`` / ``--host-index-batch-file <ids.txt>`` 批量阶段 2（实时统计）。

    Args:
        instance_id: Instance ID
        repo_path: 宿主机上的仓库路径
        output_base: 索引输出根目录（宿主机）
        build_base: 各实例 build 的父目录（宿主机），默认 ``INTENT2FIX_BUILD_DIR``
    """
    # 确保仓库处于初始状态
    if os.path.exists(os.path.join(repo_path, ".git")):
        print(f"🧹 [Batch] Resetting repository to initial state: {instance_id}")
        try:
            # git reset --hard HEAD 恢复文件，git clean -fd 删除未追踪的文件和目录
            subprocess.check_call(["git", "reset", "--hard", "HEAD"], cwd=repo_path)
            subprocess.check_call(["git", "clean", "-fd"], cwd=repo_path)
        except subprocess.CalledProcessError as e:
            print(f"⚠️ [Batch] Git reset failed for {instance_id}, continuing anyway...")

    repo_cfg = get_repo_config(instance_id)

    docker_repo = _docker_repo_path(instance_id)
    build_dir = repo_cfg["build_dir"] or os.path.join(build_base, instance_id)
    os.makedirs(build_dir, exist_ok=True)
    print(f"🧹 [Batch] Clearing build cache for: {instance_id}")
    clear_build_cache(build_dir)

    # 清除 index cache (file_hashes.json 和 cache 目录)
    instance_output_dir = Path(output_base) / instance_id / "index_output"
    print(f"🧹 [Batch] Clearing index cache for: {instance_id}")
    clear_index_cache(instance_output_dir)

    if (
        _env_bool("BATCH_INDEX_COMPILE_DB_ONLY")
        and "simdjson__simdjson" in instance_id
        and not _simdjson_supports_library_only(Path(repo_path))
    ):
        print(
            "📦 [Batch] legacy simdjson has no SIMDJSON_JUST_LIBRARY; "
            "using a production-only host compile database"
        )
        _write_legacy_simdjson_compile_database(Path(repo_path), Path(build_dir))
        _run_batch_host_index_phase(
            instance_id, repo_path, output_base, build_dir, docker_repo, repo_cfg
        )
        return

    cmd = [
        sys.executable,
        DEMO_SCRIPT,
        "--instance-id", instance_id,
        "--file-path", docker_repo,
        "--output-path", output_base,
        "--build-dir", build_dir,
    ]

    # 添加 CMake 参数
    for arg in repo_cfg["cmake_args"]:
        cmd.append(f"--cmake-arg={arg}")

    if repo_cfg.get("build_backend"):
        cmd.extend(["--build-backend", repo_cfg["build_backend"]])
    if repo_cfg.get("make_build_shell"):
        cmd.extend(["--make-build-shell", repo_cfg["make_build_shell"]])
    
    # 添加构建目标（如果指定）
    if repo_cfg.get("build_targets"):
        for target in repo_cfg["build_targets"]:
            cmd.extend(["--build-target", target])

    if _env_bool("BATCH_INDEX_COMPILE_DB_ONLY"):
        cmd.append("--compile-db-only")

    # 检查并添加 extra-header-dir（与 build_intent.py 逻辑一致）
    include_dir = Path(docker_repo) / "include"
    if include_dir.exists():
        cmd.extend(["--extra-header-dir", str(include_dir)])
        print(f"📁 [Batch] Found include directory: {include_dir}")

    abi_include_dir = Path(docker_repo) / "tests" / "abi" / "include"
    if abi_include_dir.exists():
        cmd.extend(["--extra-header-dir", str(abi_include_dir)])
        print(f"📁 [Batch] Found ABI include directory: {abi_include_dir}")

    print(f"🚀 [Batch] Starting indexing for instance: {instance_id}")
    if _env_bool("BATCH_INDEX_COMPILE_DB_ONLY"):
        print("📦 [Batch] 两阶段模式：阶段 1/2 容器内 --compile-db-only，结束后自动阶段 2 宿主机 --host-index")
    print(f"📂 [Batch] 容器源码: {docker_repo}（宿主机 clone: {repo_path}）")
    print(f"🏗️ [Batch] 构建目录: {build_dir}")
    if repo_cfg["cmake_args"]:
        print(f"🔧 [Batch] Extra CMake args: {repo_cfg['cmake_args']}")
    print(f"💻 [Batch] Running command: {' '.join(cmd)}")
    
    # 供容器内 --compile-db-only 写 instance_meta 时生成 path-map 建议（容器源码根|本路径）
    run_env = os.environ.copy()
    run_env["CODE_INTENT_HOST_REPO_PATH_FOR_META"] = os.path.abspath(repo_path)

    # ponyc：若宿主 clone 已含 lib/llvm/src（子模块在 materialize 阶段拉好），挂载进容器，
    # 否则容器内 make 仍会走 get-submodule 从 GitHub 再 clone LLVM。
    host_repo_abs = Path(repo_path).expanduser().resolve()
    llvm_src = host_repo_abs / "lib" / "llvm" / "src"
    bind_key = "CODE_INTENT_HOST_REPO_BIND"
    if not (run_env.get(bind_key) or "").strip():
        if (
            instance_id.startswith("ponylang__ponyc")
            and host_repo_abs.is_dir()
            and (host_repo_abs / ".git").exists()
            and llvm_src.is_dir()
            and any(llvm_src.iterdir())
        ):
            run_env[bind_key] = str(host_repo_abs)
            print(
                f"📂 [Batch] ponyc：宿主已检出 LLVM，自动 {bind_key} → {host_repo_abs} "
                f"（挂载到容器 {docker_repo}，避免容器内重复 clone）"
            )

    # ccache：若设置 BATCH_INDEX_CCACHE_DIR，传给容器让 demo.py 挂载并启用。
    ccache_host_dir = (os.environ.get("BATCH_INDEX_CCACHE_DIR") or "").strip()
    if ccache_host_dir:
        run_env.setdefault("CODE_INTENT_CCACHE_DIR", ccache_host_dir)
        run_env.setdefault("CODE_INTENT_USE_CCACHE", "1")
        Path(ccache_host_dir).expanduser().mkdir(parents=True, exist_ok=True)

    subprocess.check_call(cmd, env=run_env)

    if _env_bool("BATCH_INDEX_COMPILE_DB_ONLY"):
        _run_batch_host_index_phase(
            instance_id, repo_path, output_base, build_dir, docker_repo, repo_cfg
        )


def get_pr_number(instance_id: str) -> int:
    """提取 instance_id 中的 PR 号码并转为整数"""
    match = re.search(r"-(\d+)$", instance_id)
    return int(match.group(1)) if match else 0

def is_index_complete(instance_output_dir: Path) -> bool:
    """
    检查实例的索引是否完整（用于断点续跑）。
    
    Args:
        instance_output_dir: 实例的 index_output 目录路径
        
    Returns:
        True 如果索引文件存在且看起来完整
    """
    if not instance_output_dir.exists():
        return False
    
    # 检查关键文件是否存在
    required_files = [
        "functions.jsonl",
        "callgraph.jsonl",
        "classes.jsonl",
    ]
    
    for filename in required_files:
        file_path = instance_output_dir / filename
        if not file_path.exists():
            return False
        # 检查文件大小，空文件可能表示未完成
        if file_path.stat().st_size == 0:
            return False
    
    return True

def _normalize_batch_instance_id(raw: str) -> str:
    """日志里可能出现 ``swe-bench_org__repo-123``，与 JSONL 的 ``org__repo-123`` 对齐。"""
    s = (raw or "").strip()
    if s.startswith("swe-bench_"):
        return s[len("swe-bench_") :]
    return s


def _instance_in_log(instance_id: str, logged_ids: set[str]) -> bool:
    return _normalize_batch_instance_id(instance_id) in logged_ids


def load_completed_instances_from_log(log_path: Path) -> set:
    """
    从日志文件中加载已成功完成的实例列表（用于断点续跑）。
    
    Args:
        log_path: 日志文件路径
        
    Returns:
        已成功完成的实例 ID 集合
    """
    completed = set()
    if not log_path.exists():
        return completed
    
    try:
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                # 匹配 [SUCCESS] 开头的行
                match = re.search(r"\[SUCCESS\]\s+(\S+)", line)
                if match:
                    completed.add(_normalize_batch_instance_id(match.group(1)))
    except Exception as e:
        print(f"⚠️ [Batch] Failed to load completed instances from log: {e}")
    
    return completed


def load_failed_instances_from_log(log_path: Path) -> set[str]:
    """
    从 ``batch_index_summary.log`` 等日志中解析曾标记为 ``[FAILURE]`` 的实例 ID
    （兼容旧日志中的 ``[ERROR]``），供 ``--retry-failed`` 只重跑这些实例。
    """
    failed: set[str] = set()
    if not log_path.exists():
        return failed
    try:
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                # 与写日志统一为 [FAILURE]；仍兼容旧日志里的 [ERROR]。
                m = re.search(r"\[(?:FAILURE|ERROR)\]\s+(\S+)", line)
                if m:
                    failed.add(_normalize_batch_instance_id(m.group(1)))
    except Exception as e:
        print(f"⚠️ [Batch] Failed to load failed instances from log: {e}")
    return failed


def _iter_log_files_in_dir(dir_path: Path, *, recursive: bool) -> List[Path]:
    if not dir_path.is_dir():
        return []
    if recursive:
        return sorted(p for p in dir_path.rglob("*.log") if p.is_file())
    return sorted(p for p in dir_path.glob("*.log") if p.is_file())


def load_completed_instances_from_log_dir(dir_path: Path, *, recursive: bool = False) -> set[str]:
    """合并目录下多个 summary ``*.log`` 中的 ``[SUCCESS]`` id（去重）。"""
    out: set[str] = set()
    for p in _iter_log_files_in_dir(dir_path, recursive=recursive):
        out |= load_completed_instances_from_log(p)
    return out


def load_failed_instances_from_log_dir(dir_path: Path, *, recursive: bool = False) -> set[str]:
    """合并目录下多个 ``*.log`` 中的 ``[FAILURE]`` / ``[ERROR]`` id。"""
    out: set[str] = set()
    for p in _iter_log_files_in_dir(dir_path, recursive=recursive):
        out |= load_failed_instances_from_log(p)
    return out


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


def _env_str(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _parallel_instances_default() -> int:
    """BATCH_INDEX_PARALLEL_INSTANCES 优先；否则兼容 BATCH_INDEX_MAX_WORKERS。"""
    p = _env_int("BATCH_INDEX_PARALLEL_INSTANCES", 0)
    if p > 0:
        return max(1, p)
    m = _env_int("BATCH_INDEX_MAX_WORKERS", 0)
    if m > 0:
        return max(1, m)
    return 1


def git_clone_url(repo: str, git_base_url: str) -> str:
    """repo 如 jqlang/jq → https://github.com/jqlang/jq.git"""
    base = (git_base_url or "https://github.com").rstrip("/")
    r = repo.strip()
    if r.endswith(".git"):
        return f"{base}/{r}"
    return f"{base}/{r}.git"


def _sanitize_upstream_dirname(repo: str) -> str:
    """jqlang/jq → jqlang__jq，用作缓存目录名。"""
    return str(repo).strip().replace("/", "__").replace(":", "_")


def _unique_repos_preserve_order(rows: List[Dict[str, Any]]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for r in rows:
        rp = str(r.get("repo", "")).strip()
        if not rp or rp in seen:
            continue
        seen.add(rp)
        out.append(rp)
    return out


def _git_environ() -> dict[str, str]:
    """传给 git 子进程的环境（可扩展）。"""
    return os.environ.copy()


def _git_submodule_should_update(repo: Path, *, upstream_repo_slug: str | None = None) -> bool:
    """
    是否在宿主执行 ``git submodule update``。

    - ``ponylang__ponyc-<PR>`` 实例目录或上游 ``ponylang/ponyc``：默认 **会** 拉子模块（LLVM），
      避免进容器 ``make`` 时再全量 clone。
    - 其它仓库：默认不拉；设 ``BATCH_INDEX_FETCH_SUBMODULES=1`` 则 **任何** 仓库都拉。
    - ``BATCH_INDEX_SKIP_SUBMODULES=1``：强制不拉（覆盖 ponyc 默认）。
    """
    if _env_bool("BATCH_INDEX_SKIP_SUBMODULES"):
        return False
    if _env_bool("BATCH_INDEX_FETCH_SUBMODULES"):
        return True
    name = repo.resolve().name
    if name.startswith("ponylang__ponyc"):
        return True
    slug = (upstream_repo_slug or "").strip().lower().replace(" ", "")
    return slug == "ponylang/ponyc"


def _ponyc_repo_for_llvm_reference(repo: Path, *, upstream_repo_slug: str | None = None) -> bool:
    """是否对 ``--reference`` 使用 LLVM 镜像（仅 ponyc，避免误伤其它仓库的子模块）。"""
    if repo.resolve().name.startswith("ponylang__ponyc"):
        return True
    slug = (upstream_repo_slug or "").strip().lower().replace(" ", "")
    return slug == "ponylang/ponyc"


def _llvm_submodule_reference_argv(repo: Path, *, upstream_repo_slug: str | None = None) -> List[str]:
    """
    若设置 ``BATCH_INDEX_LLVM_SUBMODULE_REFERENCE`` 且为 ponyc，则附加
    ``git submodule update … --reference <mirror>``，复用本地 ``git clone --mirror`` 的 llvm 对象库。
    """
    raw = (os.environ.get("BATCH_INDEX_LLVM_SUBMODULE_REFERENCE") or "").strip()
    if not raw or not _ponyc_repo_for_llvm_reference(repo, upstream_repo_slug=upstream_repo_slug):
        return []
    refp = Path(raw).expanduser().resolve()
    if not refp.exists():
        print(f"⚠️ [Clone] BATCH_INDEX_LLVM_SUBMODULE_REFERENCE 路径不存在，忽略 --reference: {refp}")
        return []
    print(f"📥 [Clone] LLVM 子模块将使用 --reference: {refp}")
    return ["--reference", str(refp)]


def _git_submodule_update_safe(repo: Path, *, upstream_repo_slug: str | None = None) -> None:
    """在宿主同步 git 子模块（见 ``_git_submodule_should_update``）。"""
    if not _git_submodule_should_update(repo, upstream_repo_slug=upstream_repo_slug):
        return
    print(f"📥 [Clone] 宿主同步子模块（含 LLVM 等），可能需要较长时间: {repo}")
    ref_args = _llvm_submodule_reference_argv(repo, upstream_repo_slug=upstream_repo_slug)
    try:
        subprocess.check_call(
            ["git", "-C", str(repo), "submodule", "sync", "--recursive"],
            timeout=7200,
            env=_git_environ(),
        )
        subprocess.check_call(
            ["git", "-C", str(repo), "submodule", "update", "--init", "--recursive", *ref_args],
            timeout=7200,
            env=_git_environ(),
        )
    except subprocess.CalledProcessError:
        print(f"⚠️ [Clone] submodule update 失败（可忽略）: {repo}")


def ensure_upstream_mirror(
    cache_root: Path,
    repo: str,
    *,
    git_base_url: str,
    force_upstream: bool,
) -> Path:
    """对单个上游 repo 做一次全量 clone，供各实例本地 clone。"""
    cache_root.mkdir(parents=True, exist_ok=True)
    name = _sanitize_upstream_dirname(repo)
    dest = (cache_root / name).resolve()
    url = git_clone_url(repo, git_base_url)

    if force_upstream and dest.exists():
        print(f"🧹 [Upstream] BATCH_INDEX_FORCE_UPSTREAM=1，删除缓存 {dest}")
        shutil.rmtree(dest)

    if (dest / ".git").is_dir():
        if _env_bool("BATCH_INDEX_SKIP_UPSTREAM_FETCH"):
            print(
                f"📦 [Upstream] 已存在，跳过 git fetch（BATCH_INDEX_SKIP_UPSTREAM_FETCH=1）：{repo} → {dest}"
            )
            return dest
        print(f"📦 [Upstream] 已存在，git fetch origin：{repo} → {dest}")
        subprocess.check_call(
            ["git", "-C", str(dest), "fetch", "origin"],
            timeout=7200,
            env=_git_environ(),
        )
        return dest

    if dest.exists():
        shutil.rmtree(dest)
    print(f"📦 [Upstream] 全量 git clone：{repo} → {dest}")
    subprocess.check_call(
        ["git", "clone", "--progress", url, str(dest)],
        timeout=7200,
        env=_git_environ(),
    )
    return dest


def materialize_instance_clone_from_upstream(
    instance_dest: Path,
    upstream_path: Path,
    base_commit: str,
    *,
    force_reclone: bool,
    upstream_repo_slug: str,
) -> None:
    """从上游全量镜像本地 git clone 到实例目录，再 checkout 到 base_commit。"""
    instance_dest = instance_dest.resolve()
    upstream_path = upstream_path.resolve()
    if not (upstream_path / ".git").is_dir():
        raise RuntimeError(f"上游缓存不是 git 仓库: {upstream_path}")

    if force_reclone and instance_dest.exists():
        print(f"🧹 [Clone] --force-reclone 删除实例目录 {instance_dest}")
        shutil.rmtree(instance_dest)

    if (instance_dest / ".git").is_dir():
        print(f"📥 [Clone] 实例已存在，fetch + checkout：{instance_dest}")
        subprocess.check_call(
            ["git", "-C", str(instance_dest), "fetch", "--all"],
            timeout=7200,
            env=_git_environ(),
        )
        subprocess.check_call(
            [
                "git",
                "-c",
                "advice.detachedHead=false",
                "-C",
                str(instance_dest),
                "checkout",
                "--force",
                base_commit,
            ],
            timeout=600,
            env=_git_environ(),
        )
        _git_submodule_update_safe(instance_dest, upstream_repo_slug=upstream_repo_slug)
        return

    if instance_dest.exists():
        shutil.rmtree(instance_dest)

    print(f"📥 [Clone] git clone（本地上游 {upstream_path.name}）→ {instance_dest}")
    subprocess.check_call(
        ["git", "clone", "--progress", str(upstream_path), str(instance_dest)],
        timeout=7200,
        env=_git_environ(),
    )
    subprocess.check_call(
        [
            "git",
            "-c",
            "advice.detachedHead=false",
            "-C",
            str(instance_dest),
            "checkout",
            "--force",
            base_commit,
        ],
        timeout=600,
        env=_git_environ(),
    )
    _git_submodule_update_safe(instance_dest, upstream_repo_slug=upstream_repo_slug)


def load_instances_from_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            iid = rec.get("instance_id")
            repo = rec.get("repo")
            commit = rec.get("base_commit")
            if not iid or not repo or not commit:
                raise ValueError(
                    f"{path}:{line_no}: 需要字段 instance_id, repo, base_commit，当前: "
                    f"instance_id={iid!r} repo={repo!r} base_commit={commit!r}"
                )
            rows.append(rec)
    return rows


def ensure_repo_at_base_commit(
    dest: Path,
    repo: str,
    base_commit: str,
    *,
    git_base_url: str,
    force_reclone: bool,
) -> None:
    """
    新目录：git clone 完整仓库后 checkout <base_commit>。
    （不能用 ``git clone -b <40位 SHA>``：远端分支名不是 commit SHA，会 clone 失败。）
    已存在 git 仓库：git fetch origin <base_commit> 后 checkout。
    """
    dest = dest.resolve()
    url = git_clone_url(repo, git_base_url)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if force_reclone and dest.exists():
        print(f"🧹 [Clone] --force-reclone 删除 {dest}")
        shutil.rmtree(dest)

    if dest.exists() and (dest / ".git").is_dir():
        print(f"📥 [Clone] 已存在仓库，fetch origin {base_commit[:12]}… + checkout : {dest}")
        subprocess.check_call(
            ["git", "-C", str(dest), "fetch", "origin", base_commit],
            timeout=3600,
            env=_git_environ(),
        )
        subprocess.check_call(
            [
                "git",
                "-c",
                "advice.detachedHead=false",
                "-C",
                str(dest),
                "checkout",
                "--force",
                base_commit,
            ],
            timeout=600,
            env=_git_environ(),
        )
        _git_submodule_update_safe(dest, upstream_repo_slug=repo)
        return

    if dest.exists():
        raise RuntimeError(f"路径已存在但不是 git 仓库，请删除或换目录: {dest}")

    print(f"📥 [Clone] git clone → {dest}，随后 checkout {base_commit[:12]}…")
    subprocess.check_call(
        ["git", "clone", "--progress", url, str(dest)],
        timeout=7200,
        env=_git_environ(),
    )
    subprocess.check_call(
        [
            "git",
            "-c",
            "advice.detachedHead=false",
            "-C",
            str(dest),
            "checkout",
            "--force",
            base_commit,
        ],
        timeout=600,
        env=_git_environ(),
    )
    _git_submodule_update_safe(dest, upstream_repo_slug=repo)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Batch index all repositories in a directory. 路径默认读 REPO_DIR / INDEX_STORE_DIR / "
            "INTENT2FIX_BUILD_DIR；其余 BATCH_INDEX_* 见仓库根 .env。"
            "若 BATCH_INDEX_COMPILE_DB_ONLY=1，每个实例会先容器录 compile_commands 再自动宿主机 --host-index。"
        )
    )
    parser.add_argument(
        "--repo-dir",
        default=BASE_REPO_DIR,
        help="实例仓库父目录；默认读 REPO_DIR",
    )
    parser.add_argument(
        "--output-dir",
        default=BASE_OUTPUT_DIR,
        help="索引输出根；默认读 INDEX_STORE_DIR",
    )
    parser.add_argument(
        "--build-dir",
        default=BASE_BUILD_DIR,
        help="各实例 build 根；默认读 INTENT2FIX_BUILD_DIR",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=_env_int("BATCH_INDEX_LIMIT", 0),
        help="最多处理实例数，0 为全部；或 BATCH_INDEX_LIMIT",
    )
    _add_boolean_optional_argument(
        parser,
        "skip-existing",
        default=_env_bool("BATCH_INDEX_SKIP_EXISTING"),
        help_text="跳过已有完整索引；或 BATCH_INDEX_SKIP_EXISTING=1",
    )
    _add_boolean_optional_argument(
        parser,
        "resume",
        default=_env_bool("BATCH_INDEX_RESUME"),
        help_text="断点续跑：跳过 summary 及 --resume-log-dir 中曾 [SUCCESS] 的实例；或 BATCH_INDEX_RESUME=1",
    )
    _add_boolean_optional_argument(
        parser,
        "retry-failed",
        default=_env_bool("BATCH_INDEX_RETRY_FAILED"),
        help_text=(
            "只重跑 summary 日志里曾出现 [FAILURE] 的实例（与当前 --jsonl 或目录扫描的交集）；"
            "或 BATCH_INDEX_RETRY_FAILED=1。可与 --resume 同用。"
        ),
    )
    parser.add_argument(
        "--resume-log-dir",
        default=_env_str("BATCH_INDEX_RESUME_LOG_DIR"),
        metavar="DIR",
        help=(
            "与 --resume 合用：合并该目录下所有 .log 中的 [SUCCESS] 后一并跳过（主 summary 被覆盖时，把历次日志复制到此）；"
            "或 BATCH_INDEX_RESUME_LOG_DIR"
        ),
    )
    _add_boolean_optional_argument(
        parser,
        "resume-log-recursive",
        default=_env_bool("BATCH_INDEX_RESUME_LOG_RECURSIVE"),
        help_text="与 --resume-log-dir 合用：递归扫描 **/*.log；或 BATCH_INDEX_RESUME_LOG_RECURSIVE=1",
    )
    parser.add_argument(
        "--retry-failed-log-dir",
        default=_env_str("BATCH_INDEX_RETRY_FAILED_LOG_DIR"),
        metavar="DIR",
        help=(
            "与 --retry-failed 合用：合并该目录下 .log 中的 [FAILURE]/[ERROR]；或 BATCH_INDEX_RETRY_FAILED_LOG_DIR"
        ),
    )
    _add_boolean_optional_argument(
        parser,
        "retry-failed-log-recursive",
        default=_env_bool("BATCH_INDEX_RETRY_FAILED_LOG_RECURSIVE"),
        help_text="与 --retry-failed-log-dir 合用：递归 **/*.log；或 BATCH_INDEX_RETRY_FAILED_LOG_RECURSIVE=1",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        help="额外排除的 instance id（仍可与 BATCH_INDEX_EXCLUDE 合并）",
    )
    parser.add_argument(
        "--parallel-instances",
        type=int,
        default=_parallel_instances_default(),
        metavar="N",
        help=(
            "同时索引的实例数（线程池；每实例独立 docker）。1=串行。"
            "或 BATCH_INDEX_PARALLEL_INSTANCES；未设时兼容 BATCH_INDEX_MAX_WORKERS。"
        ),
    )
    parser.add_argument(
        "--jsonl",
        default=_env_str("BATCH_INDEX_JSONL"),
        metavar="PATH",
        help="数据集 JSONL（每行 instance_id, repo, base_commit）；与 --repo-dir 组合：每实例目录为 repo-dir/instance_id",
    )
    parser.add_argument(
        "--git-base-url",
        default=_env_str("BATCH_INDEX_GIT_BASE_URL", "https://github.com"),
        help="git clone 基址，最终 URL={基址}/{repo}.git；或 BATCH_INDEX_GIT_BASE_URL",
    )
    parser.add_argument(
        "--no-clone-missing",
        action="store_true",
        help="与 --jsonl 连用：不执行 clone/checkout，要求各实例目录已存在",
    )
    parser.add_argument(
        "--force-reclone",
        action="store_true",
        help="与 --jsonl 连用：删除已有 instance 目录后重新 clone；或 BATCH_INDEX_FORCE_RECLONE=1",
    )
    parser.add_argument(
        "--upstream-cache",
        default=_env_str("BATCH_INDEX_UPSTREAM_CACHE"),
        metavar="DIR",
        help="JSONL：按 repo 去重后全量克隆到此，再为每个实例从本地 clone+checkout；默认 <repo-dir>/_upstream；或 BATCH_INDEX_UPSTREAM_CACHE",
    )
    _add_boolean_optional_argument(
        parser,
        "jsonl-preserve-order",
        default=_env_bool("BATCH_INDEX_JSONL_PRESERVE_ORDER"),
        help_text="按 JSONL 文件行顺序处理（与 --limit 1 合用可只跑首条）；默认按 repo/PR 排序；或 BATCH_INDEX_JSONL_PRESERVE_ORDER=1",
    )
    parser.add_argument(
        "--host-index-only",
        action="store_true",
        help=(
            "仅执行阶段 2（单实例）：要求 <build-dir>/<instance-id>/compile_commands.json 已存在。"
            "批量+实时统计请用 --host-index-from-summary-log 或 --host-index-batch-file。"
        ),
    )
    parser.add_argument(
        "--instance-id",
        default=_env_str("BATCH_INDEX_INSTANCE_ID"),
        metavar="ID",
        help="与 --host-index-only 合用；或环境变量 BATCH_INDEX_INSTANCE_ID",
    )
    parser.add_argument(
        "--host-index-from-summary-log",
        type=Path,
        default=None,
        metavar="PATH",
        help="批量阶段 2：从 batch 摘要里「Successful Instances:」下列出的 id 逐条宿主机索引（实时统计成功/失败）",
    )
    parser.add_argument(
        "--host-index-batch-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="批量阶段 2：每行一个 instance_id（# 注释）；可与 --host-index-from-summary-log 合用，去重后顺序执行",
    )

    args = parser.parse_args()

    batch_ids: List[str] = []
    if args.host_index_from_summary_log is not None:
        p = args.host_index_from_summary_log.expanduser().resolve()
        if not p.is_file():
            print(f"❌ --host-index-from-summary-log 不是有效文件: {p}")
            return
        batch_ids.extend(load_instance_ids_from_summary_log(p))
    if args.host_index_batch_file is not None:
        p = args.host_index_batch_file.expanduser().resolve()
        if not p.is_file():
            print(f"❌ --host-index-batch-file 不是有效文件: {p}")
            return
        batch_ids.extend(load_instance_ids_from_batch_file(p))

    wants_host_batch = (
        args.host_index_from_summary_log is not None or args.host_index_batch_file is not None
    )
    if wants_host_batch:
        if not batch_ids:
            print("❌ [HostBatch] 未从摘要日志/批处理文件解析到任何 instance_id")
            return
        repo_base = Path(args.repo_dir).expanduser().resolve()
        output_base = Path(args.output_dir).expanduser().resolve()
        build_base = Path(args.build_dir).expanduser().resolve()
        run_host_index_batch(batch_ids, repo_base, output_base, build_base)
        return

    if args.host_index_only:
        iid = (args.instance_id or "").strip()
        if not iid:
            print("❌ --host-index-only 需要 --instance-id（或 BATCH_INDEX_INSTANCE_ID）")
            return
        repo_base = Path(args.repo_dir).expanduser().resolve()
        output_base = Path(args.output_dir).expanduser().resolve()
        build_base = Path(args.build_dir).expanduser().resolve()
        repo_path = str(repo_base / iid)
        if not Path(repo_path).is_dir():
            print(f"❌ 仓库目录不存在: {repo_path}")
            return
        repo_cfg = get_repo_config(iid)
        build_dir = repo_cfg["build_dir"] or str(build_base / iid)
        cc = Path(build_dir) / "compile_commands.json"
        if not cc.is_file():
            print(f"❌ 未找到 {cc}，请先完成阶段 1（容器内生成 compile_commands.json）")
            return
        docker_repo = _docker_repo_path(iid)
        print(f"🖥️ [Batch] --host-index-only：{iid}（build_dir={build_dir}）")
        _run_batch_host_index_phase(
            iid, repo_path, str(output_base), build_dir, docker_repo, repo_cfg
        )
        print(f"✅ [Batch] --host-index-only 完成: {iid}")
        return

    extra_exclude = (os.environ.get("BATCH_INDEX_EXCLUDE") or "").strip()
    if extra_exclude:
        merged = list(args.exclude)
        merged.extend(x.strip() for x in extra_exclude.split(",") if x.strip())
        args.exclude = merged

    force_reclone = bool(args.force_reclone or _env_bool("BATCH_INDEX_FORCE_RECLONE"))

    jsonl_raw = (args.jsonl or "").strip()
    jsonl_path: Path | None = None
    if jsonl_raw:
        jsonl_path = Path(jsonl_raw).expanduser()
        if not jsonl_path.is_file():
            print(f"❌ --jsonl 不是有效文件: {jsonl_path}")
            return
        jsonl_path = jsonl_path.resolve()

    repo_base = Path(args.repo_dir).expanduser()
    output_base = Path(args.output_dir).expanduser()
    build_base = Path(args.build_dir).expanduser()

    output_base.mkdir(parents=True, exist_ok=True)
    build_base.mkdir(parents=True, exist_ok=True)

    excluded_set = set(EXCLUDED_INSTANCES + args.exclude)
    work_plan: List[Tuple[str, Path]] = []
    excluded_count = 0

    if jsonl_path:
        repo_base.mkdir(parents=True, exist_ok=True)
        clone_missing = not args.no_clone_missing
        try:
            all_rows = load_instances_from_jsonl(jsonl_path)
        except Exception as exc:
            print(f"❌ 读取 jsonl 失败: {exc}")
            return
        excluded_count = sum(1 for r in all_rows if r["instance_id"] in excluded_set)
        rows = [r for r in all_rows if r["instance_id"] not in excluded_set]
        if not args.jsonl_preserve_order:
            rows.sort(
                key=lambda r: (
                    str(r.get("repo", "")),
                    get_pr_number(str(r.get("instance_id", ""))),
                )
            )
        uc_raw = (args.upstream_cache or "").strip()
        upstream_root = (
            Path(uc_raw).expanduser().resolve() if uc_raw else (repo_base / "_upstream")
        )
        if clone_missing:
            unique_repos = _unique_repos_preserve_order(rows)
            force_upstream = _env_bool("BATCH_INDEX_FORCE_UPSTREAM")
            print(
                f"📦 [Upstream] 阶段 1/2：全量克隆 {len(unique_repos)} 个上游仓库 → {upstream_root}"
            )
            upstream_root.mkdir(parents=True, exist_ok=True)
            try:
                for rp in unique_repos:
                    ensure_upstream_mirror(
                        upstream_root,
                        rp,
                        git_base_url=args.git_base_url,
                        force_upstream=force_upstream,
                    )
            except Exception as exc:
                print(f"❌ [Upstream] 失败: {exc}")
                return

        for row in rows:
            iid = str(row["instance_id"])
            dest = (repo_base / iid).resolve()
            if clone_missing:
                try:
                    up_mirror = upstream_root / _sanitize_upstream_dirname(str(row["repo"]))
                    materialize_instance_clone_from_upstream(
                        dest,
                        up_mirror,
                        str(row["base_commit"]),
                        force_reclone=force_reclone,
                        upstream_repo_slug=str(row["repo"]),
                    )
                except Exception as exc:
                    print(f"❌ [Clone] {iid}: {exc}")
                    continue
            else:
                if not dest.is_dir() or not (dest / ".git").exists():
                    print(f"⚠️ [Batch] 跳过 {iid}：{dest} 不存在且使用了 --no-clone-missing")
                    continue
            work_plan.append((iid, dest))
        print(
            f"📄 [Batch] --jsonl {jsonl_path}：数据 {len(all_rows)} 条，排除 {excluded_count}，"
            f"clone_missing={clone_missing}，进入索引队列 {len(work_plan)}"
        )
    else:
        if not repo_base.exists():
            print(f"❌ Base repo directory does not exist: {repo_base}")
            return
        instances = [d for d in repo_base.iterdir() if d.is_dir() and is_valid_instance(d.name)]
        instances.sort(key=lambda d: (d.name.rsplit("-", 1)[0], get_pr_number(d.name)))
        original_count = len(instances)
        instances = [d for d in instances if d.name not in excluded_set]
        excluded_count = original_count - len(instances)
        work_plan = [(d.name, d.resolve()) for d in instances]
        print(f"🔍 Found {len(work_plan)} valid instances in {repo_base} (swe-bench 目录扫描)")

    if excluded_count > 0:
        print(f"🚫 [Batch] Excluded {excluded_count} instance(s) from processing list")
        if excluded_set:
            print(f"   Excluded instances: {', '.join(sorted(excluded_set))}")

    summary_path = Path(BATCH_SUMMARY_LOG)
    completed_from_log: set[str] = set()
    failed_from_log: set[str] = set()

    resume_log_dir_raw = (args.resume_log_dir or "").strip()
    resume_log_dir = (
        Path(resume_log_dir_raw).expanduser().resolve() if resume_log_dir_raw else None
    )
    retry_failed_log_dir_raw = (args.retry_failed_log_dir or "").strip()
    retry_failed_log_dir = (
        Path(retry_failed_log_dir_raw).expanduser().resolve() if retry_failed_log_dir_raw else None
    )

    if args.resume:
        completed_from_log = load_completed_instances_from_log(summary_path)
        n_single = len(completed_from_log)
        if resume_log_dir and resume_log_dir.is_dir():
            files = _iter_log_files_in_dir(resume_log_dir, recursive=args.resume_log_recursive)
            from_hist = load_completed_instances_from_log_dir(
                resume_log_dir, recursive=args.resume_log_recursive
            )
            before = len(completed_from_log)
            completed_from_log |= from_hist
            added = len(completed_from_log) - before
            print(
                f"📋 [Batch] --resume：主日志 {summary_path} 中 [SUCCESS] {n_single} 个；"
                f"--resume-log-dir {resume_log_dir}（{len(files)} 个 .log）合并新增 {added} 个；"
                f"合计跳过 {len(completed_from_log)} 个已成功实例"
            )
        elif resume_log_dir_raw:
            print(f"⚠️ [Batch] --resume-log-dir 不是目录: {resume_log_dir}")
            if completed_from_log:
                print(f"📋 [Batch] 仅从主日志读取 {len(completed_from_log)} 个 [SUCCESS]")
        elif completed_from_log:
            print(f"📋 [Batch] --resume：从 {summary_path} 读取 {len(completed_from_log)} 个 [SUCCESS]，将跳过")
    elif resume_log_dir_raw:
        print(
            "⚠️ [Batch] 已指定 --resume-log-dir / BATCH_INDEX_RESUME_LOG_DIR，但未开启 --resume；"
            "历史 [SUCCESS] 不会用于跳过。请加上 --resume 或 BATCH_INDEX_RESUME=1"
        )

    if args.retry_failed:
        failed_from_log = load_failed_instances_from_log(summary_path)
        n_single_f = len(failed_from_log)
        if retry_failed_log_dir and retry_failed_log_dir.is_dir():
            files = _iter_log_files_in_dir(
                retry_failed_log_dir, recursive=args.retry_failed_log_recursive
            )
            from_hist = load_failed_instances_from_log_dir(
                retry_failed_log_dir, recursive=args.retry_failed_log_recursive
            )
            before = len(failed_from_log)
            failed_from_log |= from_hist
            added = len(failed_from_log) - before
            print(
                f"📋 [Batch] --retry-failed：主日志中失败 {n_single_f} 个；"
                f"--retry-failed-log-dir（{len(files)} 个 .log）合并新增 {added} 个；"
                f"合计 {len(failed_from_log)} 个失败 id"
            )
        elif retry_failed_log_dir_raw:
            print(f"⚠️ [Batch] --retry-failed-log-dir 不是目录: {retry_failed_log_dir}")
        if not failed_from_log:
            print(
                f"⚠️ [Batch] --retry-failed：在 {summary_path} 及扩展目录中未解析到任何 [FAILURE] 行，退出。"
            )
            return
        print(f"📋 [Batch] --retry-failed：将只处理其中与任务列表交集的实例")

    if not work_plan:
        print("⚠️ [Batch] 没有待处理的实例，退出。")
        return

    success_list = []
    failure_list = []
    skipped_count = 0
    
    start_time = time.time()
    
    # 检查是否是断点续跑
    is_resume = args.resume and summary_path.exists()
    
    # Initialize log file with header
    if is_resume:
        # 断点续跑：追加到现有日志
        with summary_path.open("a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Resumed at: {time.ctime(start_time)}\n")
            f.write(f"Processing Log (continued):\n")
            f.write(f"{'='*60}\n")
        print(f"📋 [Batch] Resuming from previous run (appending to existing log)")
    else:
        # 新运行：创建新日志
        with summary_path.open("w", encoding="utf-8") as f:
            f.write(f"Batch Indexing Summary Report\n")
            f.write(f"============================\n")
            f.write(f"Start Time: {time.ctime(start_time)}\n")
            f.write(f"Total valid instances found: {len(work_plan)}\n")
            if excluded_count > 0:
                f.write(f"Excluded instances: {excluded_count}\n")
            f.write(f"\nProcessing Log:\n")
            f.write(f"{'='*60}\n")
    
    runs: List[Tuple[str, Path]] = []
    for instance_id, instance_path in work_plan:
        if instance_id in excluded_set:
            print(f"🚫 [Batch] Skipping {instance_id} (excluded).")
            skipped_count += 1
            _append_summary_line(summary_path, f"[EXCLUDED] {instance_id} - In exclusion list\n")
            continue
        if args.resume and _instance_in_log(instance_id, completed_from_log):
            print(f"⏭️ [Batch] Skipping {instance_id} (already completed in previous run).")
            skipped_count += 1
            _append_summary_line(
                summary_path, f"[SKIP] {instance_id} - Already completed in previous run\n"
            )
            continue
        if args.retry_failed and not _instance_in_log(instance_id, failed_from_log):
            continue
        instance_output_dir = output_base / instance_id / "index_output"
        if args.skip_existing:
            if is_index_complete(instance_output_dir):
                print(f"⏭️ [Batch] Skipping {instance_id} (index already complete).")
                skipped_count += 1
                _append_summary_line(summary_path, f"[SKIP] {instance_id} - Index already complete\n")
                continue
            if instance_output_dir.exists() and any(instance_output_dir.iterdir()):
                print(f"⏭️ [Batch] Skipping {instance_id} (output directory exists).")
                skipped_count += 1
                _append_summary_line(summary_path, f"[SKIP] {instance_id} - Output directory exists\n")
                continue
        runs.append((instance_id, instance_path))
        if args.limit > 0 and len(runs) >= args.limit:
            break

    if args.retry_failed and not runs:
        dir_ids = {iid for iid, _ in work_plan}
        inter = sorted(dir_ids & failed_from_log)
        print(
            "⚠️ [Batch] --retry-failed：队列为空（当前 --jsonl/目录 与日志中的失败 id 无交集，"
            "或失败实例均被 --resume / --skip-existing 跳过）。"
        )
        if inter:
            print(
                f"ℹ️ [Batch] 诊断：日志失败 id 与当前任务列表交集有 {len(inter)} 个 "
                f"（示例 {inter[:5]}{'…' if len(inter) > 5 else ''}），"
                "但均未进入索引队列；可尝试关闭续跑/跳过："
                "`--no-resume`、`--no-skip-existing` 或 `BATCH_INDEX_SKIP_EXISTING=0`。"
            )
        elif work_plan and failed_from_log:
            samp_d = sorted(dir_ids)[:8]
            samp_f = sorted(failed_from_log)[:8]
            print(
                f"ℹ️ [Batch] 诊断：当前目录下实例 id 共 {len(dir_ids)} 个，"
                f"summary 中失败 id 共 {len(failed_from_log)} 个，二者交集为空。"
            )
            print(f"   当前 --repo-dir 下实例样例：{samp_d}")
            print(f"   日志中失败 id 样例：{samp_f}")
            print(
                "   请改用与当时批跑相同的 --jsonl（推荐），"
                "或将 --repo-dir 指到包含这些失败 id 子目录的父目录。"
            )
        return

    processed_count = len(runs)
    parallel = max(1, int(args.parallel_instances))
    print(
        f"⚙️ [Batch] 将索引 {processed_count} 个实例；实例级并行={parallel} "
        f"（每实例独立 docker，过大易占满 CPU/磁盘请酌情调小）"
    )
    if args.limit > 0 and len(runs) >= args.limit:
        print(f"🛑 [Batch] 已达 --limit {args.limit}（仅统计实际进入索引队列的实例）。")
        _append_summary_line(summary_path, f"\n[INFO] Reached limit of {args.limit} instances.\n")

    out_s = str(output_base.resolve())
    build_s = str(build_base.resolve())

    def _run_one(instance_id: str, instance_path: Path) -> Tuple[str, str, float, str]:
        """返回 (status, instance_id, seconds, detail)；status 为 ok / fail / err。"""
        t0 = time.time()
        try:
            run_indexing(instance_id, str(instance_path.resolve()), out_s, build_s)
            return ("ok", instance_id, time.time() - t0, "")
        except subprocess.CalledProcessError as e:
            return ("fail", instance_id, time.time() - t0, f"Exit code {e.returncode}")
        except Exception as e:
            return ("err", instance_id, time.time() - t0, str(e))

    if parallel <= 1:
        for instance_id, instance_path in runs:
            st, iid, dur, detail = _run_one(instance_id, instance_path)
            if st == "ok":
                print(f"✅ [Batch] Successfully indexed: {iid} (took {dur:.1f}s)")
                success_list.append(iid)
                _append_summary_line(summary_path, f"[SUCCESS] {iid} - Completed in {dur:.1f}s\n")
            elif st == "fail":
                print(f"❌ [Batch] Failed to index {iid} ({detail}, took {dur:.1f}s)")
                failure_list.append((iid, detail))
                code = detail.replace("Exit code ", "").strip()
                _append_summary_line(
                    summary_path, f"[FAILURE] {iid} - Exit code {code} (took {dur:.1f}s)\n"
                )
            else:
                print(f"❌ [Batch] Unexpected error for {iid}: {detail} (took {dur:.1f}s)")
                failure_list.append((iid, detail))
                _append_summary_line(
                    summary_path, f"[FAILURE] {iid} - {detail} (took {dur:.1f}s)\n"
                )
    else:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            future_map = {
                pool.submit(_run_one, iid, pth): (iid, pth) for iid, pth in runs
            }
            for fut in as_completed(future_map):
                st, iid, dur, detail = fut.result()
                if st == "ok":
                    print(f"✅ [Batch] Successfully indexed: {iid} (took {dur:.1f}s)")
                    success_list.append(iid)
                    _append_summary_line(summary_path, f"[SUCCESS] {iid} - Completed in {dur:.1f}s\n")
                elif st == "fail":
                    print(f"❌ [Batch] Failed to index {iid} ({detail}, took {dur:.1f}s)")
                    failure_list.append((iid, detail))
                    code = detail.replace("Exit code ", "").strip()
                    _append_summary_line(
                        summary_path, f"[FAILURE] {iid} - Exit code {code} (took {dur:.1f}s)\n"
                    )
                else:
                    print(f"❌ [Batch] Unexpected error for {iid}: {detail} (took {dur:.1f}s)")
                    failure_list.append((iid, detail))
                    _append_summary_line(
                        summary_path, f"[FAILURE] {iid} - {detail} (took {dur:.1f}s)\n"
                    )

    duration = time.time() - start_time
    
    # Append final summary to log
    with summary_path.open("a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Final Summary:\n")
        f.write(f"Duration: {duration:.2f} seconds ({duration/60:.1f} minutes)\n")
        f.write(f"Instances skipped: {skipped_count}\n")
        f.write(f"Instances processed: {processed_count}\n")
        f.write(f"Successes: {len(success_list)}\n")
        f.write(f"Failures: {len(failure_list)}\n\n")
        
        if failure_list:
            f.write(f"Failed Instances:\n")
            for inst_id, err in failure_list:
                f.write(f" - {inst_id}: {err}\n")
            f.write("\n")
            
        if success_list:
            f.write(f"Successful Instances:\n")
            for inst_id in success_list:
                f.write(f" - {inst_id}\n")
        
        f.write(f"\nEnd Time: {time.ctime(time.time())}\n")

    print(f"\n🏁 [Batch] Completed batch indexing.")
    print(f"📊 Summary: {len(success_list)} success, {len(failure_list)} failure, {skipped_count} skipped.")
    print(f"⏱️  Total duration: {duration:.1f}s ({duration/60:.1f} minutes)")
    print(f"📝 Detailed report written to: {BATCH_SUMMARY_LOG}")
    if failure_list:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
