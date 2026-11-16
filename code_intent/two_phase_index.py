#!/usr/bin/env python3
"""
两阶段索引流水线：容器内仅生成 compile_commands.json，宿主机执行 libclang 索引。

默认 **不做 chown**：CDB 多为 root 拥有但 **644**，宿主只读即可跑阶段 2；``index_store`` 由阶段 2 创建，无 root 目录问题。

若环境把构建目录设成不可读，可显式加 ``--chown-build-dir``：仅尝试 ``chown`` / ``sudo -n chown``（**从不**交互式要密码）；失败则打印可手抄命令并退出。

用法（在仓库根 INTENT2FIX 下）::

    python3 code_intent/two_phase_index.py --instance-id jqlang__jq-2157

路径默认读 .env / 环境变量：REPO_DIR、INDEX_STORE_DIR、INTENT2FIX_BUILD_DIR。
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

# 与 batch_index 一致：仓库根、加载 .env
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_CODE_INTENT_DIR = Path(__file__).resolve().parent
if str(_CODE_INTENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_INTENT_DIR))

import batch_index  # noqa: E402

from INTENT2FIX.env_paths import (  # noqa: E402
    build_instances_parent,
    index_store_parent,
    repo_instances_parent,
)

DEMO_PY = _CODE_INTENT_DIR / "demo.py"


def _maybe_git_reset(repo: Path) -> None:
    if not (repo / ".git").is_dir():
        return
    print(f"🧹 [Pipeline] git reset --hard + clean -fd: {repo}")
    try:
        subprocess.check_call(["git", "reset", "--hard", "HEAD"], cwd=str(repo))
        subprocess.check_call(["git", "clean", "-fd"], cwd=str(repo))
    except subprocess.CalledProcessError as e:
        print(f"⚠️ [Pipeline] git reset 失败，继续: {e}")


def _chown_build_dir_noninteractive(paths: list[Path]) -> None:
    """仅非交互：直接 chown，不成则 ``sudo -n``。不做交互式 sudo。"""
    uid, gid = os.getuid(), os.getgid()
    targets = [str(p.resolve()) for p in paths if p.exists()]
    if not targets:
        return
    cmd = ["chown", "-R", f"{uid}:{gid}", *targets]
    manual = "sudo " + " ".join(shlex.quote(x) for x in cmd)

    r0 = subprocess.run(cmd, capture_output=True, text=True)
    if r0.returncode == 0:
        print("✅ [Pipeline] 已修正构建目录属主（直接 chown）")
        return
    if (r0.stderr or r0.stdout or "").strip():
        print(f"ℹ️ [Pipeline] 直接 chown 未成功: {(r0.stderr or r0.stdout)[:400].strip()}")

    r1 = subprocess.run(["sudo", "-n", *cmd], capture_output=True, text=True)
    if r1.returncode == 0:
        print("✅ [Pipeline] 已修正构建目录属主（sudo -n）")
        return

    print(
        "❌ [Pipeline] 已指定 --chown-build-dir，但无法在免交互下修正属主。\n"
        f"   请手动执行: {manual}\n"
        "   或配置 NOPASSWD 后重试 ``sudo -n chown …``。"
    )
    raise SystemExit(1)


def _demo_cmd() -> list[str]:
    return [sys.executable, str(DEMO_PY)]


def run_pipeline(args: argparse.Namespace) -> int:
    instance_id = args.instance_id.strip()
    if not instance_id:
        print("❌ [Pipeline] --instance-id 不能为空")
        return 2

    repo_path = Path(args.repo_path.strip() or (repo_instances_parent() / instance_id)).expanduser().resolve()
    output_base = Path(args.output_path.strip() or index_store_parent()).expanduser().resolve()
    cfg = batch_index.get_repo_config(instance_id)
    if args.build_dir.strip():
        build_dir = Path(args.build_dir.strip()).expanduser().resolve()
    else:
        build_base = Path(args.build_base.strip() or build_instances_parent()).expanduser().resolve()
        build_dir = Path(cfg["build_dir"] or (build_base / instance_id)).expanduser().resolve()

    instance_index_root = output_base / instance_id
    docker_src = batch_index._docker_repo_path(instance_id)
    path_map = f"{docker_src}|{repo_path}"

    if not repo_path.is_dir():
        print(f"❌ [Pipeline] 仓库目录不存在: {repo_path}")
        return 2

    if args.git_reset:
        _maybe_git_reset(repo_path)
    env = os.environ.copy()
    env["CODE_INTENT_HOST_REPO_PATH_FOR_META"] = str(repo_path)

    if not args.phase2_only:
        build_dir.mkdir(parents=True, exist_ok=True)

        cmd1 = _demo_cmd() + [
            "--compile-db-only",
            "--instance-id",
            instance_id,
            "--file-path",
            docker_src,
            "--output-path",
            str(output_base),
            "--build-dir",
            str(build_dir),
            "--host-repo-bind",
            str(repo_path),
        ]
        for a in cfg.get("cmake_args") or []:
            cmd1.append(f"--cmake-arg={a}")
        if cfg.get("build_backend"):
            cmd1.extend(["--build-backend", cfg["build_backend"]])
        if cfg.get("make_build_shell"):
            cmd1.extend(["--make-build-shell", cfg["make_build_shell"]])

        print("━━━ [Pipeline] 阶段 1/2：容器生成 compile_commands.json ━━━")
        print(f"📂 宿主仓库: {repo_path}")
        print(f"📂 容器路径: {docker_src}")
        print(f"🏗️  构建目录: {build_dir}")
        try:
            subprocess.check_call(cmd1, cwd=str(_REPO_ROOT), env=env)
        except subprocess.CalledProcessError as e:
            print(f"❌ [Pipeline] 阶段 1 失败: exit {e.returncode}")
            return e.returncode

        if args.chown_build_dir:
            print("━━━ [Pipeline] --chown-build-dir：修正构建目录属主（仅非交互）━━━")
            _chown_build_dir_noninteractive([build_dir])

    if args.phase1_only:
        print("✅ [Pipeline] 仅阶段 1，结束。")
        return 0

    cc = build_dir / "compile_commands.json"
    if not cc.is_file():
        print(f"❌ [Pipeline] 未找到 {cc}，无法执行阶段 2（请先跑阶段 1）")
        return 2

    if args.phase2_only and args.chown_build_dir:
        print("━━━ [Pipeline] --chown-build-dir：阶段 2 前修正构建目录 ━━━")
        _chown_build_dir_noninteractive([build_dir])

    cmd2 = _demo_cmd() + [
        "--host-index",
        "--instance-id",
        instance_id,
        "--file-path",
        str(repo_path),
        "--output-path",
        str(output_base),
        "--build-dir",
        str(build_dir),
        "--path-map",
        path_map,
    ]
    for a in cfg.get("cmake_args") or []:
        cmd2.append(f"--cmake-arg={a}")

    print("━━━ [Pipeline] 阶段 2/2：宿主 index_repo ━━━")
    print(f"🗺️  path-map: {path_map!r}")
    try:
        subprocess.check_call(cmd2, cwd=str(_REPO_ROOT), env=env)
    except subprocess.CalledProcessError as e:
        print(f"❌ [Pipeline] 阶段 2 失败: exit {e.returncode}")
        return e.returncode

    print(f"✅ [Pipeline] 完成: {instance_index_root / 'index_output'}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="两阶段索引：Docker compile-db-only + 宿主 --host-index（默认不 chown）")
    p.add_argument("--instance-id", required=True, help="SweBench instance_id，如 jqlang__jq-2157")
    p.add_argument(
        "--repo-path",
        default="",
        help="宿主 clone 根目录；默认 REPO_DIR/<instance-id>",
    )
    p.add_argument("--output-path", default="", help="索引根；默认 INDEX_STORE_DIR")
    p.add_argument(
        "--build-base",
        default="",
        help="构建目录父路径；默认 INTENT2FIX_BUILD_DIR；与 batch_index 一致时拼 <instance-id>",
    )
    p.add_argument(
        "--build-dir",
        default="",
        help="完整构建目录（若指定则覆盖 --build-base + instance_id / get_repo_config）",
    )
    p.add_argument(
        "--git-reset",
        action="store_true",
        help="阶段 1 前对宿主仓库执行 git reset --hard 与 git clean -fd（与 batch_index 行为类似，慎用）",
    )
    p.add_argument(
        "--chown-build-dir",
        action="store_true",
        help="阶段 1 后（及 --phase2-only 时阶段 2 前）尝试 chown 构建目录；仅非交互（sudo -n），默认关闭",
    )
    p.add_argument("--phase1-only", action="store_true", help="仅阶段 1")
    p.add_argument("--phase2-only", action="store_true", help="仅阶段 2（需已有 compile_commands.json）")
    args = p.parse_args()
    if args.phase1_only and args.phase2_only:
        print("❌ [Pipeline] 不能同时指定 --phase1-only 与 --phase2-only")
        sys.exit(2)
    raise SystemExit(run_pipeline(args))


if __name__ == "__main__":
    main()
