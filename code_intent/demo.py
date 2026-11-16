from __future__ import annotations

# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, List, Tuple

from ponyc_build_from_run_sh import ponyc_make_build_shell_for_instance

# 仓库根加入 sys.path，容器内 /workspace 才能 ``import INTENT2FIX.*``（与 batch_index 一致）。
_REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORT))

# --- 配置区域 ---
DEFAULT_DOCKER_IMAGE = "mswebench/fmtlib_m_fmt:pr-3732"
CONTAINER_WORKSPACE = "/workspace"
# 宿主 Conda 环境挂载到容器内的固定前缀（与 CODE_INTENT_DOCKER_CONDA_PREFIX 联用）
CONTAINER_CONDA_ENV_PREFIX = "/opt/code_intent_conda"
# 仅约束「容器内跑 code_intent（demo / indexer）」；与仓库根 INTENT2FIX agent（pyproject 的 3.11+）无关
CODE_INTENT_CONTAINER_PYTHON_MIN = (3, 8)
# jqlang__jq 专用：make-bear 录制（builtin oniguruma）；batch_index 须与之字面一致。
JQLANG_JQ_MAKE_BUILD_SHELL = (
    "autoreconf -i && ./configure --with-oniguruma=builtin && make clean && make -j8"
)
# ----------------


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _configure_edge_debug_log(output_dir: Path) -> Path | None:
    """Enable the per-edge trace only when the caller explicitly requests it."""
    configured = (os.environ.get("EDGE_LOG_PATH") or "").strip()
    if configured:
        return Path(configured)
    if not _truthy_env("EDGE_DEBUG"):
        return None
    edge_log = output_dir / "edge_debug.log"
    if edge_log.exists():
        edge_log.unlink()
    os.environ["EDGE_LOG_PATH"] = str(edge_log.resolve())
    return edge_log


def _ensure_git_safe_directory(repo: Path) -> None:
    """宿主挂载仓库在容器内 UID 常与 .git 属主不一致，Git 2.35+ 报 safe.directory；Makefile 调 git 会失败。"""
    if not (repo / ".git").exists():
        return
    try:
        subprocess.run(
            ["git", "config", "--global", "--add", "safe.directory", str(repo.resolve())],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        pass


def _ensure_git_safe_directory_wildcard_in_docker() -> None:
    """
    子模块会在构建过程中克隆到仓库下新路径（如 ``/home/ponyc/lib/llvm/src``），属主与执行 ``git`` 的用户
    可能不一致，Git 2.35+ 报 ``dubious ownership``。构建前无法预知路径，容器内添加 ``safe.directory '*'``
    （Git ≥2.35.3）；过旧 git 忽略失败即可。
    """
    if os.environ.get("RUNNING_IN_DOCKER") != "true":
        return
    try:
        r = subprocess.run(
            ["git", "config", "--global", "--add", "safe.directory", "*"],
            check=False,
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            print("🔓 [Container] 已 ``git config --global --add safe.directory '*'``（避免子模块路径 dubious ownership）")
    except OSError:
        pass


def _ensure_git_proxy_in_container() -> None:
    """
    Git 子模块（如 ponyc 拉 llvm）不总尊重环境变量里的 ``http_proxy``；按 ``HTTPS_PROXY`` / ``https_proxy``
    等为 ``https://github.com``、``https://llvm.org`` 等设置 ``http.<url>.proxy``。

    依赖宿主启动 docker 前已 ``export`` 代理，且 ``launch_docker_from_host`` 已用 ``-e`` 传入（见 ``_docker_append_proxy_env``）。
    """
    p = (
        (os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or "").strip()
        or (os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy") or "").strip()
    )
    if not p:
        return
    # 与 git-config 文档一致：对以该前缀开头的 URL 使用代理
    keys = [
        "http.https://github.com.proxy",
        "http.https://github.com/.proxy",
        "http.https://llvm.org.proxy",
        "http.https://github.com/llvm/llvm-project.proxy",
    ]
    extra = (os.environ.get("CODE_INTENT_GIT_PROXY_URL_PREFIXES") or "").strip()
    if extra:
        for prefix in extra.split(","):
            prefix = prefix.strip()
            if prefix:
                keys.append(f"http.{prefix}.proxy")
    try:
        for k in keys:
            subprocess.run(
                ["git", "config", "--global", k, p],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except OSError:
        return
    print(f"🌐 [Container] 已为 git 子模块设置 HTTPS 代理（GitHub/llvm 等）: {p!r}")


def _bear_major_version() -> int | None:
    try:
        raw = subprocess.check_output(["bear", "--version"], text=True, stderr=subprocess.DEVNULL, timeout=8).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    m = re.search(r"(\d+)\.", raw)
    return int(m.group(1)) if m else None


def _docker_demo_argv_tail(
    args: argparse.Namespace,
    script_in_container: str,
    file_path_arg: str,
    output_path_arg: str,
    build_dir_arg: str,
) -> list[str]:
    tail = [
        script_in_container,
        "--instance-id",
        args.instance_id,
        "--file-path",
        file_path_arg,
        "--output-path",
        output_path_arg,
    ]
    if build_dir_arg:
        tail += ["--build-dir", build_dir_arg]
    for extra in args.cmake_arg:
        tail.append(f"--cmake-arg={extra}")
    for target in getattr(args, "build_target", []):
        tail.extend(["--build-target", target])
    tail.extend(["--build-backend", _effective_build_backend_choice(args)])
    mbs = (getattr(args, "make_build_shell", "") or "").strip()
    if not mbs:
        mbs = (os.environ.get("CODE_INTENT_MAKE_BUILD_SHELL") or "").strip()
    if mbs:
        tail += ["--make-build-shell", mbs]
    if getattr(args, "compile_db_only", False) or (
        os.environ.get("CODE_INTENT_COMPILE_DB_ONLY", "").strip().lower() in ("1", "true", "yes")
    ):
        tail.append("--compile-db-only")
    return tail


def _host_conda_env_dir() -> Path | None:
    """``CODE_INTENT_DOCKER_CONDA_PREFIX``：宿主 Conda 环境根目录（含 ``bin/python*``）。"""
    raw = (os.environ.get("CODE_INTENT_DOCKER_CONDA_PREFIX") or "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser().resolve()
    if not p.is_dir():
        print(f"❌ [Host] CODE_INTENT_DOCKER_CONDA_PREFIX 不是目录: {p}")
        sys.exit(1)
    return p


def _conda_python_container_path(host_env: Path, container_prefix: str) -> str:
    """容器内用于 ``exec`` 的解释器路径（与挂载前缀对应）。"""
    override = (os.environ.get("CODE_INTENT_DOCKER_CONDA_PYBIN") or "").strip()
    if override:
        if override.startswith("/"):
            return override
        return f"{container_prefix}/bin/{override.lstrip('./')}"
    for name in ("python3", "python"):
        hp = host_env / "bin" / name
        if hp.is_file() or hp.is_symlink():
            return f"{container_prefix}/bin/{name}"
    return f"{container_prefix}/bin/python3"


def _docker_inner_cmd_run_demo(
    docker_image: str,
    args: argparse.Namespace,
    script_in_container: str,
    file_path_arg: str,
    output_path_arg: str,
    build_dir_arg: str,
    forced_python: str | None = None,
) -> list[str]:
    """
    Build ``docker run ... IMAGE <CMD>`` argv tail.

    Some mswebench images have no ``python3`` on PATH, or ``python`` is Python 2 only.
    ``forced_python``（显式传入时）优先；否则 ``CODE_INTENT_DOCKER_PYTHON``；
    否则 ``bash -c`` 在容器内查找满足 ``CODE_INTENT_CONTAINER_PYTHON_MIN`` 的 Python（code_intent 与 INTENT2FIX agent 版本要求分离）。

    ``ponylang__ponyc`` 需在宿主设置 ``CODE_INTENT_DOCKER_CONDA_PREFIX``（挂载至 ``CONTAINER_CONDA_ENV_PREFIX``）
    或 ``CODE_INTENT_DOCKER_PYTHON``（容器内解释器路径）；由 ``launch_docker_from_host`` 传入 ``forced_python``。
    """
    tail = _docker_demo_argv_tail(
        args, script_in_container, file_path_arg, output_path_arg, build_dir_arg
    )
    forced = (forced_python or "").strip() or (os.environ.get("CODE_INTENT_DOCKER_PYTHON") or "").strip()
    if forced:
        # 不用 ``docker ... IMAGE <py> arg...`` 直接 exec：部分环境下对 bind mount 内符号链接的
        # 解析会报 ``stat ... no such file``；经 ``/bin/bash -c 'cd /workspace && exec ...'`` 更稳。
        print(f"🐍 [Host] container Python (forced): {forced!r}")
        ws = shlex.quote(CONTAINER_WORKSPACE)
        cmd = " ".join([shlex.quote(forced)] + [shlex.quote(x) for x in tail])
        inner_script = f"set -e; cd {ws}; exec {cmd}"
        return [docker_image, "/bin/bash", "-c", inner_script]

    q_tail = " ".join(shlex.quote(x) for x in tail)
    ws = shlex.quote(CONTAINER_WORKSPACE)
    py_maj, py_minv = CODE_INTENT_CONTAINER_PYTHON_MIN
    ver_ge = f"sys.version_info>=({py_maj},{py_minv})"
    need_msg = f"code_intent: need Python >={py_maj}.{py_minv} in container"
    script = (
        f"set -e; cd {ws}; "
        "PY=; "
        "for c in python3 python3.11 python3.10 python3.9 python3.8; do "
        f'  if command -v "$c" >/dev/null 2>&1 && "$c" -c "import sys; sys.exit(0 if {ver_ge} else 1)"; then '
        '    PY=$c; break; '
        "  fi; "
        "done; "
        'if [ -z "${PY:-}" ]; then '
        "  for c in /usr/bin/python3 /usr/local/bin/python3; do "
        f'    if [ -x "$c" ] && "$c" -c "import sys; sys.exit(0 if {ver_ge} else 1)"; then '
        '      PY=$c; break; '
        "    fi; "
        "  done; "
        "fi; "
        f'if [ -z "${{PY:-}}" ]; then echo {shlex.quote(need_msg)} >&2; exit 127; fi; '
        f'exec "$PY" {q_tail}'
    )
    print(
        f"🐍 [Host] container Python: auto (bash finds python3.x >= {py_maj}.{py_minv} for code_intent); "
        "override with CODE_INTENT_DOCKER_PYTHON or mount Conda via CODE_INTENT_DOCKER_CONDA_PREFIX"
    )
    return [docker_image, "/bin/bash", "-c", script]


def _docker_append_proxy_env(cmd: list[str]) -> None:
    """把当前 shell 里的代理变量显式传给容器（配合 --network=host 使用 127.0.0.1:12312）。"""
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "all_proxy",
    ):
        v = os.environ.get(name)
        if v:
            cmd.extend(["-e", f"{name}={v}"])


# apt 走系统代理时常见 502/证书异常；pip 等仍可用代理。设 CODE_INTENT_APT_USE_PROXY=1 则 apt 也继承代理。
_APT_SUBPROC_STRIP_PROXY_KEYS = frozenset(
    {
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "FTP_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "ftp_proxy",
        "no_proxy",
    }
)


def _env_for_apt_subprocess() -> dict[str, str]:
    if os.environ.get("CODE_INTENT_APT_USE_PROXY", "").strip() == "1":
        return dict(os.environ)
    return {k: v for k, v in os.environ.items() if k not in _APT_SUBPROC_STRIP_PROXY_KEYS}


# 容器内 apt 换国内源（仅显式开启时执行）。宿主机需在启动 docker 前导出，或由 batch_index 加载 .env 后传入。
_APT_UBUNTU_CN_ALIASES: dict[str, str] = {
    "aliyun": "http://mirrors.aliyun.com/ubuntu/",
    "ali": "http://mirrors.aliyun.com/ubuntu/",
    "tsinghua": "https://mirrors.tuna.tsinghua.edu.cn/ubuntu/",
    "tuna": "https://mirrors.tuna.tsinghua.edu.cn/ubuntu/",
    "ustc": "https://mirrors.ustc.edu.cn/ubuntu/",
    "tencent": "http://mirrors.tencent.com/ubuntu/",
    "163": "http://mirrors.163.com/ubuntu/",
    "sjtu": "https://mirror.sjtu.edu.cn/ubuntu/",
}


def _read_os_release() -> dict[str, str]:
    p = Path("/etc/os-release")
    if not p.is_file():
        return {}
    out: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip('"')
    return out


def _resolve_ubuntu_apt_mirror_base(spec: str) -> str | None:
    s = spec.strip()
    if not s:
        return None
    key = s.lower()
    if key in _APT_UBUNTU_CN_ALIASES:
        return _APT_UBUNTU_CN_ALIASES[key]
    if s.startswith("http://") or s.startswith("https://"):
        return s.rstrip("/") + "/"
    print(
        f"⚠️ [Container] CODE_INTENT_APT_MIRROR={spec!r} 未识别；"
        f"可用别名 {', '.join(sorted(_APT_UBUNTU_CN_ALIASES))} 或完整 URL（须含 …/ubuntu/）"
    )
    return None


def _ubuntu_sources_list_for_mirror(base: str, codename: str) -> str:
    """与官方 Ubuntu sources 等价的四条（main / security / updates / backports）。"""
    b = base.rstrip("/") + "/"
    lines = [
        f"deb {b} {codename} main restricted universe multiverse",
        f"deb {b} {codename}-security main restricted universe multiverse",
        f"deb {b} {codename}-updates main restricted universe multiverse",
        f"deb {b} {codename}-backports main restricted universe multiverse",
    ]
    return "\n".join(lines) + "\n"


def _backup_and_quiesce_official_ubuntu_apt_sources() -> None:
    """避免 list.d 里官方源与新的 sources.list 混用导致仍走 archive.ubuntu.com。"""
    bak_suffix = ".bak_code_intent"
    sl = Path("/etc/apt/sources.list")
    if sl.is_file():
        try:
            Path("/etc/apt/sources.list" + bak_suffix).write_bytes(sl.read_bytes())
        except OSError:
            pass
    d = Path("/etc/apt/sources.list.d")
    if not d.is_dir():
        return
    for p in sorted(d.iterdir()):
        if p.suffix not in (".list", ".sources"):
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "archive.ubuntu.com" in txt or "security.ubuntu.com" in txt:
            try:
                p.rename(p.with_name(p.name + bak_suffix))
            except OSError:
                pass


def _maybe_configure_apt_cn_mirror() -> None:
    """若设置 CODE_INTENT_APT_MIRROR，则在容器内重写 Ubuntu apt 源（需 root）。"""
    spec = os.environ.get("CODE_INTENT_APT_MIRROR", "").strip()
    if not spec:
        return
    rel = _read_os_release()
    if rel.get("ID") != "ubuntu":
        print(
            f"⚠️ [Container] CODE_INTENT_APT_MIRROR 当前仅自动处理 Ubuntu（/etc/os-release ID="
            f"{rel.get('ID', '?')!r}），跳过换源"
        )
        return
    codename = (rel.get("VERSION_CODENAME") or rel.get("UBUNTU_CODENAME") or "").strip()
    if not codename:
        print("⚠️ [Container] 无法读取 Ubuntu VERSION_CODENAME，跳过换源")
        return
    base = _resolve_ubuntu_apt_mirror_base(spec)
    if not base:
        return
    try:
        _backup_and_quiesce_official_ubuntu_apt_sources()
        Path("/etc/apt/sources.list").write_text(
            _ubuntu_sources_list_for_mirror(base, codename),
            encoding="utf-8",
        )
    except OSError as e:
        print(f"⚠️ [Container] 写入 apt 源失败（非 root 或只读文件系统？）: {e}")
        return
    print(f"🌏 [Container] 已切换 Ubuntu apt 源 → {base}（codename={codename}）")


def _is_mswebench_style_home_source_path(p: str) -> bool:
    """
    ``batch_index`` 传入的容器内逻辑路径形如 ``/home/zstd``、``/home/ponyc``。
    若用 ``Path(...).exists()`` 在宿主机判断，宿主机上恰好存在同名目录时会误挂载并
    **覆盖镜像内** 已准备好的完整源码；此类路径不应走「宿主机存在则挂载」逻辑。
    """
    try:
        s = str(Path(p).expanduser())
    except Exception:
        s = p
    return bool(re.match(r"^/home/[^/]+$", s))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SweBench 实例索引构建脚本")
    parser.add_argument("--instance-id", required=True, help="SweBench instance_id（用于区分输出目录）")
    parser.add_argument("--file-path", required=True, help="实例源码所在路径（容器内可访问）")
    parser.add_argument("--output-path", required=True, help="实例的输出根目录")
    parser.add_argument(
        "--build-dir",
        default="",
        help="（可选）CMake 构建目录，默认使用源码目录",
    )
    parser.add_argument(
        "--cmake-arg",
        action="append",
        default=[],
        help="额外透传给 cmake 的参数，可重复使用，多参数用引号括起来",
    )
    parser.add_argument(
        "--build-target",
        action="append",
        default=[],
        help="仅构建指定的 CMake target，可重复使用",
    )
    parser.add_argument(
        "--docker-shell",
        action="store_true",
        help="只启动容器并进入 /bin/bash，方便调试（不会自动执行索引）",
    )
    parser.add_argument(
        "--host-repo-bind",
        default="",
        metavar="HOST_DIR",
        help=(
            "宿主机仓库目录，挂载到 --file-path 指定的容器路径（-v HOST:CONTAINER）。"
            "需要显式用本机 clone 覆盖镜像内源码时使用；不设则对 /home/<repo> 不自动挂载。"
            "也可用环境变量 CODE_INTENT_HOST_REPO_BIND。"
        ),
    )
    parser.add_argument(
        "--build-backend",
        default="auto",
        choices=("auto", "cmake", "make-bear"),
        help=(
            "如何生成 compile_commands.json：cmake、"
            "make-bear（bear 拦截 make/autotools）、"
            "auto（有 CMakeLists 用 cmake，否则有 Makefile 用 make-bear）。"
            "未传本参数时可用环境变量 CODE_INTENT_BUILD_BACKEND。"
        ),
    )
    parser.add_argument(
        "--make-build-shell",
        default="",
        help=(
            "make-bear 下在源码根目录执行的 shell 字符串（如 ponyc 默认仅 ``make`` 以录 compile_commands；"
            "需跑测试再加 ``make test`` 请用 CODE_INTENT_MAKE_BUILD_SHELL）。"
            "未传时可读环境变量 CODE_INTENT_MAKE_BUILD_SHELL。"
        ),
    )
    parser.add_argument(
        "--compile-db-only",
        action="store_true",
        help="仅在容器内：生成 compile_commands.json 后退出，索引在宿主机用 --host-index 执行",
    )
    parser.add_argument(
        "--host-index",
        action="store_true",
        help="仅在宿主机：对已有 compile_commands.json 运行 libclang 索引（需 --build-dir、--file-path 为宿主机路径）",
    )
    parser.add_argument(
        "--path-map",
        default="",
        metavar="OLD|NEW",
        help=(
            "与 --host-index 联用：将 compile_commands 内路径前缀 OLD 换为 NEW（容器源码根→宿主 clone 根），"
            "如 /home/ponyc|/data/... ；亦可设环境变量 CODE_INTENT_COMPILE_COMMANDS_PATH_MAP"
        ),
    )
    return parser.parse_args()


def _prepare_instance_dirs(output_path: str, instance_id: str) -> Tuple[Path, Path]:
    base = Path(output_path).expanduser()
    base.mkdir(parents=True, exist_ok=True)
    instance_root = base / instance_id
    instance_root.mkdir(parents=True, exist_ok=True)
    index_dir = instance_root / "index_output"
    index_dir.mkdir(parents=True, exist_ok=True)
    return instance_root, index_dir


def _write_instance_meta(instance_root: Path, payload: dict[str, Any]) -> None:
    meta_path = instance_root / "instance_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _docker_registry_prefix() -> str:
    """镜像仓库前缀；留空则不加。可用 CODE_INTENT_REGISTRY_PREFIX 覆盖默认。"""
    return os.environ.get(
        "CODE_INTENT_REGISTRY_PREFIX", "6pqg5gclf6yaz1ch6g.xuanyuan.run"
    ).strip().rstrip("/")


def _with_registry(image: str) -> str:
    p = _docker_registry_prefix()
    if not p:
        return image
    img = (image or "").strip()
    if not img:
        return img
    if img.startswith(f"{p}/"):
        return img
    return f"{p}/{img}"


def _resolve_docker_image(instance_id: str) -> str:
    override = os.environ.get("CODE_INTENT_IMAGE")
    if override:
        return _with_registry(override.strip())
    # swe-bench_catchorg__Catch2-2849 → mswebench/catchorg_m_catch2:pr-2849
    match = re.match(r"swe-bench_([^_]+)__([^-]+)-(\d+)", instance_id)
    if match:
        org, repo, pr_id = match.groups()
        return _with_registry(f"mswebench/{org.lower()}_m_{repo.lower()}:pr-{pr_id}")
    # multi_swebench 等：jqlang__jq-3238 → mswebench/jqlang_m_jq:pr-3238（与 mswebench 命名一致）
    match = re.match(r"^([^_]+)__(.+)-(\d+)$", instance_id)
    if match:
        org, repo, pr_id = match.groups()
        return _with_registry(f"mswebench/{org.lower()}_m_{repo.lower()}:pr-{pr_id}")
    return _with_registry(DEFAULT_DOCKER_IMAGE)


def _resolve_cmake_source_dir(repo_root: Path) -> Path:
    """定位传给 ``cmake -S`` 的目录。老版本 zstd 等在 ``build/cmake``，不在仓库根。"""
    r = repo_root.resolve()
    if (r / "CMakeLists.txt").is_file():
        return r
    nested = r / "build" / "cmake" / "CMakeLists.txt"
    if nested.is_file():
        return r / "build" / "cmake"
    return r


def _has_makefile(repo_root: Path) -> bool:
    """是否像「可在源码根跑 make / bear」的工程：已有 Makefile，或为 Autotools（configure 前常无 Makefile）。"""
    r = repo_root.resolve()
    if (r / "Makefile").is_file() or (r / "makefile").is_file():
        return True
    # Autotools：./configure 之前通常只有 Makefile.am / Makefile.in / configure.ac
    if (r / "Makefile.am").is_file() or (r / "Makefile.in").is_file():
        return True
    if (r / "configure.ac").is_file() or (r / "configure").is_file():
        return True
    return False


def _argv_has_flag(argv: list[str], flag: str) -> bool:
    """是否显式传入 ``--flag`` 或 ``--flag=value``（用于 CLI 优先于环境变量）。"""
    p = flag + "="
    return any(a == flag or a.startswith(p) for a in argv)


def _effective_build_backend_choice(args: argparse.Namespace) -> str:
    if _argv_has_flag(sys.argv, "--build-backend"):
        raw = str(getattr(args, "build_backend", "auto")).strip().lower()
    else:
        raw = (
            (os.environ.get("CODE_INTENT_BUILD_BACKEND") or "").strip().lower()
            or str(getattr(args, "build_backend", "auto")).strip().lower()
        )
    if raw not in ("auto", "cmake", "make-bear"):
        print(f"⚠️ [Container] 未知 build-backend {raw!r}，按 auto 处理")
        raw = "auto"
    return raw


def _resolve_build_backend(args: argparse.Namespace, cmake_src: Path, source_dir: Path) -> str:
    raw = _effective_build_backend_choice(args)
    if raw == "cmake":
        return "cmake"
    if raw == "make-bear":
        return "make-bear"
    # auto
    has_cmake = (cmake_src / "CMakeLists.txt").is_file()
    has_make = _has_makefile(source_dir)
    if has_make and not has_cmake:
        return "make-bear"
    return "cmake"


def _apt_install_line(packages: List[str]) -> None:
    """在容器内安装一组 deb 包（需 root）。"""
    ps = " ".join(packages)
    subprocess.check_call(
        f"DEBIAN_FRONTEND=noninteractive apt-get update -qq && apt-get install -y -qq {ps}",
        shell=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=_env_for_apt_subprocess(),
    )


def _compiledb_importable() -> bool:
    """当前 ``sys.executable`` 下是否已有 compiledb（宿主 Conda 预装 + 挂载进容器时可避免 pip/SSL）。"""
    try:
        import importlib.util

        return importlib.util.find_spec("compiledb") is not None
    except Exception:
        return False


def _pip_install_quiet(pkgs: List[str]) -> None:
    """pip 安装包；在需要时加 ``--break-system-packages``（Ubuntu 24+ 等）。"""
    cmd = [sys.executable, "-m", "pip", "install", "-q", *pkgs]
    try:
        subprocess.check_call(cmd, stdout=sys.stdout, stderr=sys.stderr)
    except subprocess.CalledProcessError:
        try:
            h = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--help"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        except (subprocess.CalledProcessError, OSError):
            h = ""
        if "--break-system-packages" in h:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "--break-system-packages", *pkgs],
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
        else:
            raise


def _missing_container_build_packages(*, backend: str) -> List[str]:
    """Map missing build capabilities to Debian packages.

    Benchmark images often provide a compiler under ``/usr/local`` without
    installing the ``build-essential`` meta-package.  Package metadata is not
    a reliable capability check, especially for an intentionally offline
    compile-database container.
    """
    if backend == "cmake":
        requirements = (
            ("findutils", ("find",)),
            ("build-essential", ("c++", "g++", "clang++")),
            ("make", ("make",)),
            ("cmake", ("cmake",)),
        )
    else:
        requirements = (("bear", ("bear",)),)
    return [
        package
        for package, executables in requirements
        if not any(shutil.which(executable) for executable in executables)
    ]


def _ensure_container_build_tools(*, backend: str) -> None:
    """
    安装生成 compile_commands 所需工具。
    cmake：findutils / build-essential / make / cmake；make-bear：仅 bear（镜像里已有编译链则不再装一堆 deb）。

    该阶段不导入或安装 Python 包；compile-db-only 只需要原生构建工具，
    因此镜像没有 pip 不能成为离线结构索引的失败条件。
    """
    _maybe_configure_apt_cn_mirror()
    missing_deb = _missing_container_build_packages(backend=backend)
    if missing_deb:
        print(f"🔧 [Container] apt 安装: {', '.join(missing_deb)}")
        _apt_install_line(missing_deb)

    if backend == "cmake" and not shutil.which("cmake"):
        print("⚠️ [Container] 仍无 cmake 可执行文件，强制 apt 安装 cmake …")
        _apt_install_line(["cmake"])


def _ensure_ccache_installed() -> bool:
    """容器内确保 ccache 已安装；返回是否可用。"""
    if shutil.which("ccache"):
        return True
    print("📦 [Container] 安装 ccache …")
    try:
        subprocess.check_call(
            "apt-get install -y -q ccache",
            shell=True,
            stdout=subprocess.DEVNULL,
            env=_env_for_apt_subprocess(),
        )
        return bool(shutil.which("ccache"))
    except subprocess.CalledProcessError:
        print("⚠️ [Container] ccache 安装失败，跳过 ccache 加速")
        return False


def _inject_ccache_into_make_shell(make_shell: str) -> str:
    """
    在 make shell 前注入 ccache 支持。

    对 ponyc 等使用 cmake 的项目，通过设置
    ``CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache``
    让 cmake 在所有 C/C++ 编译命令前自动插入 ccache，无需修改 Makefile / compiler 路径。

    bear 仍会拦截并记录所有被 ccache 包装后的编译调用，compile_commands.json 依然完整；
    ccache 缓存命中时编译瞬间完成，对 LLVM 等大型子库效果显著。
    """
    if not _ensure_ccache_installed():
        return make_shell
    ccache_prefix = (
        "export CMAKE_C_COMPILER_LAUNCHER=ccache "
        "CMAKE_CXX_COMPILER_LAUNCHER=ccache "
        "CCACHE_BASEDIR=/home"
    )
    ccache_dir = (os.environ.get("CCACHE_DIR") or "").strip()
    if ccache_dir:
        ccache_prefix += f" CCACHE_DIR={ccache_dir}"
    # 若 make_shell 已经有前缀，不重复插入
    if "CMAKE_C_COMPILER_LAUNCHER" in make_shell:
        return make_shell
    result = f"{ccache_prefix} && {make_shell}"
    print(f"♻️ [Container] ccache 已注入 make shell（CMAKE_*_COMPILER_LAUNCHER=ccache）")
    return result


def _bear_record_compile_commands(source_dir: Path, out_json: Path, inner_shell: str) -> None:
    """用 bear 拦截编译调用，写出 ``compile_commands.json``。"""
    out_json.parent.mkdir(parents=True, exist_ok=True)
    if out_json.exists():
        out_json.unlink()
    cwd = str(source_dir.resolve())
    sh_wrap = ["sh", "-c", inner_shell]
    # bear 2.x：仅 ``--cdb <file>`` + ``sh -c …``（勿加 ``--output``，否则会 unrecognized arguments）。
    # bear 2.x 若写 ``--cdb <file> -- sh -c …`` 可能把子进程 argv[0] 弄成 ``--`` 触发 FileNotFoundError。
    # bear 3.x：可额外尝试 ``--output`` / ``-o`` 等。
    variants: list[list[str]] = [["bear", "--cdb", str(out_json)] + sh_wrap]
    if _bear_major_version() != 2:
        variants.extend(
            [
                ["bear", "--output", str(out_json)] + sh_wrap,
                ["bear", "-o", str(out_json)] + sh_wrap,
                ["bear", "--output", str(out_json), "--"] + sh_wrap,
                ["bear", "-o", str(out_json), "--"] + sh_wrap,
            ]
        )
    last_err: Exception | None = None
    for bear_cmd in variants:
        try:
            subprocess.check_call(bear_cmd, cwd=cwd, stdout=sys.stdout, stderr=sys.stderr)
            if out_json.is_file() and out_json.stat().st_size > 0:
                return
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            last_err = e
            continue
    # pip compiledb：仅适合「单条 make 风格」命令；0.10.x 为 ``compiledb make …``，无法包装含 ``&&`` 的任意 shell。
    print("⚠️ [Container] bear 未生成有效 compile_commands.json，尝试使用 compiledb …")
    skip_pip = (os.environ.get("CODE_INTENT_SKIP_PIP_COMPILEDB") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if _compiledb_importable():
        print("ℹ️ [Container] 已检测到 compiledb（跳过 pip install）")
    elif skip_pip:
        print(
            "❌ [Container] 未检测到 compiledb 且 CODE_INTENT_SKIP_PIP_COMPILEDB=1，无法安装；"
            "请在宿主 Conda 中执行: pip install compiledb==0.10.7"
        )
        sys.exit(1)
    else:
        _pip_install_quiet(["compiledb==0.10.7"])
    if "&&" in inner_shell or "|" in inner_shell or ";" in inner_shell:
        print(
            "❌ [Container] 当前 make-bear 的 shell 含 &&/|/;，compiledb 无法替代 bear；"
            "请在容器内修复 bear（优先已支持的 ``bear --cdb …``）或拆分构建命令。"
        )
        print(f"   末次 bear 错误: {last_err!r}")
        sys.exit(1)
    mk_parts = shlex.split(inner_shell)
    if mk_parts and mk_parts[0] == "make":
        mk_parts = mk_parts[1:]
    cmd = [
        sys.executable,
        "-m",
        "compiledb",
        "-f",
        "-o",
        str(out_json),
        "make",
        "--",
        *mk_parts,
    ]
    subprocess.check_call(cmd, cwd=cwd, stdout=sys.stdout, stderr=sys.stderr)
    if not out_json.is_file() or out_json.stat().st_size == 0:
        print(f"❌ [Container] 未能生成 {out_json}（bear / compiledb 均失败）: {last_err!r}")
        sys.exit(1)


def _compile_db_only_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "compile_db_only", False)) or os.environ.get(
        "CODE_INTENT_COMPILE_DB_ONLY", ""
    ).strip().lower() in ("1", "true", "yes")


def _write_compile_db_only_meta_and_return(
    args: argparse.Namespace,
    *,
    source_dir: Path,
    output_dir: Path,
    final_build_dir: Path,
    backend: str,
    make_shell_used: str,
) -> bool:
    """
    若启用仅生成编译数据库：将 meta 写在 ``final_build_dir/instance_meta.json``（与 CDB 同卷、便于宿主查看），
    不在 ``output_path/<instance_id>`` 下建目录。打印宿主机后续命令，并通知调用方应 ``return``。
    Returns True 表示应结束 ``run_in_docker_container``。
    """
    if not _compile_db_only_enabled(args):
        return False
    cc_path = final_build_dir / "compile_commands.json"
    host_hint = (os.environ.get("CODE_INTENT_HOST_REPO_PATH_FOR_META") or "").strip()
    path_suggestion = ""
    if host_hint:
        try:
            hp = str(Path(host_hint).expanduser().resolve())
            path_suggestion = f"{source_dir}|{hp}"
        except OSError:
            path_suggestion = f"{source_dir}|{host_hint}"
    pm_arg = shlex.quote(path_suggestion) if path_suggestion else "'/home/<repo>|<宿主机源码根>'"
    hint_cmd = (
        f"python code_intent/demo.py --host-index --instance-id {shlex.quote(args.instance_id)} "
        f"--file-path <宿主机源码根> --build-dir {shlex.quote(str(final_build_dir))} "
        f"--output-path {shlex.quote(args.output_path)} --path-map {pm_arg}"
    )
    meta: dict[str, Any] = {
        "instance_id": args.instance_id,
        "compile_db_only": True,
        "container_source_dir": str(source_dir),
        "build_root": str(final_build_dir),
        "compile_commands": str(cc_path),
        "index_dir": str(output_dir),
        "build_backend": backend,
        "make_build_shell": make_shell_used or None,
        "path_map_suggestion": path_suggestion or None,
        "host_index_command_hint": hint_cmd,
    }
    _write_instance_meta(final_build_dir, meta)
    print("✅ [Container] 已生成 compile_commands.json（--compile-db-only / CODE_INTENT_COMPILE_DB_ONLY）：跳过容器内索引。")
    print(f"📎 [Container] instance_meta.json → {final_build_dir / 'instance_meta.json'}")
    if path_suggestion:
        print(f"📎 [Container] 建议 --path-map: {path_suggestion}")
    print(f"ℹ️ [Container] 在仓库根执行宿主机索引，例如:\n   {hint_cmd}")
    return True


def run_host_index_only(args: argparse.Namespace) -> None:
    """宿主机：读取 compile_commands.json + 路径映射后执行 index_repo。"""
    build_dir = (args.build_dir or "").strip()
    if not build_dir:
        print("❌ [Host] --host-index 需要 --build-dir（其下须有 compile_commands.json）")
        sys.exit(2)
    bd = Path(build_dir).expanduser().resolve()
    cc = bd / "compile_commands.json"
    if not cc.is_file():
        print(f"❌ [Host] 未找到 {cc}")
        sys.exit(2)
    src = Path(args.file_path).expanduser().resolve()
    if not src.is_dir():
        print(f"❌ [Host] --file-path 须为宿主机上的源码根目录: {src}")
        sys.exit(2)

    os.environ["PROJECT_ROOT_OVERRIDE"] = str(src)
    pm = (args.path_map or "").strip() or (os.environ.get("CODE_INTENT_COMPILE_COMMANDS_PATH_MAP") or "").strip()
    if pm:
        os.environ["CODE_INTENT_COMPILE_COMMANDS_PATH_MAP"] = pm
        print(f"🗺️ [Host] 路径映射: {pm!r}")
    else:
        print(
            "⚠️ [Host] 未设置 --path-map / CODE_INTENT_COMPILE_COMMANDS_PATH_MAP；"
            "若 JSON 内仍为容器路径，将无法打开源文件。"
        )

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    instance_root, output_dir = _prepare_instance_dirs(args.output_path, args.instance_id)

    from code_intent.env_utils import load_toenv
    from code_intent.code_indexer import index_repo

    load_toenv()
    _configure_edge_debug_log(output_dir)

    iid = (args.instance_id or "").strip()
    if iid.startswith("ponylang__ponyc"):
        ex_key = "CODE_INTENT_INDEX_EXCLUDE_PATH_SUBSTR"
        cur = (os.environ.get(ex_key) or "").strip()
        parts = [x.strip() for x in cur.split(",") if x.strip()]
        llvm_tok = "lib/llvm/"
        if not any(llvm_tok in x.replace("\\", "/").lower() for x in parts):
            parts.append(llvm_tok)
            os.environ[ex_key] = ",".join(parts)
            print(f"🔧 [Host] ponyc 索引默认排除 LLVM: {ex_key}={os.environ[ex_key]!r}")

    clang_so = (
        (os.environ.get("LIBCLANG_LIBRARY_FILE") or os.environ.get("CLANG_LIBRARY_FILE") or "").strip() or None
    )
    print(f"🚀 [Host] index_repo: build_dir={bd}, output_dir={output_dir}")
    index_repo(str(bd), str(output_dir), clang_library_path=clang_so)
    dump_entities_per_file(str(output_dir))

    cmake_src = _resolve_cmake_source_dir(src)
    backend = _resolve_build_backend(args, cmake_src, src)
    mbs = (args.make_build_shell or os.environ.get("CODE_INTENT_MAKE_BUILD_SHELL") or "").strip() or None
    meta = {
        "instance_id": args.instance_id,
        "source_root": str(src),
        "cmake_source_dir": str(cmake_src if backend == "cmake" else src),
        "build_backend": backend,
        "make_build_shell": mbs,
        "build_root": str(bd),
        "compile_commands_dir": str(bd),
        "index_dir": str(output_dir),
        "compile_commands": str(cc),
        "cmake_args": args.cmake_arg,
        "indexed_on_host": True,
        "path_map": pm or None,
    }
    _write_instance_meta(instance_root, meta)
    print(f"✅ [Host] 索引完成: {output_dir}")


def run_in_docker_container(args: argparse.Namespace) -> None:
    print("🐳 [Container] 成功进入容器环境！")

    _ensure_git_proxy_in_container()
    _ensure_git_safe_directory_wildcard_in_docker()

    source_dir = Path(args.file_path).expanduser()
    if not source_dir.exists():
        print(f"❌ 找不到源码目录: {source_dir}")
        sys.exit(1)
    source_dir = source_dir.resolve()
    _ensure_git_safe_directory(source_dir)
    build_dir = (
        Path(args.build_dir).expanduser().resolve()
        if args.build_dir
        else source_dir
    )
    if not build_dir.exists():
        build_dir.mkdir(parents=True, exist_ok=True)
    # compile-db-only：不在挂载的 index_store 下 mkdir（避免容器 root 属主）；索引目录由阶段 2 宿主创建。
    if _compile_db_only_enabled(args):
        out_base = Path(args.output_path).expanduser().resolve()
        instance_root = out_base / args.instance_id
        output_dir = instance_root / "index_output"
    else:
        instance_root, output_dir = _prepare_instance_dirs(args.output_path, args.instance_id)

    cmake_src = _resolve_cmake_source_dir(source_dir)
    backend = _resolve_build_backend(args, cmake_src, source_dir)
    print(f"🔧 [Container] 构建后端: {backend}")

    final_build_dir: Path
    make_shell_used = ""

    if backend == "make-bear":
        if _argv_has_flag(sys.argv, "--make-build-shell"):
            make_shell_used = (getattr(args, "make_build_shell", "") or "").strip()
        else:
            make_shell_used = (
                (getattr(args, "make_build_shell", "") or "").strip()
                or (os.environ.get("CODE_INTENT_MAKE_BUILD_SHELL") or "").strip()
            )
        if not make_shell_used and "jqlang__jq" in (args.instance_id or ""):
            make_shell_used = JQLANG_JQ_MAKE_BUILD_SHELL
            print(
                "🔧 [Container] jqlang__jq 使用固定 make-bear shell（可用 --make-build-shell / "
                "CODE_INTENT_MAKE_BUILD_SHELL 覆盖）。"
            )
        if not make_shell_used and "ponylang__ponyc" in (args.instance_id or ""):
            make_shell_used = ponyc_make_build_shell_for_instance(args.instance_id) or "make"
            print(
                "🔧 [Container] ponylang__ponyc：使用 ponyc_image_run_sh.json 推导的 make-bear shell"
                "（无则 ``make``；可用 --make-build-shell / CODE_INTENT_MAKE_BUILD_SHELL 覆盖）。"
            )
        if not make_shell_used:
            print(
                "❌ [Container] make-bear 需要 --make-build-shell 或环境变量 CODE_INTENT_MAKE_BUILD_SHELL\n"
                "   示例（ponyc）：--make-build-shell 'cd /home/ponyc && make && make test'（索引请去掉 test）"
            )
            sys.exit(1)
        if not _has_makefile(source_dir):
            print(
                f"❌ [Container] make-bear 需要在源码根目录存在 Makefile，或 Autotools 迹象（Makefile.am / configure.ac 等）: {source_dir}"
            )
            sys.exit(1)
        _ensure_container_build_tools(backend=backend)
        out_json = build_dir / "compile_commands.json"
        reuse_cc = (
            _truthy_env("CODE_INTENT_SKIP_BEAR_IF_COMPILE_COMMANDS_EXISTS")
            and out_json.is_file()
            and out_json.stat().st_size > 8
        )
        if reuse_cc:
            print(
                f"♻️ [Container] 已存在 compile_commands.json，跳过 bear 录制（"
                f"CODE_INTENT_SKIP_BEAR_IF_COMPILE_COMMANDS_EXISTS=1）：{out_json}"
            )
        else:
            if _truthy_env("CODE_INTENT_USE_CCACHE"):
                make_shell_used = _inject_ccache_into_make_shell(make_shell_used)
            print(f"🔧 [Container] 使用 bear 在源码目录录制编译命令（输出 → {out_json}）…")
            _bear_record_compile_commands(source_dir, out_json, make_shell_used)
        final_build_dir = build_dir
        print(f"✅ 编译数据库就绪: {out_json}")
    else:
        if not (cmake_src / "CMakeLists.txt").is_file():
            print(
                f"❌ [Container] cmake 后端未找到 CMakeLists.txt，已检查：\n"
                f"   {source_dir}\n"
                f"   {source_dir / 'build' / 'cmake'}\n"
                f"   纯 Makefile 项目请使用 --build-backend make-bear 与 --make-build-shell。"
            )
            sys.exit(1)
        if cmake_src.resolve() != source_dir.resolve():
            print(f"🔧 [Container] CMake -S 使用子目录: {cmake_src}")

        _ensure_container_build_tools(backend=backend)

        print("🔧 [Container] 生成编译配置…")
        cache_candidates = {source_dir / "CMakeCache.txt", build_dir / "CMakeCache.txt"}
        for cache_file in cache_candidates:
            if cache_file.exists():
                print(f"🧹 删除旧的 {cache_file} ...")
                cache_file.unlink()
        try:
            cmake_cmd = [
                "cmake",
                "-S",
                str(cmake_src),
                "-B",
                str(build_dir),
                "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            ]
            for extra in args.cmake_arg:
                cmake_cmd.extend(shlex.split(extra))
            print(f"🔧 [Container] CMake 命令: {cmake_cmd}")
            subprocess.check_call(cmake_cmd, stdout=sys.stdout, stderr=sys.stderr)
            build_cmd = ["cmake", "--build", str(build_dir)]
            if args.build_target:
                build_cmd.extend(["--target", *args.build_target])
            print(f"🔧 [Container] 使用 build 命令: {build_cmd}")
            subprocess.check_call(build_cmd, stdout=sys.stdout, stderr=sys.stderr)
        except subprocess.CalledProcessError:
            print("❌ CMake 或构建阶段失败")
            sys.exit(1)

        expected_path = build_dir / "compile_commands.json"
        final_build_dir = build_dir
        if not expected_path.exists():
            print("⚠️ 在构建目录未找到 compile_commands.json，正在搜索...")
            try:
                search_roots = {build_dir, source_dir}
                found_paths: list[str] = []
                for root in search_roots:
                    output = (
                        subprocess.check_output(
                            ["find", str(root), "-maxdepth", "6", "-name", "compile_commands.json"],
                            text=True,
                        )
                        .strip()
                    )
                    if output:
                        found_paths = output.splitlines()
                        break
                if found_paths:
                    actual_json_path = found_paths[0]
                    final_build_dir = Path(actual_json_path).parent
                    print(f"✅ 找到了！实际位置在: {final_build_dir}")
                else:
                    print("❌ 致命错误：未找到 compile_commands.json")
                    subprocess.call(["ls", "-F", str(build_dir)])
                    sys.exit(1)
            except Exception as e:
                print(f"❌ 搜索出错: {e}")
                sys.exit(1)
        else:
            print(f"✅ 编译数据库已生成: {expected_path}")

    if _write_compile_db_only_meta_and_return(
        args,
        source_dir=source_dir,
        output_dir=output_dir,
        final_build_dir=final_build_dir,
        backend=backend,
        make_shell_used=make_shell_used,
    ):
        return

    print("📦 [Container] 安装系统依赖...")
    if shutil.which("clang") and shutil.which("clang++"):
        print("📦 [Container] 镜像内已有 clang，跳过 apt install")
    else:
        if any(os.environ.get(k) for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")):
            print(
                "📦 [Container] apt 子进程已去掉 HTTP(S) 代理（避免 Ubuntu 源经代理 502）；"
                "pip 仍用当前环境。需要 apt 也走代理请设 CODE_INTENT_APT_USE_PROXY=1"
            )
        subprocess.check_call(
            "apt-get update && apt-get install -y clang",
            shell=True,
            stdout=subprocess.DEVNULL,
            env=_env_for_apt_subprocess(),
        )

    print("📦 [Container] 安装索引阶段 Python 依赖（仅 clang / libclang / python-dotenv；embedding 与 intent 在宿主机）...")
    pkgs = [
        "clang==18.1.8",
        "libclang==18.1.1",
        "python-dotenv",
    ]
    cmd = [sys.executable, "-m", "pip", "install"]
    try:
        h = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--help"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if "--break-system-packages" in h:
            cmd.append("--break-system-packages")
    except (subprocess.CalledProcessError, OSError):
        pass
    cmd.extend(pkgs)
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        subprocess.call(cmd)
        sys.exit(1)

    print("🔍 [Container] 定位依赖位置...")
    try:
        find_cmd = "find /usr -type d -name 'libclang' | grep 'packages' | head -n 1"
        libclang_pkg_path = subprocess.check_output(find_cmd, shell=True, text=True).strip()
        if libclang_pkg_path:
            site_pkgs_path = os.path.dirname(libclang_pkg_path)
            if site_pkgs_path not in sys.path:
                sys.path.insert(0, site_pkgs_path)
    except Exception:
        pass

    import importlib

    importlib.invalidate_caches()

    clang_lib_so = None
    try:
        find_so_cmd = "find /usr -name 'libclang.so' | head -n 1"
        so_path = subprocess.check_output(find_so_cmd, shell=True, text=True).strip()
        if so_path:
            clang_lib_so = so_path
            print(f"✅ 找到动态库: {clang_lib_so}")
    except Exception:
        pass

    print("🚀 [Container] 开始执行索引...")
    sys.path.append(CONTAINER_WORKSPACE)

    from code_intent.env_utils import load_toenv
    from code_intent.code_indexer import index_repo

    load_toenv()

    _configure_edge_debug_log(output_dir)

    index_repo(str(final_build_dir), str(output_dir), clang_library_path=clang_lib_so)
    dump_entities_per_file(str(output_dir))

    meta = {
        "instance_id": args.instance_id,
        "source_root": str(source_dir),
        "cmake_source_dir": str(cmake_src if backend == "cmake" else source_dir),
        "build_backend": backend,
        "make_build_shell": make_shell_used or None,
        "build_root": str(final_build_dir),
        "cmake_build_dir": str(build_dir),
        "compile_commands_dir": str(final_build_dir),
        "index_dir": str(output_dir),
        "compile_commands": str((final_build_dir / "compile_commands.json")),
        "cmake_args": args.cmake_arg,
    }
    _write_instance_meta(instance_root, meta)

    print(f"✅ [Container] 索引构建完成，输出目录: {output_dir}")
    print("ℹ️ 退出容器后，可在宿主机运行 python example.py 构建 intent/feature 并测试检索。")


def dump_entities_per_file(output_dir: str) -> None:
    base = Path(output_dir)
    per_file: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"functions": [], "classes": []}
    )

    func_path = base / "functions.jsonl"
    if func_path.exists():
        with func_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                per_file[rec.get("def_file", "unknown")]["functions"].append(rec)

    class_path = base / "classes.jsonl"
    if class_path.exists():
        with class_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                per_file[rec.get("def_file", "unknown")]["classes"].append(rec)

    debug_dir = base / "per_file_entities"
    debug_dir.mkdir(parents=True, exist_ok=True)

    for file_path, content in per_file.items():
        safe_name = (
            file_path.replace(":", "").replace("/", "__").replace("\\", "__")
            if file_path
            else "unknown"
        )
        out_path = debug_dir / f"{safe_name}.json"
        payload = {"source_file": file_path, **content}
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


def launch_docker_from_host(args: argparse.Namespace) -> None:
    print("💻 [Host] 准备启动 Docker 环境...")
    cwd = os.getcwd()
    pip_cache = os.path.expanduser("~/.cache/pip")
    compile_db_only = _compile_db_only_enabled(args)
    current_script_name = os.path.basename(__file__)
    # 仓库挂载在 /workspace，脚本在 code_intent/ 下
    script_in_container = f"code_intent/{current_script_name}"
    docker_image = _resolve_docker_image(args.instance_id)
    print(f"🐳 [Host] 选用镜像: {docker_image}")
    tty_flag = ["-it"] if (sys.stdin.isatty() and sys.stdout.isatty()) else ["-i"]
    docker_network = (
        os.environ.get("CODE_INTENT_DOCKER_NETWORK", "host").strip() or "host"
    )
    workspace_mount = f"{cwd}:{CONTAINER_WORKSPACE}"
    if compile_db_only:
        workspace_mount += ":ro"
    docker_cmd = [
        "docker",
        "run",
        "--rm",
        f"--network={docker_network}",
        *tty_flag,
        "-v",
        workspace_mount,
        "-w",
        CONTAINER_WORKSPACE,
        "-e",
        "RUNNING_IN_DOCKER=true",
        "-e",
        "GIT_TERMINAL_PROMPT=0",
    ]
    if _truthy_env("EDGE_DEBUG"):
        docker_cmd.extend(["-e", "EDGE_DEBUG=1"])
    if compile_db_only:
        docker_cmd.extend(["-e", "PYTHONDONTWRITEBYTECODE=1"])
    else:
        docker_cmd.extend(["-v", f"{pip_cache}:/root/.cache/pip"])
        if os.environ.get("OPENAI_API_KEY"):
            docker_cmd.extend(["-e", "OPENAI_API_KEY"])
    if docker_network != "none":
        _docker_append_proxy_env(docker_cmd)
    for _pass in (
        "CODE_INTENT_APT_MIRROR",
        "CODE_INTENT_APT_USE_PROXY",
        "CODE_INTENT_COMPILE_DB_ONLY",
        "CODE_INTENT_SKIP_BEAR_IF_COMPILE_COMMANDS_EXISTS",
        "CODE_INTENT_USE_CCACHE",
        "CMAKE_BUILD_PARALLEL_LEVEL",
    ):
        if _pass in os.environ:
            docker_cmd.extend(["-e", _pass])

    # ccache：挂载宿主机缓存目录并设 CCACHE_DIR
    ccache_host = (os.environ.get("CODE_INTENT_CCACHE_DIR") or "").strip()
    if ccache_host:
        ccache_host_p = Path(ccache_host).expanduser().resolve()
        ccache_host_p.mkdir(parents=True, exist_ok=True)
        container_ccache = str(ccache_host_p)  # 容器内路径与宿主相同（绝对路径）
        docker_cmd.extend(["-v", f"{ccache_host_p}:{container_ccache}"])
        docker_cmd.extend(["-e", f"CCACHE_DIR={container_ccache}"])
        print(f"♻️ [Host] ccache 目录: {ccache_host_p} → 容器 CCACHE_DIR={container_ccache}")

    bind_src = (getattr(args, "host_repo_bind", "") or "").strip() or os.environ.get(
        "CODE_INTENT_HOST_REPO_BIND", ""
    ).strip()
    file_path_arg = args.file_path

    if bind_src:
        hp = Path(bind_src).expanduser().resolve()
        if not hp.is_dir():
            print(f"❌ [Host] --host-repo-bind / CODE_INTENT_HOST_REPO_BIND 不是目录: {hp}")
            sys.exit(1)
        container_dest = str(Path(args.file_path).expanduser())
        bind_mount = f"{hp}:{container_dest}"
        if compile_db_only:
            bind_mount += ":ro"
        docker_cmd.extend(["-v", bind_mount])
        file_path_arg = container_dest
        print(f"📂 [Host] 挂载宿主机仓库 → 容器: {hp} → {container_dest}")
    elif _is_mswebench_style_home_source_path(args.file_path):
        # 不把宿主机上偶然存在的 /home/foo 当成数据源（与 batch_index 的容器逻辑路径约定一致）
        print(
            f"ℹ️ [Host] {args.file_path} 按容器内路径处理，不自动挂载；"
            f"索引使用镜像内该路径下的源码。若必须用宿主机 clone，请传 --host-repo-bind。"
        )
    else:
        host_file_path = Path(args.file_path).expanduser()
        if host_file_path.exists():
            file_path_arg = str(host_file_path.resolve())
            source_mount = f"{file_path_arg}:{file_path_arg}"
            if compile_db_only:
                source_mount += ":ro"
            docker_cmd.extend(["-v", source_mount])
        else:
            print(f"⚠️ [Host] {args.file_path} 在宿主机不存在，假定容器镜像内已包含源码。")

    build_dir_arg = args.build_dir
    if build_dir_arg:
        host_build_dir = Path(build_dir_arg).expanduser()
        host_build_dir.mkdir(parents=True, exist_ok=True)
        build_dir_arg = str(host_build_dir.resolve())
        docker_cmd.extend(["-v", f"{build_dir_arg}:{build_dir_arg}"])

    host_output_path = Path(args.output_path).expanduser()
    host_output_path.mkdir(parents=True, exist_ok=True)
    output_path_arg = str(host_output_path.resolve())
    docker_cmd.extend(["-v", f"{output_path_arg}:{output_path_arg}"])

    conda_host = _host_conda_env_dir()
    explicit_py = (os.environ.get("CODE_INTENT_DOCKER_PYTHON") or "").strip()
    forced_inner: str | None = explicit_py or None

    if conda_host is not None:
        conda_mount = f"{conda_host}:{CONTAINER_CONDA_ENV_PREFIX}"
        if compile_db_only:
            conda_mount += ":ro"
        docker_cmd.extend(["-v", conda_mount])
        if not forced_inner:
            forced_inner = _conda_python_container_path(conda_host, CONTAINER_CONDA_ENV_PREFIX)
        print(
            f"📦 [Host] 已挂载 Conda 环境: {conda_host} → {CONTAINER_CONDA_ENV_PREFIX}，使用 Python: {forced_inner}"
        )

    if "ponylang__ponyc" in args.instance_id and not args.docker_shell:
        if not forced_inner:
            print(
                "❌ [Host] ponylang__ponyc 需在宿主指定 Conda 环境或容器内 Python：\n"
                "   export CODE_INTENT_DOCKER_CONDA_PREFIX=/path/to/conda/env\n"
                "   或 export CODE_INTENT_DOCKER_PYTHON=/容器内解释器路径（挂载 Conda 后常为 "
                f"{CONTAINER_CONDA_ENV_PREFIX}/bin/python3）"
            )
            sys.exit(1)

    if args.docker_shell:
        inner_cmd = [docker_image, "/bin/bash"]
    else:
        inner_cmd = _docker_inner_cmd_run_demo(
            docker_image,
            args,
            script_in_container,
            file_path_arg,
            output_path_arg,
            build_dir_arg or "",
            forced_python=forced_inner,
        )

    docker_cmd.extend(inner_cmd)
    print(
        "🚀 [Host] 运行脚本: /bin/bash (debug mode)"
        if args.docker_shell
        else f"🚀 [Host] 运行脚本: {current_script_name}"
    )
    try:
        subprocess.check_call(docker_cmd)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        pass


def main() -> None:
    args = _parse_args()
    if getattr(args, "host_index", False):
        if os.environ.get("RUNNING_IN_DOCKER") == "true":
            print("❌ --host-index 仅供宿主机使用（勿在容器内加此参数）")
            sys.exit(2)
        run_host_index_only(args)
        return
    if os.environ.get("RUNNING_IN_DOCKER") == "true":
        run_in_docker_container(args)
    else:
        launch_docker_from_host(args)


if __name__ == "__main__":
    main()
