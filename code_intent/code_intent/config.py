from __future__ import annotations

import json
import os
import shlex
import site
import sys
import sysconfig
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from clang.cindex import Config, Index
from clang.cindex import conf, Cursor

_CLANG_CONFIGURED = False

DEFAULT_LLVM14_LIB = "/usr/lib/llvm-14/lib/libclang.so.1"


def _discover_libclang_shared_object() -> Optional[str]:
    """
    在 pip 安装的 ``libclang`` wheel（site-packages/clang/native/…）或常见系统路径下查找 ``libclang.so*``。
    避免容器内仅有 pip 安装的 libclang 而无系统 ``/usr/lib/llvm-*`` 时误用硬编码路径。
    """
    roots: list[Path] = []
    try:
        for key in ("platlib", "purelib"):
            try:
                roots.append(Path(sysconfig.get_path(key)))
            except Exception:
                pass
        for p in site.getsitepackages():
            roots.append(Path(p))
        u = site.getusersitepackages()
        if u:
            roots.append(Path(u))
    except Exception:
        pass
    seen: set[str] = set()
    ordered_roots: list[Path] = []
    for r in roots:
        try:
            rp = r.resolve()
        except Exception:
            rp = r
        s = str(rp)
        if s in seen or not rp.is_dir():
            continue
        seen.add(s)
        ordered_roots.append(rp)

    preferred_suffixes = (
        "clang/native/libclang.so",
        "clang/native/libclang.so.1",
        "libclang/libclang.so",
        "libclang/libclang.so.1",
    )
    for base in ordered_roots:
        for suf in preferred_suffixes:
            cand = base / suf
            if cand.is_file():
                return str(cand.resolve())
        try:
            for cand in sorted(base.rglob("libclang.so*"), key=lambda p: len(str(p))):
                if cand.is_file() and "site-packages" in str(cand):
                    return str(cand.resolve())
        except OSError:
            continue
    if os.path.isfile(DEFAULT_LLVM14_LIB):
        return DEFAULT_LLVM14_LIB
    return None


def _patch_specialized_cursor_template() -> None:
    if hasattr(Cursor, "get_specialized_cursor_template"):
        return
    try:
        c_func = conf.lib.clang_getSpecializedCursorTemplate
        c_func.argtypes = [Cursor]
        c_func.restype = Cursor

        def impl(self: Cursor):
            res = c_func(self)
            return res if res and not res.kind.is_invalid() else None

        Cursor.get_specialized_cursor_template = impl  # type: ignore[attr-defined]
    except AttributeError:
        pass


def init_clang(clang_library_path: Optional[str] = None) -> Index:
    global _CLANG_CONFIGURED
    candidate = (
        clang_library_path
        or os.environ.get("CLANG_LIBRARY_FILE")
        or os.environ.get("LIBCLANG_LIBRARY_FILE")
        or _discover_libclang_shared_object()
    )
    try:
        if candidate and os.path.isfile(candidate):
            Config.set_library_file(candidate)
        else:
            search_path = os.environ.get("CLANG_LIBRARY_PATH")
            if search_path:
                Config.set_library_path(search_path)
        
        Config.set_compatibility_check(False)
        _CLANG_CONFIGURED = True
    except Exception as e:
        print(f"Config libclang error (candidate={candidate}): {e}")
    try:
        _patch_specialized_cursor_template()
        return Index.create()
    except Exception as e:
        print(f"Index.create() failed: {e}")
        raise


def get_clang_resource_dir() -> Optional[str]:
    try:
        lib_path = Config.library_file
        if not lib_path:
            lib_path = (
                os.environ.get("CLANG_LIBRARY_FILE")
                or os.environ.get("LIBCLANG_LIBRARY_FILE")
                or _discover_libclang_shared_object()
                or (DEFAULT_LLVM14_LIB if os.path.isfile(DEFAULT_LLVM14_LIB) else None)
            )
        
        if lib_path and os.path.exists(lib_path):
            p = Path(lib_path).resolve()
            curr = p.parent
            while curr and curr != curr.parent:
                clang_dir = curr / "clang"
                if not clang_dir.exists() and curr.name == "x86_64-linux-gnu":
                    clang_dir = curr.parent / "clang"
                if clang_dir.exists():
                    versions = sorted([d for d in clang_dir.iterdir() if d.is_dir()], reverse=True)
                    for v in versions:
                        inc = v / "include"
                        if inc.exists() and (inc / "stddef.h").exists():
                            return str(inc)
                curr = curr.parent
        common_paths = [
            "/usr/lib/clang/17/include",
            "/usr/lib/clang/16/include",
            "/usr/lib/clang/15/include",
            "/usr/lib/clang/14/include",
            "/usr/lib/clang/13/include",
            "/usr/lib/clang/12/include",
            "/usr/lib/clang/11/include",
            "/usr/local/lib/clang/16.0.0/include",
            "/usr/local/lib/clang/15.0.0/include",
            "/usr/local/lib/clang/14.0.0/include",
        ]
        for cp in common_paths:
            if os.path.exists(cp) and os.path.exists(os.path.join(cp, "stddef.h")):
                return cp
        try:
            import subprocess
            res = subprocess.check_output(
                "find /usr/lib/clang /usr/local/lib/clang -name stddef.h 2>/dev/null | head -n 1",
                shell=True, text=True
            ).strip()
            if res:
                return str(Path(res).parent)
        except Exception:
            pass
        try:
            import subprocess
            gcc_res = subprocess.check_output(
                "find /usr/lib/gcc -name stddef.h 2>/dev/null | head -n 1",
                shell=True, text=True
            ).strip()
            if gcc_res:
                gcc_include_dir = str(Path(gcc_res).parent)
                if os.path.exists(gcc_include_dir) and os.path.exists(os.path.join(gcc_include_dir, "stddef.h")):
                    return gcc_include_dir
        except Exception:
            pass
        gcc_common_paths = [
            "/usr/lib/gcc/x86_64-linux-gnu/12/include",
            "/usr/lib/gcc/x86_64-linux-gnu/11/include",
            "/usr/lib/gcc/x86_64-linux-gnu/10/include",
            "/usr/lib/gcc/x86_64-linux-gnu/9/include",
        ]
        for gcc_path in gcc_common_paths:
            if os.path.exists(gcc_path) and os.path.exists(os.path.join(gcc_path, "stddef.h")):
                return gcc_path
                
    except Exception:
        pass
    return None


def _resolve_compile_commands_path(build_dir: str) -> Path:
    base = Path(build_dir)
    if base.is_file() and base.name == "compile_commands.json":
        return base
    candidate = base / "compile_commands.json"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"compile_commands.json not found (root: {build_dir})")


def _extract_args(entry: Dict[str, object]) -> List[str]:
    arguments = entry.get("arguments")
    if isinstance(arguments, list) and arguments:
        return [str(arg) for arg in arguments]
    command = entry.get("command")
    if isinstance(command, str) and command.strip():
        return shlex.split(command)
    raise ValueError("Compile command missing arguments/command field")


def load_compile_commands(build_dir: str) -> Dict[str, Tuple[str, List[str]]]:
    path = _resolve_compile_commands_path(build_dir)
    with path.open("r", encoding="utf-8") as f:
        entries = json.load(f)

    if not isinstance(entries, list):
        raise ValueError("compile_commands.json format invalid")

    comp_map: Dict[str, Tuple[str, List[str]]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        file_path = entry.get("file")
        if not isinstance(file_path, str):
            continue
        abs_file = os.path.abspath(file_path)
        directory = entry.get("directory")
        if not isinstance(directory, str) or not directory:
            directory = os.path.dirname(abs_file)
        try:
            args = _extract_args(entry)
        except ValueError:
            continue
        comp_map[abs_file] = (directory, args)

    if not comp_map:
        raise ValueError("compile_commands.json has no valid compile entries")

    return _apply_compile_commands_path_map_env(comp_map)


def _rew_path_under_prefix(path: str, old_p: str, new_p: str) -> str:
    """若 ``path`` 在 ``old_p`` 下，则换成 ``new_p`` 前缀；否则原样返回。"""
    if not path:
        return path
    s = path.replace("\\", "/")
    op = old_p.replace("\\", "/").rstrip("/")
    np = new_p.replace("\\", "/").rstrip("/") or "/"
    if s == op:
        return np
    if s.startswith(op + "/"):
        return np + s[len(op) :]
    return path


def _rewrite_compiler_arg_for_path_map(arg: str, old_p: str, new_p: str) -> str:
    """重写参数里可能出现的容器内绝对路径（``-I`` / ``-isystem`` / 源文件路径等）。"""
    if old_p not in arg:
        return arg
    for prefix in (
        "-I",
        "-iquote",
        "-isystem",
        "-idirafter",
        "-iframeworkwithsysroot",
        "-iframework",
        "-F",
        "-isysroot",
    ):
        if arg.startswith(prefix) and len(arg) > len(prefix):
            body = arg[len(prefix) :]
            if body.startswith("="):
                rest = body[1:]
                r2 = _rew_path_under_prefix(rest, old_p, new_p)
                if r2 != rest:
                    return prefix + "=" + r2
            else:
                r2 = _rew_path_under_prefix(body, old_p, new_p)
                if r2 != body:
                    return prefix + r2
    if arg.startswith("-") and "=" in arg:
        k, _, v = arg.partition("=")
        v2 = _rew_path_under_prefix(v, old_p, new_p)
        if v2 != v:
            return f"{k}={v2}"
    return _rew_path_under_prefix(arg, old_p, new_p)


def _apply_compile_commands_path_map_env(
    comp_map: Dict[str, Tuple[str, List[str]]],
) -> Dict[str, Tuple[str, List[str]]]:
    """
    环境变量 ``CODE_INTENT_COMPILE_COMMANDS_PATH_MAP``：``容器前缀|宿主机前缀``，
    将 compile_commands 中的 file/directory/参数里的绝对路径映射到宿主机，便于「容器录库、宿主解析」。
    """
    spec = (os.environ.get("CODE_INTENT_COMPILE_COMMANDS_PATH_MAP") or "").strip()
    if not spec or "|" not in spec:
        return comp_map
    old_raw, new_raw = spec.split("|", 1)
    old_p = old_raw.strip()
    new_p = new_raw.strip()
    if not old_p:
        return comp_map
    out: Dict[str, Tuple[str, List[str]]] = {}
    for src, (cwd, args) in comp_map.items():
        nk = _rew_path_under_prefix(src, old_p, new_p)
        ncwd = _rew_path_under_prefix(cwd, old_p, new_p)
        nargs = [_rewrite_compiler_arg_for_path_map(a, old_p, new_p) for a in args]
        out[nk] = (ncwd, nargs)
    print(
        f"[Indexer] Applied CODE_INTENT_COMPILE_COMMANDS_PATH_MAP: "
        f"{old_p!r} → {new_p!r} ({len(comp_map)} entries)"
    )
    return out

