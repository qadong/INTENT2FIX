"""
从 ``ponyc_image_run_sh.json``（instance_id → 镜像内 ``/home/run.sh`` 全文）推导 bear 用的 ``make_build_shell``：

去掉 **末尾** 所有「明显是跑测试」的行（如 ``make test``、``make test config=debug``、
``make -f Makefile-lib-llvm test`` 等），再把剩余可执行行用 `` && `` 拼成一行给 ``sh -c``。

默认 JSON 路径：仓库根下 ``INTENT2FIX/benchmark/ponyc_image_run_sh.json``；
可用环境变量 ``CODE_INTENT_PONYC_RUN_SH_JSON`` 覆盖。
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

# code_intent/ponyc_build_from_run_sh.py → 仓库根
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_JSON = _REPO_ROOT / "INTENT2FIX" / "benchmark" / "ponyc_image_run_sh.json"

# 末尾剥离：mswebench ponyc 常见最后一行是 ``make … test`` / ``ninja test`` / ``ctest``
def _line_is_trailing_test_command(line: str) -> bool:
    s = line.strip()
    if not s or s.startswith("#"):
        return False
    if re.match(r"^ninja\s+test\b", s, re.I):
        return True
    if re.match(r"^ctest\b", s, re.I):
        return True
    if re.match(r"^make\b", s, re.I):
        return bool(re.search(r"(^|\s)test(\s|$|\s+)", s))
    return False


def run_sh_to_make_build_shell(run_sh: str) -> str:
    """
    将 run.sh 全文转为单行 shell（去尾部的 test 相关行；shebang / 纯 ``set -e`` 可省略）。
    """
    text = run_sh.replace("\r\n", "\n")
    lines = text.split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and _line_is_trailing_test_command(lines[-1]):
        lines.pop()
    while lines and not lines[-1].strip():
        lines.pop()

    body: list[str] = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("#!"):
            continue
        if re.fullmatch(r"set\s+[-+a-zA-Z0-9\s]+\s*", s, re.I):
            continue
        body.append(s)

    if not body:
        return "true"
    return " && ".join(body)


@lru_cache(maxsize=1)
def _load_run_sh_map(path: str) -> dict[str, str | None]:
    p = Path(path)
    if not p.is_file():
        return {}
    with p.open(encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[str, str | None] = {}
    for k, v in raw.items():
        if not k.startswith("ponylang__ponyc-"):
            continue
        if v is None:
            out[k] = None
        else:
            out[k] = str(v)
    return out


def ponyc_make_build_shell_for_instance(instance_id: str) -> str | None:
    """
    若 JSON 中有该 instance 的 run.sh，返回推导后的 ``make_build_shell``；
    无文件、无键、或值为 null 时返回 ``None``（调用方自行 fallback）。
    """
    if "ponylang__ponyc" not in instance_id:
        return None
    json_path = (os.environ.get("CODE_INTENT_PONYC_RUN_SH_JSON") or "").strip()
    path = json_path or str(_DEFAULT_JSON)
    mp = _load_run_sh_map(path)
    run_sh = mp.get(instance_id)
    if not run_sh:
        return None
    return run_sh_to_make_build_shell(run_sh)


def invalidate_run_sh_cache() -> None:
    _load_run_sh_map.cache_clear()
