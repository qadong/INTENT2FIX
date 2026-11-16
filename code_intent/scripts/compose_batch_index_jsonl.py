#!/usr/bin/env python3
"""
从主 JSONL 中筛出「可跑 batch_index」的子集：

1. **ponyc 仅 ``make`` 一把梭**：由 ``INTENT2FIX/benchmark/ponyc_image_run_sh.json`` 推导的
   ``make_build_shell`` 中 **不含** ``make libs``、**不含** ``Makefile-lib-llvm``（与
   ``ponyc_build_from_run_sh.run_sh_to_make_build_shell`` 规则一致）。
2. **其它失败实例**：从日志里解析 ``[FAILURE]`` / ``[ERROR]``（与 ``batch_index`` 一致）。
   单文件用 ``--summary-log``；**若主日志会被覆盖**，可把历次导出的日志放在同一目录下，用
   ``--summary-log-dir`` **聚合统计**（默认扫 ``*.log``，``--summary-log-recursive`` 则 ``**/*.log``）。
3. 可选 ``--extra-instances-file``：每行一个 ``instance_id``，若在主 JSONL 中存在则并入。

用法（在仓库根 INTENT2FIX 下）::

    python3 code_intent/scripts/compose_batch_index_jsonl.py \\
      --source-jsonl INTENT2FIX/benchmark/multi_swebench_c_change.jsonl \\
      --output INTENT2FIX/benchmark/batch_index_ponyc_simple_plus_failures.jsonl \\
      --summary-log-dir code_intent/batch_index_logs

然后::

    python3 code_intent/batch_index.py \\
      --jsonl INTENT2FIX/benchmark/batch_index_ponyc_simple_plus_failures.jsonl \\
      --no-clone-missing \\
      --jsonl-preserve-order

（若仓库尚未 clone，去掉 ``--no-clone-missing`` 让 batch 拉代码。）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_CODE_INTENT = _SCRIPT_DIR.parent
_REPO_ROOT = _CODE_INTENT.parent
if str(_CODE_INTENT) not in sys.path:
    sys.path.insert(0, str(_CODE_INTENT))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ponyc_build_from_run_sh import _DEFAULT_JSON, run_sh_to_make_build_shell  # noqa: E402


def _load_run_sh_map(json_path: Path | None = None) -> dict[str, str]:
    p = json_path if json_path is not None else Path(_DEFAULT_JSON)
    if not p.is_file():
        return {}
    with p.open(encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[str, str] = {}
    for k, v in raw.items():
        if k.startswith("ponylang__ponyc-") and isinstance(v, str):
            out[k] = v
    return out


def _is_ponyc_simple_make(run_sh: str) -> bool:
    sh = run_sh_to_make_build_shell(run_sh)
    if "make libs" in sh or "Makefile-lib-llvm" in sh:
        return False
    return True


def _ponyc_simple_instance_ids(run_sh_map: dict[str, str]) -> set[str]:
    return {iid for iid, text in run_sh_map.items() if _is_ponyc_simple_make(text)}


def _normalize_failure_id(raw: str) -> str:
    s = raw.strip()
    if s.startswith("swe-bench_"):
        s = s[len("swe-bench_") :]
    return s


# 与 batch_index.load_failed_instances_from_log 一致
_FAILURE_LINE = re.compile(r"\[(?:FAILURE|ERROR)\]\s+(\S+)")


def _parse_failure_ids_from_log_file(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    out: set[str] = set()
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set()
    for line in text.splitlines():
        m = _FAILURE_LINE.search(line)
        if m:
            out.add(_normalize_failure_id(m.group(1)))
    return out


def _parse_failure_ids_from_log_dir(dir_path: Path, *, recursive: bool) -> tuple[set[str], int]:
    """
    扫描目录下所有 ``*.log``，合并其中的失败 id。
    返回 (去重后的 id 集合, 参与扫描的文件数)。
    """
    if not dir_path.is_dir():
        return set(), 0
    pattern = "**/*.log" if recursive else "*.log"
    files = sorted(p for p in dir_path.glob(pattern) if p.is_file())
    out: set[str] = set()
    for p in files:
        out |= _parse_failure_ids_from_log_file(p)
    return out, len(files)


def _load_source_rows(path: Path) -> tuple[list[dict], dict[str, dict]]:
    """按文件行顺序保留 ``rows``，便于与 ``--jsonl-preserve-order`` 一致。"""
    rows: list[dict] = []
    by_id: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            iid = str(rec.get("instance_id", ""))
            if not iid:
                raise ValueError(f"{path}:{line_no}: missing instance_id")
            rows.append(rec)
            by_id[iid] = rec
    return rows, by_id


def _read_extra_ids(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="组合 batch_index 用 JSONL：ponyc 简单 make + 失败实例等")
    ap.add_argument(
        "--source-jsonl",
        type=Path,
        default=_REPO_ROOT / "INTENT2FIX" / "benchmark" / "multi_swebench_c_change.jsonl",
        help="主数据集 JSONL（需含 instance_id, repo, base_commit）",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "INTENT2FIX" / "benchmark" / "batch_index_ponyc_simple_plus_failures.jsonl",
        help="写出路径",
    )
    ap.add_argument(
        "--summary-log",
        type=Path,
        default=None,
        help="单个摘要日志；解析 [FAILURE]/[ERROR]（可与 --summary-log-dir 同时使用）",
    )
    ap.add_argument(
        "--summary-log-dir",
        type=Path,
        default=None,
        help=(
            "目录：聚合其下所有 .log 中的失败实例（适合主日志被覆盖、你另存了多份 summary）"
        ),
    )
    ap.add_argument(
        "--summary-log-recursive",
        action="store_true",
        help="与 --summary-log-dir 合用：递归扫描子目录中的 .log",
    )
    ap.add_argument(
        "--extra-instances-file",
        type=Path,
        default=None,
        help="可选：每行一个 instance_id，# 开头为注释",
    )
    ap.add_argument(
        "--ponyc-run-sh-json",
        type=Path,
        default=None,
        help="覆盖默认 ponyc_image_run_sh.json（一般不用）",
    )
    args = ap.parse_args()

    run_sh_json = args.ponyc_run_sh_json.expanduser().resolve() if args.ponyc_run_sh_json else None
    run_sh_map = _load_run_sh_map(run_sh_json)
    simple_ponyc = _ponyc_simple_instance_ids(run_sh_map)

    failures: set[str] = set()
    if args.summary_log or args.summary_log_dir:
        if args.summary_log:
            fp = args.summary_log.expanduser().resolve()
            n = _parse_failure_ids_from_log_file(fp)
            failures |= n
            if fp.is_file():
                print(f"📄 --summary-log {fp}: 解析到 {len(n)} 个失败 id")
            else:
                print(f"⚠️ --summary-log 不存在，跳过: {fp}")
        if args.summary_log_dir:
            d = args.summary_log_dir.expanduser().resolve()
            n, nfiles = _parse_failure_ids_from_log_dir(d, recursive=bool(args.summary_log_recursive))
            failures |= n
            if d.is_dir():
                print(
                    f"📂 --summary-log-dir {d}: 扫描 {nfiles} 个 .log，"
                    f"解析到 {len(n)} 个失败 id（写入前与别来源合并去重）"
                )
            else:
                print(f"⚠️ --summary-log-dir 不是目录，跳过: {d}")
    else:
        default_log = (_CODE_INTENT / "batch_index_summary.log").resolve()
        n = _parse_failure_ids_from_log_file(default_log)
        failures |= n
        if default_log.is_file() and n:
            print(f"📄 未指定 --summary-log / --summary-log-dir，使用默认 {default_log}: {len(n)} 个失败 id")

    src = args.source_jsonl.expanduser().resolve()
    rows_in_order, by_id = _load_source_rows(src)

    order: list[str] = []
    seen: set[str] = set()

    def add(iid: str) -> None:
        if iid in seen or iid not in by_id:
            return
        seen.add(iid)
        order.append(iid)

    for rec in rows_in_order:
        iid = str(rec["instance_id"])
        if iid in simple_ponyc:
            add(iid)

    for iid in sorted(failures):
        add(iid)

    if args.extra_instances_file:
        for iid in _read_extra_ids(args.extra_instances_file.expanduser().resolve()):
            add(iid)

    out_path = args.output.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for iid in order:
            f.write(json.dumps(by_id[iid], ensure_ascii=False) + "\n")

    miss_fail = sorted(failures - set(by_id.keys()))
    miss_simple_in_dataset = sorted(simple_ponyc - set(by_id.keys()))

    print(f"✅ 写出 {out_path}，共 {len(order)} 条")
    print(f"   - ponyc 简单 make 且在数据集中: {sum(1 for i in order if i in simple_ponyc)}")
    print(f"   - 来自日志的失败 id 且命中数据集: {sum(1 for i in order if i in failures)}（日志来源合计 {len(failures)} 个去重 id）")
    if miss_fail:
        print(
            f"⚠️  日志中共有 {len(miss_fail)} 个失败 id 不在主 JSONL（已跳过）: "
            f"{miss_fail[:8]}{'…' if len(miss_fail) > 8 else ''}"
        )
    if miss_simple_in_dataset:
        print(
            f"ℹ️  run_sh 标为简单 make 但有 {len(miss_simple_in_dataset)} 个不在主 JSONL（未写入）"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
