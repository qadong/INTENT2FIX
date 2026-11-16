#!/usr/bin/env python3
"""
统计目录下（可递归）所有 ``batch_index`` 风格 ``*.log`` 中的成功 / 失败实例 id。

主日志常被覆盖时，把每次跑完的 ``batch_index_summary.log`` 复制到同一目录（如
``code_intent/batch_index_logs/``），再用本脚本汇总；``batch_index --resume`` 可用
``--resume-log-dir`` 指向该目录以跳过已成功实例。

用法::

    python3 code_intent/scripts/summarize_batch_index_logs.py code_intent/batch_index_logs
    python3 code_intent/scripts/summarize_batch_index_logs.py code_intent/batch_index_logs --recursive
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_CODE_INTENT = _SCRIPT_DIR.parent
_REPO_ROOT = _CODE_INTENT.parent
if str(_CODE_INTENT) not in sys.path:
    sys.path.insert(0, str(_CODE_INTENT))

from batch_index import (  # noqa: E402
    _iter_log_files_in_dir,
    load_completed_instances_from_log,
    load_failed_instances_from_log,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="汇总 batch_index 历史 .log 中的 [SUCCESS] / [FAILURE]")
    ap.add_argument(
        "log_dir",
        type=Path,
        nargs="?",
        default=_CODE_INTENT / "batch_index_logs",
        help="含若干 .log 的目录；默认 code_intent/batch_index_logs",
    )
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="递归扫描 **/*.log",
    )
    ap.add_argument(
        "--list-ids",
        action="store_true",
        help="打印每个 id 一行（默认只打印数量与样例）",
    )
    args = ap.parse_args()
    d = args.log_dir.expanduser().resolve()
    if not d.is_dir():
        print(f"❌ 不是目录: {d}")
        return 2

    files = _iter_log_files_in_dir(d, recursive=args.recursive)
    success: set[str] = set()
    failed: set[str] = set()
    for p in files:
        success |= load_completed_instances_from_log(p)
        failed |= load_failed_instances_from_log(p)

    both = success & failed
    only_success = success - failed
    only_failed = failed - success

    print(f"📂 目录: {d}")
    print(f"📄 扫描 {len(files)} 个 .log（recursive={args.recursive}）")
    print(f"✅ [SUCCESS] 去重: {len(success)} 个")
    print(f"❌ [FAILURE]/[ERROR] 去重: {len(failed)} 个")
    print(f"⚖️  既成功又失败过（不同批次）: {len(both)} 个 —— batch_index --resume 以成功为准会跳过")
    if args.list_ids:
        print("\n--- SUCCESS ids ---")
        for x in sorted(success):
            print(x)
        print("\n--- FAILURE ids ---")
        for x in sorted(failed):
            print(x)
    else:
        if success:
            samp = ", ".join(sorted(success)[:12])
            print(f"   成功样例: {samp}{' …' if len(success) > 12 else ''}")
        if failed:
            samp = ", ".join(sorted(failed)[:12])
            print(f"   失败样例: {samp}{' …' if len(failed) > 12 else ''}")

    print(
        "\n续跑跳过已成功: "
        f"python3 code_intent/batch_index.py --resume --resume-log-dir {d} "
        f"{'--resume-log-recursive ' if args.recursive else ''}..."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
