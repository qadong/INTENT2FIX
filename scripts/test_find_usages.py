#!/usr/bin/env python3
"""
Quick manual test for code_intent.entity_tools.find_usages.

Example:
  python scripts/test_find_usages.py \
    --type function \
    --name "foo" \
    --namespace "ns" \
    --class-name "Bar" \
    --limit 20 \
    --instance "swe-bench_XXX" \
    --index-store-dir "/path/to/index"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test find_usages()")
    parser.add_argument("--type", required=True, choices=["function", "class"])
    parser.add_argument("--name", required=True, help="Symbol name (exact preferred)")
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--class-name", default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--instance", default=None, help="SWEBENCH_INSTANCE_NAME")
    parser.add_argument("--index-store-dir", default=None, help="INDEX_STORE_DIR")
    return parser.parse_args()


def main() -> int:
    _ensure_repo_on_path()
    args = _parse_args()

    if args.instance:
        os.environ["SWEBENCH_INSTANCE_NAME"] = args.instance
    if args.index_store_dir:
        os.environ["INDEX_STORE_DIR"] = args.index_store_dir

    from code_intent.code_intent.entity_tools import find_usages

    result = find_usages(
        type=args.type,
        name=args.name,
        namespace=args.namespace,
        class_name=args.class_name,
        limit=args.limit,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
