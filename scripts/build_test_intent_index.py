#!/usr/bin/env python3
"""Build the standalone test-case intent index for one repository instance."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

TEST_ARTIFACTS = (
    "test_cases.jsonl",
    "test_index.faiss",
    "test_intents.jsonl",
    "test_meta.jsonl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract repository C/C++ test cases, enrich them with direct production "
            "entities, and build the test-intent FAISS index."
        )
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        required=True,
        help="Existing CodeIndex output containing functions.jsonl and callgraph.jsonl.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        required=True,
        help="Repository checkout at the indexed revision.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Stable cache namespace, normally the benchmark instance ID.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Discard existing test artifacts before rebuilding.",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow heuristic intents and local stub embeddings (not recommended for retrieval evaluation).",
    )
    return parser.parse_args()


def jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    args = parse_args()
    if not args.allow_fallback:
        required = (
            "LLM_API_URL",
            "LLM_API_KEY",
            "LLM_MODEL",
            "VOYAGE_API_KEY",
            "VOYAGE_EMBED_MODEL",
        )
        missing = [name for name in required if not os.getenv(name)]
        if missing:
            raise SystemExit(
                "Semantic test-intent build requires: " + ", ".join(missing)
                + ". Use --allow-fallback only for local smoke tests."
            )
        os.environ["INTENT_ABORT_ON_FALLBACK"] = "1"
        os.environ["CODE_INTENT_EMBED_ABORT_ON_FALLBACK"] = "1"
    from code_intent.code_intent.intent_extractor import (
        build_test_intents,
        intent_test_artifacts_ready,
    )

    index_dir = args.index_dir.resolve()
    repo_root = args.repo_root.resolve()
    if not (index_dir / "functions.jsonl").is_file():
        raise SystemExit(f"functions.jsonl not found in {index_dir}")
    if not repo_root.is_dir():
        raise SystemExit(f"Repository root not found: {repo_root}")

    if args.rebuild:
        for name in TEST_ARTIFACTS:
            path = index_dir / name
            if path.exists():
                path.unlink()

    build_test_intents(
        index_dir=index_dir,
        repo_id=args.repo_id or repo_root.name,
        repo_root=repo_root,
        reuse_cache=not args.rebuild,
        fallback_allowed=args.allow_fallback,
    )
    summary = {
        "index_dir": str(index_dir),
        "repo_root": str(repo_root),
        "test_cases": jsonl_count(index_dir / "test_cases.jsonl"),
        "test_intents": jsonl_count(index_dir / "test_intents.jsonl"),
        "ready": intent_test_artifacts_ready(index_dir),
        "fallback_allowed": args.allow_fallback,
    }
    print(json.dumps(summary, indent=2))
    return 0 if summary["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
