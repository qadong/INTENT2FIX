#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from INTENT2FIX.patch_intent_audit.trajectory_leads import (
    TrajectoryLeadResult,
    extract_trajectory_leads,
    load_patch,
    load_trajectory,
    slice_trajectory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Slice a persisted Repair trajectory and optionally extract "
            "non-authoritative Patch Audit leads with one LLM request."
        )
    )
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--patch", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        default=(
            os.getenv("COTEST_PATCH_INTENT_AUDIT_MODEL")
            or os.getenv("COTEST_REPAIR_MODEL")
            or "deepseek-v4-flash"
        ),
    )
    parser.add_argument("--slice-only", action="store_true")
    parser.add_argument("--max-events", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trajectory = load_trajectory(args.trajectory)
    patch = load_patch(args.patch)
    sliced = slice_trajectory(
        trajectory,
        patch,
        max_events=max(1, min(args.max_events, 32)),
    )
    if args.slice_only:
        result = TrajectoryLeadResult(trajectory_slice=sliced)
    else:
        result = extract_trajectory_leads(sliced, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.model_dump(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"{sliced.instance_id or args.trajectory.stem}: "
        f"finished={sliced.repair_finished}, events={sliced.stats.selected_events}/"
        f"{sliced.stats.events_created}, leads={len(result.leads)}, "
        f"status={result.extraction_status}"
    )
    return 0 if result.extraction_status != "failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
