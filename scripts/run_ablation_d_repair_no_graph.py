#!/usr/bin/env python3
"""Ablation arm D: Repair-only with all dual-layer graph access removed."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_repair_only_ablation as runner  # noqa: E402


def main() -> int:
    return runner.main(
        runner.ARM_D_REPAIR_NO_GRAPH,
        entrypoint=Path(__file__).resolve(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
