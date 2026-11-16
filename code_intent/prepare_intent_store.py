#!/usr/bin/env python3
"""Create an isolated intent output store backed by immutable structural links."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _instance_ids(path: Path) -> set[str]:
    selected = set()
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            instance_id = str(record["instance_id"])
            selected.add(
                instance_id
                if instance_id.startswith("swe-bench_")
                else f"swe-bench_{instance_id}"
            )
    return selected


def prepare_instance(source: Path, destination: Path) -> None:
    source_index = source / "index_output"
    destination_index = destination / "index_output"
    destination_index.mkdir(parents=True, exist_ok=True)

    metadata = source / "instance_meta.json"
    destination_metadata = destination / metadata.name
    if metadata.exists() and not destination_metadata.exists():
        destination_metadata.symlink_to(metadata.resolve())

    for source_entry in source_index.iterdir():
        destination_entry = destination_index / source_entry.name
        if destination_entry.exists() or destination_entry.is_symlink():
            continue
        destination_entry.symlink_to(
            source_entry.resolve(), target_is_directory=source_entry.is_dir()
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Link audited structural artifacts into an isolated intent store."
    )
    parser.add_argument("--structural-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--instances-file", type=Path, required=True)
    args = parser.parse_args()

    selected = _instance_ids(args.instances_file)
    missing = []
    prepared = 0
    for instance_id in sorted(selected):
        source = args.structural_root / instance_id
        if not (source / "index_output" / "functions.jsonl").exists():
            missing.append(instance_id)
            continue
        prepare_instance(source, args.output_root / instance_id)
        prepared += 1
    if missing:
        raise FileNotFoundError(
            "missing structural indexes: " + ", ".join(missing)
        )
    print(f"Prepared {prepared} isolated structural index directories.")


if __name__ == "__main__":
    main()
