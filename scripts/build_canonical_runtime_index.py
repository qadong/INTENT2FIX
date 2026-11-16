#!/usr/bin/env python3
"""Build one audited runtime view over multiple disjoint intent stores."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


INDEX_ARTIFACTS = {
    "functions": ("func_index.faiss", "func_meta.jsonl"),
    "features": ("feature_index.faiss", "feature_meta.jsonl"),
    "tests": ("test_index.faiss", "test_meta.jsonl"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def normalize(instance_id: str) -> str:
    prefix = "swe-bench_"
    return instance_id[len(prefix):] if instance_id.startswith(prefix) else instance_id


def directory_name(instance_id: str) -> str:
    return instance_id if instance_id.startswith("swe-bench_") else f"swe-bench_{instance_id}"


def load_instances(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            instance_id = normalize(str(record.get("instance_id") or ""))
            base_commit = str(record.get("base_commit") or "")
            if not instance_id or not base_commit:
                raise ValueError(f"missing instance_id/base_commit at {path}:{line_number}")
            if instance_id in records:
                raise ValueError(f"duplicate instance {instance_id} in {path}")
            records[instance_id] = record
    return records


def git_head(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def validate_instance(
    instance_id: str,
    record: dict[str, Any],
    source_store: Path,
    repo_root: Path,
    model: str,
    dimension: int,
    required_indexes: list[str],
) -> tuple[Path, dict[str, Any]]:
    source = source_store / directory_name(instance_id)
    index_dir = source / "index_output"
    metadata_path = source / "instance_meta.json"
    embedding_path = index_dir / "embedding_manifest.json"
    hashes_path = index_dir / "file_hashes.json"
    for required in (source, index_dir, metadata_path, embedding_path, hashes_path):
        if not required.exists():
            raise FileNotFoundError(f"missing runtime artifact for {instance_id}: {required}")

    embedding = read_object(embedding_path)
    entries = embedding.get("indexes")
    if not isinstance(entries, dict):
        raise ValueError(f"embedding manifest has no indexes: {embedding_path}")
    for name in required_indexes:
        entry = entries.get(name)
        actual = (
            entry.get("model"),
            entry.get("dimension"),
            entry.get("metric"),
            entry.get("normalized"),
        ) if isinstance(entry, dict) else None
        expected = (model, dimension, "inner_product", True)
        if actual != expected:
            raise ValueError(
                f"embedding mismatch for {instance_id}/{name}: expected {expected}, found {actual}"
            )
        for artifact_name in INDEX_ARTIFACTS.get(name, ()):
            artifact = index_dir / artifact_name
            if not artifact.is_file():
                raise FileNotFoundError(
                    f"embedding artifact missing for {instance_id}/{name}: {artifact}"
                )

    repo = repo_root / instance_id
    base_commit = str(record["base_commit"])
    actual_commit = git_head(repo)
    if actual_commit != base_commit:
        raise ValueError(
            f"base commit mismatch for {instance_id}: expected {base_commit}, found {actual_commit}"
        )

    metadata = read_object(metadata_path)
    source_root_raw = metadata.get("source_root")
    if not isinstance(source_root_raw, str) or not source_root_raw:
        raise ValueError(f"missing source_root: {metadata_path}")
    source_root = Path(source_root_raw)
    file_hashes = read_object(hashes_path).get("files")
    if not isinstance(file_hashes, dict) or not file_hashes:
        raise ValueError(f"empty file hash manifest: {hashes_path}")
    for indexed_name, expected_hash in file_hashes.items():
        indexed_path = Path(indexed_name)
        relative = indexed_path.relative_to(source_root) if indexed_path.is_absolute() else indexed_path
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe indexed path for {instance_id}: {indexed_name}")
        active_path = repo / relative
        if not active_path.is_file():
            raise FileNotFoundError(f"indexed source missing for {instance_id}: {relative}")
        if sha256(active_path) != expected_hash:
            raise ValueError(f"indexed source mismatch for {instance_id}: {relative}")

    return source, {
        "base_commit": base_commit,
        "directory": directory_name(instance_id),
        "source_store": str(source_store.resolve()),
        "source_directory": str(source.resolve()),
        "embedding_manifest_sha256": sha256(embedding_path),
        "file_hashes_sha256": sha256(hashes_path),
        "instance_meta_sha256": sha256(metadata_path),
        "indexed_file_count": len(file_hashes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-store", type=Path, action="append", required=True)
    parser.add_argument("--instances-file", type=Path, action="append", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--embedding-model", default="voyage-2")
    parser.add_argument("--embedding-dimension", type=int, default=1024)
    args = parser.parse_args()
    if len(args.source_store) != len(args.instances_file):
        parser.error("provide one --instances-file for every --source-store, in matching order")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")

    selected: dict[str, tuple[dict[str, Any], Path]] = {}
    dataset_hashes = []
    for source_store, instances_file in zip(args.source_store, args.instances_file):
        source_store = source_store.resolve()
        instances_file = instances_file.resolve()
        dataset_hashes.append({"path": str(instances_file), "sha256": sha256(instances_file)})
        for instance_id, record in load_instances(instances_file).items():
            if instance_id in selected:
                raise ValueError(f"instance appears in more than one source store: {instance_id}")
            selected[instance_id] = (record, source_store)
    if len(selected) != args.expected_count:
        raise ValueError(f"expected {args.expected_count} instances, found {len(selected)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "strict_runtime_preflight": True,
        "instance_count": len(selected),
        "expected_embedding": {
            "model": args.embedding_model,
            "dimension": args.embedding_dimension,
            "metric": "inner_product",
            "normalized": True,
            "required_indexes": ["functions", "features", "tests"],
        },
        "datasets": dataset_hashes,
        "instances": {},
    }
    try:
        for instance_id in sorted(selected):
            record, source_store = selected[instance_id]
            source, audit = validate_instance(
                instance_id,
                record,
                source_store,
                args.repo_root.resolve(),
                args.embedding_model,
                args.embedding_dimension,
                manifest["expected_embedding"]["required_indexes"],
            )
            (temporary / directory_name(instance_id)).symlink_to(source.resolve(), target_is_directory=True)
            manifest["instances"][instance_id] = audit
        manifest_path = temporary / "runtime_store_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "instances": len(selected),
                "indexed_files": sum(
                    item["indexed_file_count"] for item in manifest["instances"].values()
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
