"""Fail-closed validation for a canonical code-intent runtime store."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Any


RUNTIME_MANIFEST_NAME = "runtime_store_manifest.json"
_INDEX_ARTIFACTS = {
    "functions": ("func_index.faiss", "func_meta.jsonl"),
    "features": ("feature_index.faiss", "feature_meta.jsonl"),
    "tests": ("test_index.faiss", "test_meta.jsonl"),
}


class RuntimeIndexPreflightError(RuntimeError):
    """The selected runtime index cannot be trusted for the active checkout."""


_VALIDATION_CACHE: dict[tuple[str, str, str], dict[str, Any]] = {}
_VALIDATION_LOCK = threading.Lock()


def normalize_instance_name(instance_name: str) -> str:
    prefix = "swe-bench_"
    return instance_name[len(prefix):] if instance_name.startswith(prefix) else instance_name


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise RuntimeIndexPreflightError(f"required runtime artifact is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeIndexPreflightError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeIndexPreflightError(f"invalid {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeIndexPreflightError(f"invalid {label}: expected a JSON object: {path}")
    return value


def _git_head(repo_dir: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeIndexPreflightError(
            f"cannot read active repository commit: {repo_dir}: {exc}"
        ) from exc


def _resolve_instance_dir(root: Path, instance_name: str, record: dict[str, Any]) -> Path:
    directory = record.get("directory")
    if not isinstance(directory, str) or not directory:
        raise RuntimeIndexPreflightError(
            f"runtime manifest has no directory for {normalize_instance_name(instance_name)}"
        )
    instance_dir = root / directory
    if not instance_dir.is_dir():
        raise RuntimeIndexPreflightError(
            f"runtime index directory is missing for {instance_name}: {instance_dir}"
        )
    return instance_dir


def _validate_embedding_manifest(
    index_dir: Path,
    record: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    manifest_path = index_dir / "embedding_manifest.json"
    recorded_sha = record.get("embedding_manifest_sha256")
    if not isinstance(recorded_sha, str) or _sha256(manifest_path) != recorded_sha:
        raise RuntimeIndexPreflightError(
            f"embedding manifest changed or is unrecorded: {manifest_path}"
        )
    manifest = _read_object(manifest_path, "embedding manifest")
    entries = manifest.get("indexes")
    if not isinstance(entries, dict):
        raise RuntimeIndexPreflightError(
            f"embedding manifest has no index provenance: {manifest_path}"
        )

    required = expected.get("required_indexes")
    if not isinstance(required, list) or not required:
        raise RuntimeIndexPreflightError("runtime manifest has no required embedding indexes")
    model = expected.get("model")
    dimension = expected.get("dimension")
    metric = expected.get("metric", "inner_product")
    for name in required:
        entry = entries.get(name)
        if not isinstance(entry, dict):
            raise RuntimeIndexPreflightError(
                f"embedding provenance for {name!r} is missing: {manifest_path}"
            )
        actual = (
            entry.get("model"),
            entry.get("dimension"),
            entry.get("metric"),
            entry.get("normalized"),
        )
        wanted = (model, dimension, metric, True)
        if actual != wanted:
            raise RuntimeIndexPreflightError(
                f"embedding provenance mismatch for {name!r}: "
                f"expected {wanted}, found {actual}"
            )
        for artifact_name in _INDEX_ARTIFACTS.get(name, ()):
            if not (index_dir / artifact_name).is_file():
                raise RuntimeIndexPreflightError(
                    f"embedding artifact for {name!r} is missing: {index_dir / artifact_name}"
                )

    runtime_model = (os.environ.get("VOYAGE_EMBED_MODEL") or "").strip()
    runtime_dimension_raw = (os.environ.get("CODE_INTENT_EMBED_DIM") or "").strip()
    try:
        runtime_dimension = int(runtime_dimension_raw)
    except ValueError as exc:
        raise RuntimeIndexPreflightError(
            f"invalid CODE_INTENT_EMBED_DIM={runtime_dimension_raw!r}"
        ) from exc
    if (runtime_model, runtime_dimension) != (model, dimension):
        raise RuntimeIndexPreflightError(
            "runtime embedding configuration does not match the canonical index: "
            f"runtime={(runtime_model, runtime_dimension)}, index={(model, dimension)}"
        )


def _validate_file_hashes(
    instance_dir: Path,
    repo_dir: Path,
    record: dict[str, Any],
) -> int:
    index_dir = instance_dir / "index_output"
    hashes_path = index_dir / "file_hashes.json"
    recorded_sha = record.get("file_hashes_sha256")
    if not isinstance(recorded_sha, str) or _sha256(hashes_path) != recorded_sha:
        raise RuntimeIndexPreflightError(
            f"file hash manifest changed or is unrecorded: {hashes_path}"
        )
    hashes = _read_object(hashes_path, "file hash manifest").get("files")
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeIndexPreflightError(f"file hash manifest is empty: {hashes_path}")

    meta_path = instance_dir / "instance_meta.json"
    recorded_meta_sha = record.get("instance_meta_sha256")
    if not isinstance(recorded_meta_sha, str) or _sha256(meta_path) != recorded_meta_sha:
        raise RuntimeIndexPreflightError(
            f"instance metadata changed or is unrecorded: {meta_path}"
        )
    metadata = _read_object(meta_path, "instance metadata")
    source_root_raw = metadata.get("source_root")
    if not isinstance(source_root_raw, str) or not source_root_raw:
        raise RuntimeIndexPreflightError(f"instance metadata has no source_root: {meta_path}")
    source_root = Path(source_root_raw)
    repo_root = repo_dir.resolve()

    for indexed_name, expected_hash in hashes.items():
        if not isinstance(indexed_name, str) or not isinstance(expected_hash, str):
            raise RuntimeIndexPreflightError(f"invalid file hash entry in {hashes_path}")
        indexed_path = Path(indexed_name)
        try:
            relative = indexed_path.relative_to(source_root) if indexed_path.is_absolute() else indexed_path
        except ValueError as exc:
            raise RuntimeIndexPreflightError(
                f"indexed file is outside recorded source_root: {indexed_path}"
            ) from exc
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeIndexPreflightError(f"unsafe indexed path: {indexed_path}")
        active_path = repo_root / relative
        if not active_path.is_file():
            raise RuntimeIndexPreflightError(
                f"indexed source is missing from active checkout: {relative}"
            )
        if _sha256(active_path) != expected_hash:
            raise RuntimeIndexPreflightError(
                f"indexed source differs from active checkout: {relative}"
            )
    return len(hashes)


def validate_runtime_index(
    instance_name: str,
    repo_dir: str | Path | None,
    index_store_dir: str | Path,
) -> dict[str, Any] | None:
    """Validate a canonical instance once per process before graph tools run.

    Legacy roots without a runtime manifest remain usable for structural-only
    development unless ``CODE_INTENT_INDEX_STRICT_PREFLIGHT=1`` is set. A
    canonical root always opts into strict validation through its manifest.
    """

    root = Path(index_store_dir).expanduser().resolve()
    manifest_path = root / RUNTIME_MANIFEST_NAME
    force_strict = (os.environ.get("CODE_INTENT_INDEX_STRICT_PREFLIGHT") or "").lower() in {
        "1", "true", "yes",
    }
    if not manifest_path.is_file():
        if force_strict:
            raise RuntimeIndexPreflightError(
                f"strict index preflight requires {manifest_path}"
            )
        return None

    manifest = _read_object(manifest_path, "runtime store manifest")
    if manifest.get("strict_runtime_preflight") is not True:
        if force_strict:
            raise RuntimeIndexPreflightError(
                f"runtime store is not marked strict: {manifest_path}"
            )
        return None
    if repo_dir is None:
        raise RuntimeIndexPreflightError(
            f"strict index preflight requires an active repository for {instance_name}"
        )

    normalized = normalize_instance_name(instance_name)
    instances = manifest.get("instances")
    record = instances.get(normalized) if isinstance(instances, dict) else None
    if not isinstance(record, dict):
        raise RuntimeIndexPreflightError(
            f"instance {normalized!r} is not present in canonical runtime store {root}"
        )

    repo_path = Path(repo_dir).expanduser().resolve()
    cache_key = (str(root), normalized, str(repo_path))
    with _VALIDATION_LOCK:
        cached = _VALIDATION_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    instance_dir = _resolve_instance_dir(root, instance_name, record)
    expected_commit = record.get("base_commit")
    if not isinstance(expected_commit, str) or not expected_commit:
        raise RuntimeIndexPreflightError(
            f"runtime manifest has no base_commit for {normalized}"
        )
    actual_commit = _git_head(repo_path)
    if actual_commit != expected_commit:
        raise RuntimeIndexPreflightError(
            f"active checkout mismatch for {normalized}: "
            f"expected {expected_commit}, found {actual_commit}"
        )

    expected_embedding = manifest.get("expected_embedding")
    if not isinstance(expected_embedding, dict):
        raise RuntimeIndexPreflightError("runtime manifest has no expected_embedding object")
    _validate_embedding_manifest(instance_dir / "index_output", record, expected_embedding)
    file_count = _validate_file_hashes(instance_dir, repo_path, record)

    report = {
        "instance": normalized,
        "base_commit": actual_commit,
        "indexed_file_count": file_count,
        "index_store_dir": str(root),
        "status": "validated",
    }
    with _VALIDATION_LOCK:
        _VALIDATION_CACHE[cache_key] = report
    return dict(report)
