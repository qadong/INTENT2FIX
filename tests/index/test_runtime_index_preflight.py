import hashlib
import json
import subprocess

import pytest

from INTENT2FIX.index import runtime_preflight
from INTENT2FIX.index.runtime_preflight import (
    RuntimeIndexPreflightError,
    validate_runtime_index,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def _runtime_fixture(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    source = repo / "src.cpp"
    source.write_text("int answer() { return 42; }\n", encoding="utf-8")
    _git(repo, "add", "src.cpp")
    _git(repo, "commit", "-qm", "base")
    commit = _git(repo, "rev-parse", "HEAD")

    root = tmp_path / "store"
    instance = root / "swe-bench_org__repo-1"
    index = instance / "index_output"
    index.mkdir(parents=True)
    metadata = instance / "instance_meta.json"
    metadata.write_text(json.dumps({"source_root": str(repo)}), encoding="utf-8")
    hashes = index / "file_hashes.json"
    hashes.write_text(
        json.dumps({"schema_version": 9, "files": {str(source): _sha256(source)}}),
        encoding="utf-8",
    )
    embedding = index / "embedding_manifest.json"
    embedding.write_text(
        json.dumps({
            "indexes": {
                name: {
                    "model": "voyage-2",
                    "dimension": 1024,
                    "metric": "inner_product",
                    "normalized": True,
                }
                for name in ("functions", "features", "tests")
            }
        }),
        encoding="utf-8",
    )
    for artifact in (
        "func_index.faiss", "func_meta.jsonl",
        "feature_index.faiss", "feature_meta.jsonl",
        "test_index.faiss", "test_meta.jsonl",
    ):
        (index / artifact).write_text("fixture", encoding="utf-8")
    manifest = root / "runtime_store_manifest.json"
    manifest.write_text(
        json.dumps({
            "strict_runtime_preflight": True,
            "expected_embedding": {
                "model": "voyage-2",
                "dimension": 1024,
                "metric": "inner_product",
                "normalized": True,
                "required_indexes": ["functions", "features", "tests"],
            },
            "instances": {
                "org__repo-1": {
                    "directory": instance.name,
                    "base_commit": commit,
                    "embedding_manifest_sha256": _sha256(embedding),
                    "file_hashes_sha256": _sha256(hashes),
                    "instance_meta_sha256": _sha256(metadata),
                }
            },
        }),
        encoding="utf-8",
    )
    return root, repo, source, manifest


@pytest.fixture(autouse=True)
def _clear_preflight_cache(monkeypatch):
    runtime_preflight._VALIDATION_CACHE.clear()
    monkeypatch.setenv("VOYAGE_EMBED_MODEL", "voyage-2")
    monkeypatch.setenv("CODE_INTENT_EMBED_DIM", "1024")


def test_preflight_accepts_matching_commit_embeddings_and_sources(tmp_path):
    root, repo, _, _ = _runtime_fixture(tmp_path)

    report = validate_runtime_index("org__repo-1", repo, root)

    assert report["status"] == "validated"
    assert report["indexed_file_count"] == 1


def test_preflight_rejects_wrong_base_commit(tmp_path):
    root, repo, _, manifest_path = _runtime_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["instances"]["org__repo-1"]["base_commit"] = "0" * 40
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeIndexPreflightError, match="active checkout mismatch"):
        validate_runtime_index("org__repo-1", repo, root)


def test_preflight_rejects_source_drift(tmp_path):
    root, repo, source, _ = _runtime_fixture(tmp_path)
    source.write_text("int answer() { return 0; }\n", encoding="utf-8")

    with pytest.raises(RuntimeIndexPreflightError, match="indexed source differs"):
        validate_runtime_index("org__repo-1", repo, root)


def test_forced_preflight_rejects_unmanifested_root(tmp_path, monkeypatch):
    monkeypatch.setenv("CODE_INTENT_INDEX_STRICT_PREFLIGHT", "1")

    with pytest.raises(RuntimeIndexPreflightError, match="requires"):
        validate_runtime_index("org__repo-1", tmp_path, tmp_path / "legacy")
