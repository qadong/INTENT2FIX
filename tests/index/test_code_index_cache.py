import json

from code_intent.code_intent.index_cache import (
    CODE_INDEX_CACHE_SCHEMA_VERSION,
    load_file_hash_cache,
    write_file_hash_cache,
)


def test_code_index_cache_rejects_legacy_schema(tmp_path):
    cache = tmp_path / "file_hashes.json"
    cache.write_text(json.dumps({"source.cpp": "old-hash"}), encoding="utf-8")

    assert load_file_hash_cache(cache) == {}


def test_code_index_cache_loads_current_schema(tmp_path):
    cache = tmp_path / "file_hashes.json"
    cache.write_text(json.dumps({
        "schema_version": CODE_INDEX_CACHE_SCHEMA_VERSION,
        "files": {"source.cpp": "current-hash"},
    }), encoding="utf-8")

    assert load_file_hash_cache(cache) == {"source.cpp": "current-hash"}


def test_code_index_cache_checkpoint_is_loadable_and_atomic(tmp_path):
    # Keep the public helper covered because index_repo calls it before worker
    # completion to make per-file caches interruption-safe.
    cache = tmp_path / "file_hashes.json"
    write_file_hash_cache(cache, {"source.cpp": "current-hash"})

    assert load_file_hash_cache(cache) == {"source.cpp": "current-hash"}
    assert not (tmp_path / ".file_hashes.json.tmp").exists()
