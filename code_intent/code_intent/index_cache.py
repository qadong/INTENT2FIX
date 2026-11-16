import json
from pathlib import Path


CODE_INDEX_CACHE_SCHEMA_VERSION = 10


def load_file_hash_cache(path: Path) -> dict[str, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != CODE_INDEX_CACHE_SCHEMA_VERSION
        or not isinstance(payload.get("files"), dict)
    ):
        return {}
    return payload["files"]


def file_hash_cache_payload(files: dict[str, str]) -> dict:
    return {
        "schema_version": CODE_INDEX_CACHE_SCHEMA_VERSION,
        "files": files,
    }


def write_file_hash_cache(path: Path, files: dict[str, str]) -> None:
    """Atomically checkpoint hashes so completed per-file caches survive interruption."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(file_hash_cache_payload(files), indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
