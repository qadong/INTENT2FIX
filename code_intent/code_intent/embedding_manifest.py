from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


MANIFEST_NAME = "embedding_manifest.json"
FUNCTION_INTENT_ARTIFACT_VERSION = 4
FUNCTION_INTENT_BUILDER = f"function-intent-v{FUNCTION_INTENT_ARTIFACT_VERSION}"
TEST_INTENT_ARTIFACT_VERSION = 5
TEST_INTENT_BUILDER = f"test-intent-v{TEST_INTENT_ARTIFACT_VERSION}"
FEATURE_INTENT_ARTIFACT_VERSION = 5
FEATURE_INTENT_BUILDER = f"feature-intent-v{FEATURE_INTENT_ARTIFACT_VERSION}"


def load_embedding_manifest(index_dir: str | Path) -> Dict[str, Any]:
    path = Path(index_dir) / MANIFEST_NAME
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def write_embedding_manifest_entry(
    index_dir: str | Path,
    index_name: str,
    *,
    model: Optional[str],
    dimension: int,
    metric: str = "inner_product",
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    base = Path(index_dir)
    base.mkdir(parents=True, exist_ok=True)
    path = base / MANIFEST_NAME
    manifest = load_embedding_manifest(base)
    manifest["version"] = 1
    indexes = manifest.setdefault("indexes", {})
    indexes[index_name] = {
        "provider": "voyage" if model and model.startswith("voyage") else "unknown",
        "model": model,
        "dimension": dimension,
        "metric": metric,
        "normalized": True,
    }
    if metadata:
        indexes[index_name].update(metadata)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)
