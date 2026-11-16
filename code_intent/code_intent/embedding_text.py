from __future__ import annotations

from typing import Any, Dict


def build_feature_embedding_text(feature: Dict[str, Any]) -> str:
    return "\n".join(
        [
            feature["feature_name"],
            f"Feature type: {feature.get('feature_type', '')}",
            f"Summary role: {feature.get('summary_role', '')}",
            feature.get("feature_description", ""),
            " ".join(feature.get("tags", [])),
        ]
    ).strip()
