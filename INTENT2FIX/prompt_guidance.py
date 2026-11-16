"""Optional stage guidance, loaded at prompt construction for ablation."""

import os
from pathlib import Path


def load_stage_guidance(stage: str) -> str:
    if stage not in {"repair", "hypothesis"}:
        raise ValueError(f"Unknown guidance stage: {stage}")
    if os.environ.get(f"COTEST_{stage.upper()}_EVIDENCE_GUIDANCE", "1") == "0":
        return ""
    return (Path(__file__).with_name("prompts") / f"{stage}_evidence.md").read_text(
        encoding="utf-8"
    ).strip()
