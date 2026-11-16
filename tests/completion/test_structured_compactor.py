import json

import pytest
from pydantic import BaseModel, ConfigDict, Field

from INTENT2FIX.completion.structured_compactor import (
    StructuredCompactionError,
    compact_structured_payload_once,
)


class CompactReport(BaseModel):
    status: str
    summary: str = Field(max_length=20)
    facts: list[str] = Field(max_length=2)

    model_config = ConfigDict(extra="forbid")


class CompactActionReport(BaseModel):
    status: str
    summary: str = Field(max_length=20)
    facts: list[str] = Field(max_length=2)
    thoughts: str = ""


def _source() -> dict:
    return {
        "status": "partial",
        "summary": "A distinct but unnecessarily verbose supported conclusion",
        "facts": ["SEARCH-1", "SEARCH-2", "SEARCH-3"],
        "note": "redundant metadata",
    }


def test_compactor_preserves_ids_and_locked_fields():
    compacted = {
        "status": "partial",
        "summary": "Supported boundary",
        "facts": ["SEARCH-1 SEARCH-2", "SEARCH-3"],
    }

    result, audit = compact_structured_payload_once(
        _source(),
        CompactReport,
        lambda prompt: json.dumps(compacted),
    )

    assert result == CompactReport(**compacted)
    assert audit.reference_ids == ("SEARCH-1", "SEARCH-2", "SEARCH-3")
    assert set(audit.mutable_paths) == {"summary", "facts", "note"}


def test_compactor_rejects_changed_evidence_ids():
    compacted = {
        "status": "partial",
        "summary": "Supported boundary",
        "facts": ["SEARCH-1 SEARCH-2", "SEARCH-4"],
    }

    with pytest.raises(StructuredCompactionError, match="reference IDs"):
        compact_structured_payload_once(
            _source(),
            CompactReport,
            lambda prompt: json.dumps(compacted),
        )


def test_compactor_rejects_changes_outside_failed_paths():
    compacted = {
        "status": "supported",
        "summary": "Supported boundary",
        "facts": ["SEARCH-1 SEARCH-2", "SEARCH-3"],
    }

    with pytest.raises(StructuredCompactionError, match="outside"):
        compact_structured_payload_once(
            _source(),
            CompactReport,
            lambda prompt: json.dumps(compacted),
        )


def test_compactor_does_not_guess_missing_required_fields():
    source = {"facts": []}

    with pytest.raises(StructuredCompactionError, match="non-compaction"):
        compact_structured_payload_once(
            source,
            CompactReport,
            lambda prompt: json.dumps({}),
        )


def test_compactor_separates_transport_thoughts_from_semantic_payload():
    source = {
        "status": "partial",
        "summary": "A distinct but unnecessarily verbose conclusion",
        "facts": ["SEARCH-1", "RUN-2", "TEST-CAND-3"],
        "thoughts": "private reasoning mentioning unrelated SEARCH-99",
    }
    compacted = {
        "status": "partial",
        "summary": "Supported boundary",
        "facts": ["SEARCH-1 RUN-2", "TEST-CAND-3"],
    }

    result, audit = compact_structured_payload_once(
        source,
        CompactActionReport,
        lambda prompt: json.dumps(compacted),
    )

    assert result.thoughts == ""
    assert audit.reference_ids == ("RUN-2", "SEARCH-1", "TEST-CAND-3")


def test_compactor_prompt_exposes_path_specific_hard_limits():
    prompts = []
    compacted = {
        "status": "partial",
        "summary": "Supported boundary",
        "facts": ["SEARCH-1 SEARCH-2", "SEARCH-3"],
    }

    compact_structured_payload_once(
        _source(),
        CompactReport,
        lambda prompt: prompts.append(prompt) or json.dumps(compacted),
    )

    assert "`summary`: hard maximum 20 characters" in prompts[0]
    assert "target at most 10 characters" in prompts[0]
    assert "`facts`: hard maximum 2 list items" in prompts[0]


def test_compactor_merges_only_returned_field_patch():
    patch = {
        "replacements": [
            {"path": ["summary"], "value": "Supported boundary"},
            {
                "path": ["facts"],
                "value": ["SEARCH-1 SEARCH-2", "SEARCH-3"],
            },
        ],
        "remove_paths": [["note"]],
    }

    result, audit = compact_structured_payload_once(
        _source(),
        CompactReport,
        lambda prompt: json.dumps(patch),
    )

    assert result.status == "partial"
    assert result.summary == "Supported boundary"
    assert result.facts == ["SEARCH-1 SEARCH-2", "SEARCH-3"]
    assert set(audit.mutable_paths) == {"summary", "facts", "note"}


def test_compactor_ignores_provider_metadata_beside_allowlisted_patch():
    patch = {
        "replacements": [
            {"path": ["summary"], "value": "Supported boundary"},
            {
                "path": ["facts"],
                "value": ["SEARCH-1 SEARCH-2", "SEARCH-3"],
            },
        ],
        "remove_paths": [["note"]],
        "explanation": "Condensed the rejected fields only.",
    }

    result, _ = compact_structured_payload_once(
        _source(),
        CompactReport,
        lambda prompt: json.dumps(patch),
    )

    assert result.status == "partial"
    assert result.summary == "Supported boundary"
    assert result.facts == ["SEARCH-1 SEARCH-2", "SEARCH-3"]


def test_string_list_overflow_has_lossless_fallback_when_llm_patch_is_malformed():
    source = {
        "status": "partial",
        "summary": "Short summary",
        "facts": ["first SEARCH-1", "second SEARCH-2", "third SEARCH-3"],
    }
    malformed_patch = {"replacements": [], "remove_paths": []}

    result, audit = compact_structured_payload_once(
        source,
        CompactReport,
        lambda prompt: json.dumps(malformed_patch),
    )

    assert result.facts == [
        "first SEARCH-1",
        "second SEARCH-2\nthird SEARCH-3",
    ]
    assert audit.reference_ids == ("SEARCH-1", "SEARCH-2", "SEARCH-3")
    assert audit.strategy == "deterministic_string_list_merge"
