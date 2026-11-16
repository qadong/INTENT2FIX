import hashlib
import json

import pytest
from pydantic import ValidationError

from INTENT2FIX.scout_probe_handoff import (
    build_probe_selection_prompt, probe_selection_schema,
)


def test_selection_prompt_preserves_all_records_and_exact_source_versions():
    records = [
        {"evidence_id": "RUN-1", "command": "c++ probe.cpp -o /tmp/p && /tmp/p",
         "exit_code": 1, "evidence_role": "premise",
         "frozen_probe_files": {"probe.cpp": "assert(original_input);\n"}},
        {"evidence_id": "RUN-2", "command": "grep predicate header.hpp",
         "exit_code": 0, "evidence_role": "reproduction",
         "frozen_probe_files": {"probe.cpp": "assert(reduced_input);\n"}},
        {"evidence_id": "RUN-3", "command": "c++ probe.cpp -o /tmp/p && /tmp/p",
         "exit_code": 0, "frozen_probe_files": {"probe.cpp": "assert(reduced_input);\n"}},
    ]
    original = json.dumps(records)
    issue = "Full original Issue " + "public-operation " * 1000
    prompt = build_probe_selection_prompt(issue, records, {"RUN-1": "full output\n" * 1000})
    payload = json.loads(prompt.split("\n", 1)[1])
    assert payload["original_issue"] == issue
    assert [item["evidence_id"] for item in payload["executions"]] == ["RUN-1", "RUN-2", "RUN-3"]
    assert payload["executions"][0]["observation"] == "full output\n" * 1000
    assert len(payload["frozen_source_versions"]) == 2
    for record, execution in zip(records, payload["executions"]):
        assert execution["command"] == record["command"]
        for path, content in record["frozen_probe_files"].items():
            version = execution["file_versions"][path]
            assert version == hashlib.sha256(content.encode()).hexdigest()
            assert payload["frozen_source_versions"][version] == content
    assert json.dumps(records) == original


@pytest.mark.parametrize("selected", ["RUN-1", "RUN-2", None])
def test_schema_allows_only_existing_execution_or_explicit_none(selected):
    schema = probe_selection_schema(["RUN-1", "RUN-2"])
    decision = schema.model_validate({"selected_evidence_id": selected, "reason": "Relevant diagnostic."})
    assert decision.selected_evidence_id == selected


@pytest.mark.parametrize("payload", [
    {"selected_evidence_id": "RUN-99", "reason": "Invented"},
    {"selected_evidence_id": "RUN-1", "reason": "Rewrite", "command": "new command"},
    {"selected_evidence_id": "RUN-1", "reason": "Rewrite", "files": {"probe.cpp": "new source"}},
])
def test_schema_rejects_invented_record_or_rewritten_artifacts(payload):
    with pytest.raises(ValidationError):
        probe_selection_schema(["RUN-1", "RUN-2"]).model_validate(payload)
