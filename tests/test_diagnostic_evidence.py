from copy import deepcopy
import json

import pytest

from INTENT2FIX.diagnostic_evidence import original_diagnostics, summarize_diagnostics


def record(**changes):
    return {"id": "RUN-1", "path": "probe.cpp", "locator": "c++ -std=c++20 probe.cpp",
            "observation": "error: document has no member at_pointer", "exit_code": 1, **changes}


def summary(prompt, schema):
    return schema(conditions="C++20 document call", finding="Compilation failed: missing member",
                  limits="Runtime behavior not tested", evidence_quotes=["no member at_pointer"])


def test_restore_requires_same_version_command_prefix_and_length(tmp_path):
    full = record()
    handoff = record(observation=full["observation"][:10], observation_truncated=True,
                     original_chars=len(full["observation"]))
    (tmp_path / "intent_scout_checkpoint.json").write_text(json.dumps({"base_commit": "base"}))
    raw_path = tmp_path / "intent_grounding_result.json"
    raw_path.write_text(json.dumps({"investigation": {"repository_evidence": [full]}}))
    restored = original_diagnostics([handoff], tmp_path, "base")["RUN-1"]
    assert restored["observation"] == full["observation"]
    assert not restored["observation_truncated"] and restored["exit_code"] == 1
    assert handoff["observation_truncated"]
    with pytest.raises(ValueError, match="different base"):
        original_diagnostics([handoff], tmp_path, "other")
    for changes in ({"locator": "other command"}, {"observation": "other text"}, {"path": "other.cpp"}):
        raw_path.write_text(json.dumps({"investigation": {"repository_evidence": [{**full, **changes}]}}))
        with pytest.raises(ValueError, match="does not match"):
            original_diagnostics([handoff], tmp_path, "base")
    assert original_diagnostics([handoff], None, "base")["RUN-1"] == handoff


def test_summary_cache_is_shared_but_never_reused_for_different_evidence(tmp_path):
    calls = []
    def summarize(prompt, schema):
        calls.append(prompt)
        return summary(prompt, schema)
    originals = {"RUN-1": record()}
    before = deepcopy(originals)
    first, _ = summarize_diagnostics(originals, tmp_path, summarize, "model-A")
    again, _ = summarize_diagnostics(originals, tmp_path, summarize, "model-A")
    assert first == again and len(calls) == 1
    assert originals == before and first["RUN-1"]["exit_code"] == 1
    assert first["RUN-1"]["summary_status"] == "model"
    summarize_diagnostics(originals, tmp_path, summarize, "model-B")
    for changes in ({"locator": "c++ -std=c++17 probe.cpp"}, {"observation": "error: no member at_pointer\nextra"}):
        summarize_diagnostics({"RUN-1": record(**changes)}, tmp_path, summarize, "model-A")
    assert len(calls) == 4


@pytest.mark.parametrize("failure", ["invented_quote", "provider_error", "empty_summary"])
def test_failed_summary_keeps_honest_preview_and_original(tmp_path, failure):
    raw = record(observation="前缀" + "a" * 12000 + "actual error", observation_truncated=True)
    def invalid(prompt, schema):
        if failure == "provider_error":
            raise RuntimeError("Unavailable")
        return {"conditions": "compile", "finding": "" if failure == "empty_summary" else "Failed",
                "limits": "unknown", "evidence_quotes": ["invented error"]}
    cards, _ = summarize_diagnostics({"RUN-1": raw}, tmp_path, invalid, "test")
    card = cards["RUN-1"]
    assert card["summary_status"] == "fallback" and card["original_truncated"]
    assert card["observation"].endswith("actual error") and len(card["observation"]) < 600
    def no_call(*args):
        raise AssertionError("Fallback should also be cached")
    assert summarize_diagnostics({"RUN-1": raw}, tmp_path, no_call, "test")[0] == cards


def test_exit_zero_and_short_output_are_preserved(tmp_path):
    raw = record(observation="result=true", exit_code=0)
    def summarize(prompt, schema):
        return schema(conditions="One probe", finding="result=true", limits="No other inputs tested",
                      evidence_quotes=["result=true"])
    cards, _ = summarize_diagnostics({"RUN-1": raw}, tmp_path, summarize, "test")
    assert cards["RUN-1"]["exit_code"] == 0
    assert json.loads(cards["RUN-1"]["observation"])["finding"] == "result=true"


def test_saved_omission_marker_is_not_presented_as_complete_output():
    raw = record(observation="first\n... [21 lines omitted] ...\nlast")
    restored = original_diagnostics([raw], None, "base")["RUN-1"]
    assert restored["observation"] == raw["observation"]
    assert restored["observation_truncated"]
    assert restored["truncation_stage"] == "upstream_saved_output"
    assert "observation_truncated" not in raw
