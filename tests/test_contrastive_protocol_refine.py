import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from INTENT2FIX.intent_dialogue.schema import ContrastiveIntentDecision


SPEC = importlib.util.spec_from_file_location(
    "contrastive_refine_runner",
    Path(__file__).parents[1] / "scripts/run_no_memory_cotest.py",
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def invalid_witness_refinement() -> dict:
    return {
        "witness_valid": False,
        "next_action": "refine_test_pair",
        "followup_request": {
            "objective": "Find a different local witness.",
            "why_current_evidence_is_insufficient": "The current pair is invalid.",
            "evidence_needed": "A controlled public observation.",
        },
        "witnessed_delta": "",
        "reason": "The current pair does not provide a valid witness.",
    }


def validation_error(payload: dict) -> ValidationError:
    with pytest.raises(ValidationError) as caught:
        ContrastiveIntentDecision.model_validate(payload)
    return caught.value


def test_light_refine_repairs_invalid_witness_protocol_without_new_semantics():
    payload = invalid_witness_refinement()
    prompts = []

    def complete(prompt: str) -> str:
        prompts.append(prompt)
        return json.dumps({
            "next_action": "reject_witness",
            "drop_repository_effect_indexes": [],
        })

    result = RUNNER._refine_contrastive_protocol_once(
        json.dumps(payload),
        validation_error(payload),
        complete,
    )

    assert result.next_action == "reject_witness"
    assert result.witness_valid is False
    assert result.followup_request is None
    assert result.reason == payload["reason"]
    assert len(prompts) == 1


def test_light_refine_drops_graph_contract_effect_without_relabeling_it():
    payload = {
        "witness_valid": False,
        "next_action": "leave_open",
        "witnessed_delta": "",
        "repository_effects": [{
            "kind": "open_decision",
            "statement": "The graph suggests another related path.",
            "evidence": [{"evidence_id": "GRAPH-1", "quote": "related path"}],
        }],
        "reason": "No local witness was established.",
    }
    result = RUNNER._refine_contrastive_protocol_once(
        json.dumps(payload),
        validation_error(payload),
        lambda _prompt: json.dumps({
            "next_action": None,
            "drop_repository_effect_indexes": [0],
        }),
    )

    assert result.next_action == "leave_open"
    assert result.repository_effects == []
    assert result.reason == payload["reason"]


def test_light_refine_ignores_unrequired_semantic_changes():
    payload = {
        "witness_valid": False,
        "next_action": "leave_open",
        "witnessed_delta": "",
        "repository_effects": [{
            "kind": "open_decision",
            "statement": "The graph suggests another related path.",
            "evidence": [{"evidence_id": "GRAPH-1", "quote": "related path"}],
        }],
        "reason": "No local witness was established.",
    }
    result = RUNNER._refine_contrastive_protocol_once(
        json.dumps(payload),
        validation_error(payload),
        lambda _prompt: json.dumps({
            "next_action": "reject_witness",
            "drop_repository_effect_indexes": [0],
        }),
    )

    assert result.next_action == "leave_open"
    assert result.repository_effects == []


def test_light_refine_uses_safe_protocol_closure_for_malformed_patch():
    payload = invalid_witness_refinement()

    result = RUNNER._refine_contrastive_protocol_once(
        json.dumps(payload),
        validation_error(payload),
        lambda _prompt: "not json",
    )

    assert result.next_action == "leave_open"
    assert result.followup_request is None
    assert result.witness_valid is False


def test_light_refine_closes_latent_invalid_witness_after_effect_drop():
    payload = invalid_witness_refinement()
    payload["repository_effects"] = [{
        "kind": "open_decision",
        "statement": "The graph suggests another related path.",
        "evidence": [{"evidence_id": "GRAPH-1", "quote": "related path"}],
    }]

    result = RUNNER._refine_contrastive_protocol_once(
        json.dumps(payload),
        validation_error(payload),
        lambda _prompt: json.dumps({"next_action": None}),
    )

    assert result.next_action == "leave_open"
    assert result.followup_request is None
    assert result.repository_effects == []


def test_light_refine_drops_effect_with_empty_quote_and_graph_id():
    payload = {
        "witness_valid": False,
        "next_action": "leave_open",
        "witnessed_delta": "",
        "repository_effects": [{
            "kind": "open_decision",
            "statement": "The graph suggests another related path.",
            "evidence": [
                {"evidence_id": "SEARCH-1", "quote": ""},
                {"evidence_id": "GRAPH-2", "quote": "related path"},
            ],
        }],
        "reason": "No local witness was established.",
    }

    result = RUNNER._refine_contrastive_protocol_once(
        json.dumps(payload),
        validation_error(payload),
        lambda _prompt: json.dumps({"next_action": None}),
    )

    assert result.repository_effects == []


def test_protocol_error_routes_directly_to_one_low_refine(monkeypatch):
    payload = invalid_witness_refinement()
    settings = []
    prompts = []

    class Provider:
        def __init__(self, **kwargs):
            settings.append(kwargs)

        def generate(self, prompt):
            prompts.append(prompt)
            if len(prompts) == 1:
                return SimpleNamespace(message=json.dumps(payload))
            return SimpleNamespace(message=json.dumps({
                "next_action": "reject_witness",
                "drop_repository_effect_indexes": [],
            }))

    monkeypatch.setattr(RUNNER, "_load_env", lambda: None)
    monkeypatch.setattr(
        RUNNER,
        "get_model_config",
        lambda _model: {"model": "openai/glm-5.3-flash"},
    )
    monkeypatch.setattr(RUNNER, "LiteLLM", Provider)

    result = RUNNER._run_structured_extraction(
        "Adjudicate the evidence.",
        ContrastiveIntentDecision,
        model_override="glm-5.3-flash",
    )

    assert result.next_action == "reject_witness"
    assert len(prompts) == 2
    assert settings[1]["max_tokens"] == 2048
    assert settings[1]["max_retries"] == 0
    assert settings[1]["timeout"] == 60
    assert settings[1]["extra_body"] == {
        "thinking": {"type": "enabled", "clear_thinking": False},
        "reasoning_effort": "low",
    }


def test_malformed_protocol_refine_uses_safe_closure_without_full_retry(monkeypatch):
    payload = invalid_witness_refinement()
    prompts = []

    class Provider:
        def __init__(self, **_kwargs):
            pass

        def generate(self, prompt):
            prompts.append(prompt)
            return SimpleNamespace(
                message=(json.dumps(payload) if len(prompts) == 1 else "not json")
            )

    monkeypatch.setattr(RUNNER, "_load_env", lambda: None)
    monkeypatch.setattr(
        RUNNER,
        "get_model_config",
        lambda _model: {"model": "openai/glm-5.3-flash"},
    )
    monkeypatch.setattr(RUNNER, "LiteLLM", Provider)

    result = RUNNER._run_structured_extraction(
        "Adjudicate the evidence.",
        ContrastiveIntentDecision,
        model_override="glm-5.3-flash",
        max_attempts=2,
    )

    assert result.next_action == "leave_open"
    assert len(prompts) == 2
