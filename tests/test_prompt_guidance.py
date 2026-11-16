import pytest

from INTENT2FIX.prompt_guidance import load_stage_guidance
from INTENT2FIX.intent_portfolio import build_hypothesis_prompt
from scripts.run_repair_only_ablation import unified_repair_system_prompt


def test_stage_guidance_can_be_ablated_independently(monkeypatch):
    monkeypatch.setenv("COTEST_REPAIR_EVIDENCE_GUIDANCE", "0")
    monkeypatch.setenv("COTEST_HYPOTHESIS_EVIDENCE_GUIDANCE", "1")
    assert load_stage_guidance("repair") == ""
    assert "hypothesis_evidence_guidance" in load_stage_guidance("hypothesis")
    assert "repair_evidence_guidance" not in unified_repair_system_prompt(False)
    monkeypatch.setenv("COTEST_REPAIR_EVIDENCE_GUIDANCE", "1")
    monkeypatch.setenv("COTEST_HYPOTHESIS_EVIDENCE_GUIDANCE", "0")
    assert "repair_evidence_guidance" in unified_repair_system_prompt(False)
    assert load_stage_guidance("hypothesis") == ""


@pytest.mark.parametrize("policy", [False, True])
def test_hypothesis_prompt_dynamically_loads_guidance(monkeypatch, policy):
    arguments = dict(issue="Public issue", issue_evidence={}, repair_intent_bundle={},
                     repair_intent_contract={}, scout_summary="", expanded_pool=policy,
                     policy_prompt=policy)
    monkeypatch.setenv("COTEST_HYPOTHESIS_EVIDENCE_GUIDANCE", "1")
    prompt = build_hypothesis_prompt(**arguments)
    assert "hypothesis_evidence_guidance" in prompt
    assert "untested literal predicate" in prompt
    monkeypatch.setenv("COTEST_HYPOTHESIS_EVIDENCE_GUIDANCE", "0")
    assert "hypothesis_evidence_guidance" not in build_hypothesis_prompt(**arguments)
