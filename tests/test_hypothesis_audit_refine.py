import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from INTENT2FIX.intent_portfolio import HypothesisPoolAudit, IntentHypothesisPool

SPEC = importlib.util.spec_from_file_location(
    'audit_refine_runner', Path(__file__).parents[1] / 'scripts/run_no_memory_cotest.py')
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


@pytest.fixture
def audit_case():
    pool = IntentHypothesisPool.model_validate({
        'obligations': [{'id': 'PRESERVE-1', 'kind': 'preserve', 'strength': 'soft',
                         'authority': 'inferred', 'statement': 'Preserve supported inputs'}],
        'hypotheses': [{'id': f'H{i}', 'title': f'Policy {i}',
                        'branch_assumption': f'Policy {i}'} for i in range(4)],
    })
    payload = {
        'decision': 'ACCEPT', 'reason': 'Select H0, H1 and H2; defer H3.',
        'hypothesis_reviews': {h.id: {
            'premise_status': 'supported', 'premise_check': f'Observed {h.id} input gate',
            'assigned_outcome_status': 'consistent', 'hard_contract_check': 'Complete scope',
            'assigned_outcome_check': 'Original fixture preserved',
            'implementation_feasibility_check': 'Existing public route is viable',
        } for h in pool.hypotheses},
        'selection_reasons': {f'H{i}': f'Existing support for policy {i}' for i in range(3)},
        'unselected_reasons': {'H3': 'Deferred for budget'},
        'joint_policy_losses': {'H3': 'The H3 joint combination will not be executed'},
        'nonblocking_notes': ['Runtime confirmation remains unavailable'],
    }
    return pool, payload


@pytest.mark.parametrize('selected', [None, [], ['H0'], ['H0', 'H1']])
def test_refine_only_recovers_selection_already_explained(audit_case, selected):
    pool, payload = audit_case
    if selected is not None:
        payload['selected_hypothesis_ids'] = selected
    original = deepcopy(payload)
    calls = []
    def complete(prompt):
        calls.append(prompt)
        return json.dumps({'selected_hypothesis_ids': ['H0', 'H1', 'H2']})
    result = RUNNER._refine_audit_selection_once(json.dumps(payload), pool, complete)
    expected = HypothesisPoolAudit.model_validate({**payload, 'selected_hypothesis_ids': ['H0', 'H1', 'H2']})
    assert result.model_dump() == expected.model_dump()
    result.validate_pool(pool)
    assert payload == original
    assert len(calls) == 1
    assert 'hypothesis_reviews' not in calls[0]


@pytest.mark.parametrize('defect', ['missing_review', 'contradiction', 'revise', 'unknown_id',
                                  'incomplete_reasons', 'missing_loss', 'valid'])
def test_refine_does_not_invent_or_override_judgments(audit_case, defect):
    pool, payload = audit_case
    if defect == 'missing_review':
        payload['hypothesis_reviews']['H1'].pop('premise_check')
    elif defect == 'contradiction':
        payload['hypothesis_reviews']['H1']['conflicts'] = ['Original scenario contradicts H1']
    elif defect == 'revise':
        payload['decision'] = 'REVISE'
    elif defect == 'unknown_id':
        payload['selected_hypothesis_ids'] = ['H9']
    elif defect == 'incomplete_reasons':
        payload['selection_reasons'].pop('H1')
    elif defect == 'missing_loss':
        payload.pop('joint_policy_losses')
    else:
        payload['selected_hypothesis_ids'] = ['H0', 'H1', 'H2']
    def forbidden(_prompt):
        pytest.fail('A lightweight call must not invent an audit decision')
    assert RUNNER._refine_audit_selection_once(json.dumps(payload), pool, forbidden) is None


@pytest.mark.parametrize('patch', [
    {'selected_hypothesis_ids': ['H0', 'H1', 'H3']},
    {'selected_hypothesis_ids': ['H0', 'H1', 'H1']},
    {'selected_hypothesis_ids': ['H0', 'H1']},
    {'selected_hypothesis_ids': ['H0', 'H1', 'H2'], 'decision': 'REVISE'},
    {'selected_hypothesis_ids': ['H1', 'H0', 'H2']},
])
def test_refine_rejects_changed_selection_or_unrelated_fields(audit_case, patch):
    pool, payload = audit_case
    payload['selected_hypothesis_ids'] = ['H0', 'H1']
    with pytest.raises(ValueError):
        RUNNER._refine_audit_selection_once(json.dumps(payload), pool, lambda _: json.dumps(patch))


@pytest.mark.parametrize('first_disconnect', [False, True])
@pytest.mark.parametrize('with_evidence_tools', [False, True])
def test_extraction_refines_without_repeating_audit(audit_case, monkeypatch, first_disconnect, with_evidence_tools):
    pool, payload = audit_case
    expensive, light = [], []
    def audit(prompt):
        expensive.append(prompt)
        if first_disconnect and len(expensive) == 1:
            raise ConnectionError('incomplete chunked read')
        return json.dumps(payload)
    def complete(**kwargs):
        raw = audit(kwargs['messages'][0]['content'])
        call = SimpleNamespace(function=SimpleNamespace(name='SubmitHypothesisAudit', arguments=raw))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[call], content=''))])
    class Lightweight:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def generate(self, prompt):
            light.append((prompt, self.kwargs))
            return SimpleNamespace(message=json.dumps({'selected_hypothesis_ids': ['H0', 'H1', 'H2']}))
    monkeypatch.setattr(RUNNER, '_load_env', lambda: None)
    monkeypatch.setattr(RUNNER, 'get_model_config', lambda _: {'model': 'openai/glm-5.3-flash'})
    monkeypatch.setattr(RUNNER, 'LiteLLM', Lightweight)
    monkeypatch.setattr('litellm.completion', complete)
    evidence = SimpleNamespace(complete=lambda kwargs, prompt, schema: audit(prompt)) if with_evidence_tools else None
    result = RUNNER._run_structured_extraction('Audit original pool', HypothesisPoolAudit,
        model_override='glm-5.3-flash', max_tokens=16384, glm_reasoning_effort='high',
        audit_pool=pool, audit_tools=evidence)
    assert result is not None
    assert result.selected_hypothesis_ids == ['H0', 'H1', 'H2']
    assert len(expensive) == (2 if first_disconnect else 1)
    assert len(light) == 1
    kwargs = light[0][1]
    assert kwargs['max_tokens'] == 2048 and kwargs['max_retries'] == 0 and kwargs['timeout'] == 60
    assert kwargs['extra_body'] == {
        'thinking': {'type': 'enabled', 'clear_thinking': False}, 'reasoning_effort': 'low'}
    assert 'response_format' not in kwargs


def test_failed_refine_is_bounded_and_original_retry_still_runs(audit_case, monkeypatch):
    pool, payload = audit_case
    expensive, light = [], []
    def audit(_kwargs, prompt, _schema):
        expensive.append(prompt)
        return json.dumps(payload)
    class FailingLightweight:
        def __init__(self, **kwargs):
            pass
        def generate(self, prompt):
            light.append(prompt)
            raise TimeoutError('lightweight call timed out')
    monkeypatch.setattr(RUNNER, '_load_env', lambda: None)
    monkeypatch.setattr(RUNNER, 'get_model_config', lambda _: {'model': 'openai/glm-5.3-flash'})
    monkeypatch.setattr(RUNNER, 'LiteLLM', FailingLightweight)
    result = RUNNER._run_structured_extraction('Audit original pool', HypothesisPoolAudit,
        model_override='glm-5.3-flash', audit_pool=pool,
        audit_tools=SimpleNamespace(complete=audit))
    assert result is None
    assert len(expensive) == 2 and len(light) == 1


def test_shape_refine_closes_review_map_without_changing_judgments(audit_case):
    pool, payload = audit_case
    payload['selected_hypothesis_ids'] = ['H0', 'H1', 'H2']
    raw = json.dumps(payload)
    marker = ', "selection_reasons":'
    offset = raw.index(marker)
    assert raw[offset - 1] == '}'
    broken = raw[:offset - 1] + raw[offset:]
    result = RUNNER._refine_audit_shape_once(broken, pool, lambda _: json.dumps({
        'before': marker, 'insert': '}'}))
    assert result.model_dump() == HypothesisPoolAudit.model_validate(payload).model_dump()


@pytest.mark.parametrize('blocker', [False, True])
def test_shape_refine_relocates_only_redundant_references(audit_case, blocker):
    pool, payload = audit_case
    payload['selected_hypothesis_ids'] = ['H0', 'H1', 'H2']
    if blocker:
        payload['internal_conflicts'] = ['An actual contradictory prediction']
    original = deepcopy(payload['hypothesis_reviews'])
    payload['hypothesis_reviews']['H0-note'] = 'See H0 entry above.'
    result = RUNNER._refine_audit_shape_once(json.dumps(payload), pool,
        lambda _: '{"move_review_notes":["H0-note"]}')
    assert result.decision == ('REVISE' if blocker else 'ACCEPT')
    assert {k:v.model_dump() for k,v in result.hypothesis_reviews.items()} == {
        k:RUNNER.HypothesisPoolAudit.model_validate({**payload, 'hypothesis_reviews':original}).hypothesis_reviews[k].model_dump()
        for k in original}
    assert 'H0-note: See H0 entry above.' in result.nonblocking_notes


@pytest.mark.parametrize('defect', ['semantic_note', 'missing_status', 'real_gap'])
def test_shape_refine_never_invents_semantic_cells_or_routes_blockers(audit_case, defect):
    pool, payload = audit_case
    payload['selected_hypothesis_ids'] = ['H0', 'H1', 'H2']
    if defect == 'semantic_note':
        payload['hypothesis_reviews']['H0-note'] = 'H0 has an actual contradiction'
    elif defect == 'missing_status':
        payload['hypothesis_reviews']['H1'].pop('assigned_outcome_status')
    else:
        payload['causal_coverage_gaps'] = ['Self-described nonblocking sacrifice, needs review']
    def forbidden(_):
        pytest.fail('No semantic repair is allowed')
    assert RUNNER._refine_audit_shape_once(json.dumps(payload), pool, forbidden) is None


@pytest.mark.parametrize('patch', [
    {'before': ', "selection_reasons":', 'insert': '}, "internal_conflicts":[]'},
    {'before': 'not in source', 'insert': '}'},
    {'before': '"', 'insert': '}'},
    {'before': ', "selection_reasons":', 'insert': '}', 'decision': 'ACCEPT'},
])
def test_shape_refine_rejects_content_changes_or_ambiguous_anchor(audit_case, patch):
    pool, payload = audit_case
    payload['selected_hypothesis_ids'] = ['H0', 'H1', 'H2']
    raw = json.dumps(payload)
    offset = raw.index(', "selection_reasons":')
    broken = raw[:offset-1] + raw[offset:]
    with pytest.raises((ValueError, TypeError)):
        RUNNER._refine_audit_shape_once(broken, pool, lambda _: json.dumps(patch))


def test_extraction_syntax_refine_after_transport_failure(audit_case, monkeypatch):
    pool, payload = audit_case
    payload['selected_hypothesis_ids'] = ['H0', 'H1', 'H2']
    raw = json.dumps(payload)
    marker = ', "selection_reasons":'
    offset = raw.index(marker)
    broken = raw[:offset-1] + raw[offset:]
    calls=[]
    def audit(*_):
        calls.append('audit')
        if len(calls)==1:
            raise ConnectionError('first stream disconnected')
        return broken
    class Light:
        def __init__(self, **kwargs): pass
        def generate(self, prompt):
            calls.append('light')
            return SimpleNamespace(message=json.dumps({'before':marker,'insert':'}'}))
    monkeypatch.setattr(RUNNER, '_load_env', lambda: None)
    monkeypatch.setattr(RUNNER, 'get_model_config', lambda _: {'model':'openai/glm-5.3-flash'})
    monkeypatch.setattr(RUNNER, 'LiteLLM', Light)
    result=RUNNER._run_structured_extraction('Audit', HypothesisPoolAudit,
        model_override='glm-5.3-flash', audit_pool=pool, audit_tools=SimpleNamespace(complete=audit))
    assert result is not None and result.decision=='ACCEPT'
    assert calls==['audit','audit','light']


def test_delimiter_refine_cannot_insert_into_judgment_text(audit_case):
    pool, payload = audit_case
    payload['selected_hypothesis_ids'] = ['H0', 'H1', 'H2']
    raw=json.dumps(payload)
    offset=raw.index(', \"selection_reasons\":')
    raw=raw[:offset-1]+raw[offset:]
    with pytest.raises(ValueError, match='delimiter insertion'):
        RUNNER._refine_audit_shape_once(raw,pool,lambda _:json.dumps({
            'before':'Select H0, H1 and H2; defer H3.','insert':'}'}))
