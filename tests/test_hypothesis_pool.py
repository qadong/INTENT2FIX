import pytest
from pydantic import ValidationError

from INTENT2FIX.intent_portfolio import (
    HypothesisPoolAudit,
    IntentHypothesisPool,
    assigned_dimension_contracts,
    build_candidate_repair_contract,
    build_hypothesis_context,
    build_hypothesis_prompt,
    build_portfolio_coverage_audit_prompt,
    build_portfolio_revision_prompt,
    hypothesis_review_cards,
    select_hypothesis_pool,
)


def test_expanded_generation_and_audit_do_not_inherit_scout_interpretations():
    args = dict(
        issue="ORIGINAL_PUBLIC_REQUEST",
        issue_evidence={"claims": ["ISSUE_ONLY_CLAIM"]},
        repair_intent_bundle={
            "cited_repository_evidence": [{"id": "SEARCH-1", "observation": "RAW_TYPE_GUARD"}],
            "intent_evidence_index": [{
                "focus_intent_ids": ["INTENT-2"],
                "repository_evidence_ids": ["SEARCH-1"],
            }],
            "input_path_check": {"evaluation": "SCOUT_CAUSAL_GUESS"},
            "repair_surface_leads": [{"statement": "SCOUT_DERIVED_CLOSURE"}],
        },
        repair_intent_contract={"open_decisions": ["SCOUT_PROPOSED_AXIS"]},
        scout_summary="SCOUT_SUMMARY",
        merge_adjudication={"decisions": ["SCOUT_MERGE_VERDICT"]},
    )
    for prompt in [
        build_hypothesis_prompt(**args, expanded_pool=True),
        build_portfolio_coverage_audit_prompt(**args, portfolio=make_pool()),
    ]:
        assert "ORIGINAL_PUBLIC_REQUEST" in prompt
        assert "RAW_TYPE_GUARD" in prompt
        assert "ISSUE_ONLY_CLAIM" in prompt
        assert "INTENT-2" in prompt
        for marker in ["SCOUT_CAUSAL_GUESS", "SCOUT_DERIVED_CLOSURE",
                       "SCOUT_PROPOSED_AXIS", "SCOUT_SUMMARY", "SCOUT_MERGE_VERDICT"]:
            assert marker not in prompt
    assert "SCOUT_CAUSAL_GUESS" in build_hypothesis_prompt(**args)


def test_policy_generation_and_revision_share_primary_evidence_and_schema():
    import json

    args = dict(
        issue='ORIGINAL_PUBLIC_REQUEST', issue_evidence={'claims': ['ISSUE_CLAIM']},
        repair_intent_bundle={
            'cited_repository_evidence': [{'id': 'SOURCE-1', 'observation': 'RAW_DEFINITION'}],
            'intent_evidence_index': [{
                'focus_intent_ids': ['INTENT-3'],
                'repository_evidence_ids': ['SOURCE-1'],
            }],
            'input_path_check': {'evaluation': 'SCOUT_CAUSAL_GUESS'},
        },
        repair_intent_contract={'open_decisions': ['SCOUT_PROPOSED_AXIS']},
        scout_summary='SCOUT_SUMMARY', merge_adjudication={'decisions': []},
    )
    generation = build_hypothesis_prompt(**args, expanded_pool=True, policy_prompt=True)
    revision = build_portfolio_revision_prompt(
        **args, portfolio=make_pool(), audit=make_audit(), policy_prompt=True)
    assert revision.startswith(generation)
    assert json.dumps(IntentHypothesisPool.model_json_schema(), ensure_ascii=False) in generation
    for prompt in [generation, revision]:
        assert 'ORIGINAL_PUBLIC_REQUEST' in prompt
        assert 'RAW_DEFINITION' in prompt
        assert 'INTENT-3' in prompt
        for marker in ['SCOUT_CAUSAL_GUESS', 'SCOUT_PROPOSED_AXIS', 'SCOUT_SUMMARY']:
            assert marker not in prompt
        assert 'mentally apply only' not in prompt
        assert 'Every branch satisfies every hard FIX/CLOSURE/PRESERVE' in prompt
    assert build_hypothesis_prompt(**args, policy_prompt=True) == build_hypothesis_prompt(**args)


def test_same_issue_outcome_keeps_distinct_causal_paths_without_fake_discriminator():
    pool = make_pool().model_dump()
    pool['hypotheses'] = pool['hypotheses'][:2]
    for h, cause, scope in zip(pool['hypotheses'],
            ['Producer selects the wrong representation', 'Consumer rejects a supported representation'],
            ['Repair producer selection', 'Repair consumer conversion']):
        h.update(root_cause=cause, repair_scope=[scope], expected_behavior=['Return the key'])
    pool['unresolved_dimensions'] = [{
        'id': 'DIM-1', 'statement': 'Which causal path explains the input failure?',
        'impact': 'high', 'disposition': 'deferred', 'hypothesis_ids': ['H0', 'H1'],
        'reason': 'Missing an isolated dispatch observation; both paths remain candidates.',
    }]
    validated = IntentHypothesisPool.model_validate(pool)
    assert len(validated.hypotheses) == 2
    assert not assigned_dimension_contracts(validated, validated.hypotheses[0])
    assert not assigned_dimension_contracts(validated, validated.hypotheses[1])


def make_pool():
    return IntentHypothesisPool.model_validate({
        "obligations": [{"id": "FIX-1", "kind": "fix", "statement": "Return the key",
                         "authority": "issue_explicit", "strength": "hard"}],
        "hypotheses": [
            {"id": f"H{i}", "title": f"policy {i}", "branch_assumption": f"policy {i}",
             "expected_behavior": [f"outcome {i}"], "covered_obligation_ids": ["FIX-1"],
             "hard_obligation_coverage": {"FIX-1": f"Return the key using policy {i}."}}
            for i in range(8)
        ],
        "unresolved_dimensions": [{
            "id": "DIM-1", "statement": "Key policy", "impact": "high",
            "disposition": "branched", "hypothesis_ids": [f"H{i}" for i in range(8)],
            "shared_discriminator": "Read the same escaped key through the public wrapper",
            "outcomes": [{"hypothesis_id": f"H{i}", "predicted_outcome": f"outcome {i}"}
                         for i in range(8)], "reason": "Unresolved key semantics",
        }],
    })


def make_audit(ids=("H7", "H3", "H0")):
    return HypothesisPoolAudit(
        decision="ACCEPT", reason="Independent policies cover the target.",
        hypothesis_reviews={f"H{i}": {
            "premise_status": "supported", "assigned_outcome_status": "consistent",
            "premise_check": "Input and receiver match the supported path.",
            "hard_contract_check": "Concrete target handled by this policy.",
            "assigned_outcome_check": "Assigned outcome agrees with policy.",
            "implementation_feasibility_check": "One supported return protocol.",
            "conflicts": [],
        } for i in range(8)},
        discriminator_reviews={"DIM-1": {"reviewed_scenario": "Read the same escaped key through the public wrapper", "compared_hypothesis_ids": ["H0", "H1"], "scenario_status": "distinguishes", "compared_outcomes": {"H0": "outcome 0", "H1": "outcome 1"}, "policy_difference": "Different key result", "comparison": "Same input, distinct outputs.", "conflicts": []}},
        joint_policy_losses={f"H{i}": f"Complete policy {i} not executed."
                             for i in range(8) if f"H{i}" not in ids},
        selected_hypothesis_ids=list(ids),
        selection_reasons={i: "Adds a supported policy." for i in ids},
        unselected_reasons={f"H{i}": "Budget deferred." for i in range(8) if f"H{i}" not in ids},
    )


def test_top_k_keeps_all_policies_and_discriminator_outcomes():
    pool = make_pool()
    selected = select_hypothesis_pool(pool, make_audit())
    assert [h.id for h in selected.repair_hypotheses] == ["H7", "H3", "H0"]
    assert selected.hypotheses == pool.hypotheses
    assert selected.unresolved_dimensions == pool.unresolved_dimensions
    assert not pool.selected_hypothesis_ids
    contracts = assigned_dimension_contracts(selected, selected.repair_hypotheses[0])
    assert contracts[0]["assigned_outcome"] == "outcome 7"
    assert IntentHypothesisPool.model_validate_json(selected.model_dump_json()) == selected


def test_shared_scenario_is_the_only_copy_in_audit_and_repair_payloads():
    import json

    payload = make_pool().model_dump()
    dimension = payload['unresolved_dimensions'][0]
    dimension['hypothesis_ids'] = ['H0', 'H1']
    dimension['outcomes'] = dimension['outcomes'][:2]
    payload['hypotheses'][0].update(
        discriminator='WRONG_RECEIVER_COPY', predicted_outcome='WRONG_RESULT_COPY')
    payload['hypotheses'][7].update(
        discriminator='standalone check', predicted_outcome='standalone result')
    pool = IntentHypothesisPool.model_validate(payload)

    assert pool.unresolved_dimensions[0].model_dump() == dimension
    assert len(pool.hypotheses) == len(payload['hypotheses'])
    for actual, original in zip(pool.hypotheses, payload['hypotheses']):
        assert actual.model_dump(exclude={'discriminator', 'predicted_outcome'}) == {
            k: v for k, v in original.items()
            if k not in {'discriminator', 'predicted_outcome'}
        }
    assert pool.hypotheses[7].discriminator == 'standalone check'
    assert pool.hypotheses[7].predicted_outcome == 'standalone result'
    hypothesis = pool.hypotheses[0]
    for card in [hypothesis_review_cards(pool)[0],
                 build_hypothesis_context(pool, hypothesis),
                 build_candidate_repair_contract(pool, hypothesis)]:
        serialized = json.dumps(card)
        assert 'WRONG_RECEIVER_COPY' not in serialized
        assert 'WRONG_RESULT_COPY' not in serialized
        assert card['assigned_dimensions'][0]['shared_discriminator'] == dimension['shared_discriminator']
        assert card['assigned_dimensions'][0]['assigned_outcome'] == 'outcome 0'
    assert payload['hypotheses'][0]['discriminator'] == 'WRONG_RECEIVER_COPY'
    assert IntentHypothesisPool.model_validate_json(pool.model_dump_json()) == pool


@pytest.mark.parametrize('group,missing,unexpected', [
    ('hypothesis_reviews', 'H0', 'H99'),
    ('discriminator_reviews', 'DIM-1', 'DIM-99'),
])
def test_review_key_error_identifies_exact_retry_correction(group, missing, unexpected):
    payload = make_audit().model_dump()
    payload[group][unexpected] = payload[group].pop(missing)
    audit = HypothesisPoolAudit.model_validate(payload)
    with pytest.raises(ValueError) as error:
        audit.validate_pool(make_pool())
    assert f'{group} expected keys=' in str(error.value)
    assert f"missing=['{missing}']" in str(error.value)
    assert f"unexpected=['{unexpected}']" in str(error.value)
    assert missing not in getattr(audit, group)
    assert unexpected in getattr(audit, group)


@pytest.mark.parametrize('branched', [False, True])
def test_audit_inventory_names_only_branched_reviews_without_dropping_deferred_notes(branched):
    import json

    payload = make_pool().model_dump()
    if not branched:
        payload['unresolved_dimensions'] = []
    payload['unresolved_dimensions'].append({
        'id': 'DIM-2', 'statement': 'Unresolved causal path', 'impact': 'high',
        'disposition': 'deferred', 'hypothesis_ids': ['H0', 'H1'],
        'reason': 'DEFERRED_PATH_CHOICE_MUST_STILL_BE_REVIEWED',
    })
    pool = IntentHypothesisPool.model_validate(payload)
    prompt = build_portfolio_coverage_audit_prompt(
        issue='Original request', issue_evidence={}, repair_intent_bundle={},
        repair_intent_contract={}, scout_summary='', merge_adjudication={}, portfolio=pool,
    )
    inventory_text = prompt.split('Required review table keys for THIS pool (exact sets):\n')[1]
    inventory = json.loads(inventory_text.splitlines()[0])
    assert inventory == {'hypothesis_reviews': [f'H{i}' for i in range(8)],
                         'discriminator_reviews': ['DIM-1'] if branched else []}
    assert 'DEFERRED_PATH_CHOICE_MUST_STILL_BE_REVIEWED' in prompt
    assert 'Fewer than three is valid' in prompt
    assert 'it is not a gate that disables Repair' in prompt
    assert len(pool.hypotheses) == 8


def test_soft_tradeoff_does_not_block_or_excuse_a_hard_conflict():
    payload = make_audit().model_dump()
    payload['hypothesis_reviews']['H0']['soft_tradeoffs'] = [
        'This policy intentionally changes an optional legacy presentation.'
    ]
    audit = HypothesisPoolAudit.model_validate(payload)
    assert audit.decision == 'ACCEPT'
    assert select_hypothesis_pool(make_pool(), audit)
    payload['hypothesis_reviews']['H0']['conflicts'] = [
        'The branch itself also mandates retaining that exact presentation.'
    ]
    audit = HypothesisPoolAudit.model_validate(payload)
    assert audit.decision == 'REVISE'
    assert audit.hypothesis_reviews['H0'].soft_tradeoffs
    assert any('also mandates' in conflict for conflict in audit.internal_conflicts)


@pytest.mark.parametrize("ids", [("H0", "H0", "H1"), ("H0", "H1", "H99")])
def test_rerank_cannot_schedule_duplicate_or_unknown_ids(ids):
    with pytest.raises(ValueError):
        select_hypothesis_pool(make_pool(), make_audit(ids))


@pytest.mark.parametrize('ids', [('H0',), ('H0', 'H1')])
def test_rerank_can_schedule_fewer_than_three_coherent_branches(ids):
    selected = select_hypothesis_pool(make_pool(), make_audit(ids))
    assert selected.selected_hypothesis_ids == list(ids)


@pytest.mark.parametrize('defect', ['duplicate', 'unknown', 'selection_reason',
                                  'unselected_reason', 'empty_reason'])
def test_accept_checks_final_assignment_before_leaving_extraction(defect):
    audit = make_audit()
    if defect == 'duplicate':
        audit.selected_hypothesis_ids[1] = audit.selected_hypothesis_ids[0]
    elif defect == 'unknown':
        audit.selected_hypothesis_ids[0] = 'H99'
    elif defect == 'selection_reason':
        audit.selection_reasons.pop('H7')
    elif defect == 'unselected_reason':
        audit.unselected_reasons.pop('H6')
    else:
        audit.selection_reasons['H7'] = ' '
    with pytest.raises(ValueError):
        audit.validate_pool(make_pool())


@pytest.mark.parametrize('omit_ranking', [False, True])
def test_complete_revise_feedback_does_not_require_final_ranking(omit_ranking):
    payload = make_audit(('H0',)).model_dump()
    payload['causal_coverage_gaps'] = ['FIX-1: the shared entry gate remains closed']
    payload.pop('joint_policy_losses')
    if omit_ranking:
        for field in ['selected_hypothesis_ids', 'selection_reasons', 'unselected_reasons']:
            payload.pop(field)
    audit = HypothesisPoolAudit.model_validate(payload)
    audit.validate_pool(make_pool())
    assert audit.decision == 'REVISE'
    assert not audit.joint_policy_losses  # No invented final-policy explanations.
    selected = select_hypothesis_pool(make_pool(), audit)
    assert selected.selected_hypothesis_ids == (
        ['H0', 'H1', 'H2'] if omit_ranking else ['H0']
    )
    del payload['hypothesis_reviews']['H0']['premise_check']
    with pytest.raises(ValidationError, match='premise_check'):
        HypothesisPoolAudit.model_validate(payload)


def test_nonblocking_notes_survive_without_weakening_gap_or_conflict_veto():
    payload = make_audit().model_dump()
    note = 'Newer compiler probe passed; the reported older compiler is unavailable.'
    payload['nonblocking_notes'] = [note]
    audit = HypothesisPoolAudit.model_validate(payload)
    assert audit.model_dump()['nonblocking_notes'] == [note]
    assert select_hypothesis_pool(make_pool(), audit)
    payload['causal_coverage_gaps'] = ['FIX-1: a required caller still fails']
    assert HypothesisPoolAudit.model_validate(payload).decision == 'REVISE'
    payload['causal_coverage_gaps'] = []
    payload['hypothesis_reviews']['H0']['premise_status'] = 'contradicted'
    assert HypothesisPoolAudit.model_validate(payload).decision == 'REVISE'


def test_pool_without_rerank_cannot_schedule_all_eight_repairs():
    with pytest.raises(ValueError, match="has not been reranked"):
        make_pool().repair_hypotheses


def test_rerank_cannot_silently_drop_unselected_explanations():
    audit = make_audit()
    audit.unselected_reasons.pop("H6")
    with pytest.raises(ValueError, match="every unselected"):
        select_hypothesis_pool(make_pool(), audit)


def test_expanded_pool_rejects_auto_filled_hard_coverage():
    payload = make_pool().model_dump()
    payload["hypotheses"][0]["hard_obligation_coverage"] = {}
    with pytest.raises(ValidationError, match="automatically filled placeholder"):
        IntentHypothesisPool.model_validate(payload)


def test_branch_conflict_overrides_optimistic_accept_but_keeps_other_repairs():
    payload = make_audit().model_dump()
    payload['hypothesis_reviews']['H0']['conflicts'] = [
        'repair_scope resets before lookup but DIM-1 invalidates the new return at call end'
    ]
    audit = HypothesisPoolAudit.model_validate(payload)
    assert audit.decision == 'REVISE'
    assert any('H0' in conflict for conflict in audit.internal_conflicts)
    selected = select_hypothesis_pool(make_pool(), audit)
    assert selected.selected_hypothesis_ids == ['H7', 'H3', 'H0']


def test_nondiscriminating_scenario_overrides_accept_without_blocking_repair():
    payload = make_audit().model_dump()
    payload['discriminator_reviews']['DIM-1']['conflicts'] = [
        'Scenario contains only the common input class; both policies predict the same output'
    ]
    audit = HypothesisPoolAudit.model_validate(payload)
    assert audit.decision == 'REVISE'
    selected = select_hypothesis_pool(make_pool(), audit)
    assert selected.selected_hypothesis_ids == ['H7', 'H3', 'H0']


@pytest.mark.parametrize('field,key', [
    ('hypothesis_reviews', 'H4'),
    ('discriminator_reviews', 'DIM-1'),
    ('joint_policy_losses', 'H6'),
])
def test_incomplete_reviews_or_whole_policy_losses_cannot_schedule(field, key):
    audit = make_audit()
    getattr(audit, field).pop(key)
    with pytest.raises(ValueError):
        select_hypothesis_pool(make_pool(), audit)


def test_policy_loss_failure_identifies_missing_extra_and_empty_entries():
    audit = make_audit(('H0', 'H1', 'H2'))
    audit.joint_policy_losses.pop('H6')
    audit.joint_policy_losses['H0'] = 'Selected policy is not a loss.'
    audit.joint_policy_losses['H4'] = ' '
    with pytest.raises(ValueError) as error:
        select_hypothesis_pool(make_pool(), audit)
    assert "missing=['H6']" in str(error.value)
    assert "unexpected=['H0']" in str(error.value)
    assert "empty=['H4']" in str(error.value)


def test_review_card_keeps_conflicting_assigned_outcome_visible_with_its_branch():
    pool = make_pool()
    pool.hypotheses[0].expected_behavior = ['returned value usable until next lookup']
    pool.unresolved_dimensions[0].outcomes[0].predicted_outcome = 'return already invalid at call end'
    card = hypothesis_review_cards(pool)[0]
    assert card['hypothesis']['expected_behavior'] == ['returned value usable until next lookup']
    assert card['assigned_dimensions'][0]['assigned_outcome'] == 'return already invalid at call end'
    assert 'competing_outcomes' not in card['assigned_dimensions'][0]


def test_factor_coverage_does_not_remove_deferred_complete_policy_loss():
    pool = make_pool()
    pool.hypotheses[0].branch_assumption = 'A with X'
    pool.hypotheses[1].branch_assumption = 'B with Y'
    pool.hypotheses[6].branch_assumption = 'A with Y'
    audit = make_audit(('H0', 'H1', 'H2'))
    audit.joint_policy_losses['H6'] = 'A with Y is not executed, although its factors appear separately.'
    selected = select_hypothesis_pool(pool, audit)
    assert 'H6' not in selected.selected_hypothesis_ids
    assert selected.hypotheses[6].branch_assumption == 'A with Y'
    del audit.joint_policy_losses['H6']
    with pytest.raises(ValueError, match='unexecuted complete policy'):
        select_hypothesis_pool(pool, audit)


@pytest.mark.parametrize('field', ['premise_status', 'assigned_outcome_status'])
def test_negative_judgment_cannot_be_overridden_by_empty_conflicts(field):
    payload = make_audit().model_dump()
    payload['hypothesis_reviews']['H0'][field] = 'contradicted'
    audit = HypothesisPoolAudit.model_validate(payload)
    assert audit.decision == 'REVISE'
    assert HypothesisPoolAudit.model_validate_json(audit.model_dump_json()) == audit
    selected = select_hypothesis_pool(make_pool(), audit)
    assert selected.selected_hypothesis_ids == ['H7', 'H3', 'H0']


def test_revise_keeps_ranked_conditional_branches_for_repair():
    payload = make_audit(('H7', 'H3', 'H0')).model_dump()
    payload['hypothesis_reviews']['H3']['premise_status'] = 'contradicted'
    payload['hypothesis_reviews']['H0']['conflicts'] = [
        'The proposed edit omits a necessary public wrapper.'
    ]
    audit = HypothesisPoolAudit.model_validate(payload)
    assert audit.decision == 'REVISE'
    selected = select_hypothesis_pool(make_pool(), audit)
    assert selected.selected_hypothesis_ids == ['H7', 'H3', 'H0']


def test_revise_without_any_clean_branch_still_schedules_one_repair():
    payload = make_audit().model_dump()
    payload['selected_hypothesis_ids'] = []
    for review in payload['hypothesis_reviews'].values():
        review['premise_status'] = 'contradicted'
    audit = HypothesisPoolAudit.model_validate(payload)
    assert audit.decision == 'REVISE'
    selected = select_hypothesis_pool(make_pool(), audit)
    assert selected.selected_hypothesis_ids == ['H0']


@pytest.mark.parametrize('status', ['same_outcome', 'invalid_fixture', 'undetermined'])
def test_invalid_discrimination_judgment_blocks_optimistic_accept(status):
    payload = make_audit().model_dump()
    payload['discriminator_reviews']['DIM-1']['scenario_status'] = status
    audit = HypothesisPoolAudit.model_validate(payload)
    assert audit.decision == 'REVISE'


def test_identical_observations_override_claimed_distinction():
    payload = make_audit().model_dump()
    payload['discriminator_reviews']['DIM-1']['compared_outcomes'] = {
        'H0': 'Compiles successfully', 'H1': ' compiles   successfully ',
    }
    audit = HypothesisPoolAudit.model_validate(payload)
    assert audit.discriminator_reviews['DIM-1'].scenario_status == 'same_outcome'
    assert audit.decision == 'REVISE'


def test_unverified_premise_alone_does_not_erase_plausible_policy():
    payload = make_audit().model_dump()
    payload['hypothesis_reviews']['H0']['premise_status'] = 'unverified'
    assert select_hypothesis_pool(make_pool(), HypothesisPoolAudit.model_validate(payload))


def test_missing_empty_unselected_map_is_safe_only_when_all_policies_selected():
    pool = make_pool()
    payload = make_audit().model_dump()
    del payload['unselected_reasons']
    with pytest.raises(ValueError, match='every unselected'):
        select_hypothesis_pool(pool, HypothesisPoolAudit.model_validate(payload))
    pool.hypotheses = pool.hypotheses[:3]
    pool.unresolved_dimensions = []
    payload.update(selected_hypothesis_ids=['H0', 'H1', 'H2'],
                   selection_reasons={h.id: 'Chosen' for h in pool.hypotheses},
                   joint_policy_losses={}, discriminator_reviews={})
    payload['hypothesis_reviews'] = {h.id: payload['hypothesis_reviews'][h.id] for h in pool.hypotheses}
    assert select_hypothesis_pool(pool, HypothesisPoolAudit.model_validate(payload))


def test_uniquely_misnested_judgment_keeps_negative_meaning():
    payload = make_audit().model_dump()
    del payload['discriminator_reviews']['DIM-1']['scenario_status']
    payload['discriminator_reviews']['scenario_status'] = 'same_outcome'
    audit = HypothesisPoolAudit.model_validate(payload)
    assert audit.decision == 'REVISE'
    assert audit.discriminator_reviews['DIM-1'].scenario_status == 'same_outcome'
    assert 'scenario_status' in payload['discriminator_reviews']  # input not mutated


@pytest.mark.parametrize('field', ['discriminator_reviews', 'joint_policy_losses'])
def test_omitted_review_table_must_be_empty_for_the_actual_pool(field):
    payload = make_audit().model_dump()
    del payload[field]
    with pytest.raises(ValueError):
        select_hypothesis_pool(make_pool(), HypothesisPoolAudit.model_validate(payload))

    pool = make_pool()
    pool.hypotheses = pool.hypotheses[:3]
    pool.unresolved_dimensions = []
    payload.update(selected_hypothesis_ids=['H0', 'H1', 'H2'],
                   selection_reasons={h.id: 'Chosen' for h in pool.hypotheses},
                   unselected_reasons={})
    payload.pop('discriminator_reviews', None)
    payload.pop('joint_policy_losses', None)
    payload['hypothesis_reviews'] = {
        h.id: payload['hypothesis_reviews'][h.id] for h in pool.hypotheses
    }
    selected = select_hypothesis_pool(pool, HypothesisPoolAudit.model_validate(payload))
    assert selected.selected_hypothesis_ids == ['H0', 'H1', 'H2']


@pytest.mark.parametrize('key,value', [
    ('H2_note', 'placeholder'), ('hard_contract_check', ''),
    ('unused_note', None), ('unused_note', []), ('unused_note', {}),
])
def test_empty_non_review_metadata_does_not_erase_real_reviews(key, value):
    from copy import deepcopy

    payload = make_audit().model_dump()
    payload['hypothesis_reviews'][key] = value
    original = deepcopy(payload)
    audit = HypothesisPoolAudit.model_validate(payload)
    assert audit == make_audit()
    assert select_hypothesis_pool(make_pool(), audit)
    assert payload == original


@pytest.mark.parametrize('key,value', [
    ('H2_note', 'H2 contradicts the original Issue'),
    ('hard_contract_check', 'The required input still fails'),
    ('H0', 'placeholder'), ('H0', {}),
])
def test_meaningful_extra_notes_and_malformed_real_reviews_are_not_discarded(key, value):
    payload = make_audit().model_dump()
    payload['hypothesis_reviews'][key] = value
    with pytest.raises(ValidationError):
        HypothesisPoolAudit.model_validate(payload)


@pytest.mark.parametrize('ambiguous', [False, True])
def test_misnested_conflict_is_preserved_only_with_a_unique_destination(ambiguous):
    payload = make_audit().model_dump()
    reviews = payload['hypothesis_reviews']
    del reviews['H0']['conflicts']
    reviews['conflicts'] = ['Original Issue input still fails under FIX-1']
    if ambiguous:
        del reviews['H1']['conflicts']
        with pytest.raises(ValidationError):
            HypothesisPoolAudit.model_validate(payload)
    else:
        audit = HypothesisPoolAudit.model_validate(payload)
        assert audit.hypothesis_reviews['H0'].conflicts == reviews['conflicts']
        assert audit.decision == 'REVISE'
        selected = select_hypothesis_pool(make_pool(), audit)
        assert selected.selected_hypothesis_ids == ['H7', 'H3', 'H0']
        assert 'conflicts' not in reviews['H0']


def test_extra_real_dimension_comparison_cannot_be_dropped():
    payload = make_audit().model_dump()
    payload['discriminator_reviews']['DIM-1_h1_h2'] = dict(
        payload['discriminator_reviews']['DIM-1'])
    audit = HypothesisPoolAudit.model_validate(payload)
    with pytest.raises(ValueError, match="unexpected=\\['DIM-1_h1_h2'\\]"):
        select_hypothesis_pool(make_pool(), audit)


def test_ambiguous_misnested_judgment_is_not_guessed():
    payload = make_audit().model_dump()
    row = payload['discriminator_reviews']['DIM-1']
    del row['scenario_status']
    payload['discriminator_reviews']['DIM-2'] = dict(row)
    payload['discriminator_reviews']['scenario_status'] = 'distinguishes'
    with pytest.raises(ValidationError):
        HypothesisPoolAudit.model_validate(payload)


@pytest.mark.parametrize('changed_field', ['reviewed_scenario', 'compared_outcomes'])
def test_audit_cannot_accept_a_replacement_fixture_or_prediction(changed_field):
    pool = make_pool()
    audit = make_audit()
    row = audit.discriminator_reviews['DIM-1']
    if changed_field == 'reviewed_scenario':
        row.reviewed_scenario = 'Call a new option instead of the original default API'
    else:
        row.compared_outcomes['H0'] = 'Succeeds only after changing the input mode'
    audit.validate_pool(pool)
    assert audit.decision == 'REVISE'
    assert any('DIM-1' in c and 'original' in c for c in audit.internal_conflicts)
    conflicts = list(audit.internal_conflicts)
    audit.validate_pool(pool)
    assert audit.internal_conflicts == conflicts
    assert select_hypothesis_pool(pool, audit).selected_hypothesis_ids == [
        'H7', 'H3', 'H0'
    ]


def test_audit_binding_is_checked_against_revised_pool_even_with_same_ids():
    pool = make_pool()
    audit = make_audit()
    pool.unresolved_dimensions[0].shared_discriminator = 'Read an unescaped key instead'
    audit.validate_pool(pool)
    assert audit.decision == 'REVISE'


@pytest.mark.parametrize('negative', [False, True])
def test_omitted_empty_conflicts_preserve_explicit_judgments(negative):
    payload = make_audit().model_dump()
    for review in payload['hypothesis_reviews'].values():
        del review['conflicts']
    for review in payload['discriminator_reviews'].values():
        del review['conflicts']
    if negative:
        payload['hypothesis_reviews']['H0']['assigned_outcome_status'] = 'contradicted'
        payload['discriminator_reviews']['DIM-1']['scenario_status'] = 'same_outcome'
    audit = HypothesisPoolAudit.model_validate(payload)
    if negative:
        assert audit.decision == 'REVISE'
        selected = select_hypothesis_pool(make_pool(), audit)
        assert selected.selected_hypothesis_ids == ['H7', 'H3', 'H0']
    else:
        assert select_hypothesis_pool(make_pool(), audit)


@pytest.mark.parametrize('field', ['premise_status', 'assigned_outcome_status'])
def test_empty_conflicts_default_does_not_invent_missing_judgments(field):
    payload = make_audit().model_dump()
    del payload['hypothesis_reviews']['H0']['conflicts']
    del payload['hypothesis_reviews']['H0'][field]
    with pytest.raises(ValidationError):
        HypothesisPoolAudit.model_validate(payload)


@pytest.mark.parametrize('field', ['premise_status', 'assigned_outcome_status'])
@pytest.mark.parametrize('ambiguous', [False, True])
def test_misnested_hypothesis_judgment_requires_one_destination(field, ambiguous):
    payload = make_audit().model_dump()
    reviews = payload['hypothesis_reviews']
    del reviews['H0'][field]
    reviews[field] = 'contradicted'
    if ambiguous:
        del reviews['H1'][field]
        with pytest.raises(ValidationError):
            HypothesisPoolAudit.model_validate(payload)
    else:
        audit = HypothesisPoolAudit.model_validate(payload)
        assert getattr(audit.hypothesis_reviews['H0'], field) == 'contradicted'
        assert audit.decision == 'REVISE'
        assert reviews[field] == 'contradicted'  # input is not mutated
