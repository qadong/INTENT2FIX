import hashlib
import json
from types import SimpleNamespace

import pytest

from tests.test_cotest_patch_selection import RUNNER
from INTENT2FIX.intent_dialogue import IntentDialogueResult, RepairIntentContract
from INTENT2FIX.intent_portfolio import (
    IntentPortfolio, IntentHypothesis, IntentObligation,
    branch_correction_clauses, build_hypothesis_context,
)
from INTENT2FIX.memory.memory import Memory
from INTENT2FIX.reproduce.intent_schema import IssueEvidenceFrame


@pytest.mark.parametrize("enabled", [False, True])
def test_repair_registers_diagnostic_reader_only_with_summary_archive(monkeypatch, tmp_path, enabled):
    from types import SimpleNamespace
    from INTENT2FIX import branch_context as bc
    from INTENT2FIX.diagnostic_evidence import record_digest

    iid = "org__repo-1"
    case = tmp_path / iid
    output = case / "portfolio/H0/continuation_1"
    output.mkdir(parents=True)
    archive = {"base_commit": "base", "contract_sha256": "contract", "records": {}, "aliases": {}}
    monkeypatch.setenv("COTEST_BRANCH_CONTEXT", "text")
    monkeypatch.setenv("COTEST_BRANCH_DIAGNOSTIC_SUMMARIES", str(int(enabled)))
    monkeypatch.setenv("INDEX_STORE_DIR", str(tmp_path / "index"))
    monkeypatch.setattr(RUNNER, "_load_env", lambda: None)
    monkeypatch.setattr(RUNNER, "get_benchmark_instance", lambda _: {"base_commit": "base"})
    monkeypatch.setattr(RUNNER, "create_repository", lambda _: object())
    monkeypatch.setattr(RUNNER.CodeIndex, "from_index_name", lambda **kwargs: object())
    monkeypatch.setattr(RUNNER, "FileContext", lambda **kwargs: SimpleNamespace(verification={}))
    monkeypatch.setattr(RUNNER, "_register_intent_verification_contract", lambda *args: None)
    monkeypatch.setattr(RUNNER, "_completion_protocol_kwargs", lambda *args: {})
    monkeypatch.setattr(RUNNER, "_model_kwargs", lambda *args: {})
    captured = {}
    def prepare(memory, *args, **kwargs):
        if enabled:
            assert kwargs["diagnostic_source_dir"] == case
            assert kwargs["diagnostic_cache_dir"] == case / "diagnostic_summaries"
            assert kwargs["summary_model_key"] == "model:low:2048"
        else:
            assert "summarize" not in kwargs
        (output / "branch_evidence_archive.json").write_text(json.dumps(archive))
        return memory, "/tmp/archive.json", archive
    monkeypatch.setattr(bc, "prepare_branch_context", prepare)
    class AtAgentCreation(Exception):
        pass
    def create(**kwargs):
        captured.update(kwargs)
        raise AtAgentCreation
    monkeypatch.setattr(RUNNER.CodingAgent, "create", create)
    class Docker:
        container_name = "test"
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def write_file(self, path, content):
            assert json.loads(content) == archive
        def run_bash(self, *args, **kwargs):
            return "", 0
    with pytest.raises(AtAgentCreation):
        RUNNER.run_repair_step(iid, "model", Docker(), Memory(instance_id=iid, problem_statement="Issue"),
                              str(output), candidate_mode=True)
    if enabled:
        assert captured["evidence_archive_path"] == str(output / "branch_evidence_archive.json")
        assert captured["evidence_archive_sha256"] == record_digest(archive)
    else:
        assert "evidence_archive_path" not in captured


def portfolio():
    return IntentPortfolio(
        obligations=[IntentObligation(id='FIX-1', kind='fix', statement='Implement selected behavior',
                                      authority='issue_explicit', strength='hard')],
        hypotheses=[IntentHypothesis(id='H0', title='Selected policy', branch_assumption='Selected behavior',
                                     covered_obligation_ids=['FIX-1'],
                                     hard_obligation_coverage={'FIX-1': 'Implement selected behavior'})],
    )


def _json_block(text, tag):
    return json.loads(text.split(f'<{tag}>\n', 1)[1].split(f'\n</{tag}>', 1)[0])


def test_implementation_claims_are_separate_without_changing_assigned_contract(tmp_path):
    p = portfolio()
    h = p.hypotheses[0]
    h.root_cause = 'Possibly a missing wrapper forwarding call in src/value.cc.'
    h.repair_scope = ['Expose required public get() API.', 'Try forwarding to value_impl::get().']
    h.expected_behavior = ['get() returns the selected value.']
    h.preservation_behavior = ['Existing callers retain their behavior.']
    h.hard_obligation_coverage = {'FIX-1': 'Proposed wrapper forwarding should cover this.'}
    h.evidence_basis = ['SEARCH-1']
    original = p.model_dump()
    canonical = build_hypothesis_context(p, h)
    clauses = branch_correction_clauses(canonical)
    memory = Memory(instance_id='org__repo-1', problem_statement='Original issue')

    branch = RUNNER._build_portfolio_branch_memory(memory, p, h, str(tmp_path))
    assigned = _json_block(branch.repair_summary, 'intent_hypothesis')
    notes = _json_block(branch.repair_summary, 'hypothesis_implementation_notes')

    assert 'root_cause' not in assigned['branch']
    assert 'hard_obligation_coverage' not in assigned['branch']
    assert notes == {'root_cause': h.root_cause, 'hard_obligation_coverage': h.hard_obligation_coverage}
    assert assigned['branch']['repair_scope'] == h.repair_scope
    assert assigned['branch']['preservation_behavior'] == h.preservation_behavior
    assert branch_correction_clauses(assigned) == clauses
    assigned['branch'].update(notes)
    assert assigned == canonical
    assert build_hypothesis_context(p, h) == canonical
    assert p.model_dump() == original


def test_handoff_retains_raw_relationship_evidence_and_uncertainty(tmp_path):
    observations = [
        {'id': 'SEARCH-1', 'observation': 'wrapper() forwards to impl() at src/value.cc:12.',
         'source': {'path': 'src/value.cc', 'start_line': 12}},
        {'id': 'SEARCH-2', 'observation': 'No indexed usages found; target definition shown.'},
        {'id': 'RUN-1', 'observation': 'Compiler not available', 'exit_code': 127},
    ]
    extra = {'id': 'SOURCE-1', 'observation': 'Exact overload declaration; selection untested.'}
    (tmp_path / 'hypothesis_source_evidence.json').write_text(json.dumps({'evidence': [extra]}))
    p = portfolio()
    memory = Memory(instance_id='org__repo-1', problem_statement='Original issue',
                    repair_intent_bundle={'cited_repository_evidence': observations})
    before = memory.model_dump()

    branch = RUNNER._build_portfolio_branch_memory(memory, p, p.hypotheses[0], str(tmp_path))

    assert branch.repair_intent_bundle['cited_repository_evidence'] == observations + [extra]
    assert memory.model_dump() == before


def test_poc_handoff_preserves_frozen_source_without_acceptance_gate(tmp_path):
    snapshot = {"evidence_id": "RUN-1", "command": "c++ probe.cpp -o /tmp/p && /tmp/p",
                "files": {"probe.cpp": "int main() { return 1; }"},
                "exit_code": 1, "evidence_role": "reproduction"}
    memory = Memory(instance_id="org__repo-1", problem_statement="Public issue",
                    repair_intent_bundle={"scout_probe_snapshot": snapshot})
    p = portfolio()
    branch = RUNNER._build_portfolio_branch_memory(memory, p, p.hypotheses[0], str(tmp_path))
    assert branch.repair_intent_bundle["scout_probe_snapshot"] == snapshot
    assert branch.repair_intent_contract["acceptance_checks"] == []
    branch.repair_intent_bundle["scout_probe_snapshot"]["files"]["probe.cpp"] = "changed"
    assert memory.repair_intent_bundle["scout_probe_snapshot"] == snapshot


@pytest.mark.parametrize("selected_id", ["RUN-1", None, "api_failure"])
def test_poc_snapshot_uses_llm_not_role_or_latest_record(monkeypatch, tmp_path, selected_id):
    actions = [
        dict(action="RunInvestigationCommand", evidence_id="RUN-1", evidence_role="premise",
             command="c++ probe.cpp -o /tmp/p && /tmp/p", exit_code=1,
             frozen_probe_files={"probe.cpp": "exact original version\n"}),
        dict(action="RunInvestigationCommand", evidence_id="RUN-2", evidence_role="reproduction",
             command="grep -n predicate header.hpp", exit_code=0,
             frozen_probe_files={"probe.cpp": "later version\n"}),
    ]
    report = SimpleNamespace(
        input_path_check=None, repository_evidence_ids=["RUN-1", "RUN-2"],
        preserve_evidence_ids=[], repository_facts=[], candidate_boundaries=[],
        open_decisions=[], contrastive_witness=None,
    )
    result = SimpleNamespace(
        render_for_repair=lambda: "Scout summary", intent_contract=RepairIntentContract(),
        exploration_plan=SimpleNamespace(mode="poc"),
        investigation=SimpleNamespace(report=report, actions=actions, repository_evidence=[
            SimpleNamespace(id=item["evidence_id"], path="probe.cpp", locator=item["command"],
                            observation="actual failing output" if i == 0 else "grep output")
            for i, item in enumerate(actions)
        ]),
        contrastive_decision=None, pair_status="", stop_reason="",
    )
    calls = []

    def select(prompt, schema, **kwargs):
        calls.append((prompt, kwargs))
        if selected_id == "api_failure":
            raise RuntimeError("API unavailable")
        return schema.model_validate({"selected_evidence_id": selected_id, "reason": "Selection rationale"})

    monkeypatch.setattr(RUNNER, "_run_structured_extraction", select)
    memory = RUNNER._memory_from_intent_dialogue_result(
        instance_name="org__repo-1", problem_statement="Complete public Issue",
        issue_evidence=IssueEvidenceFrame(), result=result, output_dir=str(tmp_path),
    )
    assert len(calls) == 1
    assert calls[0][1]["glm_reasoning_effort"] == "low"
    assert "Complete public Issue" in calls[0][0]
    assert "RUN-2" in calls[0][0]
    trace = json.loads((tmp_path / "scout_probe_selection.json").read_text())
    if selected_id == "RUN-1":
        snapshot = memory.repair_intent_bundle["scout_probe_snapshot"]
        assert snapshot["evidence_id"] == "RUN-1"
        assert snapshot["command"] == actions[0]["command"]
        assert snapshot["files"] == actions[0]["frozen_probe_files"]
        assert snapshot["baseline_observation"] == "actual failing output"
        assert "Selection rationale" not in json.dumps(memory.repair_intent_bundle)
        assert trace["status"] == "selected"
    else:
        assert "scout_probe_snapshot" not in memory.repair_intent_bundle
        assert trace["status"] == ("unavailable" if selected_id else "no_useful_poc")
    assert memory.repair_intent_contract["acceptance_checks"] == []
    assert actions[0]["frozen_probe_files"]["probe.cpp"] == "exact original version\n"


def test_revise_audit_reaches_repair_as_nonblocking_advisory(tmp_path):
    (tmp_path / 'intent_hypothesis_execution.json').write_text(json.dumps({
        'audit_decision': 'REVISE',
        'selection_mode': 'audit_review_fallback',
        'audit_reason': 'One branch has incomplete closure.',
        'selected_hypothesis_reviews': {
            'H0': {'premise_status': 'supported', 'conflicts': []},
        },
        'blocking_findings': {
            'repair_surface_gaps': ['Verify the public include closure.'],
        },
    }))
    p = portfolio()
    memory = Memory(instance_id='org__repo-1', problem_statement='Original issue')

    branch = RUNNER._build_portfolio_branch_memory(
        memory, p, p.hypotheses[0], str(tmp_path)
    )

    advisory = _json_block(branch.repair_summary, 'hypothesis_audit_advisory')
    assert advisory['audit_decision'] == 'REVISE'
    assert advisory['this_branch_review']['premise_status'] == 'supported'
    assert advisory['portfolio_findings']['repair_surface_gaps']
    assert 'diagnostic, not a Repair gate' in branch.repair_summary


def test_branch_uses_final_contract_and_preserves_diagnostic_probe(tmp_path):
    contract = RepairIntentContract(target_behavior=['Stale target'], preserve_behavior=['Stale preserve'],
        acceptance_checks=[dict(description='Frozen oracle', command='c++ probe.cpp && ./a.out',
            files={'probe.cpp': 'int main() { return 1; }'}, source_evidence_id='RUN-1',
            baseline_exit_code=1, authority='normative')])
    memory = Memory(instance_id='org__repo-1', problem_statement='Original issue',
        repair_summary='STALE SCOUT MANDATORY POLICY', repair_intent_contract=contract.model_dump(),
        repair_intent_bundle={'cited_repository_evidence': [{'id': 'SEARCH-1', 'observation': 'Raw code'}],
                              'open_decisions': ['Ask Repair to choose another policy']})
    for filename, evidence in [('hypothesis_source_evidence.json', [{'id': 'SOURCE-1', 'observation': 'Expanded definition'}]),
                               ('hypothesis_audit_evidence.json', [{'id': 'AUDIT-RUN-1', 'observation': 'Compiler output', 'exit_code': 1}])]:
        (tmp_path / filename).write_text(json.dumps({'evidence': evidence}))
    frozen = memory.model_dump()
    p = portfolio()
    branch = RUNNER._build_portfolio_branch_memory(memory, p, p.hypotheses[0], str(tmp_path))
    assert 'STALE SCOUT' not in branch.repair_summary
    assert 'Selected behavior' in branch.repair_summary
    assert 'Stale target' not in branch.repair_summary
    assert 'open_decisions' not in branch.repair_intent_bundle
    assert {e['id'] for e in branch.repair_intent_bundle['cited_repository_evidence']} == {'SEARCH-1', 'SOURCE-1', 'AUDIT-RUN-1'}
    spec = RepairIntentContract.model_validate(branch.repair_intent_contract).verification_spec()
    assert spec['target_ids'] == spec['preserve_ids'] == spec['acceptance_ids'] == []
    assert spec['diagnostic_ids'] == ['ACCEPT-1']
    assert spec['probes']['ACCEPT-1']['command'] == contract.acceptance_checks[0].command
    assert spec['probes']['ACCEPT-1']['files'] == contract.acceptance_checks[0].files
    assert memory.model_dump() == frozen


def test_unreplayable_scout_probe_is_reported_unavailable_without_changing_it(tmp_path):
    contract = RepairIntentContract(acceptance_checks=[dict(
        description='Historical temporary header', command='c++ -I /tmp/gen probe.cpp -o /tmp/probe && /tmp/probe',
        files={'probe.cpp': 'int main() {}'}, source_evidence_id='RUN-1', baseline_exit_code=1)])
    p = portfolio()
    memory = Memory(instance_id='org__repo-1', problem_statement='Issue', repair_intent_contract=contract.model_dump())
    branch = RUNNER._build_portfolio_branch_memory(memory, p, p.hypotheses[0], str(tmp_path))
    from types import SimpleNamespace
    context = SimpleNamespace(verification={})
    RUNNER._register_intent_verification_contract(context, branch)
    spec = context.verification['intent_contract']
    assert spec['probes'] == {}
    assert spec['diagnostic_ids'] == []
    assert 'ACCEPT-1' in spec['unavailable_checks']
    assert branch.repair_intent_contract['acceptance_checks'][0]['command'] == contract.acceptance_checks[0].command
    assert 'unavailable_checks' in branch.repair_summary


@pytest.mark.parametrize('admitted', [True, False])
def test_repair_only_saves_pool_without_generation_cross_execution_or_selector(monkeypatch, tmp_path, admitted):
    def forbidden(*args, **kwargs):
        pytest.fail('Repair-only crossed a stage boundary')
    for name in ['_generate_intent_portfolio_with_verification', '_cross_execute_portfolio_candidate', '_run_structured_extraction']:
        monkeypatch.setattr(RUNNER, name, forbidden)
    monkeypatch.setenv('COTEST_PORTFOLIO_REPAIR_ONLY', '1')
    monkeypatch.setattr(RUNNER, 'DockerEnv', lambda **kwargs: object())
    def repair(**kwargs):
        patch = 'diff --git a/src/a.cc b/src/a.cc\n'
        from pathlib import Path
        (Path(kwargs['output_dir']) / 'portfolio_candidate_status.json').write_text(json.dumps({
            'admitted': admitted, 'candidate_variants': [{'variant_id': 'initial', 'patch': patch,
            'admitted': admitted, 'finish_succeeded': admitted, 'kept_paths': ['src/a.cc']}]}))
        return patch if admitted else None
    monkeypatch.setattr(RUNNER, 'run_repair_step', repair)
    result = RUNNER.run_intent_portfolio_repair_step(instance_name='org__repo-1', model_name='model',
        image_name='image', work_dir='/repo', base_commit='base',
        memory=Memory(instance_id='org__repo-1', problem_statement='Issue'),
        output_dir=str(tmp_path), container_suffix='run', portfolio=portfolio())
    assert result is None
    pool = json.loads((tmp_path / 'portfolio_candidate_pool.json').read_text())
    assert len(pool) == int(admitted)
    status = json.loads((tmp_path / 'portfolio_repair_status.json').read_text())
    assert status['status'] == 'completed'
    assert status['selector_ran'] is False
    assert not (tmp_path / 'portfolio_candidate_audit.json').exists()


@pytest.mark.parametrize('defect', ['missing', 'wrong_issue', 'tampered_pool', 'unreviewed'])
def test_repair_checkpoint_fails_closed(monkeypatch, tmp_path, defect):
    case = 'org__repo-1'
    instance = {'instance_id': case, 'base_commit': 'base', 'problem_statement': 'Issue'}
    checkpoint = RUNNER._intent_scout_checkpoint_manifest(case, instance)
    directory = tmp_path / case
    directory.mkdir()
    payloads = {'intent_scout_checkpoint.json': checkpoint,
                'intent_grounding_result.json': IntentDialogueResult().model_dump(),
                'intent_portfolio.json': portfolio().model_dump(),
                'intent_hypothesis_rerank.json': {'decision': 'REVISE', 'reason': 'Incomplete'}}
    if defect == 'wrong_issue':
        payloads['intent_scout_checkpoint.json'] = {**checkpoint, 'issue_sha256': 'wrong'}
    hashes = {}
    for name, payload in payloads.items():
        data = json.dumps(payload).encode()
        (directory / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    (directory / 'repair_checkpoint.json').write_text(json.dumps({'artifact_sha256': hashes}))
    if defect == 'missing':
        (directory / 'intent_portfolio.json').unlink()
    if defect == 'tampered_pool':
        (directory / 'intent_portfolio.json').write_text('{}')
    monkeypatch.setenv('COTEST_INTENT_PORTFOLIO_CACHE_DIR', str(tmp_path))
    with pytest.raises((ValueError, FileNotFoundError)):
        RUNNER._load_portfolio_repair_checkpoint(case, instance, str(tmp_path / 'out'))


def test_checkpoint_reuses_final_pool_without_loading_obsolete_scout_decisions(monkeypatch, tmp_path):
    case = 'org__repo-1'
    instance = {'instance_id': case, 'base_commit': 'base', 'problem_statement': 'Issue'}
    raw = IntentDialogueResult().model_dump()
    raw['contrastive_decision'] = {'obsolete_unused_scout_schema': True}
    pool = {**portfolio().model_dump(), 'selected_hypothesis_ids': ['H0']}
    audit = {'decision': 'ACCEPT', 'reason': 'Complete review',
        'selected_hypothesis_ids': ['H0'], 'selection_reasons': {'H0': 'Complete policy'},
        'hypothesis_reviews': {'H0': {'premise_status': 'supported', 'premise_check': 'Input verified',
            'assigned_outcome_status': 'consistent', 'hard_contract_check': 'All targets covered',
            'assigned_outcome_check': 'Behavior consistent', 'implementation_feasibility_check': 'Supported by source'}}}
    execution = {'audit_decision': 'REVISE', 'audit_reason': 'Inspect the sibling path'}
    payloads = {'intent_scout_checkpoint.json': RUNNER._intent_scout_checkpoint_manifest(case, instance),
        'intent_grounding_result.json': raw, 'intent_portfolio.json': pool,
        'intent_hypothesis_rerank.json': audit, 'intent_hypothesis_execution.json': execution}
    directory = tmp_path / case
    directory.mkdir()
    hashes = {}
    for name, payload in payloads.items():
        data = json.dumps(payload).encode()
        (directory / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    (directory / 'repair_checkpoint.json').write_text(json.dumps({'artifact_sha256': hashes}))
    monkeypatch.setenv('COTEST_INTENT_PORTFOLIO_CACHE_DIR', str(tmp_path))
    memory, loaded = RUNNER._load_portfolio_repair_checkpoint(case, instance, str(tmp_path / 'out'))
    assert memory.problem_statement == 'Issue'
    assert loaded.selected_hypothesis_ids == ['H0']
    assert loaded.hypotheses[0].branch_assumption == 'Selected behavior'
    assert (tmp_path / 'out' / 'intent_grounding_result.json').read_bytes() == (directory / 'intent_grounding_result.json').read_bytes()
    assert json.loads((tmp_path / 'out' / 'intent_hypothesis_execution.json').read_text()) == execution


def test_repair_graph_toggle_is_independent_of_scout_graph_toggle(monkeypatch):
    monkeypatch.setenv('COTEST_ABLATE_DUAL_LAYER_GRAPH', '0')
    monkeypatch.setenv('COTEST_SCOUT_GRAPH_SEARCH', '1')
    monkeypatch.setenv('COTEST_REPAIR_GRAPH_SEARCH', '0')
    assert RUNNER._repair_graph_enabled() is False
    monkeypatch.setenv('COTEST_REPAIR_GRAPH_SEARCH', '1')
    assert RUNNER._repair_graph_enabled() is True


@pytest.mark.parametrize('selected_ids', [['H0'], ['H1']])
def test_checkpoint_accepts_only_matching_generated_pool_fallback(monkeypatch, tmp_path, selected_ids):
    case = 'org__repo-1'
    instance = {'instance_id': case, 'base_commit': 'base', 'problem_statement': 'Issue'}
    pool = {**portfolio().model_dump(), 'selected_hypothesis_ids': ['H0']}
    payloads = {
        'intent_scout_checkpoint.json': RUNNER._intent_scout_checkpoint_manifest(case, instance),
        'intent_grounding_result.json': IntentDialogueResult().model_dump(),
        'intent_portfolio.json': pool,
        'intent_hypothesis_execution.json': {
            'audit_available': False,
            'selection_mode': 'generated_pool_fallback',
            'selected_hypothesis_ids': selected_ids,
        },
    }
    directory = tmp_path / case
    directory.mkdir()
    hashes = {}
    for name, payload in payloads.items():
        data = json.dumps(payload).encode()
        (directory / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    (directory / 'repair_checkpoint.json').write_text(json.dumps({'artifact_sha256': hashes}))
    monkeypatch.setenv('COTEST_INTENT_PORTFOLIO_CACHE_DIR', str(tmp_path))
    if selected_ids == ['H0']:
        _, loaded = RUNNER._load_portfolio_repair_checkpoint(case, instance, str(tmp_path / 'out'))
        assert loaded.selected_hypothesis_ids == ['H0']
    else:
        with pytest.raises(ValueError, match='fallback'):
            RUNNER._load_portfolio_repair_checkpoint(case, instance, str(tmp_path / 'out'))
