import json
import hashlib
from types import SimpleNamespace

import pytest

from code_intent.code_intent import entity_tools
from code_intent.code_intent import embedding_manifest
from code_intent.code_intent.graph_builder import Graph
from INTENT2FIX.intent_dialogue import developer_agent as scout


@pytest.fixture(autouse=True)
def ready_semantic_index(monkeypatch, tmp_path):
    monkeypatch.setattr(entity_tools, "get_graph_rag", lambda: SimpleNamespace(
        func_index=SimpleNamespace(index_path=tmp_path / "func_index.faiss"),
    ))
    monkeypatch.setattr(embedding_manifest, "load_embedding_manifest", lambda path: {
        "indexes": {name: {"model": "voyage-2"} for name in ["functions", "features"]},
    })


def graph_action():
    return scout.RepositoryGraphQuery(
        code_index=SimpleNamespace(_activate_instance=lambda: None),
        session=scout.InvestigationSession(),
    )


@pytest.mark.parametrize('indexes,valid', [([1, 0], True), ([0, 0], False), ([3, 0], False), ([-1, 0], False), ([0], True), ([], True)])
def test_llm_companion_selection_is_bounded_to_supplied_candidates(indexes, valid):
    calls = []

    def create_completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            structured_output=scout.StructuralCompanionSelection(indexes=indexes),
            completion=None,
        )

    completion = SimpleNamespace(
        model='fixture', initialize=lambda **kw: None,
        create_completion=create_completion,
    )
    action = scout.RepositoryGraphQuery(code_index=SimpleNamespace(),
        session=scout.InvestigationSession(), structural_completion=completion)
    documents = [json.dumps({'companion': name}) for name in ['producer', 'consumer']]
    if valid:
        result = action._rerank_companions('trace state', documents, top_k=2)
        assert [row['index'] for row in result] == indexes
        assert action._rank_trace['candidate_count'] == 2
    else:
        with pytest.raises(ValueError, match='Invalid structural'):
            action._rerank_companions('trace state', documents, top_k=2)
    assert len(calls) == (1 if valid else 2)


def test_graph_bridge_selection_must_be_within_selected_frontier():
    completion = SimpleNamespace(
        model="fixture", initialize=lambda **kw: None,
        create_completion=lambda **kw: SimpleNamespace(
            structured_output=scout.StructuralCompanionSelection(
                indexes=[0], expand_indexes=[1],
            ), completion=None,
        ),
    )
    action = scout.RepositoryGraphQuery(
        code_index=SimpleNamespace(), session=scout.InvestigationSession(),
        structural_completion=completion,
    )
    with pytest.raises(ValueError, match="Invalid structural"):
        action._rerank_companions(
            "trace state", [json.dumps({"name": name}) for name in ("owner", "member")],
            top_k=2,
        )


@pytest.mark.parametrize("invalid,rejected", [
    (scout.StructuralCompanionSelection(indexes=[2]),
     {"indexes": [2], "expand_indexes": []}),
    (scout.StructuralCompanionSelection(indexes=[0], expand_indexes=[1]),
     {"indexes": [0], "expand_indexes": [1]}),
    (None, {"response_type": "NoneType"}),
])
def test_llm_companion_selection_retries_with_index_feedback(invalid, rejected):
    calls = []
    selections = iter([invalid, scout.StructuralCompanionSelection(
        indexes=[1], expand_indexes=[1],
    )])

    def create_completion(**kwargs):
        calls.append(kwargs["messages"])
        return SimpleNamespace(structured_output=next(selections), completion=None)

    completion = SimpleNamespace(
        model="fixture", initialize=lambda **kw: None,
        create_completion=create_completion,
    )
    action = scout.RepositoryGraphQuery(
        code_index=SimpleNamespace(), session=scout.InvestigationSession(),
        structural_completion=completion,
    )
    documents = [json.dumps({"name": name}) for name in ("producer", "consumer")]

    assert action._rerank_companions("trace state", documents, top_k=2) == [
        {"index": 1, "score": 1.0, "expand": True},
    ]
    assert len(calls) == 2
    assert "retry_feedback" not in json.loads(calls[0][0]["content"])
    retry = json.loads(calls[1][0]["content"])
    assert retry["retry_feedback"]["invalid_selection"] == rejected
    assert "[0, 1]" in retry["retry_feedback"]["requirement"]
    assert action._rank_trace["retry_count"] == 1


def test_llm_companion_selection_stops_after_one_retry():
    calls = []

    def create_completion(**kwargs):
        calls.append(kwargs["messages"])
        return SimpleNamespace(
            structured_output=scout.StructuralCompanionSelection(indexes=[2]),
            completion=None,
        )

    completion = SimpleNamespace(
        model="fixture", initialize=lambda **kw: None,
        create_completion=create_completion,
    )
    action = scout.RepositoryGraphQuery(
        code_index=SimpleNamespace(), session=scout.InvestigationSession(),
        structural_completion=completion,
    )
    with pytest.raises(ValueError, match="after retry.*indexes"):
        action._rerank_companions("trace state", [json.dumps({"name": "producer"})], top_k=1)
    assert len(calls) == 2


def test_graph_companion_handoff_includes_current_source_and_real_path(monkeypatch):
    source = 'int prepare() { return 7; }\n'
    path = [{'seed_id': 'seed', 'edges': [
        {'source': 'seed', 'edge': 'IN_CLASS', 'target': 'owner'},
        {'source': 'companion', 'edge': 'IN_CLASS', 'target': 'owner'}]}]
    monkeypatch.setattr(entity_tools, 'search_by_intent', lambda **kw: {'results': [{
        'func_id': 'companion', 'file': 'x.h', 'line_range': [1, 1],
        'content_hash': hashlib.sha256(source.encode()).hexdigest(),
        'retrieval_role': 'structural_companion', 'graph_paths': path,
    }]})
    action = scout.RepositoryGraphQuery(
        code_index=SimpleNamespace(_file_repo=SimpleNamespace(get_file_content=lambda _: source)),
        session=scout.InvestigationSession())
    action.execute(scout.RepositoryGraphQueryArgs(mode='function', query='state'))
    record = json.loads(action._session.evidence[0].observation)[0]
    assert record['source_status'] == 'verified'
    assert record['source_excerpt'] == '1: ' + source
    assert record['graph_paths'] == path
    stale = action._source_excerpt({'file': 'x.h', 'line_range': [1, 1], 'content_hash': 'old'})
    assert stale['source_status'] == 'index_mismatch' and 'source_excerpt' not in stale
    assert action._source_excerpt({'file': '/old/x.h', 'line_range': [1, 1]})['source_status'] == 'non_repository_path'


def test_neighbor_followup_delivers_verified_source_and_directed_edge(monkeypatch):
    source = 'int consume() { return state; }\n'
    indexed = {'id': 'consumer', 'fq_name': 'C::consume', 'def_file': 'impl.h',
               'def_start_line': 1, 'def_end_line': 1,
               'content_hash': hashlib.sha256(source.encode()).hexdigest()}
    graph = SimpleNamespace(neighbors=lambda eid, edge, direction: {'consumer'} if direction == 'in' else set())
    monkeypatch.setattr(entity_tools, 'get_graph_rag', lambda: SimpleNamespace(graph_ctx=SimpleNamespace(
        functions={'consumer': indexed}, classes={}, graph=graph)))
    monkeypatch.setattr(entity_tools, 'get_entity_neighbors', lambda **kw: {'neighbors': [
        {'id': 'consumer', 'file': 'stale.h', 'edge_types': ['CALLS']}]})
    action = scout.RepositoryGraphQuery(code_index=SimpleNamespace(
        _file_repo=SimpleNamespace(get_file_content=lambda path: source if path == 'impl.h' else None)),
        session=scout.InvestigationSession())
    result = action.execute(scout.RepositoryGraphQueryArgs(mode='neighbors', entity_id='producer', direction='both'))
    assert not result.expect_correction
    row = json.loads(action._session.evidence[0].observation)[0]
    assert row['file'] == 'impl.h' and row['source_excerpt'] == '1: ' + source
    assert row['graph_paths'] == [{'seed_id': 'producer', 'edges': [
        {'source': 'consumer', 'edge': 'CALLS', 'target': 'producer'}]}]
    indexed['content_hash'] = 'stale'
    other = action._source_excerpt({**indexed, 'file': 'impl.h', 'line_range': [1, 1]})
    assert other['source_status'] == 'index_mismatch'


@pytest.mark.parametrize('change,valid', [({}, True), ({'source_quote': 'invented()'}, False),
    ({'source_evidence_id': 'GRAPH-1', 'source_quote': 'edge'}, False),
    ({'evidence_ids': ['SEARCH-99']}, False),
    ({'status': 'unresolved', 'source_evidence_id': '', 'source_quote': '', 'missing_evidence': 'runtime gate'}, True),
    ({'status': 'unresolved', 'missing_evidence': ''}, False)])
def test_investigation_paths_validate_source_and_preserve_open_alternatives(change, valid):
    session = scout.InvestigationSession()
    session.record('cat impl.h', 'return consumer(state);', 0)
    session.record('graph query', 'edge', 0, prefix='GRAPH')
    path = dict(question='Does the producer feed the public consumer?', finding='The wrapper forwards state.',
                status='supported', evidence_ids=['GRAPH-1'], source_evidence_id='SEARCH-1',
                source_quote='return consumer(state);', missing_evidence='')
    path.update(change)
    result = scout.SubmitInvestigationReport(session=session).execute(scout.SubmitInvestigationReportArgs(
        status='partial', answer='Source inspected; runtime remains unchecked.',
        investigation_paths=[scout.InvestigationPath(**path)]))
    assert result.expect_correction == (not valid)
    if valid:
        report = result.properties['investigation_report']
        assert report['investigation_paths'][0] == scout.InvestigationPath(**path).model_dump()
        assert 'GRAPH-1' in report['repository_evidence_ids']
        if path['source_evidence_id']:
            assert 'SEARCH-1' in report['repository_evidence_ids']


def test_negative_search_can_verify_absence_without_invented_source_quote():
    session = scout.InvestigationSession()
    session.record("grep -RInF -- 'at_pointer' include", "", 1)
    path = scout.InvestigationPath(
        path_id="PATH-1", role="ingress",
        question="Does the public front-end expose at_pointer?",
        finding="The focused public-header search found no declaration.",
        status="supported", evidence_ids=["SEARCH-1"],
        verification_kind="absence", source_evidence_id="SEARCH-1",
        missing_evidence="",
    )
    result = scout.SubmitInvestigationReport(session=session).execute(
        scout.SubmitInvestigationReportArgs(
            status="partial", answer="The API is absent; implementation path remains open.",
            investigation_paths=[path],
        )
    )
    assert not result.expect_correction
    assert result.properties["investigation_report"]["investigation_paths"][0]["source_quote"] == ""


def test_scout_report_refine_downgrades_invalid_path_verification_without_retry():
    session = scout.InvestigationSession(
        conservatively_refine_report_paths=True,
    )
    session.record("graph query", "CALLS edge", 0, prefix="GRAPH")
    result = scout.SubmitInvestigationReport(session=session).execute(
        scout.SubmitInvestigationReportArgs(
            status="supported",
            answer="The graph suggests a caller relation.",
            investigation_paths=[scout.InvestigationPath(
                path_id="PATH-1", role="consumer",
                question="Does the caller reach this consumer?",
                finding="The graph reports a CALLS edge.", status="supported",
                evidence_ids=["GRAPH-1"], verification_kind="source",
                source_evidence_id="GRAPH-1", source_quote="CALLS edge",
            )],
        )
    )
    assert not result.expect_correction
    report = result.properties["investigation_report"]
    assert report["status"] == "partial"
    path = report["investigation_paths"][0]
    assert path["status"] == "unresolved"
    assert path["source_evidence_id"] == ""
    assert path["source_quote"] == ""
    assert path["missing_evidence"]
    assert "downgraded" in report["open_decisions"][-1]


def test_scout_report_refine_fills_missing_evidence_without_retry():
    session = scout.InvestigationSession(
        conservatively_refine_report_paths=True,
    )
    session.record("grep -RInF -- 'missing_api' include", "", 1)
    result = scout.SubmitInvestigationReport(session=session).execute(
        scout.SubmitInvestigationReportArgs(
            status="partial",
            answer="The mechanism remains unresolved.",
            investigation_paths=[scout.InvestigationPath(
                path_id="PATH-1", role="consumer",
                question="Which consumer owns the missing behavior?",
                finding="No direct source has established the consumer.",
                status="unresolved", evidence_ids=["SEARCH-1"],
                missing_evidence="",
            )],
        )
    )
    assert not result.expect_correction
    path = result.properties["investigation_report"]["investigation_paths"][0]
    assert path["missing_evidence"] == (
        "Direct source or execution evidence that resolves this path."
    )


def test_scout_report_refine_restores_exact_issue_quote_without_retry():
    issue = (
        "Implement JSON Pointer on top of On Demand\n\n"
        "When the pointer names an array or object, iterate the returned value."
    )
    session = scout.InvestigationSession(
        require_input_path_check=True,
        conservatively_refine_report_protocol=True,
        issue_text=issue,
    )
    session.record("sed -n '1,20p' impl.h", "if (is_object()) consume();", 0)
    result = scout.SubmitInvestigationReport(session=session).execute(
        scout.SubmitInvestigationReportArgs(
            status="partial",
            answer="The public receiver is absent; its consumer remains unresolved.",
            input_path_check=scout.InvestigationInputPathCheck(
                issue_input_quote="Paraphrased On Demand pointer request",
                actual_input="On Demand document and JSON Pointer",
                consumer_input="object consumer",
                condition_evidence_id="SEARCH-1",
                condition_quote="is_object()",
                evaluation="The object consumer is representation-compatible.",
                matches_input="unknown",
                alternative_path="The array consumer remains uninspected.",
            ),
        )
    )
    assert not result.expect_correction
    quote = result.properties["investigation_report"]["input_path_check"]["issue_input_quote"]
    assert quote in issue
    assert quote != "Paraphrased On Demand pointer request"
    assert session.actions[-2] == {
        "action": "RefineInvestigationReportProtocol",
        "field": "input_path_check.issue_input_quote",
        "strategy": "deterministic_exact_issue_excerpt",
    }


def test_scout_report_uses_light_selector_for_an_immutable_condition_quote():
    source = (
        "template <typename T>\n"
        "requires is_supported_input<T>::value\n"
        "void consume(T value);\n"
    )
    selected = "requires is_supported_input<T>::value"
    calls = []

    def select_quote(field_name, relevance, candidates):
        calls.append((field_name, relevance, candidates))
        return selected

    session = scout.InvestigationSession(
        require_input_path_check=True,
        conservatively_refine_report_protocol=True,
        report_quote_selector=select_quote,
    )
    session.record("sed -n '1,20p' impl.h", source, 0)
    result = scout.SubmitInvestigationReport(session=session).execute(
        scout.SubmitInvestigationReportArgs(
            status="partial",
            answer="The source gate was inspected.",
            input_path_check=scout.InvestigationInputPathCheck(
                actual_input="requested T",
                consumer_input="supported T",
                condition_evidence_id="SEARCH-1",
                condition_quote="T must satisfy the supported-input trait",
                evaluation="The trait controls participation for the requested T.",
                matches_input="unknown",
                alternative_path="No competing consumer was established.",
            ),
        )
    )

    assert not result.expect_correction
    assert len(calls) == 1
    assert selected in calls[0][2]
    report = result.properties["investigation_report"]
    assert report["input_path_check"]["condition_quote"] == selected
    assert any(
        action.get("strategy") == "lightweight_exact_quote_selection"
        for action in session.actions
    )


def test_light_quote_selector_is_one_low_bounded_id_only_call(monkeypatch):
    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                tool_calls=[SimpleNamespace(function=SimpleNamespace(
                    arguments=json.dumps({"quote_candidate_id": "QUOTE-2"})
                ))],
                content=None,
            ))],
            usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 37}),
        )

    import litellm

    monkeypatch.setattr(litellm, "completion", complete)
    completion = SimpleNamespace(
        model="openai/glm-5.3-flash",
        metadata={},
        temperature=1.0,
        model_base_url="https://example.invalid/v1",
        model_api_key="fixture-key",
    )
    investigator = scout.ContainerRepositoryInvestigator(
        agent=SimpleNamespace(completion=completion),
        session=scout.InvestigationSession(),
        repository=SimpleNamespace(),
        docker_env=SimpleNamespace(),
        investigation_actions=[],
        submit_action=SimpleNamespace(),
    )
    candidates = ["void declaration();", "if (gate) consume();"]

    assert investigator._select_report_quote(
        "input_path_check.condition_quote",
        "The gate controls consumption.",
        candidates,
    ) == candidates[1]
    assert investigator._select_report_quote("field", "meaning", candidates) == ""
    assert len(calls) == 1
    assert calls[0]["max_tokens"] == 256
    assert calls[0]["max_retries"] == 0
    assert calls[0]["timeout"] == 60.0
    assert calls[0]["extra_body"]["reasoning_effort"] == "low"
    enum = calls[0]["tools"][0]["function"]["parameters"]["properties"][
        "quote_candidate_id"
    ]["enum"]
    assert enum == ["QUOTE-1", "QUOTE-2"]
    assert investigator.session.report_quote_refine_trace["selected_id"] == "QUOTE-2"


def test_scout_report_rejects_unfreezeable_witness_without_downgrading_it():
    session = scout.InvestigationSession(
        conservatively_refine_report_protocol=True,
    )
    session.record(
        "sed -n '1,20p' tests/example.cpp",
        "TEST(example, baseline) { check(); }",
        0,
    )
    for name, exit_code in (("anchor", 0), ("target", 1)):
        session.record(
            f"c++ {name}.cpp $(find src -name '*.cpp') "
            f"-o /tmp/{name} && /tmp/{name}",
            "pass" if exit_code == 0 else "expected failure",
            exit_code,
            prefix="RUN",
            action="RunInvestigationCommand",
            extra={
                "frozen_probe_files": {
                    f"{name}.cpp": f"int main() {{ return {exit_code}; }}",
                },
            },
        )
    witness = scout.ContrastiveWitnessProposal.model_validate({
        "issue_claim_ids": ["EXP-1"],
        "anchor_test_path": "tests/example.cpp",
        "anchor_test_locator": "TEST(example, baseline)",
        "anchor_source_evidence_id": "SEARCH-1",
        "anchor_selection_rationale": "Same public operation and oracle.",
        "anchor": {
            "id": "ANCHOR-1",
            "description": "The baseline input passes.",
            "run_evidence_id": "RUN-1",
        },
        "target": {
            "id": "TARGET-1",
            "description": "The Issue input fails on the baseline.",
            "run_evidence_id": "RUN-2",
        },
        "mutation_summary": "Change only the Issue input.",
        "observed_trace_delta": "The anchor passes and the target fails.",
        "discrimination_analysis": "This is a local input witness.",
    })
    args = scout.SubmitInvestigationReportArgs(
        status="partial",
        answer="The executions provide a contrastive witness.",
        evidence_ids=["SEARCH-1", "RUN-1", "RUN-2"],
        contrastive_witness=witness,
    )

    result = scout.SubmitInvestigationReport(session=session).execute(args)

    assert result.expect_correction
    assert "Command substitution" in result.message
    assert args.contrastive_witness is witness


def test_trace_resolves_symbol_and_one_hop_with_live_path(monkeypatch):
    seed_source = "int Owner::produce() { return consume(); }\n"
    neighbor_source = "int Owner::consume() { return state; }\n"
    downstream_source = "int Owner::read_state() { return state; }\n"
    functions = {
        "seed": {"id": "seed", "name": "produce", "fq_name": "Owner::produce",
                 "def_file": "owner.cc", "def_start_line": 1, "def_end_line": 1,
                 "content_hash": hashlib.sha256(seed_source.encode()).hexdigest()},
        "seed-alias": {"id": "seed-alias", "name": "produce",
                       "fq_name": "macro_namespace::Owner::produce",
                       "def_file": "owner.cc", "def_start_line": 1,
                       "def_end_line": 1,
                       "content_hash": hashlib.sha256(seed_source.encode()).hexdigest()},
        "neighbor": {"id": "neighbor", "name": "consume", "fq_name": "Owner::consume",
                     "def_file": "consumer.cc", "def_start_line": 1, "def_end_line": 1,
                     "content_hash": hashlib.sha256(neighbor_source.encode()).hexdigest()},
        "downstream": {"id": "downstream", "name": "read_state",
                       "fq_name": "Owner::read_state", "def_file": "state.cc",
                       "def_start_line": 1, "def_end_line": 1,
                       "content_hash": hashlib.sha256(downstream_source.encode()).hexdigest()},
    }
    graph = SimpleNamespace(neighbors=lambda eid, edge, direction:
                            {"neighbor"} if eid == "seed" and edge == "CALLS" and direction == "out"
                            else {"downstream"} if eid == "neighbor" and edge == "CALLS"
                            and direction == "out" else set())
    monkeypatch.setattr(entity_tools, "get_graph_rag", lambda: SimpleNamespace(
        graph_ctx=SimpleNamespace(functions=functions, classes={}, graph=graph)))
    monkeypatch.setattr(entity_tools, "search_symbol", lambda **kw: {
        "entities": [functions["seed"], functions["seed-alias"]]
        if kw["type"] == "function" else []})
    monkeypatch.setattr(entity_tools, "get_entity_neighbors", lambda **kw: {
        "neighbors": [{"id": "neighbor", "edge_types": ["CALLS"]}]
        if kw["entity_id"] == "seed" else
        [{"id": "downstream", "edge_types": ["CALLS"]}]
        if kw["entity_id"] == "neighbor" else []})
    session = scout.InvestigationSession()
    session.record("sed -n '1,20p' owner.cc", seed_source, 0)
    repo = SimpleNamespace(get_file_content=lambda path: {
        "owner.cc": seed_source, "consumer.cc": neighbor_source,
        "state.cc": downstream_source,
    }.get(path))
    graph_action = scout.RepositoryGraphQuery(
        code_index=SimpleNamespace(_file_repo=repo, _activate_instance=lambda: None),
        session=session,
    )
    trace = scout.TraceInvestigationPath(graph=graph_action, session=session)
    result = trace.execute(scout.TraceInvestigationPathArgs(
        question="Where does the producer pass state?", role="state_producer",
        anchor_evidence_ids=["SEARCH-1"], missing_relation="Confirm its consumer.",
        symbol_name="produce", class_name="Owner", direction="out",
        edge_types=["CALLS"], limit=4,
    ))
    assert not result.expect_correction
    records = json.loads(session.evidence[-1].observation)
    assert [row["retrieval_role"] for row in records] == ["trace_seed", "trace_neighbor"]
    assert records[0]["alias_entity_ids"] == ["seed", "seed-alias"]
    assert records[1]["graph_paths"] == [{"seed_id": "seed", "edges": [
        {"source": "seed", "edge": "CALLS", "target": "neighbor"},
    ]}]
    assert all(row["source_status"] == "verified" for row in records)
    assert "--CALLS-->" in result.message
    assert "GREF-" in result.message
    assert "source=verified" in result.message
    assert session.live_investigation_paths[0]["path_id"] == "PATH-1"
    assert session.actions[-1]["graph_mode"] == "trace"

    second = trace.execute(scout.TraceInvestigationPathArgs(
        question="Which state reader does the consumer call?",
        entity_ref=records[1]["entity_ref"], direction="out",
        edge_types=["CALLS"],
    ))
    assert not second.expect_correction
    second_records = json.loads(session.evidence[-1].observation)
    assert second_records[1]["fq_name"] == "Owner::read_state"
    assert second_records[1]["source_status"] == "verified"
    assert "--CALLS-->" in second.message
    assert session.live_investigation_paths[-1]["path_id"] == "PATH-2"


def test_live_frontier_must_survive_final_submission():
    session = scout.InvestigationSession()
    session.record("sed -n '1,20p' owner.cc", "void consume();", 0)
    session.open_investigation_path(
        question="Does another ingress share the consumer?", role="ingress",
        evidence_ids=["SEARCH-1"], missing_evidence="Inspect public callers.",
        decision="traced",
    )
    submit = scout.SubmitInvestigationReport(session=session)
    missing = submit.execute(scout.SubmitInvestigationReportArgs(
        status="unresolved", answer="Caller remains unknown."))
    assert not missing.expect_correction
    inserted = missing.properties["investigation_report"]["investigation_paths"][0]
    assert inserted["path_id"] == "PATH-1"
    assert inserted["status"] == "unresolved"
    live = {key: value for key, value in session.live_investigation_paths[0].items()
            if key != "checkpoint_decision"}
    changed_identity = submit.execute(scout.SubmitInvestigationReportArgs(
        status="unresolved", answer="Caller remains unknown.",
        investigation_paths=[scout.InvestigationPath(**{
            **live, "role": "consumer", "question": "A different question?",
        })]))
    assert not changed_identity.expect_correction
    normalized = changed_identity.properties["investigation_report"]["investigation_paths"][0]
    assert normalized["role"] == live["role"]
    assert normalized["question"] == live["question"]
    assert normalized["evidence_ids"] == live["evidence_ids"]
    accepted = submit.execute(scout.SubmitInvestigationReportArgs(
        status="unresolved", answer="Caller remains unknown.",
        investigation_paths=[scout.InvestigationPath(**live)]))
    assert not accepted.expect_correction


def test_live_frontier_retains_accepted_resolution_across_report_revisions():
    session = scout.InvestigationSession()
    session.record("sed -n '1,20p' owner.cc", "void consume();", 0)
    session.open_investigation_path(
        question="Does another ingress share the consumer?", role="ingress",
        evidence_ids=["SEARCH-1"], missing_evidence="Inspect public callers.",
        decision="traced",
    )
    submit = scout.SubmitInvestigationReport(session=session)
    resolved = submit.execute(scout.SubmitInvestigationReportArgs(
        status="partial", answer="The public declaration closes this path.",
        investigation_paths=[scout.InvestigationPath(
            path_id="PATH-1", role="ingress",
            question="Does another ingress share the consumer?",
            finding="The public declaration exposes the consumer.",
            status="supported", evidence_ids=["SEARCH-1"],
            verification_kind="source", source_evidence_id="SEARCH-1",
            source_quote="void consume();", missing_evidence="",
        )],
    ))
    assert not resolved.expect_correction

    revised = submit.execute(scout.SubmitInvestigationReportArgs(
        status="partial", answer="A different report field was revised.",
    ))
    assert not revised.expect_correction
    retained = revised.properties["investigation_report"]["investigation_paths"][0]
    assert retained["status"] == "supported"
    assert retained["finding"] == "The public declaration exposes the consumer."
    assert retained["source_evidence_id"] == "SEARCH-1"
    assert retained["source_quote"] == "void consume();"


def test_trace_is_available_before_reproduction_and_has_no_graph_quota():
    session = scout.InvestigationSession()

    class FakeGraph:
        def execute(self, args):
            evidence = session.record(
                f"trace {args.symbol_name}",
                json.dumps([{"entity_ref": f"GREF-{len(session.evidence) + 1}"}]),
                0,
                prefix="GRAPH",
                action="RepositoryGraphQuery",
            )
            return scout.Observation(
                message=evidence.observation,
                properties={"evidence_id": evidence.id},
            )

    trace = scout.TraceInvestigationPath(graph=FakeGraph(), session=session)
    for index in range(4):
        result = trace.execute(scout.TraceInvestigationPathArgs(
            question=f"Which consumer receives value {index}?",
            symbol_name=f"producer_{index}",
        ))
        assert not result.expect_correction

    assert len(session.live_investigation_paths) == 4
    assert all(action["targeted"] for action in session.actions)
    assert not any(
        action["action"] in {"RunInvestigationCommand", "ReadOnlySearch"}
        for action in session.actions
    )


def test_trace_schema_requires_only_a_question_and_one_runtime_anchor():
    schema = scout.TraceInvestigationPathArgs.model_json_schema()
    assert set(schema["required"]) == {"thoughts", "question"}
    assert "anchor_evidence_ids" not in schema["required"]
    assert "missing_relation" not in schema["required"]
    args = scout.TraceInvestigationPathArgs(
        question="Which caller exposes this failure?",
        symbol_name="produce",
    )
    assert args.anchor_evidence_ids == []
    assert args.missing_relation == ""


def test_trace_uses_gref_when_model_repeats_its_source_and_symbol():
    session = scout.InvestigationSession()
    captured = []

    class FakeGraph:
        def execute(self, args):
            captured.append(args)
            evidence = session.record(
                "graph:trace", "[]", 0, prefix="GRAPH",
                action="RepositoryGraphQuery",
            )
            return scout.Observation(
                message="[GRAPH-1] graph relation",
                properties={"evidence_id": evidence.id},
            )

    result = scout.TraceInvestigationPath(
        graph=FakeGraph(), session=session,
    ).execute(scout.TraceInvestigationPathArgs(
        question="Does this sibling use the same representation?",
        entity_ref="GREF-7", symbol_name="diff",
        source_path="include/nlohmann/json.hpp", source_line=100,
    ))
    assert not result.expect_correction
    assert captured[0].entity_ref == "GREF-7"
    assert not captured[0].symbol_name
    assert not captured[0].source_path
    assert "ignored redundant anchor fields" in result.message


def test_report_without_grounding_terminates_as_partial_instead_of_looping():
    session = scout.InvestigationSession(require_input_path_check=True)
    session.record(
        "c++ control.cc", "passed", 0, prefix="RUN",
        action="RunInvestigationCommand", extra={"evidence_role": "control"},
    )
    session.record(
        "c++ target.cc", "failed", 1, prefix="RUN",
        action="RunInvestigationCommand", extra={"evidence_role": "reproduction"},
    )
    result = scout.SubmitInvestigationReport(session=session).execute(
        scout.SubmitInvestigationReportArgs(
            status="supported",
            answer=(
                "The behavioral contrast is real, but its implementation path is open."
            ),
            evidence_ids=["RUN-1", "RUN-2"],
        )
    )

    assert result.terminal
    report = result.properties["investigation_report"]
    assert report["status"] == "partial"
    assert any("production C/C++ definition" in item
               for item in report["open_decisions"])
    assert any("original Issue input" in item
               for item in report["open_decisions"])


def test_source_grounding_is_semantic_not_ordered_after_the_latest_probe():
    session = scout.InvestigationSession(require_input_path_check=True)
    source = session.record(
        "sed -n '1,20p' include/owner.h",
        "int Owner::produce() { return consume(); }", 0,
    )
    session.record(
        "c++ control.cc", "passed", 0, prefix="RUN",
        action="RunInvestigationCommand", extra={"evidence_role": "control"},
    )
    session.record(
        "c++ target.cc", "failed", 1, prefix="RUN",
        action="RunInvestigationCommand", extra={"evidence_role": "reproduction"},
    )
    result = scout.SubmitInvestigationReport(session=session).execute(
        scout.SubmitInvestigationReportArgs(
            status="partial",
            answer="The pre-existing source read grounds the causal lead.",
            evidence_ids=[source.id, "RUN-1", "RUN-2"],
        )
    )

    assert result.terminal
    report = result.properties["investigation_report"]
    assert source.id in report["repository_evidence_ids"]
    assert not any("No cited production C/C++ definition" in item
                   for item in report["open_decisions"])


def test_best_effort_report_retains_a_rejected_candidate_without_unsafe_fields():
    session = scout.InvestigationSession()
    source = session.record(
        "sed -n '1,20p' include/owner.h",
        "int Owner::produce() { return consume(); }", 0,
    )
    session.planned_task_ids = ["TASK-1"]
    session.latest_report_candidate = {
        "status": "supported",
        "answer": "The source identifies the producer and consumer.",
        "repository_facts": ["Owner::produce forwards to consume."],
        "evidence_ids": [source.id, "SEARCH-404"],
        "investigation_paths": [],
        "candidate_boundaries": [],
        "task_findings": [{
            "task_id": "TASK-1",
            "status": "supported",
            "finding": "The forwarding definition was located.",
            "evidence_ids": [source.id, "SEARCH-404"],
        }],
        "conflicts_with_issue": [],
        "open_decisions": [],
    }
    investigator = scout.ContainerRepositoryInvestigator(
        agent=SimpleNamespace(),
        session=session,
        repository=SimpleNamespace(),
        docker_env=SimpleNamespace(),
        investigation_actions=[],
        submit_action=SimpleNamespace(),
    )

    report = investigator._best_effort_report("No accepted report.")

    assert report.status == "partial"
    assert report.answer == "The source identifies the producer and consumer."
    assert report.repository_evidence_ids == [source.id]
    assert report.task_findings[0].evidence_ids == [source.id]
    assert report.contrastive_witness is None
    assert session.actions[-1]["action"] == "FinalizeBestEffortInvestigationReport"


def test_run_evidence_role_is_required_for_new_tools_and_legacy_defaults_to_premise():
    schema = scout.RunInvestigationCommandArgs.model_json_schema()
    assert "evidence_role" in schema["required"]
    assert schema["properties"]["evidence_role"]["enum"] == [
        "reproduction", "control", "premise",
    ]
    legacy = scout.RunInvestigationCommandArgs(command="c++ -fsyntax-only probe.cpp")
    assert legacy.evidence_role == "premise"


def test_read_only_command_selected_as_run_is_recorded_as_search_evidence():
    docker = SimpleNamespace(
        work_dir="/home/repo",
        run_bash=lambda command, timeout: ("1: int child();", 0),
    )
    session = scout.InvestigationSession()
    run = scout.RunInvestigationCommand(docker_env=docker, session=session)
    result = run.execute(scout.RunInvestigationCommandArgs(
        command="sed -n '1,5p' include/value_iterator.h",
        evidence_role="premise",
        target="value_iterator::child",
        purpose="Read the state producer definition",
    ))
    assert result.properties["evidence_id"] == "SEARCH-1"
    assert session.actions[-1]["action"] == "ReadOnlySearch"
    assert session.evidence[-1].locator == "sed -n '1,5p' include/value_iterator.h"


@pytest.mark.parametrize("role", ["reproduction", "control"])
@pytest.mark.parametrize("command", [
    "c++ probe.cpp -o /tmp/p && /tmp/p; echo exit=$?",
    "c++ probe.cpp -o /tmp/p || true",
    "c++ probe.cpp -o /tmp/p | head",
])
def test_scout_behavioral_commands_cannot_mask_exit_status(role, command):
    def unexpected_execution(*args, **kwargs):
        pytest.fail("Masked verification must not execute")

    session = scout.InvestigationSession()
    run = scout.RunInvestigationCommand(
        docker_env=SimpleNamespace(work_dir="/repo", run_bash=unexpected_execution),
        session=session,
    )
    result = run.execute(scout.RunInvestigationCommandArgs(command=command, evidence_role=role))
    assert result.expect_correction
    assert session.actions == []


def test_scout_behavioral_command_stops_on_newline_failure_and_records_original_command():
    import subprocess

    def execute(command, timeout):
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.stdout + result.stderr, result.returncode

    session = scout.InvestigationSession()
    session.investigation_files["probe.cpp"] = "exact frozen source\n"
    run = scout.RunInvestigationCommand(
        docker_env=SimpleNamespace(work_dir="/repo", run_bash=execute), session=session,
    )
    command = "false\nprintf 'misleading success'"
    result = run.execute(scout.RunInvestigationCommandArgs(command=command, evidence_role="reproduction"))
    assert result.properties["exit_code"] == 1
    assert "misleading success" not in session.evidence[-1].observation
    assert session.actions[-1]["command"] == command
    assert session.actions[-1]["frozen_probe_files"] == {"probe.cpp": "exact frozen source\n"}


def test_cpp_grep_match_suggests_a_complete_source_read_without_mutating_evidence():
    output = "69:  value find_field_unordered(string_view key);"
    docker = SimpleNamespace(
        work_dir="/home/repo",
        run_bash=lambda command, timeout: (output, 0),
    )
    session = scout.InvestigationSession()
    search = scout.ReadOnlySearch(docker_env=docker, session=session)

    result = search.execute(scout.ReadOnlySearchArgs(
        command='grep -n "find_field_unordered" include/ondemand/object.h',
        target="object lookup declaration",
        purpose="Locate the existing lookup before reading its implementation",
    ))

    assert session.evidence[-1].observation == output
    assert result.properties["source_followup"] == {
        "path": "include/ondemand/object.h",
        "line": 69,
        "command": "sed -n '64,94p' include/ondemand/object.h",
    }
    assert "match listing, not a complete function definition" in result.message
    assert "sed -n '64,94p' include/ondemand/object.h" in result.message


def test_relation_trace_rejects_unknown_or_lowercase_edge_names():
    with pytest.raises(ValueError):
        scout.TraceInvestigationPathArgs(
            question="Which operation consumes state?", role="consumer",
            anchor_evidence_ids=["SEARCH-1"], missing_relation="Trace the call.",
            symbol_name="consume", edge_types=["calls", "RETURNS"],
        )
    accepted = scout.TraceInvestigationPathArgs(
        question="Which dependent template call is selected?", role="consumer",
        anchor_evidence_ids=["SEARCH-1"], missing_relation="Trace dependent dispatch.",
        symbol_name="parse", edge_types=["DEP_CALL"],
    )
    assert accepted.edge_types == ["DEP_CALL"]


def test_failed_relation_trace_remains_visible_as_targeted_attempt():
    session = scout.InvestigationSession()
    session.record("sed -n '1p' x.cc", "int f();", 0)
    graph = scout.RepositoryGraphQuery(
        code_index=SimpleNamespace(_activate_instance=lambda: None),
        session=session,
    )
    trace = scout.TraceInvestigationPath(graph=graph, session=session)
    result = trace.execute(scout.TraceInvestigationPathArgs(
        question="Which operation consumes state?", role="consumer",
        anchor_evidence_ids=["SEARCH-1"], missing_relation="Trace the call.",
        symbol_name="consume", edge_types=["CALLS"],
    ))
    assert result.expect_correction
    assert session.actions[-1]["action"] == "RepositoryGraphQuery"
    assert session.actions[-1]["targeted"] is True
    assert session.actions[-1]["trace_failed"] is True
    assert "checkpoint" not in session.actions[-1]


def test_relation_trace_requires_a_real_one_hop_neighbor(monkeypatch):
    source = "int Owner::produce();\n"
    seed = {
        "id": "seed", "name": "produce", "fq_name": "Owner::produce",
        "def_file": "owner.cc", "def_start_line": 1, "def_end_line": 1,
        "content_hash": hashlib.sha256(source.encode()).hexdigest(),
    }
    monkeypatch.setattr(entity_tools, "get_graph_rag", lambda: SimpleNamespace(
        graph_ctx=SimpleNamespace(
            functions={"seed": seed}, classes={},
            graph=SimpleNamespace(
                neighbors=lambda *args: set(),
                edge_buckets=lambda entity_id, direction: (
                    {"DEP_TEMPLATE_REF": {"trait"}}
                    if entity_id == "seed" and direction == "out"
                    else {}
                ),
            ),
        ),
    ))
    monkeypatch.setattr(entity_tools, "search_symbol", lambda **kwargs: {
        "entities": [seed] if kwargs["type"] == "function" else [],
    })
    monkeypatch.setattr(entity_tools, "get_entity_neighbors", lambda **kwargs: {
        "neighbors": [],
    })
    session = scout.InvestigationSession()
    session.record("sed -n '1p' owner.cc", source, 0)
    graph = scout.RepositoryGraphQuery(
        code_index=SimpleNamespace(
            _activate_instance=lambda: None,
            _file_repo=SimpleNamespace(get_file_content=lambda path: source),
        ),
        session=session,
    )
    result = scout.TraceInvestigationPath(graph=graph, session=session).execute(
        scout.TraceInvestigationPathArgs(
            question="Which operation consumes state?", role="consumer",
            anchor_evidence_ids=["SEARCH-1"], missing_relation="Trace the call.",
            symbol_name="produce", class_name="Owner", edge_types=["CALLS"],
        )
    )
    assert result.expect_correction
    assert "No real one-hop relation" in result.message
    assert "out:DEP_TEMPLATE_REF" in result.message
    assert "re-anchor at the concrete" in result.message
    assert session.actions[-1]["trace_failed"] is True
    assert session.live_investigation_paths == []


def test_relation_trace_retries_opposite_direction_when_requested_side_is_empty(
    monkeypatch,
):
    seed_source = "int TestHasher::operator()() const;\n"
    caller_source = "void sortTests() { TestHasher{}(); }\n"
    functions = {
        "hasher": {
            "id": "hasher", "name": "operator()",
            "fq_name": "Catch::TestHasher::operator()",
            "def_file": "registry.cpp", "def_start_line": 1, "def_end_line": 1,
            "content_hash": hashlib.sha256(seed_source.encode()).hexdigest(),
        },
        "sort": {
            "id": "sort", "name": "sortTests", "fq_name": "Catch::sortTests",
            "def_file": "sort.cpp", "def_start_line": 1, "def_end_line": 1,
            "content_hash": hashlib.sha256(caller_source.encode()).hexdigest(),
        },
    }
    graph = SimpleNamespace(neighbors=lambda entity_id, edge, direction:
                            {"sort"} if entity_id == "hasher" and edge == "CALLS"
                            and direction == "in" else set())
    monkeypatch.setattr(entity_tools, "get_graph_rag", lambda: SimpleNamespace(
        graph_ctx=SimpleNamespace(functions=functions, classes={}, graph=graph),
    ))
    monkeypatch.setattr(entity_tools, "search_symbol", lambda **kwargs: {
        "entities": [functions["hasher"]] if kwargs["type"] == "function" else [],
    })
    calls = []

    def neighbors(**kwargs):
        calls.append(kwargs["direction"])
        return {"neighbors": ([{"id": "sort", "edge_types": ["CALLS"]}]
                              if kwargs["direction"] == "in" else [])}

    monkeypatch.setattr(entity_tools, "get_entity_neighbors", neighbors)
    session = scout.InvestigationSession()
    session.record("sed -n '1p' registry.cpp", seed_source, 0)
    action = scout.RepositoryGraphQuery(
        code_index=SimpleNamespace(
            _activate_instance=lambda: None,
            _file_repo=SimpleNamespace(get_file_content=lambda path: {
                "registry.cpp": seed_source, "sort.cpp": caller_source,
            }.get(path)),
        ),
        session=session,
    )
    result = scout.TraceInvestigationPath(graph=action, session=session).execute(
        scout.TraceInvestigationPathArgs(
            question="Which sort path uses the hasher?", role="consumer",
            anchor_evidence_ids=["SEARCH-1"],
            missing_relation="Trace the hashing caller.",
            symbol_name="operator()", class_name="TestHasher",
            direction="out", edge_types=["CALLS"], limit=4,
        )
    )
    assert not result.expect_correction
    assert calls == ["out", "in"]
    assert "opposite `in` direction" in result.message
    records = json.loads(session.evidence[-1].observation)
    assert records[-1]["direction_fallback"] is True
    assert records[-1]["trace_requested_direction"] == "out"
    assert records[-1]["trace_direction_used"] == "in"
    assert records[-1]["graph_paths"] == [{"seed_id": "hasher", "edges": [{
        "source": "sort", "edge": "CALLS", "target": "hasher",
    }]}]
    assert session.actions[-1]["trace_direction"] == {
        "requested": "out", "used": "in", "fallback": True,
    }


def test_relation_trace_resolves_class_source_anchor_to_member_definitions(
    monkeypatch,
):
    class_source = "class document { public: value find_field(string_view); };\n"
    member_source = "value document::find_field(string_view key) { return resume_value().find_field(key); }\n"
    owner = {
        "id": "document", "kind": "class", "fq_name": "ondemand::document",
        "def_file": "document.h", "def_start_line": 1, "def_end_line": 1,
        "content_hash": hashlib.sha256(class_source.encode()).hexdigest(),
    }
    member = {
        "id": "find-field", "name": "find_field",
        "fq_name": "ondemand::document::find_field",
        "def_file": "document-inl.h", "def_start_line": 1, "def_end_line": 1,
        "content_hash": hashlib.sha256(member_source.encode()).hexdigest(),
    }
    relation_graph = SimpleNamespace(
        neighbors=lambda entity_id, edge, direction: (
            {"find-field"}
            if entity_id == "document" and edge == "IN_CLASS" and direction == "in"
            else set()
        )
    )
    monkeypatch.setattr(entity_tools, "get_graph_rag", lambda: SimpleNamespace(
        graph_ctx=SimpleNamespace(
            functions={"find-field": member},
            classes={"document": owner},
            graph=relation_graph,
        ),
    ))
    calls = []

    def neighbors(**kwargs):
        calls.append((kwargs["direction"], tuple(kwargs["edge_types"])))
        return {"neighbors": [
            {"id": "find-field", "edge_types": ["IN_CLASS"]}
        ] if kwargs["direction"] == "in" and kwargs["edge_types"] == ["IN_CLASS"]
            else []}

    monkeypatch.setattr(entity_tools, "get_entity_neighbors", neighbors)
    completion = SimpleNamespace(
        model="fixture",
        initialize=lambda **_kwargs: None,
        create_completion=lambda **_kwargs: SimpleNamespace(
            structured_output=scout.StructuralCompanionSelection(indexes=[0]),
            completion=None,
        ),
    )
    session = scout.InvestigationSession()
    session.record("sed -n '1,20p' document.h", class_source, 0)
    action = scout.RepositoryGraphQuery(
        code_index=SimpleNamespace(
            _activate_instance=lambda: None,
            _file_repo=SimpleNamespace(get_file_content=lambda path: {
                "document.h": class_source,
                "document-inl.h": member_source,
            }.get(path)),
        ),
        session=session,
        structural_completion=completion,
    )
    result = scout.TraceInvestigationPath(graph=action, session=session).execute(
        scout.TraceInvestigationPathArgs(
            question="Which document lookup implements key navigation?",
            role="forwarder",
            anchor_evidence_ids=["SEARCH-1"],
            missing_relation="Resolve the class declaration to a member implementation.",
            source_path="document.h", source_line=1,
            direction="out", edge_types=["CALLS"], limit=4,
        )
    )

    assert not result.expect_correction
    assert calls == [
        ("out", ("CALLS",)),
        ("in", ("CALLS",)),
        ("in", ("IN_CLASS",)),
    ]
    assert "returned real `IN_CLASS` member definitions" in result.message
    records = json.loads(session.evidence[-1].observation)
    assert [row["fq_name"] for row in records] == [
        "ondemand::document", "ondemand::document::find_field",
    ]
    assert records[-1]["graph_paths"] == [{
        "seed_id": "document",
        "edges": [{
            "source": "find-field", "edge": "IN_CLASS", "target": "document",
        }],
    }]
    assert records[-1]["trace_relation_fallback"] == "class_membership"
    assert session.actions[-1]["trace_direction"] == {
        "requested": "out",
        "used": "in",
        "fallback": True,
        "edge_types_requested": ["CALLS"],
        "edge_types_used": ["IN_CLASS"],
        "class_membership_fallback": True,
    }


def test_class_trace_ranks_members_before_return_limit(monkeypatch):
    owner_source = "class document {};\n"
    owner = {
        "id": "document", "kind": "class", "fq_name": "document",
        "def_file": "document.h", "def_start_line": 1, "def_end_line": 1,
        "content_hash": hashlib.sha256(owner_source.encode()).hexdigest(),
    }
    member_ids = [f"member-{index:02d}" for index in range(30)]
    member_sources = {
        f"member-{index:02d}.h": f"void document::member_{index:02d}() {{}}\n"
        for index in range(30)
    }
    members = {
        member_id: {
            "id": member_id,
            "fq_name": "document::diff" if index == 20 else f"document::member_{index:02d}",
            "def_file": f"{member_id}.h", "def_start_line": 1, "def_end_line": 1,
            "content_hash": hashlib.sha256(
                member_sources[f"{member_id}.h"].encode()
            ).hexdigest(),
            "short_intent": "Creates a patch" if index == 20 else "Other operation",
        }
        for index, member_id in enumerate(member_ids)
    }
    graph = SimpleNamespace(neighbors=lambda entity_id, edge, direction: (
        set(member_ids) if entity_id == "document"
        and edge in (None, "IN_CLASS") and direction in ("in", "both")
        else set()
    ))
    monkeypatch.setattr(entity_tools, "get_graph_rag", lambda: SimpleNamespace(
        graph_ctx=SimpleNamespace(functions=members, classes={"document": owner}, graph=graph),
    ))
    requested_limits = []

    def neighbors(**kwargs):
        requested_limits.append(kwargs["max_nodes"])
        return {"neighbors": [
            {"id": member_id, "edge_types": ["IN_CLASS"]}
            for member_id in member_ids[:kwargs["max_nodes"]]
        ]}

    monkeypatch.setattr(entity_tools, "get_entity_neighbors", neighbors)

    def complete(**kwargs):
        payload = json.loads(kwargs["messages"][0]["content"])
        index = next(
            item["index"] for item in payload["candidates"]
            if item["qualified_name_tail"] == "document::diff"
        )
        return SimpleNamespace(
            structured_output=scout.StructuralCompanionSelection(indexes=[index]),
            completion=None,
        )

    completion = SimpleNamespace(
        model="fixture", initialize=lambda **_kwargs: None,
        create_completion=complete,
    )
    session = scout.InvestigationSession()
    action = scout.RepositoryGraphQuery(
        code_index=SimpleNamespace(
            _activate_instance=lambda: None,
            _file_repo=SimpleNamespace(get_file_content=lambda path: {
                "document.h": owner_source, **member_sources,
            }.get(path)),
        ),
        session=session, structural_completion=completion,
    )
    result = action.execute(scout.RepositoryGraphQueryArgs(
        mode="trace", entity_id="document", direction="in",
        edge_types=["IN_CLASS"], target="Which member creates a patch?",
    ))

    assert not result.expect_correction
    assert requested_limits == [30]
    assert action._rank_trace["candidate_count"] == 30
    records = json.loads(session.evidence[-1].observation)
    assert [row["fq_name"] for row in records] == ["document", "document::diff"]
    assert records[-1]["source_status"] == "verified"


def test_direct_call_trace_also_exposes_ranked_owner_sibling_lead(monkeypatch):
    source = (
        "class document {\n"
        "  void patch() { consume_path(); }\n"
        "  void diff() { make_path(); }\n"
        "};\n"
    )
    owner = {
        "id": "document", "kind": "class", "fq_name": "document",
        "def_file": "document.h", "def_start_line": 1, "def_end_line": 4,
        "content_hash": hashlib.sha256(source.encode()).hexdigest(),
    }
    patch = {
        "id": "patch", "fq_name": "document::patch",
        "def_file": "document.h", "def_start_line": 2, "def_end_line": 2,
        "content_hash": hashlib.sha256(source.splitlines(keepends=True)[1].encode()).hexdigest(),
    }
    diff = {
        "id": "diff", "fq_name": "document::diff",
        "def_file": "document.h", "def_start_line": 3, "def_end_line": 3,
        "short_intent": "Produces the same path representation",
        "content_hash": hashlib.sha256(source.splitlines(keepends=True)[2].encode()).hexdigest(),
    }
    helper_source = "void consume_path() {}\n"
    helper = {
        "id": "helper", "fq_name": "consume_path",
        "def_file": "helper.h", "def_start_line": 1, "def_end_line": 1,
        "content_hash": hashlib.sha256(helper_source.encode()).hexdigest(),
    }

    def connected(entity_id, edge, direction):
        if entity_id == "patch" and edge == "CALLS" and direction == "out":
            return {"helper"}
        if entity_id == "document" and edge == "IN_CLASS" and direction == "in":
            return {"patch", "diff"}
        return set()

    monkeypatch.setattr(entity_tools, "get_graph_rag", lambda: SimpleNamespace(
        graph_ctx=SimpleNamespace(
            functions={"patch": patch, "diff": diff, "helper": helper},
            classes={"document": owner}, graph=SimpleNamespace(neighbors=connected),
        ),
    ))
    monkeypatch.setattr(entity_tools, "get_entity_neighbors", lambda **kwargs: {
        "neighbors": [
            {"id": entity_id, "edge_types": kwargs["edge_types"]}
            for edge in kwargs["edge_types"]
            for entity_id in connected(kwargs["entity_id"], edge, kwargs["direction"])
        ],
    })

    def complete(**kwargs):
        candidates = json.loads(kwargs["messages"][0]["content"])["candidates"]
        selected = [
            item["index"] for item in candidates
            if item["qualified_name_tail"] == "document::diff"
        ]
        return SimpleNamespace(
            structured_output=scout.StructuralCompanionSelection(indexes=selected),
            completion=None,
        )

    completion = SimpleNamespace(
        model="fixture", initialize=lambda **_kwargs: None,
        create_completion=complete,
    )
    session = scout.InvestigationSession(issue_text="Custom path type fails in patch")
    session.record("sed -n '1,4p' document.h", source, 0)
    action = scout.RepositoryGraphQuery(
        code_index=SimpleNamespace(
            _activate_instance=lambda: None,
            _file_repo=SimpleNamespace(get_file_content=lambda path: {
                "document.h": source, "helper.h": helper_source,
            }.get(path)),
        ),
        session=session, structural_completion=completion,
    )
    result = action.execute(scout.RepositoryGraphQueryArgs(
        mode="trace", source_path="document.h", source_line=2,
        direction="in", edge_types=["CALLS"], limit=4,
        target="Which independent member produces a custom path?",
    ))

    assert not result.expect_correction
    records = json.loads(session.evidence[-1].observation)
    assert [row["fq_name"] for row in records] == [
        "document::patch", "document", "document::diff",
    ]
    assert records[-1]["retrieval_role"] == "scope_sibling_lead"
    assert records[-1]["graph_paths"] == [{
        "seed_id": "document", "edges": [{
            "source": "diff", "edge": "IN_CLASS", "target": "document",
        }],
    }]
    assert records[-1]["source_status"] == "verified"
    assert "not independent members" in result.message
    assert "scope_sibling_lead" in result.message
    assert action._direction_trace["scope_sibling_leads"] == 1


@pytest.mark.parametrize("explicit_expand", [True, False])
def test_trace_expands_issue_relevant_bridge_past_large_class_frontier(
    monkeypatch, explicit_expand,
):
    issue = "Custom StringType must also work when a JSON Pointer path is created."
    patch_source = "void basic_json::patch_inplace() { use_path(); }\n"
    owner_source = "class basic_json {};\n"
    member_lines = [
        f"void basic_json::member_{index:03d}() {{}}\n"
        for index in range(530)
    ]
    member_lines[-1] = "void basic_json::diff() { make_pointer_path(); }\n"
    member_source = "".join(member_lines)
    functions = {
        "patch": {
            "id": "patch", "fq_name": "basic_json::patch_inplace",
            "def_file": "patch.hpp", "def_start_line": 1, "def_end_line": 1,
            "content_hash": hashlib.sha256(patch_source.encode()).hexdigest(),
        },
    }
    graph = Graph()
    graph.add_node("patch", "FUNC", functions["patch"])
    owner = {
        "id": "owner", "fq_name": "basic_json", "def_file": "owner.hpp",
        "def_start_line": 1, "def_end_line": 1,
        "content_hash": hashlib.sha256(owner_source.encode()).hexdigest(),
    }
    graph.add_node("owner", "CLASS", owner)
    graph.add_edge("patch", "owner", "IN_CLASS")
    for index, line in enumerate(member_lines):
        entity_id = f"member-{index:03d}"
        functions[entity_id] = {
            "id": entity_id,
            "fq_name": "basic_json::diff" if index == 529
            else f"basic_json::member_{index:03d}",
            "def_file": "members.hpp", "def_start_line": index + 1,
            "def_end_line": index + 1,
            "short_intent": "Construct a JSON Pointer path" if index == 529
            else "Independent operation",
            "content_hash": hashlib.sha256(line.encode()).hexdigest(),
        }
        graph.add_node(entity_id, "FUNC", functions[entity_id])
        graph.add_edge(entity_id, "owner", "IN_CLASS")
    rag = SimpleNamespace(graph_ctx=SimpleNamespace(
        functions=functions, classes={"owner": owner}, graph=graph,
    ))
    monkeypatch.setattr(entity_tools, "get_graph_rag", lambda: rag)
    requested_limits = []

    def neighbors(**kwargs):
        entity_id = kwargs["entity_id"]
        edge_types = kwargs["edge_types"]
        direction = kwargs["direction"]
        requested_limits.append((entity_id, kwargs["max_nodes"]))
        ids = set().union(*(
            graph.neighbors(entity_id, edge_type, direction)
            for edge_type in edge_types
        ))
        return {"neighbors": [
            {"id": neighbor_id, "edge_types": [edge_type for edge_type in edge_types
                                           if neighbor_id in graph.neighbors(
                                               entity_id, edge_type, direction)]}
            for neighbor_id in sorted(ids)[:kwargs["max_nodes"]]
        ]}

    monkeypatch.setattr(entity_tools, "get_entity_neighbors", neighbors)
    rank_inputs = []

    def complete(**kwargs):
        payload = json.loads(kwargs["messages"][0]["content"])
        context = json.loads(payload["query"])
        assert context["issue_description"] == issue
        assert context["question"] == "Which other operation creates a pointer path?"
        rank_inputs.append(payload)
        names = [item["qualified_name_tail"] for item in payload["candidates"]]
        if "basic_json" in names:
            index = names.index("basic_json")
            selection = scout.StructuralCompanionSelection(
                indexes=[index],
                expand_indexes=[index] if explicit_expand else [],
            )
        elif "basic_json::diff" in names:
            index = names.index("basic_json::diff")
            selection = scout.StructuralCompanionSelection(
                indexes=[index], expand_indexes=[index],
            )
        else:
            selection = scout.StructuralCompanionSelection(indexes=[])
        return SimpleNamespace(structured_output=selection, completion=None)

    completion = SimpleNamespace(
        model="fixture", initialize=lambda **_kwargs: None,
        create_completion=complete,
    )
    session = scout.InvestigationSession(issue_text=issue)
    session.record("sed -n '1p' patch.hpp", patch_source, 0)
    action = scout.RepositoryGraphQuery(
        code_index=SimpleNamespace(
            _activate_instance=lambda: None,
            _file_repo=SimpleNamespace(get_file_content=lambda path: {
                "patch.hpp": patch_source,
                "owner.hpp": owner_source,
                "members.hpp": member_source,
            }.get(path)),
        ),
        session=session, structural_completion=completion,
    )
    result = scout.TraceInvestigationPath(graph=action, session=session).execute(
        scout.TraceInvestigationPathArgs(
            question="Which other operation creates a pointer path?",
            source_path="patch.hpp", source_line=1,
        )
    )

    assert not result.expect_correction
    assert ("owner", 531) in requested_limits
    assert len(rank_inputs) >= 4  # owner, three complete member batches
    assert sum(
        "basic_json" in [item["qualified_name_tail"] for item in payload["candidates"]]
        for payload in rank_inputs
    ) == 1  # expanding diff must not revisit its already-expanded owner
    records = json.loads(session.evidence[-1].observation)
    assert [row["fq_name"] for row in records] == [
        "basic_json::patch_inplace", "basic_json::diff",
    ]
    assert records[-1]["source_status"] == "verified"
    assert records[-1]["graph_paths"][0]["edges"] == [
        {"source": "patch", "edge": "IN_CLASS", "target": "owner"},
        {"source": "member-529", "edge": "IN_CLASS", "target": "owner"},
    ]
    assert "make_pointer_path" in result.message


def test_trace_revisits_deferred_neighbor_when_bridge_has_no_source_lead(
    monkeypatch,
):
    sources = {
        "anchor.hpp": "void Owner::anchor() { helper(); }\n",
        "owner.hpp": "class Owner {};\n",
        "member.hpp": "void Owner::unrelated() {}\n",
        "helper.hpp": "void helper() { use_state(); }\n",
    }
    entities = {
        name: {
            "id": name, "fq_name": name, "def_file": path,
            "def_start_line": 1, "def_end_line": 1,
            "content_hash": hashlib.sha256(sources[path].encode()).hexdigest(),
        }
        for name, path in (
            ("anchor", "anchor.hpp"), ("owner", "owner.hpp"),
            ("member", "member.hpp"), ("helper", "helper.hpp"),
        )
    }
    graph = Graph()
    for name, entity in entities.items():
        graph.add_node(name, "CLASS" if name == "owner" else "FUNC", entity)
    for source, target, edge in (
        ("anchor", "owner", "IN_CLASS"),
        ("member", "owner", "IN_CLASS"),
        ("anchor", "helper", "CALLS"),
    ):
        graph.add_edge(source, target, edge)
    monkeypatch.setattr(entity_tools, "get_graph_rag", lambda: SimpleNamespace(
        graph_ctx=SimpleNamespace(
            functions={key: value for key, value in entities.items() if key != "owner"},
            classes={"owner": entities["owner"]}, graph=graph,
        ),
    ))

    def neighbors(**kwargs):
        entity_id = kwargs["entity_id"]
        direction = kwargs["direction"]
        edge_types = kwargs["edge_types"]
        ids = set().union(*(
            graph.neighbors(entity_id, edge, direction) for edge in edge_types
        ))
        return {"neighbors": [
            {"id": neighbor_id, "edge_types": [edge for edge in edge_types
                                           if neighbor_id in graph.neighbors(
                                               entity_id, edge, direction)]}
            for neighbor_id in sorted(ids)[:kwargs["max_nodes"]]
        ]}

    monkeypatch.setattr(entity_tools, "get_entity_neighbors", neighbors)
    queries = []

    def complete(**kwargs):
        payload = json.loads(kwargs["messages"][0]["content"])
        context = json.loads(payload["query"])
        queries.append(context)
        names = [item["qualified_name_tail"] for item in payload["candidates"]]
        selected = (
            names.index("owner") if "owner" in names and "recovery_note" not in context
            else names.index("helper") if "helper" in names
            else None
        )
        return SimpleNamespace(
            structured_output=scout.StructuralCompanionSelection(
                indexes=[selected] if selected is not None else [],
            ), completion=None,
        )

    completion = SimpleNamespace(
        model="fixture", initialize=lambda **_kwargs: None,
        create_completion=complete,
    )
    action = scout.RepositoryGraphQuery(
        code_index=SimpleNamespace(
            _activate_instance=lambda: None,
            _file_repo=SimpleNamespace(get_file_content=lambda path: sources.get(path)),
        ),
        session=scout.InvestigationSession(issue_text="State use fails"),
        structural_completion=completion,
    )
    result = action.execute(scout.RepositoryGraphQueryArgs(
        mode="trace", entity_id="anchor", target="Which path uses state?",
    ))
    assert not result.expect_correction
    assert any("recovery_note" in context for context in queries)
    assert action._direction_trace["deferred_recovery"] is True
    assert "helper() { use_state(); }" in result.message


def test_semantic_result_retains_intent_and_stable_id_for_neighbor_lookup(monkeypatch):
    search_options = []

    def search_code(query, top_k_funcs, rerank_first=False, expand_structure=False, structural_reranker=None):
        search_options.append((rerank_first, expand_structure))
        return [{
            "func_id": "FUNC:opaque-id", "score": 0.7,
            "short_intent": "Formats nested elements", "feature_id": "FEATURE:ranges",
            "feature_name": "Range formatting",
        }]

    monkeypatch.setattr(entity_tools, "_get_rag_instance", lambda: SimpleNamespace(
        search_code=search_code,
    ))
    monkeypatch.setattr(entity_tools, "get_entity_by_id", lambda *a, **kw: {
        "id": "FUNC:opaque-id", "short_name": "format", "fq_name": "formatter::format",
        "def_file": "include/ranges.h", "def_start_line": 20, "def_end_line": 30,
        "code": "int format() { return 0; }",
    })
    received = []

    def neighbors(**kwargs):
        received.append(kwargs)
        return {"neighbors": [{"id": "FUNC:consumer", "edge_types": ["CALLS"],
                               "file": "include/ranges.h", "start_line": 40}]}

    monkeypatch.setattr(entity_tools, "get_entity_neighbors", neighbors)
    action = graph_action()
    first = action.execute(scout.RepositoryGraphQueryArgs(mode="function", query="nested elements"))
    candidate = json.loads(action._session.evidence[0].observation)[0]
    assert search_options == [(True, True)]
    assert candidate["short_intent"] == "Formats nested elements"
    assert candidate["feature_id"] == "FEATURE:ranges"
    assert candidate["line_range"] == [20, 30]
    second = action.execute(scout.RepositoryGraphQueryArgs(
        mode="neighbors", entity_id=candidate["id"], edge_types=["CALLS"], direction="out",
    ))
    assert received == [{"entity_id": "FUNC:opaque-id", "edge_types": ["CALLS"],
                         "direction": "out", "max_depth": 1, "max_nodes": 10}]
    assert first.properties["candidate_count"] == second.properties["candidate_count"] == 1
    assert len(action._session.evidence) == 2
    assert "CALLS" in action._session.evidence[1].observation


def test_semantic_entity_ref_survives_opaque_id_and_rejects_conflicting_anchor(monkeypatch):
    source = "int Owner::produce() { return consume(); }\n"
    consumer_source = "int Owner::consume() { return 1; }\n"
    opaque_id = "c:@N@detail@S@Owner@F@produce#&$@malformed-if-copied"
    functions = {
        opaque_id: {
            "id": opaque_id, "name": "produce", "fq_name": "detail::Owner::produce",
            "def_file": "owner.cc", "def_start_line": 1, "def_end_line": 1,
            "content_hash": hashlib.sha256(source.encode()).hexdigest(),
        },
        "consumer": {
            "id": "consumer", "name": "consume", "fq_name": "detail::Owner::consume",
            "def_file": "consumer.cc", "def_start_line": 1, "def_end_line": 1,
            "content_hash": hashlib.sha256(consumer_source.encode()).hexdigest(),
        },
    }
    graph = SimpleNamespace(neighbors=lambda entity_id, edge, direction:
                            {"consumer"} if entity_id == opaque_id and edge == "CALLS"
                            and direction == "out" else set())
    rag = SimpleNamespace(
        func_index=SimpleNamespace(index_path=SimpleNamespace(parent="indexes")),
        graph_ctx=SimpleNamespace(functions=functions, classes={}, graph=graph),
    )
    monkeypatch.setattr(entity_tools, "get_graph_rag", lambda: rag)
    monkeypatch.setattr(entity_tools, "search_by_intent", lambda **kwargs: {
        "results": [functions[opaque_id]],
    })
    calls = []

    def neighbors(**kwargs):
        calls.append(kwargs)
        return {"neighbors": [{"id": "consumer", "edge_types": ["CALLS"]}]}

    monkeypatch.setattr(entity_tools, "get_entity_neighbors", neighbors)
    repo = SimpleNamespace(get_file_content=lambda path: {
        "owner.cc": source, "consumer.cc": consumer_source,
    }.get(path))
    action = scout.RepositoryGraphQuery(
        code_index=SimpleNamespace(_activate_instance=lambda: None, _file_repo=repo),
        session=scout.InvestigationSession(),
    )

    action.execute(scout.RepositoryGraphQueryArgs(mode="function", query="produce state"))
    semantic = json.loads(action._session.evidence[-1].observation)[0]
    assert semantic["entity_ref"] == "GREF-1"
    traced = action.execute(scout.RepositoryGraphQueryArgs(
        mode="trace", entity_ref=semantic["entity_ref"],
        edge_types=["CALLS"], direction="out",
    ))
    assert not traced.expect_correction
    assert calls[0]["entity_id"] == opaque_id
    assert calls[0]["max_nodes"] == 1
    trace_records = json.loads(action._session.evidence[-1].observation)
    assert trace_records[0]["entity_ref"] == semantic["entity_ref"]

    conflicting = scout.RepositoryGraphQuery(
        code_index=SimpleNamespace(_activate_instance=lambda: None),
        session=scout.InvestigationSession(),
    ).execute(scout.RepositoryGraphQueryArgs(
        mode="trace", entity_ref="GREF-1", symbol_name="produce",
    ))
    assert conflicting.expect_correction
    assert "exactly one anchor" in conflicting.message


def test_semantic_entity_ref_includes_byte_identical_packaged_graph_alias(monkeypatch):
    source = "int ns::adapter::convert() { return detect(); }\n"
    consumer_source = "int ns::detect() { return 1; }\n"
    modular_id = "modular:adapter::convert"
    packaged_id = "single_include:adapter::convert"
    abi_packaged_id = "single_include:abi::adapter::convert"
    functions = {
        modular_id: {
            "id": modular_id, "name": "convert",
            "fq_name": "ns::adapter::convert",
            "def_file": "include/ns/adapter.hpp",
            "def_start_line": 1, "def_end_line": 1,
            "return_type": "int", "parameter_types": [],
            "content_hash": hashlib.sha256(source.encode()).hexdigest(),
        },
        packaged_id: {
            "id": packaged_id, "name": "convert",
            "fq_name": "ns::adapter::convert",
            "def_file": "single_include/ns.hpp",
            "def_start_line": 1, "def_end_line": 1,
            "return_type": "int", "parameter_types": [],
            "content_hash": hashlib.sha256(source.encode()).hexdigest(),
        },
        abi_packaged_id: {
            "id": abi_packaged_id, "name": "convert",
            "fq_name": "ns::abi::adapter::convert",
            "def_file": "single_include/abi/ns.hpp",
            "def_start_line": 1, "def_end_line": 1,
            "return_type": "int", "parameter_types": [],
            "content_hash": hashlib.sha256(source.encode()).hexdigest(),
        },
        "detect": {
            "id": "detect", "name": "detect", "fq_name": "ns::detect",
            "def_file": "include/ns/detect.hpp",
            "def_start_line": 1, "def_end_line": 1,
            "content_hash": hashlib.sha256(consumer_source.encode()).hexdigest(),
        },
    }

    def graph_neighbors(entity_id, edge, direction):
        if (entity_id == packaged_id and edge == "DEP_TEMPLATE_REF"
                and direction == "out"):
            return {"detect"}
        return set()

    graph = SimpleNamespace(neighbors=graph_neighbors)
    rag = SimpleNamespace(
        func_index=SimpleNamespace(index_path=SimpleNamespace(parent="indexes")),
        graph_ctx=SimpleNamespace(functions=functions, classes={}, graph=graph),
    )
    monkeypatch.setattr(entity_tools, "get_graph_rag", lambda: rag)
    monkeypatch.setattr(entity_tools, "search_by_intent", lambda **kwargs: {
        "results": [functions[modular_id]],
    })
    calls = []

    def neighbors(**kwargs):
        calls.append(kwargs["entity_id"])
        return {"neighbors": (
            [{"id": "detect", "edge_types": ["DEP_TEMPLATE_REF"]}]
            if kwargs["entity_id"] == packaged_id else []
        )}

    monkeypatch.setattr(entity_tools, "get_entity_neighbors", neighbors)
    repo = SimpleNamespace(get_file_content=lambda path: {
        "include/ns/adapter.hpp": source,
        "single_include/ns.hpp": source,
        "single_include/abi/ns.hpp": source,
        "include/ns/detect.hpp": consumer_source,
    }.get(path))
    action = scout.RepositoryGraphQuery(
        code_index=SimpleNamespace(
            _activate_instance=lambda: None,
            _file_repo=repo,
        ),
        session=scout.InvestigationSession(),
    )

    search = action.execute(scout.RepositoryGraphQueryArgs(
        mode="function", query="adapter conversion detector",
    ))
    assert "byte-identical source form: single_include/abi/ns.hpp:1" in search.message
    semantic = json.loads(action._session.evidence[-1].observation)[0]
    assert semantic["alias_entity_ids"] == [modular_id, packaged_id]
    assert semantic["equivalent_source_forms"] == [
        {"path": "single_include/ns.hpp", "line": 1, "source_status": "verified"},
        {"path": "single_include/abi/ns.hpp", "line": 1, "source_status": "verified"},
    ]

    traced = action.execute(scout.RepositoryGraphQueryArgs(
        mode="trace", entity_ref=semantic["entity_ref"], direction="out",
        edge_types=["DEP_TEMPLATE_REF"],
    ))
    assert not traced.expect_correction
    assert calls[:2] == [modular_id, packaged_id]
    rows = json.loads(action._session.evidence[-1].observation)
    assert any(row.get("fq_name") == "ns::detect" for row in rows)


def test_source_location_reanchors_trace_to_smallest_containing_definition(monkeypatch):
    source = "class range_formatter {\n  int parse() { return underlying(); }\n};\n"
    parse = {
        "id": "parse", "name": "parse", "fq_name": "fmt::range_formatter::parse",
        "def_file": "include/fmt/ranges.h", "def_start_line": 2, "def_end_line": 2,
        "content_hash": hashlib.sha256(
            "  int parse() { return underlying(); }\n".encode()
        ).hexdigest(),
    }
    owner = {
        "id": "owner", "name": "range_formatter", "fq_name": "fmt::range_formatter",
        "def_file": "include/fmt/ranges.h", "def_line": 1, "def_end_line": 3,
        "content_hash": hashlib.sha256(source.encode()).hexdigest(),
    }
    underlying = {
        "id": "underlying", "name": "underlying", "fq_name": "fmt::underlying",
        "def_file": "include/fmt/ranges.h", "def_start_line": 2, "def_end_line": 2,
        "content_hash": parse["content_hash"],
    }
    graph = SimpleNamespace(neighbors=lambda entity_id, edge, direction:
                            {"underlying"} if entity_id == "parse"
                            and edge == "DEP_MEMBER_REF" and direction == "out" else set())
    rag = SimpleNamespace(graph_ctx=SimpleNamespace(
        functions={"parse": parse, "underlying": underlying},
        classes={"owner": owner}, graph=graph,
    ))
    monkeypatch.setattr(entity_tools, "get_graph_rag", lambda: rag)
    received = []

    def neighbors(**kwargs):
        received.append(kwargs["entity_id"])
        return {"neighbors": ([{"id": "underlying", "edge_types": ["DEP_MEMBER_REF"]}]
                              if kwargs["entity_id"] == "parse" else [])}

    monkeypatch.setattr(entity_tools, "get_entity_neighbors", neighbors)
    session = scout.InvestigationSession()
    session.record("sed -n '1,3p' include/fmt/ranges.h", source, 0)
    action = scout.RepositoryGraphQuery(
        code_index=SimpleNamespace(
            _activate_instance=lambda: None,
            _file_repo=SimpleNamespace(get_file_content=lambda path: source),
        ),
        session=session,
    )
    result = scout.TraceInvestigationPath(graph=action, session=session).execute(
        scout.TraceInvestigationPathArgs(
            question="Which formatter state does parse delegate to?",
            role="consumer", anchor_evidence_ids=["SEARCH-1"],
            missing_relation="Trace the member dependency.",
            source_path="include/fmt/ranges.h", source_line=2,
            direction="out", edge_types=["DEP_MEMBER_REF"], limit=4,
        )
    )
    assert not result.expect_correction
    records = json.loads(session.evidence[-1].observation)
    assert records[0]["fq_name"] == "fmt::range_formatter::parse"
    assert records[-1]["fq_name"] == "fmt::underlying"
    assert received[0] == "parse"
    assert records[-1]["graph_paths"][0]["edges"][0] == {
        "source": "parse", "edge": "DEP_MEMBER_REF", "target": "underlying",
    }


def test_feature_members_are_bounded_and_keep_cross_layer_ids(monkeypatch):
    monkeypatch.setattr(entity_tools, "list_features", lambda **kw: {"features": [{
        "feature_id": "FEATURE:1", "feature_name": "Formatting",
        "feature_description": "description " * 100,
        "members": [{"func_id": f"FUNC:{i}", "file": "include/a.h", "start_line": i,
                     "code": "private large body"} for i in range(8)],
    }] * 4})
    action = graph_action()
    action.execute(scout.RepositoryGraphQueryArgs(mode="intent", query="formatting", limit=2))
    records = json.loads(action._session.evidence[0].observation)
    assert len(records) == 2
    assert len(records[0]["members"]) == 3
    assert records[0]["members"][0]["func_id"] == "FUNC:0"
    assert len(records[0]["feature_description"]) <= 800
    assert "private large body" not in action._session.evidence[0].observation


def test_real_feature_member_label_is_not_used_as_a_graph_id(monkeypatch):
    monkeypatch.setattr(entity_tools, "list_features", lambda **kw: {"features": [{
        "feature_id": "feat_1", "feature_description": "Formats ranges",
        "members": [{"func_id": "format(range) — include/ranges.h",
                     "raw_func_id": "c:@opaque-compiler-id", "file": "include/ranges.h"}],
    }]})
    action = graph_action()
    action.execute(scout.RepositoryGraphQueryArgs(mode="intent", query="ranges"))
    member = json.loads(action._session.evidence[0].observation)[0]["members"][0]
    assert member["func_id"] == "c:@opaque-compiler-id"


def test_scout_can_requery_code_intent_without_opening_an_unverified_path(monkeypatch):
    queries = []
    monkeypatch.setattr(entity_tools, "search_by_intent", lambda **kwargs: (
        queries.append(kwargs) or {"results": [{"func_id": "FUNC:1", "short_intent": "creates state"}]}
    ))
    graph = graph_action()
    search = scout.SearchCodeIntent(graph=graph, session=graph._session)
    result = search.execute(scout.SearchCodeIntentArgs(
        behavior="public operation creates state for a later consumer",
        purpose="Find the producer before tracing its consumer",
    ))
    assert not result.expect_correction
    assert queries[0]["query"] == "public operation creates state for a later consumer"
    assert graph._session.actions[-1]["semantic_requery"] is True
    assert graph._session.actions[-1]["graph_mode"] == "function"
    assert graph._session.live_investigation_paths == []


@pytest.mark.parametrize("scenario", ["expand", "abstain", "trace_abstain", "no_edges"])
def test_code_intent_search_auto_expansion_and_fallback(
    monkeypatch, tmp_path, scenario,
):
    sources = {
        "patch.hpp": "void basic_json::patch_inplace() { make_pointer(); }\n",
        "owner.hpp": "class basic_json {};\n",
        "diff.hpp": "void basic_json::diff() { make_pointer(); }\n",
    }
    functions = {
        name: {
            "id": name, "fq_name": f"basic_json::{name}", "def_file": path,
            "def_start_line": 1, "def_end_line": 1,
            "content_hash": hashlib.sha256(sources[path].encode()).hexdigest(),
            "short_intent": "Creates a pointer path",
        }
        for name, path in (("patch_inplace", "patch.hpp"), ("diff", "diff.hpp"))
    }
    owner = {
        "id": "owner", "fq_name": "basic_json", "def_file": "owner.hpp",
        "def_start_line": 1, "def_end_line": 1,
        "content_hash": hashlib.sha256(sources["owner.hpp"].encode()).hexdigest(),
    }
    graph_index = Graph()
    graph_index.add_node("owner", "CLASS", owner)
    for name, function in functions.items():
        graph_index.add_node(name, "FUNC", function)
        if scenario != "no_edges":
            graph_index.add_edge(name, "owner", "IN_CLASS")
    rag = SimpleNamespace(
        func_index=SimpleNamespace(index_path=tmp_path / "func_index.faiss"),
        graph_ctx=SimpleNamespace(
            functions=functions, classes={"owner": owner}, graph=graph_index,
        ),
    )
    monkeypatch.setattr(entity_tools, "get_graph_rag", lambda: rag)
    monkeypatch.setattr(entity_tools, "search_by_intent", lambda **_kwargs: {
        "results": [{**functions["patch_inplace"], "retrieval_role": "semantic_seed"}],
    })

    def neighbors(**kwargs):
        ids = set().union(*(
            graph_index.neighbors(kwargs["entity_id"], edge, kwargs["direction"])
            for edge in kwargs["edge_types"]
        ))
        return {"neighbors": [
            {"id": entity_id, "edge_types": [edge for edge in kwargs["edge_types"]
             if entity_id in graph_index.neighbors(
                 kwargs["entity_id"], edge, kwargs["direction"])]}
            for entity_id in sorted(ids)
        ]}

    monkeypatch.setattr(entity_tools, "get_entity_neighbors", neighbors)
    rank_contexts = []

    def complete(**kwargs):
        payload = json.loads(kwargs["messages"][0]["content"])
        context = json.loads(payload["query"])
        rank_contexts.append(context)
        names = [row["qualified_name_tail"] for row in payload["candidates"]]
        if ((context["mode"] == "graph_search_anchor" and scenario == "abstain")
                or (context["mode"] == "graph_trace" and scenario == "trace_abstain")):
            selected = None
        else:
            selected = next((names.index(name) for name in (
                "basic_json::diff", "basic_json", "basic_json::patch_inplace",
            ) if name in names), None)
        return SimpleNamespace(
            structured_output=scout.StructuralCompanionSelection(
                indexes=[selected] if selected is not None else [],
            ), completion=None,
        )

    completion = SimpleNamespace(
        model="fixture", initialize=lambda **_kwargs: None,
        create_completion=complete,
    )
    session = scout.InvestigationSession(
        issue_text="Custom StringType must work in every public JSON pointer producer."
    )
    session.record("grep -n pointer patch.hpp", sources["patch.hpp"], 0)
    graph = scout.RepositoryGraphQuery(
        code_index=SimpleNamespace(
            _activate_instance=lambda: None,
            _file_repo=SimpleNamespace(get_file_content=lambda path: sources.get(path)),
        ),
        session=session, structural_completion=completion,
    )
    result = scout.SearchCodeIntent(graph=graph, session=session).execute(
        scout.SearchCodeIntentArgs(
            behavior="public operation constructs a JSON pointer",
            purpose="Which other public operation shares the pointer construction?",
        )
    )

    assert not result.expect_correction
    assert "basic_json::patch_inplace" in result.message
    if scenario == "expand":
        assert "basic_json::diff" in result.message, [
            (item.id, item.observation[:300]) for item in session.evidence
        ]
        assert "patch_inplace (patch.hpp:1) --IN_CLASS-->" in result.message
        assert "diff (diff.hpp:1) --IN_CLASS-->" in result.message
        assert "Initial verified semantic anchors" in result.message
        assert len([item for item in session.evidence if item.id.startswith("GRAPH-")]) == 2
        assert result.properties["internal_trace_evidence_ids"] == ["GRAPH-2"]
        assert not session.live_investigation_paths
        assert session.actions[-1]["internal_search_expansion"] is True
    elif scenario == "abstain":
        assert "No issue-relevant source path was selected" in result.message
        assert len([item for item in session.evidence if item.id.startswith("GRAPH-")]) == 1
        assert not session.live_investigation_paths
    elif scenario == "no_edges":
        assert "No issue-relevant source path was selected" in result.message
        assert len([item for item in session.evidence if item.id.startswith("GRAPH-")]) == 2
        assert "No real one-hop relation" in session.evidence[-1].observation
        assert not session.live_investigation_paths
    else:
        assert "No issue-relevant source path was selected" in result.message
        assert len([item for item in session.evidence if item.id.startswith("GRAPH-")]) == 2
        assert not json.loads(session.evidence[-1].observation)[0].get("graph_paths")
        assert not session.live_investigation_paths
    assert all(context["issue_description"] == session.issue_text
               for context in rank_contexts)
    assert rank_contexts[0]["search_purpose"] == (
        "Which other public operation shares the pointer construction?"
    )
    assert rank_contexts[0]["question"].startswith("Which connected producer")


def test_function_search_exposes_traceable_shared_helper_route(monkeypatch, tmp_path):
    sources = {
        "patch.hpp": "void patch_inplace() { concat(); }\n",
        "strings.hpp": "void concat() {}\n",
        "diff.hpp": "void diff() { concat(); }\n",
    }
    functions = {
        name: {
            "id": name, "fq_name": f"basic_json::{name}", "def_file": path,
            "def_start_line": 1, "def_end_line": 1,
            "content_hash": hashlib.sha256(source.encode()).hexdigest(),
        }
        for name, path, source in (
            ("patch_inplace", "patch.hpp", sources["patch.hpp"]),
            ("concat", "strings.hpp", sources["strings.hpp"]),
            ("diff", "diff.hpp", sources["diff.hpp"]),
        )
    }
    monkeypatch.setattr(entity_tools, "get_graph_rag", lambda: SimpleNamespace(
        func_index=SimpleNamespace(index_path=tmp_path / "functions.faiss"),
        graph_ctx=SimpleNamespace(functions=functions, classes={}),
    ))
    monkeypatch.setattr(entity_tools, "search_by_intent", lambda **kwargs: {
        "results": [functions["patch_inplace"], {
            **functions["concat"], "retrieval_role": "structural_companion",
            "graph_paths": [
                {"edges": [{"source": "patch_inplace", "edge": "CALLS", "target": "concat"}]},
                {"edges": [{"source": "diff", "edge": "CALLS", "target": "concat"}]},
            ],
        }],
    })
    graph = scout.RepositoryGraphQuery(
        code_index=SimpleNamespace(
            _activate_instance=lambda: None,
            _file_repo=SimpleNamespace(get_file_content=lambda path: sources.get(path)),
        ),
        session=scout.InvestigationSession(),
    )
    result = scout.SearchCodeIntent(graph=graph, session=graph._session).execute(
        scout.SearchCodeIntentArgs(behavior="patch custom string conversion")
    )
    assert not result.expect_correction
    assert "diff (diff.hpp:1) --CALLS-->" in result.message
    assert "TraceInvestigationPath" in result.message
    assert result.message.index("Connected definitions") < result.message.index(
        "GREF-1 basic_json::patch_inplace"
    )
    assert "void diff() { concat(); }" in result.message
    records = json.loads(graph._session.evidence[-1].observation)
    assert records[0]["scope_frontier"][-1]["symbol"] == "basic_json::diff"
    assert "basic_json::diff" in graph._session.evidence[-1].observation[:2500]
    assert "diff" in records[1]["relation_frontier"][1]
    assert any("diff" in ids for ids in graph._entity_refs.values())
    assert records[0]["source_status"] == "verified"


def test_function_search_flags_multiple_source_definitions_without_claiming_scope():
    action = graph_action()
    view = action._function_relation_frontier([
        {"entity_ref": "GREF-1", "fq_name": "json::patch_inplace",
         "file": "include/json.hpp", "retrieval_role": "semantic_seed"},
        {"entity_ref": "GREF-2", "fq_name": "json::patch_inplace",
         "file": "single_include/json.hpp", "retrieval_role": "semantic_seed"},
    ], None)
    assert "definitions in multiple files" in view
    assert "check whether" in view
    assert "need separate changes" in view


def test_function_search_exposes_owner_as_scope_navigation_lead():
    action = graph_action()
    candidates = [{
        "id": "method", "entity_ref": "GREF-1", "fq_name": "Owner::method",
        "file": "owner.hpp", "retrieval_role": "semantic_seed",
        "graph_paths": [{"edges": [{
            "source": "method", "edge": "IN_CLASS", "target": "owner",
        }]}],
    }]
    ctx = SimpleNamespace(functions={}, classes={
        "owner": {"id": "owner", "fq_name": "Owner", "def_file": "owner.hpp"},
    })
    view = action._function_relation_frontier(candidates, ctx)
    assert candidates[0]["scope_frontier"][0]["kind"] == "class"
    assert "Owner (owner.hpp)" in view
    assert "--IN_CLASS-->" in view


def test_test_intent_search_returns_navigation_leads_not_a_test_oracle():
    session = scout.InvestigationSession()
    search = scout.SearchTestIntent(
        repository=SimpleNamespace(), session=session, index_name=None,
    )
    search._retriever = SimpleNamespace(retrieve=lambda *args: SimpleNamespace(
        candidates=[SimpleNamespace(
            test_name="keeps the state", path="tests/state.cpp",
            locator="tests/state.cpp:12", retrieval_ranks={"test_intent": 1},
            snippet="assert(state);",
        )],
    ))
    args = scout.SearchTestIntentArgs(behavior="public operation retains state")
    result = search.execute(args)
    assert not result.expect_correction
    assert "Inspect or run a candidate" in result.message
    assert session.actions[-1]["action"] == "SearchTestIntent"
    assert session.evidence[-1].id == "SEARCH-1"
    assert search.execute(args).expect_correction
    assert len(session.actions) == 1


def test_graph_search_is_not_forced_before_scout_actions(monkeypatch):
    calls = []

    def search(**kwargs):
        calls.append(kwargs)
        return {"results": [{"func_id": "FUNC:1", "short_intent": "format ranges"}]}

    monkeypatch.setattr(entity_tools, "search_by_intent", search)
    action = graph_action()
    node = SimpleNamespace(observation=None, is_terminal=lambda: True)
    messages = []
    monkeypatch.setattr(scout, "Node", SimpleNamespace(stub=lambda **kw: (
        messages.append(kw["user_message"]), node,
    )[1]))
    monkeypatch.setattr(scout, "FileContext", lambda **kw: None)
    monkeypatch.setattr(scout, "build_poc_investigation_task", lambda *args: "POC task")
    investigator = scout.ContainerRepositoryInvestigator(
        agent=SimpleNamespace(), session=action._session,
        repository=SimpleNamespace(), docker_env=SimpleNamespace(),
        investigation_actions=[action], submit_action=SimpleNamespace(),
        max_steps=60, mode="poc",
    )
    monkeypatch.setattr(investigator, "_run_investigation_step", lambda current: node)
    report = SimpleNamespace()
    monkeypatch.setattr(investigator, "_best_effort_report", lambda reason: report)
    result, evidence, actions = investigator.investigate_contrastively(
        "Range formatting fails", SimpleNamespace(), {},
    )
    assert result is report
    assert evidence == actions == calls == []
    assert investigator.steps_used == 1
    assert "POC task" in messages[0]


def test_repository_graph_has_no_graph_specific_call_cap(monkeypatch):
    monkeypatch.setattr(entity_tools, "search_by_intent", lambda **kwargs: {
        "results": [{
            "func_id": "FUNC:" + kwargs["query"],
            "short_intent": "candidate for " + kwargs["query"],
        }],
    })
    action = graph_action()

    for index in range(5):
        result = action.execute(scout.RepositoryGraphQueryArgs(
            mode="function",
            query=f"distinct relation question {index}",
        ))
        assert not result.expect_correction

    assert len(action._session.actions) == 5
    assert all(not item.get("graph_budget_exhausted")
               for item in action._session.actions)


def test_same_trace_anchor_accepts_a_new_causal_question(monkeypatch):
    graph = SimpleNamespace(neighbors=lambda entity_id, edge, direction: (
        {"consumer"} if entity_id == "anchor" and edge == "CALLS"
        and direction == "out" else set()
    ))
    monkeypatch.setattr(entity_tools, "get_graph_rag", lambda: SimpleNamespace(
        graph_ctx=SimpleNamespace(
            functions={
                "anchor": {"id": "anchor", "fq_name": "Owner::anchor"},
                "consumer": {"id": "consumer", "fq_name": "Owner::consumer"},
            },
            classes={}, graph=graph,
        ),
    ))
    monkeypatch.setattr(entity_tools, "get_entity_neighbors", lambda **kwargs: {
        "neighbors": [{"id": "consumer", "edge_types": ["CALLS"]}],
    })
    action = graph_action()
    first = scout.RepositoryGraphQueryArgs(
        mode="trace", entity_id="anchor", direction="out",
        edge_types=["CALLS"], target="Which consumer receives the state?",
    )
    second = first.model_copy(update={"target": "Does the consumer expose the failure?"})
    assert not action.execute(first).expect_correction
    assert not action.execute(second).expect_correction
    assert action.execute(first).expect_correction


@pytest.mark.parametrize("mode,kwargs,backend,key", [
    ("function", {"query": "format"}, "search_by_intent", "results"),
    ("intent", {"query": "format"}, "list_features", "features"),
    ("neighbors", {"entity_id": "bad-id"}, "get_entity_neighbors", "neighbors"),
    ("symbol", {"symbol_name": "format"}, "search_symbol", "entities"),
])
def test_backend_errors_are_recorded_failures_not_empty_success(monkeypatch, mode, kwargs, backend, key):
    monkeypatch.setattr(entity_tools, backend, lambda **kw: {"error": "index unavailable", key: []})
    action = graph_action()
    result = action.execute(scout.RepositoryGraphQueryArgs(mode=mode, **kwargs))
    assert result.expect_correction
    assert result.properties["query_failed"]
    assert action._session.actions[0]["exit_code"] == 1
    assert "index unavailable" in action._session.evidence[0].observation


def test_empty_neighbors_remain_valid_and_missing_id_is_retryable(monkeypatch):
    monkeypatch.setattr(entity_tools, "get_entity_neighbors", lambda **kw: {"neighbors": []})
    action = graph_action()
    failed = action.execute(scout.RepositoryGraphQueryArgs(mode="neighbors"))
    assert failed.expect_correction
    valid = action.execute(scout.RepositoryGraphQueryArgs(mode="neighbors", entity_id="FUNC:1"))
    assert not valid.expect_correction
    assert valid.properties["candidate_count"] == 0
    assert [a["exit_code"] for a in action._session.actions] == [1, 0]


def test_missing_embedding_provenance_does_not_produce_semantic_candidates(monkeypatch):
    monkeypatch.setattr(embedding_manifest, "load_embedding_manifest", lambda path: {})

    def unexpected_search(**kwargs):
        raise AssertionError("unverified vectors must not be queried")

    monkeypatch.setattr(entity_tools, "search_by_intent", unexpected_search)
    action = graph_action()
    result = action.execute(scout.RepositoryGraphQueryArgs(mode="function", query="format ranges"))
    assert result.expect_correction
    assert "embedding provenance is missing" in result.message
    assert action._session.actions[0]["exit_code"] == 1


@pytest.mark.parametrize("enabled", [True, False])
def test_scout_graph_toggle_changes_tools_and_guidance_only(monkeypatch, enabled):
    monkeypatch.setattr(scout, "get_model_config", lambda name: {"model": "unused"})
    monkeypatch.setattr(scout.BaseCompletionModel, "create", lambda **kw: SimpleNamespace(
        response_format=scout.LLMResponseFormat.TOOLS,
        initialize=lambda **kw: None,
    ))
    monkeypatch.setattr(scout, "ActionAgent", lambda **kw: SimpleNamespace(**kw))
    investigator = scout.ContainerRepositoryInvestigator.create(
        repository=SimpleNamespace(), docker_env=SimpleNamespace(work_dir="/repo"),
        code_index=SimpleNamespace(), model="unused", mode="legacy", max_steps=60,
        enable_graph_search=enabled,
    )
    names = {type(a).__name__ for a in investigator.investigation_actions}
    agent_names = {type(a).__name__ for a in investigator.agent.actions}
    assert ("RepositoryGraphQuery" in names) is enabled
    assert "RepositoryGraphQuery" not in agent_names
    assert ("TraceInvestigationPath" in agent_names) is enabled
    assert ("SearchCodeIntent" in agent_names) is enabled
    assert ("SearchTestIntent" in agent_names) is enabled
    assert ("Repository relation graph" in investigator.agent.system_prompt) is enabled
    guidance = " ".join(investigator.agent.system_prompt.split())
    assert (
        "must call either SearchCodeIntent or TraceInvestigationPath before "
        "submitting the investigation report" in guidance
    ) is enabled
    assert "You decide whether and when these tools" not in guidance
    assert "skip the graph and state the reason" not in guidance
    assert ("after\ngrep locates a fault" in investigator.agent.system_prompt) is enabled
    assert ("A list of direct callers cannot answer" in investigator.agent.system_prompt) is enabled
    assert ('edge_types=["IN_CLASS"]' in investigator.agent.system_prompt) is enabled
    assert ("`equivalent_source_forms`" in investigator.agent.system_prompt) is enabled
    assert "ReadOnlySearch" in names
    assert investigator.max_steps == 60
    assert investigator.code_index is not None  # Test-anchor retrieval unchanged.


def test_poc_scout_keeps_source_backed_scope_suspicions(monkeypatch):
    monkeypatch.setattr(scout, "get_model_config", lambda name: {"model": "unused"})
    monkeypatch.setattr(scout.BaseCompletionModel, "create", lambda **kw: SimpleNamespace(
        response_format=scout.LLMResponseFormat.TOOLS,
        initialize=lambda **kw: None,
    ))
    monkeypatch.setattr(scout, "ActionAgent", lambda **kw: SimpleNamespace(**kw))
    investigator = scout.ContainerRepositoryInvestigator.create(
        repository=SimpleNamespace(), docker_env=SimpleNamespace(work_dir="/repo"),
        code_index=None, model="unused", mode="poc", max_steps=60,
        enable_graph_search=False,
    )
    prompt = " ".join(investigator.agent.system_prompt.split())
    assert "Keep source-backed suspected siblings" in prompt
    assert "not decide the final edit set" in prompt
    assert "No direct call to the faulty function" in prompt
    assert "Do not turn suspicion into a requirement" in prompt
    assert "A proxy that changes the Issue's input representation" in prompt


def test_graph_tool_schemas_explain_when_to_call_them():
    code = scout.SearchCodeIntentArgs.tool_schema()["function"]
    trace = scout.TraceInvestigationPathArgs.tool_schema()["function"]
    test = scout.SearchTestIntentArgs.tool_schema()["function"]

    assert "after grep found the" in code["description"]
    assert "related producer, consumer, or sibling" in code["description"]
    assert "production functions, not tests" in code["description"]
    assert "Use when one observed graph or source anchor" in trace["description"]
    assert "independent public operations" in trace["description"]
    assert "use ReadOnlySearch" in trace["description"]
    assert "even without a prior semantic graph search" in trace["parameters"]["properties"]["source_path"]["description"]
    assert "not a passing oracle" in test["description"]
    assert "unknown fixture" in test["description"]


def test_scout_report_schema_accepts_uncertain_source_backed_scope_leads():
    boundary = scout.CandidateRepairBoundary(
        id="BOUNDARY-1",
        anchor_operation="Widget::decode",
        candidate_public_forms=["Widget::decode", "Widget::encode"],
        evidence_ids=["SEARCH-2", "GRAPH-1"],
        rationale=(
            "Both use the custom representation; inspect encode's body "
            "before deciding whether its path construction also fails."
        ),
    )
    assert boundary.candidate_public_forms[-1] == "Widget::encode"
    description = scout.SubmitInvestigationReportArgs.tool_schema()[
        "function"
    ]["parameters"]["properties"]["candidate_boundaries"]["description"]
    assert "membership remains uncertain" in description


def test_contrastive_scout_prompt_resolves_contracts_on_the_exact_receiver(
    monkeypatch,
):
    monkeypatch.setattr(scout, "get_model_config", lambda name: {"model": "unused"})
    monkeypatch.setattr(
        scout.BaseCompletionModel,
        "create",
        lambda **kw: SimpleNamespace(
            response_format=scout.LLMResponseFormat.TOOLS,
            initialize=lambda **kw: None,
        ),
    )
    monkeypatch.setattr(scout, "ActionAgent", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(scout, "available_dynamic_evidence_tools", lambda env: set())

    investigator = scout.ContainerRepositoryInvestigator.create(
        repository=SimpleNamespace(),
        docker_env=SimpleNamespace(work_dir="/repo"),
        code_index=SimpleNamespace(),
        model="unused",
        mode="contrastive",
        max_steps=60,
        enable_graph_search=True,
    )

    prompt = investigator.agent.system_prompt
    assert "competing repair interpretations coequal" in prompt
    assert "exact public operation, input, and error mode" in prompt
    assert "receiver actually produced by the public ingress" in prompt
    assert "A working inner method" in prompt
    assert "or alternate getter" in prompt
    assert "template/SFINAE failure" in prompt
    assert "zero-edge result as proof of absence" in prompt
    assert "current-environment\npass alone" in prompt
    assert investigator.session.require_input_path_check is False
    assert "Do not fill\nthis record merely to satisfy a template" in prompt
    assert "independent evidence reviewer decides" in prompt


def test_poc_scout_keeps_graph_but_drops_required_test_pair(monkeypatch):
    monkeypatch.setattr(scout, "get_model_config", lambda name: {"model": "unused"})
    monkeypatch.setattr(
        scout.BaseCompletionModel,
        "create",
        lambda **kw: SimpleNamespace(
            response_format=scout.LLMResponseFormat.TOOLS,
            initialize=lambda **kw: None,
        ),
    )
    monkeypatch.setattr(scout, "ActionAgent", lambda **kw: SimpleNamespace(**kw))
    investigator = scout.ContainerRepositoryInvestigator.create(
        repository=SimpleNamespace(),
        docker_env=SimpleNamespace(work_dir="/repo"),
        code_index=SimpleNamespace(),
        model="unused",
        mode="poc",
        enable_graph_search=True,
    )

    assert investigator.mode == "poc"
    assert "baseline-failing POC" in investigator.agent.system_prompt
    assert "never construct a mandatory pass/fail pair" in (
        investigator.agent.system_prompt
    )
    assert "Repository relation graph" in investigator.agent.system_prompt
    assert "SearchCodeIntent" in investigator.agent.system_prompt
    assert "must call either SearchCodeIntent or TraceInvestigationPath" in (
        " ".join(investigator.agent.system_prompt.split())
    )
    assert "SearchTestIntent" in investigator.agent.system_prompt
    assert any(
        type(action).__name__ == "TraceInvestigationPath"
        for action in investigator.agent.actions
    )


def test_scout_reserves_submission_and_correction_without_extra_steps():
    from INTENT2FIX.node import Node
    from INTENT2FIX.actions.schema import Observation
    from INTENT2FIX.reproduce.intent_schema import IssueEvidenceFrame

    submit = SimpleNamespace(name='SubmitInvestigationReport')
    search = SimpleNamespace(name='ReadOnlySearch')

    class PersistentAgent:
        def __init__(self):
            self.actions = [search, submit]
            self.calls = []
            self.submissions = 0

        def set_actions(self, actions):
            self.actions = actions

        def run(self, node):
            node.action_steps = [SimpleNamespace(observation=Observation(message='pending'))]
            self.calls.append([action.name for action in self.actions])
            if search in self.actions:
                node.observation = Observation(message='More source inspected')
                return
            self.submissions += 1
            if self.submissions == 1:
                node.observation = Observation(message='Correct the report citation', expect_correction=True)
                return
            report = scout.DeveloperAnswer(status='partial', answer='The observed result still differs; cause unresolved.')
            node.observation = Observation(message='Report submitted', terminal=True,
                properties={'investigation_report': report.model_dump()})
            node.terminal = True

    agent = PersistentAgent()
    investigator = scout.ContainerRepositoryInvestigator(
        agent=agent, session=scout.InvestigationSession(), repository=SimpleNamespace(),
        docker_env=SimpleNamespace(), investigation_actions=[], submit_action=submit, max_steps=6)
    closing = PersistentAgent()
    closing.actions = [submit]
    closing.calls = agent.calls
    investigator._submission_agent = closing
    report, _, _ = investigator.investigate_contrastively('Fix output', IssueEvidenceFrame(), {})
    assert report.status == 'partial'
    assert investigator.steps_used == 6
    assert agent.calls[:4] == [['ReadOnlySearch', 'SubmitInvestigationReport']] * 4
    assert agent.calls[4:] == [['SubmitInvestigationReport']] * 2
    assert agent.actions == [search, submit]
    assert investigator._current_node.is_terminal()

    # An exception during the closing call must not restrict a later round.
    investigator.steps_used = 5
    def fail(node):
        raise RuntimeError('provider unavailable')
    closing.run = fail
    with pytest.raises(RuntimeError, match='provider unavailable'):
        investigator._run_investigation_step(Node.stub())
    assert agent.actions == [search, submit]


def test_budget_prompt_matches_the_actual_submission_reserve():
    investigator = scout.ContainerRepositoryInvestigator(
        agent=SimpleNamespace(), session=scout.InvestigationSession(),
        repository=SimpleNamespace(), docker_env=SimpleNamespace(),
        investigation_actions=[], submit_action=SimpleNamespace(), max_steps=60,
    )
    assert investigator._submission_reserve == 5
    assert "final 5 steps are reserved" in investigator._plan_budget_prompt()



def test_submission_completion_has_its_own_schema_and_keeps_original_configuration(monkeypatch):
    from INTENT2FIX.agent.agent import ActionAgent
    from INTENT2FIX.completion.base import BaseCompletionModel, LLMResponseFormat
    from INTENT2FIX.node import Node

    session = scout.InvestigationSession()
    submit = scout.SubmitInvestigationReport(session=session)
    search = scout.ReadOnlySearch(docker_env=SimpleNamespace(), session=session)
    completion = BaseCompletionModel.create(
        model="fixture", response_format=LLMResponseFormat.TOOLS, model_api_key="fixture-only-key")
    agent = ActionAgent(completion=completion, actions=[search, submit],
                        system_prompt="Investigate", use_few_shots=False)
    investigator = scout.ContainerRepositoryInvestigator(
        agent=agent, session=session, repository=SimpleNamespace(), docker_env=SimpleNamespace(),
        investigation_actions=[search], submit_action=submit, max_steps=6)
    investigator.steps_used = 4
    current = Node.stub(user_message="Existing evidence")
    seen = []
    monkeypatch.setattr(investigator, '_run_with_no_action_retries', lambda selected, node: seen.append(selected) or node)
    assert investigator._run_investigation_step(current) is current
    closing = seen[0]
    assert closing.actions == [submit]
    assert closing.completion.response_schema == [scout.SubmitInvestigationReportArgs]
    assert agent.actions == [search, submit]
    assert agent.completion.response_schema == [scout.ReadOnlySearchArgs, scout.SubmitInvestigationReportArgs]
    assert closing.completion.model_api_key == completion.model_api_key
    assert closing.completion.model == completion.model
    assert closing.message_generator is agent.message_generator


def test_long_scout_reserves_five_steps_and_stall_forces_submission(monkeypatch):
    from INTENT2FIX.node import Node

    submit = SimpleNamespace(name="SubmitInvestigationReport")
    ordinary = SimpleNamespace(name="ReadOnlySearch")
    selected = []
    agent = SimpleNamespace(
        completion=SimpleNamespace(), actions=[ordinary, submit],
        system_prompt="Investigate", message_generator=SimpleNamespace(),
        disable_thoughts=True,
    )
    investigator = scout.ContainerRepositoryInvestigator(
        agent=agent, session=scout.InvestigationSession(), repository=SimpleNamespace(),
        docker_env=SimpleNamespace(), investigation_actions=[], submit_action=submit,
        max_steps=60,
    )
    closing = SimpleNamespace(name="closing")
    investigator._submission_agent = closing
    monkeypatch.setattr(investigator, "_run_with_no_action_retries",
                        lambda chosen, node: selected.append(chosen) or node)
    investigator.steps_used = 55
    investigator._run_investigation_step(Node.stub())
    assert selected[-1] is closing
    investigator.session.record("sed -n '1p' x.cc", "int x;", 0)
    investigator._last_evidence_count = 1
    investigator.steps_used = 20
    investigator._last_evidence_step = 14
    investigator._run_investigation_step(Node.stub())
    assert selected[-1] is closing
