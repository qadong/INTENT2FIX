import json
from types import SimpleNamespace

import faiss
import numpy as np
import pytest

from INTENT2FIX.index import entity_code_index
from INTENT2FIX.index.entity_code_index import EntityCodeIndex, _matches_file_pattern
from code_intent.code_intent.embedding_manifest import (
    load_embedding_manifest,
    write_embedding_manifest_entry,
)
from code_intent.code_intent import search_api
from code_intent.code_intent.graph_builder import (
    EDGE_CALLS,
    EDGE_DEP_CALL,
    EDGE_DEP_MEMBER,
    EDGE_DEP_TEMPLATE,
    EDGE_IN_CLASS,
    Graph,
    GraphContext,
    NODE_CLASS,
    NODE_FUNC,
)
from code_intent.code_intent.search_api import (
    CodeIntentRAG,
    _FunctionHit,
    _inner_product_similarity,
    _VectorIndex,
)


@pytest.mark.parametrize(
    ("path", "pattern", "expected"),
    [
        ("src/catch2/internal/catch_textflow.cpp", "src/catch2/internal/*", True),
        ("src/catch2/internal/catch_textflow.cpp", "src/catch2/internal", True),
        ("src/catch2/catch_assertion_result.hpp", "src/catch2/internal/*", False),
        ("tests/TextFlow.tests.cpp", "tests/**/*.cpp", True),
        ("include/fmt/core.h", "include/fmt/core.h", True),
    ],
)
def test_matches_file_pattern(path, pattern, expected):
    assert _matches_file_pattern(path, pattern) is expected


def test_semantic_search_strictly_filters_after_overfetch(monkeypatch, tmp_path):
    monkeypatch.setenv("CODE_INTENT_INDEX_STRICT_PREFLIGHT", "0")
    captured = {}

    def fake_search_by_intent(*, query, top_k):
        captured.update(query=query, top_k=top_k)
        return {
            "results": [
                {
                    "id": "wrong",
                    "name": "wrong",
                    "file": "src/unrelated.cpp",
                    "def_start_line": 1,
                    "score": 0.99,
                },
                {
                    "id": "right",
                    "name": "right",
                    "file": "src/catch2/internal/right.cpp",
                    "def_start_line": 7,
                    "score": 0.50,
                },
            ]
        }

    monkeypatch.setattr(entity_code_index, "search_by_intent", fake_search_by_intent)
    monkeypatch.setattr(entity_code_index, "set_current_instance", lambda _: None)
    index = EntityCodeIndex(index_name="test", index_store_dir=str(tmp_path))

    response = index.semantic_search(
        query="target behavior",
        file_pattern="src/catch2/internal/*",
        max_results=1,
    )

    assert captured == {"query": "target behavior", "top_k": 50}
    assert [hit.file_path for hit in response.hits] == [
        "src/catch2/internal/right.cpp"
    ]


def test_inner_product_similarity_preserves_faiss_order():
    assert _inner_product_similarity(0.9) > _inner_product_similarity(0.2)
    assert _inner_product_similarity(2.0) == 1.0
    assert _inner_product_similarity(-2.0) == -1.0


def test_vector_index_rejects_query_dimension_mismatch(tmp_path):
    index_path = tmp_path / "func_index.faiss"
    meta_path = tmp_path / "func_meta.jsonl"
    index = faiss.IndexFlatIP(3)
    index.add(np.array([[1.0, 0.0, 0.0]], dtype="float32"))
    faiss.write_index(index, str(index_path))
    meta_path.write_text(json.dumps({"index": 0, "func_id": "f"}) + "\n")

    vector_index = _VectorIndex(index_path, meta_path, "func_id")
    with pytest.raises(ValueError, match="query=2, index=3"):
        vector_index.search(np.array([1.0, 0.0], dtype="float32"), 1)


def test_embedding_manifest_merges_index_entries(tmp_path):
    write_embedding_manifest_entry(
        tmp_path, "functions", model="voyage-2", dimension=1024
    )
    write_embedding_manifest_entry(
        tmp_path, "features", model="voyage-2", dimension=1024
    )

    manifest = load_embedding_manifest(tmp_path)
    assert manifest["indexes"]["functions"]["metric"] == "inner_product"
    assert manifest["indexes"]["features"]["dimension"] == 1024


def test_manifestless_semantic_index_is_rejected(monkeypatch, tmp_path):
    index = faiss.IndexFlatIP(3)
    index.add(np.array([[1.0, 0.0, 0.0]], dtype="float32"))
    for kind, id_key in (("func", "func_id"), ("feature", "feature_id")):
        faiss.write_index(index, str(tmp_path / f"{kind}_index.faiss"))
        (tmp_path / f"{kind}_meta.jsonl").write_text(
            json.dumps({"index": 0, id_key: kind}) + "\n"
        )

    monkeypatch.setattr(search_api, "build_graph", lambda _: object())
    monkeypatch.setenv("CODE_INTENT_EMBED_DIM", "3")
    monkeypatch.setattr(
        search_api, "embed_query", lambda _: np.array([1.0, 0.0, 0.0])
    )
    rag = CodeIntentRAG(tmp_path)

    assert rag.graph_ctx is not None
    with pytest.raises(RuntimeError, match="embedding provenance"):
        rag._encode_query("query")


def test_graph_expansion_adds_callers_callees_and_class_siblings():
    graph = Graph()
    functions = {
        name: {"id": name}
        for name in ("seed", "caller", "callee", "dependent", "sibling")
    }
    for name, data in functions.items():
        graph.add_node(name, NODE_FUNC, data)
    graph.add_node("class", NODE_CLASS, {"id": "class"})
    graph.add_edge("caller", "seed", EDGE_CALLS)
    graph.add_edge("seed", "callee", EDGE_CALLS)
    graph.add_edge("seed", "dependent", EDGE_DEP_CALL)
    graph.add_edge("seed", "class", EDGE_IN_CLASS)
    graph.add_edge("sibling", "class", EDGE_IN_CLASS)

    rag = CodeIntentRAG.__new__(CodeIntentRAG)
    rag.graph_ctx = GraphContext(
        graph=graph,
        functions=functions,
        features={},
        classes={"class": {"id": "class"}},
        files={},
    )

    scores = rag._expand_graph_candidates([_FunctionHit("seed", 0.8)], seed_limit=1)

    assert scores["caller"] == pytest.approx(0.8 * rag.GRAPH_CALL_DECAY)
    assert scores["callee"] == pytest.approx(0.8 * rag.GRAPH_CALL_DECAY)
    assert scores["dependent"] == pytest.approx(0.8 * rag.GRAPH_CALL_DECAY)
    assert scores["sibling"] == pytest.approx(0.8 * rag.GRAPH_SIBLING_DECAY)


def test_rank_fusion_uses_same_score_scale_for_all_candidates():
    rag = CodeIntentRAG.__new__(CodeIntentRAG)
    base = [("first", 0.9), ("second", 0.8), ("outside", 0.7)]
    reranked = [
        {"index": 1, "score": 0.9},
        {"index": 0, "score": 0.8},
    ]

    scores = rag._rank_fusion_scores(base, reranked, ["first", "second"])

    assert scores["first"] > scores["second"] > scores["outside"]
    assert scores["second"] - 1 / (rag.RRF_K + 2) > 0
    assert max(scores.values()) < 0.1


@pytest.mark.parametrize("rerank_first", [False, True])
def test_rerank_first_keeps_relevant_candidate_in_final_results(monkeypatch, rerank_first):
    ids = [f"macro-{i}" for i in range(8)] + ["range-format"]
    rag = CodeIntentRAG.__new__(CodeIntentRAG)
    rag.graph_ctx = SimpleNamespace(
        functions={fid: {"file": "include/format.h", "short_intent": fid} for fid in ids},
        features={}, func_to_features={}, feature_to_funcs={},
        graph=SimpleNamespace(neighbors=lambda *args, **kwargs: set()),
    )
    rag._encode_query = lambda _: np.zeros(3)
    rag._search_classes_with_vector = lambda *args, **kwargs: []
    rag._search_features_with_vector = lambda *args, **kwargs: []
    rag._search_functions_with_vector = lambda *args: [
        _FunctionHit(fid, 0.8 - i * 0.01) for i, fid in enumerate(ids)
    ]
    calls = []

    def rerank(query, documents, top_k, **kwargs):
        calls.append((len(documents), top_k))
        return [{"index": i, "score": 0.9 - rank * 0.01}
                for rank, i in enumerate([8, *range(8)])]

    monkeypatch.setattr(search_api, "rerank_documents", rerank)
    result = rag.search_code("format nested characters", top_k_funcs=5, rerank_first=rerank_first)

    assert calls == [(9, rag.RERANK_TOP_K)]  # One existing rerank; no added model call.
    assert len(result) == 5
    if rerank_first:
        assert result[0]["func_id"] == "range-format"
        assert result[0]["score"] == pytest.approx(0.9)
    else:
        assert "range-format" not in [row["func_id"] for row in result]


def test_post_rerank_expansion_keeps_seeds_and_crosses_class_membership(monkeypatch):
    rag = CodeIntentRAG.__new__(CodeIntentRAG)
    graph = Graph()
    functions = {fid: {"signature": fid, "file": "x.h", "start_line": line,
                       "code": "body", "short_intent": fid}
                 for line, fid in enumerate(['seed', 'second', 'z_prepare', 'consumer', 'callee'], 1)}
    for fid, data in functions.items():
        graph.add_node(fid, NODE_FUNC, data)
    graph.add_node('owner', NODE_CLASS, {})
    graph.add_edge('seed', 'owner', EDGE_IN_CLASS)
    graph.add_edge('z_prepare', 'owner', EDGE_IN_CLASS)
    graph.add_edge('consumer', 'seed', EDGE_CALLS)
    graph.add_edge('seed', 'callee', EDGE_DEP_MEMBER)
    rag.graph_ctx = SimpleNamespace(graph=graph, functions=functions)
    calls = []

    def rerank(query, documents, top_k, **kwargs):
        calls.append((query, [json.loads(doc) for doc in documents], top_k))
        return [{'index': i, 'score': 0.8} for i in reversed(range(len(documents)))]

    monkeypatch.setattr(search_api, 'rerank_documents', rerank)
    ranked, metadata = rag._structural_companions('prepare consumer', [('seed', 1), ('second', .9), ('other', .8)], 5)
    assert [fid for fid, _ in ranked][:2] == ['seed', 'second']
    assert {fid for fid, _ in ranked} == set(functions)
    assert len(calls) == 1 and calls[0][2] == 3
    assert len(metadata['z_prepare']['graph_paths'][0]['edges']) == 2
    assert metadata['consumer']['graph_paths'][0]['edges'] == [
        {'source': 'consumer', 'edge': EDGE_CALLS, 'target': 'seed'}]
    assert metadata['callee']['graph_paths'][0]['edges'] == [
        {'source': 'seed', 'edge': EDGE_DEP_MEMBER, 'target': 'callee'}]
    assert all(doc['connections'][0]['seed'] == 'seed' for doc in calls[0][1])


def test_structural_pool_is_bounded_by_relevance_not_usr_order(monkeypatch):
    rag = CodeIntentRAG.__new__(CodeIntentRAG)
    graph = Graph()
    functions = {fid: {'signature': 'restore nested state' if fid == 'zz' else 'unrelated',
                       'file': 'x.h', 'start_line': n}
                 for n, fid in enumerate(['seed', 'second', *[f'a{i}' for i in range(30)], 'zz'], 1)}
    graph.add_node('owner', NODE_CLASS, {})
    for fid, data in functions.items():
        graph.add_node(fid, NODE_FUNC, data)
        graph.add_edge(fid, 'owner', EDGE_IN_CLASS)
    rag.graph_ctx = SimpleNamespace(graph=graph, functions=functions)
    seen = []

    def rerank(query, documents, top_k, **kwargs):
        seen.extend(json.loads(d) for d in documents)
        return [{'index': 0, 'score': 1}]

    monkeypatch.setattr(search_api, 'rerank_documents', rerank)
    result, _ = rag._structural_companions('restore nested state', [('seed', 1), ('second', .8)], 5)
    assert len(seen) == 10 and seen[0]['companion'] == 'restore nested state'
    assert result[2][0] == 'zz'
    monkeypatch.setattr(search_api, 'rerank_documents', lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('unavailable')))
    original = [('seed', 1), ('second', .8)]
    result, metadata = rag._structural_companions('restore nested state', original, 5)
    assert result == original and metadata['seed']['structural_status'] == 'rerank_failed'


def test_structural_pool_admits_executable_paths_before_overload_families():
    rag = CodeIntentRAG.__new__(CodeIntentRAG)
    graph = Graph()
    functions = {
        'seed': {'fq_name': 'api::json::operator[]', 'file': 'json.h',
                 'start_line': 100, 'code': 'return set_parent(value);'},
        'second': {'fq_name': 'api::ordered_map::operator[]', 'file': 'map.h',
                   'start_line': 20, 'code': 'return emplace(key, value);'},
        'set_parent': {'fq_name': 'api::json::set_parent', 'file': 'json.h',
                       'start_line': 40, 'code': 'set_parents();'},
        'set_parents': {'fq_name': 'api::json::set_parents', 'file': 'json.h',
                        'start_line': 30, 'code': 'child.parent = this;'},
        'emplace': {'fq_name': 'api::ordered_map::emplace', 'file': 'map.h',
                    'start_line': 10, 'code': 'storage.emplace_back(value);'},
        **{
            f'overload_{index}': {
                'fq_name': 'api::json::operator[]', 'file': 'json.h',
                'start_line': 200 + index, 'code': 'return adjacent_value;'
            }
            for index in range(20)
        },
    }
    for function_id, data in functions.items():
        graph.add_node(function_id, NODE_FUNC, data)
    graph.add_node('json_owner', NODE_CLASS, {})
    graph.add_node('map_owner', NODE_CLASS, {})
    for function_id in ('seed', 'set_parent', 'set_parents', *[
        f'overload_{index}' for index in range(20)
    ]):
        graph.add_edge(function_id, 'json_owner', EDGE_IN_CLASS)
    for function_id in ('second', 'emplace'):
        graph.add_edge(function_id, 'map_owner', EDGE_IN_CLASS)
    graph.add_edge('seed', 'set_parent', EDGE_CALLS)
    graph.add_edge('second', 'emplace', EDGE_CALLS)
    graph.add_edge('set_parent', 'set_parents', EDGE_DEP_CALL)
    rag.graph_ctx = SimpleNamespace(graph=graph, functions=functions)
    seen = []

    def rerank(query, documents, top_k, **kwargs):
        seen.extend(json.loads(document) for document in documents)
        return [{'index': index, 'score': 1 - index / 10}
                for index in range(top_k)]

    rag._structural_companions(
        'ordered json insertion must refresh child parent pointers',
        [('seed', 1), ('second', .9)], 5, reranker=rerank,
    )

    names = [document['companion'] for document in seen]
    assert {'api::json::set_parent', 'api::json::set_parents',
            'api::ordered_map::emplace'} <= set(names)
    assert names.count('api::json::operator[]') <= 2
    assert any(
        [edge['edge'] for edge in connection['path']] == [
            EDGE_CALLS, EDGE_DEP_CALL,
        ]
        for document in seen if document['companion'] == 'api::json::set_parents'
        for connection in document['connections']
    )


def test_structural_pool_reserves_one_relevant_api_sibling():
    rag = CodeIntentRAG.__new__(CodeIntentRAG)
    graph = Graph()
    functions = {
        'seed': {'fq_name': 'api::result::is', 'file': 'result.h', 'start_line': 1},
        'second': {
            'fq_name': 'api::result::operator bool', 'file': 'result.h',
            'start_line': 2,
        },
        'named': {
            'fq_name': 'api::result::is_bool', 'file': 'result.h',
            'start_line': 3,
        },
        'unrelated_sibling': {
            'fq_name': 'api::result::reset', 'file': 'result.h',
            'start_line': 4,
        },
        **{
            f'direct_{index}': {
                'fq_name': f'api::element::predicate_{index}',
                'file': 'element.h', 'start_line': index + 1,
            }
            for index in range(9)
        },
    }
    for function_id, data in functions.items():
        graph.add_node(function_id, NODE_FUNC, data)
    graph.add_node('owner', NODE_CLASS, {})
    for function_id in ('seed', 'second', 'named', 'unrelated_sibling'):
        graph.add_edge(function_id, 'owner', EDGE_IN_CLASS)
    for index in range(9):
        graph.add_edge('seed', f'direct_{index}', EDGE_CALLS)
    rag.graph_ctx = SimpleNamespace(graph=graph, functions=functions)
    seen = []

    def rerank(query, documents, top_k, **kwargs):
        seen.extend(json.loads(document) for document in documents)
        return []

    rag._structural_companions(
        'keep the is bool predicate consistent',
        [('seed', 1), ('second', .9)], 5, reranker=rerank,
    )

    names = {document['companion'] for document in seen}
    assert len(seen) == 10
    assert 'api::result::is_bool' in names
    assert 'api::result::reset' not in names
    assert {
        f'api::element::predicate_{index}' for index in range(9)
    } <= names


def test_structural_pool_crosses_template_classes_and_prefers_api_family():
    rag = CodeIntentRAG.__new__(CodeIntentRAG)
    graph = Graph()
    functions = {
        'seed': {'fq_name': 'api::result::is', 'signature': 'bool is<T>()',
                 'file': 'inline.h', 'start_line': 10, 'short_intent': 'check type'},
        'second': {'fq_name': 'api::result::operator bool', 'signature': 'operator bool()',
                   'file': 'inline.h', 'start_line': 50, 'short_intent': 'convert to bool'},
        'named': {'fq_name': 'api::result::is_bool', 'signature': 'bool is_bool()',
                  'file': 'inline.h', 'start_line': 20, 'short_intent': 'check boolean'},
        'named_array': {'fq_name': 'api::result::is_array', 'signature': 'bool is_array()',
                        'file': 'inline.h', 'start_line': 21,
                        'short_intent': 'check array'},
        'noise': {'fq_name': 'api::result::get_bool', 'signature': 'bool get_bool()',
                  'file': 'inline.h', 'start_line': 11, 'short_intent': 'get boolean'},
        'underlying': {'fq_name': 'api::element::is', 'signature': 'bool is<T>()',
                       'file': 'element.h', 'start_line': 100, 'short_intent': 'check type'},
        'named_underlying': {'fq_name': 'api::element::is_bool', 'signature': 'bool is_bool()',
                             'file': 'element.h', 'start_line': 110,
                             'short_intent': 'check boolean'},
    }
    for fid, data in functions.items():
        graph.add_node(fid, NODE_FUNC, data)
    for cid in ('result_owner', 'template', 'element_owner'):
        graph.add_node(cid, NODE_CLASS, {'fq_name': cid})
    for fid in ('seed', 'second', 'named', 'named_array', 'noise'):
        graph.add_edge(fid, 'result_owner', EDGE_IN_CLASS)
    for fid in ('underlying', 'named_underlying'):
        graph.add_edge(fid, 'element_owner', EDGE_IN_CLASS)
    graph.add_edge('seed', 'template', EDGE_DEP_TEMPLATE)
    graph.add_edge('template', 'element_owner', EDGE_DEP_TEMPLATE)
    graph.add_edge('named', 'named_underlying', EDGE_DEP_MEMBER)
    rag.graph_ctx = SimpleNamespace(graph=graph, functions=functions)
    seen = []

    def rerank(query, documents, top_k, **kwargs):
        seen.extend(json.loads(document) for document in documents)
        return [{'index': i, 'score': 1 - i / 10} for i in range(top_k)]

    ranked, metadata = rag._structural_companions(
        'is<T> should return bool directly', [('seed', 1), ('second', .9)], 5,
        reranker=rerank,
    )

    assert [document['companion'] for document in seen[:3]] == [
        'api::element::is', 'api::element::is_bool', 'api::result::is_bool']
    assert [fid for fid, _ in ranked[2:]] == [
        'underlying', 'named_underlying', 'named']
    assert 'api::result::is_array' not in {
        document['companion'] for document in seen
    }
    assert len(metadata['underlying']['graph_paths'][0]['edges']) == 3
    assert metadata['named_underlying']['graph_paths'][0]['edges'][-1]['edge'] == EDGE_DEP_MEMBER
    assert all(len(item['connections']) <= 4 for item in seen)
