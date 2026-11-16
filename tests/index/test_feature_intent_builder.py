import json
from types import SimpleNamespace

import faiss
import numpy as np

from code_intent.code_intent import feature_builder, llm_utils
from code_intent.code_intent.embedding_manifest import load_embedding_manifest


class _MemoryEmbeddingCache:
    def __init__(self):
        self.records = {}

    def get(self, text):
        return self.records.get(text)

    def update(self, records):
        self.records.update(records)

    def close(self):
        pass


def _write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_test_grouping_is_deterministic():
    coverage = {
        "test-b": {"f1": 1, "f2": 1},
        "test-a": {"f1": 1, "f2": 1},
        "test-c": {"f3": 1},
    }

    expected = [["test-a", "test-b"], ["test-c"]]
    assert feature_builder._group_tests_within_bucket(
        ["test-c", "test-b", "test-a"], coverage
    ) == expected
    assert feature_builder._group_tests_within_bucket(
        ["test-a", "test-c", "test-b"], coverage
    ) == expected


def test_source_level_test_case_bridges_to_callgraph_function_ids(monkeypatch):
    func_lookup = {
        "prod": {
            "func_id": "prod",
            "short_intent": "Parse the value.",
            "raw_summary": "Parses an input value.",
        }
    }
    tests = [
        {
            "func_id": "testcase:detached-id",
            "source_func_ids": ["indexed-test-function"],
            "short_intent": "Validate parsing.",
            "raw_summary": "The parser accepts the value.",
        }
    ]
    monkeypatch.setattr(
        feature_builder,
        "summarize_test_feature",
        lambda *_args: {
            "feature_name": "Parsing",
            "feature_description": "Validates parsing.",
            "tags": ["parse"],
            "_intent_generation": "heuristic",
        },
    )

    payloads, covered = feature_builder._build_test_feature_payloads(
        func_lookup,
        tests,
        {"indexed-test-function": ["prod"]},
    )

    assert covered == {"prod"}
    assert len(payloads) == 1
    assert payloads[0][0]["source_tests"] == ["testcase:detached-id"]
    assert payloads[0][0]["member_funcs"] == ["prod"]


def test_two_public_methods_form_a_class_feature(monkeypatch):
    members = [
        {
            "func_id": "parse",
            "class_name": "fmt::range_formatter",
            "is_internal": False,
        },
        {
            "func_id": "format",
            "class_name": "fmt::range_formatter",
            "is_internal": False,
        },
    ]
    summary_inputs = []

    def summarize(summary_members, cluster_context=None):
        summary_inputs.append((summary_members, cluster_context))
        return {
            "feature_name": "Range formatting",
            "feature_description": "Parses and formats ranges.",
            "tags": ["range"],
            "_intent_generation": "heuristic",
        }

    monkeypatch.setattr(feature_builder, "summarize_cluster", summarize)

    payloads = feature_builder._build_class_feature_payloads(members)

    assert len(payloads) == 1
    feature, memberships = payloads[0]
    assert feature["source_class"] == "fmt::range_formatter"
    assert feature["member_funcs"] == ["format", "parse"]
    assert memberships == {"format": 1.0, "parse": 1.0}
    assert len(summary_inputs) == 1
    assert all(member["_cluster_representative"] for member in summary_inputs[0][0])
    assert summary_inputs[0][1]["cluster_kind"] == "class_component"


def test_large_cluster_summary_samples_across_full_membership():
    members = [f"f{index:03d}" for index in range(100)]
    intents = {
        fid: {
            "func_id": fid,
            "file": "src/component.cpp",
            "signature": f"api::{fid}",
        }
        for fid in members
    }

    selected = feature_builder._select_cluster_summary_ids(
        members, ["f000"], intents, limit=50
    )

    assert len(selected) == 50
    assert len(set(selected)) == 50
    assert selected[0] == "f000"
    assert "f099" in selected
    assert any(fid > "f050" for fid in selected)


def test_refined_parent_is_summarized_as_navigation_scope(monkeypatch):
    intents = [
        {
            "func_id": f"f{index:03d}",
            "signature": f"api::f{index:03d}",
            "file": "src/api.cpp",
            "action": "handle value",
            "object": "value",
        }
        for index in range(60)
    ]
    member_ids = [intent["func_id"] for intent in intents]
    contexts = []

    monkeypatch.setattr(
        feature_builder,
        "_build_graph_communities",
        lambda *_args, **_kwargs: [member_ids],
    )

    def refine(
        _members,
        _callgraph,
        _vectors,
        _depth,
        parent_id,
        _counter,
        hierarchy,
    ):
        child_id = f"{parent_id}_child"
        hierarchy.append(
            {
                "parent_cluster_id": parent_id,
                "child_cluster_id": child_id,
                "reason": "test",
                "depth": 1,
            }
        )
        return [{"cluster_id": child_id, "member_funcs": member_ids[:8]}]

    def summarize(_members, cluster_context=None):
        contexts.append(cluster_context)
        return {
            "feature_name": "Grounded scope",
            "feature_description": "Grounded description.",
            "tags": ["grounded"],
            "_intent_generation": "heuristic",
        }

    monkeypatch.setattr(feature_builder, "_refine_clusters_recursive", refine)
    monkeypatch.setattr(feature_builder, "summarize_cluster", summarize)

    clusters, hierarchy = feature_builder._build_intent_clusters(
        intents,
        {},
        {
            intent["func_id"]: np.asarray([1.0, 0.0], dtype="float32")
            for intent in intents
        },
    )

    assert hierarchy
    assert contexts[0] == {
        "cluster_kind": "graph_community",
        "summary_role": "coarse_navigation",
        "total_member_count": 60,
        "sampled_member_count": 50,
        "has_refined_children": True,
    }
    assert contexts[1]["cluster_kind"] == "graph_community_refined"
    assert contexts[1]["summary_role"] == "functional_cluster"
    assert clusters[0]["summary_role"] == "coarse_navigation"
    assert clusters[1]["summary_role"] == "functional_cluster"


def test_cluster_summary_retries_ungrounded_qualified_names(monkeypatch):
    members = [
        {
            "func_id": "child",
            "signature": "simdjson::fallback::ondemand::value_iterator::child",
            "short_intent": "Return the child value iterator.",
        }
    ]
    responses = [
        {
            "feature_name": "simd navigation",
            "feature_description": "Uses simd::ondemand::raw__string.",
            "tags": ["simd"],
        },
        {
            "feature_name": "simdjson navigation",
            "feature_description": (
                "Uses simdjson::fallback::ondemand::value_iterator::child."
            ),
            "tags": ["simdjson"],
        },
    ]
    contexts = []

    def summarize(_members, context):
        contexts.append(context)
        return responses.pop(0)

    refine_calls = []

    def refine(generated, issues, evidence):
        refine_calls.append((generated, issues, evidence))
        return {"generated_": responses.pop(0)}

    monkeypatch.setattr(
        llm_utils._LLM_CLIENT, "config", SimpleNamespace(enabled=True)
    )
    monkeypatch.setattr(llm_utils._LLM_CLIENT, "summarize_cluster", summarize)
    monkeypatch.setattr(
        llm_utils._LLM_CLIENT, "refine_identifier_grounding", refine
    )
    monkeypatch.setattr(llm_utils, "INTENT_MAX_RETRIES", 2)
    monkeypatch.setattr(llm_utils, "INTENT_RETRY_BASE_SECONDS", 0.0)
    monkeypatch.setattr(llm_utils, "INTENT_RETRY_MAX_SECONDS", 0.0)

    summary = llm_utils.summarize_cluster(
        members, {"summary_role": "coarse_navigation"}
    )

    assert summary["feature_name"] == "simdjson navigation"
    assert summary["_intent_grounding_retries"] == 0
    assert summary["_intent_grounding_refinements"] == 1
    assert summary["_intent_identifier_repairs"] == {"simd": "simdjson"}
    assert len(contexts) == 1
    assert refine_calls[0][1] == [
        "simdjson::ondemand::raw__string",
    ]


def test_identifier_root_repair_does_not_rewrite_plain_action_words():
    generated = {
        "feature_name": "Parse input token",
        "feature_description": "Use simd::ondemand::value_iterator::child.",
        "tags": ["parse", "input", "token"],
    }
    members = [
        {
            "signature": "simdjson::ondemand::value_iterator::child",
            "grounded_identifiers": [
                "parse_error",
                "input_format_t",
                "token_type",
            ],
        }
    ]

    repaired, replacements = llm_utils._repair_cluster_summary_root_names(
        generated, members
    )

    assert replacements == {"simd": "simdjson"}
    assert repaired["feature_name"] == "Parse input token"
    assert repaired["tags"] == ["parse", "input", "token"]
    assert not llm_utils._cluster_summary_grounding_issues(repaired, members)


def test_identifier_redaction_removes_placeholder_feature_tags():
    summary = {
        "feature_name": "Iterator navigation",
        "feature_description": "Uses _iterator for navigation.",
        "tags": ["navigation", "_iterator"],
    }

    redacted, replacements = llm_utils._redact_ungrounded_qualified_identifiers(
        summary, ["_iterator"], {"signature": "api::navigate"}
    )

    assert replacements == ["_iterator"]
    assert redacted["feature_description"] == (
        "Uses the referenced internal helper for navigation."
    )
    assert redacted["tags"] == ["navigation"]


def test_feature_build_uses_one_cluster_tree_and_indexes_parent_and_child(
    tmp_path, monkeypatch
):
    intents = [
        {
            "func_id": f"f{index}",
            "signature": f"api::f{index}",
            "file": "src/api.cpp",
            "short_intent": f"Handle value {index}.",
            "raw_summary": f"Handles value {index}.",
            "action": "handle value",
            "object": "value",
            "features": [{"feature_id": "legacy", "score": 1.0}],
        }
        for index in range(4)
    ]
    meta = [
        {"index": index, "func_id": f"f{index}"} for index in range(4)
    ]
    _write_jsonl(tmp_path / "function_intents.jsonl", intents)
    _write_jsonl(tmp_path / "func_meta.jsonl", meta)
    _write_jsonl(
        tmp_path / "callgraph.jsonl",
        [
            {"caller": "f0", "callee": "f1", "edge_type": "call"},
            {"caller": "f1", "callee": "f2", "edge_type": "call"},
        ],
    )
    (tmp_path / "test_intents.jsonl").write_text("", encoding="utf-8")
    function_index = faiss.IndexFlatIP(2)
    function_index.add(
        np.asarray(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
            dtype="float32",
        )
    )
    faiss.write_index(function_index, str(tmp_path / "func_index.faiss"))

    calls = []

    def build_clusters(function_intents, callgraph, vector_lookup, _summary_cache):
        calls.append((function_intents, callgraph, vector_lookup))
        return (
            [
                {
                    "cluster_id": "cluster_00001",
                    "cluster_name": "Public API",
                    "cluster_description": "Handles public values.",
                    "tags": ["api"],
                    "member_funcs": ["f0", "f1", "f2", "f3"],
                    "representative_funcs": ["f0"],
                    "cluster_type": "hybrid_leiden",
                    "summary_generation": "heuristic",
                },
                {
                    "cluster_id": "cluster_00001_00002",
                    "cluster_name": "Value parsing",
                    "cluster_description": "Parses value inputs.",
                    "tags": ["parse"],
                    "member_funcs": ["f0", "f1"],
                    "representative_funcs": ["f1"],
                    "cluster_type": "hybrid_leiden_refined",
                    "summary_generation": "heuristic",
                },
            ],
            [
                {
                    "parent_cluster_id": "cluster_00001",
                    "child_cluster_id": "cluster_00001_00002",
                    "reason": "bimodal_refine",
                    "depth": 1,
                }
            ],
        )

    embedded_batches = []

    def embed(texts):
        embedded_batches.append(list(texts))
        return np.asarray(
            [[1.0, float(index + 1)] for index, _ in enumerate(texts)],
            dtype="float32",
        )

    monkeypatch.setattr(feature_builder, "EMBED_DIM", 2)
    monkeypatch.setattr(feature_builder, "_build_intent_clusters", build_clusters)
    monkeypatch.setattr(feature_builder, "embed_texts", embed)
    embedding_cache = _MemoryEmbeddingCache()
    monkeypatch.setattr(
        feature_builder,
        "GlobalEmbeddingCache",
        lambda *_args, **_kwargs: embedding_cache,
    )
    monkeypatch.setenv("VOYAGE_EMBED_MODEL", "voyage-2")
    monkeypatch.setenv("CODE_INTENT_EMBED_ABORT_ON_FALLBACK", "1")
    monkeypatch.delenv("LLM_MODEL", raising=False)

    feature_builder.build_feature_intents(tmp_path, repo_id="project")

    assert len(calls) == 1
    features = feature_builder._load_jsonl(tmp_path / "features.jsonl")
    assert [feature["feature_type"] for feature in features] == [
        "graph_community",
        "graph_community_refined",
    ]
    assert {feature["source_cluster_id"] for feature in features} == {
        "cluster_00001",
        "cluster_00001_00002",
    }
    assert all(feature["feature_type"] != "cluster" for feature in features)
    assert len(embedded_batches) == 1
    assert len(embedded_batches[0]) == 2
    assert faiss.read_index(str(tmp_path / "feature_index.faiss")).ntotal == 2

    manifest = load_embedding_manifest(tmp_path)["indexes"]["features"]
    assert manifest["builder"] == "feature-intent-v5"
    assert manifest["feature_count"] == 2
    assert manifest["cluster_count"] == 2
    assert manifest["hierarchy_count"] == 1
    assert manifest["fallback_count"] == 0
    assert manifest["embedding_cache_hits"] == 0
    assert manifest["embedding_requests"] == 2
    assert feature_builder.intent_feature_artifacts_ready(tmp_path)
    with monkeypatch.context() as prompt_patch:
        prompt_patch.setattr(
            llm_utils,
            "_CLUSTER_SYSTEM_PROMPT",
            llm_utils._CLUSTER_SYSTEM_PROMPT + "\nChanged prompt.",
        )
        assert not feature_builder.intent_feature_artifacts_ready(tmp_path)

    updated_intents = feature_builder._load_jsonl(
        tmp_path / "function_intents.jsonl"
    )
    build_id = manifest["build_id"]
    assert {record["_feature_build_id"] for record in updated_intents} == {
        build_id
    }
    assert [item["feature_id"] for item in updated_intents[0]["features"]] == [
        "feat_00000",
        "feat_00001",
    ]

    with (tmp_path / "callgraph.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {"caller": "f2", "callee": "f3", "edge_type": "call"}
            )
            + "\n"
        )
    assert not feature_builder.intent_feature_artifacts_ready(tmp_path)
    lines = (tmp_path / "callgraph.jsonl").read_text(encoding="utf-8").splitlines()
    (tmp_path / "callgraph.jsonl").write_text(
        "\n".join(lines[:-1]) + "\n", encoding="utf-8"
    )
    assert feature_builder.intent_feature_artifacts_ready(tmp_path)
    metadata = feature_builder._load_jsonl(tmp_path / "feature_meta.jsonl")
    metadata[0]["_build_id"] = "interrupted"
    _write_jsonl(tmp_path / "feature_meta.jsonl", metadata)
    assert not feature_builder.intent_feature_artifacts_ready(tmp_path)

    embedded_batches.clear()
    feature_builder.build_feature_intents(tmp_path, repo_id="project")
    assert embedded_batches == []
    manifest = load_embedding_manifest(tmp_path)["indexes"]["features"]
    assert manifest["embedding_cache_hits"] == 2
    assert manifest["embedding_requests"] == 0


def test_feature_summary_cache_reuses_exact_prompt(tmp_path, monkeypatch):
    calls = []

    def summarize(members, cluster_context=None):
        calls.append((members, cluster_context))
        return {
            "feature_name": "Range formatting",
            "feature_description": "Parses and formats ranges.",
            "tags": ["range"],
            "_intent_generation": "heuristic",
        }

    monkeypatch.setenv("INTENT_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setattr(feature_builder, "summarize_cluster", summarize)
    members = [{"func_id": "parse", "short_intent": "Parse ranges."}]

    first = feature_builder._FeatureSummaryCache("fmt")
    assert first.summarize_cluster(members)["feature_name"] == "Range formatting"
    first.close()
    second = feature_builder._FeatureSummaryCache("fmt")
    assert second.summarize_cluster(members)["feature_name"] == "Range formatting"
    second.close()
    third = feature_builder._FeatureSummaryCache("fmt")
    assert (
        third.summarize_cluster(
            members, {"summary_role": "coarse_navigation"}
        )["feature_name"]
        == "Range formatting"
    )
    third.close()

    assert len(calls) == 2
    assert second.hits == 1
    assert second.requests == 0
    assert third.requests == 1


def test_feature_summary_cache_keeps_paid_summary_after_interruption(tmp_path, monkeypatch):
    monkeypatch.setenv("INTENT_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_MODEL", "intent-model")
    calls = []

    def summarize(members, _context=None):
        calls.append(members)
        return {"feature_name": "Read one value", "feature_description": "Return the value.",
                "tags": [], "_intent_generation": "llm"}

    monkeypatch.setattr(feature_builder, "summarize_cluster", summarize)
    members = [{"func_id": "read", "signature": "api::read", "short_intent": "Read one value."}]
    first = feature_builder._FeatureSummaryCache("project")
    first.summarize_cluster(members)
    first.connection.close()  # Simulate interruption without the builder's normal close.
    second = feature_builder._FeatureSummaryCache("project")
    assert second.summarize_cluster(members)["feature_name"] == "Read one value"
    second.close()
    assert len(calls) == 1


def test_feature_summary_cache_rejects_old_ungrounded_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("INTENT_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("LLM_MODEL", raising=False)
    members = [
        {
            "func_id": "child",
            "signature": "simdjson::ondemand::value_iterator::child",
        }
    ]
    cache_key = feature_builder.cluster_summary_input_hash(members)
    cache = feature_builder._FeatureSummaryCache("simdjson")
    assert cache.connection is not None
    cache.connection.execute(
        "INSERT INTO summaries (cache_key, payload) VALUES (?, ?)",
        (
            cache_key,
            json.dumps(
                {
                    "feature_name": "simd navigation",
                    "feature_description": "Uses simd::ondemand::raw__string.",
                    "tags": ["simd"],
                    "_intent_generation": "heuristic",
                }
            ),
        ),
    )
    cache.connection.commit()

    calls = []

    def summarize(_members, _context=None):
        calls.append(True)
        return {
            "feature_name": "simdjson navigation",
            "feature_description": (
                "Uses simdjson::ondemand::value_iterator::child."
            ),
            "tags": ["simdjson"],
            "_intent_generation": "heuristic",
        }

    monkeypatch.setattr(feature_builder, "summarize_cluster", summarize)

    assert cache.summarize_cluster(members)["feature_name"] == "simdjson navigation"
    cache.close()

    assert calls == [True]
    assert cache.hits == 0
    assert cache.requests == 1
