import asyncio
import json
from types import SimpleNamespace

import faiss
import numpy as np
import pytest

from code_intent.code_intent import (
    edge_utils,
    embedding_cache as persistent_embedding_cache,
    intent_extractor,
    llm_utils,
)
from code_intent.code_intent.embedding_manifest import load_embedding_manifest


class _MemoryIntentCache:
    def __init__(self, cache_type="function", *_args, **_kwargs):
        self.cache_type = cache_type
        self.records = {}

    def get(self, cache_key):
        return self.records.get(cache_key)

    def update(self, records):
        for record in records:
            key = (
                record.get("intent_input_hash")
                if self.cache_type == "function"
                else record.get("content_hash")
            )
            self.records[key] = record


class _MemoryEmbeddingCache:
    def __init__(self, *_args, **_kwargs):
        self.records = {}

    def get(self, text):
        return self.records.get(text)

    def update(self, records):
        self.records.update(records)

    def close(self):
        pass


def test_embedding_cache_persists_and_separates_models(tmp_path, monkeypatch):
    monkeypatch.setenv("INTENT_CACHE_DIR", str(tmp_path))
    vector = np.asarray([0.6, 0.8], dtype="float32")

    first = persistent_embedding_cache.GlobalEmbeddingCache(
        "test", repo_id="fmt", model="voyage-2", dimension=2
    )
    first.update([("same text", vector)])
    first.close()

    second = persistent_embedding_cache.GlobalEmbeddingCache(
        "test", repo_id="fmt", model="voyage-2", dimension=2
    )
    assert np.allclose(second.get("same text"), vector)
    second.close()

    other_model = persistent_embedding_cache.GlobalEmbeddingCache(
        "test", repo_id="fmt", model="voyage-3", dimension=2
    )
    assert other_model.get("same text") is None
    other_model.close()


def test_function_intent_repairs_roots_and_retries_unknown_identifiers(monkeypatch):
    function = {
        "id": "child",
        "fq_name": "simdjson::fallback::ondemand::value_iterator::child",
        "short_name": "child",
        "namespaces": ["simdjson", "fallback", "ondemand"],
        "classes": ["value_iterator"],
        "return_type": "simdjson::fallback::ondemand::value_iterator",
        "parameter_types": [],
        "code": "value_iterator child() { return value_iterator{}; }",
    }
    context = {
        "callees": [
            "simdjson::fallback::ondemand::value_iterator::assert_at_child"
        ]
    }
    responses = [
        {
            "short_intent": "Build a simd child iterator.",
            "action": "construct simd::fallback::ondemand::value_iterator::raw__child",
            "object": "simd::fallback::ondemand::value_iterator",
            "conditions": [],
            "side_effects": [],
            "raw_summary": "Calls simd::fallback::ondemand::value_iterator::assert_at_child.",
            "usage_context": "",
        },
        {
            "short_intent": "Build a simdjson child iterator.",
            "action": "construct simdjson::fallback::ondemand::value_iterator::child",
            "object": "simdjson::fallback::ondemand::value_iterator",
            "conditions": [],
            "side_effects": [],
            "raw_summary": (
                "Calls simdjson::fallback::ondemand::value_iterator::assert_at_child."
            ),
            "usage_context": "",
        },
    ]
    contexts = []

    async def generate(_function, generated_context):
        contexts.append(generated_context)
        return responses.pop(0)

    refine_calls = []

    async def refine(generated, issues, evidence):
        refine_calls.append((generated, issues, evidence))
        return {"generated_": responses.pop(0)}

    monkeypatch.setattr(
        llm_utils._LLM_CLIENT, "config", SimpleNamespace(enabled=True)
    )
    monkeypatch.setattr(
        llm_utils._LLM_CLIENT, "generate_function_intent_async", generate
    )
    monkeypatch.setattr(
        llm_utils._LLM_CLIENT, "refine_identifier_grounding_async", refine
    )
    monkeypatch.setattr(llm_utils, "INTENT_MAX_RETRIES", 2)
    monkeypatch.setattr(llm_utils, "INTENT_RETRY_BASE_SECONDS", 0.0)
    monkeypatch.setattr(llm_utils, "INTENT_RETRY_MAX_SECONDS", 0.0)

    intent = asyncio.run(
        llm_utils.generate_intent_from_code_async(function, context)
    )

    assert intent["short_intent"] == "Build a simdjson child iterator."
    assert intent["_intent_grounding_retries"] == 0
    assert intent["_intent_grounding_refinements"] == 1
    assert intent["_intent_identifier_repairs"] == {"simd": "simdjson"}
    assert len(contexts) == 1
    assert refine_calls[0][1] == [
        "simdjson::fallback::ondemand::value_iterator::raw__child"
    ]


def test_function_grounding_catches_unqualified_helper_typo_and_suggests_source_name():
    function = {
        "id": "size",
        "fq_name": "nlohmann::detail::binary_reader::get_ubjson_size_type",
        "code": "return get_ubjson_array(result);",
    }
    intent = {
        "short_intent": "Read the size marker.",
        "action": "call get_ub_array",
        "object": "the marker",
        "conditions": [],
        "side_effects": [],
        "raw_summary": "Delegates to get_ub_array.",
        "usage_context": "",
    }

    issues = llm_utils.function_intent_grounding_issues(intent, function)
    evidence = llm_utils._function_identifier_evidence(function)
    catalog = llm_utils._identifier_evidence_catalog(evidence, issues)

    assert issues == ["get_ub_array"]
    assert "get_ubjson_array" in catalog["suggested_replacements"]["get_ub_array"]
    repaired, replacements = llm_utils._repair_close_identifier_names(
        intent, issues, evidence
    )
    assert replacements == {"get_ub_array": "get_ubjson_array"}
    assert repaired["action"] == "call get_ubjson_array"
    assert not llm_utils.function_intent_grounding_issues(repaired, function)


def test_identifier_redaction_does_not_corrupt_compound_identifiers():
    generated = {
        "action": "advance iterator through value_iterator",
        "raw_summary": "Calls get_root_value_iterator after checking iterator.",
    }

    redacted, replacements = llm_utils._redact_ungrounded_qualified_identifiers(
        generated,
        ["iterator"],
        {"signature": "document::resume_value"},
    )

    assert replacements == ["iterator"]
    assert redacted["action"] == (
        "advance the referenced internal helper through value_iterator"
    )
    assert redacted["raw_summary"] == (
        "Calls get_root_value_iterator after checking "
        "the referenced internal helper."
    )


def test_relative_identifier_repair_does_not_duplicate_existing_namespace():
    generated = {
        "usage_context": (
            "Calls array_iterator::operator and "
            "simdjson::fallback::ondemand::array_iterator::operator."
        )
    }
    evidence = {
        "relations": [
            "simdjson::fallback::ondemand::array_iterator::operator"
        ]
    }

    repaired, replacements = llm_utils._repair_close_identifier_names(
        generated, ["array_iterator::operator"], evidence
    )

    expected = "simdjson::fallback::ondemand::array_iterator::operator"
    assert replacements == {"array_iterator::operator": expected}
    assert repaired["usage_context"] == f"Calls {expected} and {expected}."
    assert "ondemand::simdjson" not in repaired["usage_context"]


def test_scoped_identifier_repair_converges_across_two_bad_components():
    generated = {
        "raw_summary": (
            "Calls nlohmann::basic_::from_ub for input_format_t::ub."
        )
    }
    evidence = {
        "callers": ["nlohmann::basic_json::from_ubjson"],
        "source": "case input_format_t::ubjson:",
    }
    issues = llm_utils._identifier_grounding_issues(generated, evidence)

    repaired, replacements = llm_utils._repair_close_identifier_names(
        generated, issues, evidence
    )

    assert repaired["raw_summary"] == (
        "Calls nlohmann::basic_json::from_ubjson for input_format_t::ubjson."
    )
    assert replacements["from_ub"] == "from_ubjson"
    assert replacements["input_format_t::ub"] == "input_format_t::ubjson"
    assert replacements["nlohmann::basic_::from_ubjson"] == (
        "nlohmann::basic_json::from_ubjson"
    )
    assert not llm_utils._identifier_grounding_issues(repaired, evidence)


def test_identifier_grounding_repairs_dangling_scope_and_leading_underscore():
    generated = {
        "conditions": ["input_format_t:: reaches _sax_t"],
        "raw_summary": "Calls parse_ub_internal.",
        "_intent_identifier_redactions": ["invalid::provenance_name"],
    }
    evidence = {
        "source": (
            "input_format_t format; json_sax_t sax; "
            "return parse_ubjson_internal();"
        )
    }
    issues = llm_utils._identifier_grounding_issues(generated, evidence)

    repaired, replacements = llm_utils._repair_close_identifier_names(
        generated, issues, evidence
    )

    assert issues == ["_sax_t", "input_format_t::", "parse_ub_internal"]
    assert repaired["conditions"] == ["input_format_t reaches json_sax_t"]
    assert repaired["raw_summary"] == "Calls parse_ubjson_internal."
    assert replacements == {
        "_sax_t": "json_sax_t",
        "input_format_t::": "input_format_t",
        "parse_ub_internal": "parse_ubjson_internal",
    }
    assert not llm_utils._identifier_grounding_issues(repaired, evidence)


def test_close_identifier_repair_does_not_record_noop_replacement():
    generated = {"raw_summary": "Calls root_checkpoint."}
    repaired, replacements = llm_utils._repair_close_identifier_names(
        generated,
        ["root_checkpoint"],
        {"source": "iter.root_checkpoint()"},
    )

    assert repaired == generated
    assert replacements == {}


def test_grounding_repairs_close_unqualified_code_identifiers():
    generated = {
        "feature_description": "Wrap simd_result and call from_ub.",
        "tags": ["simd_result", "from_ub"],
    }
    evidence = {
        "grounded_identifiers": ["simdjson_result", "from_ubjson"]
    }
    issues = llm_utils._identifier_grounding_issues(generated, evidence)

    repaired, replacements = llm_utils._repair_close_identifier_names(
        generated, issues, evidence
    )

    assert issues == ["from_ub", "simd_result"]
    assert replacements == {
        "from_ub": "from_ubjson",
        "simd_result": "simdjson_result",
    }
    assert repaired["tags"] == ["simdjson_result", "from_ubjson"]
    assert not llm_utils._identifier_grounding_issues(repaired, evidence)


def test_short_public_header_wrapper_remains_embedding_eligible():
    function = {
        "id": "document-count-elements",
        "short_name": "count_elements",
        "is_method": True,
        "def_file": "include/simdjson/generic/ondemand/document-inl.h",
        "def_start_line": 20,
        "def_end_line": 22,
        "caller_count": 0,
    }

    assert intent_extractor._should_embed_function(function)


def test_short_internal_method_is_still_treated_as_helper():
    function = {
        "id": "internal-helper",
        "short_name": "advance",
        "is_method": True,
        "def_file": "src/internal/parser.cpp",
        "def_start_line": 20,
        "def_end_line": 22,
        "caller_count": 0,
    }

    assert not intent_extractor._should_embed_function(function)


def test_intent_context_keeps_only_unique_executable_edges(tmp_path):
    functions = [
        {"id": "target", "fq_name": "api::target"},
        {"id": "caller-b", "fq_name": "api::caller_b"},
        {"id": "caller-a", "fq_name": "api::caller_a"},
        {"id": "caller-a-overload", "fq_name": "api::caller_a"},
        {"id": "callee", "fq_name": "api::callee"},
        {"id": "noise", "fq_name": "api::noise"},
        {"id": "test", "fq_name": "test_target", "is_test": True},
    ]
    edges = [
        {"caller": "caller-b", "callee": "target", "edge_type": "call"},
        {"caller": "caller-b", "callee": "target", "edge_type": "call"},
        {
            "caller": "caller-a", "callee": "target",
            "edge_type": "dependent_call",
        },
        {
            "caller": "caller-a-overload", "callee": "target",
            "edge_type": "call",
        },
        {"caller": "noise", "callee": "target", "edge_type": "member_ref"},
        {"caller": "test", "callee": "target", "edge_type": "decl_ref"},
        {"caller": "test", "callee": "target", "edge_type": "call"},
        {"caller": "target", "callee": "callee", "edge_type": "call"},
        {"caller": "target", "callee": "missing", "edge_type": "call"},
    ]
    (tmp_path / "callgraph.jsonl").write_text(
        "".join(json.dumps(edge) + "\n" for edge in edges),
        encoding="utf-8",
    )

    callers, callees, tests, _, _, _ = intent_extractor._load_context_data(
        tmp_path, functions
    )

    assert callers["target"] == [
        "caller-a", "caller-a-overload", "caller-b", "test"
    ]
    assert callees["target"] == ["callee"]
    assert tests["target"] == ["test"]
    assert "noise" not in callees
    assert intent_extractor._function_context(
        functions[0], (callers, callees, tests, {
            function["id"]: function["fq_name"] for function in functions
        }, {"test"}, {})
    )["callers"] == ["api::caller_a", "api::caller_b"]


def test_edge_reader_keeps_only_executable_edges(tmp_path):
    path = tmp_path / "callgraph.jsonl"
    path.write_text(
        "".join(json.dumps(edge) + "\n" for edge in [
            {"caller": "a", "callee": "b", "edge_type": "call"},
            {"caller": "a", "callee": "b", "edge_type": "call"},
            {"caller": "a", "callee": "c", "edge_type": "dependent_call"},
            {"caller": "a", "callee": "d", "edge_type": "member_ref"},
        ]),
        encoding="utf-8",
    )

    assert [
        (edge["caller"], edge["callee"])
        for edge in edge_utils.iter_executable_edges(path)
    ] == [("a", "b"), ("a", "b"), ("a", "c")]


def test_test_eligibility_requires_an_executable_production_edge(tmp_path):
    functions = {
        "test": {"id": "test", "is_test": True},
        "prod": {"id": "prod", "is_test": False},
    }
    path = tmp_path / "callgraph.jsonl"
    path.write_text(
        json.dumps({
            "caller": "test",
            "callee": "prod",
            "edge_type": "decl_ref",
        }) + "\n",
        encoding="utf-8",
    )
    assert intent_extractor._tests_touching_prod(path, functions) == set()

    path.write_text(
        json.dumps({
            "caller": "test",
            "callee": "prod",
            "edge_type": "call",
        }) + "\n",
        encoding="utf-8",
    )
    assert intent_extractor._tests_touching_prod(path, functions) == {"test"}


def test_function_intent_hash_changes_with_source_context_and_model_profile(monkeypatch):
    function = {
        "id": "target",
        "fq_name": "api::target",
        "code": "int target() { return 1; }",
        "content_hash": "source-v1",
    }
    no_context = intent_extractor.function_intent_input_hash(function, None)
    same_input = intent_extractor.function_intent_input_hash(dict(function), None)
    with_caller = intent_extractor.function_intent_input_hash(
        function, {"callers": ["api::caller"], "callees": [], "related_tests": []}
    )
    changed_source = intent_extractor.function_intent_input_hash(
        {**function, "content_hash": "source-v2"}, None
    )
    monkeypatch.setenv("LLM_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("LLM_THINKING_TYPE", "disabled")
    nonthinking = intent_extractor.function_intent_input_hash(function, None)
    monkeypatch.setenv("LLM_THINKING_TYPE", "enabled")
    monkeypatch.setenv("LLM_REASONING_EFFORT", "low")
    low_thinking = intent_extractor.function_intent_input_hash(function, None)

    assert no_context == same_input
    assert with_caller != no_context
    assert changed_source != no_context
    assert nonthinking != no_context
    assert low_thinking != nonthinking


def test_function_embedding_text_keeps_grounded_fields_without_summary_duplication():
    text = intent_extractor._build_embedding_text(
        {
            "id": "parser",
            "fq_name": "api::parser::parse",
            "namespaces": ["api"],
            "classes": ["parser"],
        },
        {
            "short_intent": "Parse one framed value.",
            "action": "parse framed value",
            "object": "input frame",
            "conditions": ["a complete frame is available"],
            "side_effects": ["advances the input cursor"],
            "usage_context": "Called by the stream decoder.",
            "raw_summary": "Consumes one frame and returns its decoded value.",
        },
    )

    assert text.count("Consumes one frame and returns its decoded value.") == 1
    assert "Action: parse framed value" in text
    assert "Conditions: a complete frame is available" in text
    assert "Side effects: advances the input cursor" in text
    assert "api::parser::parse" in text


def test_test_intent_hash_changes_with_focal_evidence():
    test = {
        "id": "test-widget",
        "content_hash": "same-test-source",
        "code": "TEST(Widget, Parse) {}",
        "focal_entities": [
            {
                "name": "Widget::parse",
                "relations": ["call"],
                "source": "bool Widget::parse() { return true; }",
            }
        ],
    }
    original = llm_utils.test_intent_input_hash(test)
    changed = llm_utils.test_intent_input_hash(
        {
            **test,
            "focal_entities": [
                {
                    "name": "Widget::parse",
                    "relations": ["call"],
                    "source": "bool Widget::parse() { return false; }",
                }
            ],
        }
    )

    assert original != changed


def test_test_intent_hash_ignores_source_locations_and_location_hashes():
    test = {
        "id": "testcase:old-location",
        "fq_name": "Widget parses valid input",
        "def_file": "tests/widget_test.cpp",
        "def_start_line": 20,
        "def_end_line": 24,
        "content_hash": "includes-old-focal-location",
        "code": "TEST_CASE(\"Widget parses valid input\") { CHECK(parse()); }",
        "inherited_setup": "Widget widget;",
        "oracle_code": "CHECK(parse());",
        "focal_entities": [{
            "name": "Widget::parse",
            "file": "src/widget.cpp",
            "start_line": 80,
            "relations": ["call"],
            "source": "bool Widget::parse() { return true; }",
        }],
    }
    moved = {
        **test,
        "id": "testcase:new-location",
        "def_start_line": 120,
        "def_end_line": 124,
        "content_hash": "includes-new-focal-location",
        "focal_entities": [{
            **test["focal_entities"][0],
            "start_line": 180,
        }],
    }

    assert llm_utils.test_intent_input_hash(test) == llm_utils.test_intent_input_hash(
        moved
    )
    assert llm_utils._legacy_test_intent_input_hash_v4(
        test
    ) != llm_utils._legacy_test_intent_input_hash_v4(moved)


def test_async_intent_generation_honors_concurrency_limit(monkeypatch):
    active = 0
    max_active = 0

    async def generate(function, _context):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return {
            "short_intent": function["id"],
            "_intent_generation": "heuristic",
        }

    monkeypatch.setattr(intent_extractor, "INTENT_CONCURRENCY", 2)
    monkeypatch.setattr(intent_extractor, "generate_intent_from_code_async", generate)
    result = asyncio.run(
        intent_extractor._generate_intents_for_batch_async(
            [{"id": f"f{index}"} for index in range(7)],
            respect_internal=False,
        )
    )

    assert len(result) == 7
    assert max_active == 2


def test_failed_intent_batch_cancels_pending_requests_before_closing_client(monkeypatch):
    events = []
    pending_started = asyncio.Event()

    async def generate(func, _context):
        if func['id'] == 'fail':
            await pending_started.wait()
            raise RuntimeError('provider unavailable')
        pending_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            events.append('pending_cancelled')

    async def close():
        events.append('client_closed')

    monkeypatch.setenv('INTENT_ABORT_ON_FALLBACK', '1')
    monkeypatch.setattr(intent_extractor, 'INTENT_CONCURRENCY', 2)
    monkeypatch.setattr(intent_extractor, 'generate_intent_from_code_async', generate)
    monkeypatch.setattr(intent_extractor, 'close_llm_async_client', close)
    with pytest.raises(RuntimeError, match='provider unavailable'):
        asyncio.run(intent_extractor._generate_intents_for_batch_async(
            [{'id': 'fail'}, {'id': 'pending'}], respect_internal=False
        ))
    assert events == ['pending_cancelled', 'client_closed']


def test_llm_async_client_is_recreated_after_batch_loop_closes(monkeypatch):
    created_loops = []
    closed_loops = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "{}"}}]}

    class LoopBoundAsyncClient:
        def __init__(self, **_kwargs):
            self.loop = asyncio.get_running_loop()
            created_loops.append(self.loop)

        async def post(self, *_args, **_kwargs):
            assert asyncio.get_running_loop() is self.loop
            return FakeResponse()

        async def aclose(self):
            assert asyncio.get_running_loop() is self.loop
            closed_loops.append(self.loop)

    monkeypatch.setattr(llm_utils.httpx, "AsyncClient", LoopBoundAsyncClient)
    client = llm_utils.LLMClient(
        llm_utils.LLMConfig(
            base_url="https://example.invalid",
            api_key="test-key",
            model="test-model",
        )
    )

    async def request_batch():
        await client._request_async({})
        await client.aclose()

    asyncio.run(request_batch())
    asyncio.run(request_batch())
    client._client.close()

    assert len(created_loops) == 2
    assert created_loops == closed_loops
    assert created_loops[0] is not created_loops[1]


def test_llm_client_sends_deepseek_nonthinking_profile(monkeypatch):
    monkeypatch.setenv("LLM_THINKING_TYPE", "disabled")
    monkeypatch.setenv("LLM_REASONING_EFFORT", "low")
    monkeypatch.setenv("LLM_MAX_TOKENS", "2000")
    captured = {}
    client = llm_utils.LLMClient(
        llm_utils.LLMConfig(
            base_url="https://api.deepseek.com",
            api_key="test-key",
            model="deepseek-v4-flash",
        )
    )

    def request(payload):
        captured.update(payload)
        return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    monkeypatch.setattr(client, "_request", request)
    assert client._chat("system", "user") == '{"ok": true}'
    client._client.close()

    assert captured["thinking"] == {"type": "disabled"}
    assert captured["max_tokens"] == 2000
    assert captured["temperature"] == 0.2
    assert "reasoning_effort" not in captured


def test_retry_delay_respects_429_retry_after(monkeypatch):
    request = llm_utils.httpx.Request("POST", "https://example.invalid")
    response = llm_utils.httpx.Response(
        429, headers={"Retry-After": "17"}, request=request
    )
    error = llm_utils.httpx.HTTPStatusError(
        "rate limited", request=request, response=response
    )
    monkeypatch.setattr(llm_utils, "INTENT_RETRY_BASE_SECONDS", 1.0)
    monkeypatch.setattr(llm_utils, "INTENT_RETRY_MAX_SECONDS", 30.0)

    assert llm_utils._retry_delay_seconds(error, 1) == 17.0
    assert llm_utils._retry_delay_seconds(RuntimeError("transient"), 3) == 4.0


def test_function_intent_origin_must_match_configured_generator(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "intent-model")
    assert intent_extractor._function_intent_generation_reusable(
        {"_intent_generation": "llm"}
    )
    assert not intent_extractor._function_intent_generation_reusable(
        {"_intent_generation": "heuristic"}
    )

    monkeypatch.delenv("LLM_MODEL")
    assert intent_extractor._function_intent_generation_reusable(
        {"_intent_generation": "heuristic"}
    )


def test_function_intent_build_reuses_exact_input_and_invalidates_graph_change(
    tmp_path,
    monkeypatch,
):
    functions = [
        {
            "id": "target",
            "fq_name": "api::target",
            "short_name": "target",
            "def_file": "src/api.cpp",
            "def_start_line": 1,
            "def_end_line": 20,
            "content_hash": "target-v1",
            "code": "int target() { return 1; }",
        },
        {
            "id": "target-test",
            "fq_name": "test_target",
            "short_name": "test_target",
            "def_file": "tests/api_test.cpp",
            "def_start_line": 1,
            "def_end_line": 10,
            "content_hash": "test-v1",
            "code": "TEST(api, target) { EXPECT_EQ(target(), 1); }",
            "is_test": True,
        },
    ]
    (tmp_path / "functions.jsonl").write_text(
        "".join(json.dumps(function) + "\n" for function in functions),
        encoding="utf-8",
    )

    generated_batches = []
    embedded_batches = []

    def generate(batch, **_kwargs):
        generated_batches.append([function["id"] for function in batch])
        return [(
            function,
            {
                "short_intent": "Return the target value.",
                "action": "return target",
                "object": "target value",
                "conditions": [],
                "side_effects": [],
                "raw_summary": "Returns the API target value.",
                "usage_context": "Used by callers and tests.",
                "_intent_generation": "llm",
            },
            True,
        ) for function in batch]

    def embed(texts):
        embedded_batches.append(list(texts))
        return np.asarray([[1.0, 0.0] for _ in texts], dtype="float32")

    monkeypatch.setenv("VOYAGE_EMBED_MODEL", "voyage-2")
    monkeypatch.setenv("LLM_MODEL", "intent-model")
    monkeypatch.setattr(intent_extractor, "EMBED_DIM", 2)
    monkeypatch.setattr(intent_extractor, "GlobalIntentCache", _MemoryIntentCache)
    monkeypatch.setattr(
        intent_extractor, "GlobalEmbeddingCache", _MemoryEmbeddingCache
    )
    monkeypatch.setattr(intent_extractor, "_generate_intents_for_batch", generate)
    monkeypatch.setattr(intent_extractor, "embed_texts", embed)

    intent_extractor.build_function_intents(tmp_path, repo_id="instance")
    first_record = json.loads((tmp_path / "function_intents.jsonl").read_text())
    first_hash = first_record["intent_input_hash"]
    assert generated_batches == [["target"]]
    assert len(embedded_batches) == 1
    assert intent_extractor.intent_function_artifacts_ready(tmp_path)

    original_prompt = llm_utils._FUNC_SYSTEM_PROMPT
    monkeypatch.setattr(
        llm_utils,
        "_FUNC_SYSTEM_PROMPT",
        original_prompt + "\nPrompt drift for readiness regression.",
    )
    assert not intent_extractor.intent_function_artifacts_ready(tmp_path)
    monkeypatch.setattr(llm_utils, "_FUNC_SYSTEM_PROMPT", original_prompt)
    assert intent_extractor.intent_function_artifacts_ready(tmp_path)

    generated_batches.clear()
    embedded_batches.clear()
    intent_extractor.build_function_intents(tmp_path, repo_id="instance")
    assert generated_batches == []
    assert embedded_batches == []

    (tmp_path / "callgraph.jsonl").write_text(
        json.dumps({
            "caller": "target-test",
            "callee": "target",
            "edge_type": "call",
        }) + "\n",
        encoding="utf-8",
    )
    assert not intent_extractor.intent_function_artifacts_ready(tmp_path)
    intent_extractor.build_function_intents(tmp_path, repo_id="instance")
    updated_record = json.loads((tmp_path / "function_intents.jsonl").read_text())
    manifest = load_embedding_manifest(tmp_path)["indexes"]["functions"]

    assert generated_batches == [["target"]]
    assert len(embedded_batches) == 1
    assert updated_record["intent_input_hash"] != first_hash
    assert updated_record["intent_builder"] == "function-intent-v4"
    assert manifest["builder"] == "function-intent-v4"
    assert manifest["model"] == "voyage-2"
    assert manifest["embedded_count"] == 1
    assert manifest["fallback_count"] == 0
    assert faiss.read_index(str(tmp_path / "func_index.faiss")).ntotal == 1
    assert intent_extractor.intent_function_artifacts_ready(tmp_path)


def test_function_build_deduplicates_exact_prompts_and_embedding_texts(
    tmp_path, monkeypatch
):
    functions = [
        {
            "id": func_id,
            "fq_name": "api::same_function",
            "short_name": "same_function",
            "def_file": "include/api.hpp",
            "def_start_line": 1,
            "def_end_line": 20,
            "content_hash": "same-source",
            "code": "int same_function() { return 1; }",
        }
        for func_id in ("variant-a", "variant-b")
    ]
    (tmp_path / "functions.jsonl").write_text(
        "".join(json.dumps(function) + "\n" for function in functions),
        encoding="utf-8",
    )
    generated = []
    embedded = []

    def generate(batch, **_kwargs):
        generated.extend(function["id"] for function in batch)
        return [
            (
                function,
                {
                    "short_intent": "Return one.",
                    "action": "return",
                    "object": "one",
                    "conditions": [],
                    "side_effects": [],
                    "raw_summary": "Returns one.",
                    "usage_context": "Public API.",
                    "_intent_generation": "llm",
                },
                True,
            )
            for function in batch
        ]

    def embed(texts):
        embedded.extend(texts)
        return np.asarray([[1.0, 0.0] for _ in texts], dtype="float32")

    monkeypatch.setenv("LLM_MODEL", "intent-model")
    monkeypatch.setenv("VOYAGE_EMBED_MODEL", "voyage-2")
    monkeypatch.setenv("CODE_INTENT_EMBED_ABORT_ON_FALLBACK", "1")
    monkeypatch.setattr(intent_extractor, "EMBED_DIM", 2)
    monkeypatch.setattr(intent_extractor, "GlobalIntentCache", _MemoryIntentCache)
    monkeypatch.setattr(
        intent_extractor, "GlobalEmbeddingCache", _MemoryEmbeddingCache
    )
    monkeypatch.setattr(intent_extractor, "_generate_intents_for_batch", generate)
    monkeypatch.setattr(intent_extractor, "embed_texts", embed)

    intent_extractor.build_function_intents(tmp_path, repo_id="project")

    records = [
        json.loads(line)
        for line in (tmp_path / "function_intents.jsonl").read_text().splitlines()
    ]
    manifest = load_embedding_manifest(tmp_path)["indexes"]["functions"]
    assert generated == ["variant-a"]
    assert len(embedded) == 1
    assert len(records) == 2
    assert len({record["intent_input_hash"] for record in records}) == 1
    assert faiss.read_index(str(tmp_path / "func_index.faiss")).ntotal == 2
    assert manifest["intent_duplicate_reuse"] == 1
    assert manifest["embedding_duplicate_reuse"] == 1


@pytest.mark.parametrize("kind", ["function", "test"])
def test_failed_generation_batch_keeps_successful_intents_for_resume(
    tmp_path, monkeypatch, kind
):
    functions = [
        {"id": name, "fq_name": f"api::{name}", "short_name": name,
         "def_file": "tests/api.cpp" if kind == "test" else "include/api.hpp",
         "def_start_line": index + 1, "def_end_line": index + 2,
         "content_hash": name, "code": f"int {name}() {{ return 1; }}",
         "is_test": kind == "test"}
        for index, name in enumerate(("first", "second"))
    ]
    (tmp_path / "functions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in functions)
    )
    monkeypatch.setenv("LLM_MODEL", "intent-model")
    monkeypatch.setenv("VOYAGE_EMBED_MODEL", "voyage-2")
    monkeypatch.setenv("INTENT_ABORT_ON_FALLBACK", "1")
    monkeypatch.setattr(intent_extractor, "GLOBAL_INTENT_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(intent_extractor, "INTENT_CONCURRENCY", 1)
    monkeypatch.setattr(intent_extractor, "EMBED_DIM", 2)
    monkeypatch.setattr(intent_extractor, "_should_embed_function", lambda _func: True)
    monkeypatch.setattr(intent_extractor, "GlobalEmbeddingCache", _MemoryEmbeddingCache)
    monkeypatch.setattr(intent_extractor, "embed_texts", lambda texts: np.asarray(
        [[1.0, 0.0] for _ in texts], dtype="float32"
    ))
    calls = []
    failing = True

    async def generate(func, *_args):
        calls.append(func["id"])
        if failing and func["id"] == "second":
            raise RuntimeError("provider unavailable")
        return {"short_intent": "Return one.", "action": "return", "object": "one",
                "conditions": [], "side_effects": [], "raw_summary": "Returns one.",
                "usage_context": "", "_intent_generation": "llm"}

    monkeypatch.setattr(intent_extractor, "generate_intent_from_code_async", generate)
    monkeypatch.setattr(intent_extractor, "generate_test_intent_from_code_async", generate)
    build = (intent_extractor.build_test_intents if kind == "test"
             else intent_extractor.build_function_intents)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        build(tmp_path, repo_id="project")
    assert calls == ["first", "second"]
    cache = intent_extractor.GlobalIntentCache(kind, repo_id="project")
    key = (llm_utils.test_intent_input_hash(functions[0]) if kind == "test"
           else llm_utils.function_intent_input_hash(functions[0]))
    assert cache.get(key)["_intent_generation"] == "llm"
    assert not (tmp_path / f"{'test' if kind == 'test' else 'function'}_intents.jsonl").exists()

    failing = False
    calls.clear()
    build(tmp_path, repo_id="project")
    assert calls == ["second"]


def test_test_build_generates_one_intent_for_identical_semantic_inputs(
    tmp_path, monkeypatch
):
    tests = [
        {"id": test_id, "fq_name": "Validate one", "def_file": "tests/api.cpp",
         "def_start_line": line, "def_end_line": line + 1, "is_test": True,
         "code": "CHECK(value() == 1);", "content_hash": "same-source"}
        for test_id, line in (("first", 1), ("second", 20))
    ]
    (tmp_path / "functions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in tests))
    monkeypatch.setenv("LLM_MODEL", "intent-model")
    monkeypatch.setenv("VOYAGE_EMBED_MODEL", "voyage-2")
    monkeypatch.setattr(intent_extractor, "GLOBAL_INTENT_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(intent_extractor, "EMBED_DIM", 2)
    monkeypatch.setattr(intent_extractor, "GlobalEmbeddingCache", _MemoryEmbeddingCache)
    monkeypatch.setattr(intent_extractor, "embed_texts", lambda texts: np.asarray(
        [[1.0, 0.0] for _ in texts], dtype="float32"
    ))
    calls = []

    async def generate(test):
        calls.append(test["id"])
        return {"short_intent": "Validate one.", "_intent_generation": "llm"}

    monkeypatch.setattr(intent_extractor, "generate_test_intent_from_code_async", generate)
    intent_extractor.build_test_intents(tmp_path, repo_id="project")

    assert calls == ["first"]
    records = [json.loads(line) for line in (tmp_path / "test_intents.jsonl").read_text().splitlines()]
    assert {row["func_id"] for row in records} == {"first", "second"}
    assert {row["start_line"] for row in records} == {1, 20}
    assert faiss.read_index(str(tmp_path / "test_index.faiss")).ntotal == 2


def test_generation_checkpoints_do_not_limit_embedding_batch_size(
    tmp_path,
    monkeypatch,
):
    generated_batches = []
    embedded_batches = []

    def generate(batch, **_kwargs):
        generated_batches.append([record["id"] for record in batch])
        return [
            (
                record,
                {
                    "short_intent": "Validate or return the requested value.",
                    "action": "validate value" if record.get("is_test") else "return value",
                    "object": "requested value",
                    "conditions": [],
                    "side_effects": [],
                    "raw_summary": "Handles the requested value.",
                    "usage_context": "Used by public callers.",
                    "oracle": "The result matches the expected value.",
                    "fixture": "",
                    "covered_symbols": [],
                    "_intent_generation": "llm",
                },
                True,
            )
            for record in batch
        ]

    def embed(texts):
        embedded_batches.append(list(texts))
        return np.asarray([[1.0, 0.0] for _ in texts], dtype="float32")

    monkeypatch.setenv("LLM_MODEL", "intent-model")
    monkeypatch.setenv("VOYAGE_EMBED_MODEL", "voyage-2")
    monkeypatch.setattr(intent_extractor, "BATCH_SIZE", 2)
    monkeypatch.setattr(intent_extractor, "EMBED_BATCH_SIZE", 3)
    monkeypatch.setattr(intent_extractor, "EMBED_DIM", 2)
    monkeypatch.setattr(intent_extractor, "GlobalIntentCache", _MemoryIntentCache)
    monkeypatch.setattr(
        intent_extractor,
        "GlobalEmbeddingCache",
        _MemoryEmbeddingCache,
    )
    monkeypatch.setattr(intent_extractor, "_generate_intents_for_batch", generate)
    monkeypatch.setattr(intent_extractor, "embed_texts", embed)

    function_dir = tmp_path / "functions"
    function_dir.mkdir()
    functions = [
        {
            "id": f"function-{index}",
            "fq_name": f"api::function_{index}",
            "short_name": f"function_{index}",
            "def_file": "include/api.hpp",
            "def_start_line": index + 1,
            "def_end_line": index + 2,
            "content_hash": f"function-hash-{index}",
            "code": f"int function_{index}() {{ return {index}; }}",
        }
        for index in range(5)
    ]
    (function_dir / "functions.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in functions),
        encoding="utf-8",
    )

    intent_extractor.build_function_intents(function_dir, repo_id="project")

    assert [len(batch) for batch in generated_batches] == [2, 2, 1]
    assert [len(batch) for batch in embedded_batches] == [3, 2]

    generated_batches.clear()
    embedded_batches.clear()
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    tests = [
        {
            "id": f"test-{index}",
            "fq_name": f"test_value_{index}",
            "def_file": "tests/value_test.cpp",
            "def_start_line": index + 1,
            "def_end_line": index + 2,
            "is_test": True,
            "framework": "catch2",
            "content_hash": f"test-hash-{index}",
            "code": f"TEST_CASE(\"value {index}\") {{ CHECK(value() == {index}); }}",
        }
        for index in range(5)
    ]
    (test_dir / "functions.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in tests),
        encoding="utf-8",
    )

    intent_extractor.build_test_intents(test_dir, repo_id="project")

    assert [len(batch) for batch in generated_batches] == [2, 2, 1]
    assert [len(batch) for batch in embedded_batches] == [3, 2]


def test_build_test_intents_writes_searchable_vector_artifacts(
    tmp_path,
    monkeypatch,
):
    functions = [
        {
            "id": "test-widget",
            "fq_name": "test_widget_parse",
            "def_file": "tests/widget_test.cpp",
            "def_start_line": 5,
            "def_end_line": 9,
            "is_test": True,
            "framework": "gtest",
            "oracle_code": "EXPECT_TRUE(Widget::parse(input));",
            "focal_entities": [{
                "name": "Widget::parse",
                "relations": ["call"],
                "short_intent": "Parse Widget input.",
                "raw_summary": "Validate and parse the supplied input.",
                "source": "bool Widget::parse(Input input) { return input.valid(); }",
            }],
            "code": "void test_widget_parse() { Widget::parse(input); }",
            "content_hash": "test-hash",
        },
        {
            "id": "widget-parse",
            "fq_name": "Widget::parse",
            "def_file": "src/widget.cpp",
            "def_start_line": 10,
            "def_end_line": 20,
            "is_test": False,
            "content_hash": "prod-hash",
        },
    ]
    with (tmp_path / "functions.jsonl").open("w", encoding="utf-8") as stream:
        for record in functions:
            stream.write(json.dumps(record) + "\n")
    (tmp_path / "callgraph.jsonl").write_text(
        json.dumps({"caller": "test-widget", "callee": "widget-parse"}) + "\n",
        encoding="utf-8",
    )

    def generate(batch, **_kwargs):
        return [(
            function,
            {
                "short_intent": "Validate parsing string input.",
                "action": "parse input",
                "object": "Widget",
                "conditions": ["input is a string"],
                "side_effects": [],
                "raw_summary": "Checks the public parser.",
                "usage_context": "Public API validation.",
                "oracle": "Parsing succeeds and returns true.",
                "fixture": "Widget fixture",
                "covered_symbols": ["Widget::parse"],
                "_intent_generation": "heuristic",
            },
            True,
        ) for function in batch]

    embedded_texts = []

    def embed(texts):
        embedded_texts.extend(texts)
        return np.asarray([[1.0, 0.0] for _text in texts], dtype="float32")

    monkeypatch.setattr(intent_extractor, "EMBED_DIM", 2)
    monkeypatch.setattr(intent_extractor, "GlobalIntentCache", _MemoryIntentCache)
    embedding_cache = _MemoryEmbeddingCache()
    monkeypatch.setattr(
        intent_extractor,
        "GlobalEmbeddingCache",
        lambda *_args, **_kwargs: embedding_cache,
    )
    monkeypatch.setenv("CODE_INTENT_EMBED_ABORT_ON_FALLBACK", "1")
    monkeypatch.setattr(intent_extractor, "_generate_intents_for_batch", generate)
    monkeypatch.setattr(intent_extractor, "embed_texts", embed)

    intent_extractor.build_test_intents(tmp_path, repo_id="instance")

    assert intent_extractor.intent_test_artifacts_ready(tmp_path)
    assert len((tmp_path / "test_intents.jsonl").read_text().splitlines()) == 1
    assert json.loads((tmp_path / "test_meta.jsonl").read_text())["record_index"] == 0
    assert faiss.read_index(str(tmp_path / "test_index.faiss")).ntotal == 1
    assert "Oracle: Parsing succeeds and returns true." in embedded_texts[-1]
    assert "Widget fixture" in embedded_texts[-1]
    assert "Validation intent: Validate parsing string input." in embedded_texts[0]
    assert "Widget::parse(input)" in embedded_texts[0]
    assert "Oracle: Parsing succeeds and returns true." in embedded_texts[0]
    assert "Production intent: Parse Widget input." in embedded_texts[0]
    assert "Production source skeleton:" in embedded_texts[0]

    def unexpected_generation(*_args, **_kwargs):
        raise AssertionError("existing test intents should be reused")

    monkeypatch.setattr(
        intent_extractor,
        "_generate_intents_for_batch",
        unexpected_generation,
    )
    first_embedding_call_count = len(embedded_texts)
    intent_extractor.build_test_intents(tmp_path, repo_id="instance")

    assert faiss.read_index(str(tmp_path / "test_index.faiss")).ntotal == 1
    assert len(embedded_texts) == first_embedding_call_count
    manifest = load_embedding_manifest(tmp_path)["indexes"]["tests"]
    assert manifest["embedding_cache_hits"] == 1
    assert manifest["embedding_requests"] == 0
    build_id = manifest["build_id"]
    for name in ("test_cases.jsonl", "test_intents.jsonl", "test_meta.jsonl"):
        records = [
            json.loads(line)
            for line in (tmp_path / name).read_text(encoding="utf-8").splitlines()
            if line
        ]
        assert {record["_build_id"] for record in records} == {build_id}

    with monkeypatch.context() as runtime:
        runtime.setenv("LLM_MODEL", "different-runtime-model")
        assert not intent_extractor.intent_test_artifacts_ready(tmp_path)
        assert intent_extractor.intent_test_artifacts_ready(
            tmp_path,
            require_current_generation=False,
        )

    metadata = json.loads((tmp_path / "test_meta.jsonl").read_text())
    metadata["_build_id"] = "interrupted-build"
    (tmp_path / "test_meta.jsonl").write_text(
        json.dumps(metadata) + "\n",
        encoding="utf-8",
    )
    assert not intent_extractor.intent_test_artifacts_ready(tmp_path)


def test_strict_test_intent_generation_propagates_provider_failures(monkeypatch):
    async def fail_generation(_function):
        raise RuntimeError("provider unavailable")

    monkeypatch.setenv("INTENT_ABORT_ON_FALLBACK", "1")
    monkeypatch.setattr(
        intent_extractor,
        "generate_test_intent_from_code_async",
        fail_generation,
    )

    with pytest.raises(RuntimeError, match="provider unavailable"):
        intent_extractor._generate_intents_for_batch(
            [{
                "id": "test-widget",
                "fq_name": "test_widget",
                "is_test": True,
                "code": "TEST(Widget, Parses) {}",
            }],
            respect_internal=False,
        )


def test_strict_function_intent_generation_propagates_provider_failures(monkeypatch):
    async def fail_generation(_function, _context):
        raise RuntimeError("provider unavailable")

    monkeypatch.setenv("INTENT_ABORT_ON_FALLBACK", "1")
    monkeypatch.setattr(
        intent_extractor,
        "generate_intent_from_code_async",
        fail_generation,
    )

    with pytest.raises(RuntimeError, match="provider unavailable"):
        intent_extractor._generate_intents_for_batch(
            [{
                "id": "widget",
                "fq_name": "Widget::parse",
                "short_name": "parse",
                "def_file": "src/widget.cpp",
                "def_start_line": 1,
                "def_end_line": 20,
                "code": "bool Widget::parse() { return true; }",
            }],
            respect_internal=False,
        )
