import json

import faiss
import numpy as np
import pytest

from code_intent.code_intent.embedding_manifest import (
    TEST_INTENT_BUILDER,
    write_embedding_manifest_entry,
)
from code_intent.code_intent.intent_extractor import intent_test_artifacts_ready
from code_intent.code_intent.test_case_extractor import extract_test_cases
from INTENT2FIX.intent_dialogue.dialogue import IntentDialogue
from INTENT2FIX.intent_dialogue.schema import (
    IntentAnchorSelectionGroup,
    IntentTestAnchorGroup,
    TestAnchorAssessment as AnchorAssessment,
    TestAnchorCandidate as AnchorCandidate,
    TestAnchorRetrieval as AnchorRetrieval,
    TestAnchorSelection as AnchorSelection,
)
from INTENT2FIX.intent_dialogue.test_retrieval import (
    _rrf_score,
    TestAnchorRetriever as AnchorRetriever,
    IntentEmbeddingUnavailableError,
)
from INTENT2FIX.repository.repository import InMemRepository
from INTENT2FIX.reproduce.intent_schema import (
    IssueClaim,
    IssueEvidenceFrame,
    IssueIntentUnit,
)


def issue_evidence() -> IssueEvidenceFrame:
    return IssueEvidenceFrame(
        claims=[
            IssueClaim(
                id="EXP-1",
                kind="expected_behavior",
                text="Widget::parse should accept string_view input.",
                issue_span="Widget::parse should accept string_view input.",
                search_terms=["Widget::parse", "string_view"],
            )
        ],
        symbols=["Widget::parse"],
    )


def test_anchor_retrieval_prefers_issue_matching_test_source():
    repository = InMemRepository(files={
        "tests/widget_parse_test.cpp": """
TEST(Widget, ExistingInput) {
  EXPECT_TRUE(Widget::parse(existing_input));
}
""",
        "tests/unrelated_test.cpp": """
TEST(Storage, OpensFile) {
  EXPECT_TRUE(Storage::open(path));
}
""",
        "src/widget.cpp": "bool Widget::parse(Input input) { return true; }",
    })

    result = AnchorRetriever(
        repository,
    ).retrieve(
        "Widget::parse should accept string_view input.",
        issue_evidence(),
    )

    assert result.status == "candidates_found"
    assert result.candidates[0].path == "tests/widget_parse_test.cpp"
    assert result.candidates[0].claim_ids == ["EXP-1"]
    assert result.candidates[0].retrieval_ranks == {"lexical": 1}
    assert "Widget::parse" in result.candidates[0].snippet
    assert all(candidate.path != "src/widget.cpp" for candidate in result.candidates)


def test_anchor_retrieval_keeps_recall_groups_per_atomic_intent():
    repository = InMemRepository(files={
        "tests/parse_text_test.cpp": """
TEST(Widget, ParseText) { EXPECT_TRUE(Widget::parse_text(input)); }
""",
        "tests/cursor_state_test.cpp": """
TEST(Widget, CursorState) { EXPECT_EQ(Widget::cursor_position(), 0); }
""",
    })
    evidence = IssueEvidenceFrame(
        claims=[
            IssueClaim(
                id="EXP-1",
                kind="expected_behavior",
                text="Widget::parse_text accepts string input.",
                issue_span="Widget::parse_text accepts string input.",
                search_terms=["Widget::parse_text"],
            ),
            IssueClaim(
                id="PRES-1",
                kind="preservation",
                text="Widget::cursor_position remains unchanged after failure.",
                issue_span="Widget::cursor_position remains unchanged after failure.",
                search_terms=["Widget::cursor_position"],
            ),
        ],
        intent_units=[
            IssueIntentUnit(
                id="INTENT-1",
                role="target",
                text="Widget::parse_text accepts string input.",
                issue_span="Widget::parse_text accepts string input.",
                source_claim_ids=["EXP-1"],
            ),
            IssueIntentUnit(
                id="INTENT-2",
                role="preserve",
                text="Widget::cursor_position remains unchanged after failure.",
                issue_span="Widget::cursor_position remains unchanged after failure.",
                source_claim_ids=["PRES-1"],
            ),
        ],
    )

    result = AnchorRetriever(repository).retrieve("Two-part issue", evidence)

    groups = {group.intent_id: group for group in result.intent_groups}
    candidates = {candidate.id: candidate for candidate in result.candidates}
    assert set(groups) == {"INTENT-1", "INTENT-2"}
    assert groups["INTENT-1"].role == "target"
    assert groups["INTENT-2"].role == "preserve"
    assert any(
        candidates[candidate_id].path == "tests/parse_text_test.cpp"
        for candidate_id in groups["INTENT-1"].candidate_ids
    )
    assert any(
        candidates[candidate_id].path == "tests/cursor_state_test.cpp"
        for candidate_id in groups["INTENT-2"].candidate_ids
    )
    assert all(
        group.intent_id in candidates[candidate_id].intent_ids
        for group in result.intent_groups
        for candidate_id in group.candidate_ids
    )


def test_anchor_selection_cannot_cross_atomic_intent_groups():
    retrieval = AnchorRetrieval(
        status="candidates_found",
        candidates=[
            AnchorCandidate(
                id="TEST-CAND-1",
                path="tests/first.cpp",
                locator="tests/first.cpp:1",
                intent_ids=["INTENT-1"],
            ),
            AnchorCandidate(
                id="TEST-CAND-2",
                path="tests/second.cpp",
                locator="tests/second.cpp:1",
                intent_ids=["INTENT-2"],
            ),
        ],
        intent_groups=[
            IntentTestAnchorGroup(
                intent_id="INTENT-1",
                role="target",
                candidate_ids=["TEST-CAND-1"],
            ),
            IntentTestAnchorGroup(
                intent_id="INTENT-2",
                role="target",
                candidate_ids=["TEST-CAND-2"],
            ),
        ],
    )
    selection = AnchorSelection(
        status="selected",
        selected_candidate_ids=["TEST-CAND-1"],
        primary_candidate_id="TEST-CAND-1",
        rationale="The first intent provides the strongest initial witness.",
        intent_selections=[
            IntentAnchorSelectionGroup(
                intent_id="INTENT-1",
                selected_candidate_ids=["TEST-CAND-1", "TEST-CAND-2"],
                primary_candidate_id="TEST-CAND-1",
                rationale="Only the first candidate belongs to this intent.",
            ),
            IntentAnchorSelectionGroup(
                intent_id="INTENT-2",
                selected_candidate_ids=["TEST-CAND-2"],
                primary_candidate_id="TEST-CAND-2",
                rationale="The second candidate exercises the second intent.",
            ),
        ],
    )

    normalized = IntentDialogue._normalize_test_anchor_selection(
        selection,
        retrieval,
    )

    assert normalized is not None
    groups = {item.intent_id: item for item in normalized.intent_selections}
    assert groups["INTENT-1"].selected_candidate_ids == ["TEST-CAND-1"]
    assert groups["INTENT-2"].selected_candidate_ids == ["TEST-CAND-2"]


def test_anchor_retrieval_discovers_wrapper_from_project_header():
    repository = InMemRepository(files={
        "include/project/macros.hpp": (
            "#define PROJECT_TEST(suite, name) TEST(suite, name)"
        ),
        "tests/widget_parse_test.cpp": """
#include "project/macros.hpp"
PROJECT_TEST(Widget, ExistingInput) {
  EXPECT_TRUE(Widget::parse(existing_input));
}
""",
    })

    result = AnchorRetriever(repository).retrieve(
        "Widget::parse should accept string_view input.",
        issue_evidence(),
    )

    assert result.candidates[0].test_name == "Widget.ExistingInput"


def test_anchor_retrieval_returns_at_most_five_files():
    repository = InMemRepository(files={
        f"tests/widget_parse_{index}_test.cpp": (
            f"TEST(Widget, Case{index}) {{ Widget::parse(existing_input); }}"
        )
        for index in range(7)
    })

    result = AnchorRetriever(repository, max_candidates=10).retrieve(
        "Widget::parse should accept string_view input.",
        issue_evidence(),
    )

    assert len(result.candidates) == 5
    assert [candidate.id for candidate in result.candidates] == [
        "TEST-CAND-1",
        "TEST-CAND-2",
        "TEST-CAND-3",
        "TEST-CAND-4",
        "TEST-CAND-5",
    ]


def test_anchor_retrieval_uses_identifier_components_for_analogs():
    repository = InMemRepository(files={
        "tests/object_count_test.cpp": """
TEST(Object, ExistingFields) {
  EXPECT_EQ(array.count_elements(), 2);
  EXPECT_EQ(array.count_elements(), 2);
}
""",
        "tests/unrelated_test.cpp": "Storage::open(existing_input);",
    })

    result = AnchorRetriever(repository).retrieve(
        "Add count_fields for objects.",
        IssueEvidenceFrame(claims=[
            IssueClaim(
                id="EXP-1",
                kind="expected_behavior",
                text="Add count_fields for objects.",
                issue_span="Add count_fields for objects.",
                search_terms=["count_fields", "objects"],
            )
        ], symbols=["count_fields"]),
    )

    candidate = result.candidates[0]
    assert candidate.path == "tests/object_count_test.cpp"
    assert candidate.retrieval_ranks == {"lexical": 1}
    assert "count" in result.search_terms


def test_anchor_retrieval_excludes_production_files_with_test_in_stem():
    repository = InMemRepository(files={
        "src/catch_test_spec_parser.cpp": "CHECKED_IF(flag);",
        "tests/Misc.tests.cpp": (
            'TEST_CASE("checked if branch") { CHECKED_IF(flag); }'
        ),
    })

    result = AnchorRetriever(repository).retrieve(
        "CHECKED_IF should be usable.",
        IssueEvidenceFrame(symbols=["CHECKED_IF"]),
    )

    assert [candidate.path for candidate in result.candidates] == [
        "tests/Misc.tests.cpp"
    ]


def test_anchor_retrieval_caps_claim_bindings_and_prefers_exact_symbol_line():
    claims = [
        IssueClaim(
            id=f"EXP-{index}",
            kind="expected_behavior",
            text="CHECKED_IF should preserve the requested behavior.",
            issue_span="CHECKED_IF should preserve the requested behavior.",
            search_terms=["CHECKED_IF", "requested behavior"],
        )
        for index in range(1, 14)
    ]
    repository = InMemRepository(files={
        "tests/Misc.tests.cpp": """
TEST_CASE("requested behavior") {
  CHECKED_IF(flag);
  run_branch();
}
""",
    })

    result = AnchorRetriever(repository).retrieve(
        "CHECKED_IF should preserve the requested behavior.",
        IssueEvidenceFrame(claims=claims, symbols=["CHECKED_IF"]),
    )

    candidate = result.candidates[0]
    assert len(candidate.claim_ids) == 10
    assert candidate.locator.startswith("tests/Misc.tests.cpp:2-")
    assert "CHECKED_IF(flag)" in candidate.snippet


def test_anchor_selection_normalization_rejects_unknown_candidates():
    retrieval = AnchorRetrieval(
        status="candidates_found",
        query="Widget::parse string_view",
        candidates=[
            AnchorCandidate(
                id="TEST-CAND-1",
                path="tests/widget_parse_test.cpp",
                locator="tests/widget_parse_test.cpp:1-10",
            )
        ],
    )
    selection = AnchorSelection(
        status="selected",
        selected_candidate_ids=["TEST-CAND-9", "TEST-CAND-1"],
        primary_candidate_id="TEST-CAND-9",
        rationale="The first candidate matches the public operation and oracle.",
        assessments=[
            AnchorAssessment(
                candidate_id="TEST-CAND-9",
                suitable=True,
                reason="Invented candidate.",
            ),
            AnchorAssessment(
                candidate_id="TEST-CAND-1",
                suitable=True,
                reason="Uses the public operation and an observable assertion.",
            ),
        ],
    )

    normalized = IntentDialogue._normalize_test_anchor_selection(
        selection,
        retrieval,
    )

    assert normalized is not None
    assert normalized.selected_candidate_ids == ["TEST-CAND-1"]
    assert normalized.primary_candidate_id == "TEST-CAND-1"
    assert [item.candidate_id for item in normalized.assessments] == [
        "TEST-CAND-1"
    ]


def test_anchor_retrieval_returns_explicit_empty_result():
    result = AnchorRetriever(
        InMemRepository(files={"src/widget.cpp": "Widget::parse(value);"}),
    ).retrieve("Widget parse failure", IssueEvidenceFrame())

    assert result.status == "no_candidates"


def test_anchor_retrieval_query_keeps_original_issue_when_claims_are_generic():
    repository = InMemRepository(
        files={"tests/count_test.cpp": "TEST(Count, Empty) {}"}
    )
    evidence = IssueEvidenceFrame.model_validate({
        "claims": [{
            "id": "OBS-1",
            "kind": "observed_behavior",
            "text": "The reported behavior is incorrect.",
            "issue_span": "Add count_fields method for objects",
        }],
    })

    result = AnchorRetriever(repository).retrieve(
        "Add count_fields method for objects",
        evidence,
    )

    assert "count_fields" in result.query
    assert result.candidates[0].test_name == "Count.Empty"


def test_anchor_retrieval_uses_test_intent_index_without_source_keywords(tmp_path):
    build_id = "semantic-build"
    source = "TEST(Api, Behavior) { EXPECT_TRUE(run_case()); }"
    case = extract_test_cases("tests/semantic_test.cpp", source)[0]
    case["_build_id"] = build_id
    repository = InMemRepository(files={
        "tests/semantic_test.cpp": source,
        "tests/unrelated_test.cpp": "TEST(Storage, File) { EXPECT_TRUE(run_case()); }",
    })
    records = [{
        "_build_id": build_id,
        "func_id": case["id"],
        "signature": "Api_Behavior_Test",
        "file": "tests/semantic_test.cpp",
        "start_line": 1,
        "short_intent": "Validate that Widget::parse accepts string_view input.",
        "action": "parse string_view input",
        "object": "Widget",
    }]
    meta = [{
        "_build_id": build_id,
        "index": 0,
        "record_index": 0,
        "func_id": case["id"],
        "file": "tests/semantic_test.cpp",
        "start_line": 1,
    }]
    (tmp_path / "test_intents.jsonl").write_text(
        json.dumps(records[0]) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "test_meta.jsonl").write_text(
        json.dumps(meta[0]) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "test_cases.jsonl").write_text(
        json.dumps(case) + "\n",
        encoding="utf-8",
    )
    index = faiss.IndexFlatIP(2)
    index.add(np.asarray([[1.0, 0.0]], dtype="float32"))
    faiss.write_index(index, str(tmp_path / "test_index.faiss"))
    write_embedding_manifest_entry(
        tmp_path,
        "tests",
        model=None,
        dimension=2,
        metadata={
            "builder": TEST_INTENT_BUILDER,
            "build_id": build_id,
            "case_count": 1,
            "fallback_count": 0,
        },
    )

    result = AnchorRetriever(
        repository,
        index_dir=tmp_path,
        query_embedder=lambda _query: np.asarray([1.0, 0.0], dtype="float32"),
    ).retrieve(
        "Widget::parse should accept string_view input.",
        issue_evidence(),
    )

    assert result.candidates[0].path == "tests/semantic_test.cpp"
    assert result.candidates[0].retrieval_ranks == {"test_intent": 1}
    assert result.candidates[0].claim_ids == ["EXP-1"]
    assert result.query.startswith("Original issue:")
    assert "Requested outcome:" in result.query
    assert "Public symbols: Widget::parse" in result.query


def test_anchor_retrieval_raises_when_strict_semantic_embedding_fails(
    tmp_path,
    monkeypatch,
):
    build_id = "strict-semantic-build"
    source = "TEST(Api, Behavior) { EXPECT_TRUE(run_case()); }"
    repository = InMemRepository(files={"tests/api_test.cpp": source})
    case = extract_test_cases("tests/api_test.cpp", source)[0]
    case["_build_id"] = build_id
    intent = {
        "_build_id": build_id,
        "func_id": case["id"],
        "test_case_id": case["id"],
        "file": "tests/api_test.cpp",
        "start_line": 1,
        "short_intent": "Exercise the public API behavior.",
    }
    (tmp_path / "test_cases.jsonl").write_text(
        json.dumps(case) + "\n", encoding="utf-8"
    )
    (tmp_path / "test_intents.jsonl").write_text(
        json.dumps(intent) + "\n", encoding="utf-8"
    )
    (tmp_path / "test_meta.jsonl").write_text(
        json.dumps({"record_index": 0, **intent}) + "\n",
        encoding="utf-8",
    )
    index = faiss.IndexFlatIP(2)
    index.add(np.asarray([[1.0, 0.0]], dtype="float32"))
    faiss.write_index(index, str(tmp_path / "test_index.faiss"))
    write_embedding_manifest_entry(
        tmp_path,
        "tests",
        model=None,
        dimension=2,
        metadata={
            "builder": TEST_INTENT_BUILDER,
            "build_id": build_id,
            "case_count": 1,
            "fallback_count": 0,
        },
    )
    monkeypatch.setenv("CODE_INTENT_EMBED_ABORT_ON_FALLBACK", "1")

    def fail_embedding(_query):
        raise ConnectionError("embedding endpoint unavailable")

    with pytest.raises(IntentEmbeddingUnavailableError, match="strict embedding mode"):
        AnchorRetriever(
            repository,
            index_dir=tmp_path,
            query_embedder=fail_embedding,
        ).retrieve("Public API behavior should change.", IssueEvidenceFrame())


def test_anchor_retrieval_replaces_stale_test_case_metadata(tmp_path):
    source = "TEST(Api, CurrentBehavior) { EXPECT_TRUE(run_current_behavior()); }"
    repository = InMemRepository(files={"tests/api_test.cpp": source})
    stale_case = extract_test_cases("tests/api_test.cpp", source)[0]
    stale_case["extractor_version"] = 3
    stale_case["fq_name"] = "stale metadata"
    stale_case["short_name"] = "stale metadata"
    stale_case["code"] = "TEST_CASE(\"stale metadata\") {}"
    (tmp_path / "test_cases.jsonl").write_text(
        json.dumps(stale_case) + "\n",
        encoding="utf-8",
    )

    result = AnchorRetriever(repository, index_dir=tmp_path).retrieve(
        "Current behavior should remain available.",
        IssueEvidenceFrame(symbols=["run_current_behavior"]),
    )

    assert result.candidates[0].test_name == "Api.CurrentBehavior"
    assert "stale metadata" not in result.candidates[0].snippet


def test_anchor_retrieval_uses_test_to_production_focal_edges(tmp_path):
    repository = InMemRepository(files={
        "tests/api_test.cpp": "TEST(Api, Behavior) { EXPECT_TRUE(run_case()); }",
        "tests/unrelated_test.cpp": "TEST(Storage, File) { EXPECT_TRUE(run_case()); }",
    })
    functions = [{
        "id": "test-api",
        "def_file": "tests/api_test.cpp",
        "def_start_line": 1,
        "is_test": True,
    }]
    callgraph = [{
        "caller": "test-api",
        "caller_file": "tests/api_test.cpp",
        "callee": "widget-parse",
        "callee_name": "Widget::parse",
        "callee_file": "src/widget.cpp",
    }]
    (tmp_path / "functions.jsonl").write_text(
        json.dumps(functions[0]) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "callgraph.jsonl").write_text(
        json.dumps(callgraph[0]) + "\n",
        encoding="utf-8",
    )

    result = AnchorRetriever(repository, index_dir=tmp_path).retrieve(
        "Widget::parse should accept string_view input.",
        issue_evidence(),
    )

    assert result.candidates[0].path == "tests/api_test.cpp"
    assert result.candidates[0].retrieval_ranks == {"focal_entity": 1}
    assert "Widget::parse" in result.candidates[0].label


def test_anchor_retrieval_recovers_focal_binding_when_callgraph_is_sparse(tmp_path):
    repository = InMemRepository(files={
        "tests/api_test.cpp": "TEST(Api, Behavior) { EXPECT_TRUE(run_case()); }",
    })
    functions = [
        {
            "id": "test-api",
            "def_file": "tests/api_test.cpp",
            "def_start_line": 1,
            "is_test": True,
            "code": "TEST(Api, Behavior) { Widget::parse(input); }",
        },
        {
            "id": "widget-parse",
            "fq_name": "Widget::parse",
            "short_name": "parse",
            "def_file": "src/widget.cpp",
            "is_test": False,
        },
    ]
    with (tmp_path / "functions.jsonl").open("w", encoding="utf-8") as stream:
        for record in functions:
            stream.write(json.dumps(record) + "\n")

    result = AnchorRetriever(repository, index_dir=tmp_path).retrieve(
        "Widget::parse should accept string_view input.",
        issue_evidence(),
    )

    assert result.candidates[0].path == "tests/api_test.cpp"
    assert result.candidates[0].retrieval_ranks == {"focal_entity": 1}
    assert "parse" in result.candidates[0].label


def test_focal_fallback_ignores_symbols_only_in_comments_and_strings(tmp_path):
    repository = InMemRepository(files={
        "tests/api_test.cpp": '''TEST(Api, DescriptionOnly) {
  // Widget::parse(input);
  const char* note = "Widget::parse(input)";
  EXPECT_TRUE(true);
}''',
    })
    (tmp_path / "functions.jsonl").write_text(
        json.dumps({
            "id": "widget-parse",
            "fq_name": "Widget::parse",
            "short_name": "parse",
            "def_file": "src/widget.cpp",
            "is_test": False,
        }) + "\n",
        encoding="utf-8",
    )

    result = AnchorRetriever(repository, index_dir=tmp_path).retrieve(
        "Widget::parse should accept string_view input.",
        issue_evidence(),
    )

    assert "focal_entity" not in result.candidates[0].retrieval_ranks


def test_anchor_retrieval_uses_precomputed_case_focals_without_callgraph(tmp_path):
    source = "TEST(Api, Behavior) { EXPECT_TRUE(run_case()); }"
    repository = InMemRepository(files={"tests/api_test.cpp": source})
    case = extract_test_cases("tests/api_test.cpp", source)[0]
    case["focal_entities"] = [{
        "name": "Widget::parse",
        "file": "src/widget.cpp",
        "short_intent": "Parse string input.",
        "raw_summary": "Accept and validate a string view.",
    }]
    (tmp_path / "test_cases.jsonl").write_text(
        json.dumps(case) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "functions.jsonl").write_text("", encoding="utf-8")

    result = AnchorRetriever(repository, index_dir=tmp_path).retrieve(
        "Widget::parse should accept string_view input.",
        issue_evidence(),
    )

    assert result.candidates[0].test_case_id == case["id"]
    assert result.candidates[0].retrieval_ranks == {"focal_entity": 1}


def test_anchor_retrieval_keeps_multiple_cases_from_one_test_file(tmp_path):
    repository = InMemRepository(files={
        "tests/api_test.cpp": """
TEST(Api, Existing) { EXPECT_TRUE(Widget::parse(existing)); }
TEST(Api, Candidate) { EXPECT_TRUE(Widget::parse(candidate)); }
""",
    })
    functions = [
        {
            "id": "test-existing",
            "def_file": "tests/api_test.cpp",
            "def_start_line": 2,
            "is_test": True,
            "code": "TEST(Api, Existing) { Widget::parse(existing); }",
        },
        {
            "id": "test-candidate",
            "def_file": "tests/api_test.cpp",
            "def_start_line": 3,
            "is_test": True,
            "code": "TEST(Api, Candidate) { Widget::parse(candidate); }",
        },
        {
            "id": "widget-parse",
            "fq_name": "Widget::parse",
            "short_name": "parse",
            "def_file": "src/widget.cpp",
            "is_test": False,
        },
    ]
    with (tmp_path / "functions.jsonl").open("w", encoding="utf-8") as stream:
        for record in functions:
            stream.write(json.dumps(record) + "\n")

    result = AnchorRetriever(repository, index_dir=tmp_path).retrieve(
        "Widget::parse should accept string_view input.",
        issue_evidence(),
    )

    assert len(result.candidates) == 2
    assert [candidate.test_name for candidate in result.candidates] == [
        "Api.Existing",
        "Api.Candidate",
    ]
    assert result.candidates[0].retrieval_ranks == {
        "lexical": 1,
        "focal_entity": 1,
    }


def test_anchor_retrieval_does_not_spend_slots_on_parent_and_child():
    repository = InMemRepository(files={
        "tests/access.tests.cpp": r'''
TEST_CASE("element access") {
  json value = json::object();
  SECTION("numeric key") {
    CHECK_THROWS_WITH(value[0], "numeric key rejected");
  }
  SECTION("string key") {
    CHECK(value["name"] == nullptr);
  }
}
''',
    })
    evidence = IssueEvidenceFrame(claims=[
        IssueClaim(
            id="EXP-1",
            kind="expected_behavior",
            text="Element access with a numeric key should report a numeric-key error.",
            issue_span="numeric key",
            search_terms=["element access", "numeric key"],
        )
    ])

    result = AnchorRetriever(repository).retrieve(
        "Element access with a numeric key should report a numeric-key error.",
        evidence,
    )

    names = [candidate.test_name for candidate in result.candidates]
    assert names[0] == "element access / numeric key"
    assert "element access" not in names
    assert "related hierarchy" in result.candidates[0].label


def test_anchor_retrieval_keeps_unrelated_tests_with_hierarchical_names():
    repository = InMemRepository(files={
        "tests/names.tests.cpp": r'''
TEST_CASE("shared name") { CHECK(run_first()); }
TEST_CASE("shared name / nested spelling") { CHECK(run_second()); }
''',
    })

    result = AnchorRetriever(repository).retrieve(
        "shared name behavior",
        IssueEvidenceFrame(symbols=["shared name"]),
    )

    assert [candidate.test_name for candidate in result.candidates] == [
        "shared name",
        "shared name / nested spelling",
    ]


def test_focal_rank_is_a_tiebreaker_for_strong_lexical_matches():
    lexical_second = _rrf_score({"lexical": 2})
    weak_lexical_with_focal = _rrf_score({
        "lexical": 13,
        "focal_entity": 1,
    })

    assert lexical_second > weak_lexical_with_focal


def test_test_intent_cache_requires_complete_index_artifacts(tmp_path):
    build_id = "empty-build"
    (tmp_path / "test_cases.jsonl").write_text("", encoding="utf-8")
    intents_path = tmp_path / "test_intents.jsonl"
    intents_path.write_text("", encoding="utf-8")

    assert not intent_test_artifacts_ready(tmp_path)

    (tmp_path / "test_meta.jsonl").write_text("", encoding="utf-8")
    faiss.write_index(faiss.IndexFlatIP(2), str(tmp_path / "test_index.faiss"))

    assert not intent_test_artifacts_ready(tmp_path)

    write_embedding_manifest_entry(
        tmp_path,
        "tests",
        model=None,
        dimension=2,
        metadata={
            "builder": TEST_INTENT_BUILDER,
            "build_id": build_id,
            "case_count": 0,
            "fallback_count": 0,
        },
    )

    assert intent_test_artifacts_ready(tmp_path)
