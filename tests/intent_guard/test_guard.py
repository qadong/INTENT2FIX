from types import SimpleNamespace

from INTENT2FIX.intent_guard.guard import IntentGuard
from INTENT2FIX.intent_guard.schema import (
    BlindIntentAdjudication,
    EvidenceAssessment,
    EvidenceRequest,
    IntentCandidate,
    IntentChallenge,
)
from INTENT2FIX.reproduce.intent_schema import (
    IssueClaim,
    IssueEvidenceFrame,
    RepairIntentBundle,
    RepairObligation,
)


class ScriptedExtractor:
    def __init__(self, *responses):
        self.responses = list(responses)

    def __call__(self, prompt, schema):
        assert prompt
        response = self.responses.pop(0)
        assert isinstance(response, schema)
        return response


def _issue_evidence(*claims: IssueClaim) -> IssueEvidenceFrame:
    return IssueEvidenceFrame(claims=list(claims))


def _claim(
    claim_id: str = "EXP-1",
    kind: str = "expected_behavior",
    text: str = "String-compatible comparisons compile.",
) -> IssueClaim:
    return IssueClaim(
        id=claim_id,
        kind=kind,
        text=text,
        issue_span=text,
    )


def _obligation(
    obligation_id: str = "OBL-1",
    text: str = "String-compatible comparisons compile.",
) -> RepairObligation:
    return RepairObligation(
        id=obligation_id,
        kind="behavioral",
        text=text,
        support="issue_explicit",
        provenance=["EXP-1"],
    )


def _bundle(*obligations: RepairObligation) -> RepairIntentBundle:
    return RepairIntentBundle(
        status="STRONG",
        reproduction_status="MATCHED",
        summary="Restore string-compatible comparison support.",
        behavioral_delta="The reported comparison compiles.",
        hard_obligations=list(obligations),
    )


def _reproduction():
    return SimpleNamespace(
        repository_evidence=[],
        execution_records=[],
        graph_bindings=[],
        matched_claim_ids=[],
    )


def _assessment(candidate_id: str, obligation_id: str) -> EvidenceAssessment:
    return EvidenceAssessment(
        evidence_id="ISSUE-EXP-1",
        role="NORMATIVE",
        relation="SUPPORTS",
        candidate_ids=[candidate_id],
        obligation_ids=[obligation_id],
    )


def test_accept_uses_exact_current_candidate_and_requires_evidence_closure():
    issue = "String-compatible comparisons compile."
    proposed = _bundle(_obligation())
    challenge = IntentChallenge()
    guard = IntentGuard(lambda *_: None)
    _, mapping = guard._anonymous_candidates(issue, proposed, challenge)
    current_id = next(key for key, value in mapping.items() if value == "CURRENT")
    adjudication = BlindIntentAdjudication(
        selection="SELECTED",
        selected_candidate_id=current_id,
        effective_intent=IntentCandidate(),
        evidence_assessments=[_assessment(current_id, "OBL-1")],
        decision_reason="The issue directly supports the current contract.",
    )
    guard = IntentGuard(ScriptedExtractor(challenge, adjudication))

    result = guard.validate(
        proposed,
        issue,
        _issue_evidence(_claim()),
        _reproduction(),
        code_index=None,
        repository=None,
    )

    assert result.execution_status == "OK"
    assert result.decision == "ACCEPT_CURRENT"
    assert result.effective_bundle == proposed


def test_revision_selects_the_anonymous_challenger():
    issue = "String-compatible comparisons compile."
    proposed = _bundle(_obligation("OBL-OLD", "All arbitrary objects compare."))
    challenger = IntentCandidate(
        summary="Limit the contract to string-compatible operands.",
        behavioral_delta="String-compatible comparisons compile.",
        hard_obligations=[_obligation()],
        rejected_interpretations=["Arbitrary objects must compare."],
    )
    challenge = IntentChallenge(
        risk_kind="SCOPE_OVERGENERALIZATION",
        challenged_obligation_ids=["OBL-OLD"],
        challenger=challenger,
        observable_difference="Arbitrary non-string objects are outside the requested scope.",
    )
    guard = IntentGuard(lambda *_: None)
    _, mapping = guard._anonymous_candidates(issue, proposed, challenge)
    challenger_id = next(
        key for key, value in mapping.items() if value == "CHALLENGER"
    )
    adjudication = BlindIntentAdjudication(
        selection="SELECTED",
        selected_candidate_id=challenger_id,
        effective_intent=IntentCandidate(),
        evidence_assessments=[_assessment(challenger_id, "OBL-1")],
        decision_reason="Only the narrower candidate follows the issue.",
    )
    guard = IntentGuard(ScriptedExtractor(challenge, adjudication))

    result = guard.validate(
        proposed,
        issue,
        _issue_evidence(_claim()),
        _reproduction(),
        code_index=None,
        repository=None,
    )

    assert result.decision == "REVISE_CURRENT"
    assert result.effective_bundle.status == "STRONG"
    assert [item.id for item in result.effective_bundle.hard_obligations] == ["OBL-1"]


def test_semantic_evidence_gap_revises_to_supported_common_core():
    issue = "String-compatible comparisons compile."
    proposed = _bundle(
        _obligation(),
        _obligation("OBL-2", "Invalid strings are silently treated as empty."),
    )
    challenge = IntentChallenge(risk_kind="UNSUPPORTED_SEMANTICS")
    guard = IntentGuard(lambda *_: None)
    _, mapping = guard._anonymous_candidates(issue, proposed, challenge)
    current_id = next(key for key, value in mapping.items() if value == "CURRENT")
    adjudication = BlindIntentAdjudication(
        selection="SELECTED",
        selected_candidate_id=current_id,
        effective_intent=IntentCandidate.from_bundle(proposed),
        evidence_assessments=[_assessment(current_id, "OBL-1")],
        decision_reason="The current candidate is preferred.",
    )
    guard = IntentGuard(ScriptedExtractor(challenge, adjudication))

    result = guard.validate(
        proposed,
        issue,
        _issue_evidence(_claim()),
        _reproduction(),
        code_index=None,
        repository=None,
    )

    assert result.execution_status == "OK"
    assert result.decision == "REVISE_CURRENT"
    assert result.effective_bundle.status == "WEAK"
    assert [item.id for item in result.effective_bundle.hard_obligations] == ["OBL-1"]
    assert any("OBL-2" in item for item in result.effective_bundle.unresolved)
    assert any("OBL-2" in item for item in result.effective_bundle.rejected_interpretations)


def test_keyword_shape_does_not_override_llm_adjudication():
    issue = "String-compatible comparisons compile."
    proposed = _bundle(
        _obligation(),
        _obligation("OBL-2", "Add a friend operator overload."),
    )
    challenge = IntentChallenge(risk_kind="IMPLEMENTATION_AS_REQUIREMENT")
    guard = IntentGuard(lambda *_: None)
    _, mapping = guard._anonymous_candidates(issue, proposed, challenge)
    current_id = next(key for key, value in mapping.items() if value == "CURRENT")
    adjudication = BlindIntentAdjudication(
        selection="SELECTED",
        selected_candidate_id=current_id,
        effective_intent=IntentCandidate.from_bundle(proposed),
        evidence_assessments=[
            _assessment(current_id, "OBL-1"),
            _assessment(current_id, "OBL-2"),
        ],
        decision_reason="Both obligations were claimed to be supported.",
    )
    guard = IntentGuard(ScriptedExtractor(challenge, adjudication))

    result = guard.validate(
        proposed,
        issue,
        _issue_evidence(_claim()),
        _reproduction(),
        code_index=None,
        repository=None,
    )

    assert result.decision == "ACCEPT_CURRENT"
    assert result.effective_bundle.status == "STRONG"
    assert [item.id for item in result.effective_bundle.hard_obligations] == [
        "OBL-1",
        "OBL-2",
    ]


def test_evidence_retrieval_is_bounded_and_preserves_source_roles():
    class CodeIndex:
        def __init__(self):
            self.queries = []

        def semantic_search(self, query, file_pattern, max_results):
            self.queries.append((query, file_pattern, max_results))
            hit = SimpleNamespace(
                file_path=f"tests/guard_{len(self.queries)}.cpp",
                spans=[],
            )
            return SimpleNamespace(hits=[hit])

    class Repository:
        def get_file_content(self, path):
            return "int main() { return 0; }\n"

    issue = "String-compatible comparisons compile."
    proposed = _bundle(_obligation())
    challenge = IntentChallenge(
        risk_kind="PRESERVATION_OMISSION",
        evidence_requests=[
            EvidenceRequest(query=f"query {index}", target_symbol="Widget")
            for index in range(5)
        ],
    )
    guard = IntentGuard(lambda *_: None)
    _, mapping = guard._anonymous_candidates(issue, proposed, challenge)
    current_id = next(key for key, value in mapping.items() if value == "CURRENT")
    adjudication = BlindIntentAdjudication(
        selection="SELECTED",
        selected_candidate_id=current_id,
        effective_intent=IntentCandidate.from_bundle(proposed),
        evidence_assessments=[_assessment(current_id, "OBL-1")],
        decision_reason="The issue supports the current contract.",
    )
    code_index = CodeIndex()
    guard = IntentGuard(ScriptedExtractor(challenge, adjudication))

    result = guard.validate(
        proposed,
        issue,
        _issue_evidence(_claim()),
        _reproduction(),
        code_index=code_index,
        repository=Repository(),
    )

    assert len(code_index.queries) == 3
    assert code_index.queries[0] == ("query 0 Widget", None, 4)
    retrieved = [item for item in result.evidence if item.id.startswith("GUARD-")]
    assert len(retrieved) == 3
    assert all(item.source_type == "TEST" for item in retrieved)
    assert all(item.role == "PRESERVATION" for item in retrieved)


def test_evidence_retrieval_enforces_target_directory_and_real_test_paths():
    class CodeIndex:
        def semantic_search(self, query, file_pattern, max_results):
            return SimpleNamespace(hits=[
                SimpleNamespace(
                    file_path="src/catch2/catch_test_case_info.hpp",
                    spans=[],
                ),
                SimpleNamespace(file_path="tests/real_test.cpp", spans=[]),
            ])

    class Repository:
        def get_file_content(self, path):
            return "int main() { return 0; }\n"

    issue = "String-compatible comparisons compile."
    proposed = _bundle(_obligation())
    challenge = IntentChallenge(
        evidence_requests=[EvidenceRequest(query="comparison", target_file="tests/")]
    )
    guard = IntentGuard(lambda *_: None)
    _, mapping = guard._anonymous_candidates(issue, proposed, challenge)
    current_id = next(key for key, value in mapping.items() if value == "CURRENT")
    adjudication = BlindIntentAdjudication(
        selection="SELECTED",
        selected_candidate_id=current_id,
        effective_intent=IntentCandidate.from_bundle(proposed),
        evidence_assessments=[_assessment(current_id, "OBL-1")],
        decision_reason="The issue supports the current contract.",
    )
    guard = IntentGuard(ScriptedExtractor(challenge, adjudication))

    result = guard.validate(
        proposed,
        issue,
        _issue_evidence(_claim()),
        _reproduction(),
        code_index=CodeIndex(),
        repository=Repository(),
    )

    retrieved = [item for item in result.evidence if item.id.startswith("GUARD-")]
    assert [item.provenance.split(":", 1)[0] for item in retrieved] == [
        "tests/real_test.cpp"
    ]
    assert IntentGuard._source_type_for_path(
        "src/catch2/catch_test_case_info.hpp"
    ) == "REPOSITORY"


def test_exact_file_evidence_uses_bounded_direct_scan():
    class CodeIndex:
        def semantic_search(self, **kwargs):
            raise AssertionError("exact file requests must not use semantic search")

    class Repository:
        def get_file_content(self, path):
            return "\n".join(
                ["// filler"] * 100
                + ["friend bool operator!=(json_pointer const&, string_t const&);"]
                + ["// tail"] * 100
            )

    issue = "String-compatible comparisons compile."
    proposed = _bundle(_obligation())
    challenge = IntentChallenge(evidence_requests=[EvidenceRequest(
        query="json_pointer operator!= string_t",
        target_symbol="operator!=",
        target_file="include/json_pointer.hpp",
    )])
    guard = IntentGuard(lambda *_: None)
    _, mapping = guard._anonymous_candidates(issue, proposed, challenge)
    current_id = next(key for key, value in mapping.items() if value == "CURRENT")
    adjudication = BlindIntentAdjudication(
        selection="SELECTED",
        selected_candidate_id=current_id,
        effective_intent=IntentCandidate.from_bundle(proposed),
        evidence_assessments=[_assessment(current_id, "OBL-1")],
        decision_reason="The issue supports the current contract.",
    )
    guard = IntentGuard(ScriptedExtractor(challenge, adjudication))

    result = guard.validate(
        proposed,
        issue,
        _issue_evidence(_claim()),
        _reproduction(),
        code_index=CodeIndex(),
        repository=Repository(),
    )

    retrieved = [item for item in result.evidence if item.id.startswith("GUARD-")]
    assert len(retrieved) == 1
    assert "operator!=" in retrieved[0].content
    assert len(retrieved[0].content.splitlines()) <= 61


def test_empty_supported_core_is_unresolved():
    issue = "String-compatible comparisons compile."
    proposed = _bundle(_obligation())
    challenge = IntentChallenge(risk_kind="UNSUPPORTED_SEMANTICS")
    guard = IntentGuard(lambda *_: None)
    _, mapping = guard._anonymous_candidates(issue, proposed, challenge)
    current_id = next(key for key, value in mapping.items() if value == "CURRENT")
    adjudication = BlindIntentAdjudication(
        selection="SELECTED",
        selected_candidate_id=current_id,
        effective_intent=IntentCandidate.from_bundle(proposed),
        evidence_assessments=[],
        decision_reason="No normative evidence was found.",
    )
    guard = IntentGuard(ScriptedExtractor(challenge, adjudication))

    result = guard.validate(
        proposed,
        issue,
        _issue_evidence(_claim()),
        _reproduction(),
        code_index=None,
        repository=None,
    )

    assert result.decision == "UNRESOLVED"
    assert result.effective_bundle.status == "UNRESOLVED"
    assert result.effective_bundle.hard_obligations == []


def test_evidence_closure_keeps_supported_behavior_and_opens_unsupported_clause():
    issue = "The test should compile. This broke a test for bitfields."
    expected_claim = _claim(
        claim_id="EXP-FALLBACK-1",
        text="The test should compile.",
    )
    preservation_claim = _claim(
        claim_id="PRES-FALLBACK-1",
        kind="preservation",
        text="This broke a test for bitfields.",
    )
    proposed = _bundle(
        _obligation("H1", "The reported equality expression must compile."),
        _obligation("H2", "Existing bitfield comparisons must keep compiling."),
        _obligation(
            "H3",
            "The expression passed to handleExpr must be an accepted internal type.",
        ),
    )
    challenge = IntentChallenge(risk_kind="IMPLEMENTATION_AS_REQUIREMENT")
    guard = IntentGuard(lambda *_: None)
    _, mapping = guard._anonymous_candidates(issue, proposed, challenge)
    current_id = next(key for key, value in mapping.items() if value == "CURRENT")
    adjudication = BlindIntentAdjudication(
        selection="SELECTED",
        selected_candidate_id=current_id,
        effective_intent=IntentCandidate.from_bundle(proposed),
        evidence_assessments=[
            EvidenceAssessment(
                evidence_id="ISSUE-EXP-FALLBACK-1",
                role="NORMATIVE",
                relation="SUPPORTS",
                candidate_ids=[current_id],
                obligation_ids=["H1"],
            ),
            EvidenceAssessment(
                evidence_id="ISSUE-PRES-FALLBACK-1",
                role="PRESERVATION",
                relation="SUPPORTS",
                candidate_ids=[current_id],
                obligation_ids=["H2"],
            ),
        ],
        decision_reason="The observable behavior is issue-supported.",
    )
    guard = IntentGuard(ScriptedExtractor(challenge, adjudication))

    result = guard.validate(
        proposed,
        issue,
        _issue_evidence(expected_claim, preservation_claim),
        _reproduction(),
        code_index=None,
        repository=None,
    )

    assert result.decision == "REVISE_CURRENT"
    assert result.effective_bundle.status == "WEAK"
    assert [item.id for item in result.effective_bundle.hard_obligations] == [
        "H1",
        "H2",
    ]
    assert any("H3" in item for item in result.effective_bundle.unresolved)


def test_technical_failure_is_the_only_fail_open_path():
    proposed = _bundle(_obligation())
    guard = IntentGuard(lambda *_: None)

    result = guard.validate(
        proposed,
        "String-compatible comparisons compile.",
        _issue_evidence(_claim()),
        _reproduction(),
        code_index=None,
        repository=None,
    )

    assert result.execution_status == "TECHNICAL_FAILURE"
    assert result.decision == "ACCEPT_CURRENT"
    assert result.effective_bundle == proposed


def test_technical_failure_does_not_apply_keyword_based_rewriting():
    proposed = _bundle(
        _obligation(),
        _obligation("OBL-2", "The operator must not be removed or changed."),
    )
    guard = IntentGuard(lambda *_: None)

    result = guard.validate(
        proposed,
        "String-compatible comparisons compile.",
        _issue_evidence(_claim()),
        _reproduction(),
        code_index=None,
        repository=None,
    )

    assert result.execution_status == "TECHNICAL_FAILURE"
    assert result.decision == "ACCEPT_CURRENT"
    assert result.proposed_intent == proposed
    assert [item.id for item in result.effective_bundle.hard_obligations] == [
        "OBL-1",
        "OBL-2",
    ]
