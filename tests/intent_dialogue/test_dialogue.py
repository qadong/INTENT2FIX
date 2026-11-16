import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from INTENT2FIX.intent_dialogue import (
    IntentDialogue,
    compact_reproduction_context,
    read_only_command_rejection,
)
from INTENT2FIX.intent_dialogue.developer_agent import (
    ContainerDeveloperAgent,
    DeveloperGraphQuery,
    DeveloperGraphQueryArgs,
    DeveloperSession,
    ReadOnlySearch,
    ReadOnlySearchArgs,
    ReplaceInvestigationText,
    ReplaceInvestigationTextArgs,
    RunInvestigationCommand,
    RunInvestigationCommandArgs,
    SubmitDeveloperAnswer,
    SubmitDeveloperAnswerArgs,
    WriteInvestigationFile,
    WriteInvestigationFileArgs,
    frozen_probe_command_rejection,
    investigation_command_rejection,
)
from INTENT2FIX.intent_dialogue.prompts import (
    build_contrastive_investigation_task,
    build_poc_investigation_task,
    build_contrastive_intent_decision_prompt,
    build_contrastive_refinement_task,
    build_evidence_sufficiency_refinement_task,
    build_evidence_sufficiency_review_prompt,
    build_exploration_plan_prompt,
    build_intent_boundary_test_refinement_task,
    build_intent_boundary_test_assessment_prompt,
    build_intent_boundary_test_task,
    build_intent_evidence_adjudication_prompt,
    build_intent_property_extraction_prompt,
    build_intent_resolution_prompt,
    build_intent_synthesis_prompt,
    build_intent_validation_routing_prompt,
)
from INTENT2FIX.intent_dialogue.schema import (
    AdjudicationEvidenceCitation,
    CandidateRepairBoundary,
    ContrastiveAttempt,
    ContrastiveFollowupRequest,
    ContrastiveIntentDecision,
    ContrastiveProbeCandidate,
    ContrastiveWitnessProposal,
    DeveloperAnswer,
    DeveloperAcceptanceCheck,
    EvidenceSufficiencyFollowup,
    EvidenceSufficiencyReview,
    FrozenAcceptanceCheck,
    HypothesisEvidenceEffect,
    IntentHypothesis,
    IntentDimensionAdjudication,
    IntentEvidenceAdjudication,
    IntentBoundaryTestAssessment,
    IntentBoundaryTestAssessmentItem,
    IntentContractClauseDecision,
    IntentContractSemanticAudit,
    IntentEvidenceResolution,
    IntentExplorationObjective,
    IntentExplorationPlan,
    IntentDialogueResult,
    IntentBehaviorProperty,
    IntentPropertyDimension,
    InvestigationTaskFinding,
    IntentSynthesisSelection,
    IntentTaskResolution,
    IntentPropertyExtraction,
    IntentValidationRoute,
    IntentValidationRouting,
    RepairIntentContract,
    RepairIntentLedgerEntry,
    RepositoryContractCitation,
    RepositoryContractEffect,
    RepositoryEvidence,
    TestAnchorCandidate as AnchorCandidate,
    TestAnchorRetrieval as AnchorRetrieval,
    TestAnchorSelection as AnchorSelection,
)
from INTENT2FIX.exceptions import RejectError
from INTENT2FIX.node import Node
from INTENT2FIX.reproduce.intent_schema import (
    ExecutionRecord,
    GraphBinding,
    IssueEvidenceFrame,
)


def proposed_check(
    description: str,
    run_evidence_id: str = "RUN-1",
) -> DeveloperAcceptanceCheck:
    return DeveloperAcceptanceCheck(
        description=description,
        run_evidence_id=run_evidence_id,
    )


def objective(
    question: str,
    question_type: str = "acceptance_example",
    related_issue_ids: list[str] | None = None,
) -> IntentExplorationObjective:
    return IntentExplorationObjective(
        question=question,
        question_type=question_type,
        related_issue_ids=related_issue_ids or [],
        rationale="This evidence rules out a materially different observable repair.",
        done_when="The relevant public behavior is established from repository evidence.",
    )


def frozen_check(description: str) -> FrozenAcceptanceCheck:
    return FrozenAcceptanceCheck(
        description=description,
        command="c++ intent_probe.cpp -o /tmp/intent_probe && /tmp/intent_probe",
        files={"intent_probe.cpp": "int main() { return 0; }"},
        source_evidence_id="RUN-1",
        baseline_exit_code=0,
    )


def issue_evidence() -> IssueEvidenceFrame:
    return IssueEvidenceFrame.model_validate({
        "claims": [
            {
                "id": "EXP-1",
                "kind": "expected_behavior",
                "text": "Widget::parse should accept string input.",
                "issue_span": "Widget::parse should accept string input.",
            }
        ]
    })


def issue_evidence_with_intent() -> IssueEvidenceFrame:
    return IssueEvidenceFrame.model_validate({
        "claims": [
            {
                "id": "EXP-1",
                "kind": "expected_behavior",
                "text": "Widget::parse should accept string input.",
                "issue_span": "Widget::parse should accept string input.",
            }
        ],
        "intent_units": [
            {
                "id": "INTENT-1",
                "role": "target",
                "text": "Widget::parse should accept string input.",
                "issue_span": "Widget::parse should accept string input.",
                "source_claim_ids": ["EXP-1"],
            }
        ],
    })


def issue_evidence_with_two_target_intents() -> IssueEvidenceFrame:
    return IssueEvidenceFrame.model_validate({
        "claims": [
            {
                "id": "EXP-1",
                "kind": "expected_behavior",
                "text": "Widget::parse accepts the requested input.",
                "issue_span": "accepts the requested input",
            },
            {
                "id": "EXP-2",
                "kind": "expected_behavior",
                "text": "A failed parse preserves the cursor state.",
                "issue_span": "preserves the cursor state",
            },
        ],
        "intent_units": [
            {
                "id": "INTENT-1",
                "role": "target",
                "text": "Widget::parse accepts the requested input.",
                "issue_span": "accepts the requested input",
                "source_claim_ids": ["EXP-1"],
            },
            {
                "id": "INTENT-2",
                "role": "target",
                "text": "A failed parse preserves the cursor state.",
                "issue_span": "preserves the cursor state",
                "source_claim_ids": ["EXP-2"],
            },
        ],
    })


def test_final_semantic_audit_downgrades_local_witness_clauses():
    audit = IntentContractSemanticAudit(
        consistent=True,
        decisions=[
            IntentContractClauseDecision(
                clause_id="PROP-1",
                disposition="diagnostic",
                evidence_ids=["RUN-1"],
                reason="The mutation invented the exact public API spelling.",
            ),
            IntentContractClauseDecision(
                clause_id="ACCEPT-1",
                disposition="diagnostic",
                evidence_ids=["RUN-1"],
                reason="The check establishes only one local witness.",
            ),
            IntentContractClauseDecision(
                clause_id="DELTA-1",
                disposition="diagnostic",
                evidence_ids=["RUN-1"],
                reason="The observed delta does not settle the public boundary.",
            ),
        ],
        summary="The Issue target remains binding, but the derived API choice does not.",
    )
    dialogue = IntentDialogue(
        extractor=ScriptedExtractor(audit),
        developer=SimpleNamespace(),
    )
    contract = RepairIntentContract(
        intent_ledger=[
            RepairIntentLedgerEntry(
                id="INTENT-1",
                role="target",
                text="Widget::parse should accept string input.",
                issue_span="Widget::parse should accept string input.",
                source_claim_ids=["EXP-1"],
                status="resolved",
                evidence_ids=["RUN-1"],
                witness_ids=["RUN-1"],
                witnessed_delta="One mutated probe fails.",
                supported_delta="The new Widget::is_supported API must return true.",
            )
        ],
        target_behavior=["Widget::parse should accept string input."],
        properties=[
            IntentBehaviorProperty(
                id="PROP-1",
                role="target",
                authority="issue_explicit",
                issue_intent_ids=["INTENT-1"],
                predicate="Widget::is_supported returns true for strings.",
                evidence_ids=["RUN-1"],
            )
        ],
        acceptance_checks=[
            FrozenAcceptanceCheck(
                description="The invented Widget::is_supported probe passes.",
                command="c++ probe.cpp && ./a.out",
                files={"probe.cpp": "int main() { return 1; }"},
                source_evidence_id="RUN-1",
                baseline_exit_code=1,
                kind="f2p",
                authority="normative",
                covered_issue_intent_ids=["INTENT-1"],
                covered_property_ids=["PROP-1"],
            )
        ],
        behavioral_delta=[
            "The new Widget::is_supported API must return true."
        ],
    )

    result = dialogue._finalize_semantic_contract(
        "Widget::parse should accept string input.",
        issue_evidence_with_intent(),
        contract,
        [
            RepositoryEvidence(
                id="RUN-1",
                path="probe.cpp",
                observation="The locally mutated probe exits 1.",
            )
        ],
    )

    assert result == audit
    assert contract.properties == []
    assert contract.acceptance_checks[0].authority == "illustrative"
    assert contract.acceptance_checks[0].kind == "diagnostic"
    assert contract.behavioral_delta == []
    assert contract.intent_ledger[0].supported_delta == ""
    assert contract.intent_ledger[0].status == "local_witness"


def test_mechanism_validation_requires_a_same_fixture_discriminator():
    contract = RepairIntentContract()
    routing_prompt = build_intent_validation_routing_prompt(
        "The public operation must use the customization mechanism.",
        issue_evidence(),
        contract,
        [],
        [],
    )
    materialization_prompt = build_intent_boundary_test_task(
        "The public operation must use the customization mechanism.",
        issue_evidence(),
        contract,
        ContrastiveIntentDecision(
            witness_valid=False,
            witnessed_delta="",
            reason="No discriminating witness exists yet.",
        ),
        IntentValidationRouting(routes=[]),
    )

    assert "same public fixture" in routing_prompt
    assert "individually callable" in routing_prompt
    assert "coexist on the same public input" in materialization_prompt
    assert "independent old-path control" in materialization_prompt

    decision_prompt = build_contrastive_intent_decision_prompt(
        "The public operation must use the customization mechanism.",
        issue_evidence(),
        {},
        {},
        [],
        [],
    )
    property_prompt = build_intent_property_extraction_prompt(
        "The public operation must use the customization mechanism.",
        issue_evidence(),
        contract,
        [],
        [],
    )
    assessor_prompt = build_intent_boundary_test_assessment_prompt(
        "The public operation must use the customization mechanism.",
        issue_evidence(),
        contract,
        ContrastiveIntentDecision(
            witness_valid=False,
            witnessed_delta="",
            reason="No discriminating witness exists yet.",
        ),
        IntentValidationRouting(routes=[]),
        [],
        {},
        [],
    )

    assert "private helper, trait, or adjacent operation" in decision_prompt
    assert "Keep such implementation choices available to\nRepair" in decision_prompt
    assert "conditional\nfallback to M" in decision_prompt
    assert "Issue's rhetorical structure" in decision_prompt
    assert "kind is `example` is never the Issue-named mechanism" in decision_prompt
    assert "newly\ninvented fixture" in property_prompt
    assert "fallback availability" in property_prompt
    assert "property is conformance to M" in property_prompt
    assert "typed `example`\nclaim" in property_prompt
    assert "pre-existing public documentation" in assessor_prompt
    assert "rhetorical\nstructure" in assessor_prompt
    assert "assessment reason calls an\nexample" in assessor_prompt

    assessment = IntentBoundaryTestAssessment(
        contract_consistent=True,
        boundary_complete=False,
        evidence_ids=["EXP-1", "RUN-3"],
        missing_discriminators=[
            "Make the requested and competing mechanisms observably disagree."
        ],
        reason="The independent probes do not reveal which mechanism wins.",
    )
    refinement_prompt = build_intent_boundary_test_refinement_task(
        "The public operation must use the customization mechanism.",
        issue_evidence(),
        contract,
        ContrastiveIntentDecision(
            witness_valid=False,
            witnessed_delta="",
            reason="No discriminating witness exists yet.",
        ),
        IntentValidationRouting(routes=[]),
        DeveloperAnswer(status="partial", answer="Independent probes only."),
        assessment,
    )

    assert "same public fixture" in refinement_prompt
    assert "surviving_wrong_repair" in refinement_prompt
    assert "existing investigation budget" in refinement_prompt
    assert "original Issue, not from a nearby analogy" in refinement_prompt
    assert "Treat typed `example` claims as fixture evidence only" in refinement_prompt


def test_candidate_repair_boundary_is_deferred_to_patch_review():
    result = IntentDialogueResult(
        intent_contract=RepairIntentContract(
            target_behavior=["Widget::parse should accept string input."],
            candidate_boundaries=[
                CandidateRepairBoundary(
                    id="BOUNDARY-1",
                    anchor_operation="Widget::validate across public receivers",
                    candidate_public_forms=[
                        "Widget::parse",
                        "WidgetView::parse",
                    ],
                    evidence_ids=["SEARCH-1"],
                    rationale=(
                        "Both forms expose the closest established validation "
                        "operation."
                    ),
                )
            ],
        )
    )

    repair_rendered = result.render_for_repair()
    review_rendered = result.intent_contract.render_for_review()

    assert "BOUNDARY-1" not in repair_rendered
    assert "WidgetView::parse" not in repair_rendered
    assert "Repository-supported surface leads" in review_rendered
    assert "Widget::parse" in review_rendered
    assert "WidgetView::parse" in review_rendered
    assert "not requirements by themselves" in review_rendered


def repository_effect(
    kind: str,
    statement: str,
    *,
    evidence_id: str = "SEARCH-1",
    quote: str = "Public contract: existing input remains accepted.",
) -> RepositoryContractEffect:
    return RepositoryContractEffect(
        kind=kind,
        statement=statement,
        evidence=[
            RepositoryContractCitation(
                evidence_id=evidence_id,
                quote=quote,
            )
        ],
    )


class ScriptedExtractor:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts = []

    def __call__(self, prompt, schema):
        assert prompt
        self.prompts.append(prompt)
        if not self.responses and schema is IntentPropertyExtraction:
            return IntentPropertyExtraction()
        if not self.responses and schema is IntentContractSemanticAudit:
            payload = prompt.split(
                "Merged candidate clauses:\n", 1
            )[1].split("\n\nBounded evidence packet:", 1)[0]
            clauses = json.loads(payload)
            issue_payload = prompt.split(
                "Protected atomic Issue evidence:\n", 1
            )[1].split("\n\nMerged candidate clauses:", 1)[0]
            protected = json.loads(issue_payload)
            fallback_evidence_id = protected["claims"][0]["id"]
            return IntentContractSemanticAudit(
                consistent=True,
                decisions=[
                    IntentContractClauseDecision(
                        clause_id=item["id"],
                        disposition=(
                            "diagnostic"
                            if item.get("proposed_authority") == "illustrative"
                            else "binding"
                        ),
                        evidence_ids=(
                            item.get("source_evidence_ids")
                            or [fallback_evidence_id]
                        )[:1],
                        reason="Preserve the pre-audited fixture semantics.",
                    )
                    for item in clauses
                ],
                summary="Test fixture accepts the pre-audited clauses.",
            )
        response = self.responses.pop(0)
        assert response is None or isinstance(response, schema)
        return response


def version_space_plan() -> IntentExplorationPlan:
    return IntentExplorationPlan(
        mode="version_space",
        objectives=[
            IntentExplorationObjective(
                goal="Determine whether ordered behavior is default or opt-in.",
                task_type="compatibility_boundary",
                related_issue_ids=["EXP-1"],
                rationale="The two meanings expose different public defaults.",
                done_when="An explicit public contract distinguishes the meanings.",
                discriminator="A public guarantee for the default object's ordering.",
                hypotheses=[
                    IntentHypothesis(
                        id="H1",
                        description="The existing default object becomes ordered.",
                        qualifiers=["Apply ordering to the existing default object."],
                    ),
                    IntentHypothesis(
                        id="H2",
                        description="A distinct opt-in object is ordered.",
                        qualifiers=["Provide ordering through a distinct opt-in object."],
                    ),
                ],
            )
        ]
    )


class FakeVersionSpaceInvestigator:
    def __init__(
        self,
        *,
        evidence_id: str,
        observation: str,
    ):
        self.evidence_id = evidence_id
        self.observation = observation

    def investigate(self, issue, issue_evidence, reproduction, plan):
        dimension = plan.tasks[0]
        return (
            DeveloperAnswer(
                status="supported",
                answer="The repository evidence was classified.",
                repository_evidence_ids=[self.evidence_id],
                task_findings=[
                    InvestigationTaskFinding(
                        task_id=dimension.id,
                        status="supported",
                        finding="The default has an explicit ordering contract.",
                        evidence_ids=[self.evidence_id],
                    )
                ],
            ),
            [
                RepositoryEvidence(
                    id=self.evidence_id,
                    path="docs/object.md",
                    locator="grep -RInF -- 'sorted order' docs",
                    observation=self.observation,
                )
            ],
            [],
        )


def resolved_adjudication(
    quote: str = "The default object stores keys in sorted order.",
) -> IntentEvidenceAdjudication:
    return IntentEvidenceAdjudication(
        decisions=[
            IntentDimensionAdjudication(
                dimension_id="DIM-1",
                disambiguated=True,
                selected_hypothesis_id="DIM-1-H2",
                evidence=[
                    AdjudicationEvidenceCitation(
                        evidence_id="SEARCH-1",
                        quote=quote,
                    )
                ],
                reason=(
                    "The explicit default guarantee leaves the opt-in "
                    "interpretation as the only compatible hypothesis."
                ),
            )
        ]
    )


def test_version_space_uses_binary_adjudicator_without_synthesizer():
    extractor = ScriptedExtractor(version_space_plan(), resolved_adjudication())
    result = IntentDialogue(
        extractor=extractor,
        investigator=FakeVersionSpaceInvestigator(
            evidence_id="SEARCH-1",
            observation="The default object stores keys in sorted order.",
        ),
    ).run("Preserve object key definition order.", issue_evidence(), {})

    assert extractor.responses == []
    assert result.stop_reason == "version_space_resolved"
    assert result.version_space[0].remaining_hypothesis_ids == ["DIM-1-H2"]
    assert result.version_space[0].eliminated[0].hypothesis_id == "DIM-1-H1"
    assert result.adjudication.decisions[0].disambiguated
    assert "distinct opt-in object" in result.render_for_repair()


def test_version_space_does_not_fall_back_to_open_legacy_tasks():
    malformed_plan = IntentExplorationPlan(
        mode="version_space",
        tasks=[
            objective(
                "Search for more context about the requested behavior.",
                question_type="behavior_boundary",
            )
        ]
    )
    extractor = ScriptedExtractor(malformed_plan)
    result = IntentDialogue(
        extractor=extractor,
        investigator=SimpleNamespace(),
    ).run("Widget::parse should accept string input.", issue_evidence(), {})

    assert malformed_plan.mode == "version_space"
    assert result.stop_reason == "no_objectives"
    assert result.exploration_plan.tasks == []


@pytest.mark.parametrize("adjudication", [
    IntentEvidenceAdjudication(
        decisions=[
            IntentDimensionAdjudication(
                dimension_id="DIM-1",
                disambiguated=False,
                reason="The repository does not encode the product choice.",
            )
        ]
    ),
    resolved_adjudication("A quote not in the observation."),
])
def test_version_space_retains_ambiguity_when_adjudicator_abstains_or_miscites(
    adjudication,
):
    result = IntentDialogue(
        extractor=ScriptedExtractor(version_space_plan(), adjudication),
        investigator=FakeVersionSpaceInvestigator(
            evidence_id="SEARCH-1",
            observation="The default object stores keys in sorted order.",
        ),
    ).run("Preserve object key definition order.", issue_evidence(), {})

    resolution = result.version_space[0]
    assert result.stop_reason == "version_space_unresolved"
    assert resolution.remaining_hypothesis_ids == ["DIM-1-H1", "DIM-1-H2"]
    assert resolution.eliminated == []
    if adjudication.decisions[0].disambiguated:
        assert resolution.rejected_effects
    rendered = result.render_for_repair()
    assert "Open Intent decisions" in rendered
    assert "minimum semantic impact" not in rendered
    assert "DIM-1 remains unresolved" in (
        result.intent_contract.render_for_review()
    )
    assert "only as a tie-breaker" not in rendered


def test_version_space_keeps_all_hypotheses_when_adjudicator_is_unavailable():
    result = IntentDialogue(
        extractor=ScriptedExtractor(version_space_plan(), None),
        investigator=FakeVersionSpaceInvestigator(
            evidence_id="SEARCH-1",
            observation="The default object stores keys in sorted order.",
        ),
    ).run("Preserve object key definition order.", issue_evidence(), {})

    assert result.stop_reason == "version_space_adjudicator_unavailable"
    assert result.version_space[0].remaining_hypothesis_ids == [
        "DIM-1-H1",
        "DIM-1-H2",
    ]


def test_planner_prompt_creates_bounded_repository_objectives():
    prompt = build_exploration_plan_prompt(
        "Assigning one public JSON type to another should not recurse.",
        IssueEvidenceFrame(),
        {"failure_class": "runtime", "execution_records": []},
    )

    assert "Evidence TODO Planner" in prompt
    assert "compatibility_boundary" in prompt
    assert "original Issue is authoritative" in prompt
    assert "Patch Reviewer" in prompt
    assert "generated" in prompt
    assert "zero to ten non-overlapping TODOs" in prompt
    assert "Ten is a hard upper bound, not a" in prompt
    assert "TASK-1" in prompt
    assert "Leave `hypotheses` empty" in prompt
    assert "do not invent A/B hypotheses" in prompt
    assert "complete symbol" in prompt
    assert "counterexamples" in prompt
    assert "representation_or_error_contract" in prompt
    assert "diagnostic convention" in prompt


def evidence_todo_plan(count: int = 1) -> IntentExplorationPlan:
    goals = [
        "Establish accepted public input categories.",
        "Establish the returned public representation.",
        "Identify the documented error category.",
        "Determine the documented default behavior.",
        "Check whether an opt-in mode is promised.",
        "Establish state retained after the public call.",
        "Identify the adjacent exception-safety guarantee.",
        "Inspect the neighboring public argument-order convention.",
        "Find an Issue-aligned public documentation example.",
        "Check the repository's public deprecation policy.",
        "This eleventh task must exceed the schema limit.",
    ]
    return IntentExplorationPlan(
        mode="evidence_todos",
        tasks=[
            IntentExplorationObjective(
                id=f"TASK-{index}",
                goal=goals[index - 1],
                task_type="repository_convention",
                related_issue_ids=["EXP-1"],
                rationale="The convention may clarify an observable boundary.",
                done_when="Public evidence and counterexamples are inspected.",
            )
            for index in range(1, count + 1)
        ],
    )


class FakeEvidenceTodoInvestigator:
    def investigate(self, issue, issue_evidence, reproduction, plan):
        observation = "Adjacent public APIs place FILE* before the format argument."
        return (
            DeveloperAnswer(
                status="supported",
                answer="The adjacent public convention was inspected.",
                repository_facts=[observation],
                repository_evidence_ids=["SEARCH-1"],
                task_findings=[
                    InvestigationTaskFinding(
                        task_id=task.id,
                        status="supported",
                        finding=observation,
                        evidence_ids=["SEARCH-1"],
                    )
                    for task in plan.tasks
                ],
            ),
            [
                RepositoryEvidence(
                    id="SEARCH-1",
                    path="include/public.h",
                    observation=observation,
                )
            ],
            [],
        )


def test_evidence_todo_resolution_adds_only_grounded_behavior_qualifier():
    quote = "Adjacent public APIs place FILE* before the format argument."
    resolution = IntentEvidenceResolution(
        resolutions=[
            IntentTaskResolution(
                task_id="TASK-1",
                resolved=True,
                resolution_kind="behavior_qualifier",
                contract_text="Place FILE* before the format argument.",
                evidence_type="consistent_public_api_convention",
                evidence=[
                    AdjudicationEvidenceCitation(
                        evidence_id="SEARCH-1",
                        quote=quote,
                    )
                ],
                alternatives_considered=[
                    "Place FILE* after the format argument."
                ],
                reason="The adjacent public family consistently uses FILE-first.",
            )
        ]
    )
    result = IntentDialogue(
        extractor=ScriptedExtractor(evidence_todo_plan(), resolution),
        investigator=FakeEvidenceTodoInvestigator(),
    ).run("Widget::parse should accept string input.", issue_evidence(), {})

    assert result.stop_reason == "evidence_resolution_resolved"
    assert result.resolution == resolution
    assert result.adjudication is None
    assert result.version_space == []
    assert "Repository-resolved qualifier: Place FILE* before" in (
        result.render_for_repair()
    )


def test_evidence_todo_resolution_rejects_a_non_exact_evidence_quote():
    resolution = IntentEvidenceResolution(
        resolutions=[
            IntentTaskResolution(
                task_id="TASK-1",
                resolved=True,
                resolution_kind="behavior_qualifier",
                contract_text="Place FILE* before the format argument.",
                evidence_type="consistent_public_api_convention",
                evidence=[
                    AdjudicationEvidenceCitation(
                        evidence_id="SEARCH-1",
                        quote="This quote is not in the observation.",
                    )
                ],
                reason="The convention appears consistent.",
            )
        ]
    )
    result = IntentDialogue(
        extractor=ScriptedExtractor(evidence_todo_plan(), resolution),
        investigator=FakeEvidenceTodoInvestigator(),
    ).run("Widget::parse should accept string input.", issue_evidence(), {})

    assert result.stop_reason == "evidence_resolution_no_effect"
    assert result.resolution.resolutions[0].resolved is False
    assert "Repository-resolved qualifier" not in result.render_for_repair()


def test_evidence_todo_plan_runs_up_to_ten_tasks_without_truncation():
    plan = evidence_todo_plan(10)
    resolution = IntentEvidenceResolution(
        resolutions=[
            IntentTaskResolution(
                task_id=f"TASK-{index}",
                resolved=False,
                reason="Repository evidence did not settle this TODO.",
            )
            for index in range(1, 11)
        ]
    )
    result = IntentDialogue(
        extractor=ScriptedExtractor(plan, resolution),
        investigator=FakeEvidenceTodoInvestigator(),
        max_tasks=10,
    ).run("Widget::parse should accept string input.", issue_evidence(), {})

    assert len(result.exploration_plan.tasks) == 10
    assert len(result.resolution.resolutions) == 10
    with pytest.raises(ValidationError):
        evidence_todo_plan(11)


def test_resolution_prompt_forms_alternatives_after_repository_search():
    prompt = build_intent_resolution_prompt(
        "Widget::parse should accept string input.",
        issue_evidence(),
        evidence_todo_plan(),
        {"task_findings": []},
        [{"id": "SEARCH-1", "observation": "public convention"}],
        {"preserve": [], "acceptance": [], "warning": []},
    )

    assert "after considering the supplied repository evidence" in prompt
    assert "consistent_public_api_convention" in prompt
    assert "form them from evidence" in prompt
    assert "must not become Repair instructions" in prompt


def test_adjudicator_prompt_is_binary_and_cannot_write_contract():
    prompt = build_intent_evidence_adjudication_prompt(
        "Preserve object key definition order.",
        issue_evidence(),
        version_space_plan(),
        {"task_findings": []},
        [{
            "id": "SEARCH-1",
            "observation": "The default object stores keys in sorted order.",
        }],
    )

    assert "exactly one binary decision" in prompt
    assert "disambiguated: true" in prompt
    assert "disambiguated: false" in prompt
    assert "may not rewrite the Issue" in prompt
    assert "Several evidence items may be\ncombined" in prompt
    assert "cannot by" in prompt
    assert "themselves determine a new intended behavior" in prompt


def test_synthesis_prompt_rejects_surface_inventory_and_defaults_sparse():
    prompt = build_intent_synthesis_prompt(
        "Widget::parse should accept string input.",
        issue_evidence(),
        [
            objective(
                "Does parsing preserve the existing error category?",
                question_type="compatibility_boundary",
            ).model_dump()
        ],
        [],
        {"preserve": [], "acceptance": [], "warning": []},
    )

    assert "Default to an empty selection" in prompt
    assert "Normally select at most two preserve candidates" in prompt
    assert "lists of files, declarations, classes, overloads" in prompt
    assert "every sibling/public type must receive an API" in prompt
    assert "Structural completeness belongs" in prompt
    assert "open_decisions" in prompt
    assert "Current behavior does" in prompt
    assert "intended post-repair behavior" in prompt
    assert "Several mechanisms producing similar output" in prompt
    assert "change to a documented default" in prompt
    assert "candidate-level evidence gate" in prompt
    assert "one new public contract must cover all of them" in prompt


def test_investigation_task_can_express_a_non_binary_compound_goal():
    task = objective(
        question=(
            "Establish accepted inputs, returned representation, and errors "
            "for the focused public scenario."
        ),
    )

    assert task.goal.startswith("Establish accepted inputs")


class FakeDeveloper:
    def __init__(self):
        self.questions = []

    def answer(self, issue, issue_evidence, reproduction, question):
        self.questions.append(question.question)
        return (
            DeveloperAnswer(
                status="supported",
                answer="String-compatible inputs preserve existing errors.",
                repository_facts=["Adjacent parsers preserve domain errors."],
                preserve_behavior=["Existing domain errors remain unchanged."],
                acceptance_checks=[
                    proposed_check(
                        "Parse a string-compatible input and a malformed contrast."
                    )
                ],
                repository_evidence_ids=["SEARCH-1", "RUN-1", "MADE-UP"],
            ),
            [
                RepositoryEvidence(
                    id="SEARCH-1",
                    path="<container>",
                    locator="rg -n 'Widget::parse' include tests",
                    observation="include/api.hpp:12: void parse(string input);",
                ),
                RepositoryEvidence(
                    id="RUN-1",
                    path="<container>",
                    locator=(
                        "c++ intent_probe.cpp -o /tmp/intent_probe && "
                        "/tmp/intent_probe"
                    ),
                    observation="",
                ),
            ],
            [
                {
                    "action": "ReadOnlySearch",
                    "command": "rg -n 'Widget::parse' include tests",
                },
                {
                    "action": "RunInvestigationCommand",
                    "command": (
                        "c++ intent_probe.cpp -o /tmp/intent_probe && "
                        "/tmp/intent_probe"
                    ),
                    "exit_code": 0,
                    "evidence_id": "RUN-1",
                    "frozen_probe_files": {
                        "intent_probe.cpp": "int main() { return 0; }"
                    },
                },
            ],
        )


def test_dialogue_uses_container_developer_and_filters_provenance():
    issue = "Widget::parse should accept string input."
    reproduction = {
        "execution_records": [{"run_id": "RUN-4", "exit_code": 1}]
    }
    extractor = ScriptedExtractor(
        IntentExplorationPlan(
            objectives=[
                objective(
                    question="Must string-compatible inputs preserve domain errors?",
                    question_type="compatibility_boundary",
                    related_issue_ids=["EXP-1"],
                )
            ]
        ),
        IntentSynthesisSelection(
            preserve_candidate_ids=["PRESERVE-CAND-1"],
            acceptance_candidate_ids=["ACCEPT-CAND-1"],
        ),
    )
    developer = FakeDeveloper()

    result = IntentDialogue(
        extractor=extractor,
        developer=developer,
        max_rounds=2,
    ).run(issue, issue_evidence(), reproduction)

    assert result.stop_reason == "synthesized"
    assert developer.questions == [
        "Must string-compatible inputs preserve domain errors?"
    ]
    answer = result.exchanges[0].answer
    assert answer.repository_evidence_ids == ["SEARCH-1", "RUN-1"]
    assert result.exchanges[0].developer_actions[0]["action"] == "ReadOnlySearch"
    assert result.exchanges[0].objective.id == "TODO-1"
    rendered = result.render_for_repair()
    assert "<repair_intent_contract>" in rendered
    assert "Widget::parse should accept string input." in rendered
    assert "Existing domain errors remain unchanged." in rendered
    assert "Parse a string-compatible input" in rendered
    assert "Adjacent parsers preserve domain errors." not in rendered
    assert "String-compatible inputs preserve existing errors." not in rendered
    assert "rg -n 'Widget::parse'" not in rendered
    assert "SEARCH-1" not in rendered
    assert "MADE-UP" not in rendered


def test_repair_handoff_keeps_both_dialogue_rounds():
    result = IntentDialogueResult(
        intent_contract=RepairIntentContract(
            target_behavior=["Add the new syntax."],
            preserve_behavior=["Public docs still teach the old syntax."],
            acceptance_checks=[frozen_check("Parse both the new and old syntax.")],
        ),
    )

    rendered = result.render_for_repair()

    assert "Add the new syntax." in rendered
    assert "Public docs still teach the old syntax." in rendered
    assert "Parse both the new and old syntax." in rendered
    assert "[TARGET-1]" in rendered
    assert "[PRESERVE-1]" in rendered
    assert "[ACCEPT-1]" in rendered


class ConflictingDeveloper:
    def answer(self, issue, issue_evidence, reproduction, question):
        return (
            DeveloperAnswer(
                status="conflict",
                answer="Existing tests expect an exception.",
                conflicts_with_issue=[
                    "[EXP-1] Existing tests expect an exception instead."
                ],
                repository_evidence_ids=["SEARCH-1"],
            ),
            [
                RepositoryEvidence(
                    id="SEARCH-1",
                    path="tests/api.cpp",
                    observation="CHECK_THROWS(call());",
                )
            ],
            [],
        )


def test_repository_conflict_is_audit_only_without_decisive_delta():
    issue = "The call must return false instead of throwing."
    evidence = IssueEvidenceFrame.model_validate({
        "claims": [
            {
                "id": "EXP-1",
                "kind": "expected_behavior",
                "text": "The call must return false instead of throwing.",
                "issue_span": "The call must return false instead of throwing.",
            }
        ],
        "intent_units": [
            {
                "id": "INTENT-1",
                "role": "target",
                "text": "The call must return false instead of throwing.",
                "issue_span": "The call must return false instead of throwing.",
                "source_claim_ids": ["EXP-1"],
            }
        ],
    })
    result = IntentDialogue(
        extractor=ScriptedExtractor(
            IntentExplorationPlan(
                objectives=[
                    objective(
                        question="Does the API promise non-throwing membership checks?",
                        question_type="compatibility_boundary",
                        related_issue_ids=["EXP-1", "UNKNOWN-1"],
                    )
                ]
            ),
            IntentSynthesisSelection(
                warning_candidate_ids=["WARNING-CAND-1"],
            ),
        ),
        developer=ConflictingDeveloper(),
    ).run(issue, evidence, {})

    assert result.exchanges[0].objective.related_issue_ids == ["EXP-1"]
    assert result.exchanges[0].objective.related_intent_ids == ["INTENT-1"]
    assert result.intent_contract.target_behavior == [issue]
    assert result.intent_contract.preserve_behavior == []
    assert result.intent_contract.compatibility_warnings == []
    assert result.stop_reason == "synthesized_no_delta"
    assert result.exchanges[0].answer.conflicts_with_issue == [
        "[EXP-1] Existing tests expect an exception instead."
    ]
    rendered = result.render_for_repair()
    assert "must not weaken, replace, narrow, or expand" in rendered
    assert "Existing tests expect" not in rendered
    assert "Compatibility warnings" not in rendered


def test_partial_answer_remains_unresolved_instead_of_rewriting_intent():
    class PartialDeveloper:
        def answer(self, issue, issue_evidence, reproduction, question):
            return (
                DeveloperAnswer(
                    status="partial",
                    answer="The repository exposes an option, but not its policy.",
                    repository_facts=["ParserOptions contains a mode flag."],
                    acceptance_checks=[
                        proposed_check(
                            "Exercise explicit alternate mode and the existing default."
                        )
                    ],
                    open_decisions=[
                        "The repository does not establish the new default."
                    ],
                    repository_evidence_ids=["SEARCH-1", "RUN-1"],
                ),
                [
                    RepositoryEvidence(
                        id="SEARCH-1",
                        path="include/options.hpp",
                        observation="bool alternate_mode = false;",
                    ),
                    RepositoryEvidence(
                        id="RUN-1",
                        path="<container>",
                        locator=(
                            "c++ intent_probe.cpp -o /tmp/intent_probe && "
                            "/tmp/intent_probe"
                        ),
                        observation="",
                    ),
                ],
                [
                    {
                        "action": "RunInvestigationCommand",
                        "command": (
                            "c++ intent_probe.cpp -o /tmp/intent_probe && "
                            "/tmp/intent_probe"
                        ),
                        "exit_code": 0,
                        "evidence_id": "RUN-1",
                        "frozen_probe_files": {
                            "intent_probe.cpp": "int main() { return 0; }"
                        },
                    }
                ],
            )

    issue = "Support the alternate parser behavior."
    evidence = IssueEvidenceFrame.model_validate({
        "claims": [
            {
                "id": "EXP-1",
                "kind": "expected_behavior",
                "text": issue,
                "issue_span": issue,
            }
        ]
    })
    result = IntentDialogue(
        extractor=ScriptedExtractor(
            IntentExplorationPlan(
                objectives=[
                    objective(
                        question="Does an existing option determine the default?",
                        question_type="behavior_boundary",
                    )
                ]
            ),
            IntentSynthesisSelection(
                acceptance_candidate_ids=["ACCEPT-CAND-1"],
            ),
        ),
        developer=PartialDeveloper(),
    ).run(issue, evidence, {})

    assert result.intent_contract.target_behavior == [issue]
    assert result.intent_contract.acceptance_checks == []
    assert result.intent_contract.open_decisions == []
    assert result.exchanges[0].answer.open_decisions == [
        "The repository does not establish the new default."
    ]
    assert result.stop_reason == "investigation_open_decisions"
    assert "Open Intent decisions" not in result.render_for_repair()
    assert "does not establish the new default" not in result.render_for_repair()
    assert "ParserOptions contains a mode flag." not in result.render_for_repair()
    assert "The repository exposes an option" not in result.render_for_repair()


def test_explicit_non_goal_is_rendered_as_a_rejected_alternative():
    issue = "Add an umbrella header without replacing internal headers."
    evidence = IssueEvidenceFrame.model_validate({
        "claims": [
            {
                "id": "EXP-1",
                "kind": "expected_behavior",
                "text": "Add an umbrella header.",
                "issue_span": "Add an umbrella header",
            },
            {
                "id": "NONGOAL-1",
                "kind": "non_goal",
                "text": "Remove the existing internal headers.",
                "issue_span": "without replacing internal headers",
            },
        ]
    })

    result = IntentDialogue(
        extractor=ScriptedExtractor(IntentExplorationPlan()),
        developer=SimpleNamespace(),
    ).run(issue, evidence, {})

    assert result.intent_contract.preserve_behavior == [
        "Do not implement this explicitly rejected alternative: "
        "Remove the existing internal headers."
    ]
    rendered = result.render_for_repair()
    assert "Do not implement this explicitly rejected alternative" in rendered


def test_planner_deduplicates_semantically_overlapping_objectives():
    developer = FakeDeveloper()
    result = IntentDialogue(
        extractor=ScriptedExtractor(
            IntentExplorationPlan(
                objectives=[
                    objective(
                        question="Does the public API provide an opt-in parser mode?",
                        question_type="behavior_boundary",
                        related_issue_ids=["EXP-1"],
                    ),
                    objective(
                        question="Does this public API expose an opt-in parsing mode?",
                        question_type="behavior_boundary",
                        related_issue_ids=["EXP-1"],
                    ),
                ]
            ),
            IntentSynthesisSelection(),
        ),
        developer=developer,
    ).run(
        "Widget::parse should accept string input.",
        issue_evidence(),
        {},
    )

    assert result.stop_reason == "synthesized_no_delta"
    assert len(result.exploration_plan.objectives) == 1
    assert len(result.exchanges) == 1
    assert len(developer.questions) == 1


def test_planner_filters_patch_surface_inventory_tasks():
    developer = FakeDeveloper()
    result = IntentDialogue(
        extractor=ScriptedExtractor(
            IntentExplorationPlan(
                objectives=[
                    objective(
                        question=(
                            "Determine which public object classes the term "
                            "objects refers to."
                        ),
                        question_type="compatibility_boundary",
                    ),
                    objective(
                        question=(
                            "Establish which public fmt::format_to_n entry "
                            "points expose the internal iterator."
                        ),
                        question_type="compatibility_boundary",
                    ),
                    objective(
                        question=(
                            "Establish whether counting preserves later public "
                            "iteration behavior."
                        ),
                        question_type="behavior_boundary",
                    ),
                    objective(
                        question=(
                            "Determine the exact set of observable iterator "
                            "concept requirements currently violated."
                        ),
                        question_type="behavior_boundary",
                    ),
                ]
            ),
            IntentSynthesisSelection(),
        ),
        developer=developer,
    ).run(
        "Add a count_fields operation for objects.",
        issue_evidence(),
        {},
    )

    assert [task.goal for task in result.exploration_plan.tasks] == [
        "Establish whether counting preserves later public iteration behavior.",
        "Determine the exact set of observable iterator concept requirements "
        "currently violated.",
    ]
    assert developer.questions == [
        "Establish whether counting preserves later public iteration behavior.",
        "Determine the exact set of observable iterator concept requirements "
        "currently violated.",
    ]


def test_planner_runs_all_distinct_objectives_before_one_synthesis():
    developer = FakeDeveloper()
    extractor = ScriptedExtractor(
        IntentExplorationPlan(
            objectives=[
                objective(
                    question="Which existing input forms remain accepted?",
                    question_type="compatibility_boundary",
                    related_issue_ids=["EXP-1"],
                ),
                objective(
                    question="Which public error category is used for malformed input?",
                    question_type="representation_or_error_contract",
                    related_issue_ids=["EXP-1"],
                ),
            ]
        ),
        IntentSynthesisSelection(
            preserve_candidate_ids=["PRESERVE-CAND-1"],
            acceptance_candidate_ids=["ACCEPT-CAND-1"],
        ),
    )

    result = IntentDialogue(
        extractor=extractor,
        developer=developer,
    ).run(
        "Widget::parse should accept string input.",
        issue_evidence(),
        {},
    )

    assert developer.questions == [
        "Which existing input forms remain accepted?",
        "Which public error category is used for malformed input?",
    ]
    assert [item.id for item in result.exploration_plan.objectives] == [
        "TODO-1",
        "TODO-2",
    ]
    assert result.stop_reason == "synthesized"


def test_investigator_executes_the_whole_todo_plan_before_one_synthesis():
    class FakeInvestigator:
        def __init__(self):
            self.calls = 0

        def investigate(self, issue, evidence, reproduction, plan):
            self.calls += 1
            assert [task.id for task in plan.tasks] == ["TODO-1", "TODO-2"]
            repository_evidence = [
                RepositoryEvidence(
                    id="SEARCH-1",
                    path="docs/api.md",
                    observation="Existing inputs retain domain errors.",
                )
            ]
            return (
                DeveloperAnswer(
                    status="supported",
                    answer="The plan established one compatibility boundary.",
                    preserve_behavior=["Existing domain errors remain unchanged."],
                    repository_evidence_ids=["SEARCH-1"],
                    task_findings=[
                        InvestigationTaskFinding(
                            task_id="TODO-1",
                            status="supported",
                            finding="Existing accepted inputs are documented.",
                            evidence_ids=["SEARCH-1"],
                        ),
                        InvestigationTaskFinding(
                            task_id="TODO-2",
                            status="supported",
                            finding="Existing failures retain domain errors.",
                            evidence_ids=["SEARCH-1"],
                        ),
                    ],
                ),
                repository_evidence,
                [],
            )

    investigator = FakeInvestigator()
    extractor = ScriptedExtractor(
        IntentExplorationPlan(
            objectives=[
                objective("Establish the adjacent accepted input forms."),
                objective(
                    "Establish the adjacent public error contract.",
                    question_type="representation_or_error_contract",
                ),
            ]
        ),
        IntentSynthesisSelection(
            preserve_candidate_ids=["PRESERVE-CAND-1"],
        ),
    )

    result = IntentDialogue(
        extractor=extractor,
        investigator=investigator,
    ).run(
        "Widget::parse should accept string input.",
        issue_evidence(),
        {},
    )

    assert investigator.calls == 1
    assert result.exchanges == []
    assert result.investigation is not None
    assert len(result.investigation.report.task_findings) == 2
    assert result.intent_contract.preserve_behavior == [
        "Existing domain errors remain unchanged."
    ]


def test_shared_investigation_keeps_open_decision_audit_only():
    class PartialInvestigator:
        def investigate(self, issue, evidence, reproduction, plan):
            repository_evidence = [
                RepositoryEvidence(
                    id="SEARCH-1",
                    path="docs/api.md",
                    observation="The current default is sorted.",
                ),
                RepositoryEvidence(
                    id="RUN-1",
                    path="<container>",
                    observation="default=sorted",
                ),
            ]
            return (
                DeveloperAnswer(
                    status="partial",
                    answer="The repository cannot choose default versus opt-in.",
                    acceptance_checks=[
                        proposed_check(
                            "Change the default container to preserve insertion order."
                        )
                    ],
                    open_decisions=[
                        "The repository cannot establish whether the new behavior "
                        "is default or opt-in."
                    ],
                    repository_evidence_ids=["SEARCH-1", "RUN-1"],
                    task_findings=[
                        InvestigationTaskFinding(
                            task_id="TODO-1",
                            status="partial",
                            finding="The current default is documented as sorted.",
                            evidence_ids=["SEARCH-1"],
                        )
                    ],
                ),
                repository_evidence,
                [
                    {
                        "action": "RunInvestigationCommand",
                        "command": (
                            "c++ intent_probe.cpp -o /tmp/intent_probe && "
                            "/tmp/intent_probe"
                        ),
                        "exit_code": 0,
                        "evidence_id": "RUN-1",
                        "frozen_probe_files": {
                            "intent_probe.cpp": "int main() { return 0; }"
                        },
                    }
                ],
            )

    result = IntentDialogue(
        extractor=ScriptedExtractor(
            IntentExplorationPlan(
                objectives=[objective("Determine default versus opt-in behavior.")]
            ),
            IntentSynthesisSelection(
                acceptance_candidate_ids=["ACCEPT-CAND-1"],
            ),
        ),
        investigator=PartialInvestigator(),
    ).run(
        "Widget::parse should accept string input.",
        issue_evidence(),
        {},
    )

    assert result.intent_contract.acceptance_checks == []
    assert result.intent_contract.open_decisions == []
    assert result.investigation.report.open_decisions == [
        "The repository cannot establish whether the new behavior is default "
        "or opt-in."
    ]
    assert result.stop_reason == "synthesized_no_delta"
    assert "default or opt-in" not in result.render_for_repair()


def test_partial_shared_report_can_preserve_supported_todo_behavior():
    class PartialInvestigator:
        def investigate(self, issue, evidence, reproduction, plan):
            repository_evidence = [
                RepositoryEvidence(
                    id="SEARCH-1",
                    path="tests/public_api.cpp",
                    observation="Existing stream output remains accepted.",
                ),
                RepositoryEvidence(
                    id="SEARCH-2",
                    path="docs/new_mode.md",
                    observation="The new default is not specified.",
                ),
            ]
            return (
                DeveloperAnswer(
                    status="partial",
                    answer=(
                        "The old stream behavior is established; the new "
                        "default remains open."
                    ),
                    preserve_behavior=[
                        "Existing unstyled stream output remains accepted."
                    ],
                    open_decisions=[
                        "The repository does not establish the new default."
                    ],
                    repository_evidence_ids=["SEARCH-1", "SEARCH-2"],
                    task_findings=[
                        InvestigationTaskFinding(
                            task_id="TODO-1",
                            status="supported",
                            finding="Existing stream behavior is pinned by tests.",
                            evidence_ids=["SEARCH-1"],
                        ),
                        InvestigationTaskFinding(
                            task_id="TODO-2",
                            status="supported",
                            finding=(
                                "Repository policy conclusively leaves the new "
                                "default unspecified."
                            ),
                            evidence_ids=["SEARCH-2"],
                        ),
                    ],
                ),
                repository_evidence,
                [],
            )

    result = IntentDialogue(
        extractor=ScriptedExtractor(
            IntentExplorationPlan(
                objectives=[
                    objective("Establish adjacent stream compatibility."),
                    objective("Determine the new default."),
                ]
            ),
            IntentSynthesisSelection(
                preserve_candidate_ids=["PRESERVE-CAND-1"],
            ),
        ),
        investigator=PartialInvestigator(),
    ).run(
        "Widget::print should support styled streams.",
        issue_evidence(),
        {},
    )

    assert result.stop_reason == "synthesized"
    assert result.intent_contract.preserve_behavior == [
        "Existing unstyled stream output remains accepted."
    ]
    assert result.intent_contract.open_decisions == []


def test_synthesizer_cannot_select_unproposed_contract_items():
    result = IntentDialogue(
        extractor=ScriptedExtractor(
            IntentExplorationPlan(
                objectives=[
                    objective(
                        question="Which existing input form remains accepted?",
                        question_type="compatibility_boundary",
                    )
                ]
            ),
            IntentSynthesisSelection(
                preserve_candidate_ids=["PRESERVE-CAND-99"],
                acceptance_candidate_ids=["ACCEPT-CAND-99"],
                warning_candidate_ids=["WARNING-CAND-99"],
            ),
        ),
        developer=FakeDeveloper(),
    ).run(
        "Widget::parse should accept string input.",
        issue_evidence(),
        {},
    )

    assert result.stop_reason == "synthesized_no_delta"
    assert result.intent_contract.target_behavior == [
        "Widget::parse should accept string input."
    ]
    assert result.intent_contract.preserve_behavior == []
    assert result.intent_contract.acceptance_checks == []


def test_fallback_contract_does_not_duplicate_or_pollute_original_issue():
    issue = "Add the requested behavior even though extraction failed."
    fallback = IssueEvidenceFrame(
        extraction_status="fallback",
        ambiguities=["Structured extraction failed."],
    )
    result = IntentDialogue(
        extractor=ScriptedExtractor(),
        developer=SimpleNamespace(),
    ).run(issue, fallback, {})

    assert result.stop_reason == "missing_issue_target"
    assert result.intent_contract == RepairIntentContract()
    assert result.render_for_repair() == ""


def test_contract_keeps_behavior_and_drops_environment_metadata():
    evidence = IssueEvidenceFrame.model_validate({
        "claims": [
            {
                "id": "EXP-1",
                "kind": "expected_behavior",
                "text": "Cross-specialization conversion must complete.",
                "issue_span": "Cross-specialization conversion must complete.",
            },
            {
                "id": "SCOPE-1",
                "kind": "affected_scope",
                "text": "The report was reproduced with GCC 5.4.",
                "issue_span": "reproduced with GCC 5.4",
            },
        ],
        "ambiguities": ["The exact branch is not specified."],
    })
    result = IntentDialogue(
        extractor=ScriptedExtractor(IntentExplorationPlan()),
        developer=SimpleNamespace(),
    ).run("Issue", evidence, {})

    assert result.stop_reason == "no_objectives"
    assert result.intent_contract.target_behavior == [
        "Cross-specialization conversion must complete."
    ]
    rendered = result.render_for_repair()
    assert "GCC 5.4" not in rendered
    assert "exact branch" not in rendered


def test_contract_verification_spec_uses_stable_ids():
    contract = RepairIntentContract(
        target_behavior=["Target works."],
        preserve_behavior=["Existing path remains valid."],
        acceptance_checks=[frozen_check("Exercise target and contrast.")],
    )

    assert contract.verification_spec() == {
        "checks": {
            "TARGET-1": "Target works.",
            "PRESERVE-1": "Existing path remains valid.",
            "ACCEPT-1": "Exercise target and contrast.",
        },
        "target_ids": ["TARGET-1"],
        "preserve_ids": ["PRESERVE-1"],
        "acceptance_ids": ["ACCEPT-1"],
        "diagnostic_ids": [],
        "probes": {
            "ACCEPT-1": frozen_check(
                "Exercise target and contrast."
                ).model_dump(
                        exclude={
                            "candidate_id",
                            "covered_issue_claim_ids",
                        "covered_issue_intent_ids",
                        "covered_property_ids",
                    "covered_route_ids",
                }
            )
        },
    }


def test_property_extraction_keeps_only_authority_bound_properties():
    proposed = IntentPropertyExtraction(properties=[
        IntentBehaviorProperty(
            id="PROP-8",
            role="target",
            authority="issue_explicit",
            predicate="Widget::parse accepts the requested string input.",
            oracle="Widget::parse succeeds for each Issue-requested string.",
            dimensions=[
                IntentPropertyDimension(
                    name="string form",
                    domain="Issue-requested string inputs",
                    representatives=["name", "name=value"],
                    evidence_ids=["EXP-1", "MADE-UP"],
                )
            ],
            scope=["Widget::parse", "Widget::parse"],
            evidence_ids=["EXP-1", "RUN-2", "MADE-UP"],
        ),
        IntentBehaviorProperty(
            id="PROP-4",
            role="preserve",
            authority="repository_contract",
            predicate="Existing accepted inputs remain accepted.",
            scope=["existing Widget::parse inputs"],
            evidence_ids=["SEARCH-1"],
        ),
        IntentBehaviorProperty(
            id="PROP-3",
            role="target",
            authority="repository_contract",
            predicate="Current implementation behavior becomes the target.",
            evidence_ids=["SEARCH-1"],
        ),
    ])
    dialogue = IntentDialogue(
        extractor=ScriptedExtractor(proposed),
        investigator=SimpleNamespace(),
        mode="contrastive",
    )

    result = dialogue._extract_intent_properties(
        "Widget::parse should accept string input.",
        issue_evidence(),
        RepairIntentContract(
            target_behavior=["Widget::parse should accept string input."]
        ),
        [
            RepositoryEvidence(
                id="SEARCH-1",
                path="tests/widget.cpp",
                observation="Existing accepted inputs remain accepted.",
            )
        ],
        [{"run_evidence_id": "RUN-2"}],
    )

    assert result is not None
    assert [item.id for item in result.properties] == ["PROP-1", "PROP-2"]
    assert result.properties[0].scope == ["Widget::parse"]
    assert result.properties[0].evidence_ids == ["EXP-1", "RUN-2"]
    assert result.properties[0].dimensions[0].evidence_ids == ["EXP-1"]


def test_target_property_inherits_issue_claim_from_bound_atomic_intent():
    proposed = IntentPropertyExtraction(properties=[
        IntentBehaviorProperty(
            id="PROP-7",
            role="target",
            authority="issue_explicit",
            predicate="Widget::parse accepts the requested string input.",
            oracle="Parsing succeeds.",
            issue_intent_ids=["INTENT-1"],
            evidence_ids=["INTENT-1", "RUN-2"],
        )
    ])
    dialogue = IntentDialogue(
        extractor=ScriptedExtractor(proposed),
        investigator=SimpleNamespace(),
        mode="contrastive",
    )

    result = dialogue._extract_intent_properties(
        "Widget::parse should accept string input.",
        issue_evidence_with_intent(),
        RepairIntentContract(
            target_behavior=["Widget::parse should accept string input."]
        ),
        [],
        [{"run_evidence_id": "RUN-2"}],
    )

    assert result is not None
    assert len(result.properties) == 1
    assert result.properties[0].issue_intent_ids == ["INTENT-1"]
    assert result.properties[0].evidence_ids == ["INTENT-1", "RUN-2", "EXP-1"]


def test_validation_router_normalizes_bindings_and_keeps_omissions_text_only():
    contract = RepairIntentContract(properties=[
        IntentBehaviorProperty(
            id="PROP-1",
            role="target",
            authority="issue_explicit",
            predicate="Widget::parse accepts the requested string input.",
            oracle="Parsing succeeds.",
            evidence_ids=["EXP-1"],
        ),
        IntentBehaviorProperty(
            id="PROP-2",
            role="preserve",
            authority="repository_contract",
            predicate="Existing accepted inputs remain accepted.",
            evidence_ids=["SEARCH-1"],
        ),
    ])
    proposed = IntentValidationRouting(routes=[
        IntentValidationRoute(
            id="ROUTE-8",
            property_id="PROP-1",
            mode="fixed_example",
            executable_oracle="The Issue example parses successfully.",
            generation_plan="Run the exact Issue example.",
            rationale="Only one exact example is established.",
        ),
        IntentValidationRoute(
            id="ROUTE-9",
            property_id="PROP-999",
            mode="compile_matrix",
            executable_oracle="Every cell compiles.",
            generation_plan="Compile all cells.",
            rationale="This route has an unknown property binding.",
        ),
    ])
    dialogue = IntentDialogue(
        extractor=ScriptedExtractor(proposed),
        investigator=SimpleNamespace(),
        mode="contrastive",
    )

    result = dialogue._route_intent_properties(
        "Widget::parse should accept string input.",
        issue_evidence(),
        contract,
        [],
        [],
    )

    assert result is not None
    assert [(item.id, item.property_id, item.mode) for item in result.routes] == [
        ("ROUTE-1", "PROP-1", "fixed_example"),
        ("ROUTE-2", "PROP-2", "text_only"),
    ]


def test_read_only_command_policy_allows_search_and_rejects_mutation():
    work_dir = "/home/repo"

    assert not read_only_command_rejection(
        "grep -RIn 'Widget::parse' include tests | head -50",
        work_dir,
    )
    assert not read_only_command_rejection(
        "sed -n '1,120p' include/api.hpp",
        work_dir,
    )
    assert not read_only_command_rejection(
        "grep -RInE 'operator<|operator>' include tests | head -50",
        work_dir,
    )
    assert not read_only_command_rejection(
        "grep -RInF -- 'value > limit' include src",
        work_dir,
    )
    assert read_only_command_rejection(
        "git log --oneline -5",
        work_dir,
    )
    assert "future fixes" in read_only_command_rejection(
        "curl https://github.com/org/repo/commit/fix.patch",
        work_dir,
    )
    assert read_only_command_rejection(
        "printf x > include/api.hpp",
        work_dir,
    )
    assert read_only_command_rejection(
        "python -c 'open(\"x\", \"w\").write(\"x\")'",
        work_dir,
    )
    assert read_only_command_rejection(
        "cat /etc/passwd",
        work_dir,
    )
    assert read_only_command_rejection(
        "rg -n Widget include tests",
        work_dir,
    )
    assert read_only_command_rejection(
        "grep -RIn Widget .",
        work_dir,
    )
    assert read_only_command_rejection(
        "grep -RIn operator< include",
        work_dir,
    )
    assert read_only_command_rejection(
        "cd /home/repo && grep -RIn Widget include",
        work_dir,
    )
    assert read_only_command_rejection(
        "ls -la",
        work_dir,
    )


def test_refined_report_merge_preserves_distinct_investigation_paths():
    from INTENT2FIX.intent_dialogue.dialogue import _merge_investigation_paths
    from INTENT2FIX.intent_dialogue.schema import InvestigationPath

    first = InvestigationPath(
        path_id="PATH-1", question="Where is state produced?", finding="open",
        status="unresolved", evidence_ids=["SEARCH-1"], missing_evidence="body",
    )
    second = InvestigationPath(
        path_id="PATH-2", question="Where is state consumed?", finding="checked",
        status="supported", evidence_ids=["SEARCH-2"], source_evidence_id="SEARCH-2",
        source_quote="consume(state);",
    )
    updated = first.model_copy(update={
        "finding": "producer body checked", "evidence_ids": ["SEARCH-3"],
    })
    merged = _merge_investigation_paths([first, second], [updated])
    assert [path.path_id for path in merged] == ["PATH-1", "PATH-2"]
    assert merged[0].finding == "producer body checked"
    assert merged[0].evidence_ids == ["SEARCH-1", "SEARCH-3"]


def test_frozen_probe_requires_standalone_compilation_to_tmp():
    files = {"intent_probe.cpp": "int main() { return 0; }"}

    assert not frozen_probe_command_rejection(
        "c++ intent_probe.cpp -Iinclude -o /tmp/intent_probe && "
        "/tmp/intent_probe",
        files,
    )
    assert frozen_probe_command_rejection(
        "make test",
        files,
    )
    assert frozen_probe_command_rejection(
        "c++ intent_probe.cpp -o intent_probe",
        files,
    )
    assert frozen_probe_command_rejection(
        "c++ other.cpp -o /tmp/intent_probe",
        files,
    )
    assert frozen_probe_command_rejection(
        "c++ intent_probe.cpp -o /tmp/intent_probe && echo passed",
        files,
    )
    assert "cannot depend on /tmp build artifacts" in (
        frozen_probe_command_rejection(
            "c++ intent_probe.cpp /tmp/build/libproduct.a "
            "-o /tmp/intent_probe && /tmp/intent_probe",
            files,
        )
    )
    assert frozen_probe_command_rejection(
        "c++ intent_probe.cpp -I/tmp/build/include "
        "-o /tmp/intent_probe && /tmp/intent_probe",
        files,
    )
    assert "Command substitution" in frozen_probe_command_rejection(
        "c++ intent_probe.cpp $(find src -name '*.cpp') "
        "-o /tmp/intent_probe && /tmp/intent_probe",
        files,
    )
    assert "mask the compiler or probe exit status" in (
        frozen_probe_command_rejection(
            "c++ intent_probe.cpp -o /tmp/intent_probe && "
            "/tmp/intent_probe; echo exit=$?",
            files,
        )
    )


class FakeDocker:
    work_dir = "/home/repo"

    def run_bash(self, command, timeout):
        assert command == "grep -RIn Widget include tests"
        assert timeout == 120
        return "include/api.hpp:12:void parse(string input);", 0


def test_developer_actions_require_real_search_evidence():
    session = DeveloperSession()
    finish = SubmitDeveloperAnswer(session=session)
    assert finish.name == "SubmitInvestigationReport"
    premature = finish.execute(
        SubmitDeveloperAnswerArgs(
            thoughts="",
            status="unresolved",
            answer="Unknown.",
        )
    )
    assert not premature.terminal
    assert premature.expect_correction

    search = ReadOnlySearch(
        docker_env=FakeDocker(),
        session=session,
    )
    observation = search.execute(
        ReadOnlySearchArgs(
            thoughts="",
            command="grep -RIn Widget include tests",
        )
    )
    assert observation.properties["evidence_id"] == "SEARCH-1"

    invalid = finish.execute(
        SubmitDeveloperAnswerArgs(
            thoughts="",
            status="partial",
            answer="The API is declared in the public header.",
            evidence_ids=["SEARCH-99"],
        )
    )
    assert invalid.expect_correction

    uncited = finish.execute(
        SubmitDeveloperAnswerArgs(
            thoughts="",
            status="supported",
            answer="The API is declared in the public header.",
        )
    )
    assert uncited.expect_correction
    assert "must cite" in uncited.message

    accepted = finish.execute(
        SubmitDeveloperAnswerArgs(
            thoughts="",
            status="partial",
            answer="The API is declared in the public header.",
            preserve_behavior=["The documented public API remains available."],
            evidence_ids=["SEARCH-1"],
        )
    )
    assert accepted.terminal
    assert (
        accepted.properties["developer_answer"]["repository_evidence_ids"]
        == ["SEARCH-1"]
    )


def test_investigation_report_requires_one_finding_per_planned_todo():
    session = DeveloperSession(planned_task_ids=["TODO-1", "TODO-2"])
    session.record("grep -RInF -- 'parse' include", "include/api.hpp:12", 0)
    submit = SubmitDeveloperAnswer(session=session)

    incomplete = submit.execute(
        SubmitDeveloperAnswerArgs(
            thoughts="",
            status="partial",
            answer="One task is established.",
            task_findings=[
                InvestigationTaskFinding(
                    task_id="TODO-1",
                    status="supported",
                    finding="The public parser is declared in the header.",
                    evidence_ids=["SEARCH-1"],
                )
            ],
        )
    )
    assert incomplete.expect_correction
    assert "missing: TODO-2" in incomplete.message

    accepted = submit.execute(
        SubmitDeveloperAnswerArgs(
            thoughts="",
            status="partial",
            answer="One task is established; one remains unresolved.",
            task_findings=[
                InvestigationTaskFinding(
                    task_id="TODO-1",
                    status="supported",
                    finding="The public parser is declared in the header.",
                    evidence_ids=["SEARCH-1"],
                ),
                InvestigationTaskFinding(
                    task_id="TODO-2",
                    status="unresolved",
                    finding="The repository does not document the new default.",
                ),
            ],
        )
    )
    assert accepted.terminal
    assert accepted.properties["investigation_report"][
        "repository_evidence_ids"
    ] == ["SEARCH-1"]


def test_investigation_report_rejects_non_normative_hypothesis_elimination():
    hypothesis_ids = {"DIM-1-H1", "DIM-1-H2"}
    session = DeveloperSession(
        planned_task_ids=["DIM-1"],
        planned_hypothesis_ids={"DIM-1": hypothesis_ids},
    )
    session.record(
        "grep -RInF -- 'sorted order' docs",
        "The default object stores keys in sorted order.",
        0,
    )
    submit = SubmitDeveloperAnswer(session=session)

    rejected = submit.execute(
        SubmitDeveloperAnswerArgs(
            thoughts="",
            status="supported",
            answer="The current checkout sorts keys.",
            task_findings=[
                InvestigationTaskFinding(
                    task_id="DIM-1",
                    status="supported",
                    finding="Observed the current default behavior.",
                    evidence_ids=["SEARCH-1"],
                    evidence_effects=[
                        HypothesisEvidenceEffect(
                            evidence_id="SEARCH-1",
                            authority="behavioral_baseline",
                            supports_hypothesis_ids=["DIM-1-H2"],
                            eliminates_hypothesis_ids=["DIM-1-H1"],
                            contract_quote=(
                                "The default object stores keys in sorted order."
                            ),
                            explanation="The checkout currently sorts keys.",
                        )
                    ],
                )
            ],
        )
    )

    assert rejected.expect_correction
    assert "descriptive evidence cannot eliminate intent" in rejected.message


def test_investigation_report_accepts_exact_public_contract_effect():
    hypothesis_ids = {"DIM-1-H1", "DIM-1-H2"}
    session = DeveloperSession(
        planned_task_ids=["DIM-1"],
        planned_hypothesis_ids={"DIM-1": hypothesis_ids},
    )
    session.record(
        "grep -RInF -- 'sorted order' docs",
        "The default object stores keys in sorted order.",
        0,
    )
    submit = SubmitDeveloperAnswer(session=session)
    accepted = submit.execute(
        SubmitDeveloperAnswerArgs(
            thoughts="",
            status="supported",
            answer="Public documentation fixes the default ordering contract.",
            task_findings=[
                InvestigationTaskFinding(
                    task_id="DIM-1",
                    status="supported",
                    finding="The default ordering is explicitly documented.",
                    evidence_ids=["SEARCH-1"],
                    evidence_effects=[
                        HypothesisEvidenceEffect(
                            evidence_id="SEARCH-1",
                            authority="public_contract",
                            supports_hypothesis_ids=["DIM-1-H2"],
                            eliminates_hypothesis_ids=["DIM-1-H1"],
                            contract_quote=(
                                "The default object stores keys in sorted order."
                            ),
                            explanation="The default guarantee excludes changing it.",
                        )
                    ],
                )
            ],
        )
    )

    assert accepted.terminal
    effect = accepted.properties["investigation_report"]["task_findings"][0][
        "evidence_effects"
    ][0]
    assert effect["eliminates_hypothesis_ids"] == ["DIM-1-H1"]


def test_supported_intent_answer_requires_actionable_contract_evidence():
    session = DeveloperSession()
    run = session.record(
        "c++ intent_probe.cpp -o /tmp/intent_probe && /tmp/intent_probe",
        "ok",
        0,
        prefix="RUN",
        action="RunInvestigationCommand",
        extra={
            "frozen_probe_files": {
                "intent_probe.cpp": "int main() { return 0; }"
            }
        },
    )
    finish = SubmitDeveloperAnswer(session=session)

    accepted = finish.execute(
        SubmitDeveloperAnswerArgs(
            thoughts="",
            status="supported",
            answer="Sibling public parsers accept string input.",
            repository_facts=[
                "Sibling public parsers accept string input."
            ],
            preserve_behavior=[
                "Sibling public parsers continue accepting string input."
            ],
            acceptance_checks=[
                proposed_check(
                    "Compile and run the sibling parser string-input example.",
                    run.id,
                )
            ],
            evidence_ids=[run.id],
        )
    )
    assert accepted.terminal
    assert accepted.properties["developer_answer"]["status"] == "supported"
    assert accepted.properties["developer_answer"]["repository_facts"] == [
        "Sibling public parsers accept string input."
    ]
    assert accepted.properties["developer_answer"]["acceptance_checks"] == [
        {
            "description": (
                "Compile and run the sibling parser string-input example."
            ),
            "run_evidence_id": "RUN-1",
            "kind": "diagnostic",
                "covered_issue_claim_ids": [],
                "covered_issue_intent_ids": [],
                "covered_property_ids": [],
            "covered_route_ids": [],
        }
    ]


@pytest.mark.parametrize(
    ("kind", "exit_code", "message"),
    [
        ("f2p", 0, "must fail on the buggy baseline"),
        ("p2p", 1, "must pass on the buggy baseline"),
    ],
)
def test_boundary_check_kind_must_match_baseline_result(
    kind,
    exit_code,
    message,
):
    session = DeveloperSession()
    run = session.record(
        "c++ intent_probe.cpp -o /tmp/intent_probe && /tmp/intent_probe",
        "result",
        exit_code,
        prefix="RUN",
        action="RunInvestigationCommand",
        extra={
            "frozen_probe_files": {
                "intent_probe.cpp": "int main() { return 0; }"
            }
        },
    )

    observation = SubmitDeveloperAnswer(session=session).execute(
        SubmitDeveloperAnswerArgs(
            thoughts="",
            status="partial",
            answer="Candidate boundary check was executed.",
            acceptance_checks=[
                DeveloperAcceptanceCheck(
                    description="Exercise the confirmed public boundary.",
                    run_evidence_id=run.id,
                    kind=kind,
                )
            ],
            evidence_ids=[run.id],
        )
    )

    assert observation.expect_correction
    assert message in observation.message


def test_supported_answer_allows_facts_without_contract_effect():
    session = DeveloperSession()
    evidence = session.record("grep -RIn Widget include", "hit", 0)
    finish = SubmitDeveloperAnswer(session=session)

    observation = finish.execute(
        SubmitDeveloperAnswerArgs(
            thoughts="",
            status="supported",
            answer="The repository declares Widget.",
            repository_facts=["Widget is declared."],
            evidence_ids=[evidence.id],
        )
    )

    assert observation.terminal
    assert not observation.expect_correction
    answer = observation.properties["developer_answer"]
    assert answer["preserve_behavior"] == []
    assert answer["acceptance_checks"] == []


def test_developer_submit_schema_rejects_free_form_intent_rewrite():
    with pytest.raises(ValidationError, match="intent_implication"):
        SubmitDeveloperAnswerArgs.model_validate({
            "status": "supported",
            "answer": "Current tests exercise the old syntax.",
            "intent_implication": "Ignore the issue and preserve old behavior.",
        })


def test_developer_rejects_duplicate_search_and_overlapping_read():
    class SearchDocker:
        work_dir = "/home/repo"

        def run_bash(self, command, timeout):
            return f"result for {command}", 0

    session = DeveloperSession()
    search = ReadOnlySearch(docker_env=SearchDocker(), session=session)
    first = search.execute(
        ReadOnlySearchArgs(
            thoughts="",
            target="API",
            purpose="locate declaration",
            command="grep -RInF -- 'Widget' include",
        )
    )
    duplicate = search.execute(
        ReadOnlySearchArgs(
            thoughts="",
            target="API",
            purpose="repeat",
            command="grep -RInF -- 'Widget' include",
        )
    )
    first_range = search.execute(
        ReadOnlySearchArgs(
            thoughts="",
            command="sed -n '1,100p' include/api.hpp",
        )
    )
    overlap = search.execute(
        ReadOnlySearchArgs(
            thoughts="",
            command="sed -n '50,120p' include/api.hpp",
        )
    )

    assert first.properties["evidence_id"] == "SEARCH-1"
    assert duplicate.expect_correction
    assert duplicate.properties["duplicate_evidence_id"] == "SEARCH-1"
    assert first_range.properties["evidence_id"] == "SEARCH-2"
    assert overlap.expect_correction
    assert overlap.properties["duplicate_evidence_id"] == "SEARCH-2"


def test_investigation_allows_pipeline_and_records_observation():
    class PipelineDocker:
        work_dir = "/home/repo"

        def run_bash(self, command, timeout):
            assert command == "c++ -fsyntax-only probe.cpp 2>&1 | head -60"
            return "ok", 0

    assert not investigation_command_rejection(
        "g++ -std=c++11 repro.cpp 2>&1 | head -60"
    )
    assert "future fixes" in investigation_command_rejection(
        "git clone https://github.com/org/repo.git /tmp/upstream"
    )
    assert "Do not delete investigation files" in investigation_command_rejection(
        "rm -f probe/target.cc"
    )
    docker = PipelineDocker()
    session = DeveloperSession()
    run = RunInvestigationCommand(
        docker_env=docker,
        session=session,
    )
    observation = run.execute(
        RunInvestigationCommandArgs(
            thoughts="",
            command="c++ -fsyntax-only probe.cpp 2>&1 | head -60",
            target="probe",
            purpose="compile candidate",
        )
    )

    assert not observation.expect_correction
    assert observation.properties["evidence_id"] == "RUN-1"
    assert session.evidence[0].observation == "ok"


def test_developer_can_refine_and_run_standalone_probe():
    class InvestigationDocker:
        work_dir = "/home/repo"

        def __init__(self):
            self.files = {}

        def write_file(self, path, content):
            self.files[path] = content

        def run_bash(self, command, timeout):
            if command.startswith("test -e "):
                path = command.removeprefix("test -e ")
                return "", int(f"/home/repo/{path}" not in self.files)
            if command.startswith("cat -- "):
                path = command.removeprefix("cat -- ")
                return self.files[f"/home/repo/{path}"], 0
            assert command == "c++ -fsyntax-only intent_probe.cpp"
            return "probe compiled", 0

    docker = InvestigationDocker()
    session = DeveloperSession()
    write = WriteInvestigationFile(docker_env=docker, session=session)
    replace = ReplaceInvestigationText(docker_env=docker, session=session)
    run = RunInvestigationCommand(docker_env=docker, session=session)

    write.execute(
        WriteInvestigationFileArgs(
            thoughts="",
            path="intent_probe.cpp",
            content="static_assert(false);",
            purpose="create trait probe",
        )
    )
    replace.execute(
        ReplaceInvestigationTextArgs(
            thoughts="",
            path="intent_probe.cpp",
            old="false",
            new="true",
            purpose="test the alternative behavior",
        )
    )
    observation = run.execute(
        RunInvestigationCommandArgs(
            thoughts="",
            command="c++ -fsyntax-only intent_probe.cpp",
            target="CAND-1",
            purpose="compile-time trait check",
        )
    )

    assert docker.files["/home/repo/intent_probe.cpp"] == (
        "static_assert(true);"
    )
    assert observation.properties["evidence_id"] == "RUN-1"
    assert session.evidence[0].observation == "probe compiled"


def test_refined_probe_can_rerun_the_same_compile_command():
    class InvestigationDocker:
        work_dir = "/home/repo"

        def __init__(self):
            self.files = {}

        def write_file(self, path, content):
            self.files[path] = content

        def run_bash(self, command, timeout):
            if command.startswith("test -e "):
                path = command.removeprefix("test -e ")
                return "", int(f"/home/repo/{path}" not in self.files)
            if command.startswith("cat -- "):
                path = command.removeprefix("cat -- ")
                return self.files[f"/home/repo/{path}"], 0
            assert command == "c++ -fsyntax-only probe.cpp"
            return self.files["/home/repo/probe.cpp"], 0

    docker = InvestigationDocker()
    session = DeveloperSession()
    write = WriteInvestigationFile(docker_env=docker, session=session)
    replace = ReplaceInvestigationText(docker_env=docker, session=session)
    run = RunInvestigationCommand(docker_env=docker, session=session)
    command = "c++ -fsyntax-only probe.cpp"

    write.execute(
        WriteInvestigationFileArgs(
            thoughts="",
            path="probe.cpp",
            content="first",
            purpose="create probe",
        )
    )
    first = run.execute(
        RunInvestigationCommandArgs(thoughts="", command=command)
    )
    duplicate = run.execute(
        RunInvestigationCommandArgs(thoughts="", command=command)
    )
    replace.execute(
        ReplaceInvestigationTextArgs(
            thoughts="",
            path="probe.cpp",
            old="first",
            new="second",
            purpose="refine probe",
        )
    )
    refined = run.execute(
        RunInvestigationCommandArgs(thoughts="", command=command)
    )

    assert first.properties["evidence_id"] == "RUN-1"
    assert duplicate.expect_correction
    assert refined.properties["evidence_id"] == "RUN-2"
    assert session.evidence[-1].observation == "second"


def test_investigation_write_recovers_stale_probe_registration():
    class MissingFileDocker:
        work_dir = "/home/repo"

        def __init__(self):
            self.files = {}

        def run_bash(self, command, timeout):
            assert command == "test -e probe/recreated.cc"
            return "", 1

        def write_file(self, path, content):
            self.files[path] = content

    docker = MissingFileDocker()
    session = DeveloperSession(
        investigation_files={"probe/recreated.cc": "old"}
    )
    observation = WriteInvestigationFile(
        docker_env=docker,
        session=session,
    ).execute(
        WriteInvestigationFileArgs(
            thoughts="",
            path="probe/recreated.cc",
            content="int main() {}",
            purpose="recreate missing standalone probe",
        )
    )

    assert not observation.expect_correction
    assert session.investigation_files["probe/recreated.cc"] == "int main() {}"
    assert docker.files["/home/repo/probe/recreated.cc"] == "int main() {}"


def test_investigation_probe_cannot_overwrite_repository_file():
    class ExistingFileDocker:
        work_dir = "/home/repo"

        def run_bash(self, command, timeout):
            assert command == "test -e include/api.hpp"
            assert timeout == 30
            return "", 0

        def write_file(self, path, content):
            raise AssertionError("an existing repository file must not be written")

    observation = WriteInvestigationFile(
        docker_env=ExistingFileDocker(),
        session=DeveloperSession(),
    ).execute(
        WriteInvestigationFileArgs(
            thoughts="",
            path="include/api.hpp",
            content="replacement",
            purpose="invalid probe path",
        )
    )

    assert observation.expect_correction
    assert "already exists" in observation.message


def test_investigation_replace_cannot_modify_repository_file():
    class ExistingFileDocker:
        work_dir = "/home/repo"

        def run_bash(self, command, timeout):
            raise AssertionError("an unregistered repository file must not be read")

        def write_file(self, path, content):
            raise AssertionError("an unregistered repository file must not be written")

    observation = ReplaceInvestigationText(
        docker_env=ExistingFileDocker(),
        session=DeveloperSession(),
    ).execute(
        ReplaceInvestigationTextArgs(
            thoughts="",
            path="include/api.hpp",
            old="old implementation",
            new="candidate repair",
            purpose="invalid production edit",
        )
    )

    assert observation.expect_correction
    assert "standalone probe" in observation.message


def test_developer_graph_query_returns_bounded_skeleton(monkeypatch):
    from code_intent.code_intent import entity_tools

    monkeypatch.setattr(
        entity_tools,
        "search_symbol",
        lambda **kwargs: {
            "entities": [
                {
                    "id": "function:Widget::parse",
                    "type": "function",
                    "fq_name": "Widget::parse",
                    "file": "include/widget.hpp",
                    "start_line": 12,
                    "body": "not returned",
                }
            ]
        },
    )

    class FakeCodeIndex:
        def __init__(self):
            self.activated = False

        def _activate_instance(self):
            self.activated = True

    index = FakeCodeIndex()
    session = DeveloperSession()
    graph = DeveloperGraphQuery(code_index=index, session=session)
    observation = graph.execute(
        DeveloperGraphQueryArgs(
            thoughts="",
            mode="symbol",
            symbol_type="function",
            symbol_name="parse",
            limit=5,
            target="public API",
            purpose="locate overload set",
        )
    )

    assert index.activated
    assert observation.properties["evidence_id"] == "GRAPH-1"
    assert "Widget::parse" in observation.message
    assert "not returned" not in observation.message


def test_developer_warns_before_last_step_without_changing_actions():
    node = SimpleNamespace(
        observation=SimpleNamespace(message="[SEARCH-19] evidence")
    )

    ContainerDeveloperAgent._append_submission_budget_notice(node)

    assert node.observation.message.startswith("[SEARCH-19] evidence")
    assert "one model step remaining" in node.observation.message
    assert "call SubmitInvestigationReport" in node.observation.message


def test_developer_followup_reuses_trajectory_and_session_cache():
    session = DeveloperSession()
    developer = ContainerDeveloperAgent(
        agent=SimpleNamespace(),
        session=session,
        repository=SimpleNamespace(),
        docker_env=SimpleNamespace(),
        investigation_actions=[],
        submit_action=SimpleNamespace(),
        max_steps=30,
    )
    first_question = objective(
        question="Must existing unqualified literals remain available?",
        question_type="compatibility_boundary",
    )
    first = developer._start_round(
        "Issue", IssueEvidenceFrame(), {}, first_question
    )
    evidence = session.record(
        "grep -RInF -- '_json' docs tests include",
        "docs/example.cpp: using namespace literals",
        0,
    )
    session.register_command(evidence.locator, evidence.id)
    developer._current_node = first

    second = developer._start_round(
        "Issue",
        IssueEvidenceFrame(),
        {},
        objective(
            question="Is compatibility controlled by configuration?",
            question_type="compatibility_boundary",
        ),
    )

    assert second.parent is first
    assert second.file_context is first.file_context
    assert "Continue the shared repository-evidence trajectory" in second.user_message
    assert "repeat completed searches" in second.user_message
    assert "30 action steps remain" in second.user_message
    assert session.evidence == [evidence]
    assert session.duplicate_command(evidence.locator) == evidence.id
    assert second.get_trajectory() == [first, second]


def test_developer_questions_share_one_step_budget():
    class BudgetAgent:
        def __init__(self):
            self.calls = 0

        def run(self, node):
            self.calls += 1
            if self.calls not in {2, 5}:
                return
            answer = DeveloperAnswer(
                status="supported",
                answer=f"answer after action {self.calls}",
                acceptance_checks=[
                    proposed_check("Exercise the selected public behavior.")
                ],
            )
            node.action_steps = [
                SimpleNamespace(
                    observation=SimpleNamespace(
                        properties={"developer_answer": answer.model_dump()}
                    )
                )
            ]
            node.terminal = True

    agent = BudgetAgent()
    developer = ContainerDeveloperAgent(
        agent=agent,
        session=DeveloperSession(),
        repository=SimpleNamespace(),
        docker_env=SimpleNamespace(),
        investigation_actions=[],
        submit_action=SimpleNamespace(),
        max_steps=6,
    )
    developer._submission_agent = agent

    developer.answer(
        "Issue",
        IssueEvidenceFrame(),
        {},
        objective(question="First question?"),
    )
    assert developer.steps_used == 2
    assert developer.remaining_steps == 4

    developer.answer(
        "Issue",
        IssueEvidenceFrame(),
        {},
        objective(question="Second question?"),
    )
    assert developer.steps_used == 5
    assert developer.remaining_steps == 1
    assert agent.calls == 5


def test_contrastive_refinement_reuses_agent_trajectory_and_step_budget():
    report = DeveloperAnswer(
        status="partial",
        answer="The focused public-contract search remains unresolved.",
        repository_evidence_ids=["SEARCH-1"],
        open_decisions=["The repository does not state the default policy."],
    )

    class RefinementAgent:
        def __init__(self):
            self.nodes = []

        def run(self, node):
            self.nodes.append(node)
            node.action_steps = [
                SimpleNamespace(
                    observation=SimpleNamespace(
                        properties={
                            "investigation_report": report.model_dump()
                        }
                    )
                )
            ]
            node.terminal = True

    session = DeveloperSession()
    session.record(
        "grep -RInF -- 'default policy' docs",
        "No explicit default policy was found.",
        1,
    )
    agent = RefinementAgent()
    investigator = ContainerDeveloperAgent(
        agent=agent,
        session=session,
        repository=SimpleNamespace(),
        docker_env=SimpleNamespace(),
        investigation_actions=[],
        submit_action=SimpleNamespace(),
        max_steps=10,
    )
    previous_node = Node.stub(
        user_message="Initial contrastive investigation",
        file_context=None,
    )
    investigator._current_node = previous_node
    investigator.steps_used = 3
    decision = ContrastiveIntentDecision(
        witness_valid=True,
        boundary_resolved=False,
        next_action="seek_repository_contract",
        followup_request=ContrastiveFollowupRequest(
            claim_ids=["EXP-1"],
            objective="Determine whether the behavior is default or opt-in.",
            why_current_evidence_is_insufficient="The pair permits both.",
            evidence_needed="Direct public contract text for the default.",
        ),
        selected_anchor_id="ANCHOR-1",
        selected_target_id="TARGET-1",
        witnessed_delta="The requested behavior is absent on the baseline.",
        covered_issue_claim_ids=["EXP-1"],
        reason="A focused contract search could resolve the choice.",
    )

    refined, evidence, _ = investigator.refine_contrastively(
        "Widget::parse should accept string input.",
        issue_evidence(),
        contrastive_report(),
        decision,
        "validated",
    )

    assert refined == report
    assert evidence[0].id == "SEARCH-1"
    assert investigator.steps_used == 4
    assert investigator.remaining_steps == 6
    assert investigator._current_node.parent is previous_node
    assert "Continue the existing Intent Evidence investigation" in (
        investigator._current_node.user_message
    )
    assert "use at most 7 action steps" in investigator._current_node.user_message


def test_contrastive_refinement_has_a_short_focused_step_limit():
    report = DeveloperAnswer(
        status="partial",
        answer="The focused evidence request remains unresolved.",
        open_decisions=["The requested distinction remains open."],
    )

    class LateSubmittingAgent:
        def __init__(self):
            self.calls = 0
            self.initial_prompt = ""

        def run(self, node):
            self.calls += 1
            if self.calls == 1:
                self.initial_prompt = node.user_message
            properties = {}
            if self.calls == 12:
                properties["investigation_report"] = report.model_dump()
                node.terminal = True
            node.action_steps = [
                SimpleNamespace(
                    observation=SimpleNamespace(
                        message=f"step {self.calls}",
                        properties=properties,
                    )
                )
            ]

    agent = LateSubmittingAgent()
    investigator = ContainerDeveloperAgent(
        agent=agent,
        session=DeveloperSession(),
        repository=SimpleNamespace(),
        docker_env=SimpleNamespace(),
        investigation_actions=[],
        submit_action=SimpleNamespace(),
        max_steps=60,
    )
    investigator._current_node = Node.stub(
        user_message="Initial contrastive investigation",
        file_context=None,
    )
    decision = ContrastiveIntentDecision(
        witness_valid=True,
        boundary_resolved=False,
        next_action="refine_test_pair",
        followup_request=ContrastiveFollowupRequest(
            claim_ids=["EXP-1"],
            objective="Distinguish the two outcomes on one shared fixture.",
            why_current_evidence_is_insufficient="The first target is local.",
            evidence_needed="One same-claim conflict witness.",
        ),
        selected_anchor_id="ANCHOR-1",
        selected_target_id="TARGET-1",
        witnessed_delta="The local target fails on the baseline.",
        covered_issue_claim_ids=["EXP-1"],
        reason="One focused witness can distinguish the alternatives.",
    )

    refined, _, _ = investigator.refine_contrastively(
        "Issue",
        issue_evidence(),
        contrastive_report(),
        decision,
        "validated",
    )

    assert refined == report
    assert agent.calls == 12
    assert investigator.steps_used == 12
    assert investigator.remaining_steps == 48
    assert "use at most 12 action steps" in agent.initial_prompt


def test_evidence_sufficiency_refinement_gets_a_dedicated_continuation_window():
    report = DeveloperAnswer(
        status="partial",
        answer="The public caller and private helper relationship is now known.",
        repository_evidence_ids=["SEARCH-1"],
    )

    class ReviewingAgent:
        def __init__(self):
            self.prompt = ""

        def run(self, node):
            self.prompt = node.user_message
            node.action_steps = [
                SimpleNamespace(
                    observation=SimpleNamespace(
                        properties={
                            "investigation_report": report.model_dump()
                        }
                    )
                )
            ]
            node.terminal = True

    agent = ReviewingAgent()
    investigator = ContainerDeveloperAgent(
        agent=agent,
        session=DeveloperSession(),
        repository=SimpleNamespace(),
        docker_env=SimpleNamespace(),
        investigation_actions=[],
        submit_action=SimpleNamespace(),
        max_steps=60,
    )
    previous_node = Node.stub(
        user_message="Initial contrastive investigation",
        file_context=None,
    )
    investigator._current_node = previous_node
    investigator.steps_used = 60
    review = EvidenceSufficiencyReview(
        decision="continue",
        reason="The report stops at a copied private helper.",
        preserved_alternatives=["The production public caller owns the fix."],
        next_investigation=EvidenceSufficiencyFollowup(
            question="Which production entry calls this helper?",
            why_material="That relation changes the repair locus and scope.",
            known_anchor="SEARCH-1 private helper definition",
        ),
    )

    refined, _, _ = investigator.refine_evidence_sufficiency(
        "Public sorting should use the requested policy.",
        issue_evidence(),
        contrastive_report(),
        review,
    )

    assert refined == report
    assert investigator.max_steps == 72
    assert investigator.steps_used == 61
    assert investigator.remaining_steps == 11
    assert investigator._current_node.parent is previous_node
    assert "Which production entry calls this helper?" in agent.prompt
    assert "Choose the investigation tools yourself" in agent.prompt


def test_developer_retries_missing_tool_call_twice(caplog):
    class MissingToolAgent:
        def __init__(self):
            self.calls = 0

        def run(self, node):
            self.calls += 1
            if self.calls <= 2:
                node.assistant_message = f"plain response {self.calls}"
                raise RejectError("No action found")

    agent = MissingToolAgent()
    root = Node.stub(user_message="Investigate", file_context=None)

    with caplog.at_level("WARNING"):
        current = ContainerDeveloperAgent._run_with_no_action_retries(
            agent,
            root,
            max_retries=2,
        )

    assert agent.calls == 3
    assert current is not root
    assert "Call exactly one available tool" in current.user_message
    assert "plain response 1" in caplog.text
    assert "plain response 2" in caplog.text


def test_developer_raises_after_two_missing_tool_call_retries():
    class MissingToolAgent:
        def __init__(self):
            self.calls = 0

        def run(self, node):
            self.calls += 1
            node.assistant_message = f"plain response {self.calls}"
            raise RejectError("No action found")

    agent = MissingToolAgent()
    root = Node.stub(user_message="Investigate", file_context=None)

    with pytest.raises(RejectError, match="No action found"):
        ContainerDeveloperAgent._run_with_no_action_retries(
            agent,
            root,
            max_retries=2,
        )

    assert agent.calls == 3


def test_compact_reproduction_context_keeps_recent_bounded_evidence():
    reproduction = SimpleNamespace(
        is_reproduced=True,
        failure_class="compilation",
        test_command="c++ repro.cpp",
        test_files=[
            {"path": "repro.cpp", "content": "int main() { return 1; }"},
        ],
        matched_claim_ids=["EXP-1"],
        repository_evidence=[{"fact": str(index)} for index in range(12)],
        graph_bindings=[
            GraphBinding(
                claim_ids=["EXP-1"],
                entity="Widget::parse",
                file="include/api.hpp",
                relation="LIKELY_REALIZES",
            )
        ],
        execution_records=[
            ExecutionRecord(
                run_id=f"RUN-{index}",
                command="c++ repro.cpp",
                exit_code=1,
                stderr="x" * 2000,
            )
            for index in range(6)
        ],
        unresolved=[],
    )

    context = compact_reproduction_context(reproduction)

    assert len(context["repository_evidence"]) == 8
    assert [item["run_id"] for item in context["execution_records"]] == [
        "RUN-2",
        "RUN-3",
        "RUN-4",
        "RUN-5",
    ]
    assert len(context["execution_records"][0]["stderr"]) == 1200
    assert context["graph_bindings"][0]["entity"] == "Widget::parse"
    assert context["reproduction_status"] == "reproduced"
    assert context["evidence_validated"] is True
    assert context["test_files"][0]["path"] == "repro.cpp"


class FakeContrastiveInvestigator:
    def __init__(self, report: DeveloperAnswer):
        self.report = report

    def investigate_contrastively(
        self,
        issue,
        issue_evidence,
        reproduction,
    ):
        evidence = [
            RepositoryEvidence(
                id="SEARCH-1",
                path="tests/widget_parse_test.cpp",
                locator="sed -n 10,35p tests/widget_parse_test.cpp",
                observation=(
                    "Public contract: existing input remains accepted. "
                    "TEST(Parse, ExistingInput) { "
                    "EXPECT_TRUE(Widget::parse(existing_input)); }"
                ),
            ),
            RepositoryEvidence(
                id="RUN-1",
                path="<container>",
                locator="c++ anchor.cpp -o /tmp/anchor && /tmp/anchor",
                observation="anchor completed",
            ),
            RepositoryEvidence(
                id="RUN-2",
                path="<container>",
                locator="c++ target.cpp -o /tmp/target && /tmp/target",
                observation="target exposed the reported failure",
            ),
        ]
        actions = [
            {
                "action": "RunInvestigationCommand",
                "evidence_id": "RUN-1",
                "command": "c++ anchor.cpp -o /tmp/anchor && /tmp/anchor",
                "exit_code": 0,
                "frozen_probe_files": {
                    "anchor.cpp": "int main() { return 0; }",
                },
            },
            {
                "action": "RunInvestigationCommand",
                "evidence_id": "RUN-2",
                "command": "c++ target.cpp -o /tmp/target && /tmp/target",
                "exit_code": 1,
                "frozen_probe_files": {
                    "target.cpp": "int main() { return 1; }",
                },
            },
        ]
        return self.report, evidence, actions


class NextIntentInvestigator(FakeContrastiveInvestigator):
    def __init__(self, report: DeveloperAnswer, next_report: DeveloperAnswer):
        super().__init__(report)
        self.next_report = next_report
        self.focused_intents = []

    @property
    def remaining_steps(self):
        return 20

    def investigate_next_intent(self, issue, issue_evidence, intent_unit):
        self.focused_intents.append(intent_unit.id)
        _, evidence, actions = self.investigate_contrastively(
            issue,
            issue_evidence,
            {},
        )
        return self.next_report, evidence, actions


def contrastive_report() -> DeveloperAnswer:
    return DeveloperAnswer(
        status="supported",
        answer="A passing anchor test and its Issue mutation were executed.",
        repository_evidence_ids=["SEARCH-1", "RUN-1", "RUN-2"],
        contrastive_witness=ContrastiveWitnessProposal(
            issue_claim_ids=["EXP-1"],
            anchor_test_path="tests/widget_parse_test.cpp",
            anchor_test_locator="TEST(Parse, ExistingInput)",
            anchor_source_evidence_id="SEARCH-1",
            anchor_selection_rationale=(
                "This is the closest passing test because it uses the same public "
                "operation and success oracle."
            ),
            anchor=ContrastiveProbeCandidate(
                id="ANCHOR-1",
                description=(
                    "The existing Widget::parse input remains accepted."
                ),
                run_evidence_id="RUN-1",
            ),
            target=ContrastiveProbeCandidate(
                id="TARGET-1",
                description=(
                    "Widget::parse accepts the additional input named by the Issue."
                ),
                run_evidence_id="RUN-2",
            ),
            mutation_summary=(
                "Replace only the existing accepted input with the additional "
                "string input requested by the Issue."
            ),
            preserved_context=[
                "The same Widget::parse API and success assertion are used."
            ],
            observed_trace_delta=(
                "ANCHOR-1 reaches the success assertion; TARGET-1 reaches the "
                "same call but reports that the requested string input is rejected."
            ),
            discrimination_analysis=(
                "The target distinguishes acceptance of the Issue-named input, "
                "but does not define behavior for other input families."
            ),
        ),
    )


def test_contrastive_mode_uses_one_adjudication_call_and_freezes_witness():
    decision = ContrastiveIntentDecision(
        witness_valid=True,
        boundary_resolved=True,
        selected_anchor_id="ANCHOR-1",
        selected_target_id="TARGET-1",
        witnessed_delta=(
            "The requested string input is rejected where the adjacent accepted "
            "input succeeds."
        ),
        supported_delta=(
            "Accept the Issue-requested input without changing the existing "
            "accepted input behavior."
        ),
        covered_issue_claim_ids=["EXP-1"],
        acceptance_authority="normative",
        repository_effects=[
            repository_effect(
                "preserve",
                "Existing accepted inputs remain accepted.",
            ),
            repository_effect(
                "compatibility_risk",
                "A broad input conversion could change documented handling "
                "of existing accepted inputs.",
            ),
        ],
        scope_evidence=[
            "Coverage first diverges after Widget::parse enters input dispatch."
        ],
        uncertainty=["The repository does not define additional input families."],
        reason="The target is a minimal Issue-guided mutation of the anchor.",
    )
    extractor = ScriptedExtractor(decision)

    result = IntentDialogue(
        extractor=extractor,
        investigator=FakeContrastiveInvestigator(contrastive_report()),
        mode="contrastive",
    ).run(
        "Widget::parse should accept string input.",
        issue_evidence(),
        {"reproduction_status": "reproduced"},
    )

    assert extractor.responses == []
    assert "source_diff_from_anchor" in extractor.prompts[0]
    assert "-int main() { return 0; }" in extractor.prompts[0]
    assert "+int main() { return 1; }" in extractor.prompts[0]
    assert result.stop_reason == "contrastive_boundary_resolved"
    assert result.exploration_plan.mode == "contrastive"
    assert result.intent_contract.behavioral_delta == [
        decision.supported_delta
    ]
    assert result.intent_contract.local_behavior_witnesses == [
        decision.witnessed_delta
    ]
    assert len(result.intent_contract.acceptance_checks) == 1
    assert result.intent_contract.acceptance_checks[0].baseline_exit_code == 1
    assert result.intent_contract.acceptance_checks[0].authority == "normative"
    assert result.intent_contract.witness_limitations == []
    assert "Issue-required target" in (
        result.intent_contract.acceptance_checks[0].description
    )
    assert "Existing accepted inputs" in result.intent_contract.preserve_behavior[0]
    assert "broad input conversion" in (
        result.intent_contract.compatibility_warnings[0]
    )
    assert "input dispatch" in result.intent_contract.scope_evidence[0]
    assert "additional input families" in result.intent_contract.open_decisions[0]
    assert "Established boundary:" in result.render_for_repair()
    assert "Observed execution boundaries" not in result.render_for_repair()
    assert "input dispatch" in result.intent_contract.render_for_review()


def test_evidence_review_drives_autonomous_followup_and_final_readiness():
    initial_review = EvidenceSufficiencyReview(
        decision="continue",
        reason="The report stops at a private helper without its public caller.",
        preserved_alternatives=["Repair the public caller's selection path."],
        next_investigation=EvidenceSufficiencyFollowup(
            question="Which production entry calls the observed helper?",
            why_material="The answer changes causal locus and repair scope.",
            known_anchor="SEARCH-1 helper definition",
        ),
    )
    final_review = EvidenceSufficiencyReview(
        decision="ready",
        reason="The production caller and competing locus are now grounded.",
        preserved_alternatives=["Repair the helper's shared implementation."],
    )
    decision = ContrastiveIntentDecision(
        witness_valid=True,
        boundary_resolved=False,
        next_action="leave_open",
        selected_anchor_id="ANCHOR-1",
        selected_target_id="TARGET-1",
        witnessed_delta="The requested input is rejected on the baseline.",
        covered_issue_claim_ids=["EXP-1"],
        uncertainty=["Either grounded causal locus may implement the request."],
        reason="Both repository-grounded loci remain compatible with the Issue.",
    )

    class EvidenceReviewInvestigator(FakeContrastiveInvestigator):
        def __init__(self, report):
            super().__init__(report)
            self.review_phases = []
            self.followups = []

        def record_evidence_sufficiency_review(self, review, *, phase):
            self.review_phases.append((phase, review.decision))

        def refine_evidence_sufficiency(
            self,
            issue,
            evidence,
            previous_report,
            review,
        ):
            self.followups.append(review.next_investigation.question)
            _, repository_evidence, actions = (
                super().investigate_contrastively(issue, evidence, {})
            )
            repository_evidence.append(RepositoryEvidence(
                id="SEARCH-2",
                path="src/widget.cpp",
                locator="sed -n 20,40p src/widget.cpp",
                observation=(
                    "Widget::parse is the production entry that calls "
                    "detail::parse_input."
                ),
            ))
            refined_report = previous_report.model_copy(update={
                "status": "partial",
                "answer": "The production caller relationship is grounded.",
                "repository_facts": [
                    "[SEARCH-2] Widget::parse calls detail::parse_input."
                ],
                "repository_evidence_ids": ["SEARCH-2"],
                "contrastive_witness": None,
            })
            return refined_report, repository_evidence, actions

    investigator = EvidenceReviewInvestigator(contrastive_report())
    reviewer = ScriptedExtractor(initial_review, final_review)
    result = IntentDialogue(
        extractor=ScriptedExtractor(decision),
        evidence_sufficiency_reviewer=reviewer,
        investigator=investigator,
        mode="contrastive",
    ).run(
        "Widget::parse should accept string input.",
        issue_evidence(),
        {},
    )

    assert len(reviewer.prompts) == 2
    assert "Which production entry calls the observed helper?" in (
        investigator.followups
    )
    assert "Widget::parse is the production entry" in reviewer.prompts[1]
    assert initial_review.next_investigation.question in reviewer.prompts[1]
    assert "Prior evidence-sufficiency reviews" in reviewer.prompts[1]
    assert investigator.review_phases == [
        ("initial", "continue"),
        ("final_handoff", "ready"),
    ]
    assert result.evidence_sufficiency_followup_performed is True
    assert result.hypothesis_ready is True
    assert result.evidence_sufficiency_review.decision == "ready"
    assert result.evidence_sufficiency_review.preserved_alternatives == [
        "Repair the public caller's selection path.",
        "Repair the helper's shared implementation.",
    ]
    assert result.investigation.report.contrastive_witness is not None
    assert "SEARCH-2" in result.investigation.report.repository_evidence_ids
    assert [
        action["phase"]
        for action in result.investigation.actions
        if action.get("action") == "ReviewEvidenceSufficiency"
    ] == ["initial", "final_handoff"]


def test_evidence_review_continue_requires_one_material_question():
    with pytest.raises(ValidationError, match="continue requires"):
        EvidenceSufficiencyReview(
            decision="continue",
            reason="One repository fact is missing.",
        )


def test_evidence_review_reason_accepts_5000_characters_but_not_more():
    review = EvidenceSufficiencyReview(
        decision="ready",
        reason="x" * 5000,
    )

    assert len(review.reason) == 5000
    with pytest.raises(ValidationError, match="at most 5000 characters"):
        EvidenceSufficiencyReview(
            decision="ready",
            reason="x" * 5001,
        )


def test_final_evidence_review_marks_a_saved_report_not_hypothesis_ready():
    initial_review = EvidenceSufficiencyReview(
        decision="ready",
        reason="The initial packet appears grounded.",
    )
    final_review = EvidenceSufficiencyReview(
        decision="continue",
        reason="The final packet still lacks the producer-to-consumer handoff.",
        next_investigation=EvidenceSufficiencyFollowup(
            question="Where is the state first produced before this consumer?",
            why_material="That fact can move the causal repair locus upstream.",
            known_anchor="SEARCH-1 consumer definition",
        ),
    )
    decision = ContrastiveIntentDecision(
        witness_valid=True,
        boundary_resolved=False,
        next_action="leave_open",
        selected_anchor_id="ANCHOR-1",
        selected_target_id="TARGET-1",
        witnessed_delta="The Issue input fails where the anchor succeeds.",
        covered_issue_claim_ids=["EXP-1"],
        reason="The pair is a valid local witness.",
    )

    result = IntentDialogue(
        extractor=ScriptedExtractor(decision),
        evidence_sufficiency_reviewer=ScriptedExtractor(
            initial_review,
            final_review,
        ),
        investigator=FakeContrastiveInvestigator(contrastive_report()),
        mode="contrastive",
    ).run(
        "Widget::parse should accept string input.",
        issue_evidence(),
        {},
    )

    assert result.investigation.report.status == "partial"
    assert result.hypothesis_ready is False
    assert result.evidence_sufficiency_review == final_review
    assert any(
        final_review.next_investigation.question in item
        for item in result.investigation.report.open_decisions
    )


def test_confirmed_intent_boundaries_become_f2p_and_p2p_checks():
    decision = ContrastiveIntentDecision(
        witness_valid=True,
        boundary_resolved=True,
        selected_anchor_id="ANCHOR-1",
        selected_target_id="TARGET-1",
        witnessed_delta="The requested string input fails on the baseline.",
        supported_delta="Widget::parse accepts the Issue-requested string input.",
        covered_issue_claim_ids=["EXP-1"],
        repository_effects=[
            repository_effect(
                "preserve",
                "Existing accepted inputs remain accepted.",
            )
        ],
        acceptance_authority="normative",
        reason="The Issue and public contract establish both boundaries.",
    )
    assessment = IntentBoundaryTestAssessment(
        contract_consistent=True,
        boundary_complete=True,
        evidence_ids=["EXP-1", "RUN-3", "RUN-4"],
        reason="The candidate set matches the explicit target and preservation boundary.",
        checks=[
            IntentBoundaryTestAssessmentItem(
                candidate_id="CHECK-1",
                run_evidence_id="RUN-3",
                kind="f2p",
                normative=True,
                discriminating=True,
                scope_resolved=True,
                evidence_ids=["EXP-1", "RUN-3"],
                covered_issue_claim_ids=["EXP-1"],
                covered_issue_intent_ids=["INTENT-1"],
                covered_property_ids=["PROP-1"],
                covered_route_ids=["ROUTE-1"],
                reason="The probe directly exercises the explicit target.",
            ),
            IntentBoundaryTestAssessmentItem(
                candidate_id="CHECK-2",
                run_evidence_id="RUN-4",
                kind="p2p",
                normative=True,
                discriminating=True,
                scope_resolved=True,
                evidence_ids=["RUN-4"],
                covered_property_ids=["PROP-2"],
                covered_route_ids=["ROUTE-2"],
                reason="The probe preserves the cited existing behavior.",
            ),
        ]
    )

    class MaterializingInvestigator(FakeContrastiveInvestigator):
        remaining_steps = 10

        def materialize_intent_boundary(
            self,
            issue,
            issue_evidence,
            contract,
            final_decision,
            routing,
        ):
            _, evidence, actions = self.investigate_contrastively(
                issue,
                issue_evidence,
                {},
            )
            evidence.extend([
                RepositoryEvidence(
                    id="RUN-3",
                    path="<container>",
                    locator="c++ f2p.cpp -o /tmp/f2p && /tmp/f2p",
                    observation="requested input rejected",
                ),
                RepositoryEvidence(
                    id="RUN-4",
                    path="<container>",
                    locator="c++ p2p.cpp -o /tmp/p2p && /tmp/p2p",
                    observation="existing input accepted",
                ),
            ])
            actions.extend([
                {
                    "action": "RunInvestigationCommand",
                    "evidence_id": "RUN-3",
                    "command": "c++ f2p.cpp -o /tmp/f2p && /tmp/f2p",
                    "exit_code": 1,
                    "frozen_probe_files": {
                        "f2p.cpp": "int main() { return 1; }",
                    },
                },
                {
                    "action": "RunInvestigationCommand",
                    "evidence_id": "RUN-4",
                    "command": "c++ p2p.cpp -o /tmp/p2p && /tmp/p2p",
                    "exit_code": 0,
                    "frozen_probe_files": {
                        "p2p.cpp": "int main() { return 0; }",
                    },
                },
            ])
            report = DeveloperAnswer(
                status="supported",
                answer="Materialized one target and one preservation boundary.",
                repository_evidence_ids=["RUN-3", "RUN-4"],
                acceptance_checks=[
                    DeveloperAcceptanceCheck(
                        description="Requested string input is accepted.",
                        run_evidence_id="RUN-3",
                        kind="f2p",
                        covered_issue_claim_ids=["EXP-1"],
                        covered_issue_intent_ids=["INTENT-1"],
                        covered_property_ids=["PROP-1"],
                        covered_route_ids=["ROUTE-1"],
                    ),
                    DeveloperAcceptanceCheck(
                        description="Existing accepted input remains accepted.",
                        run_evidence_id="RUN-4",
                        kind="p2p",
                        covered_property_ids=["PROP-2"],
                        covered_route_ids=["ROUTE-2"],
                    ),
                ],
            )
            return report, evidence, actions

    properties = IntentPropertyExtraction(properties=[
        IntentBehaviorProperty(
            id="PROP-7",
            role="target",
            authority="issue_explicit",
            issue_intent_ids=["INTENT-1"],
            predicate="Widget::parse accepts the Issue-requested string input.",
            scope=["Widget::parse string inputs"],
            evidence_ids=["EXP-1", "RUN-2"],
        ),
        IntentBehaviorProperty(
            id="PROP-9",
            role="preserve",
            authority="repository_contract",
            predicate="Existing accepted inputs remain accepted.",
            scope=["existing Widget::parse inputs"],
            evidence_ids=["SEARCH-1"],
        ),
    ])
    routing = IntentValidationRouting(routes=[
        IntentValidationRoute(
            id="ROUTE-7",
            property_id="PROP-1",
            mode="fixed_example",
            executable_oracle="The requested string is accepted.",
            generation_plan="Compile and run the exact Issue example.",
            rationale="The Issue establishes one exact input example.",
        ),
        IntentValidationRoute(
            id="ROUTE-3",
            property_id="PROP-2",
            mode="generative_property",
            executable_oracle="Every existing accepted input remains accepted.",
            generation_plan="Exercise the repository-established input partitions.",
            rationale="The public contract establishes a bounded input domain.",
        ),
    ])
    extractor = ScriptedExtractor(decision, properties, routing, assessment)
    result = IntentDialogue(
        extractor=extractor,
        investigator=MaterializingInvestigator(contrastive_report()),
        mode="contrastive",
    ).run(
        "Widget::parse should accept string input.",
        issue_evidence_with_intent(),
        {},
    )

    assert [check.kind for check in result.intent_contract.acceptance_checks] == [
        "f2p",
        "p2p",
    ]
    assert [
        check.baseline_exit_code
        for check in result.intent_contract.acceptance_checks
    ] == [1, 0]
    assert all(
        check.authority == "normative"
        for check in result.intent_contract.acceptance_checks
    )
    assert [item.id for item in result.intent_contract.properties] == [
        "PROP-1",
        "PROP-2",
    ]
    assert [
        check.covered_property_ids
        for check in result.intent_contract.acceptance_checks
    ] == [["PROP-1"], ["PROP-2"]]
    assert [
        check.covered_route_ids
        for check in result.intent_contract.acceptance_checks
    ] == [["ROUTE-1"], ["ROUTE-2"]]
    assert result.property_extraction is not None
    assert result.validation_routing is not None
    assert [route.id for route in result.validation_routing.routes] == [
        "ROUTE-1",
        "ROUTE-2",
    ]
    assert result.boundary_test_report is not None
    assert result.boundary_test_assessment == assessment
    assert 'current absence is "by design"' in extractor.prompts[0]
    assert "cannot exhibit the named target" in extractor.prompts[0]
    assert "Intent Property Extractor" in extractor.prompts[1]
    assert "compact, quantified behavior relation" in extractor.prompts[1]
    assert "counterfactual discriminator" in extractor.prompts[1]
    assert "internal mechanism" in extractor.prompts[1]
    assert "parallel evidence channels" in extractor.prompts[1]
    assert "descriptive evidence" in extractor.prompts[1]
    assert "experimental setup" in extractor.prompts[1]
    assert "not Issue authority" in extractor.prompts[1]
    assert "Cover every protected target INTENT-N" in extractor.prompts[1]
    assert "mechanism named directly by the Issue" in extractor.prompts[1]
    assert "fallback availability" in extractor.prompts[1]
    assert "independently evaluated mechanism result" in extractor.prompts[1]
    assert "Intent Validation Router" in extractor.prompts[2]
    assert "plausible narrow repair" in extractor.prompts[2]
    assert "changing only names" in extractor.prompts[2]
    assert "allocates validation only" in extractor.prompts[2]
    assert "mechanism's standard selection rules" in extractor.prompts[2]
    assert "Semantic Admission Assessor" in extractor.prompts[3]
    assert "Audit the authority of `open_decisions`" in extractor.prompts[3]
    assert "cannot exhibit the mechanism" in extractor.prompts[3]
    assert "experimental fixture separate from its oracle" in extractor.prompts[3]
    assert "copy the winner from an analogous helper" in extractor.prompts[3]
    assert "set `boundary_complete=false`" in extractor.prompts[3]
    assert "Do not return `boundary_complete=true`" in extractor.prompts[3]
    assert "Repository evidence packet" in extractor.prompts[3]
    assert '"id": "SEARCH-1"' in extractor.prompts[3]
    assert "`normative`" in extractor.prompts[3]
    assert "`discriminating`" in extractor.prompts[3]
    assert "`scope_resolved`" in extractor.prompts[3]
    assert "[F2P]" in result.render_for_repair()
    assert "[P2P]" in result.render_for_repair()

    decision_with_open_choice = decision.model_copy(update={
        "uncertainty": [
            "The requested behavior may belong to the default or an opt-in API."
        ]
    })
    diagnostic_assessment = IntentBoundaryTestAssessment(
        contract_consistent=False,
        boundary_complete=False,
        evidence_ids=["EXP-1", "RUN-3", "RUN-4"],
        diagnostic_property_ids=["PROP-1"],
        missing_discriminators=[
            "Distinguish the default API from the still-viable opt-in API."
        ],
        reason="The target property selects an unresolved public API policy.",
        checks=[
            assessment.checks[0].model_copy(update={
                "normative": False,
                "discriminating": False,
                "surviving_wrong_repair": (
                    "A separate opt-in API can satisfy the Issue."
                ),
                "missing_discriminator": (
                    "Exercise both still-viable public API choices."
                ),
            }),
            assessment.checks[1],
        ],
    )
    diagnostic_result = IntentDialogue(
        extractor=ScriptedExtractor(
            decision_with_open_choice,
            properties,
            routing,
            diagnostic_assessment,
        ),
        investigator=MaterializingInvestigator(contrastive_report()),
        mode="contrastive",
    ).run(
        "Widget::parse should accept string input.",
        issue_evidence(),
        {},
    )

    assert [item.id for item in diagnostic_result.intent_contract.properties] == [
        "PROP-2"
    ]
    assert [
        check.kind for check in diagnostic_result.intent_contract.acceptance_checks
    ] == ["p2p"]
    assert [
        check.authority
        for check in diagnostic_result.intent_contract.acceptance_checks
    ] == ["normative"]
    assert any(
        "PROP-1" in gap
        for gap in diagnostic_result.intent_contract.validation_gaps
    )
    assert "Unresolved Issue dimensions" in diagnostic_result.render_for_repair()
    assert "default API from the still-viable opt-in API" not in (
        diagnostic_result.render_for_repair()
    )
    assert "default API from the still-viable opt-in API" in (
        diagnostic_result.intent_contract.render_for_review()
    )

    incomplete_target_assessment = assessment.model_copy(update={
        "boundary_complete": False,
        "reason": (
            "The exact target is required, but the materialized property check "
            "does not cover the complete supported input partition."
        ),
        "checks": [
            assessment.checks[0].model_copy(update={
                "discriminating": False,
                "surviving_wrong_repair": (
                    "A repair can special-case the materialized property fixture."
                ),
                "missing_discriminator": (
                    "Exercise the remaining evidence-backed input partition."
                ),
            }),
            assessment.checks[1],
        ],
    })
    incomplete_result = IntentDialogue(
        extractor=ScriptedExtractor(
            decision,
            properties,
            routing,
            incomplete_target_assessment,
        ),
        investigator=MaterializingInvestigator(contrastive_report()),
        mode="contrastive",
    ).run(
        "Widget::parse should accept string input.",
        issue_evidence(),
        {},
    )

    assert [
        (check.kind, check.authority, check.source_evidence_id)
        for check in incomplete_result.intent_contract.acceptance_checks
    ] == [
        ("f2p", "normative", "RUN-2"),
        ("p2p", "normative", "RUN-4"),
    ]
    assert "Required executable checks" in incomplete_result.render_for_repair()


def test_boundary_check_without_matching_property_binding_is_rejected():
    report = DeveloperAnswer(
        status="supported",
        answer="Submitted a candidate with an unknown property binding.",
        repository_evidence_ids=["RUN-3"],
        acceptance_checks=[
            DeveloperAcceptanceCheck(
                description="Requested string input is accepted.",
                run_evidence_id="RUN-3",
                kind="f2p",
                covered_issue_claim_ids=["EXP-1"],
                covered_property_ids=["PROP-999"],
            )
        ],
    )

    class MismatchedPropertyInvestigator:
        remaining_steps = 1

        def materialize_intent_boundary(self, *args):
            return (
                report,
                [
                    RepositoryEvidence(
                        id="RUN-3",
                        path="<container>",
                        observation="requested input rejected",
                    )
                ],
                [
                    {
                        "action": "RunInvestigationCommand",
                        "evidence_id": "RUN-3",
                        "command": "c++ f2p.cpp -o /tmp/f2p && /tmp/f2p",
                        "exit_code": 1,
                        "frozen_probe_files": {
                            "f2p.cpp": "int main() { return 1; }",
                        },
                    }
                ],
            )

    assessment = IntentBoundaryTestAssessment(
        contract_consistent=True,
        boundary_complete=True,
        evidence_ids=["EXP-1", "RUN-3"],
        reason="The submitted candidate appears complete.",
        checks=[
            IntentBoundaryTestAssessmentItem(
                candidate_id="CHECK-1",
                run_evidence_id="RUN-3",
                kind="f2p",
                normative=True,
                discriminating=True,
                scope_resolved=True,
                evidence_ids=["EXP-1", "RUN-3"],
                covered_issue_claim_ids=["EXP-1"],
                covered_property_ids=["PROP-999"],
                reason="The candidate claims an unknown property.",
            )
        ],
    )
    contract = RepairIntentContract(properties=[
        IntentBehaviorProperty(
            id="PROP-1",
            role="target",
            authority="issue_explicit",
            predicate="Widget::parse accepts the requested string input.",
            evidence_ids=["EXP-1"],
        )
    ])
    dialogue = IntentDialogue(
        extractor=ScriptedExtractor(assessment),
        investigator=MismatchedPropertyInvestigator(),
        mode="contrastive",
    )

    _, _, frozen_checks, _, _ = dialogue._materialize_boundary_tests(
        "Widget::parse should accept string input.",
        issue_evidence(),
        contract,
        ContrastiveIntentDecision(
            witness_valid=False,
            witnessed_delta="",
            reason="The decision content is irrelevant to this binding test.",
        ),
        IntentValidationRouting(routes=[
            IntentValidationRoute(
                id="ROUTE-1",
                property_id="PROP-1",
                mode="fixed_example",
                executable_oracle="The requested string input is accepted.",
                generation_plan="Run the exact Issue example.",
                rationale="The Issue establishes an exact example.",
            )
        ]),
    )

    assert frozen_checks == []


def test_boundary_admission_distinguishes_candidates_sharing_one_run():
    report = DeveloperAnswer(
        status="supported",
        answer="One executable contains independent presence and value checks.",
        repository_evidence_ids=["RUN-3"],
        acceptance_checks=[
            DeveloperAcceptanceCheck(
                description="The requested method is callable.",
                run_evidence_id="RUN-3",
                kind="f2p",
                covered_issue_claim_ids=["EXP-1"],
                covered_issue_intent_ids=["INTENT-1"],
                covered_property_ids=["PROP-1"],
                covered_route_ids=["ROUTE-1"],
            ),
            DeveloperAcceptanceCheck(
                description="The requested method returns the expected value.",
                run_evidence_id="RUN-3",
                kind="f2p",
                covered_issue_claim_ids=["EXP-1"],
                covered_issue_intent_ids=["INTENT-1"],
                covered_property_ids=["PROP-2"],
                covered_route_ids=["ROUTE-2"],
            ),
        ],
    )

    class SharedRunInvestigator:
        remaining_steps = 1

        def materialize_intent_boundary(self, *args):
            return (
                report,
                [RepositoryEvidence(
                    id="RUN-3",
                    path="<container>",
                    observation="the requested method is missing",
                )],
                [{
                    "action": "RunInvestigationCommand",
                    "evidence_id": "RUN-3",
                    "command": "c++ check.cpp -o /tmp/check && /tmp/check",
                    "exit_code": 1,
                    "frozen_probe_files": {
                        "check.cpp": "int main() { return 1; }",
                    },
                }],
            )

    assessment = IntentBoundaryTestAssessment(
        contract_consistent=False,
        boundary_complete=False,
        evidence_ids=["EXP-1", "RUN-3"],
        reason="Presence is grounded, but the submitted value oracle is absent.",
        checks=[
            IntentBoundaryTestAssessmentItem(
                candidate_id="CHECK-1",
                run_evidence_id="RUN-3",
                kind="f2p",
                normative=True,
                discriminating=True,
                scope_resolved=True,
                evidence_ids=["EXP-1", "RUN-3"],
                covered_issue_claim_ids=["EXP-1"],
                covered_issue_intent_ids=["INTENT-1"],
                covered_property_ids=["PROP-1"],
                covered_route_ids=["ROUTE-1"],
                reason="The presence check is directly required by the Issue.",
            ),
            IntentBoundaryTestAssessmentItem(
                candidate_id="CHECK-2",
                run_evidence_id="RUN-3",
                kind="f2p",
                normative=False,
                discriminating=False,
                scope_resolved=True,
                evidence_ids=["EXP-1", "RUN-3"],
                covered_issue_claim_ids=["EXP-1"],
                covered_issue_intent_ids=["INTENT-1"],
                covered_property_ids=["PROP-2"],
                covered_route_ids=["ROUTE-2"],
                reason="The executable never evaluates the claimed value oracle.",
            ),
        ],
    )
    contract = RepairIntentContract(properties=[
        IntentBehaviorProperty(
            id="PROP-1",
            role="target",
            authority="issue_explicit",
            issue_intent_ids=["INTENT-1"],
            predicate="The requested method is callable.",
            evidence_ids=["EXP-1"],
        ),
        IntentBehaviorProperty(
            id="PROP-2",
            role="target",
            authority="issue_explicit",
            issue_intent_ids=["INTENT-1"],
            predicate="The requested method returns the expected value.",
            evidence_ids=["EXP-1"],
        ),
    ])
    routing = IntentValidationRouting(routes=[
        IntentValidationRoute(
            id="ROUTE-1",
            property_id="PROP-1",
            mode="compile_matrix",
            executable_oracle="The method is callable.",
            generation_plan="Compile the public call.",
            rationale="Callability is compile-time observable.",
        ),
        IntentValidationRoute(
            id="ROUTE-2",
            property_id="PROP-2",
            mode="fixed_example",
            executable_oracle="The result equals the expected value.",
            generation_plan="Execute one grounded example.",
            rationale="The Issue supplies the expected result.",
        ),
    ])
    dialogue = IntentDialogue(
        extractor=ScriptedExtractor(assessment),
        investigator=SharedRunInvestigator(),
        mode="contrastive",
    )

    _, _, frozen_checks, _, _ = dialogue._materialize_boundary_tests(
        "Add the requested method.",
        issue_evidence_with_intent(),
        contract,
        ContrastiveIntentDecision(
            witness_valid=False,
            witnessed_delta="",
            reason="The decision is not needed for this protocol test.",
        ),
        routing,
    )

    assert len(frozen_checks) == 1
    assert frozen_checks[0].candidate_id == "CHECK-1"
    assert frozen_checks[0].covered_property_ids == ["PROP-1"]
    assert frozen_checks[0].covered_route_ids == ["ROUTE-1"]

    contract.acceptance_checks = frozen_checks
    dialogue._reconcile_semantic_admission(contract, assessment)

    assert len(contract.acceptance_checks) == 1
    assert contract.acceptance_checks[0].candidate_id == "CHECK-1"


def test_diagnostic_property_pruning_removes_dependent_routes_and_checks():
    contract = RepairIntentContract(
        properties=[
            IntentBehaviorProperty(
                id="PROP-1",
                role="target",
                authority="issue_explicit",
                predicate="Keep this property.",
                evidence_ids=["EXP-1"],
            ),
            IntentBehaviorProperty(
                id="PROP-2",
                role="preserve",
                authority="repository_contract",
                predicate="Reject this unsupported property.",
                evidence_ids=["RUN-2"],
            ),
        ],
        validation_routes=[
            IntentValidationRoute(
                id="ROUTE-1",
                property_id="PROP-1",
                mode="fixed_example",
                executable_oracle="Keep this oracle.",
                generation_plan="Run the grounded example.",
                rationale="The property is retained.",
            ),
            IntentValidationRoute(
                id="ROUTE-2",
                property_id="PROP-2",
                mode="fixed_example",
                executable_oracle="Drop this oracle.",
                generation_plan="Run the unsupported example.",
                rationale="The property will be rejected.",
            ),
        ],
        acceptance_checks=[FrozenAcceptanceCheck(
            description="Unsupported check.",
            command="c++ check.cpp -o /tmp/check && /tmp/check",
            files={"check.cpp": "int main() { return 0; }"},
            source_evidence_id="RUN-2",
            baseline_exit_code=0,
            kind="p2p",
            covered_property_ids=["PROP-2"],
            covered_route_ids=["ROUTE-2"],
        )],
    )

    IntentDialogue._prune_diagnostic_contract(contract, {"PROP-2"})

    assert [item.id for item in contract.properties] == ["PROP-1"]
    assert [item.id for item in contract.validation_routes] == ["ROUTE-1"]
    assert contract.acceptance_checks == []


def test_semantic_admission_blocks_open_choice_and_unavailable_evidence():
    report = DeveloperAnswer(
        status="supported",
        answer="Materialized the default-behavior interpretation.",
        repository_evidence_ids=["RUN-3"],
        acceptance_checks=[
            DeveloperAcceptanceCheck(
                description="The default public type uses insertion order.",
                run_evidence_id="RUN-3",
                kind="f2p",
                covered_issue_claim_ids=["EXP-1"],
                covered_property_ids=["PROP-1"],
                covered_route_ids=["ROUTE-1"],
            )
        ],
    )

    class OpenChoiceInvestigator:
        remaining_steps = 1

        def materialize_intent_boundary(self, *args):
            return (
                report,
                [
                    RepositoryEvidence(
                        id="RUN-3",
                        path="<container>",
                        observation="default behavior does not preserve order",
                    )
                ],
                [
                    {
                        "action": "RunInvestigationCommand",
                        "evidence_id": "RUN-3",
                        "command": "c++ f2p.cpp -o /tmp/f2p && /tmp/f2p",
                        "exit_code": 1,
                        "frozen_probe_files": {
                            "f2p.cpp": "int main() { return 1; }",
                        },
                    }
                ],
            )

    contract = RepairIntentContract(
        properties=[
            IntentBehaviorProperty(
                id="PROP-1",
                role="target",
                authority="issue_explicit",
                predicate="Object iteration preserves definition order.",
                evidence_ids=["EXP-1"],
            )
        ],
        open_decisions=[
            "The Issue does not settle changing the default type versus an opt-in type."
        ],
    )
    routing = IntentValidationRouting(routes=[
        IntentValidationRoute(
            id="ROUTE-1",
            property_id="PROP-1",
            mode="fixed_example",
            executable_oracle="The default type preserves insertion order.",
            generation_plan="Run the exact Issue example on the default type.",
            rationale="The candidate selected one interpretation.",
        )
    ])
    decision = ContrastiveIntentDecision(
        witness_valid=False,
        witnessed_delta="",
        reason="The local witness does not settle the public type choice.",
    )
    open_choice_assessment = IntentBoundaryTestAssessment(
        contract_consistent=False,
        boundary_complete=False,
        evidence_ids=["EXP-1", "RUN-3"],
        diagnostic_property_ids=[],
        missing_discriminators=[
            "Compare an unchanged default type with a separate opt-in type."
        ],
        reason="The F2P selects one explicitly open public type policy.",
        checks=[
            IntentBoundaryTestAssessmentItem(
                candidate_id="CHECK-1",
                run_evidence_id="RUN-3",
                kind="f2p",
                normative=False,
                discriminating=False,
                scope_resolved=False,
                evidence_ids=["EXP-1", "RUN-3"],
                covered_issue_claim_ids=["EXP-1"],
                covered_property_ids=["PROP-1"],
                covered_route_ids=["ROUTE-1"],
                surviving_wrong_repair="A separate opt-in type satisfies the Issue.",
                missing_discriminator="Exercise both viable public type choices.",
                reason="The exact oracle requires changing the default type.",
            )
        ],
    )
    dialogue = IntentDialogue(
        extractor=ScriptedExtractor(open_choice_assessment),
        investigator=OpenChoiceInvestigator(),
        mode="contrastive",
    )

    _, saved_assessment, frozen_checks, _, _ = (
        dialogue._materialize_boundary_tests(
            "Preserve object definition order.",
            issue_evidence(),
            contract,
            decision,
            routing,
        )
    )

    assert frozen_checks == []
    assert saved_assessment is not None
    assert not saved_assessment.contract_consistent
    assert not saved_assessment.boundary_complete
    # The per-check semantic scope decision is authoritative even when the
    # model omits the redundant top-level diagnostic list.
    assert saved_assessment.diagnostic_property_ids == ["PROP-1"]

    miscited_assessment = open_choice_assessment.model_copy(update={
        "contract_consistent": True,
        "evidence_ids": ["EXP-1", "SEARCH-999"],
    })
    miscited_dialogue = IntentDialogue(
        extractor=ScriptedExtractor(miscited_assessment),
        investigator=OpenChoiceInvestigator(),
        mode="contrastive",
    )
    _, saved_assessment, frozen_checks, _, _ = (
        miscited_dialogue._materialize_boundary_tests(
            "Preserve object definition order.",
            issue_evidence(),
            contract,
            decision,
            routing,
        )
    )

    assert frozen_checks == []
    assert saved_assessment is not None
    assert not saved_assessment.contract_consistent
    assert any(
        "outside the supplied packet" in gap
        for gap in saved_assessment.missing_discriminators
    )


def test_semantic_admission_feedback_refines_rejected_boundary_check_once():
    class RefiningInvestigator:
        remaining_steps = 5

        def __init__(self):
            self.refinement_calls = 0

        @staticmethod
        def _result(run_id: str, description: str):
            report = DeveloperAnswer(
                status="supported",
                answer=description,
                repository_evidence_ids=[run_id],
                acceptance_checks=[
                    DeveloperAcceptanceCheck(
                        description=description,
                        run_evidence_id=run_id,
                        kind="f2p",
                        covered_issue_claim_ids=["EXP-1"],
                        covered_property_ids=["PROP-1"],
                        covered_route_ids=["ROUTE-1"],
                    )
                ],
            )
            return (
                report,
                [
                    RepositoryEvidence(
                        id=run_id,
                        path="<container>",
                        observation=description,
                    )
                ],
                [
                    {
                        "action": "RunInvestigationCommand",
                        "evidence_id": run_id,
                        "command": "c++ target.cpp -o /tmp/target && /tmp/target",
                        "exit_code": 1,
                        "frozen_probe_files": {
                            "target.cpp": "int main() { return 1; }",
                        },
                    }
                ],
            )

        def materialize_intent_boundary(self, *args):
            return self._result("RUN-3", "Independent mechanism probes.")

        def refine_intent_boundary(
            self,
            issue,
            issue_frame,
            contract,
            decision,
            routing,
            previous_report,
            assessment,
        ):
            self.refinement_calls += 1
            assert previous_report.repository_evidence_ids == ["RUN-3"]
            assert assessment.missing_discriminators
            return self._result(
                "RUN-5",
                "One fixture makes both mechanisms viable and distinguishes the winner.",
            )

    initial_assessment = IntentBoundaryTestAssessment(
        contract_consistent=True,
        boundary_complete=False,
        evidence_ids=["EXP-1", "RUN-3"],
        missing_discriminators=[
            "Exercise both viable mechanisms in one observable fixture."
        ],
        reason="Independent probes do not establish dispatch precedence.",
        checks=[
            IntentBoundaryTestAssessmentItem(
                candidate_id="CHECK-1",
                run_evidence_id="RUN-3",
                kind="f2p",
                normative=True,
                discriminating=False,
                scope_resolved=True,
                evidence_ids=["EXP-1", "RUN-3"],
                covered_issue_claim_ids=["EXP-1"],
                covered_property_ids=["PROP-1"],
                covered_route_ids=["ROUTE-1"],
                surviving_wrong_repair="The old mechanism can still win.",
                missing_discriminator=(
                    "Exercise both mechanisms in one observable fixture."
                ),
                reason="The candidate proves availability, not precedence.",
            )
        ],
    )
    refined_assessment = IntentBoundaryTestAssessment(
        contract_consistent=True,
        boundary_complete=True,
        evidence_ids=["EXP-1", "RUN-5"],
        reason="The replacement check distinguishes the requested mechanism.",
        checks=[
            IntentBoundaryTestAssessmentItem(
                candidate_id="CHECK-1",
                run_evidence_id="RUN-5",
                kind="f2p",
                normative=True,
                discriminating=True,
                scope_resolved=True,
                evidence_ids=["EXP-1", "RUN-5"],
                covered_issue_claim_ids=["EXP-1"],
                covered_property_ids=["PROP-1"],
                covered_route_ids=["ROUTE-1"],
                reason="The requested and competing paths coexist and disagree.",
            )
        ],
    )
    investigator = RefiningInvestigator()
    dialogue = IntentDialogue(
        extractor=ScriptedExtractor(initial_assessment, refined_assessment),
        investigator=investigator,
        mode="contrastive",
    )
    contract = RepairIntentContract(properties=[
        IntentBehaviorProperty(
            id="PROP-1",
            role="target",
            authority="issue_explicit",
            predicate="The public operation uses the requested mechanism.",
            evidence_ids=["EXP-1"],
        )
    ])
    routing = IntentValidationRouting(routes=[
        IntentValidationRoute(
            id="ROUTE-1",
            property_id="PROP-1",
            mode="fixed_example",
            executable_oracle="The requested mechanism wins.",
            generation_plan="Run one fixture where both paths are viable.",
            rationale="The Issue explicitly requests this mechanism.",
        )
    ])

    _, assessment, checks, _, _ = dialogue._materialize_boundary_tests(
        "The public operation must use the customization mechanism.",
        issue_evidence(),
        contract,
        ContrastiveIntentDecision(
            witness_valid=False,
            witnessed_delta="",
            reason="No local contrast is admitted yet.",
        ),
        routing,
    )

    assert investigator.refinement_calls == 1
    assert assessment == refined_assessment
    assert [check.source_evidence_id for check in checks] == ["RUN-5"]


def test_repair_handoff_marks_semantic_validation_gaps_as_non_completion():
    result = IntentDialogueResult(intent_contract=RepairIntentContract(
        validation_gaps=[
            "The current check does not distinguish member-first lookup from ADL precedence."
        ]
    ))

    rendered = result.render_for_repair()

    assert "not semantically admitted" in rendered
    assert "ADL precedence" not in rendered
    assert "ADL precedence" in result.intent_contract.render_for_review()


def test_repair_handoff_keeps_oracle_free_assessor_discriminator_audit_only():
    result = IntentDialogueResult(
        intent_contract=RepairIntentContract(
            validation_gaps=[
                "The candidate does not distinguish member lookup from ADL."
            ]
        ),
        boundary_test_assessment=IntentBoundaryTestAssessment(
            contract_consistent=True,
            boundary_complete=False,
            evidence_ids=["RUN-3"],
            missing_discriminators=[
                "Use one fixture where member lookup and ADL are both viable."
            ],
            reason="The local target leaves the dispatch winner untested.",
        ),
    )

    rendered = result.render_for_repair()

    assert "Assessor-confirmed unresolved validation limitation" in rendered
    assert "member lookup and ADL are both viable" not in rendered
    assert "Do not turn this gap into a Repair task" in rendered
    assert "buggy repository cannot resolve a product-policy choice" in rendered
    assert "proposed oracles are withheld" in rendered


def test_repair_handoff_prioritizes_concise_reusable_investigation_evidence():
    result = IntentDialogueResult(intent_contract=RepairIntentContract(
        intent_ledger=[
            RepairIntentLedgerEntry(
                id="INTENT-1",
                role="target",
                text="The public operation accepts ADL-only ranges.",
                issue_span="accepts ADL-only ranges",
                source_claim_ids=["EXP-1"],
                status="local_witness",
                witnessed_delta="The ADL-only probe fails in join.",
                evidence_ids=["SEARCH-1", "RUN-2"],
                evidence_notes=["A long raw compiler observation."],
            )
        ],
        local_behavior_witnesses=[
            "The ADL-only probe fails at the single-range overload."
        ],
        scope_evidence=[
            "The failure is pinned to include/fmt/ranges.h:598."
        ],
        validation_gaps=[
            "Member-first and free-function-first dispatch remain distinct."
        ],
    ))

    rendered = result.render_for_repair()

    assert "Immediate repair brief" in rendered
    assert "reuse before doing any new localization" in rendered
    assert "include/fmt/ranges.h:598" not in rendered
    assert "include/fmt/ranges.h:598" in (
        result.intent_contract.render_for_review()
    )
    assert "first candidate" in rendered
    assert "A long raw compiler observation" not in rendered
    assert "Evidence: SEARCH-1" not in rendered
    assert rendered.count("The ADL-only probe fails in join.") == 0


def test_semantic_admission_removes_rejected_policy_branch_from_handoff():
    contract = RepairIntentContract.model_validate({
        "intent_ledger": [
            {
                "id": "INTENT-1",
                "role": "target",
                "text": "The public operation uses ADL for customization.",
                "issue_span": "uses ADL",
                "source_claim_ids": ["EXP-1"],
                "status": "resolved",
                "witnessed_delta": "ADL-only input fails on baseline.",
                "supported_delta": (
                    "Member lookup must take precedence over free ADL lookup."
                ),
            }
        ],
        "target_behavior": [
            "The public operation uses ADL for customization."
        ],
        "properties": [
            {
                "id": "PROP-1",
                "role": "target",
                "authority": "issue_explicit",
                "issue_intent_ids": ["INTENT-1"],
                "predicate": "ADL-only input is accepted.",
                "evidence_ids": ["EXP-1"],
            },
            {
                "id": "PROP-2",
                "role": "target",
                "authority": "repository_contract",
                "issue_intent_ids": ["INTENT-1"],
                "predicate": "Member lookup wins over free ADL lookup.",
                "evidence_ids": ["RUN-2"],
            },
        ],
        "behavioral_delta": [
            "[INTENT-1] Member lookup must take precedence over free ADL lookup."
        ],
    })
    assessment = IntentBoundaryTestAssessment(
        contract_consistent=False,
        boundary_complete=False,
        evidence_ids=["EXP-1", "RUN-2"],
        diagnostic_property_ids=["PROP-2"],
        missing_discriminators=["The precedence policy remains unresolved."],
        reason="The proposed oracle chooses an unsupported policy branch.",
        checks=[
            IntentBoundaryTestAssessmentItem(
                candidate_id="CHECK-1",
                run_evidence_id="RUN-2",
                kind="f2p",
                normative=False,
                discriminating=True,
                scope_resolved=False,
                evidence_ids=["EXP-1", "RUN-2"],
                covered_issue_intent_ids=["INTENT-1"],
                covered_property_ids=["PROP-2"],
                reason="The Issue does not settle precedence.",
            )
        ],
    )

    IntentDialogue._reconcile_semantic_admission(contract, assessment)

    entry = contract.intent_ledger[0]
    assert entry.status == "local_witness"
    assert entry.supported_delta == ""
    assert contract.behavioral_delta == []
    rendered = IntentDialogueResult(intent_contract=contract).render_for_repair()
    assert "The public operation uses ADL for customization." in rendered
    assert "Member lookup must take precedence" not in rendered


def test_semantic_admission_packet_keeps_uncited_investigation_evidence():
    contract = RepairIntentContract(properties=[
        IntentBehaviorProperty(
            id="PROP-1",
            role="target",
            authority="issue_explicit",
            predicate="The requested input is accepted.",
            evidence_ids=["EXP-1", "SEARCH-1"],
        )
    ])
    report = DeveloperAnswer(
        status="supported",
        answer="Materialized one boundary candidate.",
        repository_evidence_ids=["RUN-3"],
    )
    packet = IntentDialogue._semantic_admission_evidence_packet(
        contract,
        ContrastiveIntentDecision(
            witness_valid=False,
            witnessed_delta="",
            reason="No test-pair claim is needed for this packet test.",
        ),
        report,
        [
            RepositoryEvidence(
                id="SEARCH-2",
                path="tests/sibling.cpp",
                observation="An uncited sibling exposes another public partition.",
            ),
            RepositoryEvidence(
                id="SEARCH-1",
                path="tests/target.cpp",
                observation="The cited target behavior.",
            ),
            RepositoryEvidence(
                id="RUN-3",
                path="<container>",
                observation="The boundary candidate failed on baseline.",
            ),
        ],
    )

    assert [item["id"] for item in packet] == [
        "SEARCH-1",
        "RUN-3",
        "SEARCH-2",
    ]
    assert packet[-1]["cited_by_proposed_contract"] is False


def test_contrastive_mode_preselects_and_persists_test_anchor_candidates():
    retrieval = AnchorRetrieval(
        status="candidates_found",
        query="Widget::parse string input",
        candidates=[
            AnchorCandidate(
                id="TEST-CAND-1",
                path="tests/widget_parse_test.cpp",
                locator="tests/widget_parse_test.cpp:10-35",
                label="TEST(Parse, ExistingInput)",
                claim_ids=["EXP-1"],
                retrieval_ranks={"lexical": 1, "graph": 1},
                rrf_score=0.03,
                snippet="EXPECT_TRUE(Widget::parse(existing_input));",
            )
        ],
    )
    selection = AnchorSelection(
        status="selected",
        selected_candidate_ids=["TEST-CAND-1"],
        primary_candidate_id="TEST-CAND-1",
        rationale="It uses the same public operation and success oracle.",
    )
    decision = ContrastiveIntentDecision(
        witness_valid=True,
        boundary_resolved=True,
        selected_anchor_id="ANCHOR-1",
        selected_target_id="TARGET-1",
        witnessed_delta="The requested string input is rejected on the baseline.",
        supported_delta="Accept the Issue-requested string input.",
        covered_issue_claim_ids=["EXP-1"],
        acceptance_authority="normative",
        reason="The selected test anchors the same operation and oracle.",
    )

    class RetrievingInvestigator(FakeContrastiveInvestigator):
        anchor_context = None

        def retrieve_test_anchor_candidates(self, issue, issue_evidence):
            return retrieval

        def set_test_anchor_context(self, retrieved, selected):
            self.anchor_context = (retrieved, selected)

    report = contrastive_report().model_copy(deep=True)
    report.contrastive_witness.anchor_candidate_id = "TEST-CAND-1"
    investigator = RetrievingInvestigator(report)
    extractor = ScriptedExtractor(selection, decision)
    result = IntentDialogue(
        extractor=extractor,
        investigator=investigator,
        mode="contrastive",
    ).run(
        "Widget::parse should accept string input.",
        issue_evidence(),
        {},
    )

    assert "Test Anchor Reranker" in extractor.prompts[0]
    assert "Recall@5" in extractor.prompts[0]
    assert "Assess every retrieved candidate" in extractor.prompts[0]
    assert "TEST-CAND-1" in extractor.prompts[0]
    assert "Candidate executions" in extractor.prompts[1]
    assert investigator.anchor_context == (retrieval, selection)
    assert result.investigation.anchor_retrieval == retrieval
    assert result.investigation.anchor_selection == selection
    assert (
        result.investigation.report.contrastive_witness.anchor_candidate_id
        == "TEST-CAND-1"
    )


def test_contrastive_test_anchor_retrieval_can_be_disabled_as_one_ablation():
    decision = ContrastiveIntentDecision(
        witness_valid=False,
        witnessed_delta="",
        reason="The witness is intentionally rejected in this control test.",
    )

    class DisabledRetrievalInvestigator(FakeContrastiveInvestigator):
        def retrieve_test_anchor_candidates(self, issue, issue_evidence):
            raise AssertionError("retrieval must not run when disabled")

    extractor = ScriptedExtractor(decision)
    result = IntentDialogue(
        extractor=extractor,
        investigator=DisabledRetrievalInvestigator(contrastive_report()),
        mode="contrastive",
        enable_test_anchor_retrieval=False,
    ).run(
        "Widget::parse should accept string input.",
        issue_evidence(),
        {},
    )

    assert len(extractor.prompts) == 2
    assert "Intent Property Extractor" in extractor.prompts[1]
    assert result.investigation.anchor_retrieval is None
    assert result.investigation.anchor_selection is None


def test_contrastive_witness_preserves_issue_claims_it_does_not_exercise():
    evidence = IssueEvidenceFrame.model_validate({
        "claims": [
            {
                "id": "EXP-1",
                "kind": "expected_behavior",
                "text": "Widget::parse accepts the requested input.",
                "issue_span": "accepts the requested input",
            },
            {
                "id": "OBS-1",
                "kind": "observed_behavior",
                "text": "A later state transition also fails.",
                "issue_span": "later state transition also fails",
            },
        ]
    })
    decision = ContrastiveIntentDecision(
        witness_valid=True,
        boundary_resolved=True,
        selected_anchor_id="ANCHOR-1",
        selected_target_id="TARGET-1",
        witnessed_delta="The requested input is rejected on the baseline.",
        supported_delta="Accept the requested input.",
        covered_issue_claim_ids=["EXP-1"],
        acceptance_authority="illustrative",
        reason="The pair exercises input acceptance but not later state.",
    )

    result = IntentDialogue(
        extractor=ScriptedExtractor(decision),
        investigator=FakeContrastiveInvestigator(contrastive_report()),
        mode="contrastive",
    ).run("Accept input and preserve later state.", evidence, {})

    assert result.contrastive_decision.unresolved_issue_claim_ids == ["OBS-1"]
    assert result.intent_contract.acceptance_checks[0].authority == "illustrative"
    assert result.intent_contract.witness_limitations == [
        "[OBS-1; observed_behavior] A later state transition also fails."
    ]
    rendered = result.render_for_repair()
    assert "Localization witnesses" in rendered
    assert decision.witnessed_delta in rendered
    assert "completion requirements" in rendered


def test_local_witness_cannot_discharge_an_unrelated_atomic_intent():
    evidence = IssueEvidenceFrame.model_validate({
        "claims": [
            {
                "id": "EXP-1",
                "kind": "expected_behavior",
                "text": "Widget::parse accepts the requested input.",
                "issue_span": "accepts the requested input",
            },
            {
                "id": "EXP-2",
                "kind": "expected_behavior",
                "text": "A failed parse preserves the cursor state.",
                "issue_span": "preserves the cursor state",
            },
        ],
        "intent_units": [
            {
                "id": "INTENT-1",
                "role": "target",
                "text": "Widget::parse accepts the requested input.",
                "issue_span": "accepts the requested input",
                "source_claim_ids": ["EXP-1"],
            },
            {
                "id": "INTENT-2",
                "role": "target",
                "text": "A failed parse preserves the cursor state.",
                "issue_span": "preserves the cursor state",
                "source_claim_ids": ["EXP-2"],
            },
        ],
    })
    report = contrastive_report()
    report.contrastive_witness.issue_intent_ids = ["INTENT-1"]
    decision = ContrastiveIntentDecision(
        witness_valid=True,
        boundary_resolved=True,
        selected_anchor_id="ANCHOR-1",
        selected_target_id="TARGET-1",
        witnessed_delta="The requested input is rejected on the baseline.",
        supported_delta="Accept the requested input.",
        covered_issue_claim_ids=["EXP-1"],
        covered_issue_intent_ids=["INTENT-1"],
        acceptance_authority="illustrative",
        reason="The pair exercises input acceptance only.",
    )

    result = IntentDialogue(
        extractor=ScriptedExtractor(decision),
        investigator=FakeContrastiveInvestigator(report),
        mode="contrastive",
    ).run("Accept input and preserve cursor state.", evidence, {})

    ledger = {item.id: item for item in result.intent_contract.intent_ledger}
    assert ledger["INTENT-1"].status == "resolved"
    assert ledger["INTENT-2"].status == "unresolved"
    assert result.contrastive_decision.unresolved_issue_intent_ids == ["INTENT-2"]
    assert result.intent_contract.target_behavior == [
        "Widget::parse accepts the requested input.",
        "A failed parse preserves the cursor state.",
    ]
    rendered = result.render_for_repair()
    assert "[INTENT-2] [TARGET] [UNRESOLVED]" in rendered
    assert "remain in scope" in rendered


def test_uncovered_atomic_intent_gets_an_independent_shared_budget_witness():
    evidence = issue_evidence_with_two_target_intents()
    first_report = contrastive_report()
    first_report.contrastive_witness.issue_intent_ids = ["INTENT-1"]
    first_report.repository_facts = [
        "[SEARCH-1] The parser input-acceptance path reaches Widget::parse."
    ]
    second_report = DeveloperAnswer(
        status="supported",
        answer="A separate cursor-state contrast was executed.",
        repository_facts=[
            "[SEARCH-2] Rejected input has an observable cursor-state contract."
        ],
        repository_evidence_ids=["SEARCH-2", "RUN-3", "RUN-4"],
        contrastive_witness=ContrastiveWitnessProposal(
            issue_claim_ids=["EXP-2"],
            issue_intent_ids=["INTENT-2"],
            anchor_test_path="tests/widget_cursor_test.cpp",
            anchor_test_locator="TEST(Parse, FailedInputKeepsCursor)",
            anchor_source_evidence_id="SEARCH-2",
            anchor_selection_rationale=(
                "This test observes cursor state around the same failed parse."
            ),
            anchor=ContrastiveProbeCandidate(
                id="ANCHOR-1",
                description="An adjacent failed parse preserves cursor state.",
                run_evidence_id="RUN-3",
            ),
            target=ContrastiveProbeCandidate(
                id="TARGET-1",
                description=(
                    "The Issue-reported failed parse preserves cursor state."
                ),
                run_evidence_id="RUN-4",
            ),
            mutation_summary=(
                "Change only the rejected input while retaining the cursor oracle."
            ),
            preserved_context=[
                "The same parser instance and cursor assertion are retained."
            ],
            observed_trace_delta=(
                "The anchor retains the cursor; the target advances it."
            ),
            discrimination_analysis=(
                "This pair checks state preservation and does not establish "
                "which additional inputs must be accepted."
            ),
        ),
    )

    class MultiIntentInvestigator(FakeContrastiveInvestigator):
        def __init__(self, report):
            super().__init__(report)
            self.focused_intents = []

        @property
        def remaining_steps(self):
            return 20

        def investigate_next_intent(self, issue, issue_evidence, intent_unit):
            self.focused_intents.append(intent_unit.id)
            _, repository_evidence, actions = (
                super().investigate_contrastively(issue, issue_evidence, {})
            )
            repository_evidence.extend([
                RepositoryEvidence(
                    id="SEARCH-2",
                    path="tests/widget_cursor_test.cpp",
                    locator="TEST(Parse, FailedInputKeepsCursor)",
                    observation=(
                        "The public parser contract checks cursor preservation "
                        "after rejected input."
                    ),
                ),
                RepositoryEvidence(
                    id="RUN-3",
                    path="<container>",
                    locator="c++ cursor_anchor.cpp -o /tmp/a && /tmp/a",
                    observation="cursor anchor completed",
                ),
                RepositoryEvidence(
                    id="RUN-4",
                    path="<container>",
                    locator="c++ cursor_target.cpp -o /tmp/t && /tmp/t",
                    observation="target advanced the cursor",
                ),
            ])
            actions.extend([
                {
                    "action": "RunInvestigationCommand",
                    "evidence_id": "RUN-3",
                    "command": "c++ cursor_anchor.cpp -o /tmp/a && /tmp/a",
                    "exit_code": 0,
                    "frozen_probe_files": {
                        "cursor_anchor.cpp": "int main() { return 0; }",
                    },
                },
                {
                    "action": "RunInvestigationCommand",
                    "evidence_id": "RUN-4",
                    "command": "c++ cursor_target.cpp -o /tmp/t && /tmp/t",
                    "exit_code": 1,
                    "frozen_probe_files": {
                        "cursor_target.cpp": "int main() { return 1; }",
                    },
                },
            ])
            return second_report, repository_evidence, actions

    first_decision = ContrastiveIntentDecision(
        witness_valid=True,
        boundary_resolved=True,
        selected_anchor_id="ANCHOR-1",
        selected_target_id="TARGET-1",
        witnessed_delta="The requested input is rejected on the baseline.",
        supported_delta="Accept the requested input.",
        covered_issue_claim_ids=["EXP-1"],
        covered_issue_intent_ids=["INTENT-1"],
        acceptance_authority="normative",
        reason="The pair establishes only input acceptance.",
    )
    second_decision = ContrastiveIntentDecision(
        witness_valid=True,
        boundary_resolved=True,
        selected_anchor_id="ANCHOR-1",
        selected_target_id="TARGET-1",
        witnessed_delta="The reported failure advances the cursor.",
        supported_delta="Preserve cursor state after the failed parse.",
        covered_issue_claim_ids=["EXP-2"],
        covered_issue_intent_ids=["INTENT-2"],
        acceptance_authority="normative",
        reason="The independent pair establishes state preservation.",
    )
    investigator = MultiIntentInvestigator(first_report)
    result = IntentDialogue(
        extractor=ScriptedExtractor(first_decision, second_decision),
        investigator=investigator,
        mode="contrastive",
    ).run("Accept input and preserve cursor state.", evidence, {})

    ledger = {item.id: item for item in result.intent_contract.intent_ledger}
    assert investigator.focused_intents == ["INTENT-2"]
    assert len(result.contrastive_attempts) == 2
    assert result.contrastive_decision.witnessed_delta == (
        second_decision.witnessed_delta
    )
    assert result.investigation.report.contrastive_witness == (
        second_report.contrastive_witness
    )
    assert [
        item.focus_intent_ids for item in result.investigation.report_history
    ] == [["INTENT-1", "INTENT-2"], ["INTENT-2"]]
    assert result.investigation.report_history[0].report.repository_facts == (
        first_report.repository_facts
    )
    assert result.investigation.report_history[1].report.repository_facts == (
        second_report.repository_facts
    )
    assert ledger["INTENT-1"].status == "resolved"
    assert ledger["INTENT-2"].status == "resolved"
    assert ledger["INTENT-1"].witness_ids == [
        "PAIR-1:ANCHOR-1",
        "PAIR-1:TARGET-1",
    ]
    assert ledger["INTENT-2"].witness_ids == [
        "PAIR-2:ANCHOR-1",
        "PAIR-2:TARGET-1",
    ]
    assert ledger["INTENT-1"].supported_delta == "Accept the requested input."
    assert ledger["INTENT-2"].supported_delta == (
        "Preserve cursor state after the failed parse."
    )
    assert len(result.intent_contract.acceptance_checks) == 2
    assert result.intent_contract.witness_limitations == []
    assert result.stop_reason == "contrastive_boundary_resolved"
    rendered = result.render_for_repair()
    assert "PAIR-1" not in rendered
    assert "Accept the requested input." in rendered
    assert "Preserve cursor state after the failed parse." in rendered
    assert "Grounding: [RUN-1]" not in rendered
    assert "Grounding: [RUN-3]" not in rendered


def test_invalid_independent_attempt_does_not_erase_selected_valid_witness():
    evidence = issue_evidence_with_two_target_intents()
    first_report = contrastive_report()
    first_report.contrastive_witness.issue_intent_ids = ["INTENT-1"]
    invalid_report = DeveloperAnswer(
        status="partial",
        answer="No honest independent cursor-state witness was established.",
        repository_evidence_ids=["SEARCH-1"],
        open_decisions=["A cursor-state discriminator remains unavailable."],
    )
    first_decision = ContrastiveIntentDecision(
        witness_valid=True,
        boundary_resolved=True,
        selected_anchor_id="ANCHOR-1",
        selected_target_id="TARGET-1",
        witnessed_delta="The requested input is rejected on the baseline.",
        supported_delta="Accept the requested input.",
        covered_issue_claim_ids=["EXP-1"],
        covered_issue_intent_ids=["INTENT-1"],
        acceptance_authority="normative",
        reason="The first pair establishes input acceptance.",
    )
    invalid_decision = ContrastiveIntentDecision(
        witness_valid=False,
        witnessed_delta="",
        reason="The second investigation did not establish a valid pair.",
    )
    investigator = NextIntentInvestigator(first_report, invalid_report)

    result = IntentDialogue(
        extractor=ScriptedExtractor(first_decision, invalid_decision),
        investigator=investigator,
        mode="contrastive",
    ).run("Accept input and preserve cursor state.", evidence, {})

    assert investigator.focused_intents == ["INTENT-2"]
    assert [
        attempt.decision.witness_valid
        for attempt in result.contrastive_attempts
    ] == [True, False]
    assert result.contrastive_decision.witnessed_delta == (
        first_decision.witnessed_delta
    )
    assert result.investigation.report.contrastive_witness == (
        first_report.contrastive_witness
    )


def test_later_valid_attempt_replaces_an_initial_invalid_witness():
    evidence = issue_evidence_with_intent()
    first_report = contrastive_report()
    first_report.contrastive_witness.issue_intent_ids = ["INTENT-1"]
    next_report = first_report.model_copy(deep=True)
    next_report.contrastive_witness.observed_trace_delta = (
        "The replacement target exposes the Issue-reported input boundary."
    )
    first_decision = ContrastiveIntentDecision(
        witness_valid=False,
        witnessed_delta="",
        reason="The initial pair did not preserve the Issue receiver.",
    )
    next_decision = ContrastiveIntentDecision(
        witness_valid=True,
        boundary_resolved=True,
        selected_anchor_id="ANCHOR-1",
        selected_target_id="TARGET-1",
        witnessed_delta="The replacement target rejects the requested input.",
        supported_delta="Accept the requested input.",
        covered_issue_claim_ids=["EXP-1"],
        covered_issue_intent_ids=["INTENT-1"],
        acceptance_authority="normative",
        reason="The replacement pair preserves the exact public receiver.",
    )
    investigator = NextIntentInvestigator(first_report, next_report)

    result = IntentDialogue(
        extractor=ScriptedExtractor(first_decision, next_decision),
        investigator=investigator,
        mode="contrastive",
    ).run("Widget::parse should accept string input.", evidence, {})

    assert investigator.focused_intents == ["INTENT-1"]
    assert [
        attempt.decision.witness_valid
        for attempt in result.contrastive_attempts
    ] == [False, True]
    assert result.contrastive_decision.witnessed_delta == (
        next_decision.witnessed_delta
    )
    assert result.investigation.report.contrastive_witness == (
        next_report.contrastive_witness
    )


def test_valid_local_witness_does_not_become_a_repair_boundary():
    decision = ContrastiveIntentDecision(
        witness_valid=True,
        boundary_resolved=False,
        selected_anchor_id="ANCHOR-1",
        selected_target_id="TARGET-1",
        witnessed_delta=(
            "The requested input is rejected while the adjacent anchor input "
            "is accepted."
        ),
        covered_issue_claim_ids=["EXP-1"],
        acceptance_authority="normative",
        repository_effects=[
            repository_effect(
                "preserve",
                "The public contract explicitly requires the anchor input to "
                "remain accepted.",
            )
        ],
        scope_evidence=["The target diverges at public input dispatch."],
        uncertainty=[
            "The evidence does not resolve whether the behavior is default or opt-in."
        ],
        reason=(
            "The pair is Issue-related, but more than one public behavior can "
            "make the local target pass."
        ),
    )

    result = IntentDialogue(
        extractor=ScriptedExtractor(decision),
        investigator=FakeContrastiveInvestigator(contrastive_report()),
        mode="contrastive",
    ).run(
        "Widget::parse should accept string input.",
        issue_evidence(),
        {},
    )

    assert result.stop_reason == "contrastive_witness_only"
    assert result.intent_contract.behavioral_delta == []
    assert result.intent_contract.local_behavior_witnesses == [
        decision.witnessed_delta
    ]
    assert result.intent_contract.preserve_behavior == [
        decision.repository_effects[0].statement
    ]
    assert result.intent_contract.scope_evidence == [
        "The target diverges at public input dispatch."
    ]
    assert result.intent_contract.open_decisions == decision.uncertainty
    assert result.intent_contract.acceptance_checks[0].authority == "illustrative"
    spec = result.intent_contract.verification_spec()
    assert spec["acceptance_ids"] == []
    assert spec["diagnostic_ids"] == ["ACCEPT-1"]
    assert spec["probes"]["ACCEPT-1"]["authority"] == "illustrative"
    assert "not a complete repair boundary" in (
        result.intent_contract.acceptance_checks[0].description
    )
    rendered = result.render_for_repair()
    assert "useful evidence" in rendered
    assert decision.witnessed_delta in rendered


def test_contrastive_refinement_closes_same_claim_boundary():
    first_decision = ContrastiveIntentDecision(
        witness_valid=True,
        boundary_resolved=False,
        next_action="refine_test_pair",
        followup_request=ContrastiveFollowupRequest(
            claim_ids=["EXP-1"],
            objective=(
                "Distinguish broad input acceptance from acceptance of only "
                "the sampled literal."
            ),
            why_current_evidence_is_insufficient=(
                "The first target exercises only one requested string value."
            ),
            evidence_needed=(
                "A same-claim target that checks two Issue-entitled string values."
            ),
            must_preserve=["Widget::parse public operation"],
            forbidden_inference="Do not define behavior for non-string inputs.",
        ),
        selected_anchor_id="ANCHOR-1",
        selected_target_id="TARGET-1",
        witnessed_delta="One requested string value is rejected on the baseline.",
        covered_issue_claim_ids=["EXP-1"],
        uncertainty=["The first target does not exclude literal-only acceptance."],
        reason="A stronger same-claim target can distinguish the remaining gap.",
    )
    refined_decision = ContrastiveIntentDecision(
        witness_valid=True,
        boundary_resolved=True,
        next_action="accept_boundary",
        selected_anchor_id="ANCHOR-1",
        selected_target_id="TARGET-1",
        witnessed_delta="Requested string inputs are rejected on the baseline.",
        supported_delta="Accept the Issue-requested class of string inputs.",
        covered_issue_claim_ids=["EXP-1"],
        acceptance_authority="normative",
        reason="The refined target distinguishes literal-only acceptance.",
    )

    class RefiningInvestigator(FakeContrastiveInvestigator):
        def __init__(self, report):
            super().__init__(report)
            self.refinement_requests = []

        @property
        def remaining_steps(self):
            return 20

        def refine_contrastively(
            self,
            issue,
            issue_evidence,
            previous_report,
            previous_decision,
            pair_status,
        ):
            self.refinement_requests.append(
                (previous_decision.followup_request, pair_status)
            )
            _, evidence, actions = super().investigate_contrastively(
                issue,
                issue_evidence,
                {},
            )
            report = contrastive_report().model_copy(deep=True)
            report.contrastive_witness.target.run_evidence_id = "RUN-3"
            report.contrastive_witness.mutation_summary = (
                "Strengthened the same string-input oracle with a second value."
            )
            report.repository_evidence_ids.append("RUN-3")
            evidence.append(
                RepositoryEvidence(
                    id="RUN-3",
                    path="<container>",
                    locator="c++ refined.cpp -o /tmp/refined && /tmp/refined",
                    observation="refined target exposed the reported failure",
                )
            )
            actions.append({
                "action": "RunInvestigationCommand",
                "evidence_id": "RUN-3",
                "command": "c++ refined.cpp -o /tmp/refined && /tmp/refined",
                "exit_code": 1,
                "frozen_probe_files": {
                    "refined.cpp": "int main() { return 1; }",
                },
            })
            return report, evidence, actions

    investigator = RefiningInvestigator(contrastive_report())
    extractor = ScriptedExtractor(first_decision, refined_decision)
    result = IntentDialogue(
        extractor=extractor,
        investigator=investigator,
        mode="contrastive",
    ).run(
        "Widget::parse should accept string input.",
        issue_evidence(),
        {},
    )

    assert len(result.contrastive_attempts) == 2
    assert result.contrastive_attempts[0].decision == first_decision
    assert (
        result.contrastive_attempts[1].witness.target.run_evidence_id
        == "RUN-3"
    )
    assert result.stop_reason == "contrastive_boundary_resolved"
    assert result.intent_contract.behavioral_delta == [
        refined_decision.supported_delta
    ]
    assert result.intent_contract.acceptance_checks[0].source_evidence_id == (
        "RUN-3"
    )
    assert investigator.refinement_requests[0][1] == "validated"
    assert "Candidate executions" in extractor.prompts[1]
    assert "RUN-3" in extractor.prompts[1]


def test_contrastive_leave_open_does_not_trigger_refinement():
    decision = ContrastiveIntentDecision(
        witness_valid=True,
        boundary_resolved=False,
        next_action="leave_open",
        selected_anchor_id="ANCHOR-1",
        selected_target_id="TARGET-1",
        witnessed_delta="The requested input is rejected on the baseline.",
        covered_issue_claim_ids=["EXP-1"],
        uncertainty=["The repository does not determine the public default."],
        reason="A stronger target would invent unspecified product policy.",
    )

    class NoRefinementInvestigator(FakeContrastiveInvestigator):
        @property
        def remaining_steps(self):
            return 20

        def refine_contrastively(self, *args, **kwargs):
            raise AssertionError("leave_open must not resume investigation")

    result = IntentDialogue(
        extractor=ScriptedExtractor(decision),
        investigator=NoRefinementInvestigator(contrastive_report()),
        mode="contrastive",
    ).run(
        "Widget::parse should accept string input.",
        issue_evidence(),
        {},
    )

    assert len(result.contrastive_attempts) == 1
    assert result.stop_reason == "contrastive_witness_only"
    assert result.intent_contract.open_decisions == decision.uncertainty


def test_contract_followup_reuses_prior_witness_and_new_repository_evidence():
    first_decision = ContrastiveIntentDecision(
        witness_valid=True,
        boundary_resolved=False,
        next_action="seek_repository_contract",
        followup_request=ContrastiveFollowupRequest(
            claim_ids=["EXP-1"],
            objective="Determine whether the requested behavior is opt-in.",
            why_current_evidence_is_insufficient="The pair permits both defaults.",
            evidence_needed="Direct public documentation of the exposure policy.",
        ),
        selected_anchor_id="ANCHOR-1",
        selected_target_id="TARGET-1",
        witnessed_delta="The requested behavior is absent on the baseline.",
        covered_issue_claim_ids=["EXP-1"],
        uncertainty=["Default versus opt-in exposure is unresolved."],
        reason="The public contract can settle the remaining choice.",
    )
    final_decision = ContrastiveIntentDecision(
        witness_valid=True,
        boundary_resolved=True,
        next_action="accept_boundary",
        selected_anchor_id="ANCHOR-1",
        selected_target_id="TARGET-1",
        witnessed_delta="The requested behavior is absent on the baseline.",
        supported_delta="Provide the requested behavior through the opt-in API.",
        covered_issue_claim_ids=["EXP-1"],
        acceptance_authority="illustrative",
        repository_effects=[
            RepositoryContractEffect(
                kind="target_qualifier",
                statement="Expose the requested behavior through the opt-in API.",
                evidence=[
                    RepositoryContractCitation(
                        evidence_id="SEARCH-2",
                        quote="The behavior is available through the opt-in API.",
                    )
                ],
            )
        ],
        reason="The follow-up contract evidence resolves the exposure policy.",
    )

    class ContractRefiningInvestigator(FakeContrastiveInvestigator):
        @property
        def remaining_steps(self):
            return 20

        def refine_contrastively(self, issue, evidence, report, decision, status):
            _, repository_evidence, actions = (
                super().investigate_contrastively(issue, evidence, {})
            )
            repository_evidence.append(
                RepositoryEvidence(
                    id="SEARCH-2",
                    path="docs/widget.md",
                    locator="sed -n 10,20p docs/widget.md",
                    observation=(
                        "The behavior is available through the opt-in API."
                    ),
                )
            )
            return (
                DeveloperAnswer(
                    status="supported",
                    answer="The public exposure policy is documented.",
                    repository_evidence_ids=["SEARCH-2"],
                ),
                repository_evidence,
                actions,
            )

    extractor = ScriptedExtractor(first_decision, final_decision)
    result = IntentDialogue(
        extractor=extractor,
        investigator=ContractRefiningInvestigator(contrastive_report()),
        mode="contrastive",
    ).run(
        "Widget::parse should accept string input.",
        issue_evidence(),
        {},
    )

    assert len(result.contrastive_attempts) == 2
    assert result.contrastive_attempts[1].witness is not None
    assert "SEARCH-2" in extractor.prompts[1]
    assert "Prior adjudication" in extractor.prompts[1]
    assert result.stop_reason == "contrastive_boundary_resolved"
    assert result.intent_contract.behavioral_delta == [
        final_decision.supported_delta
    ]
    assert result.intent_contract.target_behavior[-1] == (
        final_decision.repository_effects[0].statement
    )


def test_invalid_refined_target_does_not_erase_first_local_witness():
    request = ContrastiveFollowupRequest(
        claim_ids=["EXP-1"],
        objective="Distinguish the remaining same-claim behavior.",
        why_current_evidence_is_insufficient="The first target is too local.",
        evidence_needed="A stronger self-verifying target.",
    )
    first_decision = ContrastiveIntentDecision(
        witness_valid=True,
        boundary_resolved=False,
        next_action="refine_test_pair",
        followup_request=request,
        selected_anchor_id="ANCHOR-1",
        selected_target_id="TARGET-1",
        witnessed_delta="The first local target exposes the requested failure.",
        covered_issue_claim_ids=["EXP-1"],
        uncertainty=["The remaining same-claim behavior is unresolved."],
        reason="The local witness is valid but incomplete.",
    )
    rejected_refinement = ContrastiveIntentDecision(
        witness_valid=False,
        boundary_resolved=False,
        next_action="reject_witness",
        witnessed_delta="",
        reason="The replacement target passes on the baseline.",
    )

    class InvalidRefinementInvestigator(FakeContrastiveInvestigator):
        @property
        def remaining_steps(self):
            return 20

        def refine_contrastively(self, issue, evidence, report, decision, status):
            report, repository_evidence, actions = (
                super().investigate_contrastively(issue, evidence, {})
            )
            actions[-1]["exit_code"] = 0
            return report, repository_evidence, actions

    result = IntentDialogue(
        extractor=ScriptedExtractor(first_decision, rejected_refinement),
        investigator=InvalidRefinementInvestigator(contrastive_report()),
        mode="contrastive",
    ).run(
        "Widget::parse should accept string input.",
        issue_evidence(),
        {},
    )

    assert len(result.contrastive_attempts) == 2
    assert result.contrastive_attempts[1].pair_status == (
        "contrastive_invalid_execution_pair"
    )
    assert result.contrastive_decision == first_decision
    assert result.intent_contract.local_behavior_witnesses == [
        first_decision.witnessed_delta
    ]


def test_contrastive_refinement_prompt_keeps_intent_boundary():
    decision = ContrastiveIntentDecision(
        witness_valid=True,
        boundary_resolved=False,
        next_action="seek_repository_contract",
        followup_request=ContrastiveFollowupRequest(
            claim_ids=["EXP-1"],
            objective="Determine whether the behavior is default or opt-in.",
            why_current_evidence_is_insufficient="The test pair allows both.",
            evidence_needed="Direct public documentation for the default.",
        ),
        selected_anchor_id="ANCHOR-1",
        selected_target_id="TARGET-1",
        witnessed_delta="The requested behavior is absent on the baseline.",
        covered_issue_claim_ids=["EXP-1"],
        reason="Public contract evidence could settle the remaining choice.",
    )

    prompt = build_contrastive_refinement_task(
        "Widget::parse should accept string input.",
        issue_evidence(),
        contrastive_report().model_dump(),
        decision,
        "validated",
    )

    assert "not a request to design or implement the production patch" in prompt
    assert "observable alternatives named by the focused request" in prompt
    assert "repository-unresolvable" in prompt
    assert "two strongest distinct direct sources" in prompt
    assert "do not keep mining changelogs" in prompt


def test_contrastive_discards_preservation_with_uncited_contract_evidence():
    decision = ContrastiveIntentDecision(
        witness_valid=True,
        boundary_resolved=False,
        selected_anchor_id="ANCHOR-1",
        selected_target_id="TARGET-1",
        witnessed_delta="The requested input is rejected on the baseline.",
        covered_issue_claim_ids=["EXP-1"],
        repository_effects=[
            repository_effect(
                "preserve",
                "An adjacent input must remain accepted.",
                evidence_id="SEARCH-999",
            )
        ],
        reason="The target is a valid local witness.",
    )

    result = IntentDialogue(
        extractor=ScriptedExtractor(decision),
        investigator=FakeContrastiveInvestigator(contrastive_report()),
        mode="contrastive",
    ).run(
        "Widget::parse should accept string input.",
        issue_evidence(),
        {},
    )

    assert result.stop_reason == "contrastive_witness_only"
    assert result.intent_contract.local_behavior_witnesses == [
        decision.witnessed_delta
    ]
    assert result.intent_contract.preserve_behavior == []
    assert result.contrastive_decision.repository_effects == []


def test_contrastive_retains_more_than_three_open_intent_decisions():
    open_choices = [
        f"Observable policy choice {index} remains unresolved."
        for index in range(1, 7)
    ]
    decision = ContrastiveIntentDecision(
        witness_valid=True,
        boundary_resolved=False,
        selected_anchor_id="ANCHOR-1",
        selected_target_id="TARGET-1",
        witnessed_delta="The requested input is rejected on the baseline.",
        covered_issue_claim_ids=["EXP-1"],
        uncertainty=open_choices,
        reason="The local witness leaves several independent choices open.",
    )

    result = IntentDialogue(
        extractor=ScriptedExtractor(decision),
        investigator=FakeContrastiveInvestigator(contrastive_report()),
        mode="contrastive",
    ).run(
        "Widget::parse should accept string input.",
        issue_evidence(),
        {},
    )

    assert result.intent_contract.open_decisions == open_choices


def test_contrastive_decision_schema_separates_witness_and_boundary():
    assert (
        "witnessed_delta"
        in ContrastiveIntentDecision.model_json_schema()["required"]
    )
    with pytest.raises(ValidationError, match="unresolved boundary"):
        ContrastiveIntentDecision(
            witness_valid=True,
            boundary_resolved=False,
            selected_anchor_id="ANCHOR-1",
            selected_target_id="TARGET-1",
            witnessed_delta="The requested input is rejected on the baseline.",
            supported_delta="Silently promote the local target.",
            covered_issue_claim_ids=["EXP-1"],
            reason="The witness is local.",
        )

    with pytest.raises(ValidationError, match="supported behavior delta"):
        ContrastiveIntentDecision(
            witness_valid=True,
            boundary_resolved=True,
            selected_anchor_id="ANCHOR-1",
            selected_target_id="TARGET-1",
            witnessed_delta="The requested input is rejected on the baseline.",
            covered_issue_claim_ids=["EXP-1"],
            reason="The boundary is claimed to be resolved.",
        )

    with pytest.raises(ValidationError):
        RepositoryContractEffect(
            kind="compatibility_risk",
            statement="An existing public behavior could change.",
            evidence=[],
        )

    repository_only = ContrastiveIntentDecision(
        witness_valid=False,
        witnessed_delta="",
        repository_effects=[
            repository_effect(
                "target_qualifier",
                "Apply the documented string-input contract.",
            ),
            repository_effect(
                "preserve",
                "Documented existing inputs remain accepted.",
            ),
            repository_effect(
                "compatibility_risk",
                "A broad conversion could change documented existing inputs.",
            ),
            repository_effect(
                "open_decision",
                "Default versus opt-in behavior remains open.",
            ),
        ],
        reason="No valid pair exists, but direct public-contract evidence does.",
    )
    assert not repository_only.witness_valid
    assert len(repository_only.repository_effects) == 4

    with pytest.raises(ValidationError, match="witness-derived intent content"):
        ContrastiveIntentDecision(
            witness_valid=False,
            witnessed_delta="",
            uncertainty=["An uncited repository-only choice remains open."],
            reason="Repository-only uncertainty must use a cited effect.",
        )

    with pytest.raises(ValidationError, match="witness-derived intent content"):
        ContrastiveIntentDecision(
            witness_valid=False,
            witnessed_delta="",
            scope_evidence=["Execution diverges in one internal helper."],
            reason="An invalid pair cannot provide execution scope.",
        )


def test_invalid_pair_normalization_does_not_reintroduce_witness_coverage():
    frame = IssueEvidenceFrame.model_validate({
        "claims": [
            {
                "id": "EXP-1",
                "kind": "expected_behavior",
                "text": "Widget::parse should accept string input.",
                "issue_span": "Widget::parse should accept string input.",
            }
        ],
        "intent_units": [
            {
                "id": "INTENT-1",
                "role": "target",
                "text": "Widget::parse accepts string input.",
                "issue_span": "Widget::parse should accept string input.",
                "source_claim_ids": ["EXP-1"],
            }
        ],
    })
    decision = ContrastiveIntentDecision(
        witness_valid=True,
        boundary_resolved=False,
        selected_anchor_id="ANCHOR-1",
        selected_target_id="TARGET-1",
        witnessed_delta="The requested input fails where the anchor succeeds.",
        covered_issue_claim_ids=["EXP-1"],
        covered_issue_intent_ids=["INTENT-1"],
        repository_effects=[
            repository_effect(
                "preserve",
                "Existing accepted inputs remain accepted.",
            )
        ],
        reason="The local witness is controlled.",
    )

    rejected = IntentDialogue._enforce_pair_status(
        decision,
        "contrastive_invalid_execution_pair",
    )
    normalized = IntentDialogue._normalize_intent_coverage(
        frame,
        contrastive_report().contrastive_witness,
        rejected,
    )
    attempt = ContrastiveAttempt(
        attempt=1,
        pair_status="contrastive_invalid_execution_pair",
        witness=contrastive_report().contrastive_witness,
        executions=[],
        decision=normalized,
    )

    assert attempt.decision.witness_valid is False
    assert attempt.decision.unresolved_issue_intent_ids == []
    assert len(attempt.decision.repository_effects) == 1


def test_contrastive_claim_completeness_is_not_limited_by_task_budget():
    claims = [
        {
            "id": "EXP-1",
            "kind": "expected_behavior",
            "text": "The requested operation succeeds.",
            "issue_span": "operation succeeds",
        }
    ]
    claims.extend(
        {
            "id": f"OBS-{index}",
            "kind": "observed_behavior",
            "text": f"Observable condition {index} is reported by the Issue.",
            "issue_span": f"condition {index}",
        }
        for index in range(1, 12)
    )
    decision = ContrastiveIntentDecision(
        witness_valid=True,
        boundary_resolved=True,
        selected_anchor_id="ANCHOR-1",
        selected_target_id="TARGET-1",
        witnessed_delta="The requested operation fails on the baseline.",
        supported_delta="Make the requested operation succeed.",
        covered_issue_claim_ids=["EXP-1"],
        reason="The witness exercises only the operation result.",
    )

    result = IntentDialogue(
        extractor=ScriptedExtractor(decision),
        investigator=FakeContrastiveInvestigator(contrastive_report()),
        mode="contrastive",
    ).run(
        "Repair every reported observable condition.",
        IssueEvidenceFrame(claims=claims),
        {},
    )

    assert len(result.intent_contract.witness_limitations) == 11
    assert result.intent_contract.witness_limitations[-1].startswith("[OBS-11;")


def test_contrastive_investigation_can_ground_question_form_issue():
    issue = "Should Widget::parse accept string input?"
    evidence = IssueEvidenceFrame.model_validate({
        "claims": [
            {
                "id": "OBS-1",
                "kind": "observed_behavior",
                "text": "Widget::parse rejects string input.",
                "issue_span": issue,
            }
        ]
    })
    report = contrastive_report().model_copy(deep=True)
    report.contrastive_witness.issue_claim_ids = ["OBS-1"]
    decision = ContrastiveIntentDecision(
        witness_valid=True,
        boundary_resolved=True,
        selected_anchor_id="ANCHOR-1",
        selected_target_id="TARGET-1",
        witnessed_delta="The named operation rejects the Issue input.",
        supported_delta="Accept string input through the named public operation.",
        covered_issue_claim_ids=["OBS-1"],
        acceptance_authority="illustrative",
        reason="The target directly exercises the reported rejection.",
    )

    result = IntentDialogue(
        extractor=ScriptedExtractor(decision),
        investigator=FakeContrastiveInvestigator(report),
        mode="contrastive",
    ).run(issue, evidence, {})

    assert result.stop_reason == "contrastive_boundary_resolved"
    assert result.intent_contract.target_behavior == []
    assert result.intent_contract.behavioral_delta == [decision.supported_delta]


def test_contrastive_adjudication_uses_only_cited_repository_evidence():
    report = contrastive_report().model_copy(deep=True)
    report.repository_evidence_ids = ["SEARCH-1", "RUN-1", "RUN-2"]

    class InvestigatorWithUncitedNoise(FakeContrastiveInvestigator):
        def investigate_contrastively(self, issue, issue_evidence, reproduction):
            returned_report, evidence, actions = super().investigate_contrastively(
                issue,
                issue_evidence,
                reproduction,
            )
            evidence.append(
                RepositoryEvidence(
                    id="SEARCH-2",
                    path="src/unrelated.cpp",
                    locator="grep unrelated",
                    observation="UNCITED_EXPLORATORY_NOISE",
                )
            )
            return returned_report, evidence, actions

    extractor = ScriptedExtractor(
        ContrastiveIntentDecision(
            witness_valid=False,
            witnessed_delta="",
            reason="The target is not sufficiently discriminating.",
        )
    )
    IntentDialogue(
        extractor=extractor,
        investigator=InvestigatorWithUncitedNoise(report),
        mode="contrastive",
    ).run(
        "Widget::parse should accept string input.",
        issue_evidence(),
        {"reproduction_status": "reproduced"},
    )

    assert len(extractor.prompts) == 2
    assert all(
        "UNCITED_EXPLORATORY_NOISE" not in prompt
        for prompt in extractor.prompts
    )


def test_contrastive_rejects_claim_coverage_outside_the_witness_proposal():
    decision = ContrastiveIntentDecision(
        witness_valid=True,
        boundary_resolved=True,
        selected_anchor_id="ANCHOR-1",
        selected_target_id="TARGET-1",
        witnessed_delta="The requested input is rejected on the baseline.",
        supported_delta="Accept the requested input.",
        covered_issue_claim_ids=["EXP-999"],
        reason="The model cited a claim that the witness did not bind.",
    )

    result = IntentDialogue(
        extractor=ScriptedExtractor(decision),
        investigator=FakeContrastiveInvestigator(contrastive_report()),
        mode="contrastive",
    ).run(
        "Widget::parse should accept string input.",
        issue_evidence(),
        {},
    )

    assert result.stop_reason == "contrastive_invalid_claim_coverage"
    assert result.intent_contract.behavioral_delta == []
    assert result.intent_contract.acceptance_checks == []


def test_contrastive_candidate_bindings_are_independent():
    with pytest.raises(ValidationError, match="independent RUN-N"):
        ContrastiveWitnessProposal(
            issue_claim_ids=["EXP-1"],
            anchor_test_path="tests/widget_parse_test.cpp",
            anchor_test_locator="TEST(Parse, ExistingInput)",
            anchor_source_evidence_id="SEARCH-1",
            anchor_selection_rationale=(
                "The selected test uses the same public operation and oracle."
            ),
            anchor=ContrastiveProbeCandidate(
                id="ANCHOR-1",
                description="Existing scenario remains valid.",
                run_evidence_id="RUN-1",
            ),
            target=ContrastiveProbeCandidate(
                id="TARGET-1",
                description="Requested scenario becomes valid.",
                run_evidence_id="RUN-1",
            ),
            mutation_summary="Change only the Issue-named public input.",
            observed_trace_delta="The target rejects the changed input.",
            discrimination_analysis=(
                "The target distinguishes the requested input acceptance."
            ),
        )


def test_boundary_assessment_allows_one_run_to_cover_multiple_checks():
    assessment = IntentBoundaryTestAssessment(
        contract_consistent=True,
        boundary_complete=True,
        evidence_ids=["RUN-3"],
        reason="One combined executable validates both preserved properties.",
        checks=[
            IntentBoundaryTestAssessmentItem(
                candidate_id="CHECK-1",
                run_evidence_id="RUN-3",
                kind="p2p",
                normative=True,
                discriminating=True,
                scope_resolved=True,
                evidence_ids=["RUN-3"],
                covered_property_ids=["PROP-1"],
                covered_route_ids=["ROUTE-1"],
                reason="The first assertion preserves the first property.",
            ),
            IntentBoundaryTestAssessmentItem(
                candidate_id="CHECK-2",
                run_evidence_id="RUN-3",
                kind="p2p",
                normative=True,
                discriminating=True,
                scope_resolved=True,
                evidence_ids=["RUN-3"],
                covered_property_ids=["PROP-2"],
                covered_route_ids=["ROUTE-2"],
                reason="The second assertion preserves the second property.",
            ),
        ],
    )

    assert [check.run_evidence_id for check in assessment.checks] == [
        "RUN-3",
        "RUN-3",
    ]


def test_rejected_contrast_has_no_repository_contract_effect():
    decision = ContrastiveIntentDecision(
        witness_valid=False,
        witnessed_delta="",
        reason="The proposed target failure is not clearly tied to the Issue.",
    )

    result = IntentDialogue(
        extractor=ScriptedExtractor(decision),
        investigator=FakeContrastiveInvestigator(contrastive_report()),
        mode="contrastive",
    ).run(
        "Widget::parse should accept string input.",
        issue_evidence(),
        {"reproduction_status": "not_reproduced"},
    )

    assert result.stop_reason == "contrastive_no_effect"
    assert result.intent_contract.behavioral_delta == []
    assert result.intent_contract.acceptance_checks == []
    assert result.intent_contract.preserve_behavior == []


def test_repository_contract_evidence_survives_without_test_pair():
    report = DeveloperAnswer(
        status="partial",
        answer=(
            "No faithful executable pair was available, but public behavior "
            "was documented."
        ),
        repository_facts=[
            "The public documentation states that existing inputs remain accepted."
        ],
        open_decisions=[
            "The repository does not choose default versus opt-in behavior."
        ],
        repository_evidence_ids=["SEARCH-1"],
    )

    class RepositoryOnlyInvestigator:
        def investigate_contrastively(self, issue, evidence, reproduction):
            return (
                report,
                [
                    RepositoryEvidence(
                        id="SEARCH-1",
                        path="docs/widget.md",
                        locator="sed -n 10,30p docs/widget.md",
                        observation=(
                            "Public contract: Widget::parse accepts any string "
                            "input and preserves its parsed value. The public "
                            "documentation does not choose default versus "
                            "opt-in exposure for additional forms."
                        ),
                    )
                ],
                [],
            )

    decision = ContrastiveIntentDecision(
        witness_valid=False,
        witnessed_delta="",
        repository_effects=[
            repository_effect(
                "target_qualifier",
                "Apply the documented any-string input contract.",
                quote="Widget::parse accepts any string input",
            ),
            repository_effect(
                "preserve",
                "Preserve the parsed value of accepted string inputs.",
                quote="preserves its parsed value",
            ),
            repository_effect(
                "compatibility_risk",
                "A broad parser change could alter documented existing inputs.",
                quote="Widget::parse accepts any string input",
            ),
            repository_effect(
                "open_decision",
                "Default versus opt-in exposure remains unresolved.",
                quote=(
                    "does not choose default versus opt-in exposure for "
                    "additional forms"
                ),
            ),
        ],
        reason=(
            "The pair is absent, while the cited public documentation remains "
            "independently relevant."
        ),
    )
    extractor = ScriptedExtractor(decision)

    result = IntentDialogue(
        extractor=extractor,
        investigator=RepositoryOnlyInvestigator(),
        mode="contrastive",
    ).run(
        "Widget::parse should accept another string input.",
        issue_evidence(),
        {},
    )

    assert result.stop_reason == "contrastive_repository_evidence_only"
    assert result.pair_status == "contrastive_no_witness"
    assert "documented any-string" in result.intent_contract.target_behavior[-1]
    assert result.intent_contract.preserve_behavior == [
        decision.repository_effects[1].statement
    ]
    assert result.intent_contract.compatibility_warnings == [
        decision.repository_effects[2].statement
    ]
    assert result.intent_contract.open_decisions == [
        decision.repository_effects[3].statement
    ]
    assert result.intent_contract.behavioral_delta == []
    assert result.intent_contract.local_behavior_witnesses == []
    assert result.intent_contract.acceptance_checks == []
    assert "Mechanical pair status is `contrastive_no_witness`" in (
        extractor.prompts[0]
    )


def test_poc_mode_keeps_source_evidence_without_pair_or_acceptance_gate():
    report = contrastive_report().model_copy(update={
        "acceptance_checks": [proposed_check("A Scout-authored check")],
    })

    class POCInvestigator(FakeContrastiveInvestigator):
        def retrieve_test_anchor_candidates(self, *_args):
            raise AssertionError("POC mode must not retrieve a passing anchor")

        def investigate_next_intent(self, *_args):
            raise AssertionError("POC mode must not request another pair")

    decision = ContrastiveIntentDecision(
        witness_valid=False,
        witnessed_delta="",
        repository_effects=[repository_effect(
            "compatibility_risk",
            "The existing public parse test may be affected.",
        )],
        reason="The POC is diagnostic; the source test remains relevant.",
    )
    extractor = ScriptedExtractor(decision)
    result = IntentDialogue(
        extractor=extractor,
        investigator=POCInvestigator(report),
        mode="poc",
    ).run("Widget::parse should accept string input.", issue_evidence(), {})

    assert result.exploration_plan.mode == "poc"
    assert result.pair_status == "poc_only"
    assert result.investigation.report.contrastive_witness is None
    assert result.investigation.report.acceptance_checks == []
    assert result.intent_contract.acceptance_checks == []
    assert result.intent_contract.compatibility_warnings == [
        "The existing public parse test may be affected."
    ]
    assert "there is no test pair" in extractor.prompts[0]
    assert "construct an ANCHOR-1/TARGET-1 pair" in (
        build_poc_investigation_task("Issue", issue_evidence(), {})
    )
    poc_task = build_poc_investigation_task("Issue", issue_evidence(), {})
    assert "A sibling need not call the fault site" in " ".join(poc_task.split())
    assert "precise missing check" in poc_task
    assert "not a requirement" in poc_task


def test_invalid_execution_pair_discards_only_witness_derived_content():
    class PassingTargetInvestigator(FakeContrastiveInvestigator):
        def investigate_contrastively(self, issue, evidence, reproduction):
            report, repository_evidence, actions = (
                super().investigate_contrastively(issue, evidence, reproduction)
            )
            actions[1]["exit_code"] = 0
            return report, repository_evidence, actions

    decision = ContrastiveIntentDecision(
        witness_valid=True,
        boundary_resolved=True,
        selected_anchor_id="ANCHOR-1",
        selected_target_id="TARGET-1",
        witnessed_delta="The requested input is rejected on the baseline.",
        supported_delta="Accept the requested input.",
        covered_issue_claim_ids=["EXP-1"],
        repository_effects=[
            repository_effect(
                "compatibility_risk",
                "A broad parser change could alter existing accepted inputs.",
                quote="EXPECT_TRUE(Widget::parse(existing_input))",
            ),
        ],
        uncertainty=["The exact compatibility boundary remains unresolved."],
        reason="The model attempted to accept a mechanically invalid pair.",
    )

    result = IntentDialogue(
        extractor=ScriptedExtractor(decision),
        investigator=PassingTargetInvestigator(contrastive_report()),
        mode="contrastive",
    ).run(
        "Widget::parse should accept string input.",
        issue_evidence(),
        {},
    )

    assert result.stop_reason == "contrastive_repository_evidence_only"
    assert result.pair_status == "contrastive_invalid_execution_pair"
    assert not result.contrastive_decision.witness_valid
    assert result.intent_contract.behavioral_delta == []
    assert result.intent_contract.acceptance_checks == []
    assert result.intent_contract.preserve_behavior == []
    assert result.intent_contract.compatibility_warnings == [
        decision.repository_effects[0].statement
    ]
    assert result.intent_contract.open_decisions == []


def test_contrastive_submission_binds_real_run_evidence_without_semantic_rules():
    session = DeveloperSession()
    session.record(
        "sed -n 10,35p tests/widget_parse_test.cpp",
        "TEST(Parse, ExistingInput)",
        0,
        prefix="SEARCH",
        action="ReadOnlySearch",
    )
    session.record(
        "c++ anchor.cpp -o /tmp/anchor && /tmp/anchor",
        "anchor passed",
        0,
        prefix="RUN",
        action="RunInvestigationCommand",
        extra={
            "frozen_probe_files": {
                "anchor.cpp": "int main() { return 0; }",
            }
        },
    )
    session.record(
        "c++ target.cpp -o /tmp/target && /tmp/target",
        "target failed",
        1,
        prefix="RUN",
        action="RunInvestigationCommand",
        extra={
            "frozen_probe_files": {
                "target.cpp": "int main() { return 1; }",
            }
        },
    )
    submit = SubmitDeveloperAnswer(session=session)

    observation = submit.execute(
        SubmitDeveloperAnswerArgs(
            thoughts="",
            status="supported",
            answer="The executable contrast was collected.",
            contrastive_witness=contrastive_report().contrastive_witness,
        )
    )

    assert observation.terminal
    witness = observation.properties["investigation_report"][
        "contrastive_witness"
    ]
    assert witness["anchor"]["run_evidence_id"] == "RUN-1"
    assert witness["target"]["run_evidence_id"] == "RUN-2"


def test_contrastive_submission_rebinds_split_runs_for_repair_verification():
    session = DeveloperSession()
    source = session.record(
        "sed -n 10,35p tests/widget_parse_test.cpp",
        "TEST(Parse, ExistingInput)",
        0,
        prefix="SEARCH",
        action="ReadOnlySearch",
    )
    for name, run_exit in (("anchor", 0), ("target", 1)):
        files = {f"{name}.cpp": f"int main() {{ return {run_exit}; }}"}
        session.record(
            f"c++ {name}.cpp -o /tmp/{name}",
            "compiled",
            0,
            prefix="RUN",
            action="RunInvestigationCommand",
            extra={"frozen_probe_files": files},
        )
        session.record(
            f"/tmp/{name}",
            "executed",
            run_exit,
            prefix="RUN",
            action="RunInvestigationCommand",
            extra={"frozen_probe_files": files},
        )

    report = contrastive_report().model_copy(deep=True)
    report.repository_evidence_ids = [
        source.id,
        "RUN-1",
        "RUN-2",
        "RUN-3",
        "RUN-4",
    ]
    report.contrastive_witness.anchor.run_evidence_id = "RUN-2"
    report.contrastive_witness.target.run_evidence_id = "RUN-4"
    submit_payload = report.model_dump(exclude={"repository_evidence_ids"})
    submit_payload["evidence_ids"] = report.repository_evidence_ids
    observation = SubmitDeveloperAnswer(session=session).execute(
        SubmitDeveloperAnswerArgs(thoughts="", **submit_payload)
    )

    assert observation.terminal
    assert session.actions[2]["frozen_probe_command"] == (
        "c++ anchor.cpp -o /tmp/anchor && /tmp/anchor"
    )
    assert session.actions[4]["frozen_probe_command"] == (
        "c++ target.cpp -o /tmp/target && /tmp/target"
    )

    class SplitRunInvestigator:
        def investigate_contrastively(self, issue, evidence, reproduction):
            return report, list(session.evidence), list(session.actions)

    result = IntentDialogue(
        extractor=ScriptedExtractor(
            ContrastiveIntentDecision(
                witness_valid=True,
                boundary_resolved=True,
                selected_anchor_id="ANCHOR-1",
                selected_target_id="TARGET-1",
                witnessed_delta="The requested public input is rejected.",
                supported_delta="Accept the requested public input.",
                covered_issue_claim_ids=["EXP-1"],
                acceptance_authority="illustrative",
                reason="The split execution pair is a controlled witness.",
            )
        ),
        investigator=SplitRunInvestigator(),
        mode="contrastive",
    ).run("Widget::parse should accept string input.", issue_evidence(), {})

    check = result.intent_contract.acceptance_checks[0]
    assert check.command == "c++ target.cpp -o /tmp/target && /tmp/target"
    assert check.files == {"target.cpp": "int main() { return 1; }"}


def test_contrastive_submission_rejects_target_that_passes_on_baseline():
    session = DeveloperSession()
    session.record(
        "sed -n 10,35p tests/widget_parse_test.cpp",
        "TEST(Parse, ExistingInput)",
        0,
        prefix="SEARCH",
        action="ReadOnlySearch",
    )
    for name in ("anchor", "target"):
        session.record(
            f"c++ {name}.cpp -o /tmp/{name} && /tmp/{name}",
            f"{name} returned success",
            0,
            prefix="RUN",
            action="RunInvestigationCommand",
            extra={
                "frozen_probe_files": {
                    f"{name}.cpp": "int main() { return 0; }",
                }
            },
        )

    observation = SubmitDeveloperAnswer(session=session).execute(
        SubmitDeveloperAnswerArgs(
            thoughts="",
            status="supported",
            answer="The target printed an exception but returned success.",
            contrastive_witness=contrastive_report().contrastive_witness,
        )
    )

    assert observation.expect_correction
    assert "TARGET-1 must be a self-verifying" in observation.message


def test_contrastive_prompts_require_anchor_mutation_and_trace_delta():
    investigation_prompt = build_contrastive_investigation_task(
        "Widget::parse should accept string input.",
        issue_evidence(),
        {"reproduction_status": "not_reproduced"},
    )
    assert "execute the selected" in investigation_prompt
    assert "native test/build context" in investigation_prompt
    assert "Preserve test semantics rather than copying its harness" in investigation_prompt
    assert "Never compile framework runners" in investigation_prompt
    assert "report that limitation instead of rebuilding" in investigation_prompt
    assert "After one setup, compile, or link failure" in investigation_prompt
    assert "repeated build-command guessing does not investigate" in investigation_prompt
    assert "`$()` command substitution" in investigation_prompt
    assert "diagnostic-only" in investigation_prompt
    assert "Derive ANCHOR-1 faithfully" in investigation_prompt
    assert "changing one observable predicate" in investigation_prompt
    assert "do not broaden" in investigation_prompt
    assert "TARGET-1 to" in investigation_prompt
    assert "decide other claims" in investigation_prompt
    assert "Coverage and XRay collectors" in investigation_prompt

    adjudication_prompt = build_contrastive_intent_decision_prompt(
        "Widget::parse should accept string input.",
        issue_evidence(),
        {},
        contrastive_report().model_dump(),
        [],
        [],
    )
    assert "Repository-and-Test Intent Evidence" in adjudication_prompt
    assert "exact Issue spans and their typed claims outrank" in adjudication_prompt
    assert "not a second normative source" in adjudication_prompt
    assert "two independent evidence channels" in adjudication_prompt
    assert "remains admissible when no valid test" in adjudication_prompt
    assert "submit useful repository facts" in investigation_prompt
    assert "supported_delta" in adjudication_prompt
    assert "witness_valid" in adjudication_prompt
    assert "boundary_resolved" in adjudication_prompt
    assert "refine_test_pair" in adjudication_prompt
    assert "seek_repository_contract" in adjudication_prompt
    assert "leave_open" in adjudication_prompt
    assert "uncertainty_routes" not in adjudication_prompt
    assert "without inventing maintainer policy" in adjudication_prompt
    assert "repository_effects" in adjudication_prompt
    assert "target_qualifier" in adjudication_prompt
    assert "intentionally broader than the" in adjudication_prompt
    assert "likely repair" in adjudication_prompt
    assert "exact quotes" in adjudication_prompt
    assert "scope_evidence" in adjudication_prompt
    assert "covered_issue_claim_ids" in adjudication_prompt
    assert "unresolved_issue_claim_ids" in adjudication_prompt
    assert "acceptance_authority" in adjudication_prompt
    assert "too local to specify the repair" in adjudication_prompt
    assert "different observable behavior" in adjudication_prompt
    assert "plausible wrong repairs" in investigation_prompt
    assert "Keep a discriminator experimentally controlled" in investigation_prompt
    assert "equivalent public inputs" in investigation_prompt
    assert "unrelated conversion, fixture, or environment differences" in investigation_prompt
    assert "implementation_suggestions" in investigation_prompt
    assert "compatibility_risk" in adjudication_prompt
    assert "warning, not a requirement" in adjudication_prompt
    assert "Patch Reviewer owns structural" in adjudication_prompt
    assert "Do not consider files, wrappers, generated mirrors" in adjudication_prompt
    assert "lower-bound evidence" in investigation_prompt
    assert "not the stopping condition" in investigation_prompt
    assert "Hypothesis would still have to" in investigation_prompt
    assert "submit immediately" not in investigation_prompt
    assert "Repair and Patch Reviewer own" not in investigation_prompt
    assert "materially narrower repair" not in investigation_prompt
    assert "Do not turn the path selected on the failing baseline" in investigation_prompt
    assert "public ingress, seed" in investigation_prompt
    assert "same `candidate_boundaries` entry" in investigation_prompt
    assert "one direct-entry discriminator" in investigation_prompt
    assert "which concrete representation it" in investigation_prompt
    assert "regression route exercises that mirror" in investigation_prompt
    assert "not authority to require every nearby copy" in investigation_prompt
    assert "has exactly one\ncompiler invocation" in investigation_prompt
    assert "Keep `repository_facts` atomic" in investigation_prompt
    assert "winner is causal localization" in adjudication_prompt
    assert "outcome-compatible alternative" in adjudication_prompt
    assert "finite repository-evidenced public family" in adjudication_prompt

    review_prompt = build_evidence_sufficiency_review_prompt(
        "Widget::parse should accept string input.",
        issue_evidence(),
        contrastive_report().model_dump(),
        [{
            "id": "SEARCH-1",
            "path": "src/widget.cpp",
            "locator": "sed -n 1,20p src/widget.cpp",
            "observation": "detail::parse_input handles the value",
        }],
        [],
        report_history=[{
            "focus_intent_ids": ["INTENT-1"],
            "report": {"repository_facts": ["Earlier causal fact"]},
        }],
        prior_adjudications=[{
            "attempt": 1,
            "boundary_resolved": False,
            "reason": "The public receiver remains unresolved.",
        }],
    )
    assert "without guessing a repository fact" in review_prompt
    assert "single highest-value investigation question" in review_prompt
    assert "Do not prescribe grep, graph, compilation" in review_prompt
    assert "examples of reasoning, not mandatory checklist items" in review_prompt
    assert "Do not equate being able to describe several plausible branches" in (
        review_prompt
    )
    assert "challenge the report with the most consequential" in review_prompt
    assert "downstream failure\nsite" in review_prompt
    assert "Exact future edit shape" in review_prompt
    assert "Earlier causal fact" in review_prompt
    assert "The public receiver remains unresolved." in review_prompt
    assert "concrete critique, not an automatic veto" in review_prompt
    assert "rehearse each live interpretation" in review_prompt
    assert "Apply one counterfactual to each branch" in review_prompt
    assert "not automatically a\npreservation requirement" in review_prompt
    assert "An absent direct call to the fault site does not rule out" in review_prompt
    assert "ingress returns a wrapper" in review_prompt
    assert "packaged or generated form" in review_prompt
    assert "Issue asks to change\n  that default" in review_prompt

    refinement_prompt = build_evidence_sufficiency_refinement_task(
        "Widget::parse should accept string input.",
        issue_evidence(),
        contrastive_report().model_dump(),
        EvidenceSufficiencyReview(
            decision="continue",
            reason="The production caller is unknown.",
            next_investigation=EvidenceSufficiencyFollowup(
                question="Which public entry calls detail::parse_input?",
                why_material="The caller changes repair scope.",
                known_anchor="SEARCH-1 detail::parse_input",
            ),
        ),
    )
    assert "reviewer did not select a repair" in refinement_prompt
    assert "according\nto your own judgment" in refinement_prompt
    assert "Which public entry calls detail::parse_input?" in refinement_prompt

    no_llvm_prompt = build_contrastive_investigation_task(
        "Widget::parse should accept string input.",
        issue_evidence(),
        {},
        set(),
    )
    assert "LLVM dynamic collection is unavailable" in no_llvm_prompt
    assert "CollectCoverageContrast" not in no_llvm_prompt


def test_unreproduced_context_is_explicitly_unvalidated_in_prompts():
    context = compact_reproduction_context(
        SimpleNamespace(
            is_reproduced=False,
            failure_class="compilation",
            test_command="c++ attempted_repro.cpp",
            execution_records=[],
            unresolved=["The observed failure did not match the issue."],
        )
    )

    assert context["reproduction_status"] == "not_reproduced"
    assert context["evidence_validated"] is False

    prompt = build_exploration_plan_prompt(
        "Issue text", IssueEvidenceFrame(), context
    )
    assert "Reproduction attempt context" in prompt
    assert "investigation" in prompt
    assert "history only" in prompt


class FallbackDeveloper:
    def answer(self, issue, issue_evidence, reproduction, question):
        return (
            DeveloperAnswer(
                status="unresolved",
                answer=question.question,
                open_decisions=[
                    "Developer Agent exhausted its search steps."
                ],
            ),
            [],
            [],
        )


def test_exhausted_investigation_does_not_create_contract_delta():
    extractor = ScriptedExtractor(
        IntentExplorationPlan(
            objectives=[
                objective(
                    question="Where is the behavior defined?"
                )
            ]
        ),
    )

    result = IntentDialogue(
        extractor=extractor,
        developer=FallbackDeveloper(),
    ).run(
        "Widget::parse should accept string input.",
        issue_evidence(),
        {"execution_records": []},
    )

    assert result.stop_reason == "investigation_no_delta"
    assert result.exchanges[0].answer.status == "unresolved"
    rendered = result.render_for_repair()
    assert "Developer Agent exhausted its search steps." not in rendered
    assert "Widget::parse should accept string input." in rendered


class SubmittedOpenQuestionDeveloper:
    def answer(self, issue, issue_evidence, reproduction, question):
        answer = DeveloperAnswer(
            status="unresolved",
            answer="The repository does not define this product decision.",
            open_decisions=[
                "The repository does not define this product decision."
            ],
        )
        return (
            answer,
            [],
            [
                {
                    "action": "SubmitDeveloperAnswer",
                    "answer": answer.model_dump(),
                }
            ],
        )


def test_submitted_unresolved_answer_remains_audit_only():
    extractor = ScriptedExtractor(
        IntentExplorationPlan(
            objectives=[
                objective(
                    question="Is this product behavior specified?"
                )
            ]
        ),
    )

    result = IntentDialogue(
        extractor=extractor,
        developer=SubmittedOpenQuestionDeveloper(),
    ).run(
        "Widget::parse should accept string input.",
        issue_evidence(),
        {"execution_records": []},
    )

    assert result.stop_reason == "investigation_no_delta"
    rendered = result.render_for_repair()
    assert "The repository does not define this product decision." not in rendered
    assert "Widget::parse should accept string input." in rendered
    assert "repository_facts" not in rendered


def test_contrastive_report_downgrades_a_missing_or_unmatched_input_path():
    from INTENT2FIX.intent_dialogue.schema import InvestigationInputPathCheck
    session = DeveloperSession(require_input_path_check=True)
    session.record('read consumer', 'if (!j.is_array()) reject();', 0)
    submit = SubmitDeveloperAnswer(session=session)
    args = SubmitDeveloperAnswerArgs(thoughts='', status='partial',
        answer='A downstream failure is observed.', evidence_ids=['SEARCH-1'])
    missing = submit.execute(args)
    assert missing.terminal
    missing_report = missing.properties['investigation_report']
    assert missing_report['status'] == 'partial'
    assert missing_report['input_path_check'] is None
    assert any('original Issue input' in item
               for item in missing_report['open_decisions'])
    args.input_path_check = InvestigationInputPathCheck(
        actual_input='JSON object', consumer_input='JSON array',
        condition_evidence_id='SEARCH-1', condition_quote='!j.is_array()',
        evaluation='The array guard rejects the object input.', matches_input='no',
        alternative_path='Object consumer participation condition not yet inspected.',
    )
    args.status = 'supported'
    accepted = submit.execute(args)
    assert accepted.terminal
    report = accepted.properties['investigation_report']
    assert report['status'] == 'partial'
    assert report['input_path_check']['matches_input'] == 'no'
    assert report['repository_evidence_ids'] == ['SEARCH-1']


def test_input_path_check_cannot_cite_fabricated_condition():
    from INTENT2FIX.intent_dialogue.schema import InvestigationInputPathCheck
    session = DeveloperSession(require_input_path_check=True)
    session.record('read consumer', 'if (!j.is_array()) reject();', 0)
    submit = SubmitDeveloperAnswer(session=session)
    args = SubmitDeveloperAnswerArgs(thoughts='', status='partial', answer='Checked.',
        input_path_check=InvestigationInputPathCheck(
            actual_input='object', consumer_input='array', condition_evidence_id='SEARCH-1',
            condition_quote='j.is_object()', evaluation='Fabricated guard',
            matches_input='yes', alternative_path='',
        ))
    assert submit.execute(args).expect_correction
    session.reset()
    assert session.require_input_path_check


def test_input_path_cannot_relabel_diagnostic_variant_as_original_issue():
    from INTENT2FIX.intent_dialogue.schema import InvestigationInputPathCheck
    session = DeveloperSession(require_input_path_check=True,
        issue_text='Convert the JSON object {"key": 1} to Map.')
    session.record('read consumer', 'if (!j.is_array()) reject();', 0)
    args = SubmitDeveloperAnswerArgs(thoughts='', status='partial', answer='Checked.',
        input_path_check=InvestigationInputPathCheck(
            issue_input_quote='JSON array', actual_input='array', consumer_input='array',
            condition_evidence_id='SEARCH-1', condition_quote='!j.is_array()',
            evaluation='Array probe passed', matches_input='yes', alternative_path='',
        ))
    submit = SubmitDeveloperAnswer(session=session)
    assert submit.execute(args).expect_correction
    args.input_path_check.issue_input_quote = 'JSON object {"key": 1}'
    args.input_path_check.actual_input = 'object'
    args.input_path_check.matches_input = 'no'
    assert submit.execute(args).terminal


def test_input_path_quotes_allow_line_ending_transport_changes_only():
    from INTENT2FIX.intent_dialogue.schema import InvestigationInputPathCheck
    session = DeveloperSession(require_input_path_check=True,
        issue_text='JSON object\r\nconvert to Map')
    session.record('read consumer', 'if (gate)\r\nconsume_object();', 0)
    args = SubmitDeveloperAnswerArgs(thoughts='', status='partial', answer='Checked.',
        input_path_check=InvestigationInputPathCheck(
            issue_input_quote='JSON object\nconvert to Map', actual_input='object to Map',
            consumer_input='object', condition_evidence_id='SEARCH-1',
            condition_quote='if (gate)\nconsume_object();', evaluation='source reasoning',
            matches_input='unknown', alternative_path='unknown',
        ))
    submit = SubmitDeveloperAnswer(session=session)
    assert submit.execute(args).terminal
    args.input_path_check.issue_input_quote = 'JSON array\nconvert to Map'
    assert submit.execute(args).expect_correction


@pytest.mark.parametrize('quote', ['', 'gate=false', 'exit_code=0', 'invented compiler output'])
def test_input_path_premise_probe_is_bound_to_real_run_and_retained(quote):
    from INTENT2FIX.intent_dialogue.schema import InvestigationInputPathCheck
    session = DeveloperSession(require_input_path_check=True)
    session.record('read consumer', 'enable_if<gate>', 0)
    session.record('run focused trait probe', 'gate=false', 0,
                   prefix='RUN', action='RunInvestigationCommand')
    args = SubmitDeveloperAnswerArgs(thoughts='', status='partial', answer='Checked.',
        input_path_check=InvestigationInputPathCheck(
            actual_input='object', consumer_input='object', condition_evidence_id='SEARCH-1',
            condition_quote='enable_if<gate>', evaluation='gate=false',
            verification_run_id='RUN-1', verification_quote=quote,
            matches_input='no', alternative_path='Fallback is selected.',
        ))
    submit = SubmitDeveloperAnswer(session=session)
    result = submit.execute(args)
    assert result.terminal
    assert 'RUN-1' in result.properties['investigation_report']['repository_evidence_ids']
    assert args.input_path_check.verification_quote == '[RUN-1] exit_code=0\ngate=false'
    args.input_path_check.verification_quote = 'enable_if<gate>'
    args.input_path_check.verification_run_id = 'SEARCH-1'
    assert submit.execute(args).expect_correction
