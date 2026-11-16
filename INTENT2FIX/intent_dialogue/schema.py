from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from INTENT2FIX.reproduce.intent_schema import IssueEvidenceFrame


# Provider reasoning is stored separately as `reasoning_content` and is governed
# by the completion token budget. These limits apply only to the structured
# report emitted by SubmitInvestigationReport.
INVESTIGATION_REPORT_SUMMARY_MAX_CHARS = 3000
INVESTIGATION_WITNESS_ANALYSIS_MAX_CHARS = 2400


IntentTaskType = Literal[
    "behavior_boundary",
    "compatibility_boundary",
    "representation_or_error_contract",
    "acceptance_example",
    "repository_convention",
]
AnswerStatus = Literal["supported", "partial", "unresolved", "conflict"]
EvidenceAuthority = Literal[
    "issue_referenced_policy",
    "public_contract",
    "behavioral_baseline",
    "implementation_structure",
]
VersionSpaceStatus = Literal["resolved", "unresolved", "conflict"]
IntentResolutionKind = Literal[
    "behavior_qualifier",
    "preservation_boundary",
    "acceptance_check",
    "no_effect",
]
IntentResolutionEvidenceType = Literal[
    "explicit_contract",
    "consistent_public_api_convention",
    "issue_example_alignment",
    "insufficient",
]
AcceptanceAuthority = Literal["normative", "illustrative"]
IntentCheckKind = Literal["f2p", "p2p", "diagnostic"]
IntentPropertyRole = Literal["target", "preserve"]
IntentPropertyAuthority = Literal["issue_explicit", "repository_contract"]
IntentValidationMode = Literal[
    "generative_property",
    "fixed_example",
    "compile_matrix",
    "structural_review",
    "text_only",
]
IntentLedgerStatus = Literal[
    "uninvestigated",
    "repository_evidence",
    "local_witness",
    "resolved",
    "unresolved",
    "conflict",
]
RepositoryContractEffectKind = Literal[
    "target_qualifier",
    "preserve",
    "compatibility_risk",
    "open_decision",
]
ContrastiveNextAction = Literal[
    "accept_boundary",
    "refine_test_pair",
    "seek_repository_contract",
    "leave_open",
    "reject_witness",
]
class IntentHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default="", pattern=r"^(?:(?:DIM-\d+-H\d+)|(?:H\d+))?$")
    description: str = Field(..., min_length=1, max_length=400)
    qualifiers: list[str] = Field(
        default_factory=list,
        max_length=4,
        description=(
            "Atomic observable qualifiers unique to this interpretation. "
            "Do not include files, symbols, or implementation mechanisms."
        ),
    )


class IntentInvestigationTask(BaseModel):
    model_config = ConfigDict(extra="forbid", title="IntentEvidenceTodo")

    id: str = Field(default="", pattern=r"^(?:(?:TASK|DIM|TODO|OBJ)-\d+)?$")
    goal: str = Field(..., min_length=1, max_length=500)
    task_type: IntentTaskType
    related_issue_ids: list[str] = Field(default_factory=list)
    related_intent_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Atomic INTENT-N obligations whose missing boundary this TODO "
            "investigates. One TODO may bind multiple intents only when the "
            "same repository fact distinguishes them together."
        ),
    )
    rationale: str = Field(..., min_length=1, max_length=500)
    done_when: str = Field(..., min_length=1, max_length=500)
    hypotheses: list[IntentHypothesis] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "Legacy version-space alternatives. New evidence TODO plans leave "
            "this empty and form alternatives only after repository search."
        ),
    )
    discriminator: str = Field(
        default="",
        max_length=500,
        description=(
            "The normative repository evidence that could distinguish the "
            "hypotheses. This is an evidence condition, not a search command."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_objective_fields(cls, value):
        if not isinstance(value, dict):
            return value
        value = dict(value)
        if "goal" not in value and "question" in value:
            value["goal"] = value.pop("question")
        if "task_type" not in value and "question_type" in value:
            value["task_type"] = value.pop("question_type")
        return value

    @model_validator(mode="after")
    def validate_task(self) -> "IntentInvestigationTask":
        if not self.goal.strip():
            raise ValueError("goal is required for an investigation task")
        return self

    @property
    def question(self) -> str:
        """Compatibility accessor for older result readers."""
        return self.goal

    @property
    def question_type(self) -> IntentTaskType:
        """Compatibility accessor for older result readers."""
        return self.task_type


class IntentInvestigationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", title="IntentEvidenceTodoPlan")

    mode: Literal[
        "contrastive",
        "poc",
        "evidence_todos",
        "version_space",
        "legacy",
    ] = "evidence_todos"
    tasks: list[IntentInvestigationTask] = Field(
        default_factory=list,
        max_length=10,
    )

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_plan_field(cls, value):
        if not isinstance(value, dict):
            return value
        value = dict(value)
        if "tasks" not in value and "objectives" in value:
            value.setdefault("mode", "legacy")
            value["tasks"] = value.pop("objectives")
        return value

    @property
    def objectives(self) -> list[IntentInvestigationTask]:
        """Compatibility accessor for older result readers."""
        return self.tasks


class IntentSynthesisSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preserve_candidate_ids: list[str] = Field(default_factory=list, max_length=4)
    acceptance_candidate_ids: list[str] = Field(default_factory=list, max_length=4)
    warning_candidate_ids: list[str] = Field(default_factory=list, max_length=3)


# Compatibility aliases for persisted callers. New prompts and results use the
# task-oriented names above.
IntentExplorationObjective = IntentInvestigationTask
IntentExplorationPlan = IntentInvestigationPlan
ExecutorTurn = IntentInvestigationTask
IntentAmbiguityDimension = IntentInvestigationTask
IntentVersionSpacePlan = IntentInvestigationPlan


class RepositoryEvidence(BaseModel):
    id: str
    path: str
    locator: str = ""
    observation: str


class HypothesisEvidenceEffect(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(..., pattern=r"^(?:SEARCH|RUN|GRAPH)-\d+$")
    authority: EvidenceAuthority
    supports_hypothesis_ids: list[str] = Field(default_factory=list, max_length=3)
    eliminates_hypothesis_ids: list[str] = Field(default_factory=list, max_length=3)
    contract_quote: str = Field(
        default="",
        max_length=500,
        description=(
            "Exact quote from the cited observation. Required when evidence "
            "eliminates a hypothesis."
        ),
    )
    explanation: str = Field(..., min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_discrimination(self) -> "HypothesisEvidenceEffect":
        overlap = set(self.supports_hypothesis_ids) & set(
            self.eliminates_hypothesis_ids
        )
        if overlap:
            raise ValueError("one hypothesis cannot be both supported and eliminated")
        if self.eliminates_hypothesis_ids and not self.supports_hypothesis_ids:
            raise ValueError("elimination requires at least one supported alternative")
        return self


class InvestigationAcceptanceCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(
        ...,
        max_length=1600,
        description=(
            "Executable public scenario directly entailed by an explicit "
            "Issue target. Do not use a probe to define a new default, type "
            "category, API shape, or other maintainer decision. Do not bind "
            "an ambiguous behavior to the existing default public type."
        ),
    )
    run_evidence_id: str = Field(..., pattern=r"^RUN-\d+$")
    kind: IntentCheckKind = Field(
        default="diagnostic",
        description=(
            "f2p for a confirmed target that fails on the buggy baseline and "
            "must pass after repair; p2p for a confirmed preservation boundary "
            "that passes before and after repair; diagnostic for legacy checks."
        ),
    )
    covered_issue_claim_ids: list[str] = Field(default_factory=list, max_length=10)
    covered_issue_intent_ids: list[str] = Field(default_factory=list, max_length=10)
    covered_property_ids: list[str] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "PROP-N behavior properties exercised by this exact probe. Boundary "
            "materialization should leave the check empty rather than inventing "
            "a property binding."
        ),
    )
    covered_route_ids: list[str] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "ROUTE-N validation plans exercised by this exact probe. "
            "Executable boundary checks must bind both their properties and "
            "the route that authorized this validation form."
        ),
    )


class FrozenAcceptanceCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(default="", pattern=r"^(?:CHECK-\d+)?$")
    description: str = Field(..., max_length=1600)
    command: str
    files: dict[str, str] = Field(default_factory=dict)
    source_evidence_id: str = Field(..., pattern=r"^RUN-\d+$")
    baseline_exit_code: int
    kind: IntentCheckKind = "diagnostic"
    authority: AcceptanceAuthority = Field(
        default="normative",
        description=(
            "Whether the exact probe oracle is required by the Issue/public "
            "contract or is only one executable illustration of the semantic "
            "behavior change."
        ),
    )
    covered_issue_claim_ids: list[str] = Field(default_factory=list, max_length=10)
    covered_issue_intent_ids: list[str] = Field(default_factory=list, max_length=10)
    covered_property_ids: list[str] = Field(default_factory=list, max_length=10)
    covered_route_ids: list[str] = Field(default_factory=list, max_length=10)


class IntentPropertyDimension(BaseModel):
    """One evidence-backed dimension over which a predicate may vary."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=120)
    domain: str = Field(
        ...,
        min_length=1,
        max_length=400,
        description=(
            "Evidence-supported value, type, state, or configuration domain. "
            "Do not claim an unbounded domain from one example."
        ),
    )
    representatives: list[str] = Field(default_factory=list, max_length=8)
    evidence_ids: list[str] = Field(default_factory=list, min_length=1, max_length=10)


class IntentBehaviorProperty(BaseModel):
    """One quantified observable behavior relation backed by authority."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., pattern=r"^PROP-\d+$")
    role: IntentPropertyRole
    authority: IntentPropertyAuthority
    issue_intent_ids: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Atomic Issue intents whose behavior this property validates.",
    )
    predicate: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description=(
            "Implementation-independent relation over the smallest "
            "evidence-supported public domain. Never a private mechanism, "
            "diagnostic cause, candidate representation, or one witness "
            "presented as a universal rule."
        ),
    )
    oracle: str = Field(
        default="",
        max_length=500,
        description=(
            "Independent observable relation or outcome that decides every "
            "case in the stated domain without using the behavior under test "
            "to compute its own expected result. Empty when evidence does not "
            "establish a sound executable oracle."
        ),
    )
    dimensions: list[IntentPropertyDimension] = Field(
        default_factory=list,
        max_length=5,
        description=(
            "Independent evidence-backed input, public type/API form, state, "
            "sequence, or configuration axes. Representatives seed bounded "
            "validation; one witness is not proof of a generative domain "
            "unless its authority independently quantifies that domain."
        ),
    )
    scope: list[str] = Field(
        default_factory=list,
        max_length=4,
        description=(
            "Public inputs, states, operations, or API family to which the "
            "predicate applies; never implementation files or helper symbols."
        ),
    )
    evidence_ids: list[str] = Field(default_factory=list, min_length=1, max_length=10)


class IntentPropertyExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    properties: list[IntentBehaviorProperty] = Field(
        default_factory=list,
        max_length=10,
    )

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "IntentPropertyExtraction":
        property_ids = [item.id for item in self.properties]
        if len(property_ids) != len(set(property_ids)):
            raise ValueError("intent property IDs must be unique")
        return self


class IntentValidationRoute(BaseModel):
    """One LLM-selected validation form for one grounded property."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., pattern=r"^ROUTE-\d+$")
    property_id: str = Field(..., pattern=r"^PROP-\d+$")
    mode: IntentValidationMode
    executable_oracle: str = Field(default="", max_length=1000)
    generation_plan: str = Field(
        default="",
        max_length=1600,
        description=(
            "A behavior-level validation plan, never production files, patch "
            "mechanisms, or implementation edits."
        ),
    )
    rationale: str = Field(..., min_length=1, max_length=1000)


class IntentValidationRouting(BaseModel):
    model_config = ConfigDict(extra="forbid")

    routes: list[IntentValidationRoute] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def validate_unique_bindings(self) -> "IntentValidationRouting":
        route_ids = [item.id for item in self.routes]
        property_ids = [item.property_id for item in self.routes]
        if len(route_ids) != len(set(route_ids)):
            raise ValueError("intent validation route IDs must be unique")
        if len(property_ids) != len(set(property_ids)):
            raise ValueError("each property must have at most one validation route")
        return self


class ContrastiveProbeCandidate(BaseModel):
    """One frozen execution in an anchor/mutated-test intent witness."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., pattern=r"^(?:ANCHOR-1|TARGET-1)$")
    description: str = Field(..., min_length=1, max_length=500)
    run_evidence_id: str = Field(..., pattern=r"^RUN-\d+$")


class TestAnchorCandidate(BaseModel):
    """One bounded test/example candidate retrieved before investigation."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., pattern=r"^TEST-CAND-\d+$")
    test_case_id: str = Field(default="", max_length=160)
    test_name: str = Field(default="", max_length=500)
    framework: str = Field(default="", max_length=80)
    path: str = Field(..., min_length=1, max_length=400)
    locator: str = Field(..., min_length=1, max_length=500)
    label: str = Field(default="", max_length=500)
    intent_ids: list[str] = Field(default_factory=list, max_length=10)
    claim_ids: list[str] = Field(default_factory=list, max_length=10)
    retrieval_ranks: dict[str, int] = Field(default_factory=dict)
    rrf_score: float = 0.0
    snippet: str = Field(default="", max_length=5000)


class IntentTestAnchorGroup(BaseModel):
    """Recall candidates produced for one atomic Issue intent."""

    model_config = ConfigDict(extra="forbid")

    intent_id: str = Field(..., pattern=r"^INTENT-\d+$")
    role: Literal["target", "preserve"]
    query: str = Field(default="", max_length=1600)
    candidate_ids: list[str] = Field(default_factory=list, max_length=5)
    limitation: str = Field(default="", max_length=600)


class TestAnchorRetrieval(BaseModel):
    """Auditable result; current retrieval emits Recall@5 candidates."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["candidates_found", "no_candidates"] = "no_candidates"
    query: str = Field(default="", max_length=1600)
    search_terms: list[str] = Field(default_factory=list, max_length=24)
    candidates: list[TestAnchorCandidate] = Field(
        default_factory=list,
        description="At most five deduplicated recall candidates per atomic intent.",
    )
    intent_groups: list[IntentTestAnchorGroup] = Field(
        default_factory=list,
    )
    limitation: str = Field(default="", max_length=600)


class TestAnchorAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(..., pattern=r"^TEST-CAND-\d+$")
    suitable: bool
    reason: str = Field(..., min_length=1, max_length=500)


class IntentAnchorSelectionGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent_id: str = Field(..., pattern=r"^INTENT-\d+$")
    selected_candidate_ids: list[str] = Field(default_factory=list, max_length=3)
    primary_candidate_id: str = Field(
        default="",
        pattern=r"^(?:TEST-CAND-\d+)?$",
    )
    rationale: str = Field(..., min_length=1, max_length=600)

    @model_validator(mode="after")
    def validate_group(self) -> "IntentAnchorSelectionGroup":
        self.selected_candidate_ids = list(dict.fromkeys(
            self.selected_candidate_ids
        ))
        if self.selected_candidate_ids:
            if self.primary_candidate_id not in self.selected_candidate_ids:
                raise ValueError("group primary candidate must be selected")
        elif self.primary_candidate_id:
            raise ValueError("empty group cannot define a primary candidate")
        return self


class TestAnchorSelection(BaseModel):
    """LLM shortlist for baseline validation by the Investigator."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["selected", "no_suitable_candidate"]
    selected_candidate_ids: list[str] = Field(default_factory=list, max_length=3)
    primary_candidate_id: str = Field(
        default="",
        pattern=r"^(?:TEST-CAND-\d+)?$",
    )
    rationale: str = Field(..., min_length=1, max_length=1600)
    assessments: list[TestAnchorAssessment] = Field(
        default_factory=list,
        description="One assessment for each recalled candidate.",
    )
    intent_selections: list[IntentAnchorSelectionGroup] = Field(
        default_factory=list,
    )

    @model_validator(mode="after")
    def validate_selection(self) -> "TestAnchorSelection":
        selected = list(dict.fromkeys(self.selected_candidate_ids))
        if self.status == "selected":
            if not selected:
                raise ValueError("selected status requires at least one candidate")
            if self.primary_candidate_id not in selected:
                raise ValueError("primary candidate must be in the selected shortlist")
        elif selected or self.primary_candidate_id:
            raise ValueError(
                "no_suitable_candidate cannot contain a selected candidate"
            )
        self.selected_candidate_ids = selected
        return self


class ContrastiveWitnessProposal(BaseModel):
    """A passing repository test paired with its Issue-guided mutation."""

    model_config = ConfigDict(extra="forbid")

    issue_claim_ids: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
    )
    issue_intent_ids: list[str] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "Atomic INTENT-N obligations actually exercised by this pair. "
            "Omitting an intent leaves it unresolved."
        ),
    )
    anchor_candidate_id: str = Field(
        default="",
        pattern=r"^(?:TEST-CAND-\d+)?$",
        description=(
            "Pre-retrieved candidate used as the anchor, or empty when the "
            "shortlist was unavailable and the Investigator found it directly."
        ),
    )
    anchor_test_path: str = Field(..., min_length=1, max_length=400)
    anchor_test_locator: str = Field(..., min_length=1, max_length=400)
    anchor_source_evidence_id: str = Field(..., pattern=r"^SEARCH-\d+$")
    considered_anchor_evidence_ids: list[str] = Field(
        default_factory=list,
        max_length=5,
        description=(
            "Other SEARCH-N focal or analogous references compared before "
            "selecting the anchor. Empty only when no credible alternative was found."
        ),
    )
    anchor_selection_rationale: str = Field(
        ...,
        min_length=1,
        max_length=INVESTIGATION_REPORT_SUMMARY_MAX_CHARS,
        description=(
            "Why the selected anchor is closest in public setup, operation, and "
            "oracle, including how other considered references differ."
        ),
    )
    anchor: ContrastiveProbeCandidate
    target: ContrastiveProbeCandidate
    mutation_summary: str = Field(..., min_length=1, max_length=600)
    preserved_context: list[str] = Field(
        default_factory=list,
        max_length=5,
        description=(
            "Observable setup, API usage, and assertions intentionally kept "
            "identical between the anchor and Issue-mutated test."
        ),
    )
    observed_trace_delta: str = Field(
        ...,
        min_length=1,
        max_length=INVESTIGATION_WITNESS_ANALYSIS_MAX_CHARS,
    )
    discrimination_analysis: str = Field(
        ...,
        min_length=1,
        max_length=INVESTIGATION_WITNESS_ANALYSIS_MAX_CHARS,
        description=(
            "Whether a materially narrower or Issue-incompatible behavior could "
            "also make TARGET-1 pass, and which public decisions remain unresolved."
        ),
    )
    limitations: list[str] = Field(default_factory=list, max_length=5)

    @model_validator(mode="after")
    def validate_candidate_bindings(self) -> "ContrastiveWitnessProposal":
        if self.anchor.id != "ANCHOR-1":
            raise ValueError("the repository passing test must use ANCHOR-1")
        if self.target.id != "TARGET-1":
            raise ValueError("the Issue-mutated test must use TARGET-1")
        if self.anchor.run_evidence_id == self.target.run_evidence_id:
            raise ValueError(
                "anchor and target must cite independent RUN-N executions"
            )
        return self


class RepositoryContractCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(..., pattern=r"^SEARCH-\d+$")
    quote: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description=(
            "Exact quote copied from the cited repository observation."
        ),
    )


class RepositoryContractEffect(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: RepositoryContractEffectKind
    statement: str = Field(..., min_length=1, max_length=600)
    issue_intent_ids: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Atomic Issue intents clarified or bounded by this effect.",
    )
    evidence: list[RepositoryContractCitation] = Field(
        ...,
        min_length=1,
        max_length=3,
        description=(
            "Direct public-contract evidence supporting this one effect."
        ),
    )


class ContrastiveFollowupRequest(BaseModel):
    """One evidence objective requested by the Intent Adjudicator."""

    model_config = ConfigDict(extra="forbid")

    claim_ids: list[str] = Field(default_factory=list, max_length=10)
    objective: str = Field(
        ...,
        min_length=1,
        max_length=700,
        description=(
            "The observable intent question that another focused repository "
            "investigation could answer."
        ),
    )
    why_current_evidence_is_insufficient: str = Field(
        ...,
        min_length=1,
        max_length=700,
    )
    evidence_needed: str = Field(
        ...,
        min_length=1,
        max_length=700,
        description=(
            "A discriminating public observation, repository contract, or "
            "same-claim target oracle to seek next."
        ),
    )
    must_preserve: list[str] = Field(
        default_factory=list,
        max_length=5,
        description=(
            "Issue-authorized setup, operation, oracle, or claim boundaries "
            "that the follow-up must not change."
        ),
    )
    forbidden_inference: str = Field(
        default="",
        max_length=500,
        description=(
            "A product-policy or patch-scope conclusion that the requested "
            "evidence is not allowed to invent."
        ),
    )


class EvidenceSufficiencyFollowup(BaseModel):
    """One repository fact whose absence can still change a repair branch."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1, max_length=900)
    why_material: str = Field(..., min_length=1, max_length=700)
    known_anchor: str = Field(
        default="",
        max_length=500,
        description=(
            "An existing symbol, path, source location, or evidence ID from which "
            "Scout can continue. This is navigation context, not a tool command."
        ),
    )


class EvidenceSufficiencyReview(BaseModel):
    """Independent semantic review of a candidate Scout evidence handoff."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["ready", "continue"]
    reason: str = Field(..., min_length=1, max_length=5000)
    preserved_alternatives: list[str] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "Evidence-supported causal, scope, or public-policy alternatives that "
            "must remain visible to downstream Hypothesis generation."
        ),
    )
    next_investigation: EvidenceSufficiencyFollowup | None = None

    @model_validator(mode="after")
    def require_followup_for_continuation(self) -> "EvidenceSufficiencyReview":
        if self.decision == "continue" and self.next_investigation is None:
            raise ValueError("continue requires one focused next investigation")
        return self


class ContrastiveIntentDecision(BaseModel):
    """Single LLM decision over repository and test-derived intent evidence."""

    model_config = ConfigDict(extra="forbid")

    witness_valid: bool = Field(
        description=(
            "Whether the anchor/target pair is a controlled, Issue-related "
            "behavioral witness for at least one cited claim."
        ),
    )
    boundary_resolved: bool = Field(
        default=False,
        description=(
            "Whether the evidence also resolves every public-behavior decision "
            "needed by supported_delta. A valid local witness can leave this false."
        ),
    )
    next_action: ContrastiveNextAction = Field(
        default="leave_open",
        description=(
            "What the evidence pipeline should do next. Request another "
            "investigation only when the checked-out repository can plausibly "
            "supply evidence without inventing product policy."
        ),
    )
    followup_request: ContrastiveFollowupRequest | None = Field(
        default=None,
        description=(
            "Focused evidence request required by refine_test_pair or "
            "seek_repository_contract."
        ),
    )
    selected_anchor_id: str = Field(
        default="",
        pattern=r"^(?:ANCHOR-1)?$",
    )
    selected_target_id: str = Field(
        default="",
        pattern=r"^(?:TARGET-1)?$",
    )
    witnessed_delta: str = Field(
        ...,
        max_length=800,
        description=(
            "The narrow observable difference established by this exact "
            "anchor/target execution pair. This is a local lower-bound witness, "
            "not a complete repair contract or API-scope decision."
        ),
    )
    supported_delta: str = Field(
        default="",
        max_length=800,
        description=(
            "Observable test-derived behavior change supported by the evidence. "
            "Must be empty unless boundary_resolved is true."
        ),
    )
    covered_issue_claim_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Issue claims actually exercised by the accepted witness. A "
            "passing witness is not evidence about claims omitted here."
        ),
    )
    covered_issue_intent_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Atomic Issue intents actually exercised by the accepted witness."
        ),
    )
    unresolved_issue_claim_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Issue claims not settled by this one anchor/target execution "
            "pair, including claims the proposal failed to exercise."
        ),
    )
    unresolved_issue_intent_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Atomic Issue intents not settled by this local witness."
        ),
    )
    acceptance_authority: AcceptanceAuthority = Field(
        default="illustrative",
        description=(
            "Use normative only when the Issue or an explicit public contract "
            "requires the probe's exact oracle. Use illustrative when literals, "
            "representations, wording, or one selected public path are examples."
        ),
    )
    repository_effects: list[RepositoryContractEffect] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "Independently admissible repository-backed target qualifiers, "
            "preservation boundaries, compatibility risks, and open decisions. "
            "Every item carries its own exact SEARCH-N citation."
        ),
    )
    scope_evidence: list[str] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "Observed execution boundaries useful for localization. These are "
            "diagnostic evidence, not mandatory patch locations."
        ),
    )
    uncertainty: list[str] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "Observable choices exposed by an otherwise valid test witness. "
            "Repository-only open decisions belong in repository_effects."
        ),
    )
    reason: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_decision_shape(self) -> "ContrastiveIntentDecision":
        if self.boundary_resolved and not self.witness_valid:
            raise ValueError("a resolved boundary requires a valid witness")
        if self.witness_valid and (
            not self.selected_anchor_id or not self.selected_target_id
        ):
            raise ValueError(
                "a valid test pair requires selected anchor and target IDs"
            )
        if self.witness_valid and not self.covered_issue_claim_ids:
            raise ValueError(
                "a valid test pair must cover at least one cited Issue claim"
            )
        if self.witness_valid and not self.witnessed_delta.strip():
            raise ValueError(
                "a valid test pair requires its local witnessed behavior delta"
            )
        if self.boundary_resolved and not self.supported_delta.strip():
            raise ValueError(
                "a resolved boundary requires a supported behavior delta"
            )
        if not self.boundary_resolved and self.supported_delta:
            raise ValueError(
                "an unresolved boundary cannot define a complete behavior delta"
            )
        if self.next_action == "accept_boundary" and not self.boundary_resolved:
            raise ValueError("accept_boundary requires a resolved boundary")
        if self.boundary_resolved and self.next_action in {
            "refine_test_pair",
            "seek_repository_contract",
        }:
            raise ValueError("a resolved boundary cannot request more evidence")
        if self.next_action in {
            "refine_test_pair",
            "seek_repository_contract",
        } and self.followup_request is None:
            raise ValueError(
                "an evidence follow-up action requires followup_request"
            )
        if self.next_action == "refine_test_pair" and not self.witness_valid:
            raise ValueError("test-pair refinement requires a valid local witness")
        if not self.witness_valid and (
            self.selected_anchor_id
            or self.selected_target_id
            or self.boundary_resolved
            or self.witnessed_delta
            or self.supported_delta
            or self.covered_issue_claim_ids
            or self.covered_issue_intent_ids
            or self.unresolved_issue_claim_ids
            or self.unresolved_issue_intent_ids
            or self.scope_evidence
            or self.uncertainty
        ):
            raise ValueError(
                "an invalid test pair cannot select witness-derived intent content"
            )
        return self


class ContrastiveAttempt(BaseModel):
    """Auditable proposal, execution, and adjudication for one witness attempt."""

    model_config = ConfigDict(extra="forbid")

    attempt: int = Field(..., ge=1)
    pair_status: str
    witness: ContrastiveWitnessProposal | None = None
    executions: list[dict] = Field(default_factory=list, max_length=2)
    decision: ContrastiveIntentDecision


class IntentBoundaryTestAssessmentItem(BaseModel):
    """One LLM judgment over an executed boundary-test candidate."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(..., pattern=r"^CHECK-\d+$")
    run_evidence_id: str = Field(..., pattern=r"^RUN-\d+$")
    kind: Literal["f2p", "p2p"]
    normative: bool = Field(
        ...,
        description=(
            "Whether this exact oracle must hold under every behavior "
            "interpretation still compatible with the Issue and cited "
            "repository contract."
        ),
    )
    discriminating: bool = Field(
        ...,
        description=(
            "Whether the exact candidate rules out a plausible narrower or "
            "incompatible repair instead of merely demonstrating one local "
            "witness."
        ),
    )
    scope_resolved: bool = Field(
        ...,
        description=(
            "Whether the public receiver, API family, default/opt-in choice, "
            "signature, and state policy encoded by this exact check are "
            "settled by explicit Issue text or independent public-contract "
            "evidence. A local witness alone cannot settle these choices."
        ),
    )
    evidence_ids: list[str] = Field(..., min_length=1, max_length=10)
    covered_issue_claim_ids: list[str] = Field(default_factory=list, max_length=10)
    covered_issue_intent_ids: list[str] = Field(default_factory=list, max_length=10)
    covered_property_ids: list[str] = Field(default_factory=list, max_length=10)
    covered_route_ids: list[str] = Field(default_factory=list, max_length=10)
    surviving_wrong_repair: str = Field(default="", max_length=600)
    missing_discriminator: str = Field(default="", max_length=600)
    reason: str = Field(..., min_length=1, max_length=1000)


class IntentBoundaryTestAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_consistent: bool = Field(
        ...,
        description=(
            "Whether the candidate requirements avoid selecting an open "
            "policy choice or contradicting another supplied contract clause."
        ),
    )
    boundary_complete: bool = Field(
        ...,
        description=(
            "Whether the submitted candidate set covers the evidence-backed "
            "behavior dimensions needed to distinguish completion of the "
            "Issue, not every imaginable input."
        ),
    )
    evidence_ids: list[str] = Field(..., min_length=1, max_length=10)
    diagnostic_property_ids: list[str] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "PROP-N predicates that themselves select an unresolved choice or "
            "claim a scope not established by their cited authority. They "
            "remain in the persisted extraction artifact but are not Repair "
            "requirements."
        ),
    )
    missing_discriminators: list[str] = Field(default_factory=list, max_length=4)
    reason: str = Field(..., min_length=1, max_length=1800)
    checks: list[IntentBoundaryTestAssessmentItem] = Field(
        default_factory=list,
        max_length=4,
    )


class IntentContractClauseDecision(BaseModel):
    """Final authority decision for one repository-derived contract clause."""

    model_config = ConfigDict(extra="forbid")

    clause_id: str = Field(
        ...,
        pattern=r"^(?:PROP|ACCEPT|DELTA)-\d+$",
    )
    disposition: Literal["binding", "diagnostic", "reject"]
    evidence_ids: list[str] = Field(default_factory=list, max_length=10)
    reason: str = Field(..., min_length=1, max_length=800)


class IntentContractSemanticAudit(BaseModel):
    """One final LLM audit after every evidence channel has been merged."""

    model_config = ConfigDict(extra="forbid")

    consistent: bool
    decisions: list[IntentContractClauseDecision] = Field(
        default_factory=list,
        max_length=30,
    )
    summary: str = Field(..., min_length=1, max_length=1200)

    @model_validator(mode="after")
    def validate_unique_clause_ids(self) -> "IntentContractSemanticAudit":
        clause_ids = [item.clause_id for item in self.decisions]
        if len(clause_ids) != len(set(clause_ids)):
            raise ValueError("semantic audit clause IDs must be unique")
        return self


class InvestigationTaskFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(
        ...,
        pattern=r"^(?:TASK|DIM|TODO|OBJ)-\d+$",
        description="Exact task ID from the supplied investigation plan.",
    )
    status: AnswerStatus = Field(
        default="unresolved",
        description="Whether repository evidence answers this TODO.",
    )
    finding: str = Field(
        ...,
        min_length=1,
        max_length=600,
        description=(
            "Concise repository-grounded result for this TODO; state the "
            "evidence limitation when unresolved."
        ),
    )
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="Direct SEARCH-N, RUN-N, or GRAPH-N support for this finding.",
    )
    evidence_effects: list[HypothesisEvidenceEffect] = Field(
        default_factory=list,
        max_length=4,
        description=(
            "Legacy field. Leave empty in new runs; the separate Evidence "
            "Adjudicator decides whether observations disambiguate hypotheses."
        ),
    )


class EliminatedHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str
    evidence_id: str
    authority: EvidenceAuthority | None = None
    reason: str


class IntentDimensionResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension_id: str
    question: str
    status: VersionSpaceStatus
    initial_hypothesis_ids: list[str] = Field(default_factory=list)
    remaining_hypothesis_ids: list[str] = Field(default_factory=list)
    eliminated: list[EliminatedHypothesis] = Field(default_factory=list)
    resolved_qualifiers: list[str] = Field(default_factory=list)
    rejected_effects: list[str] = Field(default_factory=list)


class AdjudicationEvidenceCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(..., pattern=r"^(?:SEARCH|RUN|GRAPH)-\d+$")
    quote: str = Field(..., min_length=1, max_length=500)


class IntentDimensionAdjudication(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension_id: str = Field(..., pattern=r"^DIM-\d+$")
    disambiguated: bool
    selected_hypothesis_id: str = Field(default="", max_length=40)
    evidence: list[AdjudicationEvidenceCitation] = Field(
        default_factory=list,
        max_length=4,
    )
    reason: str = Field(..., min_length=1, max_length=600)

    @model_validator(mode="after")
    def validate_binary_decision(self) -> "IntentDimensionAdjudication":
        if self.disambiguated and (
            not self.selected_hypothesis_id or not self.evidence
        ):
            raise ValueError(
                "a disambiguated dimension requires one hypothesis and evidence"
            )
        if not self.disambiguated and self.selected_hypothesis_id:
            raise ValueError(
                "an unresolved dimension cannot select a hypothesis"
            )
        return self


class IntentEvidenceAdjudication(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decisions: list[IntentDimensionAdjudication] = Field(
        default_factory=list,
        max_length=3,
    )


class IntentTaskResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(..., pattern=r"^TASK-\d+$")
    resolved: bool
    resolution_kind: IntentResolutionKind = "no_effect"
    contract_text: str = Field(default="", max_length=600)
    contract_candidate_id: str = Field(
        default="",
        pattern=r"^(?:PRESERVE-CAND-\d+|ACCEPT-CAND-\d+)?$",
    )
    evidence_type: IntentResolutionEvidenceType = "insufficient"
    evidence: list[AdjudicationEvidenceCitation] = Field(
        default_factory=list,
        max_length=4,
    )
    alternatives_considered: list[str] = Field(
        default_factory=list,
        max_length=3,
    )
    reason: str = Field(..., min_length=1, max_length=600)

    @model_validator(mode="after")
    def validate_resolution(self) -> "IntentTaskResolution":
        if not self.resolved:
            if self.resolution_kind != "no_effect":
                raise ValueError("an unresolved task must use no_effect")
            if self.contract_text or self.contract_candidate_id:
                raise ValueError("an unresolved task cannot add a contract effect")
            if self.evidence_type != "insufficient":
                raise ValueError("an unresolved task must use insufficient evidence")
            return self

        if self.resolution_kind == "no_effect":
            raise ValueError("a resolved task requires a contract effect")
        if not self.contract_text.strip() or not self.evidence:
            raise ValueError("a resolved task requires contract text and evidence")
        if self.evidence_type == "insufficient":
            raise ValueError("a resolved task requires a sufficient evidence type")
        candidate_kinds = {"preservation_boundary", "acceptance_check"}
        if (
            self.resolution_kind in candidate_kinds
            and not self.contract_candidate_id
        ):
            raise ValueError(
                "preservation and acceptance effects require a candidate ID"
            )
        if (
            self.resolution_kind == "behavior_qualifier"
            and self.contract_candidate_id
        ):
            raise ValueError("a behavior qualifier cannot select a candidate")
        return self


class IntentEvidenceResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resolutions: list[IntentTaskResolution] = Field(
        default_factory=list,
        max_length=10,
    )


class CandidateRepairBoundary(BaseModel):
    """One evidence-backed possible repair surface, not a required edit."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., pattern=r"^BOUNDARY-\d+$")
    issue_intent_ids: list[str] = Field(default_factory=list, max_length=10)
    anchor_operation: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description=(
            "Public operation anchoring this possible scope. It may be the "
            "reported entry or a source-backed sibling sharing a relevant "
            "type, state, helper, or contract; it need not call the fault site."
        ),
    )
    candidate_public_forms: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description=(
            "Concrete observed public entries and shipped, generated, or "
            "forwarded forms possibly affected by this lead. Keep distinct "
            "causal families separate; a listed form is not yet a required edit."
        ),
    )
    evidence_ids: list[str] = Field(
        ...,
        min_length=1,
        max_length=5,
        description=(
            "SEARCH-N or GRAPH-N observations showing the operation and its "
            "Issue-relevant relation; adjacency alone is not proof of scope."
        ),
    )
    rationale: str = Field(
        ..., min_length=1, max_length=1600,
        description=(
            "Why this surface might share the cause or repair contract, what "
            "source/execution confirms, and which material fact remains open. "
            "Do not claim it is required before that fact is checked."
        ),
    )


class InvestigationPath(BaseModel):
    """A checked or still-open mechanism/scope, not a behavioral requirement."""

    model_config = ConfigDict(extra="forbid")

    path_id: str = Field(default="", pattern=r"^(?:PATH-[0-9]+)?$")
    role: Literal[
        "ingress", "state_producer", "consumer", "failure_exposure",
        "forwarder", "other",
    ] = "other"
    question: str = Field(min_length=1, max_length=500)
    finding: str = Field(min_length=1, max_length=1200,
                         description="Concrete entry, relation and implication; distinguish indexed adjacency from an observed call/state path.")
    status: Literal["supported", "ruled_out", "unresolved"]
    evidence_ids: list[str] = Field(min_length=1, max_length=6)
    verification_kind: Literal["source", "absence", "execution"] = "source"
    source_evidence_id: str = Field(
        default="",
        description="Evidence that verifies the path: SEARCH-N source/negative search or RUN-N execution.",
    )
    source_quote: str = Field(
        default="", max_length=800,
        description="Exact contiguous excerpt for source/execution evidence; empty is valid only for a negative search with no output.",
    )
    missing_evidence: str = Field(default="", max_length=600,
                                  description="Specific remaining fact needed to settle this path; required for unresolved.")


class InvestigationInputPathCheck(BaseModel):
    issue_input_quote: str = Field(
        default="", max_length=800,
        description="Exact contiguous original Issue excerpt identifying its input/receiver, copied without changing whitespace, punctuation, or line prefixes; not a later diagnostic variant."
    )
    actual_input: str = Field(
        min_length=1, max_length=400,
        description="Original public-call input/receiver BEFORE library dispatch: include supplied value representation and requested result type. An internal element or invented probe type is not the Issue input; say unknown for types the Issue does not specify."
    )
    consumer_input: str = Field(
        min_length=1, max_length=400,
        description="Representation and exact types accepted by the selected library consumer; distinguish these downstream operands from the original public input."
    )
    condition_evidence_id: str = Field(min_length=1)
    condition_quote: str = Field(
        min_length=1, max_length=800,
        description="Exact contiguous excerpt copied from condition_evidence_id without changing whitespace, punctuation, or line prefixes; it must show the consumer's participation predicate or input guard, not just the failing assignment or a function declaration."
    )
    evaluation: str = Field(min_length=1, max_length=800)
    verification_run_id: str = Field(
        default="", description="Focused RUN directly testing this premise, not merely reproducing the original error; empty means source reasoning only."
    )
    verification_quote: str = Field(
        default="", max_length=800, description="Filled by the system from the cited RUN; leave empty rather than reconstructing compiler output."
    )
    matches_input: Literal["yes", "no", "unknown"]
    alternative_path: str = Field(
        max_length=800,
        description="Competing consumer for the SAME original public input and target type, with its gate or missing evidence. A passing type-mutated diagnostic is not such a competing consumer."
    )


class InvestigationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: AnswerStatus = "unresolved"
    answer: str = Field(
        ...,
        max_length=INVESTIGATION_REPORT_SUMMARY_MAX_CHARS,
        description=(
            "Final structured summary of what the current repository "
            "establishes. Do not copy provider reasoning_content or a search "
            "narrative here; put evidence details in their dedicated fields. "
            "This is descriptive evidence, not a rewritten requirement."
        ),
    )
    repository_facts: list[str] = Field(default_factory=list, max_length=8)
    investigation_paths: list[InvestigationPath] = Field(default_factory=list, max_length=8)
    input_path_check: InvestigationInputPathCheck | None = None
    preserve_behavior: list[str] = Field(default_factory=list, max_length=2)
    preserve_evidence_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Direct SEARCH-N or RUN-N support shared by preserve_behavior "
            "items. Required for preservation handoff in version-space mode."
        ),
    )
    acceptance_checks: list[InvestigationAcceptanceCheck] = Field(
        default_factory=list,
        max_length=4,
    )
    conflicts_with_issue: list[str] = Field(default_factory=list, max_length=2)
    open_decisions: list[str] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "Repository-unresolved intent decisions retained for audit. They "
            "do not become Repair requirements or instructions."
        ),
    )
    repository_evidence_ids: list[str] = Field(default_factory=list)
    task_findings: list[InvestigationTaskFinding] = Field(
        default_factory=list,
        max_length=10,
    )
    candidate_boundaries: list[CandidateRepairBoundary] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "Evidence-backed suspicious repair surfaces, including related "
            "public operations or shipped forms not yet proven necessary. "
            "Preserve material leads for Hypothesis; they are not requirements."
        ),
    )
    contrastive_witness: ContrastiveWitnessProposal | None = Field(
        default=None,
        description=(
            "Optional existing passing anchor test paired with its minimal "
            "Issue-guided failing mutation. A separate LLM adjudicates its "
            "local witnessed delta independently from repository contract "
            "evidence."
        ),
    )

    @model_validator(mode="after")
    def validate_answer_state(self) -> "InvestigationReport":
        if self.status == "supported" and (
            self.conflicts_with_issue or self.open_decisions
        ):
            raise ValueError(
                "supported answers cannot contain conflicts or open decisions"
            )
        return self


DeveloperAcceptanceCheck = InvestigationAcceptanceCheck
DeveloperAnswer = InvestigationReport


class FocusedInvestigationReport(BaseModel):
    """One Scout report and the atomic Issue intents it investigated."""

    model_config = ConfigDict(extra="forbid")

    focus_intent_ids: list[str] = Field(default_factory=list, max_length=10)
    report: InvestigationReport


class DialogueExchange(BaseModel):
    objective: IntentExplorationObjective
    repository_evidence: list[RepositoryEvidence] = Field(default_factory=list)
    answer: DeveloperAnswer
    developer_actions: list[dict] = Field(default_factory=list)


class RepositoryInvestigation(BaseModel):
    plan: IntentInvestigationPlan
    anchor_retrieval: TestAnchorRetrieval | None = None
    anchor_selection: TestAnchorSelection | None = None
    repository_evidence: list[RepositoryEvidence] = Field(default_factory=list)
    report: InvestigationReport
    report_history: list[FocusedInvestigationReport] = Field(
        default_factory=list,
        max_length=12,
        description=(
            "Scout reports retained across independently investigated atomic "
            "intents. `report` remains the latest report for compatibility."
        ),
    )
    actions: list[dict] = Field(default_factory=list)


class RepairIntentLedgerEntry(BaseModel):
    """Persistent per-intent evidence state; Issue text remains authoritative."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., pattern=r"^INTENT-\d+$")
    role: Literal["target", "preserve"]
    text: str = Field(..., min_length=1, max_length=800)
    issue_span: str = Field(..., min_length=1, max_length=1200)
    source_claim_ids: list[str] = Field(default_factory=list, max_length=10)
    status: IntentLedgerStatus = "uninvestigated"
    evidence_ids: list[str] = Field(default_factory=list, max_length=10)
    evidence_notes: list[str] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "Bounded repository observations copied into the Repair handoff "
            "so evidence IDs remain useful outside the investigation trace."
        ),
    )
    witness_ids: list[str] = Field(default_factory=list, max_length=5)
    witnessed_delta: str = Field(
        default="",
        max_length=800,
        description=(
            "Narrow observable difference established by a test pair bound "
            "to this atomic intent. It remains local evidence unless the "
            "same pair also resolves the behavior boundary."
        ),
    )
    supported_delta: str = Field(
        default="",
        max_length=800,
        description=(
            "Observable behavior boundary established for this atomic intent "
            "by Issue-authorized and repository-grounded evidence."
        ),
    )
    limitations: list[str] = Field(default_factory=list, max_length=5)


class RepairIntentContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent_ledger: list[RepairIntentLedgerEntry] = Field(
        default_factory=list,
    )
    candidate_boundaries: list[CandidateRepairBoundary] = Field(
        default_factory=list,
        max_length=10,
    )
    target_behavior: list[str] = Field(default_factory=list)
    preserve_behavior: list[str] = Field(default_factory=list)
    properties: list[IntentBehaviorProperty] = Field(
        default_factory=list,
        max_length=10,
    )
    validation_routes: list[IntentValidationRoute] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "Validation allocation metadata. Executable routes are consumed "
            "by boundary-test materialization; structural routes are leads "
            "for Patch Reviewer, never repair requirements."
        ),
    )
    acceptance_checks: list[FrozenAcceptanceCheck] = Field(
        default_factory=list,
        max_length=10,
    )
    compatibility_warnings: list[str] = Field(default_factory=list, max_length=3)
    open_decisions: list[str] = Field(default_factory=list, max_length=10)
    local_behavior_witnesses: list[str] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "Validated local behavior differences from contrastive tests. "
            "They are lower-bound evidence and never completion requirements."
        ),
    )
    behavioral_delta: list[str] = Field(default_factory=list, max_length=10)
    scope_evidence: list[str] = Field(default_factory=list, max_length=3)
    witness_limitations: list[str] = Field(default_factory=list)
    validation_gaps: list[str] = Field(
        default_factory=list,
        max_length=4,
        description=(
            "Semantic-admission findings that explain why passing the current "
            "executable checks does not by itself prove the complete repair "
            "boundary."
        ),
    )

    def render_for_review(self) -> str:
        """Render scope evidence that is useful after a candidate patch exists."""
        if not (
            self.intent_ledger
            or self.target_behavior
            or self.preserve_behavior
            or self.candidate_boundaries
            or self.scope_evidence
            or self.compatibility_warnings
            or self.open_decisions
            or self.validation_gaps
        ):
            return ""

        lines = [
            "<repair_intent_review_context>",
            "The original Issue remains authoritative. Use the following as "
            "post-patch scope and consistency evidence, not as permission to "
            "redesign the requested behavior.",
        ]
        if self.intent_ledger:
            lines.append("Protected atomic Issue intents:")
            lines.extend(
                f"- [{item.id}] [{item.role.upper()}] "
                f"[{item.status.upper()}] {item.text}"
                for item in self.intent_ledger
            )
        else:
            if self.target_behavior:
                lines.append("Requested target behavior:")
                lines.extend(f"- {item}" for item in self.target_behavior)
            if self.preserve_behavior:
                lines.append("Explicit preservation behavior:")
                lines.extend(f"- {item}" for item in self.preserve_behavior)
        if self.candidate_boundaries:
            lines.append(
                "Repository-supported surface leads. These are conservative "
                "review candidates, not requirements by themselves:"
            )
            for boundary in self.candidate_boundaries:
                binding = (
                    f" [{','.join(boundary.issue_intent_ids)}]"
                    if boundary.issue_intent_ids
                    else ""
                )
                lines.append(
                    f"- [{boundary.id}]{binding} Closest established operation: "
                    f"{boundary.anchor_operation}. Candidate public forms: "
                    f"{'; '.join(boundary.candidate_public_forms)}. "
                    f"Repository rationale: {boundary.rationale}"
                )
        if self.scope_evidence:
            lines.append("Observed localization and implementation-surface evidence:")
            lines.extend(f"- {item}" for item in self.scope_evidence)
        if self.validation_gaps:
            lines.append("Unverified semantic boundaries:")
            lines.extend(f"- {item}" for item in self.validation_gaps)
        if self.open_decisions:
            lines.append("Repository-unresolved product decisions:")
            lines.extend(f"- {item}" for item in self.open_decisions)
        if self.compatibility_warnings:
            lines.append("Compatibility evidence:")
            lines.extend(f"- {item}" for item in self.compatibility_warnings)
        lines.extend([
            "Require a missing surface only when the candidate patch and direct "
            "repository evidence establish a concrete public inconsistency.",
            "</repair_intent_review_context>",
        ])
        return "\n".join(lines)

    def verification_spec(self) -> dict:
        target_ids = [
            f"TARGET-{index}"
            for index in range(1, len(self.target_behavior) + 1)
        ]
        preserve_ids = [
            f"PRESERVE-{index}"
            for index in range(1, len(self.preserve_behavior) + 1)
        ]
        witness_ids = [
            f"ACCEPT-{index}"
            for index in range(1, len(self.acceptance_checks) + 1)
        ]
        acceptance_ids = [
            check_id
            for check_id, check in zip(
                witness_ids,
                self.acceptance_checks,
                strict=True,
            )
            if check.authority == "normative"
        ]
        checks = dict(
            zip(target_ids, self.target_behavior, strict=True)
        )
        checks.update(
            zip(preserve_ids, self.preserve_behavior, strict=True)
        )
        checks.update({
            check_id: check.description
            for check_id, check in zip(
                witness_ids,
                self.acceptance_checks,
                strict=True,
            )
        })
        return {
            "checks": checks,
            "target_ids": target_ids,
            "preserve_ids": preserve_ids,
            "acceptance_ids": acceptance_ids,
            "diagnostic_ids": [
                check_id
                for check_id, check in zip(
                    witness_ids,
                    self.acceptance_checks,
                    strict=True,
                )
                if check.authority == "illustrative"
            ],
            "probes": {
                check_id: check.model_dump(
                    exclude={
                        "candidate_id",
                        "covered_issue_claim_ids",
                        "covered_issue_intent_ids",
                        "covered_property_ids",
                        "covered_route_ids",
                    }
                )
                for check_id, check in zip(
                    witness_ids,
                    self.acceptance_checks,
                    strict=True,
                )
            },
        }


class IntentDialogueResult(BaseModel):
    issue_evidence: IssueEvidenceFrame = Field(
        default_factory=IssueEvidenceFrame
    )
    intent_contract: RepairIntentContract = Field(
        default_factory=RepairIntentContract
    )
    exploration_plan: IntentExplorationPlan = Field(
        default_factory=IntentExplorationPlan
    )
    investigation: RepositoryInvestigation | None = None
    exchanges: list[DialogueExchange] = Field(default_factory=list)
    synthesis: IntentSynthesisSelection | None = None
    resolution: IntentEvidenceResolution | None = None
    adjudication: IntentEvidenceAdjudication | None = None
    contrastive_decision: ContrastiveIntentDecision | None = None
    contrastive_attempts: list[ContrastiveAttempt] = Field(
        default_factory=list,
        max_length=11,
    )
    evidence_sufficiency_review: EvidenceSufficiencyReview | None = None
    evidence_sufficiency_followup_performed: bool = False
    hypothesis_ready: bool | None = Field(
        default=None,
        description=(
            "The independent evidence-sufficiency verdict. None means that this "
            "dialogue path did not run the reviewer."
        ),
    )
    property_extraction: IntentPropertyExtraction | None = None
    validation_routing: IntentValidationRouting | None = None
    boundary_test_report: InvestigationReport | None = None
    boundary_test_assessment: IntentBoundaryTestAssessment | None = None
    contract_semantic_audit: IntentContractSemanticAudit | None = None
    version_space: list[IntentDimensionResolution] = Field(default_factory=list)
    pair_status: str = ""
    stop_reason: str = ""

    def render_for_repair(self) -> str:
        contract = self.intent_contract
        if not (
            contract.intent_ledger
            or contract.target_behavior
            or contract.preserve_behavior
            or contract.candidate_boundaries
            or contract.properties
            or contract.acceptance_checks
            or contract.compatibility_warnings
            or contract.open_decisions
            or contract.local_behavior_witnesses
            or contract.behavioral_delta
            or contract.scope_evidence
            or contract.witness_limitations
            or contract.validation_gaps
        ):
            return ""

        if self.contract_semantic_audit is not None:
            lines = [
                "<repair_intent_contract>",
                "The original Issue is authoritative. The items below are the "
                "repository-derived clauses that survived one final semantic "
                "authority audit; diagnostic investigation details are omitted.",
                "Required behavior:",
            ]
            lines.extend(
                f"- [{item.id}] [{item.role.upper()}] "
                f"[{item.status.upper()}] {item.text}"
                + (
                    f" Established boundary: {item.supported_delta}"
                    if item.supported_delta
                    else ""
                )
                for item in contract.intent_ledger
            )
            if contract.properties:
                lines.append("Evidence-supported behavioral boundaries:")
                lines.extend(
                    f"- [{item.id}] [{item.role.upper()}] {item.predicate}"
                    for item in contract.properties
                )
            normative_checks = [
                (index, check)
                for index, check in enumerate(contract.acceptance_checks, 1)
                if check.authority == "normative"
            ]
            if normative_checks:
                lines.append("Required executable checks:")
                lines.extend(
                    f"- [ACCEPT-{index}] [{check.kind.upper()}] "
                    f"{check.description}"
                    for index, check in normative_checks
                )
            if contract.local_behavior_witnesses:
                lines.append(
                    "Localization witnesses (useful evidence, not completion "
                    "requirements):"
                )
                lines.extend(
                    f"- {item}"
                    for item in contract.local_behavior_witnesses[:3]
                )
            unresolved = [
                item
                for item in contract.intent_ledger
                if item.status in {"uninvestigated", "local_witness", "unresolved"}
            ]
            if unresolved:
                lines.append(
                    "Unresolved Issue dimensions remain in scope; do not infer "
                    "their policy from the current implementation:"
                )
                lines.extend(f"- [{item.id}] {item.text}" for item in unresolved)
            lines.extend([
                "After the latest production edit, run every required ACCEPT-N "
                "with RunIntentCheck and run regression verification.",
                "</repair_intent_contract>",
            ])
            return "\n".join(lines)

        lines = [
            "<repair_intent_contract>",
            "Authority: The original issue defines the requested behavior. "
            "Contrastive tests provide local change witnesses. Repository "
            "evidence may add preservation boundaries, and property-based or "
            "fixed checks may validate supported dimensions, "
            "but must not weaken, replace, narrow, or expand an explicit request.",
        ]
        if contract.intent_ledger:
            lines.append("Atomic Issue Intent Ledger:")
            for item in contract.intent_ledger:
                line = (
                    f"- [{item.id}] [{item.role.upper()}] "
                    f"[{item.status.upper()}] {item.text}"
                )
                if item.supported_delta:
                    line += f" Established boundary: {item.supported_delta}"
                if item.limitations:
                    line += f" Limitations: {'; '.join(item.limitations)}"
                lines.append(line)
            lines.append(
                "A witness or check settles only the INTENT-N entries explicitly "
                "bound to it. Uninvestigated or unresolved entries remain part of "
                "the original requested behavior and must not be treated as out "
                "of scope."
            )
        if contract.local_behavior_witnesses or contract.scope_evidence:
            lines.append(
                "Immediate repair brief (reuse before doing any new localization):"
            )
            if contract.local_behavior_witnesses:
                lines.append(
                    "Validated local behavior witness (minimum evidence, not "
                    "the complete repair scope or a completion condition):"
                )
                lines.extend(
                    f"- {item}" for item in contract.local_behavior_witnesses
                )
            if contract.scope_evidence and not contract.local_behavior_witnesses:
                lines.append(
                    "Observed execution boundaries (localization evidence, not "
                    "required patch locations):"
                )
                lines.extend(f"- {item}" for item in contract.scope_evidence)
            lines.append(
                "View the cited implementation and closest public analogue, then "
                "form and test the smallest Issue-consistent candidate. Do not "
                "repeat anchor discovery, recreate an executed witness, or map "
                "the full wrapper/generated surface before the first candidate."
            )
        ledger_text = {item.text for item in contract.intent_ledger}
        sections = (
            (
                (
                    "Additional target qualifiers:"
                    if contract.intent_ledger
                    else "Target behavior:"
                ),
                "TARGET",
                [item for item in contract.target_behavior if item not in ledger_text],
            ),
            (
                (
                    "Additional preservation boundaries:"
                    if contract.intent_ledger
                    else "Preserve behavior:"
                ),
                "PRESERVE",
                [
                    item
                    for item in contract.preserve_behavior
                    if item not in ledger_text
                ],
            ),
        )
        for title, prefix, items in sections:
            if items:
                lines.append(title)
                lines.extend(
                    f"- [{prefix}-{index}] {item}"
                    for index, item in enumerate(items, 1)
                )
        if contract.properties:
            lines.append("Observable behavior properties:")
            for item in contract.properties:
                scope = f" Scope: {', '.join(item.scope)}." if item.scope else ""
                lines.append(
                    f"- [{item.id}] [{item.role.upper()}] {item.predicate}{scope}"
                )
        normative_checks = [
            (index, check)
            for index, check in enumerate(contract.acceptance_checks, 1)
            if check.authority == "normative"
        ]
        if normative_checks:
            lines.append("Required executable acceptance checks:")
            lines.extend(
                f"- [ACCEPT-{index}]"
                + (
                    f" [{check.kind.upper()}]"
                    if check.kind in {"f2p", "p2p"}
                    else ""
                )
                + f" {check.description}"
                for index, check in normative_checks
            )
            lines.append(
                "These checks supplement the Issue target; passing preservation "
                "checks alone never proves that the requested change is complete."
            )
        semantic_candidates_rejected = (
            self.boundary_test_assessment is not None
            and not self.boundary_test_assessment.contract_consistent
        )
        illustrative_checks = [
            (index, check)
            for index, check in enumerate(contract.acceptance_checks, 1)
            if check.authority == "illustrative"
            and not semantic_candidates_rejected
        ]
        if illustrative_checks:
            if contract.behavioral_delta:
                lines.append(
                    "Optional diagnostic witnesses: the semantic behavior delta "
                    "below is resolved, but each concrete probe oracle is only an "
                    "illustration and is not a completion condition."
                )
            else:
                lines.append(
                    "Optional local diagnostic witnesses: the test pair is valid "
                    "but does not resolve the complete public behavior boundary. "
                    "These checks are not completion conditions."
                )
            lines.extend(
                f"- [ACCEPT-{index}] {check.description}"
                for index, check in illustrative_checks
            )
        if contract.behavioral_delta:
            lines.append("Test-derived Change Intent:")
            lines.extend(f"- {item}" for item in contract.behavioral_delta)
        assessed_discriminators = (
            self.boundary_test_assessment.missing_discriminators
            if self.boundary_test_assessment is not None
            and self.boundary_test_assessment.contract_consistent
            else []
        )
        if assessed_discriminators:
            lines.append(
                "Assessor-confirmed unresolved validation limitation "
                f"({len(assessed_discriminators)} diagnostic gap(s), audit "
                "only):"
            )
            lines.append(
                "- Do not turn this gap into a Repair task unless the original "
                "Issue or a cited public contract already supplies the expected "
                "winner. The buggy repository cannot resolve a product-policy "
                "choice by further probing. Assessor fixture details and proposed "
                "oracles are withheld because they are not semantic authority."
            )
        elif contract.validation_gaps:
            lines.append(
                "Some generated validation candidates were not semantically "
                "admitted and are excluded from this Repair brief. Follow the "
                "Issue and admitted checks rather than reconstructing rejected "
                "oracles."
            )
        if contract.witness_limitations:
            lines.append(
                "Issue evidence not settled by the contrastive witness "
                "(do not silently discard or interpret as out of scope):"
            )
            lines.extend(f"- {item}" for item in contract.witness_limitations)
        if contract.open_decisions:
            lines.extend([
                "Open Intent decisions (audit only):",
                "The investigation retained repository-unresolved product choices "
                "for post-patch audit. They are not Repair requirements and must "
                "not delay the first Issue-consistent candidate."
            ])
        lines.extend([
            "Verification: run every required ACCEPT-N oracle once with "
            "RunIntentCheck after the latest production-code edit. Optional "
            "diagnostic ACCEPT-N witnesses may also be run for feedback; their "
            "failure never blocks completion.",
            "</repair_intent_contract>",
        ])
        return "\n".join(lines)
