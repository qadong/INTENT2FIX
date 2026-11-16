from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from INTENT2FIX.reproduce.intent_schema import (
    RepairIntentBundle,
    RepairObligation,
)


EvidenceSource = Literal[
    "ISSUE",
    "TEST",
    "DOC",
    "REPOSITORY",
    "EXECUTION",
    "INFERENCE",
]
EvidenceRole = Literal["NORMATIVE", "DESCRIPTIVE", "PRESERVATION"]
EvidenceRelation = Literal["SUPPORTS", "CONTRADICTS", "IRRELEVANT"]
GuardDecision = Literal["ACCEPT_CURRENT", "REVISE_CURRENT", "UNRESOLVED"]
BlindSelection = Literal["SELECTED", "UNRESOLVED"]


class IntentCandidate(BaseModel):
    summary: str = ""
    behavioral_delta: str = ""
    hard_obligations: List[RepairObligation] = Field(default_factory=list)
    soft_hypotheses: List[str] = Field(default_factory=list)
    rejected_interpretations: List[str] = Field(default_factory=list)
    unresolved: List[str] = Field(default_factory=list)

    @classmethod
    def from_bundle(cls, bundle: RepairIntentBundle) -> "IntentCandidate":
        return cls(
            summary=bundle.summary,
            behavioral_delta=bundle.behavioral_delta,
            hard_obligations=bundle.hard_obligations,
            soft_hypotheses=bundle.soft_hypotheses,
            rejected_interpretations=bundle.rejected_interpretations,
            unresolved=bundle.unresolved,
        )


class EvidenceRequest(BaseModel):
    query: str
    target_symbol: str = ""
    target_file: str = ""
    purpose: Literal["NORMATIVE", "PRESERVATION"] = "PRESERVATION"


class IntentChallenge(BaseModel):
    risk_kind: Literal[
        "NONE",
        "PRESERVATION_OMISSION",
        "SCOPE_OVERGENERALIZATION",
        "SCOPE_UNDERGENERALIZATION",
        "OBSERVED_EXPECTED_CONFUSION",
        "UNSUPPORTED_SEMANTICS",
        "IMPLEMENTATION_AS_REQUIREMENT",
    ] = "NONE"
    challenged_obligation_ids: List[str] = Field(default_factory=list)
    challenger: Optional[IntentCandidate] = None
    observable_difference: str = ""
    evidence_requests: List[EvidenceRequest] = Field(default_factory=list)


class IntentEvidence(BaseModel):
    id: str
    source_type: EvidenceSource
    role: EvidenceRole
    relation: EvidenceRelation = "SUPPORTS"
    provenance: str
    content: str
    claim_ids: List[str] = Field(default_factory=list)


class EvidenceAssessment(BaseModel):
    evidence_id: str
    role: EvidenceRole
    relation: EvidenceRelation
    candidate_ids: List[str] = Field(default_factory=list)
    obligation_ids: List[str] = Field(default_factory=list)


class BlindIntentAdjudication(BaseModel):
    selection: BlindSelection
    selected_candidate_id: Optional[str] = None
    effective_intent: IntentCandidate
    evidence_assessments: List[EvidenceAssessment] = Field(default_factory=list)
    remaining_uncertainty: List[str] = Field(default_factory=list)
    decision_reason: str


class IntentGuardResult(BaseModel):
    execution_status: Literal["OK", "TECHNICAL_FAILURE"]
    decision: GuardDecision
    proposed_intent: RepairIntentBundle
    effective_bundle: RepairIntentBundle
    challenge: Optional[IntentChallenge] = None
    evidence: List[IntentEvidence] = Field(default_factory=list)
    adjudication: Optional[BlindIntentAdjudication] = None
    candidate_mapping: Dict[str, Literal["CURRENT", "CHALLENGER"]] = Field(
        default_factory=dict
    )
    decision_reason: str
