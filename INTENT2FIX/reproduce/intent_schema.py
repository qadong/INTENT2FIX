from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


ClaimKind = Literal[
    "trigger",
    "observed_behavior",
    "expected_behavior",
    "affected_scope",
    "preservation",
    "non_goal",
    "example",
]
IssueIntentRole = Literal["target", "preserve"]
SupportKind = Literal[
    "issue_explicit",
    "repository_supported",
    "execution_confirmed",
    "inferred",
    "unresolved",
    "contradicted",
]


class IssueClaim(BaseModel):
    id: str
    kind: ClaimKind
    text: str
    issue_span: str
    search_terms: List[str] = Field(default_factory=list)


class IssueIntentUnit(BaseModel):
    """One atomic observable obligation stated by the original Issue."""

    id: str = Field(..., pattern=r"^INTENT-\d+$")
    role: IssueIntentRole
    text: str
    issue_span: str
    source_claim_ids: List[str] = Field(default_factory=list)


class IssueEvidenceFrame(BaseModel):
    claims: List[IssueClaim] = Field(default_factory=list)
    intent_units: List[IssueIntentUnit] = Field(default_factory=list)
    symbols: List[str] = Field(default_factory=list)
    error_signatures: List[str] = Field(default_factory=list)
    implementation_suggestions: List[str] = Field(default_factory=list)
    ambiguities: List[str] = Field(default_factory=list)
    extraction_status: Literal["structured", "fallback"] = "structured"


class GraphBinding(BaseModel):
    claim_ids: List[str] = Field(default_factory=list)
    entity: str
    file: str = ""
    relation: Literal[
        "MENTIONS",
        "LIKELY_REALIZES",
        "PRESERVES",
        "VERIFIES",
        "MAY_AFFECT",
    ]
    evidence: str = ""


class ExecutionRecord(BaseModel):
    run_id: str
    command: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""

    @property
    def output(self) -> str:
        return "\n".join(part for part in (self.stdout, self.stderr) if part)


class TestArtifact(BaseModel):
    path: str
    obligation_ids: List[str] = Field(default_factory=list)
    purpose: str = "symptom"


class RepairObligation(BaseModel):
    id: str
    kind: str
    text: str
    support: SupportKind
    provenance: List[str] = Field(default_factory=list)


class RepairIntentBundle(BaseModel):
    status: Literal["STRONG", "WEAK", "UNRESOLVED"] = "UNRESOLVED"
    reproduction_status: Literal["MATCHED", "PARTIAL", "MISMATCHED", "UNRESOLVED"] = "UNRESOLVED"
    summary: str = ""
    behavioral_delta: str = ""
    hard_obligations: List[RepairObligation] = Field(default_factory=list)
    soft_hypotheses: List[str] = Field(default_factory=list)
    rejected_interpretations: List[str] = Field(default_factory=list)
    graph_bindings: List[GraphBinding] = Field(default_factory=list)
    tests: List[TestArtifact] = Field(default_factory=list)
    execution_records: List[ExecutionRecord] = Field(default_factory=list)
    unresolved: List[str] = Field(default_factory=list)

    def compact_dump(self) -> Dict[str, Any]:
        data = self.model_dump(
            include={
                "status",
                "reproduction_status",
                "summary",
                "behavioral_delta",
                "hard_obligations",
                "soft_hypotheses",
                "rejected_interpretations",
                "graph_bindings",
                "tests",
                "unresolved",
            },
            exclude_none=True,
        )
        return {
            key: value
            for key, value in data.items()
            if key in {"status", "reproduction_status"} or value not in ("", [], {})
        }
