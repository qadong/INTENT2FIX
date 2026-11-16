from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SurfaceCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    surface: str = Field(
        ...,
        description=(
            "Concrete evidenced adjacent public surface inspected, such as "
            "a result wrapper, reference adapter, generated header, or config path."
        ),
    )
    disposition: Literal[
        "required", "advisory", "not_required", "not_applicable"
    ] = Field(
        ...,
        description=(
            "required only when omission causes an evidenced public entry path "
            "to remain incorrect or breaks an established synchronization or "
            "same-semantic-family relationship; advisory for a plausible adjacent "
            "surface without enough evidence to block; not_required with positive "
            "evidence of an intentional API difference; not_applicable when the "
            "item is not a structural surface relevant to this review."
        ),
    )
    rationale: str = Field(
        ...,
        description="Concise repository-backed reason for the disposition.",
    )
    counterexample: str = Field(
        default="",
        max_length=800,
        description=(
            "For a required omission, the concrete public call, state, branch, "
            "or generated surface that remains wrong under the candidate patch."
        ),
    )
    evidence_ids: list[str] = Field(
        ...,
        min_length=1,
        description="SEARCH-N or GRAPH-N evidence supporting this disposition.",
    )

    @model_validator(mode="after")
    def require_blocking_counterexample(self) -> "SurfaceCheck":
        if self.disposition == "required" and not self.counterexample.strip():
            raise ValueError("a required surface needs a concrete counterexample")
        return self


class PatchReviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: bool
    report: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    surface_checks: list[SurfaceCheck] = Field(default_factory=list)
    actions: list[dict] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_rejection_report(self) -> "PatchReviewResult":
        if not self.approved and not self.report.strip():
            raise ValueError("a rejected patch review requires a report")
        return self
