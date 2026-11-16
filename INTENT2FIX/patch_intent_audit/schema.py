from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PatchIntentSemanticAdmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    admitted: bool
    available: bool = True
    authority_entails_target_expectation: bool = Field(
        ...,
        description=(
            "Whether the quoted authority supports the exact success condition "
            "encoded by the target source, either directly, through the "
            "deterministic standard semantics of the specific language mechanism "
            "and semantic dimension that the authority explicitly requires, or "
            "through a cited normative public API contract. An example or analogy "
            "does not import unrelated semantics from the compared construct."
        ),
    )
    authority_identifies_disputed_dimension: bool = Field(
        ...,
        description=(
            "Whether the selected authority makes the disputed public behavior "
            "dimension relevant to the requested repair, even when it does not "
            "select one exact outcome for that dimension. This is weaker than "
            "authority_entails_target_expectation and cannot admit or block a "
            "counterexample by itself."
        ),
    )
    pair_exposes_observable_choice: bool = Field(
        ...,
        description=(
            "Whether the frozen pair makes the competing mechanisms produce "
            "different publicly observable outcomes on one valid input, "
            "regardless of which outcome the target assertion accepts."
        ),
    )
    target_distinguishes_predictions: bool = Field(
        ...,
        description=(
            "Whether the target source accepts the alternative prediction and "
            "rejects the candidate prediction on the distinguishing input."
        ),
    )
    pair_is_comparable: bool = Field(
        ...,
        description=(
            "Whether control and target share a valid environment and public "
            "operation while varying only the disputed behavior dimension."
        ),
    )
    target_audits_modified_mechanism: bool = Field(
        ...,
        description=(
            "Whether the target exercises the semantic mechanism introduced "
            "or changed by the candidate patch, rather than only exposing an "
            "unmodified sibling, wrapper, generated mirror, or configuration "
            "surface. For lookup, dispatch, overload, or fallback precedence, "
            "the input must make the competing paths jointly applicable."
        ),
    )
    candidate_prediction_matches_observation: bool = Field(
        ...,
        description=(
            "Whether the observed target compile/run result matches the stated "
            "candidate prediction closely enough to validate the candidate "
            "mechanism being audited."
        ),
    )
    reason: str = Field(..., min_length=1, max_length=1200)

    @model_validator(mode="after")
    def validate_availability(self) -> "PatchIntentSemanticAdmission":
        if self.admitted and not self.available:
            raise ValueError("an unavailable admission cannot admit a pair")
        checks = {
            "authority_entails_target_expectation": (
                self.authority_entails_target_expectation
            ),
            "authority_identifies_disputed_dimension": (
                self.authority_identifies_disputed_dimension
            ),
            "pair_exposes_observable_choice": (
                self.pair_exposes_observable_choice
            ),
            "target_distinguishes_predictions": (
                self.target_distinguishes_predictions
            ),
            "pair_is_comparable": self.pair_is_comparable,
            "target_audits_modified_mechanism": (
                self.target_audits_modified_mechanism
            ),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if self.admitted and failed:
            self.admitted = False
            self.reason = (
                "Semantic consistency failed ("
                + ", ".join(failed)
                + "). "
                + self.reason
            )[:1200]
        return self


class FrozenPatchIntentProbe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    probe_name: str
    fixture_path: str
    fixture_content: str
    control_path: str
    control_content: str
    target_path: str
    target_content: str
    control_compile_command: str
    control_run_command: str
    target_compile_command: str
    target_run_command: str
    oracle: str
    oracle_mode: Literal["runtime_behavior", "compile_acceptance"]
    distinguishing_input: str
    candidate_prediction: str
    alternative_prediction: str
    purpose: str = ""
    oracle_source: Literal["issue_explicit", "accepted_contract"]
    oracle_reference: str
    control_compile_exit_code: int
    control_run_exit_code: int | None
    target_compile_exit_code: int | None
    target_run_exit_code: int | None
    output: str = ""


class PatchIntentAuditResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal[
        "underfit",
        "no_counterexample_found",
        "inconclusive",
    ]
    blocking: bool = False
    report: str = ""
    modified_behavior: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    counterexample: FrozenPatchIntentProbe | None = None
    unresolved_discriminator: FrozenPatchIntentProbe | None = None
    semantic_admission: PatchIntentSemanticAdmission | None = None
    actions: list[dict] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_blocking_evidence(self) -> "PatchIntentAuditResult":
        if self.blocking != (self.verdict == "underfit"):
            raise ValueError("only an evidenced underfit verdict may block repair")
        if self.blocking and self.counterexample is None:
            raise ValueError("a blocking underfit verdict requires a frozen probe")
        if self.blocking and not self.report.strip():
            raise ValueError("a blocking underfit verdict requires a report")
        if self.blocking and (
            self.semantic_admission is None
            or not self.semantic_admission.available
            or not self.semantic_admission.admitted
        ):
            raise ValueError(
                "a blocking underfit verdict requires semantic admission"
            )
        return self


class PatchIntentProbeRecheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    control_compile_exit_code: int
    control_run_exit_code: int | None = None
    target_compile_exit_code: int | None = None
    target_run_exit_code: int | None = None
    conclusive: bool = True
    passed: bool
    output: str = ""
    worktree_restored: bool = False
