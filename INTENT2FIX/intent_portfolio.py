from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator
from INTENT2FIX.utils.patch import production_patch_sha256
from INTENT2FIX.prompt_guidance import load_stage_guidance


class IntentObligation(BaseModel):
    id: str = Field(pattern=r"^(FIX|PRESERVE|CLOSURE)-[0-9]+$")
    kind: Literal["fix", "preserve", "closure"]
    statement: str
    authority: Literal[
        "issue_explicit",
        "public_contract",
        "execution_backed",
        "repository_structure",
        "legacy_behavior",
        "inferred",
    ]
    strength: Literal["hard", "soft"]
    evidence_basis: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_preservation_authority(self) -> "IntentObligation":
        if (
            self.kind == "preserve"
            and self.strength == "hard"
            and self.authority not in {"issue_explicit", "public_contract"}
        ):
            self.strength = "soft"
        return self


class IntentHypothesis(BaseModel):
    id: str = Field(pattern=r"^H[0-9]+$")
    title: str
    branch_assumption: str = Field(
        description=(
            "One complete evidence-supported repair interpretation. H0 may use "
            "'none' for the comprehensive common contract; alternatives must "
            "state their distinct assumption."
        )
    )
    root_cause: str = ""
    repair_scope: List[str] = Field(default_factory=list)
    covered_obligation_ids: List[str] = Field(default_factory=list)
    hard_obligation_coverage: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "For each hard obligation, explain the concrete change or result "
            "in this branch that satisfies it. An obligation ID alone is not "
            "evidence of coverage."
        ),
    )
    discriminator: str = Field(
        default="",
        description=(
            "A concrete public, compile, configuration, or structural scenario "
            "that distinguishes this branch from at least one alternative."
        ),
    )
    predicted_outcome: str = Field(
        default="",
        description=(
            "For a multi-branch portfolio, this branch's outcome for its "
            "discriminator."
        ),
    )
    expected_behavior: List[str] = Field(default_factory=list)
    preservation_behavior: List[str] = Field(default_factory=list)
    evidence_basis: List[str] = Field(default_factory=list)


class DimensionOutcome(BaseModel):
    hypothesis_id: str = Field(pattern=r"^H[0-9]+$")
    predicted_outcome: str


class UnresolvedDimension(BaseModel):
    id: str = Field(pattern=r"^DIM-[0-9]+$")
    statement: str
    impact: Literal["high", "low"]
    disposition: Literal["branched", "soft_constraint", "deferred"]
    hypothesis_ids: List[str] = Field(default_factory=list)
    shared_discriminator: str = Field(
        default="",
        description=(
            "For a branched dimension, one candidate-independent fixture or "
            "scenario executed unchanged for every named hypothesis."
        ),
    )
    outcomes: List[DimensionOutcome] = Field(
        default_factory=list,
        description=(
            "For a branched dimension, the definite observable outcome "
            "assigned to each named hypothesis on the shared discriminator."
        ),
    )
    reason: str


class MergeBoundaryAlternative(BaseModel):
    label: str
    selection_rule: str
    contract_viability: Literal["plausible", "contradicted", "uncertain"]
    viability_reason: str
    predicted_outcomes: Dict[str, str] = Field(default_factory=dict)
    evidence_basis: List[str] = Field(default_factory=list)


class MergeBoundaryDecision(BaseModel):
    id: str = Field(pattern=r"^MERGE-[0-9]+$")
    question: str
    verdict: Literal[
        "keep_separate",
        "merge_safe",
        "inconclusive_keep",
        "single_viable",
    ]
    alternatives: List[MergeBoundaryAlternative] = Field(
        min_length=2, max_length=3
    )
    boundary_scenarios: List[str] = Field(default_factory=list, max_length=4)
    reason: str
    evidence_basis: List[str] = Field(default_factory=list)


class HypothesisMergeAdjudication(BaseModel):
    decisions: List[MergeBoundaryDecision] = Field(
        default_factory=list, max_length=3
    )


class RepairSurfaceDisposition(BaseModel):
    lead_id: str = Field(pattern=r"^SURFACE-[0-9]+$")
    disposition: Literal["included", "excluded", "deferred"]
    hypothesis_ids: List[str] = Field(default_factory=list)
    reason: str


class IntentPortfolio(BaseModel):
    obligations: List[IntentObligation] = Field(min_length=1, max_length=16)
    hypotheses: List[IntentHypothesis] = Field(min_length=1, max_length=3)
    unresolved_dimensions: List[UnresolvedDimension] = Field(default_factory=list)
    surface_dispositions: List[RepairSurfaceDisposition] = Field(
        default_factory=list,
        max_length=15,
        description=(
            "One explicit include/exclude/defer decision for every supplied "
            "SURFACE-N structural lead. Leads are evidence to assess, not "
            "requirements by themselves."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_recoverable_structure(cls, value: Any) -> Any:
        """Repair transport-level ID drift without changing semantic claims."""
        if not isinstance(value, dict):
            return value
        payload = dict(value)

        def numbered_id(raw: object, prefix: str, fallback: int) -> str:
            text = str(raw or "").strip().upper().replace("_", "-")
            match = re.search(r"([0-9]+)$", text)
            return f"{prefix}-{match.group(1) if match else fallback}"

        obligation_aliases: Dict[str, str] = {}
        obligations = []
        used_obligations = set()
        for index, raw in enumerate(payload.get("obligations") or [], start=1):
            if isinstance(raw, BaseModel):
                raw = raw.model_dump()
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            kind = str(item.get("kind") or "fix").lower()
            prefix = {
                "preserve": "PRESERVE",
                "closure": "CLOSURE",
            }.get(kind, "FIX")
            original_id = str(item.get("id") or "")
            canonical_id = numbered_id(original_id, prefix, index)
            if canonical_id in used_obligations:
                continue
            item["id"] = canonical_id
            obligation_aliases[original_id] = canonical_id
            obligation_aliases[original_id.upper()] = canonical_id
            used_obligations.add(canonical_id)
            obligations.append(item)
        payload["obligations"] = obligations

        hypothesis_aliases: Dict[str, str] = {}
        hypotheses = []
        used_hypotheses = set()
        for index, raw in enumerate(payload.get("hypotheses") or []):
            if isinstance(raw, BaseModel):
                raw = raw.model_dump()
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            original_id = str(item.get("id") or "")
            match = re.search(r"([0-9]+)$", original_id.strip())
            candidate_id = f"H{match.group(1) if match else index}"
            while candidate_id in used_hypotheses:
                index += 1
                candidate_id = f"H{index}"
            hypothesis_aliases[original_id] = candidate_id
            hypothesis_aliases[original_id.upper()] = candidate_id
            item["id"] = candidate_id
            used_hypotheses.add(candidate_id)
            hypotheses.append(item)

        hard_ids = {
            item["id"]
            for item in obligations
            if item.get("strength") == "hard"
        }
        for item in hypotheses:
            covered = {
                obligation_aliases.get(str(raw), obligation_aliases.get(
                    str(raw).upper(), str(raw)
                ))
                for raw in item.get("covered_obligation_ids") or []
            }
            covered = (covered & used_obligations) | hard_ids
            item["covered_obligation_ids"] = sorted(covered)
            claims = {
                obligation_aliases.get(str(raw), obligation_aliases.get(
                    str(raw).upper(), str(raw)
                )): str(explanation or "")
                for raw, explanation in (
                    item.get("hard_obligation_coverage") or {}
                ).items()
            }
            claims = {
                key: explanation
                for key, explanation in claims.items()
                if key in used_obligations and explanation.strip()
            }
            for obligation_id in hard_ids:
                claims.setdefault(
                    obligation_id,
                    "Evaluate this Issue target during Repair and final "
                    "candidate comparison.",
                )
            item["hard_obligation_coverage"] = claims
        payload["hypotheses"] = hypotheses

        dimensions = []
        for index, raw in enumerate(
            payload.get("unresolved_dimensions") or [], start=1
        ):
            if isinstance(raw, BaseModel):
                raw = raw.model_dump()
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            item["id"] = numbered_id(item.get("id"), "DIM", index)
            item["hypothesis_ids"] = list(dict.fromkeys(
                hypothesis_aliases.get(
                    str(raw_id),
                    hypothesis_aliases.get(str(raw_id).upper(), str(raw_id)),
                )
                for raw_id in item.get("hypothesis_ids") or []
                if hypothesis_aliases.get(
                    str(raw_id),
                    hypothesis_aliases.get(str(raw_id).upper(), str(raw_id)),
                ) in used_hypotheses
            ))
            outcomes = []
            for raw_outcome in item.get("outcomes") or []:
                if isinstance(raw_outcome, BaseModel):
                    raw_outcome = raw_outcome.model_dump()
                if not isinstance(raw_outcome, dict):
                    continue
                outcome = dict(raw_outcome)
                raw_id = str(outcome.get("hypothesis_id") or "")
                outcome["hypothesis_id"] = hypothesis_aliases.get(
                    raw_id, hypothesis_aliases.get(raw_id.upper(), raw_id)
                )
                if outcome["hypothesis_id"] in item["hypothesis_ids"]:
                    outcomes.append(outcome)
            item["outcomes"] = outcomes
            distinct_outcomes = {
                " ".join(
                    str(outcome.get("predicted_outcome") or "").lower().split()
                )
                for outcome in outcomes
                if str(outcome.get("predicted_outcome") or "").strip()
            }
            if (
                item.get("disposition") == "branched"
                and (
                    len(item["hypothesis_ids"]) < 2
                    or not str(item.get("shared_discriminator") or "").strip()
                    or {outcome["hypothesis_id"] for outcome in outcomes}
                    != set(item["hypothesis_ids"])
                    or len(distinct_outcomes) < 2
                )
            ):
                item["disposition"] = "deferred"
                item["outcomes"] = []
            dimensions.append(item)
        payload["unresolved_dimensions"] = dimensions

        surfaces = []
        for index, raw in enumerate(
            payload.get("surface_dispositions") or [], start=1
        ):
            if isinstance(raw, BaseModel):
                raw = raw.model_dump()
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            item["lead_id"] = numbered_id(
                item.get("lead_id"), "SURFACE", index
            )
            item["hypothesis_ids"] = [
                hypothesis_aliases.get(
                    str(raw_id),
                    hypothesis_aliases.get(str(raw_id).upper(), str(raw_id)),
                )
                for raw_id in item.get("hypothesis_ids") or []
                if hypothesis_aliases.get(
                    str(raw_id),
                    hypothesis_aliases.get(str(raw_id).upper(), str(raw_id)),
                ) in used_hypotheses
            ]
            if item.get("disposition") == "included" and not item[
                "hypothesis_ids"
            ]:
                item["disposition"] = "deferred"
            if not str(item.get("reason") or "").strip():
                item["reason"] = "Preserved from portfolio extraction."
            surfaces.append(item)
        payload["surface_dispositions"] = surfaces
        return payload

    @model_validator(mode="after")
    def validate_unique_hypotheses(self) -> "IntentPortfolio":
        ids = [item.id for item in self.hypotheses]
        if len(ids) != len(set(ids)):
            raise ValueError("hypothesis IDs must be unique")
        obligation_ids = [item.id for item in self.obligations]
        if len(obligation_ids) != len(set(obligation_ids)):
            raise ValueError("obligation IDs must be unique")
        known_obligations = set(obligation_ids)
        hard_obligations = {
            item.id for item in self.obligations if item.strength == "hard"
        }
        for hypothesis in self.hypotheses:
            covered = set(hypothesis.covered_obligation_ids)
            unknown = covered - known_obligations
            if unknown:
                raise ValueError(
                    f"{hypothesis.id} names unknown obligations: "
                    + ", ".join(sorted(unknown))
                )
            missing = hard_obligations - covered
            if missing:
                raise ValueError(
                    f"{hypothesis.id} must cover every hard obligation; missing: "
                    + ", ".join(sorted(missing))
                )
            coverage_claims = set(hypothesis.hard_obligation_coverage)
            missing_claims = hard_obligations - coverage_claims
            unknown_claims = coverage_claims - known_obligations
            if missing_claims or unknown_claims:
                details = []
                if missing_claims:
                    details.append(
                        "missing " + ", ".join(sorted(missing_claims))
                    )
                if unknown_claims:
                    details.append(
                        "unknown " + ", ".join(sorted(unknown_claims))
                    )
                raise ValueError(
                    f"{hypothesis.id} coverage claims must include every hard "
                    "obligation and name only known obligations: "
                    + "; ".join(details)
                )
            empty_claims = [
                obligation_id
                for obligation_id, explanation in (
                    hypothesis.hard_obligation_coverage.items()
                )
                if not explanation.strip()
            ]
            if empty_claims:
                raise ValueError(
                    f"{hypothesis.id} has empty hard-obligation coverage "
                    "claims: " + ", ".join(sorted(empty_claims))
                )

        signatures = [
            json.dumps(
                {
                    "branch_assumption": item.branch_assumption.strip().lower(),
                    "root_cause": item.root_cause.strip().lower(),
                    "repair_scope": sorted(
                        value.strip().lower() for value in item.repair_scope
                    ),
                    "discriminator": item.discriminator.strip().lower(),
                    "predicted_outcome": item.predicted_outcome.strip().lower(),
                    "expected_behavior": sorted(
                        value.strip().lower() for value in item.expected_behavior
                    ),
                    "preservation_behavior": sorted(
                        value.strip().lower()
                        for value in item.preservation_behavior
                    ),
                },
                sort_keys=True,
            )
            for item in self.hypotheses
        ]
        if len(signatures) != len(set(signatures)):
            unique = {}
            for signature, hypothesis in zip(signatures, self.hypotheses):
                unique.setdefault(signature, hypothesis)
            self.hypotheses = list(unique.values())
        if len(self.hypotheses) > 1:
            for item in self.hypotheses:
                if (
                    item.id != "H0"
                    and item.branch_assumption.strip().lower() == "none"
                ):
                    item.branch_assumption = (
                        item.title.strip() or "alternative search hypothesis"
                    )
        known_hypotheses = set(ids)
        surface_ids = [item.lead_id for item in self.surface_dispositions]
        if len(surface_ids) != len(set(surface_ids)):
            raise ValueError("surface disposition IDs must be unique")
        for disposition in self.surface_dispositions:
            if set(disposition.hypothesis_ids) - known_hypotheses:
                raise ValueError(
                    f"{disposition.lead_id} names unknown hypotheses"
                )
            if disposition.disposition == "included" and not (
                disposition.hypothesis_ids and disposition.reason.strip()
            ):
                raise ValueError(
                    f"{disposition.lead_id} included disposition requires "
                    "hypotheses and a reason"
                )
        dimension_ids = [item.id for item in self.unresolved_dimensions]
        if len(dimension_ids) != len(set(dimension_ids)):
            raise ValueError("unresolved dimension IDs must be unique")
        for dimension in self.unresolved_dimensions:
            dimension_hypotheses = set(dimension.hypothesis_ids)
            if dimension_hypotheses - known_hypotheses:
                raise ValueError(
                    f"{dimension.id} names unknown hypotheses"
                )
            if dimension.disposition == "branched" and len(
                dimension_hypotheses
            ) < 2:
                raise ValueError(
                    f"{dimension.id} marked branched must name at least two "
                    "hypotheses"
                )
            if dimension.disposition == "branched":
                if not dimension.shared_discriminator.strip():
                    raise ValueError(
                        f"{dimension.id} branched dimension requires one "
                        "shared discriminator"
                    )
                outcome_ids = [
                    item.hypothesis_id for item in dimension.outcomes
                ]
                if len(outcome_ids) != len(set(outcome_ids)):
                    raise ValueError(
                        f"{dimension.id} has duplicate hypothesis outcomes"
                    )
                if set(outcome_ids) != dimension_hypotheses:
                    raise ValueError(
                        f"{dimension.id} outcomes must cover exactly its "
                        "hypothesis IDs"
                    )
                normalized_outcomes = {
                    " ".join(item.predicted_outcome.lower().split())
                    for item in dimension.outcomes
                    if item.predicted_outcome.strip()
                }
                if len(normalized_outcomes) < 2:
                    raise ValueError(
                        f"{dimension.id} shared discriminator requires at "
                        "least two distinct definite outcomes"
                    )
            if dimension.impact == "high" and not dimension.reason.strip():
                raise ValueError(
                    f"{dimension.id} high-impact disposition requires a reason"
                )
        return self

    def obligations_of_kind(self, kind: str) -> List[IntentObligation]:
        return [item for item in self.obligations if item.kind == kind]

    @property
    def repair_hypotheses(self) -> List[IntentHypothesis]:
        return self.hypotheses

    def coverage_report(self) -> dict:
        rows = []
        for obligation in self.obligations:
            covered_by = [
                item.id
                for item in self.hypotheses
                if obligation.id in item.covered_obligation_ids
            ]
            rows.append(
                {
                    "obligation_id": obligation.id,
                    "kind": obligation.kind,
                    "strength": obligation.strength,
                    "authority": obligation.authority,
                    "covered_by": covered_by,
                    "missing_from": [
                        item.id
                        for item in self.hypotheses
                        if item.id not in covered_by
                    ],
                }
            )
        return {
            "valid": all(
                row["strength"] != "hard" or not row["missing_from"]
                for row in rows
            ),
            "obligations": rows,
            "unresolved_dimensions": [
                item.model_dump() for item in self.unresolved_dimensions
            ],
            "surface_dispositions": [
                item.model_dump() for item in self.surface_dispositions
            ],
        }


class PortfolioCoverageAudit(BaseModel):
    decision: Literal["ACCEPT", "REVISE"]
    uncovered_issue_obligations: List[str] = Field(default_factory=list)
    repair_surface_gaps: List[str] = Field(default_factory=list)
    causal_coverage_gaps: List[str] = Field(default_factory=list)
    unsupported_hard_preserves: List[str] = Field(default_factory=list)
    internal_conflicts: List[str] = Field(default_factory=list)
    nonblocking_notes: List[str] = Field(
        default_factory=list,
        description=(
            "Evidence updates, unavailable environments, unproven equivalence or "
            "implementation uncertainty without a concrete contract defect. These "
            "do not request revision; never put actual contradictions here."
        ),
    )
    reason: str

    @model_validator(mode="after")
    def normalize_decision(self) -> "PortfolioCoverageAudit":
        if self.decision == "ACCEPT" and any(
            (
                self.uncovered_issue_obligations,
                self.repair_surface_gaps,
                self.causal_coverage_gaps,
                self.unsupported_hard_preserves,
                self.internal_conflicts,
            )
        ):
            self.decision = "REVISE"
        return self


class IntentHypothesisPool(IntentPortfolio):
    """Keep the full semantic pool while budgeting at most three Repairs."""

    hypotheses: List[IntentHypothesis] = Field(min_length=1, max_length=8)
    selected_hypothesis_ids: List[str] = Field(
        default_factory=list,
        max_length=3,
        description="Leave empty during generation; assigned only by pool reranking.",
    )

    @model_validator(mode="after")
    def validate_selection(self) -> "IntentHypothesisPool":
        hard_ids = {item.id for item in self.obligations if item.strength == "hard"}
        for hypothesis in self.hypotheses:
            for obligation_id in hard_ids:
                if hypothesis.hard_obligation_coverage[obligation_id].startswith(
                    "Evaluate this Issue target during Repair"
                ):
                    raise ValueError(
                        f"{hypothesis.id} needs concrete coverage for {obligation_id}; "
                        "an automatically filled placeholder is not a repair plan"
                    )
        selected = self.selected_hypothesis_ids
        known = {item.id for item in self.hypotheses}
        if len(selected) != len(set(selected)) or set(selected) - known:
            raise ValueError("Pool selection must name unique existing hypotheses")
        if selected and not 1 <= len(selected) <= min(3, len(known)):
            raise ValueError("Pool selection must contain one to three hypotheses")
        assigned = {
            hypothesis_id
            for dimension in self.unresolved_dimensions
            if dimension.disposition == "branched"
            for hypothesis_id in dimension.hypothesis_ids
        }
        # The shared dimension is the authoritative scenario/outcome mapping.
        # Do not pass a second, potentially conflicting copy to Audit or Repair.
        self.hypotheses = [
            hypothesis.model_copy(update={"discriminator": "", "predicted_outcome": ""})
            if hypothesis.id in assigned else hypothesis
            for hypothesis in self.hypotheses
        ]
        return self

    @property
    def repair_hypotheses(self) -> List[IntentHypothesis]:
        if not self.selected_hypothesis_ids:
            raise ValueError("Hypothesis pool has not been reranked; Repair is disabled")
        by_id = {item.id: item for item in self.hypotheses}
        return [by_id[item_id] for item_id in self.selected_hypothesis_ids]


class HypothesisConsistencyReview(BaseModel):
    premise_status: Literal["supported", "unverified", "contradicted"]
    premise_check: str = Field(
        min_length=1,
        description="Actual Issue input/receiver and concrete code gate evaluated independently of branch claims."
    )
    assigned_outcome_status: Literal["consistent", "contradicted"]
    hard_contract_check: str = Field(min_length=1)
    assigned_outcome_check: str = Field(min_length=1)
    implementation_feasibility_check: str = Field(min_length=1)
    soft_tradeoffs: List[str] = Field(
        default_factory=list,
        description="Intentional losses of soft evidence that this branch does not make mandatory; these are not contradictions."
    )
    conflicts: List[str] = Field(
        default_factory=list,
        description="Concrete incompatible requirements with field/obligation IDs; empty only if none found."
    )


class DiscriminatorReview(BaseModel):
    reviewed_scenario: str = Field(
        min_length=1, description="Copy this dimension's shared_discriminator verbatim; never replace its receiver, input, API or mode."
    )
    compared_hypothesis_ids: List[str] = Field(min_length=2, max_length=2)
    scenario_status: Literal["distinguishes", "same_outcome", "invalid_fixture", "undetermined"]
    compared_outcomes: Dict[str, str] = Field(
        description="Exactly the two compared IDs mapped to their ORIGINAL predicted_outcome strings copied verbatim from this dimension. Evaluate those claims in comparison; do not repair them here."
    )
    policy_difference: str = Field(min_length=1)
    comparison: str = Field(
        min_length=1,
        description="Compare actual receiver, input, mode and observable outcomes under the same scenario."
    )
    conflicts: List[str] = Field(
        default_factory=list,
        description="Wrong/non-distinguishing input, inconsistent predictions or candidate-dependent fixture."
    )


class HypothesisPoolAudit(PortfolioCoverageAudit):
    hypothesis_reviews: Dict[str, HypothesisConsistencyReview]
    discriminator_reviews: Dict[str, DiscriminatorReview] = Field(default_factory=dict)
    selected_hypothesis_ids: List[str] = Field(
        default_factory=list, max_length=3,
        description=(
            "Rank one to three repairable hypotheses when possible. REVISE records "
            "audit concerns but does not prevent a best-effort Repair assignment."
        ),
    )
    selection_reasons: Dict[str, str] = Field(
        default_factory=dict,
        description="For ACCEPT, cite support and marginal semantic coverage for each selected ID."
    )
    unselected_reasons: Dict[str, str] = Field(
        default_factory=dict,
        description="For every unselected ID, explain the budget tradeoff; not a truth verdict."
    )
    joint_policy_losses: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "For EVERY unselected ID, state the complete policy/behavior combination "
            "that will NOT be executed. Factors appearing in different selected "
            "branches do not execute this combination. Empty only when all are selected."
        )
    )

    @model_validator(mode="before")
    @classmethod
    def recover_unambiguous_review_shape(cls, data):
        if not isinstance(data, dict):
            return data
        for group, review_model, prefix in (
            ("discriminator_reviews", DiscriminatorReview, "DIM-"),
            ("hypothesis_reviews", HypothesisConsistencyReview, "H"),
        ):
            reviews = data.get(group)
            if not isinstance(reviews, dict):
                continue
            reviews = dict(reviews)
            review_ids = {key for key in reviews
                          if key.startswith(prefix) and key[len(prefix):].isdigit()}
            for field in review_model.model_fields:
                if field not in reviews:
                    continue
                missing = [key for key in review_ids
                           if isinstance(reviews[key], dict) and field not in reviews[key]]
                if len(missing) == 1:
                    # Move only an existing value to its uniquely missing slot.
                    # Required or contradictory judgments are never synthesized.
                    reviews[missing[0]] = {**reviews[missing[0]], field: reviews.pop(field)}
                elif not missing and reviews[field] in (None, "", "placeholder", [], {}):
                    del reviews[field]
            for key in set(reviews) - review_ids - set(review_model.model_fields):
                if reviews[key] in (None, "", "placeholder", [], {}):
                    del reviews[key]
            data = {**data, group: reviews}
        return data

    @model_validator(mode="after")
    def collect_review_conflicts(self) -> "HypothesisPoolAudit":
        conflicts = {item_id: list(review.conflicts) for item_id, review in {
            **self.hypothesis_reviews, **self.discriminator_reviews
        }.items()}
        for item_id, review in self.hypothesis_reviews.items():
            if review.premise_status == "contradicted":
                conflicts[item_id].append(f"Contradicted premise: {review.premise_check}")
            if review.assigned_outcome_status == "contradicted":
                conflicts[item_id].append(f"Contradicted assigned outcome: {review.assigned_outcome_check}")
        for item_id, review in self.discriminator_reviews.items():
            pair = review.compared_hypothesis_ids
            if set(review.compared_outcomes) != set(pair) or any(
                not outcome.strip() for outcome in review.compared_outcomes.values()
            ):
                raise ValueError("Discriminator results must cover exactly the compared pair")
            outcomes = {" ".join(outcome.split()).casefold()
                        for outcome in review.compared_outcomes.values()}
            if len(outcomes) == 1 and review.scenario_status == "distinguishes":
                review.scenario_status = "same_outcome"
            if review.scenario_status != "distinguishes":
                conflicts[item_id].append(
                    f"Discriminator {review.scenario_status}: {review.comparison}"
                )
        for item_id, messages in conflicts.items():
            for conflict in messages:
                message = f"{item_id}: {conflict}"
                if message not in self.internal_conflicts:
                    self.internal_conflicts.append(message)
        if self.internal_conflicts:
            self.decision = "REVISE"
        return self

    def validate_pool(self, pool: IntentHypothesisPool) -> None:
        known = {item.id for item in pool.hypotheses}
        dimensions = {d.id for d in pool.unresolved_dimensions if d.disposition == "branched"}
        if set(self.hypothesis_reviews) != known:
            actual = set(self.hypothesis_reviews)
            raise ValueError(
                "Audit must review every complete hypothesis exactly once; "
                f"hypothesis_reviews expected keys={sorted(known)}, "
                f"missing={sorted(known - actual)}, unexpected={sorted(actual - known)}"
            )
        if set(self.discriminator_reviews) != dimensions:
            actual = set(self.discriminator_reviews)
            raise ValueError(
                "Audit must review every branched discriminator exactly once; "
                f"discriminator_reviews expected keys={sorted(dimensions)}, "
                f"missing={sorted(dimensions - actual)}, unexpected={sorted(actual - dimensions)}"
            )
        for dimension in pool.unresolved_dimensions:
            if dimension.disposition != "branched":
                continue
            pair = self.discriminator_reviews[dimension.id].compared_hypothesis_ids
            if len(set(pair)) != 2 or set(pair) - set(dimension.hypothesis_ids):
                raise ValueError("Discriminator comparison must name two distinct participants")
            review = self.discriminator_reviews[dimension.id]
            original_outcomes = {item.hypothesis_id: item.predicted_outcome
                                 for item in dimension.outcomes}
            if (review.reviewed_scenario != dimension.shared_discriminator
                    or review.compared_outcomes != {key: original_outcomes[key] for key in pair}):
                conflict = (
                    f"{dimension.id}: Audit substituted the original scenario or assigned "
                    "predictions. Review the unchanged fixture; propose any correction "
                    "explicitly in revision feedback, not as an ACCEPT justification."
                )
                if conflict not in self.internal_conflicts:
                    self.internal_conflicts.append(conflict)
                self.decision = "REVISE"
        selected = set(self.selected_hypothesis_ids)
        if len(selected) != len(self.selected_hypothesis_ids) or selected - known:
            raise ValueError("Pool selection must name unique existing hypotheses")
        # REVISE feedback remains useful even without a complete budget
        # assignment. Repair scheduling has a deterministic best-effort fallback.
        if self.decision == "REVISE":
            return
        if not 1 <= len(selected) <= min(3, len(known)):
            raise ValueError("Pool selection must contain one to three hypotheses")
        if set(self.selection_reasons) != selected:
            raise ValueError("Rerank selection_reasons must cover exactly the selected hypotheses")
        if set(self.unselected_reasons) != known - selected:
            raise ValueError("Rerank must explain every unselected hypothesis")
        if any(not value.strip() for value in [
            *self.selection_reasons.values(), *self.unselected_reasons.values()
        ]):
            raise ValueError("Rerank explanations must not be empty")
        deferred = known - selected
        if set(self.joint_policy_losses) != deferred or any(
            not value.strip() for value in self.joint_policy_losses.values()
        ):
            actual = set(self.joint_policy_losses)
            empty = sorted(key for key, value in self.joint_policy_losses.items()
                           if not value.strip())
            raise ValueError(
                "Rerank must record every unexecuted complete policy loss; "
                f"joint_policy_losses expected keys={sorted(deferred)}, "
                f"missing={sorted(deferred - actual)}, unexpected={sorted(actual - deferred)}, "
                f"empty={empty}"
            )


def select_hypothesis_pool(
    pool: IntentHypothesisPool, audit: HypothesisPoolAudit
) -> IntentHypothesisPool:
    audit.validate_pool(pool)
    known_ids = [item.id for item in pool.hypotheses]
    repairable_ids = [
        item_id for item_id in known_ids
        if (
            audit.hypothesis_reviews[item_id].premise_status != "contradicted"
            and audit.hypothesis_reviews[item_id].assigned_outcome_status != "contradicted"
            and not audit.hypothesis_reviews[item_id].conflicts
        )
    ]
    ranked = audit.selected_hypothesis_ids[:3]
    selected = [item_id for item_id in ranked if item_id in repairable_ids]
    if not selected:
        selected = repairable_ids[:3]
    if audit.decision == "REVISE":
        # A review can disprove a proposed edit without disproving its public
        # repair intent. Pass ranked conditional branches to Repair with the
        # concrete review findings instead of silently collapsing the pool.
        for item_id in ranked:
            if len(selected) >= 3:
                break
            if item_id not in selected:
                selected.append(item_id)
    if not selected:
        # A complete generated branch is still more useful to Repair than a
        # workflow stop. The Audit findings remain available as an advisory.
        selected = [known_ids[0]]
    payload = pool.model_dump()
    payload["selected_hypothesis_ids"] = selected
    return IntentHypothesisPool.model_validate(payload)


def hypothesis_review_cards(portfolio: IntentPortfolio) -> List[dict]:
    """Align each complete policy with all the outcomes Repair will receive."""
    return [
        {
            "hypothesis": h.model_dump(),
            "assigned_dimensions": [
                {k: v for k, v in item.items() if k != "competing_outcomes"}
                for item in assigned_dimension_contracts(portfolio, h)
            ],
        }
        for h in portfolio.hypotheses
    ]


def assigned_dimension_contracts(
    portfolio: IntentPortfolio,
    hypothesis: IntentHypothesis,
) -> List[dict]:
    contracts: List[dict] = []
    for dimension in portfolio.unresolved_dimensions:
        if (
            dimension.disposition != "branched"
            or hypothesis.id not in dimension.hypothesis_ids
        ):
            continue
        assigned_outcome = next(
            item.predicted_outcome
            for item in dimension.outcomes
            if item.hypothesis_id == hypothesis.id
        )
        contracts.append(
            {
                "dimension_id": dimension.id,
                "statement": dimension.statement,
                "shared_discriminator": dimension.shared_discriminator,
                "assigned_outcome": assigned_outcome,
                "competing_outcomes": [
                    item.model_dump()
                    for item in dimension.outcomes
                    if item.hypothesis_id != hypothesis.id
                ],
            }
        )
    return contracts


def render_assigned_discriminator(
    portfolio: IntentPortfolio,
    hypothesis: IntentHypothesis,
) -> str:
    contracts = assigned_dimension_contracts(portfolio, hypothesis)
    if not contracts:
        return hypothesis.discriminator
    return json.dumps(
        [
            {
                "dimension_id": item["dimension_id"],
                "shared_discriminator": item["shared_discriminator"],
            }
            for item in contracts
        ],
        ensure_ascii=False,
        sort_keys=True,
    )


def render_assigned_outcome(
    portfolio: IntentPortfolio,
    hypothesis: IntentHypothesis,
) -> str:
    contracts = assigned_dimension_contracts(portfolio, hypothesis)
    if not contracts:
        return hypothesis.predicted_outcome
    return json.dumps(
        [
            {
                "dimension_id": item["dimension_id"],
                "assigned_outcome": item["assigned_outcome"],
            }
            for item in contracts
        ],
        ensure_ascii=False,
        sort_keys=True,
    )


class ProbeExecutionResult(BaseModel):
    check_id: str
    authority: str = "normative"
    kind: str = ""
    origin: str = "scout"
    source_hypothesis_id: str = ""
    expected_outcome: str = ""
    description: str = ""
    command: str = ""
    baseline_expected_exit_code: Optional[int] = None
    expected_exit_code: int = 0
    required_output_regex: str = ""
    status: Literal["passed", "failed", "infrastructure_error"]
    exit_code: Optional[int] = None
    output: str = ""


class RepairProbeEvidence(BaseModel):
    probe_id: str
    source_hypothesis_id: str
    authority: Literal["hypothesis"] = "hypothesis"
    kind: Literal["focused", "discriminator", "soft_regression"] = "focused"
    description: str = ""
    expected_outcome: str = ""
    command: str
    files: Dict[str, str] = Field(default_factory=dict)
    expected_exit_code: int = 0
    required_output_regex: str = ""
    observed_status: Literal["passed", "failed", "inconclusive"]
    observed_exit_code: Optional[int] = None
    observed_output: str = ""
    production_patch_sha256: str


class BranchViolationEvidence(BaseModel):
    clause_id: str
    clause_quote: str
    scenario: str
    expected_behavior: str
    observed_behavior: str
    evidence_ids: List[str] = Field(default_factory=list)


class BranchFailureExplanation(BaseModel):
    failure_id: str
    clause_id: str
    clause_quote: str
    scenario: str
    expected_old_behavior: str
    observed_behavior: str
    evidence_quote: str


class BranchComplianceDecision(BaseModel):
    status: Literal["compliant", "noncompliant", "inconclusive"]
    observed_behavior: str
    reason: str
    actionable_correction: bool = False
    violated_clause_ids: List[str] = Field(default_factory=list)
    suggested_correction: str = ""
    production_patch_sha256: str = ""
    violations: List[BranchViolationEvidence] = Field(default_factory=list)
    failure_explanations: List[BranchFailureExplanation] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_actionable_correction(self) -> "BranchComplianceDecision":
        if not self.actionable_correction:
            return self
        if self.status != "noncompliant":
            raise ValueError(
                "Only a concrete noncompliance can request correction."
            )
        if not self.violated_clause_ids or not self.suggested_correction.strip():
            raise ValueError(
                "An actionable correction requires violated clause IDs and "
                "a bounded suggested correction."
            )
        return self


class BranchCorrectionRefinement(BaseModel):
    violations: List[BranchViolationEvidence]
    model_config = {"extra": "forbid"}


class CrossExecutionResult(BaseModel):
    status: Literal[
        "passed",
        "failed",
        "apply_failed",
        "unavailable",
        "infrastructure_error",
    ]
    command: str = ""
    exit_code: Optional[int] = None
    output: str = ""
    patch_applied: Optional[bool] = None
    probe_results: List[ProbeExecutionResult] = Field(default_factory=list)


class PortfolioCandidate(BaseModel):
    hypothesis_id: str
    variant_id: str = "final"
    branch_assumption: str
    repair_contract: Dict[str, object] = Field(default_factory=dict)
    discriminator: str = ""
    predicted_outcome: str = ""
    patch: str
    patch_sha256: str
    kept_paths: List[str] = Field(default_factory=list)
    excluded_paths: Dict[str, str] = Field(default_factory=dict)
    finish_succeeded: bool
    basic_tests_passed: bool
    branch_compliance: Literal[
        "compliant", "noncompliant", "inconclusive", "not_checked"
    ] = "not_checked"
    discriminator_verified: bool = False
    repair_probes: List[RepairProbeEvidence] = Field(default_factory=list)
    cross_execution: CrossExecutionResult

    @classmethod
    def create(
        cls,
        *,
        hypothesis_id: str,
        variant_id: str = "final",
        branch_assumption: str,
        repair_contract: Dict[str, object] | None = None,
        discriminator: str = "",
        predicted_outcome: str = "",
        patch: str,
        kept_paths: List[str],
        excluded_paths: Dict[str, str],
        finish_succeeded: bool,
        basic_tests_passed: bool,
        branch_compliance: str = "not_checked",
        discriminator_verified: bool = False,
        repair_probes: List[RepairProbeEvidence] | None = None,
        cross_execution: CrossExecutionResult,
    ) -> "PortfolioCandidate":
        return cls(
            hypothesis_id=hypothesis_id,
            variant_id=variant_id,
            branch_assumption=branch_assumption,
            repair_contract=repair_contract or {},
            discriminator=discriminator,
            predicted_outcome=predicted_outcome,
            patch=patch,
            patch_sha256=production_patch_sha256(patch),
            kept_paths=kept_paths,
            excluded_paths=excluded_paths,
            finish_succeeded=finish_succeeded,
            basic_tests_passed=basic_tests_passed,
            branch_compliance=branch_compliance,
            discriminator_verified=discriminator_verified,
            repair_probes=repair_probes or [],
            cross_execution=cross_execution,
        )


class CandidateAssessment(BaseModel):
    candidate_id: str
    target_coverage: Literal["complete", "partial", "missing", "unclear"]
    preservation: Literal["supported", "risk", "violated", "unclear"]
    production_scope: Literal["focused", "questionable", "unrelated", "unclear"]
    legacy_test_interpretation: Literal[
        "supports_candidate",
        "issue_authorized_failure",
        "likely_regression",
        "inconclusive",
        "unavailable",
    ]
    execution_support: Literal[
        "supported", "contradicted", "static_only", "unavailable"
    ] = "unavailable"
    concerns: List[str] = Field(default_factory=list)


class CandidateAuditDecision(BaseModel):
    selection: Literal["SELECTED", "ABSTAIN"]
    selected_candidate_id: Optional[str] = None
    assessments: List[CandidateAssessment] = Field(default_factory=list)
    decision_reason: str

    @model_validator(mode="after")
    def validate_selection(self) -> "CandidateAuditDecision":
        if self.selection == "SELECTED" and not self.selected_candidate_id:
            raise ValueError("SELECTED requires selected_candidate_id")
        if self.selection == "ABSTAIN" and self.selected_candidate_id is not None:
            raise ValueError("ABSTAIN cannot name a selected candidate")
        return self


def build_hypothesis_merge_adjudication_prompt(
    *,
    issue: str,
    issue_evidence: dict,
    repair_intent_bundle: dict,
    repair_intent_contract: dict,
) -> str:
    """Build one bounded semantic-equivalence adjudication request."""
    contract = repair_intent_contract or {}
    bundle = repair_intent_bundle or {}
    evidence = {
        "issue_evidence": issue_evidence,
        "hypothesis_ready": bundle.get("hypothesis_ready"),
        "evidence_sufficiency_review": bundle.get(
            "evidence_sufficiency_review"
        ),
        "open_decisions": (
            contract.get("open_decisions")
            or bundle.get("open_decisions")
            or []
        ),
        "candidate_boundaries": (
            contract.get("candidate_boundaries")
            or bundle.get("candidate_boundaries")
            or []
        ),
        "acceptance_checks": contract.get("acceptance_checks") or [],
        "scope_evidence": contract.get("scope_evidence") or [],
        "input_path_check": bundle.get("input_path_check"),
        "intent_evidence_index": bundle.get("intent_evidence_index", []),
        "investigation_paths": bundle.get("investigation_paths", []),
        "cited_repository_evidence": (
            bundle.get("cited_repository_evidence") or []
        ),
    }
    schema = json.dumps(
        HypothesisMergeAdjudication.model_json_schema(), ensure_ascii=False
    )
    return f"""Adjudicate only whether grounded repair alternatives are safe to merge.

This is a recall-preserving equivalence check before Repair. The original Issue
is normative. Use only the supplied Issue/Scout/repository evidence. Do not use
future patches, hidden tests, golden implementations, or current buggy behavior
as product authority.

Scout open_decisions are an incomplete inventory, not the boundary of the
interpretation space. First compare the raw Issue with factual call/selection
evidence: distinguish the desired result from the observed failing path and
tentative author proposals. If a relevant upstream participation condition,
representation choice, receiver-specific protocol, or default-versus-opt-in
policy is still unsettled, include it as an open decision. Do not require a
baseline failure to prove a missing interpretation or forbid an additive API
because it is absent. Do not invent alternatives to an explicit requirement.
When `hypothesis_ready` is false, retain the reviewer's missing repository fact
as unresolved and do not use its absence to merge or eliminate alternatives.
The readiness value is diagnostic only, never a generation gate: complete this
adjudication and pass the strongest evidence-grounded alternatives onward for
Hypothesis and Repair whether it is true, false, or unavailable.

For each supplied or newly identified open decision that can change public behavior, compilation,
selection winner, precedence, fallback participation, repair locus, or
evidence-backed scope, describe two or three concrete alternatives. Compare
complete semantic policies, not deliberately broken syntax fragments: include
the ordinary fallback, compatibility path, or supported configuration each
policy needs to satisfy the common Issue contract. For example, an unqualified
lookup policy may need an established standard fallback; that completed policy
is the alternative, not a raw call that loses built-in or standard inputs.

Before deciding whether alternatives merge, judge `contract_viability`:
- `plausible` means the complete policy can satisfy every supplied target and
  supported build/API constraint while retaining required fallback domains.
- `contradicted` means supplied execution, language level, public contract, or
  repository evidence proves it cannot satisfy a common target or supported
  environment. A mechanism that only works after dropping an evidenced support
  level is contradicted unless the Issue authorizes that compatibility change.
- `uncertain` means it could satisfy the common contract but evidence is
  insufficient. Missing evidence is not contradiction.

Keep fallback coverage separate from coexistence precedence. Evidence that a
member-only, built-in, default, or legacy-provider input must remain supported
requires a fallback path; it does not prove that provider must still win when a
new extension/customization provider is simultaneously applicable. For an
extensible lookup, dispatch, or overload policy, retain the complete alternative
that introduces the standard/default provider into ordinary unqualified
resolution as a fallback while still allowing associated or extension
candidates to compete. Do not mislabel that complete policy as extension-only
or reject it because the deliberately incomplete extension-only spelling loses
the fallback. Compare it separately with an internal helper that suppresses
extension discovery whenever a member/default provider exists. A coexistence
scenario decides precedence; member-only and extension-only scenarios decide
fallback coverage.

When the current qualified/default-only lookup excludes associated extension
candidates, always evaluate these as distinct complete policies when the
supplied language and repository evidence make both expressible:
1. an internal helper that pre-checks a member/default provider and suppresses
   extension discovery when that check succeeds; and
2. ordinary unqualified resolution with the default provider introduced into
   the candidate set as a fallback, so default and associated extension
   candidates participate in the same ranking.
An internal helper that performs a member-presence pre-check is not equivalent
to language-level unqualified resolution merely because its final fallback call
is unqualified. Use a coexistence witness with distinguishable providers to
compare the two policies.

Apply supported-environment evidence strictly. If a policy is unavailable at
an evidenced supported language level, configuration, platform, or API mode,
mark that direct policy `contradicted` unless the Issue authorizes dropping the
environment. Adding a conditional fallback to a different viable policy does
not rescue it as an independent hypothesis for that environment; it simply
collapses to the fallback policy there.

Construct up to four evidence-grounded boundary scenarios that expose the
difference, including coexistence of competing providers when
lookup/dispatch/overload rules differ. Give each alternative a definite
predicted outcome per scenario when evidence supports one; use an explicit
unknown when it does not. A contradicted alternative may be recorded to explain
why it is discarded, but it is not a downstream Repair hypothesis.

First check that the distinction concerns the Issue's operation, input domain,
or an independently established consumer/compatibility contract. A new trait
probe invented only to expose incidental differences between implementation
styles does not establish a different repair intent. If two viable mechanisms
implement the same intended policy, keep both as implementation latitude inside
one hypothesis; do not manufacture a user requirement for their incidental
helper/member access differences. Actual default modes, error protocols and
evidence-backed public ingress differences remain meaningful alternatives.

Within those relevant distinctions, the burden of proof is asymmetric:
- `merge_safe` requires the same predicted observable/compile outcome across
  every evidence-backed boundary and a difference only in source-code shape.
- `keep_separate` applies when any supported scenario changes winner,
  precedence, fallback domain, compilation, public result, causal locus, or
  contract scope.
- `inconclusive_keep` applies when equivalence cannot be proved. Missing or
  unexecuted discriminators are uncertainty, never evidence of equivalence.
- `single_viable` applies when evidence leaves fewer than two non-contradicted
  alternatives. Keep the viable policy without manufacturing a branch.

Do not manufacture alternatives for wording, helper names, or stylistic source
shape. Do not prefer a policy merely because it resembles nearby code. Among
non-contradicted policies, do not decide which is correct; preserve distinct
plausible outcomes for downstream Repair and cross-execution. Return an empty
decisions list when the supplied open decisions contain no semantic merge
question.

Return JSON matching this schema:
{schema}

Original Issue:
{issue}

Bounded evidence packet:
{json.dumps(evidence, ensure_ascii=False)}
"""


_OBSERVATION_FIDELITY_GUIDANCE = """
Check baseline observations against the original Issue before explaining a fix.
For each hard FIX, distinguish the exact expected result, the literal observed
result on the same input/mode, and the remaining difference. In the existing
coverage/check fields, cite the decisive evidence and explain which scoped
change closes that difference. Quoting, escaping, types, ordering and error
behavior can be the entire defect; similar-looking output is not equality.
A successful command exit is not a passing output assertion. A nonzero exit
with empty output is still a failed command, not evidence that the target works;
its cause remains unknown unless the command or diagnostics establish it.
If the baseline already satisfies a subgoal, identify matching evidence before
claiming preservation, and still account for every remaining Issue failure.
Use the actual source predicate with the exact input types, qualifiers and
negation. Constructibility does not imply implicit convertibility, and one true
term does not prove an entire conjunction or actual overload participation.
An entity name or graph edge without its source body is a navigation lead,
not proof of complete control flow. Mark missing or truncated evidence unknown.
"""


_SCOUT_EVIDENCE_ID = re.compile(r"(?<![A-Z0-9_-])(?:SEARCH|RUN|GRAPH)-\d+(?![A-Z0-9_-])")


def _scout_policy_evidence(bundle: dict) -> dict:
    """Carry cited Scout conclusions without promoting graph leads to facts."""
    observed_ids = {
        item.get("id") for item in bundle.get("cited_repository_evidence", [])
    }

    def has_observation(text: str) -> bool:
        return any(match.group(0) in observed_ids
                   for match in _SCOUT_EVIDENCE_ID.finditer(text))

    checked_paths = []
    open_questions = []
    for path in bundle.get("investigation_paths", []):
        source_id = path.get("source_evidence_id")
        if (path.get("status") in {"supported", "ruled_out"}
                and source_id in observed_ids):
            checked_paths.append(path)
        else:
            open_questions.append({
                "question": path.get("question", ""),
                "missing_evidence": path.get("missing_evidence", ""),
                "evidence_ids": path.get("evidence_ids", []),
            })
    leads = [
        lead for lead in bundle.get("repair_surface_leads", [])
        if set(lead.get("evidence_ids", [])) & observed_ids
    ][:15]
    return {
        "scout_report_summaries": [
            {
                "focus_intent_ids": report.get("focus_intent_ids", []),
                "status": report.get("status", ""),
                "answer": report.get("answer", ""),
            }
            for report in bundle.get("intent_reports", [])
            if has_observation(report.get("answer", ""))
        ],
        "repository_facts": [
            fact for fact in bundle.get("repository_facts", [])
            if has_observation(fact)
        ],
        "input_path_checks": [
            check for check in bundle.get("input_path_checks", [])
            if check.get("condition_evidence_id") in observed_ids
        ],
        "investigation_paths": checked_paths,
        "open_investigation_questions": open_questions,
        "repair_surface_leads": leads,
        "open_decisions": bundle.get("open_decisions", []),
    }


def _build_policy_pool_prompt(issue: str, issue_evidence: dict, bundle: dict) -> str:
    evidence = {
        "issue_evidence": issue_evidence,
        "hypothesis_ready": bundle.get("hypothesis_ready"),
        "evidence_sufficiency_review": bundle.get(
            "evidence_sufficiency_review"
        ),
        "repository_observations": bundle.get("cited_repository_evidence", []),
        "scout_probe_snapshot": bundle.get("scout_probe_snapshot"),
        "intent_evidence_index": bundle.get("intent_evidence_index", []),
        "candidate_boundaries": bundle.get("candidate_boundaries", []),
        **_scout_policy_evidence(bundle),
    }
    return f"""Derive one to eight complete repair-policy hypotheses from the original
Issue and primary repository observations. Return IntentHypothesisPool JSON.
{load_stage_guidance("hypothesis")}
The pool covers evidence-supported alternatives in intended behavior, root
cause, repair location and repair path. The same result on the original Issue
example does not prove two repair paths equivalent. Retain distinct plausible
causal paths as separate hypotheses so Repair receives a concrete direction.
Merge only confirmed equivalent duplicates or purely stylistic variations.
A later Audit selects up to three; leave selected_hypothesis_ids empty.
Do not fill a quota or invent a capability merely to distinguish paths.
Uncertainty about which Issue-related cause or scope is correct is a reason to
branch, not a reason to erase every alternative. A concrete source/graph lead
may support a complete conditional hypothesis even when its causal premise is
not yet verified. State that premise and the decisive missing fact explicitly;
give the branch a definite repair locus and public scope. Keep the competing
local policy as another branch when it is also plausible. Do not make Repair
choose between them, and do not promote either premise to a common hard fact.
Bare adjacency, a similar name, or a speculative new capability is not enough
to create a branch.
The conditional premise concerns the cause or scope, not whether the original
Issue succeeds: every branch must still give a concrete, branch-specific action
for every shared hard FIX/CLOSURE/PRESERVE in `hard_obligation_coverage`.
Never leave that map empty or use a placeholder such as "evaluate during Repair".
If no coherent action can be stated, record the missing fact as deferred instead
of emitting an incomplete branch.
Example: if the reported public entry and a second public entry are observed
to handle the same Issue-relevant representation, but the second body is not
fully checked, keep a local branch and a conditional shared-scope branch. The
second branch names both entries and the missing source fact; neither asserts
that the second entry is already proven broken. A merely adjacent method with
no such observed connection does not earn a branch.
`intent_evidence_index` only maps an atomic Issue focus to raw observation IDs;
it does not assert that an intent is covered or that Scout's causal explanation
is correct. Inspect the referenced observations before using the mapping.
Scout report summaries and repository facts are cited interpretations, not
independent authority. Check each material causal/scope relation against its
raw observation. `open_investigation_questions` are unverified graph/source
frontiers, not proved absences or required repair surfaces; when one describes
a coherent Issue-related alternative, retain it conditionally rather than
treating missing verification as evidence against that branch.
When `hypothesis_ready` is false, the Scout packet is explicitly incomplete.
Keep the reviewer's missing repository fact visible as an uncertainty;
do not silently answer it, use it to exclude a causal path, or present the
resulting pool as exhaustive. Preserve the reviewer's grounded alternatives.
Readiness is diagnostic only, never a stop instruction. Generate the strongest
evidence-grounded pool from the available packet even when it is false or
unavailable so selected branches can proceed to Repair.

{_OBSERVATION_FIDELITY_GUIDANCE}

1. Establish the common contract from the original Issue. FIX obligations are
the requested public behavior. A failing internal operation or diagnostic input
is evidence about the bug, not a requirement to make that operation or input
supported. Author implementation suggestions are suggestions unless explicitly
required. Distinguish the original input, receiver and operation from controlled
variants added during investigation.
Do not give a diagnostic proxy an `issue_explicit` FIX obligation merely
because its run failed. Name the original Issue input separately from a proxy
with a different type, character representation, or build mode; the latter
can motivate an optional branch only if source and public contract support it.
For an executable Issue example, preserve the requested successful operation
on its original input, including runtime behavior. Removing a compile error
while retaining failure of that operation is not a complete alternative unless
the Issue explicitly permits it. Do not downgrade original-example success to
an optional interpretation merely to make a candidate satisfy the common core.

2. Derive the causal repair surface from actual definitions and observations.
Use investigation_paths as a checked inventory of candidate mechanisms and
scope, not as a prescribed hypothesis list. Recheck Scout findings against their
raw evidence and verification_kind: source/execution claims need exact quotes,
while a recorded negative search can establish bounded absence. Keep unresolved paths open when their
missing fact could change the intended repair; omitted_evidence_ids marks raw
evidence absent from this handoff, not proof that a path is wrong. A ruled_out
path rules out only its stated premise on the checked input. Graph adjacency
cannot establish desired behavior or mandatory scope. Several paths can belong
to one complete hypothesis; do not create new public behavior to separate them.
Trace the original input through selection/dispatch and the consuming operation.
An error at a consumer does not prove that the input should reach that consumer.
Check the full enabling conditions and their direction, not a predicate's name.
A currently false participation condition may itself be the bug to repair.
Distinguish a gate the proposed scope changes from one it leaves unchanged;
baseline dispatch is not a permanent restriction on the repaired dispatch.
Include required siblings, wrappers, alternate ingress forms and shipped copies
when a concrete shared faulty rule or forwarding/build dependency demonstrates
that leaving them untouched keeps the requested behavior broken. State that
dependency, the affected entries, and the counterfactual. Proximity, identical
wording and diagnostic coverage alone do not establish a required closure.
Keep a CLOSURE obligation separate from an optional scope expansion.
Scout's claim that a fix is confined to one function is not an exclusion proof:
check any observed sibling against the same faulty rule before dropping it.
For a `candidate_boundaries` lead, do not equate "does not call the fault site"
with "cannot share the fault": compare the public operation's own use of the
Issue-relevant type, state, helper, or representation. For a configurable type,
trace the sibling's parameter types, intermediates, and values it inserts or
returns; matching the exact failing expression is not the causal test. For
example, one entry may fail while converting to a fixed default text type,
while another independently constructs that default type and emits it into a
configured-text result. That is a conditional shared-scope lead, not proof that
both entries must be edited. If the observed relation
supports a plausible shared failure mode that a narrow fix could leave broken,
retain a complete broader
repair-scope hypothesis, even when the relation remains unverified. State the
branch's conditional premise and missing fact rather than silently narrowing
every branch. A graph neighbor without source or Issue relevance is only a
navigation lead.
For platform- or template-dependent types, establish the concrete instantiation
from this checkout or a compiler trace, rather than assuming another platform's
type alias. When competing lookup paths can coexist on one input, a probe with
only one path available does not establish their priority.

3. Construct H0 as the smallest complete intended policy satisfying that common
contract. Additional hypotheses represent supported alternative interpretations
of root cause, repair path or observable policy: default versus explicit opt-in
behavior, input semantics, public receiver/scope, return/error behavior or
compatibility. Preserve complete
combinations of independent policy choices when supported; do not merely list
isolated dimensions. For every evidence-supported scope option, pair it with
each compatible causal mechanism already retained in the pool. Do not attach a
family-wide or alternate-entry scope only to one mechanism while another
mechanism is represented only by its narrow scope. This applies only when the
supplied source evidence completely enumerates that scope and demonstrates the
shared forwarding, construction, or error-propagation rule; similar names or
proximity are insufficient. Every branch satisfies every hard FIX/CLOSURE/PRESERVE.
For each hard obligation, hard_obligation_coverage names a concrete result or
action already in that branch. Do not make speculative feasibility a mandatory
capability. Each distinct repair path needs its own evidence and concrete scope;
do not add a public API requirement to make its output differ artificially.

4. A branch's assumption, root_cause, repair_scope and expected_behavior must
describe the same policy. Scope must name the public entry forms and required
source closure; it must not silently narrow the assumption. Keep root-cause
facts separate from proposed changes. Preserve established behavior within its
actual supported domain: an overload present in source does not prove every
type or runtime representation can use it. Diagnostic variants are not new
supported inputs. Exact output strings/error codes require evidence for that
exact input and state. Baseline tests alone are soft preservation evidence;
only Issue/public-contract authority supports a hard PRESERVE. Intentional
behavior changes can change incidental baseline outputs.

5. Record unresolved dimensions. Multiple hypotheses may share the same desired
public outcome while representing different supported causal repair paths.
They need not manufacture different post-repair outputs. If no defensible
discriminating observation is available, retain the candidate paths and record
the missing evidence in a deferred dimension naming them; do not silently merge
them or delegate choosing between those paths to Repair.
Defer a material lead out of every hypothesis only when its observed relation
is too weak to state a coherent conditional policy, not merely because its
premise is unverified.
A branched dimension needs at least two actual
hypotheses and one concrete candidate-independent public scenario with a
different definite outcome for the compared policies. Specify the same input,
receiver, state, requested operation and mode for every branch. State the
observable difference implied by the policy, not an invented implementation
detail. A scenario can describe the public operation without guessing the C++
spelling of an API that does not yet exist. Include code only when its types,
input shape and call signature are established. Proposed API additions must
account for the existing signature; a call unavailable under a policy cannot
have a runtime result under that policy. Never introduce an extra parameter,
conversion, output promise or internal-member capability merely to separate
implementations. Do not infer an exact compile result from name visibility.

For a hypothesis assigned to a branched dimension, leave its branch-local
discriminator/predicted_outcome empty: runtime retains only the authoritative
shared dimension, so Audit and Repair receive one scenario/outcome mapping.
Use branch-local fields only when no branched dimension is assigned.
Apply each branch's policy to the SAME scenario; do not substitute a
different call or fixture to rescue its prediction. Do not guess unknown
outcomes: keep an unresolved evidence gap deferred with the precise missing
observation, rather than converting it into a false contract. This does not
establish equivalence or disprove an alternative. Known policy differences
must remain represented; soft_constraint is for behavior-equivalent latitude.

6. Check the complete result before submission: no invented authority, no
diagnostic promoted to mandatory behavior, no branch contradicting its assigned
outcomes, and no missing hard obligation or demonstrated dependency. Include
one surface_dispositions decision per supplied SURFACE-N lead, with the actual
action in each included branch; leads are not requirements by themselves.
Exclude a source-backed sibling only with evidence against its proposed shared
failure mode; an absent direct call is not enough. A negative search for one
code spelling does not rule out a different use of the same Issue-relevant representation.
If its relevant body still
needs inspection, mark the premise unverified and retain a coherent conditional
scope branch rather than presenting every hypothesis as local. `included` means
represented in a branch, not proven mandatory in every repair. A branch that
includes an independent public operation must account for
its internal production and consumption of the Issue-relevant representation,
not just change its public signature.
Use exact public identifiers and evidence IDs. Do not ask Repair to revise a
hypothesis; it implements the selected intended policy using repository tools.

All required schema fields, including titles, statements, impacts and reasons,
must be present. Do not return an Audit or an explanation.
Schema:
{json.dumps(IntentHypothesisPool.model_json_schema(), ensure_ascii=False)}

Original Issue:
{issue}

Primary evidence (observations are facts, not independent product authority):
{json.dumps(evidence, ensure_ascii=False)}
"""


def build_hypothesis_prompt(
    *,
    issue: str,
    issue_evidence: dict,
    repair_intent_bundle: dict,
    repair_intent_contract: dict,
    scout_summary: str,
    merge_adjudication: dict | None = None,
    expanded_pool: bool = False,
    policy_prompt: bool = False,
) -> str:
    if expanded_pool and policy_prompt:
        return _build_policy_pool_prompt(issue, issue_evidence, repair_intent_bundle)
    portfolio_schema = IntentHypothesisPool if expanded_pool else IntentPortfolio
    schema = json.dumps(portfolio_schema.model_json_schema(), ensure_ascii=False)
    pool_guidance = """
Generate a recall-oriented pool of up to eight complete hypotheses. Three is
the later Repair budget, NOT the limit on expressing plausible interpretations.
Do not fill the pool with low-evidence inventions or near-duplicate variants.
Represent evidence-supported combinations of unresolved dimensions as complete
branches, including receiver-specific return/error contracts. Do not assume a
single return policy must apply identically to an object and its result wrapper.
Preserve a plausible competing causal explanation when a failing downstream
operation could instead be avoided by repairing upstream selection or state.

Re-derive the common hard core from the original Issue and cited public
contracts. Scout targets, preserves, probes and merge verdicts are evidence
summaries, not independent authority. A shared causal guess must not become a
hard constraint just because every current branch repeats it. Before excluding
an alternative as contradicted, identify the independently authorized common
obligation its COMPLETE repaired policy necessarily violates. Baseline failure,
an absent API, a legacy prohibition or an added diagnostic witness alone cannot
prove that a proposed new behavior is forbidden. Reconsider a merge exclusion
whose only support is one of those facts, preserving the uncertainty explicitly.
Keep exact public identifiers and source paths; never normalize substrings in
API/type/header names. Missing evidence of equivalence preserves alternatives.
Leave selected_hypothesis_ids empty; a separate budget decision selects Repairs.

Check the completeness of the EXPLANATIONS before forming cross-products of
Scout's existing axes. For each candidate cause, inspect the cited upstream
gate and actual input representation: an error at the selected consumer does
not establish that consumer as the correct path. Separate raw representation
from decoding, each receiver's return/error protocol, default behavior from
an explicit option, and tentative author suggestions from requirements when
those distinctions are relevant. Preserve a supported complete alternative
missing from Scout's list; do not freeze a diagnostic witness as shared intent.

Within one hypothesis, branch_assumption defines the policy; repair_scope and
expected/preservation behavior must implement that SAME policy. Dimension
outcomes apply it to their named scenario; they may not introduce a new policy.
Read the complete branch with ALL assigned outcomes before submitting. In
particular, a soft baseline output must not become mandatory in a discriminator
when the new policy can legitimately change it. One callable C++ signature has
one return type; runtime data cannot choose whether a call compiles. A claimed
distinguishing scenario must activate the dimension being compared, with the
same receiver, input, mode and observation for every named hypothesis.
State only the observation needed to separate policies, not a checklist of
unproved compile/output promises. Public accessibility does not remove a base
member's type constraints; name lookup and call viability are different checks.
Keep alternative implementations of the SAME intended observable policy within
one hypothesis. A compiler diagnostic naming an internal member does not make
direct calls to that member on newly invented receiver types a hard FIX. Ground
the hard contract in the enclosing public operation requested by the Issue;
retain implementation suggestions as flexible seeds. Do not add an incidental
API/trait requirement merely to distinguish those source edits.
Do not turn an unexecuted language-level prediction into a universal hard
contract. Separate the Issue's required enclosing operation from an auxiliary
member/trait expression used to investigate it. If its exact compile result or
output is unknown, keep that uncertainty explicit and leave the extra probe
out of mandatory branch outcomes. Execution of the original failure is not
execution of the proposed predicate or the post-repair expression.
Use the exact public type and actual value category in every fixture. If an
observation cannot distinguish the policies, merge that implementation detail
or mark the dimension soft/deferred instead of inventing different outputs.
For a public operation family named in a branch, explicitly say which ingress
forms expose its new policy; do not name the entire family in the assumption
and silently restrict repair_scope to the one diagnostic entry.
""" if expanded_pool else ""
    evidence = {
        "issue_evidence": issue_evidence,
        "hypothesis_ready": repair_intent_bundle.get("hypothesis_ready"),
        "evidence_sufficiency_review": repair_intent_bundle.get(
            "evidence_sufficiency_review"
        ),
        "repair_intent_bundle": repair_intent_bundle,
        "repair_intent_contract": repair_intent_contract,
        "scout_summary": scout_summary,
        "merge_adjudication": merge_adjudication or {"decisions": []},
    }
    if expanded_pool:
        # Generate from primary observations, not a second retelling of the
        # Scout's causal conclusion and its derived target/preserve clauses.
        evidence = {
            "issue_evidence": issue_evidence,
            "hypothesis_ready": repair_intent_bundle.get(
                "hypothesis_ready"
            ),
            "evidence_sufficiency_review": repair_intent_bundle.get(
                "evidence_sufficiency_review"
            ),
            "repository_observations": repair_intent_bundle.get("cited_repository_evidence", []),
            "intent_evidence_index": repair_intent_bundle.get(
                "intent_evidence_index", []
            ),
            **_scout_policy_evidence(repair_intent_bundle),
        }
    return f"""Build a small portfolio of complete repair-contract hypotheses.

{load_stage_guidance("hypothesis")}

{pool_guidance}

If `hypothesis_ready` is false, the Scout packet is explicitly incomplete.
Keep the reviewer's missing repository fact visible as a deferred uncertainty;
do not silently answer it, eliminate a causal path from its absence, or present
the generated portfolio as exhaustive. Preserve grounded alternatives named by
the review.
Readiness is diagnostic only, never a stop instruction. Generate the strongest
evidence-grounded portfolio from the available packet even when it is false or
unavailable so its selected branches can proceed to Repair.

{_OBSERVATION_FIDELITY_GUIDANCE}

The original Issue is normative. Repository evidence may add repair surfaces
only through the evidence-backed relations defined below. It may not silently
turn current buggy behavior, proximity, or a neighboring implementation pattern
into product intent.

Do not turn a failure trace into post-fix intent. A foreign value, callback,
provider, overload, route, or intermediate result observed on the failing path
is causal evidence, not behavior that the repair must preserve or make
acceptable. When the Issue requires only an end result, but a Scout-derived
delta additionally says that the currently selected foreign path or result must
be accepted, keep the Issue result as hard and demote the unsupported path claim.
The portfolio must then retain a hypothesis that repairs the upstream selection
so the library's established path reaches the requested result.
Absence of an Issue-specified selection winner is not positive authority to
broaden the downstream consumer. A downstream-acceptance hypothesis is
admissible only when an explicit Issue clause or direct public contract says
that consumer accepts the foreign result family. Otherwise keep the concrete
upstream selection outcome flexible within one upstream hypothesis and do not
branch on an unauthorized downstream contract. The downstream compile error is
still useful localization evidence.
Likewise, a Scout may add controlled variants to diagnose value category,
provider coexistence, routing, or precedence. A variant not present in the
Issue is diagnostic unless direct public-contract evidence authorizes it; do
not copy it into a hard FIX merely because it appears in an executed target.

First extract atomic obligations:
- FIX: requested observable behavior and exact public API contract when the
  Issue or repository evidence supports it.
- CLOSURE: the minimum additional repair surface required by an evidence-backed
  relation and its counterfactual necessity. Do not prescribe exact
  golden-looking code.
- PRESERVE: behavior that must remain. Mark it hard only when the Issue says so
  or an established public contract requires it. Existing/basic-test behavior,
  execution alone, sibling analogy, and model inference are soft evidence and
  must never become a hard veto.

A legacy test's exact expected string, snapshot, fixture value, or assertion is
verification evidence, not by itself a product obligation or hard CLOSURE.
Express the underlying Issue/public behavior as the obligation and keep
mechanical pin synchronization soft. Promote a test artifact itself only when
the Issue or an established public contract explicitly makes that artifact a
deliverable. A failing old pin may be the expected evidence of an intentional
behavior change.
Do not put bulk synchronization of legacy pins into a hypothesis's repair_scope.
Record the predicted old-test conflict in soft evidence or the branch outcome
instead. A bounded candidate-authored test edit may be used only to exercise the
new behavior; it is verification evidence and is excluded from submission.

The Scout evidence may contain a numbered `repair_surface_leads` list. Return
exactly one `surface_dispositions` item for every supplied SURFACE-N ID. Mark a
lead `included` only when Issue scope plus its cited dependency make it part of
one or more hypotheses; name those hypotheses and the concrete causal reason.
Mark it `excluded` when it is localization, analogy, or an unrelated surface,
and explain the source or execution evidence against the proposed connection;
an absent direct call is not enough when the lead concerns a shared type,
state, helper, or packaged representation. Use `deferred` for a material
unresolved reachability or shared-cause question; preserve a supported broader
branch when the observed source already shows the same failure mechanism.
A disposition is not itself a
hard obligation, but silence is forbidden: this checklist prevents discovered
public entries, wrappers, seeds, and downstream symptoms from disappearing
between Scout and Repair.

Derive repair-surface closure from evidence-backed relations. Use these reusable
patterns; they describe how to reason, not which APIs or language mechanisms a
patch must contain:
1. Shared-cause equivalence: expand from the Issue example only to forms proven
   to use the same faulty rule or to implement the same explicit public
   contract. Similar names, neighboring code, or membership in a broad family
   are not enough.
2. Dependency propagation: follow a data, control, call, type, or build edge
   only when leaving the connected node unchanged would keep the Issue failure,
   break compilation/linking, or violate an established public contract.
3. Synchronized representations: include a derived, generated, packaged, or
   alternate representation only when repository evidence shows that it is a
   shipped or independently built representation of the changed contract.
4. Compatibility boundary: when the Issue changes a public contract, identify
   affected consumers and supported modes from observed repository usage. Do
   not invent a generic checklist of language operations, failure sources, or
   configurations that the evidence never connects to the change. For a public
   return-type, wrapper, trait, or customization change, turn each distinct
   existing call shape actually found in the repository into compatibility
   evidence. A new target-only call cannot prove that unchanged callers still
   compile or retain their supported access and transfer behavior.
5. Conjunctive causality: when evidence shows independently necessary causal
   sites, a complete branch includes all of them. When independence is still
   uncertain, represent that uncertainty as alternatives or an unresolved
   dimension rather than silently widening H0.
6. Competing resolution semantics: when plausible lookup, dispatch, overload,
   precedence, fallback, or provider-selection mechanisms differ only when two
   providers are simultaneously applicable, branch that uncertainty with a
   conflict witness. Make the providers return distinguishable values or take
   distinguishable paths, then state exactly which provider wins in each
   branch. A witness where only one provider exists cannot distinguish these
   mechanisms.
7. Selection point before symptom point: when an extensible lookup, overload,
   dispatch, conversion, callback, or routing decision selects a foreign result
   or path and a later consumer rejects it, do not assume the consumer is the
   root cause. Trace the exact Issue witness through both points. If changing
   the earlier evidence-supported candidate set, participation rule, ranking,
   or route can retain the intended path, preserve at least one complete
   upstream-selection hypothesis. A downstream adapter/generalization is a
   separate hypothesis only when the Issue or public contract plausibly
   authorizes the consumer to accept that broader result family. State for each
   branch which path is selected before and after the repair and why that makes
   the exact witness reach the established contract.
   Do not hide this causal choice inside a disjunctive scope such as "repair
   selection or make the downstream consumer accept the foreign result". Those
   alternatives change different contracts and must be separate hypotheses;
   the downstream alternative needs its own Issue or public-contract authority.
8. Fallback-domain preservation: a competing-provider discriminator establishes
   precedence only for its coexistence input. Separately inspect the old
   selection path's evidence-backed caller domain and preserve supported inputs
   that have no competing provider, such as built-in, member-only, or standard
   library forms found in repository usage/tests. Do not turn these examples
   into a generic hard checklist; create PRESERVE obligations only for categories
   connected to the changed public operation by supplied repository evidence.
9. Selection-chain closure: for lookup, overload, dispatch, conversion, or
   routing defects, trace the exact Issue witness through every library-owned
   selection point in the enclosing public expression or call path. Fixing an
   inner sibling family is incomplete when an outer entry, wrapper, adapter, or
   forwarding layer can still be hijacked by the same competing rule. Include
   such a point only when the exact witness or repository usage proves it is on
   the path; do not expand to unrelated wrappers by analogy. Preserve at least
   one evidence-backed supported input through the complete chain, rather than
   validating only a reduced inner operation. For an extensible public API,
   explicitly inspect the first library-owned ingress or seed operation that
   introduces a user-associated value into that chain. When the same observed
   competing-provider rule can hijack that ingress, represent its repair or
   preservation separately from the inner sibling operation; an inner-only
   fixture cannot establish complete closure. Also inspect intersections
   between an Issue-authorized operation family and the library's public ingress
   machinery: if one family operation is itself used to seed or enter the public
   chain, and repository usage admits the same user-associated operand there,
   leaving that entry unchanged violates the family contract even when the
   original example first exposes an inner operation.
10. Diagnostic argument-role closure: when an Issue asks an error or diagnostic
   to identify which argument role, key kind, operand, option, or input form is
   invalid, inspect whether repository evidence shows one shared diagnostic
   construction rule across multiple public overloads or argument roles. If it
   does, and the Issue example cannot decide whether only its trigger or the
   whole shared rule should gain that context, retain both observable scopes as
   hypotheses. Use one shared matrix containing the Issue trigger and at least
   one case for every distinct argument or operand role evidenced by the
   candidate public forms; predict the exact role distinction each branch
   reports. One sibling role cannot stand in for a larger overload family. Do
   not confound role with receiver or surrounding context: when the public forms
   permit it, include a counterfactual pair that holds the receiver/context
   constant while varying the argument or operand role, in addition to the
   Issue trigger. Otherwise a receiver-specific special case can masquerade as
   argument-role closure. Do
   not broaden merely for consistent wording: without both a shared construction
   rule and distinct evidenced public roles, keep the repair at the Issue
   trigger.
11. Artifact-route closure: distinguish the editable source from the concrete
   representation compiled, linked, imported, packaged, or shipped by the
   available verification route. When supplied repository/build evidence shows
   that a generated, amalgamated, vendored, or mirrored artifact independently
   carries the changed public contract, include the evidenced synchronization
   edge as CLOSURE or explicitly defer it. A green regression run against an
   unchanged alternate representation is not evidence that the edited source
   works, and a Finish summary cannot establish synchronization that is absent
   from the frozen production diff. Do not require mirrors by convention: the
   repository/build edge and independent use must both be evidenced.
12. Equivalent access-form closure: when repository evidence shows that the
   same public operation is exposed through paired access forms (for example
   mutable/const, lvalue/rvalue, synchronous/asynchronous, or mode-specific
   wrappers), include every form that accepts the Issue's same receiver/input
   relation and repeats the same faulty construction rule. State the concrete
   counterfactual: leaving one form unchanged reproduces the Issue behavior
   through that form. Do not infer closure from a qualifier or neighboring
   signature alone; require both the shared public contract and shared-cause
   repository evidence.
13. Baseline-value fidelity: an exact preserved string, value, selected path,
   or byte-identical outcome in a discriminator must be cited from execution or
   a repository pin for that same receiver/input/provider scenario. Do not copy
   an exact value from a neighboring role or infer it from implementation
   intuition. When same-scenario evidence is absent, state only the supported
   semantic relation or mark the exact baseline outcome unresolved.
14. Existing-contract promotion: when the Issue asks to expose or query a
   capability and repository evidence already contains a non-public predicate,
   alias, or operation with the requested semantics, retain a hypothesis that
   promotes that established contract while preserving its evidenced name,
   parameters, and result semantics. Invent a different public vocabulary only
   as a real alternative with an observable reason; do not discard the closest
   existing contract merely because a new API also satisfies a reduced probe.
15. Compositional behavior: when two public modifiers, modes, or state flags
   are simultaneously active, treat their effects as conjunctive unless Issue
   or public-contract evidence says one disables the other. A candidate that
   honors a new type or mode by bypassing the existing modifier has not covered
   the combined scenario. Trace the established shared renderer, state update,
   or accounting path and use a witness where both effects are observable;
   do not branch between composition and silent disabling based only on the
   current buggy shortcut.

Mandatory family-ingress intersection audit: for every hard operation family,
inspect the supplied candidate boundaries and complete enclosing definitions
for a family operation that is also used by the library to seed, enter, or
route the public expression/call. When that generic entry accepts the Issue's
public input domain, the family scope plus repository reachability are enough
to require a separate CLOSURE; the original example need not fail at that entry
first. A branch that lists only inner family members while omitting such an
evidenced entry is incomplete. Do not infer entries that are absent from the
supplied repository evidence.

Mandatory artifact-route audit: for every supplied packaging/build fact, check
whether the hypothesis names the representation actually exercised by its
verification route and disposes of every evidenced independently built or
shipped mirror. Require revision when a hypothesis relies on a green run that
compiled an unchanged alternate representation, or claims synchronization
without a corresponding production hunk or an executed repository generator.
Do not invent this requirement when the supplied evidence contains no such
route or mirror.

Mandatory equivalent-access audit: for each hard public operation, inspect the
supplied complete definition/candidate boundary for paired access forms that
accept the same Issue receiver/input relation. When evidence shows those forms
share the faulty construction or contract, require every hypothesis to cover
them or explain executed evidence that makes one unreachable. Reject a branch
that fixes one form while the unchanged equivalent form reproduces the same
Issue behavior. Do not manufacture const/mutable, value-category, async, or
mode closure without the two evidence conditions.
Require focused verification to exercise every access form promoted into that
hard closure; a mutable-only witness cannot prove the paired const form, and
vice versa.

Apply a counterfactual necessity test to every CLOSURE obligation. Its
evidence_basis must identify the relation and explain the concrete failure that
remains if that surface is left unchanged. A repository_structure closure is
hard only when the supplied evidence establishes that dependency; analogy,
symmetry, naming, stylistic consistency, or a list of possible risks stays soft
or unresolved. Enumerate only the evidence-justified nodes, not an entire family
by default.

An Issue-author-proposed implementation may seed a mechanism alternative, but
it is not itself a product obligation unless the Issue explicitly requires that
mechanism. Do not turn a suggestion into a common hard CLOSURE, and do not force
it into the portfolio solely to reproduce the author's code shape. Judge every
branch by the observable contract and evidence-backed closure above.
Separate implementation shape from stated behavioral breadth. When the Issue
explicitly says that the requested behavior applies to a named family of
operations, modes, or public entry points, that family-wide scope is Issue
authority even if it appears beside a proposed implementation. Preserve the
named breadth while leaving the proposed signatures, helpers, and source shape
optional. This is different from expanding to siblings merely because they are
nearby or similarly named.
Also retain family breadth when two independent facts coincide: the Issue says
the same defect or proposed repair applies to the named sibling operations, and
repository evidence shows those siblings repeat the same faulty participation,
dispatch, propagation, or state-update rule. The Issue statement supplies the
intended breadth and the repository supplies shared-cause evidence. Promote the
family scope, not the author's concrete recipe. When both facts are present,
this breadth is a common hard CLOSURE, not a branch dimension; do not call it
open merely because the breadth statement appears beside a proposed fix or an
earlier Scout summary called it uncertain. An explicit statement that the same
defect or repair applies "similarly", to named siblings, or to all other members
of a family is breadth authority; it does not need to appear under an Expected
Behavior heading. Separate that applicability statement from the optional code
recipe surrounding it. Without both facts, keep sibling expansion soft.
For a family closure, require the evidence to enumerate the concrete family or
to cite a complete enclosing definition. Do not treat "and siblings", "and
similar", a visibly truncated source range, or a partial member list as proof
that the tail was inspected. Ask for revision or leave the unenumerated tail
unresolved instead of claiming complete hard coverage.
Do not prescribe an exact overload signature, trait gate, helper shape, call
qualification, or routing primitive as the branch contract unless supplied
execution or direct language/repository evidence proves that exact mechanism
selects the required path. For an unproven implementation suggestion, constrain
the causal locus and observable before/after selection instead, leaving Repair
free to establish the concrete candidate ranking with the compiler or runtime.
Never claim that one overload or provider wins from signature intuition alone.

Reconcile conflicting evidence explicitly. A compatibility warning, sibling
mechanism, or current implementation does not establish a hard preservation
rule when the supplied contract also marks that same winner, precedence, scope,
or behavior as an open decision. In that situation demote the one-sided rule
and, when the choice changes observable behavior or compilation, retain one
complete hypothesis for each plausible outcome. Lack of normative evidence for
either winner is the reason to branch; it is not permission to delete one side
or silently choose the current behavior.

Return {'one to eight' if expanded_pool else 'one to three'} hypotheses. H0 is the smallest evidence-complete repair
contract, not the narrowest symptom patch or the broadest plausible cleanup.
This is the raw behavior-hypothesis phase. Do not merge alternatives merely
because the currently supplied Issue probe exercises only one provider or
cannot distinguish them. For every `merge_adjudication` decision marked
`keep_separate` or `inconclusive_keep`, retain separate hypotheses for its
non-contradicted alternatives and carry one boundary scenario into a shared
discriminator. Never create a Repair branch for an alternative marked
`contract_viability=contradicted`; mention its exclusion only in the dimension
reason. A `single_viable` decision produces one hypothesis for its sole
non-contradicted policy. `merge_safe` alternatives may share one
mechanism-flexible hypothesis.
Every hypothesis
must cover every hard FIX, CLOSURE, and PRESERVE obligation, so every branch is
independently capable of a complete repair. Alternatives may differ on a
genuinely unresolved observable behavior, root cause, mechanism, API scope, or
compatibility realization. Every unresolved dimension marked branched must
define one
shared_discriminator that is executed unchanged for every named hypothesis,
plus one definite outcome per named hypothesis on that exact same input and
setup. The shared fixture must make all competing paths simultaneously
applicable and observably distinct; branch-local examples may supplement it but
are optional and cannot replace it. Separate unresolved dimensions may use separate shared
fixtures. A predicted outcome must be definite and observable: do not use may,
might, could, possibly, unknown, or merely say that branches differ. Do not
create alternatives merely to reach a count.
Absence of a supplied discriminator is not evidence of behavioral equivalence.
When plausible mechanisms differ on any repository-supported input in
selection winner, precedence, fallback participation, compilation, causal
locus, public scope, or public behavior, retain separate hypotheses and
construct one shared discriminator. Merge only when the mechanisms predict the
same outcome across every evidence-backed boundary and differ solely in source
shape. Do not manufacture a branched dimension for source-shape alternatives. A
dimension marked branched must name at least two real hypotheses. Otherwise
classify the uncertainty as a soft constraint or deferred implementation choice.
When branch capacity is limited, prioritize distinct observable/compile
outcomes over stylistic, structural-only, or behavior-equivalent scope variants.

For every hypothesis, hard_obligation_coverage must map every hard obligation
ID to a concrete action or result already present in that branch's repair_scope
or expected behavior. Treat the mapping as a causal proof: mentally apply only
that branch's scoped edits and confirm the Issue would be fixed and every hard
closure would hold. Never claim coverage by repeating the obligation text or by
saying a surface will be inspected or audited.

Treat any evidence-backed relation whose inclusion changes observable behavior,
compilation, or an established public contract as high impact. Every unresolved
dimension must
be marked branched, soft_constraint, or deferred with a reason; never leave a
high-impact item as an informational note that no hypothesis handles.
A high-impact observable scope, winner, precedence, or compatibility choice is
not a soft constraint: when two plausible outcomes can be stated on one shared
scenario, retain them as hypotheses even if checked-out tests favor the current
outcome, because those tests are preservation evidence rather than Issue
authority. Use deferred only when the supplied evidence cannot define a shared
discriminator, and name that missing evidence in the reason. Reserve
soft_constraint for low-impact wording, style, or behavior-equivalent latitude.

Before returning, verify that at least one coherent hypothesis covers the full
evidence-supported contract, that in fact every branch contains all hard
obligations, and that no hard preserve contradicts a branch. Evidence basis
entries must identify Issue wording or supplied evidence; do not invent
citations.
Also verify that `surface_dispositions` covers every supplied SURFACE-N exactly
once. For each included lead, its named hypotheses must contain the corresponding
action or preservation result in `repair_scope`; merely citing the lead is not
coverage.

Return JSON matching this schema:
{schema}

Original Issue:
{issue}

Shared Scout evidence:
{json.dumps(evidence, ensure_ascii=False)}
"""


def build_portfolio_coverage_audit_prompt(
    *,
    issue: str,
    issue_evidence: dict,
    repair_intent_bundle: dict,
    repair_intent_contract: dict,
    scout_summary: str,
    merge_adjudication: dict,
    portfolio: IntentPortfolio,
) -> str:
    expanded_pool = isinstance(portfolio, IntentHypothesisPool)
    audit_schema = HypothesisPoolAudit if expanded_pool else PortfolioCoverageAudit
    schema = json.dumps(audit_schema.model_json_schema(), ensure_ascii=False)
    selection_guidance = """
For every decision, rank one to three hypotheses jointly for the Repair budget whenever
the review provides a defensible ordering. Fewer than three is valid when only one or
two branches remain coherent; never manufacture a third branch to fill the budget.
This is listwise portfolio selection, not a verdict that only these hypotheses are true.
Consider evidence support, a complete executable contract and marginal coverage
of causal explanations, behaviors and public-interface combinations. Avoid
spending every slot on overlapping variants of the same explanation when a
comparably supported alternative covers a different plausible interpretation.
An Issue-related conditional branch with an unverified premise can provide more
marginal coverage than another well-supported variant of the same local repair.
Two branches that differ only in the local implementation mechanism have less
marginal scope coverage than a source-backed, coherent conditional alternative.
When reviewing a sibling that uses the same configurable representation, compare
its signature-to-output type flow; absence of the original failing expression
alone does not refute a distinct representation mismatch in that sibling.
Do not demand proof that its premise is true before ranking it for Repair; keep
the uncertainty visible and reject it only for a concrete contradiction or an
incoherent repair policy. Do not select a bare graph neighbor for diversity.
Do not select an independently contradicted policy merely for diversity; report
its material contract defect in the coverage audit so the pool can be revised.
Use the raw Issue and factual evidence to review Scout-derived hard clauses and
merge exclusions; their labels do not establish independent product authority.
An unexecuted discriminator is not a reason to erase a plausible interpretation.

Use the aligned complete-policy cards below, not a count of covered individual
dimensions. Each card's ENTIRE vector of assumptions and assigned outcomes is
one executable choice. Selecting (A,X) and (B,Y) does NOT execute (A,Y) or (B,X).
Do not call a deferred cross-combination redundant just because each factor
appears somewhere in the selected set. Compare its whole policy and supporting
evidence with the whole policies you actually select. Joint coverage may be
sacrificed for stronger support under three slots; record that loss honestly.

Complete hypothesis_reviews for EVERY hypothesis before ranking:
- premise_status is supported, unverified or contradicted. Substitute the
  fixture's actual types/input into the cited predicate. A concrete false
  predicate that the proposed scope leaves unchanged but still requires, or a
  wrong accepted representation under that scope, is contradicted. If the scope
  repairs that participation condition, distinguish the observed baseline from
  the proposed condition; baseline failure alone does not disprove the repair.
  A suggestion working on another input is not proof. Missing evidence alone
  is unverified.
- assigned_outcome_status is contradicted whenever a mandatory observation
  disagrees with the supplied code or the policy elsewhere in the card.
- premise_check starts from the raw Issue's actual input/receiver and quoted
  code. Evaluate the relevant participation predicate with those concrete types
  and the input representation accepted by the proposed consumer. Do not cite
  the candidate's root_cause or hard_obligation_coverage as proof of feasibility;
  these are claims under review. A negated condition must be evaluated with its
  negation; const subobjects do not become assignable by avoiding outer-object
  assignment. Unknown code is a missing premise, not confirmed success;
- hard_contract_check compares scope and behavior with each shared hard clause,
  including whether that clause has independent Issue/public authority;
- assigned_outcome_check compares the hypothesis's own expectations/preserves
  against ALL assigned outcomes shown in the same card. Quote the conflicting
  clauses when a soft preservation becomes mandatory elsewhere;
- implementation_feasibility_check checks the public input/receiver, upstream
  participation constraints, return/error protocol, lifetime and timing. Use
  supplied code facts; do not assume a proposed mechanism works because its
  title says it does. Missing implementation evidence is uncertainty, not by
  itself a contradiction.
Record concrete contradictions in conflicts even if the branch contains a
correct edit. A supported branch cannot simultaneously require incompatible
outputs or signatures. Do not let a correct-looking mechanism mask that.
Record an intentional loss of soft evidence in soft_tradeoffs, not conflicts.
This includes material losses explicitly permitted by the branch's policy.
A soft clause repeated as a mandatory commitment in that same branch still
binds the branch: violating that commitment is a conflict, not a soft tradeoff.

Review deferred/soft dimension statements and reasons as part of the eventual
Repair assignment. Uncertainty about which complete causal path or policy is
right stays with generation and ranking; each Repair receives an already chosen
branch. If a note asks Repair to investigate and choose BETWEEN candidate paths,
causal loci or policies before implementing, report the dimension ID in
internal_conflicts. Correct that delegation upstream while retaining supported
alternatives, rather than merging them or imposing one supposedly proven cause.
Repair may choose concrete code, predicates and helpers WITHIN its assigned
locus and behavior. Missing proof of that implementation is not itself a
contradiction. A deferred evidence gap must not become a new Repair task.

Complete discriminator_reviews for EVERY branched dimension: compare the SAME
scenario across hypotheses; show why it triggers the stated policy difference.
Name exactly two compared_hypothesis_ids and their policy_difference for THIS
dimension. Where the matrix permits, choose two policies that agree on the
other dimensions. Work out each result on the supplied input before concluding
they differ. If both support the input, it cannot distinguish narrower versus
broader support: mark same_outcome and suggest a changed input only as revision
feedback. Never evaluate that replacement as if it were the original. Different outcomes due
to another axis do not validate this dimension's claimed discriminator.
Flag a fixture that never exercises the differing behavior, a runtime input
purporting to change static callability, a default-vs-explicit-mode mismatch,
or observations inconsistent with the receiver/state actually constructed.
No execution yet means predicted, not verified; it does not require discarding
a coherent, genuinely distinguishing scenario.
Copy reviewed_scenario from this dimension's shared_discriminator VERBATIM.
Copy compared_outcomes from the two original predicted_outcome fields VERBATIM.
Runtime checks these against the pool, including after revision. These are
claims being reviewed, not observations you may rewrite. In comparison, evaluate
only what that original fixture observes. Ignore off-fixture clauses such as
'would succeed with another option': default parse and flagged parse are
different calls. If the original strings imply the same actual result despite
different explanations, mark same_outcome. If a prediction is false, mark
invalid_fixture or record its contradiction; do not quietly correct it.
Set scenario_status to distinguishes ONLY if the unchanged fixture observes a
difference on THIS axis; same_outcome if both behave alike; invalid_fixture if
it names a wrong type/API or incompatible setup; undetermined if no defensible
prediction can be made. Non-distinguishing/invalid/undetermined dimensions must
be repaired, merged or deferred upstream. Runtime enforces these judgments even
if decision says ACCEPT and conflicts is empty. A predicted but well-supported
distinction may still be distinguishes without execution.
Review the literal fixture: do not silently correct identifiers or substitute
a more favorable type, input or mode while describing an ACCEPT verdict.
Put only concrete contradictions in conflicts, NEVER a note saying 'unverified'
or 'no execution yet'. Keep uncertainty in the comparison/check explanations.
If a shared hard clause wrongly converts a diagnostic sample into a universal
target, report that clause for revision. Do not reinterpret its words silently
to declare every branch compliant. Prefer branches unaffected by that defect;
if none exist, preserve the conflict as explicit Repair advisory instead of
pretending the portfolio is coherent or stopping the workflow.

Return selected_hypothesis_ids in budget-priority order, selection_reasons keyed
by every selected ID (support plus incremental coverage), and unselected_reasons
keyed by every other ID. Unselected means budget-deferred, not disproven.
In joint_policy_losses, record every unselected ID's complete policy combination
that will not be executed. Do not report factor-wise union as zero policy loss.
Do not rewrite hypotheses, combine pieces from different branches, rename IDs
or refer to patches/hidden tests. If REVISE is needed, complete the semantic
review tables and concrete revision feedback, then still rank one to three
non-contradicted branches when any exist. REVISE is diagnostic and may trigger
one upstream revision, but it is not a gate that disables Repair. If every
branch has a defect, rank the least contradicted complete branch as a best-effort
starting point and state the defect plainly; do not invent support or suppress
a real defect merely to produce a ranking.
Keep each check/comparison under 60 words, cite only decisive facts, and avoid
restating entire hypotheses. Return every required schema field, including
empty maps/lists. Detailed private analysis must not inflate the audit payload.
""" if expanded_pool else "Do not select which plausible hypothesis is the\ntrue intent and do not rank hypotheses."
    evidence = {
        "issue_evidence": issue_evidence,
        "repair_intent_bundle": repair_intent_bundle,
        "repair_intent_contract": repair_intent_contract,
        "scout_summary": scout_summary,
        "merge_adjudication": merge_adjudication,
    }
    if expanded_pool:
        # Review primary observations and source-cited investigation leads;
        # their supported/ruled_out labels remain claims to check, not authority.
        evidence = {
            "issue_evidence": issue_evidence,
            "hypothesis_ready": repair_intent_bundle.get(
                "hypothesis_ready"
            ),
            "evidence_sufficiency_review": repair_intent_bundle.get(
                "evidence_sufficiency_review"
            ),
            "repository_observations": repair_intent_bundle.get("cited_repository_evidence", []),
            "intent_evidence_index": repair_intent_bundle.get(
                "intent_evidence_index", []
            ),
            **_scout_policy_evidence(repair_intent_bundle),
        }
    review_inventory = ""
    if expanded_pool:
        review_inventory = """
Required review table keys for THIS pool (exact sets):
""" + json.dumps({
            "hypothesis_reviews": [h.id for h in portfolio.hypotheses],
            "discriminator_reviews": [d.id for d in portfolio.unresolved_dimensions
                                      if d.disposition == "branched"],
        }) + """
Do not create discriminator_reviews for deferred or soft dimensions. Review
their coverage and assignment problems in the global gap/conflict fields.
An empty required key set means an empty table, not invented comparisons.
"""
    return f"""Audit whether this intent-hypothesis portfolio preserves repair recall.

{load_stage_guidance("hypothesis")}

This is a pre-Repair coverage and consistency Audit. The original Issue is
normative. Use only the supplied Issue, Scout, repository, execution, and merge
evidence. Do not use hidden tests, golden patches, future Repair candidates, or
implementation resemblance.

{_OBSERVATION_FIDELITY_GUIDANCE}

{selection_guidance}


Return ACCEPT only when all of these conditions hold:
1. Every Issue-explicit or established-public-contract target is represented as
   a hard FIX, and every hypothesis gives a concrete, non-contradictory account
   of how it satisfies every hard FIX/CLOSURE/PRESERVE obligation. Merely naming
   an obligation ID, promising to inspect it, or using the generic placeholder
   "Evaluate this Issue target during Repair" is not coverage.
2. Every repair surface supported by counterfactual necessity is included:
   upstream selection points before downstream symptoms, independently
   necessary sites, enumerated shared-cause siblings, public ingress/caller
   closure, equivalent access forms, and independently built or shipped mirror
   representations. Do not demand a surface from proximity, naming similarity,
   current buggy behavior, or an unverified family tail.
3. Every supplied SURFACE-N lead has one evidence-based disposition. An excluded
   or deferred lead must state the missing causal or authority relation; an
   included lead must map to concrete branch scope or preservation behavior.
   A materially plausible but unverified scope relation belongs in a conditional
   branch when a coherent policy can be stated; `deferred` is not a substitute
   for representing that alternative. Do not mistake an unverified premise for
   a contradiction, or turn an included conditional scope into a common hard
   obligation.
4. Every high-impact open decision that changes public behavior, compilation,
   API name/surface, scope, selected provider/path, precedence, fallback domain,
   exception behavior, supported mode, or causal locus is either represented by
   distinct complete hypotheses with one shared discriminator, or explicitly
   deferred because the supplied evidence cannot define the alternatives. Honor
   merge decisions marked keep_separate or inconclusive_keep. Do not manufacture
   branches for source-shape-only alternatives.
5. Hard PRESERVE obligations have Issue-explicit or public-contract authority.
   Checked-out tests, snapshots, execution, and current implementation are soft
   evidence unless independently authorized.
6. Obligations, branch assumptions, scopes, discriminators, predicted outcomes,
   and unresolved-dimension dispositions do not contradict one another. An
   Issue-author implementation suggestion is a falsifiable repair seed, not a
   hard mechanism or locus unless evidence proves that exact choice is required.

Use uncovered_issue_obligations for missing or falsely claimed normative target
coverage; repair_surface_gaps for missing evidence-backed sites or invalid
SURFACE dispositions; causal_coverage_gaps for incomplete causal/family/access/
artifact/API closure or lost high-impact alternatives; unsupported_hard_preserves
for preservation overreach; and internal_conflicts for contradictions. Every
reported gap must name the affected obligation, hypothesis, dimension, or
SURFACE ID and cite the supplied evidence that makes it material. Do not request
revision for wording, style, helper choice, speculative generality, or a test
conflict that the Issue authorizes. These gap/conflict lists are BLOCKING fields:
any nonempty list overrides ACCEPT to REVISE. Put evidence updates, unavailable
toolchains, unproven equivalence, and implementation uncertainty with no concrete
contract defect in nonblocking_notes instead. Notes are retained but do not
consume a revision. Real contradictory clauses or necessary missing paths must
still go in the blocking fields; the notes channel cannot excuse them.

Return JSON matching this schema:
{schema}

Original Issue:
{issue}

Shared evidence:
{json.dumps(evidence, ensure_ascii=False)}

Initial portfolio (hypotheses appear in aligned cards in expanded mode):
{json.dumps(portfolio.model_dump(exclude={'hypotheses'} if expanded_pool else set()), ensure_ascii=False)}

Aligned complete-policy cards (shared obligations above apply to every card):
{json.dumps(hypothesis_review_cards(portfolio), ensure_ascii=False) if expanded_pool else 'See portfolio above.'}
{review_inventory}
"""


def build_portfolio_revision_prompt(
    *,
    issue: str,
    issue_evidence: dict,
    repair_intent_bundle: dict,
    repair_intent_contract: dict,
    scout_summary: str,
    merge_adjudication: dict,
    portfolio: IntentPortfolio,
    audit: PortfolioCoverageAudit,
    policy_prompt: bool = False,
) -> str:
    generation_prompt = build_hypothesis_prompt(
        issue=issue,
        issue_evidence=issue_evidence,
        repair_intent_bundle=repair_intent_bundle,
        repair_intent_contract=repair_intent_contract,
        scout_summary=scout_summary,
        merge_adjudication=merge_adjudication,
        expanded_pool=isinstance(portfolio, IntentHypothesisPool),
        policy_prompt=policy_prompt,
    )
    return f"""{generation_prompt}

This is one bounded revision of an already generated portfolio. Preserve every
valid obligation, hypothesis distinction, evidence citation, and surface
disposition from the initial portfolio. Correct the material coverage gaps
identified by the Audit, but treat the Audit as review feedback rather than new
product evidence: ignore any suggestion that is not supported by the original
Issue or supplied evidence. Do not broaden scope, prescribe a golden-looking
implementation, merge an uncertain observable alternative, or add hypotheses
merely to reach a count. Return one complete corrected IntentPortfolio JSON
object, not an Audit object or an explanation.

Initial portfolio:
{json.dumps(portfolio.model_dump(), ensure_ascii=False)}

Coverage Audit:
{json.dumps(audit.model_dump(), ensure_ascii=False)}
"""


def build_hypothesis_context(
    portfolio: IntentPortfolio,
    hypothesis: IntentHypothesis,
) -> dict:
    return {
        "hard_obligations": [
            item.model_dump()
            for item in portfolio.obligations
            if item.strength == "hard"
        ],
        "soft_evidence": [
            item.model_dump()
            for item in portfolio.obligations
            if item.strength == "soft"
        ],
        "branch": hypothesis.model_dump(),
        "assigned_dimensions": assigned_dimension_contracts(
            portfolio, hypothesis
        ),
        "non_goals_or_soft_constraints": [
            {
                "statement": item.statement,
                "disposition": item.disposition,
                "reason": item.reason,
            }
            for item in portfolio.unresolved_dimensions
            if item.disposition != "branched"
        ],
        "competing_predictions": [
            {
                "id": item.id,
                "title": item.title,
                "discriminator": item.discriminator,
                "predicted_outcome": item.predicted_outcome,
            }
            for item in portfolio.hypotheses
            if item.id != hypothesis.id
        ],
    }


def build_candidate_repair_contract(
    portfolio: IntentPortfolio,
    hypothesis: IntentHypothesis,
) -> dict:
    """Return the full assigned contract without hypothesis identity labels."""
    return {
        "hard_obligations": [
            item.model_dump()
            for item in portfolio.obligations
            if item.strength == "hard"
        ],
        "branch": hypothesis.model_dump(exclude={"id"}),
        "assigned_dimensions": [
            {
                "statement": item["statement"],
                "shared_discriminator": item["shared_discriminator"],
                "assigned_outcome": item["assigned_outcome"],
                "competing_outcomes": [
                    outcome["predicted_outcome"]
                    for outcome in item["competing_outcomes"]
                ],
            }
            for item in assigned_dimension_contracts(portfolio, hypothesis)
        ],
        "non_goals_or_soft_constraints": [
            {
                "statement": item.statement,
                "disposition": item.disposition,
                "reason": item.reason,
            }
            for item in portfolio.unresolved_dimensions
            if item.disposition != "branched"
        ],
    }


def render_hypothesis_for_repair(
    portfolio: IntentPortfolio,
    hypothesis: IntentHypothesis,
) -> str:
    payload = build_hypothesis_context(portfolio, hypothesis)
    # Keep the canonical context used by Audit/Selector unchanged. These fields
    # explain a proposed implementation; they do not establish source coverage.
    implementation_notes = {
        field: payload["branch"].pop(field)
        for field in ("root_cause", "hard_obligation_coverage")
    }
    discriminator_marker = f"INTENT2FIX_DISCRIMINATOR_{hypothesis.id}_PASS"
    return (
        "<intent_hypothesis>\n"
        + json.dumps(payload, ensure_ascii=False)
        + "\n</intent_hypothesis>\n"
        "<hypothesis_implementation_notes>\n"
        + json.dumps(implementation_notes, ensure_ascii=False)
        + "\n</hypothesis_implementation_notes>\n"
        "The implementation notes contain a causal explanation and proposed "
        "coverage rationale, not additional requirements or proof that the "
        "implementation chain has been checked. Reuse supplied locations and "
        "source observations. The assigned repair_scope remains part of the "
        "contract: preserve its required APIs, behavior, and scope boundaries; "
        "an edit recipe within it does not establish that the suggested edit is "
        "sufficient. Do not change the assigned behavior to fit that recipe.\n"
        "<implementation_evidence_check>\n"
        "Before the first production edit, identify the source relationships "
        "needed for this branch: public operation to implementation/dispatch, "
        "and relevant callers, wrappers, or competing overloads. For each "
        "applicable relationship, reuse a cited source observation that "
        "establishes it or state the specific missing link and inspect it. "
        "A source path or coverage claim alone is not relationship evidence. "
        "Use the available source or graph tools for missing incoming users, "
        "owners/definitions, and relevant shared helpers. Read "
        "the relevant source; focused grep is available when an index is "
        "incomplete. Verify C++ overload/constraint selection with the relevant "
        "call shapes when source alone does not settle it. Record the resulting "
        "locations and any unresolved link concisely in the visible response. "
        "Existing evidence can satisfy this check without another search. "
        "Investigate only relationships required for the assigned implementation; "
        "graph adjacency does not expand patch scope.\n"
        "</implementation_evidence_check>\n"
        "This is one complete bounded repair contract, not permission to broaden "
        "the Issue. Implement every hard FIX/CLOSURE/PRESERVE obligation plus this "
        "branch's behavior. Soft evidence informs comparison but is not a hard "
        "veto. Treat listed non_goals_or_soft_constraints as scope guards, not as "
        "new repair targets. For a multi-branch portfolio, every "
        "assigned_dimensions outcome is a hard proposal constraint: implement the "
        "assigned observable behavior and do not silently switch to a competing "
        "branch. The shared_discriminator is diagnostic evidence, not a separate "
        "repair target or a Finish gate. Run it at most once when it is already "
        "directly executable or can be materialized as one small focused check. Do "
        "not spend proposal steps building a complex harness, decomposing one "
        "negative-compilation scenario into many probes, or repeatedly rewriting "
        "equivalent fixtures. When you run it, preserve its receiver, context, "
        "provider set, input, and varied axis; a one-provider or substituted-object "
        "fixture cannot establish branch compliance. Emit the exact marker "
        f"{discriminator_marker} only after the branch-specific oracle succeeds. "
        "Never emit the marker after a failed compile, failed assertion, or partial "
        "probe. Missing, unavailable, or inconclusive discriminator execution is "
        "honest selector evidence and does not excuse changing the assigned outcome "
        "or prevent submitting the production proposal. The "
        "discriminator oracle must be no stronger than the assigned observable "
        "contract. When exact wording or representation is left open, assert the "
        "required semantic property and the branch-relative difference; do not make "
        "the candidate's own exact output its oracle. Use the branch's "
        "evidence-backed relations to bound the production diff. A mentioned "
        "surface is not patch scope unless its closure obligation states the "
        "counterfactual failure that requires changing it. The implementation "
        "strategy remains free as long as it realizes the assigned observable "
        "contract without unrelated production changes. Preserve the semantic "
        "dimension named by the contract: type, category, role, identity, and "
        "concrete runtime value are different observables. In particular, do not "
        "implement a type/category/role requirement by echoing the fixture's literal "
        "value unless the contract explicitly requires that value. Treat an "
        "available regression pass as support only after checking which concrete "
        "source, generated, amalgamated, vendored, or mirrored representation the "
        "build actually compiled. If repository evidence says another independently "
        "built or shipped representation carries the changed contract, synchronize "
        "it through the repository's normal mechanism or include its required "
        "production change. A green run against an unchanged alternate "
        "representation is not evidence for the edited artifact. Do not add mirrors "
        "without direct repository/build evidence. Tests remain soft evidence, but "
        "still run the repository regression after the final production/test "
        "revision. Do not exhaust the proposal budget mechanically rewriting every "
        "legacy exact-output or exact-string pin merely to make the working-tree "
        "suite green. Prefer one focused candidate-authored check for the intended "
        "behavior and retain conflicts with unchanged legacy pins as explicit soft "
        "evidence. Make only the smallest test edit needed to exercise the candidate "
        "or a repository-required coordinated test surface; test edits remain "
        "excluded from the submitted production patch. Treat an Issue-author "
        "implementation recipe as a falsifiable seed, not the default answer: test "
        "it against the exact full-context witness before committing to it. If that "
        "witness preserves the same selection or dispatch failure after one coherent "
        "attempt, change the candidate participation, visibility, ranking, or "
        "routing mechanism instead of repeatedly tuning reduced signature sketches. "
        "If one valid focused execution contradicts the assigned outcome, make at "
        "most one coherent production correction and rerun that same check once. If "
        "the evidence remains contradictory or inconclusive, do not start another "
        "probe family: preserve the current bounded proposal, report the limitation "
        "honestly, and call Finish so Branch Compliance and the Selector can assess "
        "it. These limits bound repeated diagnostic attempts; they do not skip "
        "the initial implementation evidence check or verification of a changed "
        "shared path. A candidate is ready to compare after the necessary "
        "implementation relationships and relevant preservation checks have been "
        "examined, with unavailable checks or unresolved links reported honestly. "
        "Then call Finish without redundant searches or equivalent probes."
    )


def branch_correction_clauses(context: Dict[str, Any]) -> Dict[str, str]:
    """Only fixed requirements can authorize a same-branch code correction."""
    clauses = {
        item["id"]: item["statement"]
        for item in context.get("hard_obligations", [])
        if item.get("strength", "hard") == "hard" and item.get("statement")
    }
    for index, item in enumerate(context.get("assigned_dimensions", []), 1):
        if item.get("assigned_outcome"):
            clauses[item.get("dimension_id", f"DIM-{index}")] = item["assigned_outcome"]
    branch = context.get("branch") or {}
    for field, prefix in (("expected_behavior", "BRANCH-EXPECTED"), ("repair_scope", "BRANCH-SCOPE")):
        for index, statement in enumerate(branch.get(field, []), 1):
            clauses[f"{prefix}-{index}"] = statement
    return clauses


def branch_failure_explanation_clauses(context: Dict[str, Any]) -> Dict[str, str]:
    """An explicit observable outcome can explain a conflict with an old test."""
    return {
        cid: quote for cid, quote in branch_correction_clauses(context).items()
        if cid.startswith(("FIX-", "DIM-", "BRANCH-EXPECTED-"))
    }


def branch_failure_evidence(
    patch: str, repair_probes: List[RepairProbeEvidence], regression_evidence: Dict[str, Any],
) -> Dict[str, str]:
    """Inventory current failed executions; stale/unproven records cannot justify a waiver."""
    digest = production_patch_sha256(patch)
    failures = {
        p.probe_id: p.observed_output for p in repair_probes
        if p.production_patch_sha256 == digest and p.observed_status == "failed"
        and p.observed_exit_code is not None
    }
    failures.update({
        key: "\n".join(str(record.get(field) or "") for field in ("summary", "output"))
        for key, record in regression_evidence.items()
        if isinstance(record, dict) and record.get("production_patch_sha256") == digest
        and record.get("verification_view") == "production"
        and record.get("exit_code") is not None
        and (record.get("passed") is False or record["exit_code"] != 0)
    })
    return failures


def validate_branch_correction(
    decision: BranchComplianceDecision,
    *,
    hypothesis_context: Dict[str, Any],
    patch: str,
    repair_probes: List[RepairProbeEvidence],
    regression_evidence: Dict[str, Any],
    verification_test_patch: str = "",
) -> BranchComplianceDecision:
    """Check correction/compliance provenance, without claiming semantic proof."""
    if decision.status == "compliant":
        failures = branch_failure_evidence(patch, repair_probes, regression_evidence)
        explanations = decision.failure_explanations
        if failures or explanations:
            allowed = branch_failure_explanation_clauses(hypothesis_context)
            error = ""
            if decision.production_patch_sha256 != production_patch_sha256(patch):
                error = "Compliance does not identify the current production patch."
            elif len(explanations) != len(failures) or {e.failure_id for e in explanations} != set(failures):
                error = "Every current failed execution needs exactly one explicit branch explanation."
            else:
                for explanation in explanations:
                    quote = " ".join(explanation.clause_quote.split())
                    observed_quote = " ".join(explanation.evidence_quote.split())
                    if (
                        explanation.clause_id not in allowed
                        or quote != " ".join(allowed[explanation.clause_id].split())
                        or not all(s.strip() for s in (
                            explanation.scenario, explanation.expected_old_behavior, explanation.observed_behavior,
                        ))
                        or not observed_quote
                        or observed_quote not in " ".join(failures[explanation.failure_id].split())
                    ):
                        error = "A failure explanation lacks an explicit outcome clause or exact current evidence."
                        break
            if error:
                return decision.model_copy(update={
                    "status": "inconclusive", "actionable_correction": False, "suggested_correction": "",
                    "reason": f"Compliance withheld: {error} Original audit: {decision.reason}",
                })
    if not decision.actionable_correction:
        return decision
    clauses = branch_correction_clauses(hypothesis_context)
    digest = production_patch_sha256(patch)
    evidence_ids = {"PATCH"} if patch.strip() else set()
    if verification_test_patch.strip():
        evidence_ids.add("VERIFICATION_TEST_PATCH")
    evidence_ids.update(
        p.probe_id for p in repair_probes
        if p.production_patch_sha256 == digest
        and p.observed_status in {"passed", "failed"}
        and p.observed_exit_code is not None
    )
    evidence_ids.update(
        key for key, record in regression_evidence.items()
        if record.get("verification_view") == "production"
        and record.get("production_patch_sha256") == digest
        and record.get("exit_code") is not None
    )
    error = ""
    if any(cid not in clauses for cid in decision.violated_clause_ids):
        error = "A soft legacy, deferred, or unknown clause cannot authorize correction."
    elif decision.production_patch_sha256 != digest:
        error = "Correction does not identify the current production patch."
    elif not decision.violations or set(decision.violated_clause_ids) != {
        v.clause_id for v in decision.violations
    }:
        error = "Every violated clause needs a concrete scenario and evidence."
    else:
        for violation in decision.violations:
            if (
                " ".join(violation.clause_quote.split()) != " ".join(clauses[violation.clause_id].split())
                or not all(s.strip() for s in (violation.scenario, violation.expected_behavior, violation.observed_behavior))
                or not violation.evidence_ids
                or not set(violation.evidence_ids) <= evidence_ids
            ):
                error = "Correction has an invalid clause quotation or missing/stale evidence."
                break
    if not error:
        return decision
    return decision.model_copy(update={
        "status": "inconclusive", "actionable_correction": False,
        "suggested_correction": "", "reason": f"Correction withheld: {error} Original audit: {decision.reason}",
    })


def build_branch_compliance_prompt(
    *,
    hypothesis_context: Dict[str, object],
    patch: str,
    repair_probes: List[RepairProbeEvidence],
    regression_evidence: Dict[str, object] | None,
    verification_test_patch: str = "",
) -> str:
    schema = json.dumps(
        BranchComplianceDecision.model_json_schema(), ensure_ascii=False
    )
    payload = {
        "hypothesis_context": hypothesis_context,
        "correction_authority_clauses": branch_correction_clauses(hypothesis_context),
        "failure_explanation_clauses": branch_failure_explanation_clauses(hypothesis_context),
        "current_failed_executions": branch_failure_evidence(patch, repair_probes, regression_evidence or {}),
        "production_patch_sha256": production_patch_sha256(patch),
        "sanitized_production_patch": patch,
        "verification_test_patch": verification_test_patch,
        "final_patch_repair_probes": [
            item.model_dump() for item in repair_probes
        ],
        "repository_regression_evidence": regression_evidence or {},
    }
    return f"""Audit only whether this finished production patch implements its assigned repair hypothesis.

{load_stage_guidance("hypothesis")}

The normalized hypothesis context is the complete contract for this check.
Do not decide whether that hypothesis is the true intent of the original
Issue, do not rank it against another hypothesis or candidate, and do not
reject a patch merely because a different behavior might better solve the
Issue. Those decisions belong exclusively to the later candidate Selector.
Judge the actual sanitized production diff and final-revision execution
evidence against the assigned hypothesis clauses, not the candidate's Finish
explanation or implementation resemblance.

For each reported pass, inspect its frozen files and the executed command to see
what code and assertions actually ran. A passed exit/output oracle does not
establish Issue behavior by itself. An empty program, printed success marker,
test filename or Finish claim cannot prove the assigned behavior. If temporary
fixture source was not captured, do not invent its contents or infer coverage
from its name. Preserve the distinction between an actual behavioral test,
an environment/setup failure and an original-suite preservation check.

Return noncompliant only for a concrete mismatch with an assigned hypothesis
clause or an executed outcome selecting a competing branch. Return
inconclusive when evidence is missing and the assigned behavior cannot be
established statically. Set actionable_correction=true only when the evidence
identifies a bounded local change or verification action that can align this
patch with the assigned hypothesis; list the exact hypothesis clause IDs and a
concise correction. Uncertainty alone is not actionable.

For each assigned dimension, require the unchanged shared discriminator to
exercise all named competing providers, paths, participants, and relations in
one behavior-equivalent scenario. A reduced one-provider or substituted-object
fixture cannot establish that branch. If an executed exact discriminator
selects a competing outcome, return noncompliant. If the necessary evidence is
missing or invalid and the patch cannot establish the assigned behavior
statically, return inconclusive rather than guessing.

VERIFICATION_TEST_PATCH contains current test edits deliberately excluded from
the production submission. Use it to check clauses requiring tests to be written;
their absence from PATCH is not a missing implementation. Never request shipping
these tests as production files. Their contents alone do not prove execution or
test success; those require current execution evidence.

Legacy/basic tests are supporting evidence, not an automatic conformance gate.
Do not reinterpret their failure against the raw Issue; check only preservation
or closure clauses explicitly present in the assigned hypothesis. A compile
failure can make conformance inconclusive, but compilation repair is handled by
the orchestrator. The repair locus and mechanism may vary when the resulting
patch still realizes the complete assigned observable contract.

Only correction_authority_clauses may authorize an actionable correction.
soft_evidence and deferred notes cannot alone override the assigned branch's
explicit changed behavior. A legacy assertion that expects precisely the old
behavior being changed is a legacy-expectation conflict, not a reason to restore
that behavior. If the relation is uncertain, return inconclusive without a code
change instruction. Existing public compatibility requirements still apply.
BRANCH-SCOPE clauses describe the assigned scope; a different implementation
location is not by itself a violation of an equivalent observable contract.

Before returning compliant, account for EVERY current_failed_executions entry
in failure_explanations, exactly once per failure_id. Cover every failure shown
in that execution, not just one convenient assertion. Give the concrete input
scenario, expected_old_behavior, observed_behavior, an exact evidence_quote
from that entry (one short contiguous substring; no added ellipses, paraphrase,
or joining separate lines), and the exact clause_id/clause_quote from
failure_explanation_clauses that explicitly assigns that observed outcome to
that input. Keep explanations concise. A correctly quoted clause is necessary
but is not sufficient: its input scope AND observable result must match.
Compile-time rejection, runtime rejection, and successful formatting with a
different output are distinct outcomes; an exemption for one does not authorize
the others. A shared mechanism or affected type family does not prove the same
authorized behavior change. Soft notes, implementation recipes, absence of a
hard preservation clause, and anticipated side effects cannot justify a waiver.
If any failure has no matching explicit outcome, has incomplete evidence, or
could be an unrelated compatibility regression, return inconclusive and retain
the conflict in reason, with actionable_correction=false and an empty
suggested_correction; do not guess an exemption or rewrite the hypothesis.
If instead it proves a concrete assigned-contract violation, use the existing
noncompliant/correction rules. Do not request code changes merely because an
exemption is unproven. With no current failures, failure_explanations is empty.

For each actionable violation, provide its exact clause_quote, concrete input
scenario, expected_behavior, observed_behavior and evidence_ids. Copy the
production_patch_sha256. Evidence IDs are PATCH for the frozen diff, probe_id
for a current executed probe, or the repository_regression_evidence key for a
production-view execution. Unexecuted probes and working-tree results containing
excluded files do not establish the submitted patch. Do not change a probe's
expected behavior merely to make a correction appear successful.

Return JSON matching this schema:
{schema}

Assigned-hypothesis evidence:
{json.dumps(payload, ensure_ascii=False)}
"""


def deduplicate_candidates(
    candidates: List[PortfolioCandidate],
) -> List[PortfolioCandidate]:
    unique: Dict[str, PortfolioCandidate] = {}
    for candidate in candidates:
        unique.setdefault(candidate.patch_sha256, candidate)
    return list(unique.values())


def _portfolio_source_mapping(
    ordered: List[PortfolioCandidate],
) -> Dict[str, str]:
    source_mapping = {
        f"{candidate.hypothesis_id}@{candidate.variant_id}": (
            f"CANDIDATE-{index}"
        )
        for index, candidate in enumerate(ordered, start=1)
    }
    hypothesis_counts: Dict[str, int] = {}
    for candidate in ordered:
        hypothesis_counts[candidate.hypothesis_id] = (
            hypothesis_counts.get(candidate.hypothesis_id, 0) + 1
        )
    for index, candidate in enumerate(ordered, start=1):
        if hypothesis_counts[candidate.hypothesis_id] == 1:
            source_mapping[candidate.hypothesis_id] = f"CANDIDATE-{index}"
    return source_mapping


def _anonymize_probe_payload(
    payload: Dict[str, object],
    source_mapping: Dict[str, str],
    *,
    fallback_source_id: str = "SOURCE-CANDIDATE",
) -> Dict[str, object]:
    anonymized = dict(payload)
    raw_source_id = str(anonymized.get("source_hypothesis_id") or "")
    if not raw_source_id:
        return anonymized
    source_id = source_mapping.get(raw_source_id, fallback_source_id)
    base_hypothesis_id = raw_source_id.split("@", 1)[0]
    anonymized["source_hypothesis_id"] = source_id
    for field in ("check_id", "probe_id"):
        value = str(anonymized.get(field) or "")
        if not value:
            continue
        anonymized[field] = value.replace(
            raw_source_id, source_id
        ).replace(
            f"-{base_hypothesis_id}-", f"-{source_id}-"
        )
    for field in (
        "command",
        "output",
        "observed_output",
        "required_output_regex",
    ):
        value = str(anonymized.get(field) or "")
        if not value:
            continue
        anonymized[field] = value.replace(
            raw_source_id, source_id
        ).replace(
            f"INTENT2FIX_DISCRIMINATOR_{base_hypothesis_id}",
            f"INTENT2FIX_DISCRIMINATOR_{source_id}",
        )
    return anonymized


def _anonymize_cross_execution(
    execution: CrossExecutionResult,
    source_mapping: Dict[str, str],
) -> CrossExecutionResult:
    payload = execution.model_dump()
    probe_results = [
        _anonymize_probe_payload(item, source_mapping)
        for item in payload.get("probe_results") or []
    ]
    payload["probe_results"] = probe_results
    if probe_results:
        payload["command"] = "\n".join(
            str(item.get("command") or "") for item in probe_results
        )
        payload["output"] = "\n\n".join(
            f"[{item.get('check_id', '')}] {item.get('output', '')}"
            for item in probe_results
        )
    return CrossExecutionResult.model_validate(payload)


def anonymize_baseline_execution(
    baseline_execution: CrossExecutionResult,
    candidates: List[PortfolioCandidate],
) -> CrossExecutionResult:
    ordered = sorted(candidates, key=lambda item: item.patch_sha256)
    return _anonymize_cross_execution(
        baseline_execution,
        _portfolio_source_mapping(ordered),
    )


def anonymize_candidates(
    candidates: List[PortfolioCandidate],
) -> tuple[List[dict], Dict[str, PortfolioCandidate]]:
    anonymous: List[dict] = []
    mapping: Dict[str, PortfolioCandidate] = {}
    ordered = sorted(candidates, key=lambda item: item.patch_sha256)
    source_mapping = _portfolio_source_mapping(ordered)
    for index, candidate in enumerate(ordered, start=1):
        candidate_id = f"CANDIDATE-{index}"
        mapping[candidate_id] = candidate
        repair_verification = []
        for item in candidate.repair_probes:
            payload = _anonymize_probe_payload(
                item.model_dump(),
                source_mapping,
                fallback_source_id=(
                    candidate_id
                    if item.source_hypothesis_id == candidate.hypothesis_id
                    else "SOURCE-CANDIDATE"
                ),
            )
            repair_verification.append(payload)
        cross_execution = _anonymize_cross_execution(
            candidate.cross_execution, source_mapping
        ).model_dump()
        anonymous.append(
            {
                "candidate_id": candidate_id,
                "proposed_behavior": candidate.branch_assumption,
                "repair_contract": candidate.repair_contract,
                "assigned_discriminator": candidate.discriminator,
                "predicted_outcome": candidate.predicted_outcome,
                "patch": candidate.patch,
                "kept_paths": candidate.kept_paths,
                "mechanically_excluded_paths": candidate.excluded_paths,
                "finish_succeeded": candidate.finish_succeeded,
                "basic_tests_passed": candidate.basic_tests_passed,
                "branch_compliance": candidate.branch_compliance,
                "discriminator_verified": candidate.discriminator_verified,
                "repair_verification": repair_verification,
                "cross_execution": cross_execution,
            }
        )
    return anonymous, mapping


def build_candidate_audit_prompt(
    *,
    issue: str,
    portfolio: IntentPortfolio,
    anonymous_candidates: List[dict],
    baseline_execution: CrossExecutionResult,
    scout_repair_intent: dict | None = None,
) -> str:
    schema = json.dumps(CandidateAuditDecision.model_json_schema(), ensure_ascii=False)
    probe_matrix: Dict[str, dict] = {}
    for candidate in anonymous_candidates:
        candidate_id = str(candidate.get("candidate_id") or "")
        probe_results = (
            candidate.get("cross_execution", {}).get("probe_results", [])
        )
        for result in probe_results:
            check_id = str(result.get("check_id") or "")
            if not check_id:
                continue
            row = probe_matrix.setdefault(
                check_id,
                {
                    "authority": result.get("authority", ""),
                    "origin": result.get("origin", ""),
                    "source_candidate": result.get(
                        "source_hypothesis_id", ""
                    ),
                    "expected_outcome": result.get("expected_outcome", ""),
                    "description": result.get("description", ""),
                    "results": {},
                },
            )
            row["results"][candidate_id] = {
                "status": result.get("status", ""),
                "exit_code": result.get("exit_code"),
            }
    payload = {
        "scout_repair_intent": {
            key: (scout_repair_intent or {}).get(key, [])
            for key in ("target_behavior", "preserve_behavior", "properties",
                        "scope_evidence", "open_decisions", "validation_gaps")
        },
        "obligations": [item.model_dump() for item in portfolio.obligations],
        "coverage_report": portfolio.coverage_report(),
        "unresolved_dimensions": [
            item.model_dump() for item in portfolio.unresolved_dimensions
        ],
        "clean_baseline_execution": baseline_execution.model_dump(),
        "candidate_probe_matrix": probe_matrix,
        "candidates": anonymous_candidates,
    }
    return f"""Audit anonymous candidate patches and select one finalist.

The original Issue is normative. First reject a candidate that misses an
Issue-required target or makes an Issue-contradicting behavior change. Then
compare preservation evidence and unsupported production scope. Mechanically
excluded test/probe files are not part of the submitted patch. Do not infer
correctness from implementation resemblance or candidate order.

For each candidate, trace the Issue trigger through the defective call or state
transition, the patch's changed mechanism, and the supported affected callers,
overloads, or shipped representations. The Scout repair intent is a scope map,
not an exclusive checklist; verify its claims against the Issue, source, and
execution. A fix of one reproduction is partial if the same supported cause
remains on another path. Justified additional edits that close that cause or
preserve valid uses are acceptable; a shorter diff is not inherently safer.
Explain concrete omitted paths and residual failure mechanisms in concerns,
not merely that one patch is larger than another.

Each anonymous candidate also carries the hypothesis that guided its search.
Its root cause, repair scope, locus, and predicted mechanism are provenance,
not hard requirements. Do not reject a candidate merely for departing from
them. Judge the resulting patch against the original Issue targets,
public-contract preserves, repository/build closure, and execution evidence.
A changed locus is acceptable when it fixes the target without introducing
unsupported scope.

`branch_compliance` is a unary provenance label: it says only whether the patch
faithfully implemented the hypothesis assigned to that Repair branch. It is
not evidence that the hypothesis itself is the original Issue intent. Never
prefer a compliant candidate merely for fidelity, and never reject a
noncompliant or inconclusive candidate solely for that label. A branch-deviating
patch may still be the best Issue-aligned candidate; decide from the original
Issue and the cross-execution evidence below.

Legacy/basic test status is evidence, never an automatic gate. A failure may be
Issue-authorized, a likely unrelated regression, or inconclusive; classify it
from the Issue and patch. A passing old test cannot compensate for missing the
requested behavior. Compare each frozen probe row across the clean baseline and
all candidates. A normative ACCEPT row is strong evidence only when it directly
tests the shared Issue target; an infrastructure error is inconclusive, not a
candidate failure. Prefer a smaller diff only after semantic coverage
and preservation are equivalent. Mark production files/hunks unrelated only
when no Issue, Scout, compile-closure, generated-mirror, or compatibility reason
supports them. Select exactly one candidate when evidence supports one. ABSTAIN
when every candidate violates a hard Issue obligation, when the only candidate
is contradicted by target execution, or when evidence cannot safely distinguish
serious conflicting risks.

Do not classify an existing-caller compile/link failure as an authorized legacy
conflict unless the Issue explicitly removes that caller contract. Likewise, a
failed final-revision preservation probe needs Issue or cited public-contract
authority before it can be dismissed; a candidate's own Finish explanation is
not authority. Pure probe setup failures remain inconclusive.

The execution matrix includes probes from multiple authorities. Issue
reproduction and normative Scout probes are strong target evidence.
Illustrative Scout probes and Repair-authored hypothesis probes are soft but
must not be discarded: use them to expose observable differences, then judge
those differences against the Issue and repository contract. A candidate's own
Repair probe is not self-validating merely because it passed. Compare that same
probe across every candidate. Do not call candidates semantically equivalent
when their probe-result vectors differ. `discriminator_verified` only records
that Repair executed its own conflict oracle and emitted its marker; it is soft,
self-authored evidence, not branch compliance. Compare the marked probe across
all candidates. Set execution_support to
static_only or unavailable when no executed row supports a completeness claim;
never infer lookup, overload, dispatch, or precedence behavior from names,
comments, or patch inspection. A failing full-context final-patch probe for the
Issue behavior outweighs a passing reduced mechanism probe that omits an actual
competing provider, resolution path, or enclosing call shape. If no candidate
closes that behavior, set its execution_support to contradicted and ABSTAIN;
having only one candidate is not a reason to select it. Diff size is only a
final tie breaker after the execution evidence is genuinely equivalent.

When inspecting a discriminator, preserve its relational identity: a scenario
that holds one receiver, context, or provider set fixed while varying one axis
is not equivalent if the probe substitutes a different participant. Likewise,
do not conflate type, category, role, identity, or concrete runtime value. A
patch that echoes a witness literal does not establish a contract requiring
stable type/category context unless the Issue or repository evidence
authorizes that value-specific behavior.

Treat execution provenance as part of the evidence. A green regression run
supports a candidate only when its build/import route exercised that
candidate's changed production representation. If it compiled an unchanged
generated, amalgamated, vendored, or mirrored copy instead, classify the run as
non-supporting and assess whether the candidate omitted an evidenced shipped or
independently built representation. Do not infer such a mirror solely from a
similar filename or common project convention.

Return JSON matching this schema and use the exact anonymous candidate IDs:
{schema}

Original Issue:
{issue}

Portfolio evidence:
{json.dumps(payload, ensure_ascii=False)}
"""


def resolve_audit_selection(
    decision: CandidateAuditDecision,
    mapping: Dict[str, PortfolioCandidate],
) -> Optional[PortfolioCandidate]:
    if decision.selection != "SELECTED":
        return None
    selected_id = decision.selected_candidate_id or ""
    selected = mapping.get(selected_id)
    if selected is None:
        return None
    assessment = next(
        (
            item
            for item in decision.assessments
            if item.candidate_id == selected_id
        ),
        None,
    )
    if assessment is None:
        return None
    if assessment.target_coverage in {"missing", "partial"}:
        return None
    if assessment.preservation == "violated":
        return None
    if assessment.production_scope == "unrelated":
        return None
    if assessment.execution_support == "contradicted":
        return None
    if selected.cross_execution.patch_applied is not True:
        return None
    return selected
