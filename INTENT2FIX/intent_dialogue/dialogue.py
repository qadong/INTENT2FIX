from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher, unified_diff
from typing import Any, Callable, Type

from pydantic import BaseModel

from INTENT2FIX.intent_dialogue.prompts import (
    build_contrastive_intent_decision_prompt,
    build_poc_intent_decision_prompt,
    build_evidence_sufficiency_review_prompt,
    build_intent_boundary_test_assessment_prompt,
    build_intent_contract_semantic_audit_prompt,
    build_intent_property_extraction_prompt,
    build_intent_validation_routing_prompt,
    build_test_anchor_selection_prompt,
    build_intent_evidence_adjudication_prompt,
    build_intent_resolution_prompt,
    build_intent_task_plan_prompt,
    build_intent_synthesis_prompt,
)
from INTENT2FIX.intent_dialogue.schema import (
    CandidateRepairBoundary,
    ContrastiveAttempt,
    ContrastiveIntentDecision,
    ContrastiveProbeCandidate,
    DialogueExchange,
    EvidenceSufficiencyReview,
    EliminatedHypothesis,
    FocusedInvestigationReport,
    FrozenAcceptanceCheck,
    IntentBoundaryTestAssessment,
    IntentContractSemanticAudit,
    IntentAnchorSelectionGroup,
    IntentEvidenceAdjudication,
    IntentEvidenceResolution,
    IntentDimensionResolution,
    IntentExplorationObjective,
    IntentExplorationPlan,
    IntentDialogueResult,
    IntentBehaviorProperty,
    InvestigationReport,
    InvestigationPath,
    InvestigationTaskFinding,
    IntentSynthesisSelection,
    IntentTaskResolution,
    IntentPropertyExtraction,
    IntentPropertyDimension,
    IntentValidationRoute,
    IntentValidationRouting,
    RepositoryEvidence,
    RepositoryInvestigation,
    RepairIntentLedgerEntry,
    RepairIntentContract,
    TestAnchorRetrieval,
    TestAnchorSelection,
)
from INTENT2FIX.reproduce.intent_schema import IssueEvidenceFrame
from INTENT2FIX.intent_dialogue.test_retrieval import (
    IntentEmbeddingUnavailableError,
)


StructuredExtractor = Callable[[str, Type[BaseModel]], BaseModel | None]
logger = logging.getLogger(__name__)


_TARGET_CLAIM_TYPES = {"expected_behavior"}
_PRESERVATION_CLAIM_TYPES = {"preservation"}
_SURFACE_INVENTORY_TASK = re.compile(
    r"\b(?:wrapper|forwarding path|generated (?:header|artifact)|mirror)\b|"
    r"\bcomplete (?:list|inventory) of\b|"
    r"\bwhich public\b.{0,100}\b(?:classes|types|apis|overloads|entry points)\b|"
    r"\bwhich (?:public )?(?:classes|types|apis|overloads|entry points)\b|"
    r"\b(?:class|type|api)s?\b.{0,100}\b(?:or both|all of them|must expose)\b",
    re.IGNORECASE,
)


def _unique(items: list[str], limit: int | None = None) -> list[str]:
    values = [item.strip() for item in items if item and item.strip()]
    values = list(dict.fromkeys(values))
    return values if limit is None else values[:limit]


def _bounded_text(value: str, limit: int = 8000) -> str:
    if len(value) <= limit:
        return value
    half = limit // 2
    return (
        value[:half]
        + "\n... [bounded semantic-audit payload] ...\n"
        + value[-half:]
    )


def _evidence_sufficiency_packet(
    report: InvestigationReport,
    repository_evidence: list[RepositoryEvidence],
    report_history: list[FocusedInvestigationReport] | None = None,
) -> list[dict]:
    """Bound raw observations for an independent semantic review."""
    history = report_history or []
    cited = {
        evidence_id
        for item in history
        for evidence_id in item.report.repository_evidence_ids
    }
    cited.update(report.repository_evidence_ids)
    ordered = [
        *[item for item in repository_evidence if item.id in cited],
        *[item for item in repository_evidence if item.id not in cited],
    ]
    packet = []
    remaining_chars = 90000
    seen = set()
    for item in ordered:
        if item.id in seen or remaining_chars <= 0:
            continue
        seen.add(item.id)
        observation = _bounded_text(item.observation, 5000)
        if len(observation) > remaining_chars:
            observation = _bounded_text(observation, remaining_chars)
        packet.append({
            "id": item.id,
            "path": item.path,
            "locator": item.locator,
            "observation": observation,
            "cited_by_report": item.id in cited,
            "observation_truncated": len(observation) < len(item.observation),
        })
        remaining_chars -= len(observation)
    return packet


def _merge_acceptance_checks(
    *groups: list[FrozenAcceptanceCheck],
    limit: int = 10,
) -> list[FrozenAcceptanceCheck]:
    """Merge independent evidence channels without replacing earlier checks."""
    merged: list[FrozenAcceptanceCheck] = []
    seen: set[tuple[str, str, str, str]] = set()
    for group in groups:
        for check in group:
            key = (
                check.candidate_id,
                check.source_evidence_id,
                check.command,
                check.kind,
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(check)
            if len(merged) == limit:
                return merged
    return merged


def _merge_investigation_paths(
    *groups: list[InvestigationPath],
    limit: int = 8,
) -> list[InvestigationPath]:
    """Retain distinct checked/open paths across focused report replacements."""
    merged: dict[str, InvestigationPath] = {}
    for group in groups:
        for path in group:
            key = path.path_id or "question:" + " ".join(
                path.question.lower().split()
            )
            prior = merged.get(key)
            if prior is not None:
                path = path.model_copy(update={
                    "evidence_ids": list(dict.fromkeys(
                        prior.evidence_ids + path.evidence_ids
                    ))[:6],
                })
            merged[key] = path
    return list(merged.values())[:limit]


def _merge_candidate_boundaries(
    issue_evidence: IssueEvidenceFrame,
    *groups: list[CandidateRepairBoundary],
    limit: int = 10,
) -> list[CandidateRepairBoundary]:
    """Merge non-normative API-family leads without making scope decisions."""
    known_intent_ids = {unit.id for unit in issue_evidence.intent_units}
    merged: dict[tuple[str, tuple[str, ...]], CandidateRepairBoundary] = {}
    for group in groups:
        for boundary in group:
            intent_ids = _unique([
                intent_id
                for intent_id in boundary.issue_intent_ids
                if not known_intent_ids or intent_id in known_intent_ids
            ], limit=10)
            key = (
                " ".join(boundary.anchor_operation.lower().split()),
                tuple(sorted(intent_ids)),
            )
            existing = merged.get(key)
            if existing is None:
                merged[key] = boundary.model_copy(update={
                    "id": f"BOUNDARY-{len(merged) + 1}",
                    "issue_intent_ids": intent_ids,
                    "candidate_public_forms": _unique(
                        boundary.candidate_public_forms,
                        limit=10,
                    ),
                    "evidence_ids": _unique(boundary.evidence_ids, limit=5),
                })
            else:
                merged[key] = existing.model_copy(update={
                    "candidate_public_forms": _unique(
                        existing.candidate_public_forms
                        + boundary.candidate_public_forms,
                        limit=10,
                    ),
                    "evidence_ids": _unique(
                        existing.evidence_ids + boundary.evidence_ids,
                        limit=5,
                    ),
                })
            if len(merged) >= limit:
                break
        if len(merged) >= limit:
            break
    return list(merged.values())


def build_repair_intent_contract(
    issue_evidence: IssueEvidenceFrame,
    candidates: dict[str, dict[str, Any]] | None = None,
    synthesis: IntentSynthesisSelection | None = None,
    open_decisions: list[str] | None = None,
    resolved_qualifiers: list[str] | None = None,
    compatibility_warnings: list[str] | None = None,
) -> RepairIntentContract:
    intent_units = list(issue_evidence.intent_units)
    if not intent_units:
        sequence = 1
        for claim in issue_evidence.claims:
            if claim.kind not in (
                _TARGET_CLAIM_TYPES | _PRESERVATION_CLAIM_TYPES | {"non_goal"}
            ):
                continue
            intent_units.append({
                "id": f"INTENT-{sequence}",
                "role": (
                    "target"
                    if claim.kind in _TARGET_CLAIM_TYPES
                    else "preserve"
                ),
                "text": claim.text,
                "issue_span": claim.issue_span,
                "source_claim_ids": [claim.id],
            })
            sequence += 1

    intent_ledger = [
        RepairIntentLedgerEntry.model_validate({
            **(
                unit.model_dump()
                if hasattr(unit, "model_dump")
                else unit
            ),
            "status": "uninvestigated",
        })
        for unit in intent_units
    ]
    target_behavior = [
        unit.text for unit in intent_ledger if unit.role == "target"
    ]
    preserve_behavior = [
        unit.text for unit in intent_ledger if unit.role == "preserve"
    ]
    non_goal_claim_ids = {
        claim.id for claim in issue_evidence.claims if claim.kind == "non_goal"
    }
    preserve_behavior = [
        (
            f"Do not implement this explicitly rejected alternative: {unit.text}"
            if set(unit.source_claim_ids) & non_goal_claim_ids
            else unit.text
        )
        for unit in intent_ledger
        if unit.role == "preserve"
    ]

    # Repository-derived constraints are unsafe without an extracted Issue
    # target. In that case the raw issue remains authoritative and this module
    # deliberately becomes a no-op.
    if not target_behavior:
        return RepairIntentContract()

    target_behavior.extend(
        f"Repository-resolved qualifier: {qualifier}"
        for qualifier in resolved_qualifiers or []
    )

    candidates = candidates or {}
    preserve_candidates = candidates.get("preserve", {})
    acceptance_candidates = candidates.get("acceptance", {})
    warning_candidates = candidates.get("warning", {})
    synthesis = synthesis or IntentSynthesisSelection()

    preserve_behavior.extend(
        preserve_candidates[candidate_id]["text"]
        for candidate_id in synthesis.preserve_candidate_ids
        if candidate_id in preserve_candidates
    )
    acceptance_checks = [
        acceptance_candidates[candidate_id]["check"]
        for candidate_id in synthesis.acceptance_candidate_ids
        if candidate_id in acceptance_candidates
    ]
    selected_warnings = [
        warning_candidates[candidate_id]["text"]
        for candidate_id in synthesis.warning_candidate_ids
        if candidate_id in warning_candidates
    ]
    selected_warnings.extend(compatibility_warnings or [])

    return RepairIntentContract(
        intent_ledger=intent_ledger,
        target_behavior=_unique(target_behavior),
        preserve_behavior=_unique(preserve_behavior),
        acceptance_checks=list({
            (item.description, item.command): item
            for item in acceptance_checks
        }.values())[:4],
        compatibility_warnings=_unique(selected_warnings, limit=3),
        open_decisions=_unique(open_decisions or [], limit=10),
    )


def _grounded_open_decisions(
    reports: list[InvestigationReport],
) -> list[str]:
    return _unique([
        decision
        for report in reports
        if report.status in {"partial", "conflict"}
        and report.repository_evidence_ids
        for decision in report.open_decisions
    ], limit=10)


def _normalized_question(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9_]+", text.lower()))


def _questions_overlap(left: str, right: str) -> bool:
    left_normalized = _normalized_question(left)
    right_normalized = _normalized_question(right)
    left_tokens = set(left_normalized.split())
    right_tokens = set(right_normalized.split())
    union = left_tokens | right_tokens
    jaccard = (
        len(left_tokens & right_tokens) / len(union)
        if union
        else 1.0
    )
    return (
        jaccard >= 0.65
        or SequenceMatcher(
            None,
            left_normalized,
            right_normalized,
        ).ratio() >= 0.78
    )


def _normalize_plan(
    plan: IntentExplorationPlan,
    issue_evidence: IssueEvidenceFrame,
    limit: int,
) -> IntentExplorationPlan:
    valid_claim_ids = {claim.id for claim in issue_evidence.claims}
    valid_intent_ids = {unit.id for unit in issue_evidence.intent_units}
    intent_ids_by_claim: dict[str, list[str]] = {}
    for unit in issue_evidence.intent_units:
        for claim_id in unit.source_claim_ids:
            intent_ids_by_claim.setdefault(claim_id, []).append(unit.id)
    version_space_mode = plan.mode == "version_space"
    evidence_todo_mode = plan.mode == "evidence_todos"
    effective_limit = min(limit, 3) if version_space_mode else min(limit, 10)
    objectives: list[IntentExplorationObjective] = []
    for proposed in plan.objectives:
        if version_space_mode and (
            len(proposed.hypotheses) < 2 or not proposed.discriminator.strip()
        ):
            continue
        task_text = "\n".join((
            proposed.goal,
            proposed.rationale,
            proposed.done_when,
            proposed.discriminator,
        ))
        if _SURFACE_INVENTORY_TASK.search(task_text):
            continue
        if any(
            _questions_overlap(proposed.question, prior.question)
            for prior in objectives
        ):
            continue
        objective = proposed.model_copy(deep=True)
        objective.related_issue_ids = [
            claim_id
            for claim_id in objective.related_issue_ids
            if claim_id in valid_claim_ids
        ]
        objective.related_intent_ids = _unique([
            intent_id
            for intent_id in objective.related_intent_ids
            if intent_id in valid_intent_ids
        ] + [
            intent_id
            for claim_id in objective.related_issue_ids
            for intent_id in intent_ids_by_claim.get(claim_id, [])
        ])
        if version_space_mode:
            descriptions: set[str] = set()
            hypotheses = []
            for proposed_hypothesis in objective.hypotheses:
                description = proposed_hypothesis.description.strip()
                normalized = _normalized_question(description)
                if not normalized or normalized in descriptions:
                    continue
                descriptions.add(normalized)
                hypothesis = proposed_hypothesis.model_copy(deep=True)
                hypothesis.id = f"DIM-{len(objectives) + 1}-H{len(hypotheses) + 1}"
                hypothesis.description = description
                hypothesis.qualifiers = _unique(hypothesis.qualifiers, limit=4)
                hypotheses.append(hypothesis)
            if len(hypotheses) < 2:
                continue
            objective.id = f"DIM-{len(objectives) + 1}"
            objective.hypotheses = hypotheses[:3]
        elif evidence_todo_mode:
            objective.id = f"TASK-{len(objectives) + 1}"
            objective.hypotheses = []
            objective.discriminator = ""
        else:
            objective.id = f"TODO-{len(objectives) + 1}"
        objectives.append(objective)
        if len(objectives) >= effective_limit:
            break
    return IntentExplorationPlan(mode=plan.mode, tasks=objectives)


def _is_version_space_plan(plan: IntentExplorationPlan) -> bool:
    return plan.mode == "version_space" and bool(plan.tasks) and all(
        len(task.hypotheses) >= 2 for task in plan.tasks
    )


def _common_qualifiers(hypotheses: list[Any]) -> set[str]:
    qualifier_sets = [
        {item.strip() for item in hypothesis.qualifiers if item.strip()}
        for hypothesis in hypotheses
    ]
    if not qualifier_sets:
        return set()
    return set.intersection(*qualifier_sets)


def _reduce_version_space(
    plan: IntentExplorationPlan,
    adjudication: IntentEvidenceAdjudication,
    repository_evidence: list[Any],
) -> list[IntentDimensionResolution]:
    evidence_by_id = {item.id: item for item in repository_evidence}
    decisions_by_dimension: dict[str, Any] = {}
    duplicate_decisions: set[str] = set()
    for decision in adjudication.decisions:
        if decision.dimension_id in decisions_by_dimension:
            duplicate_decisions.add(decision.dimension_id)
            continue
        decisions_by_dimension[decision.dimension_id] = decision
    resolutions: list[IntentDimensionResolution] = []

    for dimension in plan.tasks:
        hypothesis_by_id = {
            hypothesis.id: hypothesis for hypothesis in dimension.hypotheses
        }
        initial_ids = list(hypothesis_by_id)
        remaining = set(initial_ids)
        initial_common = _common_qualifiers(list(hypothesis_by_id.values()))
        eliminated: list[EliminatedHypothesis] = []
        rejected_effects: list[str] = []
        decision = decisions_by_dimension.get(dimension.id)
        if dimension.id in duplicate_decisions:
            rejected_effects.append(
                f"{dimension.id}: adjudicator returned duplicate decisions"
            )
            decision = None

        valid_decision = bool(decision and decision.disambiguated)
        if valid_decision and decision.selected_hypothesis_id not in initial_ids:
            rejected_effects.append(
                f"{dimension.id}: selected an unknown hypothesis"
            )
            valid_decision = False
        if valid_decision:
            for citation in decision.evidence:
                evidence = evidence_by_id.get(citation.evidence_id)
                if evidence is None:
                    rejected_effects.append(
                        f"{dimension.id}/{citation.evidence_id}: unknown evidence"
                    )
                    valid_decision = False
                elif citation.quote not in evidence.observation:
                    rejected_effects.append(
                        f"{dimension.id}/{citation.evidence_id}: quote is absent "
                        "from the observation"
                    )
                    valid_decision = False

        if valid_decision:
            selected_id = decision.selected_hypothesis_id
            remaining = {selected_id}
            evidence_id = decision.evidence[0].evidence_id
            eliminated = [
                EliminatedHypothesis(
                    hypothesis_id=hypothesis_id,
                    evidence_id=evidence_id,
                    reason=decision.reason,
                )
                for hypothesis_id in initial_ids
                if hypothesis_id != selected_id
            ]
            status = "resolved"
            surviving = [hypothesis_by_id[item] for item in remaining]
            resolved_qualifiers = sorted(
                _common_qualifiers(surviving) - initial_common
            )
            if not resolved_qualifiers:
                resolved_qualifiers = [surviving[0].description]
        else:
            status = "unresolved"
            surviving = [hypothesis_by_id[item] for item in remaining]
            resolved_qualifiers = sorted(
                _common_qualifiers(surviving) - initial_common
            )

        resolutions.append(
            IntentDimensionResolution(
                dimension_id=dimension.id,
                question=dimension.goal,
                status=status,
                initial_hypothesis_ids=initial_ids,
                remaining_hypothesis_ids=[
                    item for item in initial_ids if item in remaining
                ],
                eliminated=eliminated,
                resolved_qualifiers=resolved_qualifiers,
                rejected_effects=rejected_effects,
            )
        )

    return resolutions


def _version_space_contract_inputs(
    plan: IntentExplorationPlan,
    resolutions: list[IntentDimensionResolution],
) -> tuple[list[str], list[str], list[str]]:
    dimensions = {item.id: item for item in plan.tasks}
    resolved_qualifiers: list[str] = []
    open_decisions: list[str] = []
    warnings: list[str] = []

    for resolution in resolutions:
        dimension = dimensions[resolution.dimension_id]
        hypotheses = {
            item.id: item.description for item in dimension.hypotheses
        }
        resolved_qualifiers.extend(resolution.resolved_qualifiers)
        if resolution.status == "unresolved":
            remaining = "; ".join(
                f"[{hypothesis_id}] {hypotheses[hypothesis_id]}"
                for hypothesis_id in resolution.remaining_hypothesis_ids
            )
            open_decisions.append(
                f"{resolution.dimension_id} remains unresolved: "
                f"{dimension.goal} Remaining interpretations: {remaining}"
            )
        elif resolution.status == "conflict":
            warnings.append(
                f"{resolution.dimension_id} has conflicting normative "
                f"repository evidence for: {dimension.goal}"
            )

    return (
        _unique(resolved_qualifiers, limit=4),
        _unique(open_decisions, limit=10),
        _unique(warnings, limit=3),
    )


def _candidate_registry(
    exchanges: list[DialogueExchange],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    registry: dict[str, dict[str, Any]] = {
        "preserve": {},
        "acceptance": {},
        "warning": {},
    }
    prompt_candidates: dict[str, list[dict[str, Any]]] = {
        "preserve": [],
        "acceptance": [],
        "warning": [],
    }
    seen_preserve: set[str] = set()
    seen_acceptance: set[tuple[str, str]] = set()
    seen_warning: set[str] = set()

    for exchange in exchanges:
        answer = exchange.answer
        supported_task_ids = [
            finding.task_id
            for finding in answer.task_findings
            if finding.status == "supported" and finding.evidence_ids
        ]
        all_tasks_supported = bool(answer.task_findings) and all(
            finding.status == "supported" and finding.evidence_ids
            for finding in answer.task_findings
        )
        # A shared report can be partial because one TODO remains open while a
        # different TODO established a stable compatibility boundary. Admit
        # only preservation candidates from that supported portion. Frozen
        # acceptance checks remain restricted to fully supported reports so an
        # unresolved product choice cannot be encoded by a probe.
        if answer.status == "supported" or (
            answer.status == "partial" and all_tasks_supported
        ):
            for text in answer.preserve_behavior:
                text = text.strip()
                if not text or text in seen_preserve:
                    continue
                seen_preserve.add(text)
                candidate_id = f"PRESERVE-CAND-{len(seen_preserve)}"
                value = {
                    "text": text,
                    "objective_id": exchange.objective.id,
                    "supported_task_ids": supported_task_ids,
                }
                registry["preserve"][candidate_id] = value
                prompt_candidates["preserve"].append({
                    "id": candidate_id,
                    **value,
                })

        if answer.status == "supported":
            run_actions = {
                action.get("evidence_id"): action
                for action in exchange.developer_actions
                if action.get("action") == "RunInvestigationCommand"
            }
            for proposed in answer.acceptance_checks:
                action = run_actions.get(proposed.run_evidence_id) or {}
                command = action.get("command")
                files = action.get("frozen_probe_files") or {}
                exit_code = action.get("exit_code")
                key = (proposed.description, command or "")
                if (
                    not command
                    or not files
                    or not isinstance(exit_code, int)
                    or key in seen_acceptance
                ):
                    continue
                seen_acceptance.add(key)
                candidate_id = f"ACCEPT-CAND-{len(seen_acceptance)}"
                check = FrozenAcceptanceCheck(
                    description=proposed.description,
                    command=command,
                    files=files,
                    source_evidence_id=proposed.run_evidence_id,
                    baseline_exit_code=exit_code,
                )
                registry["acceptance"][candidate_id] = {
                    "check": check,
                    "objective_id": exchange.objective.id,
                }
                prompt_candidates["acceptance"].append({
                    "id": candidate_id,
                    "description": check.description,
                    "objective_id": exchange.objective.id,
                })

        if answer.status not in {"partial", "conflict"}:
            continue
        for text in answer.conflicts_with_issue:
            text = text.strip()
            if not text or text in seen_warning:
                continue
            seen_warning.add(text)
            candidate_id = f"WARNING-CAND-{len(seen_warning)}"
            value = {
                "text": text,
                "objective_id": exchange.objective.id,
            }
            registry["warning"][candidate_id] = value
            prompt_candidates["warning"].append({
                "id": candidate_id,
                **value,
            })

    return registry, prompt_candidates


def _normalize_evidence_resolution(
    plan: IntentExplorationPlan,
    resolution: IntentEvidenceResolution,
    repository_evidence: list[Any],
    candidates: dict[str, dict[str, Any]],
) -> IntentEvidenceResolution:
    evidence_by_id = {item.id: item for item in repository_evidence}
    proposed_by_task: dict[str, IntentTaskResolution] = {}
    duplicate_task_ids: set[str] = set()
    for proposed in resolution.resolutions:
        if proposed.task_id in proposed_by_task:
            duplicate_task_ids.add(proposed.task_id)
            continue
        proposed_by_task[proposed.task_id] = proposed

    normalized: list[IntentTaskResolution] = []
    for task in plan.tasks:
        proposed = proposed_by_task.get(task.id)
        rejection = ""
        if task.id in duplicate_task_ids:
            rejection = "Resolver returned duplicate resolutions for this task."
        elif proposed is None:
            rejection = "Resolver omitted this evidence TODO."
        elif proposed.resolved:
            for citation in proposed.evidence:
                evidence = evidence_by_id.get(citation.evidence_id)
                if evidence is None:
                    rejection = (
                        f"Resolver cited unknown evidence {citation.evidence_id}."
                    )
                    break
                if citation.quote not in evidence.observation:
                    rejection = (
                        f"Resolver quote is absent from {citation.evidence_id}."
                    )
                    break

            if not rejection and proposed.resolution_kind == "preservation_boundary":
                candidate = candidates.get("preserve", {}).get(
                    proposed.contract_candidate_id
                )
                if candidate is None or proposed.contract_text != candidate["text"]:
                    rejection = "Resolver selected an invalid preservation candidate."
            elif not rejection and proposed.resolution_kind == "acceptance_check":
                candidate = candidates.get("acceptance", {}).get(
                    proposed.contract_candidate_id
                )
                if (
                    candidate is None
                    or proposed.contract_text != candidate["check"].description
                ):
                    rejection = "Resolver selected an invalid acceptance candidate."

        if rejection:
            normalized.append(
                IntentTaskResolution(
                    task_id=task.id,
                    resolved=False,
                    reason=rejection,
                )
            )
        else:
            normalized.append(proposed)

    return IntentEvidenceResolution(resolutions=normalized)


def _resolution_contract_inputs(
    resolution: IntentEvidenceResolution,
) -> tuple[list[str], IntentSynthesisSelection]:
    qualifiers: list[str] = []
    preserve_ids: list[str] = []
    acceptance_ids: list[str] = []
    for item in resolution.resolutions:
        if not item.resolved:
            continue
        if item.resolution_kind == "behavior_qualifier":
            qualifiers.append(item.contract_text)
        elif item.resolution_kind == "preservation_boundary":
            preserve_ids.append(item.contract_candidate_id)
        elif item.resolution_kind == "acceptance_check":
            acceptance_ids.append(item.contract_candidate_id)
    return (
        _unique(qualifiers, limit=4),
        IntentSynthesisSelection(
            preserve_candidate_ids=_unique(preserve_ids, limit=4),
            acceptance_candidate_ids=_unique(acceptance_ids, limit=4),
        ),
    )


def _normalize_synthesis(
    synthesis: IntentSynthesisSelection,
    candidates: dict[str, dict[str, Any]],
) -> IntentSynthesisSelection:
    return IntentSynthesisSelection(
        preserve_candidate_ids=_unique([
            candidate_id
            for candidate_id in synthesis.preserve_candidate_ids
            if candidate_id in candidates.get("preserve", {})
        ], limit=4),
        acceptance_candidate_ids=_unique([
            candidate_id
            for candidate_id in synthesis.acceptance_candidate_ids
            if candidate_id in candidates.get("acceptance", {})
        ], limit=4),
        warning_candidate_ids=_unique([
            candidate_id
            for candidate_id in synthesis.warning_candidate_ids
            if candidate_id in candidates.get("warning", {})
        ], limit=3),
    )


def _has_decisive_contract_selection(
    synthesis: IntentSynthesisSelection,
) -> bool:
    """Return whether investigation added a concrete Repair obligation."""
    return bool(
        synthesis.preserve_candidate_ids
        or synthesis.acceptance_candidate_ids
    )


def compact_reproduction_context(reproduction: Any) -> dict:
    is_reproduced = bool(getattr(reproduction, "is_reproduced", False))
    execution_records = []
    for record in getattr(reproduction, "execution_records", [])[-4:]:
        execution_records.append(
            {
                "run_id": record.run_id,
                "command": record.command,
                "exit_code": record.exit_code,
                "stdout": record.stdout[-1200:],
                "stderr": record.stderr[-1200:],
            }
        )

    return {
        "reproduction_status": (
            "reproduced" if is_reproduced else "not_reproduced"
        ),
        "evidence_validated": is_reproduced,
        "failure_class": getattr(reproduction, "failure_class", None),
        "test_command": getattr(reproduction, "test_command", None),
        "test_files": [
            {
                "path": item.get("path", ""),
                "content": item.get("content", "")[:6000],
            }
            for item in getattr(reproduction, "test_files", [])[:3]
            if isinstance(item, dict)
        ],
        "matched_claim_ids": list(
            getattr(reproduction, "matched_claim_ids", [])
        ),
        "repository_evidence": list(
            getattr(reproduction, "repository_evidence", [])
        )[:8],
        "graph_bindings": [
            item.model_dump()
            for item in getattr(reproduction, "graph_bindings", [])[:8]
        ],
        "execution_records": execution_records,
        "unresolved": list(getattr(reproduction, "unresolved", [])),
    }


class IntentDialogue:
    def __init__(
        self,
        extractor: StructuredExtractor,
        investigator: Any | None = None,
        *,
        developer: Any | None = None,
        evidence_sufficiency_reviewer: StructuredExtractor | None = None,
        max_tasks: int | None = None,
        max_objectives: int = 10,
        max_rounds: int | None = None,
        mode: str = "legacy",
        enable_test_anchor_retrieval: bool = True,
        enable_property_materialization: bool = True,
        max_contrast_refinements: int = 1,
        max_intent_witnesses: int = 10,
    ):
        self.extractor = extractor
        self.evidence_sufficiency_reviewer = evidence_sufficiency_reviewer
        self.investigator = investigator or developer
        if self.investigator is None:
            raise ValueError("IntentDialogue requires a repository investigator")
        if max_tasks is not None:
            max_objectives = max_tasks
        elif max_rounds is not None:
            max_objectives = max_rounds
        self.max_objectives = min(10, max(1, max_objectives))
        self.mode = mode
        self.enable_test_anchor_retrieval = enable_test_anchor_retrieval
        self.enable_property_materialization = enable_property_materialization
        self.max_contrast_refinements = max(0, max_contrast_refinements)
        self.max_intent_witnesses = min(10, max(1, max_intent_witnesses))

    def _review_scout_evidence(
        self,
        issue: str,
        issue_evidence: IssueEvidenceFrame,
        report: InvestigationReport,
        repository_evidence: list[RepositoryEvidence],
        executions: list[dict],
        *,
        phase: str,
        report_history: list[FocusedInvestigationReport] | None = None,
        prior_attempts: list[ContrastiveAttempt] | None = None,
        prior_reviews: list[EvidenceSufficiencyReview] | None = None,
    ) -> EvidenceSufficiencyReview | None:
        if self.evidence_sufficiency_reviewer is None:
            return None
        try:
            review = self.evidence_sufficiency_reviewer(
                build_evidence_sufficiency_review_prompt(
                    issue,
                    issue_evidence,
                    report.model_dump(),
                    _evidence_sufficiency_packet(
                        report,
                        repository_evidence,
                        report_history,
                    ),
                    executions,
                    report_history=[
                        item.model_dump() for item in (report_history or [])
                    ],
                    prior_adjudications=[
                        {
                            "attempt": attempt.attempt,
                            "pair_status": attempt.pair_status,
                            "focus_intent_ids": (
                                list(attempt.witness.issue_intent_ids)
                                if attempt.witness is not None
                                else []
                            ),
                            "witness_valid": attempt.decision.witness_valid,
                            "boundary_resolved": (
                                attempt.decision.boundary_resolved
                            ),
                            "next_action": attempt.decision.next_action,
                            "followup_request": (
                                attempt.decision.followup_request.model_dump()
                                if attempt.decision.followup_request is not None
                                else None
                            ),
                            "covered_issue_intent_ids": list(
                                attempt.decision.covered_issue_intent_ids
                            ),
                            "unresolved_issue_intent_ids": list(
                                attempt.decision.unresolved_issue_intent_ids
                            ),
                            "uncertainty": list(attempt.decision.uncertainty),
                            "reason": _bounded_text(
                                attempt.decision.reason,
                                2000,
                            ),
                        }
                        for attempt in (prior_attempts or [])
                    ],
                    prior_reviews=[
                        item.model_dump() for item in (prior_reviews or [])
                    ],
                ),
                EvidenceSufficiencyReview,
            )
        except Exception:
            logger.exception(
                "Scout evidence-sufficiency review failed: phase=%s",
                phase,
            )
            return None
        return review if isinstance(review, EvidenceSufficiencyReview) else None

    def _record_scout_evidence_review(
        self,
        actions: list[dict],
        review: EvidenceSufficiencyReview,
        *,
        phase: str,
        evidence_count: int,
    ) -> list[dict]:
        record_review = getattr(
            self.investigator,
            "record_evidence_sufficiency_review",
            None,
        )
        if callable(record_review):
            record_review(review, phase=phase)
        action = {
            "action": "ReviewEvidenceSufficiency",
            "phase": phase,
            "review": review.model_dump(),
            "evidence_count": evidence_count,
        }
        return actions if action in actions else [*actions, action]

    @staticmethod
    def _mark_report_not_hypothesis_ready(
        report: InvestigationReport,
        review: EvidenceSufficiencyReview | None,
    ) -> InvestigationReport:
        missing_fact = (
            review.next_investigation.question
            if review is not None and review.next_investigation is not None
            else "The independent evidence-sufficiency verdict is unavailable."
        )
        return report.model_copy(update={
            "status": "partial" if report.status == "supported" else report.status,
            "open_decisions": list(dict.fromkeys([
                *report.open_decisions,
                f"Evidence packet is not ready for Hypothesis: {missing_fact}",
            ]))[:10],
        })

    @staticmethod
    def _normalize_test_anchor_selection(
        selection: TestAnchorSelection | None,
        retrieval: TestAnchorRetrieval,
    ) -> TestAnchorSelection | None:
        if selection is None:
            return None
        candidate_ids = {candidate.id for candidate in retrieval.candidates}
        if selection.status == "no_suitable_candidate":
            return selection
        selected_ids = [
            candidate_id
            for candidate_id in selection.selected_candidate_ids
            if candidate_id in candidate_ids
        ]
        if not selected_ids:
            return None
        primary_id = (
            selection.primary_candidate_id
            if selection.primary_candidate_id in selected_ids
            else selected_ids[0]
        )
        assessments = [
            assessment
            for assessment in selection.assessments
            if assessment.candidate_id in candidate_ids
        ]
        proposed_groups = {
            group.intent_id: group for group in selection.intent_selections
        }
        normalized_groups = []
        for retrieval_group in retrieval.intent_groups:
            allowed_ids = set(retrieval_group.candidate_ids)
            proposed = proposed_groups.get(retrieval_group.intent_id)
            selected_group_ids = (
                [
                    candidate_id
                    for candidate_id in proposed.selected_candidate_ids
                    if candidate_id in allowed_ids
                ]
                if proposed is not None
                else []
            )
            group_primary = (
                proposed.primary_candidate_id
                if proposed is not None
                and proposed.primary_candidate_id in selected_group_ids
                else (selected_group_ids[0] if selected_group_ids else "")
            )
            normalized_groups.append(IntentAnchorSelectionGroup(
                intent_id=retrieval_group.intent_id,
                selected_candidate_ids=selected_group_ids,
                primary_candidate_id=group_primary,
                rationale=(
                    proposed.rationale
                    if proposed is not None
                    else "The reranker omitted this atomic intent group."
                ),
            ))
        return selection.model_copy(update={
            "selected_candidate_ids": selected_ids,
            "primary_candidate_id": primary_id,
            "assessments": assessments,
            "intent_selections": normalized_groups,
        })

    @staticmethod
    def _contrastive_executions(
        proposal,
        actions: list[dict],
    ) -> list[dict]:
        if proposal is None:
            return []
        action_by_run = {
            action.get("evidence_id"): action
            for action in actions
            if action.get("action") == "RunInvestigationCommand"
        }
        executions = []
        candidates = [proposal.anchor, proposal.target]
        for candidate in candidates:
            run_id = candidate.run_evidence_id
            action = action_by_run.get(run_id)
            if action is None:
                continue
            files = dict(action.get("frozen_probe_files") or {})
            executions.append({
                "candidate_id": candidate.id,
                "description": candidate.description,
                "run_evidence_id": run_id,
                "command": action.get("frozen_probe_command")
                or action.get("command", ""),
                "exit_code": action.get("exit_code"),
                "files": files,
                "run_binding": {
                    "compile": action.get("compile_run_id", run_id),
                    "execution": action.get("execution_run_id", run_id),
                },
            })
        if len(executions) == 2:
            anchor_files = executions[0]["files"]
            target_files = executions[1]["files"]

            def combined_source(files: dict[str, str]) -> str:
                if len(files) == 1:
                    return next(iter(files.values()))
                return "\n".join(
                    f"// FILE: {path}\n{content}"
                    for path, content in sorted(files.items())
                )

            anchor_source = combined_source(anchor_files)
            target_source = combined_source(target_files)
            source_diff = "".join(unified_diff(
                anchor_source.splitlines(keepends=True),
                target_source.splitlines(keepends=True),
                fromfile="ANCHOR-1",
                tofile="TARGET-1",
                n=3,
            ))
            executions[1]["source_diff_from_anchor"] = source_diff[:12000]
        return executions

    @staticmethod
    def _boundary_test_executions(
        report: InvestigationReport,
        evidence: list,
        actions: list[dict],
    ) -> list[dict]:
        action_by_run = {
            action.get("evidence_id"): action
            for action in actions
            if action.get("action") == "RunInvestigationCommand"
        }
        observation_by_id = {item.id: item.observation for item in evidence}
        executions = []
        for index, check in enumerate(report.acceptance_checks, 1):
            action = action_by_run.get(check.run_evidence_id)
            if action is None:
                continue
            executions.append({
                "candidate_id": f"CHECK-{index}",
                "run_evidence_id": check.run_evidence_id,
                "kind": check.kind,
                "description": check.description,
                "covered_issue_claim_ids": check.covered_issue_claim_ids,
                "covered_issue_intent_ids": check.covered_issue_intent_ids,
                "covered_property_ids": check.covered_property_ids,
                "covered_route_ids": check.covered_route_ids,
                "command": action.get("frozen_probe_command")
                or action.get("command", ""),
                "exit_code": action.get("exit_code"),
                "files": dict(action.get("frozen_probe_files") or {}),
                "output": observation_by_id.get(check.run_evidence_id, ""),
            })
        return executions

    @staticmethod
    def _semantic_admission_evidence_packet(
        contract: RepairIntentContract,
        decision: ContrastiveIntentDecision,
        report: InvestigationReport,
        repository_evidence: list[RepositoryEvidence],
    ) -> list[dict]:
        cited_ids = set(report.repository_evidence_ids)
        for property_item in contract.properties:
            cited_ids.update(property_item.evidence_ids)
            for dimension in property_item.dimensions:
                cited_ids.update(dimension.evidence_ids)
        for effect in decision.repository_effects:
            cited_ids.update(
                citation.evidence_id for citation in effect.evidence
            )

        cited_evidence = [
            item for item in repository_evidence if item.id in cited_ids
        ]
        uncited_evidence = [
            item for item in repository_evidence if item.id not in cited_ids
        ]
        packet = []
        seen_ids: set[str] = set()
        for item in [*cited_evidence, *uncited_evidence]:
            if item.id in seen_ids:
                continue
            seen_ids.add(item.id)
            packet.append({
                "id": item.id,
                "path": item.path[:500],
                "locator": item.locator[:800],
                "observation": item.observation[:1600],
                "cited_by_proposed_contract": item.id in cited_ids,
            })
            if len(packet) == 40:
                break
        return packet

    def _extract_intent_properties(
        self,
        issue: str,
        issue_evidence: IssueEvidenceFrame,
        contract: RepairIntentContract,
        repository_evidence: list[RepositoryEvidence],
        executions: list[dict],
    ) -> IntentPropertyExtraction | None:
        if not self.enable_property_materialization:
            logger.info(
                "Intent property extraction and boundary materialization "
                "skipped for bounded portfolio Scout mode."
            )
            return None
        proposed = self.extractor(
            build_intent_property_extraction_prompt(
                issue,
                issue_evidence,
                contract,
                [item.model_dump() for item in repository_evidence],
                executions,
            ),
            IntentPropertyExtraction,
        )
        if proposed is None:
            logger.warning(
                "Intent property extraction was unavailable; executable "
                "property checks will not be materialized."
            )
            return None

        issue_ids = {claim.id for claim in issue_evidence.claims}
        issue_kind_by_id = {
            claim.id: claim.kind for claim in issue_evidence.claims
        }
        intent_ids = {unit.id for unit in issue_evidence.intent_units}
        target_claim_ids_by_intent = {
            unit.id: [
                claim_id
                for claim_id in unit.source_claim_ids
                if issue_kind_by_id.get(claim_id) == "expected_behavior"
            ]
            for unit in issue_evidence.intent_units
            if unit.role == "target"
        }
        repository_ids = {item.id for item in repository_evidence}
        execution_ids = {
            item.get("run_evidence_id")
            for item in executions
            if item.get("run_evidence_id")
        }
        valid_evidence_ids = issue_ids | intent_ids | repository_ids | execution_ids
        normalized: list[IntentBehaviorProperty] = []
        for candidate in proposed.properties:
            evidence_ids = _unique([
                evidence_id
                for evidence_id in candidate.evidence_ids
                if evidence_id in valid_evidence_ids
            ])
            cited_issue_ids = issue_ids & set(evidence_ids)
            cited_intent_ids = [
                intent_id
                for intent_id in candidate.issue_intent_ids
                if intent_id in intent_ids
            ]
            if candidate.role == "target" and cited_intent_ids:
                # INTENT-N already carries validated Issue provenance. Recover
                # an omitted duplicate EXP-N citation without changing the
                # candidate's semantics or accepting an unbound target.
                evidence_ids = _unique([
                    *evidence_ids,
                    *(
                        claim_id
                        for intent_id in cited_intent_ids
                        for claim_id in target_claim_ids_by_intent.get(intent_id, [])
                    ),
                ])
                cited_issue_ids = issue_ids & set(evidence_ids)
            if not evidence_ids:
                continue
            if (
                candidate.role == "target"
                and (
                    candidate.authority != "issue_explicit"
                    or not cited_issue_ids
                    or (intent_ids and not cited_intent_ids)
                )
            ):
                continue
            if (
                candidate.authority == "issue_explicit"
                and not cited_issue_ids
            ):
                continue
            dimensions: list[IntentPropertyDimension] = []
            for dimension in candidate.dimensions:
                dimension_evidence_ids = _unique([
                    evidence_id
                    for evidence_id in dimension.evidence_ids
                    if evidence_id in valid_evidence_ids
                ])
                if not dimension_evidence_ids:
                    continue
                dimensions.append(
                    dimension.model_copy(update={
                        "representatives": _unique(
                            dimension.representatives,
                            limit=8,
                        ),
                        "evidence_ids": dimension_evidence_ids,
                    })
                )
            normalized.append(
                candidate.model_copy(update={
                    "id": f"PROP-{len(normalized) + 1}",
                    "scope": _unique(candidate.scope, limit=4),
                    "issue_intent_ids": cited_intent_ids,
                    "evidence_ids": evidence_ids,
                    "dimensions": dimensions[:5],
                })
            )
            if len(normalized) == 10:
                break
        result = IntentPropertyExtraction(properties=normalized)
        logger.info(
            "Intent property extraction completed: proposed=%d accepted=%d "
            "target=%d preserve=%d",
            len(proposed.properties),
            len(normalized),
            sum(item.role == "target" for item in normalized),
            sum(item.role == "preserve" for item in normalized),
        )
        return result

    def _route_intent_properties(
        self,
        issue: str,
        issue_evidence: IssueEvidenceFrame,
        contract: RepairIntentContract,
        repository_evidence: list[RepositoryEvidence],
        executions: list[dict],
    ) -> IntentValidationRouting | None:
        if not contract.properties:
            return IntentValidationRouting()
        proposed = self.extractor(
            build_intent_validation_routing_prompt(
                issue,
                issue_evidence,
                contract,
                [item.model_dump() for item in repository_evidence],
                executions,
            ),
            IntentValidationRouting,
        )
        if proposed is None:
            logger.warning(
                "Intent validation routing was unavailable; properties remain "
                "text-only and no executable checks will be materialized."
            )
            return None

        property_ids = {item.id for item in contract.properties}
        normalized: list[IntentValidationRoute] = []
        routed_property_ids: set[str] = set()
        for candidate in proposed.routes:
            if (
                candidate.property_id not in property_ids
                or candidate.property_id in routed_property_ids
            ):
                continue
            normalized.append(candidate.model_copy(update={
                "id": f"ROUTE-{len(normalized) + 1}",
            }))
            routed_property_ids.add(candidate.property_id)

        for property_id in sorted(
            property_ids - routed_property_ids,
            key=lambda value: int(value.split("-")[-1]),
        ):
            normalized.append(IntentValidationRoute(
                id=f"ROUTE-{len(normalized) + 1}",
                property_id=property_id,
                mode="text_only",
                rationale=(
                    "The router omitted this grounded property, so it remains "
                    "repair guidance without an invented executable oracle."
                ),
            ))
        result = IntentValidationRouting(routes=normalized[:10])
        logger.info(
            "Intent validation routing completed: total=%d generative=%d "
            "fixed=%d compile=%d structural=%d text_only=%d",
            len(result.routes),
            sum(item.mode == "generative_property" for item in result.routes),
            sum(item.mode == "fixed_example" for item in result.routes),
            sum(item.mode == "compile_matrix" for item in result.routes),
            sum(item.mode == "structural_review" for item in result.routes),
            sum(item.mode == "text_only" for item in result.routes),
        )
        return result

    def _materialize_boundary_tests(
        self,
        issue: str,
        issue_evidence: IssueEvidenceFrame,
        contract: RepairIntentContract,
        decision: ContrastiveIntentDecision,
        routing: IntentValidationRouting | None,
        repository_evidence: list[RepositoryEvidence] | None = None,
        refinement_source: tuple[
            InvestigationReport,
            IntentBoundaryTestAssessment,
        ] | None = None,
        allow_refinement: bool = True,
    ) -> tuple[
        InvestigationReport | None,
        IntentBoundaryTestAssessment | None,
        list[FrozenAcceptanceCheck],
        list[RepositoryEvidence] | None,
        list[dict] | None,
    ]:
        executable_modes = {
            "generative_property",
            "fixed_example",
            "compile_matrix",
        }
        if (
            not contract.properties
            or routing is None
            or not any(route.mode in executable_modes for route in routing.routes)
        ):
            logger.info(
                "Intent boundary-test materialization skipped: no grounded "
                "property has an executable validation route."
            )
            return None, None, [], None, None

        method_name = (
            "refine_intent_boundary"
            if refinement_source is not None
            else "materialize_intent_boundary"
        )
        materialize = getattr(self.investigator, method_name, None)
        remaining_steps = getattr(self.investigator, "remaining_steps", 1)
        if not callable(materialize) or remaining_steps <= 0:
            return None, None, [], None, None

        materialize_args = [
            issue,
            issue_evidence,
            contract,
            decision,
            routing,
        ]
        if refinement_source is not None:
            materialize_args.extend(refinement_source)
        report, evidence, actions = materialize(*materialize_args)
        executions = self._boundary_test_executions(report, evidence, actions)
        if not executions:
            logger.info(
                "Intent boundary-test materialization produced no executable "
                "candidates."
            )
            return report, None, [], evidence, actions

        combined_evidence_by_id = {
            item.id: item
            for item in [*(repository_evidence or []), *evidence]
        }
        combined_evidence = list(combined_evidence_by_id.values())
        admission_evidence = self._semantic_admission_evidence_packet(
            contract,
            decision,
            report,
            combined_evidence,
        )

        assessment = self.extractor(
            build_intent_boundary_test_assessment_prompt(
                issue,
                issue_evidence,
                contract,
                decision,
                routing,
                admission_evidence,
                report.model_dump(),
                executions,
            ),
            IntentBoundaryTestAssessment,
        )
        if assessment is None:
            logger.warning(
                "Intent boundary-test assessment was unavailable; no new "
                "required checks were admitted."
            )
            return report, None, [], evidence, actions

        verdict_by_candidate = {
            item.candidate_id: item for item in assessment.checks
        }
        submitted_candidate_ids = {
            item["candidate_id"] for item in executions
        }
        duplicate_verdict_candidate_ids = sorted({
            item.candidate_id
            for item in assessment.checks
            if sum(
                other.candidate_id == item.candidate_id
                for other in assessment.checks
            ) > 1
        })
        unknown_verdict_candidate_ids = sorted(
            set(verdict_by_candidate) - submitted_candidate_ids
        )
        missing_verdict_candidate_ids = sorted(
            submitted_candidate_ids - set(verdict_by_candidate)
        )
        candidate_bindings_valid = not (
            duplicate_verdict_candidate_ids
            or unknown_verdict_candidate_ids
            or missing_verdict_candidate_ids
        )
        valid_claim_ids = {claim.id for claim in issue_evidence.claims}
        valid_intent_ids = {unit.id for unit in issue_evidence.intent_units}
        property_roles = {
            item.id: item.role for item in contract.properties
        }
        unknown_diagnostic_property_ids = sorted(
            set(assessment.diagnostic_property_ids) - set(property_roles)
        )
        diagnostic_property_ids = {
            property_id
            for property_id in assessment.diagnostic_property_ids
            if property_id in property_roles
        }
        unresolved_scope_property_ids = {
            property_id
            for verdict in assessment.checks
            if not verdict.scope_resolved
            for property_id in verdict.covered_property_ids
            if property_id in property_roles
        }
        diagnostic_property_ids.update(unresolved_scope_property_ids)
        if unresolved_scope_property_ids != set(
            assessment.diagnostic_property_ids
        ).intersection(property_roles):
            assessment = assessment.model_copy(update={
                "diagnostic_property_ids": sorted(diagnostic_property_ids),
            })
        route_by_id = {item.id: item for item in routing.routes}
        valid_admission_evidence_ids = (
            valid_claim_ids
            | valid_intent_ids
            | {item["id"] for item in admission_evidence}
            | {
                candidate["run_evidence_id"]
                for candidate in executions
            }
        )
        assessment_evidence_valid = bool(assessment.evidence_ids) and set(
            assessment.evidence_ids
        ).issubset(valid_admission_evidence_ids)
        invalid_verdict_citations = [
            verdict.candidate_id
            for verdict in assessment.checks
            if (
                not verdict.evidence_ids
                or not set(verdict.evidence_ids).issubset(
                    valid_admission_evidence_ids
                )
            )
        ]
        if (
            not assessment_evidence_valid
            or invalid_verdict_citations
            or unknown_diagnostic_property_ids
            or duplicate_verdict_candidate_ids
            or unknown_verdict_candidate_ids
            or missing_verdict_candidate_ids
        ):
            citation_gaps = list(assessment.missing_discriminators)
            if not assessment_evidence_valid:
                citation_gaps.append(
                    "Semantic admission cited evidence outside the supplied "
                    "packet, so its global completeness claim is unverified."
                )
            if invalid_verdict_citations:
                citation_gaps.append(
                    "Candidate judgments with unavailable evidence: "
                    + ", ".join(invalid_verdict_citations)
                )
            if unknown_diagnostic_property_ids:
                citation_gaps.append(
                    "Semantic admission referenced unknown properties: "
                    + ", ".join(unknown_diagnostic_property_ids)
                )
            if duplicate_verdict_candidate_ids:
                citation_gaps.append(
                    "Semantic admission returned duplicate candidate IDs: "
                    + ", ".join(duplicate_verdict_candidate_ids)
                )
            if unknown_verdict_candidate_ids:
                citation_gaps.append(
                    "Semantic admission referenced unknown candidates: "
                    + ", ".join(unknown_verdict_candidate_ids)
                )
            if missing_verdict_candidate_ids:
                citation_gaps.append(
                    "Semantic admission omitted submitted candidates: "
                    + ", ".join(missing_verdict_candidate_ids)
                )
            assessment = assessment.model_copy(update={
                "contract_consistent": (
                    assessment.contract_consistent
                    and assessment_evidence_valid
                    and candidate_bindings_valid
                ),
                "boundary_complete": False,
                "diagnostic_property_ids": sorted(diagnostic_property_ids),
                "missing_discriminators": _unique(citation_gaps, limit=4),
            })
            verdict_by_candidate = {
                item.candidate_id: item for item in assessment.checks
            }
        contract_consistent = (
            assessment.contract_consistent
            and assessment_evidence_valid
            and candidate_bindings_valid
        )
        frozen_checks = []
        for candidate in executions:
            verdict = verdict_by_candidate.get(candidate["candidate_id"])
            candidate_property_ids = set(candidate["covered_property_ids"])
            candidate_route_ids = set(candidate["covered_route_ids"])
            covered_property_ids = [
                property_id
                for property_id in verdict.covered_property_ids
                if property_id in candidate_property_ids
                and property_id in property_roles
            ] if verdict is not None else []
            covered_route_ids = [
                route_id
                for route_id in verdict.covered_route_ids
                if route_id in candidate_route_ids
                and route_id in route_by_id
                and route_by_id[route_id].mode in executable_modes
            ] if verdict is not None else []
            property_intent_ids = {
                intent_id
                for property_id in covered_property_ids
                for intent_id in next(
                    item.issue_intent_ids
                    for item in contract.properties
                    if item.id == property_id
                )
            }
            covered_intent_ids = [
                intent_id
                for intent_id in verdict.covered_issue_intent_ids
                if intent_id in valid_intent_ids
                and intent_id in candidate["covered_issue_intent_ids"]
                and intent_id in property_intent_ids
            ] if verdict is not None else []
            expected_role = (
                "target" if candidate["kind"] == "f2p" else "preserve"
            )
            if (
                verdict is None
                or not assessment_evidence_valid
                or not verdict.normative
                or not verdict.discriminating
                or not verdict.scope_resolved
                or any(
                    property_id in diagnostic_property_ids
                    for property_id in covered_property_ids
                )
                or not verdict.evidence_ids
                or not set(verdict.evidence_ids).issubset(
                    valid_admission_evidence_ids
                )
                or verdict.kind != candidate["kind"]
                or not candidate["command"]
                or not candidate["files"]
                or not covered_property_ids
                or not covered_route_ids
                or (
                    candidate["kind"] == "f2p"
                    and bool(valid_intent_ids)
                    and not covered_intent_ids
                )
                or any(
                    property_roles[property_id] != expected_role
                    for property_id in covered_property_ids
                )
                or any(
                    route_by_id[route_id].property_id not in covered_property_ids
                    for route_id in covered_route_ids
                )
            ):
                continue
            exit_code = candidate["exit_code"]
            if (
                not isinstance(exit_code, int)
                or (candidate["kind"] == "f2p" and exit_code == 0)
                or (candidate["kind"] == "p2p" and exit_code != 0)
            ):
                continue
            frozen_checks.append(
                FrozenAcceptanceCheck(
                    candidate_id=candidate["candidate_id"],
                    description=candidate["description"][:500],
                    command=candidate["command"],
                    files=candidate["files"],
                    source_evidence_id=candidate["run_evidence_id"],
                    baseline_exit_code=exit_code,
                    kind=candidate["kind"],
                    authority="normative",
                    covered_issue_claim_ids=[
                        claim_id
                        for claim_id in verdict.covered_issue_claim_ids
                        if claim_id in valid_claim_ids
                    ],
                    covered_issue_intent_ids=covered_intent_ids,
                    covered_property_ids=covered_property_ids,
                    covered_route_ids=covered_route_ids,
                )
            )
        logger.info(
            "Intent boundary-test materialization completed: candidates=%d "
            "accepted=%d f2p=%d p2p=%d contract_consistent=%s "
            "boundary_complete=%s",
            len(executions),
            len(frozen_checks),
            sum(check.kind == "f2p" for check in frozen_checks),
            sum(check.kind == "p2p" for check in frozen_checks),
            contract_consistent,
            assessment.boundary_complete,
        )
        missing_discriminators = _unique([
            *assessment.missing_discriminators,
            *(
                item.missing_discriminator
                for item in assessment.checks
                if item.missing_discriminator
            ),
        ])
        refine = getattr(self.investigator, "refine_intent_boundary", None)
        if (
            allow_refinement
            and missing_discriminators
            and (
                not assessment.contract_consistent
                or not assessment.boundary_complete
            )
            and callable(refine)
            and getattr(self.investigator, "remaining_steps", 0) > 0
        ):
            logger.info(
                "Intent boundary-test refinement requested within the "
                "remaining shared budget: %s",
                "; ".join(missing_discriminators[:2]),
            )
            (
                refined_report,
                refined_assessment,
                refined_checks,
                refined_evidence,
                refined_actions,
            ) = self._materialize_boundary_tests(
                issue,
                issue_evidence,
                contract,
                decision,
                routing,
                repository_evidence,
                refinement_source=(report, assessment),
                allow_refinement=False,
            )
            if refined_assessment is not None:
                return (
                    refined_report,
                    refined_assessment,
                    _merge_acceptance_checks(
                        refined_checks,
                        frozen_checks,
                        limit=4,
                    ),
                    refined_evidence,
                    refined_actions,
                )
        return report, assessment, frozen_checks[:4], evidence, actions

    def _contrastive_pair_state(
        self,
        report: InvestigationReport,
        actions: list[dict],
        anchor_retrieval: TestAnchorRetrieval | None,
    ) -> tuple[str, list[dict]]:
        proposal = report.contrastive_witness
        if proposal is None:
            return "contrastive_no_witness", []
        known_anchor_ids = {
            candidate.id
            for candidate in (
                anchor_retrieval.candidates if anchor_retrieval else []
            )
        }
        if (
            proposal.anchor_candidate_id
            and proposal.anchor_candidate_id not in known_anchor_ids
        ):
            return "contrastive_invalid_anchor_candidate_binding", []
        executions = self._contrastive_executions(proposal, actions)
        exit_codes = {
            item["candidate_id"]: item.get("exit_code")
            for item in executions
        }
        if (
            exit_codes.get("ANCHOR-1") != 0
            or exit_codes.get("TARGET-1") in {None, 0}
        ):
            return "contrastive_invalid_execution_pair", executions
        return "", executions

    @staticmethod
    def _without_witness_content(
        decision: ContrastiveIntentDecision,
        **updates,
    ) -> ContrastiveIntentDecision:
        payload = decision.model_dump()
        payload.update({
            "witness_valid": False,
            "boundary_resolved": False,
            "selected_anchor_id": "",
            "selected_target_id": "",
            "witnessed_delta": "",
            "supported_delta": "",
            "covered_issue_claim_ids": [],
            "covered_issue_intent_ids": [],
            "unresolved_issue_claim_ids": [],
            "unresolved_issue_intent_ids": [],
            "scope_evidence": [],
            "uncertainty": [],
        })
        payload.update(updates)
        return ContrastiveIntentDecision.model_validate(payload)

    @classmethod
    def _enforce_pair_status(
        cls,
        decision: ContrastiveIntentDecision,
        pair_stop_reason: str,
    ) -> ContrastiveIntentDecision:
        if not pair_stop_reason or not decision.witness_valid:
            return decision
        return cls._without_witness_content(
            decision,
            next_action="reject_witness",
            followup_request=None,
        )

    @staticmethod
    def _normalize_intent_coverage(
        issue_evidence: IssueEvidenceFrame,
        proposal,
        decision: ContrastiveIntentDecision,
    ) -> ContrastiveIntentDecision:
        if not decision.witness_valid:
            return IntentDialogue._without_witness_content(decision)
        known_ids = {unit.id for unit in issue_evidence.intent_units}
        if not known_ids:
            return decision
        proposed_ids = {
            intent_id
            for intent_id in (
                proposal.issue_intent_ids if proposal is not None else []
            )
            if intent_id in known_ids
        }
        covered_ids = _unique([
            intent_id
            for intent_id in decision.covered_issue_intent_ids
            if intent_id in proposed_ids
        ])
        unresolved_ids = _unique(
            [
                intent_id
                for intent_id in decision.unresolved_issue_intent_ids
                if intent_id in known_ids and intent_id not in covered_ids
            ]
            + [
                unit.id
                for unit in issue_evidence.intent_units
                if unit.id not in covered_ids
            ]
        )
        return decision.model_copy(update={
            "covered_issue_intent_ids": covered_ids,
            "unresolved_issue_intent_ids": unresolved_ids,
        })

    @staticmethod
    def _apply_intent_ledger_evidence(
        contract: RepairIntentContract,
        proposal,
        decision: ContrastiveIntentDecision,
        *,
        attempt: int = 1,
    ) -> None:
        effects_by_intent: dict[str, list] = {}
        for effect in decision.repository_effects:
            for intent_id in effect.issue_intent_ids:
                effects_by_intent.setdefault(intent_id, []).append(effect)
        covered_ids = set(decision.covered_issue_intent_ids)
        if (
            proposal is not None
            and not covered_ids
            and not proposal.issue_intent_ids
        ):
            covered_claim_ids = set(decision.covered_issue_claim_ids)
            covered_ids = {
                entry.id
                for entry in contract.intent_ledger
                if covered_claim_ids.intersection(entry.source_claim_ids)
            }
        unresolved_ids = set(decision.unresolved_issue_intent_ids)
        if proposal is not None:
            unresolved_ids.update(
                entry.id
                for entry in contract.intent_ledger
                if entry.id not in covered_ids
            )
        witness_evidence_ids = (
            [proposal.anchor.run_evidence_id, proposal.target.run_evidence_id]
            if proposal is not None
            else []
        )
        updated = []
        status_rank = {
            "uninvestigated": 0,
            "unresolved": 1,
            "repository_evidence": 2,
            "local_witness": 3,
            "resolved": 4,
            "conflict": 4,
        }
        for entry in contract.intent_ledger:
            effects = effects_by_intent.get(entry.id, [])
            evidence_ids = _unique(
                entry.evidence_ids
                + [
                    citation.evidence_id
                    for effect in effects
                    for citation in effect.evidence
                ]
                + (witness_evidence_ids if entry.id in covered_ids else []),
                limit=10,
            )
            limitations = [
                item
                for item in entry.limitations
                if item != (
                    "No accepted witness or repository effect settled this "
                    "atomic intent."
                )
            ]
            proposed_status = entry.status
            if entry.id in covered_ids:
                proposed_status = (
                    "resolved" if decision.boundary_resolved else "local_witness"
                )
                if not decision.boundary_resolved:
                    limitations.append(
                        "The current test pair is a local witness, not a "
                        "complete boundary."
                    )
            elif any(effect.kind == "open_decision" for effect in effects):
                proposed_status = "unresolved"
            elif effects:
                proposed_status = "repository_evidence"
            elif entry.id in unresolved_ids:
                proposed_status = "unresolved"
                limitations.append(
                    "No accepted witness or repository effect settled this "
                    "atomic intent."
                )
            status = (
                proposed_status
                if status_rank[proposed_status] > status_rank[entry.status]
                else entry.status
            )
            if status in {"local_witness", "resolved"}:
                limitations = [
                    item
                    for item in limitations
                    if item != (
                        "No accepted witness or repository effect settled this "
                        "atomic intent."
                    )
                ]
            updated.append(entry.model_copy(update={
                "status": status,
                "evidence_ids": evidence_ids,
                "witness_ids": _unique(
                    entry.witness_ids
                    + (
                        [
                            f"PAIR-{attempt}:ANCHOR-1",
                            f"PAIR-{attempt}:TARGET-1",
                        ]
                        if entry.id in covered_ids
                        else []
                    ),
                    limit=5,
                ),
                "witnessed_delta": (
                    decision.witnessed_delta
                    if entry.id in covered_ids and decision.witnessed_delta
                    else entry.witnessed_delta
                ),
                "supported_delta": (
                    decision.supported_delta
                    if entry.id in covered_ids and decision.boundary_resolved
                    else entry.supported_delta
                ),
                "limitations": _unique(limitations, limit=5),
            }))
        contract.intent_ledger = updated

    @staticmethod
    def _attach_intent_evidence_notes(
        contract: RepairIntentContract,
        repository_evidence: list[RepositoryEvidence],
    ) -> None:
        """Carry a small amount of cited evidence into the Repair handoff."""
        evidence_by_id = {item.id: item for item in repository_evidence}
        updated = []
        for entry in contract.intent_ledger:
            notes = list(entry.evidence_notes)
            for evidence_id in entry.evidence_ids:
                evidence = evidence_by_id.get(evidence_id)
                if evidence is None:
                    continue
                observation = " ".join(evidence.observation.split())
                if not observation:
                    continue
                location = evidence.path or evidence.locator
                notes.append(
                    f"[{evidence_id}] {location}: {observation[:360]}"
                )
                if len(_unique(notes)) >= 3:
                    break
            updated.append(entry.model_copy(update={
                "evidence_notes": _unique(notes, limit=3),
            }))
        contract.intent_ledger = updated

    @staticmethod
    def _reconcile_semantic_admission(
        contract: RepairIntentContract,
        assessment: IntentBoundaryTestAssessment,
    ) -> None:
        """Keep rejected policy branches out of the Repair handoff."""
        diagnostic_property_ids = set(assessment.diagnostic_property_ids)
        rejected_candidate_ids = {
            verdict.candidate_id
            for verdict in assessment.checks
            if (
                not verdict.normative
                or not verdict.discriminating
                or not verdict.scope_resolved
                or bool(
                    diagnostic_property_ids
                    & set(verdict.covered_property_ids)
                )
            )
        }
        contract.acceptance_checks = [
            check
            for check in contract.acceptance_checks
            if (
                check.candidate_id not in rejected_candidate_ids
                and not bool(
                    diagnostic_property_ids
                    & set(check.covered_property_ids)
                )
            )
        ]
        if not assessment.contract_consistent:
            contract.acceptance_checks = [
                check
                for check in contract.acceptance_checks
                if check.authority == "normative"
            ]
        disputed_intent_ids = {
            intent_id
            for item in contract.properties
            if item.id in diagnostic_property_ids
            for intent_id in item.issue_intent_ids
        }
        for verdict in assessment.checks:
            if (
                verdict.normative
                and verdict.scope_resolved
                and verdict.run_evidence_id
            ):
                continue
            disputed_intent_ids.update(verdict.covered_issue_intent_ids)
        if not disputed_intent_ids:
            return

        limitation = (
            "Final semantic admission rejected a proposed policy branch for "
            "this intent. Only the explicit Issue target and admitted "
            "properties are binding."
        )
        contract.intent_ledger = [
            item.model_copy(update={
                "status": (
                    "local_witness"
                    if item.witnessed_delta
                    else "unresolved"
                ),
                "supported_delta": "",
                "limitations": _unique(
                    [*item.limitations, limitation],
                    limit=5,
                ),
            })
            if item.id in disputed_intent_ids
            else item
            for item in contract.intent_ledger
        ]

        def is_disputed_delta(delta: str) -> bool:
            binding = re.match(r"^\[([^\]]+)\]", delta)
            if binding is None:
                target_ids = {
                    item.id
                    for item in contract.intent_ledger
                    if item.role == "target"
                }
                return bool(target_ids) and target_ids <= disputed_intent_ids
            bound_ids = {
                item.strip() for item in binding.group(1).split(",")
            }
            return bool(bound_ids & disputed_intent_ids)

        contract.behavioral_delta = [
            delta
            for delta in contract.behavioral_delta
            if not is_disputed_delta(delta)
        ]

    @staticmethod
    def _prune_diagnostic_contract(
        contract: RepairIntentContract,
        diagnostic_property_ids: set[str],
    ) -> None:
        """Remove protocol records that depend on rejected properties."""
        contract.properties = [
            item
            for item in contract.properties
            if item.id not in diagnostic_property_ids
        ]
        retained_property_ids = {item.id for item in contract.properties}
        contract.validation_routes = [
            route
            for route in contract.validation_routes
            if route.property_id in retained_property_ids
        ]
        retained_route_ids = {
            route.id for route in contract.validation_routes
        }
        contract.acceptance_checks = [
            check
            for check in contract.acceptance_checks
            if set(check.covered_property_ids) <= retained_property_ids
            and set(check.covered_route_ids) <= retained_route_ids
        ]

    def _finalize_semantic_contract(
        self,
        issue: str,
        issue_evidence: IssueEvidenceFrame,
        contract: RepairIntentContract,
        repository_evidence: list[RepositoryEvidence],
    ) -> IntentContractSemanticAudit | None:
        """Audit all merged derived clauses before they can instruct Repair."""
        clauses: list[dict] = []
        property_clause_ids: dict[str, str] = {}
        for item in contract.properties:
            property_clause_ids[item.id] = item.id
            clauses.append({
                "id": item.id,
                "kind": "behavior_property",
                "role": item.role,
                "proposed_authority": item.authority,
                "text": item.predicate,
                "scope": item.scope,
                "oracle": item.oracle,
                "source_evidence_ids": item.evidence_ids,
                "issue_intent_ids": item.issue_intent_ids,
            })

        check_clause_ids: dict[int, str] = {}
        for index, check in enumerate(contract.acceptance_checks, 1):
            clause_id = f"ACCEPT-{index}"
            check_clause_ids[index - 1] = clause_id
            clauses.append({
                "id": clause_id,
                "kind": "executable_check",
                "role": check.kind,
                "proposed_authority": check.authority,
                "text": check.description,
                "command": check.command,
                "files": {
                    path: _bounded_text(content)
                    for path, content in check.files.items()
                },
                "baseline_exit_code": check.baseline_exit_code,
                "source_evidence_ids": _unique([
                    check.source_evidence_id,
                    *check.covered_issue_claim_ids,
                    *check.covered_issue_intent_ids,
                ]),
                "covered_property_ids": check.covered_property_ids,
                "covered_route_ids": check.covered_route_ids,
            })

        delta_values = _unique([
            *contract.behavioral_delta,
            *(
                item.supported_delta
                for item in contract.intent_ledger
                if item.supported_delta
            ),
        ], limit=10)
        delta_clause_ids = {
            value: f"DELTA-{index}"
            for index, value in enumerate(delta_values, 1)
        }
        for value, clause_id in delta_clause_ids.items():
            bound_intents = [
                item.id
                for item in contract.intent_ledger
                if item.supported_delta == value
            ]
            source_ids = _unique([
                evidence_id
                for item in contract.intent_ledger
                if item.id in bound_intents
                for evidence_id in [
                    *item.source_claim_ids,
                    *item.evidence_ids,
                ]
            ])
            clauses.append({
                "id": clause_id,
                "kind": "test_derived_delta",
                "text": value,
                "issue_intent_ids": bound_intents,
                "source_evidence_ids": source_ids,
            })

        if not clauses:
            return None

        evidence_packet = [
            {
                "id": claim.id,
                "authority": "issue",
                "observation": claim.text,
                "issue_span": claim.issue_span,
            }
            for claim in issue_evidence.claims
        ]
        evidence_packet.extend({
            "id": unit.id,
            "authority": "issue_intent",
            "observation": unit.text,
            "issue_span": unit.issue_span,
        } for unit in issue_evidence.intent_units)
        evidence_packet.extend({
            "id": item.id,
            "authority": "repository_observation",
            "path": item.path,
            "locator": item.locator,
            "observation": item.observation[:1600],
        } for item in repository_evidence[:40])

        proposed = self.extractor(
            build_intent_contract_semantic_audit_prompt(
                issue,
                issue_evidence,
                clauses,
                evidence_packet,
            ),
            IntentContractSemanticAudit,
        )
        expected_ids = {item["id"] for item in clauses}
        valid_evidence_ids = {
            item["id"] for item in evidence_packet
        } | {
            evidence_id
            for item in clauses
            for evidence_id in item.get("source_evidence_ids", [])
        }
        audit_valid = proposed is not None
        if proposed is not None:
            decisions_by_id = {
                item.clause_id: item for item in proposed.decisions
            }
            audit_valid = (
                set(decisions_by_id) == expected_ids
                and all(
                    set(item.evidence_ids) <= valid_evidence_ids
                    for item in proposed.decisions
                )
                and all(
                    item.disposition != "binding" or item.evidence_ids
                    for item in proposed.decisions
                )
            )
        else:
            decisions_by_id = {}

        if not audit_valid:
            logger.warning(
                "Final Intent contract semantic audit was unavailable or "
                "incomplete; repository-derived clauses remain diagnostic."
            )
            decisions_by_id = {}

        def disposition(clause_id: str) -> str:
            decision = decisions_by_id.get(clause_id)
            return decision.disposition if decision is not None else "diagnostic"

        nonbinding_property_ids = {
            property_id
            for property_id, clause_id in property_clause_ids.items()
            if disposition(clause_id) != "binding"
        }
        contract.properties = [
            item
            for item in contract.properties
            if item.id not in nonbinding_property_ids
        ]
        retained_property_ids = {item.id for item in contract.properties}
        contract.validation_routes = [
            route
            for route in contract.validation_routes
            if route.property_id in retained_property_ids
        ]
        retained_route_ids = {route.id for route in contract.validation_routes}

        finalized_checks: list[FrozenAcceptanceCheck] = []
        for index, check in enumerate(contract.acceptance_checks):
            check_disposition = disposition(check_clause_ids[index])
            dependencies_binding = (
                set(check.covered_property_ids) <= retained_property_ids
                and set(check.covered_route_ids) <= retained_route_ids
            )
            if check_disposition == "reject":
                continue
            authority = (
                "normative"
                if check_disposition == "binding" and dependencies_binding
                else "illustrative"
            )
            finalized_checks.append(check.model_copy(update={
                "authority": authority,
                "kind": check.kind if authority == "normative" else "diagnostic",
            }))
        contract.acceptance_checks = finalized_checks

        binding_deltas = {
            value
            for value, clause_id in delta_clause_ids.items()
            if disposition(clause_id) == "binding"
        }
        diagnostic_deltas = {
            value
            for value, clause_id in delta_clause_ids.items()
            if disposition(clause_id) == "diagnostic"
        }
        contract.behavioral_delta = [
            value for value in contract.behavioral_delta
            if value in binding_deltas
        ]
        contract.local_behavior_witnesses = _unique([
            *contract.local_behavior_witnesses,
            *diagnostic_deltas,
        ], limit=10)
        contract.intent_ledger = [
            item.model_copy(update={
                "status": (
                    "local_witness" if item.witnessed_delta else "unresolved"
                ),
                "supported_delta": "",
                "limitations": _unique([
                    *item.limitations,
                    "The final semantic audit retained this test-derived "
                    "boundary as diagnostic rather than binding.",
                ], limit=5),
            })
            if item.supported_delta and item.supported_delta not in binding_deltas
            else item
            for item in contract.intent_ledger
        ]

        nonbinding_count = sum(
            disposition(clause_id) != "binding"
            for clause_id in expected_ids
        )
        if nonbinding_count:
            summary = (
                proposed.summary
                if proposed is not None and audit_valid
                else "Final semantic audit did not establish binding authority."
            )
            contract.validation_gaps = _unique([
                *contract.validation_gaps,
                f"Final contract audit kept {nonbinding_count} derived "
                f"clause(s) non-binding: {summary}",
            ], limit=4)
        return proposed if audit_valid else None

    def _collect_additional_intent_attempts(
        self,
        issue: str,
        issue_evidence: IssueEvidenceFrame,
        reproduction: dict,
        anchor_retrieval: TestAnchorRetrieval | None,
        anchor_selection: TestAnchorSelection | None,
        attempts: list[ContrastiveAttempt],
        report: InvestigationReport,
        repository_evidence: list[RepositoryEvidence],
        actions: list[dict],
        investigation: RepositoryInvestigation,
        report_history: list[FocusedInvestigationReport],
    ) -> tuple[
        list[ContrastiveAttempt],
        InvestigationReport,
        list[RepositoryEvidence],
        list[dict],
        RepositoryInvestigation,
        list[FocusedInvestigationReport],
    ]:
        investigate_next = getattr(
            self.investigator,
            "investigate_next_intent",
            None,
        )
        if not callable(investigate_next):
            return (
                attempts,
                report,
                repository_evidence,
                actions,
                investigation,
                report_history,
            )

        covered_intent_ids = {
            intent_id
            for attempt in attempts
            if attempt.decision.witness_valid
            for intent_id in attempt.decision.covered_issue_intent_ids
        }
        target_units = [
            unit for unit in issue_evidence.intent_units if unit.role == "target"
        ]
        scheduled_intent_ids = {
            intent_id
            for attempt in attempts
            if attempt.witness is not None
            for intent_id in attempt.witness.issue_intent_ids
        }
        for unit in target_units:
            if unit.id in covered_intent_ids:
                continue
            if len(scheduled_intent_ids) >= self.max_intent_witnesses:
                break
            remaining_steps = getattr(self.investigator, "remaining_steps", 0)
            if remaining_steps <= 1:
                break

            logger.info(
                "Starting independent contrastive witness for %s: "
                "remaining_steps=%s",
                unit.id,
                remaining_steps,
            )
            next_report, next_evidence, next_actions = investigate_next(
                issue,
                issue_evidence,
                unit,
            )
            next_proposal = next_report.contrastive_witness
            next_pair_stop, next_executions = self._contrastive_pair_state(
                next_report,
                next_actions,
                anchor_retrieval,
            )
            cited_ids = set(next_report.repository_evidence_ids)
            next_adjudication_evidence = [
                item for item in next_evidence if item.id in cited_ids
            ]
            next_decision = self.extractor(
                build_contrastive_intent_decision_prompt(
                    issue,
                    issue_evidence,
                    reproduction,
                    next_report.model_dump(),
                    [item.model_dump() for item in next_adjudication_evidence],
                    next_executions,
                    anchor_retrieval,
                    anchor_selection,
                    pair_status=(next_pair_stop or "validated"),
                ),
                ContrastiveIntentDecision,
            )
            if next_decision is None:
                logger.warning(
                    "Intent adjudication unavailable for independent witness %s",
                    unit.id,
                )
                break
            next_decision = self._enforce_pair_status(
                next_decision,
                next_pair_stop,
            )
            next_decision = self._normalize_intent_coverage(
                issue_evidence,
                next_proposal,
                next_decision,
            )
            attempt_number = len(attempts) + 1
            attempts.append(ContrastiveAttempt(
                attempt=attempt_number,
                pair_status=(next_pair_stop or "validated"),
                witness=next_proposal,
                executions=next_executions,
                decision=next_decision,
            ))
            scheduled_intent_ids.add(unit.id)
            if next_decision.witness_valid:
                covered_intent_ids.update(
                    next_decision.covered_issue_intent_ids
                )
            else:
                latest_valid_attempt = next(
                    (
                        attempt
                        for attempt in reversed(attempts)
                        if attempt.decision.witness_valid
                        and attempt.witness is not None
                    ),
                    None,
                )
                if latest_valid_attempt is not None:
                    next_report = next_report.model_copy(update={
                        "contrastive_witness": latest_valid_attempt.witness,
                        "repository_evidence_ids": list(dict.fromkeys(
                            report.repository_evidence_ids
                            + next_report.repository_evidence_ids
                        )),
                    })
            report = next_report
            report_history.append(FocusedInvestigationReport(
                focus_intent_ids=[unit.id],
                report=report,
            ))
            repository_evidence = next_evidence
            actions = next_actions
            investigation = RepositoryInvestigation(
                plan=investigation.plan,
                anchor_retrieval=anchor_retrieval,
                anchor_selection=anchor_selection,
                repository_evidence=repository_evidence,
                report=report,
                report_history=report_history,
                actions=actions,
            )
            logger.info(
                "Independent intent witness completed: intent=%s attempt=%d "
                "pair_status=%s witness_valid=%s covered=%s",
                unit.id,
                attempt_number,
                next_pair_stop or "validated",
                next_decision.witness_valid,
                ",".join(next_decision.covered_issue_intent_ids) or "none",
            )

        return (
            attempts,
            report,
            repository_evidence,
            actions,
            investigation,
            report_history,
        )

    def _run_contrastive(
        self,
        issue: str,
        issue_evidence: IssueEvidenceFrame,
        reproduction: dict,
        base_contract: RepairIntentContract,
    ) -> IntentDialogueResult:
        poc_mode = self.mode == "poc"
        plan = IntentExplorationPlan(mode="poc" if poc_mode else "contrastive")
        investigate = getattr(
            self.investigator,
            "investigate_contrastively",
            None,
        )
        if not callable(investigate):
            return IntentDialogueResult(
                issue_evidence=issue_evidence,
                intent_contract=base_contract,
                exploration_plan=plan,
                pair_status="investigator_unavailable",
                stop_reason="contrastive_investigator_unavailable",
            )

        anchor_retrieval = None
        anchor_selection = None
        retrieve_candidates = getattr(
            self.investigator,
            "retrieve_test_anchor_candidates",
            None,
        )
        if not poc_mode and self.enable_test_anchor_retrieval and callable(retrieve_candidates):
            try:
                anchor_retrieval = retrieve_candidates(issue, issue_evidence)
                logger.info(
                    "Test anchor retrieval completed: status=%s candidates=%d",
                    anchor_retrieval.status,
                    len(anchor_retrieval.candidates),
                )
            except IntentEmbeddingUnavailableError:
                raise
            except Exception:
                logger.exception(
                    "Test anchor retrieval failed; continuing with Investigator search."
                )
            if anchor_retrieval and anchor_retrieval.candidates:
                proposed_selection = self.extractor(
                    build_test_anchor_selection_prompt(
                        issue,
                        issue_evidence,
                        anchor_retrieval,
                    ),
                    TestAnchorSelection,
                )
                anchor_selection = self._normalize_test_anchor_selection(
                    proposed_selection,
                    anchor_retrieval,
                )
                logger.info(
                    "Test anchor reranking completed: status=%s primary=%s shortlist=%s",
                    (
                        anchor_selection.status
                        if anchor_selection is not None
                        else "unavailable"
                    ),
                    (
                        anchor_selection.primary_candidate_id
                        if anchor_selection is not None
                        else ""
                    ),
                    (
                        ",".join(anchor_selection.selected_candidate_ids)
                        if anchor_selection is not None
                        else ""
                    ),
                )

        set_anchor_context = getattr(
            self.investigator,
            "set_test_anchor_context",
            None,
        )
        if callable(set_anchor_context):
            set_anchor_context(anchor_retrieval, anchor_selection)

        report, repository_evidence, actions = investigate(
            issue,
            issue_evidence,
            reproduction,
        )
        if poc_mode:
            report = report.model_copy(update={
                "contrastive_witness": None,
                "acceptance_checks": [],
            })
        initial_focus_intent_ids = [
            unit.id
            for unit in issue_evidence.intent_units
            if unit.role == "target"
        ]
        report_history = [FocusedInvestigationReport(
            focus_intent_ids=initial_focus_intent_ids,
            report=report,
        )]
        candidate_boundaries = _merge_candidate_boundaries(
            issue_evidence,
            report.candidate_boundaries,
        )
        candidate_base_contract = base_contract.model_copy(
            deep=True,
            update={"candidate_boundaries": candidate_boundaries},
        )
        investigation = RepositoryInvestigation(
            plan=plan,
            anchor_retrieval=anchor_retrieval,
            anchor_selection=anchor_selection,
            repository_evidence=repository_evidence,
            report=report,
            report_history=report_history,
            actions=actions,
        )
        proposal = report.contrastive_witness
        pair_stop_reason, executions = (
            ("poc_only", [])
            if poc_mode
            else self._contrastive_pair_state(
                report, actions, anchor_retrieval,
            )
        )

        evidence_sufficiency_review = None
        evidence_sufficiency_followup_performed = False
        hypothesis_ready = (
            None if self.evidence_sufficiency_reviewer is None else False
        )
        if self.evidence_sufficiency_reviewer is not None:
            evidence_sufficiency_review = self._review_scout_evidence(
                issue,
                issue_evidence,
                report,
                repository_evidence,
                executions,
                phase="initial",
                report_history=report_history,
            )
            if evidence_sufficiency_review is not None:
                hypothesis_ready = (
                    evidence_sufficiency_review.decision == "ready"
                )
                actions = self._record_scout_evidence_review(
                    actions,
                    evidence_sufficiency_review,
                    phase="initial",
                    evidence_count=len(repository_evidence),
                )
                refine_sufficiency = getattr(
                    self.investigator,
                    "refine_evidence_sufficiency",
                    None,
                )
                if (
                    evidence_sufficiency_review.decision == "continue"
                    and callable(refine_sufficiency)
                ):
                    prior_report = report
                    (
                        refined_report,
                        refined_evidence,
                        refined_actions,
                    ) = refine_sufficiency(
                        issue,
                        issue_evidence,
                        report,
                        evidence_sufficiency_review,
                    )
                    if poc_mode:
                        refined_report = refined_report.model_copy(update={
                            "contrastive_witness": None,
                            "acceptance_checks": [],
                        })
                    report_updates: dict[str, Any] = {
                        "repository_evidence_ids": list(dict.fromkeys(
                            prior_report.repository_evidence_ids
                            + refined_report.repository_evidence_ids
                        )),
                        "investigation_paths": _merge_investigation_paths(
                            prior_report.investigation_paths,
                            refined_report.investigation_paths,
                        ),
                    }
                    if (
                        refined_report.contrastive_witness is None
                        and prior_report.contrastive_witness is not None
                    ):
                        report_updates["contrastive_witness"] = (
                            prior_report.contrastive_witness
                        )
                    report = refined_report.model_copy(update=report_updates)
                    report_history[-1] = report_history[-1].model_copy(
                        update={"report": report}
                    )
                    evidence_by_id = {
                        item.id: item for item in repository_evidence
                    }
                    evidence_by_id.update({
                        item.id: item for item in refined_evidence
                    })
                    repository_evidence = list(evidence_by_id.values())
                    merged_actions = list(actions)
                    merged_actions.extend(
                        action
                        for action in refined_actions
                        if action not in merged_actions
                    )
                    actions = merged_actions
                    evidence_sufficiency_followup_performed = True
                    candidate_boundaries = _merge_candidate_boundaries(
                        issue_evidence,
                        candidate_boundaries,
                        report.candidate_boundaries,
                    )
                    candidate_base_contract = base_contract.model_copy(
                        deep=True,
                        update={"candidate_boundaries": candidate_boundaries},
                    )
                    proposal = report.contrastive_witness
                    pair_stop_reason, executions = (
                        ("poc_only", [])
                        if poc_mode
                        else self._contrastive_pair_state(
                            report, actions, anchor_retrieval,
                        )
                    )

        investigation = RepositoryInvestigation(
            plan=plan,
            anchor_retrieval=anchor_retrieval,
            anchor_selection=anchor_selection,
            repository_evidence=repository_evidence,
            report=report,
            report_history=report_history,
            actions=actions,
        )

        cited_evidence_ids = set(report.repository_evidence_ids)
        adjudication_evidence = [
            item
            for item in repository_evidence
            if item.id in cited_evidence_ids
        ]
        if pair_stop_reason and not adjudication_evidence:
            return IntentDialogueResult(
                issue_evidence=issue_evidence,
                intent_contract=candidate_base_contract,
                exploration_plan=plan,
                investigation=investigation,
                evidence_sufficiency_review=evidence_sufficiency_review,
                evidence_sufficiency_followup_performed=(
                    evidence_sufficiency_followup_performed
                ),
                hypothesis_ready=hypothesis_ready,
                pair_status=pair_stop_reason,
                stop_reason=pair_stop_reason,
            )
        decision = self.extractor(
            (
                build_poc_intent_decision_prompt(
                    issue,
                    issue_evidence,
                    report.model_dump(),
                    [item.model_dump() for item in adjudication_evidence],
                )
                if poc_mode
                else build_contrastive_intent_decision_prompt(
                    issue,
                    issue_evidence,
                    reproduction,
                    report.model_dump(),
                    [item.model_dump() for item in adjudication_evidence],
                    executions,
                    anchor_retrieval,
                    anchor_selection,
                    pair_status=(pair_stop_reason or "validated"),
                )
            ),
            ContrastiveIntentDecision,
        )
        if decision is None:
            return IntentDialogueResult(
                issue_evidence=issue_evidence,
                intent_contract=candidate_base_contract,
                exploration_plan=plan,
                investigation=investigation,
                evidence_sufficiency_review=evidence_sufficiency_review,
                evidence_sufficiency_followup_performed=(
                    evidence_sufficiency_followup_performed
                ),
                hypothesis_ready=hypothesis_ready,
                pair_status=(pair_stop_reason or "validated"),
                stop_reason="contrastive_adjudicator_unavailable",
            )

        # Mechanical validation controls only witness-derived fields. The LLM
        # still owns every semantic evidence and refinement decision.
        decision = self._enforce_pair_status(decision, pair_stop_reason)
        decision = self._normalize_intent_coverage(
            issue_evidence,
            proposal,
            decision,
        )
        contrastive_attempts = [
            ContrastiveAttempt(
                attempt=1,
                pair_status=(pair_stop_reason or "validated"),
                witness=proposal,
                executions=executions,
                decision=decision,
            )
        ]

        refine = getattr(self.investigator, "refine_contrastively", None)
        remaining_steps = getattr(self.investigator, "remaining_steps", 1)
        should_follow_up = (
            not poc_mode
            and decision.next_action in {
                "refine_test_pair",
                "seek_repository_contract",
            }
            and decision.followup_request is not None
            and callable(refine)
            and remaining_steps > 0
            and self.max_contrast_refinements > 0
        )
        if should_follow_up:
            followup_action = decision.next_action
            prior_report = report
            prior_decision = decision
            logger.info(
                "Contrastive evidence follow-up requested: action=%s "
                "remaining_steps=%s",
                followup_action,
                remaining_steps,
            )
            refined_report, refined_evidence, refined_actions = refine(
                issue,
                issue_evidence,
                report,
                decision,
                pair_stop_reason or "validated",
            )
            merged_evidence_ids = list(dict.fromkeys(
                prior_report.repository_evidence_ids
                + refined_report.repository_evidence_ids
            ))
            report_updates: dict[str, Any] = {
                "repository_evidence_ids": merged_evidence_ids,
                "investigation_paths": _merge_investigation_paths(
                    prior_report.investigation_paths,
                    refined_report.investigation_paths,
                ),
            }
            if (
                followup_action == "seek_repository_contract"
                and refined_report.contrastive_witness is None
                and prior_report.contrastive_witness is not None
            ):
                report_updates["contrastive_witness"] = (
                    prior_report.contrastive_witness
                )
            refined_report = refined_report.model_copy(update=report_updates)
            candidate_boundaries = _merge_candidate_boundaries(
                issue_evidence,
                candidate_boundaries,
                refined_report.candidate_boundaries,
            )
            refined_pair_stop, refined_executions = (
                self._contrastive_pair_state(
                    refined_report,
                    refined_actions,
                    anchor_retrieval,
                )
            )
            refined_cited_ids = set(refined_report.repository_evidence_ids)
            refined_adjudication_evidence = [
                item
                for item in refined_evidence
                if item.id in refined_cited_ids
            ]
            refined_decision = self.extractor(
                build_contrastive_intent_decision_prompt(
                    issue,
                    issue_evidence,
                    reproduction,
                    refined_report.model_dump(),
                    [
                        item.model_dump()
                        for item in refined_adjudication_evidence
                    ],
                    refined_executions,
                    anchor_retrieval,
                    anchor_selection,
                    pair_status=(refined_pair_stop or "validated"),
                    prior_decision=prior_decision,
                ),
                ContrastiveIntentDecision,
            )
            if refined_decision is not None:
                refined_decision = self._enforce_pair_status(
                    refined_decision,
                    refined_pair_stop,
                )
                refined_decision = self._normalize_intent_coverage(
                    issue_evidence,
                    refined_report.contrastive_witness,
                    refined_decision,
                )
                logger.info(
                    "Contrastive evidence follow-up completed: pair_status=%s "
                    "witness_valid=%s boundary_resolved=%s next_action=%s",
                    refined_pair_stop or "validated",
                    refined_decision.witness_valid,
                    refined_decision.boundary_resolved,
                    refined_decision.next_action,
                )
                contrastive_attempts.append(
                    ContrastiveAttempt(
                        attempt=2,
                        pair_status=(refined_pair_stop or "validated"),
                        witness=refined_report.contrastive_witness,
                        executions=refined_executions,
                        decision=refined_decision,
                    )
                )
                # A failed replacement target cannot erase an already valid
                # local witness. A valid second witness, resolved or not,
                # becomes the final evidence considered by Repair.
                adopt_refinement = (
                    refined_decision.witness_valid
                    or (
                        followup_action == "seek_repository_contract"
                        and bool(refined_decision.repository_effects)
                    )
                )
                if adopt_refinement:
                    report = refined_report
                    report_history[-1] = report_history[-1].model_copy(
                        update={"report": report}
                    )
                    repository_evidence = refined_evidence
                    actions = refined_actions
                    investigation = RepositoryInvestigation(
                        plan=plan,
                        anchor_retrieval=anchor_retrieval,
                        anchor_selection=anchor_selection,
                        repository_evidence=repository_evidence,
                        report=report,
                        report_history=report_history,
                        actions=actions,
                    )
                    proposal = report.contrastive_witness
                    pair_stop_reason = refined_pair_stop
                    executions = refined_executions
                    adjudication_evidence = refined_adjudication_evidence
                    decision = refined_decision

        if not poc_mode:
            (
                contrastive_attempts,
                report,
                repository_evidence,
                actions,
                investigation,
                report_history,
            ) = self._collect_additional_intent_attempts(
                issue,
                issue_evidence,
                reproduction,
                anchor_retrieval,
                anchor_selection,
                contrastive_attempts,
                report,
                repository_evidence,
                actions,
                investigation,
                report_history,
            )
        candidate_boundaries = _merge_candidate_boundaries(
            issue_evidence,
            candidate_boundaries,
            *[
                item.report.candidate_boundaries
                for item in report_history
            ],
        )
        selected_attempt = next(
            (
                attempt
                for attempt in reversed(contrastive_attempts)
                if attempt.decision.witness_valid and attempt.witness is not None
            ),
            contrastive_attempts[-1],
        )
        proposal = selected_attempt.witness
        decision = selected_attempt.decision
        executions = selected_attempt.executions
        pair_stop_reason = (
            ""
            if selected_attempt.pair_status == "validated"
            else selected_attempt.pair_status
        )

        if self.evidence_sufficiency_reviewer is not None:
            final_review = self._review_scout_evidence(
                issue,
                issue_evidence,
                report,
                repository_evidence,
                [
                    execution
                    for attempt in contrastive_attempts
                    for execution in attempt.executions
                ],
                phase="final_handoff",
                report_history=report_history,
                prior_attempts=contrastive_attempts,
                prior_reviews=(
                    [evidence_sufficiency_review]
                    if evidence_sufficiency_review is not None
                    else []
                ),
            )
            hypothesis_ready = False
            if final_review is not None:
                if evidence_sufficiency_review is not None:
                    final_review = final_review.model_copy(update={
                        "preserved_alternatives": _unique(
                            evidence_sufficiency_review.preserved_alternatives
                            + final_review.preserved_alternatives,
                            limit=8,
                        )
                    })
                evidence_sufficiency_review = final_review
                hypothesis_ready = final_review.decision == "ready"
                actions = self._record_scout_evidence_review(
                    actions,
                    final_review,
                    phase="final_handoff",
                    evidence_count=len(repository_evidence),
                )
            if not hypothesis_ready:
                report = self._mark_report_not_hypothesis_ready(
                    report,
                    evidence_sufficiency_review,
                )
                report_history[-1] = report_history[-1].model_copy(
                    update={"report": report}
                )
            investigation = RepositoryInvestigation(
                plan=plan,
                anchor_retrieval=anchor_retrieval,
                anchor_selection=anchor_selection,
                repository_evidence=repository_evidence,
                report=report,
                report_history=report_history,
                actions=actions,
            )

        cited_attempt_evidence_ids = {
            citation.evidence_id
            for attempt in contrastive_attempts
            for effect in attempt.decision.repository_effects
            for citation in effect.evidence
        }
        adjudication_evidence_by_id = {
            item.id: item
            for item in repository_evidence
            if item.id in cited_attempt_evidence_ids
            or item.id in set(report.repository_evidence_ids)
        }
        known_intent_ids = {unit.id for unit in issue_evidence.intent_units}
        valid_repository_effects = []
        proposed_repository_effects = [
            effect
            for attempt in contrastive_attempts
            for effect in attempt.decision.repository_effects
        ]
        for effect in proposed_repository_effects:
            valid = True
            for citation in effect.evidence:
                evidence = adjudication_evidence_by_id.get(
                    citation.evidence_id
                )
                if (
                    evidence is None
                    or citation.quote not in evidence.observation
                ):
                    valid = False
                    break
            if valid:
                normalized_intent_ids = [
                    intent_id
                    for intent_id in effect.issue_intent_ids
                    if intent_id in known_intent_ids
                ]
                if known_intent_ids and not normalized_intent_ids:
                    valid = False
                else:
                    valid_repository_effects.append(effect.model_copy(update={
                        "issue_intent_ids": normalized_intent_ids,
                    }))
            if not valid:
                logger.warning(
                    "Discarding repository intent effect with invalid "
                    "provenance or missing atomic-intent binding: "
                    "kind=%s statement=%r",
                    effect.kind,
                    effect.statement,
                )
        decision = decision.model_copy(update={
            "repository_effects": valid_repository_effects,
            "uncertainty": _unique([
                item
                for attempt in contrastive_attempts
                for item in attempt.decision.uncertainty
            ], limit=10),
            "scope_evidence": _unique([
                item
                for attempt in contrastive_attempts
                for item in attempt.decision.scope_evidence
            ], limit=3),
        })

        repository_targets = [
            effect.statement
            for effect in decision.repository_effects
            if effect.kind == "target_qualifier"
        ]
        repository_preserves = [
            effect.statement
            for effect in decision.repository_effects
            if effect.kind == "preserve"
        ]
        repository_warnings = [
            effect.statement
            for effect in decision.repository_effects
            if effect.kind == "compatibility_risk"
        ]
        repository_open_decisions = [
            effect.statement
            for effect in decision.repository_effects
            if effect.kind == "open_decision"
        ]
        intent_contract = base_contract.model_copy(deep=True)
        intent_contract.candidate_boundaries = candidate_boundaries
        intent_contract.target_behavior = _unique(
            intent_contract.target_behavior + repository_targets,
            limit=5,
        )
        intent_contract.preserve_behavior = _unique(
            intent_contract.preserve_behavior + repository_preserves,
            limit=4,
        )
        intent_contract.compatibility_warnings = _unique(
            intent_contract.compatibility_warnings
            + repository_warnings,
            limit=3,
        )
        intent_contract.open_decisions = _unique(
            intent_contract.open_decisions
            + repository_open_decisions
            + decision.uncertainty,
            limit=10,
        )
        repository_only_decision = decision.model_copy(update={
            "witness_valid": False,
            "boundary_resolved": False,
            "next_action": "leave_open",
            "followup_request": None,
            "selected_anchor_id": "",
            "selected_target_id": "",
            "witnessed_delta": "",
            "supported_delta": "",
            "covered_issue_claim_ids": [],
            "covered_issue_intent_ids": [],
            "unresolved_issue_claim_ids": [],
            "unresolved_issue_intent_ids": [],
            "scope_evidence": [],
            "uncertainty": [],
        })
        self._apply_intent_ledger_evidence(
            intent_contract,
            None,
            repository_only_decision,
            attempt=0,
        )
        for attempt in contrastive_attempts:
            witness_only_decision = attempt.decision.model_copy(
                update={"repository_effects": []}
            )
            self._apply_intent_ledger_evidence(
                intent_contract,
                attempt.witness,
                witness_only_decision,
                attempt=attempt.attempt,
            )
        known_issue_claim_ids = {claim.id for claim in issue_evidence.claims}
        valid_witness_attempts = []
        for attempt in contrastive_attempts:
            attempt_proposal = attempt.witness
            attempt_decision = attempt.decision
            if attempt_proposal is None or not attempt_decision.witness_valid:
                continue
            proposed_claim_ids = set(attempt_proposal.issue_claim_ids)
            covered_claim_ids = [
                claim_id
                for claim_id in attempt_decision.covered_issue_claim_ids
                if claim_id in known_issue_claim_ids
                and claim_id in proposed_claim_ids
            ]
            if (
                not covered_claim_ids
                or attempt_decision.selected_anchor_id
                != attempt_proposal.anchor.id
                or attempt_decision.selected_target_id
                != attempt_proposal.target.id
            ):
                continue
            valid_witness_attempts.append(attempt)

        # Same-intent refinement attempts replace the earlier local witness;
        # independently bound intents remain separate evidence tracks.
        latest_attempt_by_binding: dict[tuple[str, ...], ContrastiveAttempt] = {}
        for attempt in valid_witness_attempts:
            binding = tuple(sorted(
                attempt.decision.covered_issue_intent_ids
                or attempt.decision.covered_issue_claim_ids
            ))
            latest_attempt_by_binding[binding] = attempt
        valid_witness_attempts = list(latest_attempt_by_binding.values())

        intent_contract.local_behavior_witnesses = _unique([
            *intent_contract.local_behavior_witnesses,
            *(
                (
                    f"[{','.join(attempt.decision.covered_issue_intent_ids)}] "
                    if attempt.decision.covered_issue_intent_ids
                    else ""
                )
                + attempt.decision.witnessed_delta
                for attempt in valid_witness_attempts
                if attempt.decision.witnessed_delta
            ),
        ], limit=10)
        intent_contract.behavioral_delta = _unique([
            *intent_contract.behavioral_delta,
            *(
                (
                    f"[{','.join(attempt.decision.covered_issue_intent_ids)}] "
                    if attempt.decision.covered_issue_intent_ids
                    else ""
                )
                + attempt.decision.supported_delta
                for attempt in valid_witness_attempts
                if attempt.decision.boundary_resolved
                and attempt.decision.supported_delta
            ),
        ], limit=10)

        property_executions = [
            execution
            for attempt in valid_witness_attempts
            for execution in attempt.executions
        ]
        property_evidence_ids = set(cited_attempt_evidence_ids)
        for attempt in valid_witness_attempts:
            attempt_proposal = attempt.witness
            property_evidence_ids.update({
                attempt_proposal.anchor_source_evidence_id,
                *attempt_proposal.considered_anchor_evidence_ids,
                attempt_proposal.anchor.run_evidence_id,
                attempt_proposal.target.run_evidence_id,
            })
        property_repository_evidence = [
            item
            for item in repository_evidence
            if item.id in property_evidence_ids
        ]
        property_extraction = self._extract_intent_properties(
            issue,
            issue_evidence,
            intent_contract,
            property_repository_evidence,
            property_executions,
        )
        if property_extraction is not None:
            intent_contract.properties = property_extraction.properties

        validation_routing = self._route_intent_properties(
            issue,
            issue_evidence,
            intent_contract,
            property_repository_evidence,
            property_executions,
        )
        (
            boundary_test_report,
            boundary_test_assessment,
            boundary_checks,
            boundary_evidence,
            boundary_actions,
        ) = self._materialize_boundary_tests(
            issue,
            issue_evidence,
            intent_contract,
            decision,
            validation_routing,
            property_repository_evidence,
        )
        intent_contract.validation_routes = (
            validation_routing.routes if validation_routing is not None else []
        )
        if boundary_checks:
            intent_contract.acceptance_checks = _merge_acceptance_checks(
                intent_contract.acceptance_checks,
                boundary_checks,
            )
        diagnostic_property_ids: set[str] = set()
        semantic_candidate_rejected = False
        if boundary_test_assessment is not None:
            semantic_candidate_rejected = (
                not boundary_test_assessment.contract_consistent
            )
            diagnostic_property_ids = set(
                boundary_test_assessment.diagnostic_property_ids
            )
            self._reconcile_semantic_admission(
                intent_contract,
                boundary_test_assessment,
            )
            validation_gaps = list(
                boundary_test_assessment.missing_discriminators
            )
            validation_gaps.extend(
                f"[{item.id}] remains diagnostic rather than a Repair "
                f"requirement: {item.predicate}"
                for item in intent_contract.properties
                if item.id in diagnostic_property_ids
            )
            self._prune_diagnostic_contract(
                intent_contract,
                diagnostic_property_ids,
            )
            validation_gaps.extend(
                item.missing_discriminator
                for item in boundary_test_assessment.checks
                if item.missing_discriminator
            )
            if not boundary_test_assessment.contract_consistent:
                validation_gaps.append(
                    "Candidate acceptance requirements are not mutually "
                    "consistent with the retained Intent alternatives: "
                    + boundary_test_assessment.reason
                )
            elif (
                not boundary_test_assessment.boundary_complete
                and not validation_gaps
            ):
                validation_gaps.append(boundary_test_assessment.reason)
            intent_contract.validation_gaps = _unique(
                validation_gaps,
                limit=4,
            )

        attempt_checks: list[FrozenAcceptanceCheck] = []
        for attempt in valid_witness_attempts:
            attempt_proposal = attempt.witness
            attempt_decision = attempt.decision
            target_execution = next(
                (
                    item
                    for item in attempt.executions
                    if item.get("candidate_id") == "TARGET-1"
                ),
                None,
            )
            if attempt_proposal is None or target_execution is None:
                continue
            command = target_execution.get("command", "")
            files = target_execution.get("files") or {}
            exit_code = target_execution.get("exit_code")
            if not command or not files or not isinstance(exit_code, int):
                continue
            covered_intent_ids = set(
                attempt_decision.covered_issue_intent_ids
            )
            covered_property_ids = [
                item.id
                for item in (
                    property_extraction.properties
                    if property_extraction is not None
                    else intent_contract.properties
                )
                if item.role == "target"
                and (
                    bool(covered_intent_ids & set(item.issue_intent_ids))
                    or (
                        not covered_intent_ids
                        and bool(
                            set(attempt_decision.covered_issue_claim_ids)
                            & set(item.evidence_ids)
                        )
                    )
                )
            ]
            authority = (
                attempt_decision.acceptance_authority
                if attempt_decision.boundary_resolved
                and not diagnostic_property_ids.intersection(
                    covered_property_ids
                )
                else "illustrative"
            )
            if authority == "illustrative" and semantic_candidate_rejected:
                # The witness remains local evidence, but its rejected exact
                # oracle must not re-enter the Repair contract as a diagnostic.
                continue
            has_materialized_target_check = any(
                check.kind == "f2p"
                and check.authority == "normative"
                and (
                    bool(
                        set(check.covered_issue_intent_ids)
                        & covered_intent_ids
                    )
                    or (
                        not covered_intent_ids
                        and bool(
                            set(check.covered_issue_claim_ids)
                            & set(attempt_decision.covered_issue_claim_ids)
                        )
                    )
                )
                for check in intent_contract.acceptance_checks
            )
            if authority == "normative" and has_materialized_target_check:
                continue
            intent_label = (
                ",".join(attempt_decision.covered_issue_intent_ids)
                or "Issue intent"
            )
            check_role = (
                "Issue-required target"
                if authority == "normative"
                else (
                    "Local Issue-related witness; not a complete repair boundary"
                )
            )
            attempt_checks.append(FrozenAcceptanceCheck(
                description=(
                    f"{check_role} [{intent_label}]: "
                    f"{attempt_proposal.target.description}"
                )[:800],
                command=command,
                files=files,
                source_evidence_id=attempt_proposal.target.run_evidence_id,
                baseline_exit_code=exit_code,
                kind="f2p" if authority == "normative" else "diagnostic",
                authority=authority,
                covered_issue_claim_ids=(
                    attempt_decision.covered_issue_claim_ids
                ),
                covered_issue_intent_ids=(
                    attempt_decision.covered_issue_intent_ids
                ),
                covered_property_ids=covered_property_ids,
            ))
        if attempt_checks:
            intent_contract.acceptance_checks = _merge_acceptance_checks(
                attempt_checks,
                intent_contract.acceptance_checks,
            )
        if poc_mode:
            intent_contract.acceptance_checks = []
        if boundary_evidence is not None and boundary_actions is not None:
            investigation = investigation.model_copy(update={
                "repository_evidence": boundary_evidence,
                "actions": boundary_actions,
            })
        handoff_evidence_by_id = {
            item.id: item
            for item in [
                *repository_evidence,
                *(boundary_evidence or []),
            ]
        }
        handoff_evidence = list(handoff_evidence_by_id.values())
        intent_contract.scope_evidence = _unique(
            decision.scope_evidence,
            limit=3,
        )
        issue_claims = {claim.id: claim for claim in issue_evidence.claims}
        aggregate_covered_claim_ids = {
            claim_id
            for attempt in valid_witness_attempts
            for claim_id in attempt.decision.covered_issue_claim_ids
            if claim_id in issue_claims
        }
        intent_contract.witness_limitations = (
            [] if poc_mode else [
                f"[{claim_id}; {issue_claims[claim_id].kind}] "
                f"{issue_claims[claim_id].text}"
                for claim_id in issue_claims
                if claim_id not in aggregate_covered_claim_ids
            ]
        )
        self._attach_intent_evidence_notes(
            intent_contract,
            handoff_evidence,
        )
        contract_semantic_audit = self._finalize_semantic_contract(
            issue,
            issue_evidence,
            intent_contract,
            handoff_evidence,
        )
        repository_effect = bool(decision.repository_effects)
        if pair_stop_reason or not decision.witness_valid or proposal is None:
            return IntentDialogueResult(
                issue_evidence=issue_evidence,
                intent_contract=intent_contract,
                exploration_plan=plan,
                investigation=investigation,
                contrastive_decision=decision,
                contrastive_attempts=contrastive_attempts,
                evidence_sufficiency_review=evidence_sufficiency_review,
                evidence_sufficiency_followup_performed=(
                    evidence_sufficiency_followup_performed
                ),
                hypothesis_ready=hypothesis_ready,
                property_extraction=property_extraction,
                validation_routing=validation_routing,
                boundary_test_report=boundary_test_report,
                boundary_test_assessment=boundary_test_assessment,
                contract_semantic_audit=contract_semantic_audit,
                pair_status=(pair_stop_reason or "rejected_by_adjudicator"),
                stop_reason=(
                    "poc_repository_evidence_only"
                    if poc_mode and repository_effect
                    else "poc_evidence_only"
                    if poc_mode
                    else
                    "contrastive_repository_evidence_only"
                    if repository_effect
                    else (pair_stop_reason or "contrastive_no_effect")
                ),
            )

        proposal_claim_ids = _unique([
            claim_id
            for claim_id in proposal.issue_claim_ids
            if claim_id in issue_claims
        ])
        covered_claim_ids = _unique([
            claim_id
            for claim_id in decision.covered_issue_claim_ids
            if claim_id in proposal_claim_ids
        ])
        if not covered_claim_ids:
            return IntentDialogueResult(
                issue_evidence=issue_evidence,
                intent_contract=intent_contract,
                exploration_plan=plan,
                investigation=investigation,
                contrastive_decision=decision,
                contrastive_attempts=contrastive_attempts,
                evidence_sufficiency_review=evidence_sufficiency_review,
                evidence_sufficiency_followup_performed=(
                    evidence_sufficiency_followup_performed
                ),
                hypothesis_ready=hypothesis_ready,
                property_extraction=property_extraction,
                validation_routing=validation_routing,
                boundary_test_report=boundary_test_report,
                boundary_test_assessment=boundary_test_assessment,
                contract_semantic_audit=contract_semantic_audit,
                pair_status="validated",
                stop_reason="contrastive_invalid_claim_coverage",
            )
        unresolved_claim_ids = _unique([
            claim_id
            for claim_id in decision.unresolved_issue_claim_ids
            if claim_id in issue_claims
            and claim_id not in covered_claim_ids
        ])
        # The adjudicator decides semantic coverage. The deterministic step is
        # only a completeness inventory: retain every Issue claim it omitted
        # instead of treating one valid local witness as a complete specification.
        unresolved_claim_ids = _unique(
            unresolved_claim_ids
            + [
                claim_id
                for claim_id in issue_claims
                if claim_id not in covered_claim_ids
                and claim_id not in unresolved_claim_ids
            ],
        )
        decision = decision.model_copy(
            update={
                "covered_issue_claim_ids": covered_claim_ids,
                "unresolved_issue_claim_ids": unresolved_claim_ids,
            }
        )
        anchor = proposal.anchor
        target = proposal.target
        if (
            decision.selected_anchor_id != anchor.id
            or decision.selected_target_id != target.id
        ):
            return IntentDialogueResult(
                issue_evidence=issue_evidence,
                intent_contract=intent_contract,
                exploration_plan=plan,
                investigation=investigation,
                contrastive_decision=decision,
                contrastive_attempts=contrastive_attempts,
                evidence_sufficiency_review=evidence_sufficiency_review,
                evidence_sufficiency_followup_performed=(
                    evidence_sufficiency_followup_performed
                ),
                hypothesis_ready=hypothesis_ready,
                property_extraction=property_extraction,
                validation_routing=validation_routing,
                boundary_test_report=boundary_test_report,
                boundary_test_assessment=boundary_test_assessment,
                contract_semantic_audit=contract_semantic_audit,
                pair_status="validated",
                stop_reason="contrastive_invalid_selection",
            )
        target_ledger = [
            item for item in intent_contract.intent_ledger
            if item.role == "target"
        ]
        all_target_boundaries_resolved = bool(target_ledger) and all(
            item.status == "resolved" for item in target_ledger
        )
        if not target_ledger:
            all_target_boundaries_resolved = decision.boundary_resolved
        return IntentDialogueResult(
            issue_evidence=issue_evidence,
            intent_contract=intent_contract,
            exploration_plan=plan,
            investigation=investigation,
            contrastive_decision=decision,
            contrastive_attempts=contrastive_attempts,
            evidence_sufficiency_review=evidence_sufficiency_review,
            evidence_sufficiency_followup_performed=(
                evidence_sufficiency_followup_performed
            ),
            hypothesis_ready=hypothesis_ready,
            property_extraction=property_extraction,
            validation_routing=validation_routing,
            boundary_test_report=boundary_test_report,
            boundary_test_assessment=boundary_test_assessment,
            contract_semantic_audit=contract_semantic_audit,
            pair_status="validated",
            stop_reason=(
                "contrastive_boundary_resolved"
                if all_target_boundaries_resolved
                else "contrastive_witness_only"
            ),
        )

    def run(
        self,
        issue: str,
        issue_evidence: IssueEvidenceFrame,
        reproduction: dict,
    ) -> IntentDialogueResult:
        exchanges: list[DialogueExchange] = []
        base_contract = build_repair_intent_contract(issue_evidence)
        if self.mode in {"contrastive", "poc"}:
            return self._run_contrastive(
                issue,
                issue_evidence,
                reproduction,
                base_contract,
            )
        if not base_contract.target_behavior:
            return IntentDialogueResult(
                issue_evidence=issue_evidence,
                intent_contract=base_contract,
                exploration_plan=IntentExplorationPlan(
                    mode="contrastive" if self.mode == "contrastive" else "legacy"
                ),
                stop_reason="missing_issue_target",
            )

        proposed_plan = self.extractor(
            build_intent_task_plan_prompt(
                issue,
                issue_evidence,
                reproduction,
            ),
            IntentExplorationPlan,
        )
        if proposed_plan is None:
            return IntentDialogueResult(
                issue_evidence=issue_evidence,
                intent_contract=base_contract,
                stop_reason="planner_unavailable",
            )

        plan = _normalize_plan(
            proposed_plan,
            issue_evidence,
            self.max_objectives,
        )
        if not plan.objectives:
            return IntentDialogueResult(
                issue_evidence=issue_evidence,
                intent_contract=base_contract,
                exploration_plan=plan,
                stop_reason="no_objectives",
            )

        investigate = getattr(self.investigator, "investigate", None)
        if callable(investigate):
            report, repository_evidence, investigation_actions = investigate(
                issue,
                issue_evidence,
                reproduction,
                plan,
            )
            valid_evidence_ids = {item.id for item in repository_evidence}
            report.repository_evidence_ids = [
                evidence_id
                for evidence_id in report.repository_evidence_ids
                if evidence_id in valid_evidence_ids
            ]
            report.preserve_evidence_ids = [
                evidence_id
                for evidence_id in report.preserve_evidence_ids
                if evidence_id in valid_evidence_ids
                and not evidence_id.startswith("GRAPH-")
            ]
            valid_task_ids = {task.id for task in plan.tasks}
            seen_task_ids: set[str] = set()
            normalized_findings = []
            for finding in report.task_findings:
                if (
                    finding.task_id not in valid_task_ids
                    or finding.task_id in seen_task_ids
                ):
                    continue
                finding.evidence_ids = [
                    evidence_id
                    for evidence_id in finding.evidence_ids
                    if evidence_id in valid_evidence_ids
                ]
                if finding.status != "unresolved" and not finding.evidence_ids:
                    finding.status = "unresolved"
                    finding.finding = (
                        "The submitted finding did not cite valid repository evidence."
                    )
                normalized_findings.append(finding)
                seen_task_ids.add(finding.task_id)
            for task in plan.tasks:
                if task.id in seen_task_ids:
                    continue
                normalized_findings.append(
                    InvestigationTaskFinding(
                        task_id=task.id,
                        status="unresolved",
                        finding="The final report omitted this investigation TODO.",
                    )
                )
            report.task_findings = normalized_findings
            if report.status != "unresolved" and not report.repository_evidence_ids:
                report.status = "unresolved"
                report.open_decisions = _unique(
                    report.open_decisions
                    + ["The report did not cite valid repository evidence."],
                    limit=2,
                )

            investigation = RepositoryInvestigation(
                plan=plan,
                repository_evidence=repository_evidence,
                report=report,
                actions=investigation_actions,
            )
            candidate_exchange = DialogueExchange(
                objective=plan.tasks[0],
                repository_evidence=repository_evidence,
                answer=report,
                developer_actions=investigation_actions,
            )
            if plan.mode == "evidence_todos":
                candidates, prompt_candidates = _candidate_registry(
                    [candidate_exchange]
                )
                for candidate_group in prompt_candidates.values():
                    for candidate in candidate_group:
                        candidate.pop("objective_id", None)
                proposed_resolution = self.extractor(
                    build_intent_resolution_prompt(
                        issue,
                        issue_evidence,
                        plan,
                        report.model_dump(),
                        [item.model_dump() for item in repository_evidence],
                        prompt_candidates,
                    ),
                    IntentEvidenceResolution,
                )
                if proposed_resolution is None:
                    return IntentDialogueResult(
                        issue_evidence=issue_evidence,
                        intent_contract=base_contract,
                        exploration_plan=plan,
                        investigation=investigation,
                        stop_reason="evidence_resolver_unavailable",
                    )

                resolution = _normalize_evidence_resolution(
                    plan,
                    proposed_resolution,
                    repository_evidence,
                    candidates,
                )
                resolved_qualifiers, synthesis = _resolution_contract_inputs(
                    resolution
                )
                resolved_count = sum(
                    item.resolved for item in resolution.resolutions
                )
                intent_contract = (
                    build_repair_intent_contract(
                        issue_evidence,
                        candidates,
                        synthesis,
                        resolved_qualifiers=resolved_qualifiers,
                    )
                    if resolved_count
                    else base_contract
                )
                if resolved_count == len(plan.tasks):
                    stop_reason = "evidence_resolution_resolved"
                elif resolved_count:
                    stop_reason = "evidence_resolution_partial"
                else:
                    stop_reason = "evidence_resolution_no_effect"
                return IntentDialogueResult(
                    issue_evidence=issue_evidence,
                    intent_contract=intent_contract,
                    exploration_plan=plan,
                    investigation=investigation,
                    synthesis=synthesis,
                    resolution=resolution,
                    stop_reason=stop_reason,
                )

            if _is_version_space_plan(plan):
                adjudication = self.extractor(
                    build_intent_evidence_adjudication_prompt(
                        issue,
                        issue_evidence,
                        plan,
                        report.model_dump(),
                        [item.model_dump() for item in repository_evidence],
                    ),
                    IntentEvidenceAdjudication,
                )
                adjudicator_available = adjudication is not None
                if adjudication is None:
                    adjudication = IntentEvidenceAdjudication()
                version_space = _reduce_version_space(
                    plan,
                    adjudication,
                    repository_evidence,
                )
                (
                    resolved_qualifiers,
                    open_decisions,
                    version_warnings,
                ) = _version_space_contract_inputs(plan, version_space)
                candidates, _ = _candidate_registry([candidate_exchange])
                if not report.preserve_evidence_ids:
                    candidates["preserve"] = {}
                if open_decisions:
                    candidates["acceptance"] = {}
                deterministic_selection = IntentSynthesisSelection(
                    preserve_candidate_ids=list(candidates["preserve"])[:2],
                    acceptance_candidate_ids=list(candidates["acceptance"])[:2],
                )
                intent_contract = build_repair_intent_contract(
                    issue_evidence,
                    candidates,
                    deterministic_selection,
                    open_decisions=open_decisions,
                    resolved_qualifiers=resolved_qualifiers,
                    compatibility_warnings=version_warnings,
                )
                statuses = {item.status for item in version_space}
                if not adjudicator_available:
                    stop_reason = "version_space_adjudicator_unavailable"
                elif "conflict" in statuses:
                    stop_reason = "version_space_conflict"
                elif "unresolved" in statuses:
                    stop_reason = "version_space_unresolved"
                else:
                    stop_reason = "version_space_resolved"
                return IntentDialogueResult(
                    issue_evidence=issue_evidence,
                    intent_contract=intent_contract,
                    exploration_plan=plan,
                    investigation=investigation,
                    synthesis=deterministic_selection,
                    adjudication=adjudication,
                    version_space=version_space,
                    stop_reason=stop_reason,
                )

            candidates, prompt_candidates = _candidate_registry(
                [candidate_exchange]
            )
            open_decisions = _grounded_open_decisions([report])
            if open_decisions:
                # A frozen probe must not silently resolve a product choice
                # that the repository investigation explicitly left open.
                candidates["acceptance"] = {}
                prompt_candidates["acceptance"] = []
            # The report is produced for the shared plan, so attributing every
            # candidate to the first TODO would create false provenance. Task
            # findings remain available to the synthesizer for real linkage.
            for candidate_group in candidates.values():
                for candidate in candidate_group.values():
                    candidate.pop("objective_id", None)
            for candidate_group in prompt_candidates.values():
                for candidate in candidate_group:
                    candidate.pop("objective_id", None)
            synthesis = self.extractor(
                build_intent_synthesis_prompt(
                    issue,
                    issue_evidence,
                    [item.model_dump() for item in plan.tasks],
                    [{
                        "status": report.status,
                        "summary": report.answer,
                        "task_findings": [
                            item.model_dump() for item in report.task_findings
                        ],
                        "repository_facts": report.repository_facts,
                        "preserve_behavior": report.preserve_behavior,
                        "acceptance_checks": [
                            item.model_dump() for item in report.acceptance_checks
                        ],
                        "conflicts_with_issue": report.conflicts_with_issue,
                        "open_decisions": report.open_decisions,
                    }],
                    prompt_candidates,
                ),
                IntentSynthesisSelection,
            )
            if synthesis is None:
                return IntentDialogueResult(
                    issue_evidence=issue_evidence,
                    intent_contract=base_contract,
                    exploration_plan=plan,
                    investigation=investigation,
                    stop_reason="synthesizer_unavailable",
                )

            synthesis = _normalize_synthesis(synthesis, candidates)
            has_decisive_delta = _has_decisive_contract_selection(synthesis)
            intent_contract = (
                build_repair_intent_contract(
                    issue_evidence,
                    candidates,
                    synthesis,
                )
                if has_decisive_delta
                else base_contract
            )
            return IntentDialogueResult(
                issue_evidence=issue_evidence,
                intent_contract=intent_contract,
                exploration_plan=plan,
                investigation=investigation,
                synthesis=synthesis,
                stop_reason=(
                    "synthesized"
                    if has_decisive_delta
                    else "synthesized_no_delta"
                ),
            )

        for objective in plan.objectives:
            answer, repository_evidence, developer_actions = (
                self.investigator.answer(
                    issue,
                    issue_evidence,
                    reproduction,
                    objective,
                )
            )

            valid_evidence_ids = {item.id for item in repository_evidence}
            answer.repository_evidence_ids = [
                item
                for item in answer.repository_evidence_ids
                if item in valid_evidence_ids
            ]
            if (
                answer.status != "unresolved"
                and not answer.repository_evidence_ids
            ):
                answer.status = "unresolved"
                answer.open_decisions = _unique(
                    answer.open_decisions
                    + ["The answer did not cite valid repository evidence."],
                    limit=2,
                )

            exchanges.append(
                DialogueExchange(
                    objective=objective,
                    repository_evidence=repository_evidence,
                    answer=answer,
                    developer_actions=developer_actions,
                )
            )

        candidates, prompt_candidates = _candidate_registry(exchanges)
        open_decisions = _grounded_open_decisions([
            exchange.answer for exchange in exchanges
        ])
        if open_decisions:
            candidates["acceptance"] = {}
            prompt_candidates["acceptance"] = []
        if not any(prompt_candidates.values()):
            return IntentDialogueResult(
                issue_evidence=issue_evidence,
                intent_contract=base_contract,
                exploration_plan=plan,
                exchanges=exchanges,
                stop_reason=(
                    "investigation_open_decisions"
                    if open_decisions
                    else "investigation_no_delta"
                ),
            )

        synthesis = self.extractor(
            build_intent_synthesis_prompt(
                issue,
                issue_evidence,
                [item.model_dump() for item in plan.objectives],
                [
                    {
                        "objective_id": exchange.objective.id,
                        "status": exchange.answer.status,
                        "finding": exchange.answer.answer,
                        "repository_facts": exchange.answer.repository_facts,
                        "preserve_behavior": exchange.answer.preserve_behavior,
                        "acceptance_checks": [
                            item.model_dump()
                            for item in exchange.answer.acceptance_checks
                        ],
                        "conflicts_with_issue": (
                            exchange.answer.conflicts_with_issue
                        ),
                        "open_decisions": exchange.answer.open_decisions,
                    }
                    for exchange in exchanges
                ],
                prompt_candidates,
            ),
            IntentSynthesisSelection,
        )
        if synthesis is None:
            return IntentDialogueResult(
                issue_evidence=issue_evidence,
                intent_contract=base_contract,
                exploration_plan=plan,
                exchanges=exchanges,
                stop_reason="synthesizer_unavailable",
            )

        synthesis = _normalize_synthesis(synthesis, candidates)
        has_decisive_delta = _has_decisive_contract_selection(synthesis)
        intent_contract = (
            build_repair_intent_contract(
                issue_evidence,
                candidates,
                synthesis,
            )
            if has_decisive_delta
            else base_contract
        )
        return IntentDialogueResult(
            issue_evidence=issue_evidence,
            intent_contract=intent_contract,
            exploration_plan=plan,
            exchanges=exchanges,
            synthesis=synthesis,
            stop_reason=(
                "synthesized"
                if has_decisive_delta
                else "synthesized_no_delta"
            ),
        )
