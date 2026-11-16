import fnmatch
import hashlib
import logging
import re
from collections.abc import Callable
from pathlib import PurePosixPath
from typing import Any

from pydantic import BaseModel

from INTENT2FIX.intent_guard.prompts import (
    build_blind_adjudication_prompt,
    build_challenge_prompt,
)
from INTENT2FIX.intent_guard.schema import (
    BlindIntentAdjudication,
    EvidenceAssessment,
    EvidenceRequest,
    IntentCandidate,
    IntentChallenge,
    IntentEvidence,
    IntentGuardResult,
)
from INTENT2FIX.reproduce.intent_pipeline import (
    issue_claim_is_non_normative,
    normalize_repair_intent_bundle,
)
from INTENT2FIX.reproduce.intent_schema import (
    IssueEvidenceFrame,
    RepairIntentBundle,
)
from INTENT2FIX.utils.file import is_test


logger = logging.getLogger(__name__)

StructuredExtractor = Callable[[str, type[BaseModel]], BaseModel | None]


class IntentGuard:
    def __init__(self, extractor: StructuredExtractor, max_evidence_requests: int = 3):
        self._extractor = extractor
        self._max_evidence_requests = max_evidence_requests

    def validate(
        self,
        proposed_intent: RepairIntentBundle,
        issue: str,
        issue_evidence: IssueEvidenceFrame,
        reproduction: Any,
        code_index: Any,
        repository: Any,
    ) -> IntentGuardResult:
        original_proposed_intent = proposed_intent
        proposed_intent = normalize_repair_intent_bundle(proposed_intent)
        prevalidated_revision = proposed_intent != original_proposed_intent
        evidence = self._initial_evidence(issue, issue_evidence, reproduction)
        try:
            challenge = self._extractor(
                build_challenge_prompt(
                    issue, issue_evidence, proposed_intent, evidence
                ),
                IntentChallenge,
            )
            if not isinstance(challenge, IntentChallenge):
                raise ValueError("challenge extraction returned no valid result")

            evidence.extend(
                self._collect_requested_evidence(
                    challenge.evidence_requests[: self._max_evidence_requests],
                    code_index,
                    repository,
                    existing_ids={item.id for item in evidence},
                )
            )

            candidates, mapping = self._anonymous_candidates(
                issue, proposed_intent, challenge
            )
            adjudication = self._extractor(
                build_blind_adjudication_prompt(
                    issue=issue,
                    issue_evidence=issue_evidence,
                    candidates=candidates,
                    observable_difference=challenge.observable_difference,
                    evidence=evidence,
                ),
                BlindIntentAdjudication,
            )
            if not isinstance(adjudication, BlindIntentAdjudication):
                raise ValueError("blind adjudication returned no valid result")

            return self._finalize(
                proposed_intent=proposed_intent,
                challenge=challenge,
                evidence=evidence,
                adjudication=adjudication,
                mapping=mapping,
                original_proposed_intent=original_proposed_intent,
                prevalidated_revision=prevalidated_revision,
            )
        except Exception as exc:
            logger.exception("IntentGuard failed open because of a technical error")
            return IntentGuardResult(
                execution_status="TECHNICAL_FAILURE",
                decision=(
                    "REVISE_CURRENT" if prevalidated_revision else "ACCEPT_CURRENT"
                ),
                proposed_intent=original_proposed_intent,
                effective_bundle=proposed_intent,
                evidence=evidence,
                decision_reason=f"IntentGuard technical failure: {exc}",
            )

    def _initial_evidence(
        self,
        issue: str,
        issue_evidence: IssueEvidenceFrame,
        reproduction: Any,
    ) -> list[IntentEvidence]:
        evidence: list[IntentEvidence] = []
        for claim in issue_evidence.claims:
            if issue_claim_is_non_normative(issue, claim):
                role = "DESCRIPTIVE"
            elif claim.kind == "preservation":
                role = "PRESERVATION"
            elif claim.kind in {"expected_behavior", "affected_scope"}:
                role = "NORMATIVE"
            else:
                role = "DESCRIPTIVE"
            evidence.append(IntentEvidence(
                id=f"ISSUE-{claim.id}",
                source_type="ISSUE",
                role=role,
                provenance=f"issue claim {claim.id}",
                content=f"{claim.text}\nExact issue span: {claim.issue_span}",
                claim_ids=[claim.id],
            ))

        for index, item in enumerate(
            getattr(reproduction, "repository_evidence", [])[:12], start=1
        ):
            evidence.append(IntentEvidence(
                id=f"REPRO-SEARCH-{index}",
                source_type="REPOSITORY",
                role="DESCRIPTIVE",
                provenance=f"reproducer {item.get('action', 'search')} observation",
                content=str(item.get("observation", ""))[:4000],
            ))

        for record in getattr(reproduction, "execution_records", [])[-4:]:
            evidence.append(IntentEvidence(
                id=f"EXEC-{record.run_id}",
                source_type="EXECUTION",
                role="DESCRIPTIVE",
                provenance=f"{record.run_id}: {record.command}",
                content=record.output[-4000:],
                claim_ids=list(getattr(reproduction, "matched_claim_ids", [])),
            ))

        for index, binding in enumerate(
            getattr(reproduction, "graph_bindings", [])[:8], start=1
        ):
            evidence.append(IntentEvidence(
                id=f"GRAPH-{index}",
                source_type="REPOSITORY",
                role="DESCRIPTIVE",
                provenance=f"{binding.entity} in {binding.file}",
                content=binding.evidence or binding.relation,
                claim_ids=binding.claim_ids,
            ))
        return evidence

    def _collect_requested_evidence(
        self,
        requests: list[EvidenceRequest],
        code_index: Any,
        repository: Any,
        existing_ids: set[str],
    ) -> list[IntentEvidence]:
        collected: list[IntentEvidence] = []
        seen_locations: set[tuple[str, int, int]] = set()
        sequence = 1

        def append_evidence(path: str, start: int, end: int, query: str) -> None:
            nonlocal sequence
            location = (path, start, end)
            if location in seen_locations:
                return
            seen_locations.add(location)
            evidence_id = f"GUARD-REPO-{sequence}"
            while evidence_id in existing_ids:
                sequence += 1
                evidence_id = f"GUARD-REPO-{sequence}"
            sequence += 1
            existing_ids.add(evidence_id)
            content = repository.get_file_content(path)
            if not content:
                return
            lines = content.splitlines()
            end = min(end, len(lines))
            source_type = self._source_type_for_path(path)
            collected.append(IntentEvidence(
                id=evidence_id,
                source_type=source_type,
                role=self._maximum_role_for_source(source_type),
                provenance=f"{path}:{start}-{end} for query {query!r}",
                content="\n".join(
                    f"{line_no}: {lines[line_no - 1]}"
                    for line_no in range(start, end + 1)
                )[:6000],
            ))

        for request in requests:
            query = " ".join(
                part for part in (request.query, request.target_symbol) if part
            )
            if self._is_exact_file_target(request.target_file):
                content = repository.get_file_content(request.target_file)
                if content:
                    for start, end in self._direct_file_spans(content, query):
                        append_evidence(request.target_file, start, end, query)
                    continue
            try:
                response = code_index.semantic_search(
                    query=query,
                    file_pattern=request.target_file or None,
                    max_results=4,
                )
            except Exception as exc:
                logger.warning("IntentGuard evidence search failed for %r: %s", query, exc)
                continue

            hits = sorted(
                (
                    hit for hit in getattr(response, "hits", [])
                    if self._path_matches_target(hit.file_path, request.target_file)
                ),
                key=lambda hit: self._path_priority(hit.file_path),
            )
            for hit in hits[:3]:
                content = repository.get_file_content(hit.file_path)
                if not content:
                    continue
                lines = content.splitlines()
                spans = hit.spans[:2] if hit.spans else [None]
                for span in spans:
                    span_start = span.start_line if span else 1
                    span_end = span.end_line if span else min(len(lines), 80)
                    start = max(1, int(span_start or 1) - 20)
                    end = min(len(lines), int(span_end or span_start or start) + 40)
                    append_evidence(hit.file_path, start, end, query)
        return collected

    @staticmethod
    def _is_exact_file_target(target: str) -> bool:
        return bool(
            target
            and not target.endswith("/")
            and not any(char in target for char in "*?[")
        )

    @staticmethod
    def _path_matches_target(path: str, target: str) -> bool:
        if not target:
            return True
        normalized_path = path.lstrip("./")
        normalized_target = target.lstrip("./")
        if normalized_target.endswith("/"):
            return normalized_path.startswith(normalized_target)
        if any(char in normalized_target for char in "*?["):
            return fnmatch.fnmatch(normalized_path, normalized_target)
        return normalized_path == normalized_target

    @staticmethod
    def _direct_file_spans(content: str, query: str) -> list[tuple[int, int]]:
        lines = content.splitlines()
        if not lines:
            return []
        terms = list(dict.fromkeys(
            term.lower() for term in re.findall(
                r"operator(?:==|!=|<=|>=|<|>)|[A-Za-z_]\w*",
                query,
            )
            if len(term) >= 3
        ))
        anchor = next((term for term in reversed(terms) if term.startswith("operator")), "")
        candidates = [
            index for index, line in enumerate(lines)
            if anchor and anchor in line.lower()
        ]
        if not candidates:
            candidates = [
                index for index, line in enumerate(lines)
                if any(term in line.lower() for term in terms)
            ]
        if not candidates:
            return [(1, min(len(lines), 80))]
        ranked = []
        for index in candidates:
            start = max(0, index - 20)
            end = min(len(lines), index + 41)
            window = "\n".join(lines[start:end]).lower()
            score = sum(window.count(term) for term in terms)
            ranked.append((-score, index, start + 1, end))
        spans = []
        for _, _, start, end in sorted(ranked):
            if any(not (end < old_start or start > old_end) for old_start, old_end in spans):
                continue
            spans.append((start, end))
            if len(spans) == 2:
                break
        return spans

    @staticmethod
    def _path_priority(path: str) -> tuple[int, str]:
        source_type = IntentGuard._source_type_for_path(path)
        if source_type == "TEST":
            return (0, path)
        if source_type == "DOC":
            return (1, path)
        return (2, path)

    @staticmethod
    def _source_type_for_path(path: str) -> str:
        if is_test(path):
            return "TEST"
        pure_path = PurePosixPath(path)
        parts = {part.lower() for part in pure_path.parts}
        if (
            parts.intersection({"doc", "docs", "documentation"})
            or pure_path.suffix.lower() in {".md", ".rst"}
            or pure_path.name.lower().startswith(("readme", "changelog"))
        ):
            return "DOC"
        return "REPOSITORY"

    @staticmethod
    def _maximum_role_for_source(source_type: str) -> str:
        if source_type == "TEST":
            return "PRESERVATION"
        if source_type == "DOC":
            return "NORMATIVE"
        return "DESCRIPTIVE"

    def _anonymous_candidates(
        self,
        issue: str,
        proposed_intent: RepairIntentBundle,
        challenge: IntentChallenge,
    ) -> tuple[list[dict], dict[str, str]]:
        current = IntentCandidate.from_bundle(proposed_intent)
        challenger = challenge.challenger
        if challenger is None:
            return (
                [{"candidate_id": "CANDIDATE_A", "intent": current.model_dump()}],
                {"CANDIDATE_A": "CURRENT"},
            )

        current_first = hashlib.sha256(issue.encode("utf-8")).digest()[0] % 2 == 0
        ordered = [current, challenger] if current_first else [challenger, current]
        origins = ["CURRENT", "CHALLENGER"] if current_first else ["CHALLENGER", "CURRENT"]
        ids = ["CANDIDATE_A", "CANDIDATE_B"]
        return (
            [
                {"candidate_id": candidate_id, "intent": candidate.model_dump()}
                for candidate_id, candidate in zip(ids, ordered)
            ],
            dict(zip(ids, origins)),
        )

    def _finalize(
        self,
        proposed_intent: RepairIntentBundle,
        challenge: IntentChallenge,
        evidence: list[IntentEvidence],
        adjudication: BlindIntentAdjudication,
        mapping: dict[str, str],
        original_proposed_intent: RepairIntentBundle,
        prevalidated_revision: bool,
    ) -> IntentGuardResult:
        known_evidence_ids = {item.id for item in evidence}
        assessments = [
            item for item in adjudication.evidence_assessments
            if item.evidence_id in known_evidence_ids
        ]
        candidate_origin = mapping.get(adjudication.selected_candidate_id or "")
        current = IntentCandidate.from_bundle(proposed_intent)
        if adjudication.selection == "SELECTED" and candidate_origin == "CURRENT":
            effective = current
        elif adjudication.selection == "SELECTED" and candidate_origin == "CHALLENGER":
            effective = challenge.challenger.model_copy(deep=True)
        else:
            effective = adjudication.effective_intent.model_copy(deep=True)

        unsupported = self._unclosed_hard_obligations(
            effective,
            assessments,
            evidence,
            adjudication.selected_candidate_id,
        )
        had_hard_obligations = bool(effective.hard_obligations)
        reason = adjudication.decision_reason

        if unsupported:
            unsupported_set = set(unsupported)
            unsupported_obligations = [
                item for item in effective.hard_obligations
                if item.id in unsupported_set
            ]
            effective.hard_obligations = [
                item for item in effective.hard_obligations
                if item.id not in unsupported_set
            ]
            effective.unresolved.extend(
                f"IntentGuard found no normative or preservation evidence for obligation {item_id}."
                for item_id in dict.fromkeys(unsupported)
            )
            effective.rejected_interpretations.extend(
                f"Unsupported hard obligation {item.id}: {item.text}"
                for item in unsupported_obligations
                if not any(item.id in rejected for rejected in effective.rejected_interpretations)
            )
            reason = "Removed hard obligations that lacked evidence closure."

        effective.unresolved.extend(
            item for item in adjudication.remaining_uncertainty
            if item not in effective.unresolved
        )

        normalized_effective = normalize_repair_intent_bundle(
            self._candidate_bundle(effective, proposed_intent, "WEAK"),
            promote_supported_core=True,
        )
        effective = IntentCandidate.from_bundle(normalized_effective)

        selection_failed = (
            adjudication.selection == "UNRESOLVED"
            or candidate_origin is None
        )
        supported_core_empty = had_hard_obligations and not effective.hard_obligations
        revised = bool(
            prevalidated_revision
            or unsupported
            or adjudication.remaining_uncertainty
        )

        if selection_failed or supported_core_empty:
            decision = "UNRESOLVED"
            status = "UNRESOLVED"
            if candidate_origin is None and adjudication.selection == "SELECTED":
                reason = "Blind adjudication selected an unknown candidate."
            elif supported_core_empty:
                reason = "No evidence-supported hard-obligation core remained."
        elif candidate_origin == "CHALLENGER" or revised:
            decision = "REVISE_CURRENT"
            status = normalized_effective.status
        else:
            decision = "ACCEPT_CURRENT"
            status = proposed_intent.status

        if decision == "ACCEPT_CURRENT":
            effective_bundle = normalize_repair_intent_bundle(
                proposed_intent,
                promote_supported_core=True,
            )
        else:
            effective_bundle = normalize_repair_intent_bundle(
                self._candidate_bundle(effective, proposed_intent, status),
                promote_supported_core=decision != "UNRESOLVED",
            )

        return IntentGuardResult(
            execution_status="OK",
            decision=decision,
            proposed_intent=original_proposed_intent,
            effective_bundle=effective_bundle,
            challenge=challenge,
            evidence=evidence,
            adjudication=adjudication,
            candidate_mapping=mapping,
            decision_reason=reason,
        )

    @staticmethod
    def _unclosed_hard_obligations(
        candidate: IntentCandidate,
        assessments: list[EvidenceAssessment],
        evidence: list[IntentEvidence],
        selected_candidate_id: str | None,
    ) -> list[str]:
        evidence_by_id = {item.id: item for item in evidence}
        unclosed = []
        for obligation in candidate.hard_obligations:
            relevant = [
                assessment
                for assessment in assessments
                if (
                    selected_candidate_id in assessment.candidate_ids
                    if selected_candidate_id
                    else not assessment.candidate_ids
                )
                and obligation.id in assessment.obligation_ids
            ]
            supports = [
                assessment for assessment in relevant
                if assessment.relation == "SUPPORTS"
                and IntentGuard._can_close_obligation(
                    assessment,
                    evidence_by_id.get(assessment.evidence_id),
                    relevant,
                    evidence_by_id,
                )
            ]
            contradictions = [
                assessment for assessment in relevant
                if assessment.relation == "CONTRADICTS"
                and IntentGuard._can_close_obligation(
                    assessment,
                    evidence_by_id.get(assessment.evidence_id),
                    relevant,
                    evidence_by_id,
                )
            ]
            if not supports or contradictions:
                unclosed.append(obligation.id)
        return unclosed

    @staticmethod
    def _can_close_obligation(
        assessment: EvidenceAssessment,
        evidence: IntentEvidence | None,
        related_assessments: list[EvidenceAssessment],
        evidence_by_id: dict[str, IntentEvidence],
    ) -> bool:
        if evidence is None or assessment.role == "DESCRIPTIVE":
            return False
        if evidence.source_type == "ISSUE":
            return evidence.role == assessment.role
        if evidence.source_type == "DOC":
            return assessment.role == "NORMATIVE"
        if evidence.source_type == "TEST":
            return assessment.role == "PRESERVATION"
        if evidence.source_type != "REPOSITORY" or assessment.role != "PRESERVATION":
            return False
        repository_ids = {
            item.evidence_id
            for item in related_assessments
            if item.role == "PRESERVATION"
            and item.relation == assessment.relation
            and evidence_by_id[item.evidence_id].source_type == "REPOSITORY"
        }
        return len(repository_ids) >= 2

    @staticmethod
    def _candidate_bundle(
        candidate: IntentCandidate,
        base: RepairIntentBundle,
        status: str,
    ) -> RepairIntentBundle:
        return RepairIntentBundle(
            status=status,
            reproduction_status=base.reproduction_status,
            summary=candidate.summary or base.summary,
            behavioral_delta=candidate.behavioral_delta,
            hard_obligations=candidate.hard_obligations,
            soft_hypotheses=candidate.soft_hypotheses,
            rejected_interpretations=candidate.rejected_interpretations,
            graph_bindings=base.graph_bindings,
            tests=base.tests,
            execution_records=base.execution_records,
            unresolved=candidate.unresolved,
        )
