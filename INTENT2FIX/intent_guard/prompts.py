import json

from INTENT2FIX.intent_guard.schema import (
    BlindIntentAdjudication,
    IntentChallenge,
    IntentEvidence,
)
from INTENT2FIX.reproduce.intent_schema import IssueEvidenceFrame, RepairIntentBundle


def build_challenge_prompt(
    issue: str,
    issue_evidence: IssueEvidenceFrame,
    proposed_intent: RepairIntentBundle,
    evidence: list[IntentEvidence],
) -> str:
    schema = json.dumps(IntentChallenge.model_json_schema(), ensure_ascii=False)
    payload = {
        "issue_evidence": issue_evidence.model_dump(),
        "proposed_intent": proposed_intent.compact_dump(),
        "available_evidence": [item.model_dump() for item in evidence],
    }
    return f"""Challenge an already-proposed repair intent for a C++ issue.

Do not regenerate the issue specification from scratch. Treat the proposed intent
as H0 and identify at most one nearest behavioral contrast. Focus on the single
most dangerous omission or unsupported generalization in H0: preserved behavior,
affected scope, observable error/return/reporting semantics, compatibility, or an
implementation suggestion incorrectly promoted to a requirement.

The challenger must differ from H0 on a concrete observable behavior. If no such
difference can be stated, set risk_kind to NONE and challenger to null. Do not
invent expected behavior from the current buggy implementation. Execution proves
the current symptom only. Request at most three targeted repository searches,
preferentially for existing tests, public documentation, sibling behavior, or
callers that depend on a preservation property. Hard obligations must describe
behavior, never a patch operation or a particular overload/line/edit.
Treat tentative or explicitly unrelated issue commentary as ambiguity, not as
normative or preservation evidence. A named function, signature, overload, file,
or line can guide evidence search but cannot by itself constrain the repair.

Return JSON matching this schema:
{schema}

Original issue:
{issue}

Input:
{json.dumps(payload, ensure_ascii=False)}
"""


def build_blind_adjudication_prompt(
    issue: str,
    issue_evidence: IssueEvidenceFrame,
    candidates: list[dict],
    observable_difference: str,
    evidence: list[IntentEvidence],
) -> str:
    schema = json.dumps(BlindIntentAdjudication.model_json_schema(), ensure_ascii=False)
    payload = {
        "issue_evidence": issue_evidence.model_dump(),
        "candidates": candidates,
        "observable_difference": observable_difference,
        "evidence": [item.model_dump() for item in evidence],
    }
    return f"""Blindly adjudicate anonymous repair-intent candidates for a C++ issue.

Candidate order does not indicate which candidate is current. Evaluate the
strongest contradiction before support. For every evidence item you rely on,
return an EvidenceAssessment using its exact evidence ID.

Evidence source alone does not determine normative strength:
- The role attached to an evidence item is the strongest role the deterministic
  gate permits, not a conclusion that the item has that role.
- DESCRIPTIVE evidence establishes current behavior, not desired behavior.
- A test is PRESERVATION evidence only when the issue does not request changing
  that tested behavior.
- Multiple independent repository observations can jointly support preservation.
- Implementation suggestions and current buggy code cannot become hard behavior.
- A hard obligation must be verifiable from inputs, outputs, errors, compatibility,
  or preserved tests without reading the patch. Requirements to keep, remove,
  select, or change a function, overload, signature, file, or internal type are
  implementation requirements and must be rejected.

Every assessment that supports or contradicts a candidate obligation must name
that candidate ID and obligation ID. Unsupported obligations will be removed by
the deterministic gate even if the final decision says ACCEPT_CURRENT.

You do not know which anonymous candidate is the current intent. Set selection to
SELECTED and return its exact candidate ID when one candidate is better supported.
Set selection to UNRESOLVED and selected_candidate_id to null only when evidence
cannot choose safely. In that case, effective_intent must contain only the
evidence-supported common core; assessments for that common core must leave
candidate_ids empty. Put undecidable observable semantics in remaining_uncertainty
and unsupported assumptions in rejected_interpretations. Implementation choices
such as overload shape, SFINAE conditions, or conversion mechanism are not
contract uncertainty and belong in soft_hypotheses, not remaining_uncertainty.
Do not downgrade a complete, reproduced behavioral contract merely because its
implementation is still unknown.
Do not return a list of alternatives for the Repair Agent to choose from.

Return JSON matching this schema:
{schema}

Original issue:
{issue}

Input:
{json.dumps(payload, ensure_ascii=False)}
"""
