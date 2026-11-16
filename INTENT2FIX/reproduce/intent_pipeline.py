import json
import re
from typing import Any, Iterable, Type, TypeVar

from pydantic import BaseModel

from INTENT2FIX.completion.structured_compactor import parse_json_object
from INTENT2FIX.reproduce.intent_schema import (
    ExecutionRecord,
    IssueEvidenceFrame,
    IssueIntentUnit,
    RepairIntentBundle,
    TestArtifact,
)


SchemaT = TypeVar("SchemaT", bound=BaseModel)


_UNCERTAIN_STATEMENT = re.compile(
    r"\b(?:not sure|unclear|perhaps|maybe|possibly)\b|"
    r"\b(?:i|we)\s+(?:do not|don't)\s+think\b|"
    r"\b(?:i am|i'm|we are|we're)\s+not\s+sure\b",
    re.IGNORECASE,
)

_ROOT_CAUSE_SCOPE = re.compile(
    r"\b(?:root cause|source of (?:the )?issue|caused by|better match)\b|"
    r"\bline\s+\d+\b",
    re.IGNORECASE,
)


def issue_claim_is_non_normative(issue: str, claim: Any) -> bool:
    statement = f"{claim.text}\n{claim.issue_span}"
    if claim.kind == "preservation":
        if _UNCERTAIN_STATEMENT.search(statement):
            return True
        span_offset = issue.find(claim.issue_span)
        if span_offset >= 0:
            preceding_context = issue[max(0, span_offset - 200):span_offset]
            if re.search(r"unrelated\s+note", preceding_context, re.IGNORECASE):
                return True
    return claim.kind == "affected_scope" and bool(
        _ROOT_CAUSE_SCOPE.search(statement)
    )


def normalize_repair_intent_bundle(
    bundle: RepairIntentBundle,
    *,
    promote_supported_core: bool = False,
) -> RepairIntentBundle:
    normalized = bundle.model_copy(deep=True)
    supported_core = bool(normalized.hard_obligations) and all(
        obligation.support not in {"inferred", "unresolved", "contradicted"}
        for obligation in normalized.hard_obligations
    )
    if normalized.status != "UNRESOLVED":
        if not normalized.hard_obligations:
            normalized.status = "UNRESOLVED"
            message = "No observable hard-obligation core remained after intent validation."
            if message not in normalized.unresolved:
                normalized.unresolved.append(message)
        elif (
            promote_supported_core
            and normalized.reproduction_status == "MATCHED"
            and supported_core
            and not normalized.unresolved
        ):
            normalized.status = "STRONG"
        elif normalized.status == "STRONG" and normalized.unresolved:
            normalized.status = "WEAK"
    return normalized


def parse_structured_response(raw: str, schema: Type[SchemaT]) -> SchemaT:
    payload = parse_json_object(raw)
    # Some JSON-mode providers echo the schema's `properties` wrapper but put
    # their actual field values inside it. Models with all-default fields can
    # otherwise accept the wrapper as an empty response because extra keys are
    # ignored. Restrict unwrapping to a schema-shaped root and let validation of
    # the inner object reject a literal JSON Schema with type descriptors.
    schema_wrapper_keys = {
        "properties",
        "required",
        "type",
        "title",
        "description",
        "additionalProperties",
    }
    if (
        isinstance(payload, dict)
        and "properties" not in schema.model_fields
        and set(payload).issubset(schema_wrapper_keys)
        and isinstance(payload.get("properties"), dict)
    ):
        return schema.model_validate(payload["properties"])
    return schema.model_validate(payload)


def build_issue_evidence_prompt(issue: str) -> str:
    schema = json.dumps(IssueEvidenceFrame.model_json_schema(), ensure_ascii=False)
    return f"""Analyze the C++ issue below without inspecting repository implementation.

Extract only immutable Issue intent, evidence, and search anchors. Do not produce
a repository-informed final repair contract, root cause, patch plan, hard/soft
decision, or unsupported generalization.
Coverage is mandatory: extract every explicit requested or expected observable
behavior as an `expected_behavior` claim, including imperative issue titles such
as "Add", "Make", "Support", or "Allow". Do not omit a requested behavior merely
because the issue also suggests how to implement it. If no explicit requested
behavior exists, do not invent one.
Also compile the normative claims into `intent_units`. Each `INTENT-N` is one
atomic, independently observable target or preservation obligation. Split a
sentence only when it explicitly states multiple outcomes that could be
implemented or verified independently. Do not split a trigger, example, scope
phrase, or implementation suggestion into a separate target. Every unit must
copy an exact supporting `issue_span`, cite its normative source claim, and may
also cite directly linked trigger, observation, scope, or example claim IDs as
retrieval context for that same obligation. Use role
`target` only for expected_behavior claims and `preserve` only for preservation
or explicit non_goal claims. Intent units preserve Issue meaning; they do not
fill unspecified defaults, API families, return representations, or error text.
An example or analogy authorizes only the exact observable relation named in
its Issue span. If the Issue says "X does Y" merely to illustrate requested Y,
the intent unit may require Y but must not add X's other ordering, precedence,
fallback, state, or error semantics. Keep the example claim as retrieval context
instead of appending qualifiers such as "consistent with all X semantics" to
the normative intent text. The `text` and `issue_span` of an intent unit must be
derived only from its normative expected_behavior, preservation, or non_goal
claim. When a unit has exactly one normative claim, use that claim's behavior
text alone; do not append the wording of linked trigger, scope, or example
claims.
Use stable claim IDs with prefixes TRG, OBS, EXP, SCOPE, PRES, NONGOAL, and EX.
`issue_span` must quote the shortest exact single-line span supporting the claim.
Do not place fenced code blocks in issue_span. Escape quotes and backslashes so
the response remains valid JSON.
Implementation suggestions and author-proposed root causes must remain separate
from requirements. `affected_scope` describes user-visible environments, input
categories, or APIs; a function, file, line, overload, or claimed root cause is a
search anchor or implementation suggestion, not affected scope. `preservation`
requires explicit compatibility language or a reported regression of existing
behavior. Statements marked unrelated, speculative, or uncertain (for example
"I don't think" or "not sure") belong in ambiguities and must not become claims.
When one sentence combines a concrete code recipe with an explicit statement
that the same defect or repair applies similarly to named operations or all
members of a public operation family, split the two: keep the code shape in
`implementation_suggestions`, and extract the applicability wording as a
`scope` claim. The scope statement does not need its own Expected Behavior
heading, but it must not make the proposed signature or helper normative.
An author asking whether the current missing behavior "is by design" expresses
uncertainty about the reason for the baseline; it does not by itself create an
alternative default, opt-in, gated, or compatibility contract. When the title
or another exact span states the expected behavior, preserve that target and do
not weaken it through an inferred product-policy ambiguity.
Use `implementation_suggestions` only for concrete causes or mechanisms the
author actually proposes. Do not place model-inferred repair policies or API
alternatives there.
Use `non_goal` only when the issue explicitly excludes a behavior or change.
Do not strengthen tentative wording when paraphrasing claim text.

Return JSON matching this schema:
{schema}

Issue:
{issue}
"""


def fallback_issue_evidence(issue: str) -> IssueEvidenceFrame:
    claims = [
        {
            "id": "OBS-FALLBACK-1",
            "kind": "observed_behavior",
            "text": "The behavior reported by the original issue must be reproduced.",
            "issue_span": issue[:1000],
            "search_terms": [],
        }
    ]
    lines = issue.splitlines()
    first_line = next((line.strip() for line in lines if line.strip()), "")
    title = re.sub(r"^#+\s*", "", first_line).strip()
    if re.match(
        r"^(?:add|allow|enable|expose|fix|implement|make|maintain|preserve|"
        r"provide|support)\b",
        title,
        re.IGNORECASE,
    ):
        claims.append({
            "id": "EXP-FALLBACK-TITLE",
            "kind": "expected_behavior",
            "text": title,
            "issue_span": first_line,
            "search_terms": [],
        })
    else:
        negative_capability = re.match(
            r"^(?P<subject>.+?)\s+is\s+no\s+(?P<capability>.+)$",
            title,
            re.IGNORECASE,
        )
        if negative_capability:
            subject = negative_capability.group("subject").strip()
            capability = negative_capability.group("capability").strip()
            claims.append({
                "id": "EXP-FALLBACK-TITLE",
                "kind": "expected_behavior",
                "text": f"{subject} should satisfy {capability}.",
                "issue_span": first_line,
                "search_terms": [subject, capability],
            })
        else:
            missing_behavior = re.match(
                r"^(?P<subject>.+?)\s+(?:does|do|did)\s+not\s+"
                r"(?P<behavior>.+)$",
                title,
                re.IGNORECASE,
            )
            if missing_behavior:
                subject = missing_behavior.group("subject").strip()
                behavior = missing_behavior.group("behavior").strip()
                claims.append({
                    "id": "EXP-FALLBACK-TITLE",
                    "kind": "expected_behavior",
                    "text": f"{subject} should {behavior}.",
                    "issue_span": first_line,
                    "search_terms": [subject, behavior],
                })

    expected_headings = {"expected", "expectedbehavior", "expectedresult"}
    for index, line in enumerate(lines):
        heading = re.sub(r"[^a-z]", "", line.lower())
        if heading not in expected_headings:
            continue
        for candidate in lines[index + 1:]:
            span = candidate.strip()
            if span and not span.startswith(("```", "#", "**")):
                claims.append({
                    "id": "EXP-FALLBACK-1",
                    "kind": "expected_behavior",
                    "text": span,
                    "issue_span": span,
                    "search_terms": [],
                })
                break
        break

    preservation_pattern = re.compile(
        r"\b(?:broke|breaks?) (?:a|an|the|existing) test\b|"
        r"\b(?:must|should) (?:still|remain|continue)\b|"
        r"\bbackwards? compatib",
        re.IGNORECASE,
    )
    for line in lines:
        span = line.strip()
        if span and preservation_pattern.search(span):
            claims.append({
                "id": "PRES-FALLBACK-1",
                "kind": "preservation",
                "text": span,
                "issue_span": span,
                "search_terms": [],
            })
            break

    return IssueEvidenceFrame.model_validate({
        "claims": claims,
        "extraction_status": "fallback",
        "ambiguities": [
            "Structured issue extraction failed; fallback claims come only from exact issue sections."
        ],
    })


def issue_evidence_is_grounded(issue: str, frame: IssueEvidenceFrame | None) -> bool:
    grounded = ground_issue_evidence(issue, frame)
    return grounded is not None and len(grounded.claims) == len(frame.claims)


def ground_issue_evidence(
    issue: str,
    frame: IssueEvidenceFrame | None,
) -> IssueEvidenceFrame | None:
    if frame is None or not frame.claims:
        return None
    expected_prefixes = {
        "trigger": "TRG-",
        "observed_behavior": "OBS-",
        "expected_behavior": "EXP-",
        "affected_scope": "SCOPE-",
        "preservation": "PRES-",
        "non_goal": "NONGOAL-",
        "example": "EX-",
    }
    seen_ids = set()
    valid_claims = []
    non_normative_claims = []
    implementation_claims = []
    for claim in frame.claims:
        if (
            claim.id not in seen_ids
            and claim.id.startswith(expected_prefixes[claim.kind])
            and bool(claim.issue_span)
            and claim.issue_span in issue
        ):
            if issue_claim_is_non_normative(issue, claim):
                if claim.kind == "preservation":
                    non_normative_claims.append(claim)
                else:
                    implementation_claims.append(claim)
                seen_ids.add(claim.id)
                continue
            valid_claims.append(claim)
            seen_ids.add(claim.id)
    if not valid_claims:
        return None
    grounded = frame.model_copy(deep=True)
    grounded.claims = valid_claims
    claim_by_id = {claim.id: claim for claim in valid_claims}
    valid_intent_units = []
    seen_intent_ids = set()
    for unit in frame.intent_units:
        source_claims = [
            claim_by_id[claim_id]
            for claim_id in unit.source_claim_ids
            if claim_id in claim_by_id
        ]
        allowed_kinds = (
            {"expected_behavior"}
            if unit.role == "target"
            else {"preservation", "non_goal"}
        )
        if (
            unit.id in seen_intent_ids
            or unit.issue_span not in issue
            or not source_claims
            or not any(claim.kind in allowed_kinds for claim in source_claims)
        ):
            continue
        normalized = unit.model_copy(deep=True)
        normalized.source_claim_ids = list(dict.fromkeys(
            claim.id for claim in source_claims
        ))
        if not normalized.source_claim_ids:
            continue
        normative_claims = [
            claim for claim in source_claims if claim.kind in allowed_kinds
        ]
        if len(normative_claims) == 1:
            # Retrieval context may cite an example or trigger, but it cannot
            # alter the authoritative text of a single normative obligation.
            normalized.text = normative_claims[0].text
            normalized.issue_span = normative_claims[0].issue_span
        valid_intent_units.append(normalized)
        seen_intent_ids.add(unit.id)

    covered_target_claim_ids = {
        claim_id
        for unit in valid_intent_units
        if unit.role == "target"
        for claim_id in unit.source_claim_ids
        if claim_by_id[claim_id].kind == "expected_behavior"
    }
    covered_preserve_claim_ids = {
        claim_id
        for unit in valid_intent_units
        if unit.role == "preserve"
        for claim_id in unit.source_claim_ids
        if claim_by_id[claim_id].kind in {"preservation", "non_goal"}
    }
    next_intent_number = 1
    while f"INTENT-{next_intent_number}" in seen_intent_ids:
        next_intent_number += 1

    # Completeness fallback preserves every normative claim when the semantic
    # extractor omitted a unit. It does not attempt semantic decomposition.
    for claim in valid_claims:
        role = None
        covered_ids = set()
        if claim.kind == "expected_behavior":
            role = "target"
            covered_ids = covered_target_claim_ids
        elif claim.kind in {"preservation", "non_goal"}:
            role = "preserve"
            covered_ids = covered_preserve_claim_ids
        if role is None or claim.id in covered_ids:
            continue
        while f"INTENT-{next_intent_number}" in seen_intent_ids:
            next_intent_number += 1
        intent_id = f"INTENT-{next_intent_number}"
        next_intent_number += 1
        valid_intent_units.append(IssueIntentUnit(
            id=intent_id,
            role=role,
            text=claim.text,
            issue_span=claim.issue_span,
            source_claim_ids=[claim.id],
        ))
        seen_intent_ids.add(intent_id)
    unique_intent_units = []
    seen_intent_content = set()
    for unit in valid_intent_units:
        key = (
            unit.role,
            unit.text.strip(),
            unit.issue_span,
            tuple(unit.source_claim_ids),
        )
        if key in seen_intent_content:
            continue
        seen_intent_content.add(key)
        unique_intent_units.append(unit)
    unique_intent_units.sort(key=lambda unit: issue.find(unit.issue_span))
    grounded.intent_units = [
        unit.model_copy(update={"id": f"INTENT-{index}"})
        for index, unit in enumerate(unique_intent_units, 1)
    ]
    rejected_count = (
        len(frame.claims)
        - len(valid_claims)
        - len(non_normative_claims)
        - len(implementation_claims)
    )
    if rejected_count:
        grounded.ambiguities.append(
            f"Discarded {rejected_count} claim(s) without unique IDs and exact issue spans."
        )
    for claim in non_normative_claims:
        grounded.ambiguities.append(
            f"Non-normative preservation statement {claim.id}: {claim.issue_span}"
        )
    for claim in implementation_claims:
        suggestion = (
            f"Author-proposed implementation/root-cause hypothesis "
            f"({claim.id}): {claim.text}"
        )
        if suggestion not in grounded.implementation_suggestions:
            grounded.implementation_suggestions.append(suggestion)

    return grounded


def merge_issue_evidence_with_fallback(
    issue: str,
    frame: IssueEvidenceFrame | None,
) -> IssueEvidenceFrame:
    grounded = ground_issue_evidence(issue, frame)
    fallback = fallback_issue_evidence(issue)
    if grounded is None:
        return ground_issue_evidence(issue, fallback) or fallback

    existing_claims = {
        (claim.kind, claim.issue_span) for claim in grounded.claims
    }
    existing_ids = {claim.id for claim in grounded.claims}
    for fallback_claim in fallback.claims:
        if fallback_claim.kind not in {"expected_behavior", "preservation"}:
            continue
        key = (fallback_claim.kind, fallback_claim.issue_span)
        if key in existing_claims:
            continue
        claim = fallback_claim.model_copy(deep=True)
        if claim.id in existing_ids:
            prefix = claim.id.rsplit("-", 1)[0]
            sequence = 2
            while f"{prefix}-{sequence}" in existing_ids:
                sequence += 1
            claim.id = f"{prefix}-{sequence}"
        grounded.claims.append(claim)
        existing_claims.add(key)
        existing_ids.add(claim.id)
    return ground_issue_evidence(issue, grounded) or grounded


def build_intent_adjudication_prompt(
    issue: str,
    issue_evidence: IssueEvidenceFrame,
    repository_evidence: Iterable[dict[str, Any]],
    execution_records: Iterable[ExecutionRecord],
    tests: Iterable[TestArtifact],
    test_files: Iterable[dict[str, str]],
    graph_bindings: Iterable[dict[str, Any]],
) -> str:
    schema = json.dumps(RepairIntentBundle.model_json_schema(), ensure_ascii=False)
    evidence_payload = {
        "issue_evidence": issue_evidence.model_dump(),
        "repository_evidence": list(repository_evidence),
        "execution_records": [
            {
                **item.model_dump(exclude={"stdout", "stderr"}),
                "stdout": item.stdout[-2000:],
                "stderr": item.stderr[-2000:],
            }
            for item in execution_records
        ],
        "tests": [item.model_dump() for item in tests],
        "test_files": [
            {"path": item["path"], "content": item["content"][:6000]}
            for item in test_files
        ],
        "reported_graph_bindings": list(graph_bindings),
    }
    return f"""Synthesize and adjudicate a grounded repair intent for this C++ issue.

Rules:
- Preserve the original issue as the authority for requested behavior.
- Hard obligations are observable behavior contracts, not patch operations such
  as "add an overload", "keep/remove a function", "select an overload", or
  "edit a file". A hard obligation must be testable without inspecting the patch.
- Every issue-explicit operand form, direction, operator, type category, version,
  and compatibility constraint must remain in a hard obligation or be listed as
  unresolved. Do not silently collapse arrays, literals, or named abstractions
  into one concrete implementation type.
- Repository tests, public API declarations, and documentation can support scope.
- For regressions from a prior version, use repository evidence to recover the
  public API category and existing language-standard branches. Mark such scope
  repository_supported; do not turn the minimal reproducer into the full API contract.
- A statement that an older version compiled supports source compatibility, but
  does not by itself define conversion direction, comparison results, validation,
  or exception behavior. Require tests/docs for those semantics; otherwise list
  them as unresolved and let repair inspect the domain constructor and invariants.
- Execution confirms triggers and observed behavior, not desired behavior by itself.
- A foreign overload, provider, callback, route, or intermediate result selected
  on the failing baseline is localization evidence, not a required post-fix
  winner. Do not require Repair to accept or preserve it unless the Issue or an
  independent public contract explicitly requires that result family.
- Implementation and root-cause claims remain soft unless independently supported.
- Author-proposed fixes, including failed attempts and suggested overload shapes,
  are repair hypotheses rather than requirements. Preserve their counterexamples
  without promoting the proposed mechanism.
- Unknown implementation details do not make an otherwise complete behavioral
  contract WEAK. Put overload shape, ownership, SFINAE, file, and consistency
  questions in soft_hypotheses. Use unresolved only for an observable behavior
  that available issue/repository evidence cannot decide.
- Reject broader semantics that lack issue or repository support.
- Keep graph bindings only when the evidence names a concrete entity or file.
- Set reproduction_status to MISMATCHED for setup, syntax, dependency, unrelated,
  or wrong-symptom failures even if the command exited nonzero.
- Return UNRESOLVED instead of inventing a repair intent.
- Leave `execution_records` and `tests` empty in the response. The runner binds
  the authoritative records after validation; echoing them wastes output tokens.

Return JSON matching this schema:
{schema}

Original issue:
{issue}

Evidence:
{json.dumps(evidence_payload, ensure_ascii=False)}
"""


def unresolved_bundle(
    reason: str,
    execution_records: Iterable[ExecutionRecord] = (),
    tests: Iterable[TestArtifact] = (),
) -> RepairIntentBundle:
    return RepairIntentBundle(
        status="UNRESOLVED",
        summary="Intent grounding was inconclusive; use the original issue and validated reproducer.",
        execution_records=list(execution_records),
        tests=list(tests),
        unresolved=[reason],
    )
