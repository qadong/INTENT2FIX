from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Literal, Mapping

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from INTENT2FIX.completion.base import BaseCompletionModel
from INTENT2FIX.completion.schema import ResponseSchema
from INTENT2FIX.model_config import get_model_config


_EDIT_ACTIONS = {
    "AppendString",
    "ClaudeEditTool",
    "CreateFile",
    "InsertLine",
    "StringReplace",
}
_SEARCH_ACTIONS = {
    "EntityFunctionSemanticSearch",
    "EntityIntentSearch",
    "EntitySearchSymbol",
    "FindEntityUsage",
    "FindTemplateInstantiation",
    "GetMacroUsages",
    "RepositoryGraphQuery",
    "ViewCode",
}
_VERIFICATION_ACTIONS = {
    "RunIntentCheck",
    "RunRegression",
    "RunSingleTest",
    "RunTests",
}
_FINISH_ACTIONS = {"Finish", "VerifiedFinish"}
_ACTION_FIELDS: dict[str, tuple[str, ...]] = {
    "RunCommand": ("command", "capture_generated_changes"),
    "RunSingleTest": ("command", "test_path", "file_path"),
    "RunRegression": ("command", "test_path", "file_path"),
    "RunTests": ("command", "test_files"),
    "RunIntentCheck": ("check_id",),
    "ViewCode": ("path", "file_path", "line_start", "line_end"),
    "EntitySearchSymbol": ("name", "type", "namespace", "limit"),
    "EntityFunctionSemanticSearch": ("query", "namespace", "limit"),
    "EntityIntentSearch": ("query", "limit"),
    "FindEntityUsage": ("name", "type", "namespace", "limit"),
    "FindTemplateInstantiation": ("name", "type", "namespace", "limit"),
    "GetMacroUsages": ("name", "limit"),
    "RepositoryGraphQuery": ("entity_name", "relation", "direction", "limit"),
    "StringReplace": ("path", "file_path"),
    "CreateFile": ("path", "file_path", "artifact_kind"),
    "InsertLine": ("path", "file_path", "line_number"),
    "AppendString": ("path", "file_path"),
    "ClaudeEditTool": ("path", "file_path", "command"),
}
_INFRASTRUCTURE_MARKERS = (
    "api connection",
    "connection error",
    "container is not running",
    "docker daemon",
    "insufficient balance",
    "rate limit",
    "service unavailable",
    "timed out waiting for provider",
)
_FAILURE_MARKERS = (
    " error:",
    " failed",
    "failure",
    "fatal error:",
    "undefined reference",
    "unresolved external",
)
_SECRET_RE = re.compile(
    r"(?i)(api[_-]?key|authorization|bearer|password|secret|token)"
    r"(\s*[:=]\s*|\s+)[^\s,;]+"
)
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ABS_REPO_RE = re.compile(
    r"/(?:data|home|workspace|workspaces|repo|tmp)/[^\s:'\"]+/(?:"
    r"include|src|source|test|tests|doc|docs|cmake|CMake|single_include)/"
)
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_:.-]{2,}")
_DIAGNOSTIC_LINE_RE = re.compile(
    r"(?i)(fatal error:|error:|warning:|note:|undefined reference|"
    r"unresolved external)"
)
_LOCATION_RE = re.compile(r"(?<=\S):\d+(?::\d+)?(?=[:\s])")
_ADDRESS_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")
_VOLATILE_ID_RE = re.compile(
    r"\b(?:call|tmp|probe|repair)[_-]?[0-9a-f]{8,}\b", re.IGNORECASE
)
_DURATION_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:ms|milliseconds?|seconds?|secs?)\b", re.I
)
_FORBIDDEN_INVESTIGATION_RE = re.compile(
    r"(?i)(?:\bgit\s+(?:log|show)\b|\bupstream\s+(?:fix|patch|commit)\b|"
    r"\bfuture\s+(?:patch|commit)\b|\bpull\s+request\b|\bPR\b|"
    r"github\.com|\bweb\s+search\b|\binternet\b)"
)
_STOPWORDS = {
    "about",
    "after",
    "also",
    "because",
    "before",
    "being",
    "could",
    "does",
    "from",
    "have",
    "into",
    "issue",
    "should",
    "that",
    "their",
    "there",
    "these",
    "this",
    "using",
    "when",
    "which",
    "with",
    "would",
}


class EvidenceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    ordinal: int
    phase_id: int
    kind: Literal[
        "command_result",
        "compile_diagnostic",
        "search_observation",
        "verification",
    ]
    action_type: str
    action_summary: str = ""
    observation_excerpt: str = ""
    diagnostic_key: str = ""
    outcome: Literal["failed", "passed", "unknown"] = "unknown"
    occurrences: int = 1
    first_ordinal: int
    last_ordinal: int
    truncated: bool = False


class EvidenceQuote(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    evidence_id: str = Field(
        validation_alias=AliasChoices("evidence_id", "id"),
    )
    quote: str = Field(
        ...,
        validation_alias=AliasChoices("quote", "excerpt"),
        min_length=1,
        max_length=800,
    )


class AuditLead(BaseModel):
    """A question for further audit, not an intent clause or patch verdict."""

    model_config = ConfigDict(extra="forbid")

    evidence_quotes: list[EvidenceQuote] = Field(..., min_length=1, max_length=4)
    observed_fact: str = Field(..., min_length=1, max_length=1000)
    repair_assumption: str = Field(..., min_length=1, max_length=1000)
    unverified_public_behavior: str = Field(..., min_length=1, max_length=800)
    suggested_investigation: str = Field(..., min_length=1, max_length=800)
    issue_claim_span: str = Field(default="", max_length=1000)


class AuditLeadBatch(ResponseSchema):
    """Non-authoritative investigation leads grounded in Repair observations."""

    model_config = ConfigDict(extra="forbid")
    leads: list[AuditLead] = Field(default_factory=list, max_length=5)


class SliceStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes_seen: int = 0
    action_steps_seen: int = 0
    events_created: int = 0
    selected_events: int = 0
    aggregated_occurrences: int = 0
    corrections_skipped: int = 0
    infrastructure_failures_skipped: int = 0
    edit_observations_skipped: int = 0
    uninformative_failures_skipped: int = 0
    post_finish_steps_skipped: int = 0
    skipped_empty: int = 0
    skipped_unknown_shape: int = 0
    truncated_events: int = 0


class TrajectorySlice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instance_id: str = ""
    issue: str = ""
    patch: str = ""
    repair_finished: bool = False
    finish_ordinal: int | None = None
    events: list[EvidenceEvent] = Field(default_factory=list)
    selected_by_slice: dict[str, list[str]] = Field(default_factory=dict)
    stats: SliceStats = Field(default_factory=SliceStats)


class TrajectoryLeadResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trajectory_slice: TrajectorySlice
    leads: list[AuditLead] = Field(default_factory=list)
    extraction_status: Literal["not_requested", "completed", "empty", "failed"] = (
        "not_requested"
    )
    validation_errors: list[str] = Field(default_factory=list)


def _bounded(value: str, limit: int) -> tuple[str, bool]:
    value = value.strip()
    if len(value) <= limit:
        return value, False
    marker = "\n... [truncated] ...\n"
    payload = max(0, limit - len(marker))
    head = (payload + 1) // 2
    tail = payload // 2
    return value[:head] + marker + value[-tail:], True


def _sanitize(value: Any, limit: int) -> tuple[str, bool]:
    if value is None:
        return "", False
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=True, sort_keys=True)
    value = _ANSI_RE.sub("", value)
    value = _CONTROL_RE.sub("", value)
    value = _SECRET_RE.sub(lambda match: f"{match.group(1)}=<redacted>", value)
    value = _ABS_REPO_RE.sub("<REPO>/", value)
    return _bounded(value, limit)


def _stable_text(value: str) -> str:
    value = _LOCATION_RE.sub(":<loc>", value)
    value = _ADDRESS_RE.sub("<addr>", value)
    value = _VOLATILE_ID_RE.sub("<id>", value)
    value = _DURATION_RE.sub("<duration>", value)
    return re.sub(r"\s+", " ", value).strip()


def _action_type(action: Mapping[str, Any]) -> str:
    class_name = str(action.get("action_args_class", "")).rsplit(".", 1)[-1]
    if class_name.endswith("Args"):
        class_name = class_name[:-4]
    return class_name or str(action.get("action", action.get("name", "Unknown")))


def _safe_action_summary(action_type: str, action: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in _ACTION_FIELDS.get(action_type, ()):
        if key not in action or action[key] in (None, "", [], {}):
            continue
        value, _ = _sanitize(action[key], 600 if key == "command" else 300)
        parts.append(f"{key}={value}")
    return " ".join(parts) or action_type


def _exit_code(observation: Mapping[str, Any]) -> int | None:
    properties = observation.get("properties")
    if isinstance(properties, Mapping):
        value = properties.get("exit_code")
        if isinstance(value, int):
            return value
    message = str(observation.get("message", ""))
    match = re.search(r"exit code\s+(-?\d+)", message, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _diagnostic_fingerprint(text: str) -> str:
    lines = [
        line.strip() for line in text.splitlines() if _DIAGNOSTIC_LINE_RE.search(line)
    ]
    if not lines:
        return ""
    stable = [_stable_text(line) for line in lines[:8]]
    return hashlib.sha256("\n".join(stable).encode()).hexdigest()[:16]


def _looks_infrastructure_failure(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _INFRASTRUCTURE_MARKERS)


def _is_direct_compile(action_summary: str) -> bool:
    return bool(
        re.search(
            r"(?:^|command=)(?:[^\n]*\s)?(?:c\+\+|cc|clang\+\+|g\+\+|gcc)"
            r"(?:-\d+)?(?:\s|$)",
            action_summary,
        )
    )


def _classify_event(
    action_type: str,
    observation: Mapping[str, Any],
    action_summary: str,
    excerpt: str,
) -> tuple[str, str, str]:
    exit_code = _exit_code(observation)
    diagnostic_context = action_type in _VERIFICATION_ACTIONS or (
        action_type == "RunCommand" and _is_direct_compile(action_summary)
    )
    diagnostic_key = _diagnostic_fingerprint(excerpt) if diagnostic_context else ""
    lowered = excerpt.lower()
    if action_type in _VERIFICATION_ACTIONS:
        explicit_failure = any(
            marker in lowered
            for marker in ("log type: compilation_error", "log type: memory_leak")
        )
        if explicit_failure:
            outcome = "failed"
        elif exit_code is None:
            outcome = "unknown"
        else:
            outcome = "passed" if exit_code == 0 else "failed"
        return "verification", outcome, diagnostic_key
    if diagnostic_key:
        failed = exit_code not in (None, 0) or any(
            marker in lowered for marker in _FAILURE_MARKERS
        )
        return "compile_diagnostic", "failed" if failed else "unknown", diagnostic_key
    if action_type in _SEARCH_ACTIONS:
        return "search_observation", "passed", ""
    if exit_code is None:
        return "command_result", "unknown", ""
    return "command_result", "passed" if exit_code == 0 else "failed", ""


def _event_key(event: EvidenceEvent) -> tuple[str, ...]:
    summary = _stable_text(event.action_summary)
    observation = _stable_text(event.observation_excerpt)
    if event.action_type in _VERIFICATION_ACTIONS and "command=" in summary:
        summary = summary.split("|", 1)[0].strip()
    return (
        str(event.phase_id),
        event.action_type,
        "" if event.diagnostic_key else summary,
        event.diagnostic_key or hashlib.sha256(observation.encode()).hexdigest(),
        event.outcome,
    )


def _is_uninformative_command_failure(
    action_type: str,
    observation: Mapping[str, Any],
    excerpt: str,
) -> bool:
    if action_type != "RunCommand" or _exit_code(observation) in (None, 0):
        return False
    normalized = re.sub(r"\s+", " ", excerpt).strip().lower()
    return normalized in {
        "command failed with exit code 1. no output.",
        "command failed. no output.",
    }


def _edit_changes_production_state(action_type: str, action: Mapping[str, Any]) -> bool:
    if action_type not in _EDIT_ACTIONS:
        return False
    if action.get("artifact_kind") == "temporary_probe":
        return False
    path = str(action.get("path") or action.get("file_path") or "")
    if not path:
        return True
    name = PurePosixPath(path).name.lower()
    stem = PurePosixPath(path).stem.lower()
    return not (
        stem.startswith(("repro", "probe", "poc", "debug"))
        or name.startswith(("repro", "probe", "poc", "debug"))
    )


def _terms(value: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN_RE.findall(value)
        if token.lower() not in _STOPWORDS and not token.isdigit()
    }


def _patch_terms_and_paths(patch: str) -> tuple[set[str], set[str]]:
    paths: set[str] = set()
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                paths.add(parts[3].removeprefix("b/"))
    return _terms(patch), paths


def _units(value: str, *, patch: bool = False) -> list[set[str]]:
    if patch:
        parts = re.split(r"(?=^@@)|(?=^diff --git )", value, flags=re.MULTILINE)
    else:
        parts = re.split(r"(?:\n\s*\n|(?<=[.!?])\s+|\n[-*]\s+)", value)
    units: list[set[str]] = []
    for part in parts:
        terms = _terms(part)
        if terms and terms not in units:
            units.append(terms)
        if len(units) >= 12:
            break
    return units or ([_terms(value)] if value.strip() else [])


def _score_overlap(event: EvidenceEvent, terms: set[str]) -> int:
    event_terms = _terms(event.action_summary + "\n" + event.observation_excerpt)
    return len(event_terms & terms)


def _pick_ranked(
    events: list[EvidenceEvent],
    score: Callable[[EvidenceEvent], int],
    limit: int,
) -> list[EvidenceEvent]:
    ranked = sorted(
        events, key=lambda event: (score(event), -event.ordinal), reverse=True
    )
    return [event for event in ranked if score(event) > 0][:limit]


def _cover_units(
    events: list[EvidenceEvent],
    units: list[set[str]],
    limit: int,
    *,
    bonus: Callable[[EvidenceEvent], int] | None = None,
) -> list[EvidenceEvent]:
    selected: dict[str, EvidenceEvent] = {}
    for unit in units:
        ranked = _pick_ranked(
            events,
            lambda event, terms=unit: _score_overlap(event, terms)
            + (bonus(event) if bonus else 0),
            1,
        )
        if ranked:
            selected.setdefault(ranked[0].id, ranked[0])
        if len(selected) >= limit:
            return list(selected.values())
    all_terms = set().union(*units) if units else set()
    for event in _pick_ranked(
        events,
        lambda item: _score_overlap(item, all_terms) + (bonus(item) if bonus else 0),
        limit,
    ):
        selected.setdefault(event.id, event)
        if len(selected) >= limit:
            break
    return list(selected.values())


def _select_events(
    events: list[EvidenceEvent],
    issue: str,
    patch: str,
    *,
    issue_budget: int,
    patch_budget: int,
    counter_budget: int,
    verification_budget: int,
    max_events: int,
) -> tuple[list[EvidenceEvent], dict[str, list[str]]]:
    issue_terms = _terms(issue)
    patch_terms, patch_paths = _patch_terms_and_paths(patch)
    issue_slice = _cover_units(events, _units(issue), issue_budget)
    patch_slice = _cover_units(
        events,
        _units(patch, patch=True),
        patch_budget,
        bonus=lambda event: 8
        * any(
            path in event.action_summary or path in event.observation_excerpt
            for path in patch_paths
        ),
    )

    def counter_score(event: EvidenceEvent) -> int:
        score = 10 if event.kind == "compile_diagnostic" else 0
        if event.outcome == "failed":
            score += 7
        issue_overlap = _score_overlap(event, issue_terms)
        patch_overlap = _score_overlap(event, patch_terms)
        if issue_overlap > patch_overlap:
            score += min(issue_overlap - patch_overlap, 6)
        return score

    counter_slice = _pick_ranked(events, counter_score, counter_budget)
    verification_slice = sorted(
        [event for event in events if event.kind == "verification"],
        key=lambda event: (event.outcome in {"failed", "unknown"}, event.ordinal),
        reverse=True,
    )[:verification_budget]
    by_name = {
        "issue": issue_slice,
        "patch": patch_slice,
        "counter": counter_slice,
        "verification": verification_slice,
    }
    selected: dict[str, EvidenceEvent] = {}
    offsets = defaultdict(int)
    while len(selected) < max_events:
        progressed = False
        for name in ("counter", "verification", "issue", "patch"):
            items = by_name[name]
            index = offsets[name]
            if index >= len(items):
                continue
            offsets[name] += 1
            selected.setdefault(items[index].id, items[index])
            progressed = True
            if len(selected) >= max_events:
                break
        if not progressed:
            break
    return (
        sorted(selected.values(), key=lambda event: event.ordinal),
        {name: [event.id for event in items] for name, items in by_name.items()},
    )


def slice_trajectory(
    trajectory: Mapping[str, Any],
    patch: str,
    *,
    max_events: int = 16,
    issue_budget: int = 6,
    patch_budget: int = 5,
    counter_budget: int = 4,
    verification_budget: int = 3,
    event_chars: int = 1000,
) -> TrajectorySlice:
    metadata = trajectory.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    issue = str(metadata.get("problem_statement", ""))
    instance_id = str(metadata.get("instance_id", ""))
    nodes = trajectory.get("nodes")
    if not isinstance(nodes, list):
        nodes = []

    stats = SliceStats(nodes_seen=len(nodes))
    events: list[EvidenceEvent] = []
    aggregate_index: dict[tuple[str, ...], int] = {}
    phase_id = 0
    ordinal = 0
    repair_finished = False
    finish_ordinal: int | None = None

    for node_index, node in enumerate(nodes):
        if repair_finished:
            action_steps = (
                node.get("action_steps", []) if isinstance(node, Mapping) else []
            )
            if isinstance(action_steps, list):
                stats.post_finish_steps_skipped += len(action_steps)
            continue
        if not isinstance(node, Mapping):
            stats.skipped_unknown_shape += 1
            continue
        action_steps = node.get("action_steps")
        if not isinstance(action_steps, list):
            stats.skipped_unknown_shape += 1
            continue
        for step_index, step in enumerate(action_steps):
            stats.action_steps_seen += 1
            ordinal += 1
            if not isinstance(step, Mapping):
                stats.skipped_unknown_shape += 1
                continue
            action = step.get("action")
            observation = step.get("observation")
            if not isinstance(action, Mapping) or not isinstance(observation, Mapping):
                stats.skipped_unknown_shape += 1
                continue
            action_type = _action_type(action)
            if action_type in _FINISH_ACTIONS:
                repair_finished = True
                finish_ordinal = ordinal
                stats.post_finish_steps_skipped += len(action_steps) - step_index - 1
                break
            if bool(observation.get("expect_correction")):
                stats.corrections_skipped += 1
                continue
            summary = _safe_action_summary(action_type, action)
            excerpt, truncated = _sanitize(observation.get("message", ""), event_chars)
            if action_type in _EDIT_ACTIONS:
                stats.edit_observations_skipped += 1
                if _edit_changes_production_state(action_type, action):
                    phase_id += 1
                continue
            if _looks_infrastructure_failure(excerpt):
                stats.infrastructure_failures_skipped += 1
                continue
            if _is_uninformative_command_failure(action_type, observation, excerpt):
                stats.uninformative_failures_skipped += 1
                continue
            if not excerpt and action_type not in _VERIFICATION_ACTIONS:
                stats.skipped_empty += 1
                continue
            kind, outcome, diagnostic_key = _classify_event(
                action_type, observation, summary, excerpt
            )
            event = EvidenceEvent(
                id=f"EV-{ordinal}",
                ordinal=ordinal,
                phase_id=phase_id,
                kind=kind,
                action_type=action_type,
                action_summary=summary,
                observation_excerpt=excerpt,
                diagnostic_key=diagnostic_key,
                outcome=outcome,
                first_ordinal=ordinal,
                last_ordinal=ordinal,
                truncated=truncated,
            )
            key = _event_key(event)
            existing_index = aggregate_index.get(key)
            if existing_index is None:
                aggregate_index[key] = len(events)
                events.append(event)
            else:
                existing = events[existing_index]
                existing.occurrences += 1
                existing.last_ordinal = ordinal
                stats.aggregated_occurrences += 1
            if truncated:
                stats.truncated_events += 1

    stats.events_created = len(events)
    selected, selected_by_slice = _select_events(
        events,
        issue,
        patch,
        issue_budget=issue_budget,
        patch_budget=patch_budget,
        counter_budget=counter_budget,
        verification_budget=verification_budget,
        max_events=max_events,
    )
    stats.selected_events = len(selected)
    bounded_patch, _ = _sanitize(patch, 12000)
    return TrajectorySlice(
        instance_id=instance_id,
        issue=issue,
        patch=bounded_patch,
        repair_finished=repair_finished,
        finish_ordinal=finish_ordinal,
        events=selected,
        selected_by_slice=selected_by_slice,
        stats=stats,
    )


_EXTRACTOR_PROMPT = """You extract non-authoritative investigation leads for a later Patch Intent Auditor.

The input contains the Issue, an untrusted candidate patch, and selected action/observation evidence from the Repair trajectory before its first completion claim. The trajectory is noisy. Treat the patch and Repair choices only as hypotheses. Existing implementation and tests describe the old repository; they are not automatically desired behavior.

Return a lead only when the observations expose a concrete assumption or a public behavior that the Repair did not settle. For each lead:
- quote exact contiguous text from one or more EvidenceEvent observation excerpts;
- state only the fact established by those quotes;
- describe the Repair assumption as tentative;
- identify one externally observable behavior still needing investigation;
- suggest a focused repository search, public probe, or comparison that could resolve it;
- copy an exact Issue span only when the Issue itself motivates that investigation, otherwise leave issue_claim_span empty.

The packet is a selected slice, not the full trajectory. Never claim that something was not searched, tested, or observed merely because it is absent. Say "the selected evidence does not establish ...". Work only from the checked-out repository. Never suggest upstream/future patches, git history, pull requests, network access, or external search.

Use this citation shape exactly:
{"evidence_quotes": [{"evidence_id": "EV-4", "quote": "exact contiguous observation text"}]}

Do not state the true intent, approve or reject the patch, prescribe a patch, or turn a local witness into a global requirement. Failed commands and compiler diagnostics may falsify an assumption but cannot define desired behavior. Return an empty list when no useful unresolved behavior is grounded. Output at most five independent leads."""


def _completion_model(model_name: str) -> BaseCompletionModel:
    config = get_model_config(model_name).copy()
    for key in (
        "compact_native_prompt",
        "message_history_max_tokens",
        "few_shot_examples",
        "message_history_type",
    ):
        config.pop(key, None)
    max_tokens = os.getenv("COTEST_AUDIT_LEAD_MAX_TOKENS")
    if max_tokens:
        config["max_tokens"] = int(max_tokens)
    if "glm-5.3-flash" in str(config.get("model", "")):
        extra_body = dict(config.get("extra_body") or {})
        extra_body["reasoning_effort"] = os.getenv(
            "COTEST_AUDIT_LEAD_REASONING_EFFORT", "low"
        )
        config["extra_body"] = extra_body
    completion = BaseCompletionModel.create(**config)
    completion.initialize(AuditLeadBatch, _EXTRACTOR_PROMPT)
    return completion


def _one_completion(
    trajectory_slice: TrajectorySlice, model_name: str
) -> AuditLeadBatch:
    completion = _completion_model(model_name)
    packet = {
        "issue": trajectory_slice.issue,
        "candidate_patch_hypothesis": trajectory_slice.patch,
        "evidence_events": [event.model_dump() for event in trajectory_slice.events],
    }
    messages = [{"role": "user", "content": json.dumps(packet, ensure_ascii=False)}]
    prepared = completion._prepare_messages(messages, completion.system_prompt or "")
    raw = completion._execute_completion(prepared)
    outputs, _, _ = completion._validate_completion(raw)
    if not outputs or not isinstance(outputs[0], AuditLeadBatch):
        raise ValueError("AuditLead extractor returned no structured output")
    return outputs[0]


def validate_audit_leads(
    batch: AuditLeadBatch,
    trajectory_slice: TrajectorySlice,
) -> tuple[list[AuditLead], list[str]]:
    events = {event.id: event for event in trajectory_slice.events}
    valid: list[AuditLead] = []
    errors: list[str] = []
    for index, lead in enumerate(batch.leads, start=1):
        prefix = f"lead {index}"
        citations: set[tuple[str, str]] = set()
        quote_error = ""
        for quote in lead.evidence_quotes:
            event = events.get(quote.evidence_id)
            if event is None:
                quote_error = f"unknown evidence ID {quote.evidence_id}"
                break
            canonical = _canonical_quote(event.observation_excerpt, quote.quote)
            if canonical is None:
                quote_error = f"quote is not exact evidence from {quote.evidence_id}"
                break
            quote.quote = canonical
            citation = (quote.evidence_id, canonical)
            if citation in citations:
                quote_error = f"duplicate evidence citation from {quote.evidence_id}"
                break
            citations.add(citation)
        if quote_error:
            errors.append(f"{prefix}: {quote_error}")
            continue
        if _FORBIDDEN_INVESTIGATION_RE.search(lead.suggested_investigation):
            errors.append(
                f"{prefix}: suggested investigation uses forbidden future evidence"
            )
            continue
        if (
            lead.issue_claim_span
            and lead.issue_claim_span not in trajectory_slice.issue
        ):
            errors.append(f"{prefix}: issue_claim_span is not an exact Issue span")
            continue
        valid.append(lead)
    return valid, errors


def _canonical_quote(observation: str, quote: str) -> str | None:
    if quote in observation:
        return quote
    parts = [part for part in re.split(r"\s+", quote.strip()) if part]
    if not parts:
        return None
    match = re.search(r"\s+".join(re.escape(part) for part in parts), observation)
    return match.group(0) if match else None


def extract_trajectory_leads(
    trajectory_slice: TrajectorySlice,
    model_name: str,
) -> TrajectoryLeadResult:
    if not trajectory_slice.repair_finished:
        return TrajectoryLeadResult(
            trajectory_slice=trajectory_slice,
            extraction_status="empty",
            validation_errors=[
                "Repair trajectory has no Finish or VerifiedFinish boundary."
            ],
        )
    if not trajectory_slice.patch.strip():
        return TrajectoryLeadResult(
            trajectory_slice=trajectory_slice,
            extraction_status="empty",
            validation_errors=["No candidate patch is available to audit."],
        )
    if not trajectory_slice.events:
        return TrajectoryLeadResult(
            trajectory_slice=trajectory_slice,
            extraction_status="empty",
            validation_errors=["No admissible trajectory evidence was selected."],
        )
    try:
        batch = _one_completion(trajectory_slice, model_name)
        leads, errors = validate_audit_leads(batch, trajectory_slice)
        return TrajectoryLeadResult(
            trajectory_slice=trajectory_slice,
            leads=leads,
            extraction_status="completed" if leads else "empty",
            validation_errors=errors,
        )
    except Exception as exc:
        return TrajectoryLeadResult(
            trajectory_slice=trajectory_slice,
            extraction_status="failed",
            validation_errors=[f"{type(exc).__name__}: {exc}"],
        )


def load_trajectory(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("trajectory JSON must contain a top-level object")
    return data


def load_patch(path: str | Path | None) -> str:
    if path is None:
        return ""
    return Path(path).read_text(encoding="utf-8")
