from __future__ import annotations

import hashlib
import json
import logging
import re
import shlex
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Callable, ClassVar, Literal, Type

from pydantic import ConfigDict, Field, PrivateAttr, model_validator

from INTENT2FIX.actions.action import Action
from INTENT2FIX.actions.command_policy import (
    future_data_command_rejection,
    verification_exit_status_rejection,
)
from INTENT2FIX.actions.schema import ActionArguments, Observation
from INTENT2FIX.agent.agent import ActionAgent
from INTENT2FIX.completion.base import BaseCompletionModel, LLMResponseFormat
from INTENT2FIX.completion.schema import ResponseSchema
from INTENT2FIX.exceptions import RejectError
from INTENT2FIX.file_context import FileContext
from INTENT2FIX.index import CodeIndex
from INTENT2FIX.intent_dialogue.prompts import (
    build_intent_boundary_test_task,
    build_intent_boundary_test_refinement_task,
    build_contrastive_investigation_task,
    build_poc_investigation_task,
    build_contrastive_refinement_task,
    build_evidence_sufficiency_refinement_task,
    build_next_intent_witness_task,
    build_developer_followup_task,
    build_developer_task,
    build_repository_investigation_task,
)
from INTENT2FIX.intent_dialogue.dynamic_evidence import (
    CollectCoverageContrast,
    CollectXRayTraceContrast,
    DynamicEvidenceUnavailable,
    _resolve_probe_run,
    available_dynamic_evidence_tools,
)
from INTENT2FIX.intent_dialogue.schema import (
    INVESTIGATION_REPORT_SUMMARY_MAX_CHARS,
    CandidateRepairBoundary,
    ContrastiveIntentDecision,
    ContrastiveWitnessProposal,
    DeveloperAnswer,
    DeveloperAcceptanceCheck,
    EvidenceSufficiencyReview,
    IntentExplorationObjective,
    IntentBoundaryTestAssessment,
    IntentInvestigationPlan,
    IntentValidationRouting,
    InvestigationInputPathCheck,
    InvestigationPath,
    InvestigationTaskFinding,
    RepairIntentContract,
    RepositoryEvidence,
    TestAnchorRetrieval,
    TestAnchorSelection,
)
from INTENT2FIX.intent_dialogue.test_inventory import RepositoryTestInventory
from INTENT2FIX.intent_dialogue.test_retrieval import TestAnchorRetriever
from INTENT2FIX.message_history import MessageHistoryGenerator
from INTENT2FIX.model_config import get_model_config
from INTENT2FIX.node import Node
from INTENT2FIX.repository.repository import Repository
from INTENT2FIX.reproduce.intent_schema import IssueEvidenceFrame, IssueIntentUnit
from INTENT2FIX.runtime.dockerenv import DockerEnv
from INTENT2FIX.schema import MessageHistoryType


logger = logging.getLogger(__name__)


_ALLOWED_COMMANDS = {
    "cat",
    "grep",
    "head",
    "ls",
    "pwd",
    "sed",
    "tail",
    "wc",
}
_ALLOWED_SEPARATORS = {"&&", "|"}
_SHELL_PUNCTUATION = "|&;<>()"
_NETWORK_COMMANDS = {
    "curl",
    "ftp",
    "nc",
    "netcat",
    "scp",
    "sftp",
    "ssh",
    "wget",
}
_PROBE_COMPILERS = {"c++", "cc", "clang++", "clang", "g++", "gcc"}


def _command_segments(command: str) -> tuple[list[list[str]], str]:
    lexer = shlex.shlex(
        command,
        posix=True,
        punctuation_chars=_SHELL_PUNCTUATION,
    )
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        shell_tokens = list(lexer)
    except ValueError:
        return [], (
            "The search command has invalid shell quoting. Use one simple "
            "command and single-quote the grep pattern."
        )

    segments: list[list[str]] = [[]]
    for token in shell_tokens:
        if token in _ALLOWED_SEPARATORS:
            if not segments[-1]:
                return [], "Empty command segments are not allowed."
            segments.append([])
            continue
        if token and all(char in _SHELL_PUNCTUATION for char in token):
            return [], (
                "Shell redirection, background execution, grouping, and `||` "
                "are not allowed. Quote `<`, `>`, or `|` when they are part "
                "of a grep pattern."
            )
        segments[-1].append(token)

    if not segments[-1]:
        return [], "Empty command segments are not allowed."
    return segments, ""


def read_only_command_rejection(command: str, work_dir: str) -> str:
    if not command.strip():
        return "The search command cannot be empty."
    future_data_rejection = future_data_command_rejection(command)
    if future_data_rejection:
        return future_data_rejection
    if any(token in command for token in ("\n", "\r", "`", "$(")):
        return "Only read-only command pipelines using `|` or `&&` are allowed."
    if ".git" in command or re.search(r"(^|\s)git(?:\s|$)", command):
        return "Git metadata and history commands are not available to the Repository Investigator."
    if re.search(r"(^|\s)(?:sudo|su|env|sh|bash|zsh|python|perl|ruby)(?:\s|$)", command):
        return "Shells, interpreters, and privilege-changing commands are not allowed."

    segments, rejection = _command_segments(command)
    if rejection:
        return rejection

    for tokens in segments:
        executable = tokens[0].rsplit("/", 1)[-1]
        if executable not in _ALLOWED_COMMANDS:
            return (
                f"`{executable}` is not a permitted read-only repository command. "
                f"Allowed commands: {', '.join(sorted(_ALLOWED_COMMANDS))}."
            )
        if executable == "sed":
            if "-i" in tokens or "--in-place" in tokens:
                return "In-place sed editing is not allowed."
            if "-n" not in tokens:
                return "sed is restricted to explicit `-n` read ranges."
        if executable == "grep":
            recursive = any(
                token == "--recursive"
                or (
                    token.startswith("-")
                    and not token.startswith("--")
                    and any(flag in token[1:] for flag in ("r", "R"))
                )
                for token in tokens[1:]
            )
            if recursive and any(
                token in {".", work_dir, work_dir.rstrip("/") + "/"}
                for token in tokens[1:]
            ):
                return (
                    "Recursive grep must name source directories explicitly "
                    "instead of searching the repository root."
                )
        if executable == "ls" and any(
            token.startswith("-") and "a" in token[1:]
            for token in tokens[1:]
        ):
            return "ls options that expose hidden metadata are not allowed."
        for token in tokens[1:]:
            if token.startswith("-"):
                continue
            if ".." in token.split("/"):
                return "Parent-directory paths are not allowed."
            if token.startswith("/") and not (
                token == work_dir or token.startswith(work_dir.rstrip("/") + "/")
            ):
                return "Absolute paths outside the repository work directory are not allowed."
    return ""


def investigation_command_rejection(command: str) -> str:
    if not command.strip():
        return "The investigation command cannot be empty."
    future_data_rejection = future_data_command_rejection(command)
    if future_data_rejection:
        return future_data_rejection
    if ".git" in command or re.search(r"(^|\s)git(?:\s|$)", command):
        return "Git metadata and history are not available to the Repository Investigator."
    if any(
        re.search(rf"(^|\s){re.escape(executable)}(?:\s|$)", command)
        for executable in _NETWORK_COMMANDS
    ):
        return "Network commands are not available to the Repository Investigator."
    if re.search(r"/home/(?:fix|test)\.patch", command):
        return "Benchmark fix and test patches are not available to the Repository Investigator."
    if re.search(r"(^|[;&|]\s*|\s)(?:rm|unlink|shred)(?:\s|$)", command):
        return (
            "Do not delete investigation files. Refine probes with "
            "ReplaceInvestigationText; disposable binaries may be overwritten "
            "by the next compile command."
        )
    return ""


def _validated_relative_path(path: str) -> tuple[str, str]:
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        return "", "Use a repository-relative path without parent traversal."
    if not candidate.parts or ".git" in candidate.parts:
        return "", "Git metadata paths are not available."
    return str(candidate), ""


def _referenced_probe_files(
    command: str,
    probe_files: dict[str, str],
) -> dict[str, str]:
    try:
        tokens = {
            token.removeprefix("./") for token in shlex.split(command)
        }
    except ValueError:
        return {}
    return {
        path: content
        for path, content in probe_files.items()
        if path in tokens
    }


def frozen_probe_command_rejection(
    command: str,
    probe_files: dict[str, str],
) -> str:
    """Accept only a standalone compile and optional run of frozen probe files."""
    if "$(" in command or "`" in command:
        return (
            "Command substitution (`$()` or backticks) is not freezeable; it can "
            "change the compiled source set when replayed after Repair."
        )
    if ";" in command or "||" in command:
        return (
            "`;` and `||` are not freezeable because a later command can mask the "
            "compiler or probe exit status; use at most `&& /tmp/binary`."
        )
    segments, rejection = _command_segments(command)
    if rejection:
        return rejection
    if not segments:
        return "A frozen acceptance probe requires a compile command."

    compiler = segments[0][0].rsplit("/", 1)[-1]
    if compiler not in _PROBE_COMPILERS:
        return "A frozen acceptance probe must start with a C/C++ compiler."

    compile_tokens = segments[0]
    output_path = ""
    for index, token in enumerate(compile_tokens):
        if token == "-o" and index + 1 < len(compile_tokens):
            output_path = compile_tokens[index + 1]
            break
        if token.startswith("-o") and len(token) > 2:
            output_path = token[2:]
            break
    if not output_path.startswith("/tmp/"):
        return "Frozen probe binaries must be written under /tmp with -o."
    allowed_tmp_tokens = {output_path, f"-o{output_path}"}
    if any(
        "/tmp/" in token and token not in allowed_tmp_tokens
        for token in compile_tokens[1:]
    ):
        return (
            "Frozen probes cannot depend on /tmp build artifacts, libraries, "
            "headers, or source files; only the output binary may use /tmp."
        )

    referenced_files = _referenced_probe_files(command, probe_files)
    if not referenced_files:
        return "The compile command must reference a WriteInvestigationFile probe."

    if len(segments) > 2:
        return "A frozen probe may contain only one compile and one run command."
    if len(segments) == 2 and segments[1][0] != output_path:
        return "The second frozen-probe command must run the compiled /tmp binary."
    return ""


def _sed_read_range(command: str) -> tuple[str, int, int] | None:
    match = re.fullmatch(
        r"\s*sed\s+-n\s+['\"]?(\d+),(\d+)p['\"]?\s+([^\s|&;]+)\s*",
        command,
    )
    if match is None:
        return None
    start, end = int(match.group(1)), int(match.group(2))
    if end < start:
        return None
    return match.group(3), start, end


_CPP_SOURCE_SUFFIXES = frozenset({
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx",
    ".inc", ".inl", ".ipp", ".tpp",
})
_NON_PRODUCTION_SOURCE_PARTS = frozenset({
    "benchmark", "benchmarks", "example", "examples", "fixture", "fixtures",
    "probe", "probes", "test", "tests", "testing",
})


def _is_concrete_cpp_source_evidence(evidence: RepositoryEvidence | None) -> bool:
    """Return whether a SEARCH observation contains an actual C/C++ source read."""
    if evidence is None or not evidence.observation.strip():
        return False
    read_range = _sed_read_range(evidence.locator)
    return bool(
        read_range is not None
        and PurePosixPath(read_range[0].strip("'\"")).suffix.lower()
        in _CPP_SOURCE_SUFFIXES
    )


def _cpp_source_evidence_paths(evidence: RepositoryEvidence | None) -> list[str]:
    if evidence is None:
        return []
    read_range = _sed_read_range(evidence.locator)
    if read_range is not None:
        return [read_range[0].strip("'\"")]
    return list(dict.fromkeys(
        match.group(1)
        for match in re.finditer(r"(?m)^([^\s:\n]+):(\d+):", evidence.observation)
        if PurePosixPath(match.group(1)).suffix.lower() in _CPP_SOURCE_SUFFIXES
    ))


def _is_concrete_production_cpp_source_evidence(
    evidence: RepositoryEvidence | None,
) -> bool:
    if not _is_concrete_cpp_source_evidence(evidence):
        return False
    return any(
        not ({part.lower() for part in PurePosixPath(path).parts}
             & _NON_PRODUCTION_SOURCE_PARTS)
        for path in _cpp_source_evidence_paths(evidence)
    )


def _grep_source_followup(command: str, output: str) -> dict | None:
    """Suggest one exact source read after a C/C++ grep match listing."""
    segments, rejection = _command_segments(command)
    if rejection or not segments or segments[0][0].rsplit("/", 1)[-1] != "grep":
        return None
    explicit_paths = [
        token.strip("'\"") for token in segments[0][1:]
        if PurePosixPath(token.strip("'\"")).suffix.lower()
        in _CPP_SOURCE_SUFFIXES
    ]
    path = ""
    line = 0
    located = re.search(r"(?m)^([^\s:\n]+):(\d+):", output)
    if located is not None:
        candidate = located.group(1)
        if PurePosixPath(candidate).suffix.lower() in _CPP_SOURCE_SUFFIXES:
            path, line = candidate, int(located.group(2))
    elif len(explicit_paths) == 1:
        line_match = re.search(r"(?m)^(\d+):", output)
        if line_match is not None:
            path, line = explicit_paths[0], int(line_match.group(1))
    if not path or line < 1:
        return None
    start, end = max(1, line - 5), line + 25
    return {
        "path": path,
        "line": line,
        "command": f"sed -n '{start},{end}p' {path}",
    }


def _best_exact_issue_excerpt(
    issue_text: str,
    relevance_text: str,
    *,
    max_chars: int = 800,
) -> str:
    """Choose a short verbatim Issue excerpt without asking Scout to resubmit."""
    normalized = issue_text.replace("\r\n", "\n")
    candidates = []
    for block in re.split(r"\n\s*\n", normalized):
        block = block.strip()
        if block and len(block) <= max_chars:
            candidates.append(block)
        candidates.extend(
            line.strip() for line in block.splitlines()
            if line.strip() and len(line.strip()) <= max_chars
        )
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        return ""
    relevance_tokens = set(re.findall(
        r"[A-Za-z_][A-Za-z0-9_]*", relevance_text.lower(),
    ))

    def score(candidate: str) -> tuple[int, int, int]:
        candidate_tokens = set(re.findall(
            r"[A-Za-z_][A-Za-z0-9_]*", candidate.lower(),
        ))
        return (
            len(relevance_tokens & candidate_tokens),
            int(any(char in candidate for char in ("::", ".", "(", ")"))),
            -len(candidate),
        )

    return max(candidates, key=score)


def _exact_observation_quote_candidates(
    observation: str,
    relevance_text: str,
    *,
    max_chars: int = 800,
    max_candidates: int = 12,
) -> list[str]:
    """Rank exact contiguous source spans without letting an LLM rewrite them."""
    normalized = observation.replace("\r\n", "\n")
    lines = normalized.splitlines(keepends=True)
    candidates: list[str] = []
    for width in (1, 2, 3):
        for index in range(len(lines) - width + 1):
            candidate = "".join(lines[index:index + width]).rstrip("\n")
            if candidate.strip() and len(candidate) <= max_chars:
                candidates.append(candidate)
    candidates = list(dict.fromkeys(candidates))
    relevance_tokens = set(re.findall(
        r"[A-Za-z_][A-Za-z0-9_]*", relevance_text.lower(),
    ))

    def score(candidate: str) -> tuple[int, int, int, int]:
        candidate_tokens = set(re.findall(
            r"[A-Za-z_][A-Za-z0-9_]*", candidate.lower(),
        ))
        return (
            len(relevance_tokens & candidate_tokens),
            int(bool(re.search(
                r"(?:if\s*\(|requires\b|enable_if|is_[A-Za-z_]|concept\b|static_assert)",
                candidate,
            ))),
            int(any(char in candidate for char in ("<", ">", "(", ")", "!"))),
            -len(candidate),
        )

    candidates.sort(key=score, reverse=True)
    return candidates[:max_candidates]


def _bounded_output(output: str, *, max_lines: int = 120) -> str:
    lines = output.splitlines()
    if len(lines) > max_lines:
        half = max_lines // 2
        lines = (
            lines[:half]
            + [f"... [{len(lines) - max_lines} lines omitted] ..."]
            + lines[-half:]
        )
    bounded = "\n".join(lines)
    if len(bounded) > 12000:
        bounded = (
            bounded[:6000]
            + "\n... [output truncated] ...\n"
            + bounded[-6000:]
        )
    return bounded


@dataclass
class InvestigationSession:
    require_input_path_check: bool = False
    conservatively_refine_report_paths: bool = False
    conservatively_refine_report_protocol: bool = False
    issue_text: str = ""
    report_quote_selector: Callable[[str, str, list[str]], str] | None = field(
        default=None,
        repr=False,
    )
    report_quote_refine_trace: dict = field(default_factory=dict)
    evidence: list[RepositoryEvidence] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)
    command_evidence: dict[str, str] = field(default_factory=dict)
    output_evidence: dict[str, str] = field(default_factory=dict)
    read_ranges: dict[str, list[tuple[int, int, str]]] = field(
        default_factory=dict
    )
    investigation_files: dict[str, str] = field(default_factory=dict)
    planned_task_ids: list[str] = field(default_factory=list)
    planned_hypothesis_ids: dict[str, set[str]] = field(default_factory=dict)
    live_investigation_paths: list[dict] = field(default_factory=list)
    latest_report_candidate: dict | None = None

    def reset(self) -> None:
        self.issue_text = ""
        self.evidence.clear()
        self.actions.clear()
        self.command_evidence.clear()
        self.output_evidence.clear()
        self.read_ranges.clear()
        self.investigation_files.clear()
        self.planned_task_ids.clear()
        self.planned_hypothesis_ids.clear()
        self.live_investigation_paths.clear()
        self.report_quote_refine_trace.clear()
        self.latest_report_candidate = None

    def open_investigation_path(
        self,
        *,
        question: str,
        role: str,
        evidence_ids: list[str],
        missing_evidence: str,
        decision: str,
        graph_evidence_id: str = "",
    ) -> dict:
        """Record a bounded live relation question before the final report."""
        path_id = f"PATH-{len(self.live_investigation_paths) + 1}"
        cited_ids = list(dict.fromkeys([
            *evidence_ids,
            *([graph_evidence_id] if graph_evidence_id else []),
        ]))
        path = {
            "path_id": path_id,
            "role": role,
            "question": question,
            "finding": (
                "A focused graph trace was collected; inspect its verified definitions "
                "before deciding whether this relation changes the cause or scope."
                if decision == "traced"
                else "No focused graph trace was justified at this checkpoint."
            ),
            "status": "unresolved",
            "evidence_ids": cited_ids,
            "verification_kind": "source",
            "source_evidence_id": "",
            "source_quote": "",
            "missing_evidence": missing_evidence,
            "checkpoint_decision": decision,
        }
        self.live_investigation_paths.append(path)
        return path

    def record(
        self,
        command: str,
        output: str,
        exit_code: int,
        *,
        prefix: str = "SEARCH",
        path: str = "<container>",
        action: str = "ReadOnlySearch",
        extra: dict | None = None,
    ) -> RepositoryEvidence:
        prefix_count = sum(
            item.id.startswith(prefix + "-") for item in self.evidence
        )
        evidence = RepositoryEvidence(
            id=f"{prefix}-{prefix_count + 1}",
            path=path,
            locator=command,
            observation=output,
        )
        self.evidence.append(evidence)
        action_record = {
            "action": action,
            "command": command,
            "exit_code": exit_code,
            "evidence_id": evidence.id,
        }
        if extra:
            action_record.update(extra)
        self.actions.append(action_record)
        return evidence

    def duplicate_command(self, command: str) -> str | None:
        try:
            fingerprint = " ".join(shlex.split(command))
        except ValueError:
            fingerprint = command.strip()
        return self.command_evidence.get(fingerprint)

    def register_command(self, command: str, evidence_id: str) -> None:
        try:
            fingerprint = " ".join(shlex.split(command))
        except ValueError:
            fingerprint = command.strip()
        self.command_evidence[fingerprint] = evidence_id

    def duplicate_probe_command(self, command: str) -> str | None:
        return self.command_evidence.get(self._probe_command_fingerprint(command))

    def register_probe_command(self, command: str, evidence_id: str) -> None:
        self.command_evidence[self._probe_command_fingerprint(command)] = evidence_id

    def _probe_command_fingerprint(self, command: str) -> str:
        try:
            normalized = " ".join(shlex.split(command))
        except ValueError:
            normalized = command.strip()
        probe_state = json.dumps(
            self.investigation_files,
            sort_keys=True,
            ensure_ascii=False,
        )
        state_digest = hashlib.sha256(probe_state.encode()).hexdigest()
        return f"probe:{state_digest}:{normalized}"

    def overlapping_read(self, command: str) -> str | None:
        read_range = _sed_read_range(command)
        if read_range is None:
            return None
        path, start, end = read_range
        length = end - start + 1
        for old_start, old_end, evidence_id in self.read_ranges.get(path, []):
            overlap = max(0, min(end, old_end) - max(start, old_start) + 1)
            if overlap / length >= 0.6:
                return evidence_id
        return None

    def register_read(self, command: str, evidence_id: str) -> None:
        read_range = _sed_read_range(command)
        if read_range is None:
            return
        path, start, end = read_range
        self.read_ranges.setdefault(path, []).append(
            (start, end, evidence_id)
        )

    def duplicate_output(self, output: str) -> str | None:
        if not output.strip():
            return None
        digest = hashlib.sha256(output.encode()).hexdigest()
        return self.output_evidence.get(digest)

    def register_output(self, output: str, evidence_id: str) -> None:
        if output.strip():
            digest = hashlib.sha256(output.encode()).hexdigest()
            self.output_evidence[digest] = evidence_id


class ReadOnlySearchArgs(ActionArguments):
    target: str = Field(
        default="",
        description="Public behavior or repository convention being inspected.",
    )
    purpose: str = Field(
        default="",
        description="How this result supplies evidence for an intent TODO.",
    )
    command: str = Field(
        ...,
        description=(
            "A read-only repository inspection command. Ripgrep (rg) is not "
            "installed and commands already run at the repository root, so do "
            "not use cd. Prefer simple fixed-string searches such as "
            "`grep -RInF -- 'literal' include src tests`. Use one search per "
            "call; grep, cat, sed -n, head, tail, ls, pwd, and wc are allowed."
        ),
    )

    model_config = ConfigDict(title="ReadOnlySearch")


class ReadOnlySearch(Action):
    args_schema: ClassVar[Type[ActionArguments]] = ReadOnlySearchArgs
    _docker_env: DockerEnv = PrivateAttr()
    _session: InvestigationSession = PrivateAttr()

    def __init__(
        self,
        docker_env: DockerEnv,
        session: InvestigationSession,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._docker_env = docker_env
        self._session = session

    def execute(self, args: ReadOnlySearchArgs, **kwargs) -> Observation:
        rejection = read_only_command_rejection(
            args.command,
            self._docker_env.work_dir,
        )
        if rejection:
            return Observation(
                message=rejection,
                expect_correction=True,
                properties={"rejected_command": args.command},
            )

        duplicate_id = self._session.duplicate_command(args.command)
        if duplicate_id:
            return Observation(
                message=(
                    f"This exact search was already captured as {duplicate_id}. "
                    "Use that evidence or inspect a non-overlapping location."
                ),
                expect_correction=True,
                properties={"duplicate_evidence_id": duplicate_id},
            )
        overlap_id = self._session.overlapping_read(args.command)
        if overlap_id:
            return Observation(
                message=(
                    f"This read overlaps at least 60% with {overlap_id}. "
                    "Use the existing evidence or inspect a different range."
                ),
                expect_correction=True,
                properties={"duplicate_evidence_id": overlap_id},
            )

        output, exit_code = self._docker_env.run_bash(
            args.command,
            timeout=120,
        )
        bounded_output = _bounded_output(output)
        duplicate_output_id = self._session.duplicate_output(bounded_output)
        if duplicate_output_id:
            return Observation(
                message=(
                    f"This search returned the same evidence as "
                    f"{duplicate_output_id}. Change the target or query."
                ),
                expect_correction=True,
                properties={"duplicate_evidence_id": duplicate_output_id},
            )
        evidence = self._session.record(
            args.command,
            bounded_output,
            exit_code,
            extra={"target": args.target, "purpose": args.purpose},
        )
        self._session.register_command(args.command, evidence.id)
        self._session.register_read(args.command, evidence.id)
        self._session.register_output(bounded_output, evidence.id)
        source_followup = _grep_source_followup(args.command, bounded_output)
        followup_message = (
            "\nThis grep output is a match listing, not a complete function "
            "definition. Before using it as a graph/source conclusion, inspect "
            "the nearby body; for example: `"
            + source_followup["command"]
            + "`."
            if source_followup else ""
        )
        return Observation(
            message=(
                f"[{evidence.id}] exit_code={exit_code}\n"
                + (bounded_output or "(no output)")
                + followup_message
            ),
            properties={
                "command": args.command,
                "exit_code": exit_code,
                "evidence_id": evidence.id,
                **({"source_followup": source_followup} if source_followup else {}),
            },
        )


class ListRepositoryTestsArgs(ActionArguments):
    pattern: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description=(
            "Case-insensitive regular expression matched against each test's "
            "name, nearby comment description, framework, and repository path. "
            "Use focused alternation such as 'count_(fields|elements)|object'; "
            "do not use '.*' unless path_pattern already bounds the list."
        ),
    )
    path_pattern: str = Field(
        default="",
        max_length=300,
        description=(
            "Optional repository-relative glob, for example "
            "'tests/ondemand/*' or 'test/src/unit-*'."
        ),
    )
    limit: int = Field(
        default=30,
        ge=1,
        le=50,
        description="Maximum number of matching test declarations to return.",
    )
    purpose: str = Field(
        default="",
        max_length=300,
        description=(
            "Which Issue behavior or analogous passing test this query should "
            "help locate."
        ),
    )

    model_config = ConfigDict(title="ListRepositoryTests", extra="forbid")


class ListRepositoryTests(Action):
    args_schema: ClassVar[Type[ActionArguments]] = ListRepositoryTestsArgs
    _inventory: RepositoryTestInventory = PrivateAttr()
    _session: InvestigationSession = PrivateAttr()

    def __init__(
        self,
        repository: Repository,
        session: InvestigationSession,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._inventory = RepositoryTestInventory(repository)
        self._session = session

    def execute(
        self,
        args: ListRepositoryTestsArgs,
        **kwargs,
    ) -> Observation:
        query = json.dumps(
            {
                "pattern": args.pattern,
                "path_pattern": args.path_pattern,
                "limit": args.limit,
            },
            sort_keys=True,
        )
        command = "list-tests:" + query
        duplicate_id = self._session.duplicate_command(command)
        if duplicate_id:
            return Observation(
                message=(
                    f"This test inventory query duplicates {duplicate_id}. "
                    "Refine the regex or path_pattern, or inspect one returned "
                    "test with ReadOnlySearch."
                ),
                expect_correction=True,
            )
        try:
            matches, total_matches = self._inventory.search(
                args.pattern,
                path_pattern=args.path_pattern,
                limit=args.limit,
            )
        except re.error as exc:
            return Observation(
                message=f"Invalid test-name regular expression: {exc}",
                expect_correction=True,
            )

        lines = [
            f"Found {total_matches} matching test declarations; "
            f"showing {len(matches)} of {len(self._inventory.entries)} indexed tests."
        ]
        serialized = []
        for index, entry in enumerate(matches, 1):
            lines.append(
                f"{index}. [{entry.framework}] {entry.name} - "
                f"{entry.path}:{entry.line}"
            )
            if entry.description:
                lines.append(f"   Description: {entry.description}")
            serialized.append({
                "name": entry.name,
                "description": entry.description,
                "framework": entry.framework,
                "path": entry.path,
                "line": entry.line,
            })
        if not matches:
            lines.append(
                "No declaration matched. Broaden the regex using an API noun, "
                "error token, or adjacent operation rather than listing all tests."
            )
        output = _bounded_output("\n".join(lines), max_lines=120)
        evidence = self._session.record(
            command,
            output,
            0,
            prefix="SEARCH",
            path="<test-inventory>",
            action="ListRepositoryTests",
            extra={
                "pattern": args.pattern,
                "path_pattern": args.path_pattern,
                "purpose": args.purpose,
                "match_count": total_matches,
            },
        )
        self._session.register_command(command, evidence.id)
        return Observation(
            message=f"[{evidence.id}]\n{output}",
            properties={
                "evidence_id": evidence.id,
                "tests": serialized,
                "total_matches": total_matches,
                "indexed_tests": len(self._inventory.entries),
            },
        )


class SearchTestIntentArgs(ActionArguments):
    """Use when behavior wording may find a relevant test/example that a literal
    source search missed, including an unknown fixture for a behavior. Returns
    test leads, not a passing oracle or a required pass/fail pair; use
    ReadOnlySearch for an exact known test name or file.
    """

    behavior: str = Field(
        min_length=1, max_length=600,
        description="Issue behavior to locate in indexed tests and examples; use a focused question, not a regex.",
    )

    model_config = ConfigDict(title="SearchTestIntent", extra="forbid")


class SearchTestIntent(Action):
    """Find analogous tests; a retrieved candidate is not a passing oracle."""

    args_schema: ClassVar[Type[ActionArguments]] = SearchTestIntentArgs
    _retriever: TestAnchorRetriever = PrivateAttr()
    _session: InvestigationSession = PrivateAttr()

    def __init__(
        self,
        repository: Repository,
        session: InvestigationSession,
        index_name: str | None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._retriever = TestAnchorRetriever(
            repository=repository, index_name=index_name, max_candidates=5,
        )
        self._session = session

    def execute(self, args: SearchTestIntentArgs, **kwargs) -> Observation:
        command = "test-intent:" + args.behavior.strip()
        duplicate_id = self._session.duplicate_command(command)
        if duplicate_id:
            return Observation(
                message=f"This test-intent query was already captured as {duplicate_id}.",
                expect_correction=True,
            )
        try:
            retrieval = self._retriever.retrieve(args.behavior, IssueEvidenceFrame())
        except Exception as exc:
            return Observation(
                message=(f"Test-intent lookup failed: {exc}. Use ListRepositoryTests "
                         "or ReadOnlySearch to inspect tests directly."),
                expect_correction=True,
            )
        candidates = [
            {
                "name": candidate.test_name,
                "path": candidate.path,
                "locator": candidate.locator,
                "retrieval_ranks": candidate.retrieval_ranks,
                "snippet": candidate.snippet[:1200],
            }
            for candidate in retrieval.candidates[:5]
        ]
        output = json.dumps(candidates, ensure_ascii=False)
        evidence = self._session.record(
            command, output, 0, prefix="SEARCH", path="<test-intent-index>",
            action="SearchTestIntent", extra={"behavior": args.behavior},
        )
        self._session.register_command(command, evidence.id)
        return Observation(
            message=(f"[{evidence.id}] {len(candidates)} test/example lead(s). "
                     "Inspect or run a candidate before treating it as behavior evidence.\n"
                     + output),
            properties={"evidence_id": evidence.id, "candidate_count": len(candidates)},
        )


class WriteInvestigationFileArgs(ActionArguments):
    path: str = Field(
        ...,
        description=(
            "Repository-relative path for a standalone minimal probe or script "
            "in this disposable investigation container. Do not use this action to "
            "write a candidate product patch. A standalone compiled probe should "
            "provide its own entry point and plain assertions/exit status instead "
            "of depending on a repository test-framework runner."
        ),
    )
    content: str = Field(..., max_length=20000)
    purpose: str = ""

    model_config = ConfigDict(title="WriteInvestigationFile")


class WriteInvestigationFile(Action):
    args_schema: ClassVar[Type[ActionArguments]] = WriteInvestigationFileArgs
    _docker_env: DockerEnv = PrivateAttr()
    _session: InvestigationSession = PrivateAttr()

    def __init__(
        self,
        docker_env: DockerEnv,
        session: InvestigationSession,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._docker_env = docker_env
        self._session = session

    def execute(
        self,
        args: WriteInvestigationFileArgs,
        **kwargs,
    ) -> Observation:
        path, rejection = _validated_relative_path(args.path)
        if rejection:
            return Observation(message=rejection, expect_correction=True)
        if path in self._session.investigation_files:
            _, registered_exit_code = self._docker_env.run_bash(
                f"test -e {shlex.quote(path)}",
                timeout=30,
            )
            if registered_exit_code == 0:
                return Observation(
                    message=(
                        f"Investigation file {path} already exists. Use "
                        "ReplaceInvestigationText to refine that standalone probe."
                    ),
                    expect_correction=True,
                )
            if registered_exit_code != 1:
                return Observation(
                    message=f"Could not verify registered investigation path {path}.",
                    expect_correction=True,
                )
            # Recover from an externally removed disposable probe instead of
            # trapping the Agent between a stale registry and a missing file.
            self._session.investigation_files.pop(path, None)
        output, exit_code = self._docker_env.run_bash(
            f"test -e {shlex.quote(path)}",
            timeout=30,
        )
        if exit_code == 0:
            return Observation(
                message=(
                    f"Cannot create investigation probe {path}: the path already "
                    "exists in the checked-out repository. Choose a new standalone "
                    "probe path; investigation must not overwrite repository files."
                ),
                expect_correction=True,
            )
        if exit_code != 1:
            return Observation(
                message=(
                    f"Could not verify that investigation path {path} is new:\n"
                    f"{_bounded_output(output)}"
                ),
                expect_correction=True,
            )
        self._docker_env.write_file(
            f"{self._docker_env.work_dir.rstrip('/')}/{path}",
            args.content,
        )
        self._session.investigation_files[path] = args.content
        self._session.actions.append(
            {
                "action": "WriteInvestigationFile",
                "path": path,
                "purpose": args.purpose,
                "bytes": len(args.content.encode()),
            }
        )
        return Observation(
            message=(
                f"Wrote {path} in the disposable investigation container. "
                "Compile or inspect it before drawing a conclusion."
            ),
            properties={"path": path},
        )


class ReplaceInvestigationTextArgs(ActionArguments):
    path: str = Field(
        ...,
        description=(
            "Repository-relative path of a standalone probe or script to refine. "
            "Do not modify product implementation to simulate a repair."
        ),
    )
    old: str
    new: str
    purpose: str = ""

    model_config = ConfigDict(title="ReplaceInvestigationText")


class ReplaceInvestigationText(Action):
    args_schema: ClassVar[Type[ActionArguments]] = ReplaceInvestigationTextArgs
    _docker_env: DockerEnv = PrivateAttr()
    _session: InvestigationSession = PrivateAttr()

    def __init__(
        self,
        docker_env: DockerEnv,
        session: InvestigationSession,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._docker_env = docker_env
        self._session = session

    def execute(
        self,
        args: ReplaceInvestigationTextArgs,
        **kwargs,
    ) -> Observation:
        path, rejection = _validated_relative_path(args.path)
        if rejection:
            return Observation(message=rejection, expect_correction=True)
        if path not in self._session.investigation_files:
            return Observation(
                message=(
                    f"Cannot modify {path}: ReplaceInvestigationText may only "
                    "refine a standalone probe created in this investigation."
                ),
                expect_correction=True,
            )
        content, exit_code = self._docker_env.run_bash(
            f"cat -- {shlex.quote(path)}",
            timeout=30,
        )
        if exit_code != 0:
            return Observation(
                message=f"Could not read {path}:\n{_bounded_output(content)}",
                expect_correction=True,
            )
        occurrences = content.count(args.old)
        if occurrences != 1:
            return Observation(
                message=(
                    f"Expected exactly one matching text block in {path}, "
                    f"found {occurrences}."
                ),
                expect_correction=True,
            )
        updated = content.replace(args.old, args.new, 1)
        self._docker_env.write_file(
            f"{self._docker_env.work_dir.rstrip('/')}/{path}",
            updated,
        )
        self._session.investigation_files[path] = updated
        self._session.actions.append(
            {
                "action": "ReplaceInvestigationText",
                "path": path,
                "purpose": args.purpose,
            }
        )
        return Observation(
            message=(
                f"Replaced one text block in {path} in the disposable "
                "investigation container. Validate the change with a command."
            ),
            properties={"path": path},
        )


class RunInvestigationCommandArgs(ActionArguments):
    command: str = Field(
        ...,
        description=(
            "A command for compilation, tests, or controlled investigation in "
            "the disposable investigation container. Git and network are unavailable. "
            "Use ReadOnlySearch for grep, sed, cat, head, tail, ls, pwd, or wc; "
            "do not combine a compiler/version command with a source read. "
            "Do not delete probes created by WriteInvestigationFile; refine them "
            "with ReplaceInvestigationText. Run native framework tests through an "
            "existing repository target; do not assemble framework mains or runner "
            "sources into a standalone probe command. If this RUN may become a "
            "contrastive witness or acceptance check, use literal command arguments "
            "only: command substitution (`$()` or backticks), `;`, `||`, pipes, and "
            "redirection make it diagnostic-only. The only permitted separator for "
            "a freezeable probe is one `&&` before its `/tmp` binary."
        ),
    )
    target: str = ""
    purpose: str = ""
    evidence_role: Literal["reproduction", "control", "premise"] = Field(
        ...,
        description=(
            "Use reproduction only when this command executes the original Issue "
            "input or failing focal target with a concrete expected/actual oracle. "
            "Use control for a passing behavior-matched anchor with a concrete oracle. "
            "Use premise for setup, tool/version/layout checks, build discovery, and "
            "focused predicate/call-viability/dispatch checks."
            " Reproduction/control commands must preserve the real compiler/test exit "
            "status: run directly, chaining prerequisites with &&, without trailing echo, "
            "pipes, semicolons or ||."
        ),
    )
    timeout: int = Field(default=180, ge=1, le=600)

    model_config = ConfigDict(title="RunInvestigationCommand")

    @model_validator(mode="before")
    @classmethod
    def default_legacy_evidence_role(cls, data):
        # Persisted trajectories predate the required role. Treat omitted roles
        # conservatively as premises; new tool schemas require an explicit choice.
        if isinstance(data, dict) and not data.get("evidence_role"):
            data = {**data, "evidence_role": "premise"}
        return data


class RunInvestigationCommand(Action):
    args_schema: ClassVar[Type[ActionArguments]] = RunInvestigationCommandArgs
    _docker_env: DockerEnv = PrivateAttr()
    _session: InvestigationSession = PrivateAttr()

    def __init__(
        self,
        docker_env: DockerEnv,
        session: InvestigationSession,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._docker_env = docker_env
        self._session = session

    def execute(
        self,
        args: RunInvestigationCommandArgs,
        **kwargs,
    ) -> Observation:
        # Preserve source reads as SEARCH evidence even when the model selects
        # the broader execution tool. This keeps exact source anchors available
        # to graph tracing and the repository-evidence validators.
        if not read_only_command_rejection(
            args.command,
            self._docker_env.work_dir,
        ):
            return ReadOnlySearch(
                docker_env=self._docker_env,
                session=self._session,
            ).execute(ReadOnlySearchArgs(
                command=args.command,
                target=args.target,
                purpose=args.purpose,
            ))
        rejection = investigation_command_rejection(args.command)
        if not rejection and args.evidence_role in {"reproduction", "control"}:
            rejection = verification_exit_status_rejection(args.command)
        if rejection:
            return Observation(message=rejection, expect_correction=True)
        duplicate_id = self._session.duplicate_probe_command(args.command)
        if duplicate_id:
            return Observation(
                message=(
                    f"This command was already captured as {duplicate_id}. "
                    "Change the probe or submit the existing result."
                ),
                expect_correction=True,
            )
        output, exit_code = self._docker_env.run_bash(
            (
                f"bash -e -o pipefail -c {shlex.quote(args.command)}"
                if args.evidence_role in {"reproduction", "control"}
                else args.command
            ),
            timeout=args.timeout,
        )
        bounded_output = _bounded_output(output)
        evidence = self._session.record(
            args.command,
            bounded_output,
            exit_code,
            prefix="RUN",
            action="RunInvestigationCommand",
            extra={
                "target": args.target,
                "purpose": args.purpose,
                "evidence_role": args.evidence_role,
                "frozen_probe_files": dict(
                    self._session.investigation_files
                ),
            },
        )
        self._session.register_probe_command(args.command, evidence.id)
        return Observation(
            message=(
                f"[{evidence.id}] exit_code={exit_code}\n"
                + (bounded_output or "(no output)")
            ),
            properties={
                "command": args.command,
                "exit_code": exit_code,
                "evidence_id": evidence.id,
            },
        )


GraphEdgeType = Literal[
    "CALLS", "IN_CLASS", "DEP_CALL", "DEP_TEMPLATE_REF", "DEP_DECL_REF",
    "DEP_MEMBER_REF", "DEP_TYPE_REF", "DEP_UNEXPOSED_REF",
]

class RepositoryGraphQueryArgs(ActionArguments):
    """Locate behavior via intent/function search, then follow returned entity IDs.

    Semantic summaries and graph edges are navigation leads. Read the actual
    source before using them to explain behavior or constrain a repair.
    """

    mode: Literal[
        "intent",
        "function",
        "trace",
        "symbol",
        "usages",
        "neighbors",
        "template_instantiations",
    ]
    query: str = Field(default="", description="Behavior to locate in intent/function mode.")
    symbol_type: Literal["function", "class", "macro"] = "function"
    symbol_name: str = ""
    namespace: str = ""
    class_name: str = ""
    entity_ref: str = Field(
        default="",
        description=(
            "Stable short entity_ref returned by a prior GRAPH result. Prefer this "
            "over copying the diagnostic compiler entity_id."
        ),
    )
    entity_id: str = Field(
        default="",
        description=(
            "Diagnostic exact id/func_id returned by a prior query. Prefer entity_ref; "
            "do not reconstruct or edit this value."
        ),
    )
    source_path: str = Field(
        default="",
        description=(
            "Repository-relative file already inspected with a source tool. In trace "
            "mode, pair with source_line to re-anchor the graph after semantic retrieval."
        ),
    )
    source_line: int | None = Field(
        default=None,
        ge=1,
        description="Exact inspected source line contained by the trace seed definition.",
    )
    template_arg: str = ""
    direction: Literal["in", "out", "both"] = "both"
    edge_types: list[GraphEdgeType] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "Optional exact graph-edge filter, e.g. CALLS, DEP_CALL, IN_CLASS, "
            "DEP_MEMBER_REF, DEP_TYPE_REF, DEP_TEMPLATE_REF. Leave empty when "
            "the indexed relation kind is uncertain; a zero-edge response reports "
            "available relation kinds so the caller can re-anchor deliberately."
        ),
    )
    limit: int = Field(default=10, ge=1, le=10)
    target: str = ""
    purpose: str = ""

    model_config = ConfigDict(title="RepositoryGraphQuery")


class StructuralCompanionSelection(ResponseSchema):
    """Select distinct source leads and optional graph bridges; do not prescribe a fix."""

    indexes: list[int] = Field(max_length=6)
    expand_indexes: list[int] = Field(default_factory=list, max_length=6)


class RepositoryGraphQuery(Action):
    args_schema: ClassVar[Type[ActionArguments]] = RepositoryGraphQueryArgs
    _code_index: CodeIndex = PrivateAttr()
    _session: InvestigationSession = PrivateAttr()
    _structural_completion: BaseCompletionModel | None = PrivateAttr(default=None)
    _rank_trace: dict = PrivateAttr(default_factory=dict)
    _entity_refs: dict[str, tuple[str, ...]] = PrivateAttr(default_factory=dict)
    _entity_ref_by_id: dict[str, str] = PrivateAttr(default_factory=dict)
    _next_entity_ref: int = PrivateAttr(default=1)

    def __init__(
        self,
        code_index: CodeIndex,
        session: InvestigationSession,
        structural_completion: BaseCompletionModel | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._code_index = code_index
        self._session = session
        self._structural_completion = structural_completion
        if structural_completion is not None:
            structural_completion.initialize(
                response_schema=StructuralCompanionSelection,
                system_prompt=(
                    "Select distinct source definitions or graph paths worth inspecting for "
                    "a C++ issue. Consider control flow, shared state, type dependencies, "
                    "and independent operations that could change causal or repair scope. "
                    "For graph_search_anchor, choose definitions whose connected relations "
                    "could answer the Issue-level scope question, not merely lexical matches. "
                    "For a graph_trace query, put in expand_indexes only selected nodes whose "
                    "next real edge could expose the answer: a class owner can be a useful "
                    "bridge to sibling methods even when its name does not match the Issue. "
                    "Keep complementary causal routes rather than near-duplicate overloads. "
                    "The Issue is authoritative; index summaries and graph edges are navigation "
                    "leads, not proof of a bug or a required fix. Return at most the requested "
                    "number of indexes in relevance order; expand_indexes must be a subset. "
                    "Return fewer or none when no candidate helps answer the question."
                ),
            )

    def _rerank_companions(self, query, documents, top_k, **kwargs):
        max_select_count = min(top_k, len(documents))
        request = {
            "query": query, "max_select_count": max_select_count,
            "candidates": [{"index": i, **json.loads(doc)} for i, doc in enumerate(documents)],
        }
        for attempt in range(2):
            response = self._structural_completion.create_completion(messages=[{
                "role": "user", "content": json.dumps(request, ensure_ascii=False),
            }])
            selection = response.structured_output
            if isinstance(selection, StructuralCompanionSelection):
                indexes = selection.indexes
                expand_indexes = selection.expand_indexes
                valid = (
                    len(indexes) <= max_select_count
                    and len(set(indexes)) == len(indexes)
                    and all(0 <= i < len(documents) for i in indexes)
                    and len(set(expand_indexes)) == len(expand_indexes)
                    and set(expand_indexes).issubset(indexes)
                )
                if valid:
                    self._rank_trace = {
                        "model": self._structural_completion.model,
                        "candidate_count": len(documents),
                        "selected_indexes": indexes, "expand_indexes": expand_indexes,
                        "retry_count": attempt,
                        "usage": response.completion.usage.model_dump()
                        if response.completion and response.completion.usage else None,
                    }
                    return [{"index": i, "score": 1 / (rank + 1),
                             "expand": i in expand_indexes} for rank, i in enumerate(indexes)]
                rejected = {"indexes": indexes, "expand_indexes": expand_indexes}
            else:
                rejected = {"response_type": type(selection).__name__}
            request["retry_feedback"] = {
                "invalid_selection": rejected,
                "requirement": (
                    f"Choose at most {max_select_count} distinct zero-based indexes in "
                    f"[0, {len(documents) - 1}]; expand_indexes must be a distinct "
                    "subset of indexes. Return empty lists if nothing qualifies."
                ),
            }
        raise ValueError(
            f"Invalid structural companion selection after retry: {rejected}"
        )

    def _trace_rank_context(self, args: RepositoryGraphQueryArgs, seeds: list[dict]) -> str:
        observations = [
            {"id": item.id, "locator": item.locator,
             "excerpt": item.observation[:320]}
            for item in self._session.evidence
            if item.id.startswith(("SEARCH-", "RUN-"))
        ][-3:]
        issue = self._session.issue_text
        if len(issue) > 3500:
            issue = issue[:2500] + "\n... [middle omitted] ...\n" + issue[-1000:]
        return json.dumps({
            "mode": "graph_trace",
            "issue_description": issue,
            "question": args.target,
            "missing_relation": args.purpose,
            "source_anchor": {"path": args.source_path, "line": args.source_line},
            "anchors": [
                {"name": str(seed.get("fq_name") or seed.get("name") or "")[-240:],
                 "file": seed.get("file") or seed.get("def_file"),
                 "intent": str(seed.get("short_intent") or "")[:240],
                 "source": self._source_excerpt(seed).get("source_excerpt", "")[:450]}
                for seed in seeds
            ],
            "recent_source_or_run_evidence": observations,
        }, ensure_ascii=False)

    def _rank_trace_frontier(
        self, query: str, candidates: list[dict], top_k: int, *,
        ctx=None, verify_source: bool = False,
    ) -> list[dict]:
        if not candidates or top_k <= 0:
            return []
        if self._structural_completion is None:
            return [{"index": i, "expand": False} for i in range(min(top_k, len(candidates)))]

        def label(entity_id: str) -> str:
            indexed = next((getattr(ctx, name, {}).get(entity_id)
                            for name in ("functions", "classes")
                            if getattr(ctx, name, {}).get(entity_id)), None)
            if indexed is None:
                return "unindexed node"
            name = (indexed.get("short_name") or indexed.get("name")
                    or indexed.get("fq_name") or entity_id)
            return str(name).rsplit("::", 1)[-1][:100]

        documents = []
        for entity in candidates:
            name = str(entity.get("fq_name") or entity.get("name") or "")
            document = {
                "name": str(entity.get("short_name") or entity.get("name")
                            or name.rsplit("::", 1)[-1])[:160],
                "qualified_name_tail": name[-220:],
                "file": entity.get("def_file") or entity.get("file"),
                "signature": str(entity.get("signature") or "")[:160],
                "short_intent": str(entity.get("short_intent") or "")[:200],
                "code_preview": str(entity.get("code") or "")[:180],
                "path": [[
                    {"source": label(edge["source"]), "edge": edge["edge"],
                     "target": label(edge["target"])}
                    for edge in path.get("edges", [])[:3]
                    if all(key in edge for key in ("source", "edge", "target"))
                ] for path in (entity.get("graph_paths") or [])[:2]],
            }
            if verify_source:
                excerpt = self._source_excerpt(entity)
                document["source_status"] = excerpt.get("source_status")
                document["source_preview"] = excerpt.get("source_excerpt", "")[:500]
            documents.append(json.dumps(document, ensure_ascii=False))

        def select(indices: list[int]) -> list[dict]:
            ranked = self._rerank_companions(
                query, [documents[index] for index in indices],
                top_k=min(top_k, len(indices)),
            )
            return [{**row, "index": indices[row["index"]]} for row in ranked]

        # Every neighbor gets considered. Large adjacency lists are ranked in
        # batches, never cut to their first lexicographically ordered members.
        batch_size = 160
        if len(documents) <= batch_size:
            return select(list(range(len(documents))))
        shortlist = []
        for start in range(0, len(documents), batch_size):
            shortlist.extend(select(list(range(start, min(start + batch_size, len(documents))))))
        selected_indices = list(dict.fromkeys(row["index"] for row in shortlist))
        if len(selected_indices) <= top_k:
            return shortlist
        ranked = select(selected_indices)
        for row in ranked:
            row["expand"] = row["expand"] or any(
                prior["index"] == row["index"] and prior["expand"]
                for prior in shortlist
            )
        return ranked

    def _register_entity_ref(self, entity: dict) -> str:
        entity_ids = []
        primary_id = (
            entity.get("raw_func_id")
            or entity.get("id")
            or entity.get("func_id")
        )
        for entity_id in [primary_id, *(entity.get("alias_entity_ids") or [])]:
            if isinstance(entity_id, str) and entity_id and entity_id not in entity_ids:
                entity_ids.append(entity_id)
        if not entity_ids:
            return ""

        entity_ref = next(
            (self._entity_ref_by_id[item] for item in entity_ids
             if item in self._entity_ref_by_id),
            "",
        )
        if not entity_ref:
            entity_ref = f"GREF-{self._next_entity_ref}"
            self._next_entity_ref += 1
        merged_ids = list(self._entity_refs.get(entity_ref, ()))
        for entity_id in entity_ids:
            if entity_id not in merged_ids:
                merged_ids.append(entity_id)
            self._entity_ref_by_id[entity_id] = entity_ref
        self._entity_refs[entity_ref] = tuple(merged_ids)
        return entity_ref

    def reset_entity_refs(self) -> None:
        self._entity_refs.clear()
        self._entity_ref_by_id.clear()
        self._next_entity_ref = 1

    def _skeleton(self, entity: dict, *, register_ref: bool = True) -> dict:
        result = {
            key: entity.get(key)
            for key in (
                "id",
                "func_id",
                "feature_id",
                "feature_name",
                "type",
                "kind",
                "category",
                "name",
                "short_name",
                "fq_name",
                "display_name",
                "file",
                "def_file",
                "trigger_file",
                "start_line",
                "def_start_line",
                "line_range",
                "short_intent",
                "skeleton",
                "score",
                "edge_types",
                "relationships",
                "retrieval_role",
                "graph_paths",
                "trace_requested_direction",
                "trace_direction_used",
                "direction_fallback",
                "trace_relation_fallback",
                "structural_candidates",
                "structural_status",
                "alias_entity_ids",
                "equivalent_source_forms",
            )
            if entity.get(key) not in (None, "", [])
        }
        for key in ("short_intent", "skeleton"):
            if key in result:
                result[key] = str(result[key])[:800]
        if entity.get("raw_func_id"):
            # Feature members use func_id for a display label in the backend.
            result["func_id"] = entity["raw_func_id"]
        entity_ref = self._register_entity_ref(entity) if register_ref else ""
        if entity_ref:
            result["entity_ref"] = entity_ref
        if "feature_description" in entity:
            result["feature_description"] = str(entity["feature_description"] or "")[:800]
            result["members"] = [
                self._skeleton(member, register_ref=register_ref)
                for member in entity.get("members", [])[:3]
            ]
        return result

    def _function_relation_frontier(
        self, candidates: list[dict], ctx, *, source_line_limit: int = 12,
    ) -> str:
        """Show semantic hits as short, followable directed graph routes."""
        by_id = {
            entity_id: candidate
            for candidate in candidates
            for entity_id in (
                candidate.get("id"), candidate.get("func_id"),
                *(candidate.get("alias_entity_ids") or []),
            )
            if entity_id
        }

        def endpoint(entity_id: str) -> str:
            indexed = next((
                getattr(ctx, collection, {}).get(entity_id)
                for collection in ("functions", "classes")
                if getattr(ctx, collection, {}).get(entity_id)
            ), None) if ctx is not None else None
            entity = indexed or by_id.get(entity_id)
            if entity is None:
                return "unindexed endpoint"
            ref = self._register_entity_ref({"id": entity_id, **entity})
            name = str(
                entity.get("fq_name") or entity.get("display_name")
                or entity.get("name") or entity.get("short_name") or "node"
            ).split("::")[-1]
            path = entity.get("def_file") or entity.get("file")
            line = entity.get("def_start_line") or entity.get("start_line")
            location = f" ({path}:{line})" if path and line else f" ({path})" if path else ""
            return f"{ref} {name}{location}"

        lines = [
            "Graph source leads and directed relation frontier. Relations are "
            "navigation leads, not proof of shared behavior or repair scope."
        ]
        peer_ids = []
        if ctx is not None:
            for candidate in candidates:
                for path_record in candidate.get("graph_paths") or []:
                    for edge in path_record.get("edges", []):
                        for entity_id in (edge.get("source"), edge.get("target")):
                            if (entity_id not in by_id
                                    and (entity_id in getattr(ctx, "functions", {})
                                         or entity_id in getattr(ctx, "classes", {}))
                                    and entity_id not in peer_ids):
                                peer_ids.append(entity_id)
        if peer_ids:
            lines.append("Connected definitions not shown as search hits; inspect or exclude issue-relevant ones before claiming local scope:")
            scope_frontier = []
            for entity_id in peer_ids:
                is_class = entity_id in getattr(ctx, "classes", {})
                peer = (ctx.classes if is_class else ctx.functions)[entity_id]
                label = endpoint(entity_id)
                lines.append("  " + label)
                excerpt = self._source_excerpt(peer)
                preview = (
                    excerpt["source_excerpt"].splitlines()[:3]
                    if excerpt.get("source_status") == "verified" else []
                )
                scope_frontier.append({
                    "entity_ref": self._entity_ref_by_id[entity_id],
                    "kind": "class" if is_class else "function",
                    "symbol": peer.get("fq_name") or peer.get("name"),
                    "file": peer.get("def_file") or peer.get("file"),
                    "source_status": excerpt.get("source_status"),
                    "source_preview": preview,
                })
                if excerpt.get("source_status") == "verified":
                    lines.extend(
                        "    " + source_line
                        for source_line in preview
                    )
            # Keep this unresolved navigation frontier visible when a later
            # reviewer receives a bounded prefix of the raw GRAPH observation.
            candidates[0] = {"scope_frontier": scope_frontier, **candidates[0]}
        seed_locations = {
            candidate.get("file") or candidate.get("def_file")
            for candidate in candidates
            if candidate.get("retrieval_role") == "semantic_seed"
        }
        if len(seed_locations) > 1:
            lines.append(
                "The search matched definitions in multiple files; check whether "
                "the relevant build/public forms share an implementation or need "
                "separate changes."
            )
        seen_candidates = set()
        for candidate in candidates:
            identity = (candidate.get("entity_ref"), candidate.get("file"),
                        candidate.get("start_line") or candidate.get("line_range"))
            if isinstance(identity[-1], list):
                identity = (*identity[:-1], tuple(identity[-1]))
            if identity in seen_candidates:
                continue
            seen_candidates.add(identity)
            name = candidate.get("fq_name") or candidate.get("display_name") or "node"
            path = candidate.get("file") or candidate.get("def_file") or "unknown source"
            line = candidate.get("start_line") or candidate.get("def_start_line")
            location = f"{path}:{line}" if line else path
            lines.append(
                f"{candidate.get('entity_ref', '')} {name} [{location}] "
                f"({candidate.get('retrieval_role', 'candidate')}, "
                f"source={candidate.get('source_status', 'unavailable')})"
            )
            if candidate.get("short_intent"):
                lines.append(f"  indexed intent: {candidate['short_intent']}")
            for form in candidate.get("equivalent_source_forms") or []:
                lines.append(
                    "  byte-identical source form: "
                    f"{form['path']}:{form['line']} "
                    f"(source={form['source_status']}; inspect build/packaging use)"
                )
            if candidate.get("source_excerpt"):
                excerpt_lines = candidate["source_excerpt"].splitlines()
                lines.append("  source: " + "\n    ".join(excerpt_lines[:source_line_limit]))
                if len(excerpt_lines) > source_line_limit:
                    lines.append("    ... inspect the full definition in source")
            routes = []
            for path_record in candidate.get("graph_paths") or []:
                edges = [
                    f"{endpoint(edge['source'])} --{edge['edge']}--> "
                    f"{endpoint(edge['target'])}"
                    for edge in path_record.get("edges", [])
                    if all(edge.get(key) for key in ("source", "edge", "target"))
                ]
                if edges:
                    route = "; ".join(edges)
                    if route not in routes:
                        routes.append(route)
            candidate["relation_frontier"] = routes
            lines.extend(f"  relation: {route}" for route in routes)
        lines.append(
            "Inspect the source of any related node that could share the failing "
            "representation, predicate, or public behavior. Use "
            "TraceInvestigationPath on its GREF only when another unanswered "
            "relation could change the causal explanation "
            "or repair scope; otherwise continue with source search."
        )
        return "\n".join(lines)

    def _source_excerpt(self, entity: dict) -> dict:
        """Verify indexed locations against the current checkout before handoff."""
        repo = getattr(self._code_index, "_file_repo", None)
        path = entity.get("file") or entity.get("def_file")
        span = entity.get("line_range") or [
            entity.get("def_start_line") or entity.get("start_line")
            or entity.get("def_line") or entity.get("line"),
            entity.get("def_end_line") or entity.get("end_line")
            or entity.get("def_line") or entity.get("line"),
        ]
        if not repo or not path or not all(isinstance(n, int) and n > 0 for n in span):
            return {"source_status": "unavailable"}
        # Index paths must refer to this checkout, never an old absolute host path.
        if PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
            return {"source_status": "non_repository_path"}
        content = repo.get_file_content(path)
        if content is None:
            return {"source_status": "unavailable"}
        lines = content.splitlines(keepends=True)
        start, end = span
        if not (start <= end <= len(lines)):
            return {"source_status": "invalid_location"}
        snippet = "".join(lines[start - 1:end])
        digest = hashlib.sha256(snippet.encode()).hexdigest()
        if not entity.get("content_hash") or digest != entity["content_hash"]:
            return {"source_status": "index_mismatch", "source_sha256": digest}
        excerpt = "".join(f"{n}: {lines[n - 1]}" for n in range(start, min(end, start + 49) + 1))
        return {"source_status": "verified", "source_sha256": digest,
                "source_excerpt": excerpt[:2400],
                "source_truncated": end >= start + 50 or len(excerpt) > 2400}

    @staticmethod
    def _source_identity(entity: dict) -> tuple:
        """Group macro-expanded graph aliases that name one source definition."""
        span = entity.get("line_range") or [
            entity.get("def_start_line") or entity.get("start_line")
            or entity.get("def_line") or entity.get("line"),
            entity.get("def_end_line") or entity.get("end_line")
            or entity.get("def_line") or entity.get("line"),
        ]
        return (
            entity.get("def_file") or entity.get("file"),
            tuple(span),
            entity.get("short_name") or entity.get("name"),
        )

    @staticmethod
    def _same_function_content(left: dict, right: dict) -> bool:
        """Match duplicated packaged definitions without conflating overloads."""
        if not left.get("content_hash") or not left.get("fq_name"):
            return False
        if (
            left.get("content_hash") != right.get("content_hash")
            or left.get("fq_name") != right.get("fq_name")
        ):
            return False
        for field_name in ("return_type",):
            if (
                left.get(field_name)
                and right.get(field_name)
                and left[field_name] != right[field_name]
            ):
                return False
        left_parameters = left.get("parameter_types") or left.get("parameters")
        right_parameters = right.get("parameter_types") or right.get("parameters")
        return not (
            left_parameters
            and right_parameters
            and list(left_parameters) != list(right_parameters)
        )

    def _attach_equivalent_function_aliases(self, entity: dict, ctx) -> None:
        """Attach graph IDs for byte-identical packaged copies of one function."""
        if ctx is None:
            return
        primary_id = (
            entity.get("raw_func_id")
            or entity.get("id")
            or entity.get("func_id")
        )
        alias_ids = list(dict.fromkeys([
            item
            for item in [
                primary_id,
                *(entity.get("_trace_alias_ids") or []),
                *(entity.get("alias_entity_ids") or []),
            ]
            if isinstance(item, str) and item
        ]))
        source_path = entity.get("def_file") or entity.get("file")
        name_tail = str(entity.get("fq_name") or "").rsplit("::", 1)[-1]
        source_forms = []
        seen_forms = set()
        for entity_id, indexed in getattr(ctx, "functions", {}).items():
            if (
                self._same_function_content(entity, indexed)
                and entity_id not in alias_ids
            ):
                alias_ids.append(entity_id)
            indexed_path = indexed.get("def_file") or indexed.get("file")
            indexed_line = indexed.get("def_start_line") or indexed.get("start_line")
            form_key = (indexed_path, indexed_line)
            if (source_path and indexed_path and indexed_path != source_path
                    and indexed_line and form_key not in seen_forms
                    and entity.get("content_hash")
                    and entity["content_hash"] == indexed.get("content_hash")
                    and name_tail
                    and name_tail == str(indexed.get("fq_name") or "").rsplit("::", 1)[-1]):
                seen_forms.add(form_key)
                if self._source_excerpt(indexed).get("source_status") == "verified":
                    source_forms.append({
                        "path": indexed_path, "line": indexed_line,
                        "source_status": "verified",
                    })
        if len(alias_ids) > 1:
            entity["alias_entity_ids"] = alias_ids
            entity["_trace_alias_ids"] = alias_ids
        if source_forms:
            entity["equivalent_source_forms"] = source_forms
        for member in entity.get("members", [])[:3]:
            if isinstance(member, dict):
                self._attach_equivalent_function_aliases(member, ctx)

    @staticmethod
    def _available_relation_summary(ctx, seeds: list[dict]) -> str:
        graph = getattr(ctx, "graph", None)
        edge_buckets = getattr(graph, "edge_buckets", None)
        if not callable(edge_buckets):
            return "unavailable from the active graph backend"
        summaries = []
        seen_ids = set()
        for seed in seeds:
            alias_ids = seed.get("_trace_alias_ids") or [
                seed.get("id") or seed.get("func_id")
            ]
            for entity_id in alias_ids:
                if not entity_id or entity_id in seen_ids:
                    continue
                seen_ids.add(entity_id)
                relation_counts = {}
                for direction in ("out", "in"):
                    try:
                        buckets = edge_buckets(entity_id, direction)
                    except Exception:
                        buckets = {}
                    for edge_type, neighbors in buckets.items():
                        relation_counts[f"{direction}:{edge_type}"] = len(neighbors)
                summaries.append({
                    "entity_id": entity_id,
                    "relations": relation_counts,
                })
        return json.dumps(summaries, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _entity_span(entity: dict) -> tuple[int, int] | None:
        span = entity.get("line_range") or [
            entity.get("def_start_line") or entity.get("start_line")
            or entity.get("def_line") or entity.get("line"),
            entity.get("def_end_line") or entity.get("end_line")
            or entity.get("def_line") or entity.get("line"),
        ]
        if (len(span) != 2
                or not all(isinstance(value, int) and value > 0 for value in span)
                or span[1] < span[0]):
            return None
        return span[0], span[1]

    @staticmethod
    def _same_source_path(indexed_path: str, requested_path: str) -> bool:
        indexed = PurePosixPath(str(indexed_path)).as_posix().removeprefix("./")
        requested = PurePosixPath(requested_path).as_posix().removeprefix("./")
        return indexed == requested or indexed.endswith("/" + requested)

    def _source_trace_seeds(self, ctx, source_path: str, source_line: int) -> list[dict]:
        normalized_path, rejection = _validated_relative_path(source_path)
        if rejection:
            raise ValueError(rejection)
        candidates = []
        for collection_rank, collection_name in enumerate(("functions", "classes")):
            for entity_id, indexed in getattr(ctx, collection_name, {}).items():
                indexed_path = indexed.get("def_file") or indexed.get("file")
                span = self._entity_span(indexed)
                if (not indexed_path or span is None
                        or not self._same_source_path(indexed_path, normalized_path)
                        or not span[0] <= source_line <= span[1]):
                    continue
                candidates.append((
                    span[1] - span[0], collection_rank, span[0],
                    int(any(
                        part.isupper() and len(part) > 2
                        for part in str(
                            indexed.get("fq_name") or indexed.get("name") or ""
                        ).split("::")[:-1]
                    )),
                    str(indexed.get("fq_name") or indexed.get("name") or ""),
                    {"id": entity_id, **indexed},
                ))
        candidates.sort(key=lambda item: item[:5])

        deduped: dict[tuple, dict] = {}
        for *_, candidate in candidates:
            entity_id = candidate.get("id") or candidate.get("func_id")
            source_key = self._source_identity(candidate)
            representative = deduped.setdefault(source_key, {
                **candidate, "_trace_alias_ids": [],
            })
            if entity_id and entity_id not in representative["_trace_alias_ids"]:
                representative["_trace_alias_ids"].append(entity_id)
        seeds = list(deduped.values())[:2]
        if not seeds:
            raise ValueError(
                f"No graph definition contains {normalized_path}:{source_line}; "
                "use a line inside the inspected function/class definition or an entity_ref."
            )
        return seeds

    def _followup_entity(self, entity: dict, args, rag) -> dict:
        """Return authoritative locations and actual one-hop edges for follow-up reads."""
        ctx = getattr(rag, "graph_ctx", None)
        eid = entity.get("id") or entity.get("func_id")
        indexed = next((getattr(ctx, name, {}).get(eid) for name in ("functions", "classes")
                        if getattr(ctx, name, {}).get(eid)), {})
        enriched = {**entity, **indexed}
        if indexed:
            enriched["file"] = (
                indexed.get("def_file") or indexed.get("file") or enriched.get("file")
            )
            enriched["line_range"] = [
                indexed.get("def_start_line") or indexed.get("start_line")
                or indexed.get("def_line") or indexed.get("line")
                or enriched.get("def_start_line") or enriched.get("start_line"),
                indexed.get("def_end_line") or indexed.get("end_line")
                or indexed.get("def_line") or indexed.get("line")
                or enriched.get("def_end_line") or enriched.get("end_line"),
            ]
        if args.mode == "neighbors" and ctx is not None:
            edges = []
            for edge in entity.get("edge_types", []):
                for direction in ("out", "in"):
                    if args.direction not in ("both", direction):
                        continue
                    if eid in ctx.graph.neighbors(args.entity_id, edge, direction):
                        source, target = (args.entity_id, eid) if direction == "out" else (eid, args.entity_id)
                        edges.append({"source": source, "edge": edge, "target": target})
            if edges:
                enriched["graph_paths"] = [{"seed_id": args.entity_id, "edges": edges}]
        return enriched

    def execute(
        self,
        args: RepositoryGraphQueryArgs,
        **kwargs,
    ) -> Observation:
        query = json.dumps(
            args.model_dump(exclude={"thoughts"} if args.mode == "trace"
                            else {"thoughts", "target", "purpose"}),
            sort_keys=True,
        )
        duplicate_id = self._session.duplicate_command("graph:" + query)
        if duplicate_id:
            return Observation(
                message=(
                    f"This graph query was already captured as {duplicate_id}."
                ),
                expect_correction=True,
            )
        try:
            self._rank_trace = {}
            self._direction_trace = {}
            from code_intent.code_intent.entity_tools import (
                find_template_instantiation,
                find_usages,
                get_entity_neighbors,
                get_graph_rag,
                list_features,
                search_by_intent,
                search_symbol,
            )

            if hasattr(self._code_index, "_activate_instance"):
                self._code_index._activate_instance()

            if args.mode in {"intent", "function"}:
                if not args.query.strip():
                    raise ValueError("query is required for intent/function mode")
                from code_intent.code_intent.embedding_manifest import load_embedding_manifest

                rag = get_graph_rag()
                if rag is None:
                    raise ValueError("Semantic index is unavailable; use symbol/source search")
                manifest = load_embedding_manifest(rag.func_index.index_path.parent)
                required = ("features",) if args.mode == "intent" else ("functions", "features")
                if any(not manifest.get("indexes", {}).get(name, {}).get("model") for name in required):
                    raise ValueError(
                        "Semantic index embedding provenance is missing. Align/rebuild "
                        "the vectors before semantic search; symbol/source search remains available."
                    )
                if args.mode == "intent":
                    result = list_features(query=args.query, limit=args.limit)
                    entities = result.get("features", [])
                else:
                    result = search_by_intent(
                        query=args.query, top_k=args.limit, rerank_first=True,
                        expand_structure=True,
                        structural_reranker=self._rerank_companions if self._structural_completion else None,
                    )
                    entities = result.get("results", [])
            elif args.mode == "trace":
                rag = get_graph_rag()
                ctx = getattr(rag, "graph_ctx", None)
                if ctx is None:
                    raise ValueError("Structural graph is unavailable for a relation trace")
                has_source_anchor = bool(args.source_path or args.source_line is not None)
                if bool(args.source_path) != (args.source_line is not None):
                    raise ValueError("source_path and source_line must be supplied together")
                selectors = [
                    bool(args.entity_ref), bool(args.entity_id),
                    has_source_anchor, bool(args.symbol_name),
                ]
                if sum(selectors) != 1:
                    raise ValueError(
                        "Trace requires exactly one anchor: entity_ref, entity_id, "
                        "source_path+source_line, or symbol_name. Do not combine them."
                    )
                if args.entity_ref:
                    entity_ids = self._entity_refs.get(args.entity_ref)
                    if not entity_ids:
                        raise ValueError(
                            "Unknown entity_ref; use a GREF-N returned by this graph session."
                        )
                    indexed_seeds = []
                    for entity_id in entity_ids:
                        seed = next((getattr(ctx, name, {}).get(entity_id)
                                     for name in ("functions", "classes")
                                     if getattr(ctx, name, {}).get(entity_id)), None)
                        if seed:
                            indexed_seeds.append({"id": entity_id, **seed})
                    if not indexed_seeds:
                        raise ValueError(
                            "The entity_ref no longer resolves in the active graph."
                        )
                    representative = {
                        **indexed_seeds[0],
                        "_trace_alias_ids": [seed["id"] for seed in indexed_seeds],
                    }
                    seeds = [representative]
                elif args.entity_id:
                    seed = next((getattr(ctx, name, {}).get(args.entity_id)
                                 for name in ("functions", "classes")
                                 if getattr(ctx, name, {}).get(args.entity_id)), None)
                    seeds = [{"id": args.entity_id, **(seed or {})}] if seed else []
                    if not seeds:
                        raise ValueError("Trace entity_id is not present in the active graph")
                elif has_source_anchor:
                    seeds = self._source_trace_seeds(
                        ctx, args.source_path, args.source_line,
                    )
                else:
                    entity_types = [args.symbol_type]
                    if args.symbol_type in {"function", "class"}:
                        entity_types.append(args.symbol_type + "_template")
                    seeds = []
                    for entity_type in entity_types:
                        found = search_symbol(
                            type=entity_type,
                            name=args.symbol_name,
                            namespace=args.namespace or None,
                            class_name=args.class_name or None,
                            limit=25,
                        )
                        if found.get("error"):
                            raise ValueError(found["error"])
                        seeds.extend(found.get("entities", []))
                    symbol = args.symbol_name.strip()
                    exact_seeds = []
                    for seed in seeds:
                        short_name = str(seed.get("short_name") or seed.get("name") or "")
                        fq_name = str(seed.get("fq_name") or seed.get("display_name") or "")
                        symbol_matches = (
                            short_name == symbol
                            or fq_name == symbol
                            or fq_name.endswith("::" + symbol)
                        )
                        class_matches = (
                            not args.class_name
                            or f"::{args.class_name}::" in fq_name
                            or fq_name.startswith(args.class_name + "::")
                            or str(seed.get("class_name") or "") == args.class_name
                            or args.class_name in (seed.get("classes") or [])
                        )
                        if symbol_matches and class_matches:
                            exact_seeds.append(seed)
                    # C++ macro namespaces can expose several graph IDs for the
                    # same physical declaration. Keep one source seed while
                    # following every alias ID, so aliases do not consume the
                    # bounded source-result slots or hide useful neighbors.
                    deduped = {}
                    for seed in exact_seeds:
                        seed_id = seed.get("id") or seed.get("func_id")
                        if not seed_id:
                            continue
                        source_key = self._source_identity(seed)
                        representative = deduped.setdefault(source_key, {
                            **seed, "_trace_alias_ids": [],
                        })
                        representative["_trace_alias_ids"].append(seed_id)
                    seed_limit = min(2, max(1, args.limit - 1))
                    seeds = list(deduped.values())[:seed_limit]
                    if not seeds:
                        raise ValueError(
                            "No exact graph symbol matched; use an exact entity_id or refine "
                            "namespace/class_name instead of tracing a fuzzy name."
                        )
                for seed in seeds:
                    self._attach_equivalent_function_aliases(seed, ctx)
                edge_types = args.edge_types or [
                    "CALLS", "DEP_CALL", "IN_CLASS", "DEP_MEMBER_REF", "DEP_TYPE_REF",
                    "DEP_TEMPLATE_REF",
                ]
                class_seed_ids = set(getattr(ctx, "classes", {}))
                class_seeds = [
                    seed for seed in seeds
                    if any(
                        entity_id in class_seed_ids
                        for entity_id in (
                            seed.get("_trace_alias_ids")
                            or [seed.get("id") or seed.get("func_id")]
                        )
                    )
                ]
                trace_seeds = []
                for seed in seeds:
                    seed_id = seed.get("id") or seed.get("func_id")
                    alias_ids = seed.get("_trace_alias_ids") or [seed_id]
                    seed_args = args.model_copy(update={"mode": "symbol", "entity_id": seed_id})
                    enriched_seed = self._followup_entity(seed, seed_args, rag)
                    enriched_seed["retrieval_role"] = "trace_seed"
                    if len(alias_ids) > 1:
                        enriched_seed["alias_entity_ids"] = alias_ids
                    trace_seeds.append(enriched_seed)

                def collect_neighbors(
                    trace_direction: str,
                    relation_edge_types: list[str],
                    relation_seeds: list[dict] | None = None,
                ) -> list[dict]:
                    trace_neighbors = []
                    seen_entity_ids = set()
                    neighbor_by_source = {}
                    for seed in relation_seeds or seeds:
                        seed_id = seed.get("id") or seed.get("func_id")
                        alias_ids = (
                            seed.get("_trace_alias_ids")
                            or seed.get("alias_entity_ids")
                            or [seed_id]
                        )
                        seen_entity_ids.update(alias_ids)
                        for alias_id in alias_ids:
                            neighbor_ids = set().union(*(
                                ctx.graph.neighbors(alias_id, edge, trace_direction)
                                for edge in relation_edge_types
                            ))
                            found = get_entity_neighbors(
                                entity_id=alias_id,
                                direction=trace_direction,
                                max_depth=1,
                                max_nodes=max(1, len(neighbor_ids)),
                                edge_types=relation_edge_types,
                            )
                            if found.get("error"):
                                raise ValueError(found["error"])
                            neighbor_args = args.model_copy(update={
                                "mode": "neighbors", "entity_id": alias_id,
                                "direction": trace_direction,
                                "edge_types": relation_edge_types,
                            })
                            for neighbor in found.get("neighbors", []):
                                enriched = self._followup_entity(
                                    neighbor, neighbor_args, rag,
                                )
                                neighbor_id = enriched.get("id") or enriched.get("func_id")
                                if (neighbor_id in seen_entity_ids
                                        or not enriched.get("graph_paths")):
                                    continue
                                if seed.get("graph_paths"):
                                    enriched["graph_paths"] = [
                                        {"seed_id": parent_path.get("seed_id", seed_id),
                                         "edges": parent_path["edges"] + child_path["edges"]}
                                        for parent_path in seed["graph_paths"][:2]
                                        for child_path in enriched["graph_paths"][:2]
                                    ]
                                source_key = self._source_identity(enriched)
                                prior_neighbor = neighbor_by_source.get(source_key)
                                if prior_neighbor is not None:
                                    aliases = prior_neighbor.setdefault(
                                        "alias_entity_ids",
                                        [prior_neighbor.get("id")
                                         or prior_neighbor.get("func_id")],
                                    )
                                    if neighbor_id not in aliases:
                                        aliases.append(neighbor_id)
                                    for graph_path in enriched["graph_paths"]:
                                        if graph_path not in prior_neighbor["graph_paths"]:
                                            prior_neighbor["graph_paths"].append(graph_path)
                                    seen_entity_ids.add(neighbor_id)
                                    continue
                                enriched["retrieval_role"] = "trace_neighbor"
                                trace_neighbors.append(enriched)
                                neighbor_by_source[source_key] = enriched
                                seen_entity_ids.add(neighbor_id)
                    return trace_neighbors

                trace_direction = args.direction
                trace_neighbors = collect_neighbors(
                    trace_direction, edge_types,
                )
                if not trace_neighbors and args.direction in {"in", "out"}:
                    trace_direction = "in" if args.direction == "out" else "out"
                    trace_neighbors = collect_neighbors(
                        trace_direction, edge_types,
                    )
                class_membership_fallback = False
                if (not trace_neighbors and class_seeds
                        and "IN_CLASS" not in edge_types):
                    # A source line inside a C++ class often resolves to the
                    # owner node even when the requested behavior is implemented
                    # by an out-of-line member. Resolve that declaration boundary
                    # to real member definitions before asking the model to choose
                    # a stable function ref for the next call/state hop.
                    for membership_direction in ("in", "out"):
                        trace_neighbors = collect_neighbors(
                            membership_direction,
                            ["IN_CLASS"],
                            class_seeds,
                        )
                        if trace_neighbors:
                            trace_direction = membership_direction
                            class_membership_fallback = True
                            break
                if not trace_neighbors:
                    available_relations = self._available_relation_summary(
                        ctx,
                        seeds,
                    )
                    raise ValueError(
                        "No real one-hop relation matched the exact seed and edge filter. "
                        f"Requested direction={args.direction}, edge_types={edge_types}. "
                        f"Available relations on exact/equivalent anchors: "
                        f"{available_relations}. A zero-edge result at this anchor is "
                        "not evidence that the causal relation is absent. If the relation "
                        "would change the explanation or scope, re-anchor at the concrete "
                        "producer/consumer/trait or retry with an observed relation kind; "
                        "otherwise continue with source tools."
                    )
                direction_fallback = trace_direction != args.direction
                self._direction_trace = {
                    "requested": args.direction,
                    "used": trace_direction,
                    "fallback": direction_fallback,
                }
                if class_membership_fallback:
                    self._direction_trace.update({
                        "edge_types_requested": edge_types,
                        "edge_types_used": ["IN_CLASS"],
                        "class_membership_fallback": True,
                    })
                for entity in [*trace_seeds, *trace_neighbors]:
                    entity["trace_requested_direction"] = args.direction
                    entity["trace_direction_used"] = trace_direction
                    if direction_fallback:
                        entity["direction_fallback"] = True
                    if class_membership_fallback:
                        entity["trace_relation_fallback"] = "class_membership"
                rank_context = self._trace_rank_context(args, trace_seeds)
                rank_steps = []
                display_slots = max(0, args.limit - len(trace_seeds))
                ranked = self._rank_trace_frontier(
                    rank_context, trace_neighbors, display_slots, ctx=ctx,
                )
                if self._structural_completion is not None:
                    rank_steps.append(self._rank_trace.copy())
                selected = {}
                visited_source_keys = {
                    self._source_identity(entity) for entity in trace_seeds
                }
                frontier = []
                seed_ids = {
                    entity_id
                    for seed in trace_seeds
                    for entity_id in (
                        seed.get("_trace_alias_ids")
                        or [seed.get("id") or seed.get("func_id")]
                    )
                }
                for row in ranked:
                    entity = trace_neighbors[row["index"]]
                    source_key = self._source_identity(entity)
                    selected[source_key] = entity
                    visited_source_keys.add(source_key)
                    entity_id = entity.get("id") or entity.get("func_id")
                    if (row.get("expand") or
                            (entity_id in class_seed_ids and
                             ctx.graph.neighbors(entity_id, "IN_CLASS", "in"))):
                        frontier.append(entity)

                # The model marks useful bridge nodes. Expand them inside this
                # action so Scout sees a source lead, not just an owning class.
                # The depth is an output-work budget, not a Scout graph-call gate.
                for depth in (2, 3):
                    if not frontier:
                        break
                    next_candidates = {}
                    expanded_class_ids = set()
                    for parent in frontier:
                        parent_id = parent.get("id") or parent.get("func_id")
                        is_class = parent_id in class_seed_ids
                        if is_class:
                            expanded_class_ids.add(parent_id)
                        children = collect_neighbors(
                            "both" if is_class else trace_direction,
                            ["IN_CLASS"] if is_class else edge_types,
                            [parent],
                        )
                        for child in children:
                            child_id = child.get("id") or child.get("func_id")
                            key = self._source_identity(child)
                            if child_id in seed_ids or key in visited_source_keys:
                                continue
                            prior = next_candidates.get(key)
                            if prior is None:
                                next_candidates[key] = child
                            else:
                                for path in child.get("graph_paths") or []:
                                    if path not in prior["graph_paths"]:
                                        prior["graph_paths"].append(path)
                    if not next_candidates:
                        break
                    candidates = list(next_candidates.values())
                    stage_ranked = self._rank_trace_frontier(
                        rank_context, candidates, display_slots, ctx=ctx,
                    )
                    if self._structural_completion is not None:
                        rank_steps.append(self._rank_trace.copy())
                    if not stage_ranked:
                        break
                    frontier = []
                    for row in stage_ranked:
                        entity = candidates[row["index"]]
                        source_key = self._source_identity(entity)
                        selected[source_key] = entity
                        visited_source_keys.add(source_key)
                        entity_id = entity.get("id") or entity.get("func_id")
                        if (row.get("expand") or
                                (entity_id in class_seed_ids and
                                 ctx.graph.neighbors(entity_id, "IN_CLASS", "in"))):
                            frontier.append(entity)
                    if expanded_class_ids:
                        for key, entity in list(selected.items()):
                            if (entity.get("id") or entity.get("func_id")) in expanded_class_ids:
                                selected.pop(key)
                    self._direction_trace["internal_hops"] = depth

                leads = list(selected.values())
                verified_functions = [
                    entity for entity in leads
                    if (entity.get("id") or entity.get("func_id")) not in class_seed_ids
                    and self._source_excerpt(entity).get("source_status") == "verified"
                ]
                if ranked and not verified_functions:
                    deferred = [
                        entity for entity in trace_neighbors
                        if self._source_identity(entity) not in visited_source_keys
                    ]
                    if deferred:
                        recovery_context = json.dumps({
                            **json.loads(rank_context),
                            "recovery_note": (
                                "Earlier paths yielded no verified function lead; "
                                "consider a different causal route if one exists."
                            ),
                        }, ensure_ascii=False)
                        recovery_ranked = self._rank_trace_frontier(
                            recovery_context, deferred, display_slots, ctx=ctx,
                        )
                        if self._structural_completion is not None:
                            rank_steps.append(self._rank_trace.copy())
                        leads.extend(deferred[row["index"]] for row in recovery_ranked)
                        self._direction_trace["deferred_recovery"] = bool(recovery_ranked)
                if len(leads) > display_slots:
                    final_ranked = self._rank_trace_frontier(
                        rank_context, leads, display_slots, ctx=ctx,
                        verify_source=True,
                    )
                    if self._structural_completion is not None:
                        rank_steps.append(self._rank_trace.copy())
                    leads = [leads[row["index"]] for row in final_ranked]
                # Direct callers cannot reveal independent members of the same
                # owner. When the exact source anchor also resolved to a class,
                # offer a separately labelled, issue-ranked member frontier.
                # These are scope leads, never additional CALLS results.
                if (class_seeds and not class_membership_fallback and edge_types
                        and set(edge_types) <= {"CALLS", "DEP_CALL"}
                        and len(leads) < display_slots):
                    member_candidates = [
                        member for member in collect_neighbors(
                            "in", ["IN_CLASS"], class_seeds,
                        )
                        if self._source_identity(member) not in visited_source_keys
                    ]
                    if member_candidates:
                        sibling_context = json.dumps({
                            **json.loads(rank_context),
                            "relation_kind": "IN_CLASS sibling, not direct caller",
                            "scope_question": (
                                "Which independent public members could produce or "
                                "consume the Issue-relevant representation? "
                                "Return only source leads worth inspecting; "
                                "membership alone is not evidence of a shared bug."
                            ),
                        }, ensure_ascii=False)
                        sibling_ranked = self._rank_trace_frontier(
                            sibling_context, member_candidates,
                            display_slots - len(leads), ctx=ctx,
                        )
                        if self._structural_completion is not None:
                            rank_steps.append(self._rank_trace.copy())
                        for row in sibling_ranked:
                            member = member_candidates[row["index"]]
                            member["retrieval_role"] = "scope_sibling_lead"
                            leads.append(member)
                        self._direction_trace["scope_sibling_leads"] = len(sibling_ranked)
                if rank_steps:
                    self._rank_trace = {
                        **rank_steps[-1],
                        "candidate_count": len(trace_neighbors),
                        "stages": rank_steps,
                    }
                entities = (trace_seeds + leads)[:args.limit]
                result = {"results": entities}
            elif args.mode == "symbol":
                if not args.symbol_name:
                    raise ValueError("symbol_name is required for symbol mode")
                entity_types = [args.symbol_type]
                if args.symbol_type in {"function", "class"}:
                    entity_types.append(args.symbol_type + "_template")
                entities = []
                for entity_type in entity_types:
                    result = search_symbol(
                        type=entity_type,
                        name=args.symbol_name,
                        namespace=args.namespace or None,
                        class_name=args.class_name or None,
                        limit=args.limit,
                    )
                    if result.get("error"):
                        raise ValueError(result["error"])
                    entities.extend(result.get("entities", []))
            elif args.mode == "usages":
                if not args.symbol_name:
                    raise ValueError("symbol_name is required for usages mode")
                result = find_usages(
                    type=args.symbol_type,
                    name=args.symbol_name,
                    namespace=args.namespace or None,
                    class_name=args.class_name or None,
                    limit=args.limit,
                )
                entities = result.get("usages", [])
            elif args.mode == "neighbors":
                if not args.entity_id:
                    raise ValueError("entity_id is required for neighbors mode")
                result = get_entity_neighbors(
                    entity_id=args.entity_id,
                    direction=args.direction,
                    max_depth=1,
                    max_nodes=args.limit,
                    edge_types=args.edge_types or None,
                )
                entities = result.get("neighbors", [])
            else:
                if not (args.symbol_name or args.template_arg):
                    raise ValueError(
                        "symbol_name or template_arg is required for "
                        "template_instantiations mode"
                    )
                result = find_template_instantiation(
                    display_name=args.symbol_name or None,
                    template_arg=args.template_arg or None,
                    limit=args.limit,
                )
                entities = result.get("entities", [])

            if result.get("error"):
                raise ValueError(result["error"])
            if args.mode in {"symbol", "neighbors"}:
                rag = get_graph_rag()
                entities = [self._followup_entity(entity, args, rag) for entity in entities[:args.limit]]
            if args.mode in {"function", "trace", "symbol", "neighbors"}:
                ctx = getattr(rag, "graph_ctx", None)
                for entity in entities:
                    self._attach_equivalent_function_aliases(entity, ctx)
            skeletons = [
                self._skeleton(entity) for entity in entities[: args.limit]
            ]
            if args.mode in {"function", "trace", "symbol", "neighbors"}:
                for skeleton, entity in zip(skeletons, entities):
                    skeleton.update(self._source_excerpt(entity))
            agent_view = (
                self._function_relation_frontier(
                    skeletons, getattr(rag, "graph_ctx", None),
                    source_line_limit=5 if args.mode == "trace" else 12,
                )
                if args.mode in {"function", "trace"} else None
            )
            ctx = getattr(rag, "graph_ctx", None)
            if (args.mode == "trace" and args.edge_types
                    and set(args.edge_types) <= {"CALLS", "DEP_CALL"}):
                owner_refs = [
                    row["entity_ref"] for row in skeletons
                    if row.get("id") in getattr(ctx, "classes", {})
                    and row.get("entity_ref")
                ]
                if owner_refs:
                    agent_view += (
                        "\nCALLS checks direct execution, not independent "
                        "members of the owner class. "
                        + (
                            "The `scope_sibling_lead` entries came from real "
                            "IN_CLASS edges; inspect their bodies before "
                            "including or excluding their repair scope."
                            if self._direction_trace.get("scope_sibling_leads") else
                            f"Trace owner {', '.join(owner_refs)} with "
                            "`edge_types=[\"IN_CLASS\"]` or search its members "
                            "in source before calling the scope local."
                        )
                    )
            output = json.dumps(skeletons, indent=2, ensure_ascii=False)
            evidence = self._session.record(
                "graph:" + query,
                output,
                0,
                prefix="GRAPH",
                path="<code-graph>",
                action="RepositoryGraphQuery",
                extra={"target": args.target, "purpose": args.purpose,
                       "graph_mode": args.mode,
                       "structural_rerank": self._rank_trace,
                       "trace_direction": self._direction_trace},
            )
            self._session.register_command("graph:" + query, evidence.id)
            return Observation(
                message=(
                    f"[{evidence.id}] {len(skeletons)} graph candidate(s)\n"
                    + (
                        "Requested function relation did not apply to the class "
                        "source anchor; returned real `IN_CLASS` member definitions "
                        "for the next exact function hop.\n"
                        if self._direction_trace.get("class_membership_fallback") else
                        "Requested direction had no real match; returned the "
                        f"opposite `{self._direction_trace['used']}` direction with "
                        "the original directed paths preserved.\n"
                        if self._direction_trace.get("fallback") else ""
                    )
                    + (agent_view if agent_view is not None else output or "[]")
                ),
                properties={
                    "evidence_id": evidence.id,
                    "candidate_count": len(skeletons),
                },
            )
        except Exception as exc:
            evidence = self._session.record(
                "graph:" + query, str(exc), 1,
                prefix="GRAPH", path="<code-graph>",
                action="RepositoryGraphQuery",
                extra={"target": args.target, "purpose": args.purpose,
                       "graph_mode": args.mode},
            )
            return Observation(
                message=f"[{evidence.id}] Graph query failed: {exc}",
                expect_correction=True,
                properties={"evidence_id": evidence.id, "query_failed": True},
            )


class SearchCodeIntentArgs(ActionArguments):
    """Use when a behavior-oriented code search could find an unknown source
    anchor or a related producer, consumer, or sibling after grep found the
    failure site. It internally expands selected graph paths using the Issue
    and Scout evidence, returning verified definitions and GREF leads, not proof
    of shared failure. It indexes production functions, not tests; use
    SearchTestIntent for an unknown test/fixture and ReadOnlySearch for exact
    literals or full bodies.
    """

    behavior: str = Field(
        min_length=1, max_length=800,
        description="Focused behavior or relationship to locate, including a related operation whose symbol is unknown.",
    )
    purpose: str = Field(
        default="", max_length=300,
        description="What missing source anchor would change the causal explanation or scope?",
    )

    model_config = ConfigDict(title="SearchCodeIntent", extra="forbid")


class SearchCodeIntent(Action):
    """Locate source anchors and expand issue-relevant relations in one Scout action."""

    args_schema: ClassVar[Type[ActionArguments]] = SearchCodeIntentArgs
    _graph: RepositoryGraphQuery = PrivateAttr()
    _session: InvestigationSession = PrivateAttr()

    def __init__(
        self,
        graph: RepositoryGraphQuery,
        session: InvestigationSession,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._graph = graph
        self._session = session

    def execute(self, args: SearchCodeIntentArgs, **kwargs) -> Observation:
        prior_actions = len(self._session.actions)
        result = self._graph.execute(RepositoryGraphQueryArgs(
            mode="function", query=args.behavior, limit=5,
            target=args.behavior, purpose=args.purpose or "Find a causal source anchor",
        ))
        if (len(self._session.actions) > prior_actions
                and self._session.actions[-1].get("action") == "RepositoryGraphQuery"):
            self._session.actions[-1].update(
                targeted=True, semantic_requery=True,
            )
        if (result.expect_correction or self._graph._structural_completion is None
                or not self._session.issue_text):
            return result

        try:
            from code_intent.code_intent.entity_tools import get_graph_rag

            ctx = getattr(get_graph_rag(), "graph_ctx", None)
            evidence_id = result.properties.get("evidence_id")
            evidence = next(
                (item for item in reversed(self._session.evidence)
                 if item.id == evidence_id), None,
            )
            if ctx is None or evidence is None:
                return result
            seen_sources = set()
            anchors = []
            for record in json.loads(evidence.observation):
                entity_id = record.get("id") or record.get("func_id")
                indexed = ctx.functions.get(entity_id)
                if (indexed is None or not record.get("entity_ref")
                        or record.get("source_status") != "verified"):
                    continue
                anchor = {
                    **indexed, "id": entity_id,
                    "entity_ref": record["entity_ref"],
                    "retrieval_role": record.get("retrieval_role"),
                    "graph_paths": record.get("graph_paths") or [],
                }
                source_key = self._graph._source_identity(anchor)
                if source_key in seen_sources:
                    continue
                seen_sources.add(source_key)
                anchors.append(anchor)
            if not anchors:
                return result

            question = (
                "Which connected producer, consumer, or sibling could change "
                "the issue's cause or repair scope around "
                + args.behavior[:300] + "?"
            )
            trace_args = RepositoryGraphQueryArgs(
                mode="trace", target=question, purpose=question,
            )
            context = json.loads(self._graph._trace_rank_context(trace_args, anchors))
            context.update(mode="graph_search_anchor", behavior_query=args.behavior,
                           search_purpose=args.purpose)
            ranked = self._graph._rank_trace_frontier(
                json.dumps(context, ensure_ascii=False), anchors,
                len(anchors), ctx=ctx, verify_source=True,
            )
        except Exception:
            logger.warning("Issue-guided graph anchor selection failed", exc_info=True)
            result.message += (
                "\nIssue-guided relation expansion was unavailable; the search "
                "leads remain navigation hints, not a closed repair scope."
            )
            return result

        traced = []
        trace_ids = []
        for row in ranked:
            anchor = anchors[row["index"]]
            trace_result = self._graph.execute(RepositoryGraphQueryArgs(
                mode="trace", entity_ref=anchor["entity_ref"],
                target=question, purpose=question, limit=6,
            ))
            if not trace_result.expect_correction:
                trace_id = trace_result.properties.get("evidence_id")
                trace_evidence = next(
                    (item for item in reversed(self._session.evidence)
                     if item.id == trace_id), None,
                )
                if trace_evidence and any(
                    row.get("graph_paths") and row.get("source_status") == "verified"
                    for row in json.loads(trace_evidence.observation)
                ):
                    self._session.actions[-1]["internal_search_expansion"] = True
                    traced.append(trace_result.message)
                    trace_ids.append(trace_id)
        if traced:
            initial_leads = "\n".join(
                f"- {anchor['entity_ref']} "
                f"{anchor.get('fq_name') or anchor.get('name') or anchor['id']} "
                f"[{anchor.get('def_file') or anchor.get('file')}:"
                f"{anchor.get('def_start_line') or anchor.get('start_line')}]; "
                f"{str(anchor.get('short_intent') or '')[:160]}"
                for anchor in anchors
            )
            result.message = (
                "Issue-guided relation paths expanded inside SearchCodeIntent:\n"
                + "\n\n".join(traced)
                + "\n\nInitial verified semantic anchors (navigation only):\n"
                + initial_leads
            )
            result.properties["internal_trace_evidence_ids"] = trace_ids
        else:
            result.message += (
                "\nNo issue-relevant source path was selected by internal relation "
                "expansion; do not infer that the repair scope is local."
            )
        return result


class TraceInvestigationPathArgs(ActionArguments):
    """Use when one observed graph or source anchor has a specific caller,
    callee, owner/member, or dependency relation that could change cause or
    repair scope. An owner's members can reveal independent public operations
    that a direct-caller search cannot find. The tool can expand selected
    relation bridges internally and returns short source-checked paths; use ReadOnlySearch
    to inspect a known body or literal instead.
    """

    question: str = Field(
        min_length=1,
        max_length=500,
        description=(
            "Concrete causal or scope question that one structural relation can answer."
        ),
    )
    role: Literal[
        "ingress", "state_producer", "consumer", "failure_exposure",
        "forwarder", "other",
    ] = "other"
    anchor_evidence_ids: list[str] = Field(
        default_factory=list,
        max_length=4,
        description=(
            "Optional SEARCH-N/GRAPH-N provenance. Matching evidence is attached "
            "automatically when the anchor already appears in Scout evidence."
        ),
    )
    missing_relation: str = Field(
        default="",
        max_length=600,
        description=(
            "Optional narrower description of the missing relation; defaults to question."
        ),
    )
    entity_ref: str = Field(
        default="",
        description=(
            "Preferred stable GREF-N from a prior GRAPH result. Copy it exactly."
        ),
    )
    entity_id: str = Field(
        default="",
        description=(
            "Fallback raw compiler ID from a prior GRAPH result. Do not reconstruct it; "
            "prefer entity_ref."
        ),
    )
    source_path: str = Field(
        default="",
        description=(
            "Repository-relative path from cited SEARCH evidence; pair with "
            "source_line to trace a known definition directly, even without a "
            "prior semantic graph search."
        ),
    )
    source_line: int | None = Field(
        default=None,
        ge=1,
        description="Exact cited line inside the source definition to trace.",
    )
    symbol_type: Literal["function", "class", "macro"] = "function"
    symbol_name: str = Field(default="", description="Exact source symbol when no graph id is available.")
    namespace: str = ""
    class_name: str = ""
    direction: Literal["in", "out", "both"] = "both"
    edge_types: list[GraphEdgeType] = Field(
        default_factory=list,
        max_length=5,
        description=(
            "Exact directed graph relations; use schema values such as CALLS, "
            "DEP_CALL, IN_CLASS, DEP_MEMBER_REF, DEP_TYPE_REF, or DEP_TEMPLATE_REF."
        ),
    )
    limit: int = Field(default=6, ge=2, le=6)

    model_config = ConfigDict(title="TraceInvestigationPath")


class TraceInvestigationPath(Action):
    """Resolve one exact anchor and inspect a question-guided relation frontier."""

    args_schema: ClassVar[Type[ActionArguments]] = TraceInvestigationPathArgs
    _graph: RepositoryGraphQuery = PrivateAttr()
    _session: InvestigationSession = PrivateAttr()

    def __init__(
        self,
        graph: RepositoryGraphQuery,
        session: InvestigationSession,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._graph = graph
        self._session = session

    def execute(self, args: TraceInvestigationPathArgs, **kwargs) -> Observation:
        # A GREF is the most precise anchor returned by this graph session.
        # Models sometimes repeat its symbol or source location as context;
        # treat those as redundant rather than losing the requested hop.
        redundant_anchors = []
        if args.entity_ref:
            redundant_anchors = [
                name for name, present in (
                    ("entity_id", args.entity_id),
                    ("source_path/source_line", args.source_path or args.source_line is not None),
                    ("symbol_name", args.symbol_name),
                ) if present
            ]
            args = args.model_copy(update={
                "entity_id": "", "source_path": "", "source_line": None,
                "symbol_name": "",
            })
        has_source_anchor = bool(args.source_path or args.source_line is not None)
        valid_ids = {item.id for item in self._session.evidence}
        unknown = [item for item in args.anchor_evidence_ids if item not in valid_ids]
        if unknown:
            return Observation(
                message="Unknown anchor evidence IDs: " + ", ".join(unknown),
                expect_correction=True,
            )
        if bool(args.source_path) != (args.source_line is not None):
            return Observation(
                message="Supply source_path and source_line together.",
                expect_correction=True,
            )
        selectors = [
            bool(args.entity_ref), bool(args.entity_id),
            has_source_anchor, bool(args.symbol_name),
        ]
        if sum(selectors) != 1:
            return Observation(
                message=(
                    "Trace exactly one anchor: entity_ref, entity_id, "
                    "source_path+source_line, or source symbol_name."
                ),
                expect_correction=True,
            )
        if has_source_anchor:
            normalized_path, rejection = _validated_relative_path(args.source_path)
            if rejection:
                return Observation(message=rejection, expect_correction=True)
            matching_source = [
                item for item in self._session.evidence
                if item.id.startswith("SEARCH-")
                and normalized_path in (item.locator + "\n" + item.observation)
            ]
            if not matching_source:
                return Observation(
                    message=(
                        "The source_path is not present in Scout SEARCH evidence. "
                        "Read that source location before tracing it."
                    ),
                    expect_correction=True,
                )
        anchor_evidence_ids = list(args.anchor_evidence_ids)
        if has_source_anchor:
            anchor_evidence_ids.extend(item.id for item in matching_source)
        elif args.entity_ref:
            anchor_evidence_ids.extend(
                item.id for item in reversed(self._session.evidence)
                if item.id.startswith("GRAPH-")
                and args.entity_ref in item.observation
            )
        elif args.symbol_name:
            anchor_evidence_ids.extend(
                item.id for item in reversed(self._session.evidence)
                if item.id.startswith(("SEARCH-", "GRAPH-"))
                and args.symbol_name in (item.locator + "\n" + item.observation)
            )
        anchor_evidence_ids = list(dict.fromkeys(anchor_evidence_ids))[:4]
        missing_relation = args.missing_relation or args.question
        result = self._graph.execute(RepositoryGraphQueryArgs(
            mode="trace", entity_ref=args.entity_ref, entity_id=args.entity_id,
            source_path=args.source_path, source_line=args.source_line,
            symbol_type=args.symbol_type, symbol_name=args.symbol_name,
            namespace=args.namespace, class_name=args.class_name,
            direction=args.direction, edge_types=args.edge_types,
            limit=args.limit, target=args.question, purpose=missing_relation,
        ))
        if (self._session.actions
                and self._session.actions[-1].get("action") == "RepositoryGraphQuery"):
            self._session.actions[-1].update(
                targeted=True,
                trace_failed=bool(result.expect_correction),
            )
        if result.expect_correction:
            return result
        graph_id = result.properties.get("evidence_id", "")
        path = self._session.open_investigation_path(
            question=args.question, role=args.role,
            evidence_ids=anchor_evidence_ids,
            missing_evidence=missing_relation,
            decision="traced", graph_evidence_id=graph_id,
        )
        self._session.actions[-1].update({
            "path_id": path["path_id"],
            "live_path": path,
            "targeted": True,
            "trace_failed": False,
        })
        result.message = f'[{path["path_id"]}] live relation frontier\n' + result.message
        if redundant_anchors:
            result.message += (
                "\nUsed the exact entity_ref; ignored redundant anchor fields: "
                + ", ".join(redundant_anchors)
            )
        result.properties["path_id"] = path["path_id"]
        return result


class SubmitInvestigationReportArgs(ActionArguments):
    status: str = Field(
        ...,
        pattern="^(supported|partial|unresolved|conflict)$",
        description=(
            "Overall investigation status. In TODO mode, use supported only "
            "when every TODO has useful, directly cited repository evidence."
        ),
    )
    answer: str = Field(
        ...,
        max_length=INVESTIGATION_REPORT_SUMMARY_MAX_CHARS,
        description=(
            "Final structured summary of what the current repository "
            "establishes. Do not copy private reasoning or the chronological "
            "search narrative into this field. Put evidence details in the "
            "dedicated fields. Do not write a replacement requirement or "
            "implementation plan."
        ),
    )
    repository_facts: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="Descriptive facts useful for audit, not repair requirements.",
    )
    input_path_check: InvestigationInputPathCheck | None = Field(
        default=None,
        description=(
            "Optional causal evidence: actual Issue input/receiver versus a "
            "selected consumer, with an exact gate quote and its evaluated "
            "direction. Use it when this distinction can change the explanation."
        ),
    )
    investigation_paths: list[InvestigationPath] = Field(
        default_factory=list, max_length=8,
        description="Distinct evidence-backed mechanism/scope questions inspected before forming Hypotheses. Preserve every live PATH-N across refinements. Source, negative-search, or execution verification must match verification_kind; graph adjacency alone remains unresolved.",
    )
    preserve_behavior: list[str] = Field(
        default_factory=list,
        max_length=2,
        description=(
            "Legacy TODO-mode contract candidate. Leave empty in contrastive "
            "mode, where the separate Adjudicator owns all contract effects."
        ),
    )
    preserve_evidence_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Legacy TODO-mode evidence for preserve_behavior. Leave empty in "
            "contrastive mode."
        ),
    )
    acceptance_checks: list[DeveloperAcceptanceCheck] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "Executable boundary-test candidates. Use diagnostic for legacy "
            "TODO checks. In boundary-test materialization, use f2p for a "
            "confirmed target that fails on the baseline or p2p for a "
            "confirmed preservation behavior that passes on the baseline."
        ),
    )
    conflicts_with_issue: list[str] = Field(
        default_factory=list,
        max_length=2,
        description="Observed repository conflicts; these never override the Issue.",
    )
    open_decisions: list[str] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "Intent decisions the checked-out repository cannot establish. "
            "These are persisted for audit and do not instruct Repair."
        ),
    )
    evidence_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Optional decisive SEARCH-N, RUN-N, GRAPH-N, COVERAGE-N, or "
            "XRAY-N evidence IDs. Task-level IDs are merged automatically."
        ),
    )
    task_findings: list[InvestigationTaskFinding] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "TODO-mode only: one concise evidence summary for each TASK-N. "
            "Leave empty in contrastive mode."
        ),
    )
    candidate_boundaries: list[CandidateRepairBoundary] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "Evidence-backed possible repair surfaces, not required edits. "
            "Retain a material sibling, shared-type/helper, forwarding, or "
            "shipped-form lead even when its membership remains uncertain. "
            "Cite the observed relation and say what still needs checking; "
            "do not list unrelated graph neighbors."
        ),
    )
    contrastive_witness: ContrastiveWitnessProposal | None = Field(
        default=None,
        description=(
            "Optional test-derived intent proposal. Bind one existing passing "
            "repository anchor test to ANCHOR-1, its minimal Issue-guided "
            "mutation to TARGET-1, and their observed trace difference. A "
            "separate LLM call decides whether it may affect Repair. Bind only "
            "the atomic INTENT-N units fully exercised by this exact pair."
        ),
    )

    model_config = ConfigDict(title="SubmitInvestigationReport", extra="forbid")


class SubmitInvestigationReport(Action):
    args_schema: ClassVar[Type[ActionArguments]] = SubmitInvestigationReportArgs
    _session: InvestigationSession = PrivateAttr()

    def __init__(self, session: InvestigationSession, **kwargs):
        super().__init__(**kwargs)
        self._session = session

    def execute(
        self,
        args: SubmitInvestigationReportArgs,
        **kwargs,
    ) -> Observation:
        if not self._session.evidence:
            return Observation(
                message=(
                    "Search the repository with ReadOnlySearch before answering. "
                    "Even an open question must report what was inspected."
                ),
                expect_correction=True,
            )
        # Preserve the latest schema-valid report so budget exhaustion can retain
        # its substantive findings even when a later evidence-protocol check fails.
        self._session.latest_report_candidate = args.model_dump()
        valid_ids = {item.id for item in self._session.evidence}
        path_check = args.input_path_check
        if self._session.require_input_path_check and path_check is None:
            if args.status == "supported":
                args.status = "partial"
            args.open_decisions = list(dict.fromkeys([
                *args.open_decisions,
                "The original Issue input was not connected to a verified production "
                "consumer path before Scout submitted the report.",
            ]))[:10]
        if path_check is not None:
            if self._session.issue_text and (
                not path_check.issue_input_quote.strip()
                or path_check.issue_input_quote.replace("\r\n", "\n")
                not in self._session.issue_text.replace("\r\n", "\n")
            ):
                exact_quote = _best_exact_issue_excerpt(
                    self._session.issue_text,
                    " ".join((
                        path_check.issue_input_quote,
                        path_check.actual_input,
                        path_check.consumer_input,
                    )),
                )
                if (self._session.conservatively_refine_report_protocol
                        and exact_quote):
                    path_check.issue_input_quote = exact_quote
                    self._session.actions.append({
                        "action": "RefineInvestigationReportProtocol",
                        "field": "input_path_check.issue_input_quote",
                        "strategy": "deterministic_exact_issue_excerpt",
                    })
                else:
                    return Observation(
                        message=(
                            "input_path_check.issue_input_quote must quote the ORIGINAL Issue's "
                            "input/receiver. Do not relabel a later diagnostic variant as the "
                            "Issue input. If the input is unspecified, quote the requested "
                            "operation and record that uncertainty. Copy an unchanged excerpt "
                            "from the original Issue below; do not paraphrase or reconstruct "
                            "its code indentation.\n" + self._session.issue_text[:1600]
                        ),
                        expect_correction=True,
                    )
            evidence = next((item for item in self._session.evidence
                             if item.id == path_check.condition_evidence_id), None)
            if evidence is None or (
                path_check.condition_quote.replace("\r\n", "\n")
                not in evidence.observation.replace("\r\n", "\n")
            ):
                quote_candidates = _exact_observation_quote_candidates(
                    evidence.observation if evidence else "",
                    " ".join((
                        path_check.condition_quote,
                        path_check.actual_input,
                        path_check.consumer_input,
                        path_check.evaluation,
                    )),
                )
                selected_quote = ""
                strategy = ""
                if (
                    self._session.conservatively_refine_report_protocol
                    and quote_candidates
                    and self._session.report_quote_selector is not None
                ):
                    selected_quote = self._session.report_quote_selector(
                        "input_path_check.condition_quote",
                        " ".join((
                            path_check.condition_quote,
                            path_check.actual_input,
                            path_check.consumer_input,
                            path_check.evaluation,
                        )),
                        quote_candidates,
                    )
                    if selected_quote in quote_candidates:
                        strategy = "lightweight_exact_quote_selection"
                if (
                    self._session.conservatively_refine_report_protocol
                    and quote_candidates
                    and selected_quote not in quote_candidates
                ):
                    selected_quote = quote_candidates[0]
                    strategy = "deterministic_exact_quote_unknown_downgrade"
                    path_check.matches_input = "unknown"
                    path_check.evaluation = (
                        "An exact cited excerpt was retained, but the isolated quote "
                        "selector did not validate that it establishes the submitted "
                        "input-path claim."
                    )
                    args.status = "partial"
                    args.open_decisions = list(dict.fromkeys([
                        *args.open_decisions,
                        "The input-path quote was repaired mechanically; its causal "
                        "alignment remains unresolved.",
                    ]))[:10]
                if selected_quote in quote_candidates:
                    path_check.condition_quote = selected_quote
                    self._session.actions.append({
                        "action": "RefineInvestigationReportProtocol",
                        "field": "input_path_check.condition_quote",
                        "strategy": strategy,
                        "source_evidence_id": path_check.condition_evidence_id,
                        "refine_trace": dict(
                            self._session.report_quote_refine_trace
                        ),
                    })
                else:
                    return Observation(
                        message=(
                            "input_path_check.condition_quote must be an exact excerpt of its cited "
                            "investigation observation. Copy one unchanged condition line; do not "
                            "rewrite the predicate or edit a probe file to repair this metadata. "
                            f"Cited observation {path_check.condition_evidence_id}: "
                            + (evidence.observation[:2000] if evidence else "unknown evidence ID")
                        ),
                        expect_correction=True,
                        properties={
                            "report_protocol_error": "condition_quote_not_exact",
                            "source_evidence_id": path_check.condition_evidence_id,
                        },
                    )
            if path_check.matches_input != "yes" and args.status == "supported":
                args.status = "partial"
                args.open_decisions = list(dict.fromkeys([
                    *args.open_decisions,
                    "The submitted input path remains unmatched or unverified.",
                ]))[:10]
            if path_check.verification_run_id or path_check.verification_quote:
                verification = next((item for item in self._session.evidence
                                     if item.id == path_check.verification_run_id), None)
                run = next((item for item in self._session.actions
                            if item.get("evidence_id") == path_check.verification_run_id
                            and item.get("action") == "RunInvestigationCommand"), None)
                if verification is None or run is None:
                    return Observation(
                        message=(
                            "Input-path verification must cite an actual RUN. The system "
                            "copies its output; do not reconstruct it or change a probe to "
                            "repair metadata. If no direct probe was run, leave both fields "
                            "empty. Available RUN IDs: " + ", ".join(
                                item["evidence_id"] for item in self._session.actions
                                if item.get("action") == "RunInvestigationCommand"
                            )
                        ),
                        expect_correction=True,
                    )
                path_check.verification_quote = (
                    f"[{verification.id}] exit_code={run.get('exit_code')}\n"
                    + verification.observation
                )[:800]
        submitted_path_ids = [path.path_id for path in args.investigation_paths if path.path_id]
        if len(submitted_path_ids) != len(set(submitted_path_ids)):
            return Observation(
                message="Each investigation_path.path_id must be unique.",
                expect_correction=True,
            )
        live_by_id = {
            path["path_id"]: path for path in self._session.live_investigation_paths
        }
        submitted_by_id = {
            path.path_id: path for path in args.investigation_paths if path.path_id
        }
        normalized_paths = []
        for live in self._session.live_investigation_paths:
            path = submitted_by_id.get(live["path_id"])
            if path is None:
                path = InvestigationPath.model_validate({
                    key: value for key, value in live.items()
                    if key != "checkpoint_decision"
                })
            else:
                path = path.model_copy(update={
                    "path_id": live["path_id"],
                    "role": live["role"],
                    "question": live["question"],
                    "evidence_ids": list(dict.fromkeys(
                        live["evidence_ids"] + path.evidence_ids
                    ))[:6],
                })
            normalized_paths.append(path)
        normalized_paths.extend(
            path for path in args.investigation_paths
            if path.path_id not in live_by_id
        )
        args.investigation_paths = normalized_paths[:8]
        downgraded_paths = []
        for path in args.investigation_paths:
            source = next((item for item in self._session.evidence if item.id == path.source_evidence_id), None)
            source_action = next((item for item in self._session.actions
                                  if item.get("evidence_id") == path.source_evidence_id), None)
            if path.status == "unresolved" and not path.missing_evidence.strip():
                if self._session.conservatively_refine_report_paths:
                    path.missing_evidence = (
                        "Direct source or execution evidence that resolves this path."
                    )
                else:
                    return Observation(message="An unresolved investigation_path must name its missing_evidence.", expect_correction=True)
            if path.status in {"supported", "ruled_out"} or path.source_evidence_id or path.source_quote:
                quote_matches = bool(
                    source and path.source_quote.strip()
                    and path.source_quote.replace("\r\n", "\n")
                    in source.observation.replace("\r\n", "\n")
                )
                if path.verification_kind == "source":
                    valid_verification = bool(
                        source and source.id.startswith("SEARCH-") and quote_matches
                    )
                elif path.verification_kind == "absence":
                    negative_search = bool(
                        source and source_action
                        and source.id.startswith("SEARCH-")
                        and source_action.get("action") == "ReadOnlySearch"
                        and source_action.get("exit_code") == 1
                        and str(source_action.get("command", "")).lstrip().startswith("grep ")
                        and (not source.observation.strip() or quote_matches)
                    )
                    execution_absence = bool(
                        source and source_action
                        and source.id.startswith("RUN-")
                        and source_action.get("action") == "RunInvestigationCommand"
                        and quote_matches
                    )
                    valid_verification = bool(
                        negative_search or execution_absence
                    )
                else:
                    valid_verification = bool(
                        source and source.id.startswith("RUN-")
                        and source_action
                        and source_action.get("action") == "RunInvestigationCommand"
                        and quote_matches
                    )
                if not valid_verification:
                    if self._session.conservatively_refine_report_paths:
                        path.status = "unresolved"
                        path.finding = (
                            path.finding[:900]
                            + " The submitted verification metadata did not match "
                              "the cited observation, so this path remains unresolved."
                        )[:1200]
                        path.source_evidence_id = ""
                        path.source_quote = ""
                        path.missing_evidence = (
                            path.missing_evidence
                            or "Direct source or execution evidence that verifies this path."
                        )
                        downgraded_paths.append(path.path_id or path.question[:80])
                        continue
                    return Observation(
                        message=(
                            "An investigation_path verification must match verification_kind: "
                            "source needs SEARCH-N plus an exact quote; absence needs a negative "
                            "SEARCH-N (empty output is allowed) or RUN-N plus an exact quote; "
                            "execution needs RUN-N plus an exact quote. Graph adjacency alone "
                            "stays unresolved."
                        ),
                        expect_correction=True,
                    )
        if downgraded_paths:
            note = (
                "Scout conservatively downgraded unverified investigation paths: "
                + ", ".join(downgraded_paths)
            )
            args.open_decisions = list(dict.fromkeys([
                *args.open_decisions, note,
            ]))[:10]
            if args.status == "supported":
                args.status = "partial"
        unknown_ids = [
            item for item in args.evidence_ids if item not in valid_ids
        ]
        unknown_ids.extend(
            item for item in args.preserve_evidence_ids if item not in valid_ids
        )
        if args.contrastive_witness is not None:
            witness = args.contrastive_witness
            if witness.anchor_source_evidence_id not in valid_ids:
                unknown_ids.append(witness.anchor_source_evidence_id)
            unknown_ids.extend(
                evidence_id
                for evidence_id in witness.considered_anchor_evidence_ids
                if evidence_id not in valid_ids
            )
            for candidate in (witness.anchor, witness.target):
                if candidate.run_evidence_id in valid_ids:
                    continue
                unknown_ids.append(
                    candidate.run_evidence_id
                )
        unknown_ids.extend(
            evidence_id
            for finding in args.task_findings
            for evidence_id in finding.evidence_ids
            if evidence_id not in valid_ids
        )
        unknown_ids.extend(
            evidence_id
            for boundary in args.candidate_boundaries
            for evidence_id in boundary.evidence_ids
            if evidence_id not in valid_ids
        )
        unknown_ids.extend(eid for path in args.investigation_paths for eid in path.evidence_ids if eid not in valid_ids)
        unknown_ids = list(dict.fromkeys(unknown_ids))
        if unknown_ids:
            return Observation(
                message=(
                    "Unknown evidence IDs: "
                    + ", ".join(unknown_ids)
                    + ". Cite only evidence observations produced by this "
                    "investigation."
                ),
                expect_correction=True,
            )
        unsupported_findings = [
            finding.task_id
            for finding in args.task_findings
            if finding.status != "unresolved" and not finding.evidence_ids
        ]
        if unsupported_findings:
            return Observation(
                message=(
                    "Each non-unresolved task finding must cite direct evidence: "
                    + ", ".join(unsupported_findings)
                ),
                expect_correction=True,
            )
        evidence_by_id = {item.id: item for item in self._session.evidence}
        invalid_effects: list[str] = []
        for finding in args.task_findings:
            valid_hypothesis_ids = self._session.planned_hypothesis_ids.get(
                finding.task_id,
                set(),
            )
            for effect in finding.evidence_effects:
                label = f"{finding.task_id}/{effect.evidence_id}"
                referenced_ids = set(effect.supports_hypothesis_ids) | set(
                    effect.eliminates_hypothesis_ids
                )
                evidence = evidence_by_id.get(effect.evidence_id)
                if effect.evidence_id not in finding.evidence_ids:
                    invalid_effects.append(
                        f"{label}: effect evidence is not cited by the finding"
                    )
                    continue
                if not referenced_ids or not referenced_ids <= valid_hypothesis_ids:
                    invalid_effects.append(
                        f"{label}: effect references unknown hypotheses"
                    )
                    continue
                if effect.eliminates_hypothesis_ids:
                    if effect.authority not in {
                        "issue_referenced_policy",
                        "public_contract",
                    }:
                        invalid_effects.append(
                            f"{label}: descriptive evidence cannot eliminate intent"
                        )
                        continue
                    if not effect.evidence_id.startswith("SEARCH-"):
                        invalid_effects.append(
                            f"{label}: only searched normative text can eliminate intent"
                        )
                        continue
                    if (
                        evidence is None
                        or not effect.contract_quote.strip()
                        or effect.contract_quote not in evidence.observation
                    ):
                        invalid_effects.append(
                            f"{label}: contract_quote is not exact observation text"
                        )
        if invalid_effects:
            return Observation(
                message=(
                    "Invalid hypothesis evidence effects:\n- "
                    + "\n- ".join(invalid_effects)
                    + "\nKeep descriptive evidence in finding/repository_facts "
                    "without eliminating hypotheses."
                ),
                expect_correction=True,
            )
        cited_ids = list(dict.fromkeys(
            args.evidence_ids
            + ([path_check.condition_evidence_id] if path_check is not None else [])
            + ([path_check.verification_run_id] if path_check is not None and path_check.verification_run_id else [])
            + args.preserve_evidence_ids
            + [eid for path in args.investigation_paths
               for eid in [*path.evidence_ids, *([path.source_evidence_id] if path.source_evidence_id else [])]]
            + [
                evidence_id
                for finding in args.task_findings
                for evidence_id in finding.evidence_ids
            ]
            + [
                evidence_id
                for boundary in args.candidate_boundaries
                for evidence_id in boundary.evidence_ids
            ]
            + (
                [
                    args.contrastive_witness.anchor_source_evidence_id,
                    *args.contrastive_witness.considered_anchor_evidence_ids,
                    args.contrastive_witness.anchor.run_evidence_id,
                    args.contrastive_witness.target.run_evidence_id,
                ]
                if args.contrastive_witness is not None
                else []
            )
        ))
        if self._session.require_input_path_check and not any(
            _is_concrete_production_cpp_source_evidence(
                evidence_by_id.get(evidence_id)
            )
            for evidence_id in cited_ids
        ):
            if args.status == "supported":
                args.status = "partial"
            args.open_decisions = list(dict.fromkeys([
                *args.open_decisions,
                "No cited production C/C++ definition grounds the reported causal "
                "path or repair scope; source may be collected before or after probes.",
            ]))[:10]
        if (
            self._session.planned_task_ids
            and args.preserve_behavior
            and not args.preserve_evidence_ids
        ):
            return Observation(
                message=(
                    "preserve_behavior requires direct preserve_evidence_ids. "
                    "Otherwise keep the observation in repository_facts."
                ),
                expect_correction=True,
            )
        invalid_preserve_ids = [
            evidence_id
            for evidence_id in args.preserve_evidence_ids
            if evidence_id.startswith("GRAPH-")
        ]
        if invalid_preserve_ids:
            return Observation(
                message=(
                    "GRAPH evidence describes structure and cannot establish "
                    "observable preservation behavior: "
                    + ", ".join(invalid_preserve_ids)
                ),
                expect_correction=True,
            )
        if args.status != "unresolved" and not cited_ids:
            return Observation(
                message=(
                    "Supported, partial, and conflict answers must cite at "
                    "least one observation from this investigation."
                ),
                expect_correction=True,
            )
        if self._session.planned_task_ids:
            submitted_task_ids = [
                finding.task_id for finding in args.task_findings
            ]
            missing_task_ids = [
                task_id
                for task_id in self._session.planned_task_ids
                if task_id not in submitted_task_ids
            ]
            unknown_task_ids = [
                task_id
                for task_id in submitted_task_ids
                if task_id not in self._session.planned_task_ids
            ]
            duplicate_task_ids = sorted({
                task_id
                for task_id in submitted_task_ids
                if submitted_task_ids.count(task_id) > 1
            })
            if missing_task_ids or unknown_task_ids or duplicate_task_ids:
                details = []
                if missing_task_ids:
                    details.append("missing: " + ", ".join(missing_task_ids))
                if unknown_task_ids:
                    details.append("unknown: " + ", ".join(unknown_task_ids))
                if duplicate_task_ids:
                    details.append("duplicate: " + ", ".join(duplicate_task_ids))
                return Observation(
                    message=(
                        "Submit exactly one task_findings item per planned TODO ("
                        + "; ".join(details)
                        + ")."
                    ),
                    expect_correction=True,
                )
        if args.status == "supported" and (
            args.conflicts_with_issue or args.open_decisions
        ):
            return Observation(
                message=(
                    "A supported answer cannot contain conflicts or open "
                    "decisions. Use partial, conflict, or unresolved."
                ),
                expect_correction=True,
            )
        if args.status == "supported" and any(
            finding.status != "supported" for finding in args.task_findings
        ):
            return Observation(
                message=(
                    "A supported report requires every TODO finding to be "
                    "supported. Use partial, conflict, or unresolved."
                ),
                expect_correction=True,
            )
        actions_by_evidence = {
            item.get("evidence_id"): item
            for item in self._session.actions
            if item.get("action") == "RunInvestigationCommand"
        }
        contrastive_candidates = (
            [
                args.contrastive_witness.anchor,
                args.contrastive_witness.target,
            ]
            if args.contrastive_witness is not None
            else []
        )
        if args.contrastive_witness is not None:
            witness = args.contrastive_witness
            if not witness.anchor_source_evidence_id.startswith("SEARCH-"):
                return Observation(
                    message=(
                        "The anchor test must cite a SEARCH-N read of the "
                        "existing repository test source."
                    ),
                    expect_correction=True,
                )
            invalid_alternatives = [
                evidence_id
                for evidence_id in witness.considered_anchor_evidence_ids
                if not evidence_id.startswith("SEARCH-")
            ]
            if invalid_alternatives:
                return Observation(
                    message=(
                        "Considered anchor references must cite SEARCH-N source "
                        "observations: " + ", ".join(invalid_alternatives)
                    ),
                    expect_correction=True,
                )
        for candidate in contrastive_candidates:
            action = actions_by_evidence.get(candidate.run_evidence_id)
            if action is None:
                return Observation(
                    message=(
                        f"Contrastive candidate {candidate.id} must cite an "
                        "actual RunInvestigationCommand RUN-N observation."
                    ),
                    expect_correction=True,
                )
            probe_files = action.get("frozen_probe_files") or {}
            try:
                resolved_run = _resolve_probe_run(
                    self._session,
                    candidate.run_evidence_id,
                )
            except DynamicEvidenceUnavailable:
                # A compile-time target has no runnable binary and therefore
                # cannot be dynamically rebound. It remains a valid frozen
                # witness when the single compiler command is safe.
                rejection = frozen_probe_command_rejection(
                    action.get("command", ""),
                    probe_files,
                )
                if rejection:
                    return Observation(
                        message=(
                            f"Cannot freeze contrastive candidate {candidate.id}: "
                            f"{rejection}"
                        ),
                        expect_correction=True,
                    )
                action["frozen_probe_command"] = action.get("command", "")
                action["frozen_probe_files"] = _referenced_probe_files(
                    action.get("command", ""),
                    probe_files,
                )
            else:
                if not resolved_run.output_path.startswith("/tmp/"):
                    return Observation(
                        message=(
                            f"Cannot freeze contrastive candidate {candidate.id}: "
                            "probe binaries must be written under /tmp with -o."
                        ),
                        expect_correction=True,
                    )
                action["frozen_probe_command"] = resolved_run.command
                action["frozen_probe_files"] = resolved_run.files
                action["compile_run_id"] = resolved_run.compile_run_id
                action["execution_run_id"] = resolved_run.execution_run_id
        if args.contrastive_witness is not None:
            anchor_action = actions_by_evidence.get(
                args.contrastive_witness.anchor.run_evidence_id,
                {},
            )
            target_action = actions_by_evidence.get(
                args.contrastive_witness.target.run_evidence_id,
                {},
            )
            if anchor_action.get("exit_code") != 0:
                return Observation(
                    message=(
                        "ANCHOR-1 must be a passing baseline execution with "
                        "exit code 0. Fix the anchor probe or choose a valid "
                        "repository passing test."
                    ),
                    expect_correction=True,
                )
            target_exit_code = target_action.get("exit_code")
            if not isinstance(target_exit_code, int) or target_exit_code == 0:
                return Observation(
                    message=(
                        "TARGET-1 must be a self-verifying Issue mutation that "
                        "exits nonzero on the baseline. Printing the failure and "
                        "returning 0 is not a valid target."
                    ),
                    expect_correction=True,
                )
        for check in args.acceptance_checks:
            if check.run_evidence_id not in cited_ids:
                return Observation(
                    message=(
                        f"Acceptance check '{check.description}' must include "
                        f"{check.run_evidence_id} in evidence_ids."
                    ),
                    expect_correction=True,
                )
            action = actions_by_evidence.get(check.run_evidence_id)
            if action is None:
                return Observation(
                    message=(
                        f"Acceptance check '{check.description}' must cite an "
                        "actual RunInvestigationCommand RUN-N observation."
                    ),
                    expect_correction=True,
                )
            exit_code = action.get("exit_code")
            if check.kind == "f2p" and (
                not isinstance(exit_code, int) or exit_code == 0
            ):
                return Observation(
                    message=(
                        f"F2P check {check.run_evidence_id} must fail on the "
                        "buggy baseline with a nonzero exit code."
                    ),
                    expect_correction=True,
                )
            if check.kind == "p2p" and exit_code != 0:
                return Observation(
                    message=(
                        f"P2P check {check.run_evidence_id} must pass on the "
                        "buggy baseline with exit code 0."
                    ),
                    expect_correction=True,
                )
            probe_files = action.get("frozen_probe_files") or {}
            rejection = frozen_probe_command_rejection(
                action.get("command", ""),
                probe_files,
            )
            if rejection:
                return Observation(
                    message=(
                        f"Cannot freeze {check.run_evidence_id} as an "
                        f"acceptance probe: {rejection}"
                    ),
                    expect_correction=True,
                )
            action["frozen_probe_files"] = _referenced_probe_files(
                action.get("command", ""),
                probe_files,
            )
        answer = DeveloperAnswer(
            status=args.status,
            answer=args.answer,
            repository_facts=args.repository_facts,
            input_path_check=args.input_path_check,
            investigation_paths=args.investigation_paths,
            preserve_behavior=args.preserve_behavior,
            preserve_evidence_ids=args.preserve_evidence_ids,
            acceptance_checks=args.acceptance_checks,
            conflicts_with_issue=args.conflicts_with_issue,
            open_decisions=args.open_decisions,
            repository_evidence_ids=cited_ids,
            task_findings=args.task_findings,
            candidate_boundaries=args.candidate_boundaries,
            contrastive_witness=args.contrastive_witness,
        )
        accepted_paths = {
            path.path_id: path.model_dump()
            for path in answer.investigation_paths
            if path.path_id in live_by_id
        }
        for index, live in enumerate(self._session.live_investigation_paths):
            accepted = accepted_paths.get(live["path_id"])
            if accepted is None:
                continue
            if "checkpoint_decision" in live:
                accepted["checkpoint_decision"] = live["checkpoint_decision"]
            self._session.live_investigation_paths[index] = accepted
        self._session.actions.append(
            {
                "action": "SubmitInvestigationReport",
                "report": answer.model_dump(),
            }
        )
        return Observation(
            message=args.answer,
            terminal=True,
            properties={
                "investigation_report": answer.model_dump(),
                "developer_answer": answer.model_dump(),
            },
        )


class ContainerRepositoryInvestigator:
    _MAX_CONTRAST_REFINEMENT_STEPS = 12

    def __init__(
        self,
        agent: ActionAgent,
        session: InvestigationSession,
        repository: Repository,
        docker_env: DockerEnv,
        investigation_actions: list[Action],
        submit_action: SubmitInvestigationReport,
        *,
        max_steps: int = 60,
        dynamic_evidence_tools: set[str] | None = None,
        code_index: CodeIndex | None = None,
        mode: str = "legacy",
    ):
        self.agent = agent
        self.session = session
        self.repository = repository
        self.docker_env = docker_env
        self.investigation_actions = investigation_actions
        self.submit_action = submit_action
        self.max_steps = max(2, max_steps)
        self.dynamic_evidence_tools = dynamic_evidence_tools or set()
        self.code_index = code_index
        self.mode = mode
        self.steps_used = 0
        self._current_node: Node | None = None
        self._anchor_retrieval: TestAnchorRetrieval | None = None
        self._anchor_selection: TestAnchorSelection | None = None
        self._submission_agent: ActionAgent | None = None
        self._graph_action = next(
            (action for action in investigation_actions
             if isinstance(action, RepositoryGraphQuery)),
            None,
        )
        self.session.conservatively_refine_report_paths = True
        self.session.conservatively_refine_report_protocol = True
        self._last_evidence_count = 0
        self._last_evidence_step = 0
        self._report_quote_refine_used = False
        self._report_quote_refine_trace: dict = {}
        self.session.report_quote_selector = self._select_report_quote

    @property
    def remaining_steps(self) -> int:
        return max(0, self.max_steps - self.steps_used)

    @property
    def _submission_reserve(self) -> int:
        return min(5, max(2, self.max_steps // 4))

    def _reset_step_controls(self) -> None:
        if self._graph_action is not None:
            self._graph_action.reset_entity_refs()
        self._last_evidence_count = len(self.session.evidence)
        self._last_evidence_step = self.steps_used
        self._report_quote_refine_used = False
        self._report_quote_refine_trace = {}

    def _select_report_quote(
        self,
        field_name: str,
        relevance_text: str,
        candidates: list[str],
    ) -> str:
        """Use one small call to select an existing span, never to author a quote."""
        if self._report_quote_refine_used or not candidates:
            return ""
        self._report_quote_refine_used = True
        candidate_by_id = {
            f"QUOTE-{index}": candidate
            for index, candidate in enumerate(candidates, start=1)
        }
        prompt = (
            "Select the one exact source excerpt that best supports the submitted "
            "input-path participation or guard claim. Return only a candidate ID. "
            "The candidates are immutable substrings copied from one cited repository "
            "observation. Do not rewrite source, invent a predicate, change the claim, "
            "or select a merely adjacent declaration when a guard is available.\n"
            + json.dumps({
                "field": field_name,
                "submitted_meaning": relevance_text[:2400],
                "candidates": candidate_by_id,
            }, ensure_ascii=False)
        )
        completion = self.agent.completion
        kwargs = {
            "model": completion.model,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "SelectExactSourceQuote",
                    "description": "Select one immutable exact source excerpt by ID.",
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "quote_candidate_id": {
                                "type": "string",
                                "enum": list(candidate_by_id),
                            },
                        },
                        "required": ["quote_candidate_id"],
                    },
                },
            }],
            "tool_choice": "auto",
            "max_tokens": 256,
            "timeout": 60.0,
            "max_retries": 0,
            "metadata": {
                **dict(completion.metadata or {}),
                "stage": "scout_report_exact_quote_refine",
            },
        }
        if completion.temperature is not None:
            kwargs["temperature"] = completion.temperature
        if completion.model_base_url:
            kwargs["api_base"] = completion.model_base_url
        if completion.model_api_key:
            kwargs["api_key"] = completion.model_api_key
        if "glm-5.3-flash" in completion.model:
            kwargs["extra_body"] = {
                "thinking": {
                    "type": "enabled",
                    "clear_thinking": False,
                },
                "reasoning_effort": "low",
            }

        try:
            import litellm

            response = litellm.completion(**kwargs)
            message = response.choices[0].message
            selection = {}
            calls = getattr(message, "tool_calls", None) or []
            if calls:
                selection = json.loads(calls[0].function.arguments)
            elif getattr(message, "content", None):
                selection = json.loads(message.content)
            selected_id = selection.get("quote_candidate_id", "")
            selected_quote = candidate_by_id.get(selected_id, "")
            usage = getattr(response, "usage", None)
            self._report_quote_refine_trace = {
                "model": completion.model,
                "candidate_count": len(candidate_by_id),
                "selected_id": selected_id,
                "usage": usage.model_dump() if hasattr(usage, "model_dump") else None,
            }
            self.session.report_quote_refine_trace = dict(
                self._report_quote_refine_trace
            )
            if selected_quote:
                logger.info(
                    "Scout report exact-quote protocol restored in one lightweight "
                    "selection call: field=%s selected=%s",
                    field_name,
                    selected_id,
                )
                return selected_quote
            logger.warning(
                "Lightweight Scout report quote selection returned an invalid ID; "
                "using the deterministic exact-span fallback."
            )
        except Exception as exc:
            self._report_quote_refine_trace = {
                "model": completion.model,
                "candidate_count": len(candidate_by_id),
                "error": type(exc).__name__,
            }
            self.session.report_quote_refine_trace = dict(
                self._report_quote_refine_trace
            )
            logger.warning(
                "Lightweight Scout report quote selection failed (%s); using the "
                "deterministic exact-span fallback.",
                type(exc).__name__,
            )
        return ""

    @staticmethod
    def _append_submission_budget_notice(node: Node) -> None:
        if not node.observation:
            return
        notice = (
            "You have reached the investigation budget and have one model "
            "step remaining. In your next response, call "
            "SubmitInvestigationReport with your best evidence-based report. "
            "Do not call another search, graph, edit, or command action."
        )
        node.observation.message = "\n\n".join(
            part for part in (node.observation.message, notice) if part
        )

    @staticmethod
    def _run_with_no_action_retries(
        agent: ActionAgent,
        current: Node,
        *,
        max_retries: int = 2,
    ) -> Node:
        retries = 0
        while True:
            try:
                agent.run(current)
                return current
            except RejectError as exc:
                if str(exc) != "No action found" or retries >= max_retries:
                    raise
                retries += 1
                content = current.assistant_message or ""
                if len(content) > 2000:
                    content = content[:2000] + "... [truncated]"
                logger.warning(
                    "Repository Investigator response contained no tool call; retrying "
                    "%d/%d. content=%r",
                    retries,
                    max_retries,
                    content,
                )
                retry_node = Node.stub(
                    parent=current,
                    user_message=(
                        "Your previous response did not call a tool. Call "
                        "exactly one available tool now. If your investigation "
                        "is complete, call SubmitInvestigationReport."
                    ),
                    file_context=current.file_context,
                )
                current.add_child(retry_node)
                current = retry_node

    def _run_investigation_step(self, current: Node) -> Node:
        evidence_count = len(self.session.evidence)
        if evidence_count > self._last_evidence_count:
            self._last_evidence_count = evidence_count
            self._last_evidence_step = self.steps_used
        stalled = (
            bool(self.session.evidence)
            and self.steps_used - self._last_evidence_step >= 6
            and self.remaining_steps > self._submission_reserve
        )
        if stalled:
            current.user_message = "\n\n".join(filter(None, [
                current.user_message,
                "Six Scout action steps produced no new repository evidence. Stop "
                "editing investigation files and submit the best supported report now; "
                "retain unresolved live PATH-N gaps explicitly.",
            ]))
        if self.remaining_steps <= self._submission_reserve:
            current.user_message = "\n\n".join(filter(None, [
                current.user_message,
                f"{self.remaining_steps} shared steps remain. These final steps "
                "are reserved for SubmitInvestigationReport and correcting its "
                "validation feedback. Summarize collected evidence now; unresolved "
                "mechanisms may remain explicitly unknown. Compare literal actual "
                "output with the Issue's expected output, including quoting and "
                "escaping, and preserve command exit codes. Successful execution "
                "alone does not establish the requested behavior.",
            ]))
        if self.remaining_steps > self._submission_reserve and not stalled:
            return self._run_with_no_action_retries(self.agent, current)
        if self._submission_agent is None:
            # Completion schemas are immutable after initialization. A fresh
            # configuration reads the same trajectory with only the report tool.
            self._submission_agent = ActionAgent(
                completion=self.agent.completion.clone(
                    response_schema=None, system_prompt=None,
                    model_api_key=self.agent.completion.model_api_key,
                ),
                actions=[self.submit_action],
                system_prompt=self.agent.system_prompt,
                message_generator=self.agent.message_generator,
                use_few_shots=False,
                disable_thoughts=self.agent.disable_thoughts,
            )
        return self._run_with_no_action_retries(self._submission_agent, current)

    def _best_effort_report(self, default_answer: str) -> DeveloperAnswer:
        """Retain a schema-valid report candidate when protocol repair runs out."""
        candidate = self.session.latest_report_candidate
        if candidate is None:
            return DeveloperAnswer(
                status="unresolved",
                answer=default_answer,
                task_findings=[
                    InvestigationTaskFinding(
                        task_id=task_id,
                        status="unresolved",
                        finding=(
                            "No final finding was submitted before budget exhaustion."
                        ),
                        evidence_ids=[],
                    )
                    for task_id in self.session.planned_task_ids
                ],
                repository_evidence_ids=[
                    item.id for item in self.session.evidence
                ],
                open_decisions=[
                    "Repository Investigator exhausted the shared step budget."
                ],
            )

        valid_ids = {item.id for item in self.session.evidence}
        cited_ids = [
            evidence_id
            for evidence_id in candidate.get("evidence_ids", [])
            if evidence_id in valid_ids
        ]
        for path in candidate.get("investigation_paths", []):
            cited_ids.extend(
                evidence_id
                for evidence_id in [
                    *path.get("evidence_ids", []),
                    path.get("source_evidence_id", ""),
                ]
                if evidence_id in valid_ids
            )
        for boundary in candidate.get("candidate_boundaries", []):
            cited_ids.extend(
                evidence_id
                for evidence_id in boundary.get("evidence_ids", [])
                if evidence_id in valid_ids
            )

        candidate_findings = {
            finding.get("task_id"): finding
            for finding in candidate.get("task_findings", [])
        }
        task_findings = []
        for task_id in self.session.planned_task_ids:
            finding = candidate_findings.get(task_id)
            if finding is None:
                task_findings.append(InvestigationTaskFinding(
                    task_id=task_id,
                    status="unresolved",
                    finding="The final report candidate omitted this investigation task.",
                    evidence_ids=[],
                ))
                continue
            finding_ids = [
                evidence_id
                for evidence_id in finding.get("evidence_ids", [])
                if evidence_id in valid_ids
            ]
            cited_ids.extend(finding_ids)
            task_findings.append(InvestigationTaskFinding(
                task_id=task_id,
                status=finding.get("status", "unresolved")
                if finding_ids else "unresolved",
                finding=finding.get("finding")
                or "The submitted finding could not be retained.",
                evidence_ids=finding_ids,
                evidence_effects=[],
            ))

        cited_ids = list(dict.fromkeys(cited_ids))
        open_decisions = list(dict.fromkeys([
            *candidate.get("open_decisions", []),
            "The last report candidate did not satisfy every evidence-protocol "
            "check before the shared budget expired; unverified structured claims "
            "and witness metadata were omitted.",
        ]))[:10]
        report = DeveloperAnswer(
            status="partial" if cited_ids or candidate.get("repository_facts") else "unresolved",
            answer=candidate.get("answer") or default_answer,
            repository_facts=candidate.get("repository_facts", [])[:8],
            conflicts_with_issue=candidate.get("conflicts_with_issue", [])[:2],
            open_decisions=open_decisions,
            repository_evidence_ids=cited_ids,
            task_findings=task_findings,
        )
        self.session.actions.append({
            "action": "FinalizeBestEffortInvestigationReport",
            "report": report.model_dump(),
        })
        return report

    @classmethod
    def create(
        cls,
        repository: Repository,
        docker_env: DockerEnv,
        code_index: CodeIndex | None = None,
        *,
        model: str,
        max_steps: int = 60,
        mode: str = "legacy",
        enable_graph_search: bool = True,
        message_history_type: MessageHistoryType | None = None,
        **kwargs,
    ) -> "ContainerRepositoryInvestigator":
        model_config = get_model_config(model).copy()
        model_config.pop("compact_native_prompt", None)
        model_config.pop("message_history_max_tokens", None)
        model_config.pop("few_shot_examples", None)
        configured_history_type = model_config.pop(
            "message_history_type",
            None,
        )
        model_config.update(kwargs)
        completion_model = BaseCompletionModel.create(**model_config)

        if message_history_type is None:
            message_history_type = configured_history_type
        if message_history_type is None:
            message_history_type = (
                MessageHistoryType.MESSAGES
                if completion_model.response_format == LLMResponseFormat.TOOLS
                else MessageHistoryType.REACT
            )

        session = InvestigationSession()
        submit_action = SubmitInvestigationReport(session=session)
        dynamic_evidence_tools = (
            available_dynamic_evidence_tools(docker_env)
            if mode == "contrastive"
            else set()
        )
        if mode == "contrastive":
            logger.info(
                "LLVM dynamic evidence tools available: %s",
                ", ".join(sorted(dynamic_evidence_tools)) or "none",
            )
        actions = [
            ListRepositoryTests(
                repository=repository,
                session=session,
            ),
            ReadOnlySearch(docker_env=docker_env, session=session),
            WriteInvestigationFile(
                docker_env=docker_env,
                session=session,
            ),
            ReplaceInvestigationText(
                docker_env=docker_env,
                session=session,
            ),
            RunInvestigationCommand(
                docker_env=docker_env,
                session=session,
            ),
        ]
        if "coverage" in dynamic_evidence_tools:
            actions.append(
                CollectCoverageContrast(
                    docker_env=docker_env,
                    session=session,
                )
            )
        if "xray" in dynamic_evidence_tools:
            actions.append(
                CollectXRayTraceContrast(
                    docker_env=docker_env,
                    session=session,
                )
            )
        if code_index is not None and enable_graph_search:
            graph_config = {**model_config, "max_tokens": 4096, "timeout": 60.0, "max_retries": 0}
            if "glm" in model_config["model"]:
                graph_config["extra_body"] = {
                    **(model_config.get("extra_body") or {}), "reasoning_effort": "low",
                }
            actions.append(
                RepositoryGraphQuery(
                    code_index=code_index,
                    session=session,
                    structural_completion=BaseCompletionModel.create(**graph_config),
                )
            )
        actions.append(submit_action)
        if mode == "poc":
            system_prompt = f"""You are the Intent Evidence Scout. Investigate the
Issue in `{docker_env.work_dir}` and submit a factual, source-cited report for
Hypothesis. Find the original trigger, observed failure, causal path, and
repository-connected possible repair surfaces; preserve supported alternatives
and mark unresolved policy choices. Do not design or edit the production patch.

The first reproduced fault is a localization anchor, not proof that only its
direct callers need repair. Your job is to map suspicious code, not decide the
final edit set. Keep source-backed suspected siblings, shared-type
consumers/producers, helpers, and shipped forms in `candidate_boundaries` until
their relation is checked. For each, distinguish the observed connection from
the still-unknown scope decision. No direct call to the faulty function rules
out only that call path; it does not rule out the same representation failure in
an independent public operation. Do not turn suspicion into a requirement or
list arbitrary nearby definitions.

Search focused source, tests, and public documentation with ReadOnlySearch.
Use RunInvestigationCommand for an Issue-faithful POC when execution could
resolve a material question. Write or refine only temporary investigation
files. Prefer a baseline-failing POC, but do not invent an expected outcome to
force one. A passing comparison is optional and can share the POC or come from
an existing test; never construct a mandatory pass/fail pair or test anchor.
After a compile/setup failure, check the build assumptions before changing the
POC. A proxy that changes the Issue's input representation or build mode is
diagnostic only; first separate its own fixture failure from the reported bug.
Once a focused POC establishes the reported failure, use the remaining
investigation to bound the cause and affected operations; another fixture
revision is useful only if it can change that conclusion. No benchmark patch,
later release, network, or production edit is allowed.

The POC is diagnostic, never an acceptance check. Submit
`contrastive_witness: null` and `acceptance_checks: []`. Cite exact SEARCH-N and
RUN-N observations in repository_facts and report evidence IDs. Use graph
traces for causal or scope relations when they help, then check material source
definitions. If no honest POC exists, submit source evidence and uncertainty.
Answer with SubmitInvestigationReport. Keep `answer` under
{INVESTIGATION_REPORT_SUMMARY_MAX_CHARS} characters; put detail in structured
report fields.
"""
        elif mode == "contrastive":
            if dynamic_evidence_tools == {"coverage", "xray"}:
                dynamic_workflow = (
                    "For a runnable target whose execution locations matter, "
                    "call CollectCoverageContrast. Call "
                    "CollectXRayTraceContrast only when call order or the first "
                    "runtime divergence remains unclear."
                )
            elif "coverage" in dynamic_evidence_tools:
                dynamic_workflow = (
                    "For a runnable target whose execution locations matter, "
                    "you may call CollectCoverageContrast. XRay is unavailable."
                )
            elif "xray" in dynamic_evidence_tools:
                dynamic_workflow = (
                    "For a runnable target whose call order matters, you may "
                    "call CollectXRayTraceContrast. Coverage is unavailable."
                )
            else:
                dynamic_workflow = (
                    "LLVM dynamic collection is unavailable. Use the source "
                    "contrast and ordinary compiler/runtime observations."
                )
            system_prompt = f"""You are the Intent Evidence Investigator in an
Intent Grounding pipeline. The repository is live in
`{docker_env.work_dir}` inside a disposable container. Build a factual evidence
packet from which Hypothesis can retain every repository-grounded,
Issue-compatible repair interpretation without guessing how this checkout
works. Investigate observable behavior, causal paths, and materially connected
scope; do not design the production patch or decide unspecified policy.

Use this workflow:
1. Read the Issue and retain its explicit changes, triggers, and unknowns.
   In parallel with executable investigation, collect direct public-contract
   evidence from docs, formal API declarations, explicit test/TODO contract
   text, focused examples, and behavior tests. Normative text can clarify a
   target or establish preservation. Ordinary passing behavior can identify a
   compatibility risk, but cannot create a new requirement. Record what this
   evidence establishes, what a repair could accidentally break, and which
   observable choices remain unresolved. This
   repository channel remains useful even if no honest test pair can be built.
   Current implementation shape, graph proximity, and old baseline behavior do
   not by themselves establish a new requirement.
   Before keeping competing repair interpretations coequal--for example,
   library behavior versus caller/test-harness adaptation, or an API change
   versus documentation/no change--inspect direct repository tests, docs, and
   declarations for the exact public operation, input, and error mode. If that
   evidence explicitly selects one observable behavior, retain that decision
   and mark contradicted alternatives as ruled out instead of leaving them
   equally open. Implementation shape alone cannot select among them.
   When the Issue explicitly says that the same defect or proposed repair
   applies across named sibling operations, inspect those concrete definitions
   for a repeated causal rule. If the rule is actually shared, record one
   repository fact that lists the affected sibling sites and the common rule;
   if it is not shared, record the counterexample. This fact is scope evidence
   for later adjudication, not permission to make the author's exact recipe
   normative, and does not require a separate executable probe for every sibling.
   Before claiming a complete sibling family, inspect the complete enclosing
   class, dispatch table, registration block, or equivalent definition. Follow
   an adjacent source range when the first read ends mid-block. Enumerate every
   inspected sibling in `repository_facts` and `candidate_public_forms`; never
   hide an incomplete range behind "and siblings" or "and similar". Keep the
   uninspected tail unresolved rather than asserting family completeness.
   For every Issue-authorized operation family, also inspect the public ingress,
   seed, route, or dispatcher that introduces a user value into that family.
   When an ingress uses one of the same operation tokens or dispatch hooks and
   admits the Issue's input domain, include it in the same
   `candidate_public_forms` boundary and cite the complete source definition.
   This is a family-selection intersection, not an unrelated API expansion.
   If repository evidence shows no such intersection, do not invent one. When
   the same selection rule could hijack direct entry, use the one allowed
   same-claim discriminator to exercise that ingress rather than merely adding
   more inner-family samples.
   Even when the Issue names only one trigger, inspect the complete enclosing
   definition when an exact search immediately shows that the same diagnostic,
   dispatch, state-update, or construction rule is repeated by distinct public
   argument roles or access forms. Record the concrete repeated forms as one
   non-normative `candidate_boundaries` entry so later hypotheses can compare a
   trigger-local repair with shared-rule closure. Do not promote those forms to
   Issue scope, and do not expand beyond that bounded enclosing definition by
   naming or proximity alone.
   Treat the applicability wording as scope evidence even when it appears next
   to a concrete code recipe; the recipe remains only a mechanism hypothesis.
2. Start with any pre-retrieved test shortlist supplied in the task. Inspect
   the selector's primary candidate first and baseline-validate it. When the
   shortlist is grouped by INTENT-N, compare candidates within each atomic
   intent and do not let one group's anchor stand for another group's behavior.
   Try the remaining shortlist before searching from scratch. Search beyond it when no
   listed candidate can preserve a sufficiently faithful public setup and oracle;
   a merely related but weak analogy does not block follow-up search. Use
   ListRepositoryTests with
   a focused regex over test names/descriptions before recursively grepping the
   test tree. The inventory is navigation only: inspect the selected source with
   ReadOnlySearch and execute it before using it as ANCHOR-1. Compare plausible
   focal and analogous references before selecting the closest passing anchor;
   explain why it best matches the public setup, operation, and oracle. Use
   other references as supporting evidence without turning similarity into a
   requirement.
   For a named API addition, do not infer the target family merely because an
   existing equivalent operation supplies an easy rename mutation. That test is
   useful for behavioral invariants, but a redundant alias needs Issue or
   repository support. Compare it against families where the requested name and
   contract complete an established counterpart or vocabulary gap. Preserve the
   best evidence from each plausible independent family before ranking them.
3. First execute the selected repository test or example in its native test/build
   context and confirm it passes. Then derive standalone ANCHOR-1 faithfully from
   that exact test slice, preserving its relevant fixture state, public setup,
   operation, language/build assumptions, and oracle. Reject an anchor whose
   context cannot be preserved. Preserve the test's semantics, not its harness:
   when the source uses a test framework, translate the focused setup, operation,
   and oracle into a self-contained program with its own entry point and simple
   return/assert checks. Do not compile or link framework runners, framework
   amalgamation sources, or repository test-main utilities into a standalone
   probe. Run the original test only through an existing repository build target;
   if that target is unavailable, record the limitation instead of reconstructing
   the framework by hand. Before compiling the standalone slice, inspect
   the anchor's includes and native target configuration and reuse the required
   language mode, defines, include paths, and link/header-only mode instead of
   guessing them. After one setup, compile, or link failure, classify it before
   editing the probe again. A missing framework main or framework symbol means the
   standalone extraction retained harness dependencies and must be simplified;
   it is not a reason to add more framework sources. Other missing declarations,
   unresolved product symbols, include paths, and language-mode errors require
   one focused check of the native target configuration. Repeated build-command
   guessing or source edits are not semantic evidence.
   Refine the same probe with ReplaceInvestigationText after a genuine fixture
   error; do not create numbered copies of an equivalent probe. Create
   TARGET-1 by changing one Issue-authorized
   observable predicate. Run both independently. The anchor must pass and the
   self-verifying target must fail for the Issue reason. Before freezing the
   target, identify plausible wrong repairs for the same cited claim and
   strengthen the existing fixture or oracle when it can distinguish one without
   adding an unrelated requirement. Prefer a probe that exposes the requested
   public affordance or dispatch semantics over one that only manufactures a
   false result inside a candidate detector. A compile and execution may be one `compile && run`
   When the Issue explicitly requires participation in a public lookup,
   dispatch, customization, conversion, or extension mechanism, prefer a target
   where that mechanism and the old path are both viable but observably disagree.
   Keep that discriminator controlled: both paths must receive the same object,
   cv/ref category, candidate signatures, and public oracle. Change only the
   semantic dimension named by the Issue; otherwise conversion ranking or a
   fixture difference can masquerade as lookup or dispatch behavior.
   Do not preserve the old winner merely because the baseline passes, transfer
   unstated precedence from an analogy, or invent an opt-in/new-type alternative
   without concrete Issue or public-contract evidence.
   A compile and execution may produce one RUN-N or two adjacent RUN-N records
   for the same binary; dynamic tools accept
   either form. Never modify production implementation files.
4. Compare source and ordinary compile/runtime outcomes first. For a compile-time
   failure, diagnostics are normally sufficient. {dynamic_workflow} Dynamic
   evidence suggests where behavior diverges; it does not define repair scope
   or intent.
5. When the Issue adds or changes a public operation, identify the closest
   established operation with the same observable purpose. Use exact symbol
   source search to enumerate the concrete public receivers, result/reference
   adapters, forwarding facades, and generated public forms on which that anchor
   operation is exposed. If an ownership/forwarding relation is material,
   retain it as the precise question for a TraceInvestigationPath call. Submit this as one
   `candidate_boundaries` entry. It is deliberately a conservative upper bound:
   include plausible observed forms even when receiver semantics may later
   exclude some of them, but do not pull in unrelated APIs merely because they
   share a broad Intent feature or nearby file. This structural evidence does
   not decide product intent; preserve factually supported alternatives so
   Hypothesis can branch instead of asking Repair to rediscover them. Keep a direct receiver together
   with the result/reference adapters and value/document facades on which the
   same anchor operation is directly forwarded; a local witness on only the
   receiver is not a reason to split that established family. Put genuinely
   independent API families with different ownership, lifetime, error, or
   traversal models in separate BOUNDARY-N entries and order them from strongest
   to weakest match. In each rationale, compare the requested operation's name
   and observable semantics with the anchor and say what evidence would justify
   selecting this family. Do not imply that multiple independent families must
   be implemented together. Do not report established same-operation forwarding
   closure as a product `open_decision`; preserve it in the non-normative upper
   bound so Repair can inspect it.
   Rank independent families using both behavioral similarity and API-gap
   evidence. An existing equivalent operation is a strong contrast case but is
   weak evidence for adding a second spelling; an established paired operation
   with a missing counterpart is stronger evidence for a named API addition,
   absent explicit alias or rename language in the Issue.
6. Submit direct repository contract evidence plus any valid pair, observed
   difference, and unresolved uncertainty. One pair is a lower-bound witness,
   not the whole Issue contract, and its absence does not discard repository
   evidence.
   The orchestrator may then focus another uncovered INTENT-N in the same
   container and shared budget. Each submitted report carries at most one
   independent pair; never reuse one pair merely to mark another atomic intent
   covered.
   Explicitly retain cited Issue claims and INTENT-N units that the pair does
   not exercise. Bind `issue_intent_ids` only to complete atomic obligations
   actually tested by the pair. Treat
   author-proposed causes and mechanisms as hypotheses to inspect when they can
   explain the missing behavior; do not silently discard them after the first
   compiler error. That error proves one failed requirement, not that other Issue
   clues are irrelevant. Before submission, ask whether a different observable behavior
   that violates the cited Issue claim could also pass TARGET-1, or whether the
   target silently chose an unresolved default, opt-in, compatibility,
   representation, state, or error policy. When the wrong behavior concerns the
   same cited claim, refine the target if the existing fixture can discriminate
   it. Otherwise report that limitation; do not add unrelated assertions or
   inspect wrapper/generated coverage to turn a product choice into a hard
   requirement. Inspect surfaces connected by observed call, forwarding,
   ownership, generation, packaging, or build relations when omitting one could
   erase a plausible repair branch. Exact patch completeness still belongs to
   Patch Reviewer. If no honest contrast exists, submit no witness but still
   cite decisive repository SEARCH-N observations and report repository facts,
   candidate boundaries, and open decisions.
   Do not turn the path selected on the failing baseline into desired behavior.
   A foreign overload, provider, callback, route, or intermediate result that
   reaches the eventual error is localization evidence. Unless the Issue or a
   direct public contract requires that result family, report the failed
   selection chain without saying Repair must accept the foreign result.

When the relation between the original input/receiver and a selected consumer
could change the causal explanation, record an input_path_check:
issue_input_quote copied from the original Issue, actual_input (including
object/array/scalar or receiver state), consumer_input, an exact condition_quote
with condition_evidence_id, evaluation with the concrete types/direction
substituted, matches_input, and alternative_path. For a runtime API use its
state/input guard; for an overload use its participation predicate. Do not fill
this record merely to satisfy a template. If the relation is immaterial or no
grounded predicate exists, leave it absent and preserve the relevant uncertainty;
the independent evidence reviewer decides whether that gap blocks Hypothesis.
For every exact quote, copy one contiguous substring exactly as displayed in the
cited observation. Preserve its line-number prefix, whitespace, punctuation, and
line breaks; do not paraphrase, reconstruct, or normalize it.
If the selected consumer accepts the wrong representation, inspect the matching
consumer before treating the downstream symptom as the cause. An unresolved
check is permitted with partial/unresolved status and explicit missing evidence.
Do not replace an unresolved cause with a list of guessed patch spellings.
Establish that original public input and requested result BEFORE choosing a
type-mutated anchor. The anchor is a diagnostic control, not a replacement for
the Issue program. A compile error does not authorize changing its runtime input.

After the original failure is established, use focused probes when they test a
premise that could CHANGE your causal explanation. For C++ dispatch or
member-access failures, query the actual traits/call expressions in the
checked-out headers with the Issue's concrete types (a small static_assert or
printed bool matrix is sufficient). Distinguish member name visibility from
call viability, and deserialization support from reverse serialization support.
Evaluate both the selected fallback and the existing consumer for the actual
input representation; do not infer its gate from the downstream element error.
If the exact historical compiler/configuration is unavailable or the current
environment accepts the original example, preserve that limitation and build a
minimal semantic discriminator by varying one relevant type, receiver,
language-mode, or configuration property while keeping the public operation and
oracle fixed. Use it to test the suspected selection rule. A current-environment
pass alone does not establish that no repository change is needed.
Do not hand-reimplement the repository predicate as your oracle. No patch or
golden implementation is needed for these original-header expression checks.
Every RunInvestigationCommand must classify its evidence_role explicitly. Use
reproduction only when the command executes the original Issue input or failing
focal target with a concrete expected/actual oracle. Use control for a passing,
behavior-matched anchor with a concrete oracle, even when it runs an analogous
existing API. Compiler/version checks, directory listings, build discovery, and
predicate checks are premise evidence even when they pass. TraceInvestigationPath
is an ordinary optional investigation tool and is available before or after
behavioral execution. A passing control, failing target, or production-source
read can strengthen its anchor or interpretation, but none unlocks or gates it.
Use ReadOnlySearch for repository reads; never combine a compiler/version check
and a source read in one RunInvestigationCommand. Pure read-only commands chosen
through the run tool are recorded as SEARCH evidence automatically.
Mark focused predicate and failed-prediction commands evidence_role=premise. They
are retained even if your final explanation does not cite them.
Early in the investigation, locate a narrow C/C++ definition for the concrete
receiver, selected consumer, state producer, or forwarding entry point. It is a
useful source-backed graph anchor: expand from it when callers, callees,
ownership, or dependencies could clarify the causal chain or affected scope.
Continue to validate graph-guided hypotheses against the true Issue reproduction
and inspected definitions when those become available.
For an access/adapter hypothesis, test the SAME concrete operation on the exact
underlying/base type. A passing control with a different mapped type cannot
prove that private inheritance or wrapping caused the original failure. If the
underlying expression also fails, inspect its own participation constraint.
When public ingress produces a wrapper, also bind the exact Issue expression to
the receiver actually produced by the public ingress and execute or compile that
observable before deciding which layer owns the defect. A working inner method
or alternate getter does not establish the wrapper's return or error behavior.
For conversions, retain the original public source value AND destination type;
a downstream container element is not the original source representation.
When you record an input_path_check, report the decisive RUN ID in its
verification_run_id; the system fills verification_quote from that run, so
leave that field empty. If no direct check is feasible, leave both empty and
call evaluation source reasoning, not an experimentally established cause.
Do not spend this probe budget merely rebuilding the original failing example.

Use the existing budget to check the premise, not just reproduce the last error:
- For overload/dispatch/trait failures, compare the actual input type and
  representation on the failing and passing paths. Inspect the participation
  condition immediately upstream of the symptom and the competing public path.
  Read the complete consumer body: does it accept the Issue's actual object,
  array, scalar or receiver state? If it accepts a different representation,
  search for the consumer of the actual representation and read THAT consumer's
  participation trait before proposing a cause. Evaluate the trait in the
  requested conversion direction; support for converting A to B does not imply
  support for converting B to A, especially for user-defined one-way adapters.
  Use a focused read-only type/dispatch probe for this premise instead of a
  second reproduction of the downstream error. In repository_facts, retain
  the concrete input type/representation, accepted representation, relevant
  predicate (including negation) and evaluation with SEARCH/RUN references.
  A downstream assignment/conversion error proves that path was selected, not
  that it should have been selected. Check conversion direction, constness and
  receiver type before describing a candidate path as viable. If this cannot
  be checked within budget, report the missing condition as unresolved.
- For a new or changed API, separate representation from transformation, the
  underlying receiver from its wrapper, and default behavior from an explicit
  mode. List only choices relevant to the Issue and inspected public surface.
  A legacy default and an additive option can coexist; an absent option is not
  proof that changing the default is the only interpretation. Distinguish an
  author's tentative proposal from an unconditional requested behavior using
  the exact Issue wording. Never weaken an explicit requirement for diversity.
- Keep repository_facts descriptive: observed call chain, gate and outcome with
  SEARCH/RUN IDs. Put untested causes, intended scope and policy choices in
  open_decisions with the missing evidence, not as established repository facts
  or new target/preserve requirements. Do not frame two guessed causes as an
  exhaustive set. Preserve a relevant competing explanation even if the local
  target cannot distinguish it. A passing baseline output does not require
  that same output after a newly enabled dispatch/debug/error policy.
This check replaces redundant reproduction work; it is not permission to
design patches or start another broad investigation. If evidence is absent,
submit the uncertainty rather than asserting that every alternative will pass.

Before choosing a single causal explanation, keep a small set of concrete
investigation_paths grounded in the original input and the entities you found.
Keep the PATH-N ids created by relation traces. Check whether a public
ingress, upstream selection/state producer, failing consumer, or observed public
forwarding form changes that explanation or its repair scope. A failure observed
in a consumer does not establish that the consumer first created the bad state.
Follow only relations that could change this decision. Update the paths as
source and probes support, rule out, or leave an explanation unresolved. One
local witness does not establish that other observed paths are irrelevant.
Use candidate_boundaries for observed API families and investigation_paths for
their causal/forwarding/state questions; do not duplicate their source bodies.
Also retain a separate candidate boundary for a source-backed sibling or
shipped form that could share the Issue's type, state, helper, or public
contract even if it does not call the first fault site. State the observed
connection and missing check rather than silently ruling it in or out.
Before closing, give each live PATH-N its finding, role, evidence and any precise
missing fact. Source verification needs an exact SEARCH quote. A negative SEARCH
with exit 1 and no output may verify absence; execution verification needs an
exact RUN quote. Graph edges alone are navigation. If no unresolved relation
could change the cause or scope, do not call the graph and do not invent a path.

Before submission, rehearse every live repair interpretation as a compact
behavior-cause-scope story without designing the edit. From cited evidence, it
should identify the original public trigger and observable result, the causal
decision/state/forwarding relation that could be repaired, any materially
different default/error/fallback/compatibility behavior, and the bounded
connected surfaces affected by that cause. If a branch still needs an obtainable
repository fact to tell that story, investigate the highest-value missing fact
or leave it explicit for the Reviewer. Do not force irrelevant categories into
the story and do not predict exact patch syntax.

For each branch, ask the counterfactual: if only its narrowest supported locus
changed, would an observed caller, sibling entry, wrapper, mirror, generated
form, configuration, or fallback remain inconsistent with the Issue? Follow
only evidence-suggested relations; this is not an exhaustive inventory. A
passing baseline test is a compatibility observation, not automatically a hard
preservation requirement when the Issue requests changing that behavior.

Examples: when public ingress creates a wrapper but a probe exercises only its
inner receiver, inspect the ingress-made wrapper's actual return/error path
before treating inner success as a complete causal branch. When a public
declaration is forwarded to a packaged or generated form, use the observed
forwarding/build relation to determine whether a local change reaches that form;
similar naming alone neither includes nor excludes it. When an Issue changes a
default pinned by an old test, compare any explicit mode or fallback contract
before deciding whether the old assertion is compatibility evidence or expected
to change. These illustrate reasoning choices, not mandatory gates.

Stopping condition: before submission, ask whether Hypothesis would still need
to guess a repository fact that could change the causal locus, public behavior,
or required scope, and whether a supported alternative would disappear from the
packet. Continue only for such a material fact. Submit when the remaining
uncertainty is a genuine intent choice and each live alternative has enough
source or execution evidence to become a separate hypothesis. Do not catalogue
unrelated code or probe every graph neighbor. If focused public-contract search
cannot resolve a policy choice, preserve the grounded alternatives as open
instead of choosing one.
Do not mistake fallback availability for conformance to a mechanism explicitly
named by the Issue. If an old-path-first implementation could pass TARGET-1,
compare the public operation with an independent evaluation of the named
mechanism in the same fixture. Make the requested and competing paths
simultaneously viable and observably different without using the repository
helper under repair as its own oracle. Continue only while another observation
could materially change the causal explanation or a supported branch.

Every command runs from the repository root. Ripgrep (`rg`), Git metadata and
history, network access, benchmark patches, later releases, and upstream source
are unavailable. Keep probe binaries under `/tmp`. Run compiler and probe
commands directly; observations already capture output, so do not append pipes,
`head`, status-printing `echo`, or shell redirection to evidence commands.
A freezeable compile command starts with the compiler and contains exactly one
compiler invocation. Compile and link every required probe/product source in
that invocation, optionally followed by `&&` running the resulting binary; do
not prepend `cd`, build intermediate objects, or append status-printing helpers.
Its arguments must be literal: `$()` command substitution, backticks, `;`, `||`,
pipes, and redirection are never freezeable. A RUN using any of these remains
diagnostic evidence and must not be placed in `contrastive_witness` or
`acceptance_checks`.

Reason internally using the provider's separate `reasoning_content` channel.
That channel uses the completion token budget and is preserved across tool
turns when required by the provider. Do not copy that chain of thought into a
tool argument. SubmitInvestigationReport is a separate, compact structured
artifact: `answer` is only the final summary (at most
{INVESTIGATION_REPORT_SUMMARY_MAX_CHARS} characters), while factual detail,
citations, unresolved choices, and pair analysis belong in their named fields.

Finish with SubmitInvestigationReport. Put the paired proposal only in
`contrastive_witness`; leave `task_findings`, `preserve_behavior`, and
`acceptance_checks` empty during the initial investigation because the
adjudicator owns contract effects. If a later Boundary Test Materialization
task explicitly supplies the confirmed contract, leave `contrastive_witness`
empty and populate only its requested f2p/p2p `acceptance_checks`. Put
concise public-contract observations in `repository_facts`, concrete unresolved
public choices in `open_decisions`, and cite their decisive SEARCH-N records in
`evidence_ids`. Keep `repository_facts` atomic: one public entry/seed, family rule,
downstream symptom, packaging/build fact, or preservation surface per item. Do
not combine independent structural relations in one sentence, because later
coverage must include or exclude each one separately. Put an observed sibling
operation family in
`candidate_boundaries`; each entry must cite the SEARCH-N/GRAPH-N evidence that
shows its concrete forms and Issue-relevant relation, and remains
non-normative. Keep competing
API families separate, but keep one family's direct receiver and established
result/reference/facade forwarding forms together. Rank the stronger semantic
analogue first, and describe selection evidence rather than recommending the
union of independent families. Record compared
anchor SEARCH-N IDs and the selection rationale
in the witness, together with the target discrimination analysis. If the anchor
came from the supplied shortlist, also record its TEST-CAND-N ID in
`anchor_candidate_id`; leave it empty only after explaining why the shortlist
was unusable and finding a better anchor through follow-up search. When a pair
exists, also cite its two RUN-N records and any decisive COVERAGE-N or XRAY-N.
Routine exploration need not be cited. A later single LLM adjudicator decides
whether either evidence channel may affect Repair.

Before submitting the initial contrastive report, check the exact tool schema:
- Set the top-level `status`; use `partial`, not `supported`, whenever
  `open_decisions` is non-empty.
- Put both nested `anchor` and nested `target` inside `contrastive_witness`.
  Never put either object at top level or inside `candidate_boundaries`.
- Cite only exact SEARCH-N/RUN-N IDs already returned by tools. Do not guess an
  ID from a command count.
- The anchor and target `run_evidence_id` values must be different RUN-N
  records. Each cited RUN command must itself start with a C/C++ compiler and
  compile (and optionally run) that candidate's frozen source; a run of an
  already-built binary and one combined anchor/target command are not freezeable.
- Leave `candidate_boundaries` empty only when no material, source-backed
  possible scope was observed. A cited open relation is valid even when its
  final repair membership is unknown; do not move unrelated fields into it.
"""
        else:
            system_prompt = f"""You are the Repository Investigator in an Intent
Grounding pipeline.

The repository is live in `{docker_env.work_dir}` inside a disposable container.
Investigate the supplied intent evidence TODOs and return one evidence-based
report. The Evidence TODO Planner identifies missing repository context without
preselecting alternatives; you decide how to inspect the repository efficiently.
You do not decide product policy and you do not design or implement the patch.

The Issue's explicit requirements are protected. Repository evidence can explain
terminology, establish adjacent public behavior, add a compatibility boundary,
or provide an acceptance example. It cannot cancel, narrow, or replace an
explicit request. If a dimension asks for a maintainer preference that the old
repository cannot encode, report it unresolved after enough inspection to show
the limitation. If an Issue term maps to multiple working repository mechanisms,
report that distinction as an open decision unless docs or tests explicitly
define which mechanism the term denotes. Do not merge all observed mechanisms
into a broader post-repair requirement. If the requested behavior conflicts
with a documented default while an additive or opt-in public surface is
plausible, report `partial` with an `open_decisions` item unless explicit
repository policy chooses one. Do not bind acceptance checks to the default
surface merely because it is the easiest type to probe.

For each TODO, preserve the strongest relevant observations and their
SEARCH-N/RUN-N/GRAPH-N IDs in its task finding. Look for explicit public
documentation, focused tests and examples, TODO contract text, declared API
equivalence, formal requirements, and consistent adjacent public conventions.
Search for material counterexamples before reporting a convention as stable.

Leave `evidence_effects` empty. A separate Evidence-Grounded Intent Resolver
receives your raw observations and decides whether they justify a contract
effect. RUN probes and current passing tests describe the checked-out baseline;
GRAPH results and source structure describe implementation context. Label these
limitations instead of turning them into product intent. Never infer intent
from which implementation is larger, older, easier, or shared by more callers.

Use the TODO list as a checklist, not as isolated conversations. Share
search results, graph queries, probe files, and command observations across
tasks. Process high-priority TODOs first, skip work already answered, and stop
when each evidence goal is supported, contradicted, or shown unavailable.
Submit exactly one final report with one `task_findings` item per TASK-N. Do not
call the submit tool after each TODO.

Start with surfaces likely to encode public intent: docs, focused behavior tests,
public headers, adjacent APIs, and small public-API probes. You may inspect
source, CMake/configuration, graph neighbors, wrappers, and generated surfaces
when they help interpret the Issue or explain a public convention. Do not turn a
source inventory into a requirement and do not infer intended scope solely from
similar names or class hierarchies. Patch-surface completeness belongs to the
post-repair Patch Reviewer.

Use ListRepositoryTests with a focused regular expression when you need to find
existing focal or analogous tests. It lists test names, nearby descriptions,
frameworks, and source locators without dumping implementations. Treat the list
as navigation only; inspect selected source with ReadOnlySearch and run a test or
probe before deriving behavioral evidence.

When reading is insufficient, write the smallest standalone probe or script that
checks a conjecture about current public behavior, then compile or run it. Use
WriteInvestigationFile and refine only the probe with ReplaceInvestigationText.
Do not modify product implementation files to simulate a patch. Repeated
commands and substantially overlapping reads are rejected.

Every command runs from the repository root. Ripgrep (`rg`) is not installed.
Prefer simple fixed-string searches such as
`grep -RInF -- 'literal' include src tests`, then inspect focused ranges with
`sed -n`. Single-quote C++ operators. Network, Git metadata/history, benchmark
fix/test patches, future commits, later releases, and upstream source are
unavailable. Reproducer-created files are investigation history, not original
repository evidence.

Example workflow:
TASK-1 asks whether adjacent FILE-oriented formatting APIs follow one public
argument-order convention. Search public headers, docs, and focused tests;
retain several relevant declarations and any counterexample. Report only what
the repository establishes. The Resolver will decide whether the convention is
strong enough to become a narrow behavior qualifier.

Use a small probe when it clarifies current observable behavior or supplies an
Issue-entailed acceptance scenario. An acceptance check must cite a RUN-N
produced by compiling a WriteInvestigationFile source to `/tmp` and optionally
running that exact binary. A standalone source must own its entry point and use
plain language assertions or exit status rather than repository test-framework
macros. The freezeable shape is `c++ -std=c++11 -Iinclude intent_probe.cpp -o
/tmp/intent_probe && /tmp/intent_probe`; add only product-library flags or links
confirmed by the native target, never framework runners or test-main sources.

Reason internally using the provider's separate `reasoning_content` channel;
do not copy that chain of thought into tool arguments. Finish with
SubmitInvestigationReport. `answer` is only the final summary (at most
{INVESTIGATION_REPORT_SUMMARY_MAX_CHARS} characters); put investigation detail
and evidence in the dedicated structured fields.
`task_findings` records the result and direct evidence for every TODO.
`repository_facts` contains supporting descriptive facts. `preserve_behavior`
contains only stable, observable behavior that materially bounds the requested
change; cite the direct SEARCH-N or RUN-N observations in
`preserve_evidence_ids`. GRAPH results cannot establish preservation.
`acceptance_checks` contains executable scenarios that discriminate the
requested behavior from a plausible wrong or incomplete repair and are directly
entailed by an explicit Issue target. A probe of current behavior is evidence,
not authority to define new accepted types, defaults, API shapes, or scope.
Default both contract-effect fields to empty. Put implementation details only in
facts, not in preservation or acceptance requirements. Preserve a tentative
author suggestion in the task finding when repository evidence cannot determine
whether the maintainer intends it as part of the repair. Use `open_decisions`
only as an audit explanation; the Resolver independently decides contract
effects. Use supported when the requested repository evidence was collected for
every TODO, partial when useful evidence was found but a
requested source remains unchecked, and unresolved when no useful evidence was
established.
"""
        if code_index is not None and enable_graph_search:
            system_prompt += """

Repository relation graph: use SearchCodeIntent to find a source anchor from a
focused behavior question. It is useful both when names are unknown and after
grep locates a fault but leaves a related producer, consumer, or sibling
unidentified. It searches production functions, not tests: use
SearchTestIntent for unknown behavior-matched tests/fixtures, and ReadOnlySearch
for exact literals and full source bodies.
SearchCodeIntent internally expands selected source anchors using the Issue
and current evidence. Its verified paths and GREF-N endpoints are navigation
leads, not a causal conclusion; re-query with the observed mechanism if the
first result misses the relationship.
Scope question: does the failure depend only on the found function's control
flow, or on a type, state, predicate, or representation shared with independent
public operations? A list of direct callers cannot answer the latter question.
Once you have inspected an Issue-relevant production-source definition, you
must call either SearchCodeIntent or TraceInvestigationPath before submitting
the investigation report. Ask which connected producer, consumer, owner, or
sibling could change the causal explanation or repair scope. A grep/source
match and a successful POC do not replace this graph check. For a source_path
and source_line already cited in SEARCH evidence, or a known GREF-N, use
TraceInvestigationPath on that exact anchor; use SearchCodeIntent when the
related operation must first be found by behavior. No semantic search or POC
is prerequisite for a direct source-anchored trace. This is a minimum check,
not a limit on useful follow-up graph calls. Inspect relevant returned source
before declaring local scope; if the graph fails or yields no useful relation,
record that limitation and continue with source search.
An entry-point match is the start of the investigation, not a closed repair
scope. The search result's `relation_frontier` renders actual directed edges
with traceable GREF-N endpoints. Ask which connected caller, callee, sibling
public operation, or representation helper could experience the same failure;
inspect the returned paths and trace a remaining plausible relation if needed,
then decide whether that surface belongs in the repair or can be excluded.
In particular, two public operations sharing a helper or representation are
worth comparing even when neither directly calls the other. This is a
question to investigate, not a rule that every connected node must be changed.
If a returned edge is
irrelevant, state the reason and use source search; do not traverse for its own
sake or infer behavior merely from graph adjacency.
An owning class in `scope_frontier` is a route to its member definitions, not
proof that the first matched method is the only affected one. If the failing
representation can be produced or consumed by another public member, trace
the owner's GREF with a focused question about that behavior, inspect the
returned member, and continue from its GREF only if another relation matters.
`equivalent_source_forms` lists byte-identical definitions in other source
paths. Check whether each is shipped or generated, and keep an Issue-relevant
form as a possible scope boundary while that packaging relation is unresolved;
the matching bytes alone do not make it a required edit.
For example, with a class owner shown as `GREF-42`, call
`TraceInvestigationPath(question="Which public members consume this shared
representation?", entity_ref="GREF-42", direction="in",
edge_types=["IN_CLASS"])`, replacing `GREF-42` with the actual owner ref.
If the graph does not reveal a relevant member, use a focused source search;
do not treat an unenumerated class as evidence of local scope.
Use SearchTestIntent when a behavior description could locate an analogous test
that lexical names miss; its result is a lead, not a passing oracle or required
pass/fail pair. After the required graph check, use further graph calls when
they answer a remaining causal or scope question. These tools are available
from the beginning
and do not depend on first producing a pass/fail pair, reading source in a
particular order, or reaching a workflow checkpoint. There is no graph-call quota
apart from the shared Scout action budget.

For example, if an Issue describes a value created by one public operation and
later consumed by another, search for the behavior that creates the value,
inspect the returned definition, then trace its actual caller/consumer relation
only if that relation changes the proposed cause or required scope. If a local
fix appears sufficient, consider whether another public entry still uses the
same faulty state, representation, or predicate; a focused semantic search can
find an unknown sibling, but only inspected source or execution can establish
that it shares the cause. When two lookup or dispatch paths could both apply,
a test with only one available path cannot establish their precedence. Read the
connected source and compare it with the original Issue input before recording
a causal claim. If a test describes the same operation in different vocabulary,
SearchTestIntent can locate it, but inspect its assertion before relying on it.

Early in the investigation, look for a concrete code anchor connected to the
Issue: an implementation location, exact symbol, or stable GREF-N from a
focused search or earlier trace. Once the source definition is inspected, use
that anchor for the required graph check to clarify the call chain, causal
path, affected consumers, ownership, type/template dependencies, compatibility
surface, or repair scope. A source_path plus exact source_line already present
in SEARCH evidence is the strongest anchor; an exact symbol or stable GREF-N
is also valid.
Only `question` and one anchor form are essential. Add direction, edge types,
role, or a narrower missing_relation when they materially disambiguate the
request, and never invent or edit an entity ID.

Each trace resolves one exact seed and may follow selected structural bridges
internally, using the Issue and your concrete question to return a small set of
directed, source-checked paths. Inspect the returned definition before treating
a path as evidence of a shared mechanism. If another relation remains material,
trace again from its GREF-N; do not repeat an identical query or collect
neighbors without a concrete question.
For example, after a result names `GREF-7`, call
`TraceInvestigationPath(question="Does this consumer share the failing state?",
entity_ref="GREF-7")`. The GREF is the anchor; do not also fill in its symbol
name or source location. If only source search found the fault, a cited
`src/handler.cpp:42` can be traced directly with
`TraceInvestigationPath(question="Which public caller reaches this handler?",
source_path="src/handler.cpp", source_line=42, direction="in")`.

Examples:
- A public ingress produces a wrapper whose method and inner receiver have
  different return or error contracts: trace outgoing CALLS or DEP_CALL from
  the wrapper, read the returned callee definition, then exercise the exact
  Issue expression on the ingress-produced receiver.
- A helper's behavior appears faulty and repair impact depends on its consumers:
  trace incoming CALLS from the helper, then inspect the callers that determine
  scope.
- A formatting or conversion path crosses an owning type or template boundary:
  trace the relevant IN_CLASS, DEP_MEMBER_REF, DEP_TYPE_REF, or DEP_TEMPLATE_REF
  relation and continue from the returned entity if another hop matters.
- A template/SFINAE failure is reported in a forwarding function but the causal
  question is which trait admitted or rejected the concrete type: use the
  compiler stack and inspected predicate to re-anchor at that trait/detector,
  then expand its template/type dependencies. No edge from the forwarder does
  not establish that the dependency is absent.
- A requested API is absent and source search already proves that absence: the
  graph cannot invent the missing entity; inspect neighboring implementations,
  tests, and public contracts instead.

A graph edge is navigation evidence, not proof of runtime behavior, desired
intent, or repair scope. Read returned definitions and confirm material claims
with source or execution evidence. Use direct source search for lexical facts in
the same definition, broad discovery, documentation/contracts, runtime truth,
or unindexed/missing entities.
Before concluding that the repair is confined to the first matched function,
revisit any graph-linked definition that plausibly shares the Issue's failing
type assumption, state, predicate, or public contract. Trace from its GREF-N
to answer the specific relation question and inspect source; if the relation
is not material, say why. If a plausible peer remains unread, label scope
uncertain instead of asserting that it is unaffected. Likewise, a second
source copy returned for the same public operation is a packaging question to
check, not evidence that the first file automatically updates the second.
If a one-way trace has no real match, the tool may return the opposite direction
once. This is marked by trace_requested_direction, trace_direction_used, and
direction_fallback, while graph_paths retain the actual directed edge. Interpret
that result by its real source/target direction; do not relabel an incoming caller
as an outgoing consumer.
If neither direction matches, use the returned available-relation diagnostic to
decide whether the anchor, relation kind, or question was wrong. Re-anchor at a
more causal symbol when that relation would change a live hypothesis; do not
treat a zero-edge result as proof of absence and do not retry merely to consume
graph results.
If a returned location is a declaration, read its implementation before marking
the path supported. Stop when the missing relation is answered or record the gap;
text search remains available for unindexed relations and source checks.
Generated intent describes existing code and may be inaccurate. Neither an
intent summary nor a graph edge establishes desired behavior or preservation;
verify the original definitions and retain the Issue's authority. If retrieval
fails or is irrelevant, record that limitation and continue with text search.
"""
        message_generator = MessageHistoryGenerator.create(
            message_history_type=message_history_type,
            include_file_context=False,
            thoughts_in_action=False,
        )
        graph_action = next(
            (action for action in actions
             if isinstance(action, RepositoryGraphQuery)),
            None,
        )
        trace_action = (
            TraceInvestigationPath(
                graph=graph_action,
                session=session,
            )
            if graph_action is not None else None
        )
        semantic_action = (
            SearchCodeIntent(graph=graph_action, session=session)
            if graph_action is not None else None
        )
        test_intent_action = (
            SearchTestIntent(
                repository=repository, session=session,
                index_name=getattr(code_index, "_index_name", None),
            )
            if graph_action is not None else None
        )
        # Raw semantic/neighbor graph modes stay hidden. Scout decides whether and
        # when exact relation traces help its current causal or scope question.
        agent_actions = [
            action for action in actions
            if not isinstance(action, RepositoryGraphQuery)
        ]
        for action in (semantic_action, test_intent_action, trace_action):
            if action is not None:
                agent_actions.insert(agent_actions.index(submit_action), action)
        agent = ActionAgent(
            completion=completion_model,
            actions=agent_actions,
            system_prompt=system_prompt,
            message_generator=message_generator,
            use_few_shots=False,
            disable_thoughts=True,
        )
        return cls(
            agent=agent,
            session=session,
            repository=repository,
            docker_env=docker_env,
            investigation_actions=actions,
            submit_action=submit_action,
            max_steps=max_steps,
            dynamic_evidence_tools=dynamic_evidence_tools,
            code_index=code_index,
            mode=mode,
        )

    def retrieve_test_anchor_candidates(
        self,
        issue: str,
        issue_evidence: IssueEvidenceFrame,
    ) -> TestAnchorRetrieval:
        return TestAnchorRetriever(
            repository=self.repository,
            index_name=getattr(self.code_index, "_index_name", None),
        ).retrieve(issue, issue_evidence)

    def set_test_anchor_context(
        self,
        retrieval: TestAnchorRetrieval | None,
        selection: TestAnchorSelection | None,
    ) -> None:
        self._anchor_retrieval = retrieval
        self._anchor_selection = selection

    def investigate(
        self,
        issue: str,
        issue_evidence: IssueEvidenceFrame,
        reproduction: dict,
        plan: IntentInvestigationPlan,
    ) -> tuple[DeveloperAnswer, list[RepositoryEvidence], list[dict]]:
        """Execute one shared dimension plan and submit one evidence report."""
        self.session.reset()
        self.session.issue_text = issue
        self.session.planned_task_ids = [task.id for task in plan.tasks]
        self.session.planned_hypothesis_ids = {
            task.id: {hypothesis.id for hypothesis in task.hypotheses}
            for task in plan.tasks
        }
        self.steps_used = 0
        self._current_node = None
        self._reset_step_controls()
        current = Node.stub(
            user_message=(
                build_repository_investigation_task(
                    issue,
                    issue_evidence,
                    reproduction,
                    plan,
                )
                + self._plan_budget_prompt()
            ),
            file_context=FileContext(
                repo=self.repository,
                runtime=self.docker_env,
            ),
        )

        for _ in range(self.remaining_steps):
            current = self._run_investigation_step(current)
            self.steps_used += 1
            if current.is_terminal():
                break
            if self.remaining_steps == 1:
                self._append_submission_budget_notice(current)
            child = Node.stub(
                parent=current,
                file_context=current.file_context,
            )
            current.add_child(child)
            current = child

        self._current_node = current
        report_data = None
        if current.observation:
            report_data = current.observation.properties.get(
                "investigation_report"
            )
        if report_data:
            report = DeveloperAnswer.model_validate(report_data)
        else:
            report = self._best_effort_report(
                "The repository investigation exhausted its shared budget "
                "without an accepted final report."
            )
        return (
            report,
            list(self.session.evidence),
            list(self.session.actions),
        )

    def investigate_contrastively(
        self,
        issue: str,
        issue_evidence: IssueEvidenceFrame,
        reproduction: dict,
    ) -> tuple[DeveloperAnswer, list[RepositoryEvidence], list[dict]]:
        """Collect repository contract evidence and an optional test witness."""
        self.session.reset()
        self.session.issue_text = issue
        self.steps_used = 0
        self._current_node = None
        self._reset_step_controls()
        current = Node.stub(
            user_message=(
                (
                    build_poc_investigation_task(
                        issue, issue_evidence, reproduction,
                    )
                    if self.mode == "poc"
                    else build_contrastive_investigation_task(
                        issue,
                        issue_evidence,
                        reproduction,
                        self.dynamic_evidence_tools,
                        self._anchor_retrieval,
                        self._anchor_selection,
                    )
                )
                + self._plan_budget_prompt()
            ),
            file_context=FileContext(
                repo=self.repository,
                runtime=self.docker_env,
            ),
        )

        for _ in range(self.remaining_steps):
            current = self._run_investigation_step(current)
            self.steps_used += 1
            if current.is_terminal():
                break
            if self.remaining_steps == 1:
                self._append_submission_budget_notice(current)
            child = Node.stub(
                parent=current,
                file_context=current.file_context,
            )
            current.add_child(child)
            current = child

        self._current_node = current
        report_data = None
        if current.observation:
            report_data = current.observation.properties.get(
                "investigation_report"
            )
        if report_data:
            report = DeveloperAnswer.model_validate(report_data)
        else:
            report = self._best_effort_report(
                "The intent evidence investigation exhausted its budget "
                "without an accepted final report."
            )
        return (
            report,
            list(self.session.evidence),
            list(self.session.actions),
        )

    def refine_contrastively(
        self,
        issue: str,
        issue_evidence: IssueEvidenceFrame,
        previous_report: DeveloperAnswer,
        previous_decision: ContrastiveIntentDecision,
        pair_status: str,
    ) -> tuple[DeveloperAnswer, list[RepositoryEvidence], list[dict]]:
        """Continue the same trajectory for one adjudicator evidence request."""
        if self._current_node is None or self.remaining_steps <= 0:
            return (
                previous_report,
                list(self.session.evidence),
                list(self.session.actions),
            )

        available_steps = min(
            self.remaining_steps,
            self._MAX_CONTRAST_REFINEMENT_STEPS,
        )
        focused_budget_prompt = (
            "\n\nFocused continuation budget: use at most "
            f"{available_steps} action steps to answer only the adjudicator's "
            "single evidence request. Reuse the existing witness and files; "
            "do not explore adjacent APIs, depths, modes, or mechanisms. "
            "Submit the best replacement report before this focused limit; "
            "unused shared investigation steps remain available."
        )
        if available_steps == 1:
            focused_budget_prompt += (
                " This is the only remaining focused step: call "
                "SubmitInvestigationReport now using the evidence already "
                "collected."
            )

        current = Node.stub(
            parent=self._current_node,
            user_message=(
                build_contrastive_refinement_task(
                    issue,
                    issue_evidence,
                    previous_report.model_dump(),
                    previous_decision,
                    pair_status,
                )
                + focused_budget_prompt
            ),
            file_context=self._current_node.file_context,
        )
        self._current_node.add_child(current)

        for step_index in range(available_steps):
            current = self._run_investigation_step(current)
            self.steps_used += 1
            if current.is_terminal():
                break
            if step_index == available_steps - 2:
                self._append_submission_budget_notice(current)
            child = Node.stub(
                parent=current,
                file_context=current.file_context,
            )
            current.add_child(child)
            current = child

        self._current_node = current
        report_data = None
        if current.observation:
            report_data = current.observation.properties.get(
                "investigation_report"
            )
        if report_data:
            report = DeveloperAnswer.model_validate(report_data)
        else:
            request = previous_decision.followup_request
            report = DeveloperAnswer(
                status="unresolved",
                answer=(
                    "The focused contrast refinement exhausted the remaining "
                    "shared budget without submitting a replacement report."
                ),
                repository_evidence_ids=[
                    item.id for item in self.session.evidence
                ],
                open_decisions=[
                    (
                        request.objective
                        if request is not None
                        else "The requested contrast refinement remains unresolved."
                    )
                ],
            )
        return (
            report,
            list(self.session.evidence),
            list(self.session.actions),
        )

    def record_evidence_sufficiency_review(
        self,
        review: EvidenceSufficiencyReview,
        *,
        phase: str,
    ) -> None:
        """Persist the independent verdict beside the Scout action history."""
        self.session.actions.append({
            "action": "ReviewEvidenceSufficiency",
            "phase": phase,
            "review": review.model_dump(),
            "evidence_count": len(self.session.evidence),
        })

    def refine_evidence_sufficiency(
        self,
        issue: str,
        issue_evidence: IssueEvidenceFrame,
        previous_report: DeveloperAnswer,
        review: EvidenceSufficiencyReview,
    ) -> tuple[DeveloperAnswer, list[RepositoryEvidence], list[dict]]:
        """Let the same Scout autonomously answer one material evidence gap."""
        if self._current_node is None or review.next_investigation is None:
            return (
                previous_report,
                list(self.session.evidence),
                list(self.session.actions),
            )

        # A terminal initial report may consume the shared Scout budget. Give
        # this independent review one real continuation window rather than
        # making readiness depend on how late the first report was submitted.
        available_steps = self._MAX_CONTRAST_REFINEMENT_STEPS
        self.max_steps += available_steps
        focused_budget_prompt = (
            "\n\nEvidence-review continuation allowance: up to "
            f"{available_steps} additional action steps are available. Choose "
            "the investigation tools yourself and submit the replacement "
            "report before this allowance ends."
        )
        current = Node.stub(
            parent=self._current_node,
            user_message=(
                build_evidence_sufficiency_refinement_task(
                    issue,
                    issue_evidence,
                    previous_report.model_dump(),
                    review,
                )
                + focused_budget_prompt
            ),
            file_context=self._current_node.file_context,
        )
        self._current_node.add_child(current)

        for step_index in range(available_steps):
            current = self._run_investigation_step(current)
            self.steps_used += 1
            if current.is_terminal():
                break
            if step_index == available_steps - 2:
                self._append_submission_budget_notice(current)
            child = Node.stub(
                parent=current,
                file_context=current.file_context,
            )
            current.add_child(child)
            current = child

        self._current_node = current
        report_data = None
        if current.observation:
            report_data = current.observation.properties.get(
                "investigation_report"
            )
        if report_data:
            report = DeveloperAnswer.model_validate(report_data)
        else:
            report = self._best_effort_report(
                "The evidence-review continuation ended without an accepted "
                "replacement report."
            )
        return (
            report,
            list(self.session.evidence),
            list(self.session.actions),
        )

    def investigate_next_intent(
        self,
        issue: str,
        issue_evidence: IssueEvidenceFrame,
        intent_unit: IssueIntentUnit,
    ) -> tuple[DeveloperAnswer, list[RepositoryEvidence], list[dict]]:
        """Continue the shared trajectory with one independent intent pair."""
        if self._current_node is None or self.remaining_steps <= 0:
            return (
                DeveloperAnswer(
                    status="unresolved",
                    answer=(
                        f"No shared investigation budget remained for "
                        f"{intent_unit.id}."
                    ),
                    repository_evidence_ids=[
                        item.id for item in self.session.evidence
                    ],
                    open_decisions=[
                        f"{intent_unit.id} has no independently investigated "
                        "test witness."
                    ],
                ),
                list(self.session.evidence),
                list(self.session.actions),
            )

        current = Node.stub(
            parent=self._current_node,
            user_message=(
                build_next_intent_witness_task(
                    issue,
                    issue_evidence,
                    intent_unit,
                    self._anchor_retrieval,
                    self._anchor_selection,
                )
                + self._plan_budget_prompt()
            ),
            file_context=self._current_node.file_context,
        )
        self._current_node.add_child(current)

        for _ in range(self.remaining_steps):
            current = self._run_investigation_step(current)
            self.steps_used += 1
            if current.is_terminal():
                break
            if self.remaining_steps == 1:
                self._append_submission_budget_notice(current)
            child = Node.stub(
                parent=current,
                file_context=current.file_context,
            )
            current.add_child(child)
            current = child

        self._current_node = current
        report_data = None
        if current.observation:
            report_data = current.observation.properties.get(
                "investigation_report"
            )
        if report_data:
            report = DeveloperAnswer.model_validate(report_data)
        else:
            report = DeveloperAnswer(
                status="unresolved",
                answer=(
                    f"The focused investigation for {intent_unit.id} "
                    "exhausted the remaining shared budget without a report."
                ),
                repository_evidence_ids=[
                    item.id for item in self.session.evidence
                ],
                open_decisions=[
                    f"{intent_unit.id} remains without an independently "
                    "submitted test witness."
                ],
            )
        return (
            report,
            list(self.session.evidence),
            list(self.session.actions),
        )

    def materialize_intent_boundary(
        self,
        issue: str,
        issue_evidence: IssueEvidenceFrame,
        contract: RepairIntentContract,
        decision: ContrastiveIntentDecision,
        routing: IntentValidationRouting,
    ) -> tuple[DeveloperAnswer, list[RepositoryEvidence], list[dict]]:
        """Continue the trajectory and encode confirmed boundaries as probes."""
        task = build_intent_boundary_test_task(
            issue,
            issue_evidence,
            contract,
            decision,
            routing,
        )
        return self._run_boundary_materialization_task(task)

    def refine_intent_boundary(
        self,
        issue: str,
        issue_evidence: IssueEvidenceFrame,
        contract: RepairIntentContract,
        decision: ContrastiveIntentDecision,
        routing: IntentValidationRouting,
        previous_report: DeveloperAnswer,
        assessment: IntentBoundaryTestAssessment,
    ) -> tuple[DeveloperAnswer, list[RepositoryEvidence], list[dict]]:
        """Revise rejected checks using semantic-admission feedback."""
        task = build_intent_boundary_test_refinement_task(
            issue,
            issue_evidence,
            contract,
            decision,
            routing,
            previous_report,
            assessment,
        )
        return self._run_boundary_materialization_task(task)

    def _run_boundary_materialization_task(
        self,
        task: str,
    ) -> tuple[DeveloperAnswer, list[RepositoryEvidence], list[dict]]:
        if self._current_node is None or self.remaining_steps <= 0:
            return (
                DeveloperAnswer(
                    status="unresolved",
                    answer=(
                        "No shared investigation budget remained for boundary-test "
                        "materialization."
                    ),
                    repository_evidence_ids=[
                        item.id for item in self.session.evidence
                    ],
                ),
                list(self.session.evidence),
                list(self.session.actions),
            )

        current = Node.stub(
            parent=self._current_node,
            user_message=task + self._plan_budget_prompt(),
            file_context=self._current_node.file_context,
        )
        self._current_node.add_child(current)

        for _ in range(self.remaining_steps):
            current = self._run_investigation_step(current)
            self.steps_used += 1
            if current.is_terminal():
                break
            if self.remaining_steps == 1:
                self._append_submission_budget_notice(current)
            child = Node.stub(
                parent=current,
                file_context=current.file_context,
            )
            current.add_child(child)
            current = child

        self._current_node = current
        report_data = None
        if current.observation:
            report_data = current.observation.properties.get(
                "investigation_report"
            )
        if report_data:
            report = DeveloperAnswer.model_validate(report_data)
        else:
            report = DeveloperAnswer(
                status="unresolved",
                answer=(
                    "Boundary-test materialization exhausted the remaining "
                    "shared budget without a submitted report."
                ),
                repository_evidence_ids=[
                    item.id for item in self.session.evidence
                ],
            )
        return (
            report,
            list(self.session.evidence),
            list(self.session.actions),
        )

    def answer(
        self,
        issue: str,
        issue_evidence: IssueEvidenceFrame,
        reproduction: dict,
        question: IntentExplorationObjective,
    ) -> tuple[DeveloperAnswer, list[RepositoryEvidence], list[dict]]:
        current = self._start_round(
            issue,
            issue_evidence,
            reproduction,
            question,
        )
        available_steps = self.remaining_steps
        for _ in range(available_steps):
            current = self._run_investigation_step(current)
            self.steps_used += 1
            if current.is_terminal():
                break
            if self.remaining_steps == 1:
                self._append_submission_budget_notice(current)
            child = Node.stub(
                parent=current,
                file_context=current.file_context,
            )
            current.add_child(child)
            current = child

        self._current_node = current

        answer_data = None
        if current.observation:
            answer_data = current.observation.properties.get("developer_answer")
        if answer_data:
            answer = DeveloperAnswer.model_validate(answer_data)
        else:
            answer = DeveloperAnswer(
                status="unresolved",
                answer=(
                    "The repository investigation did not establish an answer "
                    "before the shared step budget was exhausted."
                ),
                repository_evidence_ids=[
                    item.id for item in self.session.evidence
                ],
                open_decisions=[
                    "Repository Investigator exhausted its shared step budget without "
                    "submitting an answer."
                ],
            )
        return (
            answer,
            list(self.session.evidence),
            list(self.session.actions),
        )

    def _start_round(
        self,
        issue: str,
        issue_evidence: IssueEvidenceFrame,
        reproduction: dict,
        question: IntentExplorationObjective,
    ) -> Node:
        if self._current_node is None:
            self.session.reset()
            self.session.issue_text = issue
            self._reset_step_controls()
            return Node.stub(
                user_message=(
                    build_developer_task(
                        issue,
                        issue_evidence,
                        reproduction,
                        question,
                    )
                    + self._shared_budget_prompt()
                ),
                file_context=FileContext(
                    repo=self.repository,
                    runtime=self.docker_env,
                ),
            )

        followup = Node.stub(
            parent=self._current_node,
            user_message=(
                build_developer_followup_task(question)
                + self._shared_budget_prompt()
            ),
            file_context=self._current_node.file_context,
        )
        self._current_node.add_child(followup)
        return followup

    def _shared_budget_prompt(self) -> str:
        prompt = (
            "\n\nShared investigation budget: "
            f"{self.remaining_steps} action steps remain for this objective "
            "and any later objective. Submit as soon as this objective is "
            "answered; unused steps remain available to the next objective."
        )
        if self.remaining_steps <= 1:
            prompt += (
                " This is the final action step: do not run another search or "
                "command; call SubmitInvestigationReport using the evidence already "
                "collected."
            )
        return prompt

    def _plan_budget_prompt(self) -> str:
        return (
            "\n\nShared investigation budget: "
            f"{self.remaining_steps} action steps remain. Allocate them by "
            "evidence value and call SubmitInvestigationReport as soon as the "
            "requested evidence is established or shown unavailable; unused "
            f"steps need not be consumed. The final {self._submission_reserve} "
            "steps are reserved for "
            "report submission and validation correction. Separate literal "
            "expected and actual output on the same input/mode, including "
            "quoting, escaping and exit status; successful execution alone "
            "does not establish the requested behavior."
        )


# Compatibility aliases for external callers that have not migrated their
# imports. Runtime prompts and tool schemas use the role-neutral names above.
DeveloperSession = InvestigationSession
DeveloperGraphQueryArgs = RepositoryGraphQueryArgs
DeveloperGraphQuery = RepositoryGraphQuery
SubmitDeveloperAnswerArgs = SubmitInvestigationReportArgs
SubmitDeveloperAnswer = SubmitInvestigationReport
ContainerDeveloperAgent = ContainerRepositoryInvestigator
