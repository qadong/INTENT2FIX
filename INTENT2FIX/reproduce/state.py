import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Dict, Iterable, List, Optional

from INTENT2FIX.reproduce.intent_schema import ExecutionRecord


def normalize_artifact_path(path: str) -> Optional[str]:
    if not path or os.path.isabs(path) or "\\" in path:
        return None
    candidate = PurePosixPath(path)
    if any(part in {"", ".", ".."} for part in candidate.parts):
        return None
    normalized = str(candidate)
    return normalized if normalized not in {"", "."} else None


def is_dedicated_reproduction_path(path: str) -> bool:
    normalized = normalize_artifact_path(path)
    if not normalized:
        return False
    candidate = PurePosixPath(normalized)
    return (
        candidate.name.lower().startswith("repro")
        and candidate.suffix.lower() in {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp"}
    )


def command_references_artifact(command: str, path: str) -> bool:
    normalized = normalize_artifact_path(path)
    if not normalized:
        return False
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        tokens = command.split()
    source_suffixes = {".c", ".cc", ".cpp", ".cxx", ".C"}
    source_path = PurePosixPath(normalized)
    derived_executable = source_path.stem if source_path.suffix in source_suffixes else None
    for raw_token in tokens:
        token = raw_token.strip("'\"(){}[];,:")
        if token in {normalized, f"./{normalized}"}:
            return True
        if token.endswith(f"/{normalized}"):
            return True
        if derived_executable and token.startswith(("./", "/")):
            if PurePosixPath(token).name == derived_executable:
                return True
    return False


def command_removes_artifact(command: str, paths: Iterable[str]) -> bool:
    for segment in re.split(r"(?:&&|\|\||[;|])", command):
        removes_directly = re.search(r"(^|\s)(rm|unlink|mv)(\s|$)", segment)
        removes_via_find = (
            re.search(r"(^|\s)find(\s|$)", segment)
            and re.search(r"(^|\s)-delete(\s|$)", segment)
        )
        if not removes_directly and not removes_via_find:
            continue
        if any(command_references_artifact(segment, path) for path in paths):
            return True
    return False


def is_reproduction_command(command: str) -> bool:
    return bool(re.search(
        r"(^|[;&|]\s*)(?:(?:\S*/)?(?:g\+\+|c\+\+|clang\+\+|gcc|clang|cc)(?:\s|$)|(?:\./|/)\S+)",
        command,
    ))


def is_build_or_test_command(command: str) -> bool:
    compiler_with_source = re.search(
        r"(^|[;&|]\s*)(?:\S*/)?(?:g\+\+|c\+\+|clang\+\+|gcc|clang|cc)(?:\s|$)"
        r"(?=[^;&|]*(?:\.(?:c|cc|cpp|cxx|C)\b|\s-c(?:\s|$)))",
        command,
    )
    build_or_test_driver = re.search(
        r"(^|[;&|]\s*)(?:(?:\S*/)?cmake\s+--build\b|(?:\S*/)?(?:make|ninja|ctest)(?:\s|$))",
        command,
    )
    return bool(compiler_with_source or build_or_test_driver)


def shell_file_write_reason(command: str, artifact_paths: Iterable[str]) -> Optional[str]:
    if "<<" in command:
        return "RunCommand cannot use heredocs; use a file editing action."

    sanitized = re.sub(r"\d*>\s*&\d+", "", command)
    sanitized = re.sub(r"\d*>\s*/dev/null", "", sanitized)
    if re.search(r"(^|[\s;|&])\d*>>?\s*\S+", sanitized):
        return "RunCommand cannot redirect output to files; use a file editing action."
    if re.search(r"(^|[;|&]\s*)tee(?:\s|$)", command):
        return "RunCommand cannot write files with tee; use a file editing action."
    for segment in re.split(r"(?:&&|\|\||[;|])", command):
        try:
            tokens = shlex.split(segment, posix=True)
        except ValueError:
            tokens = segment.split()
        for index, token in enumerate(tokens):
            executable = PurePosixPath(token).name
            options = tokens[index + 1 :]
            if executable == "sed" and any(
                option == "--in-place"
                or option.startswith("--in-place=")
                or (
                    option.startswith("-")
                    and not option.startswith("--")
                    and "i" in option[1:]
                )
                for option in options
            ):
                return (
                    "RunCommand cannot edit files in place with sed; use a "
                    "file editing action."
                )
            if executable == "perl" and any(
                option.startswith("-")
                and not option.startswith("--")
                and "i" in option[1:]
                for option in options
            ):
                return (
                    "RunCommand cannot edit files in place with perl; use a "
                    "file editing action."
                )
    if command_removes_artifact(command, artifact_paths):
        return (
            "RunCommand cannot remove or move a registered verification artifact. "
            "Rewrite it with CreateFile or a file editing action."
        )
    return None


@dataclass
class ReproductionSession:
    created_artifacts: Dict[str, str] = field(default_factory=dict)
    execution_records: List[ExecutionRecord] = field(default_factory=list)
    candidate_run_ids: List[str] = field(default_factory=list)
    revised_after_candidate: bool = False
    revised_execution_attempted: bool = False

    def register_artifact(self, path: str, content: str) -> None:
        normalized = normalize_artifact_path(path)
        if normalized is None:
            raise ValueError(f"Invalid repository-relative artifact path: {path}")
        self.created_artifacts[normalized] = content

    def add_execution(self, record: ExecutionRecord) -> None:
        self.execution_records.append(record)
        if record in self.eligible_failure_records(self.created_artifacts):
            self.candidate_run_ids.append(record.run_id)

    def convergence_rejection(self, action_name: str) -> Optional[str]:
        if not self.candidate_run_ids or action_name == "Finish":
            return None

        latest_run_id = self.candidate_run_ids[-1]
        if len(self.candidate_run_ids) >= 2 or self.revised_execution_attempted:
            return (
                "Reproduction evidence is already sufficient. Call Finish now with "
                f"{latest_run_id}; no further search, file revision, or command is allowed."
            )

        if not self.revised_after_candidate:
            if action_name == "CreateFile":
                return None
            return (
                f"{latest_run_id} is candidate reproduction evidence. Call Finish with "
                f"{latest_run_id}, or use CreateFile once to revise the reproducer."
            )

        if action_name == "RunCommand":
            return None
        return (
            "The one allowed reproducer revision has been used. Run the revised repro "
            f"artifact once, or call Finish with {latest_run_id}."
        )

    def note_post_candidate_revision(self) -> None:
        self.revised_after_candidate = True

    def note_revised_execution(self) -> None:
        self.revised_execution_attempted = True

    def get_execution(self, run_id: str) -> Optional[ExecutionRecord]:
        return next(
            (record for record in self.execution_records if record.run_id == run_id),
            None,
        )

    def eligible_failure_records(self, declared_paths: Iterable[str]) -> List[ExecutionRecord]:
        paths = list(declared_paths)
        return [
            record
            for record in self.execution_records
            if record.exit_code != 0
            and is_reproduction_command(record.command)
            and any(command_references_artifact(record.command, path) for path in paths)
        ]
