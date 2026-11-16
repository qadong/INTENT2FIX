from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
import posixpath
import re

from INTENT2FIX.file_context import FileContext
from INTENT2FIX.reproduce.state import is_dedicated_reproduction_path


_TEST_PARTS = {"test", "tests", "testing", "fixture", "fixtures"}
_KNOWN_TEMPORARY_NAMES = {"fix_tests.py", "apply_test_edits.py"}


@dataclass(frozen=True)
class CandidatePatch:
    raw_patch: str
    patch: str
    kept_paths: list[str]
    excluded_paths: dict[str, str]
    excluded_test_patch: str = ""
    artifact_errors: tuple[str, ...] = ()


def _is_test_path(path: str) -> bool:
    return bool(set(PurePosixPath(path.lower()).parts) & _TEST_PARTS)


def _is_obvious_temporary_file(path: str) -> bool:
    name = PurePosixPath(path.lower()).name
    return (
        "probe" in name
        or name.startswith(("repro", "poc_"))
        or name.endswith(".patch")
        or name in _KNOWN_TEMPORARY_NAMES
    )


def needs_production_classification(path: str) -> bool:
    """Production-directory files need an explicit deliverable classification."""
    return (
        PurePosixPath(path).parts[0].lower() in {"include", "src", "lib"}
        and not _is_test_path(path)
        and not _is_obvious_temporary_file(path)
    )


def sanitize_candidate_patch(file_context: FileContext) -> CandidatePatch:
    """Remove harness-owned and verification-only edits from a candidate patch."""
    raw_patch = file_context.generate_git_patch()
    submission_context = file_context.clone()
    verification = file_context.verification or {}
    explicit_production = set(verification.get("production_paths", []))
    registered_ephemeral = set(verification.get("ephemeral_paths", []))
    registered_ephemeral.update(
        verification.get("immutable_reproduction_paths", [])
    )
    new_paths = {
        context_file.file_path
        for context_file in file_context.files
        if context_file.is_new
    }

    excluded: dict[str, str] = {}
    kept: list[str] = []
    for path in list(submission_context.get_patched_files()):
        reason = ""
        if path in registered_ephemeral:
            reason = "registered_ephemeral"
        elif path not in explicit_production and is_dedicated_reproduction_path(path):
            reason = "reproduction_artifact"
        elif _is_test_path(path):
            reason = "test_or_fixture"
        elif path in new_paths and path not in explicit_production and _is_obvious_temporary_file(path):
            reason = "temporary_probe_or_script"

        if reason:
            submission_context.remove_file(path)
            excluded[path] = reason
        else:
            kept.append(path)

    test_evidence_context = file_context.clone()
    for path in list(test_evidence_context.get_patched_files()):
        if excluded.get(path) != "test_or_fixture":
            test_evidence_context.remove_file(path)

    artifact_errors = []
    for path in sorted(new_paths & registered_ephemeral):
        if needs_production_classification(path):
            artifact_errors.append(
                f"New production-path file {path} is classified as temporary_probe. "
                "Use CreateFile with the same content and artifact_kind=production "
                "for a deliverable, or move the diagnostic to a probe/test path."
            )
    newly_excluded = new_paths & set(excluded)
    for path in kept if newly_excluded else []:
        content = file_context.get_file(path).content
        for target in re.findall(r'^\s*#\s*include\s*[<"]([^>"\n]+)[>"]', content, re.M):
            # Only detect dependencies on known newly excluded files. Compiler
            # verification handles macros and other dependency mechanisms.
            # Restrict this check to literal repository/sibling paths; suffix
            # matches would confuse different headers with the same basename.
            referenced_paths = {target, posixpath.normpath(str(PurePosixPath(path).parent / target))}
            for removed in sorted(newly_excluded & referenced_paths):
                artifact_errors.append(
                    f"Submitted file {path} includes {target}, but new file "
                    f"{removed} is excluded ({excluded[removed]}). Correct its "
                    "classification or remove the production dependency."
                )

    return CandidatePatch(
        raw_patch=raw_patch,
        patch=submission_context.generate_git_patch(),
        kept_paths=kept,
        excluded_paths=excluded,
        excluded_test_patch=test_evidence_context.generate_git_patch(),
        artifact_errors=tuple(artifact_errors),
    )
