"""Canonical locations for generated experiment artifacts."""

import os


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")
SUBMISSION_ROOT = os.path.join(PROJECT_ROOT, "submission")


def _root_artifact_path(path: str, root: str) -> str:
    """Keep relative artifacts under ``root`` while preserving absolute paths."""
    expanded = os.path.expanduser(path.strip())
    if not expanded:
        raise ValueError("Artifact path must not be empty.")
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)

    root_name = os.path.basename(root)
    normalized = os.path.normpath(expanded)
    if normalized == root_name or normalized.startswith(root_name + os.sep):
        candidate = os.path.abspath(os.path.join(PROJECT_ROOT, normalized))
    else:
        candidate = os.path.abspath(os.path.join(root, normalized))

    if os.path.commonpath((root, candidate)) != root:
        raise ValueError(
            f"Relative artifact path escapes {root_name}/: {path}. "
            "Use an absolute path for an intentional external destination."
        )
    return candidate


def results_path(path: str) -> str:
    return _root_artifact_path(path, RESULTS_ROOT)


def submission_path(path: str) -> str:
    return _root_artifact_path(path, SUBMISSION_ROOT)
