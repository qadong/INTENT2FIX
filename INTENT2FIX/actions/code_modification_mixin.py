import logging
from pathlib import Path
from typing import Optional, Tuple

from pydantic import PrivateAttr

from INTENT2FIX.actions.schema import Observation
from INTENT2FIX.file_context import FileContext
from INTENT2FIX.index import CodeIndex
from INTENT2FIX.repository.repository import Repository
from INTENT2FIX.runtime.runtime import RuntimeEnvironment
from INTENT2FIX.utils.file import is_test

logger = logging.getLogger(__name__)


class CodeModificationMixin:
    """
    A mixin that provides common functionality for actions that modify code files.
    This includes path normalization, file validation, test running, and observation handling.
    """

    _runtime: RuntimeEnvironment | None = PrivateAttr(default=None)
    _code_index: CodeIndex | None = PrivateAttr(default=None)
    _repository: Repository | None = PrivateAttr(default=None)

    def normalize_path(self, file_path: str) -> str:
        """Normalize a repository-root prefix and one leading slash."""
        if file_path == "/repo":
            return ""
        if file_path.startswith("/repo/"):
            file_path = file_path[6:]
        if file_path.startswith("/"):
            file_path = file_path[1:]
        return file_path

    def validate_repository_path(
        self, file_path: str
    ) -> Tuple[Optional[Path], Optional[Observation]]:
        """Return a normalized path only when it remains inside the repository."""
        path_str = self.normalize_path(file_path)
        path = Path(path_str)
        if not path_str or path_str == "." or path.is_absolute() or ".." in path.parts:
            return None, Observation(
                message=(
                    f"Invalid repository-relative path: {file_path}. "
                    "Use a file path inside the repository without '..'."
                ),
                properties={"fail_reason": "invalid_path", "success": False},
                expect_correction=True,
            )

        repo_path = getattr(self._repository, "repo_path", None)
        if repo_path:
            repo_root = Path(repo_path).resolve()
            resolved_path = (repo_root / path).resolve()
            if not resolved_path.is_relative_to(repo_root):
                return None, Observation(
                    message=(
                        f"Invalid repository-relative path: {file_path}. "
                        "The resolved path points outside the repository."
                    ),
                    properties={"fail_reason": "invalid_path", "success": False},
                    expect_correction=True,
                )
        return path, None

    def validate_file_access(
        self, file_path: str, file_context: FileContext
    ) -> Tuple[Optional[Path], Optional[Observation]]:
        """
        Validate file access and return either a valid Path object or an error Observation.

        Args:
            file_path: The path to validate
            file_context: The file context
        Returns:
            Tuple of (Path object if valid, Error observation if invalid)
        """
        path, path_error = self.validate_repository_path(file_path)
        if path_error:
            return None, path_error

        if not file_context.file_exists(str(path)):
            message = f"File {path} not found."
            if file_path.startswith("/"):
                message += " It seems you are using an absolute path. Please use a relative path from the repository root instead."
            return None, Observation(
                message=message,
                properties={"fail_reason": "file_not_found"},
            )

        if (
            file_context.verification.get("finish_policy")
            == "portfolio_soft_evidence"
            and (
                is_test(str(path))
                or bool(
                    {"fixture", "fixtures"}
                    & {part.lower() for part in path.parts}
                )
            )
        ):
            return None, Observation(
                message=(
                    f"Edit rejected: {path} is existing repository test evidence "
                    "and is excluded from portfolio candidate submissions. Treat "
                    "legacy expectation failures as soft evidence; fix production "
                    "code only when they reveal an Issue-inconsistent regression. "
                    "If the Issue behavior and focused discriminator are already "
                    "satisfied, report the legacy failure and call Finish. Use a "
                    "temporary_probe CreateFile for new diagnostic evidence."
                ),
                properties={
                    "fail_reason": "portfolio_repository_test_edit",
                    "success": False,
                },
                expect_correction=True,
            )

        if not file_context.has_file(str(path)):
            return None, Observation(
                message=f"You have not yet viewed the file {path}. Use ViewCode to view the parts of the file that you want to modify.",
                properties={"fail_reason": "file_not_in_context"},
            )

        return path, None

    def run_tests(
        self,
        file_path: str,
        file_context: FileContext,
    ) -> str:
        # Disable automatic test running after code modification.
        # We rely on the agent to explicitly call RunSingleTest or RunRegression
        # as per the updated workflow instructions.
        return ""

    def format_snippet_with_lines(self, snippet: str, start_line: int) -> str:
        """Format a code snippet with line numbers"""
        return "\n".join(f"{i + start_line:6}\t{line}" for i, line in enumerate(snippet.split("\n")))
