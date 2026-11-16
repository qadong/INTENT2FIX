"""Compact Repair tool surface used by the opt-in four-tool experiment."""

from __future__ import annotations

import shlex
import re
from pathlib import PurePosixPath
from typing import Literal

from pydantic import ConfigDict, Field, PrivateAttr, model_validator

from INTENT2FIX.actions.action import Action
from INTENT2FIX.actions.command_policy import (
    future_data_command_rejection,
    local_git_mutation_rejection,
    verification_exit_status_rejection,
)
from INTENT2FIX.actions.create_file import CreateFile, CreateFileArgs
from INTENT2FIX.actions.insert_line import InsertLine, InsertLinesArgs
from INTENT2FIX.actions.output_sanitizer import sanitize_action_output
from INTENT2FIX.actions.repository_write_guard import (
    restore_file_context_worktree,
    tracked_worktree_fingerprint,
)
from INTENT2FIX.actions.run_single_test import _compute_passed
from INTENT2FIX.actions.test_parser import parse_test_output
from INTENT2FIX.actions.run_command import RunCommand
from INTENT2FIX.actions.schema import ActionArguments, Observation
from INTENT2FIX.actions.string_replace import StringReplace, StringReplaceArgs
from INTENT2FIX.actions.verification_sync import sync_verification_files
from INTENT2FIX.actions.view_code import CodeSpan, ViewCode, ViewCodeArgs
from INTENT2FIX.completion import BaseCompletionModel
from INTENT2FIX.file_context import FileContext
from INTENT2FIX.index import CodeIndex
from INTENT2FIX.intent_dialogue.developer_agent import (
    GraphEdgeType,
    InvestigationSession,
    RepositoryGraphQuery,
    RepositoryGraphQueryArgs,
    SearchCodeIntent,
    SearchCodeIntentArgs,
)
from INTENT2FIX.repository.repository import Repository
from INTENT2FIX.runtime.dockerenv import DockerEnv
from INTENT2FIX.workspace import Workspace
from INTENT2FIX.patch_review.sanitizer import sanitize_candidate_patch
from INTENT2FIX.utils.patch import production_patch_sha256


class BashArgs(ActionArguments):
    """Run one shell command for source search, temporary reproduction, build, or test.

    Commands start at the repository root and may use shell composition. Put disposable
    files under /tmp. Changes to tracked files and existing Editor-created files,
    including deletions, are imported into the candidate patch; create intended new
    production files with Editor. The observation always reports the real shell exit code and bounded raw
    output. Network access, reference patches, and mutating Git commands are unavailable.

    Set verification=true for compilation/tests, with probe_paths naming their disposable
    fixtures. Create fixtures in a separate ordinary command. Verification preserves failure
    status and freezes the executed production revision and fixtures; expected nonzero exits
    require required_output_regex. A zero-exit print-only diagnostic is not a behavioral oracle.
    Previously registered fixtures are also frozen and checked on later verifications,
    even when probe_paths is omitted. Removing a fixture requires a separate ordinary command.
    """

    command: str = Field(..., description="Shell command to execute at the repository root.")
    verification: bool = Field(
        default=False, description="True for a build/test oracle; false for search, setup or editing."
    )
    expected_exit_code: int = Field(default=0, ge=0, le=255)
    required_output_regex: str = Field(
        default="", max_length=1000,
        description="Required output pattern for an expected nonzero failure."
    )
    probe_paths: list[str] = Field(
        default_factory=list,
        description=(
            "Existing disposable source/header files needed to replay this verification. "
            "Previously registered files are inherited; an empty list does not disable their checks."
        )
    )

    model_config = ConfigDict(title="Bash")


class UnifiedBash(Action):
    args_schema = BashArgs

    _runtime: DockerEnv = PrivateAttr()
    _capture: RunCommand = PrivateAttr()

    def __init__(self, runtime: DockerEnv, **data):
        super().__init__(**data)
        self._runtime = runtime
        self._capture = RunCommand(docker_env=runtime)

    def execute(
        self,
        args: BashArgs,
        file_context: FileContext | None = None,
        workspace: Workspace | None = None,
    ) -> Observation:
        if file_context is None:
            return Observation(
                message="Bash requires the current patch context.",
                expect_correction=True,
            )

        rejection = future_data_command_rejection(args.command)
        if not rejection:
            rejection = local_git_mutation_rejection(args.command)
        if not rejection and args.verification:
            rejection = verification_exit_status_rejection(args.command)
            if rejection:
                rejection = (
                    "No command was executed and no fixture was created or updated. "
                    "Prepare/edit fixtures with Bash(verification=false) first and inspect "
                    "their actual contents. Then run only the compiler/test with "
                    "Bash(verification=true), declaring the existing source files in "
                    "probe_paths. Use && for prerequisites, not pipelines, || or ;. "
                    "Use ordinary Bash for extra output inspection."
                )
            if args.expected_exit_code != 0 and not args.required_output_regex:
                rejection = "A nonzero verification oracle requires required_output_regex."
            if args.required_output_regex:
                try:
                    re.compile(args.required_output_regex)
                except re.error as exc:
                    rejection = f"Invalid verification output regex: {exc}"
        if rejection:
            return Observation(
                message=rejection,
                expect_correction=True,
                properties={"rejected_command": args.command},
            )

        sync_failure = sync_verification_files(file_context, self._runtime)
        if sync_failure:
            return Observation(
                message=sync_failure,
                expect_correction=True,
                properties={"rejected_command": args.command},
            )

        execution_snapshot = None
        before_fingerprint = ""
        if args.verification:
            before_fingerprint, error = tracked_worktree_fingerprint(self._runtime)
            probe_files = {}
            declared_paths = set(args.probe_paths)
            probe_paths = list(dict.fromkeys([
                *file_context.verification.get("bash_probe_paths", []),
                *args.probe_paths,
            ]))
            for path in probe_paths:
                parsed = PurePosixPath(path)
                read_command = f"head -c 131073 -- {shlex.quote(path)}"
                if ".." in parsed.parts or (parsed.is_absolute() and not path.startswith("/tmp/")):
                    error = f"Probe path must be repository-relative or under /tmp: {path}"
                    break
                error = error or future_data_command_rejection(read_command)
                if error:
                    break
                content, read_exit = self._runtime.run_bash(read_command)
                if read_exit and path not in declared_paths:
                    _, exists_exit = self._runtime.run_bash(f"test -e {shlex.quote(path)}")
                    if exists_exit == 1:
                        # An old fixture removed during ordinary setup is retired.
                        # Explicit inputs and unreadable/oversized files still fail.
                        continue
                if read_exit or len(content.encode("utf-8")) > 131072:
                    error = f"Could not freeze complete verification fixture: {path}"
                    break
                probe_files[path] = content
            if error:
                return Observation(message=(
                    f"{error}. No command was executed. Create or correct the fixture "
                    "with Bash(verification=false), inspect its actual contents, then "
                    "rerun verification=true with existing probe_paths."
                ), expect_correction=True,
                                   properties={"fail_reason": "verification_snapshot_unavailable"})
            candidate = sanitize_candidate_patch(file_context)
            file_context.verification["bash_probe_paths"] = list(probe_files)
            execution_snapshot = {
                "production_patch_sha256": production_patch_sha256(candidate.patch),
                "command": args.command,
                "expected_exit_code": args.expected_exit_code,
                "required_output_regex": args.required_output_regex,
                "change_seq": file_context.change_seq,
                "artifact_errors": list(candidate.artifact_errors),
                "probe_files": probe_files,
                "inherited_probe_paths": [path for path in probe_files if path not in declared_paths],
            }

        # DockerEnv's generic shell is ``sh``.  This model-facing action is
        # explicitly Bash, so enable pipefail to keep a failed compiler/test
        # from being hidden by a trailing ``head``, ``tail``, or ``grep``.
        shell_options = "-e -o pipefail" if args.verification else "-o pipefail"
        shell_command = f"bash {shell_options} -c {shlex.quote(args.command)}"
        output, exit_code = self._runtime.run_bash(shell_command)
        safe_output = sanitize_action_output(
            output,
            max_chars=12000,
            max_lines=200,
            keep="head_tail",
        )

        imported_paths, capture_error = self._capture._capture_generated_changes(
            file_context
        )
        if capture_error:
            restore_error = restore_file_context_worktree(self._runtime, file_context)
            message = (
                "Bash changed managed repository content that could not be safely "
                f"imported: {capture_error}. Those managed side effects were discarded."
            )
            if restore_error:
                message += f" Worktree restoration failed: {restore_error}"
            if safe_output.text:
                message += "\n--- output (sanitized and bounded) ---\n" + safe_output.text
            return Observation(
                message=message,
                expect_correction=True,
                properties={
                    "command": args.command,
                    "exit_code": exit_code,
                    "output": safe_output.text,
                    "output_metadata": safe_output.metadata,
                    "fail_reason": "tracked_changes_not_imported",
                },
            )

        record = {
            "command": args.command,
            "exit_code": exit_code,
            "change_seq": file_context.change_seq,
            "tracked_changes": imported_paths,
        }
        if execution_snapshot is not None:
            after_fingerprint, error = tracked_worktree_fingerprint(self._runtime)
            candidate = sanitize_candidate_patch(file_context)
            changed = (
                bool(error) or before_fingerprint != after_fingerprint
                or production_patch_sha256(candidate.patch) != execution_snapshot["production_patch_sha256"]
            )
            fixture_changed = False
            for path, content in execution_snapshot["probe_files"].items():
                current, read_exit = self._runtime.run_bash(
                    f"head -c 131073 -- {shlex.quote(path)}"
                )
                fixture_changed |= bool(read_exit) or current != content
            record.update(
                expected_exit_code=args.expected_exit_code,
                required_output_regex=args.required_output_regex,
                execution_snapshot=execution_snapshot,
                oracle_satisfied=not (changed or fixture_changed) and _compute_passed(
                    exit_code, parse_test_output(output),
                    expected_exit_code=args.expected_exit_code,
                    required_output_regex=args.required_output_regex, output=output,
                ),
            )
            if changed:
                record["fail_reason"] = "verification_modified_repository"
            elif fixture_changed:
                record["fail_reason"] = "verification_modified_fixture"
        file_context.verification["bash"] = record

        message = f"Command finished with exit code {exit_code}."
        if execution_snapshot is not None:
            message += f"\nDeclared verification oracle satisfied: {record['oracle_satisfied']}."
            message += " This checks the declared exit/output oracle, not Issue coverage."
            if record.get("fail_reason"):
                message += " Verification changed repository or fixture content; rerun separately on the final patch."
        if imported_paths:
            message += "\nRepository text changes imported: " + ", ".join(imported_paths)
        if safe_output.text:
            message += "\n--- output (sanitized and bounded) ---\n" + safe_output.text
        else:
            message += " No output."
        return Observation(
            message=message,
            summary=f"Bash exited {exit_code}.",
            expect_correction=bool(record.get("fail_reason")),
            properties={
                **record,
                "output": safe_output.text,
                "output_metadata": safe_output.metadata,
            },
        )


EditorCommand = Literal["view", "create", "replace", "insert"]


class EditorArgs(ActionArguments):
    """View or edit repository text through one persistent editor interface.

    Use view before editing an existing file. Use create only for an intended
    production deliverable; disposable reproduction files belong under /tmp via Bash.
    replace requires one exact, unique old_str. insert adds new_str after insert_line.
    """

    command: EditorCommand
    path: str = Field(..., description="Repository-relative path, optionally prefixed by /repo/.")
    view_range: list[int] | None = Field(
        default=None,
        description="For view only: inclusive [start_line, end_line], with -1 meaning EOF.",
    )
    file_text: str | None = Field(default=None, description="Complete content for create.")
    old_str: str | None = Field(default=None, description="Exact existing text for replace.")
    new_str: str | None = Field(default=None, description="Replacement or inserted text.")
    insert_line: int | None = Field(
        default=None,
        description="For insert only: insert new_str after this one-based line number.",
    )

    model_config = ConfigDict(title="Editor")

    @model_validator(mode="after")
    def validate_command_fields(self) -> "EditorArgs":
        if self.command == "create" and self.file_text is None:
            raise ValueError("create requires file_text")
        if self.command == "replace" and (self.old_str is None or self.new_str is None):
            raise ValueError("replace requires old_str and new_str")
        if self.command == "insert" and (
            self.insert_line is None or self.new_str is None
        ):
            raise ValueError("insert requires insert_line and new_str")
        if self.view_range is not None and len(self.view_range) != 2:
            raise ValueError("view_range must contain exactly two line numbers")
        return self


class UnifiedEditor(Action):
    args_schema = EditorArgs

    _view: ViewCode = PrivateAttr()
    _create: CreateFile = PrivateAttr()
    _replace: StringReplace = PrivateAttr()
    _insert: InsertLine = PrivateAttr()

    def __init__(
        self,
        repository: Repository,
        code_index: CodeIndex | None,
        completion_model: BaseCompletionModel,
        runtime: DockerEnv | None = None,
        **data,
    ):
        super().__init__(**data)
        self._view = ViewCode(
            repository=repository,
            completion_model=completion_model,
        )
        self._create = CreateFile(
            repository=repository,
            code_index=code_index,
            runtime=runtime,
        )
        self._replace = StringReplace(
            repository=repository,
            code_index=code_index,
            runtime=runtime,
        )
        self._insert = InsertLine(
            repository=repository,
            code_index=code_index,
            runtime=runtime,
        )

    @staticmethod
    def _normalized_path(path: str) -> str:
        if path.startswith("/repo/"):
            return path[6:]
        return path[1:] if path.startswith("/") else path

    def execute(
        self,
        args: EditorArgs,
        file_context: FileContext | None = None,
        workspace: Workspace | None = None,
    ) -> Observation:
        if file_context is None:
            return Observation(
                message="Editor requires the current patch context.",
                expect_correction=True,
            )
        path = self._normalized_path(args.path)
        if args.command == "view":
            span = CodeSpan(file_path=path)
            if args.view_range is not None:
                span.start_line, span.end_line = args.view_range
            return self._view.execute(
                ViewCodeArgs(thoughts=args.thoughts, files=[span]),
                file_context=file_context,
            )
        if args.command == "create":
            observation = self._create.execute(
                CreateFileArgs(
                    thoughts=args.thoughts,
                    path=path,
                    file_text=args.file_text or "",
                    artifact_kind="production",
                ),
                file_context=file_context,
            )
        elif args.command == "replace":
            observation = self._replace.execute(
                StringReplaceArgs(
                    thoughts=args.thoughts,
                    path=path,
                    old_str=args.old_str or "",
                    new_str=args.new_str or "",
                ),
                file_context=file_context,
            )
        else:
            observation = self._insert.execute(
                InsertLinesArgs(
                    thoughts=args.thoughts,
                    path=path,
                    insert_line=args.insert_line or 0,
                    new_str=args.new_str or "",
                ),
                file_context=file_context,
            )
        observation.message = observation.message.replace("ViewCode", 'Editor(command="view")')
        observation.message = observation.message.replace("CreateFile", 'Editor(command="create")')
        if observation.properties.get("fail_reason") == "file_exists":
            observation.message += ' Use Editor(command="view"), then command="replace" for an existing file.'
        return observation


GraphOperation = Literal["search", "trace"]


class CodeGraphArgs(ActionArguments):
    """Search for a behavior or trace real graph edges from an inspected anchor.

    Search internally expands Issue-relevant relations. Trace accepts one GREF,
    inspected source location, or exact symbol and follows issue-guided graph
    paths. Results are navigation leads; verify the actual source and build path.
    """

    operation: GraphOperation
    behavior: str = Field(
        default="", max_length=800,
        description="For search: focused behavior or relation whose source name is unknown.",
    )
    question: str = Field(
        default="", max_length=500,
        description="For trace: concrete causal or patch-scope question for this anchor.",
    )
    purpose: str = Field(
        default="", max_length=300,
        description="What missing relation would change the cause, implementation owner, or scope?",
    )
    entity_ref: str = Field(default="", description="GREF-N copied from a prior CodeGraph result.")
    source_path: str = Field(
        default="", description="Inspected repository-relative source path; pair with source_line.",
    )
    source_line: int | None = Field(default=None, ge=1)
    symbol_name: str = Field(default="", description="Exact source symbol if no GREF/source location is available.")
    symbol_type: Literal["function", "class", "macro"] = "function"
    namespace: str = ""
    class_name: str = ""
    direction: Literal["in", "out", "both"] = "both"
    edge_types: list[GraphEdgeType] = Field(
        default_factory=list, max_length=5,
        description="Optional exact relation kinds; leave empty when the indexed kind is uncertain.",
    )
    limit: int = Field(default=6, ge=2, le=10, description="Trace result limit, including the anchor.")

    model_config = ConfigDict(title="CodeGraph")

    @model_validator(mode="after")
    def validate_operation_fields(self) -> "CodeGraphArgs":
        if self.operation == "search":
            if not self.behavior.strip():
                raise ValueError("search requires behavior")
            return self
        if not self.question.strip():
            raise ValueError("trace requires question")
        if bool(self.source_path) != (self.source_line is not None):
            raise ValueError("trace requires source_path and source_line together")
        anchors = (bool(self.entity_ref), bool(self.source_path), bool(self.symbol_name))
        if sum(anchors) != 1:
            raise ValueError("trace requires exactly one GREF, source location, or exact symbol")
        return self


class UnifiedCodeGraph(Action):
    args_schema = CodeGraphArgs

    _session: InvestigationSession = PrivateAttr()
    _graph: RepositoryGraphQuery = PrivateAttr()
    _search: SearchCodeIntent = PrivateAttr()

    def __init__(
        self,
        code_index: CodeIndex,
        completion_model: BaseCompletionModel,
        issue_text: str = "",
        **data,
    ):
        super().__init__(**data)
        self._session = InvestigationSession(issue_text=issue_text)
        self._graph = RepositoryGraphQuery(
            code_index=code_index,
            session=self._session,
            structural_completion=completion_model.clone(
                max_tokens=4096, timeout=60.0, max_retries=0,
            ),
        )
        self._search = SearchCodeIntent(graph=self._graph, session=self._session)

    def execute(
        self,
        args: CodeGraphArgs,
        file_context: FileContext | None = None,
        workspace: Workspace | None = None,
    ) -> Observation:
        if args.operation == "search":
            result = self._search.execute(SearchCodeIntentArgs(
                thoughts=args.thoughts,
                behavior=args.behavior,
                purpose=args.purpose,
            ))
        else:
            result = self._graph.execute(RepositoryGraphQueryArgs(
                thoughts=args.thoughts,
                mode="trace",
                target=args.question,
                purpose=args.purpose or args.question,
                entity_ref=args.entity_ref,
                source_path=args.source_path,
                source_line=args.source_line,
                symbol_name=args.symbol_name,
                symbol_type=args.symbol_type,
                namespace=args.namespace,
                class_name=args.class_name,
                direction=args.direction,
                edge_types=args.edge_types,
                limit=args.limit,
            ))
        result.message = (
            result.message
            .replace("SearchCodeIntent", "CodeGraph search")
            .replace("TraceInvestigationPath", "CodeGraph trace")
        )
        return result
