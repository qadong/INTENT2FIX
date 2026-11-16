import hashlib
import json
import re
import shlex
from pathlib import PurePosixPath
from typing import ClassVar, Type

from pydantic import ConfigDict, Field, PrivateAttr

from INTENT2FIX.actions.action import Action
from INTENT2FIX.actions.output_sanitizer import OUTPUT_CAPTURE_TRUNCATION_MARKER
from INTENT2FIX.actions.schema import ActionArguments, Observation
from INTENT2FIX.actions.verification_sync import sync_verification_files
from INTENT2FIX.file_context import FileContext
from INTENT2FIX.runtime.dockerenv import DockerEnv


class RunIntentCheckArgs(ActionArguments):
    """Run one immutable ACCEPT-N probe from the repair intent contract.

    The harness restores the frozen probe and runs it against the latest patch.
    Use only an executable ACCEPT-N ID listed in <repair_intent_contract>; do not
    invent an ID or substitute a command. Required probes block Finish outside
    portfolio mode; in portfolio mode their outcomes are evidence under the Repair
    Policy. Optional diagnostic probes never block completion. This executes a
    frozen check; it does not judge or revise the Hypothesis. Report the observed
    result without treating one passing probe as proof of the entire repair.
    """

    thoughts: str = Field(
        ...,
        description="Why this required or diagnostic frozen probe should run now.",
    )
    check_id: str = Field(
        ...,
        pattern=r"^ACCEPT-\d+$",
        description="Exact executable ACCEPT-N ID listed in <repair_intent_contract>.",
    )

    model_config = ConfigDict(title="RunIntentCheck")

    @property
    def log_name(self):
        return f"RunIntentCheck(check_id='{self.check_id}')"


class RunIntentCheck(Action):
    args_schema: ClassVar[Type[ActionArguments]] = RunIntentCheckArgs
    _runtime: DockerEnv = PrivateAttr()

    def __init__(self, runtime: DockerEnv | None = None, **data):
        super().__init__(**data)
        self._runtime = runtime

    @staticmethod
    def _canonicalize_probe_sources(
        command: str,
        files: dict[str, str],
    ) -> str:
        """Bind frozen source tokens to their repository-relative files."""
        basename_counts: dict[str, int] = {}
        for path in files:
            basename = PurePosixPath(path).name
            basename_counts[basename] = basename_counts.get(basename, 0) + 1

        rewritten = command
        for path in sorted(files, key=len, reverse=True):
            aliases = [f"/tmp/{path}"]
            basename = PurePosixPath(path).name
            if basename_counts[basename] == 1:
                aliases.append(f"/tmp/{basename}")
            for alias in aliases:
                rewritten = re.sub(
                    rf"(?<![A-Za-z0-9_./-]){re.escape(alias)}"
                    rf"(?![A-Za-z0-9_./-])",
                    path,
                    rewritten,
                )
        return rewritten

    @staticmethod
    def _prepare_probe_output_directory(command: str) -> str:
        """Create a nested /tmp output directory before compiling the probe."""
        try:
            tokens = shlex.split(command)
        except ValueError:
            return command

        output_dirs: list[str] = []
        for index, token in enumerate(tokens):
            output_path = ""
            if token == "-o" and index + 1 < len(tokens):
                output_path = tokens[index + 1]
            elif token.startswith("-o") and len(token) > 2:
                output_path = token[2:]
            if not output_path.startswith("/tmp/"):
                continue
            parent = str(PurePosixPath(output_path).parent)
            if parent != "/tmp" and parent not in output_dirs:
                output_dirs.append(parent)

        if not output_dirs:
            return command
        directories = " ".join(shlex.quote(path) for path in output_dirs)
        return f"mkdir -p {directories} && {command}"

    def execute(
        self,
        args: RunIntentCheckArgs,
        file_context: FileContext,
        **kwargs,
    ) -> Observation:
        if not self._runtime:
            raise ValueError(
                "Runtime environment must be provided to run intent checks."
            )

        verification = getattr(file_context, "verification", {}) or {}
        contract = verification.get("intent_contract") or {}
        probe = (contract.get("probes") or {}).get(args.check_id)
        if probe is None:
            return Observation(
                message=(
                    f"Unknown or non-executable intent check {args.check_id}. "
                    "Use an ACCEPT-N ID listed in <repair_intent_contract>."
                ),
                summary="Unknown frozen intent check.",
                expect_correction=True,
                properties={"fail_reason": "unknown_intent_check_id"},
            )

        sync_failure = sync_verification_files(file_context, self._runtime)
        if sync_failure:
            return Observation(
                message=sync_failure,
                summary="File synchronization failed.",
            )

        files = probe.get("files") or {}
        for path, content in files.items():
            candidate = PurePosixPath(path)
            if candidate.is_absolute() or ".." in candidate.parts:
                return Observation(
                    message=f"Frozen probe contains an unsafe path: {path}",
                    summary="Invalid frozen probe path.",
                    expect_correction=False,
                    properties={"fail_reason": "invalid_frozen_probe"},
                )
            self._runtime.write_file(
                f"{self._runtime.work_dir.rstrip('/')}/{path}",
                content,
            )

        command = probe.get("command", "")
        executed_command = self._canonicalize_probe_sources(command, files)
        executed_command = self._prepare_probe_output_directory(
            executed_command
        )
        output, exit_code = self._runtime.run_bash(
            executed_command,
            timeout=600,
        )
        passed = exit_code == 0 and OUTPUT_CAPTURE_TRUNCATION_MARKER not in output
        authority = probe.get("authority", "normative")
        diagnostic_only = authority == "illustrative"
        change_seq = getattr(file_context, "change_seq", 0)
        digest = hashlib.sha256(
            json.dumps(
                {"command": command, "files": files},
                sort_keys=True,
            ).encode()
        ).hexdigest()
        file_context.verification.setdefault(
            "intent_check_results", {}
        )[args.check_id] = {
            "passed": passed,
            "exit_code": exit_code,
            "change_seq": change_seq,
            "probe_sha256": digest,
            "authority": authority,
        }

        max_chars = 3000
        if len(output) > max_chars:
            output = (
                output[: max_chars // 2]
                + "\n... [frozen probe output truncated] ...\n"
                + output[-max_chars // 2 :]
            )
        description = probe.get("description", args.check_id)
        role = "diagnostic-only witness" if diagnostic_only else "required oracle"
        return Observation(
            message=(
                f"Frozen intent check {args.check_id} finished with exit code "
                f"{exit_code}.\nRole: {role}.\nDescription: {description}\n\n"
                f"Output:\n{output or '(no output)'}"
            ),
            summary=(
                f"Frozen intent check {args.check_id} ({role}): "
                + (
                    "matched"
                    if passed and diagnostic_only
                    else "still diverges"
                    if diagnostic_only
                    else "passed"
                    if passed
                    else "failed"
                )
            ),
            expect_correction=not passed and not diagnostic_only,
            properties={
                "check_id": args.check_id,
                "exit_code": exit_code,
                "passed": passed,
                "authority": authority,
                "required": not diagnostic_only,
                "change_seq": change_seq,
                "probe_sha256": digest,
                "executed_command": executed_command,
            },
        )
