"""Read immutable archived evidence by ID without repository or shell access."""
import json
from pathlib import Path
from typing import ClassVar

from pydantic import ConfigDict, Field

from INTENT2FIX.actions.action import Action
from INTENT2FIX.actions.schema import ActionArguments, Observation
from INTENT2FIX.diagnostic_evidence import record_digest


class ReadSourceArgs(ActionArguments):
    """Read original frozen evidence behind a diagnostic summary or source reference.

    Use its exact source_id. Read the original when a decision depends on precise
    errors, tested operands, compiler/configuration details, conflicting summaries,
    or omitted source bodies. This reads pre-repair evidence, not the current tree;
    use ViewCode for current repository code. Continue from next_offset when needed.
    """
    source_id: str = Field(description="Existing evidence ID, for example RUN-14 or CODE-...; not a file path.")
    offset: int = Field(default=0, ge=0, description="Character offset in the serialized original record.")
    limit: int = Field(default=6000, ge=1, le=12000, description="Maximum original-record characters to return.")
    model_config = ConfigDict(title="read_source", extra="forbid")


class ReadSource(Action):
    args_schema: ClassVar[type[ActionArguments]] = ReadSourceArgs
    archive_path: str
    archive_sha256: str

    def execute(self, args: ReadSourceArgs, file_context=None, workspace=None):
        archive = json.loads(Path(self.archive_path).read_text())
        if record_digest(archive) != self.archive_sha256:
            return Observation(message="Evidence archive integrity check failed; original evidence unavailable.",
                               expect_correction=True, properties={"fail_reason": "archive_changed"})
        # Prefer the original legacy record over its presentation alias.
        source_id = args.source_id
        if source_id not in archive["records"]:
            source_id = archive.get("aliases", {}).get(source_id, source_id)
        record = archive["records"].get(source_id)
        if record is None:
            return Observation(message="Unknown source_id. Use an exact ID from the handoff or its source references.",
                               expect_correction=True, properties={"fail_reason": "unknown_source_id"})
        original = json.dumps(record, ensure_ascii=False, indent=2)
        if args.offset > len(original):
            return Observation(message=f"Offset exceeds original record length {len(original)}.", expect_correction=True)
        end = min(args.offset + args.limit, len(original))
        result = {"source_id": source_id, "base_commit": archive["base_commit"],
                  "record_sha256": record_digest(record), "offset": args.offset,
                  "total_chars": len(original), "next_offset": end if end < len(original) else None,
                  "original_truncated": bool(record.get("observation_truncated")),
                  "content": original[args.offset:end]}
        return Observation(message=json.dumps(result, ensure_ascii=False), properties={
            k: result[k] for k in ("source_id", "offset", "next_offset", "record_sha256")})
