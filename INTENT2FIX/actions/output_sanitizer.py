import hashlib
import re
from dataclasses import dataclass
from typing import Literal


OUTPUT_CAPTURE_TRUNCATION_MARKER = "[command output capture truncated]"


_ANSI_ESCAPE_RE = re.compile(
    r"(?:"
    r"\x1b\][^\x07]*(?:\x07|\x1b\\)"  # OSC
    r"|\x1b\[[0-?]*[ -/]*[@-~]"  # CSI
    r"|\x1b[()][0-2A-Z]"  # character-set selection
    r"|\x1b[@-_]"  # remaining two-byte escapes
    r")",
    re.DOTALL,
)
_UNSAFE_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


@dataclass(frozen=True)
class SanitizedOutput:
    text: str
    metadata: dict[str, object]


def sanitize_action_output(
    output: str,
    *,
    max_chars: int,
    max_lines: int | None = None,
    keep: Literal["tail", "head_tail"] = "tail",
) -> SanitizedOutput:
    """Make command output safe and bounded before storing it in an Observation."""
    output = output or ""
    control_chars = sum(
        1
        for char in output
        if char not in "\n\r\t" and (ord(char) < 32 or 127 <= ord(char) < 160)
    )
    replacement_chars = output.count("\ufffd")
    binary_like = "\x00" in output or (
        len(output) >= 64
        and (control_chars + replacement_chars) / len(output) > 0.02
    )
    metadata: dict[str, object] = {
        "original_chars": len(output),
        "control_chars": control_chars,
        "replacement_chars": replacement_chars,
        "sha256": hashlib.sha256(
            output.encode("utf-8", errors="replace")
        ).hexdigest(),
        "binary_suppressed": binary_like,
        "truncated": False,
    }

    if binary_like:
        return SanitizedOutput(
            text=(
                "[binary output suppressed: "
                f"{len(output)} characters; inspect the file with file, strings, "
                "readelf, or objdump instead of cat]"
            ),
            metadata=metadata,
        )

    safe = _ANSI_ESCAPE_RE.sub("", output)
    safe = _UNSAFE_CONTROL_RE.sub("", safe)

    if max_lines is not None:
        lines = safe.splitlines()
        if len(lines) > max_lines:
            safe = "\n".join(lines[-max_lines:])
            metadata["truncated"] = True

    if len(safe) > max_chars:
        marker = f"\n... [output truncated; original {len(output)} characters] ...\n"
        available = max(0, max_chars - len(marker))
        if keep == "head_tail":
            head_chars = available // 2
            tail_chars = available - head_chars
            tail = safe[-tail_chars:] if tail_chars else ""
            safe = safe[:head_chars] + marker + tail
        else:
            tail = safe[-available:] if available else ""
            safe = marker + tail
        metadata["truncated"] = True

    return SanitizedOutput(text=safe, metadata=metadata)
