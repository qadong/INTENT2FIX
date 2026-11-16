from INTENT2FIX.actions.repository_write_guard import tracked_worktree_fingerprint


class _Runtime:
    def __init__(self, output: str, exit_code: int = 0) -> None:
        self.output = output
        self.exit_code = exit_code
        self.command = ""

    def run_bash(self, command: str) -> tuple[str, int]:
        self.command = command
        return self.output, self.exit_code


def test_fingerprint_command_is_posix_shell_compatible() -> None:
    runtime = _Runtime("abc123  /tmp/intent2fix-tracked-diff.42\n")

    fingerprint, error = tracked_worktree_fingerprint(runtime)  # type: ignore[arg-type]

    assert fingerprint == "abc123"
    assert error == ""
    assert "pipefail" not in runtime.command
    assert 'git diff --no-ext-diff --binary HEAD > "$tmp"' in runtime.command
    assert 'rm -f "$tmp"' in runtime.command
    assert 'exit "$status"' in runtime.command


def test_fingerprint_reports_command_failure() -> None:
    runtime = _Runtime("fatal: not a git repository\n", exit_code=128)

    fingerprint, error = tracked_worktree_fingerprint(runtime)  # type: ignore[arg-type]

    assert fingerprint == ""
    assert "Could not fingerprint repository changes" in error
