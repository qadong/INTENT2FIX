import shlex
from pathlib import PurePosixPath

from INTENT2FIX.actions.verification_sync import sync_verification_files
from INTENT2FIX.file_context import FileContext
from INTENT2FIX.runtime.dockerenv import DockerEnv


def tracked_worktree_fingerprint(runtime: DockerEnv) -> tuple[str, str]:
    """Return a stable fingerprint of tracked worktree changes."""
    output, exit_code = runtime.run_bash(
        "tmp=/tmp/intent2fix-tracked-diff.$$; "
        'git diff --no-ext-diff --binary HEAD > "$tmp" && sha256sum "$tmp"; '
        'status=$?; rm -f "$tmp"; exit "$status"'
    )
    if exit_code != 0 or not output.strip():
        return "", f"Could not fingerprint repository changes: {output}"
    return output.split()[0], ""


def tracked_worktree_paths(runtime: DockerEnv) -> tuple[list[str], str]:
    output, exit_code = runtime.run_bash("git diff --name-only -z HEAD")
    if exit_code != 0:
        return [], f"Could not inspect repository changes: {output}"
    paths = []
    for raw_path in output.split("\0"):
        path = raw_path.strip()
        if not path:
            continue
        parsed = PurePosixPath(path)
        if parsed.is_absolute() or ".." in parsed.parts:
            return [], f"Repository command reported an unsafe path: {path}"
        paths.append(parsed.as_posix())
    return list(dict.fromkeys(paths)), ""


def restore_file_context_worktree(
    runtime: DockerEnv,
    file_context: FileContext,
) -> str:
    """Discard command side effects, then restore the tracked FileContext state."""
    paths, error = tracked_worktree_paths(runtime)
    if error:
        return error
    if paths:
        command = "git checkout -- " + " ".join(shlex.quote(path) for path in paths)
        output, exit_code = runtime.run_bash(command)
        if exit_code != 0:
            return f"Could not discard untracked command side effects: {output}"
    return sync_verification_files(file_context, runtime)
