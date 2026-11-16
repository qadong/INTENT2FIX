import os
import shlex
import tempfile
import uuid

from INTENT2FIX.file_context import FileContext
from INTENT2FIX.runtime.dockerenv import DockerEnv


_SYNCED_PATHS_KEY = "_container_synced_patched_files"
_EPHEMERAL_PATHS_KEY = "ephemeral_paths"
_PROBE_CHANGE_SEQ_KEY = "probe_change_seq"
PRODUCTION_ISOLATION = "fresh_image_production_v2"


def run_isolated_production_command(
    runtime: DockerEnv, command: str, included_paths: list[str], file_context: FileContext
) -> tuple[str, int]:
    """Verify delivered files in an initialized image, without moving live inputs."""
    with DockerEnv(
        image_name=runtime.image_name,
        container_name=f"production-verify-{uuid.uuid4().hex}",
        work_dir=runtime.work_dir,
        benchmark_base_commit=runtime.benchmark_base_commit,
    ) as isolated:
        # Older simdjson import.cmake downloads before creating its cache root.
        # Create the empty directory; never import dependencies from live Repair.
        output, exit_code = isolated.run_bash(
            "if [ -f dependencies/import.cmake ]; then mkdir -p dependencies/.cache; fi"
        )
        if exit_code != 0:
            raise RuntimeError(f"Production verification initialization failed: {output}")
        # The image supplies its own dependencies/data; no live worktree mount or
        # unregistered files are copied. Keep synchronization bookkeeping local.
        view = file_context.clone()
        view.verification.pop(_SYNCED_PATHS_KEY, None)
        failure = sync_verification_files(view, isolated, included_paths=set(included_paths))
        if failure:
            raise RuntimeError(f"Production verification setup failed: {failure}")
        return isolated.run_bash(command)


def current_probe_change_seq(file_context: FileContext) -> int:
    return int(file_context.verification.get(_PROBE_CHANGE_SEQ_KEY, 0))


def mark_file_change(file_context: FileContext, path: str) -> None:
    """Invalidate only the verification state affected by an edited path."""
    ephemeral_paths = set(
        file_context.verification.get(_EPHEMERAL_PATHS_KEY, [])
    )
    if str(path) in ephemeral_paths:
        file_context.verification[_PROBE_CHANGE_SEQ_KEY] = (
            current_probe_change_seq(file_context) + 1
        )
        return
    file_context.change_seq += 1


def sync_verification_files(
    file_context: FileContext,
    runtime: DockerEnv,
    *,
    included_paths: set[str] | None = None,
) -> str:
    """Sync selected current patches and restore files no longer selected."""
    patched_paths = set(file_context.get_patched_files())
    current_paths = (
        patched_paths
        if included_paths is None
        else patched_paths & set(included_paths)
    )
    previous_paths = set(file_context.verification.get(_SYNCED_PATHS_KEY, []))

    for file_path in sorted(current_paths | previous_paths | patched_paths):
        context_file = file_context.get_file(file_path)
        if file_path not in current_paths and (
            context_file is None or context_file.is_new
        ):
            # A new excluded header must be absent, not present as an empty file.
            container_path = os.path.join(runtime.work_dir, file_path)
            output, exit_code = runtime.run_bash(
                f"rm -f -- {shlex.quote(container_path)}", work_dir="/"
            )
            if exit_code != 0:
                return f"Failed to remove excluded new file {file_path}: {output}"
            continue
        if file_path in current_paths:
            file_content = context_file.content
        elif file_path in patched_paths:
            # The patch still exists but this verification view excludes it.
            file_content = context_file.get_base_content()
        else:
            # The patch was reverted in FileContext; content is already baseline.
            file_content = context_file.content
        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp_file:
                tmp_file.write(file_content)
                tmp_path = tmp_file.name

            container_path = os.path.join(runtime.work_dir, file_path)
            container_parent = os.path.dirname(container_path)
            if container_parent:
                output, exit_code = runtime.run_bash(
                    f"mkdir -p {shlex.quote(container_parent)}",
                    work_dir="/",
                )
                if exit_code != 0:
                    return f"Failed to create container directory {container_parent}: {output}"
            if not runtime.copy_anything_to_container(tmp_path, container_path):
                return f"Failed to copy {file_path} to container."
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    file_context.verification[_SYNCED_PATHS_KEY] = sorted(current_paths)
    return ""
