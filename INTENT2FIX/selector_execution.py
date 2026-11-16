"""Bounded, frozen probes in disposable base/candidate containers (no Repair)."""

import io
import re
import shlex
import tarfile
import uuid

from INTENT2FIX.portfolio_selector import Probe, object_hash
from INTENT2FIX.actions.output_sanitizer import OUTPUT_CAPTURE_TRUNCATION_MARKER


PROBE_PATH = ".intent2fix_selector/probe.cc"
BINARY = "/tmp/intent2fix-selector-probe"


def refinement_args(args: list[str]) -> list[str]:
    """Remove only redundant harness-owned argv, never arbitrary build inputs."""
    cleaned = []
    i = 0
    while i < len(args):
        if args[i] == PROBE_PATH:
            i += 1
        elif args[i:i+2] == ["-o", BINARY]:
            i += 2
        else:
            cleaned.append(args[i])
            i += 1
    return cleaned


def probe_commands(probe: Probe) -> tuple[list[str], list[str]]:
    # argv is quoted by the harness; forbid compiler side effects/hidden dependencies.
    for arg in probe.compile_args:
        if not arg or any(c in arg for c in ("\n", "\r", "\x00")):
            raise ValueError("Invalid compiler argument")
        if arg.startswith(("@", "-o", "--output", "-fplugin", "-Xclang", "-B", "-specs", "-save-temps")) or arg in ("-c", "-E", "-S", "-M", "-MM", "-MD", "-MMD"):
            raise ValueError("Probe must compile a runnable binary without auxiliary writes")
        if re.search(r"(?:^|/|^-I|^-L|^-include)\.\.(?:/|$)", arg) or "/tmp/" in arg or ".git" in arg:
            raise ValueError("Probe cannot depend on temporary/future/outside sources")
        if arg.startswith(("/", "-I/", "-L/", "-include/", "-isystem/", "-iquote/")) or PROBE_PATH in arg:
            raise ValueError("Compile only base-relative sources and the harness-owned probe")
    return ([probe.compiler, *probe.compile_args, PROBE_PATH, "-o", BINARY],
            [BINARY, *probe.run_args])


def _write(env, path: str, content: str):
    """SDK archive avoids shell interpolation and writes only harness-owned paths."""
    data = content.encode()
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        entry = tarfile.TarInfo(path)
        entry.size = len(data)
        entry.mode = 0o644
        archive.addfile(entry, io.BytesIO(data))
    if not env.container.put_archive(env.work_dir, stream.getvalue()):
        raise RuntimeError("Failed to materialize frozen selector input")


def _cmake_public_options(env) -> list[str]:
    """Reuse literal configure switches from the image's build route, not its tests."""
    script, code = env.run_bash(
        "if test -f /home/build_run.sh; then sed -n '1,160p' /home/build_run.sh; "
        "elif test -f /home/run.sh; then sed -n '1,160p' /home/run.sh; fi", timeout=30)
    if code != 0:
        return []
    for line in script.replace("\\\n", " ").splitlines():
        match = re.search(r"(?:^|[;&]\s*)cmake\s+([^#]*)", line)
        if not match:
            continue
        try:
            args = shlex.split(match.group(1))
        except ValueError:
            continue
        if "--build" in args:
            continue
        return [arg for arg in args if re.fullmatch(
            r"-D[A-Za-z_][A-Za-z0-9_]*(?::[A-Za-z]+)?=[A-Za-z0-9_+./-]+", arg)]
    return []


def compile_candidate(candidate: dict, *, image: str, work_dir: str,
                      base_commit: str, env_factory=None) -> dict:
    """Build the exact production patch on a clean public checkout, without tests.

    A failed or unavailable build is never a pass. Keep infrastructure/configure
    failures distinct from a completed compiler failure for later diagnosis.
    """
    if env_factory is None:
        from INTENT2FIX import actions  # noqa: F401
        from INTENT2FIX.runtime.dockerenv import DockerEnv
        env_factory = DockerEnv
    result = {"candidate_id": candidate.get("candidate_id"),
              "patch_sha256": candidate["patch_sha256"], "base_commit": base_commit,
              "image": image, "status": "unknown", "phase": "setup",
              "build_route": None, "command": None, "exit_code": None, "output": ""}
    try:
        with env_factory(image_name=image, container_name=f"selector-build-{uuid.uuid4().hex[:12]}",
                         work_dir=work_dir, benchmark_base_commit=base_commit) as env:
            _write(env, ".intent2fix-selector.patch", candidate["patch"])
            output, code = env.run_bash(
                "git apply --check .intent2fix-selector.patch && git apply .intent2fix-selector.patch",
                timeout=60)
            if code != 0:
                return {**result, "status": "failed", "phase": "apply", "exit_code": code,
                        "output": output[-6000:]}
            route, code = env.run_bash(
                "if test -f CMakeLists.txt; then echo cmake; "
                "elif test -f Makefile; then echo make; else echo unavailable; fi", timeout=30)
            route = route.strip()
            if code != 0 or route not in {"cmake", "make"}:
                return {**result, "phase": "route", "output": "No supported public build route"}
            result["build_route"] = route
            if route == "cmake":
                configure = ["cmake", "-S", ".", "-B", "/tmp/intent2fix-selector-build",
                             "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", *_cmake_public_options(env)]
                result["command"] = (shlex.join(configure) + " && "
                                     "cmake --build /tmp/intent2fix-selector-build --parallel 2")
                output, code = env.run_bash(shlex.join(configure), timeout=180)
                if code != 0:
                    return {**result, "phase": "configure", "exit_code": code,
                            "output": output[-6000:]}
                output, code = env.run_bash(
                    "cmake --build /tmp/intent2fix-selector-build --parallel 2", timeout=900)
                if code == 0:
                    _, targets = env.run_bash(
                        "grep -q '\"file\"' /tmp/intent2fix-selector-build/compile_commands.json", timeout=30)
                    if targets != 0:
                        return {**result, "phase": "build", "output":
                                "CMake succeeded without a C/C++ compilation target"}
            else:
                result["command"] = "make -B -j2"
                output, code = env.run_bash("make -B -j2", timeout=900)
            return {**result, "status": "passed" if code == 0 else
                    "unknown" if code in (124, None) else "failed",
                    "phase": "build", "exit_code": code, "output": output[-6000:]}
    except Exception as exc:
        return {**result, "phase": "infrastructure", "output": type(exc).__name__}


def execution_rows(case: dict, probe: Probe, *, image: str, env_factory=None, attempt="") -> list[dict]:
    """Fresh container per (probe, patch); save phases even when setup/compile fails.

    Only completed runs with independently bound oracles are adjudicative. Compiler
    diagnostics remain observations: candidate attribution requires a static review.
    """
    if env_factory is None:
        # Match the application's import order. actions.verification_sync imports
        # DockerEnv; importing dockerenv first creates a circular initialization.
        from INTENT2FIX import actions  # noqa: F401
        from INTENT2FIX.runtime.dockerenv import DockerEnv
        env_factory = DockerEnv
    commands = probe_commands(probe)
    probe_sha = object_hash(probe.model_dump())
    prefix = f"{probe.probe_id}:{attempt}" if attempt else probe.probe_id
    rows = [{"evidence_id": f"{prefix}:definition", "candidate_id": "BASELINE", "patch_sha256": None,
             "base_commit": case["base_commit"], "probe_id": probe.probe_id, "probe_sha256": probe_sha,
             "phase": "probe_definition", "status": "frozen", "adjudicative": False,
             "build_route": commands[0], "probe": probe.model_dump()}]
    candidates = [{"candidate_id": "BASELINE", "patch_sha256": None, "patch": ""}, *case["candidates"]]
    for candidate in candidates:
        cid = candidate["candidate_id"]
        common = {"candidate_id": cid, "patch_sha256": candidate["patch_sha256"],
                  "probe_id": probe.probe_id, "probe_sha256": probe_sha,
                  "base_commit": case["base_commit"], "clause_id": probe.clause_id,
                  "image": image, "build_route": commands[0]}
        def record(phase, status, output="", exit_code=None, adjudicative=False):
            prefix = f"{probe.probe_id}:{attempt}" if attempt else probe.probe_id
            row = {**common, "evidence_id": f"{prefix}:{cid}:{phase}",
                   "phase": phase, "status": status, "exit_code": exit_code,
                   "output": output[:6000] + ("\n[truncated]" if len(output) > 6000 else ""),
                   "adjudicative": adjudicative}
            rows.append(row)
            return row
        try:
            with env_factory(image_name=image, container_name=f"selector-v42-{uuid.uuid4().hex[:12]}",
                             work_dir=f"/home/{case['repo'].split('/')[1]}",
                             benchmark_base_commit=case["base_commit"]) as env:
                record("setup", "passed")
                if candidate["patch"]:
                    _write(env, ".intent2fix-selector.patch", candidate["patch"])
                    output, code = env.run_bash("git apply --check .intent2fix-selector.patch && git apply .intent2fix-selector.patch", timeout=60)
                    record("apply", "passed" if code == 0 else "failed", output, code)
                    if code != 0:
                        continue
                _, code = env.run_bash("test ! -e .intent2fix_selector && mkdir .intent2fix_selector", timeout=30)
                if code:
                    record("fixture", "unknown", "Probe namespace already exists", code)
                    continue
                _write(env, PROBE_PATH, probe.source)
                # Include added files too: plain `git diff HEAD` omits untracked
                # production files created by a patch. This index is container-local.
                snapshot = "git add --all && git diff --cached --binary HEAD"
                before, code = env.run_bash(snapshot, timeout=30)
                if code:
                    record("integrity", "unknown", "Cannot snapshot candidate source", code)
                    continue
                output, code = env.run_bash(shlex.join(commands[0]), timeout=180)
                record("compile", "passed" if code == 0 else "unknown", output, code)
                if code != 0:
                    continue
                output, code = env.run_bash(shlex.join(commands[1]), timeout=30)
                record("run", "observed" if code not in (124, 126, 127, None) else "unknown", output, code)
                after, diff_code = env.run_bash(snapshot, timeout=30)
                intact = (diff_code == 0 and before == after
                          and OUTPUT_CAPTURE_TRUNCATION_MARKER not in before
                          and OUTPUT_CAPTURE_TRUNCATION_MARKER not in after)
                record("integrity", "passed" if intact else "unknown",
                       "Product source unchanged during probe" if intact else "Product source mutated or integrity unavailable")
                has_oracle = probe.clause_id is not None and (
                    probe.expected_exit_code is not None or probe.required_output_regex is not None)
                if (not intact or code in (124, 126, 127, None) or not has_oracle
                        or OUTPUT_CAPTURE_TRUNCATION_MARKER in output):
                    record("assert", "unknown", "No valid completed run with an independent oracle")
                else:
                    matches = (probe.expected_exit_code is None or code == probe.expected_exit_code) and (
                        probe.required_output_regex is None or re.search(probe.required_output_regex, output) is not None)
                    record("assert", "passed" if matches else "failed", output, code, adjudicative=True)
        except Exception as exc:
            # Provider/Docker exception text can include configuration. No raw secrets/logs.
            record("infrastructure", "unknown", type(exc).__name__)
    return rows
