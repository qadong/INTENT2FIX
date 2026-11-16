"""Cross-action regressions for saved Repair mechanism failures."""
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock
from contextlib import contextmanager

import pytest

from INTENT2FIX.actions.create_file import CreateFile, CreateFileArgs
from INTENT2FIX.actions.run_single_test import RunSingleTest, RunSingleTestArgs, _compute_passed as focused_passed
from INTENT2FIX.actions.run_regression import RunRegression, RunRegressionArgs, _compute_passed as regression_passed
from INTENT2FIX.actions.test_parser import parse_test_output, LogType
from INTENT2FIX.actions.verification_sync import (
    PRODUCTION_ISOLATION, mark_file_change, sync_verification_files, run_isolated_production_command,
)
from INTENT2FIX.file_context import FileContext
from INTENT2FIX.patch_review.sanitizer import sanitize_candidate_patch
from INTENT2FIX.repository.repository import InMemRepository
from INTENT2FIX.runtime.dockerenv import DockerEnv
from tests.test_cotest_patch_selection import RUNNER


@pytest.mark.parametrize("path", ["src/reprocess.cpp", "include/probe_policy.hpp", "src/poc_module.cpp"])
def test_explicit_production_overrides_probe_name_in_saved_context(path):
    context = FileContext(repo=InMemRepository())
    CreateFile().execute(CreateFileArgs(path=path, file_text="int value;\n", artifact_kind="production"), context)
    # Rehydrate: an exception must survive cloning and trajectory persistence.
    reloaded = FileContext.from_dict(context.model_dump(), repo=InMemRepository())
    candidate = sanitize_candidate_patch(reloaded)
    assert path in candidate.kept_paths
    assert candidate.patch and not candidate.artifact_errors


@pytest.mark.parametrize("registered", [False, True])
def test_fixed_test_and_registered_reproduction_boundaries_win_over_production_name(registered):
    path = "repro.cpp" if registered else "tests/probe.cpp"
    context = FileContext(repo=InMemRepository())
    CreateFile().execute(CreateFileArgs(path=path, file_text="int main() {}\n", artifact_kind="production"), context)
    if registered:
        context.verification["immutable_reproduction_paths"] = [path]
    assert not sanitize_candidate_patch(context).patch


@pytest.mark.parametrize("output,exit_code", [
    ("[       OK ] smoke\nSegmentation fault (core dumped)\n", 139),
    ("[       OK ] smoke\nTimed out\n", 124),
    ("1/2 Test #1: smoke .... Passed\n2/2 Test #2: absent ....***Not Run\n", 8),
    ("No tests were found!!!\n", 0),
    ("[       OK ] earlier_suite\nNo tests ran.\n", 0),
])
def test_partial_missing_or_crashed_tests_do_not_pass(output, exit_code):
    summary = parse_test_output(output)
    assert not regression_passed(exit_code, summary)
    assert not focused_passed(exit_code, summary, output=output)


@pytest.mark.parametrize("output", [
    "probe.cpp:1:3: fatal error: absent.hpp: No such file or directory\n",
    "c++: command not found\n",
])
def test_setup_error_cannot_satisfy_negative_compile_oracle(output):
    assert not focused_passed(1, parse_test_output(output), expected_exit_code=1,
                              output=output, required_output_regex=".*")


def test_satisfied_negative_compile_does_not_request_compilation_repair():
    runtime = Mock(spec=DockerEnv)
    runtime.run_bash.return_value = (
        "negative.cpp:2:3: error: Formatting of non-void pointers is disallowed\n", 1
    )
    context = FileContext(repo=InMemRepository())
    action = RunSingleTest(runtime=runtime)
    args = RunSingleTestArgs(thoughts="Expect rejection", command="c++ negative.cpp",
                             expected_exit_code=1, required_output_regex="non-void pointers is disallowed")
    for _ in range(2):
        observation = action.execute(args, context)
        assert observation.properties["oracle_satisfied"]
        assert not observation.properties["compilation_failure_signature"]
        assert "Repeated compilation failure" not in observation.message


def test_make_test_failure_cannot_trigger_engineering_correction():
    output = "1/1 Test #1: legacy_message ....***Failed\nmake: *** [Makefile:71: test] Error 8\n"
    parsed = parse_test_output(output)
    assert parsed.log_type == LogType.TEST_RESULTS
    context = FileContext(repo=InMemRepository())
    context.add_file("src/fix.cpp").apply_changes("int value;\n")
    candidate = sanitize_candidate_patch(context)
    record = {"passed": False, "exit_code": 2, "log_type": parsed.log_type.value,
              "output": output, "change_seq": 0, "verification_view": "production",
              "verification_isolation": PRODUCTION_ISOLATION,
              "production_patch_sha256": RUNNER._portfolio_patch_sha256(candidate.patch)}
    context.verification["production_regression"] = record
    assert not RUNNER._portfolio_engineering_diagnostic(context, candidate)
    record.update(log_type="compilation_error", output="src/fix.cpp:1:3: error: no matching function")
    assert RUNNER._portfolio_engineering_diagnostic(context, candidate)
    record.pop("verification_isolation")
    assert RUNNER._fresh_portfolio_regression(context) is None


@pytest.mark.parametrize("error", [
    "test case THREW exception: invalid input",
    "CHECK( value == 1 ) is NOT correct!",
    "REQUIRE_THROWS_AS( parse(input), parse_error ) did NOT throw at all!",
])
def test_doctest_runtime_errors_are_not_compiler_errors(error):
    output = ("6/79 Test #6: test-bjdata_cpp11 ....***Failed\n"
              f"/home/json/tests/src/unit-bjdata.cpp:119: ERROR: {error}\n")
    summary = parse_test_output(output)
    assert summary.log_type == LogType.TEST_RESULTS
    assert summary.results[0].status.value == "failed"
    # A real compiler failure elsewhere must still be recognized.
    summary = parse_test_output(output + "src/other.cpp:2:3: error: invalid conversion\n")
    assert summary.log_type == LogType.COMPILATION_ERROR
    assert "invalid conversion" in summary.summary_line


@pytest.fixture
def local_repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/fix.cpp").write_text("int value;\n")
    (tmp_path / ".gitignore").write_text("include/ambient.hpp\nbuild/\n")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"],
                   cwd=tmp_path, check=True, capture_output=True)
    return tmp_path


@pytest.fixture
def compiler():
    executable = shutil.which("c++")
    if not executable:
        pytest.skip("C++ compiler unavailable")
    return executable


def local_runtime(directory, compiler):
    runtime = Mock(spec=DockerEnv)
    runtime.work_dir = str(directory)
    runtime.image_name = "local-image-fixture"
    runtime.benchmark_base_commit = None
    def run(command, **kwargs):
        # Replace only the unavailable benchmark entry point with real C++.
        if "/home/build_run.sh" in command:
            command = f"{compiler} -fsyntax-only -Iinclude src/fix.cpp"
        done = subprocess.run(["sh", "-c", command], cwd=directory, capture_output=True, text=True)
        return done.stdout + done.stderr, done.returncode
    def copy(source, target):
        Path(target).write_bytes(Path(source).read_bytes())
        return True
    runtime.run_bash.side_effect = run
    runtime.copy_anything_to_container.side_effect = copy
    return runtime


@pytest.fixture
def fresh_local_image(monkeypatch, tmp_path, compiler):
    instances = []
    @contextmanager
    def create(**kwargs):
        assert kwargs["image_name"] == "local-image-fixture"
        assert kwargs["benchmark_base_commit"] is None
        assert kwargs["container_name"].startswith("production-verify-")
        assert "host_work_dir" not in kwargs
        directory = tmp_path / f"fresh-image-{len(instances)}"
        directory.mkdir()
        (directory / "src").mkdir()
        (directory / "src/fix.cpp").write_text("int value;\n")
        # Model immutable image-supplied ignored test data, absent from the patch.
        (directory / "test_data").mkdir()
        (directory / "test_data/input.txt").write_text("image supplied data")
        (directory / "dependencies").mkdir()
        (directory / "dependencies/import.cmake").write_text("# baseline dependency importer\n")
        instances.append(directory)
        try:
            yield local_runtime(directory, compiler)
        finally:
            shutil.rmtree(directory)
    monkeypatch.setattr("INTENT2FIX.actions.verification_sync.DockerEnv", create)
    return instances


def test_production_regression_excludes_unknown_ignored_header_then_restores_it(local_repo, compiler, fresh_local_image):
    runtime = local_runtime(local_repo, compiler)
    context = FileContext(repo=InMemRepository({"src/fix.cpp": "int value;\n"}))
    context.add_file("src/fix.cpp").apply_changes('#include "ambient.hpp"\nint value = VALUE;\n')
    (local_repo / "include").mkdir()
    header = local_repo / "include/ambient.hpp"
    header.write_text("#define VALUE 1\n")
    candidate = sanitize_candidate_patch(context)
    context.verification["production_regression"] = {
        "passed": True, "change_seq": 0, "verification_view": "production",
        "production_patch_sha256": RUNNER._portfolio_patch_sha256(candidate.patch),
    }
    result = RunRegression(runtime=runtime).execute(RunRegressionArgs(thoughts="check delivery"), context)
    assert not result.properties.get("reused")
    record = context.verification["production_regression"]
    assert not record["passed"] and record["log_type"] == "compilation_error"
    assert record["verification_isolation"] == PRODUCTION_ISOLATION
    assert header.read_text() == "#define VALUE 1\n"
    assert fresh_local_image and all(not p.exists() for p in fresh_local_image)
    # A registered production header is included in the next isolated build.
    CreateFile().execute(CreateFileArgs(path="include/ambient.hpp", file_text=header.read_text(), artifact_kind="production"), context)
    RunRegression(runtime=runtime).execute(RunRegressionArgs(thoughts="check complete delivery"), context)
    assert context.verification["production_regression"]["passed"]


@pytest.mark.parametrize("command,exit_code", [("exit 7", 7), ("test -f test_data/input.txt && test -d dependencies/.cache && test ! -e 'notes with spaces/saved.txt'", 0)])
def test_fresh_verification_leaves_live_files_and_keeps_image_data(local_repo, compiler, fresh_local_image, command, exit_code):
    directory = local_repo / "notes with spaces"
    directory.mkdir()
    (directory / "saved.txt").write_text("keep exact contents")
    runtime = local_runtime(local_repo, compiler)
    _, actual = run_isolated_production_command(runtime, command, [], FileContext(repo=InMemRepository()))
    assert actual == exit_code
    assert (directory / "saved.txt").read_text() == "keep exact contents"
    assert all(not p.exists() for p in fresh_local_image)


def test_fresh_verification_cleanup_on_sync_failure(monkeypatch, local_repo, compiler, fresh_local_image):
    runtime = local_runtime(local_repo, compiler)
    context = FileContext(repo=InMemRepository())
    context.verification["_container_synced_patched_files"] = ["live-only.cpp"]
    monkeypatch.setattr("INTENT2FIX.actions.verification_sync.sync_verification_files", lambda *a, **k: "copy failed")
    with pytest.raises(RuntimeError, match="copy failed"):
        run_isolated_production_command(runtime, "exit 0", [], context)
    assert fresh_local_image and all(not p.exists() for p in fresh_local_image)
    assert context.verification["_container_synced_patched_files"] == ["live-only.cpp"]


def test_patch_identity_shared_by_probe_regression_candidate_and_audit(local_repo, compiler, fresh_local_image):
    from INTENT2FIX.intent_portfolio import PortfolioCandidate, CrossExecutionResult
    runtime = local_runtime(local_repo, compiler)
    context = FileContext(repo=InMemRepository({"src/fix.cpp": "int value;\n"}))
    context.add_file("src/fix.cpp").apply_changes("int value = 1;\n")
    candidate = sanitize_candidate_patch(context)
    observation = RunSingleTest(runtime=runtime).execute(
        RunSingleTestArgs(thoughts="Compile", command=f"{compiler} -fsyntax-only src/fix.cpp"), context)
    RunRegression(runtime=runtime).execute(RunRegressionArgs(thoughts="Verify delivery"), context)
    stored = PortfolioCandidate.create(
        hypothesis_id="H0", branch_assumption="", patch=candidate.patch,
        kept_paths=candidate.kept_paths, excluded_paths={}, finish_succeeded=True,
        basic_tests_passed=True, cross_execution=CrossExecutionResult(status="unavailable"))
    evidence = RUNNER._portfolio_compliance_regression_evidence(context)
    assert observation.properties["execution_snapshot"]["production_patch_sha256"] == stored.patch_sha256
    assert context.verification["production_regression"]["production_patch_sha256"] == stored.patch_sha256
    assert all(r["production_patch_sha256"] == stored.patch_sha256 and r["verification_view"] == "production"
               for r in evidence.values())
    context.verification["production_regression"]["production_patch_sha256"] = "earlier-patch"
    evidence = RUNNER._portfolio_compliance_regression_evidence(context)
    assert evidence["production_only_with_original_tests"]["verification_view"] == "unproven"


def test_test_then_edit_does_not_transfer_success_to_unexecuted_patch(local_repo, compiler):
    runtime = local_runtime(local_repo, compiler)
    context = FileContext(repo=InMemRepository({"src/fix.cpp": "int value;\n"}))
    source = context.add_file("src/fix.cpp")
    source.apply_changes("int value = 1;\n")
    mark_file_change(context, "src/fix.cpp")
    tested = sanitize_candidate_patch(context)
    args = RunSingleTestArgs(thoughts="compile", command=f"{compiler} -fsyntax-only src/fix.cpp")
    observation = RunSingleTest(runtime=runtime).execute(args, context)
    assert observation.properties["oracle_satisfied"]
    source.apply_changes("this is invalid C++;\n")
    mark_file_change(context, "src/fix.cpp")
    candidate = sanitize_candidate_patch(context)
    node = SimpleNamespace(file_context=context, action_steps=[SimpleNamespace(action=args, observation=observation)])
    trajectory = SimpleNamespace(get_all_nodes=lambda: [node])
    assert RUNNER._collect_final_patch_repair_probes(final_trajectory=trajectory, candidate=candidate, hypothesis_context={}) == []
    probes = RUNNER._collect_final_patch_repair_probes(final_trajectory=trajectory, candidate=tested, hypothesis_context={})
    assert len(probes) == 1 and probes[0].observed_status == "passed"
    sync_verification_files(context, runtime)
    assert runtime.run_bash(args.command)[1] != 0
    observation.properties.pop("execution_snapshot")
    assert RUNNER._collect_final_patch_repair_probes(final_trajectory=trajectory, candidate=tested, hypothesis_context={}) == []


def test_probe_source_and_oracle_are_frozen_at_execution():
    runtime = Mock(spec=DockerEnv)
    runtime.work_dir = "/repo"
    runtime.run_bash.return_value = ("expected\n", 0)
    runtime.copy_anything_to_container.return_value = True
    context = FileContext(repo=InMemRepository({"src/fix.cpp": "int value;\n"}))
    context.add_file("src/fix.cpp").apply_changes("int value = 1;\n")
    context.add_file("probe.cpp").apply_changes("int original;\n")
    context.verification["ephemeral_paths"] = ["probe.cpp"]
    args = RunSingleTestArgs(thoughts="run original probe", command="c++ probe.cpp", required_output_regex="expected")
    observation = RunSingleTest(runtime=runtime).execute(args, context)
    context.get_file("probe.cpp").apply_changes("int changed;\n")
    node = SimpleNamespace(file_context=context, action_steps=[SimpleNamespace(action=args, observation=observation)])
    trajectory = SimpleNamespace(get_all_nodes=lambda: [node])
    candidate = sanitize_candidate_patch(context)
    probes = RUNNER._collect_final_patch_repair_probes(final_trajectory=trajectory, candidate=candidate, hypothesis_context={})
    assert probes[0].files["probe.cpp"] == "int original;\n"
    args.required_output_regex = "different oracle"
    assert RUNNER._collect_final_patch_repair_probes(final_trajectory=trajectory, candidate=candidate, hypothesis_context={}) == []
