from types import SimpleNamespace
from pathlib import Path
import shutil
import subprocess

import pytest

from INTENT2FIX.actions.finish import FinishArgs
from INTENT2FIX.actions.run_command import RunCommand
from INTENT2FIX.actions.unified_repair import (
    BashArgs,
    CodeGraphArgs,
    EditorArgs,
    UnifiedBash,
    UnifiedCodeGraph,
    UnifiedEditor,
)
from INTENT2FIX.actions.schema import Observation
from INTENT2FIX.agent.code_agent import CodingAgent
from INTENT2FIX.completion import LLMResponseFormat
from INTENT2FIX.completion.base import BaseCompletionModel
from INTENT2FIX.index.code_index import CodeIndex
from INTENT2FIX.repository.repository import InMemRepository
from INTENT2FIX.runtime.runtime import RuntimeEnvironment
from INTENT2FIX.file_context import FileContext
from INTENT2FIX.patch_review.sanitizer import sanitize_candidate_patch
from INTENT2FIX.utils.patch import production_patch_sha256
from scripts import run_repair_only_ablation as repair_runner


class _Runtime:
    def __init__(self):
        self.commands = []

    def run_bash(self, command):
        self.commands.append(command)
        return "compiled and passed", 0


class _CodeIndex(CodeIndex):
    def __init__(self):
        pass


class _LocalRuntime:
    """Exercise actual shell/capture/sync behavior without Docker or a model."""

    def __init__(self, root):
        self.work_dir = str(root)

    def run_bash(self, command, work_dir=None):
        result = subprocess.run(
            command, shell=True, cwd=work_dir or self.work_dir, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        return result.stdout, result.returncode

    def copy_anything_to_container(self, source, target):
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        return True


@pytest.fixture
def local_tools(tmp_path):
    baseline = "int old_value;\n"
    (tmp_path / "old.hpp").write_text(baseline)
    for args in [
        ["init", "-q"], ["add", "old.hpp"],
        ["-c", "user.name=Test", "-c", "user.email=test@example.invalid",
         "commit", "-q", "-m", "baseline"],
    ]:
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    repo = InMemRepository(files={"old.hpp": baseline})
    context = FileContext(repo=repo)
    runtime = _LocalRuntime(tmp_path)
    completion = BaseCompletionModel.create(
        response_format=LLMResponseFormat.TOOLS, model="gpt-3.5-turbo",
        max_tokens=1000, model_api_key="dummy_key",
    )
    editor = UnifiedEditor(repository=repo, code_index=None,
                           completion_model=completion, runtime=runtime)
    created = editor.execute(EditorArgs(command="create", path="include/new.hpp",
                                        file_text="int original;\n"), file_context=context)
    assert created.properties["success"]
    return UnifiedBash(runtime=runtime), editor, context, tmp_path


def test_bash_edit_of_editor_created_header_survives_next_sync(local_tools):
    action, _, context, root = local_tools
    changed = action.execute(BashArgs(command="printf 'int corrected;\\n' > include/new.hpp"),
                             file_context=context)
    assert "include/new.hpp" in changed.properties["tracked_changes"]
    assert context.get_file("include/new.hpp").content == "int corrected;\n"
    read = action.execute(BashArgs(command="cat include/new.hpp"), file_context=context)
    assert read.properties["output"] == "int corrected;\n"
    assert "int corrected;" in context.generate_git_patch()
    assert (root / "include/new.hpp").read_text() == "int corrected;\n"


def test_bash_truncation_keeps_managed_file_distinct_from_deletion(local_tools):
    action, _, context, root = local_tools
    action.execute(BashArgs(command=": > include/new.hpp"), file_context=context)
    assert context.get_file("include/new.hpp").content == ""
    assert "include/new.hpp" in context.verification.get("production_paths", [])
    action.execute(BashArgs(command="test -f include/new.hpp"), file_context=context)
    assert (root / "include/new.hpp").is_file()
    assert (root / "include/new.hpp").read_text() == ""


def test_bash_imports_managed_edits_even_when_command_later_fails(local_tools):
    action, _, context, _ = local_tools
    observed = action.execute(
        BashArgs(command="printf 'int corrected;\\n' > include/new.hpp && false"),
        file_context=context,
    )
    assert observed.properties["exit_code"] == 1
    assert context.get_file("include/new.hpp").content == "int corrected;\n"
    read = action.execute(BashArgs(command="cat include/new.hpp"), file_context=context)
    assert read.properties["output"] == "int corrected;\n"


def test_bash_delete_allows_editor_recreate_without_old_header_resurrection(local_tools):
    action, editor, context, root = local_tools
    removed = action.execute(BashArgs(command="rm include/new.hpp"), file_context=context)
    assert removed.properties["exit_code"] == 0
    assert "include/new.hpp" not in context.get_patched_files()
    assert "include/new.hpp" not in context.verification.get("production_paths", [])
    read = action.execute(BashArgs(command="test ! -e include/new.hpp"), file_context=context)
    assert read.properties["exit_code"] == 0
    assert not (root / "include/new.hpp").exists()
    created = editor.execute(EditorArgs(command="create", path="include/new.hpp",
                                        file_text="int replacement;\n"), file_context=context)
    assert created.properties["success"]
    read = action.execute(BashArgs(command="cat include/new.hpp"), file_context=context)
    assert read.properties["output"] == "int replacement;\n"


def test_verification_cannot_mutate_editor_created_header_unnoticed(local_tools):
    action, _, context, _ = local_tools
    observed = action.execute(
        BashArgs(command="printf 'int changed;\\n' > include/new.hpp", verification=True),
        file_context=context,
    )
    assert observed.properties["fail_reason"] == "verification_modified_repository"
    assert observed.properties["oracle_satisfied"] is False


def test_bash_does_not_import_unregistered_disposable_files(local_tools):
    action, _, context, _ = local_tools
    action.execute(BashArgs(command="printf 'int temporary;\\n' > scratch.cpp"),
                   file_context=context)
    assert "scratch.cpp" not in context.get_patched_files()


def test_editor_errors_name_available_view_and_replace_operations(local_tools):
    _, editor, context, _ = local_tools
    existing = editor.execute(EditorArgs(command="create", path="include/new.hpp",
                                         file_text="int replacement;\n"), file_context=context)
    assert 'Editor(command="view")' in existing.message
    assert 'command="replace"' in existing.message
    assert "CreateFile" not in existing.message
    not_viewed = editor.execute(EditorArgs(command="replace", path="old.hpp",
                                           old_str="old_value", new_str="corrected"),
                                file_context=context)
    assert 'Editor(command="view")' in not_viewed.message
    assert "ViewCode" not in not_viewed.message


def test_rejected_fixture_creation_explains_actual_four_tool_recovery(verified_bash):
    action, context, runtime = verified_bash
    command = "cat > /tmp/p.cpp <<'EOF'\nint main(){return 0;}\nEOF\nc++ /tmp/p.cpp"
    observed = action.execute(BashArgs(command=command, verification=True,
                                      probe_paths=["/tmp/p.cpp"]), file_context=context)
    assert observed.expect_correction
    assert "verification=false" in observed.message
    assert "No command was executed" in observed.message
    assert "RunCommand" not in observed.message
    assert runtime.commands == []


def test_two_phase_fixture_recovery_freezes_actual_assertions(local_tools):
    if not shutil.which("c++"):
        pytest.skip("C++ compiler unavailable")
    action, _, context, root = local_tools
    source = '#include "include/new.hpp"\n#include <cassert>\nint main(){assert(original==0);}\n'
    setup = f"cat > issue.cpp <<'EOF'\n{source}EOF"
    rejected = action.execute(
        BashArgs(command=setup + "\nc++ issue.cpp -o issue-check && ./issue-check",
                 verification=True, probe_paths=["issue.cpp"]),
        file_context=context,
    )
    assert rejected.expect_correction
    assert not (root / "issue.cpp").exists()
    prepared = action.execute(BashArgs(command=setup), file_context=context)
    assert prepared.properties["exit_code"] == 0
    assert (root / "issue.cpp").read_text() == source
    checked = action.execute(
        BashArgs(command="c++ issue.cpp -o issue-check && ./issue-check",
                 verification=True, probe_paths=["issue.cpp"]),
        file_context=context,
    )
    assert checked.properties["exit_code"] == 0
    assert checked.properties["oracle_satisfied"] is True
    assert checked.properties["execution_snapshot"]["probe_files"] == {"issue.cpp": source}
    assert "issue.cpp" not in context.get_patched_files()


def test_verification_inherits_registered_source_after_ordinary_commands(local_tools):
    action, _, context, root = local_tools
    source = "original witness\n"
    (root / "witness.cpp").write_text(source)
    declared = action.execute(
        BashArgs(command="test -s witness.cpp", verification=True, probe_paths=["witness.cpp"]),
        file_context=context,
    )
    assert declared.properties["oracle_satisfied"] is True
    action.execute(BashArgs(command="cat witness.cpp"), file_context=context)
    inherited = action.execute(
        BashArgs(command="test -s witness.cpp", verification=True), file_context=context,
    )
    assert inherited.properties["execution_snapshot"]["probe_files"] == {"witness.cpp": source}
    assert inherited.properties["execution_snapshot"]["inherited_probe_paths"] == ["witness.cpp"]
    assert inherited.properties["oracle_satisfied"] is True


def test_verification_rejects_fixture_mutation_when_paths_are_omitted(local_tools):
    action, _, context, root = local_tools
    source = "original witness\n"
    (root / "witness.cpp").write_text(source)
    action.execute(
        BashArgs(command="test -s witness.cpp", verification=True, probe_paths=["witness.cpp"]),
        file_context=context,
    )
    action.execute(BashArgs(command="cat witness.cpp"), file_context=context)
    modified = action.execute(
        BashArgs(command="python3 - <<'PY'\np='witness.cpp'\ns=open(p).read()\n"
                         "open(p,'w').write(s.replace('original', 'weakened'))\nPY\n"
                         "test -s witness.cpp", verification=True),
        file_context=context,
    )
    assert modified.properties["exit_code"] == 0
    assert modified.properties["fail_reason"] == "verification_modified_fixture"
    assert modified.properties["oracle_satisfied"] is False
    assert modified.properties["execution_snapshot"]["probe_files"] == {"witness.cpp": source}
    assert modified.expect_correction


def test_verification_keeps_known_paths_when_another_fixture_is_declared(local_tools):
    action, _, context, root = local_tools
    for name in ["first.cpp", "second.cpp"]:
        (root / name).write_text(name + "\n")
        observed = action.execute(
            BashArgs(command=f"test -s {name}", verification=True, probe_paths=[name]),
            file_context=context,
        )
        assert observed.properties["oracle_satisfied"] is True
    inherited = action.execute(
        BashArgs(command="test -s first.cpp", verification=True), file_context=context,
    )
    assert inherited.properties["execution_snapshot"]["probe_files"] == {
        "first.cpp": "first.cpp\n", "second.cpp": "second.cpp\n",
    }


def test_verification_retires_deleted_remembered_fixture_not_explicit_missing_path(local_tools):
    action, _, context, root = local_tools
    (root / "witness.cpp").write_text("original witness\n")
    action.execute(
        BashArgs(command="test -s witness.cpp", verification=True, probe_paths=["witness.cpp"]),
        file_context=context,
    )
    action.execute(BashArgs(command="rm witness.cpp"), file_context=context)
    repository_check = action.execute(
        BashArgs(command="test -f old.hpp", verification=True), file_context=context,
    )
    assert repository_check.properties["oracle_satisfied"] is True
    assert context.verification["bash_probe_paths"] == []
    explicit_missing = action.execute(
        BashArgs(command="test -f old.hpp", verification=True, probe_paths=["witness.cpp"]),
        file_context=context,
    )
    assert explicit_missing.properties["fail_reason"] == "verification_snapshot_unavailable"
    assert explicit_missing.expect_correction


def test_verification_freezes_latest_source_after_separate_edit(local_tools):
    action, _, context, root = local_tools
    (root / "witness.cpp").write_text("original witness\n")
    action.execute(
        BashArgs(command="test -s witness.cpp", verification=True, probe_paths=["witness.cpp"]),
        file_context=context,
    )
    action.execute(BashArgs(command="printf 'new witness\\n' > witness.cpp"), file_context=context)
    checked = action.execute(
        BashArgs(command="test -s witness.cpp", verification=True), file_context=context,
    )
    assert checked.properties["execution_snapshot"]["probe_files"] == {"witness.cpp": "new witness\n"}
    assert checked.properties["oracle_satisfied"] is True


def test_verification_does_not_drop_oversized_remembered_fixture(local_tools):
    action, _, context, root = local_tools
    (root / "witness.cpp").write_text("original witness\n")
    action.execute(
        BashArgs(command="test -s witness.cpp", verification=True, probe_paths=["witness.cpp"]),
        file_context=context,
    )
    (root / "witness.cpp").write_text("x" * 131073)
    checked = action.execute(
        BashArgs(command="test -s witness.cpp", verification=True), file_context=context,
    )
    assert checked.properties["fail_reason"] == "verification_snapshot_unavailable"
    assert checked.expect_correction


def test_verification_does_not_retire_unreadable_remembered_fixture(local_tools, monkeypatch):
    action, _, context, root = local_tools
    (root / "witness.cpp").write_text("original witness\n")
    action.execute(
        BashArgs(command="test -s witness.cpp", verification=True, probe_paths=["witness.cpp"]),
        file_context=context,
    )
    run_bash = action._runtime.run_bash
    def unreadable(command, work_dir=None):
        if command == "head -c 131073 -- witness.cpp":
            return "Permission denied", 1
        return run_bash(command, work_dir=work_dir)
    monkeypatch.setattr(action._runtime, "run_bash", unreadable)
    checked = action.execute(
        BashArgs(command="test -f old.hpp", verification=True), file_context=context,
    )
    assert checked.properties["fail_reason"] == "verification_snapshot_unavailable"
    assert context.verification["bash_probe_paths"] == ["witness.cpp"]


def test_verification_detects_deleted_fixture_even_when_paths_are_omitted(local_tools):
    action, _, context, root = local_tools
    (root / "witness.cpp").write_text("original witness\n")
    action.execute(
        BashArgs(command="test -s witness.cpp", verification=True, probe_paths=["witness.cpp"]),
        file_context=context,
    )
    observed = action.execute(
        BashArgs(command="rm witness.cpp", verification=True), file_context=context,
    )
    assert observed.properties["exit_code"] == 0
    assert observed.properties["fail_reason"] == "verification_modified_fixture"
    assert observed.properties["oracle_satisfied"] is False


def test_verification_path_memory_survives_file_context_clone(local_tools):
    action, _, context, root = local_tools
    (root / "witness.cpp").write_text("original witness\n")
    action.execute(
        BashArgs(command="test -s witness.cpp", verification=True, probe_paths=["witness.cpp"]),
        file_context=context,
    )
    cloned = context.clone()
    checked = action.execute(
        BashArgs(command="test -s witness.cpp", verification=True), file_context=cloned,
    )
    assert checked.properties["execution_snapshot"]["probe_files"] == {"witness.cpp": "original witness\n"}


@pytest.mark.parametrize("mutate", [False, True])
def test_collector_keeps_inherited_source_but_not_mutating_verification(local_tools, mutate):
    from tests.test_cotest_patch_selection import RUNNER
    action, _, context, root = local_tools
    source = "original witness\n"
    (root / "witness.cpp").write_text(source)
    action.execute(
        BashArgs(command="test -s witness.cpp", verification=True, probe_paths=["witness.cpp"]),
        file_context=context,
    )
    command = "rm witness.cpp" if mutate else "test -s witness.cpp"
    args = BashArgs(command=command, verification=True)
    observation = action.execute(args, file_context=context)
    trajectory = SimpleNamespace(get_all_nodes=lambda: [SimpleNamespace(
        file_context=context, action_steps=[SimpleNamespace(action=args, observation=observation)],
    )])
    probes = RUNNER._collect_final_patch_repair_probes(
        final_trajectory=trajectory, candidate=sanitize_candidate_patch(context), hypothesis_context=None,
    )
    assert len(probes) == (0 if mutate else 1)
    if probes:
        assert probes[0].files == {"witness.cpp": source}
        assert probes[0].observed_status == "passed"


@pytest.fixture
def verified_bash(monkeypatch):
    context = FileContext(repo=InMemRepository(files={"src/fix.cpp": "int n=1;\n"}))
    context.add_file("src/fix.cpp").apply_changes("int n=2;\n")
    runtime = _Runtime()
    monkeypatch.setattr("INTENT2FIX.actions.unified_repair.sync_verification_files", lambda *_: "")
    monkeypatch.setattr(RunCommand, "_capture_generated_changes", lambda *_: ([], ""))
    monkeypatch.setattr("INTENT2FIX.actions.unified_repair.tracked_worktree_fingerprint",
                        lambda *_: ("same", ""))
    return UnifiedBash(runtime=runtime), context, runtime


@pytest.mark.parametrize("exit_code,expected,pattern,passed", [
    (1, 0, "", False), (0, 0, "", True), (1, 1, "controlled failure", True),
    (1, 1, "different failure", False),
])
def test_bash_verification_freezes_actual_patch_fixture_and_oracle(
    verified_bash, exit_code, expected, pattern, passed
):
    action, context, runtime = verified_bash
    source = "int main() { return 1; }\n"
    def execute(command):
        runtime.commands.append(command)
        return (source, 0) if command.startswith("head -c") else ("controlled failure", exit_code)
    runtime.run_bash = execute
    args = BashArgs(command="c++ /tmp/p.cpp -o /tmp/p && /tmp/p", verification=True,
                    probe_paths=["/tmp/p.cpp"], expected_exit_code=expected, required_output_regex=pattern)
    observation = action.execute(args, file_context=context)
    snapshot = observation.properties["execution_snapshot"]
    assert snapshot["production_patch_sha256"] == production_patch_sha256(sanitize_candidate_patch(context).patch)
    assert snapshot["probe_files"] == {"/tmp/p.cpp": source}
    assert observation.properties["oracle_satisfied"] is passed
    assert observation.properties["exit_code"] == exit_code
    assert "bash -e -o pipefail" in runtime.commands[1]


@pytest.mark.parametrize("command", ["false; echo rc=$?", "false || true", "false | head"])
def test_bash_verification_rejects_masked_failure_before_execution(verified_bash, command):
    action, context, runtime = verified_bash
    observation = action.execute(BashArgs(command=command, verification=True), file_context=context)
    assert observation.expect_correction
    assert runtime.commands == []


def test_bash_verification_preserves_newline_failure(verified_bash):
    import subprocess
    action, context, runtime = verified_bash
    runtime.run_bash = lambda command: (
        lambda result: (result.stdout, result.returncode)
    )(subprocess.run(command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT))
    observation = action.execute(BashArgs(command="false\necho success", verification=True), file_context=context)
    assert observation.properties["exit_code"] == 1
    assert observation.properties["oracle_satisfied"] is False
    assert "success" not in observation.properties["output"]


@pytest.mark.parametrize("what_changed", ["repository", "fixture"])
def test_bash_mutating_verification_cannot_be_used_as_evidence(verified_bash, monkeypatch, what_changed):
    action, context, runtime = verified_bash
    if what_changed == "repository":
        fingerprints = iter([("before", ""), ("after", "")])
        monkeypatch.setattr("INTENT2FIX.actions.unified_repair.tracked_worktree_fingerprint", lambda *_: next(fingerprints))
    sources = iter(["original", "changed" if what_changed == "fixture" else "original"])
    runtime.run_bash = lambda command: (next(sources), 0) if command.startswith("head -c") else ("", 0)
    observation = action.execute(BashArgs(command="c++ /tmp/p.cpp", verification=True,
                                          probe_paths=["/tmp/p.cpp"]), file_context=context)
    assert observation.properties["fail_reason"] == f"verification_modified_{what_changed}"
    assert observation.properties["oracle_satisfied"] is False


@pytest.mark.parametrize("stale,marked,count", [(False, True, 1), (True, True, 0), (False, False, 0)])
def test_collector_accepts_only_explicit_final_revision_bash_evidence(verified_bash, stale, marked, count):
    from tests.test_cotest_patch_selection import RUNNER
    action, context, runtime = verified_bash
    runtime.run_bash = lambda command: ("assertion failed", 1)
    args = BashArgs(command="/tmp/p", verification=True)
    observation = action.execute(args, file_context=context)
    if stale:
        context.add_file("src/fix.cpp").apply_changes("int n=3;\n")
    args.verification = marked
    trajectory = SimpleNamespace(get_all_nodes=lambda: [SimpleNamespace(
        file_context=context, action_steps=[SimpleNamespace(action=args, observation=observation)]
    )])
    probes = RUNNER._collect_final_patch_repair_probes(
        final_trajectory=trajectory, candidate=sanitize_candidate_patch(context), hypothesis_context=None,
    )
    assert len(probes) == count
    if probes:
        assert probes[0].observed_status == "failed"
        assert probes[0].observed_output == "assertion failed"


class _AgentRuntime(RuntimeEnvironment):
    def run_tests(self, test_files=None):
        return []


def test_unified_tool_schema_names_are_exactly_four():
    schemas = [
        schema.tool_schema()
        for schema in (BashArgs, EditorArgs, CodeGraphArgs, FinishArgs)
    ]
    assert [schema["function"]["name"] for schema in schemas] == [
        "Bash",
        "Editor",
        "CodeGraph",
        "Finish",
    ]


def test_coding_agent_unified_profile_exposes_only_four_tools():
    completion = BaseCompletionModel.create(
        response_format=LLMResponseFormat.TOOLS,
        model="gpt-3.5-turbo",
        max_tokens=1000,
        temperature=0.7,
        model_api_key="dummy_key",
    )
    agent = CodingAgent.create(
        repository=InMemRepository(),
        code_index=_CodeIndex(),
        runtime=_AgentRuntime(),
        completion_model=completion,
        tool_profile="unified_four",
        system_prompt_override="four tools",
        graph_issue_text="An original issue",
    )

    assert [action.args_schema.name for action in agent.actions] == [
        "Bash",
        "Editor",
        "CodeGraph",
        "Finish",
    ]
    assert next(
        action for action in agent.actions if isinstance(action, UnifiedCodeGraph)
    )._session.issue_text == "An original issue"


def test_coding_agent_unified_graph_off_exposes_only_source_tools():
    completion = BaseCompletionModel.create(
        response_format=LLMResponseFormat.TOOLS,
        model="gpt-3.5-turbo",
        max_tokens=1000,
        temperature=0.7,
        model_api_key="dummy_key",
    )
    agent = CodingAgent.create(
        repository=InMemRepository(),
        code_index=None,
        runtime=_AgentRuntime(),
        completion_model=completion,
        tool_profile="unified_four",
        enable_indexed_tools=False,
        system_prompt_override="three tools",
        graph_issue_text="This issue must not reach a graph action",
    )
    assert [action.args_schema.name for action in agent.actions] == [
        "Bash", "Editor", "Finish",
    ]


def test_bash_accepts_build_commands_and_returns_raw_exit(
    monkeypatch: pytest.MonkeyPatch,
):
    runtime = _Runtime()
    action = UnifiedBash(runtime=runtime)
    context = SimpleNamespace(change_seq=3, verification={})
    monkeypatch.setattr(
        "INTENT2FIX.actions.unified_repair.sync_verification_files",
        lambda file_context, current_runtime: "",
    )
    monkeypatch.setattr(
        RunCommand,
        "_capture_generated_changes",
        lambda self, file_context: ([], ""),
    )

    observation = action.execute(
        BashArgs(command="g++ /tmp/repro.cpp -o /tmp/repro && /tmp/repro"),
        file_context=context,
    )

    assert observation.properties["exit_code"] == 0
    assert observation.properties["output"] == "compiled and passed"
    assert runtime.commands == [
        "bash -o pipefail -c 'g++ /tmp/repro.cpp -o /tmp/repro && /tmp/repro'"
    ]
    assert context.verification["bash"]["change_seq"] == 3


def test_editor_and_graph_require_operation_specific_fields():
    with pytest.raises(ValueError, match="replace requires"):
        EditorArgs(command="replace", path="src/a.cpp")
    with pytest.raises(ValueError, match="search requires behavior"):
        CodeGraphArgs(operation="search")
    with pytest.raises(ValueError, match="trace requires question"):
        CodeGraphArgs(operation="trace", source_path="src/a.cpp", source_line=3)
    with pytest.raises(ValueError, match="source_path and source_line together"):
        CodeGraphArgs(operation="trace", question="Who calls this?", source_path="src/a.cpp")
    with pytest.raises(ValueError, match="exactly one"):
        CodeGraphArgs(operation="trace", question="Who calls this?", entity_ref="GREF-1", symbol_name="f")


def test_code_graph_uses_issue_guided_search_and_source_trace(monkeypatch):
    completion = BaseCompletionModel.create(
        response_format=LLMResponseFormat.TOOLS,
        model="gpt-3.5-turbo",
        max_tokens=1000,
        temperature=0.7,
        model_api_key="dummy_key",
    )
    action = UnifiedCodeGraph(
        code_index=_CodeIndex(),
        completion_model=completion,
        issue_text="Original issue describes a broken wrapper",
    )
    calls = []

    def search_execute(self, args):
        calls.append(("search", args, self._session.issue_text))
        return Observation(message="SearchCodeIntent: use TraceInvestigationPath")

    def graph_execute(self, args):
        calls.append(("trace", args, self._session.issue_text))
        return Observation(message="[GRAPH-2] a real relation")

    monkeypatch.setattr(
        "INTENT2FIX.actions.unified_repair.SearchCodeIntent.execute",
        search_execute,
    )
    monkeypatch.setattr(
        "INTENT2FIX.actions.unified_repair.RepositoryGraphQuery.execute",
        graph_execute,
    )
    search_result = action.execute(CodeGraphArgs(operation="search", behavior="format a range"))
    action.execute(CodeGraphArgs(
        operation="trace",
        question="Which other public entry point consumes this type?",
        source_path="include/format.h",
        source_line=42,
    ))

    assert [call[0] for call in calls] == ["search", "trace"]
    assert all(call[2] == "Original issue describes a broken wrapper" for call in calls)
    assert calls[0][1].behavior == "format a range"
    assert calls[1][1].mode == "trace"
    assert calls[1][1].source_path == "include/format.h"
    assert calls[1][1].source_line == 42
    assert calls[1][1].target.startswith("Which other")
    assert search_result.message == "CodeGraph search: use CodeGraph trace"


def test_unified_runner_records_visible_tools_and_prompt():
    config = repair_runner.apply_ablation_arm(
        {
            "repair_tool_profile": repair_runner.UNIFIED_FOUR_TOOL_PROFILE,
            "repair_steps": 200,
            "model": "glm-5.3-flash",
            "reasoning_effort": "low",
        },
        repair_runner.ARM_C_REPAIR_GRAPH,
    )

    marker = repair_runner.marker_payload(config)
    assert marker["model_visible_tools"] == [
        "Bash",
        "Editor",
        "CodeGraph",
        "Finish",
    ]
    assert marker["repair_tool_profile"] == "unified_four"
    assert "representation actually consumed" in repair_runner.repair_system_prompt(
        config
    )
    assert "never edit, weaken, or delete" in repair_runner.repair_system_prompt(config)
    assert "minimum language standard" in repair_runner.repair_system_prompt(config)
    assert "operation=\"trace\"" in repair_runner.repair_system_prompt(config)
    graph_off = repair_runner.apply_ablation_arm(config, repair_runner.ARM_D_REPAIR_NO_GRAPH)
    assert "CodeGraph" not in repair_runner.repair_system_prompt(graph_off)
