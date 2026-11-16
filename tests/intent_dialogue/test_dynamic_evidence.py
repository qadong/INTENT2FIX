import json

import pytest

from INTENT2FIX.intent_dialogue import dynamic_evidence
from INTENT2FIX.intent_dialogue.developer_agent import InvestigationSession
from INTENT2FIX.intent_dialogue.dynamic_evidence import (
    CollectCoverageContrast,
    CollectCoverageContrastArgs,
    CollectXRayTraceContrast,
    CollectXRayTraceContrastArgs,
    _coverage_difference,
    _parse_coverage_export,
    _parse_xray_trace,
    _resolve_probe_run,
    _trace_difference,
)


class FakeDocker:
    work_dir = "/home/repo"


def recorded_pair() -> InvestigationSession:
    session = InvestigationSession()
    session.investigation_files = {
        "anchor.cpp": "int main() { return 0; }",
        "target.cpp": "int main() { return 1; }",
    }
    session.record(
        "c++ -std=c++17 -Iinclude anchor.cpp -o /tmp/anchor && /tmp/anchor",
        "anchor passed",
        0,
        prefix="RUN",
        action="RunInvestigationCommand",
        extra={
            "frozen_probe_files": {
                "anchor.cpp": "int main() { return 0; }",
            }
        },
    )
    session.record(
        "c++ -std=c++17 -Iinclude target.cpp -o /tmp/target && /tmp/target",
        "target failed",
        1,
        prefix="RUN",
        action="RunInvestigationCommand",
        extra={
            "frozen_probe_files": {
                "target.cpp": "int main() { return 1; }",
            }
        },
    )
    return session


def recorded_split_pair() -> InvestigationSession:
    session = InvestigationSession()
    session.investigation_files = {
        "anchor.cpp": "int main() { return 0; }",
        "target.cpp": "int main() { return 1; }",
    }
    for name, exit_code in (("anchor", 0), ("target", 1)):
        session.record(
            f"c++ -std=c++17 -Iinclude {name}.cpp -o /tmp/{name}",
            "compiled",
            0,
            prefix="RUN",
            action="RunInvestigationCommand",
            extra={
                "frozen_probe_files": {
                    f"{name}.cpp": session.investigation_files[f"{name}.cpp"],
                }
            },
        )
        session.record(
            f"/tmp/{name}",
            f"{name} executed",
            exit_code,
            prefix="RUN",
            action="RunInvestigationCommand",
            extra={
                "frozen_probe_files": {
                    f"{name}.cpp": session.investigation_files[f"{name}.cpp"],
                }
            },
        )
    return session


def test_dynamic_run_binding_uses_frozen_probe_and_compile_options():
    run = _resolve_probe_run(recorded_pair(), "RUN-1")

    assert run.files == {"anchor.cpp": "int main() { return 0; }"}
    assert run.compile_tokens[:3] == ["c++", "-std=c++17", "-Iinclude"]
    assert run.run_tokens == ["/tmp/anchor"]


def test_dynamic_run_binding_accepts_split_compile_and_execution_records():
    session = recorded_split_pair()

    from_compile = _resolve_probe_run(session, "RUN-1")
    from_execution = _resolve_probe_run(session, "RUN-4")

    assert from_compile.compile_run_id == "RUN-1"
    assert from_compile.execution_run_id == "RUN-2"
    assert from_compile.run_tokens == ["/tmp/anchor"]
    assert from_execution.compile_run_id == "RUN-3"
    assert from_execution.execution_run_id == "RUN-4"
    assert from_execution.files == {
        "target.cpp": "int main() { return 1; }"
    }


def test_dynamic_run_binding_rejects_unbound_or_shell_control_commands():
    session = InvestigationSession()
    session.record(
        "c++ probe.cpp -o /tmp/probe | head",
        "",
        0,
        prefix="RUN",
        action="RunInvestigationCommand",
        extra={"frozen_probe_files": {"probe.cpp": "int main() {}"}},
    )

    with pytest.raises(
        dynamic_evidence.DynamicEvidenceUnavailable,
        match="shell control operators",
    ):
        _resolve_probe_run(session, "RUN-1")


def test_dynamic_tool_preflight_exposes_only_complete_collectors(monkeypatch):
    def discover(docker, name, preferred_version=None):
        if name == "llvm-xray":
            raise dynamic_evidence.DynamicEvidenceUnavailable("missing")
        return f"/toolchain/{name}"

    monkeypatch.setattr(dynamic_evidence, "_discover_tool", discover)
    monkeypatch.setattr(dynamic_evidence, "_clang_major", lambda docker, compiler: 14)

    assert dynamic_evidence.available_dynamic_evidence_tools(FakeDocker()) == {
        "coverage"
    }


def test_coverage_export_is_restricted_to_repository_product_sources():
    payload = json.dumps({
        "data": [{
            "files": [
                {
                    "filename": "/home/repo/include/widget.hpp",
                    "segments": [[10, 1, 1, True], [11, 1, 0, True]],
                },
                {
                    "filename": "/home/repo/anchor.cpp",
                    "segments": [[1, 1, 1, True]],
                },
                {
                    "filename": "/usr/include/vector",
                    "segments": [[1, 1, 1, True]],
                },
            ],
            "functions": [
                {
                    "name": "Widget::parse()",
                    "count": 1,
                    "filenames": ["/home/repo/include/widget.hpp"],
                }
            ],
        }]
    })

    parsed = _parse_coverage_export(
        payload,
        "/home/repo",
        {"anchor.cpp"},
    )

    assert parsed["lines"] == {("include/widget.hpp", 10)}
    assert parsed["functions"] == {"Widget::parse()"}


def test_coverage_and_xray_comparisons_are_deterministic():
    coverage = _coverage_difference(
        {
            "lines": {("src/a.cpp", 1), ("src/a.cpp", 2)},
            "functions": {"parse", "accept"},
        },
        {
            "lines": {("src/a.cpp", 1), ("src/b.cpp", 7)},
            "functions": {"parse", "reject"},
        },
    )
    assert coverage["common_locations"] == ["src/a.cpp:1"]
    assert coverage["anchor_only_functions"] == ["accept"]
    assert coverage["target_only_functions"] == ["reject"]

    trace = _trace_difference(
        ["parse", "lookup", "accept"],
        ["parse", "lookup", "reject"],
    )
    assert trace["common_prefix"] == ["parse", "lookup"]
    assert trace["first_divergence"] == {
        "anchor": "accept",
        "target": "reject",
    }


def test_xray_parser_uses_symbolized_function_entries_only():
    sequence = _parse_xray_trace(json.dumps({
        "traceEvents": [
            {"ts": "3.0", "ph": "B", "name": "lookup"},
            {"ts": "1.0", "ph": "B", "name": "parse"},
            {"ts": "2.0", "ph": "E", "name": "parse"},
            {"ts": "4.0", "ph": "B", "name": "lookup"},
        ]
    }))

    assert sequence == ["parse", "lookup"]


def test_coverage_action_records_bound_dynamic_evidence(monkeypatch):
    session = recorded_pair()

    def fake_collect(docker, current_session, anchor, target, timeout):
        assert current_session is session
        assert [anchor.run_id, target.run_id] == ["RUN-1", "RUN-2"]
        assert timeout == 300
        return {
            "kind": "llvm_coverage_contrast",
            "source_run_ids": ["RUN-1", "RUN-2"],
            "source_fingerprint": "abc123",
            "common_locations": ["src/a.cpp:1-3"],
            "anchor_only_locations": [],
            "target_only_locations": ["src/b.cpp:7"],
        }

    monkeypatch.setattr(
        dynamic_evidence,
        "_collect_coverage_contrast",
        fake_collect,
    )
    observation = CollectCoverageContrast(
        docker_env=FakeDocker(),
        session=session,
    ).execute(
        CollectCoverageContrastArgs(
            thoughts="",
            anchor_run_id="RUN-1",
            target_run_id="RUN-2",
        )
    )

    assert observation.properties["evidence_id"] == "COVERAGE-1"
    assert session.actions[-1]["source_run_ids"] == ["RUN-1", "RUN-2"]
    assert session.actions[-1]["source_fingerprint"] == "abc123"


def test_xray_unavailability_does_not_create_evidence(monkeypatch):
    session = recorded_pair()

    def unavailable(*args, **kwargs):
        raise dynamic_evidence.DynamicEvidenceUnavailable("llvm-xray missing")

    monkeypatch.setattr(
        dynamic_evidence,
        "_collect_xray_contrast",
        unavailable,
    )
    before = len(session.evidence)
    observation = CollectXRayTraceContrast(
        docker_env=FakeDocker(),
        session=session,
    ).execute(
        CollectXRayTraceContrastArgs(
            thoughts="",
            anchor_run_id="RUN-1",
            target_run_id="RUN-2",
        )
    )

    assert observation.properties == {"available": False}
    assert len(session.evidence) == before
