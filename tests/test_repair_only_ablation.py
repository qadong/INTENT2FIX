import json
from pathlib import Path

import pytest

from scripts import run_ablation_d_repair_no_graph as D_ENTRYPOINT
from scripts import run_unified_four_repair_no_graph as UNIFIED_D_ENTRYPOINT
from scripts import run_repair_only_ablation as RUNNER


def _config(tmp_path):
    return {
        "jsonl": str(tmp_path / "benchmark.jsonl"),
        "model": "glm-5.3-flash",
        "reasoning_effort": "low",
        "repair_steps": 200,
        "docker_registry": "registry.example",
        "workers": 1,
    }


def test_repair_only_memory_requests_reproduction_without_upstream_evidence():
    memory = RUNNER.build_repair_memory(
        "org__repo-1", {"problem_statement": "reported behavior"}
    )

    assert memory.problem_statement == "reported behavior"
    assert memory.reproduction_summary == ""
    assert memory.reproduction_result == ""
    assert memory.test_files == []
    assert "Investigate the reported behavior" in memory.repair_summary
    assert "If exact reproduction is too" in memory.repair_summary


@pytest.mark.parametrize(
    (
        "arm",
        "graph_environment",
        "graph_marker",
        "indexed_tools",
        "expected_prompt",
    ),
    (
        (
            RUNNER.ARM_C_REPAIR_GRAPH,
            "0",
            "enabled",
            True,
            RUNNER.STANDALONE_REPAIR_SYSTEM_PROMPT,
        ),
        (
            RUNNER.ARM_D_REPAIR_NO_GRAPH,
            "1",
            "disabled",
            False,
            RUNNER.STANDALONE_REPAIR_NO_GRAPH_SYSTEM_PROMPT,
        ),
    ),
)
def test_repair_only_case_calls_direct_repair_with_selected_graph_arm(
    monkeypatch,
    tmp_path,
    arm,
    graph_environment,
    graph_marker,
    indexed_tools,
    expected_prompt,
):
    config = RUNNER.apply_ablation_arm(_config(tmp_path), arm)
    patch = (
        "diff --git a/src/value.cc b/src/value.cc\n"
        "--- a/src/value.cc\n"
        "+++ b/src/value.cc\n"
        "@@ -1 +1 @@\n"
        "-return 1;\n"
        "+return 2;\n"
    )
    observed = {}

    monkeypatch.setattr(RUNNER.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(
        RUNNER,
        "configure_case_logging",
        lambda case_dir, _instance_id: case_dir / "repair-only.log",
    )
    monkeypatch.setattr(
        RUNNER.cotest,
        "get_benchmark_instance",
        lambda _instance_id: {
            "problem_statement": "reported behavior",
            "base_commit": "abc123",
        },
    )
    monkeypatch.setattr(
        RUNNER.cotest,
        "parse_instance_string",
        lambda _instance_id: {"org": "org", "repo": "repo", "number": "1"},
    )

    class FakeDockerEnv:
        def __init__(self, **kwargs):
            observed["docker"] = kwargs

    monkeypatch.setattr(RUNNER, "DockerEnv", FakeDockerEnv)

    def fake_repair_step(**kwargs):
        observed["repair"] = kwargs
        Path(kwargs["output_dir"], "solution.patch").write_text(patch, encoding="utf-8")
        return patch

    monkeypatch.setattr(RUNNER.cotest, "run_repair_step", fake_repair_step)

    outcome = RUNNER.run_case("org__repo-1", str(tmp_path / "results"), "run-1", config)

    assert outcome["status"] == "completed"
    assert outcome["patch_sha256"]
    assert observed["repair"]["model_name"] == "glm-5.3-flash"
    assert observed["repair"]["memory"].reproduction_result == ""
    assert (
        "Investigate the reported behavior"
        in observed["repair"]["memory"].repair_summary
    )
    assert observed["repair"]["agent_system_prompt"] == expected_prompt + "\n" + RUNNER.load_stage_guidance("repair")
    assert observed["repair"]["include_repository_outline"] is False
    assert RUNNER.os.environ["COTEST_ABLATE_DUAL_LAYER_GRAPH"] == graph_environment
    assert RUNNER.os.environ["COTEST_ENABLE_INTENT_PORTFOLIO"] == "0"
    assert RUNNER.os.environ["COTEST_ENABLE_LEGACY_REPRODUCER"] == "0"
    assert RUNNER.os.environ["COTEST_TEST_ENHANCEMENT_ROUNDS"] == "0"
    assert RUNNER.os.environ["COTEST_REPAIR_STEPS"] == "200"

    case_dir = tmp_path / "results" / "org__repo-1"
    marker = json.loads(
        (case_dir / "repair_only_ablation.json").read_text(encoding="utf-8")
    )
    assert marker["enabled_stages"] == ["repair"]
    assert marker["ablation_arm"] == arm
    assert marker["dual_layer_graph"] == graph_marker
    assert marker["repair_indexed_tools"] is indexed_tools
    assert marker["graph_index_loading"] is indexed_tools
    assert marker["repair_step_budget"] == 200
    assert marker["repair_system_prompt"] == "standalone_concise_v2"
    assert marker["repair_system_prompt_variant"] == (
        "graph_enabled" if indexed_tools else "source_only"
    )
    assert marker["repository_outline"] == "omitted"
    assert "scout" in marker["disabled_stages"]
    assert "hypothesis" in marker["disabled_stages"]
    assert "selector" in marker["disabled_stages"]
    assert not (case_dir / "intent_portfolio.json").exists()
    assert not (case_dir / "portfolio_candidate_pool.json").exists()


def test_no_graph_prompt_removes_only_graph_navigation_guidance(tmp_path):
    graph_config = RUNNER.apply_ablation_arm(
        _config(tmp_path), RUNNER.ARM_C_REPAIR_GRAPH
    )
    no_graph_config = RUNNER.apply_ablation_arm(
        _config(tmp_path), RUNNER.ARM_D_REPAIR_NO_GRAPH
    )

    assert "Indexed graph tools are optional" in RUNNER.repair_system_prompt(
        graph_config
    )
    no_graph_prompt = RUNNER.repair_system_prompt(no_graph_config)
    assert "Indexed graph tools" not in no_graph_prompt
    assert "through the checked-out source" in no_graph_prompt
    assert "Reproduce the reported behavior when practical" in no_graph_prompt
    assert no_graph_config["model"] == graph_config["model"]
    assert no_graph_config["reasoning_effort"] == graph_config["reasoning_effort"]
    assert no_graph_config["repair_steps"] == graph_config["repair_steps"]


@pytest.mark.parametrize("has_graph", (True, False))
def test_unified_prompt_loads_graph_guidance_only_when_available(
    tmp_path, has_graph
):
    arm = (
        RUNNER.ARM_C_REPAIR_GRAPH
        if has_graph
        else RUNNER.ARM_D_REPAIR_NO_GRAPH
    )
    config = RUNNER.apply_ablation_arm(
        {
            **_config(tmp_path),
            "repair_tool_profile": RUNNER.UNIFIED_FOUR_TOOL_PROFILE,
        },
        arm,
    )

    prompt = RUNNER.repair_system_prompt(config)
    assert ("Bash, Editor, CodeGraph, and Finish" in prompt) is has_graph
    assert ("Bash, Editor, and Finish" in prompt) is not has_graph
    assert ("issue-relevant" in prompt) is has_graph
    assert ('operation="search"' in prompt and 'operation="trace"' in prompt) is has_graph
    assert ("CodeGraph" in prompt) is has_graph
    assert ("graph" in prompt.lower()) is has_graph
    assert ("you must call CodeGraph before the" in prompt) is has_graph
    assert config["repair_system_prompt"] == (
        "unified_four_v5_required_scope_trace"
    )


def test_d_entrypoint_selects_repair_no_graph_arm(monkeypatch):
    observed = {}

    def fake_main(arm, *, entrypoint):
        observed["arm"] = arm
        observed["entrypoint"] = entrypoint
        return 17

    monkeypatch.setattr(D_ENTRYPOINT.runner, "main", fake_main)

    assert D_ENTRYPOINT.main() == 17
    assert observed["arm"] == RUNNER.ARM_D_REPAIR_NO_GRAPH
    assert observed["entrypoint"].name == "run_ablation_d_repair_no_graph.py"


def test_unified_d_entrypoint_selects_no_graph_and_four_tools(monkeypatch):
    observed = {}

    def fake_main(arm, *, entrypoint, repair_tool_profile):
        observed.update(arm=arm, entrypoint=entrypoint,
                        repair_tool_profile=repair_tool_profile)
        return 17

    monkeypatch.setattr(UNIFIED_D_ENTRYPOINT.runner, "main", fake_main)

    assert UNIFIED_D_ENTRYPOINT.main() == 17
    assert observed["arm"] == RUNNER.ARM_D_REPAIR_NO_GRAPH
    assert observed["entrypoint"].name == "run_unified_four_repair_no_graph.py"
    assert observed["repair_tool_profile"] == RUNNER.UNIFIED_FOUR_TOOL_PROFILE


def test_d_run_config_freezes_entrypoint_and_graph_off_profile(tmp_path):
    config = RUNNER.apply_ablation_arm(
        {
            **_config(tmp_path),
            "jsonl_sha256": "benchmark-sha",
        },
        RUNNER.ARM_D_REPAIR_NO_GRAPH,
    )
    entrypoint = Path(D_ENTRYPOINT.__file__).resolve()

    RUNNER.freeze_run_config(
        tmp_path,
        config,
        ["org__repo-1"],
        entrypoint,
    )

    frozen = json.loads((tmp_path / "run_config.json").read_text(encoding="utf-8"))
    assert frozen["runner"] == str(entrypoint)
    assert frozen["runner_sha256"] == RUNNER.sha256_file(entrypoint)
    assert frozen["ablation_arm"] == RUNNER.ARM_D_REPAIR_NO_GRAPH
    assert frozen["dual_layer_graph"] == "disabled"
    assert frozen["repair_indexed_tools"] is False
    assert frozen["repair_system_prompt"] == "standalone_concise_v2"
    assert frozen["repair_system_prompt_variant"] == "source_only"
    assert frozen["shared_repair_only_runner"] == str(Path(RUNNER.__file__).resolve())


def test_repair_guidance_cannot_change_mid_run(tmp_path, monkeypatch):
    config = RUNNER.apply_ablation_arm({**_config(tmp_path), "jsonl_sha256": "benchmark-sha"},
                                     RUNNER.ARM_D_REPAIR_NO_GRAPH)
    monkeypatch.setenv("COTEST_REPAIR_EVIDENCE_GUIDANCE", "1")
    entrypoint = Path(D_ENTRYPOINT.__file__).resolve()
    RUNNER.freeze_run_config(tmp_path, config, ["org__repo-1"], entrypoint)
    monkeypatch.setenv("COTEST_REPAIR_EVIDENCE_GUIDANCE", "0")
    with pytest.raises(RuntimeError, match="repair_evidence_guidance_enabled"):
        RUNNER.freeze_run_config(tmp_path, config, ["org__repo-1"], entrypoint)
