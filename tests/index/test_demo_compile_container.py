from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_demo(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "code_intent"))
    spec = importlib.util.spec_from_file_location(
        "code_intent_demo_container_test", ROOT / "code_intent/demo.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_batch_index():
    spec = importlib.util.spec_from_file_location(
        "code_intent_batch_index_test", ROOT / "code_intent/batch_index.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compile_only_container_is_offline_read_only_and_has_no_api_key(
    monkeypatch, tmp_path
):
    demo = _load_demo(monkeypatch)
    conda = tmp_path / "conda"
    (conda / "bin").mkdir(parents=True)
    (conda / "bin/python3").touch()
    output = tmp_path / "output"
    build = tmp_path / "build"
    captured = []

    monkeypatch.setenv("CODE_INTENT_DOCKER_NETWORK", "none")
    monkeypatch.setenv("CODE_INTENT_DOCKER_CONDA_PREFIX", str(conda))
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-enter-compile-container")
    monkeypatch.setenv("HTTPS_PROXY", "http://must-not-enter:1234")
    monkeypatch.delenv("CODE_INTENT_CCACHE_DIR", raising=False)
    monkeypatch.delenv("CODE_INTENT_HOST_REPO_BIND", raising=False)
    monkeypatch.setattr(demo.subprocess, "check_call", lambda argv: captured.append(argv))

    args = argparse.Namespace(
        instance_id="swe-bench_simdjson__simdjson-1615",
        file_path="/home/simdjson",
        output_path=str(output),
        build_dir=str(build),
        cmake_arg=["-DSIMDJSON_DEVELOPER_MODE=ON"],
        build_target=["simdjson"],
        build_backend="auto",
        make_build_shell="",
        compile_db_only=True,
        docker_shell=False,
        host_repo_bind="",
    )

    demo.launch_docker_from_host(args)

    assert len(captured) == 1
    command = captured[0]
    rendered = " ".join(command)
    assert "--network=none" in command
    assert "OPENAI_API_KEY" not in rendered
    assert "must-not-enter" not in rendered
    assert "EDGE_DEBUG=1" not in command
    assert f"{ROOT}:/workspace:ro" in command
    assert f"{conda.resolve()}:/opt/code_intent_conda:ro" in command
    assert "--build-target simdjson" in rendered
    assert "PYTHONDONTWRITEBYTECODE=1" in command


def test_edge_trace_is_opt_in(monkeypatch, tmp_path):
    demo = _load_demo(monkeypatch)
    monkeypatch.delenv("EDGE_DEBUG", raising=False)
    monkeypatch.delenv("EDGE_LOG_PATH", raising=False)

    assert demo._configure_edge_debug_log(tmp_path) is None
    assert "EDGE_LOG_PATH" not in demo.os.environ

    stale = tmp_path / "edge_debug.log"
    stale.write_text("stale\n")
    monkeypatch.setenv("EDGE_DEBUG", "1")
    assert demo._configure_edge_debug_log(tmp_path) == stale
    assert not stale.exists()
    assert demo.os.environ["EDGE_LOG_PATH"] == str(stale.resolve())


def test_simdjson_structural_build_avoids_developer_dependencies():
    batch_index = _load_batch_index()
    config = batch_index.get_repo_config(
        "swe-bench_simdjson__simdjson-1615"
    )

    assert config["cmake_args"] == [
        "-DSIMDJSON_DEVELOPER_MODE=OFF",
        "-DSIMDJSON_JUST_LIBRARY=ON",
        "-DSIMDJSON_GOOGLE_BENCHMARKS=OFF",
        "-DSIMDJSON_COMPETITION=OFF",
    ]
    assert config["build_targets"] == ["simdjson"]


def test_container_build_tool_check_uses_executables_not_meta_packages(monkeypatch):
    demo = _load_demo(monkeypatch)
    available = {"find", "c++", "make", "cmake"}
    monkeypatch.setattr(
        demo.shutil,
        "which",
        lambda name: f"/usr/local/bin/{name}" if name in available else None,
    )

    assert demo._missing_container_build_packages(backend="cmake") == []
    assert demo._missing_container_build_packages(backend="make-bear") == ["bear"]


def test_container_build_tool_setup_does_not_require_pip(monkeypatch):
    demo = _load_demo(monkeypatch)
    monkeypatch.setattr(demo, "_maybe_configure_apt_cn_mirror", lambda: None)
    monkeypatch.setattr(
        demo, "_missing_container_build_packages", lambda **kwargs: []
    )
    monkeypatch.setattr(
        demo.subprocess,
        "check_call",
        lambda *args, **kwargs: pytest.fail(
            f"native build tools were present but subprocess ran: {args}"
        ),
    )

    demo._ensure_container_build_tools(backend="cmake")


def test_legacy_simdjson_uses_production_only_compile_database(
    monkeypatch, tmp_path
):
    batch_index = _load_batch_index()
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "include").mkdir()
    (repo / "src/simdjson.cpp").write_text("int simdjson_anchor;\n")
    (repo / "simdjson-flags.cmake").write_text("# no library-only option\n")
    build_root = tmp_path / "builds"
    output_root = tmp_path / "output"
    calls = []
    monkeypatch.setenv("BATCH_INDEX_COMPILE_DB_ONLY", "1")
    monkeypatch.setattr(
        batch_index,
        "_run_batch_host_index_phase",
        lambda *args: calls.append(args),
    )

    instance_id = "swe-bench_simdjson__simdjson-949"
    batch_index.run_indexing(
        instance_id, str(repo), str(output_root), str(build_root)
    )

    compile_db = build_root / instance_id / "compile_commands.json"
    rows = __import__("json").loads(compile_db.read_text())
    assert rows[0]["file"] == str(repo / "src/simdjson.cpp")
    assert "-DSIMDJSON_THREADS_ENABLED=1" in rows[0]["arguments"]
    assert len(calls) == 1
