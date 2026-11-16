#!/usr/bin/env python3
"""Run the indexed 128-case Repair-only graph ablation and official evaluation."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/repair_only_pair128_unified200_isolated_low_20260929"
DATASET = ROOT / "INTENT2FIX/benchmark/multi_swebench_cpp_change.jsonl"
INDEX = ROOT / "results/intent_canonical128_20260921/index_store"
ORIGINS = Path("/data/dqa/cache/repos")
PYTHON = ROOT / ".venv/bin/python"
EXCLUDED = "yhirose__cpp-httplib-1765"
PUBLIC_KEYS = ("repo", "instance_id", "base_commit", "problem_statement", "created_at")
SOURCES = (
    "scripts/run_repair_only_pair128.py",
    "scripts/run_repair_only_ablation.py",
    "scripts/run_unified_four_repair.py",
    "scripts/run_unified_four_repair_no_graph.py",
    "scripts/run_no_memory_cotest.py",
    "scripts/evaluate_fix_patches.py",
    "INTENT2FIX/actions/unified_repair.py",
    "INTENT2FIX/agent/code_agent.py",
    "INTENT2FIX/prompt_guidance.py",
    "INTENT2FIX/prompts/repair_evidence.md",
)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare() -> None:
    if RUN.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {RUN}")
    records = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line]
    ids = [row["instance_id"] for row in records]
    assert len(ids) == len(set(ids)) == 129
    assert ids.count(EXCLUDED) == 1
    public = [{key: row[key] for key in PUBLIC_KEYS if key in row}
              for row in records if row["instance_id"] != EXCLUDED]
    selected = [row["instance_id"] for row in public]
    assert len(selected) == 128 and len(set(selected)) == 128
    index_manifest = read_json(INDEX / "runtime_store_manifest.json")
    assert index_manifest["strict_runtime_preflight"] is True
    assert set(index_manifest["instances"]) == set(selected)
    RUN.mkdir(parents=True)
    public_path = RUN / "instances_public128.jsonl"
    public_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in public),
        encoding="utf-8",
    )
    repo_names = sorted({row["repo"] for row in public})
    for arm in ("graph_on", "graph_off"):
        cache_root = RUN / "repo_cache" / arm
        for repo_name in repo_names:
            origin = ORIGINS / repo_name
            if not (origin / ".git").is_dir():
                raise FileNotFoundError(f"local repository origin missing: {origin}")
            link = cache_root / repo_name
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(origin, target_is_directory=True)
    write_json(RUN / "manifest.json", {
        "status": "PREPARED", "created_at": now(), "cases": selected,
        "excluded_case": EXCLUDED, "reason": "user requested skipping the only unindexed case",
        "arms": ["graph_on", "graph_off"], "stages": ["Repair"],
        "model": "glm-5.3-flash", "reasoning_effort": "low",
        "repair_steps": 200, "workers_per_arm": 5,
        "tool_profile": "unified_four", "selector": False,
        "public_inputs": str(public_path), "public_inputs_sha256": sha256(public_path),
        "benchmark_sha256": sha256(DATASET), "index_root": str(INDEX),
        "index_manifest_sha256": sha256(INDEX / "runtime_store_manifest.json"),
        "repo_cache_roots": {arm: str(RUN / "repo_cache" / arm)
                             for arm in ("graph_on", "graph_off")},
        "local_origin_repos": repo_names,
        "source_sha256": {name: sha256(ROOT / name) for name in SOURCES},
        "official_evaluation": "after both Repair arms finish; single submitted patch per case",
    })
    print(f"PREPARED {RUN}: 128 paired public cases, 200 Repair steps", flush=True)


def launch_child(command: list[str], log_path: Path, env: dict[str, str]) -> subprocess.Popen:
    with log_path.open("x", encoding="utf-8") as log:
        child = subprocess.Popen(
            command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    print(f"START pid={child.pid} log={log_path}", flush=True)
    return child


def run() -> int:
    manifest_path = RUN / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest["status"] != "PREPARED":
        raise RuntimeError(f"run is not PREPARED: {manifest['status']}")
    if sha256(Path(manifest["public_inputs"])) != manifest["public_inputs_sha256"]:
        raise RuntimeError("public inputs changed after preparation")
    if sha256(INDEX / "runtime_store_manifest.json") != manifest["index_manifest_sha256"]:
        raise RuntimeError("canonical graph store changed after preparation")
    for name, expected in manifest["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f"source changed after preparation: {name}")

    base_env = os.environ.copy()
    base_env.update({
        "INDEX_STORE_DIR": str(INDEX),
        "CODE_INTENT_INDEX_STRICT_PREFLIGHT": "1",
        "COTEST_REPAIR_TOOL_PROFILE": "unified_four",
        "COTEST_CODING_TOOL_PROFILE": "unified_four",
        "COTEST_GLM_REASONING_EFFORT": "low",
    })
    manifest.update(status="RUNNING", started_at=now(), controller_pid=os.getpid())
    write_json(manifest_path, manifest)
    repairs = {}
    for arm, entrypoint in (
        ("graph_on", "scripts/run_unified_four_repair.py"),
        ("graph_off", "scripts/run_unified_four_repair_no_graph.py"),
    ):
        env = {**base_env,
               "REPO_DIR": manifest["repo_cache_roots"][arm],
               "COTEST_REPAIR_GRAPH_SEARCH": "1" if arm == "graph_on" else "0"}
        repairs[arm] = launch_child([
            str(PYTHON), "-u", str(ROOT / entrypoint),
            "--jsonl", manifest["public_inputs"],
            "--output-dir", str(RUN / arm), "--workers", "5",
            "--model", "glm-5.3-flash", "--reasoning-effort", "low",
            "--repair-steps", "200",
        ], RUN / f"{arm}_launcher.log", env)
    manifest["repair_pids"] = {arm: process.pid for arm, process in repairs.items()}
    write_json(manifest_path, manifest)

    repair_exits = {arm: process.wait() for arm, process in repairs.items()}
    manifest = read_json(manifest_path)
    manifest.update(status="EVALUATING", repair_exit_codes=repair_exits,
                    repair_finished_at=now())
    write_json(manifest_path, manifest)
    evaluations = {}
    for arm in repairs:
        submission = RUN / arm / "submission.json"
        if not submission.is_file():
            continue
        evaluations[arm] = launch_child([
            str(PYTHON), "-u", str(ROOT / "scripts/evaluate_fix_patches.py"),
            str(submission), "--output-dir", str(RUN / f"official_eval_{arm}"),
            "--workers", "5", "--timeout", "3600",
        ], RUN / f"official_eval_{arm}_launcher.log", base_env)
    evaluation_exits = {arm: process.wait() for arm, process in evaluations.items()}
    summaries = {}
    for arm in repairs:
        path = RUN / f"official_eval_{arm}/summary.json"
        if path.is_file():
            summary = read_json(path)
            summaries[arm] = {"evaluated_patches": summary["total"],
                              "resolved_cases": summary["resolved"]}
    manifest = read_json(manifest_path)
    manifest.update(repair_exit_codes=repair_exits,
                    evaluation_exit_codes=evaluation_exits,
                    official_summary=summaries, finished_at=now(),
                    status="COMPLETED" if all(code == 0 for code in repair_exits.values())
                    and len(summaries) == 2 else "PARTIAL")
    write_json(manifest_path, manifest)
    return 0 if manifest["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"prepare", "run"}:
        raise SystemExit("usage: run_repair_only_pair128.py {prepare|run}")
    raise SystemExit(prepare() if sys.argv[1] == "prepare" else run())
