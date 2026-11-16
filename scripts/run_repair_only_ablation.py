#!/usr/bin/env python3
"""Shared Repair-only runner for the graph-on and graph-off ablation arms."""

from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import signal
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_no_memory_cotest as cotest  # noqa: E402
from INTENT2FIX.prompt_guidance import load_stage_guidance  # noqa: E402


# Reuse the initialized symbols from the production runner. Importing DockerEnv
# before that module establishes its action imports triggers a circular import.
Memory = cotest.Memory
DockerEnv = cotest.DockerEnv
production_patch_sha256 = cotest.production_patch_sha256


ARM_C_REPAIR_GRAPH = "C_repair_only_graph"
ARM_D_REPAIR_NO_GRAPH = "D_repair_only_no_graph"
_VALID_ARMS = {ARM_C_REPAIR_GRAPH, ARM_D_REPAIR_NO_GRAPH}
STANDARD_TOOL_PROFILE = "standard"
UNIFIED_FOUR_TOOL_PROFILE = "unified_four"


_GRAPH_GUIDANCE = """Inspect the relevant implementation and existing tests before editing. Use focused grep/find
through RunCommand and read exact source with ViewCode. Indexed graph tools are optional: use
them when ownership, callers, templates, macros, or the affected scope is unclear, then verify
their leads in source. Do not call tools merely because they are available."""


_SOURCE_ONLY_GUIDANCE = """Inspect the relevant implementation and existing tests before editing. Use focused grep/find
through RunCommand and read exact source with ViewCode. Trace ownership, callers, templates,
macros, and affected scope through the checked-out source when those relationships are unclear.
Verify every inferred relationship against source before relying on it."""


STANDALONE_REPAIR_SYSTEM_PROMPT = """You are an autonomous software engineer repairing a C++ repository.
Use the original issue and the current checkout as the source of truth. Take exactly one tool
action per response and wait for its result.

Inspect the relevant implementation and existing tests before editing. Use focused grep/find
through RunCommand and read exact source with ViewCode. Indexed graph tools are optional: use
them when ownership, callers, templates, macros, or the affected scope is unclear, then verify
their leads in source. Do not call tools merely because they are available.

Reproduce the reported behavior when practical, but do not spend the run repeatedly rebuilding
the same unchanged state or chasing an unavailable exact reproduction. Implement the smallest
complete production fix that covers all behavior stated by the issue and preserves relevant
existing callers. Do not modify tests, fixtures, generated outputs, or build configuration to
hide a defect. Do not use network sources or repository history to find the answer.

After the final edit, run the most focused useful build or test and a broader existing check when
practical. Treat failures and unavailable checks honestly; diagnose actionable failures, but do
not loop on equivalent commands. Review the resulting production patch and call VerifiedFinish
once the best supported fix is stable, even if some non-actionable verification is unavailable.
"""


STANDALONE_REPAIR_NO_GRAPH_SYSTEM_PROMPT = STANDALONE_REPAIR_SYSTEM_PROMPT.replace(
    _GRAPH_GUIDANCE,
    _SOURCE_ONLY_GUIDANCE,
)


_UNIFIED_GRAPH_GUIDANCE = """CodeGraph is available only while the dual-layer index is enabled. Bash is best for exact
names, literals, and reading or testing source. Use CodeGraph when an unknown implementation
anchor or a relation between known definitions could change the cause, patch owner, or complete
scope. It sees the original Issue automatically. Its search/trace queries select issue-relevant
real graph paths internally and return source-checked leads; they do not prove that a neighbor
shares the bug or must be edited.

After inspecting an Issue-relevant production definition, you must call CodeGraph before the
first production edit to check whether a connected caller, callee, owner, sibling public entry,
or packaged representation changes the assigned Hypothesis's implementation or repair scope.
Trace the inspected source location when you have one; search by behavior only when the anchor
is still unknown. Reusing Scout's graph findings does not replace this Repair-stage check.
Inspect issue-relevant returned definitions in source before deciding the patch scope. If the
graph fails or has no useful relation, record that limitation and continue with Bash. This is
a minimum scope check, not a quota: make further graph calls only for unresolved relationships
that could change the fix.

Use `CodeGraph(operation="search", behavior="...", purpose="...")` when issue language does not
identify the implementation. Once Bash or Editor has shown an exact function/class location, use
`CodeGraph(operation="trace", source_path="...", source_line=N, question="...")` to ask which
caller, callee, owner/member, or type dependency changes the repair scope. A returned GREF can
anchor another trace via `entity_ref`; leave `edge_types` empty unless the precise indexed edge
kind is known. Open the returned definitions with Bash/Editor and verify the actual call and
build-selected source before editing; after an edit, re-read the patched source because graph
locations describe the indexed checkout. A zero-edge result is not proof of local scope; re-anchor
at a relevant producer or consumer if the relationship still matters.

Example: a failing public wrapper calls an internal formatter, but it is unclear whether another
entry point consumes the same representation. After reading the wrapper, trace its inspected
source line with the question "Which callers or sibling entry points use this representation?".
The graph may return a shared implementation and a sibling path. Read both bodies, then decide
whether one fix covers both; do not edit the sibling merely because an edge exists.

"""


_UNIFIED_REPAIR_SYSTEM_PROMPT_TEMPLATE = """You are an autonomous software engineer repairing a C++ repository.
Use the original issue and current checkout as the source of truth. Available tools are:
{available_tools}.

Bash is the continuous investigation and verification interface. Use it for grep/find, repository
inspection, disposable reproduction files under /tmp, compilation, focused tests, broader tests,
and final diff inspection. Pipelines use pipefail, but still use && for prerequisites and do not
let a trailing echo or a later semicolon command turn a failed verification into success.

Editor is the production text interface. View an existing file before replacing or inserting
text; use create only for an intended production deliverable.

{graph_guidance}Investigate the reported behavior before editing and reproduce it when practical. Implement the
smallest complete production fix, covering every behavior and type/operand branch stated by the
issue. Existing tests and their expectations are constraints: never edit, weaken, or delete them
to make an implementation pass. Use /tmp for disposable reproduction code. Validate with the
repository's declared/default build route and minimum language standard; a manual command using a
newer standard is not sufficient evidence. Before Finish, re-read the issue and inspect the final
diff. Check whether the edited logic has other public, generated, amalgamated, mirrored, or
build-selected representations, and verify the representation actually consumed by the relevant
build/test command. Run focused and broader checks when practical. If an exact check is
unavailable, report that honestly and Finish with the best supported patch instead of looping. Do
not use network sources, reference patches, or Git history to find the answer.
"""


def unified_repair_system_prompt(has_graph: bool) -> str:
    return _UNIFIED_REPAIR_SYSTEM_PROMPT_TEMPLATE.format(
        available_tools=(
            "Bash, Editor, CodeGraph, and Finish"
            if has_graph
            else "Bash, Editor, and Finish"
        ),
        graph_guidance=_UNIFIED_GRAPH_GUIDANCE if has_graph else "",
    ) + "\n" + load_stage_guidance("repair")


REPRODUCE_THEN_FIX_GUIDANCE = """<standalone_repair>
Investigate the reported behavior and its production path before changing code. Reproduce it
with an existing test or a focused runtime check when practical. If exact reproduction is too
expensive or unavailable, proceed from direct source evidence and state that limitation.
Implement the complete issue-described behavior, then run the most relevant available checks.
</standalone_repair>"""


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def split_ids(raw: str | None) -> set[str]:
    return {item.strip() for item in (raw or "").split(",") if item.strip()}


def apply_ablation_arm(config: dict[str, Any], arm: str) -> dict[str, Any]:
    if arm not in _VALID_ARMS:
        raise ValueError(f"Unknown Repair-only ablation arm: {arm}")
    graph_enabled = arm == ARM_C_REPAIR_GRAPH
    tool_profile = config.get("repair_tool_profile", STANDARD_TOOL_PROFILE)
    unified = tool_profile == UNIFIED_FOUR_TOOL_PROFILE
    return {
        **config,
        "ablation_arm": arm,
        "dual_layer_graph": "enabled" if graph_enabled else "disabled",
        "repair_indexed_tools": graph_enabled,
        "repair_system_prompt": (
            "unified_four_v5_required_scope_trace"
            if unified
            else "standalone_concise_v2"
        ),
        "repair_system_prompt_variant": (
            "unified_graph_enabled"
            if unified and graph_enabled
            else "unified_source_only"
            if unified
            else "graph_enabled"
            if graph_enabled
            else "source_only"
        ),
    }


def graph_enabled(config: dict[str, Any]) -> bool:
    return config.get("dual_layer_graph", "enabled") == "enabled"


def repair_system_prompt(config: dict[str, Any]) -> str:
    if config.get("repair_tool_profile") == UNIFIED_FOUR_TOOL_PROFILE:
        return unified_repair_system_prompt(graph_enabled(config))
    if graph_enabled(config):
        return STANDALONE_REPAIR_SYSTEM_PROMPT + "\n" + load_stage_guidance("repair")
    return STANDALONE_REPAIR_NO_GRAPH_SYSTEM_PROMPT + "\n" + load_stage_guidance("repair")


def configure_repair_only_environment(config: dict[str, Any]) -> None:
    """Pin every stage switch so inherited experiment flags cannot leak in."""
    os.environ.update(
        {
            "COTEST_JSONL": config["jsonl"],
            "COTEST_REPAIR_MODEL": config["model"],
            "COTEST_REPAIR_STEPS": str(config["repair_steps"]),
            "COTEST_GLM_REASONING_EFFORT": config["reasoning_effort"],
            "COTEST_DOCKER_REGISTRY": config["docker_registry"],
            "COTEST_ABLATE_DUAL_LAYER_GRAPH": ("0" if graph_enabled(config) else "1"),
            "COTEST_ENABLE_INTENT_GROUNDING": "0",
            "COTEST_ENABLE_CAUSAL_INVESTIGATION": "0",
            "COTEST_ENABLE_INTENT_DIALOGUE": "0",
            "COTEST_ENABLE_LEGACY_REPRODUCER": "0",
            "COTEST_ENABLE_REPRODUCER_FALLBACK": "0",
            "COTEST_ENABLE_INTENT_PORTFOLIO": "0",
            "COTEST_PORTFOLIO_REPAIR_ONLY": "0",
            "COTEST_PORTFOLIO_SCOUT_ONLY": "0",
            "COTEST_PORTFOLIO_STOP_AFTER_REPAIR": "0",
            "COTEST_ENABLE_PATCH_REVIEWER": "0",
            "COTEST_ENABLE_PATCH_INTENT_AUDIT": "0",
            "COTEST_ENABLE_TRAJECTORY_AUDIT_LEADS": "0",
            "COTEST_TEST_ENHANCEMENT_ROUNDS": "0",
            "COTEST_BRANCH_CONTEXT": "off",
            "COTEST_CODING_TOOL_PROFILE": config.get(
                "repair_tool_profile",
                STANDARD_TOOL_PROFILE,
            ),
        }
    )


def build_repair_memory(instance_id: str, benchmark_instance: dict[str, Any]) -> Memory:
    return Memory(
        instance_id=instance_id,
        problem_statement=benchmark_instance["problem_statement"],
        repair_summary=REPRODUCE_THEN_FIX_GUIDANCE,
    )


def marker_payload(config: dict[str, Any]) -> dict[str, Any]:
    has_graph = graph_enabled(config)
    tool_profile = config.get("repair_tool_profile", STANDARD_TOOL_PROFILE)
    return {
        "schema_version": 1,
        "repair_evidence_guidance_enabled": bool(load_stage_guidance("repair")),
        "repair_prompt_sha256": hashlib.sha256(repair_system_prompt(config).encode()).hexdigest(),
        "mode": (
            "repair_only_ablation" if has_graph else "repair_only_no_graph_ablation"
        ),
        "ablation_arm": config.get(
            "ablation_arm",
            ARM_C_REPAIR_GRAPH,
        ),
        "enabled_stages": ["repair"],
        "disabled_stages": [
            "scout",
            "hypothesis",
            "selector",
            "standalone_reproducer",
            "test_agent",
            "patch_reviewer",
            "patch_intent_audit",
        ],
        "dual_layer_graph": "enabled" if has_graph else "disabled",
        "repair_indexed_tools": has_graph,
        "repair_tool_profile": tool_profile,
        "model_visible_tools": (
            ["Bash", "Editor", "CodeGraph", "Finish"]
            if tool_profile == UNIFIED_FOUR_TOOL_PROFILE and has_graph
            else ["Bash", "Editor", "Finish"]
            if tool_profile == UNIFIED_FOUR_TOOL_PROFILE
            else "standard"
        ),
        "graph_index_loading": has_graph,
        "repair_step_budget": config["repair_steps"],
        "repair_model": config["model"],
        "reasoning_effort": config["reasoning_effort"],
        "repair_prompt_protocol": "reproduce_or_establish_evidence_then_fix",
        "repair_system_prompt": config.get(
            "repair_system_prompt",
            "standalone_concise_v2",
        ),
        "repair_system_prompt_variant": config.get(
            "repair_system_prompt_variant",
            "graph_enabled" if has_graph else "source_only",
        ),
        "repository_outline": "omitted",
    }


def configure_case_logging(case_dir: Path, instance_id: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = case_dir / f"repair_only_run_{timestamp}.log"
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter(
        "%(asctime)s - %(process)d - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)
    logging.info("Starting Repair-only ablation for %s", instance_id)
    return log_path


def completed_outcome(case_dir: Path, instance_id: str) -> dict[str, Any] | None:
    status_path = case_dir / "repair_only_status.json"
    if not status_path.is_file():
        return None
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "completed":
        return None
    patch_path = case_dir / "solution.patch"
    expected_hash = status.get("patch_sha256")
    if expected_hash:
        if not patch_path.is_file():
            raise RuntimeError(f"Completed case lost solution.patch: {instance_id}")
        patch = patch_path.read_text(encoding="utf-8")
        if production_patch_sha256(patch) != expected_hash:
            raise RuntimeError(f"Completed patch drift: {instance_id}")
    elif patch_path.is_file() and patch_path.read_text(encoding="utf-8").strip():
        raise RuntimeError(
            f"Completed no-patch status conflicts with file: {instance_id}"
        )
    return {
        "instance_id": instance_id,
        "status": "completed",
        "patch_sha256": expected_hash,
        "resumed": True,
    }


def run_case(
    instance_id: str,
    output_dir: str,
    run_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    signal.signal(signal.SIGTERM, cotest._signal_handler)
    signal.signal(signal.SIGINT, cotest._signal_handler)
    configure_repair_only_environment(config)
    os.environ["SWEBENCH_INSTANCE_NAME"] = instance_id

    case_dir = Path(output_dir) / instance_id
    case_dir.mkdir(parents=True, exist_ok=True)
    prior = completed_outcome(case_dir, instance_id)
    if prior is not None:
        return prior

    log_path = configure_case_logging(case_dir, instance_id)
    marker = marker_payload(config)
    write_json(case_dir / "repair_only_ablation.json", marker)
    write_json(
        case_dir / "repair_only_status.json",
        {
            "status": "running",
            "instance_id": instance_id,
            "started_at": now(),
            "log": str(log_path),
            **marker,
        },
    )

    try:
        benchmark_instance = cotest.get_benchmark_instance(instance_id)
        parsed = cotest.parse_instance_string(instance_id)
        if parsed is None:
            raise ValueError(f"Invalid instance ID: {instance_id}")
        image = (
            f"mswebench/{parsed['org']}_m_{parsed['repo']}:pr-{parsed['number']}"
        ).lower()
        registry = config["docker_registry"].rstrip("/")
        if registry:
            image = f"{registry}/{image}"
        docker_env = DockerEnv(
            image_name=image,
            container_name=f"repair-only-{instance_id}-{run_id}",
            work_dir=f"/home/{parsed['repo']}",
            benchmark_base_commit=benchmark_instance["base_commit"],
        )
        patch = cotest.run_repair_step(
            instance_name=instance_id,
            model_name=config["model"],
            docker_env=docker_env,
            memory=build_repair_memory(instance_id, benchmark_instance),
            output_dir=str(case_dir),
            agent_system_prompt=repair_system_prompt(config),
            include_repository_outline=False,
        )
        patch = patch or ""
        patch_hash = production_patch_sha256(patch) if patch.strip() else None
        status = {
            "status": "completed",
            "instance_id": instance_id,
            "started_at": json.loads(
                (case_dir / "repair_only_status.json").read_text(encoding="utf-8")
            )["started_at"],
            "completed_at": now(),
            "has_patch": bool(patch_hash),
            "patch_sha256": patch_hash,
            "log": str(log_path),
            **marker,
        }
        write_json(case_dir / "repair_only_status.json", status)
        return {
            "instance_id": instance_id,
            "status": "completed",
            "patch_sha256": patch_hash,
            "resumed": False,
        }
    except Exception as error:
        logging.exception("Repair-only case failed: %s", instance_id)
        write_json(
            case_dir / "repair_only_status.json",
            {
                "status": "failed",
                "instance_id": instance_id,
                "failed_at": now(),
                "error": f"{type(error).__name__}: {error}",
                "log": str(log_path),
                **marker,
            },
        )
        return {
            "instance_id": instance_id,
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "resumed": False,
        }


def freeze_run_config(
    output_dir: Path,
    config: dict[str, Any],
    instance_ids: list[str],
    entrypoint: Path,
) -> None:
    path = output_dir / "run_config.json"
    payload = {
        **config,
        "instance_ids": instance_ids,
        "instance_count": len(instance_ids),
        "runner": str(entrypoint),
        "runner_sha256": sha256_file(entrypoint),
        "shared_repair_only_runner": str(Path(__file__).resolve()),
        "shared_repair_only_runner_sha256": sha256_file(Path(__file__).resolve()),
        "repair_implementation": str(Path(cotest.__file__).resolve()),
        "repair_implementation_sha256": sha256_file(Path(cotest.__file__).resolve()),
        "created_at": now(),
        "repair_evidence_guidance_enabled": bool(load_stage_guidance("repair")),
        "repair_prompt_sha256": hashlib.sha256(repair_system_prompt(config).encode()).hexdigest(),
    }
    if config.get("repair_tool_profile") == UNIFIED_FOUR_TOOL_PROFILE:
        unified_actions = ROOT / "INTENT2FIX/actions/unified_repair.py"
        coding_agent = ROOT / "INTENT2FIX/agent/code_agent.py"
        payload.update(
            {
                "unified_tool_implementation": str(unified_actions),
                "unified_tool_implementation_sha256": sha256_file(
                    unified_actions
                ),
                "coding_agent_implementation": str(coding_agent),
                "coding_agent_implementation_sha256": sha256_file(coding_agent),
            }
        )
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        for key in (
            "jsonl",
            "jsonl_sha256",
            "model",
            "reasoning_effort",
            "repair_steps",
            "docker_registry",
            "instance_ids",
            "ablation_arm",
            "dual_layer_graph",
            "repair_indexed_tools",
            "repair_system_prompt",
            "repair_system_prompt_variant",
            "repair_evidence_guidance_enabled",
            "repair_prompt_sha256",
            "repair_tool_profile",
            "unified_tool_implementation_sha256",
            "coding_agent_implementation_sha256",
        ):
            if existing.get(key) != payload.get(key):
                raise RuntimeError(f"Run configuration drift for {key}")
        return
    write_json(path, payload)


def progress_payload(
    instance_ids: list[str], outcomes: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    completed = sorted(
        instance_id
        for instance_id, result in outcomes.items()
        if result.get("status") == "completed"
    )
    failed = sorted(
        instance_id
        for instance_id, result in outcomes.items()
        if result.get("status") == "failed"
    )
    return {
        "updated_at": now(),
        "total": len(instance_ids),
        "completed": len(completed),
        "with_patch": sum(
            bool(result.get("patch_sha256")) for result in outcomes.values()
        ),
        "failed": len(failed),
        "pending": len(instance_ids) - len(outcomes),
        "completed_instance_ids": completed,
        "failed_instance_ids": failed,
    }


def parse_args(arm: str = ARM_C_REPAIR_GRAPH) -> argparse.Namespace:
    if arm not in _VALID_ARMS:
        raise ValueError(f"Unknown Repair-only ablation arm: {arm}")
    graph_description = "enabled" if arm == ARM_C_REPAIR_GRAPH else "strictly disabled"
    parser = argparse.ArgumentParser(
        description=(
            "Repair-only ablation: original Issue directly to one Repair agent; "
            f"graph/index tools {graph_description}; no Scout, Hypothesis, Reproducer, "
            "Selector, TestAgent, or Reviewer."
        )
    )
    parser.add_argument(
        "--jsonl",
        default="INTENT2FIX/benchmark/multi_swebench_cpp_change.jsonl",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", default="glm-5.3-flash")
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default="low",
    )
    parser.add_argument("--repair-steps", type=int, default=200)
    parser.add_argument("--docker-registry", default="6pqg5gclf6yaz1ch6g.xuanyuan.run")
    parser.add_argument("--only", help="Comma-separated instance IDs")
    parser.add_argument("--skip", help="Comma-separated instance IDs")
    return parser.parse_args()


def main(
    arm: str = ARM_C_REPAIR_GRAPH,
    *,
    entrypoint: str | Path | None = None,
    repair_tool_profile: str = STANDARD_TOOL_PROFILE,
) -> int:
    args = parse_args(arm)
    cotest._load_env()
    if os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
    if args.workers < 1 or args.repair_steps < 1:
        raise ValueError("workers and repair-steps must be positive")

    jsonl = Path(args.jsonl)
    if not jsonl.is_absolute():
        jsonl = ROOT / jsonl
    jsonl = jsonl.resolve()
    all_ids = cotest.extract_instance_ids_from_jsonl(str(jsonl))
    only = split_ids(args.only)
    skip = split_ids(args.skip)
    unknown = (only | skip) - set(all_ids)
    if unknown:
        raise ValueError(f"Unknown instance IDs: {sorted(unknown)}")
    instance_ids = [
        instance_id
        for instance_id in all_ids
        if (not only or instance_id in only) and instance_id not in skip
    ]
    if not instance_ids:
        raise ValueError("No instances selected")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = apply_ablation_arm(
        {
            "jsonl": str(jsonl),
            "jsonl_sha256": sha256_file(jsonl),
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "repair_steps": args.repair_steps,
            "docker_registry": args.docker_registry,
            "workers": args.workers,
            "repair_tool_profile": repair_tool_profile,
        },
        arm,
    )
    configure_repair_only_environment(config)
    runner_entrypoint = Path(entrypoint or __file__).resolve()
    freeze_run_config(output_dir, config, instance_ids, runner_entrypoint)

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    outcomes: dict[str, dict[str, Any]] = {}
    write_json(output_dir / "progress.json", progress_payload(instance_ids, outcomes))
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.workers)
    interrupted = False
    try:
        futures = {
            executor.submit(
                run_case,
                instance_id,
                str(output_dir),
                run_id,
                config,
            ): instance_id
            for instance_id in instance_ids
        }
        for future in concurrent.futures.as_completed(futures):
            instance_id = futures[future]
            try:
                outcomes[instance_id] = future.result()
            except Exception as error:
                logging.exception("Worker crashed for %s", instance_id)
                outcomes[instance_id] = {
                    "instance_id": instance_id,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            write_json(
                output_dir / "progress.json",
                progress_payload(instance_ids, outcomes),
            )
    except KeyboardInterrupt:
        interrupted = True
        cotest._shutdown_process_pool(executor, 10)
        cotest._cleanup_run_containers(run_id)
    finally:
        if not interrupted:
            executor.shutdown(wait=True, cancel_futures=False)

    submissions = []
    for instance_id in instance_ids:
        status = outcomes.get(instance_id) or completed_outcome(
            output_dir / instance_id, instance_id
        )
        if not status or not status.get("patch_sha256"):
            continue
        patch = (output_dir / instance_id / "solution.patch").read_text(
            encoding="utf-8"
        )
        submissions.append({"instance_id": instance_id, "model_patch": patch})
    write_json(output_dir / "submission.json", submissions)
    final_progress = progress_payload(instance_ids, outcomes)
    final_progress["interrupted"] = interrupted
    final_progress["submission_count"] = len(submissions)
    write_json(output_dir / "progress.json", final_progress)
    return 130 if interrupted else (1 if final_progress["failed"] else 0)


if __name__ == "__main__":
    raise SystemExit(main(ARM_C_REPAIR_GRAPH))
