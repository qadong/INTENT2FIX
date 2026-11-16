#!/usr/bin/env python3
"""Compare long-context response latency for GLM-5.3-Flash and Qwen3.8-Flash."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_SOURCES = (
    "scripts/run_no_memory_cotest.py",
    "INTENT2FIX/agent/code_prompts.py",
    "INTENT2FIX/agent/code_agent.py",
    "INTENT2FIX/patch_intent_audit/auditor.py",
    "INTENT2FIX/patch_intent_audit/trajectory_leads.py",
    "INTENT2FIX/patch_review/reviewer.py",
    "INTENT2FIX/completion/base.py",
    "INTENT2FIX/actions/run_single_test.py",
)

SYSTEM_PROMPT = """You are reviewing a long C++ repair-agent implementation snapshot.
Use only the supplied repository evidence. Separate observations from inferences. Identify the
smallest mechanism-level causes, preserve public behavior not changed by the task, and propose
focused executable checks. Finish by calling SubmitLongTaskAnalysis exactly once. Do not emit a
normal text answer."""

TASK_PROMPT = """Analyze why a candidate repair can pass focused checks but still fail official
evaluation. Trace interactions among repair completion, patch-intent audit, same-context
refinement, patch-surface review, test execution, and provider completion handling. Find at least
five concrete evidence locations from different components. Return a concise diagnosis, a minimal
repair plan, a validation plan, and residual risks. Do not assume that every listed component is
faulty, and do not recommend repository-specific hard-coded gates.

Repository snapshot follows:
"""

SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "SubmitLongTaskAnalysis",
        "description": "Submit the final evidence-grounded long-task analysis.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "diagnosis",
                "evidence",
                "repair_plan",
                "validation_plan",
                "residual_risks",
            ],
            "properties": {
                "diagnosis": {"type": "string"},
                "evidence": {
                    "type": "array",
                    "minItems": 5,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["location", "observation", "implication"],
                        "properties": {
                            "location": {"type": "string"},
                            "observation": {"type": "string"},
                            "implication": {"type": "string"},
                        },
                    },
                },
                "repair_plan": {
                    "type": "array",
                    "minItems": 3,
                    "items": {"type": "string"},
                },
                "validation_plan": {
                    "type": "array",
                    "minItems": 3,
                    "items": {"type": "string"},
                },
                "residual_risks": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
    },
}


@dataclass(frozen=True)
class ModelSpec:
    label: str
    model: str
    base_url: str
    key_env: str
    extra_body: dict[str, Any]


MODEL_SPECS = {
    "glm": ModelSpec(
        label="glm-5.3-flash",
        model="glm-5.3-flash",
        base_url="https://api.z.ai/api/paas/v4",
        key_env="ZAI_API_KEY",
        extra_body={
            "thinking": {"type": "enabled", "clear_thinking": False},
            "reasoning_effort": "high",
        },
    ),
    "qwen": ModelSpec(
        label="qwen3.8-flash",
        model="qwen3.8-flash",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        key_env="DASHSCOPE_API_KEY",
        extra_body={
            "enable_thinking": True,
            "preserve_thinking": True,
            "reasoning_effort": "xhigh",
        },
    ),
}


def _sample_file(path: Path, token_budget: int, encoding: Any) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    tokens = encoding.encode(text)
    if len(tokens) <= token_budget:
        sample = text
    else:
        half = max(1, token_budget // 2)
        sample = (
            encoding.decode(tokens[:half])
            + "\n\n# ... middle omitted by latency benchmark ...\n\n"
            + encoding.decode(tokens[-half:])
        )
    numbered = "\n".join(
        f"{line_no:06d}: {line}"
        for line_no, line in enumerate(sample.splitlines(), start=1)
    )
    return f"\n\n===== {path.relative_to(ROOT)} =====\n{numbered}"


def _synthetic_context() -> str:
    blocks = []
    for index in range(1, 1201):
        family = ("parser", "formatter", "iterator", "serializer")[index % 4]
        blocks.append(
            f"""
===== synthetic/{family}/component_{index:04d}.cpp =====
{index * 10 + 1:06d}: public_result component_{index:04d}::execute(input_view input) const {{
{index * 10 + 2:06d}:   auto state = detail::{family}_dispatch(input, policy_);
{index * 10 + 3:06d}:   return public_result{{state.value, state.status}};
{index * 10 + 4:06d}: }}
===== synthetic/include/component_{index:04d}.hpp =====
{index * 10 + 5:06d}: public_result execute(input_view input) const;
{index * 10 + 6:06d}: result<public_result> execute_checked(input_view input) const;
===== synthetic/tests/component_{index:04d}_test.cpp =====
{index * 10 + 7:06d}: CHECK(component.execute("ordinary").status == status::ok);
{index * 10 + 8:06d}: CHECK(component.execute_checked("ordinary")->status == status::ok);
===== synthetic/trace/RUN-{index:04d} =====
command: compile_and_run component_{index:04d}_test.cpp
exit: 0
observation: ordinary input follows {family}_dispatch and preserves public_result status.
candidate-note: focused test covers the direct entry only; wrapper and generated mirror remain unverified.
"""
        )
    return "".join(blocks)


def build_messages(
    target_tokens: int,
    *,
    repository_context: bool = False,
) -> tuple[list[dict[str, str]], int]:
    encoding = tiktoken.get_encoding("cl100k_base")
    fixed_tokens = len(encoding.encode(SYSTEM_PROMPT + TASK_PROMPT))
    context_budget = max(1000, target_tokens - fixed_tokens)
    if repository_context:
        per_file = max(200, context_budget // len(REPOSITORY_SOURCES))
        context = "".join(
            _sample_file(ROOT / relative_path, per_file, encoding)
            for relative_path in REPOSITORY_SOURCES
        )
    else:
        context = _synthetic_context()
    context = encoding.decode(encoding.encode(context)[:context_budget])
    user_content = TASK_PROMPT + context
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    estimated_tokens = len(encoding.encode(SYSTEM_PROMPT + user_content))
    return messages, estimated_tokens


def _delta_reasoning(delta: Any) -> str:
    reasoning = getattr(delta, "reasoning_content", None)
    if reasoning:
        return reasoning
    model_extra = getattr(delta, "model_extra", None) or {}
    return model_extra.get("reasoning_content") or model_extra.get("reasoning") or ""


def run_once(
    spec: ModelSpec,
    messages: list[dict[str, str]],
    *,
    trial: int,
    max_tokens: int,
    timeout: float,
    reasoning_effort: str,
) -> dict[str, Any]:
    api_key = os.getenv(spec.key_env)
    if not api_key:
        raise RuntimeError(f"{spec.key_env} is not configured")

    client = OpenAI(
        api_key=api_key,
        base_url=spec.base_url,
        timeout=timeout,
        max_retries=0,
    )
    started = time.perf_counter()
    metrics: dict[str, Any] = {
        "model": spec.label,
        "trial": trial,
        "first_event_seconds": None,
        "first_reasoning_seconds": None,
        "first_content_seconds": None,
        "first_tool_call_seconds": None,
        "total_seconds": None,
        "reasoning_characters": 0,
        "content_characters": 0,
        "tool_argument_characters": 0,
        "stream_chunks": 0,
        "finish_reason": None,
        "success": False,
        "error": None,
    }

    try:
        extra_body = dict(spec.extra_body)
        extra_body["reasoning_effort"] = reasoning_effort
        stream = client.chat.completions.create(
            model=spec.model,
            messages=messages,
            tools=[SUBMIT_TOOL],
            tool_choice="auto",
            temperature=1.0,
            max_tokens=max_tokens,
            stream=True,
            extra_body=extra_body,
        )
        for chunk in stream:
            metrics["stream_chunks"] += 1
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            now = time.perf_counter() - started
            reasoning = _delta_reasoning(delta)
            content = getattr(delta, "content", None) or ""
            tool_calls = getattr(delta, "tool_calls", None) or []
            has_event = bool(reasoning or content or tool_calls)
            if has_event and metrics["first_event_seconds"] is None:
                metrics["first_event_seconds"] = now
            if reasoning:
                if metrics["first_reasoning_seconds"] is None:
                    metrics["first_reasoning_seconds"] = now
                metrics["reasoning_characters"] += len(reasoning)
            if content:
                if metrics["first_content_seconds"] is None:
                    metrics["first_content_seconds"] = now
                metrics["content_characters"] += len(content)
            if tool_calls:
                if metrics["first_tool_call_seconds"] is None:
                    metrics["first_tool_call_seconds"] = now
                for tool_call in tool_calls:
                    function = getattr(tool_call, "function", None)
                    arguments = getattr(function, "arguments", None) if function else None
                    metrics["tool_argument_characters"] += len(arguments or "")
            if choice.finish_reason:
                metrics["finish_reason"] = choice.finish_reason
        metrics["success"] = metrics["first_tool_call_seconds"] is not None
    except Exception as exc:  # Preserve provider failures in the benchmark report.
        metrics["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        metrics["total_seconds"] = time.perf_counter() - started
    return metrics


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for label in ("glm-5.3-flash", "qwen3.8-flash"):
        selected = [row for row in rows if row["model"] == label and row["success"]]
        summary[label] = {
            "successful_trials": len(selected),
            "median_first_event_seconds": (
                statistics.median(row["first_event_seconds"] for row in selected)
                if selected
                else None
            ),
            "median_first_tool_call_seconds": (
                statistics.median(row["first_tool_call_seconds"] for row in selected)
                if selected
                else None
            ),
            "median_total_seconds": (
                statistics.median(row["total_seconds"] for row in selected)
                if selected
                else None
            ),
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tokens", type=int, default=32000)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--glm-reasoning-effort", default="high")
    parser.add_argument("--qwen-reasoning-effort", default="medium")
    parser.add_argument(
        "--models",
        default="glm,qwen",
        help="Comma-separated subset of: glm,qwen",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "model_latency" / "glm_vs_qwen_long_task.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--repository-context",
        action="store_true",
        help="Use current repository sources instead of the safe synthetic context.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(ROOT / ".env")
    messages, estimated_tokens = build_messages(
        args.input_tokens,
        repository_context=args.repository_context,
    )
    model_keys = tuple(key.strip() for key in args.models.split(",") if key.strip())
    unknown_models = sorted(set(model_keys) - set(MODEL_SPECS))
    if not model_keys or unknown_models:
        raise ValueError(f"Invalid --models value: {args.models}")
    print(f"Estimated input tokens: {estimated_tokens}", flush=True)
    if args.dry_run:
        return 0

    rows: list[dict[str, Any]] = []
    for trial in range(1, args.trials + 1):
        order = model_keys if trial % 2 else tuple(reversed(model_keys))
        for model_key in order:
            spec = MODEL_SPECS[model_key]
            print(f"Trial {trial}: {spec.label} ...", flush=True)
            row = run_once(
                spec,
                messages,
                trial=trial,
                max_tokens=args.max_output_tokens,
                timeout=args.timeout,
                reasoning_effort=(
                    args.glm_reasoning_effort
                    if model_key == "glm"
                    else args.qwen_reasoning_effort
                ),
            )
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "configuration": {
            "estimated_input_tokens": estimated_tokens,
            "requested_input_tokens": args.input_tokens,
            "max_output_tokens": args.max_output_tokens,
            "trials": args.trials,
            "models": list(model_keys),
            "context_source": "repository" if args.repository_context else "synthetic",
            "glm_reasoning_effort": args.glm_reasoning_effort,
            "qwen_reasoning_effort": args.qwen_reasoning_effort,
        },
        "runs": rows,
        "summary": summarize(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {args.output}", flush=True)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)
    return 0 if all(row["success"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
