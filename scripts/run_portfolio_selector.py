#!/usr/bin/env python3
"""Selector-only CLI. Input is public Issue/base evidence plus frozen actual patches."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from INTENT2FIX.portfolio_selector import (
    DISCRIMINATOR_PROMPT, INTENT_SEARCH_PROMPT, ORACLE_REVIEW_PROMPT, PROBE_PROMPT, RANK_PROMPT,
    BuildRefinement, Decision, FixtureRefinement, IntentSearch, OracleReview, ProbePlan,
    base_sources, check_apply, check_decision, check_fixture_refinement, check_probes, check_search_queries, freeze_candidates, ground_decision,
    ground_probe_route, indistinguishable_groups, issue_contract, object_hash, public_header_route,
    rank_payload, resolve, search_base_sources,
)
from INTENT2FIX.selector_execution import PROBE_PATH, compile_candidate, execution_rows, probe_commands, refinement_args


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


class Model:
    def __init__(self, alias, directory, reasoning_effort="low"):
        from dotenv import load_dotenv
        from INTENT2FIX.model_config import get_model_config
        load_dotenv(ROOT / ".env", override=False)
        config = get_model_config(alias)
        self.kwargs = {"model": config["model"], "timeout": 240, "max_retries": 0,
                       "max_tokens": 16000, "temperature": 0.0}
        if config.get("model_base_url"):
            self.kwargs["api_base"] = config["model_base_url"]
        key = config.get("model_api_key") or os.getenv(config.get("model_api_key_env") or "CUSTOM_LLM_API_KEY")
        if key:
            self.kwargs["api_key"] = key
        if "glm-5.3-flash" in config["model"]:
            self.kwargs["extra_body"] = {"thinking": {"type": "enabled", "clear_thinking": False}, "reasoning_effort": "low"}
        self.directory = directory
        self.reasoning_effort = reasoning_effort
        self.usage = []

    def stage_kwargs(self, stage):
        kwargs = dict(self.kwargs)
        if (getattr(self, "reasoning_effort", "high") == "high"
                and stage in ("probe_plan", "oracle_review", "discriminator_plan",
                              "discriminator_oracle_review", "intent_search", "rank")
                and "extra_body" in kwargs):
            kwargs["extra_body"] = {**kwargs["extra_body"], "reasoning_effort": "high"}
            kwargs["max_tokens"] = 32768
            kwargs["timeout"] = 480
        return kwargs

    def ask(self, stage, instruction, payload, schema, validate):
        import litellm
        from INTENT2FIX.completion.structured_compactor import parse_json_object
        prompt = instruction + "\nSchema:\n" + json.dumps(schema.model_json_schema()) + "\nEvidence:\n" + json.dumps(payload, ensure_ascii=False)
        (self.directory / f"{stage}.prompt.txt").write_text(prompt)
        last_error = ""
        previous_content = ""
        for attempt in range(2):
            start = time.monotonic()
            try:
                messages = [{"role": "user", "content": prompt}]
                if last_error:
                    if previous_content:
                        messages.append({"role": "assistant", "content": previous_content})
                    messages.append({"role": "user", "content": "The prior response was invalid: " + last_error + ". Return corrected complete schema JSON using only the frozen evidence."})
                kwargs = self.stage_kwargs(stage)
                stream = litellm.completion(**kwargs, messages=messages, stream=True,
                                             stream_options={"include_usage": True})
                parts, usage, finish_reason = [], None, None
                reasoning_chars = 0
                try:
                    for chunk in stream:
                        if getattr(chunk, "usage", None) is not None:
                            usage = chunk.usage.model_dump()
                        if not chunk.choices:
                            continue
                        choice = chunk.choices[0]
                        parts.append(choice.delta.content or "")
                        reasoning_chars += len(getattr(choice.delta, "reasoning_content", None) or "")
                        if choice.finish_reason:
                            finish_reason = choice.finish_reason
                finally:
                    close = getattr(stream, "close", None)
                    if close:
                        close()
                content = "".join(parts)
                previous_content = content
                self.usage.append({"stage": stage, "attempt": attempt + 1,
                                   "reasoning_effort": kwargs.get("extra_body", {}).get("reasoning_effort"),
                                   "seconds": round(time.monotonic() - start, 3),
                                   "streaming": True, "reasoning_chars_received": reasoning_chars,
                                   "usage": usage, "finish_reason": finish_reason})
                (self.directory / f"{stage}.response{attempt+1}.txt").write_text(content)
                result = schema.model_validate(parse_json_object(content))
                validate(result)
                save(self.directory / f"{stage}.json", result.model_dump())
                return result
            except (ValueError, TypeError) as exc:
                last_error = str(exc)[:1800]
            except Exception as exc:
                # Avoid leaking credentials/provider request objects through exception text.
                last_error = type(exc).__name__
                self.usage.append({"stage": stage, "attempt": attempt + 1,
                                   "seconds": round(time.monotonic() - start, 3),
                                   "error_type": last_error, "usage": None})
            finally:
                save(self.directory / "usage.json", self.usage)
            save(self.directory / f"{stage}.validation{attempt+1}.json", {"error": last_error})
        raise RuntimeError(f"{stage} failed after two attempts: {last_error}")


def prepare_case(instance, raw, repo, repair_intent=None, hypotheses=None):
    candidates = freeze_candidates(raw, hypotheses)
    for candidate in candidates:
        candidate["application"] = check_apply(repo, instance["base_commit"], candidate["patch"])
    sources = base_sources(repo, instance["base_commit"], instance["problem_statement"], candidates)
    evidence = []
    # Historical coarse labels/oracles are deliberately stripped. Raw observations can
    # suggest a fresh probe, but their fixture/phase/hash provenance was not verified.
    by_sha = {c.get("patch_sha256"): c for c in raw}
    for candidate in candidates:
        rows = by_sha.get(candidate["patch_sha256"], {}).get("cross_execution", {}).get("probe_results", [])
        for index, row in enumerate(rows[:4], 1):
            evidence.append({"evidence_id": f"LEGACY:{candidate['candidate_id']}:{index}",
                             "candidate_id": candidate["candidate_id"], "patch_sha256": candidate["patch_sha256"],
                             "phase": "legacy_unclassified", "adjudicative": False,
                             "command": row.get("command", ""), "exit_code": row.get("exit_code"),
                             "output": row.get("output", "")[:1600]})
    return {"instance_id": instance["instance_id"], "repo": instance["repo"],
            "base_commit": instance["base_commit"], "repo_path": str(repo.resolve()), "sources": sources,
            "candidates": candidates, "evidence": evidence,
            "repair_intent": repair_intent or {}}


def compilation_results(case, image):
    work_dir = f"/home/{case['repo'].split('/')[1]}"
    results = {}
    for candidate in case["candidates"]:
        cid = candidate["candidate_id"]
        if candidate["application"]["status"] == "failed":
            results[cid] = {"candidate_id": cid, "patch_sha256": candidate["patch_sha256"],
                            "status": "failed", "phase": "apply_check", "output": "Patch cannot apply at base"}
            continue
        results[cid] = compile_candidate(candidate, image=image, work_dir=work_dir,
                                         base_commit=case["base_commit"])
    return results


def run_case(case, directory, *, model_alias="glm-5.3-flash", execute=False, reverse=False,
             registry="6pqg5gclf6yaz1ch6g.xuanyuan.run", reasoning_effort="low",
             compile_runner=None):
    directory.mkdir(parents=True, exist_ok=False)
    case = json.loads(json.dumps(case))
    if reverse:
        case["candidates"].reverse()
    save(directory / "input.json", case)
    input_sha = object_hash(case)
    if not case["candidates"]:
        result = {"instance_id": case["instance_id"], "input_sha256": input_sha,
                  "status": "empty_pool", "selected_patch_sha256": None, "ranking": []}
        save(directory / "result.json", result)
        return result
    org, repo = case["repo"].lower().split("/")
    number = case["instance_id"].rsplit("-", 1)[1]
    image = f"{registry}/mswebench/{org}_m_{repo}:pr-{number}"
    compilation = (compile_runner or compilation_results)(case, image)
    if set(compilation) != {c["candidate_id"] for c in case["candidates"]} or any(
        compilation[c["candidate_id"]].get("patch_sha256") != c["patch_sha256"]
        for c in case["candidates"]
    ):
        raise ValueError("Compilation results do not bind every frozen candidate patch")
    save(directory / "compilation.json", compilation)
    if not any(row["status"] == "passed" for row in compilation.values()):
        result = {"instance_id": case["instance_id"], "input_sha256": input_sha,
                  "status": "no_compiling_patch", "selected_candidate_id": None,
                  "selected_patch_sha256": None,
                  "ranking": [c["candidate_id"] for c in case["candidates"]],
                  "probe_count": 0, "execution_rows": 0, "official_labels_used": False}
        save(directory / "result.json", result)
        save(directory / "usage.json", [])
        return result
    model = Model(model_alias, directory, reasoning_effort)
    evidence = case["evidence"][:]
    contract = issue_contract(case["sources"])
    save(directory / "contract.json", contract.model_dump())
    searched = []
    if case.get("repo_path") and Path(case["repo_path"]).is_dir():
        try:
            search = model.ask("intent_search", INTENT_SEARCH_PROMPT,
                               rank_payload(case, contract, evidence, compilation=compilation),
                               IntentSearch, lambda plan: check_search_queries(plan.queries))
            searched = search_base_sources(Path(case["repo_path"]), case["base_commit"], search.queries)
            save(directory / "source_search.json", searched)
            case["sources"].extend(searched)
        except (RuntimeError, ValueError, OSError) as exc:
            save(directory / "source_search_unavailable.json", {"error_type": type(exc).__name__})
    def validate(decision, allow_probes):
        ground_decision(decision, case, contract, evidence)
        check_decision(decision, case, contract, evidence, allow_probes=allow_probes,
                       require_comparisons=True)
        for probe in decision.probes:
            probe_commands(probe)
    probes = []
    route = public_header_route(case["sources"])
    def review_oracles(planned, stage):
        if not planned:
            return planned
        def validate_review(review):
            if len(review.reviews) != len(planned) or {
                item.probe_id for item in review.reviews
            } != {probe.probe_id for probe in planned}:
                raise ValueError("Oracle review must cover every planned probe exactly once")
            by_id = {probe.probe_id: probe for probe in planned}
            for item in review.reviews:
                probe = by_id[item.probe_id]
                item.keep_expected_exit_code = bool(
                    item.keep_expected_exit_code and probe.expected_exit_code is not None)
                item.keep_required_output_regex = bool(
                    item.keep_required_output_regex and probe.required_output_regex is not None)
        try:
            review = model.ask(stage, ORACLE_REVIEW_PROMPT,
                               {"sources": case["sources"],
                                "probes": [probe.model_dump() for probe in planned]},
                               OracleReview, validate_review)
            reviews = {item.probe_id: item for item in review.reviews}
        except RuntimeError:
            save(directory / f"{stage}_unavailable.json", {
                "status": "unknown", "reason": "No valid independent oracle review; retain observations only"})
            reviews = {}
        for probe in planned:
            item = reviews.get(probe.probe_id)
            if item is None or not item.keep_expected_exit_code:
                probe.expected_exit_code = None
            if item is None or not item.keep_required_output_regex:
                probe.required_output_regex = None
            if probe.expected_exit_code is None and probe.required_output_regex is None:
                probe.clause_id = None
        save(directory / f"{stage}_applied.json", [probe.model_dump() for probe in planned])
        return planned
    if execute and sum(row["status"] == "passed" for row in compilation.values()) >= 2:
        def validate_plan(plan, *, extra=False):
            allowed = {"P3"} if extra else {"P1", "P2"}
            if len(plan.probes) > len(allowed) or any(p.probe_id not in allowed for p in plan.probes):
                raise ValueError(f"This stage permits only probe IDs {sorted(allowed)}")
            for probe in plan.probes:
                ground_probe_route(probe, route)
                if probe.clause_id is None and (probe.expected_exit_code is not None or probe.required_output_regex is not None):
                    plan.reason += f" Unbound expectation for {probe.probe_id} retained only in raw response; execution is observation-only."
                    probe.expected_exit_code = None
                    probe.required_output_regex = None
            check_probes(plan.probes, case, contract)
            for probe in plan.probes:
                probe_commands(probe)
        try:
            plan = model.ask("probe_plan", PROBE_PROMPT,
                             rank_payload(case, contract, evidence, compilation=compilation),
                             ProbePlan, validate_plan)
            probes = review_oracles(plan.probes, "oracle_review")
        except RuntimeError:
            save(directory / "probe_unavailable.json", {"status": "unknown",
                 "reason": "No valid bounded probe plan was delivered; rank with available static evidence"})
    if probes:
        def execute_probe(probe, attempt=""):
            rows = execution_rows(case, probe, image=image, attempt=attempt)
            evidence.extend(rows)
            save(directory / "execution.json", evidence)
            return rows
        def all_builds_failed(rows):
            builds = [r for r in rows if r["phase"] == "compile"]
            return len(builds) == len(case["candidates"]) + 1 and all(r["status"] != "passed" for r in builds)
        def run_probe(probe):
            rows = execute_probe(probe)
            builds = [r for r in rows if r["phase"] == "compile"]
            if all_builds_failed(rows):
                # One shared build correction, with identical source/input/oracle.
                # Failure to repair a fixture does not invalidate the static decision.
                try:
                    def validate_refinement(refinement):
                        refinement.compile_args = refinement_args(refinement.compile_args)
                        fixed = probe.model_copy(update={"compiler": refinement.compiler,
                                                          "compile_args": refinement.compile_args})
                        ground_probe_route(fixed, route)
                        probe_commands(fixed)
                        refinement.compile_args = fixed.compile_args
                    refinement = model.ask(f"{probe.probe_id}_build_refine",
                        "Correct only the shared compiler/build arguments using public base build sources "
                        "and diagnostics. Keep the probe source, runtime inputs, assertions, expected behavior "
                        "and candidate patches fixed. Do not disable assertions or change feature macros to "
                        "bypass checks. Do not switch compilers merely to make an unsupported probe API "
                        "or format spec compile; use the public build route unless base build evidence "
                        "requires otherwise. The harness adds its probe source and -o binary: omit those from "
                        "compile_args. Example: [\"-std=c++11\", \"-Iinclude\"]. Return BuildRefinement JSON.",
                        {"sources": case["sources"], "probe": probe.model_dump(), "compile_rows": builds},
                        BuildRefinement, validate_refinement)
                    fixed = probe.model_copy(update={"compiler": refinement.compiler,
                                                     "compile_args": refinement.compile_args})
                    if fixed != probe:
                        rows = execute_probe(fixed, "build_retry")
                        probe = fixed
                except RuntimeError:
                    save(directory / f"{probe.probe_id}_build_unavailable.json",
                         {"status": "unknown", "reason": "Shared build refinement unavailable"})
            builds = [r for r in rows if r["phase"] == "compile"]
            if all_builds_failed(rows) and any(PROBE_PATH in r.get("output", "") for r in builds):
                try:
                    def validate_fixture(refinement):
                        check_fixture_refinement(probe.source, refinement.source)
                        probe_commands(probe.model_copy(update={"source": refinement.source}))
                    repair = model.ask(f"{probe.probe_id}_fixture_refine",
                        "Every base/candidate build failed, including diagnostics in the harness program. "
                        "Correct shared scaffolding mistakes by ADDING missing declarations, includes or "
                        "constructors. Preserve every original non-comment code token in order, including "
                        "the tested library call, its receiver/arguments, assertions and output. Do not rename "
                        "that call or implement its operation manually inside the probe. Baseline is allowed "
                        "to keep failing from the original Issue; do not try to repair that product failure. "
                        "Compare diagnostics across the matrix to isolate shared scaffolding errors. "
                        "Do not edit candidate sources, disable checks, "
                        "replace the program by printing an expected answer, or add new test scenarios. "
                        "Compiler arguments and the external expected exit/output are frozen. Return the complete "
                        "corrected source and explain the minimal compatibility correction; if impossible, "
                        "return the original source and explain why.",
                        {"sources": case["sources"], "probe": probe.model_dump(), "compile_rows": builds},
                        FixtureRefinement, validate_fixture)
                    fixed = probe.model_copy(update={"source": repair.source})
                    if fixed != probe:
                        execute_probe(fixed, "fixture_retry")
                        probe = fixed
                except RuntimeError:
                    save(directory / f"{probe.probe_id}_fixture_unavailable.json",
                         {"status": "unknown", "reason": "Shared fixture compilation correction unavailable"})
            return probe
        probes = [run_probe(probe) for probe in probes]
        groups = indistinguishable_groups(case, evidence, probes)
        if groups:
            try:
                extra_plan = model.ask("discriminator_plan", DISCRIMINATOR_PROMPT,
                    {**rank_payload(case, contract, evidence, compilation=compilation),
                     "earlier_probes": [p.model_dump() for p in probes], "indistinguishable_candidates": groups},
                    ProbePlan, lambda p: validate_plan(p, extra=True))
                probes.extend(run_probe(probe) for probe in review_oracles(
                    extra_plan.probes, "discriminator_oracle_review"))
            except RuntimeError:
                save(directory / "discriminator_unavailable.json", {"status": "unknown",
                     "reason": "No valid extra discriminator delivered; retain existing evidence"})
    decision = model.ask("rank", RANK_PROMPT,
                         rank_payload(case, contract, evidence, final=True, execution_disabled=not execute,
                                      compilation=compilation),
                         Decision, lambda x: validate(x, False))
    result = {**resolve(decision, case, evidence=evidence, probes=probes,
                        compilation=compilation),
              "instance_id": case["instance_id"], "input_sha256": input_sha,
              "model": model_alias, "contract_sha256": object_hash(contract.model_dump()),
              "compilation_sha256": object_hash(compilation),
              "source_search_sha256": object_hash(searched),
              "decision": decision.model_dump(),
              "probe_count": len(probes),
              "execution_rows": len(evidence) - len(case["evidence"]), "official_labels_used": False}
    save(directory / "result.json", result)
    selected = result["selected_patch_sha256"]
    if selected:
        patch = next(c["patch"] for c in case["candidates"] if c["patch_sha256"] == selected)
        (directory / "selected.patch").write_text(patch)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Frozen public input JSONL from selector_benchmark.py prepare")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--model", default="glm-5.3-flash")
    parser.add_argument("--reasoning-effort", choices=("low", "high"), default="low")
    parser.add_argument("--execute-probes", action="store_true")
    parser.add_argument("--reverse-candidate-order", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [json.loads(line) for line in args.inputs.read_text().splitlines() if line.strip()]
    if set(args.case) - {c["instance_id"] for c in cases}:
        parser.error("Requested case missing from frozen inputs")
    for case in cases:
        iid = case["instance_id"]
        if args.case and iid not in args.case:
            continue
        directory = args.output / iid
        if directory.exists():
            raise FileExistsError(f"Refusing to overwrite frozen selector run: {directory}")
        print(f"START {iid}", flush=True)
        try:
            result = run_case(case, directory, model_alias=args.model, execute=args.execute_probes,
                              reverse=args.reverse_candidate_order,
                              reasoning_effort=args.reasoning_effort)
            print(f"DONE {iid} {result['status']} {result.get('selected_patch_sha256')}", flush=True)
        except Exception as exc:
            directory.mkdir(parents=True, exist_ok=True)
            save(directory / "error.json", {"instance_id": iid, "error_type": type(exc).__name__})
            print(f"FAILED {iid} {type(exc).__name__}", flush=True)


if __name__ == "__main__":
    main()
