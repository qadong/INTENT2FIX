#!/usr/bin/env python3
"""Prepare frozen public Selector inputs; separately join official patch labels AFTER selection."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from INTENT2FIX.portfolio_selector import object_hash, patch_hash
from run_portfolio_selector import prepare_case, save


def read(path):
    return json.loads(path.read_text())


def label_ledger(summaries):
    labels = {}
    for summary_path in summaries:
        for row in read(summary_path)["results"]:
            log = Path(row["log_path"])
            if not log.is_absolute():
                log = ROOT / log
            model_path = log.with_name("model.patch")
            if not model_path.exists():
                raise ValueError(f"Cannot bind evaluation to actual submitted bytes: {model_path}")
            sha = patch_hash(model_path.read_text())
            if row.get("patch_sha256", sha) != sha:
                raise ValueError("Evaluation's declared patch hash differs from submitted bytes")
            fix = log.with_name("fix.patch")
            # Some historical submissions were wholly excluded by the evaluator.
            # Keep their recorded submission-level failure, not a made-up test failure.
            effective = patch_hash(fix.read_text()) if fix.exists() else None
            if effective is None and not (row.get("excluded_files") and not row.get("kept_files") and row["resolved"] is False):
                raise ValueError("Missing effective evaluated patch without explicit filtering evidence")
            key = row["instance_id"] + ":" + sha
            label = {"instance_id": row["instance_id"], "patch_sha256": sha,
                     "resolved": row["resolved"], "effective_patch_sha256": effective,
                     "evaluation_kind": "filtered_empty" if effective is None else "official_harness",
                     "sources": [{"summary": str(summary_path.resolve()), "model_patch": str(model_path),
                                  "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                                  "exit_code": row.get("exit_code")} ]}
            if type(label["resolved"]) is not bool:
                raise ValueError("Unfinished evaluation cannot become a false label")
            if key in labels:
                if labels[key]["resolved"] != label["resolved"] or labels[key]["effective_patch_sha256"] != effective:
                    raise ValueError(f"Conflicting official labels for {key}")
                labels[key]["sources"].extend(label["sources"])
            else:
                labels[key] = label
    return labels


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=False)
    records = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    inputs = []
    for record in records:
        iid = record["instance_id"]
        pool_path = args.pools / iid / "portfolio_candidate_pool.json"
        # Explicit public-field allowlist: dataset patch/test_patch/labels never reach inference.
        public = {key: record[key] for key in ("instance_id", "repo", "base_commit", "problem_statement")}
        portfolio_path = pool_path.with_name("intent_portfolio.json")
        portfolio = read(portfolio_path) if portfolio_path.exists() else {}
        repair_intent = {
            "common_obligations": portfolio.get("obligations", []),
            "open_questions": [
                {key: item.get(key) for key in ("statement", "impact")}
                for item in portfolio.get("unresolved_dimensions", [])
            ],
        }
        inputs.append(prepare_case(public, read(pool_path) if pool_path.exists() else [],
                                   args.repos / iid, repair_intent=repair_intent,
                                   hypotheses=portfolio.get("hypotheses", [])))
    (args.output / "inputs.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in inputs))
    labels = label_ledger([args.original, *args.alternatives])
    baseline = {}
    for row in read(args.original)["results"]:
        log = Path(row["log_path"])
        if not log.is_absolute():
            log = ROOT / log
        baseline[row["instance_id"]] = {"resolved": row["resolved"],
                                       "patch_sha256": patch_hash(log.with_name("model.patch").read_text())}
    keys = {c["instance_id"] + ":" + p["patch_sha256"] for c in inputs for p in c["candidates"]}
    orphan = set(labels) - keys
    if orphan:
        raise ValueError(f"{len(orphan)} evaluated patch hashes absent from frozen pool")
    save(args.output / "labels.json", {"labels": labels, "baseline": baseline})
    manifest = {"cases": len(inputs), "candidates": len(keys), "labeled_candidates": len(keys & set(labels)),
                "unknown_candidates": len(keys - set(labels)),
                "baseline_resolved": sum(b["resolved"] for b in baseline.values()),
                "known_pool_resolved": len({v["instance_id"] for v in labels.values() if v["resolved"]}),
                "input_sha256": hashlib.sha256((args.output / "inputs.jsonl").read_bytes()).hexdigest(),
                "labels_sha256": hashlib.sha256((args.output / "labels.json").read_bytes()).hexdigest(),
                "case_input_sha256": {c["instance_id"]: object_hash(c) for c in inputs},
                "inference_input_fields": ["public Issue", "base excerpts", "common repair intent", "candidate Hypothesis proposals", "actual patches", "apply checks", "unclassified old observations"],
                "gold_or_official_labels_in_inputs": False}
    save(args.output / "manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "case_input_sha256"}, indent=2))


def score_results(inputs, ledger, results):
    pools = {c["instance_id"]: c for c in inputs}
    labels, baseline = ledger["labels"], ledger["baseline"]
    rows = []
    for result in results:
        iid = result["instance_id"]
        if iid not in pools:
            raise ValueError("Selected case is not in frozen benchmark")
        hashes = {p["patch_sha256"] for p in pools[iid]["candidates"]}
        sha = result.get("selected_patch_sha256")
        if sha is not None and sha not in hashes:
            raise ValueError("Selected patch is absent from the frozen pool")
        label = labels.get(iid + ":" + sha) if sha else None
        value = label["resolved"] if label is not None else (False if sha is None else None)
        old = baseline.get(iid, {}).get("resolved", False)
        oracle = any(labels.get(iid + ":" + h, {}).get("resolved", False) for h in hashes)
        rows.append({"instance_id": iid, "selected_patch_sha256": sha, "resolved": value,
                     "status": result.get("status", "selected"),
                     "outcome_basis": "official_label" if label is not None else "unknown_label" if sha else "no_submission",
                     "old_resolved": old, "known_pool_resolved": oracle,
                     "transition": "unknown" if value is None else f"{int(old)}->{int(value)}"})
    if len({r["instance_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate case decisions")
    passed = sum(r["resolved"] is True for r in rows)
    unknown = sum(r["resolved"] is None for r in rows)
    old = sum(r["old_resolved"] for r in rows)
    return {"scored_cases": len(rows), "remaining_cases": len(inputs) - len(rows),
            "old_resolved_on_same_cases": old, "new_known_resolved": passed,
            "unknown_selected_labels": unknown,
            "recovered": [r["instance_id"] for r in rows if r["transition"] == "0->1"],
            "regressed": [r["instance_id"] for r in rows if r["transition"] == "1->0"],
            "net_change_bounds": [passed - old, passed + unknown - old],
            "known_pool_resolved_on_same_cases": sum(r["known_pool_resolved"] for r in rows),
            "development_set_not_holdout": True, "cases": rows}


def score(args):
    manifest = read(args.frozen / "manifest.json")
    for name, field in (("inputs.jsonl", "input_sha256"), ("labels.json", "labels_sha256")):
        if hashlib.sha256((args.frozen / name).read_bytes()).hexdigest() != manifest[field]:
            raise ValueError("Frozen input/label artifact changed")
    inputs = [json.loads(line) for line in (args.frozen / "inputs.jsonl").read_text().splitlines()]
    results = []
    result_paths = list(args.results.glob("*/result.json"))
    error_paths = [p for p in args.results.glob("*/error.json") if not p.with_name("result.json").exists()]
    for path in sorted(result_paths + error_paths):
        result = read(path)
        actual_input = read(path.with_name("input.json"))
        if path.name == "error.json":
            # A terminal pipeline failure earns zero benchmark credit. Do not hide
            # it by comparing only the successfully delivered selections.
            result = {**result, "input_sha256": object_hash(actual_input),
                      "selected_patch_sha256": None, "status": "pipeline_error"}
        if object_hash(actual_input) != result["input_sha256"]:
            raise ValueError("Decision input hash mismatch")
        # Candidate order perturbation is allowed; all bytes/evidence must otherwise match.
        actual_input["candidates"].sort(key=lambda c: c["patch_sha256"])
        if object_hash(actual_input) != manifest["case_input_sha256"][result["instance_id"]]:
            raise ValueError("Decision did not use the frozen case input")
        if result.get("selected_patch_sha256"):
            if patch_hash(path.with_name("selected.patch").read_text()) != result["selected_patch_sha256"]:
                raise ValueError("Selected artifact hash mismatch")
        results.append(result)
    ledger = read(args.frozen / "labels.json")
    pool_keys = {c["instance_id"] + ":" + p["patch_sha256"] for c in inputs for p in c["candidates"]}
    supplemental = label_ledger(args.additional_evaluation)
    for key, label in supplemental.items():
        if key not in pool_keys:
            raise ValueError("Supplemental evaluation does not match a frozen candidate")
        previous = ledger["labels"].get(key)
        if previous and (previous["resolved"] != label["resolved"] or previous["effective_patch_sha256"] != label["effective_patch_sha256"]):
            raise ValueError("Supplemental evaluation conflicts with the frozen label")
        ledger["labels"][key] = label
    report = score_results(inputs, ledger, results)
    report["label_count_used"] = len(ledger["labels"])
    report["supplemental_evaluations"] = [{"path": str(p.resolve()),
                                           "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                                          for p in args.additional_evaluation]
    report["failed_runs"] = [p.parent.name for p in args.results.glob("*/error.json")]
    save(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))


def export_unlabeled(args):
    """Export decisions with no existing official label for a separate evaluation."""
    rows = []
    for decision in read(args.score)["cases"]:
        if decision["outcome_basis"] != "unknown_label":
            continue
        if args.case and decision["instance_id"] not in args.case:
            continue
        patch = (args.results / decision["instance_id"] / "selected.patch").read_text()
        if patch_hash(patch) != decision["selected_patch_sha256"]:
            raise ValueError("Selected patch hash mismatch")
        rows.append({"instance_id": decision["instance_id"], "model_patch": patch})
    save(args.output, rows)
    print(f"Exported {len(rows)} unlabeled selections to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--pools", type=Path, required=True)
    p.add_argument("--repos", type=Path, default=Path("/data/dqa/cache/repos"))
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--alternatives", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=prepare)
    p = sub.add_parser("score")
    p.add_argument("--frozen", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--additional-evaluation", type=Path, action="append", default=[],
                   help="Official results for previously unknown selected hashes, scored only after decisions are frozen")
    p.set_defaults(func=score)
    p = sub.add_parser("export-unlabeled")
    p.add_argument("--score", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--case", action="append", default=[])
    p.set_defaults(func=export_unlabeled)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
