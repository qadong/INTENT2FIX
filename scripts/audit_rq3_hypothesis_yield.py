"""Count selected graph-on hypotheses and official passing candidate yield.

All inputs are existing local artifacts. A hypothesis is one selected branch in
one case run; multiple patch variants from the same branch count once.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = Path("/data/dqa/multi-swe-bench/dataset/multi_swebench_cpp.jsonl")
RUNS = (
    (
        "hypothesis_v4_downstream10_unified_four_graph_compare_low_20260925/graph_on",
        "fix_patch_eval/hypothesis_v4_downstream10_graph_compare_20260926",
        True,
    ),
    (
        "fullflow_graph_scope_v5_paired10_low_20260926/graph_on",
        "fix_patch_eval/fullflow_graph_scope_v5_paired10_all_candidates_20260926",
        True,
    ),
    (
        "fullflow_graph_scope_v5_round3_paired10_low_20260926/graph_on",
        "results/fullflow_graph_scope_v5_round3_paired10_low_20260926/official_eval_graph_on",
        True,
    ),
    (
        "fullflow_graph_scope_v5_round4_paired10_low_20260926/graph_on",
        "results/fullflow_graph_scope_v5_round4_paired10_low_20260926/official_eval_graph_on",
        True,
    ),
    (
        "fullflow_graph_scope_v5_remaining88_paired_low_20260927/graph_on",
        "results/fullflow_graph_scope_v5_remaining88_paired_low_20260927/official_eval_completed_20260927_1305/graph_on",
        True,
    ),
    (
        "fullflow_verification_v7_remaining34_paired_low_20260928/graph_on",
        "results/fullflow_verification_v7_all_available_official_20260928/official_eval_graph_on",
        True,
    ),
    (
        "graph_pruning_retry4_scout_hypothesis_repair_low_20260926",
        "fix_patch_eval/graph_pruning_retry4_all_candidates_20260926",
        False,
    ),
    (
        "fullflow_verification_v7_retry_hypothesis3_low_20260928/graph_on",
        "results/fullflow_verification_v7_all_available_official_20260928/official_eval_graph_on",
        True,
    ),
    (
        "fullflow_v112_regression10_graph_on_low_20260928/graph_on",
        "results/fullflow_v112_regression10_graph_on_low_20260928/official_eval_graph_on",
        False,
    ),
)


def read_json(path: Path):
    return json.loads(path.read_text())


def official_by_hash(root: Path) -> dict[tuple[str, str], bool]:
    outcomes = {}
    for path in root.glob("candidates/*/*/result.json"):
        result = read_json(path)
        case_id = result.get("instance_id") or path.parent.name
        hashes = {result.get("patch_sha256")}
        hashes.update(
            source.get("reported_sha256")
            for source in result.get("sources", [])
            if isinstance(source, dict)
        )
        for digest in hashes - {None, ""}:
            key = (case_id, digest)
            outcomes[key] = outcomes.get(key, False) or result.get("resolved") is True
    return outcomes


def audit() -> dict:
    dataset_ids = {
        json.loads(line)["instance_id"] for line in DATASET.read_text().splitlines()
    }
    eligible_ids = {
        case_id for case_id in dataset_ids
        if not case_id.startswith("yhirose__cpp-httplib-")
    }
    rows = []
    mismatches = []
    run_case_generated = {}
    outcome_cache = {}
    for run_index, (run, eval_root, original_cohort) in enumerate(RUNS):
        case_root = ROOT / "results" / run / "cases_glm-5.3-flash_cotest"
        if eval_root not in outcome_cache:
            outcome_cache[eval_root] = official_by_hash(ROOT / eval_root)
        outcomes = outcome_cache[eval_root]
        for portfolio_path in sorted(case_root.glob("*/intent_portfolio.json")):
            case_id = portfolio_path.parent.name
            portfolio = read_json(portfolio_path)
            run_case_generated[(run_index, case_id)] = len(portfolio.get("hypotheses", []))
            execution_path = portfolio_path.with_name("intent_hypothesis_execution.json")
            execution = read_json(execution_path) if execution_path.exists() else {}
            selected = execution.get("selected_hypothesis_ids")
            if selected is None:
                selected = portfolio.get("selected_hypothesis_ids", [])
            if execution and selected != portfolio.get("selected_hypothesis_ids", []):
                mismatches.append(f"{run}:{case_id}")
            if len(selected) != len(set(selected)):
                raise ValueError(f"Duplicate selected hypothesis: {run}:{case_id}")
            candidates_path = portfolio_path.with_name("portfolio_candidate_pool.json")
            candidates = read_json(candidates_path) if candidates_path.exists() else []
            for hypothesis_id in selected:
                digests = {
                    row.get("patch_sha256") for row in candidates
                    if row.get("hypothesis_id") == hypothesis_id
                    and row.get("patch_sha256")
                }
                statuses = [
                    outcomes[(case_id, digest)] for digest in digests
                    if (case_id, digest) in outcomes
                ]
                if any(statuses):
                    status = "passing"
                elif statuses:
                    status = "evaluated_not_passing"
                elif digests:
                    status = "not_officially_evaluated"
                else:
                    status = "no_patch"
                rows.append({
                    "case_id": case_id,
                    "run": run,
                    "run_index": run_index,
                    "original_cohort": original_cohort,
                    "hypothesis_id": hypothesis_id,
                    "status": status,
                    "candidate_hashes": len(digests),
                    "evaluated_hashes": len(statuses),
                })
    if {row["case_id"] for row in rows} != eligible_ids:
        raise ValueError("Hypothesis artifacts do not cover all eligible cases")

    # The initial case-complete comparison keeps one Hypothesis run per case.
    # Retry3 fills a missing original portfolio; later quality reruns do not
    # supersede a successful first Hypothesis run for this sensitivity view.
    first_run = {}
    for row in rows:
        if row["original_cohort"]:
            first_run.setdefault(row["case_id"], row["run_index"])
    one_run_rows = [
        row for row in rows if row["run_index"] == first_run.get(row["case_id"])
    ]
    latest_run = {}
    for row in rows:
        latest_run[row["case_id"]] = row["run_index"]
    latest_rows = [
        row for row in rows if row["run_index"] == latest_run[row["case_id"]]
    ]

    def summarize(items: list[dict]) -> dict:
        status_counts = Counter(row["status"] for row in items)
        attempts = {(row["run_index"], row["case_id"]) for row in items}
        return {
            "case_count": len({row["case_id"] for row in items}),
            "run_case_count": len(attempts),
            "generated_hypotheses": sum(run_case_generated[key] for key in attempts),
            "selected_hypotheses": len(items),
            "passing_hypotheses": status_counts["passing"],
            "passing_rate": round(status_counts["passing"] / len(items), 6),
            "status_counts": dict(status_counts),
            "cases_with_passing_hypothesis": len({
                row["case_id"] for row in items if row["status"] == "passing"
            }),
        }

    return {
        "dataset_cases": len(dataset_ids),
        "eligible_cases": len(eligible_ids),
        "excluded_cases": sorted(dataset_ids - eligible_ids),
        "execution_portfolio_selection_mismatches": mismatches,
        "one_run_per_case": summarize(one_run_rows),
        "latest_run_per_case": summarize(latest_rows),
        "latest_not_officially_evaluated": [
            f"{row['case_id']}:{row['hypothesis_id']}@{row['run']}"
            for row in latest_rows if row["status"] == "not_officially_evaluated"
        ],
        "all_historical_attempts": summarize(rows),
        "by_run": {
            run: summarize([row for row in rows if row["run"] == run])
            for run, _, _ in RUNS
        },
        "rows": rows,
    }


if __name__ == "__main__":
    report = audit()
    del report["rows"]
    print(json.dumps(report, ensure_ascii=False, indent=2))
