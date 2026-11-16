import sys
import json

from code_intent import batch_intent


def test_request_slots_respect_global_budget_across_project_lanes():
    assert batch_intent._allocate_request_slots(5, 4) == [2, 1, 1, 1]
    assert batch_intent._allocate_request_slots(2, 4) == [1, 1, 1, 1]
    assert batch_intent._allocate_request_slots(5, 0) == []


def test_request_slots_honor_per_case_cap():
    assert batch_intent._allocate_request_slots(30, 3, 10) == [10, 10, 10]
    assert batch_intent._allocate_request_slots(30, 2, 10) == [10, 10]
    assert batch_intent._allocate_request_slots(12, 3, 10) == [4, 4, 4]


def test_sort_instances_can_process_each_repository_newest_first():
    instances = [
        "swe-bench_fmtlib__fmt-100",
        "swe-bench_nlohmann__json-20",
        "swe-bench_fmtlib__fmt-300",
        "swe-bench_nlohmann__json-40",
    ]

    assert batch_intent._sort_instances(instances, "newest-first") == [
        "swe-bench_fmtlib__fmt-300",
        "swe-bench_fmtlib__fmt-100",
        "swe-bench_nlohmann__json-40",
        "swe-bench_nlohmann__json-20",
    ]
    assert batch_intent._sort_instances(instances, "oldest-first") == [
        "swe-bench_fmtlib__fmt-100",
        "swe-bench_fmtlib__fmt-300",
        "swe-bench_nlohmann__json-20",
        "swe-bench_nlohmann__json-40",
    ]


def test_instance_file_accepts_benchmark_jsonl_and_plain_ids(tmp_path):
    path = tmp_path / "instances.jsonl"
    path.write_text(
        json.dumps({"instance_id": "fmtlib__fmt-3158"})
        + "\n"
        + "swe-bench_nlohmann__json-3463\n",
        encoding="utf-8",
    )

    assert batch_intent._load_instance_ids(path) == {
        "swe-bench_fmtlib__fmt-3158",
        "swe-bench_nlohmann__json-3463",
    }


def test_repository_path_accepts_unprefixed_checkout_names(tmp_path):
    checkout = tmp_path / "fmtlib__fmt-3158"
    checkout.mkdir()

    assert batch_intent._repository_path(
        tmp_path, "swe-bench_fmtlib__fmt-3158"
    ) == checkout


def test_child_uses_current_interpreter_and_assigned_request_slots(
    tmp_path, monkeypatch
):
    calls = []

    def check_call(command, env):
        calls.append((command, env))

    monkeypatch.setattr(batch_intent.subprocess, "check_call", check_call)
    monkeypatch.setattr(batch_intent, "EXAMPLE_SCRIPT", "/workspace/example.py")

    batch_intent.run_intent_generation(
        "swe-bench_fmtlib__fmt-3158",
        "/workspace/repo",
        str(tmp_path),
        "functions",
        request_concurrency=2,
    )

    assert len(calls) == 1
    command, environment = calls[0]
    assert command[0] == sys.executable
    assert environment["INTENT_CONCURRENCY"] == "2"
    assert environment["INTENT_CACHE_DIR"] == str(tmp_path / ".intent_cache")
