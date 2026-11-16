import json

from scripts.audit_scout_edit_exposure import (
    baseline_hunks,
    graph_lines,
    source_lines,
    validated_targets,
)


def test_preimage_and_insertion_anchor_are_distinct() -> None:
    patch = """diff --git a/src/a.cc b/src/a.cc
--- a/src/a.cc
+++ b/src/a.cc
@@ -1,2 +1,2 @@
-old
+new
 keep
@@ -4,2 +4,3 @@
 before
+added
 after
"""
    hunks, unmappable = baseline_hunks(patch)
    assert unmappable == 0
    assert hunks[0]["targets"] == {("src/a.cc", 1, "old")}
    assert hunks[1]["targets"] == {
        ("src/a.cc", 4, "before"),
        ("src/a.cc", 5, "after"),
    }


def test_patch_line_offsets_align_to_actual_baseline() -> None:
    hunk = {
        "path": "src/a.cc", "old_start": 10,
        "old_lines": ["old", "keep"],
        "targets": {("src/a.cc", 10, "old")},
    }
    assert validated_targets(hunk, ["prefix", "old", "keep"]) == {
        ("src/a.cc", 2, "old")
    }


def test_numbered_graph_excerpt_counts_but_path_only_does_not() -> None:
    observation = json.dumps([
        {"file": "src/a.cc", "line_range": [1, 2], "source_status": "verified"},
        {"file": "src/a.cc", "source_status": "verified", "source_excerpt": "1: old\n2: keep"},
    ])
    assert graph_lines(observation, {"src/a.cc"}, mode="function") == {
        ("src/a.cc", 1, "old"),
        ("src/a.cc", 2, "keep"),
    }


def test_graph_compact_view_does_not_count_hidden_raw_excerpt_lines() -> None:
    observation = json.dumps([{
        "file": "src/a.cc", "source_status": "verified",
        "source_excerpt": "\n".join(f"{n}: line {n}" for n in range(1, 21)),
    }])
    compact = graph_lines(observation, {"src/a.cc"}, mode="trace")
    assert ("src/a.cc", 5, "line 5") in compact
    assert ("src/a.cc", 6, "line 6") not in compact
    assert ("src/a.cc", 20, "line 20") in graph_lines(
        observation, {"src/a.cc"}, mode="symbol"
    )


def test_sed_multiple_ranges_and_omitted_middle_keep_line_numbers() -> None:
    paths = {"src/a.cc"}
    assert source_lines("sed -n '10,12p;20,22p' src/a.cc", "a\nb\nc\nd\ne\nf", paths) == {
        ("src/a.cc", 10, "a"),
        ("src/a.cc", 11, "b"),
        ("src/a.cc", 12, "c"),
        ("src/a.cc", 20, "d"),
        ("src/a.cc", 21, "e"),
        ("src/a.cc", 22, "f"),
    }
    assert source_lines(
        "sed -n '10,15p' src/a.cc", "a\n... [4 lines omitted] ...\nf", paths
    ) == {("src/a.cc", 10, "a"), ("src/a.cc", 15, "f")}


def test_chained_sed_views_are_attributed_to_their_own_files() -> None:
    assert source_lines(
        "sed -n '10,11p' src/a.cc && sed -n '20,21p' src/b.cc",
        "a\nb\nc\nd",
        {"src/a.cc", "src/b.cc"},
    ) == {
        ("src/a.cc", 10, "a"), ("src/a.cc", 11, "b"),
        ("src/b.cc", 20, "c"), ("src/b.cc", 21, "d"),
    }


def test_grep_requires_returned_line_not_just_queried_path() -> None:
    paths = {"src/a.cc"}
    assert source_lines("grep -n old src/a.cc", "1:old", paths) == {
        ("src/a.cc", 1, "old")
    }
    assert source_lines("grep -n old src/a.cc", "", paths) == set()
