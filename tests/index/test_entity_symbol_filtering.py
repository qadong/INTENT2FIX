import json

from INTENT2FIX.index.entity_code_index import EntityCodeIndex
from code_intent.code_intent.entity_tools import (
    _filter_entities,
    _rank_and_diversify_entities,
)


def _entity(namespace, file_path, line, *, entity_id=None):
    owner = namespace[-1]
    return {
        "id": entity_id or f"{owner}-{line}",
        "fq_name": "::".join([*namespace, owner, "at_pointer"]),
        "short_name": "at_pointer",
        "namespaces": namespace,
        "classes": [owner],
        "def_file": file_path,
        "def_start_line": line,
        "def_end_line": line + 5,
        "parameter_types": ["std::string_view"],
    }


def test_symbol_file_filter_supports_globs():
    entities = [
        _entity(["simdjson", "dom"], "include/simdjson/dom/object-inl.h", 10),
        _entity(
            ["simdjson", "fallback", "ondemand"],
            "include/simdjson/generic/ondemand/object-inl.h",
            20,
        ),
    ]

    filtered = _filter_entities(
        entities,
        "function",
        name="at_pointer",
        file="include/simdjson/generic/ondemand/*",
    )

    assert [entity["def_start_line"] for entity in filtered] == [20]


def test_symbol_namespace_matches_across_macro_expansion_component():
    entity = _entity(
        ["simdjson", "fallback", "ondemand"],
        "include/simdjson/generic/ondemand/object-inl.h",
        20,
    )

    filtered = _filter_entities(
        [entity],
        "function",
        name="at_pointer",
        namespace="simdjson::ondemand",
    )

    assert filtered == [entity]


def test_symbol_candidates_deduplicate_and_cover_independent_scopes_early():
    entities = []
    for line in range(1, 13):
        dom = _entity(
            ["simdjson", "dom"],
            f"include/simdjson/dom/type-{line}.h",
            line,
            entity_id=f"dom-{line}-a",
        )
        duplicate = dict(dom, id=f"dom-{line}-b")
        entities.extend((dom, duplicate))
    entities.extend(
        _entity(
            ["simdjson", "fallback", "ondemand"],
            f"include/simdjson/generic/ondemand/type-{line}.h",
            100 + line,
        )
        for line in range(3)
    )

    ranked = _rank_and_diversify_entities(entities, "at_pointer")

    assert len(ranked) == 15
    assert any("ondemand" in entity["namespaces"] for entity in ranked[:10])
    identities = {
        (entity["fq_name"], entity["def_file"], entity["def_start_line"])
        for entity in ranked
    }
    assert len(identities) == len(ranked)


def test_explicit_index_root_does_not_fall_back_to_process_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("CODE_INTENT_INDEX_STRICT_PREFLIGHT", "0")
    instance = "org__repo-1"
    stale = tmp_path / "stale"
    selected = tmp_path / "selected"
    for root, symbol in ((stale, "stale_symbol"), (selected, "selected_symbol")):
        entities = root / f"swe-bench_{instance}" / "index_output" / "entities"
        entities.mkdir(parents=True)
        row = _entity(["project"], "src/file.cpp", 10)
        row["short_name"] = symbol
        row["fq_name"] = f"project::{symbol}"
        (entities / "functions.jsonl").write_text(json.dumps(row) + "\n")
        (entities / "function_templates.jsonl").write_text("")
    monkeypatch.setenv("INDEX_STORE_DIR", str(stale))

    index = EntityCodeIndex(index_name=instance, index_store_dir=str(selected))
    response = index.find_function("selected_symbol", name_exact_match=True)

    assert len(response.hits) == 1
    assert response.hits[0].spans[0].label == "project::selected_symbol"
