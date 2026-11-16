import json

from code_intent.code_intent import entity_tools
from code_intent.code_intent.graph_builder import build_graph


def _write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_build_graph_keeps_class_references_from_functions_without_intents(tmp_path):
    caller_id = "function:caller"
    no_intent_id = "function:no-intent"
    class_id = "class:Column"
    _write_jsonl(
        tmp_path / "functions.jsonl",
        [
            {
                "id": caller_id,
                "fq_name": "useColumn",
                "short_name": "useColumn",
                "def_file": "src/use.cpp",
                "def_start_line": 10,
                "def_end_line": 12,
            },
            {
                "id": no_intent_id,
                "fq_name": "useColumnWithoutIntent",
                "short_name": "useColumnWithoutIntent",
                "def_file": "src/no_intent.cpp",
                "def_start_line": 20,
                "def_end_line": 22,
            },
        ],
    )
    _write_jsonl(
        tmp_path / "classes.jsonl",
        [{
            "id": class_id,
            "fq_name": "Catch::TextFlow::Column",
            "short_name": "Column",
            "def_file": "src/column.hpp",
            "def_start_line": 5,
        }],
    )
    _write_jsonl(
        tmp_path / "function_intents.jsonl",
        [{
            "func_id": caller_id,
            "file": "/stale/build/src/use.cpp",
            "start_line": 999,
            "short_intent": "Use a column.",
        }],
    )
    _write_jsonl(
        tmp_path / "callgraph.jsonl",
        [{"caller": no_intent_id, "callee": class_id, "edge_type": "type_ref"}],
    )

    context = build_graph(tmp_path)

    assert caller_id in context.functions
    assert context.functions[caller_id]["fq_name"] == "useColumn"
    assert context.functions[caller_id]["file"] == "src/use.cpp"
    assert context.functions[caller_id]["start_line"] == 10
    assert context.functions[caller_id]["short_intent"] == "Use a column."
    assert no_intent_id in context.functions
    assert context.graph.neighbors(
        class_id, "DEP_TYPE_REF", direction="in"
    ) == {no_intent_id}


def test_find_usages_maps_constructor_to_owning_class(monkeypatch):
    constructor = {
        "id": "function:Column::Column",
        "fq_name": "Catch::TextFlow::Column::Column",
        "name": "Column",
        "def_file": "src/column.hpp",
        "def_start_line": 10,
    }
    column_class = {
        "id": "class:Column",
        "fq_name": "Catch::TextFlow::Column",
        "short_name": "Column",
        "def_file": "src/column.hpp",
        "def_start_line": 5,
    }
    calls = []

    def fake_search_symbol(type, **kwargs):
        if type == "function":
            return {"entities": [constructor]}
        if type == "class":
            return {"entities": [column_class]}
        return {"entities": []}

    def fake_get_entity_neighbors(entity_id, **kwargs):
        calls.append((entity_id, kwargs))
        if entity_id == column_class["id"]:
            return {
                "neighbors": [{
                    "id": "function:caller",
                    "type": "FUNC",
                    "fq_name": "useColumn",
                    "file": "src/use.cpp",
                    "line": 20,
                }]
            }
        return {"neighbors": []}

    monkeypatch.setattr(entity_tools, "search_symbol", fake_search_symbol)
    monkeypatch.setattr(
        entity_tools, "get_entity_neighbors", fake_get_entity_neighbors
    )

    result = entity_tools.find_usages(
        type="function",
        name="Column",
        namespace="Catch::TextFlow",
    )

    assert [usage["id"] for usage in result["usages"]] == ["function:caller"]
    assert [call[0] for call in calls] == [constructor["id"], column_class["id"]]
    assert "IN_CLASS" not in calls[0][1]["edge_types"]
    assert "DEP_TYPE_REF" in calls[0][1]["edge_types"]
