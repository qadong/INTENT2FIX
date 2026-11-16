from INTENT2FIX.actions.create_file import CreateFileArgs
from INTENT2FIX.actions.append_string import AppendStringArgs
from INTENT2FIX.actions.entity_function_semantic_search import EntityFunctionSemanticSearchArgs
from INTENT2FIX.actions.entity_intent_search import EntityIntentSearchArgs
from INTENT2FIX.actions.entity_search_symbol import EntitySearchSymbolArgs
from INTENT2FIX.actions.find_entity_usage import FindEntityUsage, FindEntityUsageArgs
from INTENT2FIX.actions.find_template_instantiation import FindTemplateInstantiationArgs
from INTENT2FIX.actions.get_macro_usages import GetMacroUsages, GetMacroUsagesArgs
from INTENT2FIX.actions.run_command import RunCommandArgs
from INTENT2FIX.actions.run_intent_check import RunIntentCheckArgs
from INTENT2FIX.actions.run_regression import RunRegressionArgs
from INTENT2FIX.actions.run_single_test import RunSingleTestArgs
from INTENT2FIX.actions.string_replace import StringReplaceArgs
from INTENT2FIX.actions.verified_finish import VerifiedFinishArgs
from INTENT2FIX.actions.view_code import ViewCodeArgs
from INTENT2FIX.completion import LLMResponseFormat
from INTENT2FIX.completion.base import BaseCompletionModel
from INTENT2FIX.index import entity_code_index as entity_code_index_module
from INTENT2FIX.index.entity_code_index import EntityCodeIndex
from INTENT2FIX.repository.repository import InMemRepository


def _function_schema(args_class):
    return args_class.tool_schema(thoughts_in_action=False)["function"]


def _completion_model():
    return BaseCompletionModel.create(
        response_format=LLMResponseFormat.TOOLS,
        model="gpt-3.5-turbo",
        max_tokens=1000,
        temperature=0,
        model_api_key="dummy",
    )


def test_native_tool_schema_keeps_defaulted_fields_optional():
    expected_required = {
        EntityFunctionSemanticSearchArgs: {"query"},
        EntityIntentSearchArgs: {"query"},
        EntitySearchSymbolArgs: {"type", "name"},
        GetMacroUsagesArgs: {"name"},
        FindTemplateInstantiationArgs: set(),
        FindEntityUsageArgs: {"type", "name"},
        ViewCodeArgs: {"files"},
        StringReplaceArgs: {"path", "old_str", "new_str"},
        CreateFileArgs: {"path", "file_text"},
        AppendStringArgs: {"path", "new_str"},
        RunCommandArgs: {"command"},
        RunRegressionArgs: set(),
        RunSingleTestArgs: {"command"},
        RunIntentCheckArgs: {"check_id"},
        VerifiedFinishArgs: {"finish_reason", "test_verification"},
    }

    for args_class, expected in expected_required.items():
        required = set(_function_schema(args_class)["parameters"]["required"])
        assert required == expected, args_class.__name__

    symbol_type = _function_schema(EntitySearchSymbolArgs)["parameters"]["properties"]["type"]
    assert symbol_type["enum"] == ["function", "class", "namespace", "macro"]
    assert EntitySearchSymbolArgs(type="function_template", name="format").type == "function"


def test_native_tool_schema_describes_every_exposed_argument():
    args_classes = [
        EntityFunctionSemanticSearchArgs,
        EntityIntentSearchArgs,
        EntitySearchSymbolArgs,
        GetMacroUsagesArgs,
        FindTemplateInstantiationArgs,
        FindEntityUsageArgs,
        ViewCodeArgs,
        StringReplaceArgs,
        CreateFileArgs,
        AppendStringArgs,
        RunCommandArgs,
        RunRegressionArgs,
        RunSingleTestArgs,
        RunIntentCheckArgs,
        VerifiedFinishArgs,
    ]

    for args_class in args_classes:
        properties = _function_schema(args_class)["parameters"]["properties"]
        missing = [name for name, value in properties.items() if not value.get("description")]
        assert not missing, f"{args_class.__name__}: {missing}"


def test_runtime_and_view_tools_are_self_describing():
    run_command = _function_schema(RunCommandArgs)
    assert "read-only repository search" in run_command["description"]
    assert "current repository directory" in run_command["parameters"]["properties"]["command"]["description"]
    capture_description = run_command["parameters"]["properties"][
        "capture_generated_changes"
    ]["description"]
    assert "repository-owned artifact generator" in capture_description
    assert "candidate-created transformation script" in capture_description
    assert "tracked generated text files" in capture_description
    assert "imported into the patch" in capture_description

    run_single = _function_schema(RunSingleTestArgs)
    assert "Preserve the real exit status" in run_single["description"]
    assert "must not generate, synchronize, or edit tracked" in (
        run_single["parameters"]["properties"]["command"]["description"]
    )

    intent_check = _function_schema(RunIntentCheckArgs)
    assert "immutable ACCEPT-N probe" in intent_check["description"]
    assert "Exact executable ACCEPT-N ID" in intent_check["parameters"]["properties"]["check_id"]["description"]

    view_code = _function_schema(ViewCodeArgs)
    span_description = (
        view_code["parameters"]["properties"]["files"]["items"]
        ["properties"]["span_ids"]["description"]
    )
    assert "copied exactly from search observations" in span_description
    assert "Do not construct or guess" in span_description


def test_entity_code_index_forwards_function_filters(monkeypatch):
    calls = []

    def fake_search_symbol(**kwargs):
        calls.append(kwargs)
        return {
            "entities": [
                {
                    "id": "function:Widget::format",
                    "fq_name": "Widget::format",
                    "file": "include/widget.hpp",
                    "line": 10,
                }
            ]
        }

    monkeypatch.setattr(entity_code_index_module, "search_symbol", fake_search_symbol)
    index = EntityCodeIndex(file_repo=None, index_name=None)

    result = index.find_function(
        "format",
        class_name="Widget",
        file_pattern="include/**",
        namespace="demo",
        name_exact_match=True,
        return_type="void",
        parameter_types=["int"],
        limit=7,
    )

    assert result.hits
    assert {call["type"] for call in calls} == {"function", "function_template"}
    assert all(call["name_exact_match"] is True for call in calls)
    assert all(call["return_type"] == "void" for call in calls)
    assert all(call["parameter_types"] == ["int"] for call in calls)
    assert all(call["limit"] == 7 for call in calls)


def test_function_signature_filter_does_not_fall_back_to_whole_class(monkeypatch):
    monkeypatch.setattr(
        entity_code_index_module,
        "search_symbol",
        lambda **kwargs: {"entities": []},
    )
    index = EntityCodeIndex(file_repo=None, index_name=None)
    fallback_calls = []

    def fake_find_class(**kwargs):
        fallback_calls.append(kwargs)
        raise AssertionError("signature-constrained searches must not return a class hit")

    monkeypatch.setattr(index, "find_class", fake_find_class)

    result = index.find_function(
        "format",
        class_name="Widget",
        return_type="void",
        parameter_types=["int"],
    )

    assert result.hits == []
    assert fallback_calls == []


def test_exact_function_search_does_not_fall_back_to_whole_class(monkeypatch):
    monkeypatch.setattr(
        entity_code_index_module,
        "search_symbol",
        lambda **kwargs: {"entities": []},
    )
    index = EntityCodeIndex(file_repo=None, index_name=None)
    fallback_calls = []

    def fake_find_class(**kwargs):
        fallback_calls.append(kwargs)
        raise AssertionError("exact function searches must not return a class hit")

    monkeypatch.setattr(index, "find_class", fake_find_class)

    result = index.find_function(
        "format",
        class_name="Widget",
        name_exact_match=True,
    )

    assert result.hits == []
    assert fallback_calls == []


def test_macro_and_usage_file_patterns_filter_graph_results(monkeypatch):
    from code_intent.code_intent import entity_tools

    monkeypatch.setattr(
        entity_tools,
        "get_macro_usages",
        lambda name: {
            "definitions": [
                {"file": "include/api.hpp", "line": 1},
                {"file": "src/api.cpp", "line": 2},
            ],
            "usages": [
                {"file": "include/use.hpp", "line": 3, "fq_name": "use_header"},
                {"file": "src/use.cpp", "line": 4, "fq_name": "use_source"},
            ],
        },
    )
    monkeypatch.setattr(
        entity_tools,
        "find_usages",
        lambda **kwargs: {
            "target": {
                "id": "function:target",
                "fq_name": "target",
                "file": "src/target.cpp",
                "line": 1,
            },
            "usages": [
                {"id": "u1", "fq_name": "header_user", "file": "include/u.hpp", "line": 2},
                {"id": "u2", "fq_name": "source_user", "file": "src/u.cpp", "line": 3},
            ],
        },
    )

    repository = InMemRepository()
    model = _completion_model()
    macro_action = GetMacroUsages(
        repository=repository,
        code_index=None,
        completion_model=model.clone(),
    )
    usage_action = FindEntityUsage(
        repository=repository,
        code_index=None,
        completion_model=model.clone(),
    )

    macro_result = macro_action._search(
        GetMacroUsagesArgs(name="API_MACRO", file_pattern="include/**")
    )
    usage_result = usage_action._search(
        FindEntityUsageArgs(
            type="function",
            name="target",
            file_pattern="include/**",
        )
    )

    assert "1 definition(s), 1 usage(s)" in macro_result.message
    assert {hit.file_path for hit in macro_result.hits} == {
        "include/api.hpp",
        "include/use.hpp",
    }
    assert {hit.file_path for hit in usage_result.hits} == {"include/u.hpp"}
