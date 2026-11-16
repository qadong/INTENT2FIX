import json

from code_intent.code_intent.test_case_extractor import (
    collect_test_cases,
    extract_test_cases,
    is_repository_test_source,
)


def test_extract_test_cases_keeps_sections_as_distinct_cases():
    cases = extract_test_cases(
        "test/src/unit-access.cpp",
        r'''
TEST_CASE("element access", "[json]") {
  json value = json::object();
  SECTION("numeric key") {
    CHECK_THROWS_WITH(value[0], "numeric key rejected");
  }
  SECTION("string key") {
    CHECK(value["name"] == nullptr);
  }
}
''',
    )

    assert [case["fq_name"] for case in cases] == [
        "element access",
        "element access / numeric key",
        "element access / string key",
    ]
    assert cases[1]["parent_name"] == "element access"
    assert cases[1]["parent_test_id"] == cases[0]["id"]
    assert cases[1]["ancestor_test_ids"] == [cases[0]["id"]]
    assert "CHECK_THROWS_WITH" in cases[1]["oracle_code"]
    assert cases[1]["def_end_line"] < cases[0]["def_end_line"]


def test_extract_test_cases_preserves_nested_section_setup():
    cases = extract_test_cases(
        "tests/scenario.tests.cpp",
        r'''
SCENARIO("editing a document") {
  Document document("seed");
  GIVEN("an open document") {
    document.open();
    WHEN("text is appended") {
      document.append("value");
      THEN("the text is visible") { CHECK(document.text() == "seedvalue"); }
    }
  }
}
''',
    )

    then = next(case for case in cases if case["framework"] == "then")
    assert then["parent_name"].endswith("/ text is appended")
    assert "Document document" in then["inherited_setup"]
    assert "document.open" in then["inherited_setup"]


def test_extract_test_cases_preserves_setup_between_sibling_sections():
    cases = extract_test_cases(
        "tests/sibling_setup.tests.cpp",
        r'''
TEST_CASE("sequential setup") {
  prepare_common();
  SECTION("first") { CHECK(first_result()); }
  prepare_second();
  SECTION("second") {
    SECTION("nested") { CHECK(second_result()); }
  }
}
''',
    )

    second = next(case for case in cases if case["fq_name"].endswith("/ second"))
    nested = next(case for case in cases if case["fq_name"].endswith("/ nested"))
    assert "prepare_common" in second["inherited_setup"]
    assert "prepare_second" in second["inherited_setup"]
    assert 'SECTION("first")' not in second["inherited_setup"]
    assert "prepare_second" in nested["inherited_setup"]
    assert len(nested["ancestor_test_ids"]) == 2


def test_extract_test_cases_discovers_project_test_macro_wrappers():
    cases = extract_test_cases(
        "tests/project_wrapper_test.cpp",
        r'''
#define PROJECT_TEST(suite, name) TEST(suite, name)
#define PROJECT_CASE PROJECT_TEST
PROJECT_TEST(Parser, AcceptsText) { EXPECT_TRUE(parse_text()); }
PROJECT_CASE(Parser, RejectsBytes) { EXPECT_FALSE(parse_bytes()); }
''',
    )

    assert [case["fq_name"] for case in cases] == [
        "Parser.AcceptsText",
        "Parser.RejectsBytes",
    ]
    assert all(case["framework"] == "gtest" for case in cases)


def test_collect_test_cases_discovers_wrapper_from_project_test_header(tmp_path):
    repo = tmp_path / "repo"
    index = tmp_path / "index"
    (repo / "tests").mkdir(parents=True)
    (repo / "include" / "project").mkdir(parents=True)
    index.mkdir()
    (repo / "include" / "project" / "macros.hpp").write_text(
        """#ifndef PROJECT_TEST_H
#define PROJECT_TEST_H
#define PROJECT_TEST(suite, name) TEST(suite, name)
#endif
""",
        encoding="utf-8",
    )
    (repo / "tests" / "parser_test.cpp").write_text(
        '''#include "project/macros.hpp"
PROJECT_TEST(Parser, AcceptsText) { EXPECT_TRUE(parse_text()); }
''',
        encoding="utf-8",
    )

    cases = collect_test_cases(index, repo)

    assert [case["fq_name"] for case in cases] == ["Parser.AcceptsText"]


def test_extract_test_cases_respects_wrapper_definition_lifetime():
    cases = extract_test_cases(
        "tests/wrapper_lifetime_test.cpp",
        r'''
#if 0
#define DISABLED_TEST(suite, name) TEST(suite, name)
#endif
DISABLED_TEST(Parser, NotDefined) { EXPECT_TRUE(parse_text()); }
#define PROJECT_TEST(suite, name) TEST(suite, name)
PROJECT_TEST(Parser, BeforeUndef) { EXPECT_TRUE(parse_text()); }
#undef PROJECT_TEST
PROJECT_TEST(Parser, AfterUndef) { EXPECT_TRUE(parse_text()); }
#define CHANGING_TEST(suite, name) TEST(suite, name)
CHANGING_TEST(Parser, BeforeRedefinition) { EXPECT_TRUE(parse_text()); }
#define CHANGING_TEST(suite, name) NOT_A_TEST(suite, name)
CHANGING_TEST(Parser, AfterRedefinition) { EXPECT_TRUE(parse_text()); }
''',
    )

    assert [case["fq_name"] for case in cases] == [
        "Parser.BeforeUndef",
        "Parser.BeforeRedefinition",
    ]


def test_extract_test_cases_ignores_non_executable_macro_text():
    cases = extract_test_cases(
        "tests/macro.tests.cpp",
        r'''
// TEST_CASE("comment") { CHECK(false); }
const char* example = R"cpp(TEST_CASE("raw") { CHECK(false); })cpp";
#define WRAPPED_TEST TEST_CASE("definition") { CHECK(false); }
#if 0
TEST_CASE("disabled") { CHECK(false); }
#else
TEST_CASE("enabled fallback") { CHECK(true); }
#endif
#if 1
TEST_CASE("enabled primary") { CHECK(true); }
#else
TEST_CASE("disabled fallback") { CHECK(false); }
#endif
TEST_CASE("real") { CHECK(true); }
''',
    )

    assert [case["fq_name"] for case in cases] == [
        "enabled fallback",
        "enabled primary",
        "real",
    ]


def test_extract_test_cases_masks_short_circuited_preprocessor_branches():
    cases = extract_test_cases(
        "tests/preprocessor.tests.cpp",
        r'''
#if 0 && FEATURE_ENABLED
TEST_CASE("disabled conjunction") { CHECK(false); }
#endif
#if 1 || FEATURE_ENABLED
TEST_CASE("enabled disjunction") { CHECK(true); }
#endif
#if FEATURE_ENABLED
TEST_CASE("unknown configuration") { CHECK(true); }
#elif 0 && OTHER_FEATURE
TEST_CASE("unreachable elif") { CHECK(false); }
#endif
#if 0 \
    && MULTILINE_FEATURE
TEST_CASE("disabled multiline conjunction") { CHECK(false); }
#endif
''',
    )

    assert [case["fq_name"] for case in cases] == [
        "enabled disjunction",
        "unknown configuration",
    ]


def test_test_source_filter_excludes_vendored_framework_but_keeps_project_tests():
    assert not is_repository_test_source("test/gtest/gmock/gmock.h")
    assert not is_repository_test_source("test/gmock-gtest-all.cc")
    assert not is_repository_test_source("tests/third_party/catch/catch.hpp")
    assert not is_repository_test_source("googletest/test/gtest_output_test_.cc")
    assert not is_repository_test_source("libs/googletest/test/gtest_test.cc")
    assert not is_repository_test_source("build/_deps/catch2-src/tests/SelfTest.cpp")
    assert is_repository_test_source("tests/SelfTest/UsageTests/Misc.tests.cpp")
    assert is_repository_test_source("test/gtest-extra-test.cc")
    assert is_repository_test_source("tests/gtest/parser_test.cpp")


def test_collect_test_cases_enriches_direct_production_context(tmp_path):
    repo = tmp_path / "repo"
    index = tmp_path / "index"
    (repo / "tests").mkdir(parents=True)
    index.mkdir()
    (repo / "tests" / "widget_test.cpp").write_text(
        "TEST(Widget, ParsesText) { EXPECT_TRUE(Widget::parse(input)); }",
        encoding="utf-8",
    )
    functions = [
        {
            "id": "test-body",
            "fq_name": "Widget_ParsesText_Test::TestBody",
            "short_name": "TestBody",
            "def_file": "tests/widget_test.cpp",
            "def_start_line": 1,
            "def_end_line": 1,
            "is_test": True,
            "code": "Widget::parse(input);",
        },
        {
            "id": "widget-parse",
            "fq_name": "Widget::parse",
            "short_name": "parse",
            "def_file": "src/widget.cpp",
            "def_start_line": 10,
            "def_end_line": 20,
            "is_test": False,
            "code": "bool Widget::parse(Input input) { return input.valid(); }",
        },
    ]
    with (index / "functions.jsonl").open("w", encoding="utf-8") as stream:
        for record in functions:
            stream.write(json.dumps(record) + "\n")
    (index / "callgraph.jsonl").write_text(
        json.dumps({
            "caller": "test-body",
            "callee": "widget-parse",
            "edge_type": "call",
        }) + "\n",
        encoding="utf-8",
    )
    (index / "function_intents.jsonl").write_text(
        json.dumps({
            "func_id": "widget-parse",
            "short_intent": "Parse a Widget input.",
            "raw_summary": "Validates and parses the supplied input.",
        }) + "\n",
        encoding="utf-8",
    )

    cases = collect_test_cases(index, repo)

    assert len(cases) == 1
    assert cases[0]["source_func_ids"] == ["test-body"]
    assert cases[0]["focal_entities"][0]["name"] == "Widget::parse"
    assert cases[0]["focal_entities"][0]["relations"] == ["call"]
    assert "input.valid" in cases[0]["focal_entities"][0]["source"]


def test_collect_test_cases_does_not_choose_an_arbitrary_overload(tmp_path):
    repo = tmp_path / "repo"
    index = tmp_path / "index"
    (repo / "tests").mkdir(parents=True)
    index.mkdir()
    (repo / "tests" / "widget_test.cpp").write_text(
        "TEST(Widget, ParsesText) { EXPECT_TRUE(Widget::parse(input)); }",
        encoding="utf-8",
    )
    functions = [
        {
            "id": f"parse-{kind}",
            "fq_name": "Widget::parse",
            "short_name": "parse",
            "def_file": "src/widget.cpp",
            "def_start_line": line,
            "is_test": False,
            "code": f"bool Widget::parse({kind} input);",
        }
        for kind, line in (("Text", 10), ("Bytes", 20))
    ]
    with (index / "functions.jsonl").open("w", encoding="utf-8") as stream:
        for record in functions:
            stream.write(json.dumps(record) + "\n")

    cases = collect_test_cases(index, repo)

    assert cases[0]["focal_entities"] == []


def test_collect_test_cases_ignores_source_references_in_comments_and_strings(tmp_path):
    repo = tmp_path / "repo"
    index = tmp_path / "index"
    (repo / "tests").mkdir(parents=True)
    index.mkdir()
    (repo / "tests" / "widget_test.cpp").write_text(
        '''TEST(Widget, DescriptionOnly) {
  // Widget::parse(input);
  const char* note = "Widget::parse(input)";
  EXPECT_TRUE(true);
}''',
        encoding="utf-8",
    )
    (index / "functions.jsonl").write_text(
        json.dumps({
            "id": "widget-parse",
            "fq_name": "Widget::parse",
            "short_name": "parse",
            "def_file": "src/widget.cpp",
            "def_start_line": 10,
            "is_test": False,
            "code": "bool Widget::parse(Input input);",
        }) + "\n",
        encoding="utf-8",
    )

    cases = collect_test_cases(index, repo)

    assert cases[0]["focal_entities"] == []


def test_collect_test_cases_uses_libclang_boundary_for_function_tests(tmp_path):
    repo = tmp_path / "repo"
    index = tmp_path / "index"
    (repo / "tests").mkdir(parents=True)
    index.mkdir()
    source = "void parser_contract() { CHECK(parse(input)); }"
    (repo / "tests" / "parser_checks.cpp").write_text(source, encoding="utf-8")
    (index / "functions.jsonl").write_text(
        json.dumps({
            "id": "parser-contract",
            "fq_name": "parser_contract",
            "short_name": "parser_contract",
            "def_file": "tests/parser_checks.cpp",
            "def_start_line": 1,
            "def_end_line": 1,
            "is_test": True,
            "code": source,
        }) + "\n",
        encoding="utf-8",
    )

    cases = collect_test_cases(index, repo)

    assert len(cases) == 1
    assert cases[0]["framework"] == "libclang-function"
    assert cases[0]["source_func_ids"] == ["parser-contract"]


def test_collect_test_cases_uses_columns_for_same_line_macro_bindings(tmp_path):
    repo = tmp_path / "repo"
    index = tmp_path / "index"
    (repo / "tests").mkdir(parents=True)
    index.mkdir()
    source = "TEST(A, One) {} TEST(B, Two) {}"
    (repo / "tests" / "same_line_test.cpp").write_text(source, encoding="utf-8")
    functions = [
        {
            "id": "first-body",
            "fq_name": "TestBody",
            "def_file": "tests/same_line_test.cpp",
            "def_start_line": 1,
            "def_end_line": 1,
            "def_start_column": 1,
            "is_test": True,
        },
        {
            "id": "second-body",
            "fq_name": "TestBody",
            "def_file": "tests/same_line_test.cpp",
            "def_start_line": 1,
            "def_end_line": 1,
            "def_start_column": source.index("TEST(B") + 1,
            "is_test": True,
        },
    ]
    with (index / "functions.jsonl").open("w", encoding="utf-8") as stream:
        for record in functions:
            stream.write(json.dumps(record) + "\n")

    cases = collect_test_cases(index, repo)

    assert [case["source_func_ids"] for case in cases] == [
        ["first-body"],
        ["second-body"],
    ]


def test_collect_test_cases_rejects_ambiguous_same_line_macro_bindings(tmp_path):
    repo = tmp_path / "repo"
    index = tmp_path / "index"
    (repo / "tests").mkdir(parents=True)
    index.mkdir()
    source = "TEST(A, One) {} TEST(B, Two) {}"
    (repo / "tests" / "same_line_test.cpp").write_text(source, encoding="utf-8")
    (index / "functions.jsonl").write_text(
        json.dumps({
            "id": "ambiguous-body",
            "fq_name": "TestBody",
            "def_file": "tests/same_line_test.cpp",
            "def_start_line": 1,
            "def_end_line": 1,
            "is_test": True,
        }) + "\n",
        encoding="utf-8",
    )

    cases = collect_test_cases(index, repo)

    assert [case["source_func_ids"] for case in cases] == [[], []]
