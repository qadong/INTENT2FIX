from INTENT2FIX.intent_dialogue.developer_agent import (
    InvestigationSession,
    ListRepositoryTests,
    ListRepositoryTestsArgs,
)
from INTENT2FIX.intent_dialogue.test_inventory import RepositoryTestInventory
from INTENT2FIX.repository.repository import InMemRepository


def test_inventory_extracts_common_cpp_test_shapes_and_descriptions():
    repository = InMemRepository(files={
        "tests/access_tests.cpp": """
// Numeric object indexing must report the rejected key category.
TEST_CASE("numeric object indexing", "[json][access]") {}

TEST(Parser, RejectsInvalidUtf8) {}

bool json_array_with_array_count() { return true; }
""",
        "tests/singleheadertest.cpp": "int main() { return 0; }",
        "src/catch_test_spec_parser.cpp": (
            'TEST_CASE("production false positive", "[internal]") {}'
        ),
        "test/thirdparty/dependency_tests.cpp": (
            'TEST_CASE("dependency test", "[dependency]") {}'
        ),
    })

    entries = RepositoryTestInventory(repository).entries

    assert [entry.name for entry in entries] == [
        "numeric object indexing",
        "Parser.RejectsInvalidUtf8",
        "json_array_with_array_count",
        "singleheadertest",
    ]
    assert entries[0].framework == "test_case"
    assert "rejected key category" in entries[0].description
    assert entries[1].framework == "gtest"
    assert entries[2].framework == "bool-function"
    assert entries[3].framework == "standalone"


def test_inventory_regex_search_prioritizes_names_and_respects_path_glob():
    repository = InMemRepository(files={
        "tests/ondemand/object_tests.cpp": """
bool json_object_iteration() { return true; }
bool json_array_with_array_count() { return true; }
""",
        "tests/dom/object_tests.cpp": (
            "bool json_object_conversion() { return true; }"
        ),
    })
    inventory = RepositoryTestInventory(repository)

    matches, total = inventory.search(
        r"array.*count|count_elements",
        path_pattern="tests/ondemand/*",
    )

    assert total == 1
    assert matches[0].name == "json_array_with_array_count"


def test_list_repository_tests_action_returns_bounded_search_evidence():
    repository = InMemRepository(files={
        "tests/widget_tests.cpp": """
TEST_CASE("widget accepts text", "[widget]") {}
TEST_CASE("widget rejects binary", "[widget]") {}
""",
    })
    session = InvestigationSession()
    action = ListRepositoryTests(repository=repository, session=session)

    observation = action.execute(ListRepositoryTestsArgs(
        pattern=r"widget.*text|accepts",
        path_pattern="tests/*",
        limit=10,
        purpose="Find the closest passing text-input test.",
    ))

    assert observation.expect_correction is False
    assert observation.properties["evidence_id"] == "SEARCH-1"
    assert observation.properties["total_matches"] == 1
    assert observation.properties["tests"][0]["name"] == "widget accepts text"
    assert "tests/widget_tests.cpp" in observation.message
    assert session.actions[0]["action"] == "ListRepositoryTests"

    duplicate = action.execute(ListRepositoryTestsArgs(
        pattern=r"widget.*text|accepts",
        path_pattern="tests/*",
        limit=10,
    ))
    assert duplicate.expect_correction is True
    assert "duplicates SEARCH-1" in duplicate.message


def test_list_repository_tests_rejects_invalid_regex():
    action = ListRepositoryTests(
        repository=InMemRepository(files={}),
        session=InvestigationSession(),
    )

    observation = action.execute(ListRepositoryTestsArgs(pattern="["))

    assert observation.expect_correction is True
    assert "Invalid test-name regular expression" in observation.message
