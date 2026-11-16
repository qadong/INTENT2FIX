import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "evaluate_fix_patches.py"
SPEC = importlib.util.spec_from_file_location("evaluate_fix_patches", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_image_for_instance_uses_multiswebench_convention():
    assert MODULE.image_for_instance(
        "nlohmann__json-609", "registry.example"
    ) == "registry.example/mswebench/nlohmann_m_json:pr-609"


def test_failed_test_detection_ignores_zero_failure_summary():
    assert not MODULE.log_has_failed_tests(
        "100% tests passed, 0 tests failed out of 10"
    )
    assert MODULE.log_has_failed_tests(
        "80% tests passed, 2 tests failed out of 10"
    )
    assert MODULE.failed_tests_from_log(
        "1/2 Test #2: unit-test ........***Failed\n"
    ) == ["unit-test"]


def test_filter_model_patch_removes_tests_and_repro_but_keeps_build_files():
    patch = """--- a/src/library.cpp
+++ b/src/library.cpp
@@ -1 +1 @@
-old
+new
--- a/test/unit.cpp
+++ b/test/unit.cpp
@@ -1 +1 @@
-old test
+new test
--- /dev/null
+++ b/repro_case.cpp
@@ -0,0 +1 @@
+repro
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -1 +1 @@
-old cmake
+new cmake
"""

    filtered, kept, excluded = MODULE.filter_model_patch(patch)

    assert kept == ["src/library.cpp", "CMakeLists.txt"]
    assert excluded == [
        "test/unit.cpp",
        "repro_case.cpp",
    ]
    assert "src/library.cpp" in filtered
    assert "CMakeLists.txt" in filtered
    assert "test/unit.cpp" not in filtered


def test_filter_model_patch_handles_test_naming_without_substring_false_positive():
    patch = """--- a/projects/SelfTest/UsageTests/Benchmark.tests.cpp
+++ b/projects/SelfTest/UsageTests/Benchmark.tests.cpp
@@ -1 +1 @@
-old test
+new test
--- a/src/framework/internal/framework_test_case_info_hasher.cpp
+++ b/src/framework/internal/framework_test_case_info_hasher.cpp
@@ -1 +1 @@
-old product code
+new product code
--- a/src/latest/parser.cpp
+++ b/src/latest/parser.cpp
@@ -1 +1 @@
-old parser
+new parser
--- a/tools/parser_test.cpp
+++ b/tools/parser_test.cpp
@@ -1 +1 @@
-old tool test
+new tool test
"""

    filtered, kept, excluded = MODULE.filter_model_patch(patch)

    assert kept == [
        "src/framework/internal/framework_test_case_info_hasher.cpp",
        "src/latest/parser.cpp",
    ]
    assert excluded == [
        "projects/SelfTest/UsageTests/Benchmark.tests.cpp",
        "tools/parser_test.cpp",
    ]
    assert "Benchmark.tests.cpp" not in filtered
    assert "framework_test_case_info_hasher.cpp" in filtered


def test_filter_model_patch_keeps_product_names_containing_repro_substring():
    patch = """--- a/include/internal/catch_preprocessor.hpp
+++ b/include/internal/catch_preprocessor.hpp
@@ -1 +1 @@
-old preprocessor
+new preprocessor
--- a/src/apocalypse.cpp
+++ b/src/apocalypse.cpp
@@ -1 +1 @@
-old product code
+new product code
--- /dev/null
+++ b/tools/bug_reproducer.cpp
@@ -0,0 +1 @@
+reproducer
"""

    filtered, kept, excluded = MODULE.filter_model_patch(patch)

    assert kept == [
        "include/internal/catch_preprocessor.hpp",
        "src/apocalypse.cpp",
    ]
    assert excluded == ["tools/bug_reproducer.cpp"]
    assert "catch_preprocessor.hpp" in filtered
    assert "apocalypse.cpp" in filtered


def test_deduplicate_test_patch_hunks_removes_only_identical_hunk():
    model_patch = """diff --git a/src/api.hpp b/src/api.hpp
--- a/src/api.hpp
+++ b/src/api.hpp
@@ -1,3 +1,3 @@
-old macro
+new macro
 keep
 extra context
@@ -10 +10 @@
-old implementation
+model implementation
"""
    test_patch = """diff --git a/src/api.hpp b/src/api.hpp
--- a/src/api.hpp
+++ b/src/api.hpp
@@ -1,2 +1,2 @@
-old macro
+new macro
 keep
@@ -10 +10 @@
-old implementation
+official implementation
"""

    deduplicated, removed = MODULE.deduplicate_test_patch_hunks(
        model_patch,
        test_patch,
    )

    assert removed == 1
    assert "new macro" not in deduplicated
    assert "model implementation" in deduplicated
    assert "official implementation" not in deduplicated


def test_remove_model_paths_from_test_patch_keeps_candidate_authoritative():
    model_patch = """--- a/src/api.hpp
+++ b/src/api.hpp
@@ -1 +1 @@
-old implementation
+alternative implementation
"""
    test_patch = """--- a/src/api.hpp
+++ b/src/api.hpp
@@ -1 +1 @@
-old implementation
+official implementation
--- a/tests/api_test.cpp
+++ b/tests/api_test.cpp
@@ -1 +1 @@
-old test
+new test
"""

    sanitized, removed_hunks, removed_files = (
        MODULE.remove_model_paths_from_test_patch(model_patch, test_patch)
    )

    assert removed_hunks == 1
    assert removed_files == ["src/api.hpp"]
    assert "official implementation" not in sanitized
    assert "tests/api_test.cpp" in sanitized
    assert "new test" in sanitized


def test_dependency_preflight_only_initializes_old_simdjson_cache():
    assert MODULE.dependency_preflight_command("simdjson__simdjson-2178") == [
        "/bin/mkdir",
        "-p",
        "/home/simdjson/dependencies/.cache",
    ]
    assert MODULE.dependency_preflight_command("fmtlib__fmt-493") is None
