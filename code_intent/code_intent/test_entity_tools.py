import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.environ["SWEBENCH_INSTANCE_NAME"] = "swe-bench_catchorg__Catch2-2849"

from code_intent.entity_tools import (
    search_symbol,
    find_template_instantiation,
    get_entity_by_id,
    get_tools_definition,
    execute_tool,
)


def test_1_search_function_by_name():
    print("=" * 80)
    print("Test 1: Search function by name")
    print("=" * 80)
    result = search_symbol(type="function", name="test", limit=5)
    print(f"Found {result['total']} functions, returned {result['count']}")
    if result['entities']:
        print("\nFirst 3 results:")
        for i, entity in enumerate(result['entities'][:3], 1):
            print(f"  {i}. {entity.get('fq_name')}")
            print(f"     file: {entity.get('def_file', 'unknown')}")
            print(f"     line: {entity.get('def_start_line', 'unknown')}")
    print()


def test_2_search_class_by_kind():
    print("=" * 80)
    print("Test 2: Search class by kind (struct)")
    print("=" * 80)
    result = search_symbol(type="class", kind="struct", limit=5)
    print(f"Found {result['total']} structs, returned {result['count']}")
    if result['entities']:
        print("\nFirst 3 results:")
        for i, entity in enumerate(result['entities'][:3], 1):
            print(f"  {i}. {entity.get('fq_name')} (kind: {entity.get('kind')})")
            print(f"     file: {entity.get('def_file', 'unknown')}")
    print()


def test_3_search_function_by_namespace():
    print("=" * 80)
    print("Test 3: Search function by namespace (Catch)")
    print("=" * 80)
    result = search_symbol(type="function", namespace="Catch", limit=5)
    print(f"Found {result['total']} functions, returned {result['count']}")
    if result['entities']:
        print("\nFirst 3 results:")
        for i, entity in enumerate(result['entities'][:3], 1):
            print(f"  {i}. {entity.get('fq_name')}")
            print(f"     namespaces: {entity.get('namespaces', [])}")
    print()


def test_4_search_class_by_base_class():
    print("=" * 80)
    print("Test 4: Search class by base (std::exception)")
    print("=" * 80)
    result = search_symbol(type="class", base_class="exception", limit=5)
    print(f"Found {result['total']} classes, returned {result['count']}")
    if result['entities']:
        print("\nFirst 3 results:")
        for i, entity in enumerate(result['entities'][:3], 1):
            print(f"  {i}. {entity.get('fq_name')}")
            print(f"     bases: {entity.get('bases', [])}")
    print()


def test_5_search_macro():
    print("=" * 80)
    print("Test 5: Search macro (CATCH_ prefix)")
    print("=" * 80)
    result = search_symbol(type="macro", name="CATCH_", limit=5)
    print(f"Found {result['total']} macros, returned {result['count']}")
    if result['entities']:
        print("\nFirst 3 results:")
        for i, entity in enumerate(result['entities'][:3], 1):
            print(f"  {i}. {entity.get('name')}")
            print(f"     file: {entity.get('file', 'unknown')}")
            print(f"     line: {entity.get('line', 'unknown')}")
    print()


def test_6_search_function_by_file():
    print("=" * 80)
    print("Test 6: Search function by file (containing 'test')")
    print("=" * 80)
    result = search_symbol(type="function", file="test", limit=5)
    print(f"Found {result['total']} functions, returned {result['count']}")
    if result['entities']:
        print("\nFirst 3 results:")
        for i, entity in enumerate(result['entities'][:3], 1):
            print(f"  {i}. {entity.get('fq_name')}")
            print(f"     file: {entity.get('def_file', 'unknown')}")
    print()


def test_7_search_function_template():
    print("=" * 80)
    print("Test 7: Search function template")
    print("=" * 80)
    result = search_symbol(type="function_template", limit=5)
    print(f"Found {result['total']} function templates, returned {result['count']}")
    if result['entities']:
        print("\nFirst 3 results:")
        for i, entity in enumerate(result['entities'][:3], 1):
            print(f"  {i}. {entity.get('fq_name')}")
            print(f"     is_template: {entity.get('is_template', False)}")
    print()


def test_8_search_class_template():
    print("=" * 80)
    print("Test 8: Search class template")
    print("=" * 80)
    result = search_symbol(type="class_template", limit=5)
    print(f"Found {result['total']} class templates, returned {result['count']}")
    
    if result['entities']:
        print("\nFirst 3 results:")
        for i, entity in enumerate(result['entities'][:3], 1):
            print(f"  {i}. {entity.get('fq_name')}")
            print(f"     kind: {entity.get('kind', 'unknown')}")
    print()


def test_9_find_template_instantiation():
    print("=" * 80)
    print("Test 9: Find template instantiation (is_callable)")
    print("=" * 80)
    result = find_template_instantiation(display_name="is_callable", limit=5)
    print(f"Found {result['total']} template instantiations, returned {result['count']}")
    if result['entities']:
        print("\nFirst 3 results:")
        for i, entity in enumerate(result['entities'][:3], 1):
            print(f"  {i}. {entity.get('display_name')}")
            print(f"     template_args: {entity.get('template_args', [])}")
            print(f"     file: {entity.get('trigger_file', 'unknown')}")
    print()


def test_10_get_entity_by_id():
    print("=" * 80)
    print("Test 10: Get entity by ID")
    print("=" * 80)
    search_result = search_symbol(type="function", name="test", limit=1)
    if not search_result['entities']:
        print("No test function found, skip this test")
        return
    entity_id = search_result['entities'][0].get('id')
    print(f"Lookup ID: {entity_id}")
    entity = get_entity_by_id(entity_id)
    if entity:
        print(f"Found entity: {entity.get('fq_name')}")
        print(f"  file: {entity.get('def_file', 'unknown')}")
        print(f"  type: function")
    else:
        print("Entity not found")
    print()


def test_11_complex_filter():
    print("=" * 80)
    print("Test 11: Complex filter (function + namespace + file)")
    print("=" * 80)
    result = search_symbol(
        type="function",
        namespace="Catch",
        file="test",
        limit=5
    )
    print(f"Found {result['total']} functions, returned {result['count']}")
    if result['entities']:
        print("\nFirst 3 results:")
        for i, entity in enumerate(result['entities'][:3], 1):
            print(f"  {i}. {entity.get('fq_name')}")
            print(f"     namespaces: {entity.get('namespaces', [])}")
            print(f"     file: {entity.get('def_file', 'unknown')}")
    print()


def test_12_pagination():
    print("=" * 80)
    print("Test 12: Pagination")
    print("=" * 80)
    result1 = search_symbol(type="function", name="test", limit=3, offset=0)
    print(f"Page 1: total {result1['total']}, returned {result1['count']}")
    if result1['entities']:
        print(f"  first: {result1['entities'][0].get('fq_name')}")
    result2 = search_symbol(type="function", name="test", limit=3, offset=3)
    print(f"Page 2: returned {result2['count']}")
    if result2['entities']:
        print(f"  first: {result2['entities'][0].get('fq_name')}")
    if result1['entities'] and result2['entities']:
        id1 = result1['entities'][0].get('id')
        id2 = result2['entities'][0].get('id')
        if id1 != id2:
            print("Pagination OK, no duplicate")
        else:
            print("Pagination error, duplicate")
    print()


def test_13_tools_definition():
    print("=" * 80)
    print("Test 13: Tools definition format")
    print("=" * 80)
    tools = get_tools_definition()
    print(f"Got {len(tools)} tool definitions")
    for tool in tools:
        func = tool["function"]
        print(f"  - {func['name']}: {func['description'][:50]}...")
    assert len(tools) == 3, "Expected 3 tools"
    assert all("type" in t and t["type"] == "function" for t in tools), "All tools should be function type"
    assert all("function" in t for t in tools), "All tools should have function field"
    print("Tool definition format OK")
    print()


def test_14_execute_tool():
    print("=" * 80)
    print("Test 14: Execute tool (simulate agent call)")
    print("=" * 80)
    result = execute_tool(
        "search_symbol",
        {
            "type": "function",
            "name": "test",
            "limit": 3
        }
    )
    print("Tool execution OK")
    print(f"  Found {result['total']} functions, returned {result['count']}")
    if result['entities']:
        print(f"  Example: {result['entities'][0].get('fq_name')}")
    print()


def run_all_tests():
    print("\n" + "=" * 80)
    print("Running entity_tools tests")
    print("=" * 80 + "\n")
    
    tests = [
        test_1_search_function_by_name,
        test_2_search_class_by_kind,
        test_3_search_function_by_namespace,
        test_4_search_class_by_base_class,
        test_5_search_macro,
        test_6_search_function_by_file,
        test_7_search_function_template,
        test_8_search_class_template,
        test_9_find_template_instantiation,
        test_10_get_entity_by_id,
        test_11_complex_filter,
        test_12_pagination,
        test_13_tools_definition,
        test_14_execute_tool,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"Test failed: {test_func.__name__}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()
    
    print("=" * 80)
    print(f"Tests done: passed {passed}, failed {failed}")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

