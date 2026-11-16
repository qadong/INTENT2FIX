"""Quick test for entity_tools - showcasing code and skeletons"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set instance
os.environ["SWEBENCH_INSTANCE_NAME"] = "swe-bench_catchorg__Catch2-2849"

from code_intent.entity_tools import search_symbol, get_entity_by_id

def test_display():
    print("=" * 80)
    print("ENTITY TOOLS EFFECTIVENESS TEST")
    print("=" * 80)

    # 1. Test Function - Should show full code with line numbers
    print("\n[CASE 1] Search Function: 'toString'")
    result = search_symbol(type="function", name="toString", limit=1)
    if result['entities']:
        entity = result['entities'][0]
        print(f"ID: {entity.get('id')}")
        print(f"FQ Name: {entity.get('fq_name')}")
        print(f"File: {entity.get('file')} (Range: {entity.get('line_range')})")
        print("-" * 40)
        print("CODE (Numbered Implementation):")
        print(entity.get('code', 'MISSING CODE!'))
        print("-" * 40)
    else:
        print("No function found for 'toString'")

    # 2. Test Class - Should show skeleton with member line ranges
    print("\n[CASE 2] Search Class: 'Session'")
    result = search_symbol(type="class", name="Session", limit=1)
    if result['entities']:
        entity = result['entities'][0]
        print(f"ID: {entity.get('id')}")
        print(f"FQ Name: {entity.get('fq_name')}")
        print(f"File: {entity.get('file')} (Range: {entity.get('line_range')})")
        print("-" * 40)
        print("SKELETON (Member Summary with Line Ranges):")
        print(entity.get('skeleton', 'MISSING SKELETON!'))
        print("-" * 40)
        
        # 3. Test Detail - Show full code for the same class with line numbers
        print("\n[CASE 3] Get Full Class Detail (using ID from Case 2)")
        full_entity = get_entity_by_id(entity.get('id'))
        if full_entity:
            print(f"Full Code Preview with Line Numbers (First 15 lines):")
            code_lines = full_entity.get('code', '').splitlines()
            print("\n".join(code_lines[:15]))
            if len(code_lines) > 15:
                print(f"    ... ({len(code_lines)-15} more lines)")
    else:
        print("No class found for 'Session'")

    # 4. Test Macro
    print("\n[CASE 4] Search Macro: 'CATCH_CONFIG'")
    result = search_symbol(type="macro", name="CATCH_CONFIG", limit=2)
    for i, m in enumerate(result['entities'], 1):
        print(f"{i}. Macro: {m.get('name')}")
        print(f"   File: {m.get('file')}:{m.get('line')}")
        print(f"   Value: {m.get('value')}")

    print("\n" + "=" * 80)
    print("All tests completed! ✓")
    print("=" * 80)

if __name__ == "__main__":
    try:
        test_display()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
