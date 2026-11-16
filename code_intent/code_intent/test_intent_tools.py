import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from code_intent.entity_tools import (
    search_by_intent, 
    list_features, 
    get_macro_usages, 
    get_tools_definition,
    get_current_instance
)

def test_intent_tools():
    # Setup environment
    instance_name = "swe-bench_catchorg__Catch2-2849"
    os.environ["SWEBENCH_INSTANCE_NAME"] = instance_name
    
    print(f"Testing tools for instance: {instance_name}")
    print("-" * 50)
    
    # 1. Test get_tools_definition
    print("\n1. Testing get_tools_definition:")
    tools = get_tools_definition()
    new_tools = [t["function"]["name"] for t in tools if t["function"]["name"] in ("search_by_intent", "list_features", "get_macro_usages")]
    print(f"Found new tools: {new_tools}")
    
    # 2. Test search_by_intent
    print("\n2. Testing search_by_intent (query='assertion handling'):")
    # Note: This requires VOYAGE_API_KEY if we want real embeddings, otherwise it uses stubs
    result = search_by_intent(query="assertion handling", top_k=3)
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Found {result['count']} results.")
        for i, res in enumerate(result["results"]):
            print(f"  [{i+1}] {res.get('name')} (Score: {res.get('score')})")
            print(f"      File: {res.get('file')}:{res.get('line_range')}")
            
    # 3. Test list_features
    print("\n3. Testing list_features:")
    result = list_features(limit=5)
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Found {result['count']} features.")
        for i, feat in enumerate(result["features"]):
            print(f"  [{i+1}] {feat.get('feature_name')}")
            print(f"      Description: {feat.get('feature_description')[:100]}...")
            
    # 4. Test get_macro_usages
    print("\n4. Testing get_macro_usages (name='CATCH_CONFIG_MAIN'):")
    result = get_macro_usages(name="CATCH_CONFIG_MAIN")
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Macro: {result.get('name')}")
        print(f"Definitions: {len(result.get('definitions', []))}")
        print(f"Usages: {len(result.get('usages', []))}")
        for i, usage in enumerate(result.get("usages", [])[:3]):
            print(f"  Usage [{i+1}]: {usage.get('fq_name')} in {usage.get('file')}")

if __name__ == "__main__":
    test_intent_tools()





