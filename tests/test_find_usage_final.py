
import os
import logging
from INTENT2FIX.benchmark.utils import get_INTENT2FIX_instance
from INTENT2FIX.benchmark.swebench import create_repository
from INTENT2FIX.index.entity_code_index import EntityCodeIndex
from INTENT2FIX.actions.find_entity_usage import FindEntityUsage, FindEntityUsageArgs
from INTENT2FIX.file_context import FileContext
from INTENT2FIX.completion import BaseCompletionModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockCompletionModel(BaseCompletionModel):
    def initialize(self, response_schema, system_prompt):
        pass
    def create_completion(self, messages, response_schema=None):
        return None
    def _validate_completion(self, completion):
        return True

def test_find_entity_usage_comprehensive():
    """
    Test the FindEntityUsage action in the fmtlib__fmt-4055 environment.
    This test verifies that we can find callers of functions and usages of classes.
    """
    instance_name = "fmtlib__fmt-4055"
    os.environ["SWEBENCH_INSTANCE_NAME"] = instance_name
    
    # Load .env
    from dotenv import load_dotenv
    load_dotenv()
    
    index_store_dir = os.getenv("INDEX_STORE_DIR")
    if not index_store_dir:
        print("ERROR: INDEX_STORE_DIR not set in environment or .env file")
        return

    instance = get_INTENT2FIX_instance(instance_name)
    repository = create_repository(instance)
    
    code_index = EntityCodeIndex.from_index_name(
        index_name=instance_name,
        file_repo=repository,
        index_store_dir=index_store_dir
    )

    # Initialize action with a mock model
    # response_format must be a valid Enum value for the real INTENT2FIX models
    mock_model = MockCompletionModel(model="gpt-4", response_format="json")
    action = FindEntityUsage(repository=repository, code_index=code_index, completion_model=mock_model)
    file_context = FileContext(repo=repository)

    print(f"\n{'='*60}")
    print(f" TESTING FindEntityUsage for {instance_name}")
    print(f"{'='*60}")

    # Case 1: Find usages of a common class/struct
    # monostate is a common struct in fmt
    print("\n>>> CASE 1: Find usages of class/struct 'monostate'")
    args1 = FindEntityUsageArgs(type="class", name="monostate")
    obs1 = action.execute(args1, file_context=file_context)
    print(f"Result Summary: {obs1.summary}")
    print(f"Message Snippet:\n{obs1.message[:500]}...")

    # Case 2: Find callers of a core function
    # vformat is a central formatting function
    print("\n>>> CASE 2: Find callers of function 'vformat'")
    args2 = FindEntityUsageArgs(type="function", name="vformat")
    obs2 = action.execute(args2, file_context=file_context)
    print(f"Result Summary: {obs2.summary}")
    if obs2.message:
        print(f"Message Snippet:\n{obs2.message[:500]}...")
    else:
        print("No callers found or search failed.")

    # Case 3: Disambiguate with namespace
    print("\n>>> CASE 3: Find callers of 'vformat' in namespace 'fmt'")
    args3 = FindEntityUsageArgs(type="function", name="vformat", namespace="fmt")
    obs3 = action.execute(args3, file_context=file_context)
    print(f"Result Summary: {obs3.summary}")

if __name__ == "__main__":
    test_find_entity_usage_comprehensive()
