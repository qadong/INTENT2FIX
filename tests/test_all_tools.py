import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

# 将项目根目录加入 path
sys.path.insert(0, str(Path(__file__).parent.parent))

# 加载环境变量
os.environ["INDEX_STORE_DIR"] = "./cache/INTENT2FIX/index_results"

from INTENT2FIX.actions.entity_search_symbol import EntitySearchSymbol, EntitySearchSymbolArgs
from INTENT2FIX.actions.entity_function_semantic_search import EntityFunctionSemanticSearch, EntityFunctionSemanticSearchArgs
from INTENT2FIX.actions.entity_intent_search import EntityIntentSearch, EntityIntentSearchArgs
from INTENT2FIX.actions.run_command import RunCommand, RunCommandArgs
from INTENT2FIX.actions.view_code import ViewCode, ViewCodeArgs
from INTENT2FIX.actions.string_replace import StringReplace, StringReplaceArgs
from INTENT2FIX.file_context import FileContext
from INTENT2FIX.repository.file import FileRepository
from INTENT2FIX.index.entity_code_index import EntityCodeIndex
from INTENT2FIX.completion.base import BaseCompletionModel, CompletionResponse, LLMResponseFormat

# ==========================================
# 配置区
# ==========================================
INSTANCE_ID = "catchorg__Catch2-2187"
TEST_REPO_PATH = "./cache/INTENT2FIX/repo/swe-bench_catchorg__Catch2-2187" 

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

class MockModel(BaseCompletionModel):
    model: str = "mock"
    response_format: LLMResponseFormat = LLMResponseFormat.TOOLS
    def create_completion(self, messages: List[dict]) -> CompletionResponse:
        return CompletionResponse(structured_outputs=[], text_response="Mock")
    def initialize(self, response_schema, system_prompt): self._initialized = True
    def _execute_completion(self, messages): pass
    def _validate_completion(self, completion_response): return [], "Mock", []

def run_tool_test(name, action, args, file_context):
    print(f"\n" + "="*60)
    print(f" TESTING TOOL: {name}")
    print("="*60)
    try:
        observation = action.execute(args, file_context=file_context)
        print(f"\n[OBSERVATION OUTPUT]:\n{observation.message}")
        return observation
    except Exception as e:
        print(f"\n[ERROR]: {str(e)}")
        return None

def main():
    os.environ["SWEBENCH_INSTANCE_NAME"] = INSTANCE_ID
    repository = FileRepository(repo_path=TEST_REPO_PATH)
    
    from unittest.mock import MagicMock
    mock_docker_env = MagicMock()
    mock_docker_env.run_bash.return_value = ("Mocked combined output", 0)
    
    code_index = EntityCodeIndex(file_repo=repository, index_name=INSTANCE_ID)
    mock_model = MockModel()
    from INTENT2FIX.actions.search_base import SearchBaseArgs
    mock_model.initialize(SearchBaseArgs, "dummy")

    file_context = FileContext(repo=repository)

    # 1. EntitySearchSymbol
    search_symbol = EntitySearchSymbol(repository=repository, code_index=code_index, completion_model=mock_model)
    run_tool_test("EntitySearchSymbol", search_symbol, 
                  EntitySearchSymbolArgs(type="class", name="ResultDisposition", name_exact_match=True), 
                  file_context)

    # 2. EntityFunctionSemanticSearch
    func_semantic_search = EntityFunctionSemanticSearch(repository=repository, code_index=code_index, completion_model=mock_model)
    run_tool_test("EntityFunctionSemanticSearch", func_semantic_search,
                  EntityFunctionSemanticSearchArgs(query="How is the test result reported?"),
                  file_context)

    # 3. EntityIntentSearch
    intent_search = EntityIntentSearch(repository=repository, code_index=code_index, completion_model=mock_model)
    run_tool_test("EntityIntentSearch", intent_search,
                  EntityIntentSearchArgs(query="Architecture of signal handling"),
                  file_context)

    # 4. RunCommand
    run_command = RunCommand(repository=repository, docker_env=mock_docker_env)
    run_tool_test("RunCommand", run_command, 
                  RunCommandArgs(command="ls -l"), 
                  file_context)

    # 4. StringReplace (正式流程：ViewCode -> StringReplace)
    dummy_file = "dummy_test_file.py"
    dummy_content = "def hello():\n    print('Hello')\n"
    full_dummy_path = os.path.join(TEST_REPO_PATH, dummy_file)
    with open(full_dummy_path, "w") as f:
        f.write(dummy_content)
    
    print("\n--- 步骤 3.1: 先调用 ViewCode 激活文件查看状态 ---")
    view_code = ViewCode(repository=repository)
    view_args = ViewCodeArgs(files=[{"file_path": dummy_file, "start_line": 1, "end_line": 2}])
    view_code.execute(view_args, file_context=file_context)

    print("\n--- 步骤 3.2: 执行 StringReplace ---")
    string_replace = StringReplace(repository=repository)
    run_tool_test("StringReplace", string_replace,
                  StringReplaceArgs(
                      path=dummy_file,
                      old_str="    print('Hello')",
                      new_str="    print('Hello INTENT2FIX')"
                  ),
                  file_context)
    
    if os.path.exists(full_dummy_path):
        os.remove(full_dummy_path)

if __name__ == "__main__":
    main()
