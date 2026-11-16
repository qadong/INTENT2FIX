import logging
import os
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from INTENT2FIX.actions.entity_search_symbol import EntitySearchSymbol, EntitySearchSymbolArgs
from INTENT2FIX.actions.schema import Observation
from INTENT2FIX.completion.base import BaseCompletionModel, CompletionResponse, LLMResponseFormat
from INTENT2FIX.file_context import FileContext
from INTENT2FIX.index.entity_code_index import EntityCodeIndex
from INTENT2FIX.repository.file import FileRepository


INSTANCE_ID = "facebook__zstd-3942"
REPO_PATH = Path("/data/dqa/cache/repos/facebook__zstd-3942")
INDEX_STORE_DIR = Path("/data/dqa/cache/index_store")


logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


class MockModel(BaseCompletionModel):
    model: str = "mock"
    response_format: LLMResponseFormat = LLMResponseFormat.TOOLS

    def create_completion(self, messages: List[dict]) -> CompletionResponse:
        return CompletionResponse(structured_outputs=[], text_response="Mock")

    def initialize(self, response_schema, system_prompt):
        self._initialized = True

    def _execute_completion(self, messages):
        return None

    def _validate_completion(self, completion_response):
        return [], "Mock", []


def _assert_search_hit(observation: Observation, expected_text: str) -> None:
    print(f"\n[OBSERVATION]\n{observation.message}\n")
    hits = observation.properties.get("search_hits") if observation.properties else None
    if not hits:
        raise AssertionError(f"Expected search hits containing {expected_text!r}, got none")
    if expected_text not in observation.message:
        raise AssertionError(f"Expected observation to mention {expected_text!r}")


def main() -> None:
    if not REPO_PATH.is_dir():
        raise FileNotFoundError(f"Missing zstd repo: {REPO_PATH}")
    if not (INDEX_STORE_DIR / INSTANCE_ID / "index_output" / "entities" / "functions.jsonl").is_file():
        raise FileNotFoundError(f"Missing zstd function index under: {INDEX_STORE_DIR / INSTANCE_ID}")

    os.environ["INDEX_STORE_DIR"] = str(INDEX_STORE_DIR)
    os.environ["SWEBENCH_INSTANCE_NAME"] = INSTANCE_ID

    repository = FileRepository(repo_path=str(REPO_PATH))
    code_index = EntityCodeIndex(file_repo=repository, index_name=INSTANCE_ID)
    file_context = FileContext(repo=repository)

    mock_model = MockModel()
    mock_model.initialize(EntitySearchSymbolArgs, "dummy")

    search_symbol = EntitySearchSymbol(
        repository=repository,
        code_index=code_index,
        completion_model=mock_model,
    )

    print("\n" + "=" * 80)
    print("TEST 1: EntitySearchSymbol function lookup: TRACE_enable")
    print("=" * 80)
    function_obs = search_symbol.execute(
        EntitySearchSymbolArgs(
            type="function",
            name="TRACE_enable",
            file_pattern="programs/",
            name_exact_match=True,
            limit=20,
        ),
        file_context=file_context,
    )
    _assert_search_hit(function_obs, "TRACE_enable")

    print("\n" + "=" * 80)
    print("TEST 2: EntitySearchSymbol macro lookup: DEBUGLOG")
    print("=" * 80)
    macro_obs = search_symbol.execute(
        EntitySearchSymbolArgs(
            type="macro",
            name="DEBUGLOG",
            file_pattern="lib/common/",
            name_exact_match=True,
            limit=20,
        ),
        file_context=file_context,
    )
    _assert_search_hit(macro_obs, "DEBUGLOG")

    print("\nAll zstd entity tool checks passed.")


if __name__ == "__main__":
    main()
