"""Semantic 'Function' search action using entity_tools"""

import os
from typing import Optional, Type, ClassVar, List

from pydantic import Field, model_validator, ConfigDict

from INTENT2FIX.actions.schema import ActionArguments, FewShotExample
from INTENT2FIX.actions.search_base import SearchBaseAction, SearchBaseArgs
from INTENT2FIX.index.types import SearchCodeResponse

logger = __import__("logging").getLogger(__name__)


class EntityFunctionSemanticSearchArgs(SearchBaseArgs):
    """Find ranked function candidates by implementation behavior.

    Use this for an implementation or related helper whose behavior is known but
    name/location is missing. Prefer EntitySearchSymbol for an already known name.
    Results are ranked indexed function locations/skeletons, not a complete call
    chain or proof of the required behavior. Inspect selected candidates with
    ViewCode; use text search when results are irrelevant or incomplete.
    """

    query: str = Field(
        ...,
        description="Describe what the target function does, not its guessed name. Example: 'formats chrono seconds with fractional precision'."
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Backend candidate cap (default 10, maximum 50); progressive output shows at most 10 skeletons."
    )

    model_config = ConfigDict(title="EntityFunctionSemanticSearch")

    @model_validator(mode="after")
    def validate_query(self) -> "EntityFunctionSemanticSearchArgs":
        if not self.query.strip():
            raise ValueError("query cannot be empty")
        return self

    def to_prompt(self):
        prompt = f"Function-level semantic search: {self.query}"
        if self.file_pattern:
            prompt += f" in files matching: {self.file_pattern}"
        return prompt

    def short_summary(self) -> str:
        params = [f"query={self.query[:50]}..."]
        if self.file_pattern:
            params.append(f"file_pattern={self.file_pattern}")
        return f"{self.name}({', '.join(params)})"


class EntityFunctionSemanticSearch(SearchBaseAction):
    """Semantic search for functions using entity_tools search_by_intent."""
    
    args_schema: ClassVar[Type[ActionArguments]] = EntityFunctionSemanticSearchArgs

    def _search(self, args: EntityFunctionSemanticSearchArgs) -> SearchCodeResponse:
        from INTENT2FIX.index.entity_code_index import EntityCodeIndex
        
        logger.info(
            f"{self.name}: query='{args.query[:50]}...', "
            f"file_pattern={args.file_pattern}, top_k={args.top_k}"
        )
        
        # Use existing code index if available, otherwise create one
        if self._code_index and isinstance(self._code_index, EntityCodeIndex):
            entity_index = self._code_index
        else:
            # Fallback to creating a new one if not injected
            instance_name = os.environ.get("SWEBENCH_INSTANCE_NAME")
            entity_index = EntityCodeIndex(file_repo=self._repository, index_name=instance_name)
        
        return entity_index.semantic_search(
            query=args.query,
            file_pattern=args.file_pattern,
            max_results=args.top_k,
        )

    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Find the functions responsible for reporting test results in XML format.",
                action=EntityFunctionSemanticSearchArgs(
                    query="XML reporting of test results",
                ),
            ),
            FewShotExample.create(
                user_input="How does Catch2 handle C++ signals and exceptions?",
                action=EntityFunctionSemanticSearchArgs(
                    query="exception handling and signal interception",
                ),
            ),
        ]
