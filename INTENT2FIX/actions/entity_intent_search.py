"""High-level 'Intent' (Feature) search action using entity_tools"""

import os
from typing import Optional, Type, ClassVar, List, Dict, Any

from pydantic import Field, model_validator, ConfigDict, PrivateAttr

from INTENT2FIX.actions.action import Action
from INTENT2FIX.actions.schema import ActionArguments, FewShotExample, Observation
from INTENT2FIX.file_context import FileContext
from INTENT2FIX.workspace import Workspace
from INTENT2FIX.repository.repository import Repository
from INTENT2FIX.index import CodeIndex

logger = __import__("logging").getLogger(__name__)


class EntityIntentSearchArgs(ActionArguments):
    """Find high-level feature clusters and member labels by existing-code behavior.

    Use this when the owning feature/module is unclear. Results contain generated
    descriptions and a bounded list of member labels, not verified entry points,
    implementations, or traversal IDs. Locate a relevant member with
    EntitySearchSymbol, then inspect its source with ViewCode. Existing-code intent
    does not establish desired behavior or authorize changing the assigned Hypothesis.
    """

    query: str = Field(
        ...,
        description="Describe the repository-level behavior or feature to locate. Example: 'ordered JSON object storage and conversion'."
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum feature candidates to return (default 10, maximum 50)."
    )

    model_config = ConfigDict(title="EntityIntentSearch")

    @model_validator(mode="after")
    def validate_query(self) -> "EntityIntentSearchArgs":
        if not self.query.strip():
            raise ValueError("query cannot be empty")
        return self

    def to_prompt(self):
        return f"Intent (Feature) search: {self.query}"


class EntityIntentSearch(Action):
    """Semantic search for high-level features using entity_tools list_features."""
    
    args_schema: ClassVar[Type[ActionArguments]] = EntityIntentSearchArgs

    _repository: Repository = PrivateAttr()
    _code_index: CodeIndex = PrivateAttr()
    progressive_disclosure: bool = Field(
        False,
        description="Return a compact feature shortlist before source code is viewed.",
    )

    def __init__(
        self,
        repository: Repository = None,
        code_index: CodeIndex | None = None,
        **data,
    ):
        super().__init__(**data)
        self._repository = repository
        self._code_index = code_index

    def _execute(
        self,
        args: EntityIntentSearchArgs,
        file_context: FileContext | None = None,
        workspace: Workspace | None = None,
    ) -> str | None:
        from INTENT2FIX.index.entity_code_index import EntityCodeIndex
        
        logger.info(f"{self.name}: query='{args.query[:50]}...', limit={args.limit}")
        
        # Use existing code index if available, otherwise create one
        if self._code_index and isinstance(self._code_index, EntityCodeIndex):
            entity_index = self._code_index
        else:
            # Fallback to creating a new one if not injected
            import os
            instance_name = os.environ.get("SWEBENCH_INSTANCE_NAME")
            entity_index = EntityCodeIndex(file_repo=self._repository, index_name=instance_name)
        
        result = entity_index.feature_search(
            query=args.query,
            max_results=args.limit,
        )
        
        if "error" in result:
            return f"Error in intent search: {result['error']}"
            
        features = result.get("features", [])
        if not features:
            return f"No high-level features found for query: '{args.query}'"
            
        # Format the output for the agent
        output = [f"Found {len(features)} high-level code features for your query:\n"]
        
        for i, feat in enumerate(features):
            name = feat.get("feature_name", "Unknown Feature")
            desc = feat.get("feature_description", "No description available.")
            if self.progressive_disclosure and len(desc) > 320:
                desc = desc[:317].rstrip() + "..."
            score = feat.get("score", 0.0)
            
            output.append(f"{i+1}. FEATURE: {name} (Score: {score})")
            output.append(f"   Description: {desc}")
            
            members = feat.get("members", [])
            if members:
                output.append("   Main member functions:")
                member_limit = 3 if self.progressive_disclosure else 5
                for member in members[:member_limit]:
                    func_label = member.get("func_id", "Unknown function")
                    output.append(f"     - {func_label}")
                if len(members) > member_limit:
                    output.append(f"     - ... and {len(members)-member_limit} more functions")
            output.append("")

        if self.progressive_disclosure:
            output.append(
                "Next: select a relevant member and use EntitySearchSymbol or ViewCode; "
                "refine the intent query if none match."
            )
            
        return "\n".join(output)

    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="How is the test reporting system structured?",
                action=EntityIntentSearchArgs(
                    query="test reporting and output formats",
                ),
            ),
            FewShotExample.create(
                user_input="What are the core modules for handling C++ exceptions and signals?",
                action=EntityIntentSearchArgs(
                    query="exception handling and signal interception",
                ),
            ),
        ]
