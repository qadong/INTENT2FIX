"""List features action using entity_tools"""

from typing import Optional, Type, ClassVar, List

from pydantic import Field, ConfigDict

from INTENT2FIX.actions.schema import ActionArguments, FewShotExample
from INTENT2FIX.actions.search_base import SearchBaseAction, SearchBaseArgs
from INTENT2FIX.index.types import SearchCodeResponse

logger = __import__("logging").getLogger(__name__)


class ListFeaturesArgs(SearchBaseArgs):
    """List or search high-level functional features (clusters of related functions).
    
    Features represent functional clusters that group related functions together.
    This action helps understand the macro architecture of the codebase and find
    entry points for specific modules or functionality.
    
    Use this when you need to:
    - Understand the high-level structure of the codebase
    - Find entry points for specific functionality
    - Discover related functions grouped by feature
    - Get an overview of what the codebase does
    """

    query: Optional[str] = Field(
        default=None,
        description="Optional search keyword to filter features by name or description. If not provided, lists all available features."
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of features to return. Default is 20."
    )

    model_config = ConfigDict(title="ListFeatures")

    def to_prompt(self):
        if self.query:
            return f"Search features: {self.query}"
        return "List all features"

    def short_summary(self) -> str:
        if self.query:
            return f"ListFeatures(query={self.query}, limit={self.limit})"
        return f"ListFeatures(limit={self.limit})"


class ListFeatures(SearchBaseAction):
    """List features using entity_tools list_features."""
    
    args_schema: ClassVar[Type[ActionArguments]] = ListFeaturesArgs

    def _search(self, args: ListFeaturesArgs) -> SearchCodeResponse:
        from INTENT2FIX.index.entity_code_index import EntityCodeIndex, _entity_to_search_hit
        
        logger.info(f"{self.name}: query={args.query}, limit={args.limit}")
        
        try:
            from code_intent.code_intent.entity_tools import list_features
            
            result = list_features(
                query=args.query,
                limit=args.limit,
            )
            
            if "error" in result:
                return SearchCodeResponse(message=f"Error: {result['error']}")
            
            # Convert features to SearchCodeResponse
            # Features contain member functions, so we create hits for each member
            hits = []
            features = result.get("features", [])
            
            for feature in features:
                # Create hits for member functions
                for member in feature.get("members", []):
                    hit = _entity_to_search_hit(member, self._repository)
                    if hit:
                        hits.append(hit)
            
            count = result.get("count", len(features))
            message = f"Found {count} feature(s)"
            if args.query:
                message += f" matching '{args.query}'"
            
            # Add feature names to message
            if features:
                feature_names = [f.get("name", "unknown") for f in features[:5]]
                message += f": {', '.join(feature_names)}"
                if len(features) > 5:
                    message += f" and {len(features) - 5} more"
            
            return SearchCodeResponse(message=message, hits=hits)
        except Exception as e:
            logger.error(f"Error in ListFeatures: {e}", exc_info=True)
            return SearchCodeResponse(message=f"Error listing features: {str(e)}")

    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="What are the main features of this codebase?",
                action=ListFeaturesArgs(
                    limit=10,
                ),
            ),
            FewShotExample.create(
                user_input="Find features related to memory management",
                action=ListFeaturesArgs(
                    query="memory management",
                    limit=10,
                ),
            ),
        ]

