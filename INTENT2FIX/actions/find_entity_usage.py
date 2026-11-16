
from typing import Literal, Optional, List, Type, ClassVar

from pydantic import Field, ConfigDict

from INTENT2FIX.actions.schema import ActionArguments, FewShotExample
from INTENT2FIX.actions.search_base import SearchBaseAction, SearchBaseArgs
from INTENT2FIX.index.types import SearchCodeResponse

logger = __import__("logging").getLogger(__name__)


class FindEntityUsageArgs(SearchBaseArgs):
    """Query indexed incoming call/reference edges for a function or class.

    Use this to inspect affected callers/consumers of a shared function or type.
    It selects the first matching indexed target, not every overload; narrow by
    namespace/class_name and check the reported target. Results are incoming usage
    locations/skeletons, not outgoing callees or an exhaustive reference list. If no
    usage is found, the target definition may be shown instead. Inspect relevant
    source with ViewCode and use grep to check missing or ambiguous references.
    """

    type: Literal["function", "class"] = Field(
        ...,
        description="Target entity kind: function or class."
    )
    name: str = Field(
        ...,
        description="Observed short or qualified target name. Exact lookup is tried first, then partial matching; verify the reported target."
    )
    namespace: Optional[str] = Field(
        default=None,
        description="Optional partial namespace filter used to disambiguate the target."
    )
    class_name: Optional[str] = Field(
        default=None,
        description="Optional owning class for a method target."
    )
    limit: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Graph candidate cap (default 50); progressive output shows at most 10 usage skeletons."
    )

    model_config = ConfigDict(title="FindEntityUsage")

    def to_prompt(self):
        prompt = f"Find usages of {self.type}: {self.name}"
        if self.class_name:
            prompt += f" in class: {self.class_name}"
        if self.namespace:
            prompt += f" in namespace: {self.namespace}"
        return prompt

    def short_summary(self) -> str:
        return f"FindEntityUsage(type={self.type}, name={self.name})"


class FindEntityUsage(SearchBaseAction):
    """Find entity usages using entity_tools find_usages."""
    
    args_schema: ClassVar[Type[ActionArguments]] = FindEntityUsageArgs

    def _search(self, args: FindEntityUsageArgs) -> SearchCodeResponse:
        from INTENT2FIX.index.entity_code_index import EntityCodeIndex, _entity_to_search_hit
        
        logger.info(f"{self.name}: type={args.type}, name='{args.name}'")
        
        try:
            from code_intent.code_intent.entity_tools import find_usages
            
            # Ensure environment
            self._ensure_entity_tools_env()
            
            result = find_usages(
                type=args.type,
                name=args.name,
                namespace=args.namespace,
                class_name=args.class_name,
                limit=args.limit
            )
            
            if "error" in result:
                return SearchCodeResponse(message=f"Error: {result['error']}")

            if args.file_pattern:
                from INTENT2FIX.index.entity_code_index import _matches_file_pattern

                result = {
                    **result,
                    "usages": [
                        item
                        for item in result.get("usages", [])
                        if _matches_file_pattern(item.get("file", ""), args.file_pattern)
                    ],
                }
            
            target = result.get("target", {})
            usages = result.get("usages", [])
            if args.file_pattern and target:
                from INTENT2FIX.index.entity_code_index import _matches_file_pattern

                if not _matches_file_pattern(
                    target.get("file", "") or target.get("def_file", ""),
                    args.file_pattern,
                ):
                    target = {}
            
            hits = []
            for usage in usages:
                # Usages from get_entity_neighbors already have some info, 
                # but we might need more to create a proper hit.
                # The 'usage' dict here has keys like 'id', 'type', 'name', 'file'.
                hit = _entity_to_search_hit(usage, self._repository)
                if hit:
                    hits.append(hit)

            if not hits and target:
                target_hit = _entity_to_search_hit(target, self._repository)
                if target_hit:
                    hits.append(target_hit)
            
            target_name = target.get("fq_name") or target.get("name") or args.name
            if usages and hits:
                message = f"Found {len(hits)} usage location(s) of {args.type} '{target_name}'"
            elif hits:
                message = (
                    f"No code usage locations found for {args.type} '{target_name}'. "
                    "Showing the target definition instead."
                )
            else:
                message = f"Found no code usage locations for {args.type} '{target_name}'"
            
            return SearchCodeResponse(message=message, hits=hits)
        except Exception as e:
            logger.error(f"Error in FindEntityUsage: {e}", exc_info=True)
            return SearchCodeResponse(message=f"Error searching usages: {str(e)}")

    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Find all callers of the 'parse_json' function",
                action=FindEntityUsageArgs(
                    type="function",
                    name="parse_json",
                ),
            ),
            FewShotExample.create(
                user_input="Where is the 'UserRepository' class used?",
                action=FindEntityUsageArgs(
                    type="class",
                    name="UserRepository",
                ),
            ),
        ]
