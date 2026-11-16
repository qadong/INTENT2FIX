"""Search symbol action using entity_tools"""

from typing import Literal, Optional, List, Type, ClassVar

from pydantic import Field, field_validator, model_validator, ConfigDict

from INTENT2FIX.actions.schema import ActionArguments, FewShotExample
from INTENT2FIX.actions.search_base import SearchBaseAction, SearchBaseArgs
from INTENT2FIX.index.types import SearchCodeResponse

logger = __import__("logging").getLogger(__name__)


class EntitySearchSymbolArgs(SearchBaseArgs):
    """Find indexed declarations by symbol name or list a namespace skeleton.

    Function and class searches include template forms. Use namespace/class_name
    to disambiguate same-name entities. Results are declaration skeletons; inspect
    the selected definition with ViewCode before editing. This finds named entities,
    not their callers; use FindEntityUsage for incoming usage evidence. An empty
    index result does not prove absence from source.
    """

    type: Literal["function", "class", "namespace", "macro"] = Field(
        ...,
        description="Search kind: function, class, namespace, or macro. Function/class include template forms."
    )
    name: str = Field(
        ...,
        description="Required symbol or namespace name. Partial match by default."
    )
    name_exact_match: bool = Field(
        default=False,
        description="Require an exact qualified, short, or entity-name match for function/class/macro searches."
    )
    namespace: Optional[str] = Field(
        default=None,
        description="Partial namespace filter used only with type=function or type=class."
    )
    class_name: Optional[str] = Field(
        default=None,
        description="Partial owning-class filter for a method search; only valid with type=function."
    )
    return_type: Optional[str] = Field(
        default=None,
        description="Exact indexed return-type filter; only valid with type=function."
    )
    parameter_types: Optional[List[str]] = Field(
        default=None,
        description="Exact ordered indexed parameter types; only valid with type=function."
    )
    kind: Optional[Literal["class", "struct"]] = Field(
        default=None,
        description="Optional class-kind filter: class or struct; only valid with type=class."
    )
    limit: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Backend candidate cap (default 50); progressive output shows at most 10 declaration skeletons."
    )

    model_config = ConfigDict(title="EntitySearchSymbol")

    @field_validator("type", mode="before")
    @classmethod
    def normalize_template_type(cls, value):
        return {
            "function_template": "function",
            "class_template": "class",
        }.get(value, value)

    @model_validator(mode="after")
    def validate_name(self) -> "EntitySearchSymbolArgs":
        if not self.name.strip():
            raise ValueError("name cannot be empty")
        return self

    def to_prompt(self):
        prompt = f"Searching for {self.type}"
        if self.name:
            prompt += f" with name: {self.name}"
        if self.class_name:
            prompt += f" in class: {self.class_name}"
        if self.file_pattern:
            prompt += f" in files matching: {self.file_pattern}"
        return prompt

    def short_summary(self) -> str:
        params = [f"type={self.type}"]
        if self.name:
            params.append(f"name={self.name}")
        if self.file_pattern:
            params.append(f"file_pattern={self.file_pattern}")
        return f"{self.name}({', '.join(params)})"


class EntitySearchSymbol(SearchBaseAction):
    """Search for code symbols using entity_tools."""
    
    args_schema: ClassVar[Type[ActionArguments]] = EntitySearchSymbolArgs

    def _search(self, args: EntitySearchSymbolArgs) -> SearchCodeResponse:
        from INTENT2FIX.index.entity_code_index import EntityCodeIndex, _entity_to_search_hit
        
        logger.info(
            f"{self.name}: type={args.type}, name={args.name}, "
            f"class_name={args.class_name}, file_pattern={args.file_pattern}"
        )

        # For namespace search, we increase the limits since it's meant to be a skeleton
        if args.type == "namespace" and not self.progressive_disclosure:
            self.max_hits = 500  # Allow more hits for skeleton
            self.max_search_tokens = 200000  # Higher token limit for skeleton
            self.max_identify_tokens = 200000
        
        # Use the code_index if available, otherwise create a new one
        if self._code_index and isinstance(self._code_index, EntityCodeIndex):
            entity_index = self._code_index
        else:
            # Create new EntityCodeIndex, try to get index_name from existing code_index
            index_name = None
            if self._code_index and hasattr(self._code_index, '_index_name'):
                index_name = self._code_index._index_name
            entity_index = EntityCodeIndex(file_repo=self._repository, index_name=index_name)
        
        if args.type == "function":
            return entity_index.find_function(
                function_name=args.name,
                class_name=args.class_name,
                file_pattern=args.file_pattern,
                namespace=args.namespace,
                name_exact_match=args.name_exact_match,
                return_type=args.return_type,
                parameter_types=args.parameter_types,
                limit=args.limit,
            )
        elif args.type == "class":
            return entity_index.find_class(
                class_name=args.name or "",
                file_pattern=args.file_pattern,
                kind=args.kind,
                namespace=args.namespace,
                name_exact_match=args.name_exact_match,
                limit=args.limit,
            )
        elif args.type == "namespace":
            return entity_index.find_namespace(
                namespace_name=args.name,
                file_pattern=args.file_pattern,
                limit=args.limit,
            )
        else:
            # Macro lookup uses the entity_tools backend directly.
            try:
                from code_intent.code_intent.entity_tools import search_symbol
                
                # Ensure environment variables are set before calling search_symbol
                self._ensure_entity_tools_env()
                
                # Build search parameters - only include relevant ones for the type
                search_params = {
                    "type": args.type,
                    "name": args.name,
                    "name_exact_match": args.name_exact_match,
                    "file": args.file_pattern,
                    "limit": args.limit,
                }
                
                # Only add function/class-specific parameters for non-macro types
                if args.type != "macro":
                    if args.namespace:
                        search_params["namespace"] = args.namespace
                    if args.class_name:
                        search_params["class_name"] = args.class_name
                    if args.return_type:
                        search_params["return_type"] = args.return_type
                    if args.parameter_types:
                        search_params["parameter_types"] = args.parameter_types
                    if args.kind:
                        search_params["kind"] = args.kind
                
                result = search_symbol(**search_params)
                logger.info(f"DEBUG [EntitySearchSymbol] Raw result count: {result.get('total', 0)}")
                
                # Convert to SearchCodeResponse
                file_hits = {}
                for entity in result.get("entities", []):
                    logger.info(f"DEBUG [EntitySearchSymbol] processing entity: name={entity.get('name') or entity.get('fq_name')}, file={entity.get('file')}")
                    hit = _entity_to_search_hit(entity, self._repository)
                    if hit:
                        if hit.file_path not in file_hits:
                            file_hits[hit.file_path] = hit
                        else:
                            # Merge spans for the same file
                            existing_span_ids = {s.span_id for s in file_hits[hit.file_path].spans}
                            for new_span in hit.spans:
                                if new_span.span_id not in existing_span_ids:
                                    file_hits[hit.file_path].spans.append(new_span)
                
                hits = list(file_hits.values())
                
                message = f"Found {len(hits)} {args.type}(s)"
                if args.name:
                    message += f" matching '{args.name}'"
                
                return SearchCodeResponse(message=message, hits=hits)
            except Exception as e:
                logger.error(f"Error in EntitySearchSymbol: {e}", exc_info=True)
                return SearchCodeResponse(message=f"Error searching: {str(e)}")

    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Find the calculate_interest function",
                action=EntitySearchSymbolArgs(
                    type="function",
                    name="calculate_interest",
                ),
            ),
            FewShotExample.create(
                user_input="Show me the UserRepository class",
                action=EntitySearchSymbolArgs(
                    type="class",
                    name="UserRepository",
                ),
            ),
        ]
