"""Find template instantiation action using entity_tools"""

from typing import Literal, Optional, Type, ClassVar, List

from pydantic import Field, ConfigDict

from INTENT2FIX.actions.schema import ActionArguments, FewShotExample
from INTENT2FIX.actions.search_base import SearchBaseAction, SearchBaseArgs
from INTENT2FIX.index.types import SearchCodeResponse

logger = __import__("logging").getLogger(__name__)


class FindTemplateInstantiationArgs(SearchBaseArgs):
    """Find indexed template instantiation and specialization records.

    Use this when a specific template form or specialization affects the repair.
    Provide an observed display name or template argument; use a definition ID only
    if already available from evidence. Results are recorded source locations, not
    every implicit instantiation, and do not prove which specialization/overload a
    compiler selects. Inspect source with ViewCode and verify relevant call shapes
    with a focused compile probe.
    """

    display_name: Optional[str] = Field(
        default=None,
        description="Partial rendered-instantiation name, for example 'vector<int>'."
    )
    template_def_id: Optional[str] = Field(
        default=None,
        description="Exact indexed template definition ID, only if already available; otherwise use display_name or template_arg. Do not invent an ID."
    )
    template_arg: Optional[str] = Field(
        default=None,
        description="Partial match against any recorded template argument."
    )
    category: Optional[Literal["Full", "Partial", "Explicit"]] = Field(
        default=None,
        description="Optional record category: Full, Partial, or Explicit."
    )
    is_explicit_specialization: Optional[bool] = Field(
        default=None,
        description="Optional exact filter for explicit-specialization records."
    )
    limit: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Backend record cap (default 50); progressive output shows at most 10 skeletons."
    )

    model_config = ConfigDict(title="FindTemplateInstantiation")

    def to_prompt(self):
        parts = ["Find template instantiations"]
        if self.display_name:
            parts.append(f"with display_name: {self.display_name}")
        if self.template_def_id:
            parts.append(f"from template: {self.template_def_id}")
        if self.template_arg:
            parts.append(f"with argument: {self.template_arg}")
        if self.category:
            parts.append(f"category: {self.category}")
        if self.file_pattern:
            parts.append(f"in files: {self.file_pattern}")
        return ", ".join(parts)

    def short_summary(self) -> str:
        params = []
        if self.display_name:
            params.append(f"display_name={self.display_name}")
        if self.template_def_id:
            params.append(f"template_def_id={self.template_def_id}")
        if self.template_arg:
            params.append(f"template_arg={self.template_arg}")
        return f"FindTemplateInstantiation({', '.join(params)})"


class FindTemplateInstantiation(SearchBaseAction):
    """Find template instantiations using entity_tools find_template_instantiation."""
    
    args_schema: ClassVar[Type[ActionArguments]] = FindTemplateInstantiationArgs

    def _search(self, args: FindTemplateInstantiationArgs) -> SearchCodeResponse:
        from INTENT2FIX.index.entity_code_index import EntityCodeIndex, _entity_to_search_hit
        
        logger.info(
            f"{self.name}: display_name={args.display_name}, "
            f"template_def_id={args.template_def_id}, template_arg={args.template_arg}, "
            f"category={args.category}, file_pattern={args.file_pattern}"
        )
        
        try:
            from code_intent.code_intent.entity_tools import find_template_instantiation

            self._ensure_entity_tools_env()
            
            result = find_template_instantiation(
                display_name=args.display_name,
                template_def_id=args.template_def_id,
                template_arg=args.template_arg,
                file=args.file_pattern,
                category=args.category,
                is_explicit_specialization=args.is_explicit_specialization,
                limit=args.limit,
            )
            
            # Convert to SearchCodeResponse
            hits = []
            for entity in result.get("entities", []):
                hit = _entity_to_search_hit(entity, self._repository)
                if hit:
                    hits.append(hit)
            
            total = result.get("total", 0)
            count = result.get("count", len(hits))
            message = f"Found {count} template instantiation(s) (total: {total})"
            if args.display_name:
                message += f" matching '{args.display_name}'"
            if args.template_def_id:
                message += f" from template '{args.template_def_id}'"
            
            return SearchCodeResponse(message=message, hits=hits)
        except Exception as e:
            logger.error(f"Error in FindTemplateInstantiation: {e}", exc_info=True)
            return SearchCodeResponse(message=f"Error searching template instantiations: {str(e)}")

    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Find all instantiations of std::vector<int>",
                action=FindTemplateInstantiationArgs(
                    display_name="vector<int>",
                ),
            ),
            FewShotExample.create(
                user_input="Find all instantiations from template definition 'template:MyClass:src/myclass.h:10'",
                action=FindTemplateInstantiationArgs(
                    template_def_id="template:MyClass:src/myclass.h:10",
                ),
            ),
        ]
