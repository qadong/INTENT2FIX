"""Get macro usages action using entity_tools"""

from typing import Type, ClassVar, List

from pydantic import Field, ConfigDict

from INTENT2FIX.actions.schema import ActionArguments, FewShotExample
from INTENT2FIX.actions.search_base import SearchBaseAction, SearchBaseArgs
from INTENT2FIX.index.types import SearchCodeResponse

logger = __import__("logging").getLogger(__name__)


class GetMacroUsagesArgs(SearchBaseArgs):
    """Find indexed macro definitions and function-level usage edges.

    Use this when a named macro controls code implicated by the repair. Results
    identify recorded definitions and functions using the macro, not expanded code
    or the active build configuration. Inspect them with ViewCode; use grep for
    missing textual or preprocessor-only references and compile relevant forms.
    """

    name: str = Field(
        ...,
        description="Exact macro identifier, without arguments or a leading #define."
    )

    model_config = ConfigDict(title="GetMacroUsages")

    def to_prompt(self):
        prompt = f"Find macro usages: {self.name}"
        if self.file_pattern:
            prompt += f" in files matching: {self.file_pattern}"
        return prompt

    def short_summary(self) -> str:
        return f"GetMacroUsages(name={self.name})"


class GetMacroUsages(SearchBaseAction):
    """Get macro usages using entity_tools get_macro_usages."""
    
    args_schema: ClassVar[Type[ActionArguments]] = GetMacroUsagesArgs

    def _search(self, args: GetMacroUsagesArgs) -> SearchCodeResponse:
        from INTENT2FIX.index.entity_code_index import EntityCodeIndex, _entity_to_search_hit
        
        logger.info(f"{self.name}: name='{args.name}'")
        
        try:
            from code_intent.code_intent.entity_tools import get_macro_usages

            self._ensure_entity_tools_env()
            
            result = get_macro_usages(name=args.name)
            
            if "error" in result:
                return SearchCodeResponse(message=f"Error: {result['error']}")

            if args.file_pattern:
                from INTENT2FIX.index.entity_code_index import _matches_file_pattern

                result = {
                    **result,
                    "definitions": [
                        item
                        for item in result.get("definitions", [])
                        if _matches_file_pattern(item.get("file", ""), args.file_pattern)
                    ],
                    "usages": [
                        item
                        for item in result.get("usages", [])
                        if _matches_file_pattern(item.get("file", ""), args.file_pattern)
                    ],
                }
            
            # Convert to SearchCodeResponse
            hits = []
            
            # Add macro definitions as hits
            for defn in result.get("definitions", []):
                if "file" in defn and "line" in defn:
                    from INTENT2FIX.index.types import SpanHit, SearchCodeHit
                    span_id = f"{defn['file']}:{defn['line']}"
                    value = defn.get("value_snippet") or defn.get("value") or ""
                    hit = SearchCodeHit(
                        file_path=defn["file"],
                        spans=[SpanHit(
                            span_id=span_id,
                            label=args.name,
                            start_line=defn["line"],
                            end_line=defn["line"],
                            tokens=0,
                            skeleton=f"#define {args.name} {value}".rstrip(),
                        )]
                    )
                    hits.append(hit)
            
            # Add macro usages (functions that use the macro) as hits
            for usage in result.get("usages", []):
                if "file" in usage and "line" in usage:
                    fq_name = usage.get("fq_name", "unknown")
                    from INTENT2FIX.index.types import SpanHit, SearchCodeHit
                    span_id = f"{usage['file']}:{usage['line']}"
                    hit = SearchCodeHit(
                        file_path=usage["file"],
                        spans=[SpanHit(
                            span_id=span_id,
                            label=fq_name,
                            start_line=usage["line"],
                            end_line=usage["line"],
                            tokens=0,
                            skeleton=f"{fq_name} uses {args.name}",
                        )]
                    )
                    hits.append(hit)
            
            def_count = len(result.get("definitions", []))
            usage_count = len(result.get("usages", []))
            message = (
                f"Found macro '{args.name}': "
                f"{def_count} definition(s), {usage_count} usage(s) in functions"
            )
            
            return SearchCodeResponse(message=message, hits=hits)
        except Exception as e:
            logger.error(f"Error in GetMacroUsages: {e}", exc_info=True)
            return SearchCodeResponse(message=f"Error searching macro usages: {str(e)}")

    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Where is the CATCH_CONFIG_MAIN macro used?",
                action=GetMacroUsagesArgs(
                    name="CATCH_CONFIG_MAIN",
                ),
            ),
            FewShotExample.create(
                user_input="Find all functions that use INTERNAL_CATCH_IF",
                action=GetMacroUsagesArgs(
                    name="INTERNAL_CATCH_IF",
                ),
            ),
        ]
