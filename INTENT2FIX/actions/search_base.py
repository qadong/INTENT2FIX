import logging
from abc import ABC
from typing import List, Optional, Type, Any, ClassVar, Tuple

from pydantic import Field, PrivateAttr, BaseModel, field_validator, model_validator

from INTENT2FIX.actions.action import Action
from INTENT2FIX.actions.schema import ActionArguments, Observation, RewardScaleEntry
from INTENT2FIX.completion import BaseCompletionModel
from INTENT2FIX.completion.model import Completion
from INTENT2FIX.completion.schema import (
    ChatCompletionAssistantMessage,
    ChatCompletionUserMessage,
)
from INTENT2FIX.completion.schema import ResponseSchema
from INTENT2FIX.exceptions import CompletionRejectError
from INTENT2FIX.file_context import FileContext
from INTENT2FIX.index import CodeIndex
from INTENT2FIX.index.types import SearchCodeResponse
from INTENT2FIX.repository.repository import Repository
from INTENT2FIX.workspace import Workspace

logger = logging.getLogger(__name__)

IDENTIFY_SYSTEM_PROMPT = """You are an autonomous AI assistant tasked with identifying relevant code in a codebase. Your goal is to select key code sections from the search results that are most relevant to the search request.

The previous messages will contain:
1. A search request from an AI assistant
2. Search results containing various code sections with their line numbers

# Your Task:

1. Understand the Search Request:
   * Analyze the previous search request to understand what code elements are being looked for
   * Identify key elements such as functions, variables, classes, or patterns that are relevant

2. Evaluate Search Results:
   * Examine each code section in the search results for alignment with the search request
   * Assess the relevance and importance of each code section
   * Consider the complete context of code sections

3. Respond with the Identify Action:
   * Select the smallest line spans that contain the relevant declaration or logic
   * Do not include broad surrounding regions when a narrower span is sufficient
   * Provide your analysis in the thoughts field
   * List the relevant file paths with start and end line numbers in the identified_spans field
"""


class SearchBaseArgs(ActionArguments, ABC):
    file_pattern: Optional[str] = Field(
        default=None,
        description=(
            "Optional single repository-relative file or glob filter, for example "
            "'include/**' or 'src/foo.cpp'; comma-separated patterns are not supported."
        ),
    )

    @field_validator("file_pattern")
    @classmethod
    def validate_file_pattern(cls, v):
        if v:
            if "," in v:
                raise ValueError("File pattern cannot contain commas")
        return v


class IdentifiedSpans(BaseModel):
    file_path: str = Field(description="The file path where the relevant code is found.")
    start_line: int = Field(description="Starting line number of the relevant code section.")
    end_line: int = Field(description="Ending line number of the relevant code section.")


class Identify(ResponseSchema):
    """Identify if the provided search result is relevant to the reported issue."""

    thoughts: Optional[str] = Field(
        None,
        description="Your thoughts and analysis on the search results and how they relate to the reported issue.",
    )

    identified_spans: Optional[list[IdentifiedSpans]] = Field(
        default=None,
        description="Files and code sections in the search results identified as relevant to the reported issue.",
    )


class SearchBaseAction(Action):
    args_schema: ClassVar[Type[ActionArguments]] = SearchBaseArgs

    max_search_tokens: int = Field(
        20000,
        description="The maximum number of tokens allowed in the search results.",
    )
    max_identify_tokens: int = Field(
        20000,
        description="The maximum number of tokens allowed in the identified code sections.",
    )
    max_identify_prompt_tokens: int = Field(
        50000,
        description="The maximum number of tokens allowed in the identify prompt.",
    )
    max_hits: int = Field(
        10,
        description="The maximum number of search hits to display.",
    )
    progressive_disclosure: bool = Field(
        False,
        description="Return declaration skeletons first; use ViewCode for implementation bodies.",
    )
    progressive_max_results: int = Field(
        10,
        description="Maximum declaration skeletons returned in progressive disclosure mode.",
    )
    completion_model: BaseCompletionModel = Field(
        ...,
        description="The completion model used to identify relevant code sections in search results.",
    )

    _repository: Repository = PrivateAttr()
    _code_index: CodeIndex = PrivateAttr()
    
    def _ensure_entity_tools_env(self):
        """Ensure environment variables are set for entity_tools."""
        import os
        try:
            from code_intent.code_intent.entity_tools import set_current_instance
        except ImportError:
            set_current_instance = None
        
        # Get index_name from code_index if available
        index_name = None
        if self._code_index and hasattr(self._code_index, '_index_name'):
            index_name = self._code_index._index_name
        
        if index_name:
            if set_current_instance is not None:
                set_current_instance(index_name)
            os.environ["SWEBENCH_INSTANCE_NAME"] = index_name

    def __init__(
        self,
        repository: Repository = None,
        code_index: CodeIndex | None = None,
        **data,
    ):
        super().__init__(**data)
        self._repository = repository
        self._code_index = code_index

    @model_validator(mode="after")
    def initalize_model(self):
        if self.completion_model:
            self.completion_model.initialize(Identify, IDENTIFY_SYSTEM_PROMPT)
        return self

    def execute(
        self,
        args: SearchBaseArgs,
        file_context: FileContext | None = None,
        workspace: Workspace | None = None,
    ) -> Observation:
        if file_context is None:
            raise ValueError("File context must be provided to execute the search action.")

        properties = {"search_hits": [], "search_tokens": 0}

        search_result, alternative_suggestion = self._search_with_fallback(args)
        if not search_result.hits:
            properties["fail_reason"] = "no_search_hits"
            message = "No search results found."
            if args.file_pattern and args.file_pattern.startswith("/"):
                message += f" It seems you are using an absolute path in file_pattern ('{args.file_pattern}'). Please use a relative path from the repository root instead (e.g. without the leading slash)."
            return Observation(message=message, properties=properties)

        if self.progressive_disclosure:
            candidates = self._progressive_candidates(search_result)
            properties["search_hits"] = candidates
            properties["search_tokens"] = 0
            properties["disclosure_level"] = "candidates"
            message = self._format_progressive_candidates(
                search_result,
                candidates,
                alternative_suggestion,
            )
            return Observation(message=message, summary=message, properties=properties)

        search_result_context = self._search_result_to_context(search_result)
        properties["search_tokens"] = search_result_context.context_size()
        properties["search_hits"] = search_result_context.model_dump(exclude_none=True)

        completion = None
        if search_result_context.span_count() == 1 and search_result_context.context_size() > self.max_identify_tokens:
            logger.warning(
                f"{self.name}: Context for {search_result_context.create_summary()} is too large ({search_result_context.context_size()} tokens)."
            )
            properties["fail_reason"] = "search_too_large"
            return Observation(
                message="Search too large. Found a single code section that is too large to view. Please refine the search query.",
                properties=properties,
            )
        elif (
            search_result_context.context_size() > self.max_search_tokens and search_result_context.span_count() > 1
        ) or search_result_context.span_count() > self.max_hits:
            logger.info(
                f"{self.name}: Search too large. {properties['search_tokens']} tokens and {search_result_context.span_count()} hits, will ask for clarification."
            )

            view_context, completion = self._identify_code(args, search_result_context)
        else:
            view_context = search_result_context

        span_count = search_result_context.span_count()
        search_result_str = f"Found {span_count} code sections."

        # Apply previous changes to view context to ensure the agent sees the latest version
        for file in file_context.files:
            if view_context.has_file(file.file_path):
                view_file = view_context.get_file(file.file_path)
                # Ensure the base content matches the current state
                view_file._cached_base_content = file.get_base_content()
                if file.patch:
                    view_file.set_patch(file.patch)

        new_span_ids = file_context.add_file_context(view_context)

        if view_context.is_empty():
            search_result_str += "\n\nNone of the search results was relevant to the task."
            summary = "Didn't find any relevant code sections in the search results."
            message = search_result_str
        else:
            viewed_str = "that has already been viewed" if not new_span_ids else ""
            
            # Only create summary if tokens exceed 4000 threshold
            view_tokens = view_context.context_size()
            has_skeletons = any(any(s.skeleton for s in f.spans) for f in view_context.files)
            
            if view_tokens > 4000 or has_skeletons:
                if alternative_suggestion:
                    summary = f"Did not find an exact match but found the following alternative suggestions {viewed_str}:\n{view_context.create_summary()}"
                else:
                    summary = f"Found the following relevant code spans {viewed_str}:\n{view_context.create_summary()}"
            else:
                # For small contexts (<= 4000 tokens), don't use summary
                summary = None

            message = "Found the following relevant code:\n"

            message += view_context.create_prompt(
                show_span_ids=False,
                show_line_numbers=True,
                exclude_comments=False,
                show_outcommented_code=True,
                only_signatures=view_tokens > 4000 or has_skeletons,
            )

        properties["new_span_ids"] = new_span_ids

        logger.info(
            f"{self.name}: Found {span_count} code sections in search results. Viewed {view_context.span_count()} code sections."
        )

        return Observation(
            message=message,
            summary=summary,
            properties=properties,
            execution_completion=completion,
        )

    def _search_with_fallback(self, args: SearchBaseArgs) -> Tuple[SearchCodeResponse, bool]:
        alternative_suggestion = False
        search_result = self._search(args)
        if not search_result.hits:
            search_result = self._search_for_alternative_suggestion(args)
            alternative_suggestion = True
            logger.info(
                f"{self.name}: No relevant search results found. Will use alternative suggestion with {search_result.hits} hits."
            )

        return search_result, alternative_suggestion

    def _search_result_to_context(self, search_result: SearchCodeResponse) -> FileContext:
        search_result_context = FileContext(repo=self._repository)
        for hit in search_result.hits:
            for span in hit.spans:
                # IMPORTANT: Do not auto-add extra spans (e.g. imports / class init spans) when
                # rendering search results. For C++ this frequently pollutes the output with file
                # headers/guards and causes large range merges. The agent can request more context
                # via ViewCode if needed.
                search_result_context.add_span_to_context(
                    hit.file_path, span.span_id, add_extra=False, skeleton=span.skeleton
                )

        return search_result_context

    def _progressive_candidates(self, search_result: SearchCodeResponse) -> List[dict]:
        candidates = []
        for hit in search_result.hits:
            for span in hit.spans:
                candidates.append(
                    {
                        "file_path": hit.file_path,
                        "span_id": span.span_id,
                        "label": span.label,
                        "start_line": span.start_line,
                        "end_line": span.end_line,
                        "estimated_tokens": span.tokens,
                        "skeleton": span.skeleton,
                    }
                )
                if len(candidates) >= self.progressive_max_results:
                    return candidates
        return candidates

    def _format_progressive_candidates(
        self,
        search_result: SearchCodeResponse,
        candidates: List[dict],
        alternative_suggestion: bool,
    ) -> str:
        total = sum(len(hit.spans) for hit in search_result.hits)
        qualifier = "alternative " if alternative_suggestion else ""
        lines = [
            search_result.message or f"Found {total} {qualifier}candidate locations.",
            f"Showing declaration skeletons for {len(candidates)} of {total} {qualifier}candidate locations. "
            "Use ViewCode with the file and span_id (or line range) to inspect implementations.",
        ]
        for index, candidate in enumerate(candidates, 1):
            location = candidate["file_path"]
            start_line = candidate.get("start_line")
            end_line = candidate.get("end_line")
            if start_line:
                location += f":{start_line}"
                if end_line and end_line != start_line:
                    location += f"-{end_line}"
            label = candidate.get("label") or candidate["span_id"]
            lines.append(f"{index}. {label} - {location} [span_id: {candidate['span_id']}]")
            skeleton = candidate.get("skeleton")
            if skeleton:
                lines.append("   Declaration skeleton:")
                lines.extend(f"     {line}" for line in skeleton.strip().splitlines())
        if total > len(candidates):
            lines.append("Refine the symbol, namespace, or file filter to reveal other candidates.")
        return "\n".join(lines)

    def _select_span_instructions(self, search_result: SearchCodeResponse) -> str:
        if not self.add_to_context:
            return f"Here's the search result with the first line of codes in each code block. Use ViewCode to view specific code sections. "

        return f"The search result is too large. You must identify the relevant code sections in the search results to use them. "

    def _select_span_response_prompt(self, search_result: SearchCodeResponse) -> str:
        search_result_context = FileContext(repo=self._repository)
        for hit in search_result.hits:
            for span in hit.spans:
                search_result_context.add_span_to_context(hit.file_path, span.span_id, add_extra=False)

        search_result_str = search_result_context.create_prompt(
            show_span_ids=False,
            show_line_numbers=True,
            exclude_comments=False,
            show_outcommented_code=True,
            outcomment_code_comment="...",
            # only_signatures=True
        )

        prompt = self._select_span_instructions(search_result)
        prompt += f"\n<search_results>\n{search_result_str}\n</search_result>\n"
        return prompt

    def _search(self, args: SearchBaseArgs) -> SearchCodeResponse:
        raise NotImplementedError("Subclasses must implement this method.")

    def _search_for_alternative_suggestion(self, args: SearchBaseArgs) -> SearchCodeResponse:
        return SearchCodeResponse()

    def _identify_code(
        self, args: SearchBaseArgs, search_result_ctx: FileContext
    ) -> Tuple[IdentifiedSpans, Completion]:
        search_result_str = search_result_ctx.create_prompt(
            show_span_ids=True,
            show_line_numbers=True,
            exclude_comments=False,
            show_outcommented_code=True,
            outcomment_code_comment="...",
            max_tokens=self.max_identify_prompt_tokens,
            only_signatures=True,
        )

        content = "Search request:"
        content += f"\n{args.to_prompt()}"

        content += "\n\nIdentify the relevant code sections in the search results to use them. "
        content += f"\n\n<search_results>\n{search_result_str}\n</search_result>\n"
        identify_message = ChatCompletionUserMessage(role="user", content=content)

        messages = [identify_message]
        completion = None

        MAX_RETRIES = 3
        for retry_attempt in range(MAX_RETRIES):
            completion_response = self.completion_model.create_completion(messages=messages)
            
            if not completion_response or not completion_response.completion:
                logger.warning(f"Identification failed: No completion response on attempt {retry_attempt + 1}")
                continue

            logger.info(
                f"Identifying relevant code sections. Attempt {retry_attempt + 1} of {MAX_RETRIES}.{len(completion_response.structured_outputs) if completion_response.structured_outputs else 0} identify requests."
            )

            view_context = FileContext(repo=self._repository)
            # Inherit ALL files from search_result_ctx to ensure in-memory changes are visible
            for file_path, context_file in search_result_ctx._files.items():
                view_context.add_file(file_path, content=context_file.get_base_content(), add_extra=False)

            if not completion_response or not completion_response.structured_outputs:
                logger.warning("No identified code in response")
                return view_context, completion_response.completion if completion_response else None

            for identified_code in completion_response.structured_outputs:
                if identified_code.identified_spans:
                    for identified_spans in identified_code.identified_spans:
                        view_context.add_line_span_to_context(
                            identified_spans.file_path,
                            identified_spans.start_line,
                            identified_spans.end_line,
                            add_extra=True,
                        )

            tokens = view_context.context_size()

            if tokens > self.max_identify_tokens:
                logger.info(f"Identified code sections are too large ({tokens} tokens).")

                messages.append(
                    ChatCompletionAssistantMessage(role="assistant", content=identified_code.model_dump_json())
                )

                messages.append(
                    ChatCompletionUserMessage(
                        role="user",
                        content=f"The identified code sections are too large ({tokens} tokens). Maximum allowed is {self.max_search_tokens} tokens. "
                        f"Please identify a smaller subset of the most relevant code sections.",
                    )
                )
            else:
                logger.info(f"Identified code sections are within the token limit ({tokens} tokens).")
                return view_context, completion_response.completion

        # If we've exhausted all retries and still too large
        raise CompletionRejectError(
            f"Unable to reduce code selection to under {self.max_search_tokens} tokens after {MAX_RETRIES} attempts",
            last_completion=completion,
        )

    @classmethod
    def get_evaluation_criteria(cls, trajectory_length) -> List[str]:
        evaluation_criteria = super().get_evaluation_criteria(trajectory_length)
        evaluation_criteria.extend(
            [
                "Query Relevance: Evaluate if the search query or parameters are well-defined and likely to find relevant code.",
                "Search Scope Appropriateness: Check if the file patterns and class/function names narrow down the search effectively.",
                "Relevance of Search Results: Assess whether the search results are directly related to the problem and useful for making progress.",
                "Size of Search Results: Ensure that the code context provided is appropriately sized—not too large to overwhelm nor too small to be unhelpful.",
            ]
        )

        return evaluation_criteria

    @classmethod
    def get_reward_scale(cls, trajectory_length) -> List[RewardScaleEntry]:
        if trajectory_length <= 3:
            return cls.generate_reward_scale_entries(
                [
                    (
                        90,
                        100,
                        "The search action is excellent, with well-defined parameters yielding only highly relevant results.",
                    ),
                    (
                        75,
                        89,
                        "The search action is good, with reasonable parameters yielding relevant results.",
                    ),
                    (
                        25,
                        74,
                        "The search action have issues with parameters or yields few or no relevant results.",
                    ),
                    (
                        0,
                        24,
                        "The action is counterproductive, with search results that are entirely irrelevant or excessively large, causing setbacks.",
                    ),
                ]
            )
        else:
            return cls.generate_reward_scale_entries(
                [
                    (
                        90,
                        100,
                        "The search action significantly advances the solution, providing highly relevant and appropriately sized search results.",
                    ),
                    (
                        75,
                        89,
                        "The search action contributes positively towards solving the problem, with relevant results and minor issues.",
                    ),
                    (
                        50,
                        74,
                        "The search action is acceptable but may have issues with relevance or provides search results that are too large or too small.",
                    ),
                    (
                        25,
                        49,
                        "The search action provides results that are not helpful due to relevance or size issues.",
                    ),
                    (
                        0,
                        24,
                        "The search action has minimal impact, providing few relevant results.",
                    ),
                    (
                        -50,
                        -1,
                        "The action is counterproductive, with search results that are entirely irrelevant or excessively large, causing setbacks.",
                    ),
                ]
            )

    @classmethod
    def model_validate(cls, obj: Any) -> "SearchBaseAction":
        if isinstance(obj, dict):
            obj = obj.copy()
            repository = obj.pop("repository")
            code_index = obj.pop("code_index")
            completion_model = BaseCompletionModel.model_validate(obj.pop("completion_model"))

            return cls(code_index=code_index, repository=repository, completion_model=completion_model, **obj)
        return super().model_validate(obj)
