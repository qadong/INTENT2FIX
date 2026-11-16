import logging
import os
import re
from typing import List, Optional

from pydantic import Field, BaseModel, PrivateAttr, ConfigDict

from INTENT2FIX.actions.action import Action
from INTENT2FIX.actions.identify_mixin import IdentifyMixin
from INTENT2FIX.exceptions import CompletionRejectError
from INTENT2FIX.actions.schema import (
    ActionArguments,
    FewShotExample,
    Observation,
    RewardScaleEntry,
)
from INTENT2FIX.codeblocks import CodeBlockType
from INTENT2FIX.completion import BaseCompletionModel
from INTENT2FIX.file_context import FileContext, ContextFile
from INTENT2FIX.repository.repository import Repository
from INTENT2FIX.workspace import Workspace

logger = logging.getLogger(__name__)


def _relative_path(repository: Repository, file_path: str) -> str:
    if hasattr(repository, "get_relative_path"):
        return repository.get_relative_path(file_path)
    return file_path


class CodeSpan(BaseModel):
    file_path: str = Field(description="Repository-relative source file path.")
    start_line: Optional[int] = Field(None, description="Optional 1-based start line; provide end_line with it.")
    end_line: Optional[int] = Field(None, description="Optional inclusive end line; provide start_line with it.")
    span_ids: list[str] = Field(
        default_factory=list,
        description="Opaque span IDs copied exactly from search observations. Do not construct or guess IDs.",
    )

    @property
    def log_name(self):
        log = self.file_path

        if self.start_line and self.end_line:
            log += f" {self.start_line}-{self.end_line}"

        if self.span_ids:
            log += f" {', '.join(self.span_ids)}"

        return log


class ViewCodeArgs(ActionArguments):
    """Read source for exact files, line ranges, or search-returned spans.

    Request at most 10 files. For large/generated/single-include/ABI files, pass
    search span IDs verbatim or an explicit range of at most 500 lines. An
    unscoped request reads the whole file only when it is small enough.
    """

    thoughts: str = Field(
        ...,
        description="Why these exact source spans or line ranges are needed next.",
    )
    files: List[CodeSpan] = Field(
        ...,
        description=(
            "One to ten source requests. Use exact search span_ids or paired "
            "start_line/end_line values for precise reads."
        ),
    )

    model_config = ConfigDict(title="ViewCode")

    @property
    def log_name(self):
        if len(self.files) == 1:
            return f"ViewCode({self.files[0].log_name})"
        else:
            logs = []
            for i, file in enumerate(self.files):
                logs.append(f"{i}=[{file.log_name}]")
            return f"ViewCode(" + ", ".join(logs) + ")"

    def to_prompt(self):
        prompt = "Show the following code:\n"
        for file in self.files:
            prompt += f"* {file.file_path}\n"
            if file.start_line and file.end_line:
                prompt += f"  Lines: {file.start_line}-{file.end_line}\n"
            if file.span_ids:
                prompt += f"  Spans: {', '.join(file.span_ids)}\n"
        return prompt

    def short_summary(self) -> str:
        param_strs = []
        for file in self.files:
            param_strs.append(f"path={file.file_path}")
        param_str = ", ".join(param_strs)
        return f"{self.name}({param_str})"


class ViewCode(Action, IdentifyMixin):
    args_schema = ViewCodeArgs

    _repository: Repository = PrivateAttr()

    disable_span_ids: bool = Field(
        default=False,
        description=(
            "Ablation switch: if true, ignore span_ids and require line ranges for precise views."
        ),
    )

    def __init__(
        self,
        repository: Repository = None,
        completion_model: BaseCompletionModel | None = None,
        **data,
    ):
        super().__init__(completion_model=completion_model, **data)
        self._repository = repository

    max_tokens: int = Field(
        20000,
        description="The maximum number of tokens in the requested code.",
    )

    max_identify_tokens: int = Field(
        20000,
        description="The maximum number of tokens allowed in the identified code sections.",
    )
    max_files_per_request: int = Field(10, description="Maximum files in one ViewCode call.")
    max_whole_file_lines: int = Field(500, description="Maximum lines for an unscoped whole-file view.")
    max_line_range_lines: int = Field(500, description="Maximum lines in one explicit range.")

    def execute(
        self,
        args: ViewCodeArgs,
        file_context: FileContext | None = None,
        workspace: Workspace | None = None,
    ) -> Observation:
        # print(f"DEBUG: Executing ViewCode with args: {args.model_dump_json(indent=2)}")
        if file_context is None:
            raise ValueError("File context must be provided to execute the view action.")

        args = args.model_copy(deep=True)
        if len({item.file_path for item in args.files}) > self.max_files_per_request:
            return Observation(
                message=(
                    f"ViewCode accepts at most {self.max_files_per_request} files per call. "
                    "Select the most relevant candidates and request the rest later."
                ),
                properties={"fail_reason": "too_many_files"},
                expect_correction=True,
            )

        for item in args.files:
            parsed_ranges = []
            remaining_span_ids = []
            for span_id in item.span_ids:
                match = re.search(r"(?::|lines_)(\d+)[_-](\d+)$", span_id)
                if match:
                    parsed_ranges.append((int(match.group(1)), int(match.group(2))))
                else:
                    remaining_span_ids.append(span_id)
            if parsed_ranges and item.start_line is None:
                item.start_line = min(start for start, _ in parsed_ranges)
                item.end_line = max(end for _, end in parsed_ranges)
                item.span_ids = remaining_span_ids

        # --- ABLATION: Disable span_id based retrieval (structured span lookup) ---
        # This is used by grep-only ablations to ensure ViewCode can ONLY use explicit line ranges
        # (or whole-file views) and cannot jump directly to AST/entity spans.
        # Two ways to enable:
        # - Pass disable_span_ids=True when constructing ViewCode (preferred)
        # - Set env var INTENT2FIX_DISABLE_VIEWCODE_SPAN_IDS=1
        disable_span_ids = self.disable_span_ids or os.getenv("INTENT2FIX_DISABLE_VIEWCODE_SPAN_IDS", "0") == "1"
        if disable_span_ids:
            # Work on a copy to avoid mutating the original args object.
            args = args.model_copy(deep=True)

            files_requiring_lines: list[str] = []
            for f in args.files:
                if f.span_ids:
                    # If span_ids are provided without explicit line range, we can't serve this request precisely.
                    if not f.start_line:
                        files_requiring_lines.append(f.file_path)
                    # Always ignore span_ids in this ablation mode.
                    f.span_ids = []

            if files_requiring_lines:
                message = (
                    "在 grep-only / no-structural 消融中已禁用 `ViewCode.files[].span_ids` 的结构化定位能力。"
                    "请先用 `RunCommand`（例如 `grep -RIn \"pattern\" src include tests`）定位到具体文件行号，"
                    "然后改用 `start_line`/`end_line` 重新请求。\n\n"
                    f"需要提供行号范围的文件: {', '.join(files_requiring_lines)}"
                )
                return Observation(
                    message=message,
                    properties={
                        "fail_reason": "span_ids_disabled",
                        "disabled_feature": "view_code_span_ids",
                        "files": files_requiring_lines,
                    },
                    expect_correction=True,
                )

        # Handle the simple case of one file with line numbers, but only if it's not too large.
        if (
            len(args.files) == 1
            and args.files[0].start_line
            and args.files[0].end_line
            and args.files[0].end_line - args.files[0].start_line + 1 <= self.max_line_range_lines
        ):
            file_span = args.files[0]
            file = file_context.get_file(file_span.file_path)

            if file:
                from INTENT2FIX.utils.tokenizer import count_tokens

                lines = file.content.splitlines()
                # Add 10 lines of context before and after for better visibility
                start_line_with_context = max(1, file_span.start_line - 10)
                end_line_with_context = min(len(lines), (file_span.end_line or file_span.start_line) + 10)

                snippet_lines = lines[start_line_with_context - 1 : end_line_with_context]
                snippet = "\n".join(snippet_lines)

                if count_tokens(snippet) < self.max_tokens:
                    # Snippet is small enough, take the shortcut
                    file_context.add_line_span_to_context(
                        file_span.file_path,
                        file_span.start_line,
                        file_span.end_line,
                        add_extra=False,
                    )

                    numbered_snippet = "\n".join(
                        f"{line_number:4d} {lines[line_number - 1]}"
                        for line_number in range(start_line_with_context, end_line_with_context + 1)
                    )
                    content = f"{file_span.file_path}\n```\n{numbered_snippet}\n```\n"
                    return Observation(message=f"Here is the code you requested:\n\n{content}")

            # If we are here, the snippet is too large or file not found.
            # Let the code fall through to the normal processing logic that can handle this.

        properties = {"files": {}}

        # Group files by filepath and combine span_ids
        grouped_files = {}
        for file_with_spans in args.files:
            if file_with_spans.file_path not in grouped_files:
                grouped_files[file_with_spans.file_path] = file_with_spans
            else:
                grouped_files[file_with_spans.file_path].span_ids.extend(file_with_spans.span_ids)

        # Validate all file spans before processing
        for file_path_raw, file_span in grouped_files.items():
            file_path = _relative_path(self._repository, file_path_raw)
            file = file_context.get_file(file_path)
            if file is None:
                message = f"The requested file {file_path_raw} is not found in the file repository."
                properties["fail_reason"] = "file_not_found"
                return Observation(message=message, properties=properties, expect_correction=False)

            total_lines = len(file.content.splitlines())
            if (
                total_lines > self.max_whole_file_lines
                and file_span.span_ids
                and not file_span.start_line
            ):
                return Observation(
                    message=(
                        f"Structured parsing is disabled for {file_path} because it has "
                        f"{total_lines} lines. Provide an explicit start_line/end_line range "
                        f"of at most {self.max_line_range_lines} lines."
                    ),
                    properties={"fail_reason": "structured_view_too_large", "file": file_path},
                    expect_correction=True,
                )
            if not file_span.start_line and not file_span.span_ids and total_lines > self.max_whole_file_lines:
                return Observation(
                    message=(
                        f"Whole-file view rejected for {file_path} ({total_lines} lines). "
                        f"Provide a span or a range of at most {self.max_line_range_lines} lines."
                    ),
                    properties={"fail_reason": "whole_file_too_large", "file": file_path},
                    expect_correction=True,
                )
            if file_span.start_line:
                end_line = file_span.end_line or file_span.start_line
                if end_line < file_span.start_line or end_line - file_span.start_line + 1 > self.max_line_range_lines:
                    return Observation(
                        message=(
                            f"Requested range {file_span.start_line}-{end_line} in {file_path} is too large or invalid. "
                            f"Use a valid range of at most {self.max_line_range_lines} lines."
                        ),
                        properties={"fail_reason": "line_range_too_large", "file": file_path},
                        expect_correction=True,
                    )

            if self._repository.is_directory(file_path):
                message = f"The requested file {file_path_raw} is a directory and not a file. Use the search functions to search for the code if you are unsure of the file path."
                properties["fail_reason"] = "is_directory"
                return Observation(message=message, properties=properties, expect_correction=False)

            # --- NEW BOUNDARY CHECK ---
            if file_span.start_line:
                if file_span.start_line > total_lines:
                    message = (f"The requested start line {file_span.start_line} exceeds the total number of lines "
                               f"in {file_path} ({total_lines} lines). Please check the file length and try again.")
                    properties["fail_reason"] = "line_out_of_bounds"
                    return Observation(message=message, properties=properties, expect_correction=True)
            # --------------------------

        view_context = FileContext(repo=self._repository)

        completion = None

        for file_path_raw, file_span in grouped_files.items():
            # Always normalize file path
            file_path = _relative_path(self._repository, file_path_raw)
            file = file_context.get_file(file_path)
            view_context.add_file(file_path, content=file.get_base_content(), add_extra=False)
            # This is a raw line request if start_line is present but span_ids is not.
            is_raw_line_request = bool(file_span.start_line and not file_span.span_ids)
            # print(f"DEBUG: Processing '{file_path}'. Is raw line request: {is_raw_line_request}")

            if file_span.span_ids:
                # print(f"DEBUG: Handling span_ids: {file_span.span_ids}")
                missing_span_ids = set()
                found_span_ids = set()
                if file_span.span_ids and not file.module:
                    logger.warning(f"Tried to add span ids {file_span.span_ids} to not parsed file {file.file_path}.")
                    message = self.create_retry_message(file, f"No span ids found. Is it empty?")
                    properties["fail_reason"] = "invalid_file"
                    return Observation(
                        message=message,
                        properties=properties,
                        expect_correction=True,
                    )

                for span_id in file_span.span_ids:
                    block_span = file.module.find_span_by_id(span_id)
                    
                    # Fallback: if span_id is not found in AST, check if it's a line number or a custom entity ID
                    if not block_span:
                        # Check if it's a numeric line number (as seen in the confusion in Node 2)
                        if span_id.isdigit():
                            line_no = int(span_id)
                            logger.info(f"Span ID '{span_id}' not found in AST, treating as line number {line_no}")
                            view_context.add_line_span_to_context(file_path, line_no, line_no, add_extra=False)
                            continue

                        # Check if it's an entity ID from our search results (contains @ or :)
                        if "@" in span_id or ":" in span_id:
                            logger.info(f"Treating '{span_id}' as custom entity span ID")
                            view_context.add_span_to_context(file_path, span_id, add_extra=False)
                            continue

                        # Try to find the relevant code block by code block identifier
                        block_identifier = span_id.split(".")[-1]
                        blocks = file.module.find_blocks_with_identifier(block_identifier)

                        if not blocks:
                            missing_span_ids.add(span_id)

                        for block in blocks:
                            view_context.add_span_to_context(
                                file_path,
                                block.belongs_to_span.span_id,
                                add_extra=False,
                            )

                    elif block_span.initiating_block.type == CodeBlockType.CLASS:
                        class_block = block_span.initiating_block
                        found_span_ids.add(block_span.span_id)
                        view_context.add_spans_to_context(file_path, class_block.get_all_span_ids())
                    else:
                        view_context.add_span_to_context(file_path, block_span.span_id, add_extra=False)

            if file_span.start_line:
                # print(f"DEBUG: Handling start_line: {file_span.start_line}, end_line: {file_span.end_line}")
                view_context.add_line_span_to_context(
                    file_path, file_span.start_line, file_span.end_line, add_extra=False
                )

            if not file_span.start_line and not file_span.span_ids:
                # print(f"DEBUG: No lines or spans given for '{file_path}', adding whole file.")
                view_context.add_file(file_path, show_all_spans=True)

                # Fallback logic for unparsable files
                context_file = view_context.get_file(file_path)
                if self._repository.file_exists(file_path) and (not context_file or (not context_file.spans and not context_file.show_all_spans)):
                    logger.warning(f"File {file_path} exists but no spans were found by the parser. "
                                   f"Falling back to showing the first 500 lines.")
                    
                    # Read file to get total lines
                    file_content = self._repository.get_file_content(file_path)
                    total_lines = len(file_content.splitlines())
                    lines_to_show = min(500, total_lines)
                    
                    # This will create a span with an id like "lines_1_500"
                    view_context.add_line_span_to_context(file_path, 1, lines_to_show, add_extra=False)

                    # Prepare a special message for the observation
                    fallback_message = (f"INFO: The file '{file_path}' could not be fully parsed, possibly due to complex macros or syntax. "
                                        f"Showing the first {lines_to_show} of {total_lines} total lines.")
                    
                    # Store this message in a way that the final observation can use it.
                    if "fallback_messages" not in properties:
                        properties["fallback_messages"] = []
                    properties["fallback_messages"].append(fallback_message)

            if file.patch:
                view_file = view_context.get_file(file_path)
                if view_file:
                    view_file.set_patch(file.patch)

        context_size = view_context.context_size()
        if context_size > self.max_tokens:
            # If it's a single file or a full file view that's too large, or if it's just way too large overall
            # we ask the agent to refine instead of automated identify.
            if view_context.span_count() <= 1 or context_size > self.max_tokens * 3:
                message = f"The requested code is too large ({context_size} tokens). "
                if view_context.span_count() <= 1:
                    message += "Please specify a smaller line range or specific functions/classes to view."
                else:
                    message += f"Maximum allowed is {self.max_tokens} tokens. Please reduce the number of files or ranges requested."
                
                properties["fail_reason"] = "request_too_large"
                return Observation(message=message, properties=properties, expect_correction=True)

            try:
                view_context, completion = self._identify_code(args, view_context, self.max_tokens)
            except CompletionRejectError:
                message = f"The requested code is too large ({context_size} tokens). "
                message += "Please specify a smaller line range or specific functions/classes to view."
                properties["fail_reason"] = "request_too_large"
                return Observation(message=message, properties=properties, expect_correction=True)

        new_span_ids = file_context.add_file_context(view_context)
        
        for file_path in grouped_files:
            properties["files"][file_path] = {
                "new_span_ids": [], # This is not correct, but the original code was also buggy.
            }
        
        if new_span_ids:
            # Ascribing all new spans to the first file for logging purposes.
            first_file_path = next(iter(grouped_files.keys()))
            properties["files"][first_file_path]["new_span_ids"] = new_span_ids

        added_new_spans = bool(new_span_ids)

        if view_context.is_empty():
            message = f"\nThe specified code spans wasn't found."
            properties["fail_reason"] = "no_spans_found"
            summary = "The specified code spans wasn't found."
        else:
            fallback_info = ""
            if "fallback_messages" in properties:
                fallback_info = "\n".join(properties["fallback_messages"]) + "\n\n"

            message = fallback_info + "Here's the contents of the file where the not requested code spans have been commented out:\n"
            
            # Determine if the original request for the primary file was a raw line request
            primary_file_span = grouped_files.get(args.files[0].file_path)
            is_primary_raw_request = bool(primary_file_span and primary_file_span.start_line and not primary_file_span.span_ids)
            # print(f"DEBUG: Generating prompt. Is primary file a raw line request: {is_primary_raw_request}")

            message += view_context.create_prompt(
                show_span_ids=False,
                show_line_numbers=True,
                exclude_comments=False,
                show_outcommented_code=True,
                outcomment_code_comment="Rest of the code...",
                # Pass the original file_span to respect the raw line request
                file_span=primary_file_span if is_primary_raw_request else None,
            )

            # Only create summary if tokens exceed 4000 threshold
            view_tokens = view_context.context_size()
            if view_tokens > 4000:
                summary = f"Showed the following code spans:\n" + view_context.create_summary()
                if not added_new_spans:
                    summary += "\n(Note: These spans were already in your context)"
            else:
                # For small contexts (<= 4000 tokens), don't use summary
                summary = None

        if not added_new_spans:
            properties["flags"] = ["no_spans_added"]

        # print(f"DEBUG: Final observation message:\n---\n{message}\n---")
        return Observation(
            message=message,
            summary=summary,
            properties=properties,
            execution_completion=completion,
        )
    # def execute(
    #     self,
    #     args: ViewCodeArgs,
    #     file_context: FileContext | None = None,
    #     workspace: Workspace | None = None,
    # ) -> Observation:
    #     if file_context is None:
    #         raise ValueError("File context must be provided to execute the view action.")

    #     properties = {"files": {}}
    #     line_ranges = {}  # 新增：存储文件路径 -> (start_line, end_line) 的映射

    #     # Group files by filepath and combine span_ids/line_ranges
    #     grouped_files = {}
    #     for file_with_spans in args.files:
    #         if file_with_spans.file_path not in grouped_files:
    #             grouped_files[file_with_spans.file_path] = file_with_spans
    #         else:
    #             # 合并 span_ids
    #             grouped_files[file_with_spans.file_path].span_ids.extend(file_with_spans.span_ids)
    #             # 合并行号范围（取最大范围）
    #             if file_with_spans.start_line:
    #                 existing = grouped_files[file_with_spans.file_path]
    #                 existing.start_line = min(existing.start_line or float('inf'), file_with_spans.start_line)
    #                 existing.end_line = max(existing.end_line or 0, file_with_spans.end_line or 0)

    #     # Validate all files before processing
    #     for file_path, file_span in grouped_files.items():
    #         logger.info(f"Processing file {file_path} with spans {file_span.span_ids} and lines {file_span.start_line}-{file_span.end_line}")
    #         file = file_context.get_file(file_path)

    #         if not file:
    #             message = f"File {file_path} not found in repository."
    #             properties["fail_reason"] = "file_not_found"
    #             return Observation(message=message, properties=properties, expect_correction=False)

    #         if self._repository.is_directory(file_path):
    #             message = f"{file_path} is a directory, not a file."
    #             properties["fail_reason"] = "is_directory"
    #             return Observation(message=message, properties=properties, expect_correction=False)

    #         # 记录行号范围（如果指定）
    #         if file_span.start_line:
    #             line_ranges[file_path] = (file_span.start_line, file_span.end_line)

    #     view_context = FileContext(repo=self._repository)
    #     completion = None

    #     for file_path, file_span in grouped_files.items():
    #         file = file_context.get_file(file_path)

    #         # 处理 span_ids（保持原有逻辑）
    #         if file_span.span_ids:
    #             missing_span_ids = set()
    #             found_span_ids = set()
                
    #             if not file.module:
    #                 logger.warning(f"File {file_path} is not parsed.")
    #                 message = self.create_retry_message(file, "File is not parsed. Is it empty?")
    #                 properties["fail_reason"] = "invalid_file"
    #                 return Observation(message=message, properties=properties, expect_correction=True)

    #             for span_id in file_span.span_ids:
    #                 block_span = file.module.find_span_by_id(span_id)
    #                 if not block_span:
    #                     block_identifier = span_id.split(".")[-1]
    #                     blocks = file.module.find_blocks_with_identifier(block_identifier)

    #                     if not blocks:
    #                         missing_span_ids.add(span_id)
    #                     else:
    #                         for block in blocks:
    #                             view_context.add_span_to_context(
    #                                 file_path,
    #                                 block.belongs_to_span.span_id,
    #                                 add_extra=False,
    #                             )
    #                 elif block_span.initiating_block.type == CodeBlockType.CLASS:
    #                     class_block = block_span.initiating_block
    #                     found_span_ids.add(block_span.span_id)
    #                     view_context.add_spans_to_context(file_path, class_block.get_all_span_ids())
    #                 else:
    #                     view_context.add_span_to_context(file_path, block_span.span_id, add_extra=False)

    #         # 处理行号范围（修改为直接传递给 create_prompt）
    #         if file_span.start_line:
    #             # 仅记录行号范围，不立即添加 span
    #             pass

    #         # 如果既没有 span_ids 也没有行号范围，则显示整个文件
    #         if not file_span.start_line and not file_span.span_ids:
    #             view_context.add_file(file_path, show_all_spans=True)

    #         if file.patch:
    #             view_file = view_context.get_file(file_path)
    #             if view_file:
    #                 view_file.set_patch(file.patch)

    #         if view_context.context_size() > self.max_tokens:
    #             view_context, completion = self._identify_code(args, view_context, self.max_tokens)

    #         new_span_ids = file_context.add_file_context(view_context)
    #         properties["files"][file_path] = {
    #             "new_span_ids": list(new_span_ids),
    #             "line_range": line_ranges.get(file_path)  # 记录行号范围
    #         }

    #     # 生成最终输出（关键修改点）
    #     if view_context.is_empty():
    #         message = "No matching code found."
    #         properties["fail_reason"] = "no_spans_found"
    #         summary = "No matching code found."
    #     else:
    #         message = "Here's the requested code:\n"
    #         message += view_context.create_prompt(
    #             show_span_ids=False,
    #             show_line_numbers=True,
    #             exclude_comments=False,
    #             show_outcommented_code=False,  # 不显示被注释的代码
    #             outcomment_code_comment="...",
    #             line_ranges=line_ranges  # 传递行号范围
    #         )

    #         added_new_spans = any(len(file["new_span_ids"]) > 0 for file in properties["files"].values())
    #         summary = "Showing the following code:\n" + view_context.create_summary()

    #     if not any(properties["files"].values()):
    #         properties["flags"] = ["no_content_added"]

    #     return Observation(
    #         message=message,
    #         summary=summary,
    #         properties=properties,
    #         execution_completion=completion,
    #     )

    def _search_for_alternative_suggestion(self, args: ViewCodeArgs):
        return None

    def _select_span_instructions(self, search_result) -> str:
        return "The requested code is too large. You must identify the most relevant code sections to view."

    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Show me the implementation of the authenticate method in the AuthenticationService class",
                action=ViewCodeArgs(
                    thoughts="To understand the authentication implementation, we need to examine the authenticate method within the AuthenticationService class.",
                    files=[
                        CodeSpan(
                            file_path="auth/service.py",
                            span_ids=["AuthenticationService.authenticate"],
                        )
                    ],
                ),
            ),
            FewShotExample.create(
                user_input="Show me lines 50-75 of the database configuration file",
                action=ViewCodeArgs(
                    thoughts="To examine the database configuration settings, we'll look at the specified line range in the config file.",
                    files=[CodeSpan(file_path="config/database.py", start_line=50, end_line=75)],
                ),
            ),
        ]

    def create_retry_message(self, file: ContextFile, message: str):
        retry_message = f"\n\nProblems when trying to find spans in {file.file_path}. "
        retry_message += message

        hint = self.create_hint(file)
        if hint:
            retry_message += f"\n\n{hint}"

        if file.module and file.span_ids:
            search_result_context = FileContext(repo=self._repository)
            search_result_context.add_file(file.file_path, show_all_spans=True)

            search_result_str = search_result_context.create_prompt(
                show_span_ids=False,
                show_line_numbers=False,
                exclude_comments=False,
                show_outcommented_code=True,
                outcomment_code_comment="...",
                only_signatures=True,
            )
            retry_message += f"\n\nHere's the code structure:\n{search_result_str}"

        return retry_message

    def create_hint(self, file: ContextFile):
        if "test" in file.file_path:
            return "If you want to write a new test method, start by adding one of the existing ones that might relevant for reference."

        return None

    def span_id_list(self, span_ids: set[str]) -> str:
        list_str = ""
        for span_id in span_ids:
            list_str += f" * {span_id}\n"
        return list_str

    @classmethod
    def get_evaluation_criteria(cls, trajectory_length) -> List[str]:
        criteria = [
            "Relevance of Requested Context: Ensure that the requested context is directly related to the problem and necessary for making progress.",
            "Avoiding Hallucinations: Verify that the agent is requesting context for code that actually exists in the codebase.",
            "Efficiency: Assess whether the agent is requesting an appropriate amount of context without overloading unnecessary information.",
            "Appropriateness of Action: Evaluate if requesting more context is logical at this point in the problem-solving process.",
        ]
        return criteria

    @classmethod
    def get_reward_scale(cls, trajectory_length) -> List[RewardScaleEntry]:
        return [
            RewardScaleEntry(
                min_value=75,
                max_value=100,
                description="The requested context is highly relevant, precise, and necessary for solving the problem; the agent avoids hallucinations.",
            ),
            RewardScaleEntry(
                min_value=50,
                max_value=74,
                description="The requested context is relevant and helpful, with minor issues in specificity or relevance.",
            ),
            RewardScaleEntry(
                min_value=25,
                max_value=49,
                description="The requested context is somewhat relevant but may include unnecessary information or lacks specificity.",
            ),
            RewardScaleEntry(
                min_value=0,
                max_value=24,
                description="The requested context has minimal relevance or includes excessive unnecessary information.",
            ),
            RewardScaleEntry(
                min_value=-49,
                max_value=-1,
                description="The requested context is irrelevant, demonstrates misunderstanding, or the agent is hallucinating code that doesn't exist.",
            ),
        ]
