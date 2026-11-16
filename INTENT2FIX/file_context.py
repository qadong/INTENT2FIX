import difflib
import io
import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional, List, Dict, Set, Tuple, Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from unidiff import PatchSet
from unidiff.errors import UnidiffParseError

from INTENT2FIX.codeblocks import CodeBlockType, get_parser_by_path
from INTENT2FIX.codeblocks.codeblocks import (
    BlockSpan,
    CodeBlock,
    CodeBlockTypeGroup,
    SpanMarker,
    SpanType,
)
from INTENT2FIX.codeblocks.module import Module
from INTENT2FIX.repository import FileRepository
from INTENT2FIX.repository.repository import Repository
from INTENT2FIX.runtime.runtime import RuntimeEnvironment, TestResult
from INTENT2FIX.runtime.runtime import TestStatus
from INTENT2FIX.schema import FileWithSpans
from INTENT2FIX.utils.file import is_test
from INTENT2FIX.utils.tokenizer import count_tokens

logger = logging.getLogger(__name__)


class ContextSpan(BaseModel):
    span_id: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    tokens: Optional[int] = None
    pinned: bool = Field(
        default=False,
        description="Whether the span is pinned and cannot be removed from context",
    )
    skeleton: Optional[str] = None


@dataclass
class CurrentPromptSpan:
    span_id: Optional[str] = None
    tokens: int = 0


class ContextFile(BaseModel):
    """
    Represents the context of a file, managing patches that reflect changes over time.

    Attributes:
        file_path (str): The path to the file within the repository.
        accumulated_patch (Optional[str]): A Git-formatted patch representing all changes from the original content.
        patch (Optional[str]): A Git-formatted patch representing the latest changes applied in this ContextFile.
        spans (List[ContextSpan]): A list of spans associated with this file.
        show_all_spans (bool): A flag to indicate whether to display all spans.
    """

    file_path: str = Field(..., description="The relative path to the file within the repository.")
    patch: Optional[str] = Field(
        None,
        description="Git-formatted patch representing the latest changes applied in this ContextFile.",
    )
    spans: List[ContextSpan] = Field(
        default_factory=list,
        description="List of context spans associated with this file.",
    )
    show_all_spans: bool = Field(False, description="Flag to indicate whether to display all context spans.")
    was_edited: bool = Field(default=False, exclude=True)
    was_viewed: bool = Field(default=False, exclude=True)

    # Private attributes
    _initial_patch: Optional[str] = PrivateAttr(None)
    _cached_base_content: Optional[str] = PrivateAttr(None)
    _cached_content: Optional[str] = PrivateAttr(None)
    _cached_module: Optional[Module] = PrivateAttr(None)

    _repo: Repository = PrivateAttr()

    _cache_valid: bool = PrivateAttr(False)

    _is_new: bool = PrivateAttr(False)

    def __init__(
        self,
        repo: Optional[Repository],
        file_path: str,
        initial_patch: Optional[str] = None,
        **data,
    ):
        """
        Initializes the ContextFile instance.

        Args:
            repo (Optional[Repository]): The repository instance, can be None when reconstructing from dict
            file_path (str): The path to the file within the repository
            initial_patch (Optional[str]): A Git-formatted patch representing accumulated changes
            **data: Additional keyword arguments
        """
        super().__init__(file_path=file_path, **data)
        self._repo = repo
        self._initial_patch = initial_patch
        self._is_new = False if repo is None else not repo.file_exists(file_path)

    def _add_import_span(self):
        # TODO: Initiate module or add this lazily?
        if self.module:
            # Always include init spans like 'imports' to context file
            for child in self.module.children:
                if (child.type == CodeBlockType.IMPORT) and child.belongs_to_span.span_id:
                    self.add_span(child.belongs_to_span.span_id, pinned=True)

    def get_base_content(self) -> str:
        """
        Retrieves the base content of the file by applying the initial_patch to the original content.

        Returns:
            str: The base content of the file.

        Raises:
            FileNotFoundError: If the file does not exist in the repository.
            Exception: If applying the initial_patch fails.
        """
        if not self._repo:
            return None

        if self._cached_base_content is not None:
            return self._cached_base_content

        if not self._repo.file_exists(self.file_path):
            original_content = ""
        else:
            original_content = self._repo.get_file_content(self.file_path)

        if self._initial_patch:
            try:
                self._cached_base_content = self.apply_patch_to_content(original_content, self._initial_patch)
            except Exception as e:
                raise Exception(f"Failed to apply initial patch: {e}")
        else:
            self._cached_base_content = original_content

        return self._cached_base_content

    @property
    def module(self) -> Module | None:
        if not self._repo:
            return None

        if self._cached_module is not None:
            return self._cached_module

        parser = get_parser_by_path(self.file_path)
        if parser:
            self._cached_module = parser.parse(self.content, self.file_path)

        return self._cached_module

    @property
    def content(self) -> str:
        """
        Retrieves the current content of the file by applying the latest patch to the base content.

        Returns:
            str: The current content of the file.
        """
        if self._cached_content is not None:
            return self._cached_content

        base_content = self.get_base_content()
        if self.patch:
            try:
                self._cached_content = self.apply_patch_to_content(base_content, self.patch)
            except Exception as e:
                logger.error(f"Failed to apply patch: {self.patch}")
                raise e
        else:
            self._cached_content = base_content

        return self._cached_content

    def apply_changes(self, updated_content: str) -> set[str]:
        """
        Applies new content to the ContextFile by generating a patch between the base content and the new content.

        Args:
            updated_content (str): The new content to apply to the file.

        Returns:
            set[str]: Set of new span IDs added to context
        """
        self.was_edited = True

        base_content = self.get_base_content()
        new_patch = self.generate_patch(base_content, updated_content)
        self.patch = new_patch


        new_span_ids = set()

        # Track modified lines from patch
        patch_set = PatchSet(io.StringIO(new_patch))
        for patched_file in patch_set:
            for hunk in patched_file:
                # Get the line range for this hunk's changes
                modified_start = None
                modified_end = None

                for line in hunk:
                    if line.is_added or line.is_removed:
                        # Convert to 0-based line numbers
                        current_line = line.target_line_no if line.is_added else line.source_line_no
                        if current_line is not None:
                            if modified_start is None:
                                modified_start = current_line
                            modified_end = current_line

                if modified_start is not None:
                    # Add the modified line span to context
                    span_ids = self.add_line_span(modified_start, modified_end)
                    new_span_ids.update(span_ids)

        # Invalidate cached content
        self._cached_content = None
        self._cached_module = None

        return new_span_ids

    def apply_patch_to_content(self, content: str, patch: str) -> str:
        """
        Applies a Git-formatted patch to the given content.

        Args:
            content (str): The original content to apply the patch to.
            patch (str): The Git-formatted patch to apply.

        Returns:
            str: The patched content.

        Raises:
            Exception: If the patch does not contain changes for the specified file or if a context mismatch occurs.
        """
        try:
            patch_set = PatchSet(io.StringIO(patch))
        except UnidiffParseError as e:
            logger.error(f"Failed to parse patch for {self.file_path}: {e}")
            logger.debug(f"Invalid patch:\n{patch}")
            # Return original content if patch is invalid
            return content

        patched_content = content

        for patched_file in patch_set:
            patched_file_path = patched_file.path
            # Correctly strip 'a/' or 'b/' prefixes
            if patched_file_path.startswith("a/"):
                patched_file_path = patched_file_path[2:]
            elif patched_file_path.startswith("b/"):
                patched_file_path = patched_file_path[2:]
            if os.path.normpath(patched_file_path) == os.path.normpath(self.file_path):
                patched_content = self._apply_patched_file(patched_content, patched_file)
                break
        else:
            raise Exception(f"Patch does not contain changes for file {self.file_path}")

        return patched_content

    def _apply_patched_file(self, content: str, patched_file) -> str:
        """
        Applies a single patched file's hunks to the content.

        Args:
            content (str): The original content.
            patched_file: The patched file object from the PatchSet.

        Returns:
            str: The patched content.

        Raises:
            Exception: If there is a context mismatch during patch application.
        """
        content_lines = content.splitlines(keepends=True)
        new_content_lines = []
        line_no = 0

        for hunk in patched_file:
            try:
                # Copy unchanged lines before the hunk
                while line_no < hunk.source_start - 1 and line_no < len(content_lines):
                    new_content_lines.append(content_lines[line_no])
                    line_no += 1

                # Apply changes from the hunk
                for line in hunk:
                    if line.is_context:
                        if line_no >= len(content_lines):
                            raise Exception(
                                f"Patch context mismatch: Line {line_no} is beyond end of file ({len(content_lines)} lines)"
                            )
                        elif line.value.strip() and content_lines[line_no].strip() != line.value.strip():
                            raise Exception(
                                f'Patch context mismatch at line {line_no}: Expected "{line.value.strip()}", got "{content_lines[line_no].strip()}"'
                            )
                        new_content_lines.append(content_lines[line_no])
                        line_no += 1
                    elif line.is_added:
                        new_content_lines.append(line.value)
                    elif line.is_removed:
                        if line_no >= len(content_lines):
                            raise Exception(
                                f"Patch context mismatch: Cannot remove line {line_no} as it is beyond end of file"
                            )
                        line_no += 1

            except Exception as e:
                raise RuntimeError(
                    f"Failed to apply patch to {self.file_path}. Line number {line_no}. Hunk: {hunk}"
                ) from e

        # Copy any remaining lines after the last hunk
        if line_no < len(content_lines):
            new_content_lines.extend(content_lines[line_no:])

        return "".join(new_content_lines)

    def generate_full_patch(self) -> str:
        """
        Generates a full Git-formatted patch from the original content to the current content.

        Returns:
            str: The generated full patch as a string.
        """
        original_content = self._repo.get_file_content(self.file_path)
        current_content = self.content

        patch = self.generate_patch(original_content, current_content)
        return patch

    def generate_patch(self, old_content: str, new_content: str) -> str:
        """
        Generates a Git-formatted unified diff patch between old_content and new_content
        by calling the system's `diff -u` command.

        Args:
            old_content (str): The original content.
            new_content (str): The new content.

        Returns:
            str: The generated patch as a string.
        """
        old_tmp_file = None
        new_tmp_file = None
        patch_content = ""

        # Helper to strip trailing whitespace from a line while preserving its line ending.
        def rstrip_line_content(line):
            if line.endswith('\r\n'):
                return line[:-2].rstrip(' \t') + '\r\n'
            if line.endswith('\n'):
                return line[:-1].rstrip(' \t') + '\n'
            return line.rstrip(' \t')

        try:
            # Create temporary files for old and new content
            old_tmp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8')
            old_tmp_file.write(old_content)
            old_tmp_file.close()

            new_tmp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8')
            # Apply rstrip_line_content to each line of new_content before writing
            cleaned_new_content = "".join([rstrip_line_content(line) for line in new_content.splitlines(keepends=True)])
            new_tmp_file.write(cleaned_new_content)
            new_tmp_file.close()

            # Determine the 'fromfile' and 'tofile' arguments for diff.
            # For new files, 'fromfile' is conventionally /dev/null.
            from_file_label = f"a/{self.file_path}"
            if not old_content:
                from_file_label = "/dev/null"
            
            to_file_label = f"b/{self.file_path}"

            # Execute the system's diff command
            # -u: Generate a unified diff
            # -L: Set labels for the files in the diff header
            command = (
                f"diff -u -L {from_file_label} -L {to_file_label} "
                f"{old_tmp_file.name} {new_tmp_file.name}"
            )
            
            process = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                check=False # Do not raise an exception for non-zero exit codes (e.g., diff finds differences)
            )

            if process.returncode not in [0, 1]: # 0 for no diff, 1 for diff found
                logger.error(f"Diff command failed with exit code {process.returncode}: {process.stderr}")
                raise RuntimeError(f"Failed to generate patch: {process.stderr}")

            patch_content = process.stdout

        finally:
            # Clean up temporary files
            if old_tmp_file and os.path.exists(old_tmp_file.name):
                os.unlink(old_tmp_file.name)
            if new_tmp_file and os.path.exists(new_tmp_file.name):
                os.unlink(new_tmp_file.name)

        return patch_content

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        # Ensure these fields are excluded even if exclude=True is not in kwargs
        data.pop("was_edited", None)
        data.pop("was_viewed", None)
        # Ensure 'patch' is always included, even if it's None
        if "patch" not in data:
            data["patch"] = None
        return data

    @property
    def span_ids(self):
        return {span.span_id for span in self.spans}

    def to_prompt(
        self,
        show_span_ids=False,
        show_line_numbers=False,
        exclude_comments=False,
        show_outcommented_code=False,
        outcomment_code_comment: str = "...",
        show_all_spans: bool = False,
        only_signatures: bool = False,
        max_tokens: Optional[int] = None,
        file_span: Optional[BlockSpan] = None,  # Actually a CodeSpan from ViewCodeArgs
    ):
        # NEW: Optimized path for C++ Search Results using pre-computed skeletons
        if only_signatures and not show_all_spans and any(s.skeleton for s in self.spans):
            logger.info(f"Using pre-computed skeletons for {self.file_path}")
            contents = []
            for span in sorted(self.spans, key=lambda x: x.start_line or 0):
                if span.skeleton:
                    skeleton_content = span.skeleton.strip()
                    if not skeleton_content.endswith("\n"):
                        skeleton_content += "\n"
                    
                    # Add simple line number if requested
                    if show_line_numbers and span.start_line:
                        formatted = ""
                        for i, line in enumerate(skeleton_content.splitlines()):
                            formatted += f"{span.start_line + i:6}\t{line}\n"
                        skeleton_content = formatted
                    
                    contents.append(skeleton_content)
            
            if contents:
                code = "\n".join(contents)
                result = f"{self.file_path}\n```\n{code}\n```\n"
                return result

        # The line-based renderer is more robust for raw line requests or mixed content.
        # We force it if any span is a raw line span (starts with lines_), 
        # OR if we want to show everything but have no module.
        force_line_renderer = not self.module or any(
            s.span_id.startswith("lines_") for s in self.spans
        ) or (show_all_spans or self.show_all_spans)

        # New logic: Only use AST renderer (skeletons) for truly large "class-like" structures.
        # For structs, enums, unions, macros, and functions, we prefer the line renderer 
        # to ensure the full definition is visible.
        if not force_line_renderer and self.module:
            skeleton_worthy_types = {
                CodeBlockType.CLASS,
                CodeBlockType.NAMESPACE,
                CodeBlockType.TEMPLATE_CLASS,
            }

            for span in self.spans:
                # If it's a custom entity span or a path:line format, check if it's "skeleton worthy"
                if ":" in span.span_id or "@" in span.span_id:
                    is_skeleton_worthy = False
                    if span.start_line:
                        # Check what type of block is at this line
                        matched_spans = self.module.find_spans_by_line_numbers(span.start_line, span.end_line)
                        if not matched_spans:
                            # If no AST span exactly matches the line, it's safer to use line renderer
                            # (e.g. for macros or minor entities not indexed by the parser)
                            is_skeleton_worthy = False
                        else:
                            # Check if the most specific match (innermost) is skeleton-worthy
                            for ms in reversed(matched_spans):
                                if ms.block_type in skeleton_worthy_types:
                                    # When the token length of a class definition is less than 3000, 
                                    # display the complete structure directly (force line renderer).
                                    if ms.tokens < 3000:
                                        is_skeleton_worthy = False
                                        # Update span range to show the full block
                                        span.start_line = ms.start_line
                                        span.end_line = ms.end_line
                                    else:
                                        is_skeleton_worthy = True
                                    break
                                
                                # If the innermost structure is NOT skeleton worthy (e.g. macro, function), 
                                # then we definitely want the full view.
                                if ms.block_type and ms.block_type.group == CodeBlockTypeGroup.STRUCTURE:
                                    is_skeleton_worthy = False
                                    # For functions, structs, etc., update the span range to show the full block
                                    span.start_line = ms.start_line
                                    span.end_line = ms.end_line
                                    break
                    
                    if not is_skeleton_worthy:
                        # If the file contains a truly large structure (>= 3000 tokens), 
                        # we prefer to stay in AST mode so those large blocks can be skeletonized.
                        # Forcing line renderer on a file with mixed spans (one huge class, one small function)
                        # would cause the huge class to be fully expanded.
                        has_large_structure = any((s.tokens or 0) >= 3000 for s in self.spans)
                        if has_large_structure:
                            logger.info(f"Staying in AST renderer for {self.file_path} despite non-skeleton-worthy span {span.span_id}, because large structures are present.")
                            continue

                        logger.debug(f"Forcing line renderer for {self.file_path} because span {span.span_id} is not skeleton-worthy.")
                        force_line_renderer = True
                        break
                elif not any(ms.span_id == span.span_id for ms in self.module.spans_by_id.values()):
                    # If the span_id is not in the AST at all, use line renderer
                    force_line_renderer = True
                    break

        if force_line_renderer:
            # Ensure all spans have line numbers before using the line-based renderer.
            for span in self.spans:
                # IMPORTANT: Never let Tree-sitter override or "fill in" line numbers for Clang spans (id with ':')
                # unless they are somehow missing (which shouldn't happen now).
                if span.start_line is None and self.module and ":" not in span.span_id:
                    block_span = self.module.find_span_by_id(span.span_id)
                    if block_span:
                        span.start_line = block_span.start_line
                        span.end_line = block_span.end_line

            code = self._to_prompt_with_line_spans(
                show_span_id=show_span_ids, 
                show_line_numbers=show_line_numbers,
                file_span=file_span,
                show_all_spans=show_all_spans or self.show_all_spans,
                max_tokens=max_tokens
            )

        elif self.module:
            # Use the AST-based renderer only when no raw line spans are present.
            if not self.show_all_spans and not self.span_ids:
                logger.warning(
                    f"No span ids provided for {self.file_path}, returning empty."
                )
                return ""

            code = self._to_prompt(
                code_block=self.module,
                show_span_id=show_span_ids,
                show_line_numbers=show_line_numbers,
                outcomment_code_comment=outcomment_code_comment,
                show_outcommented_code=show_outcommented_code,
                exclude_comments=exclude_comments,
                show_all_spans=show_all_spans or self.show_all_spans,
                only_signatures=only_signatures,
                max_tokens=max_tokens,
                file_span=file_span,
            )
        else:
            # Fallback for no module and no raw spans
            code = ""

        result = f"{self.file_path}\n```\n{code}\n```\n"

        if max_tokens and count_tokens(result) > max_tokens:
            logger.warning(
                f"Content for {self.file_path} still exceeded max_tokens ({max_tokens}) after internal limiting."
            )

            # Fallback for oversized content: try only signatures if not already doing so
            if not only_signatures:
                logger.info(f"Falling back to signature-only view for {self.file_path} due to token limit.")
                return self.to_prompt(
                    show_span_ids=show_span_ids,
                    show_line_numbers=show_line_numbers,
                    exclude_comments=exclude_comments,
                    show_outcommented_code=show_outcommented_code,
                    outcomment_code_comment=outcomment_code_comment,
                    show_all_spans=show_all_spans,
                    only_signatures=True,
                    max_tokens=max_tokens,
                    file_span=file_span,
                )

            # If still too large, return a truncated version instead of empty string
            logger.info(f"Hard truncating content for {self.file_path} as a last resort.")
            truncated_code = code[:max_tokens * 2] # Very rough character-based truncation fallback
            return f"{self.file_path}\n```\n{truncated_code}\n... (truncated)\n```\n"

        return result

        return result

    def _find_span(self, codeblock: CodeBlock) -> Optional[ContextSpan]:
        # First, try to find an EXACT match (ID or exact line range)
        # This is critical to avoid matching the wrong macro on an adjacent line.
        if codeblock.belongs_to_span:
            for span in self.spans:
                if codeblock.belongs_to_span.span_id == span.span_id:
                    return span
                if ("@" in span.span_id or ":" in span.span_id) and span.span_id.startswith(f"{codeblock.belongs_to_span.span_id}"):
                    return span

        for span in self.spans:
            if span.start_line == codeblock.start_line and span.end_line == codeblock.end_line:
                return span
            
            # Match Clang-style spans that overlap significantly with the codeblock
            if span.start_line and span.end_line and ":" in span.span_id:
                if span.start_line <= codeblock.start_line and span.end_line >= codeblock.end_line:
                    # codeblock is inside the span range
                    return span

        # If no exact match, we only allow fuzzy match for large structures (Class, Namespace)
        # where minor offsets in the start/end lines are less likely to point to a completely different entity.
        if codeblock.type in [CodeBlockType.CLASS, CodeBlockType.NAMESPACE, CodeBlockType.TEMPLATE_CLASS]:
            for span in self.spans:
                if span.start_line and span.end_line:
                    if abs(span.start_line - codeblock.start_line) <= 1 and abs(span.end_line - codeblock.end_line) <= 1:
                        return span

        return None

    def _within_span(self, line_no: int) -> Optional[ContextSpan]:
        for span in self.spans:
            if (
                span.start_line
                and span.end_line
                and span.start_line <= line_no <= span.end_line
            ):
                return span
        return None

    def _to_prompt_with_line_spans(
        self, 
        show_span_id: bool = False, 
        show_line_numbers: bool = False,
        file_span: Optional[any] = None,
        show_all_spans: bool = False,
        max_tokens: Optional[int] = None
    ) -> str:
        content_lines = self.content.split("\n")

        if not self.spans and not show_all_spans:
            return ""

        # Pre-calculate active line ranges with context
        active_ranges = []
        if show_all_spans:
            active_ranges.append((1, len(content_lines)))
        else:
            # 1. Add ranges from spans
            for span in self.spans:
                if span.start_line:
                    # If it's a pinned span (like imports), we don't want the +10 context 
                    # to avoid merging with our actual search result.
                    context_size = 0 if span.pinned else 10
                    active_ranges.append((max(1, span.start_line - context_size), min(len(content_lines), (span.end_line or span.start_line) + context_size)))
            
            # 2. Add range from file_span filter (ViewCode specific)
            if file_span and hasattr(file_span, "start_line") and file_span.start_line:
                active_ranges.append((max(1, file_span.start_line - 10), min(len(content_lines), (file_span.end_line or file_span.start_line) + 10)))

        # Sort and merge overlapping ranges
        active_ranges.sort()
        merged_ranges = []
        if active_ranges:
            curr_start, curr_end = active_ranges[0]
            for next_start, next_end in active_ranges[1:]:
                if next_start <= curr_end + 1: # Merge if adjacent or overlapping
                    curr_end = max(curr_end, next_end)
                else:
                    merged_ranges.append((curr_start, curr_end))
                    curr_start, curr_end = next_start, next_end
            merged_ranges.append((curr_start, curr_end))

        def is_line_active(l_no):
            for r_start, r_end in merged_ranges:
                if r_start <= l_no <= r_end:
                    return True
            return False

        prompt_content = ""
        outcommented = True
        current_tokens = 0
        for i, line in enumerate(content_lines):
            line_no = i + 1

            if is_line_active(line_no):
                if outcommented and line_no > 1:
                    new_line = "... other code\n"
                    if max_tokens and current_tokens + count_tokens(new_line) > max_tokens:
                        break
                    prompt_content += new_line
                    current_tokens += count_tokens(new_line)
                
                line_prefix = f"{line_no:4d} " if show_line_numbers else ""
                new_line = line_prefix + line + "\n"
                
                if max_tokens and current_tokens + count_tokens(new_line) > max_tokens:
                    prompt_content += "... (truncated due to token limit)\n"
                    break
                    
                prompt_content += new_line
                current_tokens += count_tokens(new_line)
                outcommented = False
            else:
                outcommented = True

        return prompt_content

    def _has_line_span_in_block(self, block: CodeBlock) -> bool:
        if not self.spans:
            return False
        for span in self.spans:
            if span.start_line and block.start_line <= span.start_line <= block.end_line:
                return True
        return False

    def _to_prompt(
        self,
        code_block: CodeBlock,
        current_span: Optional[CurrentPromptSpan] = None,
        show_outcommented_code: bool = True,
        outcomment_code_comment: str = "...",
        show_span_id: bool = False,
        show_line_numbers: bool = False,
        exclude_comments: bool = False,
        show_all_spans: bool = False,
        only_signatures: bool = False,
        max_tokens: Optional[int] = None,
        current_tokens: int = 0,
        file_span: any = None,
    ):
        if current_span is None:
            current_span = CurrentPromptSpan()

        contents = ""
        if not code_block.children:
            return contents

        outcommented_block = None
        for _i, child in enumerate(code_block.children):
            if exclude_comments and child.type.group == CodeBlockTypeGroup.COMMENT:
                continue

            # Check if adding this block would exceed max_tokens
            if max_tokens:
                if current_tokens + child.tokens > max_tokens:
                    logger.debug(
                        f"Stopping at child block as it would exceed max_tokens"
                    )
                    break

            show_new_span_id = False
            show_child = False
            child_span = self._find_span(child)
            
            # #region agent log
            import json, time
            try:
                with open('./INTENT2FIX/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run3","hypothesisId":"H7","location":"INTENT2FIX/file_context.py:756","message":"Checking child block","data":{"type":child.type.value,"lines":f"{child.start_line}-{child.end_line}","has_span":child_span is not None,"only_signatures":only_signatures},"timestamp":int(time.time()*1000)}) + "\n")
            except: pass
            # #endregion

            if child_span:
                if only_signatures:
                    if child_span.skeleton:
                        # Use pre-computed skeleton if available
                        skeleton_content = child_span.skeleton.strip()
                        formatted_skeleton = ""
                        for i, line in enumerate(skeleton_content.splitlines()):
                            line_no = child.start_line + i
                            prefix = f"{line_no:6}\t" if show_line_numbers else ""
                            formatted_skeleton += f"{prefix}{child.indentation}{line}\n"
                        
                        contents += formatted_skeleton
                        current_tokens += count_tokens(formatted_skeleton)
                        continue
                    
                    elif child.type.group == CodeBlockTypeGroup.STRUCTURE and (child.end_line - child.start_line) > 5:
                        # Force skeletonize large structures even without pre-computed skeleton
                        block_content = child._to_prompt_string(
                            show_span_id=show_new_span_id,
                            show_line_numbers=show_line_numbers,
                            span_marker=SpanMarker.TAG,
                        )
                        lines = block_content.split("\n")
                        if len(lines) > 1:
                            truncated = lines[0] + "\n" + child.indentation + "    ... (implementation omitted)\n"
                            if child.content_lines[-1].strip() in ["}", "};"]:
                                truncated += child.indentation + child.content_lines[-1] + "\n"
                            contents += truncated
                            current_tokens += count_tokens(truncated)
                            continue

                if child_span.span_id != current_span.span_id:
                    show_child = True
                    show_new_span_id = show_span_id
                    current_span = CurrentPromptSpan(child_span.span_id)
                elif not child_span.tokens:
                    show_child = True
                else:
                    # Count all tokens in child block if it's not a structure (function or class) or a 'compound' (like an 'if' or 'for' clause)
                    if child.type.group == CodeBlockTypeGroup.IMPLEMENTATION and child.type not in [
                        CodeBlockType.COMPOUND,
                        CodeBlockType.DEPENDENT_CLAUSE,
                    ]:
                        child_tokens = child.sum_tokens()
                    else:
                        child_tokens = child.tokens

                    if current_span.tokens + child_tokens <= child_span.tokens:
                        show_child = True

                    current_span.tokens += child_tokens

            elif (
                not child.belongs_to_span
                or child.belongs_to_any_span not in self.spans
            ) and (child.has_any_span(self.span_ids) or self._has_line_span_in_block(child)):
                show_child = True

                if (
                    child.belongs_to_span
                    and current_span.span_id != child.belongs_to_span.span_id
                ):
                    show_new_span_id = show_span_id
                    current_span = CurrentPromptSpan(child.belongs_to_span.span_id)

            if self.show_all_spans or show_all_spans:
                show_child = True

            if only_signatures and child.type.group != CodeBlockTypeGroup.STRUCTURE:
                show_child = False

            if show_child:
                if outcommented_block:
                    block_content = outcommented_block._to_prompt_string(
                        show_line_numbers=show_line_numbers,
                    )
                    contents += block_content
                    current_tokens += count_tokens(block_content)
                    outcommented_block = None

                # In signature-only mode, we only want to show the header of the structure.
                # If it's a giant block from a coarse-grained parser, we must truncate.
                if only_signatures and child.type.group == CodeBlockTypeGroup.STRUCTURE and (child.end_line - child.start_line) > 5:
                    block_content = child._to_prompt_string(
                        show_span_id=show_new_span_id,
                        show_line_numbers=show_line_numbers,
                        span_marker=SpanMarker.TAG,
                    )
                    lines = block_content.split("\n")
                    if len(lines) > 1:
                        truncated = lines[0] + "\n" + child.indentation + "    ... (implementation omitted)\n"
                        if child.content_lines[-1].strip() in ["}", "};"]:
                            truncated += child.indentation + child.content_lines[-1] + "\n"
                        contents += truncated
                        current_tokens += count_tokens(truncated)
                        continue

                block_content = child._to_prompt_string(
                    show_span_id=show_new_span_id,
                    show_line_numbers=show_line_numbers,
                    span_marker=SpanMarker.TAG,
                )
                contents += block_content
                current_tokens += count_tokens(block_content)

                child_content = self._to_prompt(
                    code_block=child,
                    exclude_comments=exclude_comments,
                    show_outcommented_code=show_outcommented_code,
                    outcomment_code_comment=outcomment_code_comment,
                    show_span_id=show_span_id,
                    current_span=current_span,
                    show_line_numbers=show_line_numbers,
                    show_all_spans=show_all_spans,
                    only_signatures=only_signatures,
                    max_tokens=max_tokens,
                    current_tokens=current_tokens,
                )
                contents += child_content
                current_tokens += count_tokens(child_content)

            elif (
                show_outcommented_code
                and not outcommented_block
                and child.type
                not in [
                    CodeBlockType.COMMENT,
                    CodeBlockType.COMMENTED_OUT_CODE,
                    CodeBlockType.SPACE,
                ]
            ):
                outcommented_block = child.create_commented_out_block(
                    outcomment_code_comment
                )
                outcommented_block.start_line = child.start_line

        if show_outcommented_code and outcommented_block:
            block_content = outcommented_block._to_prompt_string(
                show_line_numbers=show_line_numbers,
            )
            contents += block_content
            current_tokens += count_tokens(block_content)

        return contents

    def set_patch(self, patch: str):
        self.patch = patch
        self._cached_content = None
        self._cached_module = None
        self.was_edited = True

    def context_size(self):
        if self.module:
            if self.span_ids is None:
                return self.module.sum_tokens()
            else:
                tokens = 0
                for span_id in self.span_ids:
                    span = self.module.find_span_by_id(span_id)
                    if span:
                        tokens += span.tokens
                    elif span_id.startswith("lines_"):
                        # For raw line spans, estimate tokens
                        parts = span_id.split("_")
                        if len(parts) >= 3:
                            try:
                                start = int(parts[1])
                                end = int(parts[2])
                                lines = self.content.splitlines()[start-1:end]
                                tokens += count_tokens("\n".join(lines))
                            except (ValueError, IndexError):
                                pass
                    elif ":" in span_id or "@" in span_id:
                        # For custom entity spans, we might have stored line numbers
                        span_obj = self.get_span(span_id)
                        if span_obj and span_obj.start_line and span_obj.end_line:
                            lines = self.content.splitlines()[span_obj.start_line-1:span_obj.end_line]
                            tokens += count_tokens("\n".join(lines))
                
                if tokens > 0:
                    return tokens
        
        # Fallback for no module or if no tokens were counted from spans
        if self.show_all_spans:
            return count_tokens(self.content)
        
        return count_tokens(self.to_prompt())

    def has_span(self, span_id: str):
        return span_id in self.span_ids

    def add_spans(
        self,
        span_ids: Set[str],
        tokens: Optional[int] = None,
        pinned: bool = False,
        add_extra: bool = True,
        skeleton: Optional[str] = None,
    ):
        for span_id in span_ids:
            self.add_span(span_id, tokens=tokens, pinned=pinned, add_extra=add_extra, skeleton=skeleton)

    def add_span(
        self,
        span_id: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        tokens: Optional[int] = None,
        pinned: bool = False,
        add_extra: bool = True,
        skeleton: Optional[str] = None,
    ) -> bool:
        self.was_viewed = True
        existing_span = next(
            (span for span in self.spans if span.span_id == span_id), None
        )

        if existing_span:
            existing_span.tokens = tokens
            existing_span.pinned = pinned
            if skeleton:
                existing_span.skeleton = skeleton
            return False
        else:
            block_span = self.module.find_span_by_id(span_id) if self.module else None
            if block_span:
                self.spans.append(
                    ContextSpan(
                        span_id=span_id,
                        start_line=start_line
                        if start_line is not None
                        else block_span.start_line,
                        end_line=end_line
                        if end_line is not None
                        else block_span.end_line,
                        tokens=tokens,
                        pinned=pinned,
                        skeleton=skeleton,
                    )
                )
                if add_extra:
                    self._add_class_span(block_span)
                return True
            else:
                # Try to parse span_id for custom formats (e.g., fq_name@{start_line}-{end_line})
                # This supports entity_tools span_id format: fq_name@{start_line}-{end_line}
                parsed_start_line = start_line
                parsed_end_line = end_line
                
                if "@" in span_id or ":" in span_id:
                    # NEW: For Clang Index spans, we DISABLE add_extra to prevent Tree-sitter pollution
                    # We also avoid AST range adjustment.
                    add_extra = False 
                    
                    try:
                        # Improved parsing for file_path:start-end format
                        if ":" in span_id:
                            parts = span_id.rsplit(":", 1)
                        else:
                            parts = span_id.rsplit("@", 1)
                            
                        if len(parts) == 2:
                            # Parse line range: "start_line-end_line" or just "line"
                            line_part = parts[1]
                            if "-" in line_part:
                                line_range = line_part.split("-", 1)
                                if len(line_range) == 2:
                                    try:
                                        parsed_start_line = int(line_range[0])
                                        parsed_end_line = int(line_range[1])
                                        
                                        # Handle -1 or other special values
                                        if parsed_start_line < 1:
                                            parsed_start_line = 1
                                        if parsed_end_line < parsed_start_line:
                                            parsed_end_line = parsed_start_line
                                    except ValueError:
                                        pass
                                else:
                                    # Fallback: only start_line provided
                                    parsed_start_line = int(line_range[0])
                                    parsed_end_line = parsed_start_line
                            else:
                                # Fallback: only start_line
                                try:
                                    parsed_start_line = int(line_part)
                                    parsed_end_line = parsed_start_line
                                except ValueError:
                                    pass
                    except (ValueError, IndexError):
                        pass
                
                # If we have line information, create the span directly
                if parsed_start_line is not None:
                    # For Clang spans, we TRUST the line number and do NOT adjust it using AST.
                    new_span = ContextSpan(
                        span_id=span_id,
                        start_line=parsed_start_line,
                        end_line=parsed_end_line or parsed_start_line,
                        tokens=tokens,
                        pinned=pinned,
                        skeleton=skeleton,
                    )
                    self.spans.append(new_span)
                    return True
                else:
                    logger.warning(
                        f"Tried to add not existing span id {span_id} in file {self.file_path} "
                        f"and could not parse line information"
                    )
                    return False

    def _add_class_span(self, span: BlockSpan):
        if span.initiating_block.type != CodeBlockType.CLASS:
            class_block = span.initiating_block.find_type_in_parents(
                CodeBlockType.CLASS
            )
        elif span.initiating_block.type == CodeBlockType.CLASS:
            class_block = span.initiating_block
        else:
            return

        if not class_block or self.has_span(class_block.belongs_to_span.span_id):
            return

        # Always add init spans like constructors to context
        for child in class_block.children:
            if (
                child.belongs_to_span.span_type == SpanType.INITATION
                and child.belongs_to_span.span_id
                and not self.has_span(child.belongs_to_span.span_id)
            ):
                if child.belongs_to_span.span_id not in self.span_ids:
                    self.spans.append(
                        ContextSpan(span_id=child.belongs_to_span.span_id)
                    )

        if class_block.belongs_to_span.span_id not in self.span_ids:
            self.spans.append(ContextSpan(span_id=class_block.belongs_to_span.span_id))

    def add_line_span(
        self, start_line: int, end_line: int | None = None, add_extra: bool = False
    ) -> list[str]:
        self.was_viewed = True
        end_line = end_line or start_line

        # Always create a raw line span to precisely control the context for line-based requests.
        logger.info(
            f"Creating raw line span for {start_line}-{end_line} in {self.file_path}."
        )
        span_id = f"lines_{start_line}_{end_line}"

        if not any(s.span_id == span_id for s in self.spans):
            self.spans.append(
                ContextSpan(
                    span_id=span_id, start_line=start_line, end_line=end_line
                )
            )
            return [span_id]

        return []

    def lines_is_in_context(self, start_line: int, end_line: int) -> bool:
        """
        Check if the given line range's start and end points are covered by spans in the context.
        """
        if self.show_all_spans:
            return True

        start_covered = False
        end_covered = False

        for span in self.spans:
            # 1. Check direct line numbers (for raw line spans like lines_X_Y or Clang spans)
            if span.start_line is not None and span.end_line is not None:
                if span.start_line <= start_line <= span.end_line:
                    start_covered = True
                if span.start_line <= end_line <= span.end_line:
                    end_covered = True
                if start_covered and end_covered:
                    return True

            # 2. Fallback to AST blocks (if available)
            if self.module:
                block_span = self.module.find_span_by_id(span.span_id)
                if block_span:
                    if block_span.start_line <= start_line <= block_span.end_line:
                        start_covered = True
                    if block_span.start_line <= end_line <= block_span.end_line:
                        end_covered = True
                    if start_covered and end_covered:
                        return True

        return False

    def remove_span(self, span_id: str):
        self.spans = [span for span in self.spans if span.span_id != span_id]

    def remove_all_spans(self):
        self.spans = [span for span in self.spans if span.pinned]

    def get_spans(self) -> List[BlockSpan]:
        block_spans = []
        for span in self.spans:
            if not self.module:
                continue

            block_span = self.module.find_span_by_id(span.span_id)
            if block_span:
                block_spans.append(block_span)
        return block_spans

    def get_block_span(self, span_id: str) -> Optional[BlockSpan]:
        if not self.module:
            return None
        for span in self.spans:
            if span.span_id == span_id:
                block_span = self.module.find_span_by_id(span_id)
                if block_span:
                    return block_span
                else:
                    logger.warning(f"Could not find span with id {span_id} in file {self.file_path}")
        return None

    def get_span(self, span_id: str) -> Optional[ContextSpan]:
        for span in self.spans:
            if span.span_id == span_id:
                return span
        return None

    def get_patches(self) -> List[str]:
        """
        Get all patches associated with this ContextFile.
        """
        return self.patches

    @property
    def is_new(self) -> bool:
        """
        Returns whether this file is newly created in the context.

        Returns:
            bool: True if the file is new, False otherwise
        """
        return self._is_new


class TestFile(BaseModel):
    file_path: str = Field(..., description="The path to the test file.")
    test_results: List[TestResult] = Field(default_factory=list, description="List of test results.")


class FileContext(BaseModel):
    _repo: Repository | None = PrivateAttr(None)
    _runtime: RuntimeEnvironment = PrivateAttr(None)

    _files: Dict[str, ContextFile] = PrivateAttr(default_factory=dict)
    _test_files: Dict[str, TestFile] = PrivateAttr(default_factory=dict)  # Changed to Dict
    _max_tokens: int = PrivateAttr(default=20000)
    test_output: str = Field(default="", description="Test output for the context.")
    change_seq: int = Field(
        default=0,
        description="Monotonic counter incremented whenever the agent modifies files in this context. Used to gate VerifiedFinish.",
    )
    verification: Dict[str, Any] = Field(
        default_factory=dict,
        description="Verification state for gating finish (e.g., last passing RunRegression/RunSingleTest results and which change_seq they validated).",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        repo: Repository | None,
        runtime: RuntimeEnvironment | None = None,
        test_output: str = "",
        **data,
    ):
        super().__init__(test_output=test_output,**data)

        self._repo = repo
        self._runtime = runtime

        if "_files" not in self.__dict__:
            self.__dict__["_files"] = {}

        if "_test_files" not in self.__dict__:
            self.__dict__["_test_files"] = {}

        if "_max_tokens" not in self.__dict__:
            self.__dict__["_max_tokens"] = data.get("max_tokens", 20000)

    @classmethod
    def from_dir(cls, repo_dir: str, max_tokens: int = 20000):
        from INTENT2FIX.repository.file import FileRepository

        repo = FileRepository(repo_path=repo_dir)
        instance = cls(max_tokens=max_tokens, repo=repo)
        return instance

    @classmethod
    def from_json(cls, repo_dir: str, json_data: str):
        """
        Create a FileContext instance from JSON data.

        :param repo_dir: The repository directory path.
        :param json_data: A JSON string representing the FileContext data.
        :return: A new FileContext instance.
        """
        data = json.loads(json_data)
        return cls.from_dict(data, repo_dir=repo_dir)

    @classmethod
    def from_dict(
        cls,
        data: Dict,
        repo_dir: str | None = None,
        repo: Repository | None = None,
        runtime: RuntimeEnvironment | None = None,
    ):
        if not repo and repo_dir:
            repo = FileRepository(repo_path=repo_dir)
        instance = cls(
            max_tokens=data.get("max_tokens", 20000),
            repo=repo,
            runtime=runtime,
            change_seq=data.get("change_seq", 0),
            verification=data.get("verification", {}) or {},
        )
        instance.load_files_from_dict(data.get("files", []), test_files=data.get("test_files", []))
        return instance

    def load_files_from_dict(self, files: list[dict], test_files: list[dict] | None = None):
        """
        Loads files and test files from a dictionary representation.

        Args:
            files (list[dict]): List of file data dictionaries
            test_files (list[dict] | None): List of test file data dictionaries
        """
        # Load regular files
        for file_data in files:
            file_path = file_data["file_path"]
            show_all_spans = file_data.get("show_all_spans", False)
            spans = [ContextSpan(**span) for span in file_data.get("spans", [])]

            self._files[file_path] = ContextFile(
                file_path=file_path,
                spans=spans,
                show_all_spans=show_all_spans,
                patch=file_data.get("patch"),
                repo=self._repo,
            )

        # Load test files
        if test_files:
            for test_file_data in test_files:
                file_path = test_file_data["file_path"]
                self._test_files[file_path] = TestFile(**test_file_data)

    def model_dump(self, **kwargs):
        """
        Dumps the model to a dictionary, including files and test files.
        """
        if "exclude_none" not in kwargs:
            kwargs["exclude_none"] = True

        files = [file.model_dump(**kwargs) for file in self._files.values()]
        test_files = [test_file.model_dump(**kwargs) for test_file in self._test_files.values()]

        return {
            "max_tokens": self.__dict__["_max_tokens"],
            "change_seq": self.change_seq,
            "verification": self.verification,
            "files": files,
            "test_files": test_files,
        }

    def snapshot(self):
        dict = self.model_dump()
        del dict["max_tokens"]
        return dict

    def restore_from_snapshot(self, snapshot: dict):
        self._files.clear()
        self._test_files.clear()
        # Restore verification bookkeeping if present
        self.change_seq = snapshot.get("change_seq", self.change_seq)
        self.verification = snapshot.get("verification", self.verification) or {}
        self.load_files_from_dict(snapshot.get("files", []))

    def to_files_with_spans(self) -> List[FileWithSpans]:
        return [
            FileWithSpans(file_path=file_path, span_ids=list(file.span_ids)) for file_path, file in self._files.items()
        ]

    def add_files_with_spans(self, files_with_spans: List[FileWithSpans]):
        for file_with_spans in files_with_spans:
            self.add_spans_to_context(file_with_spans.file_path, set(file_with_spans.span_ids))

    def add_file(self, file_path: str, content: str | None = None, show_all_spans: bool = False, add_extra: bool = True) -> ContextFile:
        if self._repo and hasattr(self._repo, "get_relative_path"):
            file_path = self._repo.get_relative_path(file_path)

        if file_path not in self._files:
            self._files[file_path] = ContextFile(
                file_path=file_path,
                spans=[],
                show_all_spans=show_all_spans,
                repo=self._repo,
            )
            
            if content is not None:
                # Set the initial content for dynamically created files
                self._files[file_path]._cached_base_content = content
                self._files[file_path]._cached_content = content

            if add_extra:
                self._files[file_path]._add_import_span()
        else:
            if show_all_spans:
                self._files[file_path].show_all_spans = True
            if content is not None:
                self._files[file_path]._cached_base_content = content
                self._files[file_path]._cached_content = content

        return self._files[file_path]

    def add_file_with_lines(self, file_path: str, start_line: int, end_line: Optional[int] = None):
        end_line = end_line or start_line
        if file_path not in self._files:
            self.add_file(file_path)

        self._files[file_path].add_line_span(start_line, end_line)

    def remove_file(self, file_path: str):
        if file_path in self._files:
            del self._files[file_path]

    def exists(self, file_path: str):
        return file_path in self._files

    @property
    def has_runtime(self):
        return bool(self._runtime)

    @property
    def files(self):
        return list(self._files.values())

    @property
    def test_files(self):
        return list(self._test_files.values())

    def add_spans_to_context(
        self,
        file_path: str,
        span_ids: Set[str],
        tokens: Optional[int] = None,
        pinned: bool = False,
        add_extra: bool = True,
        skeleton: Optional[str] = None,
    ):
        if not self.has_file(file_path):
            context_file = self.add_file(file_path)
        else:
            context_file = self.get_context_file(file_path)

        if context_file:
            context_file.add_spans(span_ids, tokens, pinned=pinned, add_extra=add_extra, skeleton=skeleton)
        else:
            logger.warning(f"Could not find file {file_path} in the repository")

    def add_span_to_context(
        self,
        file_path: str,
        span_id: str,
        tokens: Optional[int] = None,
        pinned: bool = False,
        add_extra: bool = True,
        skeleton: Optional[str] = None,
    ) -> bool:
        if not self.has_file(file_path):
            context_file = self.add_file(file_path)
        else:
            context_file = self.get_context_file(file_path)

        if context_file:
            return context_file.add_span(span_id, tokens=tokens, pinned=pinned, add_extra=add_extra, skeleton=skeleton)
        else:
            logger.warning(f"Could not find file {file_path} in the repository")
            return False
    def add_line_span_to_context(
        self,
        file_path: str,
        start_line: int,
        end_line: int | None = None,
        add_extra: bool = True,
    ) -> List[str]:
        if not self.has_file(file_path):
            context_file = self.add_file(file_path, add_extra=add_extra)
        else:
            context_file = self.get_context_file(file_path)

        if context_file:
            return context_file.add_line_span(start_line, end_line, add_extra=add_extra)
        else:
            logger.warning(f"Could not find file {file_path} in the repository")
            return []


    def remove_span_from_context(self, file_path: str, span_id: str, remove_file: bool = False):
        context_file = self.get_context_file(file_path)
        if context_file:
            context_file.remove_span(span_id)

            if not context_file.spans and remove_file:
                self.remove_file(file_path)

    def remove_spans_from_context(self, file_path: str, span_ids: List[str], remove_file: bool = False):
        for span_id in span_ids:
            self.remove_span_from_context(file_path, span_id, remove_file)

    def get_spans(self, file_path: str) -> List[BlockSpan]:
        context_file = self.get_context_file(file_path)
        if context_file:
            return context_file.get_spans()
        return []

    def get_span(self, file_path: str, span_id: str) -> Optional[BlockSpan]:
        context_file = self.get_context_file(file_path)
        if context_file:
            return context_file.get_block_span(span_id)
        return None

    def has_span(self, file_path: str, span_id: str):
        context_file = self.get_context_file(file_path)
        if context_file:
            return span_id in context_file.span_ids
        return False

    def apply(self, file_context: "FileContext"):
        """
        Apply a list of FileContext instances, collecting their ContextFiles.
        """
        for context_file in file_context.files:
            file_path = context_file.file_path
            if file_path not in self.files:
                self._files[file_path] = ContextFile(
                    file_path=file_path,
                    repo=self.repo,
                    initial_patch=file_context.generate_full_patch(),
                )

            self._files[file_path].spans.extend(context_file.spans)
            self._files[file_path].show_all_spans = context_file.show_all_spans

    def has_file(self, file_path: str):
        if self._repo and hasattr(self._repo, "get_relative_path"):
            file_path = self._repo.get_relative_path(file_path)
        return file_path in self._files and (self._files[file_path].spans or self._files[file_path].show_all_spans)

    def get_file(self, file_path: str) -> Optional[ContextFile]:
        return self.get_context_file(file_path)

    def file_exists(self, file_path: str):
        context_file = self._files.get(file_path)
        return context_file or self._repo.file_exists(file_path)

    def is_directory(self, file_path: str):
        return self._repo.is_directory(file_path)

    def get_context_file(self, file_path: str, add_extra: bool = False) -> Optional[ContextFile]:
        if self._repo and hasattr(self._repo, "get_relative_path"):
            file_path = self._repo.get_relative_path(file_path)

        context_file = self._files.get(file_path)

        if not context_file:
            if not self._repo.file_exists(file_path):
                logger.info(f"get_context_file({file_path}) File not found")
                return None

            if self._repo.is_directory(file_path):
                logger.info(f"get_context_file({file_path}) File is a directory")
                return None

            self.add_file(file_path, add_extra=add_extra)
            context_file = self._files[file_path]

        return context_file

    def get_context_files(self) -> List[ContextFile]:
        return list(self._files.values())

    def context_size(self):
        if self._repo:
            has_skeletons = any(any(s.skeleton for s in f.spans) for f in self._files.values())
            content = self.create_prompt(
                show_span_ids=False,
                show_line_numbers=True,
                show_outcommented_code=True,
                outcomment_code_comment="...",
                only_signatures=has_skeletons,
            )
            return count_tokens(content)

        # TODO: This doesnt give accure results. Will count tokens in the generated prompt instead
        # sum(file.context_size() for file in self._files.values())
        return 0

    def available_context_size(self):
        return self._max_tokens - self.context_size()

    def save_file(self, file_path: str, updated_content: Optional[str] = None):
        self._repo.save_file(file_path, updated_content)

    def reset(self):
        self._files = {}
        self._test_files = {}

    def is_empty(self):
        return not self._files

    def strip_line_breaks_only(self, text):
        return text.lstrip("\n\r").rstrip("\n\r")

    def create_prompt(
        self,
        show_span_ids=False,
        show_line_numbers=False,
        exclude_comments=False,
        show_outcommented_code=False,
        outcomment_code_comment: str = "...",
        files: set | None = None,
        only_signatures: bool = False,
        max_tokens: Optional[int] = None,
        file_span: Optional[BlockSpan] = None,
    ):
        file_contexts = []
        current_tokens = 0

        for context_file in self.get_context_files():
            if not files or context_file.file_path in files:
                # If a specific file_span is provided, ensure it only applies to the matching file
                effective_file_span = None
                if file_span and hasattr(file_span, "file_path") and file_span.file_path == context_file.file_path:
                    effective_file_span = file_span

                content = context_file.to_prompt(
                    show_span_ids,
                    show_line_numbers,
                    exclude_comments,
                    show_outcommented_code,
                    outcomment_code_comment,
                    only_signatures=only_signatures,
                    max_tokens=max_tokens,
                    file_span=effective_file_span,
                )

                if max_tokens:
                    content_tokens = count_tokens(content)
                    if current_tokens + content_tokens > max_tokens:
                        logger.warning(f"Skipping {context_file.file_path} as it would exceed max_tokens")
                        break
                    current_tokens += content_tokens

                if content:  # Only add non-empty content
                    file_contexts.append(content)

        return "\n\n".join(file_contexts)


    def clone(self):
        dump = self.model_dump(exclude={"files": {"__all__": {"was_edited", "was_viewed"}}})
        # Preserve verification bookkeeping across node expansions
        cloned_context = FileContext(
            repo=self._repo,
            runtime=self._runtime,
            change_seq=dump.get("change_seq", 0),
            verification=dump.get("verification", {}) or {},
        )
        cloned_context.load_files_from_dict(files=dump.get("files", []), test_files=dump.get("test_files", []))
        return cloned_context

    def has_patch(self, ignore_tests: bool = False):
        return any(file.patch for file in self._files.values() if not ignore_tests or not is_test(file.file_path))

    def has_test_patch(self):
        return any(file.patch for file in self._files.values() if is_test(file.file_path))

    def generate_git_patch(self, ignore_tests: bool = False) -> str:
        """
        Generates a full patch for all files with changes in the FileContext.
        The patch is formatted like a git diff.

        Returns:
            str: A git-diff-like patch string containing all file changes.
        """
        full_patch = []
        for file_path, context_file in self._files.items():
            if ignore_tests and is_test(file_path):
                continue
            if context_file.patch:
                full_patch.append(context_file.patch)

        return "\n".join(full_patch)

    def get_updated_files(self, old_context: "FileContext", include_patches: bool = True) -> set[str]:
        """
        Compares this FileContext with an older one and returns a set of files that have been updated.
        Updates include content changes, span additions/removals, and file additions.

        Args:
            old_context: The previous FileContext to compare against

        Returns:
            set[str]: Set of file paths that have different content or spans between the two contexts
        """
        updated_files = set()

        # Check files in current context
        for file_path, current_file in self._files.items():
            old_file = old_context._files.get(file_path)

            if old_file is None:
                # New file added
                updated_files.add(file_path)
            else:
                # Check for content changes
                if include_patches and current_file.content != old_file.content:
                    updated_files.add(file_path)
                    continue

                # Check for span changes
                current_spans = current_file.span_ids
                old_spans = old_file.span_ids
                if current_spans != old_spans:
                    updated_files.add(file_path)

        return updated_files

    def get_context_diff(self, old_context: "FileContext") -> "FileContext":
        diff_context = FileContext(repo=self._repo)

        for file_path, current_file in self._files.items():
            old_file = old_context._files.get(file_path)

            if old_file is None:
                logger.info(f"File {file_path} is new")
                diff_context._files[file_path] = current_file
            else:
                current_spans = current_file.span_ids
                old_spans = old_file.span_ids
                new_spans = current_spans - old_spans
                if new_spans:
                    diff_context._files[file_path] = ContextFile(
                        file_path=file_path,
                        repo=self._repo,
                    )
                    diff_context._files[file_path].spans.extend(
                        [span for span in current_file.spans if span.span_id in new_spans]
                    )
        return diff_context
    
    
    def get_patched_files(self, include_new_files: bool = True) -> List[str]:
        """
        Returns a list of file paths that have patches (changes) or are newly created.
        
        Args:
            include_new_files: Whether to include newly created files in the result.
                            Default is True.
        
        Returns:
            List[str]: List of patched or newly created file paths.
        """
        patched_files = []
        for file_path, file in self._files.items():
            if file.patch or (include_new_files and file.is_new):
                patched_files.append(file_path)
        return patched_files

    def create_summary(self) -> str:
        """
        Creates a summary of the files and spans in the context.

        Returns:
            str: A formatted summary string listing files and their spans
        """
        if self.is_empty():
            return "No files in context"

        summary = []
        for context_file in self.get_context_files():
            # Get file stats
            tokens = context_file.context_size()

            # Get patch stats if available
            patch_stats = ""
            if context_file.patch:
                patch_lines = context_file.patch.split("\n")
                additions = sum(1 for line in patch_lines if line.startswith("+") and not line.startswith("+++"))
                deletions = sum(1 for line in patch_lines if line.startswith("-") and not line.startswith("---"))
                patch_stats = f" (+{additions}/-{deletions})"

            summary.append(f"\n### {context_file.file_path}")
            summary.append(f"- Tokens: {tokens}{patch_stats}")

            if context_file.show_all_spans:
                summary.append("- Showing all code in file")
                continue

            if context_file.spans:
                spans = []
                for span in context_file.spans:
                    if span.start_line and span.end_line:
                        # If it's a named span (not just numeric lines), show the name
                        if "@" in span.span_id:
                            name = span.span_id.split("@")[0]
                            spans.append(f"{name} ({span.start_line}-{span.end_line})")
                        elif ":" in span.span_id:
                            # For file:line format, just show the whole ID as it is self-descriptive
                            spans.append(span.span_id)
                        elif span.span_id.startswith("lines_"):
                            spans.append(f"{span.start_line}-{span.end_line}")
                        else:
                            spans.append(f"{span.span_id} ({span.start_line}-{span.end_line})")
                    else:
                        spans.append(span.span_id)
                summary.append(f"- Spans: {', '.join(spans)}")

        return "\n".join(summary)

    def add_file_context(self, other_context: "FileContext") -> List[str]:
        """
        Adds spans from another FileContext to the current one and returns newly added span IDs.

        Args:
            other_context: The FileContext to merge into this one

        Returns:
            List[str]: List of newly added span IDs. If show_all_spans was enabled for a file, 
                      "show_all_spans" is added to the list.
        """
        new_span_ids = []

        for other_file in other_context.files:
            file_path = other_file.file_path

            if not self.has_file(file_path):
                # Create new file if it doesn't exist
                context_file = self.add_file(file_path)
            else:
                context_file = self.get_context_file(file_path)

            # Add spans that don't already exist
            for span in other_file.spans:
                if context_file.add_span(
                    span.span_id,
                    start_line=span.start_line,
                    end_line=span.end_line,
                    tokens=span.tokens,
                    skeleton=span.skeleton,
                ):
                    new_span_ids.append(span.span_id)

            # Copy show_all_spans flag if either context has it enabled
            if other_file.show_all_spans and not context_file.show_all_spans:
                context_file.show_all_spans = True
                new_span_ids.append(f"show_all_spans({file_path})")
            elif other_file.show_all_spans:
                context_file.show_all_spans = True

        return new_span_ids

    def span_count(self) -> int:
        """
        Returns the total number of span IDs across all files in the context.

        Returns:
            int: Total number of span IDs
        """
        span_ids = []
        for file in self._files.values():
            span_ids.extend(file.span_ids)
        return len(span_ids)

    def get_test_counts(self) -> Tuple[int, int, int]:
        """
        Returns counts of passed, failed, and errored tests.

        Returns:
            Tuple[int, int, int]: A tuple containing (passed_count, failure_count, error_count)
        """
        all_results = []
        for test_file in self._test_files.values():
            all_results.extend(test_file.test_results)

        failure_count = sum(1 for r in all_results if r.status == TestStatus.FAILED)
        error_count = sum(1 for r in all_results if r.status == TestStatus.ERROR)
        passed_count = len(all_results) - failure_count - error_count

        return (passed_count, failure_count, error_count)

    def run_tests(self, test_files: List[str] | None = None) -> List[TestFile]:
        """
        Runs tests using the runtime environment and groups results by file.
        Preserves all test files in context, even those without results.

        Args:
            test_files (List[str] | None): Optional list of test files to run.
                                         If provided, adds these files to context before running.

        Returns:
            List[TestFile]: List of test results grouped by file
        """

        if not self._runtime:
            logger.error("Runtime environment not set for FileContext to run tests")
            return []

        if test_files:
            for test_file in test_files:
                self.add_test_file(test_file)
                if not self.has_file(test_file):
                    logger.info(f"Adding test file: {test_file} to context")
                    self.add_file(test_file, add_extra=False)
        elif self._test_files:
            test_files = list(self._test_files.keys())
        else:
            logger.warning("No test files in context to run tests")
            return []

        patch = self.generate_git_patch()

        # If specific test files provided, only run those
        if test_files:
            test_results = self._runtime.run_tests(patch=patch, test_files=test_files)
        else:
            test_results = self._runtime.run_tests(patch=patch)

        # Group results by file
        results_by_file = {}
        for result in test_results:
            if result.file_path:
                if result.file_path not in results_by_file:
                    results_by_file[result.file_path] = []
                results_by_file[result.file_path].append(result)
            else:
                logger.warning(f"Test result missing file path: {result}")

        # Update test results for existing test files
        for test_file in self._test_files.values():
            test_file.test_results = results_by_file.get(test_file.file_path, [])

        # Add summary log
        test_summary = []
        for test_file in self._test_files.values():
            status_counts = {}
            for result in test_file.test_results:
                if result.status not in status_counts:
                    status_counts[result.status] = 0
                status_counts[result.status] += 1
            test_summary.append(f" - {test_file.file_path}: {len(test_file.test_results)} tests. {status_counts}")

        logger.info(f"Test summary by file:{chr(10)}{chr(10).join(test_summary)}")

        return list(self._test_files.values())

    def add_test_file(self, file_path: str) -> TestFile:
        """Test summary
        Adds a new test file path to the context if it doesn't already exist.

        Args:
            file_path (str): Path to the test file

        Returns:
            TestFile: The new or existing TestFile object
        """
        if file_path in self._test_files:
            return self._test_files[file_path]

        test_file = TestFile(file_path=file_path, test_results=[])
        self._test_files[file_path] = test_file
        logger.debug(f"Added test file: {file_path}")

        return test_file

    def get_updated_test_results(self, previous_context: "FileContext") -> Set[str]:
        """
        Gets test files that have different results from the previous context.

        Args:
            previous_context: The previous FileContext to compare against

        Returns:
            Set[str]: Set of file paths that have updated test results
        """
        updated_files = set()

        for file_path, test_file in self._test_files.items():
            prev_test_file = previous_context._test_files.get(file_path)
            if not prev_test_file or test_file.test_results != prev_test_file.test_results:
                updated_files.add(file_path)

        return updated_files

    def get_test_summary(self) -> str:
        """
        Returns a summary of all test results in the format "X passed. Y failed. Z errors."

        Returns:
            str: Summary string of test results
        """
        from INTENT2FIX.runtime.runtime import TestStatus

        all_results = []
        for test_file in self._test_files.values():
            all_results.extend(test_file.test_results)

        failure_count = sum(1 for r in all_results if r.status == TestStatus.FAILED)
        error_count = sum(1 for r in all_results if r.status == TestStatus.ERROR)
        passed_count = len(all_results) - failure_count - error_count

        return f"{passed_count} passed. {failure_count} failed. {error_count} errors."

    def get_test_failure_details(self, max_tokens: int = 8000, max_chars_per_test: int = 2000) -> str:
        """
        Returns detailed output for each failed or errored test result.

        Returns:
            str: Formatted string containing details of failed tests
        """
        from INTENT2FIX.runtime.runtime import TestStatus

        sum_tokens = 0
        test_result_strings = []
        for test_file in self._test_files.values():
            for result in test_file.test_results:
                if result.status in [TestStatus.FAILED, TestStatus.ERROR] and result.message:
                    attributes = ""
                    if result.file_path:
                        attributes += f"{result.file_path}"
                        if result.span_id:
                            attributes += f" {result.span_id}"
                        if result.line:
                            attributes += f", line: {result.line}"

                    if len(result.message) > max_chars_per_test:
                        truncated_message = result.message[:max_chars_per_test]
                        truncated_message += "\n...truncated"
                    else:
                        truncated_message = result.message

                    test_result_str = f"* {result.status.value} {attributes}>\n```\n{truncated_message}\n```\n"
                    test_result_tokens = count_tokens(test_result_str)
                    if sum_tokens + test_result_tokens > max_tokens:
                        break

                    sum_tokens += test_result_tokens
                    test_result_strings.append(test_result_str)

        return "\n".join(test_result_strings) if test_result_strings else ""

    def get_test_status(self) -> Optional["TestStatus"]:
        """
        Returns the overall test status based on all test results.
        Returns ERROR if any test has error status,
        FAILED if any test has failed status,
        PASSED if all tests passed,
        or None if no test results exist.

        Returns:
            Optional[TestStatus]: The overall test status
        """
        all_results = []
        for test_file in self._test_files.values():
            all_results.extend(test_file.test_results)

        if not all_results:
            return None

        if any(r.status == TestStatus.ERROR for r in all_results):
            return TestStatus.ERROR
        elif any(r.status == TestStatus.FAILED for r in all_results):
            return TestStatus.FAILED
        else:
            return TestStatus.PASSED

    def was_edited(self) -> bool:
        """
        Checks if any files in the context were edited.

        Returns:
            bool: True if any file has been edited, False otherwise
        """
        return any(file.was_edited for file in self._files.values())

    def get_edited_files(self) -> List[str]:
        """
        Returns a list of file paths that have been edited in the context.
        A file is considered edited if it has changes (patch) but is not new.

        Returns:
            List[str]: List of edited file paths
        """
        return [file_path for file_path, file in self._files.items() if file.was_edited and not file.is_new]

    def get_created_files(self) -> List[str]:
        """
        Returns a list of file paths that have been newly created in the context.

        Returns:
            List[str]: List of created file paths
        """
        return [file_path for file_path, file in self._files.items() if file.is_new]


    def run_compile_and_tests(self) -> Tuple[bool, str]:
        """
        执行完整的编译和测试流程(针对C/C++/Java等需要编译的语言)
        
        Returns:
            Tuple[bool, str]: 
                - 编译是否成功
                - 测试结果摘要（包含编译错误或测试详情）
        """
        if not self._runtime:
            logger.error("Runtime environment not set")
            return False, "Runtime environment not set"

        # 1. 获取所有被修改的文件
        patched_files = [
            file_path for file_path, file in self._files.items() 
            if file.patch or file.is_new
        ]
        
        if not patched_files:
            logger.warning("No files were modified or created.")
            return False, "No files were modified or created."

        # 2. 准备临时文件
        import tempfile
        import os
        tmp_files = []
        compile_status = True
        compile_message = ""
        test_summary = ""

        try:
            # 3. 复制所有修改文件到容器
            for file_path in patched_files:
                tmp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
                tmp_files.append(tmp_file.name)
                
                file_content = self.get_file(file_path).content
                tmp_file.write(file_content)
                tmp_file.close()
                
                container_path = f"/app/cxxopts/{file_path}"
                if not self._runtime.copy_anything_to_container(tmp_file.name, container_path):
                    compile_status = False
                    compile_message = f"Failed to copy {file_path} to container"
                    break
            else:
                # 4. 执行编译（如果是需要编译的语言）
                if self._code_index._settings.language in ['c', 'cpp', 'java']:
                    logger.info("Running compilation...")
                    compile_status, compile_message = self._runtime.run_compile('/app/cxxopts')
                    logger.info(f"Compilation status: {compile_status}")
                    
                    # 5. 如果编译成功，执行测试
                    if compile_status:
                        test_summary = self._runtime.run_test(
                            language=self._code_index._settings.language
                        )
                        # 更新测试结果到上下文
                        self._update_test_results(test_summary)

        finally:
            # 6. 清理临时文件
            for tmp_file_path in tmp_files:
                try:
                    os.unlink(tmp_file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {tmp_file_path}: {str(e)}")

        # 7. 生成最终输出
        output = []
        if not compile_status:
            output.append(f"Compilation failed:\n{compile_message}")
        else:
            output.append("Compilation successful.")
            if test_summary:
                if self._runtime.is_test_failed(test_summary):
                    output.append(f"Tests failed:\n{test_summary}")
                else:
                    output.append("All tests passed!")
        
        return compile_status, "\n".join(output)

    def _update_test_results(self, raw_test_output: str) -> None:
        """
        内部方法:将原始测试输出解析后更新到_test_files
        
        Args:
            raw_test_output: 运行时返回的原始测试结果字符串
        """
        # 这里需要根据实际运行时返回的格式实现解析逻辑
        # 示例伪代码：
        test_results = []
        
        # 解析raw_test_output生成TestResult对象列表
        # 假设每个测试结果包含 file_path 和 status
        # test_results.append(TestResult(file_path="test_xxx.py", status=TestStatus.PASSED))
        
        # 更新到上下文
        for result in test_results:
            if result.file_path not in self._test_files:
                self._test_files[result.file_path] = TestFile(file_path=result.file_path)
            self._test_files[result.file_path].test_results.append(result)
