import logging
import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from INTENT2FIX.codeblocks.codeblocks import (
    CodeBlock,
    CodeBlockType,
    CodeBlockTypeGroup,
    BlockSpan,
    SpanType,
)
from INTENT2FIX.codeblocks.module import Module
# from INTENT2FIX.repository.repository import Repository # Removed to avoid circular import

logger = logging.getLogger(__name__)

from INTENT2FIX.utils.tokenizer import count_tokens

from INTENT2FIX.codeblocks.parser.parser import CodeParser

class ClangdParser(CodeParser):
    """
    A parser that uses Clangd index data (via entity_tools) instead of Tree-sitter.
    It builds a CodeBlock tree based on entities found in the Clangd index.
    """

    def __init__(self, **kwargs):
        # We do NOT call super().__init__ to avoid Tree-sitter initialization
        self.encoding = kwargs.get("encoding", "utf8")
        self.tokenizer = kwargs.get("tokenizer")
        if not self.tokenizer:
            from llama_index.core import get_tokenizer
            self.tokenizer = get_tokenizer()
        self.index_callback = kwargs.get("index_callback")

    def parse(self, content: str, file_path: Optional[str] = None) -> Module:
        """
        Parses the file content into a Module using Clangd index data.
        """
        if not file_path:
            # Fallback to a simple Module if no file path provided
            return Module(
                type=CodeBlockType.MODULE,
                content=content,
                start_line=1,
                end_line=len(content.split("\n")),
                tokens=count_tokens(content)
            )

        # 1. Get entities for this file from Clangd index
        entities = self._get_entities_for_file(file_path)
        
        # 2. Sort entities by line range (size) to handle nesting
        entities.sort(key=lambda x: (x.get("def_line") or 1, -(x.get("def_end_line") or 1)))

        # 3. Create Module
        content_lines = content.split("\n")
        module = Module(
            type=CodeBlockType.MODULE,
            content=content,
            file_path=file_path,
            start_line=1,
            end_line=len(content_lines),
            spans_by_id={},
            tokens=count_tokens(content),
            language="cpp" # Default for ClangdParser
        )

        # 4. Build CodeBlock tree
        blocks = []
        
        for entity in entities:
            start_line = entity.get("def_line") or entity.get("line") or 1
            end_line = entity.get("def_end_line") or start_line
            
            # Map Clangd kinds to CodeBlockType
            kind = entity.get("kind", "").lower()
            if kind in ["class", "class_template", "template_class"]:
                block_type = CodeBlockType.CLASS
            elif kind in ["struct", "template_struct"]:
                block_type = CodeBlockType.STRUCT
            elif kind == "enum":
                block_type = CodeBlockType.ENUM
            elif kind == "union":
                block_type = CodeBlockType.UNION
            elif kind in ["function", "function_template", "template_function", "method", "constructor", "destructor"]:
                block_type = CodeBlockType.FUNCTION
            elif kind == "macro":
                block_type = CodeBlockType.MACRO 
            elif kind == "namespace":
                block_type = CodeBlockType.NAMESPACE
            else:
                block_type = CodeBlockType.CODE

            # Create span_id in Clangd format: file_path:start_line or file_path:start_line-end_line
            if start_line == end_line:
                span_id = f"{file_path}:{start_line}"
            else:
                span_id = f"{file_path}:{start_line}-{end_line}"

            # Extract content for this block
            block_content = "\n".join(content_lines[start_line-1:end_line])

            # Improved identifier: prefer short_name, then fq_name
            identifier = entity.get("short_name") or entity.get("fq_name") or entity.get("name")

            block = CodeBlock(
                type=block_type,
                content=block_content,
                identifier=identifier,
                start_line=start_line,
                end_line=end_line,
                span_ids={span_id},
                tokens=count_tokens(block_content)
            )
            
            # Create a BlockSpan for this block
            span = BlockSpan(
                span_id=span_id,
                span_type=SpanType.IMPLEMENTATION,
                start_line=start_line,
                end_line=end_line,
                initiating_block=block,
                tokens=block.tokens,
            )
            module.spans_by_id[span_id] = span
            block.belongs_to_span = span
            
            blocks.append(block)
            
            if self.index_callback:
                self.index_callback(block)

        # 5. Nest blocks
        blocks.sort(key=lambda b: (b.end_line - b.start_line), reverse=True)
        
        root_blocks = []
        for i, block in enumerate(blocks):
            is_nested = False
            for j in range(i):
                parent = blocks[j]
                if parent.start_line <= block.start_line and parent.end_line >= block.end_line and parent != block:
                    parent.children.append(block)
                    block.parent = parent
                    is_nested = True
                    break
            if not is_nested:
                root_blocks.append(block)
        
        # Sort root blocks by start line
        root_blocks.sort(key=lambda b: b.start_line)
        module.children = root_blocks
        
        # Ensure parent-child relationships and identifiers are unique
        for b in blocks:
            b.children.sort(key=lambda c: c.start_line)
            for c in b.children:
                c.parent = b

        return module

    def _get_entities_for_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Retrieves all entities for the given file from Clangd index files.
        """
        # Try to get instance name
        instance_name = os.environ.get("SWEBENCH_INSTANCE_NAME")
        if not instance_name:
            logger.warning("SWEBENCH_INSTANCE_NAME not set, ClangdParser cannot find entities.")
            return []

        index_root = os.environ.get("INDEX_STORE_DIR")
        if not index_root:
            logger.warning("INDEX_STORE_DIR not set, ClangdParser cannot find entities.")
            return []
        
        index_dir = Path(index_root) / f"swe-bench_{instance_name}" / "index_output" / "entities"
        if not index_dir.exists():
            # Try without swe-bench_ prefix
            index_dir = Path(index_root) / instance_name / "index_output" / "entities"
            if not index_dir.exists():
                return []

        entities = []
        
        # Files to check: classes.jsonl, functions.jsonl, macros.jsonl
        jsonl_files = ["classes.jsonl", "functions.jsonl", "macros.jsonl"]
        
        # Normalize target file path for matching
        # Clangd usually has absolute paths like /home/Catch2/...
        # or relative paths if indexed differently.
        
        for jsonl_name in jsonl_files:
            file_p = index_dir / jsonl_name
            if not file_p.exists():
                continue
                
            try:
                with open(file_p, "r") as f:
                    for line in f:
                        entity = json.loads(line)
                        def_file = entity.get("def_file") or entity.get("file")
                        if not def_file:
                            continue
                            
                        # Matching logic: def_file might be absolute, file_path is relative
                        if file_path in def_file:
                            entities.append(entity)
            except Exception as e:
                logger.error(f"Error reading Clangd index file {file_p}: {e}")
                
        return entities

