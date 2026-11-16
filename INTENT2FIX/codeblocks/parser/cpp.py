import logging
from typing import Optional
from INTENT2FIX.codeblocks.parser.clangd import ClangdParser

logger = logging.getLogger(__name__)

class CppParser(ClangdParser):
    @property
    def language(self):
        return "cpp"

    def parse(self, content: str, file_path: Optional[str] = None):
        module = super().parse(content, file_path)
        # If ClangdParser returns a skeleton-only or empty module (because index is missing),
        # fallback to Tree-sitter for better structure.
        if not module or not module.children:
            try:
                from INTENT2FIX.codeblocks.parser.cpp2 import CppParser as TreeSitterCppParser
                logger.info(f"ClangdParser failed for {file_path}, falling back to Tree-sitter.")
                ts_parser = TreeSitterCppParser()
                return ts_parser.parse(content, file_path)
            except Exception as e:
                logger.error(f"Fallback to Tree-sitter failed: {e}")
        return module
