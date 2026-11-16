import tree_sitter_cpp as cpp
from tree_sitter import Language

from INTENT2FIX.codeblocks.parser.parser import CodeParser


class CppParser(CodeParser):
    def __init__(self, **kwargs):
        super().__init__(Language(cpp.language()), **kwargs)
        self.queries = []
        self.queries.extend(self._build_queries("cpp2.scm"))
        self.gpt_queries = []
