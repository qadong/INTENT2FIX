from INTENT2FIX.codeblocks.codeblocks import CodeBlock, CodeBlockType, SpanType
from INTENT2FIX.codeblocks.parser.create import create_parser, create_parser_by_ext
from INTENT2FIX.codeblocks.parser.java import JavaParser
from INTENT2FIX.codeblocks.parser.parser import CodeParser
from INTENT2FIX.codeblocks.parser.python import PythonParser
from INTENT2FIX.codeblocks.parser.c import CParser
from INTENT2FIX.codeblocks.parser.cpp import CppParser

def supports_codeblocks(path: str):
    return path.endswith((".py", ".cpp", ".hpp", ".cc", ".hh", ".cxx", ".hxx", ".c", ".h"))


def get_parser_by_path(file_path: str) -> CodeParser | None:
    if file_path.endswith(".py"):
        return PythonParser()
    elif file_path.endswith(".java"):
        return JavaParser()
    elif file_path.endswith((".c",".h")):
        return CParser()
    elif file_path.endswith((".cpp",".hpp",".cc",".hh",".cxx",".hxx")):
        return CppParser()
    else:
        return None
