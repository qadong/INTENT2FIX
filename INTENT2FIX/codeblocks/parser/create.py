from INTENT2FIX.codeblocks.parser.java import JavaParser
from INTENT2FIX.codeblocks.parser.parser import CodeParser
from INTENT2FIX.codeblocks.parser.python import PythonParser
from INTENT2FIX.codeblocks.parser.c import CParser
from INTENT2FIX.codeblocks.parser.cpp import CppParser

def is_supported(language: str) -> bool:
    return language and language in ["python", "java","c","cpp"]


def create_parser_by_ext(ext: str, **kwargs) -> CodeParser | None:
    if ext == ".py":
        return PythonParser(**kwargs)
    elif ext == ".java":
        return JavaParser(**kwargs)
    # elif ext in [".c",".h"]:
    #     return CParser(**kwargs)
    elif ext in [".cpp",".hpp",".cc",".hh",".cxx",".hxx",".c",".h"]:
        return CppParser(**kwargs)

    raise NotImplementedError(f"Extension {ext} is not supported.")


def create_parser(language: str, **kwargs) -> CodeParser | None:
    if language == "python":
        return PythonParser(**kwargs)
    elif language == "java":
        return JavaParser(**kwargs)
    elif language == "cpp":
        return CppParser(**kwargs)
    elif language == 'c':
        return CParser(**kwargs)
    raise NotImplementedError(f"Language {language} is not supported.")
