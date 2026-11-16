from INTENT2FIX.actions.append_string import AppendString
from INTENT2FIX.actions.create_file import CreateFile
from INTENT2FIX.actions.finish import Finish
# from INTENT2FIX.actions.run_tests import RunTests
from INTENT2FIX.actions.string_replace import StringReplace
from INTENT2FIX.actions.view_code import ViewCode
from INTENT2FIX.actions.bash_action import BashAction
from INTENT2FIX.actions.run_intent_check import RunIntentCheck

# Use entity_tools based search actions
from INTENT2FIX.actions.entity_search_symbol import EntitySearchSymbol
from INTENT2FIX.actions.entity_function_semantic_search import EntityFunctionSemanticSearch
from INTENT2FIX.actions.entity_intent_search import EntityIntentSearch
from INTENT2FIX.actions.get_macro_usages import GetMacroUsages
from INTENT2FIX.actions.find_template_instantiation import FindTemplateInstantiation
from INTENT2FIX.actions.find_entity_usage import FindEntityUsage
from INTENT2FIX.actions.list_features import ListFeatures

# Keep old imports for backward compatibility (but they won't be used)
# from INTENT2FIX.actions.find_class import FindClass
# from INTENT2FIX.actions.find_code_snippet import FindCodeSnippet
# from INTENT2FIX.actions.find_function import FindFunction
# from INTENT2FIX.actions.semantic_search import SemanticSearch
