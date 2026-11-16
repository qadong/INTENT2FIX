# Use EntityCodeIndex instead of CodeIndex for C++ codebases
from INTENT2FIX.index.entity_code_index import EntityCodeIndex
# Export EntityCodeIndex as CodeIndex for backward compatibility
CodeIndex = EntityCodeIndex
from INTENT2FIX.index.settings import IndexSettings
