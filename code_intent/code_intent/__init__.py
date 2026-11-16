
from importlib import import_module
from typing import Any, Dict, Tuple

__all__ = [
    "build_function_intents",
    "build_feature_intents",
    "CodeIntentRAG",
    "load_functions",
]

_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    "build_function_intents": ("code_intent.intent_extractor", "build_function_intents"),
    "load_functions": ("code_intent.intent_extractor", "load_functions"),
    "build_feature_intents": ("code_intent.feature_builder", "build_feature_intents"),
    "CodeIntentRAG": ("code_intent.search_api", "CodeIntentRAG"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if not target:
        raise AttributeError(f"module {__name__} has no attribute {name}")
    module_name, attr_name = target
    module = import_module(module_name)
    attr = getattr(module, attr_name)
    globals()[name] = attr
    return attr


def __dir__() -> Any:
    return sorted(list(globals().keys()) + list(_LAZY_ATTRS.keys()))