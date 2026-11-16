from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Literal, Optional


EntityKind = Literal[
    "file",
    "function",
    "class",
    "function_template",
    "class_template",
    "template_instantiation",
    "macro_def",
    "macro_use",
    "preprocessor_directive",
]


EdgeType = Literal[
    "call",
    "dependent_call",
    "decl_ref",
    "member_ref",
    "type_ref",
    "template_ref",
    "unexposed_ref",
    "instantiates",
    "implements",
    "includes",
]


@dataclass
class EntityBase:
    entity_id: str
    kind: EntityKind
    usr: Optional[str]
    fq_name: Optional[str]
    short_name: Optional[str]
    file: Optional[str]
    line: Optional[int]
    end_line: Optional[int]
    is_test: bool = False
    is_third_party: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FunctionEntity(EntityBase):
    signature: Optional[str] = None
    return_type: Optional[str] = None
    parameter_types: Optional[List[str]] = None
    namespaces: Optional[List[str]] = None
    classes: Optional[List[str]] = None
    is_method: bool = False
    is_static: bool = False
    is_template: bool = False
    raw_comment: str = ""


@dataclass
class ClassEntity(EntityBase):
    namespaces: Optional[List[str]] = None
    bases: Optional[List[str]] = None
    class_kind: Optional[str] = None  # class/struct/template
    raw_comment: str = ""


@dataclass
class TemplateInstantiationEntity(EntityBase):
    display_name: Optional[str] = None
    template_def_id: Optional[str] = None
    template_args: Optional[List[str]] = None
    trigger_file: Optional[str] = None
    trigger_line: Optional[int] = None
    is_explicit_specialization: bool = False


@dataclass
class EdgeRecord:
    edge_type: EdgeType
    src_entity_id: str
    dst_entity_id: str
    evidence_file: Optional[str] = None
    evidence_line: Optional[int] = None
    count: int = 1
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if data.get("extra") is None:
            data.pop("extra", None)
        return data


@dataclass
class SpecRecord:
    spec_id: str
    target_entity_id: str
    signature: str
    doc_comment: str
    constraints: List[str]
    intent_summary: str
    embedding_text: str






