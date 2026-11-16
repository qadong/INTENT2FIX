from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

NODE_FUNC = "FUNC"
NODE_CLASS = "CLASS"
NODE_FILE = "FILE"
NODE_FEATURE = "INTENT_FEATURE"
NODE_INTENT_FUNC = "INTENT_FUNCTION"
NODE_INTENT_CLUSTER = "INTENT_CLUSTER"
NODE_MACRO = "MACRO"
NODE_PREPROCESSOR = "PREPROCESSOR"

EDGE_CALLS = "CALLS"
EDGE_DEP_CALL = "DEP_CALL"
EDGE_IN_FILE = "IN_FILE"
EDGE_IN_CLASS = "IN_CLASS"
EDGE_IMPLEMENTS = "IMPLEMENTS_INTENT"
EDGE_COVERS_FILE = "COVERS_FILE"
EDGE_COVERS_CLASS = "COVERS_CLASS"
EDGE_USES_MACRO = "USES_MACRO"
EDGE_MEMBER_OF = "MEMBER_OF"
EDGE_INTERACTS = "INTERACTS"
EDGE_GROUNDS_TO = "GROUNDS_TO"
EDGE_PARENT_OF = "PARENT_OF"
EDGE_DEP_TEMPLATE = "DEP_TEMPLATE_REF"
EDGE_DEP_DECL = "DEP_DECL_REF"
EDGE_DEP_MEMBER = "DEP_MEMBER_REF"
EDGE_DEP_TYPE = "DEP_TYPE_REF"
EDGE_DEP_UNEXPOSED = "DEP_UNEXPOSED_REF"

DEPENDENCY_EDGE_MAP = {
    "dependent_call": EDGE_DEP_CALL,
    "template_ref": EDGE_DEP_TEMPLATE,
    "decl_ref": EDGE_DEP_DECL,
    "member_ref": EDGE_DEP_MEMBER,
    "type_ref": EDGE_DEP_TYPE,
    "unexposed_ref": EDGE_DEP_UNEXPOSED,
}

DEPENDENCY_EDGE_TYPES = set(DEPENDENCY_EDGE_MAP.values())
RELATION_EDGE_TYPES = DEPENDENCY_EDGE_TYPES | {EDGE_CALLS, EDGE_IMPLEMENTS}
STRUCTURE_EDGE_TYPES = {
    EDGE_IN_FILE,
    EDGE_IN_CLASS,
    EDGE_COVERS_FILE,
    EDGE_COVERS_CLASS,
}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class Graph:

    def __init__(self) -> None:
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self._edges_out: Dict[str, Dict[str, Set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self._edges_in: Dict[str, Dict[str, Set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )

    def add_node(self, node_id: str, node_type: str, data: Dict[str, Any]) -> None:
        self.nodes[node_id] = {"id": node_id, "type": node_type, **data}

    def add_edge(self, src: str, dst: str, edge_type: str) -> None:
        if src not in self.nodes or dst not in self.nodes:
            return
        self._edges_out[src][edge_type].add(dst)
        self._edges_in[dst][edge_type].add(src)

    def neighbors(
        self,
        node_id: str,
        edge_type: Optional[str] = None,
        direction: str = "out",
    ) -> Set[str]:
        targets: Set[str] = set()
        if direction in ("out", "both"):
            buckets = self._edges_out.get(node_id, {})
            if edge_type:
                targets.update(buckets.get(edge_type, set()))
            else:
                for dsts in buckets.values():
                    targets.update(dsts)
        if direction in ("in", "both"):
            buckets = self._edges_in.get(node_id, {})
            if edge_type:
                targets.update(buckets.get(edge_type, set()))
            else:
                for srcs in buckets.values():
                    targets.update(srcs)
        return targets

    def edge_buckets(
        self, node_id: str, direction: str = "out"
    ) -> Dict[str, Set[str]]:
        if direction == "out":
            return self._edges_out.get(node_id, {})
        if direction == "in":
            return self._edges_in.get(node_id, {})
        if direction == "both":
            merged: Dict[str, Set[str]] = defaultdict(set)
            out_map = self._edges_out.get(node_id, {})
            in_map = self._edges_in.get(node_id, {})
            for edge_type, targets in out_map.items():
                merged[edge_type].update(targets)
            for edge_type, targets in in_map.items():
                merged[edge_type].update(targets)
            return merged
        raise ValueError(f"Unsupported direction: {direction}")


@dataclass
class GraphContext:
    graph: Graph
    functions: Dict[str, Dict[str, Any]]
    features: Dict[str, Dict[str, Any]]
    classes: Dict[str, Dict[str, Any]]
    files: Dict[str, Dict[str, Any]]
    macros: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    preprocessor_directives: List[Dict[str, Any]] = field(default_factory=list)
    func_to_features: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    feature_to_funcs: Dict[str, List[str]] = field(default_factory=dict)

    def get_function(self, func_id: str) -> Optional[Dict[str, Any]]:
        return self.functions.get(func_id)

    def get_feature(self, feature_id: str) -> Optional[Dict[str, Any]]:
        return self.features.get(feature_id)


def _combine_function_data(
    source: Dict[str, Any], intent: Dict[str, Any]
) -> Dict[str, Any]:
    data = source.copy()
    data.update(intent)
    data.setdefault("func_id", source["id"])
    data["namespaces"] = source.get("namespaces", [])
    data["classes"] = source.get("classes", [])
    if "owner_class_id" in source:
        data["owner_class_id"] = source["owner_class_id"]
    else:
        data.pop("owner_class_id", None)
    data["file"] = source.get("def_file") or intent.get("file")
    data["start_line"] = source.get("def_start_line") or intent.get("start_line")
    data["end_line"] = source.get("def_end_line") or intent.get("end_line")
    data["signature"] = intent.get("signature") or source.get("fq_name")
    data["is_test"] = source.get("is_test", False)
    data["code"] = source.get("code", "")
    data["content_hash"] = source.get("content_hash", "")
    return data


def _owner_class_id(func: Dict[str, Any], classes: Dict[str, Dict[str, Any]]) -> Optional[str]:
    # An explicit empty/missing target must not become a name-based guess.
    if "owner_class_id" in func:
        owner = func["owner_class_id"]
        return owner if owner in classes else None
    if not func.get("classes") or not func.get("is_method"):
        return None
    name = "::".join(func.get("namespaces", []) + func["classes"])
    matches = [cls for cls in classes.values() if cls.get("fq_name") == name]
    if len(matches) != 1:
        return None
    cls = matches[0]
    # Legacy records have no semantic parent: accept only an unambiguous,
    # lexically enclosed method definition. Out-of-line definitions need reindexing.
    if (func.get("file") == cls.get("def_file")
            and cls.get("def_line", 0) <= (func.get("start_line") or 0)
            <= (func.get("end_line") or 0) <= cls.get("def_end_line", 0)):
        return cls["id"]
    return None


def _ensure_file_node(
    graph: Graph, files: Dict[str, Dict[str, Any]], path: Optional[str]
) -> Optional[str]:
    if not path:
        return None
    if path not in files:
        node_id = f"file:{path}"
        files[path] = {"id": node_id, "path": path}
        graph.add_node(node_id, NODE_FILE, {"path": path})
    return files[path]["id"]


def build_graph(index_dir: str | Path = "index_output") -> GraphContext:
    base = Path(index_dir)
    functions = _read_jsonl(base / "functions.jsonl")
    function_intents = _read_jsonl(base / "function_intents.jsonl")
    class_intents = _read_jsonl(base / "class_intents.jsonl")
    features = _read_jsonl(base / "features.jsonl")
    cluster_intents = _read_jsonl(base / "cluster_intents.jsonl")
    cluster_hierarchy = _read_jsonl(base / "cluster_hierarchy.jsonl")
    classes = _read_jsonl(base / "classes.jsonl")
    macros = _read_jsonl(base / "macros.jsonl")
    file_records = _read_jsonl(base / "entities/files.jsonl")
    preprocessor_directives = _read_jsonl(
        base / "preprocessor_directives.jsonl"
    )
    call_edges = _read_jsonl(base / "callgraph.jsonl")

    graph = Graph()
    file_nodes: Dict[str, Dict[str, Any]] = {}
    for record in file_records:
        path = record.get("file") or record.get("path")
        file_node_id = _ensure_file_node(graph, file_nodes, path)
        if not file_node_id:
            continue
        file_nodes[path].update(record)
        graph.nodes[file_node_id].update(record)

    for directive in preprocessor_directives:
        node_id = directive.get("id")
        if not node_id:
            continue
        graph.add_node(node_id, NODE_PREPROCESSOR, directive)
        file_node_id = _ensure_file_node(
            graph, file_nodes, directive.get("file")
        )
        if file_node_id:
            graph.add_edge(node_id, file_node_id, EDGE_IN_FILE)

    macro_lookup: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen_macro_nodes = set()
    for m in macros:
        mname = m["name"]
        node_id = f"macro:{mname}:{m['file']}:{m['line']}"
        if node_id not in seen_macro_nodes:
            seen_macro_nodes.add(node_id)
            macro_lookup[mname].append(m)
            graph.add_node(node_id, NODE_MACRO, m)
            file_node_id = _ensure_file_node(graph, file_nodes, m.get("file"))
            if file_node_id:
                graph.add_edge(node_id, file_node_id, EDGE_IN_FILE)

    class_intent_lookup = {item["class_id"]: item for item in class_intents}
    class_lookup: Dict[str, Dict[str, Any]] = {}
    for cls in classes:
        cls_id = cls["id"]
        intent = class_intent_lookup.get(cls_id, {})
        combined = {**cls, **intent}
        class_lookup[cls_id] = combined
        graph.add_node(cls_id, NODE_CLASS, combined)
        file_node_id = _ensure_file_node(graph, file_nodes, cls.get("def_file"))
        if file_node_id:
            graph.add_edge(cls_id, file_node_id, EDGE_IN_FILE)

    function_lookup: Dict[str, Dict[str, Any]] = {}
    intent_lookup = {item["func_id"]: item for item in function_intents}

    for func in functions:
        func_id = func["id"]
        intent = intent_lookup.get(func_id, {})
        combined = _combine_function_data(func, intent)
        function_lookup[func_id] = combined
        graph.add_node(func_id, NODE_FUNC, combined)
        for mname in func.get("macros_used", []):
            if mname in macro_lookup:
                for m_def in macro_lookup[mname]:
                    m_node_id = f"macro:{mname}:{m_def['file']}:{m_def['line']}"
                    graph.add_edge(func_id, m_node_id, EDGE_USES_MACRO)

        file_node_id = _ensure_file_node(graph, file_nodes, combined.get("file"))
        if file_node_id:
            graph.add_edge(func_id, file_node_id, EDGE_IN_FILE)
        class_id = _owner_class_id(combined, class_lookup)
        if class_id:
            graph.add_edge(func_id, class_id, EDGE_IN_CLASS)

    # Intent function nodes (GroundsTo)
    intent_func_ids: Dict[str, str] = {}
    for intent in function_intents:
        func_id = intent.get("func_id")
        if not func_id or func_id not in function_lookup:
            continue
        intent_node_id = f"intent_func:{func_id}"
        intent_func_ids[func_id] = intent_node_id
        graph.add_node(
            intent_node_id,
            NODE_INTENT_FUNC,
            {
                "func_id": func_id,
                "short_intent": intent.get("short_intent"),
                "usage_context": intent.get("usage_context"),
                "raw_summary": intent.get("raw_summary"),
            },
        )
        graph.add_edge(intent_node_id, func_id, EDGE_GROUNDS_TO)

    feature_lookup: Dict[str, Dict[str, Any]] = {}
    for feature in features:
        feature_id = feature["feature_id"]
        feature_lookup[feature_id] = feature
        graph.add_node(
            feature_id,
            NODE_FEATURE,
            {
                "feature_id": feature_id,
                "feature_name": feature["feature_name"],
                "tags": feature.get("tags", []),
            },
        )

    # Intent cluster nodes
    intent_cluster_lookup: Dict[str, Dict[str, Any]] = {}
    if not cluster_intents:
        cluster_intents = []
        for feat in features:
            if feat.get("feature_type") == "graph_community":
                cluster_intents.append(
                    {
                        "cluster_id": feat["feature_id"],
                        "cluster_name": feat.get("feature_name"),
                        "cluster_description": feat.get("feature_description"),
                        "tags": feat.get("tags", []),
                        "member_funcs": feat.get("member_funcs", []),
                        "representative_funcs": feat.get("representative_funcs", []),
                        "cluster_type": "graph_community",
                    }
                )

    for cluster in cluster_intents:
        cluster_id = cluster["cluster_id"]
        intent_cluster_lookup[cluster_id] = cluster
        graph.add_node(
            cluster_id,
            NODE_INTENT_CLUSTER,
            {
                "cluster_id": cluster_id,
                "cluster_name": cluster.get("cluster_name"),
                "tags": cluster.get("tags", []),
            },
        )

    for rel in cluster_hierarchy:
        parent_id = rel.get("parent_cluster_id")
        child_id = rel.get("child_cluster_id")
        if not parent_id or not child_id:
            continue
        if parent_id in intent_cluster_lookup and child_id in intent_cluster_lookup:
            graph.add_edge(parent_id, child_id, EDGE_PARENT_OF)

    func_to_features: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for intent in function_intents:
        func_id = intent["func_id"]
        for feature_info in intent.get("features", []):
            feature_id = feature_info["feature_id"]
            func_to_features[func_id].append(feature_info)
            graph.add_edge(func_id, feature_id, EDGE_IMPLEMENTS)

    feature_to_funcs: Dict[str, List[str]] = defaultdict(list)
    for feature in features:
        for member in feature.get("member_funcs", []):
            feature_to_funcs[feature["feature_id"]].append(member)

    for feature_id, members in feature_to_funcs.items():
        file_targets: Set[str] = set()
        class_targets: Set[str] = set()
        for func_id in members:
            func_data = function_lookup.get(func_id)
            if not func_data:
                continue
            file_node_id = _ensure_file_node(graph, file_nodes, func_data.get("file"))
            if file_node_id:
                file_targets.add(file_node_id)
            class_targets.update(graph.neighbors(func_id, EDGE_IN_CLASS, direction="out"))
        for file_node_id in file_targets:
            graph.add_edge(feature_id, file_node_id, EDGE_COVERS_FILE)
        for class_id in class_targets:
            graph.add_edge(feature_id, class_id, EDGE_COVERS_CLASS)
            graph.add_edge(class_id, feature_id, EDGE_IMPLEMENTS)

    # MemberOf: intent function -> intent cluster
    func_to_cluster: Dict[str, str] = {}
    for cluster_id, cluster in intent_cluster_lookup.items():
        for fid in cluster.get("member_funcs", []):
            func_to_cluster[fid] = cluster_id
            intent_node_id = intent_func_ids.get(fid)
            if intent_node_id:
                graph.add_edge(intent_node_id, cluster_id, EDGE_MEMBER_OF)

    code_entity_ids = set(function_lookup) | set(class_lookup)
    strict_executable_endpoints = (base / "edge_endpoint_stats.json").is_file()
    for edge in call_edges:
        caller = edge.get("caller")
        callee = edge.get("callee")
        edge_type = edge.get("edge_type", "call")
        if (
            edge_type in {"call", "dependent_call"}
            and (caller not in function_lookup or callee not in function_lookup)
        ):
            if strict_executable_endpoints:
                raise RuntimeError(
                    "Executable edge references a missing function node: "
                    f"{caller!r} -> {callee!r} ({edge_type})."
                )
            continue
        if caller not in code_entity_ids or callee not in code_entity_ids:
            continue
        if edge_type == "call":
            graph.add_edge(caller, callee, EDGE_CALLS)
            continue
        dep_edge = DEPENDENCY_EDGE_MAP.get(edge_type)
        if dep_edge:
            graph.add_edge(caller, callee, dep_edge)

    for edge in call_edges:
        caller = edge.get("caller")
        callee = edge.get("callee")
        if not caller or not callee:
            continue
        src_cluster = func_to_cluster.get(caller)
        dst_cluster = func_to_cluster.get(callee)
        if not src_cluster or not dst_cluster or src_cluster == dst_cluster:
            continue
        graph.add_edge(src_cluster, dst_cluster, EDGE_INTERACTS)

    return GraphContext(
        graph=graph,
        functions=function_lookup,
        features=feature_lookup,
        classes=class_lookup,
        files=file_nodes,
        macros=macro_lookup,
        preprocessor_directives=preprocessor_directives,
        func_to_features=func_to_features,
        feature_to_funcs=feature_to_funcs,
    )
