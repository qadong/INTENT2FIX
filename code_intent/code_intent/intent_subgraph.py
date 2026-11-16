from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .graph_builder import (
    EDGE_CALLS,
    EDGE_DEP_CALL,
    EDGE_COVERS_CLASS,
    EDGE_COVERS_FILE,
    EDGE_DEP_DECL,
    EDGE_DEP_MEMBER,
    EDGE_DEP_TEMPLATE,
    EDGE_DEP_TYPE,
    EDGE_DEP_UNEXPOSED,
    EDGE_IMPLEMENTS,
    EDGE_IN_CLASS,
    EDGE_IN_FILE,
    GraphContext,
    NODE_CLASS,
    NODE_FEATURE,
    NODE_FILE,
    NODE_FUNC,
    RELATION_EDGE_TYPES,
    STRUCTURE_EDGE_TYPES,
    build_graph,
)

NODE_LABEL_KEYS = ("signature", "feature_name", "path", "id")

NODE_STYLES: Dict[str, Dict[str, str]] = {
    "FUNC": {"shape": "box", "color": "#2563eb"},
    "INTENT_FEATURE": {"shape": "doubleoctagon", "color": "#d97706"},
    "CLASS": {"shape": "oval", "color": "#7c3aed"},
    "FILE": {"shape": "note", "color": "#6b7280"},
}

EDGE_STYLES: Dict[str, Dict[str, str]] = {
    EDGE_CALLS: {"color": "#0ea5e9"},
    EDGE_DEP_CALL: {"color": "#0284c7", "style": "dashed"},
    EDGE_IMPLEMENTS: {"color": "#10b981"},
    EDGE_DEP_TEMPLATE: {"color": "#f97316"},
    EDGE_DEP_DECL: {"color": "#fb923c"},
    EDGE_DEP_MEMBER: {"color": "#f97316"},
    EDGE_DEP_TYPE: {"color": "#fb7185"},
    EDGE_DEP_UNEXPOSED: {"color": "#94a3b8"},
    EDGE_IN_FILE: {"color": "#94a3b8", "style": "dashed"},
    EDGE_IN_CLASS: {"color": "#94a3b8", "style": "dashed"},
    EDGE_COVERS_FILE: {"color": "#6366f1", "style": "dotted"},
    EDGE_COVERS_CLASS: {"color": "#6366f1", "style": "dotted"},
}

DEFAULT_EDGE_TYPES = sorted(RELATION_EDGE_TYPES)


def _resolve_root_id(ctx: GraphContext, args: argparse.Namespace) -> str:
    if args.node_id:
        return args.node_id
    if args.feature_id:
        if args.feature_id not in ctx.features:
            raise ValueError(f"feature_id {args.feature_id} does not exist")
        return args.feature_id
    if args.func_id:
        if args.func_id not in ctx.functions:
            raise ValueError(f"func_id {args.func_id} does not exist or has no intent")
        return args.func_id
    raise ValueError("Must specify subgraph root via --feature-id / --func-id / --node-id")


def collect_subgraph(
    ctx: GraphContext,
    root_id: str,
    max_depth: int,
    allowed_edge_types: Optional[Set[str]] = None,
) -> Dict[str, List[Dict[str, object]]]:
    queue: deque[Tuple[str, int]] = deque([(root_id, 0)])
    visited: Set[str] = {root_id}
    edge_set: Set[Tuple[str, str, str]] = set()

    while queue:
        node_id, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for direction in ("out", "in"):
            buckets = ctx.graph.edge_buckets(node_id, direction)
            if not buckets:
                continue
            for edge_type, neighbors in buckets.items():
                if allowed_edge_types and edge_type not in allowed_edge_types:
                    continue
                for neighbor in neighbors:
                    if direction == "out":
                        src, dst = node_id, neighbor
                    else:
                        src, dst = neighbor, node_id
                    edge_key = (edge_type, src, dst)
                    if edge_key not in edge_set:
                        edge_set.add(edge_key)
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))

    nodes = [
        ctx.graph.nodes[node_id]
        for node_id in visited
        if node_id in ctx.graph.nodes
    ]
    edges = [
        {"type": edge_type, "source": src, "target": dst}
        for edge_type, src, dst in sorted(edge_set)
    ]
    return {"nodes": nodes, "edges": edges}


def _format_scope(node: Dict[str, object]) -> str:
    namespaces = node.get("namespaces") or []
    classes = node.get("classes") or []
    segments = [*namespaces, *classes]
    return "::".join(seg for seg in segments if seg)


def _format_func_label(node: Dict[str, object]) -> str:
    short_name = node.get("short_name") or node.get("action")
    signature = node.get("signature") or node.get("id")
    scope = _format_scope(node)
    if short_name:
        qualified = f"{scope}::{short_name}" if scope else short_name
    else:
        qualified = signature
    file_path = node.get("file")
    if file_path:
        return f"{qualified} @ {file_path}"
    return str(qualified)


def _format_feature_label(node: Dict[str, object]) -> str:
    name = node.get("feature_name") or node.get("id")
    description = node.get("feature_description")
    if description:
        trimmed = description[:120]
        if len(description) > 120:
            trimmed = trimmed.rstrip() + "..."
        return f"{name} — {trimmed}"
    return str(name)


def _format_file_label(node: Dict[str, object]) -> str:
    path = node.get("path") or node.get("file") or node.get("id")
    return str(path)


def _format_class_label(node: Dict[str, object]) -> str:
    fq_name = node.get("fq_name") or node.get("id")
    file_path = node.get("file")
    if file_path:
        return f"{fq_name} @ {file_path}"
    return str(fq_name)


def _infer_label(node: Dict[str, object]) -> str:
    node_type = node.get("type")
    if node_type == NODE_FUNC:
        return _format_func_label(node)
    if node_type == NODE_FEATURE:
        return _format_feature_label(node)
    if node_type == NODE_FILE:
        return _format_file_label(node)
    if node_type == NODE_CLASS:
        return _format_class_label(node)
    for key in NODE_LABEL_KEYS:
        value = node.get(key)
        if value:
            return str(value)
    return str(node["id"])


def to_dot(subgraph: Dict[str, List[Dict[str, object]]]) -> str:
    lines = ["digraph IntentSubgraph {"]
    lines.append('  graph [rankdir=LR];')
    lines.append('  node [fontname="Arial"];')
    lines.append('  edge [fontname="Arial"];')

    for node in subgraph["nodes"]:
        node_id = node["id"]
        node_type = node.get("type", "NODE")
        style = NODE_STYLES.get(node_type, {"shape": "ellipse", "color": "#1f2937"})
        label = _infer_label(node).replace('"', '\\"')
        lines.append(
            f'  "{node_id}" [label="{label}", shape={style["shape"]}, color="{style["color"]}"];'
        )

    for edge in subgraph["edges"]:
        edge_type = edge["type"]
        style = EDGE_STYLES.get(edge_type, {"color": "#9ca3af"})
        attrs = [f'color="{style["color"]}"', f'label="{edge_type}"']
        if "style" in style:
            attrs.append(f'style="{style["style"]}"')
        lines.append(
            f'  "{edge["source"]}" -> "{edge["target"]}" [{", ".join(attrs)}];'
        )

    lines.append("}")
    return "\n".join(lines)


def parse_edge_types(edge_arg: Optional[str], include_structure: bool) -> Set[str]:
    if edge_arg:
        selected = {item.strip() for item in edge_arg.split(",") if item.strip()}
    else:
        selected = set(DEFAULT_EDGE_TYPES)
    if include_structure:
        selected |= STRUCTURE_EDGE_TYPES
    return selected


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Export intent-related subgraph")
    parser.add_argument(
        "--index-dir", default="index_output", help="Index output dir (default index_output)"
    )
    parser.add_argument("--feature-id", help="Feature node id, e.g. feat_00012")
    parser.add_argument("--func-id", help="Function node id (with USR)")
    parser.add_argument("--node-id", help="Graph node id")
    parser.add_argument("--max-depth", type=int, default=2, help="BFS depth limit")
    parser.add_argument(
        "--edge-types",
        default=",".join(DEFAULT_EDGE_TYPES),
        help="Comma-separated edge types (see graph_builder)",
    )
    parser.add_argument(
        "--include-structure",
        action="store_true",
        help="Include FILE/CLASS structure edges",
    )
    parser.add_argument(
        "--format",
        choices=["json", "dot"],
        default="json",
        help="Output format (default json)",
    )
    parser.add_argument("-o", "--output", help="Output file path (default stdout)")

    args = parser.parse_args(argv)

    ctx = build_graph(args.index_dir)
    root_id = _resolve_root_id(ctx, args)
    allowed_edge_types = parse_edge_types(args.edge_types, args.include_structure)
    subgraph = collect_subgraph(ctx, root_id, args.max_depth, allowed_edge_types)

    if args.format == "json":
        payload = json.dumps(subgraph, ensure_ascii=False, indent=2)
    else:
        payload = to_dot(subgraph)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
        print(
            f"Written to {args.output}: {len(subgraph['nodes'])} nodes, {len(subgraph['edges'])} edges."
        )
    else:
        print(payload)


if __name__ == "__main__":
    main()

