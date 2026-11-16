from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _load_jsonl(path: Path) -> List[dict]:
    items: List[dict] = []
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL parse failed: {path} line={line[:80]}...") from exc
    return items


def _build_fmap(funcs: Iterable[dict]) -> Dict[str, dict]:
    return {item.get("id"): item for item in funcs if item.get("id")}


def _build_class_maps(classes: Iterable[dict]) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    by_id: Dict[str, dict] = {}
    by_fq: Dict[str, dict] = {}
    for item in classes:
        cid = item.get("id")
        if cid:
            by_id[cid] = item
        fq = item.get("fq_name")
        if fq and fq not in by_fq:
            by_fq[fq] = item
    return by_id, by_fq


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', r"\"").replace("\n", r"\n")


def _dot_id(kind: str, identifier: str) -> str:
    return f"\"{_escape(f'{kind}:{identifier}')!s}\""


def _func_label(info: dict) -> str:
    name = info.get("fq_name") or info.get("short_name") or info.get("id") or "unknown"
    path = info.get("def_file") or ""
    rel = path
    if path:
        try:
            rel = os.path.relpath(path)
        except ValueError:
            rel = path
    return f"{name}\\n{rel}" if rel else name


def _class_label(info: dict) -> str:
    name = info.get("fq_name") or info.get("short_name") or info.get("id") or "class"
    file_path = info.get("def_file") or ""
    if file_path:
        try:
            rel = os.path.relpath(file_path)
        except ValueError:
            rel = file_path
        return f"{name}\\n{rel}"
    return name


def _collect_functions(feature: dict) -> List[str]:
    funcs: List[str] = []
    for key in ("member_funcs", "representative_funcs"):
        for fid in feature.get(key, []) or []:
            if fid and fid not in funcs:
                funcs.append(fid)
    return funcs


def generate_dot(
    features: Iterable[dict],
    func_map: Dict[str, dict],
    class_fq_map: Dict[str, dict],
) -> str:
    lines: List[str] = [
        "digraph FeatureCoverage {",
        "  rankdir=LR;",
        "  node [shape=box, style=filled, fillcolor=\"#FFF4D6\", fontname=\"Helvetica\"];",
    ]
    defined_nodes = set()

    def ensure_node(node_id: str, label: str, shape: str, fill: str) -> None:
        if node_id in defined_nodes:
            return
        defined_nodes.add(node_id)
        lines.append(
            f'  {node_id} [label="{_escape(label)}", shape={shape}, fillcolor="{fill}"];'
        )

    for feat in features:
        feature_id = feat.get("feature_id") or feat.get("feature_name")
        if not feature_id:
            continue
        feature_node = _dot_id("feature", feature_id)
        ensure_node(
            feature_node,
            feat.get("feature_name", feature_id),
            "box",
            "#E3F2FD",
        )

        for fid in _collect_functions(feat):
            func_info = func_map.get(fid)
            func_node = _dot_id("func", fid)
            label = _func_label(func_info or {"id": fid})
            ensure_node(func_node, label, "oval", "#F8F9FA")
            lines.append(f"  {feature_node} -> {func_node} [label=\"function\"];")

        src_class = feat.get("source_class")
        if src_class:
            class_info = class_fq_map.get(src_class, {"fq_name": src_class})
            class_node = _dot_id("class", class_info.get("id") or src_class)
            ensure_node(class_node, _class_label(class_info), "folder", "#E8F5E9")
            lines.append(f"  {feature_node} -> {class_node} [label=\"class\"];")

    lines.append("}")
    return "\n".join(lines)


def _filter_features(
    features: List[dict], selected_ids: Optional[List[str]], text: Optional[str]
) -> List[dict]:
    if not selected_ids and not text:
        return features
    selected_set = {item for item in (selected_ids or []) if item}
    result: List[dict] = []
    for feat in features:
        fid = feat.get("feature_id")
        if selected_set and fid not in selected_set:
            continue
        if text:
            name = feat.get("feature_name", "")
            desc = feat.get("feature_description", "")
            if text.lower() not in name.lower() and text.lower() not in desc.lower():
                continue
        result.append(feat)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize feature vs function/class relations, output Graphviz DOT"
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("index_output/features.jsonl"),
        help="features.jsonl path",
    )
    parser.add_argument(
        "--functions",
        type=Path,
        default=Path("index_output/functions.jsonl"),
        help="functions.jsonl path",
    )
    parser.add_argument(
        "--classes",
        type=Path,
        default=Path("index_output/classes.jsonl"),
        help="classes.jsonl path",
    )
    parser.add_argument(
        "--feature-id",
        action="append",
        help="Limit to specific feature_id(s)",
    )
    parser.add_argument(
        "--contains",
        help="Filter features by name/description",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output DOT path (default stdout)",
    )

    args = parser.parse_args()

    features = _load_jsonl(args.features)
    funcs = _load_jsonl(args.functions)
    classes = _load_jsonl(args.classes)

    func_map = _build_fmap(funcs)
    _, class_fq_map = _build_class_maps(classes)

    filtered = _filter_features(features, args.feature_id, args.contains)
    if not filtered:
        raise SystemExit("No matching features found")

    dot = generate_dot(filtered, func_map, class_fq_map)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(dot, encoding="utf-8")
        print(f"Written DOT to: {args.output}")
    else:
        print(dot)


if __name__ == "__main__":
    main()







