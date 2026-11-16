from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .graph_builder import GraphContext, build_graph
from .intent_subgraph import (
    DEFAULT_EDGE_TYPES,
    collect_subgraph,
    parse_edge_types,
    to_dot,
)


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _score_feature(query_tokens: Sequence[str], feature: Dict[str, object]) -> float:
    corpus_parts: List[str] = []
    name = feature.get("feature_name")
    desc = feature.get("feature_description")
    tags = feature.get("tags") or []
    if isinstance(name, str):
        corpus_parts.append(name)
    if isinstance(desc, str):
        corpus_parts.append(desc)
    if isinstance(tags, list):
        corpus_parts.extend(str(tag) for tag in tags)
    corpus = " ".join(corpus_parts)
    tokens = _tokenize(corpus)
    if not tokens or not query_tokens:
        return 0.0
    set_query = set(query_tokens)
    set_feat = set(tokens)
    intersect = len(set_query & set_feat)
    if intersect == 0:
        substr = " ".join(query_tokens)
        return 0.6 if substr and substr in corpus.lower() else 0.0
    jaccard = intersect / (len(set_query) + len(set_feat) - intersect)
    coverage = intersect / len(set_query)
    return 0.7 * jaccard + 0.3 * coverage


def _rank_features(
    query: str, features: List[Dict[str, object]], top_k: int
) -> List[Tuple[Dict[str, object], float]]:
    tokens = _tokenize(query)
    scored: List[Tuple[Dict[str, object], float]] = []
    for feature in features:
        score = _score_feature(tokens, feature)
        if score > 0:
            scored.append((feature, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]


def _default_output_path(index_dir: Path, query: str, fmt: str) -> Path:
    slug = "_".join(_tokenize(query))[:40] or hashlib.md5(query.encode()).hexdigest()[:8]
    suffix = ".json" if fmt == "json" else ".dot"
    return index_dir / "subgraphs" / f"query_{slug}{suffix}"


def _ensure_output_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _print_feature_choices(results: Sequence[Tuple[Dict[str, object], float]]) -> None:
    print("Candidate features:")
    for feature, score in results:
        print(
            f"- {feature.get('feature_id')} | {feature.get('feature_name')} | score={score:.3f}"
        )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Query by natural language and output subgraph."
    )
    parser.add_argument("--query", required=True, help="Natural language query")
    parser.add_argument("--index-dir", default="index_output", help="Index output dir")
    parser.add_argument("--max-depth", type=int, default=2, help="Subgraph BFS depth")
    parser.add_argument("--edge-types", default=",".join(DEFAULT_EDGE_TYPES))
    parser.add_argument(
        "--include-structure",
        action="store_true",
        help="Include FILE/CLASS structure edges",
    )
    parser.add_argument("--format", choices=["json", "dot"], default="dot")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k features to keep")
    parser.add_argument(
        "-o",
        "--output",
        help="Output path (default index_output/subgraphs/query_xxx.{dot,json})",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    index_dir = Path(args.index_dir)
    features_path = index_dir / "features.jsonl"
    features = _read_jsonl(features_path)
    if not features:
        raise SystemExit(f"No feature data at {features_path}")

    ranked = _rank_features(args.query, features, args.top_k)
    if not ranked:
        raise SystemExit("Query matched no feature, try adjusting the description.")

    _print_feature_choices(ranked)

    best_feature = ranked[0][0]
    best_id = best_feature["feature_id"]
    print(f"\nSelected feature: {best_id} ({best_feature.get('feature_name')})")

    ctx: GraphContext = build_graph(index_dir)
    allowed_edge_types = parse_edge_types(args.edge_types, args.include_structure)
    subgraph = collect_subgraph(ctx, best_id, args.max_depth, allowed_edge_types)
    payload = (
        json.dumps(subgraph, ensure_ascii=False, indent=2)
        if args.format == "json"
        else to_dot(subgraph)
    )

    output_path = Path(args.output) if args.output else _default_output_path(index_dir, args.query, args.format)
    _ensure_output_dir(output_path)
    output_path.write_text(payload, encoding="utf-8")
    print(
        f"\nWritten to {output_path}: {len(subgraph['nodes'])} nodes, {len(subgraph['edges'])} edges."
    )
    if args.format == "dot":
        print("Render with Graphviz, e.g.: dot -Tpng output.dot -o output.png")


if __name__ == "__main__":
    main()

