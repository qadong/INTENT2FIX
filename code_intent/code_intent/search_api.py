from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import faiss
import numpy as np

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError:
    FastAPI = None
    HTTPException = None
    BaseModel = object
    Field = None

from .graph_builder import (
    EDGE_CALLS,
    EDGE_DEP_CALL,
    EDGE_DEP_MEMBER,
    EDGE_DEP_TEMPLATE,
    EDGE_IN_CLASS,
    GraphContext,
    build_graph,
)
from .embedding_manifest import load_embedding_manifest
from .intent_params import EXPANSION_WIDTH
from .llm_utils import embed_query, rerank_documents

INDEX_ROOT = Path("index")
logger = logging.getLogger(__name__)


def _inner_product_similarity(score: float) -> float:
    """Return cosine similarity from a normalized IndexFlatIP score."""
    return max(-1.0, min(1.0, float(score)))


def _load_meta(path: Path, id_key: str) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    records.sort(key=lambda x: x.get("index", 0))
    for rec in records:
        if id_key not in rec:
            raise ValueError(f"meta file missing {id_key}: {path}")
    return records


class _VectorIndex:
    def __init__(self, index_path: Path, meta_path: Path, id_key: str) -> None:
        self.index_path = index_path
        self.meta_path = meta_path
        self.id_key = id_key
        self.index: Optional[faiss.Index] = None
        self.meta: List[Dict[str, Any]] = []
        if index_path.exists() and meta_path.exists():
            self.index = faiss.read_index(str(index_path))
            self.meta = _load_meta(meta_path, id_key)

    def search(self, query_vec: np.ndarray, top_k: int) -> List[Tuple[Dict[str, Any], float]]:
        if self.index is None or not self.meta or self.index.ntotal == 0 or top_k <= 0:
            return []
        q = query_vec.astype("float32").reshape(1, -1)
        if q.shape[1] != self.index.d:
            raise ValueError(
                f"Embedding dimension mismatch for {self.index_path.name}: "
                f"query={q.shape[1]}, index={self.index.d}"
            )
        scores, indices = self.index.search(q, top_k)
        results: List[Tuple[Dict[str, Any], float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self.meta):
                continue
            results.append((self.meta[idx], float(score)))
        return results


@dataclass
class _FunctionHit:
    func_id: str
    score: float


class CodeIntentRAG:

    FEATURE_TOP_K = max(15, EXPANSION_WIDTH // 2)
    FUNC_CAND_MULTIPLIER = 5
    WEIGHT_FUNC = 0.6
    WEIGHT_FEATURE = 0.15
    WEIGHT_GRAPH = 0.25
    WEIGHT_CLASS = 0.1
    WEIGHT_TEXT = 0.3
    WEIGHT_RERANK = 0.35
    RERANK_TOP_K = EXPANSION_WIDTH
    GRAPH_SEED_MULTIPLIER = 2
    GRAPH_CALLER_LIMIT = 8
    GRAPH_CALLEE_LIMIT = 8
    GRAPH_SIBLING_LIMIT = 8
    GRAPH_CALL_DECAY = 0.85
    GRAPH_SIBLING_DECAY = 0.7
    RRF_K = 60
    FEATURE_MEMBER_LIMIT = 12
    STRUCTURAL_SEEDS = 2
    STRUCTURAL_CANDIDATES = 10
    STRUCTURAL_COMPANIONS = 3
    STRUCTURAL_PATHS_PER_CANDIDATE = 4
    STRUCTURAL_PER_API_FAMILY = 2
    STRUCTURAL_PER_SEED_VARIANT_FAMILY = 1
    STRUCTURAL_MIXED_PATH_RESERVE = 1

    def __init__(self, index_dir: str | Path = "index_output") -> None:
        base = Path(index_dir)
        self._embedding_compatible = True
        self.graph_ctx: GraphContext = build_graph(base)
        self.func_index = _VectorIndex(
            base / "func_index.faiss", base / "func_meta.jsonl", "func_id"
        )
        self.class_index = _VectorIndex(
            base / "class_index.faiss", base / "class_meta.jsonl", "class_id"
        )
        self.feature_index = _VectorIndex(
            base / "feature_index.faiss", base / "feature_meta.jsonl", "feature_id"
        )
        self._log_embedding_compatibility(base)

    def _log_embedding_compatibility(self, base: Path) -> None:
        manifest = load_embedding_manifest(base)
        entries = manifest.get("indexes", {})
        runtime_model = os.getenv("VOYAGE_EMBED_MODEL")
        runtime_dim = int(os.getenv("CODE_INTENT_EMBED_DIM", "768"))
        loaded = {
            "functions": self.func_index,
            "classes": self.class_index,
            "features": self.feature_index,
        }
        for name, vector_index in loaded.items():
            if vector_index.index is None:
                continue
            entry = entries.get(name)
            if not entry:
                self._embedding_compatible = False
                logger.error(
                    "Embedding provenance is missing for %s (%s, dim=%d). "
                    "Semantic vector search is disabled; rebuild or select a "
                    "manifest-backed index.",
                    name,
                    base,
                    vector_index.index.d,
                )
                continue
            index_model = entry.get("model")
            index_dim = entry.get("dimension")
            if index_model != runtime_model or index_dim != runtime_dim:
                self._embedding_compatible = False
                logger.error(
                    "Embedding configuration mismatch for %s: index model=%r "
                    "dim=%s; runtime model=%r dim=%s. Semantic vector search "
                    "is disabled.",
                    name,
                    index_model,
                    index_dim,
                    runtime_model,
                    runtime_dim,
                )
            if vector_index.index.d != index_dim:
                self._embedding_compatible = False
                logger.error(
                    "FAISS dimension mismatch for %s: manifest=%s index=%d. "
                    "Semantic vector search is disabled.",
                    name,
                    index_dim,
                    vector_index.index.d,
                )
        if entries and self._embedding_compatible:
            logger.info(
                "Semantic indexes ready: model=%s dim=%d metric=inner_product path=%s",
                runtime_model,
                runtime_dim,
                base,
            )

    def _encode_query(self, query: str) -> np.ndarray:
        if not self._embedding_compatible:
            raise RuntimeError(
                "Semantic vector search is disabled because the index embedding "
                "provenance does not match the runtime configuration. Rebuild "
                "function and feature embeddings first."
            )
        return embed_query(query)

    def _format_func_label(self, func_data: Dict[str, Any]) -> str:
        signature = func_data.get("signature") or func_data.get("short_intent")
        file_path = func_data.get("file")
        if signature and file_path:
            return f"{signature} — {file_path}"
        return signature or file_path or "unknown"

    def _feature_members(
        self, feature_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        member_ids = self.graph_ctx.feature_to_funcs.get(feature_id, [])
        members: List[Dict[str, Any]] = []
        for func_id in member_ids:
            func_data = self.graph_ctx.functions.get(func_id)
            if not func_data:
                continue
            members.append(
                {
                    "func_id": self._format_func_label(func_data),
                    "raw_func_id": func_id,
                    "short_intent": func_data.get("short_intent"),
                    "file": func_data.get("file"),
                    "start_line": func_data.get("start_line"),
                    "end_line": func_data.get("end_line"),
                }
            )
        if limit is not None and len(members) > limit:
            return members[:limit]
        return members

    def search_features(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        vec = self._encode_query(query)
        hits = self.feature_index.search(vec, top_k)
        
        results: List[Dict[str, Any]] = []
        for meta, score in hits:
            feature_id = meta["feature_id"]
            feature = self.graph_ctx.features.get(feature_id, {})
            sim = _inner_product_similarity(score)
            text_bonus = 0.0
            q_lower = query.lower()
            f_name = (feature.get("feature_name") or "").lower()
            if q_lower in f_name or f_name in q_lower:
                text_bonus = 0.2
            
            members = self._feature_members(feature_id, self.FEATURE_MEMBER_LIMIT)
            file_set = []
            seen_files = set()
            for member in members:
                path = member.get("file")
                if not path or path in seen_files:
                    continue
                seen_files.add(path)
                file_set.append(path)
            
            results.append(
                {
                    "feature_id": feature_id,
                    "feature_name": feature.get("feature_name"),
                    "feature_description": feature.get("feature_description"),
                    "tags": feature.get("tags", []),
                    "score": round(sim + text_bonus, 4),
                    "members": members,
                    "files": file_set,
                }
            )
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def search_classes(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        vec = self._encode_query(query)
        return self._search_classes_with_vector(vec, top_k)

    def _search_classes_with_vector(
        self, query_vec: np.ndarray, top_k: int
    ) -> List[Dict[str, Any]]:
        hits = self.class_index.search(query_vec, top_k)
        results = []
        for meta, score in hits:
            cls_id = meta["class_id"]
            cls_data = self.graph_ctx.classes.get(cls_id, {})
            sim = _inner_product_similarity(score)
            results.append({
                "class_id": cls_id,
                "fq_name": cls_data.get("fq_name"),
                "short_intent": cls_data.get("short_intent"),
                "responsibility": cls_data.get("responsibility"),
                "file": cls_data.get("file"),
                "score": round(sim, 4)
            })
        return results

    def get_macro_details(self, macro_name: str) -> Dict[str, Any]:
        definitions = self.graph_ctx.macros.get(macro_name, [])
        if not definitions:
            return {"name": macro_name, "definitions": [], "usages": []}
        usages = []
        for m_def in definitions:
            m_node_id = f"macro:{macro_name}:{m_def['file']}:{m_def['line']}"
            callers = self.graph_ctx.graph.neighbors(m_node_id, "USES_MACRO", direction="in")
            for func_id in callers:
                func_data = self.graph_ctx.functions.get(func_id)
                if func_data:
                    usages.append({
                        "func_id": func_id,
                        "fq_name": func_data.get("fq_name"),
                        "file": func_data.get("file"),
                        "line": func_data.get("start_line")
                    })
        seen_funcs = set()
        unique_usages = []
        for u in usages:
            if u["func_id"] not in seen_funcs:
                seen_funcs.add(u["func_id"])
                unique_usages.append(u)

        return {
            "name": macro_name,
            "definitions": definitions,
            "usages": unique_usages
        }

    def _search_features_with_vector(
        self, query_vec: np.ndarray, top_k: int
    ) -> List[Tuple[str, float]]:
        hits = self.feature_index.search(query_vec, top_k)
        return [
            (meta["feature_id"], _inner_product_similarity(score))
            for meta, score in hits
        ]

    def _search_functions_with_vector(
        self, query_vec: np.ndarray, top_k: int
    ) -> List[_FunctionHit]:
        hits = self.func_index.search(query_vec, top_k)
        return [
            _FunctionHit(
                func_id=meta["func_id"],
                score=_inner_product_similarity(score),
            )
            for meta, score in hits
        ]

    def _best_feature_for_function(
        self, func_id: str, feature_sims: Dict[str, float]
    ) -> Optional[Tuple[str, float]]:
        memberships = self.graph_ctx.func_to_features.get(func_id, [])
        best_feature: Optional[str] = None
        best_score = -1.0
        for membership in memberships:
            feat_id = membership["feature_id"]
            sim = feature_sims.get(feat_id)
            if sim is None:
                continue
            score = sim * membership.get("score", 1.0)
            if score > best_score:
                best_score = score
                best_feature = feat_id
        if best_feature is None:
            return None
        return best_feature, best_score

    def _expand_graph_candidates(
        self, func_hits: List[_FunctionHit], seed_limit: int
    ) -> Dict[str, float]:
        """Propagate semantic seed scores over bounded structural edges."""
        propagated: Dict[str, float] = {}

        def add(nodes: set[str], score: float, limit: int) -> None:
            for node_id in sorted(nodes)[:limit]:
                if node_id not in self.graph_ctx.functions:
                    continue
                propagated[node_id] = max(propagated.get(node_id, 0.0), score)

        for hit in func_hits[:seed_limit]:
            seed_score = max(hit.score, 0.0)
            graph = self.graph_ctx.graph
            for edge_type in (EDGE_CALLS, EDGE_DEP_CALL):
                add(
                    graph.neighbors(hit.func_id, edge_type, direction="in"),
                    seed_score * self.GRAPH_CALL_DECAY,
                    self.GRAPH_CALLER_LIMIT,
                )
                add(
                    graph.neighbors(hit.func_id, edge_type, direction="out"),
                    seed_score * self.GRAPH_CALL_DECAY,
                    self.GRAPH_CALLEE_LIMIT,
                )
            for class_id in graph.neighbors(
                hit.func_id, EDGE_IN_CLASS, direction="out"
            ):
                add(
                    graph.neighbors(class_id, EDGE_IN_CLASS, direction="in"),
                    seed_score * self.GRAPH_SIBLING_DECAY,
                    self.GRAPH_SIBLING_LIMIT,
                )
        return propagated

    def _rank_fusion_scores(
        self,
        scored_results: List[Tuple[str, float]],
        rerank_results: List[Dict[str, Any]],
        candidate_ids: List[str],
    ) -> Dict[str, float]:
        base_rank = {
            func_id: rank
            for rank, (func_id, _) in enumerate(scored_results, 1)
        }
        rerank_rank = {
            candidate_ids[result["index"]]: rank
            for rank, result in enumerate(rerank_results, 1)
            if 0 <= result["index"] < len(candidate_ids)
        }
        return {
            func_id: 1.0 / (self.RRF_K + rank)
            + (
                self.WEIGHT_RERANK
                / (self.RRF_K + rerank_rank[func_id])
                if func_id in rerank_rank
                else 0.0
            )
            for func_id, rank in base_rank.items()
        }

    def _structural_companions(
        self, query: str, ranked: List[Tuple[str, float]], limit: int,
        reranker: Optional[Callable] = None,
    ) -> Tuple[List[Tuple[str, float]], Dict[str, Dict[str, Any]]]:
        """Supplement final semantic seeds with bounded, explicitly traced code neighbors."""
        seeds = ranked[:self.STRUCTURAL_SEEDS]
        slots = min(self.STRUCTURAL_COMPANIONS, limit - len(seeds))
        if slots <= 0:
            return ranked, {}
        graph = self.graph_ctx.graph
        functions = self.graph_ctx.functions
        seed_ids = {fid for fid, _ in seeds}
        seed_rank = {fid: rank for rank, (fid, _) in enumerate(seeds)}
        paths: Dict[str, List[Dict[str, Any]]] = {}

        def add(target, seed_id, edges):
            if target in functions and target not in seed_ids:
                path = {"seed_id": seed_id, "edges": edges}
                candidate_paths = paths.setdefault(target, [])
                if path not in candidate_paths:
                    candidate_paths.append(path)
                    candidate_paths.sort(key=lambda item: (
                        len(item["edges"]),
                        item["seed_id"],
                        tuple(
                            (edge["source"], edge["edge"], edge["target"])
                            for edge in item["edges"]
                        ),
                    ))
                    del candidate_paths[self.STRUCTURAL_PATHS_PER_CANDIDATE:]

        def linked(node_id, edge):
            for direction in ("out", "in"):
                for target in sorted(graph.neighbors(node_id, edge, direction=direction)):
                    source, dest = (
                        (node_id, target) if direction == "out" else (target, node_id)
                    )
                    yield target, {"source": source, "edge": edge, "target": dest}

        def class_members(class_id):
            for member in sorted(graph.neighbors(
                class_id, EDGE_IN_CLASS, direction="in"
            )):
                yield member, {
                    "source": member,
                    "edge": EDGE_IN_CLASS,
                    "target": class_id,
                }

        for seed_id, _ in seeds:
            for owner in sorted(graph.neighbors(
                seed_id, EDGE_IN_CLASS, direction="out"
            )):
                owner_path = [{
                    "source": seed_id,
                    "edge": EDGE_IN_CLASS,
                    "target": owner,
                }]
                for member, member_edge in class_members(owner):
                    member_path = owner_path + [member_edge]
                    add(member, seed_id, member_path)
                    if member == seed_id:
                        continue
                    # A parallel wrapper often carries the concrete forwarding
                    # edge that a dependent template call cannot express.
                    for edge in (
                        EDGE_CALLS,
                        EDGE_DEP_CALL,
                        EDGE_DEP_MEMBER,
                        EDGE_DEP_TEMPLATE,
                    ):
                        for target, target_edge in linked(member, edge):
                            add(target, seed_id, member_path + [target_edge])
            for edge in (
                EDGE_CALLS,
                EDGE_DEP_CALL,
                EDGE_DEP_MEMBER,
                EDGE_DEP_TEMPLATE,
            ):
                for target, target_edge in linked(seed_id, edge):
                    direct_path = [target_edge]
                    add(target, seed_id, direct_path)
                    if target in functions:
                        # Follow one real callee hop from a direct seed callee.
                        # This preserves short state chains without treating a
                        # call from every class sibling as equally strong.
                        if (
                            target_edge["source"] == seed_id
                            and edge in (EDGE_CALLS, EDGE_DEP_CALL)
                        ):
                            for nested_edge_type in (EDGE_CALLS, EDGE_DEP_CALL):
                                for nested in sorted(graph.neighbors(
                                    target,
                                    nested_edge_type,
                                    direction="out",
                                )):
                                    if nested == target:
                                        continue
                                    add(nested, seed_id, direct_path + [{
                                        "source": target,
                                        "edge": nested_edge_type,
                                        "target": nested,
                                    }])
                        continue
                    for member, member_edge in class_members(target):
                        add(member, seed_id, direct_path + [member_edge])
                    # Template specialization graphs can contain an
                    # intermediate template class before the concrete class.
                    if edge == EDGE_DEP_TEMPLATE:
                        for nested, nested_edge in linked(target, edge):
                            if nested in functions:
                                add(nested, seed_id, direct_path + [nested_edge])
                                continue
                            for member, member_edge in class_members(nested):
                                add(
                                    member,
                                    seed_id,
                                    direct_path + [nested_edge, member_edge],
                                )
        if not paths:
            return ranked, {}

        def code_terms(value):
            terms = set()
            for token in re.findall(r"[a-zA-Z][a-zA-Z_0-9]*", value):
                for part in token.split("_"):
                    terms.update(
                        word.lower()
                        for word in re.findall(
                            r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", part
                        )
                        if len(word) >= 2
                    )
            return terms

        def api_name(function):
            value = str(function.get("fq_name") or function.get("name") or "")
            return value.rsplit("::", 1)[-1].split("(", 1)[0].strip().lower()

        terms = code_terms(query)

        def evidence_tier(fid):
            """Keep executable paths ahead of broader structural adjacency."""
            best = 3
            for path in paths[fid]:
                edge_types = [edge["edge"] for edge in path["edges"]]
                if not edge_types:
                    continue
                if len(edge_types) == 1 and edge_types[-1] in (
                    EDGE_CALLS, EDGE_DEP_CALL,
                ):
                    best = min(best, 0)
                elif all(
                    edge in (EDGE_CALLS, EDGE_DEP_CALL) for edge in edge_types
                ):
                    best = min(best, 1)
                elif any(
                    edge in (
                        EDGE_CALLS,
                        EDGE_DEP_CALL,
                        EDGE_DEP_MEMBER,
                        EDGE_DEP_TEMPLATE,
                    )
                    for edge in edge_types
                ):
                    best = min(best, 2)
            return best

        def priority(fid):
            f = functions[fid]
            identity_words = code_terms(
                str(f.get("fq_name", "")) + " "
                + str(f.get("signature", "")) + " "
                + str(f.get("short_intent", ""))
            )
            source_words = code_terms(str(f.get("code", ""))[:800])
            candidate_name = api_name(f)
            family_affinity = 0
            for path in paths[fid]:
                seed_name = api_name(functions[path["seed_id"]])
                exact_weight = max(1, 4 - 2 * seed_rank[path["seed_id"]])
                if candidate_name == seed_name:
                    family_affinity = max(family_affinity, exact_weight)
                elif (
                    seed_name
                    and candidate_name
                    and (
                        candidate_name.startswith(seed_name + "_")
                        or seed_name.startswith(candidate_name + "_")
                    )
                ):
                    family_affinity = max(family_affinity, exact_weight - 1)
            path_length = min(len(path["edges"]) for path in paths[fid])
            distance = min(
                abs((f.get("start_line") or 0) - (functions[p["seed_id"]].get("start_line") or 0))
                if f.get("file") == functions[p["seed_id"]].get("file") else float("inf")
                for p in paths[fid]
            )
            return (
                evidence_tier(fid),
                -len(terms & identity_words),
                -len(terms & source_words),
                -family_affinity,
                path_length,
                distance,
                f.get("file") or "",
                f.get("start_line") or 0,
                f.get("signature") or "",
                fid,
            )

        # Bound provider input by evidence and relevance, while preventing one
        # overloaded API name from consuming the whole reranker budget.
        candidates = []
        family_counts: Dict[str, int] = {}
        seed_variant_counts: Dict[Tuple[str, str, str], int] = {}
        deferred_executable = []
        executable_count = 0
        executable_limit = max(
            0,
            self.STRUCTURAL_CANDIDATES
            - self.STRUCTURAL_MIXED_PATH_RESERVE,
        )

        seed_owners = {
            seed_id: set(graph.neighbors(
                seed_id, EDGE_IN_CLASS, direction="out"
            ))
            for seed_id, _ in seeds
        }

        def seed_variant_family(fid):
            candidate_name = api_name(functions[fid])
            candidate_owners = set(graph.neighbors(
                fid, EDGE_IN_CLASS, direction="out"
            ))
            for seed_id, _ in seeds:
                seed_name = api_name(functions[seed_id])
                shared_owners = candidate_owners & seed_owners[seed_id]
                if seed_name and candidate_name.startswith(seed_name + "_"):
                    if shared_owners:
                        return (seed_id, min(shared_owners), seed_name)
            return None

        def admit(fid):
            function = functions[fid]
            family = str(
                function.get("fq_name")
                or fid
            )
            if family_counts.get(family, 0) >= self.STRUCTURAL_PER_API_FAMILY:
                return False
            variant_family = (
                seed_variant_family(fid) if evidence_tier(fid) > 1 else None
            )
            if (
                variant_family is not None
                and seed_variant_counts.get(variant_family, 0)
                >= self.STRUCTURAL_PER_SEED_VARIANT_FAMILY
            ):
                return False
            candidates.append(fid)
            family_counts[family] = family_counts.get(family, 0) + 1
            if variant_family is not None:
                seed_variant_counts[variant_family] = (
                    seed_variant_counts.get(variant_family, 0) + 1
                )
            return True

        for fid in sorted(paths, key=priority):
            executable = evidence_tier(fid) <= 1
            if executable and executable_count >= executable_limit:
                deferred_executable.append(fid)
                continue
            if admit(fid) and executable:
                executable_count += 1
            if len(candidates) >= self.STRUCTURAL_CANDIDATES:
                break
        for fid in deferred_executable:
            if len(candidates) >= self.STRUCTURAL_CANDIDATES:
                break
            admit(fid)
        documents = []
        for fid in candidates:
            f = functions[fid]
            origins = []
            for path in paths[fid]:
                seed = functions[path["seed_id"]]
                origins.append({
                    "seed": seed.get("fq_name") or seed.get("signature"),
                    "seed_intent": seed.get("short_intent"),
                    "seed_code": seed.get("code", "")[:800],
                    "path": [{
                        "source": graph.nodes[e["source"]].get("fq_name") or e["source"],
                        "edge": e["edge"],
                        "target": graph.nodes[e["target"]].get("fq_name") or e["target"],
                    } for e in path["edges"]],
                })
            documents.append(json.dumps({
                "companion": f.get("fq_name") or f.get("signature"), "intent": f.get("short_intent"),
                "file": f.get("file"), "start_line": f.get("start_line"),
                "code": f.get("code", "")[:1600], "connections": origins,
            }, ensure_ascii=False))
        try:
            results = (reranker or rerank_documents)(
                "Rank the connected code that completes the semantic seed's behavior for "
                "this issue. Prefer underlying implementations, concrete forwarding "
                "targets, and parallel entry points in the same API family. Also include "
                "state/configuration producers when they control the behavior. Use the "
                "shown graph path and source; reject methods that are merely adjacent. "
                "Issue:\n" + query,
                documents, top_k=slots, strict=True,
            )
            selected = []
            for result in results:
                fid = candidates[result["index"]]
                if fid not in {item[0] for item in selected}:
                    selected.append((fid, result["score"]))
            selected = selected[:slots]
        except Exception:
            logger.warning("Structural rerank failed; retaining semantic results", exc_info=False)
            return ranked, {seeds[0][0]: {"structural_status": "rerank_failed"}}
        metadata = {fid: {"retrieval_role": "semantic_seed"} for fid, _ in seeds}
        for fid, _ in selected:
            metadata[fid] = {"retrieval_role": "structural_companion", "graph_paths": paths[fid]}
        metadata[seeds[0][0]]["structural_candidates"] = len(candidates)
        combined = seeds + selected
        selected_ids = {fid for fid, _ in combined}
        combined += [item for item in ranked if item[0] not in selected_ids]
        return combined[:limit], metadata

    def search_code(
        self, query: str, top_k_funcs: int = 50, *, rerank_first: bool = False,
        expand_structure: bool = False,
        structural_reranker: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        if top_k_funcs <= 0:
            return []
        query_vec = self._encode_query(query)
        class_hits = self._search_classes_with_vector(query_vec, top_k=5)
        class_sim_map = {hit["class_id"]: hit["score"] for hit in class_hits}
        feature_hits = self._search_features_with_vector(
            query_vec, max(self.FEATURE_TOP_K, top_k_funcs)
        )
        feature_sims = {fid: sim for fid, sim in feature_hits}

        func_hits = self._search_functions_with_vector(
            query_vec, top_k_funcs * self.FUNC_CAND_MULTIPLIER
        )
        candidate_ids = {hit.func_id for hit in func_hits}
        for feature_id, feat_sim in feature_sims.items():
            if feat_sim < 0.1:
                continue
            members = self.graph_ctx.feature_to_funcs.get(feature_id, [])
            candidate_ids.update(members)
        func_sim_map = {hit.func_id: hit.score for hit in func_hits}
        graph_scores = self._expand_graph_candidates(
            func_hits,
            seed_limit=max(top_k_funcs * self.GRAPH_SEED_MULTIPLIER, top_k_funcs),
        )
        candidate_ids.update(graph_scores)
        scored_results: List[Tuple[str, float]] = []
        for func_id in candidate_ids:
            func_data = self.graph_ctx.functions.get(func_id)
            if not func_data:
                continue
            func_similarity = func_sim_map.get(func_id, 0.0)
            feature_bonus = 0.0
            for membership in self.graph_ctx.func_to_features.get(func_id, []):
                feat_sim = feature_sims.get(membership["feature_id"], 0.0)
                mem_score = max(membership.get("score", 1.0), 0.3)
                feature_bonus = max(feature_bonus, feat_sim * mem_score)

            graph_bonus = graph_scores.get(func_id, 0.0)
            
            # Class Bonus
            class_bonus = 0.0
            parent_classes = self.graph_ctx.graph.neighbors(
                func_id, EDGE_IN_CLASS, direction="out"
            )
            for cls_id in parent_classes:
                if cls_id in class_sim_map:
                    class_bonus = max(class_bonus, class_sim_map[cls_id])

            text_bonus = 0.0
            q_lower = query.lower()
            f_intent = (func_data.get("short_intent") or "").lower()
            f_sig = (func_data.get("signature") or "").lower()
            if len(q_lower) > 5 and q_lower in f_intent:
                text_bonus = 1.0
            elif len(f_intent) > 5 and f_intent in q_lower:
                text_bonus = 0.8
            elif q_lower in f_sig:
                text_bonus = 0.5
            base_score = (
                self.WEIGHT_FUNC * func_similarity
                + self.WEIGHT_FEATURE * feature_bonus
                + self.WEIGHT_GRAPH * graph_bonus
                + self.WEIGHT_CLASS * class_bonus
                + self.WEIGHT_TEXT * text_bonus
            )
            if func_similarity > 0.85:
                base_score += 0.3

            scored_results.append((func_id, base_score))

        scored_results.sort(key=lambda x: x[1], reverse=True)
        rerank_candidates = scored_results[:self.RERANK_TOP_K]
        if rerank_candidates:
            doc_texts = []
            cand_ids = [cid for cid, _ in rerank_candidates]
            for cid in cand_ids:
                f = self.graph_ctx.functions[cid]
                best_feature = self._best_feature_for_function(cid, feature_sims)
                feature_info = ""
                if best_feature:
                     feat_data = self.graph_ctx.features.get(best_feature[0])
                     if feat_data:
                         feature_info = f"\nFeature Context: {feat_data.get('feature_name', '')} - {feat_data.get('feature_description', '')}"

                code_snippet = f.get("code", "")[:800]
                doc_text = (
                    f"Intent: {f.get('short_intent', '')}\n"
                    f"Signature: {f.get('signature', '')}\n"
                    f"Summary: {f.get('raw_summary', '')}\n"
                    f"Code: {code_snippet}"
                    f"{feature_info}"
                )
                doc_texts.append(doc_text)
            
            try:
                rerank_results = rerank_documents(
                    query, doc_texts, top_k=self.RERANK_TOP_K
                )
                if rerank_first:
                    # Scout selects navigation leads using the reranker's final
                    # judgment; fusion can otherwise hide its strongest hit.
                    top_results = [
                        (cand_ids[result["index"]], result["score"])
                        for result in rerank_results[:top_k_funcs]
                    ]
                else:
                    fusion_scores = self._rank_fusion_scores(
                        scored_results, rerank_results, cand_ids
                    )
                    final_scored_results = [
                        (cid, fusion_scores[cid]) for cid, _ in scored_results
                    ]
                    final_scored_results.sort(key=lambda x: x[1], reverse=True)
                    top_results = final_scored_results[:top_k_funcs]
            except Exception as e:
                if "API key" in str(e):
                    print("Voyage Rerank API Key invalid, skipping rerank stage.")
                else:
                    print(f"Rerank failed: {e}")
                top_results = scored_results[:top_k_funcs]
        else:
            top_results = scored_results[:top_k_funcs]
        structural_metadata = {}
        if expand_structure:
            top_results, structural_metadata = self._structural_companions(
                query, top_results, top_k_funcs, reranker=structural_reranker)
        formatted: List[Dict[str, Any]] = []
        for func_id, score in top_results:
            func_data = self.graph_ctx.functions.get(func_id)
            if not func_data:
                continue
            best_feature = self._best_feature_for_function(func_id, feature_sims)
            feature_record = (
                self.graph_ctx.features.get(best_feature[0])
                if best_feature
                else None
            )
            formatted.append(
                {
                    "func_id": func_id,
                    "fq_name": func_data.get("fq_name"),
                    "file": func_data.get("file"),
                    "start_line": func_data.get("start_line"),
                    "end_line": func_data.get("end_line"),
                    "short_intent": func_data.get("short_intent"),
                    "feature_id": feature_record.get("feature_id") if feature_record else None,
                    "feature_name": feature_record.get("feature_name") if feature_record else None,
                    "score": round(score, 4),
                    "content_hash": func_data.get("content_hash"),
                    **structural_metadata.get(func_id, {}),
                }
            )
        return formatted


_RAG_INSTANCE: Optional[CodeIntentRAG] = None
_RAG_INDEX_DIR: Optional[str] = None


def _resolve_index_dir(
    index_dir: Optional[str], instance_id: Optional[str]
) -> str:
    if index_dir:
        return index_dir
    if instance_id:
        derived = (
            INDEX_ROOT
            / instance_id
            / "intent_output"
            / instance_id
            / "index_output"
        )
        return str(derived)
    return "index_output"


def _get_rag(index_dir: str) -> CodeIntentRAG:
    global _RAG_INSTANCE, _RAG_INDEX_DIR
    if _RAG_INSTANCE is None or _RAG_INDEX_DIR != index_dir:
        _RAG_INSTANCE = CodeIntentRAG(index_dir)
        _RAG_INDEX_DIR = index_dir
    return _RAG_INSTANCE


if FastAPI is not None:
    class _SearchRequest(BaseModel):
        query: str = Field(..., description="Natural language query", min_length=1)
        top_k: int = Field(10, description="Number of results", gt=0, le=200)
        instance_id: Optional[str] = Field(
            default=None,
            description="SweBench instance ID (derives index dir)",
        )
        index_dir: Optional[str] = Field(
            default=None,
            description="Custom index dir, overrides instance_id",
        )


    class _FeatureSearchRequest(BaseModel):
        query: str = Field(..., description="Natural language query", min_length=1)
        top_k: int = Field(10, description="Number of results", gt=0, le=200)
        instance_id: Optional[str] = Field(
            default=None,
            description="SweBench instance ID (derives index dir)",
        )
        index_dir: Optional[str] = Field(
            default=None,
            description="Custom index dir, overrides instance_id",
        )


    class _MacroSearchRequest(BaseModel):
        name: str = Field(..., description="Macro name", min_length=1)
        instance_id: Optional[str] = Field(default=None)
        index_dir: Optional[str] = Field(default=None)


    app = FastAPI(
        title="Code Intent Search API",
        description="Code search service by intent and graph",
        version="0.1.0",
    )


    @app.get("/healthz")
    def healthz() -> Dict[str, str]:
        return {"status": "ok"}


    @app.post("/search/functions")
    def search_functions(req: _SearchRequest) -> Dict[str, Any]:
        resolved_dir = _resolve_index_dir(req.index_dir, req.instance_id)
        rag = _get_rag(resolved_dir)
        try:
            results = rag.search_code(req.query, req.top_k)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"results": results}


    @app.post("/search/features")
    def search_features(req: _FeatureSearchRequest) -> Dict[str, Any]:
        resolved_dir = _resolve_index_dir(req.index_dir, req.instance_id)
        rag = _get_rag(resolved_dir)
        try:
            results = rag.search_features(req.query, req.top_k)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"results": results}


    @app.post("/search/macros")
    def search_macros(req: _MacroSearchRequest) -> Dict[str, Any]:
        resolved_dir = _resolve_index_dir(req.index_dir, req.instance_id)
        rag = _get_rag(resolved_dir)
        try:
            details = rag.get_macro_details(req.name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return details
else:
    app = None
