

from __future__ import annotations

import json
import hashlib
import os
import sqlite3
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import faiss
import numpy as np

try:
    import igraph as ig
    import leidenalg as la
except ImportError:  # pragma: no cover - exercised by dependency preflight
    ig = None
    la = None

from .embedding_manifest import (
    FEATURE_INTENT_BUILDER,
    load_embedding_manifest,
    write_embedding_manifest_entry,
)
from .embedding_cache import GlobalEmbeddingCache, embedding_cache_writes_allowed
from .edge_utils import iter_executable_edges
from .embedding_text import build_feature_embedding_text
from .intent_params import (
    BIMODAL_MIN_RATIO,
    BIMODAL_SEPARATION,
    CHILD_CLUSTER_MIN_SIZE,
    CLUSTER_ALPHA,
    LEIDEN_ITERATIONS,
    LEIDEN_RESOLUTION,
    MAX_CLUSTER_SPLIT_DEPTH,
    REPRESENTATIVE_DEGREE_WEIGHT,
    SEMANTIC_TOP_K,
    SIMILARITY_THRESHOLD,
)
from .llm_utils import (
    EMBED_DIM,
    FEATURE_SUMMARY_VALIDATION_CONTRACT,
    _cluster_summary_grounding_issues,
    cluster_summary_input_hash,
    embed_texts,
    feature_summary_prompt_contract,
    summarize_cluster,
    summarize_test_feature,
    test_feature_summary_input_hash,
)

MIN_CLUSTER_SIZE = 8
REPRESENTATIVE_TOP_K = 5
TEST_COVERAGE_MAX_HOPS = 2
TEST_COVERAGE_MAX_FUNCS = 200
CLASS_FEATURE_MIN_MEMBERS = 2
TEST_GROUP_MAX_SIZE = 5
TEST_GROUP_SIMILARITY_THRESHOLD = 0.6
FEATURE_EMBED_BATCH_SIZE = max(
    1,
    int(os.getenv("INTENT_EMBED_BATCH_SIZE", "100")),
)
FEATURE_SUMMARY_CACHE_COMMIT_BATCH = 1


class _FeatureSummaryCache:
    def __init__(self, repo_id: Optional[str]) -> None:
        cache_root = os.getenv("INTENT_CACHE_DIR") or os.getenv(
            "CODE_INTENT_CACHE_DIR"
        )
        self.connection: Optional[sqlite3.Connection] = None
        self.hits = 0
        self.requests = 0
        self.pending_writes = 0
        if not cache_root:
            return
        cache_dir = Path(cache_root) / (repo_id or "common")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(
            cache_dir / "feature_summaries.sqlite3", timeout=60.0
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS summaries (
                cache_key TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    @staticmethod
    def _reusable(summary: Dict[str, Any]) -> bool:
        expected = "llm" if os.getenv("LLM_MODEL") else "heuristic"
        return summary.get("_intent_generation") == expected

    def _get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        if self.connection is None:
            return None
        row = self.connection.execute(
            "SELECT payload FROM summaries WHERE cache_key = ?", (cache_key,)
        ).fetchone()
        if row is None:
            return None
        try:
            summary = json.loads(row[0])
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(summary, dict) or not self._reusable(summary):
            return None
        self.hits += 1
        return summary

    def _put(self, cache_key: str, summary: Dict[str, Any]) -> None:
        if self.connection is None or not self._reusable(summary):
            return
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO summaries (cache_key, payload) VALUES (?, ?)",
            (cache_key, json.dumps(summary, ensure_ascii=False, sort_keys=True)),
        )
        if cursor.rowcount:
            self.pending_writes += 1
        if self.pending_writes >= FEATURE_SUMMARY_CACHE_COMMIT_BATCH:
            self.connection.commit()
            self.pending_writes = 0

    def summarize_cluster(
        self,
        members: List[Dict[str, Any]],
        cluster_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cache_key = cluster_summary_input_hash(members, cluster_context)
        cached = self._get(cache_key)
        if cached is not None:
            if not _cluster_summary_grounding_issues(cached, members):
                return cached
            self.hits -= 1
            assert self.connection is not None
            self.connection.execute(
                "DELETE FROM summaries WHERE cache_key = ?", (cache_key,)
            )
        self.requests += 1
        summary = summarize_cluster(members, cluster_context)
        self._put(cache_key, summary)
        return summary

    def summarize_test_feature(
        self, test_intent: Dict[str, Any], members: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        cache_key = test_feature_summary_input_hash(test_intent, members)
        cached = self._get(cache_key)
        if cached is not None:
            return cached
        self.requests += 1
        summary = summarize_test_feature(test_intent, members)
        self._put(cache_key, summary)
        return summary

    def close(self) -> None:
        if self.connection is not None:
            self.connection.commit()
            self.pending_writes = 0
            self.connection.close()


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_jsonl_atomic(records: Iterable[Dict[str, Any]], destination: Path) -> None:
    _ensure_parent_dir(destination)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp_path.replace(destination)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_function_intents(path: Path) -> List[Dict[str, Any]]:
    return _load_jsonl(path)


def _load_meta(path: Path, id_key: str) -> List[Dict[str, Any]]:
    records = _load_jsonl(path)
    records.sort(key=lambda x: x.get("index", 0))
    for rec in records:
        if id_key not in rec:
            raise ValueError(f"meta file missing {id_key}: {path}")
    return records


def _load_embeddings(index_path: Path, count: int) -> np.ndarray:
    if count == 0:
        return np.zeros((0, EMBED_DIM), dtype="float32")
    index = faiss.read_index(str(index_path))
    if index.ntotal != count:
        raise ValueError(
            f"FAISS index count {index.ntotal} != meta record count {count}"
        )
    if index.d != EMBED_DIM:
        raise ValueError(f"FAISS dimension {index.d} != runtime dimension {EMBED_DIM}")
    try:
        vectors = index.reconstruct_n(0, count)
    except (AttributeError, RuntimeError):
        vectors = np.vstack([index.reconstruct(i) for i in range(count)])
    return vectors.astype("float32")


def _load_callgraph(path: Path) -> Dict[str, List[str]]:
    graph: Dict[str, Set[str]] = defaultdict(set)
    for edge in iter_executable_edges(path):
        graph[edge["caller"]].add(edge["callee"])
    return {caller: sorted(callees) for caller, callees in graph.items()}


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _collect_test_coverage(
    test_id: str,
    adjacency: Dict[str, List[str]],
    func_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, int]:
    coverage: Dict[str, int] = {}
    visited: Set[str] = set()
    queue: deque[Tuple[str, int]] = deque()
    visited.add(test_id)
    queue.append((test_id, 0))

    while queue and len(coverage) < TEST_COVERAGE_MAX_FUNCS:
        node, hop = queue.popleft()
        if hop >= TEST_COVERAGE_MAX_HOPS:
            continue
        for callee in adjacency.get(node, []):
            if callee in visited:
                continue
            visited.add(callee)
            queue.append((callee, hop + 1))
            if callee in func_lookup and callee not in coverage:
                coverage[callee] = hop + 1
    return coverage


def _combine_coverages(
    test_ids: List[str], coverage_map: Dict[str, Dict[str, int]]
) -> Dict[str, int]:
    combined: Dict[str, int] = {}
    for tid in test_ids:
        for fid, hop in coverage_map.get(tid, {}).items():
            if fid not in combined or hop < combined[fid]:
                combined[fid] = hop
    return combined


def _combine_test_intents(tests: List[Dict[str, Any]]) -> Dict[str, Any]:
    short_parts = [t.get("short_intent") for t in tests if t.get("short_intent")]
    raw_parts = [t.get("raw_summary") for t in tests if t.get("raw_summary")]
    return {
        "short_intent": " / ".join(short_parts) if short_parts else "Test-driven scenario",
        "raw_summary": " ".join(raw_parts) if raw_parts else "Combined test intent.",
    }


def _test_bucket_key(test: Optional[Dict[str, Any]]) -> str:
    if not test:
        return "bucket:unknown"
    file_path = test.get("file")
    if file_path:
        parts = Path(file_path).parts
        key = "/".join(parts[-3:]) if len(parts) >= 3 else "/".join(parts)
        return f"file:{key}"
    namespaces = "::".join(test.get("namespaces") or [])
    if namespaces:
        return f"ns:{namespaces}"
    short = (test.get("short_intent") or test.get("signature") or "").strip()
    if short:
        tokens = short.lower().split()
        return "name:" + " ".join(tokens[:3])
    return f"id:{test.get('func_id', 'unknown')}"


def _group_tests_within_bucket(
    bucket_ids: List[str], coverage_map: Dict[str, Dict[str, int]]
) -> List[List[str]]:
    groups: List[List[str]] = []
    remaining = sorted(set(bucket_ids))

    def jaccard(a: Set[str], b: Set[str]) -> float:
        if not a and not b:
            return 0.0
        return len(a & b) / len(a | b)

    while remaining:
        seed = remaining.pop(0)
        group = [seed]
        seed_cov = set(coverage_map.get(seed, {}))
        to_remove: List[str] = []
        for other in list(remaining):
            if len(group) >= TEST_GROUP_MAX_SIZE:
                break
            other_cov = set(coverage_map.get(other, {}))
            if not other_cov:
                continue
            similarity = jaccard(seed_cov, other_cov)
            if similarity >= TEST_GROUP_SIMILARITY_THRESHOLD:
                group.append(other)
                to_remove.append(other)
        for item in to_remove:
            remaining.remove(item)
        groups.append(group)
    return groups


def _group_tests(
    test_ids: List[str],
    coverage_map: Dict[str, Dict[str, int]],
    test_lookup: Dict[str, Dict[str, Any]],
) -> List[List[str]]:
    buckets: Dict[str, List[str]] = defaultdict(list)
    for tid in test_ids:
        buckets[_test_bucket_key(test_lookup.get(tid))].append(tid)

    groups: List[List[str]] = []
    for bucket_key in sorted(buckets):
        bucket_ids = buckets[bucket_key]
        groups.extend(_group_tests_within_bucket(bucket_ids, coverage_map))
    return groups


def _build_test_feature_payloads(
    func_lookup: Dict[str, Dict[str, Any]],
    test_intents: List[Dict[str, Any]],
    callgraph: Dict[str, List[str]],
    summary_cache: Optional[_FeatureSummaryCache] = None,
) -> Tuple[List[Tuple[Dict[str, Any], Dict[str, float]]], Set[str]]:
    test_lookup = {test["func_id"]: test for test in test_intents}
    coverage_map: Dict[str, Dict[str, int]] = {}
    for test in test_intents:
        source_ids = sorted(
            {
                str(func_id)
                for func_id in (test.get("source_func_ids") or [])
                if func_id
            }
        )
        source_ids = source_ids or [test["func_id"]]
        coverage: Dict[str, int] = {}
        for source_id in source_ids:
            for func_id, hop in _collect_test_coverage(
                source_id, callgraph, func_lookup
            ).items():
                coverage[func_id] = min(coverage.get(func_id, hop), hop)
        focal_ids = sorted(
            {
                str(focal.get("func_id"))
                for focal in (test.get("focal_entities") or [])
                if focal.get("func_id") in func_lookup
            }
        )
        for focal_id in focal_ids:
            coverage[focal_id] = 0
            for func_id, hop in _collect_test_coverage(
                focal_id, callgraph, func_lookup
            ).items():
                adjusted_hop = hop + 1
                coverage[func_id] = min(
                    coverage.get(func_id, adjusted_hop), adjusted_hop
                )
        if coverage:
            coverage_map[test["func_id"]] = coverage

    grouped_tests = _group_tests(
        list(coverage_map.keys()), coverage_map, test_lookup
    )

    payloads: List[Tuple[Dict[str, Any], Dict[str, float]]] = []
    covered_funcs: Set[str] = set()

    for group in grouped_tests:
        combined_coverage = _combine_coverages(group, coverage_map)
        members = [
            func_lookup[fid] for fid in sorted(combined_coverage) if fid in func_lookup
        ]
        if not members:
            continue
        combined_test_intent = _combine_test_intents(
            [test_lookup[tid] for tid in group]
        )
        summary = (
            summary_cache.summarize_test_feature(combined_test_intent, members)
            if summary_cache is not None
            else summarize_test_feature(combined_test_intent, members)
        )
        representative = sorted(
            combined_coverage.items(),
            key=lambda x: (x[1], x[0]),
        )[:REPRESENTATIVE_TOP_K]
        feature = {
            "feature_name": summary["feature_name"],
            "feature_description": summary["feature_description"],
            "tags": summary.get("tags", []),
            "member_funcs": sorted(combined_coverage),
            "representative_funcs": [fid for fid, _ in representative],
            "is_noise": False,
            "feature_type": "test",
            "source_tests": sorted(group),
            "summary_generation": summary.get("_intent_generation", "unknown"),
        }
        memberships = {
            fid: 1.0 / (hop + 1) for fid, hop in combined_coverage.items()
        }
        payloads.append((feature, memberships))
        covered_funcs.update(combined_coverage.keys())

    return payloads, covered_funcs


def _build_graph_communities(
    func_ids: List[str],
    callgraph: Dict[str, List[str]],
    vector_lookup: Optional[Dict[str, np.ndarray]] = None,
    min_cluster_size: Optional[int] = None,
) -> List[List[str]]:
    if not func_ids:
        return []
    if ig is None or la is None:
        raise RuntimeError(
            "feature intent construction requires python-igraph and leidenalg"
        )

    func_ids = sorted(set(func_ids))
    func_set = set(func_ids)
    id_to_idx = {fid: i for i, fid in enumerate(func_ids)}
    edge_data: Dict[Tuple[int, int], Dict[str, float]] = {}
    for caller in sorted(callgraph):
        callees = callgraph[caller]
        if caller not in func_set:
            continue
        u = id_to_idx[caller]
        
        for callee in sorted(set(callees)):
            if callee not in func_set:
                continue
            if caller == callee:
                continue
            
            v = id_to_idx[callee]
            edge_data[(u, v)] = {"struct": 1.0, "sem": 0.0}

    if vector_lookup:
        neighbor_ids = []
        neighbor_vecs = []
        for fid in func_ids:
            vec = vector_lookup.get(fid)
            if vec is None:
                neighbor_ids.append(None)
                neighbor_vecs.append(None)
            else:
                neighbor_ids.append(fid)
                neighbor_vecs.append(vec)

        if any(v is not None for v in neighbor_vecs):
            valid_indices = [i for i, v in enumerate(neighbor_vecs) if v is not None]
            matrix = np.vstack([neighbor_vecs[i] for i in valid_indices]).astype("float32")
            if matrix.size:
                index = faiss.IndexFlatIP(matrix.shape[1])
                index.add(matrix)
                top_k = min(SEMANTIC_TOP_K + 1, matrix.shape[0])
                scores, indices = index.search(matrix, top_k)
                for row_idx, (row_scores, row_indices) in enumerate(zip(scores, indices)):
                    src_global_idx = valid_indices[row_idx]
                    for score, col_idx in zip(row_scores, row_indices):
                        if col_idx < 0 or score < SIMILARITY_THRESHOLD:
                            continue
                        dst_global_idx = valid_indices[col_idx]
                        if src_global_idx == dst_global_idx:
                            continue
                        edge_key = (src_global_idx, dst_global_idx)
                        existing = edge_data.get(edge_key)
                        if existing:
                            existing["sem"] = max(existing["sem"], float(score))
                        else:
                            edge_data[edge_key] = {"struct": 0.0, "sem": float(score)}

    if not edge_data:
        return []

    edges: List[Tuple[int, int]] = []
    weights: List[float] = []
    for (u, v), parts in sorted(edge_data.items()):
        w_struct = parts.get("struct", 0.0)
        w_sem = parts.get("sem", 0.0)
        weight = (1.0 - CLUSTER_ALPHA) * w_struct + CLUSTER_ALPHA * w_sem
        if weight <= 0:
            continue
        edges.append((u, v))
        weights.append(weight)

    if not edges:
        return []

    g = ig.Graph(n=len(func_ids), edges=edges, directed=True)
    g.es["weight"] = weights
    partition = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        resolution_parameter=LEIDEN_RESOLUTION,
        weights="weight",
        n_iterations=LEIDEN_ITERATIONS,
        seed=42,
    )
    communities = []
    threshold = min_cluster_size or MIN_CLUSTER_SIZE
    for community_indices in partition:
        if len(community_indices) < threshold:
            continue
        members = sorted(func_ids[idx] for idx in community_indices)
        communities.append(members)

    communities.sort(key=lambda members: (members[0], len(members), members))
    return communities


def _cluster_split_bimodal(
    members_ids: List[str],
    vector_lookup: Dict[str, np.ndarray],
) -> Optional[List[List[str]]]:
    vectors = [vector_lookup.get(fid) for fid in members_ids]
    valid = [(fid, vec) for fid, vec in zip(members_ids, vectors) if vec is not None]
    if len(valid) < CHILD_CLUSTER_MIN_SIZE * 2:
        return None
    ids = [fid for fid, _ in valid]
    mat = np.vstack([vec for _, vec in valid]).astype("float32")
    centroid = mat.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm:
        centroid = centroid / norm
    sims = mat @ centroid
    if sims.size < CHILD_CLUSTER_MIN_SIZE * 2:
        return None

    # 1D 2-means
    c1, c2 = float(sims.min()), float(sims.max())
    if abs(c2 - c1) < BIMODAL_SEPARATION:
        return None
    labels = None
    for _ in range(6):
        d1 = np.abs(sims - c1)
        d2 = np.abs(sims - c2)
        labels = d1 <= d2
        if labels.all() or (~labels).all():
            return None
        c1 = float(sims[labels].mean())
        c2 = float(sims[~labels].mean())

    if labels is None:
        return None
    size_a = int(labels.sum())
    size_b = int((~labels).sum())
    min_size = min(size_a, size_b)
    if min_size < CHILD_CLUSTER_MIN_SIZE:
        return None
    if min_size / max(size_a, size_b) < BIMODAL_MIN_RATIO:
        return None
    if abs(c2 - c1) < BIMODAL_SEPARATION:
        return None

    group_a = sorted(ids[i] for i, flag in enumerate(labels) if flag)
    group_b = sorted(ids[i] for i, flag in enumerate(labels) if not flag)
    return sorted([group_a, group_b], key=lambda group: (group[0], len(group), group))


def _pick_cluster_representatives(
    members_ids: List[str],
    callgraph: Dict[str, List[str]],
    vector_lookup: Dict[str, np.ndarray],
    top_k: int,
) -> List[str]:
    if not members_ids:
        return []
    if len(members_ids) <= top_k:
        return sorted(members_ids)

    member_set = set(members_ids)
    degree_scores: Dict[str, float] = {fid: 0.0 for fid in members_ids}
    for caller in members_ids:
        for callee in callgraph.get(caller, []):
            if callee in member_set:
                degree_scores[caller] += 1.0
                degree_scores[callee] += 1.0

    max_degree = max(degree_scores.values()) if degree_scores else 0.0
    degree_norm = {
        fid: (score / max_degree if max_degree > 0 else 0.0)
        for fid, score in degree_scores.items()
    }

    vectors = [vector_lookup.get(fid) for fid in members_ids]
    valid_vecs = [vec for vec in vectors if vec is not None]
    sim_norm: Dict[str, float] = {fid: 0.0 for fid in members_ids}
    if valid_vecs:
        mat = np.vstack(valid_vecs).astype("float32")
        centroid = mat.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm:
            centroid = centroid / norm
        sims: List[Tuple[str, float]] = []
        for fid in members_ids:
            vec = vector_lookup.get(fid)
            if vec is None:
                sims.append((fid, 0.0))
                continue
            sims.append((fid, float(vec @ centroid)))
        sim_values = [s for _, s in sims]
        min_sim = min(sim_values)
        max_sim = max(sim_values)
        for fid, sim in sims:
            if max_sim > min_sim:
                sim_norm[fid] = (sim - min_sim) / (max_sim - min_sim)
            else:
                sim_norm[fid] = 0.0

    weight = REPRESENTATIVE_DEGREE_WEIGHT
    scored = []
    for fid in members_ids:
        score = weight * degree_norm.get(fid, 0.0) + (1.0 - weight) * sim_norm.get(fid, 0.0)
        scored.append((fid, score))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return [fid for fid, _ in scored[:top_k]]


def _select_cluster_summary_ids(
    members_ids: List[str],
    representatives: List[str],
    intent_lookup: Dict[str, Dict[str, Any]],
    limit: int = 50,
) -> List[str]:
    """Keep graph centers while sampling the full cluster deterministically."""
    member_set = set(members_ids)
    selected = list(
        dict.fromkeys(fid for fid in representatives if fid in member_set)
    )[:limit]
    remaining_slots = limit - len(selected)
    if remaining_slots <= 0:
        return selected

    selected_set = set(selected)
    remaining = sorted(
        (fid for fid in member_set if fid not in selected_set),
        key=lambda fid: (
            str(intent_lookup.get(fid, {}).get("file") or ""),
            str(intent_lookup.get(fid, {}).get("class_name") or ""),
            str(intent_lookup.get(fid, {}).get("signature") or ""),
            fid,
        ),
    )
    if len(remaining) <= remaining_slots:
        return [*selected, *remaining]
    if remaining_slots == 1:
        sampled = [remaining[len(remaining) // 2]]
    else:
        sampled = [
            remaining[index * (len(remaining) - 1) // (remaining_slots - 1)]
            for index in range(remaining_slots)
        ]
    return [*selected, *sampled]


def _refine_clusters_recursive(
    members_ids: List[str],
    callgraph: Dict[str, List[str]],
    vector_lookup: Dict[str, np.ndarray],
    depth: int,
    cluster_id_prefix: str,
    counter: List[int],
    hierarchy: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if depth >= MAX_CLUSTER_SPLIT_DEPTH:
        return []
    split = _cluster_split_bimodal(members_ids, vector_lookup)
    if not split:
        return []
    sub_clusters = _build_graph_communities(
        members_ids,
        callgraph,
        vector_lookup,
        min_cluster_size=CHILD_CLUSTER_MIN_SIZE,
    )
    if len(sub_clusters) <= 1:
        sub_clusters = split

    results: List[Dict[str, Any]] = []
    for sub in sorted(sub_clusters, key=lambda group: (group[0], len(group), group)):
        if len(sub) < CHILD_CLUSTER_MIN_SIZE:
            continue
        counter[0] += 1
        child_id = f"{cluster_id_prefix}_{counter[0]:05d}"
        hierarchy.append(
            {
                "parent_cluster_id": cluster_id_prefix,
                "child_cluster_id": child_id,
                "reason": "bimodal_refine",
                "depth": depth + 1,
            }
        )
        sub = sorted(sub)
        results.append({"cluster_id": child_id, "member_funcs": sub})
        results.extend(
            _refine_clusters_recursive(
                sub,
                callgraph,
                vector_lookup,
                depth + 1,
                child_id,
                counter,
                hierarchy,
            )
        )
    return results


def _build_intent_clusters(
    function_intents: List[Dict[str, Any]],
    callgraph: Dict[str, List[str]],
    vector_lookup: Dict[str, np.ndarray],
    summary_cache: Optional[_FeatureSummaryCache] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    all_func_ids = sorted({
        rec["func_id"] for rec in function_intents if not rec.get("is_test")
    })
    communities = _build_graph_communities(all_func_ids, callgraph, vector_lookup)
    cluster_records: List[Dict[str, Any]] = []
    hierarchy_records: List[Dict[str, Any]] = []
    cluster_counter = [0]

    intent_lookup = {rec["func_id"]: rec for rec in function_intents}
    for members_ids in communities:
        members_ids = sorted(members_ids)
        members = [intent_lookup[fid] for fid in members_ids if fid in intent_lookup]
        if not members:
            continue
        cluster_counter[0] += 1
        parent_id = f"cluster_{cluster_counter[0]:05d}"
        child_defs = _refine_clusters_recursive(
            members_ids,
            callgraph,
            vector_lookup,
            0,
            parent_id,
            cluster_counter,
            hierarchy_records,
        )
        representatives = _pick_cluster_representatives(
            members_ids, callgraph, vector_lookup, REPRESENTATIVE_TOP_K
        )
        summary_ids = _select_cluster_summary_ids(
            members_ids, representatives, intent_lookup
        )
        representative_set = set(representatives)
        summary_members = [
            {
                **intent_lookup[fid],
                "_cluster_representative": fid in representative_set,
                "_cluster_sample_role": (
                    "structural_representative"
                    if fid in representative_set
                    else "coverage_sample"
                ),
            }
            for fid in summary_ids
        ]
        summary_role = (
            "coarse_navigation"
            if child_defs or len(members_ids) > len(summary_ids)
            else "functional_cluster"
        )
        summary_context = {
            "cluster_kind": "graph_community",
            "summary_role": summary_role,
            "total_member_count": len(members_ids),
            "sampled_member_count": len(summary_ids),
            "has_refined_children": bool(child_defs),
        }
        summary = (
            summary_cache.summarize_cluster(summary_members, summary_context)
            if summary_cache is not None
            else summarize_cluster(summary_members, summary_context)
        )
        cluster_records.append(
            {
                "cluster_id": parent_id,
                "cluster_name": summary["feature_name"],
                "cluster_description": summary["feature_description"],
                "tags": summary.get("tags", []),
                "member_funcs": members_ids,
                "representative_funcs": representatives,
                "cluster_type": "hybrid_leiden",
                "summary_role": summary_role,
                "summary_sample_count": len(summary_ids),
                "summary_generation": summary.get("_intent_generation", "unknown"),
            }
        )

        for child in child_defs:
            child["member_funcs"] = sorted(child["member_funcs"])
            child_members = [
                intent_lookup[fid]
                for fid in child["member_funcs"]
                if fid in intent_lookup
            ]
            if not child_members:
                continue
            child_reps = _pick_cluster_representatives(
                child["member_funcs"], callgraph, vector_lookup, REPRESENTATIVE_TOP_K
            )
            child_summary_ids = _select_cluster_summary_ids(
                child["member_funcs"], child_reps, intent_lookup
            )
            child_representative_set = set(child_reps)
            child_summary_members = [
                {
                    **intent_lookup[fid],
                    "_cluster_representative": fid in child_representative_set,
                    "_cluster_sample_role": (
                        "structural_representative"
                        if fid in child_representative_set
                        else "coverage_sample"
                    ),
                }
                for fid in child_summary_ids
            ]
            child_summary_context = {
                "cluster_kind": "graph_community_refined",
                "summary_role": "functional_cluster",
                "total_member_count": len(child["member_funcs"]),
                "sampled_member_count": len(child_summary_ids),
                "has_refined_children": any(
                    relation["parent_cluster_id"] == child["cluster_id"]
                    for relation in hierarchy_records
                ),
            }
            child_summary = (
                summary_cache.summarize_cluster(
                    child_summary_members, child_summary_context
                )
                if summary_cache is not None
                else summarize_cluster(
                    child_summary_members, child_summary_context
                )
            )
            cluster_records.append(
                {
                    "cluster_id": child["cluster_id"],
                    "cluster_name": child_summary["feature_name"],
                    "cluster_description": child_summary["feature_description"],
                    "tags": child_summary.get("tags", []),
                    "member_funcs": child["member_funcs"],
                    "representative_funcs": child_reps,
                    "cluster_type": "hybrid_leiden_refined",
                    "summary_role": "functional_cluster",
                    "summary_sample_count": len(child_summary_ids),
                    "summary_generation": child_summary.get(
                        "_intent_generation", "unknown"
                    ),
                }
            )
    return cluster_records, hierarchy_records


def _build_class_feature_payloads(
    function_intents: List[Dict[str, Any]],
    summary_cache: Optional[_FeatureSummaryCache] = None,
) -> List[Tuple[Dict[str, Any], Dict[str, float]]]:
    classes: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for intent in function_intents:
        if intent.get("is_test"):
            continue
        class_name = intent.get("class_name")
        if not class_name:
            continue
        classes[class_name].append(intent)

    payloads: List[Tuple[Dict[str, Any], Dict[str, float]]] = []
    for class_name in sorted(classes):
        members = sorted(classes[class_name], key=lambda member: member["func_id"])
        if len(members) < CLASS_FEATURE_MIN_MEMBERS:
            continue
        member_ids = [member["func_id"] for member in members]
        intent_lookup = {member["func_id"]: member for member in members}
        representatives = _select_cluster_summary_ids(
            member_ids, [], intent_lookup, REPRESENTATIVE_TOP_K
        )
        summary_ids = _select_cluster_summary_ids(
            member_ids, representatives, intent_lookup
        )
        representative_set = set(representatives)
        summary_members = [
            {
                **intent_lookup[fid],
                "_cluster_representative": fid in representative_set,
                "_cluster_sample_role": (
                    "structural_representative"
                    if fid in representative_set
                    else "coverage_sample"
                ),
            }
            for fid in summary_ids
        ]
        summary_context = {
            "cluster_kind": "class_component",
            "summary_role": "functional_cluster",
            "total_member_count": len(member_ids),
            "sampled_member_count": len(summary_ids),
            "source_class": class_name,
        }
        summary = (
            summary_cache.summarize_cluster(summary_members, summary_context)
            if summary_cache is not None
            else summarize_cluster(summary_members, summary_context)
        )
        feature = {
            "feature_name": summary["feature_name"],
            "feature_description": summary["feature_description"],
            "tags": summary.get("tags", []),
            "member_funcs": [m["func_id"] for m in members],
            "representative_funcs": representatives,
            "is_noise": False,
            "feature_type": "class",
            "source_class": class_name,
            "summary_role": "functional_cluster",
            "summary_sample_count": len(summary_ids),
            "summary_generation": summary.get("_intent_generation", "unknown"),
        }
        memberships = {m["func_id"]: 1.0 for m in members}
        payloads.append((feature, memberships))
    return payloads

def _sha256_file(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "size": path.stat().st_size}


def _stable_function_intents_digest(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: str(item.get("func_id") or "")):
        stable = {
            key: value
            for key, value in record.items()
            if key not in {"features", "_feature_build_id"}
        }
        digest.update(
            json.dumps(
                stable,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return {"sha256": digest.hexdigest(), "count": len(records)}


def _feature_parameters() -> Dict[str, Any]:
    return {
        "min_cluster_size": MIN_CLUSTER_SIZE,
        "representative_top_k": REPRESENTATIVE_TOP_K,
        "test_coverage_max_hops": TEST_COVERAGE_MAX_HOPS,
        "test_coverage_max_funcs": TEST_COVERAGE_MAX_FUNCS,
        "class_feature_min_members": CLASS_FEATURE_MIN_MEMBERS,
        "test_group_max_size": TEST_GROUP_MAX_SIZE,
        "test_group_similarity_threshold": TEST_GROUP_SIMILARITY_THRESHOLD,
        "cluster_alpha": CLUSTER_ALPHA,
        "semantic_top_k": SEMANTIC_TOP_K,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "leiden_resolution": LEIDEN_RESOLUTION,
        "leiden_iterations": LEIDEN_ITERATIONS,
        "child_cluster_min_size": CHILD_CLUSTER_MIN_SIZE,
        "max_cluster_split_depth": MAX_CLUSTER_SPLIT_DEPTH,
        "bimodal_min_ratio": BIMODAL_MIN_RATIO,
        "bimodal_separation": BIMODAL_SEPARATION,
        "representative_degree_weight": REPRESENTATIVE_DEGREE_WEIGHT,
        "summary_prompt_contract": feature_summary_prompt_contract(),
        "summary_validation_contract": FEATURE_SUMMARY_VALIDATION_CONTRACT,
    }


def _feature_input_signatures(
    base: Path,
    function_intents: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    intents = (
        function_intents
        if function_intents is not None
        else _load_function_intents(base / "function_intents.jsonl")
    )
    return {
        "function_intents": _stable_function_intents_digest(intents),
        "func_meta": _sha256_file(base / "func_meta.jsonl"),
        "func_index": _sha256_file(base / "func_index.faiss"),
        "callgraph": _sha256_file(base / "callgraph.jsonl"),
        "test_intents": _sha256_file(base / "test_intents.jsonl"),
        "parameters": _feature_parameters(),
    }


def intent_feature_artifacts_ready(
    index_dir: str | Path,
    *,
    require_current_inputs: bool = True,
    require_runtime_embedding: bool = True,
) -> bool:
    base = Path(index_dir)
    paths = {
        "features": base / "features.jsonl",
        "meta": base / "feature_meta.jsonl",
        "index": base / "feature_index.faiss",
        "clusters": base / "cluster_intents.jsonl",
        "hierarchy": base / "cluster_hierarchy.jsonl",
        "function_intents": base / "function_intents.jsonl",
    }
    if not all(path.exists() for path in paths.values()):
        return False
    entry = (
        load_embedding_manifest(base).get("indexes", {}).get("features") or {}
    )
    if entry.get("builder") != FEATURE_INTENT_BUILDER:
        return False
    if require_runtime_embedding and not (
        entry.get("model") == os.getenv("VOYAGE_EMBED_MODEL")
        and entry.get("dimension") == EMBED_DIM
    ):
        return False
    if require_current_inputs and (
        entry.get("input_signatures") != _feature_input_signatures(base)
    ):
        return False
    build_id = entry.get("build_id")
    if not isinstance(build_id, str) or not build_id:
        return False

    try:
        features = _load_jsonl(paths["features"])
        metadata = _load_jsonl(paths["meta"])
        clusters = _load_jsonl(paths["clusters"])
        hierarchy = _load_jsonl(paths["hierarchy"])
        function_intents = _load_jsonl(paths["function_intents"])
        index = faiss.read_index(str(paths["index"]))
        expected_feature_count = int(entry.get("feature_count"))
        expected_cluster_count = int(entry.get("cluster_count"))
        expected_hierarchy_count = int(entry.get("hierarchy_count"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError, RuntimeError):
        return False

    if not (
        len(features)
        == len(metadata)
        == index.ntotal
        == expected_feature_count
        and len(clusters) == expected_cluster_count
        and len(hierarchy) == expected_hierarchy_count
        and index.d == entry.get("dimension")
    ):
        return False
    feature_ids = [record.get("feature_id") for record in features]
    cluster_ids = [record.get("cluster_id") for record in clusters]
    function_ids = {record.get("func_id") for record in function_intents}
    if (
        None in feature_ids
        or len(feature_ids) != len(set(feature_ids))
        or None in cluster_ids
        or len(cluster_ids) != len(set(cluster_ids))
    ):
        return False
    feature_id_set = set(feature_ids)
    cluster_id_set = set(cluster_ids)
    if any(
        record.get("_build_id") != build_id
        for record in [*features, *metadata, *clusters, *hierarchy]
    ):
        return False
    if any(
        record.get("_feature_build_id") != build_id
        for record in function_intents
    ):
        return False
    if any(
        meta.get("index") != index_position
        or meta.get("feature_id") != feature_ids[index_position]
        for index_position, meta in enumerate(metadata)
    ):
        return False
    if any(
        not set(feature.get("member_funcs") or []).issubset(function_ids)
        or not set(feature.get("representative_funcs") or []).issubset(function_ids)
        for feature in features
    ):
        return False
    searchable_cluster_ids = {
        feature.get("source_cluster_id")
        for feature in features
        if feature.get("feature_type")
        in {"graph_community", "graph_community_refined"}
    }
    if searchable_cluster_ids != cluster_id_set:
        return False
    if any(
        relation.get("parent_cluster_id") not in cluster_id_set
        or relation.get("child_cluster_id") not in cluster_id_set
        for relation in hierarchy
    ):
        return False
    if any(
        feature_ref.get("feature_id") not in feature_id_set
        for record in function_intents
        for feature_ref in (record.get("features") or [])
    ):
        return False
    return entry.get("fallback_count") == 0


def build_feature_intents(
    index_dir: str | Path = "index_output",
    repo_id: Optional[str] = None,
) -> None:
    base = Path(index_dir)
    intents_path = base / "function_intents.jsonl"
    test_intents_path = base / "test_intents.jsonl"
    func_meta_path = base / "func_meta.jsonl"
    func_index_path = base / "func_index.faiss"
    callgraph_path = base / "callgraph.jsonl"
    features_path = base / "features.jsonl"
    feature_meta_path = base / "feature_meta.jsonl"
    feature_index_path = base / "feature_index.faiss"
    cluster_intents_path = base / "cluster_intents.jsonl"
    cluster_hierarchy_path = base / "cluster_hierarchy.jsonl"

    function_intents = _load_function_intents(intents_path)
    input_signatures = _feature_input_signatures(base, function_intents)
    func_lookup = {rec["func_id"]: rec for rec in function_intents}
    test_intents = _load_jsonl(test_intents_path)
    callgraph = _load_callgraph(callgraph_path)
    func_meta = _load_meta(func_meta_path, "func_id")
    build_id = uuid.uuid4().hex
    summary_cache = _FeatureSummaryCache(repo_id)

    if any(meta.get("index") != idx for idx, meta in enumerate(func_meta)):
        raise ValueError("func_meta.jsonl indices must be contiguous and ordered")
    embeddings = _load_embeddings(func_index_path, len(func_meta))
    vectors = _normalize_rows(embeddings)
    vector_lookup = {
        meta["func_id"]: vectors[idx] for idx, meta in enumerate(func_meta)
    }
    for meta in func_meta:
        if meta["func_id"] not in func_lookup:
            raise KeyError(f"function_intents.jsonl missing {meta['func_id']}")

    feature_records: List[Dict[str, Any]] = []
    feature_meta_records: List[Dict[str, Any]] = []
    function_feature_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def register_feature(
        feature_payload: Dict[str, Any], memberships: Dict[str, float]
    ) -> str:
        feature = dict(feature_payload)
        feature_id = f"feat_{len(feature_records):05d}"
        feature["feature_id"] = feature_id
        feature["_build_id"] = build_id
        feature["member_funcs"] = sorted(set(feature.get("member_funcs") or []))
        feature["representative_funcs"] = list(
            dict.fromkeys(feature.get("representative_funcs") or [])
        )
        feature_records.append(feature)
        feature_meta_records.append(
            {
                "_build_id": build_id,
                "index": len(feature_meta_records),
                "feature_id": feature_id,
            }
        )
        for func_id, score in sorted(memberships.items()):
            if func_id not in func_lookup:
                continue
            function_feature_map[func_id].append(
                {"feature_id": feature_id, "score": float(score)}
            )
        return feature_id

    test_payloads, _covered_funcs = _build_test_feature_payloads(
        func_lookup, test_intents, callgraph, summary_cache
    )
    for feature_payload, memberships in test_payloads:
        register_feature(feature_payload, memberships)

    for feature_payload, memberships in _build_class_feature_payloads(
        function_intents, summary_cache
    ):
        register_feature(feature_payload, memberships)

    cluster_records, cluster_hierarchy = _build_intent_clusters(
        function_intents, callgraph, vector_lookup, summary_cache
    )
    for cluster in cluster_records:
        cluster["_build_id"] = build_id
        feature_type = (
            "graph_community_refined"
            if cluster.get("cluster_type") == "hybrid_leiden_refined"
            else "graph_community"
        )
        register_feature(
            {
                "feature_name": cluster.get("cluster_name") or "Unnamed Feature",
                "feature_description": cluster.get("cluster_description") or "",
                "tags": cluster.get("tags") or [],
                "member_funcs": cluster.get("member_funcs") or [],
                "representative_funcs": cluster.get("representative_funcs") or [],
                "is_noise": False,
                "feature_type": feature_type,
                "source_cluster_id": cluster["cluster_id"],
                "summary_role": cluster.get("summary_role", "functional_cluster"),
                "summary_sample_count": cluster.get("summary_sample_count", 0),
                "summary_generation": cluster.get(
                    "summary_generation", "unknown"
                ),
            },
            {func_id: 1.0 for func_id in cluster.get("member_funcs") or []},
        )
    for relation in cluster_hierarchy:
        relation["_build_id"] = build_id
    summary_cache.close()

    updated_intents: List[Dict[str, Any]] = []
    for record in function_intents:
        updated = dict(record)
        updated["features"] = sorted(
            function_feature_map.get(record["func_id"], []),
            key=lambda item: item["feature_id"],
        )
        updated["_feature_build_id"] = build_id
        updated_intents.append(updated)

    embedding_texts = [
        build_feature_embedding_text(feature) for feature in feature_records
    ]
    feature_embeddings: List[Optional[np.ndarray]] = [
        None for _ in embedding_texts
    ]
    embedding_cache = GlobalEmbeddingCache(
        "feature",
        repo_id=repo_id,
        model=os.getenv("VOYAGE_EMBED_MODEL") or "",
        dimension=EMBED_DIM,
    )
    embedding_cache_hits = 0
    missing_positions: Dict[str, List[int]] = defaultdict(list)
    for index, embedding_text in enumerate(embedding_texts):
        reusable = embedding_cache.get(embedding_text)
        if reusable is not None and reusable.shape == (EMBED_DIM,):
            feature_embeddings[index] = reusable
            embedding_cache_hits += 1
        else:
            missing_positions[embedding_text].append(index)

    missing_texts = list(missing_positions)
    embedding_batches = 0
    for start in range(0, len(missing_texts), FEATURE_EMBED_BATCH_SIZE):
        batch = missing_texts[start : start + FEATURE_EMBED_BATCH_SIZE]
        batch_vectors = embed_texts(batch)
        if batch_vectors.shape != (len(batch), EMBED_DIM):
            raise ValueError(
                "feature embedding shape mismatch: "
                f"{batch_vectors.shape} != ({len(batch)}, {EMBED_DIM})"
            )
        normalized_vectors = _normalize_rows(batch_vectors)
        cache_records: List[Tuple[str, np.ndarray]] = []
        for embedding_text, vector in zip(batch, normalized_vectors):
            for index in missing_positions[embedding_text]:
                feature_embeddings[index] = vector
            cache_records.append((embedding_text, vector))
        if embedding_cache_writes_allowed():
            embedding_cache.update(cache_records)
        embedding_batches += 1
    embedding_cache.close()

    if any(vector is None for vector in feature_embeddings):
        raise RuntimeError(
            "feature embedding generation returned fewer vectors than requested"
        )

    feature_matrix = (
        np.vstack([
            vector for vector in feature_embeddings if vector is not None
        ]).astype("float32")
        if feature_embeddings
        else np.zeros((0, EMBED_DIM), dtype="float32")
    )
    feature_index = faiss.IndexFlatIP(EMBED_DIM)
    if feature_matrix.size:
        feature_index.add(feature_matrix)

    if _feature_input_signatures(base, function_intents) != input_signatures:
        raise RuntimeError("feature intent inputs changed during construction")
    expected_generation = "llm" if os.getenv("LLM_MODEL") else "heuristic"
    fallback_count = sum(
        feature.get("summary_generation") != expected_generation
        for feature in feature_records
    )
    if os.getenv("INTENT_ABORT_ON_FALLBACK") == "1" and fallback_count:
        raise RuntimeError(
            f"feature intent construction produced {fallback_count} fallback summaries"
        )

    _write_jsonl_atomic(cluster_records, cluster_intents_path)
    _write_jsonl_atomic(cluster_hierarchy, cluster_hierarchy_path)
    _write_jsonl_atomic(feature_records, features_path)
    _write_jsonl_atomic(feature_meta_records, feature_meta_path)
    _write_jsonl_atomic(updated_intents, intents_path)
    tmp_index_path = feature_index_path.with_suffix(feature_index_path.suffix + ".tmp")
    faiss.write_index(feature_index, str(tmp_index_path))
    tmp_index_path.replace(feature_index_path)
    write_embedding_manifest_entry(
        base,
        "features",
        model=os.getenv("VOYAGE_EMBED_MODEL"),
        dimension=EMBED_DIM,
        metadata={
            "builder": FEATURE_INTENT_BUILDER,
            "intent_model": os.getenv("LLM_MODEL") or "heuristic-fallback",
            "build_id": build_id,
            "feature_count": len(feature_records),
            "cluster_count": len(cluster_records),
            "hierarchy_count": len(cluster_hierarchy),
            "summary_count": len(feature_records),
            "summary_cache_hits": summary_cache.hits,
            "summary_requests": summary_cache.requests,
            "fallback_count": fallback_count,
            "embedding_cache_hits": embedding_cache_hits,
            "embedding_requests": len(missing_texts),
            "embedding_duplicate_reuse": sum(
                len(indices) - 1 for indices in missing_positions.values()
            ),
            "embedding_batches": embedding_batches,
            "input_signatures": input_signatures,
        },
    )
    print(
        "Feature intents: "
        f"summary cache reuse {summary_cache.hits}, summaries generated "
        f"{summary_cache.requests}, embedded {len(missing_texts)}, "
        f"embedding duplicate reuse "
        f"{sum(len(indices) - 1 for indices in missing_positions.values())}, "
        f"embedding cache reuse {embedding_cache_hits}."
    )
