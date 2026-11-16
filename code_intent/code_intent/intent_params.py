from __future__ import annotations

# Retrieval / traversal hyperparameters
HOP_BUDGET = 2  # h: hop budget
EXPANSION_WIDTH = 30  # K: expansion width (rerank candidates)
NEIGHBORHOOD_SIZE = 50  # Kctx: neighborhood size for context graphs

# Intent summarization hyperparameters
TOKEN_BUDGET = 1024  # B: max token budget for caller/callee/test context

# Clustering hyperparameters
CLUSTER_ALPHA = 0.3 # α: balance structure vs semantic
SIMILARITY_THRESHOLD = 0.75  # τ: cosine similarity threshold
SEMANTIC_TOP_K = 12  # semantic neighbors per node (cost control)
LEIDEN_RESOLUTION = 1.8  # >1.0 finer, <1.0 coarser
LEIDEN_ITERATIONS = 5
REPRESENTATIVE_DEGREE_WEIGHT = 0.6  # weight for local-degree vs centroid similarity

# Cluster refinement (hierarchy)
MAX_CLUSTER_SPLIT_DEPTH = 2  # maximum recursive split depth
CHILD_CLUSTER_MIN_SIZE = 3  # minimum size for child clusters
BIMODAL_SEPARATION = 0.08  # minimum center separation to split
BIMODAL_MIN_RATIO = 0.2  # smaller cluster must be >= this ratio