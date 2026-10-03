"""Configurable thresholds for the adaptive query routing layer."""

from __future__ import annotations

import os

# ── Embedding classification ──────────────────────────────────────────────────
EMBEDDING_SIMILARITY_THRESHOLD = float(
    os.getenv("ROUTING_EMBEDDING_SIMILARITY_THRESHOLD", "0.55")
)
# Minimum cosine similarity to assign a query to the nearest cluster via embeddings.

EMBEDDING_AMBIGUITY_MARGIN = float(os.getenv("ROUTING_EMBEDDING_AMBIGUITY_MARGIN", "0.05"))
# If top-1 and top-2 cluster similarities differ by less than this, treat as ambiguous.

EMBEDDING_MODEL = os.getenv("EVAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ── Sub-cluster splitting (hybrid seed + dynamic sub-clusters) ────────────────
SUBCLUSTER_MIN_SAMPLES_FOR_SPLIT = int(
    os.getenv("ROUTING_SUBCLUSTER_MIN_SAMPLES_FOR_SPLIT", "15")
)
SUBCLUSTER_CENTROID_LOW_THRESHOLD = float(
    os.getenv("ROUTING_SUBCLUSTER_CENTROID_LOW_THRESHOLD", "0.45")
)
# When a query's similarity to its cluster centroid is below this after enough
# samples, a new sub-cluster is created (e.g. python/sub_1).

# ── Direct routing ────────────────────────────────────────────────────────────
MIN_SAMPLES_FOR_DIRECT_ROUTING = int(os.getenv("ROUTING_MIN_SAMPLES_FOR_DIRECT_ROUTING", "20"))
# Cluster must have at least this many fully-explored samples before direct routing.

# ── Re-exploration triggers (any metric breach triggers full exploration) ─────
CONFIDENCE_DROP_THRESHOLD = float(os.getenv("ROUTING_CONFIDENCE_DROP_THRESHOLD", "0.15"))
ACCURACY_DROP_THRESHOLD = float(os.getenv("ROUTING_ACCURACY_DROP_THRESHOLD", "0.15"))
LATENCY_INCREASE_THRESHOLD_MS = float(os.getenv("ROUTING_LATENCY_INCREASE_THRESHOLD_MS", "500"))
ROLLING_WINDOW_SIZE = int(os.getenv("ROUTING_ROLLING_WINDOW_SIZE", "5"))
TREND_DEGRADATION_MIN_DROP = float(os.getenv("ROUTING_TREND_DEGRADATION_MIN_DROP", "0.10"))
PERIODIC_REEXPLORATION_INTERVAL = int(os.getenv("ROUTING_PERIODIC_REEXPLORATION_INTERVAL", "50"))

# ── Seed clusters (hybrid model) ──────────────────────────────────────────────
SEED_CLUSTERS = {
    "chemistry": {"seed_domain": "chemistry", "default_model": "organic_model"},
    "python": {"seed_domain": "python", "default_model": "python_model"},
    "gita": {"seed_domain": "gita", "default_model": "gita_model"},
    "general": {"seed_domain": "general", "default_model": "groq_model"},
}

ALL_MODELS = ["organic_model", "python_model", "gita_model", "groq_model"]

# ── LLM fallback classifier ───────────────────────────────────────────────────
LLM_CLASSIFIER_ENABLED = os.getenv("ROUTING_LLM_CLASSIFIER_ENABLED", "true").lower() in {
    "1",
    "true",
    "yes",
}
LLM_CLASSIFIER_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
