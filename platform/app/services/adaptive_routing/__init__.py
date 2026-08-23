"""Adaptive query routing layer."""

from app.services.adaptive_routing.classifier import (
    ClassificationResult,
    QueryClassifier,
    classify_query,
    classify_with_fine_tuned_router,
)
from app.services.adaptive_routing.config import (
    ACCURACY_DROP_THRESHOLD,
    CONFIDENCE_DROP_THRESHOLD,
    EMBEDDING_AMBIGUITY_MARGIN,
    EMBEDDING_SIMILARITY_THRESHOLD,
    LATENCY_INCREASE_THRESHOLD_MS,
    MIN_SAMPLES_FOR_DIRECT_ROUTING,
    PERIODIC_REEXPLORATION_INTERVAL,
    ROLLING_WINDOW_SIZE,
    SUBCLUSTER_CENTROID_LOW_THRESHOLD,
    SUBCLUSTER_MIN_SAMPLES_FOR_SPLIT,
    TREND_DEGRADATION_MIN_DROP,
)
from app.services.adaptive_routing.reexploration import force_reexploration
from app.services.adaptive_routing.router import plan_query_execution, record_query_outcome
from app.services.adaptive_routing.storage import ensure_routing_files, read_routing_logs

__all__ = [
    "ACCURACY_DROP_THRESHOLD",
    "CONFIDENCE_DROP_THRESHOLD",
    "ClassificationResult",
    "EMBEDDING_AMBIGUITY_MARGIN",
    "EMBEDDING_SIMILARITY_THRESHOLD",
    "LATENCY_INCREASE_THRESHOLD_MS",
    "MIN_SAMPLES_FOR_DIRECT_ROUTING",
    "PERIODIC_REEXPLORATION_INTERVAL",
    "QueryClassifier",
    "ROLLING_WINDOW_SIZE",
    "SUBCLUSTER_CENTROID_LOW_THRESHOLD",
    "SUBCLUSTER_MIN_SAMPLES_FOR_SPLIT",
    "TREND_DEGRADATION_MIN_DROP",
    "classify_query",
    "classify_with_fine_tuned_router",
    "ensure_routing_files",
    "force_reexploration",
    "plan_query_execution",
    "read_routing_logs",
    "record_query_outcome",
]
