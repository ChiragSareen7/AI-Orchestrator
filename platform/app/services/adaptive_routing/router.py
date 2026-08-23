"""Adaptive routing decision engine."""

from __future__ import annotations

from datetime import datetime, timezone

from app.services.adaptive_routing.classifier import (
    ClassificationResult,
    classify_query,
    compute_query_embedding,
    update_cluster_centroid,
)
from app.services.adaptive_routing.config import (
    ALL_MODELS,
    MIN_SAMPLES_FOR_DIRECT_ROUTING,
    SUBCLUSTER_CENTROID_LOW_THRESHOLD,
    SUBCLUSTER_MIN_SAMPLES_FOR_SPLIT,
)
from app.services.adaptive_routing.metrics import extract_metric_bundle
from app.services.adaptive_routing.reexploration import (
    clear_force_explore,
    record_routed_query_metrics,
    should_full_explore,
)
from app.services.adaptive_routing.storage import (
    allocate_subcluster_id,
    append_routing_log,
    get_cluster,
    load_routing_table,
    read_routing_logs,
    save_routing_table,
    update_cluster,
)
from app.evaluation.embeddings import cosine_similarity_vec
import numpy as np


def _maybe_split_subcluster(cluster_id: str, query: str, query_embedding: list[float]) -> str:
    table = load_routing_table()
    cluster = table["clusters"][cluster_id]
    count = int(cluster.get("count", 0))
    if count < SUBCLUSTER_MIN_SAMPLES_FOR_SPLIT:
        return cluster_id

    centroid = cluster.get("centroid")
    if not centroid:
        return cluster_id

    sim = cosine_similarity_vec(
        np.asarray(query_embedding, dtype=np.float32),
        np.asarray(centroid, dtype=np.float32),
    )
    if sim >= SUBCLUSTER_CENTROID_LOW_THRESHOLD:
        return cluster_id

    parent_id = cluster_id.split("/")[0]
    new_id = allocate_subcluster_id(parent_id)
    new_cluster = {
        "id": new_id,
        "parent_id": parent_id,
        "seed_domain": cluster.get("seed_domain", parent_id),
        "best_model": cluster.get("best_model"),
        "avg_accuracy": 0.0,
        "avg_semantic_accuracy": 0.0,
        "avg_confidence": 0.0,
        "avg_latency": 0.0,
        "count": 0,
        "exploration_count": 0,
        "queries_since_exploration": 0,
        "last_full_exploration_at": None,
        "mode": "exploration",
        "force_explore": False,
        "centroid": query_embedding,
        "recent_routed_metrics": [],
    }
    table["clusters"][new_id] = new_cluster
    save_routing_table(table)
    return new_id


def _cluster_matches_force_target(cluster_id: str, cluster: dict, target: str) -> bool:
    if target == cluster_id:
        return True
    if target == cluster.get("seed_domain"):
        return True
    if cluster_id.startswith(f"{target}/"):
        return True
    return cluster.get("parent_id") == target


def _plan_routing_disabled(query: str, analysis: dict) -> dict:
    """Bypass adaptive routing and always call every model."""
    classification = classify_query(query, analysis)
    cluster_id = classification.cluster_id
    query_embedding = compute_query_embedding(query)
    cluster = get_cluster(cluster_id) or update_cluster(cluster_id, {})

    routing_decision = {
        "cluster_id": cluster_id,
        "classification_method": classification.method,
        "classification_confidence": round(classification.confidence, 4),
        "classification_similarities": {
            k: round(v, 4) for k, v in classification.similarities.items()
        },
        "mode": "full_exploration",
        "models": list(ALL_MODELS),
        "reason": "routing_disabled_by_user",
        "reexploration_reasons": [],
        "ready_for_direct": int(cluster.get("count", 0)) >= MIN_SAMPLES_FOR_DIRECT_ROUTING,
        "cluster_count": int(cluster.get("count", 0)),
        "cluster_best_model": cluster.get("best_model"),
        "routing_enabled": False,
    }

    append_routing_log(
        {
            "event": "routing_decision",
            "query": query,
            "analysis_domain": analysis.get("domain"),
            **routing_decision,
        }
    )

    return {
        "classification": classification,
        "cluster_id": cluster_id,
        "query_embedding": query_embedding,
        "models": list(ALL_MODELS),
        "mode": "full_exploration",
        "routing_decision": routing_decision,
    }


def get_routing_status() -> dict:
    """Return routing table summary for the dashboard."""
    from app.services.adaptive_routing.config import (
        MIN_SAMPLES_FOR_DIRECT_ROUTING,
        PERIODIC_REEXPLORATION_INTERVAL,
    )

    table = load_routing_table()
    clusters: dict = {}
    for cluster_id, cluster in table.get("clusters", {}).items():
        clusters[cluster_id] = {
            k: v for k, v in cluster.items() if k != "centroid"
        }

    logs = read_routing_logs()[-30:]
    recent_decisions = [entry for entry in logs if entry.get("event") == "routing_decision"]

    return {
        "clusters": clusters,
        "min_samples_for_direct": MIN_SAMPLES_FOR_DIRECT_ROUTING,
        "periodic_reexploration_interval": PERIODIC_REEXPLORATION_INTERVAL,
        "recent_decisions": recent_decisions[-10:],
    }


def get_ranked_models_for_cluster(cluster_id: str, exclude: list[str] | None = None) -> list[str]:
    """
    Return models ordered by cluster preference (for self-correction escalation).

    Minimal routing lookup — does not change routing decisions elsewhere.
    """
    from app.services.adaptive_routing.config import ALL_MODELS, SEED_CLUSTERS

    cluster = get_cluster(cluster_id) or {}
    seed = cluster_id.split("/")[0]
    candidates: list[str] = []

    for model in [cluster.get("best_model"), SEED_CLUSTERS.get(seed, {}).get("default_model")]:
        if model and model in ALL_MODELS and model not in candidates:
            candidates.append(model)

    for model in ALL_MODELS:
        if model not in candidates:
            candidates.append(model)

    if exclude:
        candidates = [m for m in candidates if m not in exclude]
    return candidates


def plan_query_execution(
    query: str,
    analysis: dict,
    force_explore_cluster: str | None = None,
    enable_routing: bool = True,
) -> dict:
    """
    Decide whether to run full exploration or direct routing for a query.

    Returns an execution plan consumed by the orchestrator.
    """
    if not enable_routing:
        return _plan_routing_disabled(query, analysis)

    classification = classify_query(query, analysis)
    cluster_id = classification.cluster_id
    query_embedding = compute_query_embedding(query)

    if cluster_id in load_routing_table().get("clusters", {}):
        cluster_id = _maybe_split_subcluster(cluster_id, query, query_embedding)

    cluster = get_cluster(cluster_id) or update_cluster(cluster_id, {})
    reexplore_reasons: list[str] = []

    if force_explore_cluster and _cluster_matches_force_target(cluster_id, cluster, force_explore_cluster):
        reexplore_reasons.append(f"request_force_explore_cluster:{force_explore_cluster}")

    explore_forced, trigger_reasons = should_full_explore(cluster)
    reexplore_reasons.extend(trigger_reasons)

    ready_for_direct = int(cluster.get("count", 0)) >= MIN_SAMPLES_FOR_DIRECT_ROUTING
    use_direct = ready_for_direct and not reexplore_reasons and cluster.get("mode") == "direct"

    if use_direct:
        models = [cluster.get("best_model", "groq_model")]
        mode = "direct"
        reason = "confident_cluster_direct_route"
    else:
        models = list(ALL_MODELS)
        mode = "full_exploration"
        if reexplore_reasons:
            reason = "re_exploration_triggered"
        elif not ready_for_direct:
            reason = "insufficient_cluster_samples"
        else:
            reason = "cluster_not_in_direct_mode"

    routing_decision = {
        "cluster_id": cluster_id,
        "classification_method": classification.method,
        "classification_confidence": round(classification.confidence, 4),
        "classification_similarities": {
            k: round(v, 4) for k, v in classification.similarities.items()
        },
        "mode": mode,
        "models": models,
        "reason": reason,
        "reexploration_reasons": reexplore_reasons,
        "ready_for_direct": ready_for_direct,
        "cluster_count": int(cluster.get("count", 0)),
        "cluster_best_model": cluster.get("best_model"),
        "routing_enabled": True,
    }

    append_routing_log(
        {
            "event": "routing_decision",
            "query": query,
            "analysis_domain": analysis.get("domain"),
            **routing_decision,
        }
    )

    return {
        "classification": classification,
        "cluster_id": cluster_id,
        "query_embedding": query_embedding,
        "models": models,
        "mode": mode,
        "routing_decision": routing_decision,
    }


def record_query_outcome(
    plan: dict,
    query: str,
    responses: list[dict],
    best: dict,
    analysis: dict,
) -> None:
    """Update routing table after a query completes."""
    cluster_id = plan["cluster_id"]
    query_embedding = plan["query_embedding"]
    mode = plan["mode"]
    cluster = get_cluster(cluster_id) or update_cluster(cluster_id, {})

    count = int(cluster.get("count", 0))
    new_count = count + 1
    centroid = update_cluster_centroid(cluster.get("centroid"), query_embedding, new_count)

    bundle = extract_metric_bundle(best["metrics"])
    updates: dict = {
        "centroid": centroid,
        "count": new_count,
    }

    if mode == "full_exploration":
        exploration_count = int(cluster.get("exploration_count", 0)) + 1
        acc_val = bundle["semantic_accuracy"] if bundle.get("semantic_accuracy") is not None else bundle["accuracy"]

        def _rolling_avg(prev: float, new_val: float, n: int) -> float:
            if n <= 1:
                return new_val
            return ((prev * (n - 1)) + new_val) / n

        sem_acc = bundle.get("semantic_accuracy")
        sem_acc_val = float(sem_acc) if sem_acc is not None else acc_val

        updates.update(
            {
                "best_model": best["model"],
                "avg_accuracy": round(_rolling_avg(cluster.get("avg_accuracy", 0.0), acc_val, exploration_count), 4),
                "avg_semantic_accuracy": round(
                    _rolling_avg(cluster.get("avg_semantic_accuracy", 0.0), sem_acc_val, exploration_count), 4
                ),
                "avg_confidence": round(
                    _rolling_avg(cluster.get("avg_confidence", 0.0), bundle["confidence"], exploration_count), 4
                ),
                "avg_latency": round(
                    _rolling_avg(cluster.get("avg_latency", 0.0), bundle["latency"], exploration_count), 2
                ),
                "exploration_count": exploration_count,
                "queries_since_exploration": 0,
                "last_full_exploration_at": datetime.now(timezone.utc).isoformat(),
                "mode": "direct" if new_count >= MIN_SAMPLES_FOR_DIRECT_ROUTING else "exploration",
                "force_explore": False,
                "recent_routed_metrics": [],
            }
        )
        clear_force_explore(cluster_id)
    else:
        cluster = record_routed_query_metrics(cluster_id, best["metrics"])
        explore_forced, reasons = should_full_explore(cluster, best["metrics"])
        updates["queries_since_exploration"] = cluster.get("queries_since_exploration", 0)
        if explore_forced:
            updates["mode"] = "exploration"
            updates["force_explore"] = True
        append_routing_log(
            {
                "event": "routed_query_outcome",
                "query": query,
                "cluster_id": cluster_id,
                "best_model": best["model"],
                "metrics": bundle,
                "reexploration_triggered": explore_forced,
                "reexploration_reasons": reasons,
            }
        )

    update_cluster(cluster_id, updates)

    append_routing_log(
        {
            "event": "cluster_state_updated",
            "query": query,
            "cluster_id": cluster_id,
            "mode": mode,
            "best_model": best["model"],
            "analysis_domain": analysis.get("domain"),
            "metrics": bundle,
        }
    )
