"""Bootstrap or live-seed routing clusters so adaptive routing can use direct mode."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from app.services.adaptive_routing.classifier import compute_query_embedding, update_cluster_centroid
from app.services.adaptive_routing.config import SEED_CLUSTERS
from app.services.adaptive_routing import config as routing_config
from app.services.adaptive_routing.storage import append_routing_log, load_routing_table, save_routing_table, update_cluster

SEED_QUERIES_PATH = Path(__file__).resolve().parents[3] / "data" / "routing_seed_queries.json"


def load_seed_queries() -> dict[str, list[str]]:
    if not SEED_QUERIES_PATH.exists():
        raise FileNotFoundError(f"Seed queries file not found: {SEED_QUERIES_PATH}")
    data = json.loads(SEED_QUERIES_PATH.read_text())
    return {k: list(v) for k, v in data.items()}


def bootstrap_cluster(cluster_id: str, queries: list[str]) -> dict:
    """Populate a cluster with embeddings and sample counts without calling models."""
    if cluster_id not in SEED_CLUSTERS:
        raise ValueError(f"Unknown seed cluster: {cluster_id}")

    default_model = SEED_CLUSTERS[cluster_id]["default_model"]
    cluster = update_cluster(cluster_id, {})

    for query in queries:
        embedding = compute_query_embedding(query)
        count = int(cluster.get("count", 0)) + 1
        exploration_count = int(cluster.get("exploration_count", 0)) + 1
        centroid = update_cluster_centroid(cluster.get("centroid"), embedding, count)

        cluster = update_cluster(
            cluster_id,
            {
                "count": count,
                "exploration_count": exploration_count,
                "centroid": centroid,
                "best_model": default_model,
                "avg_accuracy": 0.85,
                "avg_semantic_accuracy": 0.85,
                "avg_confidence": 0.8,
                "avg_latency": 150.0,
                "queries_since_exploration": 0,
                "force_explore": False,
                "recent_routed_metrics": [],
                "last_full_exploration_at": datetime.now(timezone.utc).isoformat(),
                "mode": "direct"
                if count >= routing_config.MIN_SAMPLES_FOR_DIRECT_ROUTING
                else "exploration",
            },
        )

        append_routing_log(
            {
                "event": "routing_seed_bootstrap",
                "cluster_id": cluster_id,
                "query": query,
                "count": count,
                "mode": cluster["mode"],
                "best_model": default_model,
            }
        )

    return cluster


def bootstrap_all_clusters(queries_by_cluster: dict[str, list[str]] | None = None) -> dict:
    """Bootstrap every seed cluster from curated queries."""
    queries_by_cluster = queries_by_cluster or load_seed_queries()
    results = {}
    for cluster_id, queries in queries_by_cluster.items():
        if cluster_id not in SEED_CLUSTERS:
            continue
        results[cluster_id] = bootstrap_cluster(cluster_id, queries)
    return results


def live_seed_cluster(cluster_id: str, queries: list[str]) -> list[dict]:
    """Run full exploration for each query to build real eval history (slow)."""
    from app.services.orchestrator import run_query_pipeline

    outcomes = []
    for query in queries:
        result = run_query_pipeline(query, enable_routing=False)
        outcomes.append(
            {
                "query": query,
                "cluster_id": result.get("routing", {}).get("cluster_id"),
                "best_model": result.get("best_model"),
                "mode": result.get("routing", {}).get("mode"),
            }
        )
    return outcomes


def live_seed_all(queries_by_cluster: dict[str, list[str]] | None = None, max_per_cluster: int | None = None) -> dict:
    queries_by_cluster = queries_by_cluster or load_seed_queries()
    all_outcomes: dict[str, list] = {}
    for cluster_id, queries in queries_by_cluster.items():
        if cluster_id not in SEED_CLUSTERS:
            continue
        batch = queries[:max_per_cluster] if max_per_cluster else queries
        all_outcomes[cluster_id] = live_seed_cluster(cluster_id, batch)
    return all_outcomes


def get_seed_status() -> dict:
    table = load_routing_table()
    min_samples = routing_config.MIN_SAMPLES_FOR_DIRECT_ROUTING
    clusters = {}
    for cid, row in table.get("clusters", {}).items():
        count = int(row.get("count", 0))
        clusters[cid] = {
            "count": count,
            "mode": row.get("mode"),
            "best_model": row.get("best_model"),
            "ready_for_direct": count >= min_samples and row.get("mode") == "direct",
            "samples_needed": max(0, min_samples - count),
            "has_centroid": row.get("centroid") is not None,
        }
    return {
        "min_samples_for_direct": min_samples,
        "clusters": clusters,
        "all_ready": all(clusters.get(cid, {}).get("ready_for_direct") for cid in SEED_CLUSTERS),
    }


def verify_direct_routing(test_queries: dict[str, str] | None = None) -> dict:
    """Run one query per cluster with routing enabled; expect direct mode when seeded."""
    from app.services.adaptive_routing.router import plan_query_execution
    from app.services.query_analyzer import analyze_query

    test_queries = test_queries or {
        "python": "Explain dict comprehension in Python.",
        "chemistry": "What is the boiling point of benzene?",
        "gita": "Explain dharma according to the Gita.",
        "general": "What is the capital of France?",
    }

    results = {}
    for cluster_id, query in test_queries.items():
        analysis = analyze_query(query)
        plan = plan_query_execution(query, analysis, enable_routing=True)
        routing = plan["routing_decision"]
        results[cluster_id] = {
            "query": query,
            "mode": routing.get("mode"),
            "models_called": routing.get("models"),
            "model_count": len(routing.get("models") or []),
            "cluster_id": routing.get("cluster_id"),
            "reason": routing.get("reason"),
            "direct_routing_worked": routing.get("mode") == "direct" and len(routing.get("models") or []) == 1,
        }
    return results
