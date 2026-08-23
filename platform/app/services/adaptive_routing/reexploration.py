"""Re-exploration trigger checks and manual override handling."""

from __future__ import annotations

from datetime import datetime, timezone

from app.services.adaptive_routing.config import (
    ACCURACY_DROP_THRESHOLD,
    CONFIDENCE_DROP_THRESHOLD,
    LATENCY_INCREASE_THRESHOLD_MS,
    MIN_SAMPLES_FOR_DIRECT_ROUTING,
    PERIODIC_REEXPLORATION_INTERVAL,
    ROLLING_WINDOW_SIZE,
    TREND_DEGRADATION_MIN_DROP,
)
from app.services.adaptive_routing.metrics import extract_metric_bundle, metric_values_for_triggers
from app.services.adaptive_routing.storage import load_routing_table, save_routing_table, update_cluster


def _rolling_trend_degrading(recent: list[dict], metric_key: str) -> bool:
    window = recent[-ROLLING_WINDOW_SIZE:]
    if len(window) < ROLLING_WINDOW_SIZE:
        return False
    values = [float(item.get(metric_key, 0.0)) for item in window if item.get(metric_key) is not None]
    if len(values) < ROLLING_WINDOW_SIZE:
        return False
    first_half = values[: len(values) // 2]
    second_half = values[len(values) // 2 :]
    if not first_half or not second_half:
        return False
    earlier_avg = sum(first_half) / len(first_half)
    later_avg = sum(second_half) / len(second_half)
    return (earlier_avg - later_avg) >= TREND_DEGRADATION_MIN_DROP


def check_reexploration_triggers(cluster: dict, latest_metrics: dict | None = None) -> list[str]:
    """Return reasons to force full exploration (any trigger is sufficient)."""
    reasons: list[str] = []

    if cluster.get("force_explore"):
        reasons.append("manual_force_explore_flag")

    if cluster.get("count", 0) < MIN_SAMPLES_FOR_DIRECT_ROUTING:
        return reasons

    if int(cluster.get("queries_since_exploration", 0)) >= PERIODIC_REEXPLORATION_INTERVAL:
        reasons.append(
            f"periodic_revalidation_after_{PERIODIC_REEXPLORATION_INTERVAL}_routed_queries"
        )

    if latest_metrics:
        bundle = extract_metric_bundle(latest_metrics)
        baseline_map = {
            "accuracy": float(cluster.get("avg_accuracy", 0.0)),
            "semantic_accuracy": float(cluster.get("avg_semantic_accuracy", cluster.get("avg_accuracy", 0.0))),
            "confidence": float(cluster.get("avg_confidence", cluster.get("avg_accuracy", 0.0))),
            "latency": float(cluster.get("avg_latency", 0.0)),
        }
        for name, value in metric_values_for_triggers(bundle):
            baseline = baseline_map[name]
            if name == "latency":
                if value > baseline + LATENCY_INCREASE_THRESHOLD_MS:
                    reasons.append(
                        f"latency_increase:{value:.2f}>{baseline + LATENCY_INCREASE_THRESHOLD_MS:.2f}"
                    )
            else:
                threshold = ACCURACY_DROP_THRESHOLD if "accuracy" in name else CONFIDENCE_DROP_THRESHOLD
                if value < baseline - threshold:
                    reasons.append(f"{name}_drop:{value:.4f}<{baseline - threshold:.4f}")

    recent = cluster.get("recent_routed_metrics", [])
    for metric_key in ("accuracy", "semantic_accuracy", "confidence"):
        if _rolling_trend_degrading(recent, metric_key):
            reasons.append(f"trend_degradation_{metric_key}_rolling_window_{ROLLING_WINDOW_SIZE}")

    return reasons


def should_full_explore(cluster: dict, latest_metrics: dict | None = None) -> tuple[bool, list[str]]:
    reasons = check_reexploration_triggers(cluster, latest_metrics)
    return bool(reasons), reasons


def force_reexploration(cluster_id: str | None = None, all_clusters: bool = False) -> dict:
    """Manually mark cluster(s) for forced re-exploration on the next matching query."""
    table = load_routing_table()
    updated: list[str] = []

    if all_clusters:
        for cid in table.get("clusters", {}):
            table["clusters"][cid]["force_explore"] = True
            table["clusters"][cid]["mode"] = "exploration"
            updated.append(cid)
    elif cluster_id:
        if cluster_id not in table.get("clusters", {}):
            raise ValueError(f"Unknown cluster: {cluster_id}")
        table["clusters"][cluster_id]["force_explore"] = True
        table["clusters"][cluster_id]["mode"] = "exploration"
        updated.append(cluster_id)
    else:
        raise ValueError("Provide cluster_id or set all_clusters=True")

    save_routing_table(table)
    return {"updated_clusters": updated, "forced_at": datetime.now(timezone.utc).isoformat()}


def clear_force_explore(cluster_id: str) -> None:
    table = load_routing_table()
    if cluster_id in table.get("clusters", {}):
        table["clusters"][cluster_id]["force_explore"] = False
        save_routing_table(table)


def record_routed_query_metrics(cluster_id: str, metrics: dict) -> dict:
    """Append routed-query metrics and return updated cluster."""
    cluster = load_routing_table()["clusters"][cluster_id]
    bundle = extract_metric_bundle(metrics)
    bundle["timestamp"] = datetime.now(timezone.utc).isoformat()

    recent = list(cluster.get("recent_routed_metrics", []))
    recent.append(bundle)
    cluster["recent_routed_metrics"] = recent[-ROLLING_WINDOW_SIZE * 2 :]

    cluster["queries_since_exploration"] = int(cluster.get("queries_since_exploration", 0)) + 1
    return update_cluster(cluster_id, cluster)
