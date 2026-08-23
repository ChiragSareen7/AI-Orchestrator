"""Extract routing-relevant metrics from evaluation results."""

from __future__ import annotations


def extract_metric_bundle(metrics: dict) -> dict:
    """Return lexical accuracy, semantic accuracy (if present), and confidence."""
    sem = metrics.get("semantic")
    semantic_accuracy = None
    if isinstance(sem, dict) and not (sem.get("metadata") or {}).get("error"):
        semantic_accuracy = float(sem.get("accuracy", metrics.get("accuracyScore", 0.0)))

    return {
        "accuracy": float(metrics.get("accuracyScore", 0.0)),
        "semantic_accuracy": semantic_accuracy,
        "confidence": float(metrics.get("confidenceScore", 0.0)),
        "latency": float(metrics.get("latency", 0.0)),
    }


def metric_values_for_triggers(bundle: dict) -> list[tuple[str, float]]:
    """Return named metrics used for re-exploration trigger checks."""
    values: list[tuple[str, float]] = [
        ("accuracy", bundle["accuracy"]),
        ("confidence", bundle["confidence"]),
    ]
    if bundle.get("semantic_accuracy") is not None:
        values.append(("semantic_accuracy", float(bundle["semantic_accuracy"])))
    return values
