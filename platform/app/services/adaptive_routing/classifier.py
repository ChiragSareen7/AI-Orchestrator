"""Query classification: embedding similarity, LLM fallback, and extension points."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import httpx
import numpy as np

from app.evaluation.embeddings import cosine_similarity_vec, embed_texts
from app.services.adaptive_routing.config import (
    EMBEDDING_AMBIGUITY_MARGIN,
    EMBEDDING_MODEL,
    EMBEDDING_SIMILARITY_THRESHOLD,
    LLM_CLASSIFIER_ENABLED,
    LLM_CLASSIFIER_MODEL,
    SEED_CLUSTERS,
)
from app.services.adaptive_routing.storage import load_routing_table


@dataclass
class ClassificationResult:
    cluster_id: str
    method: str
    confidence: float
    similarities: dict[str, float]
    ambiguous: bool = False
    llm_domain: str | None = None


class QueryClassifier(Protocol):
    """Extension point: plug in a future fine-tuned routing model here."""

    def classify(self, query: str, analysis: dict | None = None) -> ClassificationResult: ...


def classify_with_fine_tuned_router(query: str, analysis: dict | None = None) -> ClassificationResult:
    """
    Extension point for a future fine-tuned routing model.

    Replace `classify_query` to delegate here once a trained router is available.
    """
    raise NotImplementedError(
        "Fine-tuned routing model is not configured. "
        "Use embedding/LLM classification or wire a trained model into this function."
    )


def _cluster_similarities(query_embedding: np.ndarray, table: dict) -> dict[str, float]:
    scores: dict[str, float] = {}
    for cluster_id, cluster in table.get("clusters", {}).items():
        centroid = cluster.get("centroid")
        if not centroid:
            continue
        scores[cluster_id] = cosine_similarity_vec(query_embedding, np.asarray(centroid, dtype=np.float32))
    return scores


def _llm_classify_domain(query: str) -> tuple[str, float]:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return "general", 0.5

    base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    timeout = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "25"))
    prompt = (
        "Classify this user query into exactly one domain label.\n"
        "Valid labels: chemistry, python, gita, general\n"
        "Reply with only the label.\n\n"
        f"Query: {query}"
    )
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": LLM_CLASSIFIER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            label = resp.json()["choices"][0]["message"]["content"].strip().lower()
    except Exception:
        return "general", 0.5

    valid = set(SEED_CLUSTERS.keys())
    if label not in valid:
        for candidate in valid:
            if candidate in label:
                label = candidate
                break
        else:
            label = "general"
    return label, 0.75


def classify_query(query: str, analysis: dict | None = None) -> ClassificationResult:
    """
    Classify a query into a cluster using embedding similarity with LLM fallback.

    To swap in a fine-tuned router later, replace the body of this function or
    call `classify_with_fine_tuned_router` when configured.
    """
    table = load_routing_table()
    query_vec = embed_texts([query.strip()], EMBEDDING_MODEL)[0]
    similarities = _cluster_similarities(query_vec, table)

    if similarities:
        ranked = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
        best_id, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        ambiguous = (best_score - second_score) < EMBEDDING_AMBIGUITY_MARGIN
        if best_score >= EMBEDDING_SIMILARITY_THRESHOLD and not ambiguous:
            return ClassificationResult(
                cluster_id=best_id,
                method="embedding_match",
                confidence=best_score,
                similarities=similarities,
                ambiguous=False,
            )

    if LLM_CLASSIFIER_ENABLED:
        domain, llm_conf = _llm_classify_domain(query)
        return ClassificationResult(
            cluster_id=domain,
            method="llm_fallback",
            confidence=llm_conf,
            similarities=similarities,
            ambiguous=True,
            llm_domain=domain,
        )

    if analysis and analysis.get("domain") in SEED_CLUSTERS:
        domain = analysis["domain"]
    else:
        domain = "general"

    return ClassificationResult(
        cluster_id=domain,
        method="keyword_fallback",
        confidence=0.4,
        similarities=similarities,
        ambiguous=True,
        llm_domain=domain,
    )


def compute_query_embedding(query: str) -> list[float]:
    vec = embed_texts([query.strip()], EMBEDDING_MODEL)[0]
    return [float(x) for x in vec.tolist()]


def update_cluster_centroid(existing_centroid: list[float] | None, new_embedding: list[float], count: int) -> list[float]:
    arr = np.asarray(new_embedding, dtype=np.float32)
    if not existing_centroid or count <= 1:
        return [float(x) for x in arr.tolist()]
    old = np.asarray(existing_centroid, dtype=np.float32)
    updated = ((old * (count - 1)) + arr) / count
    norm = float(np.linalg.norm(updated))
    if norm > 1e-12:
        updated = updated / norm
    return [float(x) for x in updated.tolist()]
