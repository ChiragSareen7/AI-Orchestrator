from __future__ import annotations  # allows modern type hint syntax on older Python

import os         # for reading environment variables
import traceback  # for capturing detailed error information when something fails

from app.utils.similarity import jaccard_similarity, keyword_overlap
# jaccard_similarity: measures word overlap between two texts (0.0 to 1.0)
# keyword_overlap: what fraction of query keywords appear in the response


def _legacy_lexical_metrics(query: str, response: str, latency: float, token_usage: int) -> dict:
    # computes simple word-based (lexical) quality scores — fast, no AI needed
    # these are the FALLBACK scores used when semantic evaluation is disabled or fails

    relevance = keyword_overlap(query, response)
    # measures: what fraction of the query's keywords appear in the response?
    # e.g. query="boiling point benzene", response mentions "benzene boiling" → high relevance

    lexical = jaccard_similarity(query, response)
    # measures: what fraction of words are shared between query AND response (union)?
    # Jaccard = |intersection| / |union| — penalizes responses that add many off-topic words

    accuracy = min(1.0, (0.65 * relevance) + (0.35 * lexical))
    # blend the two scores: 65% weight on relevance, 35% on Jaccard similarity
    # min(1.0, ...) caps the result at 1.0 (can't be more than 100% accurate)

    low_quality_markers = ["i don't know", "uncertain", "not sure", "cannot determine"]
    # these phrases signal the model is unsure → reduce confidence, raise hallucination concern
    has_marker = any(m in response.lower() for m in low_quality_markers)
    # True if ANY of these phrases appear anywhere in the response (case-insensitive)

    hallucination = max(0.0, 1.0 - accuracy + (0.1 if has_marker else 0.0))
    # hallucination = how much of the response might be made up
    # starts as (1 - accuracy): if accuracy is high, hallucination is low
    # adds 0.1 penalty if the model expressed uncertainty
    # max(0.0, ...) ensures it never goes below 0

    confidence = max(0.0, min(1.0, accuracy - (latency / 40000.0) - (0.15 if has_marker else 0.0)))
    # confidence = how much we trust this response overall
    # starts from accuracy, then subtracts:
    #   - latency / 40000 : slow responses are slightly less trustworthy (small penalty)
    #   - 0.15 if uncertain marker found : big penalty for expressed uncertainty
    # clamped between 0.0 and 1.0

    toxicity = 0.0  # toxicity detection is not implemented — always 0

    return {
        "accuracyScore": round(accuracy, 4),         # overall correctness proxy (0-1)
        "relevanceScore": round(relevance, 4),        # keyword coverage (0-1)
        "hallucinationScore": round(hallucination, 4),# how much might be made up (0=good, 1=bad)
        "confidenceScore": round(confidence, 4),      # overall trust level (0-1)
        "latency": latency,                           # response time in milliseconds
        "tokenUsage": token_usage,                    # estimated tokens used
        "cost": round(token_usage * 0.000002, 6),    # estimated cost: $0.000002 per token
        "toxicityScore": toxicity,                    # always 0.0 (not implemented)
        "errorRate": 0.0,                             # no error (if we got here, it succeeded)
    }


def evaluate_response(
    query: str,           # the user's original question
    response: str,        # the model's answer
    latency: float,       # how long the call took (ms)
    token_usage: int,     # estimated tokens used
    error: str | None,    # None if successful, error string if the model call failed
    *,                    # everything after * must be passed as keyword argument (e.g. context=...)
    context: str | None = None,       # optional: retrieved context chunks for grounding
    ground_truth: str | None = None,  # optional: the known correct answer for accuracy
) -> dict:
    # main evaluation function — returns a full metrics dict for one model+prompt result

    if error:
        # if the model call failed entirely, return worst-case scores immediately
        return {
            "accuracyScore": 0.0,        # no accuracy — we have no answer
            "relevanceScore": 0.0,       # no relevance
            "hallucinationScore": 1.0,   # maximum hallucination (we treat error as worst case)
            "confidenceScore": 0.0,      # zero confidence
            "latency": latency,          # still record the latency (how long until failure)
            "tokenUsage": token_usage,   # still record tokens (might be > 0 for partial calls)
            "cost": round(token_usage * 0.000002, 6),
            "toxicityScore": 0.0,
            "errorRate": 1.0,            # error rate = 1.0 (this call completely failed)
            "semantic": None,            # no semantic evaluation (no response to evaluate)
        }

    # ── Tier 1: Lexical scoring (always runs) ────────────────────────────────
    legacy = _legacy_lexical_metrics(query, response, latency, token_usage)
    # compute the fast word-based metrics first

    # ── Tier 2: Semantic scoring (optional, controlled by env var) ────────────
    semantic: dict | None = None
    if os.getenv("EVAL_ENABLE_SEMANTIC", "true").lower() in ("1", "true", "yes"):
        # only run semantic evaluation if EVAL_ENABLE_SEMANTIC is set to true in .env
        # it's expensive (loads AI models) so we allow turning it off
        try:
            from app.evaluation.pipeline import load_thresholds_from_env, run_semantic_evaluation
            # lazy import inside the try block — only loads these heavy modules if needed

            semantic = run_semantic_evaluation(
                query,
                response,
                context=context or "",      # if context is None, use empty string
                ground_truth=ground_truth,
                thresholds=load_thresholds_from_env(),  # reads EVAL_* env vars for thresholds
            )
            # semantic dict contains: {relevance, groundedness, hallucination, accuracy, confidence, ...}

        except Exception:
            # if semantic evaluation fails for any reason, fall back to None
            # this ensures the whole pipeline doesn't crash just because semantic eval failed
            semantic = {
                "relevance": 0.0,
                "groundedness": 0.0,
                "hallucination": 1.0,       # worst case
                "accuracy": 0.0,
                "confidence": 0.0,
                "unsupported_sentences": [],
                "metadata": {
                    "error": traceback.format_exc()[:800],  # capture error details (first 800 chars)
                    "embedding_model": os.getenv("EVAL_EMBEDDING_MODEL", ""),
                    "model_used": "error",
                },
            }
    else:
        semantic = None  # semantic evaluation is disabled by env var

    return {**legacy, "semantic": semantic}
    # {**legacy} = spread all key-value pairs from the legacy dict into the new dict
    # then add the "semantic" key with the semantic evaluation results
    # final dict has ALL lexical metrics PLUS the "semantic" sub-dict
