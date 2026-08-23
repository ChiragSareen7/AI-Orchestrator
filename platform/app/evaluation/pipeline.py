from __future__ import annotations  # allows modern type hint syntax on older Python

from typing import Any  # Any = type hint meaning "any type is acceptable"

from app.evaluation.config import EvaluationThresholds  # all tunable thresholds/settings
from app.evaluation.embeddings import relevance_query_response, semantic_similarity_pair  # embedding similarity
from app.evaluation.grounding import chunk_context, groundedness_from_chunks, split_sentences  # grounding check
from app.evaluation.llm_judge import make_judge  # LLM-as-judge factory
from app.evaluation.logging_hooks import emit_eval_event  # optional event hooks for logging
from app.evaluation.nli import entailment_mean_score  # NLI entailment scoring

DEFAULT_THRESHOLDS = EvaluationThresholds()
# creates one shared EvaluationThresholds instance using env var values
# re-used if no custom thresholds are passed to run_semantic_evaluation


def run_semantic_evaluation(
    query: str,           # the user's original question
    response: str,        # the model's answer to evaluate
    *,                    # all following args must be keyword-only
    context: str = "",    # retrieved context text (for grounding check)
    ground_truth: str | None = None,  # known correct answer (for accuracy scoring)
    thresholds: EvaluationThresholds | None = None,  # optional custom thresholds
) -> dict[str, Any]:
    # the main semantic evaluation function — runs all quality checks on a response
    # returns a dict with: relevance, groundedness, hallucination, accuracy, confidence, ...

    t = thresholds or DEFAULT_THRESHOLDS  # use provided thresholds or defaults
    emb = t.embedding_model  # e.g. "sentence-transformers/all-MiniLM-L6-v2"
    q, r = (query or "").strip(), (response or "").strip()  # clean both inputs

    # ── Build metadata dict for debugging ────────────────────────────────────
    meta_base = {
        "model_used": emb,        # which embedding model was used
        "embedding_model": emb,
        "thresholds": {           # log what thresholds were used (useful for debugging)
            "grounding_min_sim": t.grounding_min_sim,
            "weights": {
                "groundedness": t.weight_groundedness,
                "relevance": t.weight_relevance,
                "entailment": t.weight_entailment,
            },
        },
        "grounding_skipped": False,  # will update below if context is empty
        "nli_enabled": t.enable_nli,
        "llm_judge_used": False,     # will update below if LLM judge runs
    }

    if not r:
        # ── Empty response: return worst-case scores immediately ───────────────
        out = _empty_semantic(meta_base, emb)
        emit_eval_event("semantic_eval", {"query": q, "empty_response": True})
        # fire an event so any registered hooks know evaluation happened with empty response
        return out

    # ── Step 1: Compute relevance ─────────────────────────────────────────────
    relevance = relevance_query_response(q, r, emb)
    # cosine similarity between query and response embeddings
    # measures: how semantically related is the response to the question?

    # ── Step 2: Split response into sentences for grounding ───────────────────
    sentences = split_sentences(r)
    if not sentences and r:
        sentences = [r[:2000]]  # if splitting failed, use the whole response as one sentence

    # ── Step 3: Chunk the context ─────────────────────────────────────────────
    chunks = chunk_context(context)
    # split context into 480-char paragraphs for comparison against each sentence

    grounding_skipped = not (context or "").strip()
    # True if no context was provided — can't do grounding without context

    unsupported: list[str] = []    # sentences that aren't backed by context
    entailment_avg: float | None = None  # NLI score (computed later if enabled)

    # ── Step 4: Groundedness check ────────────────────────────────────────────
    if grounding_skipped:
        groundedness = float(relevance)
        # no context → use query-response relevance as a proxy for groundedness
        # (we can't verify grounding without source material)
        meta_base["grounding_skipped"] = True
        hallucination = 0.0
        # can't detect hallucination without context, so set to 0.0 (optimistic)
    else:
        g, unsupported = groundedness_from_chunks(sentences, chunks, emb, t.grounding_min_sim)
        groundedness = float(g)  # fraction of response sentences backed by context

        n_sent = len(sentences) if sentences else 1
        hallucination = len(unsupported) / max(n_sent, 1)
        # hallucination = fraction of sentences NOT grounded in context
        # max(n_sent, 1) prevents division by zero if sentences is empty

    # ── Step 5: NLI check (optional — detects contradictions) ─────────────────
    if t.enable_nli and not grounding_skipped and sentences:
        # only run NLI if:
        # 1. enabled in config (EVAL_ENABLE_NLI=true)
        # 2. context was provided (otherwise nothing to compare against)
        # 3. there are sentences to check
        try:
            ent_avg, nli_lab_list = entailment_mean_score(context, sentences, t.nli_model)
            entailment_avg = ent_avg  # average entailment score across all sentences

            for sent, lab in zip(sentences, nli_lab_list):
                # zip() pairs each sentence with its NLI label
                if lab == "contradiction" and sent not in unsupported:
                    unsupported.append(sent)
                # if NLI found a contradiction that grounding didn't catch, add it to unsupported

            if nli_lab_list:
                contrad = sum(1 for x in nli_lab_list if x == "contradiction")
                # count sentences labeled as "contradiction"
                hallucination = max(hallucination, contrad / len(nli_lab_list))
                # take the HIGHER hallucination score between embedding-based and NLI-based
                # this is conservative: if EITHER method detects hallucination, we flag it
        except Exception:
            entailment_avg = None  # NLI failed → ignore, continue with what we have

    # ── Step 6: Accuracy scoring ──────────────────────────────────────────────
    if ground_truth and (ground_truth or "").strip():
        accuracy = semantic_similarity_pair(r, ground_truth.strip(), emb)
        # if a known correct answer was provided: compare response to ground truth
        # high similarity = response is close to the correct answer
    else:
        accuracy = float(groundedness * relevance)
        # no ground truth available → use proxy:
        # a response is "accurate" if it's both relevant to the query AND grounded in context
        # this is an approximation — not as reliable as having a ground truth

    # ── Step 7: Confidence scoring ────────────────────────────────────────────
    wg, wr, we = t.weight_groundedness, t.weight_relevance, t.weight_entailment
    # unpack weights for readability

    if entailment_avg is not None:
        # NLI ran successfully → include its score in confidence
        wsum = wg + wr + we  # total weight = 0.35 + 0.35 + 0.30 = 1.0
        confidence = (wg * groundedness + wr * relevance + we * entailment_avg) / wsum
        # weighted average of all three signals
    else:
        # NLI not run or failed → use only groundedness and relevance
        wsum = wg + wr   # total weight = 0.35 + 0.35 = 0.70
        confidence = (wg * groundedness + wr * relevance) / wsum if wsum else relevance
        # weighted average of two signals, normalized by their combined weight

    confidence = max(0.0, min(1.0, float(confidence)))
    # clamp confidence to [0.0, 1.0] range (avoid floating point edge cases)

    # ── Step 8: LLM judge (optional — most expensive) ─────────────────────────
    judge_raw: dict[str, Any] | None = None
    if t.enable_llm_judge:
        # EVAL_ENABLE_LLM_JUDGE=true → call an LLM to judge the response
        judge_fn = make_judge(None)  # get the default Groq judge function
        if judge_fn:
            judge_raw = judge_fn(q, r, context)  # call the judge
            meta_base["llm_judge_used"] = judge_raw is not None
            # record whether the judge actually ran (it might return None on error)

            if judge_raw:
                jc = float(judge_raw.get("correctness", 0.5))
                # get the judge's correctness score (0-1)
                confidence = max(0.0, min(1.0, 0.5 * confidence + 0.5 * jc))
                # blend: 50% from our computed confidence + 50% from LLM judge's correctness
                # this makes the final confidence account for both mathematical and AI assessment

    # ── Step 9: Assemble and return results ───────────────────────────────────
    result = {
        "relevance": round(float(relevance), 4),      # 0-1, how on-topic the response is
        "groundedness": round(float(groundedness), 4),# 0-1, how much is backed by context
        "hallucination": round(float(hallucination), 4),# 0-1, how much might be made up
        "accuracy": round(float(accuracy), 4),        # 0-1, how correct the answer is
        "confidence": round(float(confidence), 4),    # 0-1, overall quality signal
        "unsupported_sentences": unsupported[:50],    # list of sentences not in context (max 50)
        "metadata": {
            **meta_base,  # spread all the metadata fields we built earlier
            "entailment_avg": round(entailment_avg, 4) if entailment_avg is not None else None,
            "llm_judge": judge_raw,  # the judge's full response dict (or None)
        },
    }

    emit_eval_event("semantic_eval", {
        "query_len": len(q),
        "response_len": len(r),
        "result": result,
    })
    # fire event for any registered hooks (e.g. for logging to external systems)

    return result


def _empty_semantic(meta_base: dict, emb: str) -> dict[str, Any]:
    # returns worst-case semantic scores when the response is empty
    z = {
        "relevance": 0.0,           # no relevance (nothing to compare)
        "groundedness": 0.0,        # not grounded (nothing was said)
        "hallucination": 1.0,       # maximum hallucination (no response = failure)
        "accuracy": 0.0,            # no accuracy
        "confidence": 0.0,          # no confidence
        "unsupported_sentences": [], # no sentences to check
        "metadata": {**meta_base, "embedding_model": emb},
    }
    return z


def load_thresholds_from_env() -> EvaluationThresholds:
    # creates a fresh EvaluationThresholds instance reading all values from env vars
    # called by evaluator.py each time a response needs to be scored
    return EvaluationThresholds()
