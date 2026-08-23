"""Requirement-driven self-correction loop (4-attempt hard cap)."""

from __future__ import annotations

from uuid import uuid4

from app.services.adaptive_routing.config import ALL_MODELS
from app.services.adaptive_routing.router import get_ranked_models_for_cluster, plan_query_execution
from app.services.evaluator import evaluate_response
from app.services.model_executor import execute_model
from app.services.query_analyzer import analyze_query
from app.services.ranker import rank_responses
from app.services.self_correction.config import (
    MAX_ATTEMPTS,
    NEXT_BEST_FALLBACK_MODEL,
    REQUIREMENTS_PROFILE_REQUIRED,
)
from app.services.self_correction.judge import judge_response
from app.services.self_correction.logger import append_self_correction_log
from app.services.self_correction.prompt_enhancer import enhance_prompt
from app.services.self_correction.requirements import get_requirements_profile, resolve_requirements_profile


def _call_model(model: str, original_query: str, prompt_text: str, execute_fn=None) -> dict:
    runner = execute_fn or execute_model
    return runner(model, original_query, prompt_text)


def _run_all_models_pick_best(
    query: str,
    prompt_text: str,
    context: str | None,
    execute_fn=None,
) -> dict:
    """Call all 4 models and return the best-ranked response."""
    responses = []
    for model in ALL_MODELS:
        raw = _call_model(model, query, prompt_text, execute_fn=execute_fn)
        metrics = evaluate_response(
            query,
            raw.get("response", ""),
            raw["latency"],
            raw["tokenUsage"],
            raw.get("error"),
            context=context,
        )
        responses.append(
            {
                "response_id": str(uuid4()),
                "model": model,
                "prompt": prompt_text,
                "response": raw.get("response", ""),
                "error": raw.get("error"),
                "metrics": metrics,
            }
        )
    ranked = rank_responses(responses)
    return ranked[0]


def _run_single_model(
    model: str,
    query: str,
    prompt_text: str,
    context: str | None,
    execute_fn=None,
) -> dict:
    raw = _call_model(model, query, prompt_text, execute_fn=execute_fn)
    metrics = evaluate_response(
        query,
        raw.get("response", ""),
        raw["latency"],
        raw["tokenUsage"],
        raw.get("error"),
        context=context,
    )
    return {
        "response_id": str(uuid4()),
        "model": model,
        "prompt": prompt_text,
        "response": raw.get("response", ""),
        "error": raw.get("error"),
        "metrics": metrics,
    }


def _attempt_record(
    attempt_number: int,
    model: str,
    prompt: str,
    response: str,
    judgment: dict,
    selection_mode: str,
    metrics: dict | None = None,
) -> dict:
    record = {
        "attempt": attempt_number,
        "model": model,
        "prompt": prompt,
        "response": response,
        "selection_mode": selection_mode,
        "judge": judgment,
    }
    if metrics:
        record["metrics"] = metrics
    return record


def run_self_correction_loop(
    query: str,
    client_id: str,
    context: str | None = None,
    enable_routing: bool = True,
    judge_fn=None,
    execute_fn=None,
) -> dict:
    """
    Execute the 4-attempt self-correction sequence.

    Attempt 1: all 4 models with original query → best-ranked answer → judge.
    Attempt 2: same model, enhanced prompt from attempt 1 failures.
    Attempt 3: next-best model, original query.
    Attempt 4: same model as 3, enhanced prompt from attempt 3 failures.
    """
    if REQUIREMENTS_PROFILE_REQUIRED and not get_requirements_profile(client_id):
        raise ValueError(
            f"Requirements profile required for client_id={client_id}. "
            "Create one via PUT /requirements/{client_id} first."
        )

    profile = resolve_requirements_profile(client_id, allow_empty=True)
    analysis = analyze_query(query)
    routing_plan = plan_query_execution(
        query=query,
        analysis=analysis,
        enable_routing=enable_routing,
    )
    cluster_id = routing_plan.get("cluster_id", analysis.get("domain", "general"))

    attempts: list[dict] = []
    original_query = query

    # ── Attempt 1: all models, pick best ─────────────────────────────────────
    best_from_all = _run_all_models_pick_best(
        original_query, original_query, context, execute_fn=execute_fn
    )
    judgment_1 = judge_response(
        original_query, best_from_all.get("response", ""), profile, judge_fn=judge_fn
    )
    attempts.append(
        _attempt_record(
            1,
            best_from_all["model"],
            original_query,
            best_from_all.get("response", ""),
            judgment_1,
            "all_models_best_ranked",
            metrics=best_from_all.get("metrics"),
        )
    )
    if judgment_1.get("pass"):
        return _finalize(query, client_id, profile, cluster_id, routing_plan, attempts, 1, False)

    model_a = best_from_all["model"]

    # ── Attempt 2: same model, enhanced prompt ───────────────────────────────
    prompt_2 = enhance_prompt(original_query, profile, judgment_1.get("reasons") or [])
    result_2 = _run_single_model(model_a, original_query, prompt_2, context, execute_fn=execute_fn)
    judgment_2 = judge_response(
        original_query, result_2.get("response", ""), profile, judge_fn=judge_fn
    )
    attempts.append(
        _attempt_record(
            2,
            model_a,
            prompt_2,
            result_2.get("response", ""),
            judgment_2,
            "same_model_refined",
            metrics=result_2.get("metrics"),
        )
    )
    if judgment_2.get("pass"):
        return _finalize(query, client_id, profile, cluster_id, routing_plan, attempts, 2, False)

    # ── Attempt 3: switch model, original query ──────────────────────────────
    ranked_models = get_ranked_models_for_cluster(cluster_id, exclude=[model_a])
    model_b = ranked_models[0] if ranked_models else NEXT_BEST_FALLBACK_MODEL
    if model_b == model_a:
        model_b = next((m for m in ALL_MODELS if m != model_a), NEXT_BEST_FALLBACK_MODEL)

    result_3 = _run_single_model(model_b, original_query, original_query, context, execute_fn=execute_fn)
    judgment_3 = judge_response(
        original_query, result_3.get("response", ""), profile, judge_fn=judge_fn
    )
    attempts.append(
        _attempt_record(
            3,
            model_b,
            original_query,
            result_3.get("response", ""),
            judgment_3,
            "switched_model_original_query",
            metrics=result_3.get("metrics"),
        )
    )
    if judgment_3.get("pass"):
        return _finalize(query, client_id, profile, cluster_id, routing_plan, attempts, 3, False)

    # ── Attempt 4: same model as 3, enhanced prompt ─────────────────────────
    prompt_4 = enhance_prompt(original_query, profile, judgment_3.get("reasons") or [])
    result_4 = _run_single_model(model_b, original_query, prompt_4, context, execute_fn=execute_fn)
    judgment_4 = judge_response(
        original_query, result_4.get("response", ""), profile, judge_fn=judge_fn
    )
    attempts.append(
        _attempt_record(
            4,
            model_b,
            prompt_4,
            result_4.get("response", ""),
            judgment_4,
            "switched_model_refined",
            metrics=result_4.get("metrics"),
        )
    )
    if judgment_4.get("pass"):
        return _finalize(query, client_id, profile, cluster_id, routing_plan, attempts, 4, False)

    return _finalize(query, client_id, profile, cluster_id, routing_plan, attempts, None, True)


def _finalize(
    query: str,
    client_id: str,
    profile: dict,
    cluster_id: str,
    routing_plan: dict,
    attempts: list[dict],
    passed_on_attempt: int | None,
    requirements_not_met: bool,
) -> dict:
    if passed_on_attempt is not None:
        winner = attempts[passed_on_attempt - 1]
    else:
        winner = max(attempts, key=lambda a: float(a["judge"].get("score", 0.0)))

    outcome = {
        "query": query,
        "client_id": client_id,
        "cluster_id": cluster_id,
        "requirements_profile": profile,
        "routing_decision": routing_plan.get("routing_decision"),
        "attempts": attempts,
        "passed_on_attempt": passed_on_attempt,
        "requirements_not_met": requirements_not_met,
        "final_answer": winner.get("response", ""),
        "final_model": winner.get("model"),
        "final_judge": winner.get("judge"),
        "final_prompt": winner.get("prompt"),
        "message": (
            "Requirements met."
            if passed_on_attempt
            else "Did not fully meet requirements — best available attempt."
        ),
    }

    saved = append_self_correction_log(outcome)
    return {
        **outcome,
        "log_id": saved["id"],
        "best_answer": outcome["final_answer"],
        "best_model": outcome["final_model"],
        "requirements_met": passed_on_attempt is not None,
    }
