from __future__ import annotations  # allows modern type hint syntax on older Python

from collections import defaultdict  # defaultdict is a dict that auto-creates missing keys with a default value
from uuid import uuid4

# import all the individual service functions this orchestrator needs to call
from app.services.query_analyzer import analyze_query       # detects domain, complexity, intent
from app.services.prompt_generator import generate_prompts  # creates 3 prompt variants
from app.services.model_executor import execute_model       # actually calls the model (HTTP or Groq API)
from app.services.evaluator import evaluate_response        # scores each response (lexical + semantic)
from app.services.ranker import rank_responses              # sorts responses from best to worst
from app.services.learning import update_learning           # saves which model won, updates performance averages
from app.services.logger import append_log                  # writes the full query result to logs.json
from app.services.adaptive_routing.router import plan_query_execution, record_query_outcome
from app.utils.similarity import average_pairwise_similarity  # measures how consistent a model's responses are


def _blue_metrics(responses: list[dict]) -> dict:
    # computes cross-model aggregate metrics (BLUE = Behavior Level Understanding Evaluation)
    # 'responses' is the full list of all model+prompt combination results

    by_model = defaultdict(list)  # groups responses by model name; defaultdict creates an empty list for any new key
    for r in responses:
        by_model[r["model"]].append(r)  # add each response under its model's group

    # collect latency, cost, error rate from ALL responses (across all models and prompts)
    all_lat = [r["metrics"]["latency"] for r in responses]    # list of all latency values in milliseconds
    all_cost = [r["metrics"]["cost"] for r in responses]      # list of all cost values in dollars
    all_err = [r["metrics"]["errorRate"] for r in responses]  # list of all error rates (0.0 or 1.0)

    stabilities = []  # will hold consistency scores per model
    for vals in by_model.values():
        # for each model, look at all 3 responses (one per prompt variant)
        texts = [v.get("response", "") for v in vals if v.get("response")]  # get non-empty response texts
        if len(texts) >= 2:  # need at least 2 responses to compute pairwise similarity
            stabilities.append(average_pairwise_similarity(texts))
            # average_pairwise_similarity measures: if same model, does it give similar answers to different prompts?
            # high value = consistent model; low value = non-deterministic/unstable model

    # compute the overall behavior stability: average across all models
    behavior_stability = round(sum(stabilities) / len(stabilities), 4) if stabilities else 0.0
    # round to 4 decimal places; if no stabilities calculated, default to 0.0

    return {
        "behaviorStability": behavior_stability,  # 0-1, how consistent models are across prompt variants
        "latency": round(sum(all_lat) / len(all_lat), 2) if all_lat else 0.0,  # average latency across ALL calls
        "usageCost": round(sum(all_cost), 6),    # TOTAL cost of all model calls (sum, not average)
        "errorRate": round(sum(all_err) / len(all_err), 4) if all_err else 0.0,  # average error rate
    }


def run_query_pipeline(
    query: str,             # the user's question
    context: str | None = None,        # optional: retrieved document chunks for grounding
    ground_truth: str | None = None,   # optional: a known correct answer for accuracy scoring
    force_explore: str | None = None,  # optional: cluster-scoped force re-exploration
    enable_routing: bool = True,  # when False, always call all models
) -> dict:
    # ── Step 1: Understand the query ─────────────────────────────────────────
    analysis = analyze_query(query)
    # returns: {domain: "chemistry", complexity: "low", intent: "fact_lookup", domain_scores: {...}}
    # domain tells us WHICH model to use first

    # ── Step 2: Adaptive routing — classify cluster and choose models ────────
    execution_plan = plan_query_execution(
        query=query,
        analysis=analysis,
        force_explore_cluster=force_explore,
        enable_routing=enable_routing,
    )
    models_to_run = execution_plan["models"]
    routing_decision = execution_plan["routing_decision"]

    # ── Step 3: Generate 3 prompt variants ───────────────────────────────────
    prompts = generate_prompts(query)
    # returns: [
    #   {"version": "v1", "text": "Answer clearly and concisely: <query>"},
    #   {"version": "v2", "text": "Explain in detail with useful context: <query>"},
    #   {"version": "v3", "text": "Answer strictly with facts only: <query>"},
    # ]

    # ── Step 4: Run selected model(s) × every prompt combination ─────────────
    responses = []  # will collect all results here
    for model in models_to_run:
        for prompt in prompts:
            # call the model — this does an HTTP request to :8010 or a Groq API call
            raw = execute_model(model, query, prompt["text"])
            # raw = {model, query, response, error, latency, tokenUsage}

            # score the response — how accurate, relevant, hallucinated?
            metrics = evaluate_response(
                query,                          # the original user question
                raw.get("response", ""),        # what the model said (empty string if error)
                raw["latency"],                 # how long the call took in milliseconds
                raw["tokenUsage"],              # estimated number of tokens used
                raw.get("error"),               # error message if the call failed (None if success)
                context=context,                # optional context for grounding check
                ground_truth=ground_truth,      # optional correct answer for accuracy
            )

            # store the complete result for this model+prompt combination
            responses.append(
                {
                    "response_id": str(uuid4()),
                    "model": model,                       # e.g. "organic_model"
                    "prompt_version": prompt["version"],  # e.g. "v1"
                    "prompt": prompt["text"],             # the full prompt text sent to the model
                    "response": raw.get("response", ""),  # the model's answer
                    "error": raw.get("error"),            # error message or None
                    "metrics": metrics,                   # full scoring dict
                }
            )

    # ── Step 5: Rank all responses from best to worst ─────────────────────────
    ranked = rank_responses(responses)
    # sorts by: accuracy (descending) → hallucination (ascending) → latency (ascending)
    best = ranked[0]  # the top-ranked response is the winner

    # ── Step 6: Handle the case where ALL models failed ──────────────────────
    all_failed = all((r.get("error") is not None) for r in responses)
    # checks: does EVERY response have an error? if yes, all models failed

    if all_failed:
        errors = []
        for r in responses:
            err = r.get("error")
            if err and err not in errors:  # collect unique error messages
                errors.append(err)
        best_answer = "All model calls failed. Check model server/Groq credentials."
        # safe fallback message shown to user when every model fails
    else:
        errors = []  # no errors to report
        best_answer = best["response"]  # use the top-ranked model's response

    # ── Step 7: Update learning system ───────────────────────────────────────
    learning_domain = execution_plan.get("cluster_id", analysis["domain"]).split("/")[0]
    update_learning(learning_domain, best["model"], best["metrics"])
    record_query_outcome(execution_plan, query, responses, best, analysis)
    # tells the learning system: "for domain X, model Y won with these metrics"
    # this updates the running average accuracy/latency and saves it to model_performance.json
    # next time this domain is queried, the winning model gets promoted in the ranking

    # ── Step 8: Compute cross-model aggregate metrics ─────────────────────────
    blue = _blue_metrics(responses)
    # stability, avg latency, total cost, avg error rate across all 12 calls

    # ── Step 9: Build the log entry and save it ───────────────────────────────
    log_entry = {
        "query": query,             # the original question
        "analysis": analysis,       # domain/complexity/intent
        "routing": routing_decision,
        "responses": responses,     # model+prompt results for this execution mode
        "best_model": best["model"],        # which model won
        "best_prompt": best["prompt_version"],  # which prompt version won
        "best_response": best_answer,       # the final answer
        "metrics": best["metrics"],         # the winning model's scores
        "blue_metrics": blue,               # cross-model aggregate stats
        "errors": errors,                   # any error messages
    }
    saved_log = append_log(log_entry)  # writes this entire dict to store/logs.json

    # ── Step 10: Return the response to the API caller ────────────────────────
    return {
        "log_id": saved_log["id"],                # stable id for follow-up questions
        "best_answer": best_answer,          # the actual answer to show the user
        "best_model": best["model"],         # e.g. "organic_model"
        "best_prompt": best["prompt_version"],  # e.g. "v1"
        "metrics": best["metrics"],          # accuracy, hallucination, relevance, etc.
        "blue_metrics": blue,                # behaviorStability, avg latency, total cost
        "analysis": analysis,                # domain, complexity, intent
        "routing": routing_decision,
        "errors": errors,                    # empty list if all went well
        "all_responses": ranked,             # all 12 responses sorted best-first (for inspection)
    }
