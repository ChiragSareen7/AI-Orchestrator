from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any
from uuid import uuid4

from app.services.evaluator import evaluate_response
from app.services.logger import append_log, read_logs
from app.services.model_executor import execute_model


def _tokens(text: str) -> set[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return {word for word in cleaned.split() if len(word) > 2}


def _text_score(needle: str, haystack: str) -> float:
    if not needle.strip() or not haystack.strip():
        return 0.0

    needle_tokens = _tokens(needle)
    haystack_tokens = _tokens(haystack)
    overlap = len(needle_tokens & haystack_tokens) / max(len(needle_tokens), 1)
    sequence = SequenceMatcher(None, needle.lower(), haystack.lower()).ratio()
    return (0.75 * overlap) + (0.25 * sequence)


def _response_candidates(logs: list[dict]) -> list[dict]:
    candidates = []
    for log_index, log in enumerate(logs):
        if log.get("type") == "followup":
            continue

        for response_index, response in enumerate(log.get("responses") or []):
            if not response.get("response"):
                continue

            log_id = log.get("id") or f"legacy-log-{log_index}"
            response_id = response.get("response_id") or f"{log_id}-response-{response_index}"
            searchable = "\n".join(
                [
                    str(log.get("query", "")),
                    str(log.get("best_response", "")),
                    str(response.get("model", "")),
                    str(response.get("prompt_version", "")),
                    str(response.get("prompt", "")),
                    str(response.get("response", "")),
                ]
            )

            candidates.append(
                {
                    "log": log,
                    "log_index": log_index,
                    "log_id": log_id,
                    "response": response,
                    "response_index": response_index,
                    "response_id": response_id,
                    "searchable": searchable,
                }
            )

    return candidates


def _find_target_response(
    logs: list[dict],
    followup_question: str,
    target_reference: str | None,
    parent_log_id: str | None,
    parent_response_id: str | None,
) -> dict:
    candidates = _response_candidates(logs)
    if not candidates:
        raise ValueError("No previous model responses found in logs.json.")

    if parent_log_id and parent_response_id:
        for candidate in candidates:
            if candidate["log_id"] == parent_log_id and candidate["response_id"] == parent_response_id:
                return {**candidate, "match_score": 1.0}

    reference = " ".join(part for part in [target_reference, followup_question] if part)
    scored = [
        {
            **candidate,
            "match_score": _text_score(reference, candidate["searchable"]),
        }
        for candidate in candidates
    ]
    scored.sort(key=lambda item: (item["match_score"], item["log_index"]), reverse=True)

    best = scored[0]
    if best["match_score"] <= 0:
        raise ValueError("Could not identify which previous response the follow-up refers to.")

    return best


def _followup_instruction(prompt_version: str | None) -> str:
    if prompt_version == "v2":
        return "Explain in detail with useful context."
    if prompt_version == "v3":
        return "Answer strictly with facts only. If uncertain, say uncertain."
    return "Answer clearly and concisely."


def _build_followup_prompt(
    original_query: str,
    original_response: str,
    followup_question: str,
    prompt_version: str | None,
) -> str:
    return f"""{_followup_instruction(prompt_version)} You are answering a follow-up question about one previous model response.

Original user query:
{original_query}

Previous model response:
{original_response}

Follow-up question:
{followup_question}

Answer the follow-up directly using the original query and previous response as context."""


def answer_followup(
    followup_question: str,
    *,
    target_reference: str | None = None,
    parent_log_id: str | None = None,
    parent_response_id: str | None = None,
) -> dict[str, Any]:
    logs = read_logs()
    target = _find_target_response(
        logs,
        followup_question,
        target_reference,
        parent_log_id,
        parent_response_id,
    )

    original_log = target["log"]
    original_response = target["response"]
    model = original_response["model"]
    prompt_version = original_response.get("prompt_version")
    followup_prompt = _build_followup_prompt(
        str(original_log.get("query", "")),
        str(original_response.get("response", "")),
        followup_question,
        prompt_version,
    )

    raw = execute_model(model, followup_question, followup_prompt)
    metrics = evaluate_response(
        followup_question,
        raw.get("response", ""),
        raw["latency"],
        raw["tokenUsage"],
        raw.get("error"),
        context=f"{original_log.get('query', '')}\n{original_response.get('response', '')}",
    )

    followup_response = {
        "response_id": str(uuid4()),
        "model": model,
        "prompt_version": prompt_version,
        "prompt": followup_prompt,
        "response": raw.get("response", ""),
        "error": raw.get("error"),
        "metrics": metrics,
    }
    errors = [raw["error"]] if raw.get("error") else []
    saved_log = append_log(
        {
            "type": "followup",
            "parent_log_id": target["log_id"],
            "parent_response_id": target["response_id"],
            "parent_response_index": target["response_index"],
            "query": followup_question,
            "followup_query": followup_question,
            "original_query": original_log.get("query", ""),
            "matched_model": model,
            "matched_prompt": prompt_version,
            "matched_response_excerpt": str(original_response.get("response", ""))[:500],
            "match_score": round(float(target["match_score"]), 4),
            "responses": [followup_response],
            "best_model": model,
            "best_prompt": prompt_version,
            "best_response": raw.get("response", ""),
            "metrics": metrics,
            "errors": errors,
        }
    )

    return {
        "log_id": saved_log["id"],
        "parent_log_id": target["log_id"],
        "parent_response_id": target["response_id"],
        "parent_response_index": target["response_index"],
        "match_score": round(float(target["match_score"]), 4),
        "original_query": original_log.get("query", ""),
        "matched_model": model,
        "matched_prompt": prompt_version,
        "answer": raw.get("response", ""),
        "metrics": metrics,
        "errors": errors,
    }
