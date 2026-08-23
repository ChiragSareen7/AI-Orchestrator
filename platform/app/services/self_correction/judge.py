"""LLM-as-judge for requirement compliance (separate from lexical/segment eval)."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx

from app.services.self_correction.config import (
    JUDGE_MODEL,
    JUDGE_PASS_SCORE_THRESHOLD,
    JUDGE_TIMEOUT_SECONDS,
)
from app.services.self_correction.requirements import profile_to_prompt_context


def judge_response(
    query: str,
    response: str,
    requirements_profile: dict,
    judge_fn=None,
) -> dict[str, Any]:
    """
    Return structured judgment: pass, score, reasons, raw details.
    """
    if judge_fn is not None:
        return judge_fn(query, response, requirements_profile)

    requirements_text = profile_to_prompt_context(requirements_profile)
    system = (
        "You are a strict requirements compliance judge. "
        "Evaluate whether the response meets the user's requirements profile. "
        "Reply with ONLY valid JSON, no markdown:\n"
        '{"pass": true|false, "score": 0.0-1.0, "reasons": ["specific reason", ...]}'
    )
    user = (
        f"User query:\n{query}\n\n"
        f"Requirements profile:\n{requirements_text}\n\n"
        f"Model response:\n{response[:8000]}\n\n"
        "List specific failures in reasons if pass is false."
    )

    key = os.getenv("GROQ_API_KEY") or os.getenv("EVAL_JUDGE_API_KEY")
    if not key:
        return {
            "pass": False,
            "score": 0.0,
            "reasons": ["Judge unavailable: missing GROQ_API_KEY"],
            "error": "missing_api_key",
        }

    base = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    try:
        with httpx.Client(timeout=JUDGE_TIMEOUT_SECONDS) as client:
            resp = client.post(
                f"{base.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": JUDGE_MODEL,
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        return {
            "pass": False,
            "score": 0.0,
            "reasons": [f"Judge call failed: {exc}"],
            "error": str(exc),
        }

    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {"pass": False, "score": 0.0, "reasons": ["Judge returned invalid JSON"], "raw": text}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"pass": False, "score": 0.0, "reasons": ["Judge returned invalid JSON"], "raw": text}

    score = float(parsed.get("score", 0.0))
    reasons = parsed.get("reasons") or []
    if not isinstance(reasons, list):
        reasons = [str(reasons)]

    explicit_pass = parsed.get("pass")
    if explicit_pass is not None:
        passed = bool(explicit_pass)
    else:
        passed = score >= JUDGE_PASS_SCORE_THRESHOLD
    if score >= JUDGE_PASS_SCORE_THRESHOLD:
        passed = True

    return {
        "pass": passed,
        "score": round(score, 4),
        "reasons": [str(r) for r in reasons if str(r).strip()],
        "threshold": JUDGE_PASS_SCORE_THRESHOLD,
    }
