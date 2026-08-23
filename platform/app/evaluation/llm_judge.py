from __future__ import annotations  # allows modern type hint syntax on older Python

import json    # for parsing the LLM's JSON response
import os      # for reading environment variables
import re      # for cleaning up the LLM's response (removing markdown code fences)
from typing import Any, Callable, Protocol  # type hint utilities

import httpx  # HTTP client for calling the Groq API


class JudgeProvider(Protocol):
    # Protocol defines a "shape" for a callable — any function with this signature qualifies
    # this is called "duck typing" — if it looks like a duck (has __call__ with these args), it is one
    def __call__(self, query: str, response: str, context: str) -> dict[str, Any] | None:
        # a judge function takes (query, response, context) and returns a dict of scores
        # returns None if the evaluation failed (e.g. API error)
        ...


def default_groq_judge(query: str, response: str, context: str) -> dict[str, Any] | None:
    # asks Groq's Llama to evaluate the quality of a response
    # this is the "LLM-as-judge" pattern: use one LLM to grade another LLM's output
    # returns: {correctness: 0-1, relevance: 0-1, hallucination: 0-1} or None

    key = os.getenv("GROQ_API_KEY") or os.getenv("EVAL_JUDGE_API_KEY")
    # try to get the API key from either env var
    if not key:
        return None  # no API key → can't call Groq → return None

    base = os.getenv("EVAL_JUDGE_BASE_URL", os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"))
    # get the API base URL: check EVAL_JUDGE_BASE_URL first, fall back to GROQ_BASE_URL, then hardcode default

    model = os.getenv("EVAL_JUDGE_MODEL", os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))
    # get the model to use for judging — can be a different model from the one answering questions

    system = (
        "You are an evaluation judge. Reply with ONLY valid JSON, no markdown: "
        '{"correctness":0-1,"relevance":0-1,"hallucination":0-1} '
        "where hallucination is degree of unsupported/fabricated content (higher=worse)."
    )
    # the system prompt gives the LLM its role and tells it EXACTLY what format to use
    # "ONLY valid JSON, no markdown" = prevents the LLM from wrapping in ```json ... ```
    # scores are 0-1: correctness and relevance higher = better; hallucination higher = worse

    user = f"query:\n{query}\n\ncontext:\n{context[:6000]}\n\nresponse:\n{response[:8000]}"
    # the user message contains the query, context, and response for the judge to evaluate
    # [:6000] and [:8000] truncate to stay within Groq's token limits

    try:
        with httpx.Client(timeout=45.0) as client:
            # 45 second timeout — LLM judge calls can be slow
            r = client.post(
                f"{base.rstrip('/')}/chat/completions",
                # rstrip('/') removes trailing slash to avoid double slashes in the URL
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "temperature": 0,  # temperature=0 = fully deterministic = same judgment each time
                    "messages": [
                        {"role": "system", "content": system},  # instructions for the judge
                        {"role": "user", "content": user},      # the content to judge
                    ],
                },
            )
            r.raise_for_status()  # raise exception if HTTP error (4xx or 5xx)
            text = r.json()["choices"][0]["message"]["content"].strip()
            # extract the LLM's response text
    except Exception:
        return None  # if ANY error occurs (network, API, etc.) → return None gracefully

    # ── Clean up the response ─────────────────────────────────────────────────
    text = re.sub(r"^```json\s*", "", text)   # remove opening ```json if present
    text = re.sub(r"\s*```$", "", text)        # remove closing ``` if present
    # some LLMs still wrap their JSON in markdown code fences even when told not to

    try:
        return json.loads(text)  # parse the JSON string into a Python dict
    except json.JSONDecodeError:
        # the LLM didn't return clean JSON — try to extract JSON from within the text
        m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        # re.DOTALL makes . match newlines too
        # r"\{[^{}]+\}" matches the first { ... } block without nested braces
        if m:
            try:
                return json.loads(m.group(0))  # parse the extracted JSON fragment
            except json.JSONDecodeError:
                pass  # even the extracted fragment isn't valid JSON
    return None  # give up and return None


def make_judge(provider: Callable[..., dict[str, Any] | None] | None) -> JudgeProvider | None:
    # factory function that returns a judge provider
    # allows injecting a custom judge function (useful for testing)
    if provider is not None:
        return provider  # use the custom judge function passed in
    return default_groq_judge  # use the default Groq-based judge
