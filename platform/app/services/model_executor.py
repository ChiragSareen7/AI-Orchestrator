from __future__ import annotations  # allows modern type hint syntax on older Python

import os    # used to read environment variables
import time  # used to measure how long the model call takes
from typing import Any  # Any = a type hint meaning "could be anything"

import httpx  # httpx is a modern HTTP client library — used to call external services

from app.services.tracing import traceable_if_enabled  # optional LangSmith tracing decorator
from app.utils.token_estimator import estimate_tokens  # estimates token count from text length

# ── Configuration (all read from .env file) ───────────────────────────────────
MODELS_API_BASE_URL = os.getenv("MODELS_API_BASE_URL", "http://127.0.0.1:8010")
# the URL of the local fine-tuned models server (models_api/)
# default is localhost:8010 — change in .env if running on a different machine

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# your Groq API key from console.groq.com — REQUIRED for groq_model calls

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
# which Llama model to use on Groq — "llama-3.1-8b-instant" is fast and free

GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
# Groq's API follows the OpenAI API format, so we can use it like an OpenAI client

REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "25"))
# how many seconds to wait for a model response before giving up
# float() converts the string from env var to a decimal number

# ── Model → Endpoint Mapping ─────────────────────────────────────────────────
MODEL_ENDPOINT_MAP = {
    "organic_model": "/organic",  # chemistry model → POST http://127.0.0.1:8010/organic
    "python_model":  "/python",   # Python model    → POST http://127.0.0.1:8010/python
    "gita_model":    "/gita",     # Gita model      → POST http://127.0.0.1:8010/gita
}
# groq_model is NOT in this map because it uses a completely different API (Groq's cloud)


@traceable_if_enabled(run_type="chain", name="execute-model-call")
# this decorator optionally sends execution data to LangSmith for tracing
# if LANGCHAIN_TRACING_V2=false in .env, this decorator does nothing (no-op)
def execute_model(model_name: str, original_query: str, prompt_text: str) -> dict[str, Any]:
    # calls a single model with a single prompt and measures the result

    started = time.perf_counter()  # record the start time in high-resolution seconds
    # perf_counter is more precise than time.time() — use it for measuring elapsed time

    error = None   # will store an error message if something goes wrong
    text = ""      # will store the model's response text

    try:
        if model_name == "groq_model":
            text = _execute_groq(prompt_text)  # call Groq's cloud API
        else:
            # local fine-tuned model — call our own models_api server
            endpoint = MODEL_ENDPOINT_MAP[model_name]  # e.g. "/organic"
            payload = {"query": prompt_text}  # the request body — models_api expects {"query": "..."}

            with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
                # 'with' ensures the HTTP client is properly closed after use (even if an error occurs)
                resp = client.post(f"{MODELS_API_BASE_URL}{endpoint}", json=payload)
                # sends POST request to e.g. "http://127.0.0.1:8010/organic"
                # json=payload automatically serializes the dict to JSON and sets Content-Type header

                resp.raise_for_status()
                # if the server returned an error status (4xx or 5xx), raise an exception
                # this jumps to the 'except' block below

                body = resp.json()  # parse the JSON response body into a Python dict
                text = body.get("answer", "")
                # extract the "answer" field from the response
                # .get("answer", "") returns empty string if "answer" key doesn't exist

    except Exception as exc:
        error = str(exc)
        # if ANYTHING went wrong (network error, timeout, bad response, etc.),
        # catch the exception and save its message as the error string
        # str(exc) converts the exception object to a human-readable string

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    # calculate how long the call took:
    # perf_counter() - started = elapsed seconds (float)
    # * 1000 = convert to milliseconds
    # round(..., 2) = round to 2 decimal places

    tokens = estimate_tokens(prompt_text + " " + text)
    # estimate how many tokens were used in total (prompt + response)
    # this is used to estimate the cost of the API call

    return {
        "model": model_name,       # e.g. "organic_model"
        "query": original_query,   # the original user question (not the full prompt)
        "response": text,          # the model's answer (empty string if error)
        "error": error,            # None if success, error message string if failed
        "latency": latency_ms,     # response time in milliseconds
        "tokenUsage": tokens,      # estimated token count
    }


def _execute_groq(prompt_text: str) -> str:
    # sends the prompt to Groq's cloud API and returns the response text
    # this is a private function (underscore prefix convention means "internal use only")

    if not GROQ_API_KEY:
        raise RuntimeError("Missing GROQ_API_KEY")
        # fail immediately with a clear error if the API key isn't set in .env

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        # "Bearer token" is a standard HTTP authentication format
        # the API key goes after "Bearer "
        "Content-Type": "application/json",
        # tells the server we're sending JSON data
    }

    payload = {
        "model": GROQ_MODEL,  # e.g. "llama-3.1-8b-instant"
        "messages": [
            {"role": "user", "content": prompt_text}
            # OpenAI-format messages: list of {role, content} dicts
            # "user" role means this is the human's message
        ],
        "temperature": 0.1,
        # temperature controls randomness: 0.0 = fully deterministic, 1.0 = very random
        # 0.1 is near-deterministic — almost the same output each time, but slightly flexible
    }

    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        resp = client.post(f"{GROQ_BASE_URL}/chat/completions", headers=headers, json=payload)
        # POST to https://api.groq.com/openai/v1/chat/completions
        # this is exactly the same format as OpenAI's API

        resp.raise_for_status()  # raise an error if response is 4xx or 5xx
        body = resp.json()       # parse JSON response

        return body["choices"][0]["message"]["content"].strip()
        # OpenAI/Groq response format:
        # {"choices": [{"message": {"content": "the answer here"}}]}
        # [0] = first choice (there's only one since we didn't request multiple)
        # ["message"]["content"] = the actual text the model generated
        # .strip() removes leading/trailing whitespace
