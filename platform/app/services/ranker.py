from __future__ import annotations  # allows modern type hint syntax on older Python

import os  # to read the USE_SEMANTIC_RANKING environment variable


def _use_semantic_ranking() -> bool:
    # checks whether to rank by semantic scores or fall back to lexical scores
    return os.getenv("USE_SEMANTIC_RANKING", "true").lower() in ("1", "true", "yes")
    # reads USE_SEMANTIC_RANKING from .env; if not set, defaults to "true"
    # checks if the value is one of the "truthy" strings: "1", "true", "yes"
    # returns True or False


def _semantic_sort_key(r: dict) -> tuple:
    # creates a tuple used to sort responses when semantic scores are available
    # Python sorts tuples element by element: first element compared first, then second if tied, etc.
    m = r.get("metrics") or {}  # get the metrics dict; if None, use empty dict
    s = m.get("semantic")       # get the semantic sub-dict (None if semantic eval was disabled)

    if s and isinstance(s, dict) and "accuracy" in s:
        # semantic scores ARE available — use them (they're more meaningful than lexical)
        return (
            -float(s.get("accuracy", 0.0)),     # NEGATED: highest accuracy sorts FIRST (ascending sort)
            float(s.get("hallucination", 1.0)), # lowest hallucination sorts first (0=good, 1=bad)
            float(m.get("latency", 0.0)),        # fastest response sorts first
            -float(s.get("confidence", 0.0)),   # NEGATED: highest confidence sorts first
        )
    # fallback: semantic key missing → use lexical scores instead
    return (
        -float(m.get("accuracyScore", 0.0)),       # negate for descending sort
        float(m.get("hallucinationScore", 1.0)),   # ascending sort
        float(m.get("latency", 0.0)),              # ascending sort
        -float(m.get("confidenceScore", 0.0)),     # negate for descending sort
    )


def _lexical_sort_key(r: dict) -> tuple:
    # creates a sort key using ONLY lexical (word-based) scores
    # used when USE_SEMANTIC_RANKING=false in .env
    m = r.get("metrics") or {}  # get the metrics dict safely
    return (
        -m["accuracyScore"],       # NEGATED so highest accuracy sorts first
        m["hallucinationScore"],   # lowest hallucination first (no negation needed)
        m["latency"],              # fastest first
        -m["confidenceScore"],     # NEGATED so highest confidence sorts first
    )
    # example: accuracy=0.9 → sort key starts with -0.9
    #          accuracy=0.7 → sort key starts with -0.7
    # -0.9 < -0.7, so the 0.9 accuracy response sorts BEFORE the 0.7 one ✓


def rank_responses(responses: list[dict]) -> list[dict]:
    # takes all model+prompt responses and returns them sorted from BEST to WORST
    key = _semantic_sort_key if _use_semantic_ranking() else _lexical_sort_key
    # pick the sorting function based on configuration
    # if semantic scores available and enabled → use semantic sort key
    # otherwise → use lexical sort key

    return sorted(responses, key=key)
    # sorted() returns a NEW list in sorted order (doesn't modify original)
    # Python's sort is stable: if two responses have exactly equal sort keys,
    # they maintain their original relative order
