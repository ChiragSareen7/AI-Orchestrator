from __future__ import annotations  # allows modern type hint syntax on older Python

from app.services.learning import get_domain_bias
# get_domain_bias reads model_performance.json and returns whichever model has historically
# performed best for a given domain — used to influence the routing order


# ── Domain → Primary Model Mapping ──────────────────────────────────────────
DOMAIN_PRIMARY = {
    "chemistry": "organic_model",  # chemistry questions → try organic_model first
    "python":    "python_model",   # Python questions    → try python_model first
    "gita":      "gita_model",     # Gita questions      → try gita_model first
    "general":   "groq_model",     # everything else     → try Groq (Llama) first
}
# these are the "best default" choices before any learning has happened

ALL_MODELS = ["organic_model", "python_model", "gita_model", "groq_model"]
# the full list of all 4 available models — the orchestrator will try ALL of them


def route_models(domain: str) -> dict:
    # takes the detected domain and returns an ordered list of which models to try

    primary = DOMAIN_PRIMARY.get(domain, "groq_model")
    # look up the best default model for this domain
    # .get(domain, "groq_model") means: if domain not found in dict, use "groq_model" as fallback

    bias = get_domain_bias(domain)
    # reads model_performance.json to find the HISTORICALLY BEST model for this domain
    # e.g. if organic_model has been winning chemistry questions, bias = "organic_model"
    # returns None if there's no history yet for this domain

    # ── Build ranked list ─────────────────────────────────────────────────────
    ranked = [primary]
    # start with the primary (domain-specific) model at position 0

    if bias and bias in ALL_MODELS and bias not in ranked:
        ranked.append(bias)
        # if the historically best model is different from the primary, add it at position 1
        # this gives the learned winner an early chance to run
        # the check "bias not in ranked" avoids adding the same model twice

    for model in ALL_MODELS:
        if model not in ranked:
            ranked.append(model)
    # add all remaining models that haven't been added yet
    # this ensures ALL 4 models are always in the list (just in different positions)

    return {
        "primary": primary,   # the single best default model for this domain
        "ranked": ranked,     # all 4 models sorted: [primary, bias (if different), ...rest]
        "fallbacks": [m for m in ranked if m != primary],
        # fallbacks = all models EXCEPT the primary
        # used if the primary model fails or produces a bad response
    }
