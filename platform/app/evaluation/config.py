from __future__ import annotations  # allows modern type hint syntax on older Python

import os  # for reading environment variables
from dataclasses import dataclass, field  # dataclass decorator auto-generates __init__, __repr__ etc.


def _f(name: str, default: float) -> float:
    # helper: read a FLOAT value from an environment variable
    # if the variable is missing or invalid, use the default value
    try:
        return float(os.getenv(name, str(default)))
        # os.getenv(name, str(default)) reads the env var, defaulting to str(default) if missing
        # float(...) converts the string to a decimal number
    except ValueError:
        return default  # if the conversion fails (e.g. someone set it to "abc"), use default


def _b(name: str, default: bool) -> bool:
    # helper: read a BOOLEAN value from an environment variable
    return os.getenv(name, str(default)).lower() in ("1", "true", "yes")
    # env vars are always strings, so we check for common "truthy" string values
    # e.g. EVAL_ENABLE_NLI=true → True; EVAL_ENABLE_NLI=false → False


@dataclass  # @dataclass auto-generates __init__ so we don't have to write it manually
class EvaluationThresholds:
    # this class holds ALL configuration for the semantic evaluation system
    # every field reads its value from an environment variable when the object is created
    # if the env var is not set, the default value is used
    # this design means you can tune the system without changing any code

    grounding_min_sim: float = field(
        default_factory=lambda: _f("EVAL_GROUNDING_MIN_SIM", 0.75)
    )
    # minimum cosine similarity (0.0-1.0) for a response sentence to be considered "grounded"
    # 0.75 = a sentence must match context at 75% similarity or more to be accepted
    # lower this value to be more permissive; raise it to be stricter

    weight_groundedness: float = field(
        default_factory=lambda: _f("EVAL_W_GROUNDEDNESS", 0.35)
    )
    # how much weight to give "groundedness" when computing the confidence score
    # 0.35 = groundedness counts for 35% of confidence

    weight_relevance: float = field(
        default_factory=lambda: _f("EVAL_W_RELEVANCE", 0.35)
    )
    # how much weight to give "relevance" when computing the confidence score
    # 0.35 = relevance counts for 35% of confidence

    weight_entailment: float = field(
        default_factory=lambda: _f("EVAL_W_ENTAILMENT", 0.30)
    )
    # how much weight to give NLI "entailment" when computing the confidence score
    # 0.30 = NLI entailment counts for 30% of confidence (only used when NLI is enabled)
    # the three weights together: 0.35 + 0.35 + 0.30 = 1.0 when all are used

    enable_nli: bool = field(
        default_factory=lambda: _b("EVAL_ENABLE_NLI", False)
    )
    # whether to run Natural Language Inference (NLI) evaluation
    # NLI checks if response sentences CONTRADICT the context (not just similarity)
    # default False because it requires downloading a cross-encoder model and is slower

    enable_llm_judge: bool = field(
        default_factory=lambda: _b("EVAL_ENABLE_LLM_JUDGE", False)
    )
    # whether to call an LLM (Groq) to judge the response quality
    # the LLM judge gives a score for correctness, relevance, and hallucination
    # default False because it costs API credits and adds latency

    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EVAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    # which sentence embedding model to use for semantic similarity
    # "all-MiniLM-L6-v2" is a good balance of speed and quality:
    #   - 384 dimensions (small → fast)
    #   - trained on 1B+ sentence pairs
    #   - downloads automatically from HuggingFace on first use

    nli_model: str = field(
        default_factory=lambda: os.getenv(
            "EVAL_NLI_MODEL", "cross-encoder/nli-MiniLM-L6-H-6"
        )
    )
    # which NLI cross-encoder model to use for entailment checking
    # "nli-MiniLM-L6-H-6" is a small but effective NLI model
    # only used when enable_nli=True

    # NOTE: field(default_factory=lambda: ...) is used instead of plain defaults
    # because dataclass default values are evaluated ONCE at class definition time
    # but os.getenv reads env vars at RUNTIME (when the object is created)
    # lambda: makes a callable so it's evaluated fresh every time an instance is created
