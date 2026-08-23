# This file marks the 'evaluation' folder as a Python package
# Without this file, Python wouldn't recognize 'evaluation' as an importable package
# 'from app.evaluation.pipeline import run_semantic_evaluation' only works because this file exists

# The evaluation package contains all semantic quality scoring modules:
#   config.py         — EvaluationThresholds (all configurable via env vars)
#   embeddings.py     — sentence embedding + cosine similarity (via sentence-transformers)
#   grounding.py      — checks if response sentences are backed by context
#   nli.py            — Natural Language Inference: detects contradictions
#   llm_judge.py      — uses an LLM to judge response quality
#   pipeline.py       — orchestrates all the above into one run_semantic_evaluation() call
#   logging_hooks.py  — event hook system for monitoring evaluation runs

# Public API exposed by this package:
from app.evaluation.pipeline import run_semantic_evaluation  # the main function to call
from app.evaluation.config import EvaluationThresholds       # the settings dataclass

__all__ = ["run_semantic_evaluation", "EvaluationThresholds"]
# __all__ defines what gets exported when someone does 'from app.evaluation import *'
# this is a best practice: explicitly say what is public API vs internal
