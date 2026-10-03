"""Configuration for the requirement-driven self-correction loop."""

from __future__ import annotations

import os

# ── Judge ─────────────────────────────────────────────────────────────────────
JUDGE_PASS_SCORE_THRESHOLD = float(os.getenv("SELF_CORRECTION_JUDGE_PASS_THRESHOLD", "0.75"))
JUDGE_MODEL = os.getenv("SELF_CORRECTION_JUDGE_MODEL", os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"))
JUDGE_TIMEOUT_SECONDS = float(os.getenv("SELF_CORRECTION_JUDGE_TIMEOUT", "45"))

# ── Requirements profile ──────────────────────────────────────────────────────
REQUIREMENTS_PROFILE_REQUIRED = os.getenv("REQUIREMENTS_PROFILE_REQUIRED", "false").lower() in {
    "1",
    "true",
    "yes",
}
DEFAULT_CLIENT_ID = os.getenv("SELF_CORRECTION_DEFAULT_CLIENT_ID", "default")

# ── Model escalation fallback ───────────────────────────────────────────────────
NEXT_BEST_FALLBACK_MODEL = os.getenv("SELF_CORRECTION_NEXT_BEST_FALLBACK", "groq_model")

# ── Attempt cap (hard limit — do not change without product review) ───────────
MAX_ATTEMPTS = 4
