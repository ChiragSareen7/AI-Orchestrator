"""Requirement-driven self-correction loop."""

from app.services.self_correction.config import (
    JUDGE_PASS_SCORE_THRESHOLD,
    MAX_ATTEMPTS,
    REQUIREMENTS_PROFILE_REQUIRED,
)
from app.services.self_correction.judge import judge_response
from app.services.self_correction.loop import run_self_correction_loop
from app.services.self_correction.logger import ensure_self_correction_logs, read_self_correction_logs
from app.services.self_correction.prompt_enhancer import enhance_prompt
from app.services.self_correction.requirements import (
    ensure_profiles_file,
    get_requirements_profile,
    list_requirements_profiles,
    save_requirements_profile,
)

__all__ = [
    "JUDGE_PASS_SCORE_THRESHOLD",
    "MAX_ATTEMPTS",
    "REQUIREMENTS_PROFILE_REQUIRED",
    "ensure_profiles_file",
    "ensure_self_correction_logs",
    "enhance_prompt",
    "get_requirements_profile",
    "judge_response",
    "list_requirements_profiles",
    "read_self_correction_logs",
    "run_self_correction_loop",
    "save_requirements_profile",
]
