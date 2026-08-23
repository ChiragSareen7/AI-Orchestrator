"""Build improved prompts from judge failure reasons."""

from __future__ import annotations

from app.services.self_correction.requirements import profile_to_prompt_context


def enhance_prompt(
    original_query: str,
    requirements_profile: dict,
    failure_reasons: list[str],
) -> str:
    """
    Construct a refined prompt that explicitly addresses judge failure reasons.
    """
    requirements_text = profile_to_prompt_context(requirements_profile)
    reasons_block = "\n".join(f"  - {reason}" for reason in failure_reasons if reason.strip())

    return (
        "Revise your answer to the following question. Your previous answer did NOT meet "
        "the requirements. Fix ONLY what is listed under 'Required fixes'.\n\n"
        f"Requirements profile:\n{requirements_text}\n\n"
        f"Required fixes (address each specifically):\n{reasons_block}\n\n"
        f"Original question:\n{original_query}\n\n"
        "Provide an improved answer that satisfies the requirements and fixes above."
    )
