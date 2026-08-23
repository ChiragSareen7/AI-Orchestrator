from __future__ import annotations  # allows modern type hint syntax on older Python


def generate_prompts(query: str) -> list[dict]:
    # takes the raw user question and wraps it in 3 different instruction styles
    # why? — different prompt styles elicit different response behaviors from the same model
    # by running all 3, we capture more variety and let the evaluation pick the best one

    return [
        {
            "version": "v1",
            "text": f"Answer clearly and concisely: {query}",
            # v1 = direct and short — best for simple factual questions
            # e.g. "Answer clearly and concisely: What is the boiling point of benzene?"
        },
        {
            "version": "v2",
            "text": f"Explain in detail with useful context: {query}",
            # v2 = detailed explanation — best for complex conceptual questions
            # e.g. "Explain in detail with useful context: How does Python's GIL work?"
        },
        {
            "version": "v3",
            "text": f"Answer strictly with facts only. If uncertain, say uncertain: {query}",
            # v3 = strict facts only, no guessing — most hallucination-resistant
            # the phrase "If uncertain, say uncertain" instructs the model to admit uncertainty
            # rather than confidently making up an answer
        },
    ]
    # result: a list of 3 dicts, each with a version label and the full prompt text
    # the orchestrator loops over these and tries each one with every model
