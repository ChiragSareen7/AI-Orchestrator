from __future__ import annotations  # allows modern type hint syntax on older Python


def estimate_tokens(text: str) -> int:
    # estimates the number of tokens in a text WITHOUT actually running a tokenizer
    # why? — running a real tokenizer (like GPT's tiktoken) is slow and requires a large library
    # this is a FAST APPROXIMATION used for cost estimation and logging

    # rough heuristic: ~4 chars per token for English text
    # where does this come from?
    #   - typical English word is ~5 characters
    #   - tokens are roughly word-length (but punctuation, common words are shorter)
    #   - in practice, 1 token ≈ 3-5 characters for typical English prose
    #   - 4 chars/token is a commonly used estimate in the LLM industry
    # examples:
    #   "Hello world" → 11 chars / 4 = 2 tokens (actual: 2 tokens ✓)
    #   "What is the boiling point of benzene?" → 37 chars / 4 = 9 tokens (actual: ~8 tokens ✓)

    return max(1, len(text) // 4)
    # len(text) = number of characters
    # // 4 = integer division by 4 (no decimals)
    # max(1, ...) ensures we always return at least 1 token (never 0, even for empty strings)
