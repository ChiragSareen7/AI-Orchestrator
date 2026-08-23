from __future__ import annotations  # allows modern type hint syntax on older Python

import re  # for splitting text using regular expressions
from typing import Sequence  # type hint for any ordered sequence (list, tuple, etc.)

from app.evaluation.embeddings import embed_texts, cosine_similarity_vec
# embed_texts: converts text to embedding vectors
# cosine_similarity_vec: measures similarity between two vectors


_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
# compile the regex pattern ONCE (more efficient than compiling on every call)
# this pattern splits text at:
#   (?<=[.!?])\s+ = whitespace that comes AFTER a period, exclamation mark, or question mark
#                   (?<=...) is a "lookbehind" — matches a position, not the character itself
#                   \s+ = one or more whitespace characters (space, tab, newline)
#   | = OR
#   \n+ = one or more newline characters (paragraph breaks)
# result: splits text into sentences


def split_sentences(text: str, min_len: int = 12) -> list[str]:
    # splits a response text into individual sentences
    # short fragments are filtered out (they're usually not meaningful standalone)

    text = (text or "").strip()  # handle None, then remove surrounding whitespace
    if not text:
        return []  # empty text → no sentences

    parts = [p.strip() for p in _SPLIT_RE.split(text) if p and len(p.strip()) >= min_len]
    # _SPLIT_RE.split(text) = split text at sentence boundaries
    # [p.strip() for p in ...] = clean whitespace from each part
    # if p = skip empty strings
    # len(p.strip()) >= min_len = skip very short fragments (< 12 chars by default)
    #   e.g. "Ok." or "Yes." are filtered out — too short to be grounded independently

    return parts if parts else [text]
    # if we got at least one part → return the parts list
    # if splitting produced nothing (no sentence boundaries found) → return the whole text as one item


def chunk_context(context: str, max_chars: int = 480) -> list[str]:
    # splits the context (retrieved document text) into manageable chunks
    # this is needed because cosine similarity works best on shorter passages

    ctx = (context or "").strip()  # handle None, remove whitespace
    if not ctx:
        return []  # no context → no chunks

    blocks = [b.strip() for b in re.split(r"\n\s*\n", ctx) if b.strip()]
    # split on double newlines (paragraph breaks): "\n\s*\n" matches blank lines
    # this separates paragraphs first

    chunks: list[str] = []
    for b in blocks:
        if len(b) <= max_chars:
            chunks.append(b)  # paragraph fits in one chunk → keep as-is
        else:
            # paragraph is too long → split into fixed-size windows
            for i in range(0, len(b), max_chars):
                chunks.append(b[i : i + max_chars])
            # b[i : i+480] = slice 480 characters starting at position i
            # range(0, len(b), 480) = 0, 480, 960, ... (step of 480)
            # this creates non-overlapping windows through the long paragraph

    return [c for c in chunks if c]
    # filter out any empty chunks (shouldn't happen, but safety check)


def groundedness_from_chunks(
    sentences: Sequence[str],        # response sentences to check
    context_chunks: Sequence[str],   # context paragraphs to check against
    embedding_model: str,            # which embedding model to use
    min_sim: float,                  # minimum similarity to consider a sentence "grounded"
) -> tuple[float, list[str]]:
    # determines how much of the response is grounded (supported) by the context
    # returns: (groundedness_score, list_of_unsupported_sentences)

    if not sentences:
        return 1.0, []
    # no sentences to check → perfect groundedness (nothing to be wrong about)

    if not context_chunks:
        return 0.0, list(sentences)
    # no context provided → can't verify anything → all sentences are "unsupported"

    # ── Embed everything at once for efficiency ───────────────────────────────
    sent_embs = embed_texts(list(sentences), embedding_model)
    # shape: (n_sentences, 384) — one embedding vector per response sentence

    chunk_embs = embed_texts(list(context_chunks), embedding_model)
    # shape: (n_chunks, 384) — one embedding vector per context chunk

    # ── Check each sentence against all context chunks ────────────────────────
    unsupported: list[str] = []  # will collect sentences that can't be verified
    supported = 0  # count of sentences that ARE grounded

    for i, _ in enumerate(sentences):
        # for each response sentence, find the BEST matching context chunk
        best = 0.0  # best cosine similarity found so far
        for j in range(len(context_chunks)):
            sim = cosine_similarity_vec(sent_embs[i], chunk_embs[j])
            # how similar is sentence i to context chunk j?
            if sim > best:
                best = sim  # update if this chunk is a better match

        if best >= min_sim:
            supported += 1
            # this sentence matches a context chunk closely enough → it's grounded
        else:
            unsupported.append(sentences[i])
            # no context chunk matched this sentence closely enough → it might be hallucinated

    g = supported / len(sentences)
    # groundedness = fraction of sentences that are supported
    # e.g. 8 out of 10 sentences grounded → groundedness = 0.8

    return g, unsupported
    # g = groundedness score (0.0 to 1.0, higher is better)
    # unsupported = list of sentence strings that couldn't be verified in context
