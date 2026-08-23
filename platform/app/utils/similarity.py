from __future__ import annotations  # allows modern type hint syntax on older Python

import itertools  # provides functions for working with iterators and combinations
import re         # regular expressions for text tokenization


def _tokens(text: str) -> set[str]:
    # converts a text string into a set of unique word tokens (lowercase)
    return set(re.findall(r"[a-zA-Z0-9]+", (text or "").lower()))
    # (text or "") handles None by defaulting to empty string
    # .lower() makes everything lowercase so "Python" and "python" are the same token
    # re.findall(r"[a-zA-Z0-9]+", ...) extracts sequences of letters and digits
    #   → removes punctuation: "hello, world!" → ["hello", "world"]
    # set() removes duplicates: ["the", "cat", "the"] → {"the", "cat"}
    # result: {"hello", "world"}


def jaccard_similarity(a: str, b: str) -> float:
    # Jaccard similarity = |intersection| / |union|
    # measures how much overlap two texts have in their vocabulary
    # range: 0.0 (completely different words) to 1.0 (identical word sets)

    ta, tb = _tokens(a), _tokens(b)  # tokenize both texts into sets
    if not ta or not tb:
        return 0.0  # if either text is empty, similarity is 0 (nothing to compare)

    return len(ta & tb) / max(1, len(ta | tb))
    # ta & tb = set INTERSECTION: words that appear in BOTH texts
    # ta | tb = set UNION: all words that appear in EITHER text
    # len(intersection) / len(union) = fraction of shared vocabulary
    # max(1, ...) prevents division by zero (though unlikely since we checked for empty above)
    #
    # Example:
    #   a = "cat sat on mat", tokens = {"cat", "sat", "on", "mat"}
    #   b = "cat ran on roof", tokens = {"cat", "ran", "on", "roof"}
    #   intersection = {"cat", "on"} → size 2
    #   union = {"cat", "sat", "on", "mat", "ran", "roof"} → size 6
    #   Jaccard = 2/6 = 0.333


def keyword_overlap(query: str, response: str) -> float:
    # measures: what FRACTION of the query's keywords appear in the response?
    # range: 0.0 (none of the query words are in the response) to 1.0 (all query words found)
    # this is specifically designed for evaluating relevance: does the response address the query?

    q, r = _tokens(query), _tokens(response)  # tokenize both
    if not q:
        return 0.0  # if query has no words, return 0 (nothing to check coverage for)

    return len(q & r) / len(q)
    # q & r = query words that ALSO appear in response (intersection)
    # len(q & r) / len(q) = fraction of query keywords covered by response
    #
    # Example:
    #   query = "boiling point benzene", q = {"boiling", "point", "benzene"}
    #   response = "benzene boils at 80.1°C", r = {"benzene", "boils", "at"}
    #   intersection = {"benzene"} → size 1
    #   keyword_overlap = 1/3 = 0.333  (only "benzene" from query appears in response)
    #
    # Note: "boiling" and "boils" are DIFFERENT tokens, so they don't match
    # This limitation is why semantic evaluation (embeddings) is more powerful


def average_pairwise_similarity(texts: list[str]) -> float:
    # measures how similar a model is to ITSELF across different prompt variants
    # called "behavior stability" — a stable model gives similar answers to similar prompts
    # range: 0.0 (totally different answers) to 1.0 (identical answers)

    if len(texts) < 2:
        return 1.0
        # need at least 2 texts to compare; if only 1 text, it's trivially "stable" (perfect score)

    sims = [jaccard_similarity(a, b) for a, b in itertools.combinations(texts, 2)]
    # itertools.combinations(texts, 2) generates ALL unique pairs:
    #   e.g. [text1, text2, text3] → [(text1,text2), (text1,text3), (text2,text3)]
    # compute Jaccard similarity for each pair
    # [jaccard(a,b) for a,b in pairs] = list of similarity scores

    return sum(sims) / len(sims) if sims else 1.0
    # compute the average similarity across all pairs
    # e.g. 3 texts → 3 pairs → average of 3 similarity scores
    # high average = model gives consistent (stable) answers regardless of prompt wording
