from __future__ import annotations  # allows modern type hint syntax on older Python

import re  # re = regular expressions — used to find and extract patterns in text


# ── Domain Keywords ───────────────────────────────────────────────────────────
DOMAIN_KEYWORDS = {
    # a dictionary mapping each domain name → set of keywords that belong to that domain
    "chemistry": {"compound", "molecule", "organic", "benzene", "ethanol", "acetone",
                  "boiling", "melting", "toxicity", "solubility", "ph"},
    # if the query contains any of these words → likely a chemistry question → use organic_model

    "python": {"python", "list", "tuple", "dict", "function", "class", "loop",
               "decorator", "pandas", "numpy"},
    # if the query contains any of these words → likely a Python programming question → use python_model

    "gita": {"gita", "krishna", "arjuna", "dharma", "karma", "chapter", "verse", "bhagavad"},
    # if the query contains any of these words → likely a Bhagavad Gita question → use gita_model
}
# NOTE: if no domain keywords match, the domain defaults to "general" → use groq_model


def analyze_query(query: str) -> dict:
    # takes the raw user question and figures out: what domain, how complex, what type of question

    q = query.lower().strip()
    # .lower() converts to lowercase so "Python" matches "python"
    # .strip() removes leading/trailing spaces

    words = set(re.findall(r"[a-zA-Z0-9]+", q))
    # re.findall with pattern "[a-zA-Z0-9]+" extracts all words (letters and numbers, ignoring punctuation)
    # e.g. "What is benzene?" → {"what", "is", "benzene"}
    # set() removes duplicates so each word is counted only once

    # ── Domain Detection ─────────────────────────────────────────────────────
    domain_scores = {d: len(words & kws) for d, kws in DOMAIN_KEYWORDS.items()}
    # for each domain, count how many of the query's words appear in that domain's keyword set
    # 'words & kws' = set intersection (words that appear in BOTH sets)
    # e.g. query has "benzene" → chemistry score = 1, python score = 0, gita score = 0
    # result: {"chemistry": 1, "python": 0, "gita": 0}

    best_domain = max(domain_scores, key=domain_scores.get)
    # find the domain with the highest score
    # max(dict, key=dict.get) finds the key whose value is the largest

    domain = best_domain if domain_scores[best_domain] > 0 else "general"
    # if the best domain has score > 0 → use it
    # if the best domain has score 0 → no keywords matched → domain is "general" (use Groq)

    # ── Complexity Detection ──────────────────────────────────────────────────
    length = len(words)  # number of unique words in the query
    if length <= 8:
        complexity = "low"     # short query → simple question
    elif length <= 18:
        complexity = "medium"  # medium-length query → moderate complexity
    else:
        complexity = "high"    # long query → complex question

    # ── Intent Detection ─────────────────────────────────────────────────────
    intent = "explanation" if any(k in words for k in {"why", "how", "explain"}) else "fact_lookup"
    # if the query contains "why", "how", or "explain" → user wants an explanation
    # otherwise → user wants a specific fact (e.g. "What is the boiling point of benzene?")

    return {
        "domain": domain,             # "chemistry", "python", "gita", or "general"
        "complexity": complexity,     # "low", "medium", or "high"
        "intent": intent,             # "explanation" or "fact_lookup"
        "domain_scores": domain_scores,  # raw scores for each domain (useful for debugging)
    }
