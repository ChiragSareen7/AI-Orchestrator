from __future__ import annotations  # allows modern type hint syntax on older Python

import os  # for reading EVAL_EMBEDDING_DEVICE env var
from typing import Sequence  # Sequence = any ordered collection (list, tuple, etc.)

import numpy as np  # numpy = fast numerical computing library; arrays, vector math


# ── Singleton Model Cache ─────────────────────────────────────────────────────
_model = None        # stores the loaded SentenceTransformer model (None = not loaded yet)
_model_name: str | None = None  # stores which model name is currently loaded


def _get_model(name: str):
    # lazy-loads the embedding model: only downloads/loads it when first needed
    # also implements a "singleton" pattern: load once, reuse forever
    global _model, _model_name  # we're modifying module-level variables

    if _model is None or _model_name != name:
        # only load if: (1) nothing is loaded yet, OR (2) a different model is requested
        from sentence_transformers import SentenceTransformer
        # import here (lazy) to avoid loading the heavy library at startup

        _model = SentenceTransformer(
            name,   # e.g. "sentence-transformers/all-MiniLM-L6-v2"
            device=os.getenv("EVAL_EMBEDDING_DEVICE", "cpu")
            # which hardware to run on: "cpu" (default, always available) or "cuda" (GPU, faster)
        )
        _model_name = name  # remember which model is loaded
    return _model  # return the cached model (avoids reloading every call)


def embed_texts(texts: Sequence[str], model_name: str) -> np.ndarray:
    # converts a list of text strings into a 2D numpy array of embedding vectors
    # shape: (number_of_texts, embedding_dimensions)
    # e.g. 2 texts → shape (2, 384) for all-MiniLM-L6-v2

    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
        # if no texts provided, return an empty array (avoid divide-by-zero later)

    m = _get_model(model_name)  # get (or load) the embedding model

    arr = m.encode(
        list(texts),           # convert Sequence to list (encode requires a list)
        convert_to_numpy=True, # return numpy array instead of PyTorch tensor
        normalize_embeddings=True,
        # normalize_embeddings=True divides each vector by its magnitude (L2 norm)
        # this makes all vectors unit length (magnitude = 1.0)
        # benefit: cosine similarity = dot product when vectors are unit length
        #          this is FASTER and avoids division in cosine computation
        show_progress_bar=False,  # don't print a progress bar (cleaner output)
    )

    return np.asarray(arr, dtype=np.float32)
    # ensure the output is a float32 numpy array (consistent data type)


def cosine_similarity_vec(a: np.ndarray, b: np.ndarray) -> float:
    # computes cosine similarity between two embedding vectors
    # cosine similarity measures the ANGLE between two vectors:
    #   1.0 = identical direction (same meaning)
    #   0.0 = perpendicular (unrelated)
    #  -1.0 = opposite direction (opposite meaning)

    if a.size == 0 or b.size == 0:
        return 0.0  # can't compute similarity for empty vectors

    a = np.asarray(a).flatten()  # ensure 1D array (flatten removes extra dimensions)
    b = np.asarray(b).flatten()  # same for b

    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    # denominator = product of magnitudes (lengths) of both vectors
    # np.linalg.norm(v) = sqrt(sum of squares of all elements) = vector magnitude

    if denom < 1e-12:
        return 0.0
        # if either vector is nearly zero-length, return 0 to avoid division by zero
        # 1e-12 is a very small number (0.000000000001) used as a safety threshold

    return float(np.dot(a, b) / denom)
    # cosine_similarity = dot_product / (magnitude_a * magnitude_b)
    # np.dot(a, b) = sum of element-wise products (a[0]*b[0] + a[1]*b[1] + ...)
    # NOTE: if embeddings are already normalized (unit vectors), denom = 1.0
    #       so this simplifies to just np.dot(a, b) — very fast


def relevance_query_response(query: str, response: str, model_name: str) -> float:
    # measures how semantically relevant the response is to the query
    # uses embedding cosine similarity — captures meaning, not just keywords
    q, r = query.strip(), response.strip()  # remove leading/trailing whitespace

    if not q or not r:
        return 0.0  # can't compute relevance if either text is empty

    emb = embed_texts([q, r], model_name)
    # embed both texts together in one call (more efficient than two separate calls)
    # returns shape (2, 384): emb[0] = query embedding, emb[1] = response embedding

    return cosine_similarity_vec(emb[0], emb[1])
    # measure the angle between the query vector and response vector
    # high similarity → response is semantically relevant to the query


def semantic_similarity_pair(a: str, b: str, model_name: str) -> float:
    # measures semantic similarity between any two text strings
    # used for computing accuracy when a ground_truth reference answer is provided
    a, b = a.strip(), b.strip()  # clean up whitespace

    if not a or not b:
        return 0.0  # can't compute if either is empty

    emb = embed_texts([a, b], model_name)  # embed both texts
    return cosine_similarity_vec(emb[0], emb[1])  # cosine similarity between them
    # e.g. compare model's response to the known correct answer
    # high similarity → model's answer is close to the ground truth
