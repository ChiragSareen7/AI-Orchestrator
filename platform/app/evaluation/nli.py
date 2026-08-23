from __future__ import annotations  # allows modern type hint syntax on older Python

from typing import Literal  # Literal restricts a type to specific allowed values

import numpy as np  # numpy for numerical operations (softmax, argmax, etc.)


# ── Singleton Cross-Encoder Cache ─────────────────────────────────────────────
_ce = None        # cached CrossEncoder model (None = not loaded yet)
_ce_name: str | None = None  # name of the currently loaded model

Label = Literal["entailment", "neutral", "contradiction"]
# Label is a type that can ONLY be one of these three strings
# NLI (Natural Language Inference) has exactly 3 possible relationships:
#   - "entailment": hypothesis is supported by / follows from the premise
#   - "neutral": hypothesis neither follows from nor contradicts premise
#   - "contradiction": hypothesis contradicts / conflicts with the premise


def _get_cross_encoder(model_name: str):
    # lazy-loads the NLI cross-encoder model (singleton pattern)
    global _ce, _ce_name  # modifying module-level variables

    if _ce is None or _ce_name != model_name:
        # only load if not yet loaded or a different model is requested
        from sentence_transformers import CrossEncoder
        # CrossEncoder is different from SentenceTransformer:
        #   - SentenceTransformer encodes texts INDEPENDENTLY (bi-encoder)
        #   - CrossEncoder takes TWO texts together and classifies their relationship
        #   - CrossEncoder is more accurate but slower (can't pre-compute embeddings)

        _ce = CrossEncoder(
            model_name,  # e.g. "cross-encoder/nli-MiniLM-L6-H-6"
            device="cpu",      # always use CPU (no GPU needed for this small model)
            max_length=256,    # truncate texts longer than 256 tokens
        )
        _ce_name = model_name  # remember which model is loaded
    return _ce  # return the cached model


def _normalize_label(s: str) -> Label:
    # converts a raw model output string to one of our 3 known label values
    s = (s or "").lower()  # convert to lowercase for case-insensitive comparison
    if "entail" in s:
        return "entailment"
    if "contradict" in s:
        return "contradiction"
    return "neutral"  # default to neutral if neither entailment nor contradiction


def entailment_label(premise: str, hypothesis: str, model_name: str) -> tuple[Label, float]:
    # given a premise (context) and hypothesis (response sentence),
    # classifies their relationship and returns (label, confidence)

    premise, hypothesis = premise.strip(), hypothesis.strip()
    if not premise or not hypothesis:
        return "neutral", 1 / 3
        # if either text is empty, return "neutral" with low confidence (1/3 = random chance)

    model = _get_cross_encoder(model_name)  # get or load the NLI model

    raw = model.predict([(premise, hypothesis)], show_progress_bar=False)
    # predict takes a LIST of (premise, hypothesis) pairs
    # returns raw logit scores (one per class: [contradiction, entailment, neutral])
    # raw shape: (1, 3) or (3,) depending on batch size

    logits = np.asarray(raw[0] if isinstance(raw, (list, tuple)) else raw).flatten()
    # raw[0] gets the first (and only) result from the batch
    # .flatten() ensures 1D array: [logit_class0, logit_class1, logit_class2]

    exp = np.exp(logits - np.max(logits))
    # softmax computation (converts raw logits to probabilities)
    # subtract max(logits) first for numerical stability (prevents overflow in exp())
    # np.exp(x) = e^x for each element

    probs = exp / (np.sum(exp) + 1e-12)
    # complete softmax: divide by sum to get probabilities that sum to 1.0
    # + 1e-12 avoids division by zero

    idx = int(np.argmax(probs))  # index of the highest probability class
    conf = float(probs[idx])     # confidence = probability of the winning class

    # ── Map index to label ─────────────────────────────────────────────────────
    cfg = getattr(model.model, "config", None)  # try to get the model's HuggingFace config
    id2label = getattr(cfg, "id2label", None) if cfg is not None else None
    # id2label is a dict like {0: "contradiction", 1: "entailment", 2: "neutral"}

    if isinstance(id2label, dict) and idx in id2label:
        lab = _normalize_label(str(id2label[idx]))
        # use the model's own label mapping (most reliable)
    else:
        order = ["contradiction", "entailment", "neutral"]
        lab = order[idx] if idx < len(order) else "neutral"
        # fallback: assume standard NLI label order if model config not available

    return lab, conf  # e.g. ("contradiction", 0.87)


def entailment_mean_score(
    context: str,           # the premise = retrieved context text
    sentences: list[str],   # the hypotheses = response sentences to check
    model_name: str,        # which NLI model to use
) -> tuple[float, list[Label]]:
    # checks each response sentence against the context using NLI
    # returns: (mean_entailment_score, list_of_labels_per_sentence)

    if not (context or "").strip() or not sentences:
        return 0.5, []
        # if no context or no sentences, return 0.5 (neutral/uncertain) with empty label list

    premise = context[:4000]
    # truncate context to 4000 characters to stay within model's max_length
    # NLI models can't process very long texts; 4000 chars ≈ 800 words ≈ enough context

    scores: list[float] = []  # numeric score for each sentence
    labels: list[Label] = []  # label for each sentence

    for sent in sentences:
        lab, p = entailment_label(premise, sent, model_name)
        # check if this sentence is entailed, neutral, or contradicted by the context
        labels.append(lab)  # save the label

        if lab == "entailment":
            scores.append(1.0)     # fully supported → score 1.0 (best)
        elif lab == "contradiction":
            scores.append(0.0)     # contradicts context → score 0.0 (worst = hallucination)
        else:
            scores.append(0.5)     # neutral = uncertain → score 0.5 (middle)

    mean_s = float(sum(scores) / len(scores)) if scores else 0.5
    # compute the average entailment score across all sentences
    # high value (close to 1.0) = most sentences are entailed by context (good)
    # low value (close to 0.0) = many sentences contradict context (hallucination)

    return mean_s, labels
    # mean_s: 0.0-1.0 entailment score
    # labels: ["entailment", "neutral", "contradiction", ...] — one per sentence
