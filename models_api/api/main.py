from __future__ import annotations  # allows modern type hint syntax on older Python

from pathlib import Path         # for working with file paths
from typing import Dict, Tuple   # type hints for dict and tuple with specific types

import torch  # PyTorch: the deep learning framework our models are built on
from fastapi import FastAPI, HTTPException  # FastAPI = web framework; HTTPException = return error responses
from pydantic import BaseModel, Field  # Pydantic = data validation for request/response shapes
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# AutoModelForSeq2SeqLM = loads any seq2seq (encoder-decoder) model from HuggingFace
# AutoTokenizer = loads the corresponding tokenizer for that model
# "Auto" means it automatically detects the right model class from the config file


# ── Request / Response Models ─────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    # the user's question; ... means required; must be at least 1 character


class QueryResponse(BaseModel):
    answer: str   # the model's generated answer
    model: str    # which model endpoint answered (e.g. "python", "organic", "gita")


# ── Keyword Router ────────────────────────────────────────────────────────────
def _route_model(query: str) -> str:
    # given a query, picks which domain model to use based on keyword matching
    # used by the /ask endpoint (auto-routing)
    text = query.lower()  # lowercase for case-insensitive matching

    if any(k in text for k in {"gita", "krishna", "arjuna", "dharma", "karma", "verse", "chapter"}):
        return "gita"    # Bhagavad Gita keyword found → use gita_model
    if any(k in text for k in {"compound", "molecular", "boiling", "melting", "solubility", "toxicity", "benzene"}):
        return "organic" # chemistry keyword found → use organic_model
    return "python"      # default: if no other keywords match, assume Python question


# ── Model Loading ─────────────────────────────────────────────────────────────
def _load_model_bundle(model_dir: Path) -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    # loads a fine-tuned model and its tokenizer from a local directory
    if not model_dir.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_dir}")
        # give a clear error message if the model folder doesn't exist

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    # AutoTokenizer reads tokenizer.json and tokenizer_config.json from the folder
    # the tokenizer converts text to numbers (token IDs) and back

    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))
    # AutoModelForSeq2SeqLM reads config.json and model.safetensors from the folder
    # this loads the model architecture AND the trained weights

    model.eval()
    # set the model to evaluation mode (not training mode)
    # this disables dropout layers, which are only needed during training
    # in eval mode: same input ALWAYS produces same output (deterministic)

    return tokenizer, model  # return both as a tuple


# ── Prompt Formatting ─────────────────────────────────────────────────────────
def _prompt_for_model(model_key: str, query: str) -> str:
    # adds task-specific prefixes to the query before feeding to the model
    # why? — during training, each model was trained with these exact prefix formats
    # using the SAME format at inference ensures the model knows what task it's doing
    q = query.strip()  # remove leading/trailing whitespace
    if model_key == "python":
        return f"task: python_faq; question: {q}"
        # tells the model: "this is a Python FAQ task; answer the question"
    if model_key == "organic":
        return f"task: organic_props; question: {q}"
        # tells the model: "this is an organic chemistry properties task"
    if model_key == "gita":
        return f"task: gita_verse; question: {q}"
        # tells the model: "this is a Bhagavad Gita verse explanation task"
    return f"question: {q}"  # fallback for unknown model keys


# ── Text Generation ───────────────────────────────────────────────────────────
def _generate_answer(
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    query: str,
    model_key: str,
    max_input_length: int = 256,   # maximum tokens in the input (truncate if longer)
    max_output_length: int = 256,  # maximum tokens to generate in the output
) -> str:
    prompt = _prompt_for_model(model_key, query)
    # format the query with the correct task prefix

    encoded = tokenizer(
        prompt,
        return_tensors="pt",    # "pt" = return PyTorch tensors (not numpy or python lists)
        truncation=True,         # if prompt is too long, cut it to max_input_length
        max_length=max_input_length,
    )
    # encoded is a dict with:
    #   "input_ids": tensor of token ID numbers [[101, 2054, 2003, ...]]
    #   "attention_mask": tensor marking which tokens are real vs padding [[1, 1, 1, ...]]

    with torch.no_grad():
        # torch.no_grad() tells PyTorch: don't compute gradients (we're not training)
        # this saves memory and makes inference faster 
        output_ids = model.generate(
            **encoded,               # unpack encoded dict: input_ids + attention_mask
            max_new_tokens=max_output_length,  # generate at most 256 new tokens
            do_sample=False,         # DETERMINISTIC: always pick the highest-probability next token
                                     # "greedy decoding" — same input ALWAYS gives same output
            num_beams=1,             # beam search width = 1 → equivalent to greedy decoding
                                     # if num_beams > 1: keeps multiple candidate sequences (slower, sometimes better)
        )
    # output_ids is a tensor of generated token IDs: [[token1, token2, ...]]

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    # output_ids[0] gets the first (and only) generated sequence
    # tokenizer.decode() converts token IDs back to human-readable text
    # skip_special_tokens=True removes [PAD], [EOS], [SEP] tokens from the output
    # .strip() removes any leading/trailing whitespace


# ── App Setup & Model Registry ────────────────────────────────────────────────
app = FastAPI(title="Models QA API", version="1.0.0")
# create the FastAPI app with a descriptive title

ROOT = Path(__file__).resolve().parents[1]
# __file__ = this file: models_api/api/main.py
# .parents[0] = api/ folder
# .parents[1] = models_api/ folder ← ROOT points here

MODEL_PATHS = {
    "python":  ROOT / "python_model",   # models_api/python_model/
    "organic": ROOT / "organic_model",  # models_api/organic_model/
    "gita":    ROOT / "gita_model",     # models_api/gita_model/
}
# maps model key → folder path where the model weights and configs are stored

MODEL_REGISTRY: Dict[str, Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]] = {}
# stores loaded model bundles in memory
# Dict[str, Tuple[...]] = dictionary where keys are strings, values are (tokenizer, model) pairs
# empty at start; populated during startup event below


# ── Startup Event ─────────────────────────────────────────────────────────────
@app.on_event("startup")  # runs automatically when the server starts
def startup_event() -> None:
    for key, path in MODEL_PATHS.items():
        # loop through all 3 models
        try:
            MODEL_REGISTRY[key] = _load_model_bundle(path)
            # load the model and tokenizer from disk into memory
            # this takes 5-30 seconds per model (loading 294MB weights files)
        except Exception as exc:
            print(f"[startup] failed to load {key} model: {exc}")
            # if a model fails to load, print the error but keep going
            # the other models will still be available
            # requests for the failed model will return 503 (Service Unavailable)


# ── Request Handler ───────────────────────────────────────────────────────────
def _handle_request(model_key: str, request: QueryRequest) -> QueryResponse:
    # common logic for all model endpoints

    bundle = MODEL_REGISTRY.get(model_key)
    # look up the loaded model in the registry
    if bundle is None:
        raise HTTPException(status_code=503, detail=f"Model '{model_key}' is not loaded.")
        # 503 Service Unavailable: the model failed to load at startup or doesn't exist

    tokenizer, model = bundle  # unpack the (tokenizer, model) tuple
    return QueryResponse(
        answer=_generate_answer(tokenizer, model, request.query, model_key),
        # run the model to generate an answer
        model=model_key,
        # tell the caller which model answered
    )


# ── API Endpoints ─────────────────────────────────────────────────────────────
@app.post("/python", response_model=QueryResponse)
def ask_python(request: QueryRequest) -> QueryResponse:
    # handles POST /python — routes to the Python FAQ model
    return _handle_request("python", request)


@app.post("/organic", response_model=QueryResponse)
def ask_organic(request: QueryRequest) -> QueryResponse:
    # handles POST /organic — routes to the Organic Chemistry model
    return _handle_request("organic", request)


@app.post("/gita", response_model=QueryResponse)
def ask_gita(request: QueryRequest) -> QueryResponse:
    # handles POST /gita — routes to the Bhagavad Gita model
    return _handle_request("gita", request)


@app.post("/ask", response_model=QueryResponse)
def ask_routed(request: QueryRequest) -> QueryResponse:
    # handles POST /ask — automatically detects domain and routes to the right model
    # this is a convenience endpoint: you don't need to know which model to use
    return _handle_request(_route_model(request.query), request)
    # _route_model() detects the right model key from keywords in the query
