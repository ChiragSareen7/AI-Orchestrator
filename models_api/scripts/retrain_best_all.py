"""
Retrains all three domain models (python, organic, gita) from their CSV datasets.
Uses Flan-T5 as the base model (can be overridden with the BASE_MODEL env var).
Run with: python models_api/scripts/retrain_best_all.py
"""
from __future__ import annotations  # allows modern type hint syntax on older Python

import os     # for reading environment variables
import sys    # for modifying Python's import path
from pathlib import Path  # for file path operations

import pandas as pd   # for loading and transforming CSV files
import torch          # PyTorch: the deep learning framework
from datasets import Dataset  # HuggingFace dataset format (required by Trainer)
from transformers import (
    AutoModelForSeq2SeqLM,       # loads any seq2seq model from HuggingFace Hub or local path
    AutoTokenizer,                # loads the tokenizer for the model
    DataCollatorForSeq2Seq,       # handles batch padding for training
    Seq2SeqTrainer,               # the high-level training loop class
    Seq2SeqTrainingArguments,     # all training hyperparameters
)

ROOT = Path(__file__).resolve().parents[1]
# ROOT = models_api/ folder (go up 2 levels from scripts/retrain_best_all.py)

sys.path.insert(0, str(ROOT / "utils"))
# add models_api/utils/ to Python's import path
# this allows "from dataset_loader import ..." to find dataset_loader.py

from dataset_loader import _normalize_columns, _read_csv  # noqa: E402
# import helper functions for reading and normalizing CSV files


# ── Data Building Functions ────────────────────────────────────────────────────

def build_python_df() -> pd.DataFrame:
    # reads faq_dataset.csv and builds training data for the Python model
    p = ROOT / "faq_dataset.csv"  # path to the Python FAQ CSV
    df = _read_csv(str(p))         # read CSV (handles encoding issues)
    df = _normalize_columns(df)    # standardize column names to lowercase_with_underscores

    if "question" not in df.columns or "answer" not in df.columns:
        raise ValueError("faq_dataset.csv needs question, answer columns")
        # fail early with a clear error if the CSV doesn't have the expected columns

    rows = []  # will collect training examples as dicts
    for _, r in df.iterrows():  # iterate over each row in the DataFrame
        q = str(r["question"]).strip()  # get question, ensure it's a clean string
        a = str(r["answer"]).strip()    # get answer, ensure it's a clean string
        inp = f"task: python_faq; question: {q}"
        # add the task prefix that the model learned during pre-training
        # IMPORTANT: this exact prefix must match what was used during training
        # the model uses this prefix to know it's answering a Python FAQ

        rows.append({"input_text": inp, "target_text": a})
        # input_text = what goes INTO the model; target_text = what the model should produce

    return pd.DataFrame(rows)  # convert list of dicts to DataFrame


def build_organic_df() -> pd.DataFrame:
    # reads Organic_Compounds_Properties.csv and builds training data for the chemistry model
    p = ROOT / "Organic_Compounds_Properties.csv"
    try:
        df = pd.read_csv(str(p), encoding="utf-8", skiprows=2)
        # skiprows=2 skips the first 2 rows (they may be header/title rows, not data)
    except UnicodeDecodeError:
        df = pd.read_csv(str(p), encoding="latin-1", skiprows=2)
        # try latin-1 if utf-8 fails (Excel-exported CSVs often use latin-1)

    df = _normalize_columns(df)  # standardize column names

    if "compound_name" in df.columns:
        name_col = "compound_name"  # use this column if it exists
    else:
        name_col = next((c for c in df.columns if "compound" in c or "name" in c), df.columns[0])
        # find a column with "compound" or "name" in its name; fall back to first column
        # next() gets the first match from the generator expression

    prop_cols = [c for c in df.columns if c != name_col]
    # all columns EXCEPT the compound name column are property columns

    rows = []
    for _, r in df.iterrows():
        name = str(r.get(name_col, "")).strip()  # get compound name
        if not name:
            continue  # skip rows with no compound name

        parts = [f"{c}: {r[c]}" for c in prop_cols if pd.notna(r.get(c))]
        # build "property_name: value" strings for each non-empty property
        # pd.notna() = True if value is NOT NaN (not missing)

        props = "; ".join(parts)  # combine all properties with "; " separator
        q = f"What are the properties of {name}?"  # standardized question format
        inp = f"task: organic_props; question: {q}"  # add task prefix for model context
        rows.append({"input_text": inp, "target_text": props})

    return pd.DataFrame(rows)


def build_gita_df() -> pd.DataFrame:
    # reads Bhagvad Gita.csv and builds training data for the Gita model
    p = ROOT / "Bhagvad Gita.csv"  # note: filename has a space in it
    df = _read_csv(str(p))          # read CSV with encoding fallback
    df = _normalize_columns(df)     # standardize column names

    rows = []
    for _, r in df.iterrows():
        ch = r.get("chapter_no", r.get("chapter", r.get("ch", "")))
        # try multiple possible column names for chapter number (different CSV formats)
        # .get() returns None if key not found → falls through to the next .get()

        vs = r.get("verse_no", r.get("verse", r.get("v", "")))
        # same pattern for verse number

        trans = r.get("english_translation", r.get("translation", r.get("english", "")))
        # English translation of the Sanskrit verse

        expl = r.get("explanation", r.get("commentary", ""))
        # commentary or explanation of the verse's meaning

        q = f"Chapter {ch} Verse {vs}: explain the teaching."
        # standardized question: "Chapter 2 Verse 47: explain the teaching."

        a = f"Translation: {trans}. Explanation: {expl}"
        # standardized answer format combining translation and explanation

        inp = f"task: gita_verse; question: {q}"
        # add task prefix so the model knows it's a Gita verse task
        rows.append({"input_text": inp, "target_text": str(a).strip()})

    return pd.DataFrame(rows)


# ── Training Helpers ──────────────────────────────────────────────────────────

def _tokenize(batch, tokenizer, max_in: int, max_out: int):
    # converts a batch of text examples to token IDs
    enc = tokenizer(batch["input_text"], truncation=True, max_length=max_in, padding=False)
    # encode the input text; truncate if too long; don't pad (collator handles that)
    lab = tokenizer(batch["target_text"], truncation=True, max_length=max_out, padding=False)
    # encode the target (answer) text
    enc["labels"] = lab["input_ids"]
    # add the target token IDs as "labels" — what the model must learn to predict
    return enc


def _load_start_model(path: Path, base: str):
    # loads the model to start training from:
    # - if a previously trained model exists at 'path', start from there (continue training)
    # - otherwise, download the base model from HuggingFace Hub
    if path.exists() and (path / "config.json").exists():
        # check if there's an existing model at this path
        try:
            tok = AutoTokenizer.from_pretrained(str(path))    # load saved tokenizer
            model = AutoModelForSeq2SeqLM.from_pretrained(str(path))  # load saved model
            return tok, model
            # return the previously trained model → this is "incremental training"
        except Exception:
            pass  # if loading fails for any reason, fall through to downloading base model

    tok = AutoTokenizer.from_pretrained(base)    # download tokenizer from HuggingFace Hub
    model = AutoModelForSeq2SeqLM.from_pretrained(base)  # download base model weights
    return tok, model


def _train(name: str, df: pd.DataFrame, out_dir: Path, epochs: int, lr: float, bs: int, max_in: int, max_out: int):
    # trains a single model on the given DataFrame
    # name = human-readable name for logging ("python", "organic", "gita")
    # df = training data as a pandas DataFrame
    # out_dir = where to save the trained model
    # epochs = number of training passes
    # lr = learning rate (how fast to update weights)
    # bs = batch size (examples per update step)
    # max_in/max_out = max token lengths for input and output

    ds = Dataset.from_pandas(df)
    # convert pandas DataFrame to HuggingFace Dataset (required by Trainer)

    split = ds.train_test_split(test_size=0.2, seed=42)
    # 80% for training, 20% for evaluation
    # seed=42 = reproducible split (same split every run)

    base = os.getenv("BASE_MODEL", "google/flan-t5-base")
    # check if BASE_MODEL is set in environment; default to "google/flan-t5-base"
    # Flan-T5 is a Google seq2seq model pre-trained on instruction following tasks
    # "base" size = 250M parameters (good balance of quality and speed)

    tok, model = _load_start_model(out_dir, base)
    # load starting model (either previously trained or fresh from Hub)

    def tok_map(batch):
        return _tokenize(batch, tok, max_in, max_out)
    # inner function that captures tok (tokenizer) from the outer scope

    tr = split["train"].map(tok_map, batched=True, remove_columns=split["train"].column_names)
    # tokenize all training examples in batches (fast)
    ev = split["test"].map(tok_map, batched=True, remove_columns=split["test"].column_names)
    # tokenize all evaluation examples

    collator = DataCollatorForSeq2Seq(tok, model=model)
    # handles dynamic padding: pads each batch to the length of its longest example

    args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir / "checkpoints"),  # save checkpoints here during training
        eval_strategy="epoch",      # evaluate after each epoch
        save_strategy="epoch",      # save checkpoint after each epoch
        learning_rate=lr,           # step size for weight updates
        per_device_train_batch_size=bs,   # 8 examples per gradient update
        per_device_eval_batch_size=bs,
        num_train_epochs=epochs,    # total training passes (5 for python/gita, 4 for organic)
        weight_decay=0.01,          # L2 regularization to prevent overfitting
        logging_steps=30,           # print loss every 30 steps
        load_best_model_at_end=True, # use the best checkpoint (not necessarily the last one)
        predict_with_generate=True, # use model.generate() for evaluation predictions
        save_total_limit=2,         # only keep 2 checkpoints at a time (saves disk space)
        use_cpu=True,               # explicitly use CPU (no GPU required)
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tr,    # tokenized training data
        eval_dataset=ev,     # tokenized evaluation data
        processing_class=tok,          # tokenizer (some HF versions need this)
        data_collator=collator,  # handles per-batch padding
    )

    trainer.train()  # start the training loop (can take minutes to hours depending on data size)
    trainer.save_model(str(out_dir))    # save the best model to the output directory
    tok.save_pretrained(str(out_dir))   # save the tokenizer alongside the model


def main() -> None:
    device = torch.device("cpu")
    print(f"Using device: {device}")
    # print which hardware is being used (always CPU in this project)

    _train(
        "python",           # model name (just for display)
        build_python_df(),  # build training data from faq_dataset.csv
        ROOT / "python_model",  # save trained model here
        epochs=5,           # 5 training epochs
        lr=3e-5,            # learning rate: 0.00003 (small, typical for fine-tuning)
        bs=8,               # batch size: 8 examples at once
        max_in=256,         # max 256 input tokens
        max_out=256,        # max 256 output tokens
    )

    _train(
        "organic",
        build_organic_df(),   # build training data from Organic_Compounds_Properties.csv
        ROOT / "organic_model",
        epochs=4,   # 4 epochs (chemistry data has less variability, needs fewer passes)
        lr=3e-5,
        bs=8,
        max_in=256,
        max_out=256,
    )

    _train(
        "gita",
        build_gita_df(),    # build training data from Bhagvad Gita.csv
        ROOT / "gita_model",
        epochs=5,           # 5 epochs
        lr=1e-4,            # higher learning rate for Gita (1e-4 = 0.0001)
                            # Gita text is more stylistically different from pre-training data
                            # so we need a larger step size to adapt faster
        bs=8,
        max_in=196,         # shorter max input for Gita (verses + questions tend to be shorter)
        max_out=220,        # slightly longer output (translations + explanations can be verbose)
    )


if __name__ == "__main__":
    # only execute if this script is run directly (not imported)
    main()
