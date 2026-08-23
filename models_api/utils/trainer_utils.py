from __future__ import annotations  # allows modern type hint syntax on older Python

import os  # for creating output directories

import torch  # PyTorch — deep learning framework
from datasets import Dataset  # HuggingFace dataset format
from transformers import (
    AutoModelForSeq2SeqLM,         # loads any sequence-to-sequence model (like Flan-T5)
    AutoTokenizer,                  # loads the corresponding tokenizer
    DataCollatorForSeq2Seq,         # batches and pads sequences for efficient training
    Seq2SeqTrainer,                 # the high-level trainer class that handles the training loop
    Seq2SeqTrainingArguments,       # all training hyperparameters (learning rate, epochs, etc.)
)


def tokenize_seq2seq(
    examples: dict,         # a batch of examples: {"input_text": [...], "target_text": [...]}
    tokenizer,              # the tokenizer to use for converting text to token IDs
    max_input_length: int,  # maximum length for input sequences (truncate if longer)
    max_target_length: int, # maximum length for output/target sequences
):
    # converts a batch of text examples into tokenized format for seq2seq training
    # called by Dataset.map() which applies it to the whole dataset in batches

    model_inputs = tokenizer(
        examples["input_text"],      # list of input questions/prompts to tokenize
        max_length=max_input_length, # truncate inputs longer than this
        truncation=True,             # enable truncation (without this, max_length is ignored)
        padding=False,               # don't pad here — DataCollatorForSeq2Seq handles padding per-batch
    )
    # result: {"input_ids": [[101, 2054, ...], ...], "attention_mask": [[1, 1, ...], ...]}
    # input_ids = token ID numbers (the model sees these, not raw text)
    # attention_mask = 1 for real tokens, 0 for padding (tells model what to pay attention to)

    labels = tokenizer(
        examples["target_text"],      # list of expected answers to tokenize
        max_length=max_target_length,
        truncation=True,
        padding=False,
    )
    # result: {"input_ids": [[answer_token1, answer_token2, ...], ...]}

    model_inputs["labels"] = labels["input_ids"]
    # in seq2seq training, "labels" = the target token IDs the model must learn to predict
    # the model sees input_ids (question) and tries to generate labels (answer)

    return model_inputs


def train_model(
    base_model: str,              # HuggingFace model name or local path to start from
                                  # e.g. "google/flan-t5-base" or "./models_api/python_model"
    train_ds: Dataset,            # training dataset (80% of data)
    eval_ds: Dataset | None,      # evaluation dataset (20% of data); None to skip evaluation
    save_path: str,               # where to save the trained model weights
    epochs: int = 3,              # how many full passes through the training data
    learning_rate: float = 3e-5,  # step size for gradient updates (3e-5 = 0.00003)
    batch_size: int = 8,          # number of examples processed together per step
    max_input_length: int = 256,  # truncate inputs at 256 tokens
    max_target_length: int = 256, # truncate targets at 256 tokens
) -> None:

    os.makedirs(save_path, exist_ok=True)
    # create the output directory if it doesn't exist
    # exist_ok=True = don't raise error if it already exists

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    # download or load the tokenizer for the base model
    # if base_model = "google/flan-t5-base" → downloads from HuggingFace Hub
    # if base_model = local path → loads from disk

    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    # download or load the model weights
    # we start from a pretrained model (transfer learning) and fine-tune it on our data

    def _tok(batch):
        return tokenize_seq2seq(batch, tokenizer, max_input_length, max_target_length)
    # inner function that wraps tokenize_seq2seq with our specific tokenizer and lengths
    # used as the function to pass to dataset.map()

    train_tok = train_ds.map(_tok, batched=True, remove_columns=train_ds.column_names)
    # apply tokenization to ALL training examples
    # batched=True = process multiple examples at once (faster than one by one)
    # remove_columns = drop the original text columns (we only need input_ids and labels)

    eval_tok = None
    if eval_ds is not None:
        eval_tok = eval_ds.map(_tok, batched=True, remove_columns=eval_ds.column_names)
    # tokenize evaluation dataset if provided

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    # DataCollatorForSeq2Seq handles dynamic padding:
    # batches often have sequences of different lengths
    # it pads shorter sequences to match the longest in each batch (not the global max)
    # this is more memory-efficient than fixed-length padding

    args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(save_path, "checkpoints"),
        # where to save model checkpoints during training
        # checkpoints are snapshots saved at the end of each epoch

        eval_strategy="epoch" if eval_tok else "no",
        # "epoch" = evaluate after every training epoch; "no" = never evaluate
        # evaluation measures how well the model performs on unseen data

        save_strategy="epoch",
        # save a checkpoint after every epoch

        learning_rate=learning_rate,  # how fast to update model weights (3e-5 is typical for fine-tuning)

        per_device_train_batch_size=batch_size,  # batch size for training (8 examples at once)
        per_device_eval_batch_size=batch_size,   # batch size for evaluation

        num_train_epochs=epochs,  # how many full passes through the training data

        weight_decay=0.01,
        # L2 regularization: slightly penalizes large weights to prevent overfitting
        # makes the model generalize better to unseen questions

        logging_steps=50,
        # print training progress every 50 gradient updates

        load_best_model_at_end=bool(eval_tok),
        # if evaluation is enabled: after training, load the checkpoint with the best eval score
        # ensures we don't end up with the last checkpoint which might have overfit

        predict_with_generate=True,
        # use model.generate() for predictions (not just logits)
        # necessary for seq2seq models to properly decode output tokens

        save_total_limit=2,
        # only keep the 2 most recent checkpoints (saves disk space)
    )

    trainer = Seq2SeqTrainer(
        model=model,              # the model to train
        args=args,                # all training configuration
        train_dataset=train_tok,  # tokenized training data
        eval_dataset=eval_tok,    # tokenized evaluation data (or None)
        processing_class=tokenizer,    # tokenizer for processing (used in some HF versions)
        data_collator=data_collator,   # handles batching and padding
    )

    trainer.train()
    # THE TRAINING LOOP:
    # - for each epoch: iterate through all training batches
    #   - forward pass: model predicts output token by token
    #   - compute loss: how wrong was the prediction?
    #   - backward pass: compute gradients (how to adjust each weight)
    #   - optimizer step: update weights to reduce loss
    # - repeat for all epochs

    trainer.save_model(save_path)
    # saves the fine-tuned model weights to disk (model.safetensors + config.json)

    tokenizer.save_pretrained(save_path)
    # saves the tokenizer files to disk (tokenizer.json + tokenizer_config.json)
    # both are needed later when loading the model for inference
