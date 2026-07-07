"""
Fine-tune DeBERTa-v3-large for argument extraction (frame-element role labeling).

Frame-conditioned BIO token classification over the global FE-name vocabulary.
Input: "{frame} : {…}<t> {trigger} </t>{…}" — the trigger is wrapped inline with
predicate-position markers so the model sees *where* the predicate is (M3). At
inference (eval_args.py) the label logits are masked to the frame's FEs so only
valid roles are emitted.
"""
from __future__ import annotations

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import argparse

import numpy as np
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from args_data import IGNORE_INDEX, build_args_dataset, fe_label_maps
from data import TRIGGER_END, TRIGGER_START
from lexicon import Lexicon


def preprocess_logits_for_metrics(logits, labels):
    # argmax on-device so the Trainer stores token ids, not full (seq x ~2400) logits
    return logits.argmax(dim=-1)


def make_compute_metrics():
    """Token-level micro P/R/F1 over FE tokens (proxy for checkpoint selection).
    The authoritative weighted span F1 comes from eval_args."""

    def compute_metrics(eval_pred):
        preds, labels = eval_pred  # preds already argmaxed
        tp = fp = fn = 0
        for p_row, l_row in zip(preds, labels):
            for p, l in zip(p_row, l_row):
                if l == IGNORE_INDEX:
                    continue
                if l != 0 and p == l:
                    tp += 1
                elif l != 0 and p != l:
                    fn += 1
                elif l == 0 and p != 0:
                    fp += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {"token_f1": f1}

    return compute_metrics


def train(
    base_model: str = "microsoft/deberta-v3-large",
    output_dir: str = "outputs/args",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-5,
    max_length: int = 320,
    warmup_ratio: float = 0.06,
    weight_decay: float = 0.01,
):
    lexicon = Lexicon()
    fe_vocab = lexicon.fe_vocab()
    labels, label2id, id2label = fe_label_maps(fe_vocab)
    print(f"FE vocabulary: {len(fe_vocab)} roles -> {len(labels)} BIO labels")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    assert tokenizer.is_fast, "need a fast tokenizer for offset_mapping"
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [TRIGGER_START, TRIGGER_END]}
    )

    model = AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        torch_dtype=torch.float32,
    )
    model.resize_token_embeddings(len(tokenizer))  # for the added <t>/</t> markers

    train_ds = build_args_dataset("train", tokenizer, label2id, lexicon, max_length=max_length)
    dev_ds = build_args_dataset("dev", tokenizer, label2id, lexicon, max_length=max_length)
    print(f"train examples: {len(train_ds)}   dev examples: {len(dev_ds)}")

    collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=IGNORE_INDEX)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        bf16=use_bf16,
        fp16=use_fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="token_f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        compute_metrics=make_compute_metrics(),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return model, tokenizer, lexicon


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="microsoft/deberta-v3-large")
    p.add_argument("--output-dir", default="outputs/args")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-length", type=int, default=320)
    args = p.parse_args()

    model, tokenizer, lexicon = train(
        base_model=args.base_model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
    )

    from eval_args import evaluate_args, print_report

    metrics = evaluate_args(model, tokenizer, lexicon, split="test", max_length=args.max_length)
    print_report(metrics)
