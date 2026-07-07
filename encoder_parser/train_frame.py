"""
Fine-tune DeBERTa-v3-large as a frame classifier.

Input: a sentence with the trigger word wrapped in entity markers
('The chef <t> gave </t> food.'). Output: one of ~1221 FrameNet frames. At
inference (eval_frame.py) the logits are masked to the lexicon candidate frames,
so an invalid frame can't be produced.
"""
from __future__ import annotations

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import argparse

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from data import TRIGGER_END, TRIGGER_START, build_frame_dataset
from lexicon import Lexicon


def make_compute_metrics():
    """Unmasked top-1 accuracy over all frames — a fast proxy for checkpoint
    selection. The authoritative candidate-masked number comes from eval_frame."""

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = float((preds == labels).mean())
        return {"accuracy": acc}

    return compute_metrics


def train(
    base_model: str = "microsoft/deberta-v3-large",
    output_dir: str = "outputs/frame",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-5,
    max_length: int = 320,
    warmup_ratio: float = 0.06,
    weight_decay: float = 0.01,
):
    lexicon = Lexicon()
    frame2id = lexicon.frame2id()
    id2frame = {i: name for name, i in frame2id.items()}
    print(f"frame vocabulary: {len(frame2id)} frames")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [TRIGGER_START, TRIGGER_END]}
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(frame2id),
        id2label=id2frame,
        label2id=frame2id,
        torch_dtype=torch.float32,
    )
    model.resize_token_embeddings(len(tokenizer))  # for the 2 marker tokens

    train_ds = build_frame_dataset("train", tokenizer, frame2id, lexicon, max_length=max_length)
    dev_ds = build_frame_dataset("dev", tokenizer, frame2id, lexicon, max_length=max_length)
    print(f"train examples: {len(train_ds)}   dev examples: {len(dev_ds)}")

    collator = DataCollatorWithPadding(tokenizer)

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
        metric_for_best_model="accuracy",
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
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return model, tokenizer, lexicon


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="microsoft/deberta-v3-large")
    p.add_argument("--output-dir", default="outputs/frame")
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

    from eval_frame import evaluate_frame, print_report

    metrics = evaluate_frame(model, tokenizer, lexicon, split="test", max_length=args.max_length)
    print_report(metrics)
