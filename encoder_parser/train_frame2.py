"""
Train the frame-classification v2 marker-pooled model (model_frame2).

Custom collator to carry start_pos/end_pos (marker token indices) alongside the
padded ids; otherwise a standard Trainer (single label per example). The
authoritative candidate-masked accuracy comes from eval_frame2's sweep.
"""
from __future__ import annotations

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import argparse
import json

import numpy as np
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments

from data import TRIGGER_END, TRIGGER_START
from frame2_data import build_frame2_dataset
from lexicon import Lexicon
from model_frame2 import FrameMarkerModel


class FrameMarkerCollator:
    """Right-pad ids/mask; stack start_pos/end_pos/labels (padding on the right
    keeps the marker token indices valid)."""

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features):
        maxlen = max(len(f["input_ids"]) for f in features)
        input_ids, attn, start, end, labels = [], [], [], [], []
        for f in features:
            pad = maxlen - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * pad)
            attn.append(f["attention_mask"] + [0] * pad)
            start.append(f["start_pos"])
            end.append(f["end_pos"])
            labels.append(f["labels"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "start_pos": torch.tensor(start, dtype=torch.long),
            "end_pos": torch.tensor(end, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def make_compute_metrics():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        acc = float((np.argmax(logits, axis=-1) == labels).mean())
        return {"accuracy": acc}

    return compute_metrics


def train(
    base_model: str = "microsoft/deberta-v3-large",
    output_dir: str = "outputs/frame2",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-5,
    max_length: int = 320,
    warmup_ratio: float = 0.06,
    weight_decay: float = 0.01,
):
    lexicon = Lexicon()
    frame2id = lexicon.frame2id()
    print(f"frame vocabulary: {len(frame2id)} frames")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.add_special_tokens({"additional_special_tokens": [TRIGGER_START, TRIGGER_END]})
    start_id = tokenizer.convert_tokens_to_ids(TRIGGER_START)
    end_id = tokenizer.convert_tokens_to_ids(TRIGGER_END)

    model = FrameMarkerModel.from_pretrained(base_model, num_frames=len(frame2id))
    model.resize_token_embeddings(len(tokenizer))

    train_ds = build_frame2_dataset("train", tokenizer, frame2id, start_id, end_id, max_length)
    dev_ds = build_frame2_dataset("dev", tokenizer, frame2id, start_id, end_id, max_length)
    print(f"train examples: {len(train_ds)}   dev examples: {len(dev_ds)}")

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
        remove_unused_columns=False,  # keep start_pos/end_pos for the model
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=FrameMarkerCollator(tokenizer),
        compute_metrics=make_compute_metrics(),
    )
    trainer.train()

    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "frame2_model.pt"))
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "frame2id.json"), "w") as f:
        json.dump({"frame2id": frame2id, "base_model": base_model}, f)
    return model, tokenizer, lexicon


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="microsoft/deberta-v3-large")
    p.add_argument("--output-dir", default="outputs/frame2")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-length", type=int, default=320)
    a = p.parse_args()

    model, tokenizer, lexicon = train(
        base_model=a.base_model, output_dir=a.output_dir, epochs=a.epochs,
        batch_size=a.batch_size, lr=a.lr, max_length=a.max_length,
    )

    from eval_frame2 import evaluate_frame2, print_report

    print_report(evaluate_frame2(model, tokenizer, lexicon, split="test", max_length=a.max_length))
