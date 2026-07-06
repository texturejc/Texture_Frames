"""
Fine-tune DeBERTa-v3-large as a trigger-identification token classifier.

Runnable as a script or imported into the Colab notebook. Saves the best model
(by dev trigger-word F1) to `output_dir`.
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

from data import (
    ID2LABEL,
    IGNORE_INDEX,
    LABEL2ID,
    LABELS,
    build_trigger_dataset,
    prf1,
)


def make_compute_metrics():
    """Token-level P/R/F1 on the TRIGGER class (fast proxy used for checkpoint
    selection during training; the authoritative word-level number comes from
    eval_trigger.py after training)."""

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        tp = fp = fn = 0
        trig = LABEL2ID["TRIGGER"]
        for p_row, l_row in zip(preds, labels):
            for p, l in zip(p_row, l_row):
                if l == IGNORE_INDEX:
                    continue
                if p == trig and l == trig:
                    tp += 1
                elif p == trig and l != trig:
                    fp += 1
                elif p != trig and l == trig:
                    fn += 1
        return {"token_" + k: v for k, v in prf1(tp, fp, fn).items()}

    return compute_metrics


def train(
    base_model: str = "microsoft/deberta-v3-large",
    output_dir: str = "outputs/trigger",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-5,
    max_length: int = 320,
    warmup_ratio: float = 0.06,
    weight_decay: float = 0.01,
    fp16: bool = True,
):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    assert tokenizer.is_fast, "need a fast tokenizer for offset_mapping/word_ids"

    # Load master weights in fp32. DeBERTa-v3's checkpoint is fp16-sized; letting
    # params stay fp16 triggers "Attempting to unscale FP16 gradients" under AMP.
    model = AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        torch_dtype=torch.float32,
    )

    train_ds = build_trigger_dataset("train", tokenizer, max_length=max_length)
    dev_ds = build_trigger_dataset("dev", tokenizer, max_length=max_length)

    collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=IGNORE_INDEX)

    # Prefer bf16 on Ampere+ (A100/L4): more stable and, unlike fp16, uses no
    # gradient scaler — so it can't raise "Attempting to unscale FP16 gradients".
    # Fall back to fp16 on older GPUs, fp32 on CPU.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and fp16 and not use_bf16

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
        eval_strategy="epoch",  # transformers >=4.44 (renamed from evaluation_strategy)
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
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return model, tokenizer


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="microsoft/deberta-v3-large")
    p.add_argument("--output-dir", default="outputs/trigger")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-length", type=int, default=320)
    args = p.parse_args()

    model, tokenizer = train(
        base_model=args.base_model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
    )

    # Authoritative word-level eval on the test split.
    from eval_trigger import evaluate_trigger, print_report

    metrics = evaluate_trigger(model, tokenizer, split="test", max_length=args.max_length)
    print_report(metrics)
