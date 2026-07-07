"""
Train the argument-extraction v2 detect-then-classify model (model_args2).

Custom collator + Trainer because each example carries a ragged list of spans
(gold + sampled NULL negatives) that can't be a padded tensor. Loss is computed
inside the model (detect_CE + λ·role_CE); eval only accumulates that loss for
checkpoint selection — the authoritative weighted span F1 comes from eval_args2.
"""
from __future__ import annotations

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import argparse
import json

import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments

from args2_data import build_args2_dataset, role_label_maps
from data import TRIGGER_END, TRIGGER_START
from lexicon import Lexicon
from model_args2 import IGNORE_INDEX, Args2Model


class Args2Collator:
    """Right-pad input_ids/attention_mask/detect_labels; carry `spans` as a plain
    Python list (token indices stay valid because padding is on the right)."""

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features):
        maxlen = max(len(f["input_ids"]) for f in features)
        input_ids, attn, detect, spans = [], [], [], []
        for f in features:
            pad = maxlen - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * pad)
            attn.append(f["attention_mask"] + [0] * pad)
            detect.append(f["detect_labels"] + [IGNORE_INDEX] * pad)
            spans.append([tuple(s) for s in f["spans"]])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "detect_labels": torch.tensor(detect, dtype=torch.long),
            "spans": spans,
        }


class Args2Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # eval: only the loss — `spans` is a ragged Python list, not gatherable
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)


def train(
    base_model: str = "microsoft/deberta-v3-large",
    output_dir: str = "outputs/args2",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-5,
    max_length: int = 320,
    role_lambda: float = 1.0,
    n_negatives: int = 4,
    warmup_ratio: float = 0.06,
    weight_decay: float = 0.01,
):
    lexicon = Lexicon()
    fe_vocab = lexicon.fe_vocab()
    roles, role2id, id2role = role_label_maps(fe_vocab)
    print(f"roles: {len(roles)} (NULL + {len(fe_vocab)} FEs)")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    assert tokenizer.is_fast, "need a fast tokenizer for offset_mapping"
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [TRIGGER_START, TRIGGER_END]}
    )

    model = Args2Model.from_pretrained(base_model, num_roles=len(roles), role_lambda=role_lambda)
    model.resize_token_embeddings(len(tokenizer))

    train_ds = build_args2_dataset("train", tokenizer, role2id, lexicon, max_length, n_negatives)
    dev_ds = build_args2_dataset("dev", tokenizer, role2id, lexicon, max_length, n_negatives)
    print(f"train examples: {len(train_ds)}   dev examples: {len(dev_ds)}")

    collator = Args2Collator(tokenizer)
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
        metric_for_best_model="loss",
        greater_is_better=False,
        save_total_limit=1,
        remove_unused_columns=False,  # keep detect_labels/spans for the model
        label_names=["detect_labels", "spans"],
        report_to="none",
    )

    trainer = Args2Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
    )
    trainer.train()

    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "args2_model.pt"))
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "role2id.json"), "w") as f:
        json.dump({"role2id": role2id, "base_model": base_model}, f)
    return model, tokenizer, lexicon, role2id, id2role


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="microsoft/deberta-v3-large")
    p.add_argument("--output-dir", default="outputs/args2")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-length", type=int, default=320)
    p.add_argument("--role-lambda", type=float, default=1.0)
    a = p.parse_args()

    model, tokenizer, lexicon, role2id, id2role = train(
        base_model=a.base_model,
        output_dir=a.output_dir,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        max_length=a.max_length,
        role_lambda=a.role_lambda,
    )

    from eval_args2 import evaluate_args2, print_report

    metrics = evaluate_args2(
        model, tokenizer, lexicon, role2id, id2role, split="test", max_length=a.max_length
    )
    print_report(metrics)
