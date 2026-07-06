"""
Word-level trigger-identification F1, comparable to the upstream metric.

Runs the fine-tuned token classifier over a split and scores predicted vs. gold
trigger words on the original (un-altered) sentences.
"""
from __future__ import annotations

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import time

import torch

from data import (
    load_trigger_sentences,
    predicted_trigger_locs_from_tokens,
    prf1,
    score_trigger_words,
)


@torch.no_grad()
def evaluate_trigger(model, tokenizer, split: str = "test", max_length: int = 320):
    model.eval()
    device = model.device
    data = load_trigger_sentences(split)

    tp = fp = fn = 0
    t0 = time.time()
    for text, gold_locs in data:
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        word_ids = enc.word_ids()
        offsets = enc["offset_mapping"][0].tolist()
        logits = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        ).logits[0]
        pred_ids = logits.argmax(-1).tolist()
        token_is_trigger = [pid == 1 for pid in pred_ids]
        pred_locs = predicted_trigger_locs_from_tokens(
            offsets, word_ids, token_is_trigger
        )
        s_tp, s_fp, s_fn = score_trigger_words(text, gold_locs, pred_locs)
        tp += s_tp
        fp += s_fp
        fn += s_fn
    elapsed = time.time() - t0

    metrics = prf1(tp, fp, fn)
    metrics["true_pos"], metrics["false_pos"], metrics["false_neg"] = tp, fp, fn
    metrics["n_sentences"] = len(data)
    metrics["ms_per_sentence"] = 1000 * elapsed / max(len(data), 1)
    return metrics


def print_report(metrics: dict, reported_f1: float = 0.735) -> None:
    print("=" * 60)
    print("Trigger identification (encoder) — test split")
    print("=" * 60)
    print(f"  precision : {metrics['precision']:.3f}")
    print(f"  recall    : {metrics['recall']:.3f}")
    print(f"  f1        : {metrics['f1']:.3f}   (baseline {reported_f1:.3f})")
    print(f"  tp/fp/fn  : {metrics['true_pos']}/{metrics['false_pos']}/{metrics['false_neg']}")
    print(f"  speed     : {metrics['ms_per_sentence']:.2f} ms/sentence "
          f"over {metrics['n_sentences']} sentences")
    print("=" * 60)
