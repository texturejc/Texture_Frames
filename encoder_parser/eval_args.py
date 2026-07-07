"""
Argument-extraction weighted F1 on the test split, with FE masking.

For each (trigger, frame): logits are masked to the frame's FE roles (only O and
the frame's B-/I-{FE} labels are allowed), BIO is decoded to (FE, span_text)
pairs, and scored with the upstream weighting (non-core FEs = 0.5). Comparable to
the baseline's args_extraction F1 (0.753 reproduced).
"""
from __future__ import annotations

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import time

import torch

from args_data import (
    build_args_input,
    decode_bio_spans,
    fe_label_maps,
    load_args_examples,
    score_args,
    trigger_word_text,
)


def _allowed_label_ids(frame, lexicon, label2id):
    core, non_core = lexicon.frame_elements(frame)
    allowed = {label2id["O"]}
    for fe in [*core, *non_core]:
        for tag in (f"B-{fe}", f"I-{fe}"):
            if tag in label2id:
                allowed.add(label2id[tag])
    return allowed


@torch.no_grad()
def evaluate_args(model, tokenizer, lexicon, split: str = "test", max_length: int = 320):
    model.eval()
    device = model.device
    fe_vocab = lexicon.fe_vocab()
    _, label2id, id2label = fe_label_maps(fe_vocab)
    num_labels = len(label2id)

    # cache the allowed-label mask per frame
    frame_mask_cache: dict[str, torch.Tensor] = {}

    def frame_mask(frame: str) -> torch.Tensor:
        if frame not in frame_mask_cache:
            allowed = _allowed_label_ids(frame, lexicon, label2id)
            mask = torch.full((num_labels,), float("-inf"), device=device)
            mask[list(allowed)] = 0.0
            frame_mask_cache[frame] = mask
        return frame_mask_cache[frame]

    examples = load_args_examples(split)
    tp = fp = fn = 0.0
    t0 = time.time()
    for text, trigger_loc, frame, gold_fes in examples:
        gold_spans = [(name, text[s:e].strip()) for name, s, e in gold_fes]

        trig = trigger_word_text(text, trigger_loc)
        combined, prefix_len = build_args_input(text, frame, trig)
        enc = tokenizer(
            combined,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        logits = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        ).logits[0]
        logits = logits + frame_mask(frame)  # restrict to this frame's FE roles
        pred_ids = logits.argmax(-1).tolist()

        offsets = enc["offset_mapping"][0].tolist()
        pred_spans = decode_bio_spans(offsets, pred_ids, id2label, prefix_len, combined)

        s_tp, s_fp, s_fn = score_args(
            gold_spans, pred_spans, lambda fe: lexicon.is_non_core(frame, fe)
        )
        tp += s_tp
        fp += s_fp
        fn += s_fn
    elapsed = time.time() - t0

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_examples": len(examples),
        "ms_per_example": 1000 * elapsed / max(len(examples), 1),
        "split": split,
    }


def print_report(metrics: dict, reported_f1: float = 0.753) -> None:
    print("=" * 60)
    print(f"Argument extraction (encoder) — {metrics['split']} split")
    print("=" * 60)
    print(f"  precision : {metrics['precision']:.3f}")
    print(f"  recall    : {metrics['recall']:.3f}")
    print(f"  f1        : {metrics['f1']:.3f}   (baseline {reported_f1:.3f})"
          f"{'  (beats)' if metrics['f1'] > reported_f1 else ''}")
    print(f"  tp/fp/fn  : {metrics['tp']:.1f}/{metrics['fp']:.1f}/{metrics['fn']:.1f} "
          f"(non-core FEs weighted 0.5)")
    print(f"  speed     : {metrics['ms_per_example']:.2f} ms/example "
          f"over {metrics['n_examples']} examples")
    print("=" * 60)
