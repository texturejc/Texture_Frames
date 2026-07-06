"""
Frame-classification accuracy on the test split, with candidate masking.

For each trigger, logits are restricted to the lexicon's candidate frames before
argmax — so the model can only ever emit a valid candidate. Since every example
has exactly one gold frame and one prediction, accuracy == precision == recall ==
F1, directly comparable to the baseline's frame_classification F1 (0.887 in our
reproduction).
"""
from __future__ import annotations

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import time

import torch

from data import load_frame_examples, mark_trigger


@torch.no_grad()
def evaluate_frame(model, tokenizer, lexicon, split: str = "test", max_length: int = 320):
    model.eval()
    device = model.device
    frame2id = lexicon.frame2id()

    examples = load_frame_examples(split)
    correct = 0
    total = 0
    covered = 0  # gold frame was among the lexicon candidates
    t0 = time.time()
    for text, trigger_loc, gold_frame in examples:
        enc = tokenizer(
            mark_trigger(text, trigger_loc),
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        logits = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        ).logits[0]

        candidates = lexicon.candidate_frames(text, trigger_loc)
        cand_ids = [frame2id[c] for c in candidates if c in frame2id]
        if gold_frame in candidates:
            covered += 1

        if cand_ids:
            mask = torch.full_like(logits, float("-inf"))
            mask[cand_ids] = logits[cand_ids]
            pred_id = int(mask.argmax().item())
        else:
            pred_id = int(logits.argmax().item())

        gold_id = frame2id.get(gold_frame)
        if pred_id == gold_id:
            correct += 1
        total += 1
    elapsed = time.time() - t0

    acc = correct / total if total else 0.0
    return {
        "accuracy": acc,
        "f1": acc,  # single gold + single pred => f1 == accuracy
        "correct": correct,
        "total": total,
        "lexicon_coverage": covered / total if total else 0.0,
        "ms_per_example": 1000 * elapsed / max(total, 1),
    }


def print_report(metrics: dict, reported_f1: float = 0.887) -> None:
    print("=" * 60)
    print("Frame classification (encoder) — test split")
    print("=" * 60)
    print(f"  accuracy / f1 : {metrics['f1']:.3f}   (baseline {reported_f1:.3f})")
    print(f"  correct       : {metrics['correct']}/{metrics['total']}")
    print(f"  lexicon cover : {metrics['lexicon_coverage']:.3f} "
          f"(share of golds present in candidate set — an upper bound on accuracy)")
    print(f"  speed         : {metrics['ms_per_example']:.2f} ms/example")
    print("=" * 60)
