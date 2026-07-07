"""
Frame classification v2 (marker-pooled) accuracy with the candidate-mask sweep —
same scoring as eval_frame, just logits from the marker-pooled model. Pick the
bias on dev, report on test; comparable to the baseline's 0.887.
"""
from __future__ import annotations

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import time

import torch

from data import TRIGGER_END, TRIGGER_START, load_frame_examples, mark_trigger
from eval_frame import DEFAULT_BIASES, _predict, print_report  # reuse sweep + report
from frame2_data import find_marker_positions

__all__ = ["evaluate_frame2", "print_report"]


@torch.no_grad()
def evaluate_frame2(model, tokenizer, lexicon, split: str = "test", max_length: int = 320, biases=None):
    biases = biases if biases is not None else DEFAULT_BIASES
    model.eval()
    device = next(model.parameters()).device
    frame2id = lexicon.frame2id()
    start_id = tokenizer.convert_tokens_to_ids(TRIGGER_START)
    end_id = tokenizer.convert_tokens_to_ids(TRIGGER_END)

    examples = load_frame_examples(split)
    correct = {b: 0 for b in biases}
    covered = total = 0
    t0 = time.time()
    for text, trigger_loc, gold_frame in examples:
        enc = tokenizer(
            mark_trigger(text, trigger_loc), truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        sp, ep = find_marker_positions(enc["input_ids"][0].tolist(), start_id, end_id)
        logits = model.encode_logits(
            enc["input_ids"].to(device), enc["attention_mask"].to(device),
            torch.tensor([sp], device=device), torch.tensor([ep], device=device),
        )[0]

        candidates = lexicon.candidate_frames(text, trigger_loc)
        cand_ids = [frame2id[c] for c in candidates if c in frame2id]
        gold_id = frame2id.get(gold_frame)
        if gold_frame in candidates:
            covered += 1
        for b in biases:
            if _predict(logits, cand_ids, b) == gold_id:
                correct[b] += 1
        total += 1
    elapsed = time.time() - t0

    return {
        "by_bias": {b: correct[b] / total if total else 0.0 for b in biases},
        "total": total,
        "lexicon_coverage": covered / total if total else 0.0,
        "ms_per_example": 1000 * elapsed / max(total, 1),
        "split": split,
    }
