"""
Frame-classification accuracy on the test split, with a candidate-mask sweep.

Instead of only hard-masking (non-candidates -> -inf, capped at lexicon
coverage), we sweep a *soft* mask: add a positive bias B to candidate-frame
logits and take the global argmax.
  * B = inf  -> hard mask (candidates always win)  [previous behavior]
  * B = 0    -> no mask (pure global argmax)
  * 0<B<inf  -> candidates strongly preferred but a confident non-candidate can
                still win, recovering golds that fall outside the candidate set.

Every example has one gold + one prediction, so accuracy == precision == recall
== F1, comparable to the baseline's frame_classification F1 (0.887 reproduced).
"""
from __future__ import annotations

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import time

import torch

from data import build_frame_input, frame_candidate_hint, load_frame_examples

DEFAULT_BIASES = [float("inf"), 15.0, 10.0, 7.0, 5.0, 3.0, 0.0]


def _predict(logits: torch.Tensor, cand_ids: list[int], bias: float) -> int:
    if bias == float("inf"):
        if not cand_ids:
            return int(logits.argmax().item())
        cand = torch.tensor(cand_ids, device=logits.device)
        return int(cand[logits[cand].argmax()].item())  # argmax among candidates
    biased = logits.clone()
    if cand_ids:
        idx = torch.tensor(cand_ids, device=logits.device)
        biased[idx] += bias
    return int(biased.argmax().item())


@torch.no_grad()
def evaluate_frame(
    model, tokenizer, lexicon, split: str = "test", max_length: int = 320, biases=None
):
    biases = biases if biases is not None else DEFAULT_BIASES
    model.eval()
    device = model.device
    frame2id = lexicon.frame2id()

    examples = load_frame_examples(split)
    correct = {b: 0 for b in biases}
    covered = 0
    total = 0
    t0 = time.time()
    for text, trigger_loc, gold_frame in examples:
        candidates = lexicon.candidate_frames(text, trigger_loc)
        hint = frame_candidate_hint(candidates)
        enc = tokenizer(
            build_frame_input(text, trigger_loc, hint),
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        logits = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        ).logits[0]

        cand_ids = [frame2id[c] for c in candidates if c in frame2id]
        gold_id = frame2id.get(gold_frame)
        if gold_frame in candidates:
            covered += 1
        for b in biases:
            if _predict(logits, cand_ids, b) == gold_id:
                correct[b] += 1
        total += 1
    elapsed = time.time() - t0

    by_bias = {b: correct[b] / total if total else 0.0 for b in biases}
    return {
        "by_bias": by_bias,
        "total": total,
        "lexicon_coverage": covered / total if total else 0.0,
        "ms_per_example": 1000 * elapsed / max(total, 1),
        "split": split,
    }


def _bias_label(b: float) -> str:
    if b == float("inf"):
        return "hard"
    if b == 0.0:
        return "unmasked"
    return f"soft B={b:g}"


def print_report(metrics: dict, reported_f1: float = 0.887) -> None:
    by_bias = metrics["by_bias"]
    best_b = max(by_bias, key=by_bias.get)
    print("=" * 60)
    print(f"Frame classification (encoder) — {metrics['split']} split")
    print("=" * 60)
    print(f"{'mask':<12}{'f1/acc':>10}   vs baseline {reported_f1:.3f}")
    print("-" * 40)
    for b, acc in by_bias.items():
        star = "  <- best" if b == best_b else ""
        beat = " (beats)" if acc > reported_f1 else ""
        print(f"{_bias_label(b):<12}{acc:>10.3f}{star}{beat}")
    print("-" * 40)
    print(f"lexicon coverage : {metrics['lexicon_coverage']:.3f} (hard-mask ceiling)")
    print(f"best             : {_bias_label(best_b)} = {by_bias[best_b]:.3f}")
    print(f"speed            : {metrics['ms_per_example']:.2f} ms/example")
    print("NOTE: the winning bias should be confirmed on the dev split before")
    print("      being claimed — picking it on test is mildly optimistic.")
    print("=" * 60)
