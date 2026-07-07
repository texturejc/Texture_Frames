"""
Argument-extraction v2 weighted F1 on the test split.

Per example: encode once -> detection head argmax -> BIO-decode to spans ->
role head over those spans, masked to the frame's FEs (+ NULL so bad spans can be
rejected) -> (role, span_text). Scored with the SAME weighted metric as v1
(args_data.score_args, non-core FEs = 0.5), so the number is directly comparable
to v1's 0.628 and the baseline's 0.753.
"""
from __future__ import annotations

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import time

import torch

from args2_data import NULL_ROLE, decode_detect_spans, role_label_maps
from args_data import (
    _clean_span_text,
    build_args_input,
    frame_fe_hint,
    load_args_examples,
    score_args,
)


def _allowed_role_ids(frame, lexicon, role2id):
    """NULL (always, so spurious spans can be rejected) + this frame's FE roles."""
    core, non_core = lexicon.frame_elements(frame)
    allowed = {role2id[NULL_ROLE]}
    for fe in [*core, *non_core]:
        if fe in role2id:
            allowed.add(role2id[fe])
    return allowed


@torch.no_grad()
def evaluate_args2(
    model, tokenizer, lexicon, role2id, id2role, split: str = "test", max_length: int = 320
):
    model.eval()
    device = next(model.parameters()).device
    num_roles = len(role2id)

    mask_cache: dict[str, torch.Tensor] = {}

    def role_mask(frame: str) -> torch.Tensor:
        if frame not in mask_cache:
            m = torch.full((num_roles,), float("-inf"), device=device)
            m[list(_allowed_role_ids(frame, lexicon, role2id))] = 0.0
            mask_cache[frame] = m
        return mask_cache[frame]

    examples = load_args_examples(split)
    tp = fp = fn = 0.0
    null_id = role2id[NULL_ROLE]
    t0 = time.time()
    for text, trigger_loc, frame, gold_fes in examples:
        gold_spans = [(name, _clean_span_text(text[s:e])) for name, s, e in gold_fes]

        hint = frame_fe_hint(lexicon, frame)
        combined, prefix_len, _, _ = build_args_input(text, frame, trigger_loc, hint)
        enc = tokenizer(
            combined,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = enc["offset_mapping"][0].tolist()
        hidden, detect_logits = model.encode(
            enc["input_ids"].to(device), enc["attention_mask"].to(device)
        )
        detect_pred = detect_logits[0].argmax(-1).tolist()
        spans = decode_detect_spans(offsets, detect_pred, prefix_len)

        pred_spans = []
        if spans:
            span_bi = [(0, s_tok, e_tok) for (s_tok, e_tok, _, _) in spans]
            role_logits = model.role_logits_for_spans(hidden, span_bi) + role_mask(frame)
            role_pred = role_logits.argmax(-1).tolist()
            for (_, _, cs, ce), r in zip(spans, role_pred):
                if r == null_id:
                    continue
                pred_spans.append((id2role[r], _clean_span_text(combined[cs:ce])))

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


def print_report(metrics: dict, reported_f1: float = 0.753, v1_f1: float = 0.628) -> None:
    print("=" * 60)
    print(f"Argument extraction v2 (detect-then-classify) — {metrics['split']} split")
    print("=" * 60)
    print(f"  precision : {metrics['precision']:.3f}")
    print(f"  recall    : {metrics['recall']:.3f}")
    print(f"  f1        : {metrics['f1']:.3f}   (v1 {v1_f1:.3f}, baseline {reported_f1:.3f})"
          f"{'  (beats baseline)' if metrics['f1'] > reported_f1 else ''}")
    print(f"  tp/fp/fn  : {metrics['tp']:.1f}/{metrics['fp']:.1f}/{metrics['fn']:.1f} "
          f"(non-core FEs weighted 0.5)")
    print(f"  speed     : {metrics['ms_per_example']:.2f} ms/example "
          f"over {metrics['n_examples']} examples")
    print("=" * 60)
