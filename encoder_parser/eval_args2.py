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


# Bias added to the NULL logit before argmax. Positive -> reject more spans
# (higher precision, lower recall); negative -> keep more (higher recall). 0 is
# the plain operating point.
DEFAULT_NULL_BIASES = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]


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
    model, tokenizer, lexicon, role2id, id2role,
    split: str = "test", max_length: int = 320, null_biases=None,
):
    """Sweep the NULL-logit bias in one pass (detection + role logits are computed
    once per example; only the NULL threshold varies), returning P/R/F1 per bias."""
    null_biases = list(null_biases) if null_biases is not None else DEFAULT_NULL_BIASES
    model.eval()
    device = next(model.parameters()).device
    num_roles = len(role2id)
    null_id = role2id[NULL_ROLE]

    mask_cache: dict[str, torch.Tensor] = {}

    def role_mask(frame: str) -> torch.Tensor:
        if frame not in mask_cache:
            m = torch.full((num_roles,), float("-inf"), device=device)
            m[list(_allowed_role_ids(frame, lexicon, role2id))] = 0.0
            mask_cache[frame] = m
        return mask_cache[frame]

    examples = load_args_examples(split)
    tp = {b: 0.0 for b in null_biases}
    fp = {b: 0.0 for b in null_biases}
    fn = {b: 0.0 for b in null_biases}
    t0 = time.time()
    for text, trigger_loc, frame, gold_fes in examples:
        gold_spans = [(name, _clean_span_text(text[s:e])) for name, s, e in gold_fes]
        is_nc = lambda fe: lexicon.is_non_core(frame, fe)  # noqa: E731

        hint = frame_fe_hint(lexicon, frame)
        combined, prefix_len, _, _ = build_args_input(text, frame, trigger_loc, hint)
        enc = tokenizer(
            combined, truncation=True, max_length=max_length,
            return_offsets_mapping=True, return_tensors="pt",
        )
        offsets = enc["offset_mapping"][0].tolist()
        hidden, detect_logits = model.encode(
            enc["input_ids"].to(device), enc["attention_mask"].to(device)
        )
        spans = decode_detect_spans(offsets, detect_logits[0].argmax(-1).tolist(), prefix_len)

        role_logits = None
        if spans:
            span_bi = [(0, s_tok, e_tok) for (s_tok, e_tok, _, _) in spans]
            role_logits = model.role_logits_for_spans(hidden, span_bi) + role_mask(frame)

        for b in null_biases:
            pred_spans = []
            if role_logits is not None:
                biased = role_logits.clone()
                biased[:, null_id] += b
                role_pred = biased.argmax(-1).tolist()
                for (_, _, cs, ce), r in zip(spans, role_pred):
                    if r != null_id:
                        pred_spans.append((id2role[r], _clean_span_text(combined[cs:ce])))
            s_tp, s_fp, s_fn = score_args(gold_spans, pred_spans, is_nc)
            tp[b] += s_tp
            fp[b] += s_fp
            fn[b] += s_fn
    elapsed = time.time() - t0

    by_bias = {}
    for b in null_biases:
        p = tp[b] / (tp[b] + fp[b]) if (tp[b] + fp[b]) else 0.0
        r = tp[b] / (tp[b] + fn[b]) if (tp[b] + fn[b]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        by_bias[b] = {"precision": p, "recall": r, "f1": f1,
                      "tp": tp[b], "fp": fp[b], "fn": fn[b]}
    return {
        "by_bias": by_bias,
        "n_examples": len(examples),
        "ms_per_example": 1000 * elapsed / max(len(examples), 1),
        "split": split,
    }


def print_report(metrics: dict, reported_f1: float = 0.753, v1_f1: float = 0.628) -> None:
    by_bias = metrics["by_bias"]
    best_b = max(by_bias, key=lambda b: by_bias[b]["f1"])
    print("=" * 62)
    print(f"Argument extraction v2 (detect-then-classify) — {metrics['split']} split")
    print("=" * 62)
    print(f"{'null_bias':<11}{'P':>8}{'R':>8}{'F1':>8}   vs v1 {v1_f1:.3f} / base {reported_f1:.3f}")
    print("-" * 50)
    for b, m in by_bias.items():
        star = "  <- best" if b == best_b else ""
        beat = " (beats base)" if m["f1"] > reported_f1 else ""
        print(f"{b:<11.1f}{m['precision']:>8.3f}{m['recall']:>8.3f}{m['f1']:>8.3f}{star}{beat}")
    print("-" * 50)
    bm = by_bias[best_b]
    print(f"best null_bias {best_b:+.1f}: F1 {bm['f1']:.3f} "
          f"(tp/fp/fn {bm['tp']:.1f}/{bm['fp']:.1f}/{bm['fn']:.1f}, non-core 0.5)")
    print(f"speed {metrics['ms_per_example']:.2f} ms/example over {metrics['n_examples']} examples")
    print("NOTE: pick the bias on DEV, then report it on TEST (picking on test is")
    print("      mildly optimistic).")
    print("=" * 62)
