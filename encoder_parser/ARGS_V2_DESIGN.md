# Argument extraction v2 — detect-then-classify span head

## Why v1 (flat BIO over ~2,400 labels) tops out ~0.63

Measured error breakdown (400 examples / 704 gold FEs, FE-conditioned model):
exact 363, **missed 153 (recall)**, **boundary_off 120**, wrong_role 68, spurious 116.
Token-level acc is 0.766 but exact-span F1 is 0.628 — a ~14 pt gap that is *all*
boundary imprecision. Two root causes:

1. **2,400-way BIO starves the long tail.** ~1,200 FEs, many with few examples;
   rare roles get almost no gradient signal → they're missed.
2. **Exact-span metric punishes tagging.** One token off = total miss. The
   generative baseline is trained on the eval objective (reproduce the gold
   string); our tagger optimizes a token proxy and decodes separately.

## v2 design

Same backbone + input as v1 (keep both — they help):
`{frame} [{FE menu}] : … <t> {trigger} </t> …` on DeBERTa-v3-large.
Replace the single 2,400-label head with **two heads**:

### Head A — span detection (3-class BIO: O / B / I)
- "Is this token part of *an* argument span?" — frame/role-agnostic.
- Dense positive signal (3 classes, not 2,400) → better boundaries + recall.
- No width cap → arbitrary-length spans (long clause-args survive).
- Standard `DataCollatorForTokenClassification`, weighted CE if O dominates.

### Head B — role classification (per span, frame-masked)
- Input: a span's pooled token reps (start ⊕ end ⊕ mean-pool over the span).
- Output: softmax over **this frame's FE set only** (+ a NULL/reject class),
  masked via the lexicon — picking among ~10, not 2,400.
- Train on **gold spans** (teacher forcing); at inference feed Head-A spans.
- Kills long-tail starvation and role confusion (wrong_role bucket).

### Loss
`L = L_detect (token CE, 3-class) + λ · L_role (span CE over frame FEs)`.
Single model, single forward pass; span pooling is cheap → speed win preserved.

### Decode (inference)
1. Head A → argument spans (BIO decode, arbitrary length).
2. Head B → role for each span (argmax over frame's FEs; drop if NULL wins).
3. Enforce non-overlap per role; emit `(role, span_text)`; clean markers +
   normalize whitespace (as v1). Score with the existing weighted metric.

## Why this reaches the buckets
- **boundary_off ↓** — spans scored/decoded as units with dense detection signal.
- **missed ↓** — detection is a 3-class problem; role head sees the frame's menu.
- **wrong_role ↓** — role choice is a small frame-conditioned softmax.

## Known risk / iteration levers
- Pipeline error propagation: a span Head A misses can't be recovered by Head B.
  Mitigation: Head A recall is the priority; weighted CE / lower B-threshold.
- Train/inference span mismatch (gold vs predicted spans for Head B).
  Standard; monitor with an eval that feeds predicted spans to Head B.
- If still short of 0.753 after first run: add lightweight augmentation (recover
  the dropped nlpaug edge) + tune epochs/LR/pooling.

## Files (planned)
- `args2_data.py` — span-detection labels + gold-span records for Head B (pure
  helpers unit-tested, mirroring args_data).
- `model_args2.py` — DeBERTa backbone + detection head + biaffine/MLP role head.
- `train_args2.py`, `eval_args2.py`, `train_args2.ipynb`.
- Keep v1 (`args_data`/`train_args`/`eval_args`) intact for comparison.
