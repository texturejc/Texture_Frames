# Path B — Encoder + task-heads architecture

Replaces the T5 seq2seq pipeline with encoder models that predict structure
directly (spans / labels), reusing the upstream FrameNet loaders, Open-Sesame
splits, candidate-frame lexicon, and — for comparability — the upstream metric
definitions.

Backbone: **DeBERTa-v3-large** (`microsoft/deberta-v3-large`, ~435M).

## The three tasks (same decomposition as upstream)

| Task | Input | Head | Output |
| ---- | ----- | ---- | ------ |
| **Trigger id** | sentence | token classification (`O` / `TRIGGER`) | which words evoke a frame |
| **Frame classification** | sentence + marked trigger | pooled classifier over a **frame embedding table**, masked to lexicon candidates | one frame name |
| **Argument extraction** | sentence + trigger + frame + role | extractive span head (start/end, nullable) per frame element | role → span |

All three are **single forward passes** — no autoregressive decoding, no beam
search. That is the source of the expected ~10–40× speedup over the baseline's
196.6 ms/sample.

## Scaffold strategy — vertical slices

Because the environment can't run CUDA locally, we validate one *complete*
task end-to-end (data → train → eval-through-comparable-metric) before scaling
to the next, so bugs surface in the smallest possible surface area.

1. **Trigger id first** (this milestone's first slice) — the simplest task, but
   it exercises the entire machinery: loaders → tokenization/alignment →
   HF Trainer → eval adapter → comparable F1.
2. Frame classification — adds the candidate-masked frame-embedding head.
3. Argument extraction — the hard one; extractive span head per role.

## Separate models vs. shared backbone

The scaffold trains **one DeBERTa-v3 model per task** (standard HF recipes,
independently debuggable, each still a single fast forward pass; 3×435M in fp16
≈ 2.6 GB, trivial on an A100). This is a deliberate, reversible deviation from
the earlier "one shared backbone, three heads" sketch: consolidating into a
shared encoder (lower memory, possible accuracy gain from shared representations)
is a **Milestone 3** optimization once all three tasks are individually working
and we have head-to-head numbers. Flagging so it's a conscious choice, not drift.

## Metric comparability

Upstream trigger id is a **word-level** metric: split the sentence on
whitespace; a word is a trigger iff prefixed with `*`; align predicted vs. gold
words positionally and count TP/FP/FN (see `TriggerIdentificationSample.
evaluate_prediction`). The generative model can corrupt the text (dropped/added
words → FP/FN); the encoder never alters the text, so we compute the *same*
word-level metric directly on the original sentence — a cleaner, fair version of
the identical definition. `eval_trigger.py` implements it.

Frame and argument metrics will likewise mirror upstream's `evaluate_prediction`
definitions when those slices land.

## Files

- `data.py` — load FrameNet sentences via upstream loaders; pure-Python
  word-segmentation / trigger-word / scoring helpers (unit-tested, no torch);
  tokenizer-based label alignment for training.
- `train_trigger.py` — HF `Trainer` fine-tune of DeBERTa-v3-large token classifier.
- `eval_trigger.py` — word-level trigger P/R/F1, upstream-comparable.
- `train_encoder.ipynb` — Colab driver (installs `requirements-colab.txt`).
- `tests/` — pure-Python unit tests for the alignment/scoring core.
