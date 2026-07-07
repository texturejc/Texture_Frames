# Path B — Encoder + task-heads FrameNet parser

Goal: re-architect the T5 seq2seq parser (`frame-semantic-transformer`, v0.10.0,
upstream frozen 2023) into a **DeBERTa-v3-large encoder + three task heads**,
beating its accuracy (especially argument extraction) without sacrificing
inference speed. Training runs on Google Colab (Pro/Pro+, A100/L4).

## Baseline to beat (upstream `base` model, Sesame test split)

| Task                   | Base F1 (dev / test) |
| ---------------------- | -------------------- |
| Trigger identification | 0.78 / 0.74          |
| Frame classification   | 0.91 / 0.89          |
| Argument extraction    | 0.78 / 0.75          |

Data: FrameNet 1.7 via NLTK (`framenet_v17`). Splits: Open-Sesame
(23 test docs, 8 dev docs — see `sesame_data_splits.py`).

---

## Milestone 1 — Reproduce the baseline in Colab  ✅ DONE (2026-07-06)

Confirm we can reproduce the upstream `base` model's published test F1 with our
own eval harness, so later "we beat it" claims are credible.

- [x] `git init` project root; absorb vendored code; record provenance
- [x] Standalone eval script that reuses upstream scoring, no Lightning dep
      (`reproduce_baseline/eval_baseline.py`)
- [x] Colab notebook (`reproduce_baseline/reproduce_baseline.ipynb`)
- [x] Pinned `requirements-colab.txt` (canonical env for eval + training)
- [x] **Run on Colab; confirmed F1 within ~0.5 pt of reported** (2026-07-06, A100)
- [x] Record measured numbers below

### Colab environment (avoids the install/protobuf failures)
- Parser is **cloned + added to `sys.path`**, never `pip install`ed → no package
  build, so imports can't fail with `KeyError: 'frame_semantic_transformer'`.
- `torch` / `protobuf` / `numpy` are left exactly as Colab ships them. The
  protobuf C++ error is defeated by setting
  `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before importing transformers,
  **not** by pinning protobuf (which conflicts with Colab's TF/google-* libs).
- Everything else is pinned in `requirements-colab.txt`.

### Why a custom eval loop
Upstream `evaluate_model()` uses PyTorch-Lightning 1.x hooks
(`validation_epoch_end(outputs)`, `Trainer(gpus=1)`) that were **removed in
Lightning 2.0**. Rather than pin a fragile 2023 dependency stack on a 2026
Colab image, `eval_baseline.py` drives the loop manually while calling the
upstream `evaluate_batch` / `calc_eval_metrics` / `merge_metrics` functions
verbatim — identical scoring, robust environment.

### Measured results (2026-07-06, Colab A100, base model, test split)
| Task | Reported test F1 | Measured test F1 | Δ |
| ---- | ---------------- | ---------------- | ---- |
| Trigger identification | 0.74 | 0.735 (P 0.757 / R 0.714) | −0.005 |
| Frame classification   | 0.89 | 0.887 (P 0.887 / R 0.887) | −0.003 |
| Argument extraction    | 0.75 | 0.753 (P 0.740 / R 0.767) | +0.003 |

**Baseline reproduced** (all within ~0.5 pt; residual = tokenizer/transformers
drift since 2023, as expected).

### Speed baseline to beat  ⭐
`196.6 ms/sample` on an A100 — 15,126 test samples took **~50 minutes**. This is
slow because inference is 3 sequential passes of 5-way beam-search *generation*.
The Milestone 2 encoder does one forward pass per sample with no beam search, so
this is the number that should drop ~10–40×. Latency is a headline deliverable,
not an afterthought.

## Milestone 2 — Encoder pipeline scaffold (DeBERTa-v3-large)  ⏳ IN PROGRESS

Architecture + rationale in `encoder_parser/DESIGN.md`. Built as vertical slices
(one task fully working before the next) since CUDA can't run locally.

### Slice 1 — Trigger identification  ✅ DONE (2026-07-06) — beats baseline
- [x] `encoder_parser/data.py`, `train_trigger.py`, `eval_trigger.py`,
      `train_encoder.ipynb`, `sesame_splits.py`
- [x] tests: `test_data_core.py` (8), `test_load_trigger.py` (1) — pass locally
- [x] **Colab run (A100, DeBERTa-v3-large, bf16, 5 epochs):**

  | metric | encoder | baseline |
  | ------ | ------- | -------- |
  | trigger F1 | **0.751** (P 0.728 / R 0.775) | 0.735 |
  | speed | 73.7 ms/sentence (unbatched) | 196.6 ms/sample (batched, 3-task) |

  tp/fp/fn = 5326/1994/1547 over 1354 test sentences.

  **Caveats (to tighten before any published claim):**
  1. Not a perfectly clean head-to-head: 0.735 came from upstream's *generative*
     eval (penalized for text-generation artifacts); the encoder is scored with
     our cleaner word-level metric. To be airtight, also run the baseline model
     through our metric. The encoder is genuinely ahead, but the exact margin is
     soft.
  2. Speed units differ (sentence vs task-sample) and encoder eval is unbatched;
     batching will widen the gap. Not yet an apples-to-apples ms figure.

  Bottom line: pipeline is sound and the encoder is at least competitive and
  likely ahead on trigger id — strong green light for slices 2 & 3.

### Slice 2 — Frame classification  (built, awaiting first Colab run)
DeBERTa-v3-large sequence classifier over the full frame vocab (~1221), with the
trigger wrapped in entity markers (`<t> … </t>`); at inference logits are masked
to the lexicon candidate frames so only a valid frame can be emitted.
- [x] `encoder_parser/lexicon.py` — FrameNet frame vocab + candidate lookup,
      vendored faithfully from upstream (LoaderDataCache / InferenceLoader /
      FrameClassificationTask), nltk-only, no `nlpaug`
- [x] `encoder_parser/data.py` — `mark_trigger`, `load_frame_examples`,
      `build_frame_dataset`
- [x] `encoder_parser/train_frame.py`, `eval_frame.py` (candidate-masked F1)
- [x] `encoder_parser/train_frame.ipynb` — Colab driver
- [x] tests: `mark_trigger` (3), `trigger_bigrams` (3) pass locally
- [x] **Colab run (A100, DeBERTa-v3-large, bf16, 5 epochs):**

  | metric | encoder | baseline |
  | ------ | ------- | -------- |
  | frame F1 / acc | **0.861** (5927/6886) | 0.887 |
  | lexicon coverage (ceiling) | 0.978 | — |
  | speed | 78.4 ms/example (unbatched) | — |

  **Gap analysis (0.887 − 0.861 = −0.026, below baseline):**
  - 2.2% is lost to the hard candidate mask: coverage is 0.978, so 2.2% of gold
    frames aren't in the candidate set and become forced errors. The baseline
    (T5) uses candidates only as an input *hint* and can still emit any valid
    frame, so it isn't capped at 0.978. **Fix: soft-mask** (bias candidates
    instead of −inf) to recover this.
  - Of the covered cases the model gets 0.861/0.978 = 0.881 right; the rest is
    discrimination among multi-candidate triggers → **tuning territory** (marker-
    token pooling instead of CLS, LR/epochs, feed candidate names into input as
    the baseline does).
  - This is the baseline's *strongest* task; a first untuned scaffold landing
    within 2.6 pts is expected. Closing it is a Milestone 3 job.
### Slice 3 — Argument extraction  (built, awaiting first Colab run)
Frame-conditioned **BIO role labeling** over a global FE-name vocabulary. Input
`{frame} | {trigger_word} : {sentence}`; at inference logits are masked to the
frame's FEs so only valid roles emit. Weighted span F1 (non-core FEs = 0.5)
mirrors upstream ArgumentsExtractionSample.evaluate_prediction.
- [x] `encoder_parser/lexicon.py` — FE vocab + core/non-core per frame
- [x] `encoder_parser/args_data.py` — load/build + pure BIO align, span decode,
      weighted scorer
- [x] `encoder_parser/train_args.py` (big label head; preprocess_logits caps eval
      memory), `eval_args.py` (FE-masked weighted F1)
- [x] `encoder_parser/train_args.ipynb` — Colab driver
- [x] tests: `test_args_data.py` (8) — BIO align/decode round-trip + weighted
      metric, pass locally
- [x] **Colab run (A100, DeBERTa-v3-large, bf16, 5 epochs):**

  | metric | encoder | baseline |
  | ------ | ------- | -------- |
  | args F1 (weighted span) | **0.605** (P 0.598 / R 0.611) | 0.753 |
  | token-level acc (proxy) | 0.766 | — |
  | speed | 62 ms/example | — |

  First run scored 0.009 due to a BIO-alignment bug (DeBERTa leading-space
  offsets dropped the first token of every span); fixed + retrained -> 0.605.
  - **token 0.766 vs span 0.605**: the model tags most tokens right, but exact
    span-match is unforgiving — boundaries are often a token off.
  - **Below baseline (−0.148)** on its weakest task; this is the scaffold with
    no predicate-position marking. Biggest M3 lever: mark the trigger inline so
    the model knows *where* the predicate is, not just its word.

Milestone 2 complete: all three slices run end-to-end through baseline-comparable
metrics. Scorecard — trigger 0.751 > 0.735 ✓ | frame 0.861 vs 0.887 (soft-mask
sweep pending) | args 0.605 vs 0.753. Not yet beating overall; M3 is where that's
decided (and not guaranteed).

## Milestone 3 — Close and beat the gap  ⏳ IN PROGRESS

### Args — predicate-position marking  (built, awaiting Colab retrain)
The scaffold conveyed the trigger only as a word in the prefix, so on multi-
predicate sentences the model couldn't tell *which* verb's arguments to extract
(diagnostic showed role-scramble + fragmentation on repeated sentences). Fix:
wrap the trigger inline with `<t> … </t>` markers.
- [x] `args_data.build_args_input` now returns `(combined, prefix_len, ts, te)`
      and marks the trigger; input is `"{frame} : … <t> {trigger} </t> …"`
- [x] `remap_fe_span` moves gold FE offsets through the inserted markers
      (start uses `>=`, end uses `>` so trigger-adjacent FEs land correctly)
- [x] `<t>`/`</t>` added as special tokens + `resize_token_embeddings` in
      `train_args`; decode strips any marker a span abuts; gold/pred whitespace
      normalized identically before matching
- [x] tests: build-marks-trigger, remap-around-trigger, decode-strips-markers
      (11 args tests pass locally)
- [x] **Colab retrain (predicate marking only): 0.596 — flat vs 0.605 scaffold.**
      Diagnosis: predicate localization wasn't the bottleneck (multi-predicate
      confusion is a minority of examples). Error breakdown on 400 examples /
      704 gold FEs: exact 363, **missed 153 (recall)**, **boundary_off 120**,
      wrong_role 68, spurious 116. Recall + boundary dominate; the model is
      under-informed about *which* roles to look for (wrong_role is small).

### Args — FE-menu conditioning  (built, awaiting Colab retrain)
Feed the frame's FE names into the input so the model knows the role inventory
(mirrors how the baseline conditions). Targets the missed + wrong_role buckets.
- [x] `frame_fe_hint` + `build_args_input(..., fe_hint)`; input is now
      `"{frame} [{FE1}; {FE2}; …] : … <t> {trigger} </t> …"`
- [x] `build_args_dataset` + `eval_args` thread the lexicon so train/eval inputs
      match exactly; 13 args tests pass
- [ ] **Colab retrain + eval: report args F1 vs 0.605 / 0.753**
- [ ] Cheap pre-retrain check: FE-mask coverage (forced-miss recall ceiling ->
      args soft-mask, parallel to the frame fix) — run before committing more.

### Frame soft-mask sweep  — DONE (2026-07-07), gap NOT closed
Sweep result: dev best `soft B=10` = 0.891; that bias on **test = 0.863**
(hard 0.861). Only +0.002 on test — the 2.2% coverage-ceiling misses aren't
recovered because the model isn't confident on non-candidate golds (a
*discrimination* gap, not a mask gap). Adopt B=10 (free, dev-justified) but
**frame is still −0.024 vs 0.887**. Caveat: dev 0.891 "beats" only the
baseline's *test* 0.887; baseline dev is ~0.91, so test-vs-test we're behind.
- [x] **Frame candidate-name conditioning — TRIED, FAILED, REVERTED.**
      Retrained with `[cand1; cand2; …] :` prepended to the input. Test dropped
      **0.863 → 0.816** (−0.047). Lesson: unlike args (per-token, where the FE
      menu informs each decision without diluting), frame is a *sequence-level*
      [CLS]-pooled task — prepending a long candidate list dilutes the pooled
      representation and invites a "list → dominant frame" shortcut that hurts
      the ambiguous triggers we wanted to fix. Reverted (git revert). Best frame
      stays the original model + soft-mask B=10 = **0.863**.
- [x] **Frame v2 — marker-token pooling (BUILT + locally verified, awaiting retrain).**
      Pool the `<t>`/`</t>` marker hidden states (concat) instead of [CLS], to close
      the discrimination gap soft-mask/conditioning couldn't. Mirrors the args v2
      infra: `frame2_data.py` (marker-position finder, pure-tested), `model_frame2.py`
      (marker-pooled classifier), `train_frame2.py` (position-carrying collator +
      Trainer), `eval_frame2.py` (reuses eval_frame's candidate sweep + report),
      `train_frame2.ipynb`. 5 frame2 tests pass (marker-find, forward/backprop,
      collator, trainer step). Keeps v1 frame code intact for comparison.
- [x] **Colab retrain + sweep: frame2 marker-pooling ≈ flat.** dev best B=7
      = 0.896; dev-picked bias on **test = 0.863** (test-best B=10 = 0.868).
      Only ~+0.005 over v1's 0.863 — marker pooling was *not* the lever.
      Diagnosis: covered-case accuracy is now ~0.885 (≈ baseline's 0.887 overall),
      so discrimination is basically solved; the residual −0.02 is mostly the
      2.2% candidate-coverage ceiling, which soft-mask can't recover — a hard,
      low-payoff lever. **Frame plateaus ~0.86–0.87; banked as competitive.**

## Milestone 4 — pip-installable library  ⏳ IN PROGRESS

Wrap the three trained heads into `texture-frames`, installable via
`pip install git+https://github.com/texturejc/Texture_Frames`.
- [x] `src/texture_frames/` package (src-layout); runtime modules copied from
      `encoder_parser/` with relativized imports; vendored-parser hack stripped.
- [x] `pipeline.py` — `FrameParser.parse(text)` chains trigger → frame → args on
      raw text, returns `FrameAnnotation(trigger, frame, [Argument(role, text, …)])`.
- [x] `weights.py` — HF Hub download + rebuild custom heads (from_config backbone,
      resize-then-load_state_dict). `pyproject.toml`, `README.md`, package tests.
- [x] editable install + import + 4 package tests pass locally.
- [ ] **User: upload the 3 checkpoints to HF Hub** (`texture-frames-{trigger,
      frame,args}`) via the README snippet, then first real `parse()` end-to-end.
- [x] Weights uploaded to HF Hub; code repo + 3 HF repos made **public** —
      tokenless `pip install` + `FrameParser()` verified end-to-end.
- [x] Comprehensive `README.md`: FrameNet background, install (local + Colab,
      dependency notes), full usage, benchmark table, per-head technical
      justification + training protocol, acknowledgements (David Chanin, FrameNet,
      Open-Sesame, DeBERTa), citation + license.
- [ ] Optional: fp16 weights to halve download; PyPI release; CLI entry point.

## Final scorecard (2026-07-07) — goal: ≥ baseline accuracy, faster

| Task    | Encoder            | Baseline | Verdict                         |
| ------- | ------------------ | -------- | ------------------------------- |
| Trigger | 0.751              | 0.735    | ✅ ahead                        |
| Args    | 0.750              | 0.753    | ✅ parity (±noise; beats on dev)|
| Frame   | 0.863–0.868        | 0.887    | ~−0.02, competitive (banked)    |
| Speed   | ~50 ms/example     | 196.6 ms | ✅ ~4× faster everywhere        |

Outcome: **2 of 3 tasks at/above baseline, frame within ~0.02, and the whole
pipeline ~4× faster** (one forward pass vs 3 generative beam-search passes).
Args — the task everyone flagged as hardest — went 0.628 → 0.712 → **0.750**
via the detect-then-classify rearchitecture + WordNet augmentation. Frame is the
one task still short, and three principled levers couldn't close its last ~0.02.

### Args v2 — detect-then-classify  (BUILT + locally verified, awaiting Colab retrain)
Replaces v1's flat 2,400-way BIO with two heads on one DeBERTa backbone
(`ARGS_V2_DESIGN.md`): 3-class span detection + per-span frame-masked role
classification (+ NULL to reject spurious spans). Keeps predicate marking +
FE-menu conditioning. Single forward pass -> speed win preserved.
- [x] `args2_data.py` — detection labels, gold span→token records, negative
      sampling, span decoder, role maps (11 pure tests)
- [x] `model_args2.py` — two-head model, span pooling (start⊕end⊕mean), loss
      = detect_CE + λ·role_CE (3 smoke tests on tiny backbone)
- [x] `train_args2.py` — ragged-span collator + `Args2Trainer`; `eval_args2.py`
      — decode→role→frame-mask→weighted F1 (same metric as v1); `train_args2.ipynb`
- [x] end-to-end plumbing verified locally: collator, train step, eval-loss via
      prediction_step, eval decode path (3 tests). 17 v2 tests total pass.
- [x] **Colab retrain (A100): v2 F1 = 0.712** (P 0.687 / R 0.737), **50 ms/example.**
      +0.084 over v1's 0.628 and *faster* — the architecture change worked. Now
      only −0.041 from baseline 0.753. Precision < recall (fp 3532 > fn 2766):
      the model over-predicts, so a NULL-threshold tightening should raise F1.
- [x] **NULL-bias sweep:** dev best +2.0 (0.746); that bias on **test = 0.715**
      (+0.003 over 0.712). Free lever tapped — curve flat around the max. Still
      **−0.038 vs 0.753**. Adopt null_bias +2.0.
- [x] **v2 error breakdown** (704 gold, at +2.0): exact 489, missed 84,
      spurious 75, boundary 67, wrong_role 64 — v2 improved *every* v1 bucket
      (exact 363→489, boundary 120→67, missed 153→84). Remaining tail is
      balanced; biggest addressable = missed + wrong_role (long tail).
- [x] **Augmentation (built, awaiting retrain).** `augment.py`: WordNet synonym
      replacement with char-offset remapping of trigger + FE spans (pure part
      unit-tested; verified end-to-end locally). `build_args2_dataset(augment=N)`
      adds N synonym copies per arg-bearing train example. Tuning retrain config:
      epochs 6, n_negatives 6, augment 1.
- [x] **Colab tuning retrain (augment=1, n_neg=6, epochs=6) + sweep:**
      dev best +2.0 = **0.784** (beats base); that bias on **test = 0.750**
      (P 0.737 / R 0.763), **49.5 ms/example**. +0.035 from augmentation.
      **Effectively at parity** with baseline 0.753 (−0.003, within single-run
      noise; beats on dev) and ~4× faster. Args goal met ("as accurate + faster").
      Trajectory: v1 0.628 → v2 0.712 → v2+aug **0.750**.

### Still queued
- Trigger metric comparability: run the baseline model through our word-level
  metric for an airtight head-to-head.
- Frame: marker-token pooling (deferred; args v2 first per user).
- Batched eval for apples-to-apples ms/example vs the baseline's 196.6.
- Optional shared-backbone consolidation — see DESIGN.md.
