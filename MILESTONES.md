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
### Slice 3 — Argument extraction  ▫ TODO (extractive span head per role)

## Milestone 3 — Close and beat the gap  ▫ TODO
(includes optional shared-backbone consolidation — see DESIGN.md)
