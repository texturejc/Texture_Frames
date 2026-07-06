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

### Slice 1 — Trigger identification  (built, awaiting first Colab run)
- [x] `encoder_parser/data.py` — FrameNet loading + tokenizer label alignment;
      pure-Python core (word seg / trigger-word id / scoring) unit-tested locally
- [x] `encoder_parser/train_trigger.py` — HF Trainer fine-tune, DeBERTa-v3-large
- [x] `encoder_parser/eval_trigger.py` — word-level F1, baseline-comparable
- [x] `encoder_parser/train_encoder.ipynb` — Colab driver
- [x] `encoder_parser/tests/test_data_core.py` — 8 tests, pass locally
- [ ] **Colab run: report trigger F1 vs 0.735 + ms/sentence vs 196.6** ← user action

### Slice 2 — Frame classification  ▫ TODO (candidate-masked frame-embedding head)
### Slice 3 — Argument extraction  ▫ TODO (extractive span head per role)

## Milestone 3 — Close and beat the gap  ▫ TODO
(includes optional shared-backbone consolidation — see DESIGN.md)
