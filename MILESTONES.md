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

## Milestone 1 — Reproduce the baseline in Colab  ⏳ IN PROGRESS

Confirm we can reproduce the upstream `base` model's published test F1 with our
own eval harness, so later "we beat it" claims are credible.

- [x] `git init` project root; absorb vendored code; record provenance
- [x] Standalone eval script that reuses upstream scoring, no Lightning dep
      (`reproduce_baseline/eval_baseline.py`)
- [x] Colab notebook (`reproduce_baseline/reproduce_baseline.ipynb`)
- [x] Pinned `requirements-colab.txt` (canonical env for eval + training)
- [ ] **Run on Colab; confirm F1 within ~1 pt of the table above** ← user action
- [ ] Record measured numbers below

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

### Measured results (fill in after Colab run)
| Task | Reported test F1 | Measured test F1 |
| ---- | ---------------- | ---------------- |
| Trigger identification | 0.74 | _tbd_ |
| Frame classification   | 0.89 | _tbd_ |
| Argument extraction    | 0.75 | _tbd_ |

## Milestone 2 — Encoder pipeline scaffold (DeBERTa-v3-large)  ▫ TODO
## Milestone 3 — Close and beat the gap  ▫ TODO
