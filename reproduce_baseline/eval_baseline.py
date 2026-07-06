"""
Reproduce the upstream frame-semantic-transformer baseline F1 on the Open-Sesame
FrameNet 1.7 test/dev splits.

Drives the evaluation loop manually (no PyTorch-Lightning) so it runs on a modern
Colab image, while calling the *upstream* scoring functions verbatim
(`evaluate_batch`, `calc_eval_metrics`, `merge_metrics`) so the numbers are
directly comparable to the published results.

Usage:
    python eval_baseline.py --model base            # upstream base model from HF hub
    python eval_baseline.py --model small
    python eval_baseline.py --model /path/to/model  # a local fine-tuned t5 model
    python eval_baseline.py --model base --split test --batch-size 16
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# Must be set before transformers/sentencepiece import to avoid the protobuf
# "Descriptors cannot be created directly" C++ backend error.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5TokenizerFast

# Make the vendored package importable whether run from repo root or this dir.
REPO = Path(__file__).resolve().parent.parent / "frame-semantic-transformer"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from frame_semantic_transformer.constants import MODEL_MAX_LENGTH, MODEL_REVISION
from frame_semantic_transformer.data.LoaderDataCache import LoaderDataCache
from frame_semantic_transformer.data.TaskSampleDataset import TaskSampleDataset
from frame_semantic_transformer.data.loaders.framenet17 import (
    Framenet17InferenceLoader,
    Framenet17TrainingLoader,
)
from frame_semantic_transformer.data.tasks_from_annotated_sentences import (
    tasks_from_annotated_sentences,
)
from frame_semantic_transformer.training.evaluate_batch import (
    TaskEvalResults,
    calc_eval_metrics,
    evaluate_batch,
)
from frame_semantic_transformer.training.TrainingModelWrapper import merge_metrics

# Published upstream test-set F1, for an at-a-glance comparison.
REPORTED_TEST_F1 = {
    "trigger_identification": 0.74,
    "frame_classification": 0.89,
    "args_extraction": 0.75,
}
REPORTED_DEV_F1 = {
    "trigger_identification": 0.78,
    "frame_classification": 0.91,
    "args_extraction": 0.78,
}


def resolve_model_ref(model: str) -> tuple[str, str | None]:
    """Map the friendly 'base'/'small' names to the HF hub repo + pinned revision."""
    if model in ("base", "small"):
        return f"chanind/frame-semantic-transformer-{model}", MODEL_REVISION
    return model, None  # a local path or an arbitrary hub id


def build_dataset(split: str, tokenizer: T5TokenizerFast, loader_cache: LoaderDataCache):
    training_loader = Framenet17TrainingLoader()
    training_loader.setup()
    if split == "test":
        sentences = training_loader.load_test_data()
    elif split == "dev":
        sentences = training_loader.load_validation_data()
    else:
        raise ValueError(f"unknown split {split!r} (expected 'test' or 'dev')")
    tasks = tasks_from_annotated_sentences(sentences, loader_cache)
    dataset = TaskSampleDataset(tasks, tokenizer, balance_tasks=False)
    return sentences, tasks, dataset


def evaluate(model, tokenizer, dataset, loader_cache, batch_size, num_workers):
    """Manual replacement for TrainingModelWrapper's validate/test loop."""
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    merged: dict[str, TaskEvalResults] = defaultdict(TaskEvalResults)
    n_batches = len(loader)
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch_metrics = evaluate_batch(model, tokenizer, batch, loader_cache)
            # merge_metrics expects a list of per-batch metric dicts
            merged = merge_metrics([merged, batch_metrics])
            if (i + 1) % 10 == 0 or (i + 1) == n_batches:
                print(f"  batch {i + 1}/{n_batches}", flush=True)
    elapsed = time.time() - t0
    return merged, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="base", help="'base', 'small', or a path/hub id")
    parser.add_argument("--split", default="test", choices=["test", "dev"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    model_ref, revision = resolve_model_ref(args.model)
    print(f"Loading model: {model_ref}" + (f" @ {revision}" if revision else ""))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T5ForConditionalGeneration.from_pretrained(model_ref, revision=revision).to(device)
    tokenizer = T5TokenizerFast.from_pretrained(
        model_ref, revision=revision, model_max_length=MODEL_MAX_LENGTH, legacy=False
    )
    print(f"Device: {device}")

    inference_loader = Framenet17InferenceLoader()
    inference_loader.setup()
    loader_cache = LoaderDataCache(inference_loader)

    print(f"Building {args.split} dataset...")
    sentences, tasks, dataset = build_dataset(args.split, tokenizer, loader_cache)
    print(f"  {len(sentences)} sentences -> {len(tasks)} tasks -> {len(dataset)} samples")

    print(f"Evaluating on {args.split} set...")
    merged, elapsed = evaluate(
        model, tokenizer, dataset, loader_cache, args.batch_size, args.num_workers
    )

    reported = REPORTED_TEST_F1 if args.split == "test" else REPORTED_DEV_F1
    print("\n" + "=" * 68)
    print(f"RESULTS ({args.model}, {args.split} split)")
    print("=" * 68)
    header = f"{'task':<24} {'precision':>9} {'recall':>9} {'f1':>9} {'reported':>9}"
    print(header)
    print("-" * len(header))
    for task_name in ("trigger_identification", "frame_classification", "args_extraction"):
        if task_name not in merged:
            continue
        scores = calc_eval_metrics(merged[task_name].scores)
        rep = reported.get(task_name)
        rep_s = f"{rep:>9.2f}" if rep is not None else f"{'-':>9}"
        print(
            f"{task_name:<24} {scores['precision']:>9.3f} {scores['recall']:>9.3f} "
            f"{scores['f_score']:>9.3f}{rep_s}"
        )
    print("-" * len(header))
    print(f"eval wall-clock: {elapsed:.1f}s over {len(dataset)} samples "
          f"({1000 * elapsed / max(len(dataset), 1):.1f} ms/sample)")
    print("=" * 68)


if __name__ == "__main__":
    main()
