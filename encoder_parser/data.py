"""
Data preparation for the encoder trigger-identification head.

Split into two layers:
  * pure-Python helpers (word segmentation, trigger-word id, label alignment,
    word-level scoring) — no torch/transformers, unit-tested in tests/.
  * a builder that uses the upstream FrameNet loaders + a HF fast tokenizer to
    produce a tokenized dataset for `AutoModelForTokenClassification`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Optional

# protobuf C++ backend guard (see requirements-colab.txt / notebooks)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# Make the vendored parser importable from source (no install needed).
_REPO = Path(__file__).resolve().parent.parent / "frame-semantic-transformer"
if _REPO.exists() and str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Label scheme for the token-classification head.
LABELS = ["O", "TRIGGER"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
IGNORE_INDEX = -100  # torch CrossEntropyLoss / HF default ignore id


# --------------------------------------------------------------------------- #
# Pure-Python core (no torch/transformers) — unit-tested                       #
# --------------------------------------------------------------------------- #

def whitespace_words(text: str) -> list[tuple[int, int]]:
    """Return (start, end) char spans of maximal non-whitespace runs.

    Matches the whitespace `.split()` word definition used by the upstream
    trigger metric, but keeps char offsets so we can align to trigger_locs.
    """
    spans: list[tuple[int, int]] = []
    start: Optional[int] = None
    for i, ch in enumerate(text):
        if ch.isspace():
            if start is not None:
                spans.append((start, i))
                start = None
        elif start is None:
            start = i
    if start is not None:
        spans.append((start, len(text)))
    return spans


def snap_to_word_start(text: str, idx: int) -> int:
    """Advance `idx` to the first non-whitespace char at or after it.

    DeBERTa-v3's SentencePiece fast tokenizer can report a sub-token offset that
    starts on the leading space marker rather than the first letter; snapping
    makes trigger-word alignment robust to that quirk. A no-op for clean offsets.
    """
    n = len(text)
    while idx < n and text[idx].isspace():
        idx += 1
    return idx


def trigger_word_indices(
    words: list[tuple[int, int]], trigger_locs: Iterable[int]
) -> set[int]:
    """Indices of `words` that are frame triggers.

    A word is a trigger iff some trigger_loc falls within its char span. Upstream
    inserts a `*` at exactly the trigger_loc, which is the start of the trigger
    word, so this coincides with word-start matching while tolerating minor
    off-by-one offsets.
    """
    locs = sorted(set(trigger_locs))
    out: set[int] = set()
    for idx, (start, end) in enumerate(words):
        if any(start <= loc < end for loc in locs):
            out.add(idx)
    return out


def align_trigger_labels(
    offset_mapping: list[tuple[int, int]],
    word_ids: list[Optional[int]],
    word_is_trigger: list[bool],
) -> list[int]:
    """Per-token labels for token classification.

    Standard first-subword scheme: the first sub-token of each word carries the
    word's label; continuation sub-tokens and special tokens get IGNORE_INDEX so
    they don't contribute to the loss or to per-word scoring.

    `offset_mapping` is accepted for symmetry/validation; grouping uses word_ids.
    """
    assert len(offset_mapping) == len(word_ids)
    labels: list[int] = []
    prev_word: Optional[int] = None
    for wid in word_ids:
        if wid is None:
            labels.append(IGNORE_INDEX)
        elif wid != prev_word:
            labels.append(LABEL2ID["TRIGGER"] if word_is_trigger[wid] else LABEL2ID["O"])
        else:
            labels.append(IGNORE_INDEX)
        prev_word = wid
    return labels


def predicted_trigger_locs_from_tokens(
    offset_mapping: list[tuple[int, int]],
    word_ids: list[Optional[int]],
    token_pred_is_trigger: list[bool],
) -> set[int]:
    """Map first-subword token predictions back to word-start char offsets."""
    locs: set[int] = set()
    prev_word: Optional[int] = None
    for (start, _end), wid, is_trig in zip(
        offset_mapping, word_ids, token_pred_is_trigger
    ):
        if wid is not None and wid != prev_word and is_trig:
            locs.add(start)
        prev_word = wid
    return locs


def score_trigger_words(
    text: str,
    gold_trigger_locs: Iterable[int],
    pred_trigger_locs: Iterable[int],
) -> tuple[int, int, int]:
    """Word-level (true_pos, false_pos, false_neg), upstream-comparable.

    A word counts as a gold/predicted trigger iff a gold/predicted loc lands in
    its whitespace span.
    """
    words = whitespace_words(text)
    gold = trigger_word_indices(words, (snap_to_word_start(text, loc) for loc in gold_trigger_locs))
    pred = trigger_word_indices(words, (snap_to_word_start(text, loc) for loc in pred_trigger_locs))
    true_pos = len(gold & pred)
    false_pos = len(pred - gold)
    false_neg = len(gold - pred)
    return true_pos, false_pos, false_neg


def prf1(true_pos: int, false_pos: int, false_neg: int) -> dict[str, float]:
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) else 0.0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


# --------------------------------------------------------------------------- #
# FrameNet loading + tokenization (needs nltk + transformers)                  #
# --------------------------------------------------------------------------- #

def load_trigger_sentences(split: str) -> list[tuple[str, list[int]]]:
    """Return [(sentence_text, [trigger_locs]), ...] for a split.

    split ∈ {"train", "dev", "test"}.

    Deliberately self-contained: it reuses only the upstream *clean* modules —
    the Open-Sesame split lists and the FrameNet download helper — and reads the
    corpus via nltk directly. It does NOT import Framenet17TrainingLoader, whose
    module unconditionally imports the augmentation classes (SynonymAugmentation,
    KeyboardAugmentation) which require `nlpaug`. We never run augmentation here,
    so pulling in nlpaug (and its numpy/pandas ABI conflicts on Colab) is pure
    downside. The parsing below mirrors upstream
    `parse_annotated_sentence_from_framenet_sentence` for the trigger fields,
    so the sentence set + trigger locs match what the baseline was scored on.
    """
    from nltk.corpus import framenet as fn

    from frame_semantic_transformer.data.loaders.framenet17.ensure_framenet_downloaded import (
        ensure_framenet_downloaded,
    )
    from frame_semantic_transformer.data.loaders.framenet17.sesame_data_splits import (
        SESAME_DEV_FILES,
        SESAME_TEST_FILES,
    )

    ensure_framenet_downloaded()

    if split == "train":
        include_docs, exclude_docs = None, set(SESAME_DEV_FILES) | set(SESAME_TEST_FILES)
    elif split == "dev":
        include_docs, exclude_docs = set(SESAME_DEV_FILES), None
    elif split == "test":
        include_docs, exclude_docs = set(SESAME_TEST_FILES), None
    else:
        raise ValueError(f"unknown split {split!r}")

    out: list[tuple[str, list[int]]] = []
    for doc in fn.docs():
        fname = doc["filename"]
        if exclude_docs and fname in exclude_docs:
            continue
        if include_docs and fname not in include_docs:
            continue
        for sentence in doc["sentence"]:
            text = sentence["text"]
            locs: list[int] = []
            broken = False
            for ann in sentence["annotationSet"]:
                if "FE" in ann and "Target" in ann and "frame" in ann:
                    for target_span in ann["Target"]:
                        loc = target_span[0]
                        if loc >= len(text):
                            broken = True  # upstream drops the whole sentence
                            break
                        locs.append(loc)
                if broken:
                    break
            # upstream keeps the sentence only if it has ≥1 valid frame annotation
            if not broken and locs:
                out.append((text, sorted(set(locs))))
    return out


def build_trigger_dataset(split: str, tokenizer, max_length: int = 320):
    """Tokenize a split into a HF Dataset with input_ids/attention_mask/labels.

    Requires a *fast* tokenizer (offset_mapping + word_ids).
    """
    from datasets import Dataset

    raw = load_trigger_sentences(split)
    texts = [t for t, _ in raw]
    trigger_locs_list = [locs for _, locs in raw]

    def gen():
        for text, trigger_locs in zip(texts, trigger_locs_list):
            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_offsets_mapping=True,
            )
            word_ids = enc.word_ids()
            words = whitespace_words(text)
            trig_word_idx = trigger_word_indices(words, trigger_locs)
            # word_is_trigger indexed by the tokenizer's word ids: a tokenizer
            # "word" is a trigger iff its char-span start lands in a gold trigger
            # whitespace-word. Build per-tokenizer-word trigger flags via offsets.
            n_words = (max([w for w in word_ids if w is not None], default=-1)) + 1
            word_start = [None] * n_words
            for (s, _e), wid in zip(enc["offset_mapping"], word_ids):
                if wid is not None and word_start[wid] is None:
                    word_start[wid] = snap_to_word_start(text, s)
            word_is_trigger = []
            gold = trigger_word_indices(words, trigger_locs)
            for wid in range(n_words):
                cs = word_start[wid]
                # which whitespace word does this tokenizer-word start in?
                is_trig = False
                for gi in gold:
                    ws, we = words[gi]
                    if cs is not None and ws <= cs < we:
                        is_trig = True
                        break
                word_is_trigger.append(is_trig)
            labels = align_trigger_labels(
                enc["offset_mapping"], word_ids, word_is_trigger
            )
            yield {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": labels,
            }

    return Dataset.from_generator(gen)
