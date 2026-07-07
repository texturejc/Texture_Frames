"""
Frame classification v2 — data layer for marker-token pooling.

Same input as v1 (`mark_trigger`: '… <t> gave </t> …'), but instead of pooling
[CLS] we pool the two trigger-marker tokens' hidden states, so the classifier
focuses on the predicate in context (entity-marker pooling, à la relation
extraction). This layer just records each marker's token index. Pure helper is
unit-tested.
"""
from __future__ import annotations

from data import load_frame_examples, mark_trigger


def find_marker_positions(input_ids: list[int], start_id: int, end_id: int) -> tuple[int, int]:
    """Token indices of the <t> and </t> markers. Falls back to CLS (0) / last
    token if a marker was truncated away (rare — trigger near a long tail)."""
    start = input_ids.index(start_id) if start_id in input_ids else 0
    end = input_ids.index(end_id) if end_id in input_ids else len(input_ids) - 1
    return start, end


def build_frame2_dataset(
    split: str, tokenizer, frame2id: dict, start_id: int, end_id: int, max_length: int = 320
):
    """Torch Dataset: input_ids, attention_mask, start_pos, end_pos, labels(frame id).
    Requires the tokenizer to already have the <t>/</t> markers added."""
    import torch

    class _ListDataset(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            return self.rows[idx]

    rows = []
    for text, trigger_loc, frame in load_frame_examples(split):
        if frame not in frame2id:
            continue
        enc = tokenizer(mark_trigger(text, trigger_loc), truncation=True, max_length=max_length)
        sp, ep = find_marker_positions(enc["input_ids"], start_id, end_id)
        rows.append(
            {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "start_pos": sp,
                "end_pos": ep,
                "labels": frame2id[frame],
            }
        )
    return _ListDataset(rows)
