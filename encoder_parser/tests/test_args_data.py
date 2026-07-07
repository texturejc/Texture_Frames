"""Pure-Python tests for args_data (no torch/transformers/nltk)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from args_data import (  # noqa: E402
    IGNORE_INDEX,
    align_fe_bio,
    build_args_input,
    decode_bio_spans,
    fe_label_maps,
    score_args,
)

FE_VOCAB = ["Agent", "Theme", "Time"]
LABELS, L2I, I2L = fe_label_maps(FE_VOCAB)


def test_fe_label_maps():
    assert LABELS == ["O", "B-Agent", "I-Agent", "B-Theme", "I-Theme", "B-Time", "I-Time"]
    assert L2I["O"] == 0 and L2I["B-Agent"] == 1 and L2I["I-Time"] == 6


def test_build_args_input():
    combined, plen = build_args_input("The chef gave food.", "Giving", "gave")
    assert combined == "Giving | gave : The chef gave food."
    assert plen == len("Giving | gave : ")
    # a sentence offset maps into the sentence region of combined
    assert combined[plen:] == "The chef gave food."


# combined = "F | g : ab cd", prefix_len = 8, sentence "ab cd"
COMBINED = "F | g : ab cd"
PREFIX_LEN = 8
OFFSETS = [(0, 0), (0, 1), (2, 3), (4, 5), (6, 7), (8, 10), (11, 13), (0, 0)]


def test_align_fe_bio_marks_only_sentence_tokens():
    # Agent span "ab" at sentence [0,2) -> combined (8,10)
    fe_char_spans = [(8, 10, "Agent")]
    labels = align_fe_bio(OFFSETS, fe_char_spans, L2I, PREFIX_LEN, COMBINED)
    assert labels == [IGNORE_INDEX] * 5 + [L2I["B-Agent"], L2I["O"], IGNORE_INDEX]


def test_align_fe_bio_multi_token_span_gets_B_then_I():
    # a span covering both "ab"(8,10) and "cd"(11,13)
    fe_char_spans = [(8, 13, "Theme")]
    labels = align_fe_bio(OFFSETS, fe_char_spans, L2I, PREFIX_LEN, COMBINED)
    assert labels[5] == L2I["B-Theme"]
    assert labels[6] == L2I["I-Theme"]


def test_align_fe_bio_snaps_leading_space_offset():
    # DeBERTa quirk: the token for "cd" reports its start on the preceding space
    # (index 10, ' ') instead of 11. Without snapping, the span "cd" would drop
    # its only token and be lost. Snapping recovers it as B-Theme.
    offsets = [(0, 0), (0, 1), (2, 3), (4, 5), (6, 7), (8, 10), (10, 13), (0, 0)]
    fe_char_spans = [(11, 13, "Theme")]  # "cd" at sentence [3,5) -> combined (11,13)
    labels = align_fe_bio(offsets, fe_char_spans, L2I, PREFIX_LEN, COMBINED)
    assert labels[6] == L2I["B-Theme"]  # the leading-space token is NOT dropped


def test_decode_bio_spans_roundtrip():
    pred_ids = [0, 0, 0, 0, 0, L2I["B-Agent"], L2I["O"], 0]
    spans = decode_bio_spans(OFFSETS, pred_ids, I2L, PREFIX_LEN, COMBINED)
    assert spans == [("Agent", "ab")]


def test_decode_bio_spans_multitoken():
    pred_ids = [0, 0, 0, 0, 0, L2I["B-Theme"], L2I["I-Theme"], 0]
    spans = decode_bio_spans(OFFSETS, pred_ids, I2L, PREFIX_LEN, COMBINED)
    assert spans == [("Theme", "ab cd")]


def test_score_args_core_and_noncore_weighting():
    is_non_core = lambda fe: fe == "Time"  # noqa: E731
    gold = [("Agent", "he"), ("Time", "yesterday")]
    pred = [("Agent", "he")]  # missed the non-core Time
    tp, fp, fn = score_args(gold, pred, is_non_core)
    assert (tp, fp, fn) == (1.0, 0.0, 0.5)  # core hit 1.0, non-core miss 0.5


def test_score_args_false_positive():
    is_non_core = lambda fe: False  # noqa: E731
    gold = [("Agent", "he")]
    pred = [("Agent", "he"), ("Theme", "food")]  # extra wrong core FE
    tp, fp, fn = score_args(gold, pred, is_non_core)
    assert (tp, fp, fn) == (1.0, 1.0, 0.0)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("\ndone")
