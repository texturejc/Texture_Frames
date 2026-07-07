"""Pure-Python tests for args2_data (no torch/transformers/nltk)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from args2_data import (  # noqa: E402
    DETECT_B,
    DETECT_I,
    DETECT_O,
    NULL_ROLE,
    decode_detect_spans,
    detect_bio_labels,
    gold_span_token_indices,
    role_label_maps,
)
from args_data import IGNORE_INDEX  # noqa: E402

# combined = "F [x] : ab cd", sentence "ab cd" starts at prefix_len.
# indices: F0 sp1 [2 x3 ]4 sp5 :6 sp7 a8 b9 sp10 c11 d12
COMBINED = "F [x] : ab cd"
PREFIX_LEN = 8
# tokens: [CLS], "F", "[x]", ":", "ab"(8,10), "cd"(11,13), [SEP]
OFFSETS = [(0, 0), (0, 1), (2, 5), (6, 7), (8, 10), (11, 13), (0, 0)]


def test_role_label_maps():
    roles, r2i, i2r = role_label_maps(["Donor", "Theme"])
    assert roles == [NULL_ROLE, "Donor", "Theme"]
    assert r2i[NULL_ROLE] == 0 and r2i["Donor"] == 1
    assert i2r[2] == "Theme"


def test_detect_bio_prefix_ignored_and_bio():
    # one span covering "ab cd" -> combined (8,13)
    labels = detect_bio_labels(OFFSETS, [(8, 13)], PREFIX_LEN, COMBINED)
    assert labels == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX,
                      DETECT_B, DETECT_I, IGNORE_INDEX]


def test_detect_bio_two_adjacent_spans_each_get_B():
    # "ab" and "cd" are two separate spans -> B, B (not B, I)
    labels = detect_bio_labels(OFFSETS, [(8, 10), (11, 13)], PREFIX_LEN, COMBINED)
    assert labels[4] == DETECT_B and labels[5] == DETECT_B


def test_detect_bio_snaps_leading_space():
    # "cd" token reports start on the preceding space (index 10) — must still tag it
    offs = [(0, 0), (0, 1), (2, 5), (6, 7), (8, 10), (10, 13), (0, 0)]
    labels = detect_bio_labels(offs, [(11, 13)], PREFIX_LEN, COMBINED)
    assert labels[5] == DETECT_B  # not dropped


def test_gold_span_token_indices():
    spans = gold_span_token_indices(OFFSETS, [(8, 13, "Theme")], PREFIX_LEN, COMBINED)
    assert spans == [(4, 5, "Theme")]  # tokens 4..5 cover "ab cd"


def test_gold_span_token_indices_single_token():
    spans = gold_span_token_indices(OFFSETS, [(8, 10, "Donor")], PREFIX_LEN, COMBINED)
    assert spans == [(4, 4, "Donor")]


def test_decode_detect_spans_roundtrip():
    pred = [DETECT_O, DETECT_O, DETECT_O, DETECT_O, DETECT_B, DETECT_I, DETECT_O]
    spans = decode_detect_spans(OFFSETS, pred, PREFIX_LEN)
    assert spans == [(4, 5, 8, 13)]  # start_tok, end_tok, char_start, char_end


def test_decode_detect_spans_two_spans():
    pred = [DETECT_O, DETECT_O, DETECT_O, DETECT_O, DETECT_B, DETECT_B, DETECT_O]
    spans = decode_detect_spans(OFFSETS, pred, PREFIX_LEN)
    assert spans == [(4, 4, 8, 10), (5, 5, 11, 13)]


def test_decode_detect_spans_stray_I_opens_span():
    pred = [DETECT_O, DETECT_O, DETECT_O, DETECT_O, DETECT_I, DETECT_I, DETECT_O]
    spans = decode_detect_spans(OFFSETS, pred, PREFIX_LEN)
    assert spans == [(4, 5, 8, 13)]  # I without B still yields a span


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("\ndone")
