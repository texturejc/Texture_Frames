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
    frame_fe_hint,
    remap_fe_span,
    score_args,
)


class _FakeLexicon:
    def frame_elements(self, frame):
        return (["Donor", "Recipient"], ["Time", "Place"])  # core, non_core

FE_VOCAB = ["Agent", "Theme", "Time"]
LABELS, L2I, I2L = fe_label_maps(FE_VOCAB)


def test_fe_label_maps():
    assert LABELS == ["O", "B-Agent", "I-Agent", "B-Theme", "I-Theme", "B-Time", "I-Time"]
    assert L2I["O"] == 0 and L2I["B-Agent"] == 1 and L2I["I-Time"] == 6


def test_build_args_input_marks_trigger():
    # "gave" is the word at loc 9 in "The chef gave food."
    combined, plen, ts, te = build_args_input("The chef gave food.", "Giving", 9)
    assert combined == "Giving : The chef <t> gave </t> food."
    assert plen == len("Giving : ")
    assert (ts, te) == (9, 13)  # trigger span in the ORIGINAL text


def test_frame_fe_hint_lists_core_then_noncore():
    assert frame_fe_hint(_FakeLexicon(), "Giving") == "Donor; Recipient; Time; Place"
    assert frame_fe_hint(_FakeLexicon(), "Giving", max_fes=2) == "Donor; Recipient"


def test_build_args_input_with_fe_hint():
    hint = frame_fe_hint(_FakeLexicon(), "Giving")
    combined, plen, ts, te = build_args_input("The chef gave food.", "Giving", 9, hint)
    assert combined == "Giving [Donor; Recipient; Time; Place] : The chef <t> gave </t> food."
    assert plen == len(f"Giving [{hint}] : ")
    assert (ts, te) == (9, 13)
    # gold FE offsets still remap correctly with the longer prefix
    s, e = remap_fe_span(0, 8, ts, te, plen)
    assert combined[s:e] == "The chef"


def test_remap_fe_span_around_trigger():
    # original "The chef gave food.", trigger "gave" = (9, 13), prefix "Giving : "
    ts, te, plen = 9, 13, len("Giving : ")
    combined, *_ = build_args_input("The chef gave food.", "Giving", 9)
    # FE before the trigger: "The chef" [0,8) -> lands before <t>, text preserved
    s, e = remap_fe_span(0, 8, ts, te, plen)
    assert combined[s:e] == "The chef"
    # FE after the trigger: "food" [14,18) -> lands after </t>, text preserved
    s, e = remap_fe_span(14, 18, ts, te, plen)
    assert combined[s:e] == "food"


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


def test_decode_bio_spans_strips_predicate_markers():
    # a predicted span that abuts a marker: "<t> gave" should clean to "gave"
    combined = "Giving : The chef <t> gave </t> food."
    # token "gave" at combined[22:26]; include the marker token to prove stripping
    offsets = [(0, 0), (0, 6), (7, 8), (9, 12), (13, 17), (18, 21), (22, 26), (0, 0)]
    plen = len("Giving : ")
    # label the <t> token and the gave token both as B/I-Agent
    pred_ids = [0, 0, 0, 0, 0, L2I["B-Agent"], L2I["I-Agent"], 0]
    spans = decode_bio_spans(offsets, pred_ids, I2L, plen, combined)
    assert spans == [("Agent", "gave")]  # marker text removed, whitespace collapsed


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
