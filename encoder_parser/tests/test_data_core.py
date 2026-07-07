"""Pure-Python unit tests for the trigger data core (no torch/transformers)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import (  # noqa: E402
    IGNORE_INDEX,
    LABEL2ID,
    TRIGGER_END,
    TRIGGER_START,
    align_trigger_labels,
    build_frame_input,
    frame_candidate_hint,
    mark_trigger,
    predicted_trigger_locs_from_tokens,
    prf1,
    score_trigger_words,
    trigger_word_indices,
    whitespace_words,
)

O, T = LABEL2ID["O"], LABEL2ID["TRIGGER"]


def test_whitespace_words_basic():
    text = "The chef gave food."
    assert whitespace_words(text) == [(0, 3), (4, 8), (9, 13), (14, 19)]


def test_whitespace_words_leading_and_multiple_spaces():
    text = "  a   bb "
    assert whitespace_words(text) == [(2, 3), (6, 8)]


def test_trigger_word_indices_matches_word_start():
    text = "The chef gave food."
    words = whitespace_words(text)
    # "gave" starts at char 9
    assert trigger_word_indices(words, [9]) == {2}
    # loc inside a word (not just start) still matches its word
    assert trigger_word_indices(words, [5]) == {1}


def test_align_trigger_labels_first_subword_scheme():
    # sentence "gave food" tokenized as: [CLS] ga ##ve food [SEP]
    offsets = [(0, 0), (0, 2), (2, 4), (5, 9), (0, 0)]
    word_ids = [None, 0, 0, 1, None]
    word_is_trigger = [True, False]  # word 0 = "gave" is a trigger
    labels = align_trigger_labels(offsets, word_ids, word_is_trigger)
    assert labels == [IGNORE_INDEX, T, IGNORE_INDEX, O, IGNORE_INDEX]


def test_predicted_locs_roundtrip():
    offsets = [(0, 0), (0, 2), (2, 4), (5, 9), (0, 0)]
    word_ids = [None, 0, 0, 1, None]
    # model fired TRIGGER on the first subword of word 0 (char start 0)
    preds = [False, True, False, False, False]
    assert predicted_trigger_locs_from_tokens(offsets, word_ids, preds) == {0}
    # a continuation-token prediction must NOT create a loc (only first subword)
    preds2 = [False, False, True, False, False]
    assert predicted_trigger_locs_from_tokens(offsets, word_ids, preds2) == set()


def test_score_trigger_words_perfect():
    text = "The chef gave food."
    tp, fp, fn = score_trigger_words(text, gold_trigger_locs=[9], pred_trigger_locs=[9])
    assert (tp, fp, fn) == (1, 0, 0)


def test_score_trigger_words_fp_and_fn():
    text = "The chef gave food."
    # gold trigger = "gave"(9); pred trigger = "chef"(4). one FP, one FN.
    tp, fp, fn = score_trigger_words(text, gold_trigger_locs=[9], pred_trigger_locs=[4])
    assert (tp, fp, fn) == (0, 1, 1)


def test_mark_trigger_basic():
    assert mark_trigger("The chef gave food.", 9) == "The chef <t> gave </t> food."


def test_mark_trigger_first_word():
    assert mark_trigger("Give it back.", 0) == "<t> Give </t> it back."


def test_frame_candidate_hint_joins_and_caps():
    assert frame_candidate_hint(["Giving", "Sending", "Commerce_sell"]) == \
        "Giving; Sending; Commerce_sell"
    assert frame_candidate_hint(["A", "B", "C"], max_cands=2) == "A; B"


def test_build_frame_input_with_and_without_hint():
    hint = frame_candidate_hint(["Giving", "Sending"])
    assert build_frame_input("The chef gave food.", 9, hint) == \
        "[Giving; Sending] : The chef <t> gave </t> food."
    # no hint -> plain marked sentence (back-compatible)
    assert build_frame_input("The chef gave food.", 9) == "The chef <t> gave </t> food."


def test_mark_trigger_snaps_off_space():
    # a loc landing on the space before a word still marks that word
    assert mark_trigger("a b c", 1) == "a <t> b </t> c"


def test_prf1():
    m = prf1(true_pos=3, false_pos=1, false_neg=1)
    assert abs(m["precision"] - 0.75) < 1e-9
    assert abs(m["recall"] - 0.75) < 1e-9
    assert abs(m["f1"] - 0.75) < 1e-9
    zero = prf1(0, 0, 0)
    assert zero == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
