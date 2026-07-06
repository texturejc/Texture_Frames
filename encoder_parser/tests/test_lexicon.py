"""Pure-Python tests for the lexicon's trigger_bigrams (no nltk needed).

Locks the bigram construction to upstream FrameClassificationTask.trigger_bigrams
so candidate-frame lookups stay identical to the baseline's.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lexicon import trigger_bigrams  # noqa: E402


def test_trigger_bigrams_middle_word():
    # "The chef gave food." — trigger "gave" at char 9
    assert trigger_bigrams("The chef gave food.", 9) == [
        ["chef", "gave"],
        ["gave", "food."],
        ["gave"],
    ]


def test_trigger_bigrams_first_word():
    # no preceding token -> no prev+trigger bigram
    assert trigger_bigrams("Give it back.", 0) == [["Give", "it"], ["Give"]]


def test_trigger_bigrams_last_word():
    # no following token -> no trigger+next bigram
    assert trigger_bigrams("She left", 4) == [["She", "left"], ["left"]]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("\n3 passed")
