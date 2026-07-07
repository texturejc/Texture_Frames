"""Pure tests for the augmentation offset remapping (inject a deterministic
synonym_fn; no nltk/torch)."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from augment import _rebuild, augment_example  # noqa: E402


def _fake_syn(word, rng):
    return {"chef": "cook", "food": "meal", "strongest": "toughest"}.get(word.lower())


def test_rebuild_spans():
    text, spans = _rebuild(["The", "cook", "gave"])
    assert text == "The cook gave"
    assert spans == [(0, 3), (4, 8), (9, 13)]


def test_augment_remaps_trigger_and_fes():
    # "The chef gave food ." — trigger "gave" @9; Donor="chef"[4,8], Theme="food"[14,18]
    text = "The chef gave food ."
    fes = [("Donor", 4, 8), ("Theme", 14, 18)]
    out = augment_example(text, 9, fes, _fake_syn, random.Random(0), p_replace=1.0)
    assert out is not None
    new_text, new_loc, new_fes = out
    assert new_text == "The cook gave meal ."
    assert new_text[new_loc : new_loc + 4] == "gave"          # trigger preserved
    assert [(n, new_text[s:e]) for n, s, e in new_fes] == [
        ("Donor", "cook"), ("Theme", "meal"),                  # spans remapped
    ]


def test_augment_never_replaces_trigger():
    # even if the trigger word has a synonym, it must stay put
    text = "The chef gave food ."
    out = augment_example(text, 9, [("Theme", 14, 18)],
                          lambda w, rng: "handed" if w == "gave" else _fake_syn(w, rng),
                          random.Random(0), p_replace=1.0)
    new_text, new_loc, _ = out
    assert new_text[new_loc : new_loc + 4] == "gave"


def test_augment_returns_none_when_no_synonyms():
    out = augment_example("a b c", 0, [("X", 2, 3)], lambda w, rng: None,
                          random.Random(0), p_replace=1.0)
    assert out is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("\ndone")
