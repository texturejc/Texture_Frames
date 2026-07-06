"""Exercise data.load_trigger_sentences against a mocked FrameNet corpus.

Verifies the parsing (split filtering, trigger-loc extraction, broken-annotation
drop, no-trigger skip) WITHOUT needing nltk or the real corpus — and proves the
code path imports no `frame_semantic_transformer` / `nlpaug`.
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# A test-split doc (KBEval__atm.xml ∈ SESAME_TEST_FILES) and a non-split doc.
FAKE_DOCS = [
    {
        "filename": "KBEval__atm.xml",
        "sentence": [
            {  # "The chef gave food." — trigger "gave" at char 9
                "text": "The chef gave food.",
                "annotationSet": [
                    {"FE": [[]], "Target": [[9, 13]], "frame": {"name": "Giving"}},
                ],
            },
            {  # two triggers in one sentence
                "text": "She left and returned.",
                "annotationSet": [
                    {"FE": [[]], "Target": [[4, 8]], "frame": {"name": "Departing"}},
                    {"FE": [[]], "Target": [[13, 21]], "frame": {"name": "Arriving"}},
                ],
            },
            {  # broken annotation: loc past end of text -> whole sentence dropped
                "text": "short",
                "annotationSet": [
                    {"FE": [[]], "Target": [[99, 104]], "frame": {"name": "X"}},
                ],
            },
            {  # no valid frame annotation -> skipped
                "text": "No frames here.",
                "annotationSet": [{"Target": [[0, 2]]}],  # missing FE/frame keys
            },
        ],
    },
    {  # a doc NOT in the test split -> excluded when split="test"
        "filename": "SomeOtherDoc.xml",
        "sentence": [
            {
                "text": "Ignore me.",
                "annotationSet": [
                    {"FE": [[]], "Target": [[0, 6]], "frame": {"name": "Y"}},
                ],
            }
        ],
    },
]


def _install_fake_nltk():
    fake_nltk = types.ModuleType("nltk")
    fake_nltk.data = types.SimpleNamespace(find=lambda p: True, download=lambda p: None)
    fake_corpus = types.ModuleType("nltk.corpus")
    fake_corpus.framenet = types.SimpleNamespace(docs=lambda: FAKE_DOCS)
    sys.modules["nltk"] = fake_nltk
    sys.modules["nltk.corpus"] = fake_corpus


def test_load_trigger_sentences_test_split():
    _install_fake_nltk()
    import data

    rows = data.load_trigger_sentences("test")
    by_text = dict(rows)

    # non-test doc excluded
    assert "Ignore me." not in by_text
    # broken-annotation sentence dropped
    assert "short" not in by_text
    # no-frame sentence skipped
    assert "No frames here." not in by_text
    # valid sentences kept with correct trigger locs
    assert by_text["The chef gave food."] == [9]
    assert by_text["She left and returned."] == [4, 13]
    assert len(rows) == 2

    # and confirm we never dragged in the augmentation stack
    assert "nlpaug" not in sys.modules
    assert not any(m.startswith("frame_semantic_transformer") for m in sys.modules)


if __name__ == "__main__":
    test_load_trigger_sentences_test_split()
    print("ok  test_load_trigger_sentences_test_split")
    print("\n1 passed")
