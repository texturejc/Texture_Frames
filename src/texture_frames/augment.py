"""
Lightweight synonym-replacement augmentation for argument extraction (recovers the
long-tail edge the baseline gets from nlpaug, without the dependency). WordNet only
(nltk — already downloaded for the lexicon).

The hard part is that FrameNet annotations are CHARACTER offsets, so swapping a word
shifts everything downstream. `augment_example` replaces non-trigger content words
and remaps the trigger loc + every FE span onto the rebuilt sentence. The offset
remapping is pure (inject a synonym_fn) and unit-tested; `wordnet_synonym` is the
nltk-backed picker used at dataset-build time.
"""
from __future__ import annotations

from .data import snap_to_word_start, whitespace_words


def _rebuild(word_strs: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Join words with single spaces; return (text, [(start, end) per word])."""
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    pos = 0
    for i, w in enumerate(word_strs):
        if i > 0:
            parts.append(" ")
            pos += 1
        spans.append((pos, pos + len(w)))
        parts.append(w)
        pos += len(w)
    return "".join(parts), spans


def _match_case(src: str, repl: str) -> str:
    return repl.capitalize() if src[:1].isupper() else repl


def augment_example(text, trigger_loc, fes, synonym_fn, rng, p_replace: float = 0.3):
    """Return (new_text, new_trigger_loc, new_fes) with some non-trigger content
    words swapped for synonyms, or None if nothing was replaced. `fes` is
    [(name, start, end), ...]; synonym_fn(word, rng) -> str|None."""
    words = whitespace_words(text)
    if not words:
        return None
    word_strs = [text[s:e] for s, e in words]

    tloc = snap_to_word_start(text, trigger_loc)
    trig_idx = next((i for i, (s, e) in enumerate(words) if s <= tloc < e), None)

    new_strs = list(word_strs)
    changed = False
    for i, w in enumerate(word_strs):
        if i == trig_idx or not any(c.isalpha() for c in w):
            continue  # never touch the predicate or pure punctuation
        if rng.random() > p_replace:
            continue
        syn = synonym_fn(w, rng)
        if syn and syn.lower() != w.lower():
            new_strs[i] = _match_case(w, syn)
            changed = True
    if not changed:
        return None

    new_text, new_spans = _rebuild(new_strs)
    new_loc = new_spans[trig_idx][0] if trig_idx is not None else 0
    new_fes = []
    for name, s, e in fes:
        covered = [i for i, (ws, we) in enumerate(words) if ws < e and we > s]
        if covered:
            new_fes.append((name, new_spans[covered[0]][0], new_spans[covered[-1]][1]))
    return new_text, new_loc, new_fes


def wordnet_synonym(word: str, rng):
    """A random single-word WordNet synonym (≠ the word), or None."""
    from nltk.corpus import wordnet as wn

    syns = set()
    for syn in wn.synsets(word.lower()):
        for lemma in syn.lemmas():
            name = lemma.name()
            if "_" not in name and name.isalpha() and name.lower() != word.lower():
                syns.add(name)
    if not syns:
        return None
    return rng.choice(sorted(syns))
