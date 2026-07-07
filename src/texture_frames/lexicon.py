"""
FrameNet 1.7 lexicon: trigger word -> candidate frames, and the frame vocabulary.

Vendored faithfully from the upstream parser so the candidate sets are IDENTICAL
to what the baseline used (essential for a fair comparison):
  * frame loading + LU normalization: Framenet17InferenceLoader
  * bigram lookup map + candidate lookup: LoaderDataCache
  * trigger_bigrams: FrameClassificationTask

Reimplemented here (rather than imported) because every import path into
`frame_semantic_transformer` transitively pulls in the augmentation modules and
thus `nlpaug`. This module needs only nltk (framenet + wordnet + stemmers).
"""
from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache
from itertools import product
from typing import Iterable

LOW_PRIORITY_LONGER_LUS = {"back", "down", "make", "take", "have", "into", "come"}
WORDNET_LEMMATIZER_POS = ["a", "r", "n", "v", "s"]


def normalize_name(name: str) -> str:
    """Lowercase, drop underscores — matches upstream LoaderDataCache.normalize_name."""
    return name.lower().replace("_", "")


class Lexicon:
    """Builds (once, cached) the FrameNet frame list + LU-bigram -> frames map."""

    def __init__(self) -> None:
        self._stemmers = None
        self._lemmatizer = None

    # -- nltk setup -------------------------------------------------------- #

    def setup(self) -> None:
        import nltk

        for pkg, path in [
            ("framenet_v17", "corpora/framenet_v17"),
            ("wordnet", "corpora/wordnet"),
            ("omw-1.4", "corpora/omw-1.4"),
        ]:
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(pkg)

        from nltk.stem import (
            LancasterStemmer,
            PorterStemmer,
            SnowballStemmer,
            WordNetLemmatizer,
        )

        self._stemmers = [
            PorterStemmer(),
            LancasterStemmer(),
            SnowballStemmer("english"),
        ]
        self._lemmatizer = WordNetLemmatizer()

    # -- LU normalization (verbatim from Framenet17InferenceLoader) -------- #

    def normalize_lexical_unit_text(self, lu: str) -> set[str]:
        if self._lemmatizer is None:
            self.setup()
        normalized_lu = lu.lower()
        normalized_lu = re.sub(r"\.[a-zA-Z]+$", "", normalized_lu)
        normalized_lu = re.sub(r"[^a-z0-9 ]", "", normalized_lu)
        normalized_lu = normalized_lu.strip()
        norm_lus = {stemmer.stem(normalized_lu) for stemmer in self._stemmers}
        for pos in WORDNET_LEMMATIZER_POS:
            norm_lus.add(self._lemmatizer.lemmatize(normalized_lu, pos=pos))
        return norm_lus

    def prioritize_lexical_unit(self, lu: str) -> bool:
        norm_lu = self.normalize_lexical_unit_text(lu)
        return len(norm_lu) >= 4 and norm_lu not in LOW_PRIORITY_LONGER_LUS

    def _normalize_ngram(self, ngram: list[str]) -> set[str]:
        norm_toks = [self.normalize_lexical_unit_text(tok) for tok in ngram]
        return {"_".join(combo) for combo in product(*norm_toks)}

    # -- frames + lookup map ---------------------------------------------- #

    @lru_cache(1)
    def frames(self) -> list[dict]:
        """[{name, lexical_units, core_elements, non_core_elements}, ...]."""
        self.setup()
        from nltk.corpus import framenet as fn

        out = []
        for raw in fn.frames():
            out.append(
                {
                    "name": raw.name,
                    "lexical_units": list(raw.lexUnit.keys()),
                    "core_elements": [
                        n for (n, fe) in raw.FE.items() if fe.coreType == "Core"
                    ],
                    "non_core_elements": [
                        n for (n, fe) in raw.FE.items() if fe.coreType != "Core"
                    ],
                }
            )
        return out

    @lru_cache(1)
    def _frames_by_name(self) -> dict[str, dict]:
        return {f["name"]: f for f in self.frames()}

    @lru_cache(1)
    def fe_vocab(self) -> list[str]:
        """Sorted unique frame-element names across all frames (the BIO role space)."""
        names: set[str] = set()
        for f in self.frames():
            names.update(f["core_elements"])
            names.update(f["non_core_elements"])
        return sorted(names)

    def frame_elements(self, frame_name: str) -> tuple[list[str], list[str]]:
        """(core_elements, non_core_elements) for a frame."""
        f = self._frames_by_name().get(frame_name)
        if f is None:
            return [], []
        return f["core_elements"], f["non_core_elements"]

    def is_non_core(self, frame_name: str, fe_name: str) -> bool:
        """Sesame scoring: non-core FEs count 0.5, core 1.0."""
        f = self._frames_by_name().get(frame_name)
        return bool(f and fe_name in f["non_core_elements"])

    @lru_cache(1)
    def frame_vocab(self) -> list[str]:
        """Sorted list of all frame names — the classification label space."""
        return sorted(f["name"] for f in self.frames())

    @lru_cache(1)
    def frame2id(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.frame_vocab())}

    @lru_cache(1)
    def _bigram_to_frames(self) -> dict[str, list[str]]:
        uniq: dict[str, set[str]] = defaultdict(set)
        for frame in self.frames():
            for lu in frame["lexical_units"]:
                parts = lu.split()
                lu_bigrams: list[str] = []
                prev = None
                for part in parts:
                    if len(parts) == 1 or self.prioritize_lexical_unit(part):
                        for norm_part in self._normalize_ngram([part]):
                            lu_bigrams.append(norm_part)
                    if prev is not None:
                        for norm_parts in self._normalize_ngram([prev, part]):
                            lu_bigrams.append(norm_parts)
                    prev = part
                for bigram in lu_bigrams:
                    uniq[bigram].add(frame["name"])
        return {k: sorted(v) for k, v in uniq.items()}

    def possible_frames_for_bigrams(self, bigrams: list[list[str]]) -> list[str]:
        lookup = self._bigram_to_frames()
        possible: list[str] = []
        for bigram in bigrams:
            for norm in sorted(self._normalize_ngram(bigram)):
                if norm in lookup:
                    possible += lookup[norm]
        return list(dict.fromkeys(possible))  # dedupe, preserve order

    def candidate_frames(self, text: str, trigger_loc: int) -> list[str]:
        return self.possible_frames_for_bigrams(trigger_bigrams(text, trigger_loc))


def trigger_bigrams(text: str, trigger_loc: int) -> list[list[str]]:
    """prev+trigger, trigger+next, and trigger monogram — verbatim from
    FrameClassificationTask.trigger_bigrams."""
    pre_tokens = text[:trigger_loc].split()
    after_tokens = text[trigger_loc:].split()
    trigger = after_tokens[0]
    post_tokens = after_tokens[1:]
    bigrams: list[list[str]] = []
    if pre_tokens:
        bigrams.append([pre_tokens[-1], trigger])
    if post_tokens:
        bigrams.append([trigger, post_tokens[0]])
    bigrams.append([trigger])
    return bigrams
