"""
End-to-end FrameNet parser: raw text -> frame annotations.

Chains the three encoder heads on a sentence:
  1. trigger identification (token classification) -> trigger word locations
  2. frame classification (marker-pooled, candidate-masked) per trigger
  3. argument extraction (detect-then-classify, FE-masked) per (trigger, frame)

The operating points (frame candidate bias, args NULL bias) default to the
dev-selected values from Milestone 3.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from . import weights
from .args2_data import NULL_ROLE, decode_detect_spans
from .args_data import _clean_span_text, build_args_input, frame_fe_hint
from .data import (
    LABEL2ID,
    TRIGGER_END,
    TRIGGER_START,
    mark_trigger,
    predicted_trigger_locs_from_tokens,
    snap_to_word_start,
    whitespace_words,
)
from .frame2_data import find_marker_positions
from .lexicon import Lexicon

DEFAULT_TRIGGER_REPO = "texturejc/texture-frames-trigger"
DEFAULT_FRAME_REPO = "texturejc/texture-frames-frame"
DEFAULT_ARGS_REPO = "texturejc/texture-frames-args"


@dataclass
class Argument:
    role: str
    text: str
    start: int  # char offset in the sentence (-1 if not locatable)
    end: int


@dataclass
class FrameAnnotation:
    trigger: str
    trigger_loc: int
    frame: str
    arguments: list = field(default_factory=list)


def _ensure_nltk():
    import nltk

    for pkg, path in [
        ("framenet_v17", "corpora/framenet_v17"),
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


def _word_at(text: str, loc: int) -> str:
    loc = snap_to_word_start(text, loc)
    for s, e in whitespace_words(text):
        if s <= loc < e:
            return text[s:e]
    return ""


class FrameParser:
    """Load once, then call `.parse(text)`. Models download from the HF Hub on
    first construction and are cached by huggingface_hub."""

    def __init__(
        self,
        device: str | None = None,
        trigger_repo: str = DEFAULT_TRIGGER_REPO,
        frame_repo: str = DEFAULT_FRAME_REPO,
        args_repo: str = DEFAULT_ARGS_REPO,
        frame_bias: float = 7.0,
        null_bias: float = 2.0,
        max_length: int = 320,
    ):
        _ensure_nltk()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.frame_bias = frame_bias
        self.null_bias = null_bias

        self.lexicon = Lexicon()
        self.trigger_model, self.trigger_tok = weights.load_trigger(trigger_repo, self.device)
        self.frame_model, self.frame_tok, self.frame2id, self.id2frame = weights.load_frame(
            frame_repo, self.device
        )
        self.args_model, self.args_tok, self.role2id, self.id2role = weights.load_args(
            args_repo, self.device
        )
        self._frame_start = self.frame_tok.convert_tokens_to_ids(TRIGGER_START)
        self._frame_end = self.frame_tok.convert_tokens_to_ids(TRIGGER_END)
        self._null_id = self.role2id[NULL_ROLE]

    # -- public ------------------------------------------------------------ #
    @torch.no_grad()
    def parse(self, text: str) -> list[FrameAnnotation]:
        annotations = []
        for loc in self._triggers(text):
            frame = self._frame(text, loc)
            args = self._args(text, loc, frame)
            annotations.append(
                FrameAnnotation(
                    trigger=_word_at(text, loc), trigger_loc=loc, frame=frame, arguments=args
                )
            )
        return annotations

    # -- stages ------------------------------------------------------------ #
    def _triggers(self, text: str) -> list[int]:
        enc = self.trigger_tok(
            text, truncation=True, max_length=self.max_length,
            return_offsets_mapping=True, return_tensors="pt",
        )
        word_ids = enc.word_ids()
        logits = self.trigger_model(
            input_ids=enc["input_ids"].to(self.device),
            attention_mask=enc["attention_mask"].to(self.device),
        ).logits[0]
        is_trig = [p == LABEL2ID["TRIGGER"] for p in logits.argmax(-1).tolist()]
        offsets = enc["offset_mapping"][0].tolist()
        return sorted(predicted_trigger_locs_from_tokens(offsets, word_ids, is_trig))

    def _frame(self, text: str, loc: int) -> str:
        enc = self.frame_tok(
            mark_trigger(text, loc), truncation=True, max_length=self.max_length,
            return_tensors="pt",
        )
        sp, ep = find_marker_positions(enc["input_ids"][0].tolist(), self._frame_start, self._frame_end)
        logits = self.frame_model.encode_logits(
            enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device),
            torch.tensor([sp], device=self.device), torch.tensor([ep], device=self.device),
        )[0]
        cand_ids = [self.frame2id[c] for c in self.lexicon.candidate_frames(text, loc)
                    if c in self.frame2id]
        if cand_ids:
            logits = logits.clone()
            logits[torch.tensor(cand_ids, device=self.device)] += self.frame_bias
        return self.id2frame[int(logits.argmax())]

    def _allowed_role_ids(self, frame: str):
        core, non_core = self.lexicon.frame_elements(frame)
        allowed = {self._null_id}
        for fe in [*core, *non_core]:
            if fe in self.role2id:
                allowed.add(self.role2id[fe])
        return allowed

    def _args(self, text: str, loc: int, frame: str) -> list[Argument]:
        hint = frame_fe_hint(self.lexicon, frame)
        combined, prefix_len, _, _ = build_args_input(text, frame, loc, hint)
        enc = self.args_tok(
            combined, truncation=True, max_length=self.max_length,
            return_offsets_mapping=True, return_tensors="pt",
        )
        offsets = enc["offset_mapping"][0].tolist()
        hidden, detect_logits = self.args_model.encode(
            enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)
        )
        spans = decode_detect_spans(offsets, detect_logits[0].argmax(-1).tolist(), prefix_len)
        if not spans:
            return []

        mask = torch.full((len(self.role2id),), float("-inf"), device=self.device)
        mask[list(self._allowed_role_ids(frame))] = 0.0
        role_logits = self.args_model.role_logits_for_spans(
            hidden, [(0, s, e) for (s, e, _, _) in spans]
        ) + mask
        role_logits[:, self._null_id] += self.null_bias

        args = []
        for (_, _, cs, ce), r in zip(spans, role_logits.argmax(-1).tolist()):
            if r == self._null_id:
                continue
            span_text = _clean_span_text(combined[cs:ce])
            start = text.find(span_text)
            args.append(Argument(
                role=self.id2role[r], text=span_text,
                start=start, end=start + len(span_text) if start >= 0 else -1,
            ))
        return args
