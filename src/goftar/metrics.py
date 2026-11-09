"""Evaluation metrics (WER / CER) with Persian-aware normalisation."""

from __future__ import annotations

import difflib
import re
from collections.abc import Sequence
from dataclasses import dataclass

from goftar.core.config import PostprocessConfig
from goftar.text.normalize import ZWNJ, normalize_text

_NON_WORD = re.compile(r"[^\w\s]")
_SCORING_OPTS = PostprocessConfig(digits="persian", punctuation=False, half_space=False)


# Persian writers disagree on spacing ("کتاب ها" / "کتابها" / "کتاب‌ها", "می رود" / "میرود"),
# and that is not a recognition error. Affixes are glued to their word before comparing.
_SUFFIX_TOKENS = frozenset({"ها", "های", "هایی", "تر", "ترین", "ی"})
_PREFIX_TOKENS = frozenset({"می", "نمی"})


def _join_affixes(tokens: list[str]) -> list[str]:
    out: list[str] = []
    pending = ""
    for tok in tokens:
        if tok in _PREFIX_TOKENS:
            pending += tok
            continue
        tok, pending = pending + tok, ""
        if tok in _SUFFIX_TOKENS and out:
            out[-1] += tok
        else:
            out.append(tok)
    if pending:
        out.append(pending)
    return out


def tokenize(text: str) -> list[str]:
    """Normalise, drop punctuation, ignore spacing around affixes, split into words."""
    text = normalize_text(text, _SCORING_OPTS).replace(ZWNJ, " ")
    return _join_affixes(_NON_WORD.sub(" ", text).lower().split())


def edit_distance(a: Sequence, b: Sequence) -> int:
    """Levenshtein distance (insert / delete / substitute, each cost 1)."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        current = [i]
        for j, y in enumerate(b, start=1):
            cost = 0 if x == y else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = tokenize(reference), tokenize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = list(" ".join(tokenize(reference))), list(" ".join(tokenize(hypothesis)))
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


@dataclass(frozen=True)
class ErrorSpan:
    """A run of wrong words; start/end index the reference words (start == end: insertion)."""

    start: int
    end: int
    reference: str
    heard: str


def error_spans(reference: str, hypothesis: str) -> list[ErrorSpan]:
    ref, hyp = tokenize(reference), tokenize(hypothesis)
    matcher = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    return [
        ErrorSpan(i1, i2, " ".join(ref[i1:i2]), " ".join(hyp[j1:j2]))
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    ]


def word_errors(reference: str, hypothesis: str) -> list[tuple[str, str]]:
    """Word-level differences as (reference words, heard words); '' means nothing on that side."""
    return [(e.reference, e.heard) for e in error_spans(reference, hypothesis)]
