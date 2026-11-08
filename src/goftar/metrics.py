"""Evaluation metrics (WER / CER) with Persian-aware normalisation."""

from __future__ import annotations

import re
from collections.abc import Sequence

from goftar.core.config import PostprocessConfig
from goftar.text.normalize import ZWNJ, normalize_text

_NON_WORD = re.compile(r"[^\w\s]")
_SCORING_OPTS = PostprocessConfig(digits="persian", punctuation=False, half_space=False)


def tokenize(text: str) -> list[str]:
    """Normalise, drop punctuation, treat half-space like a space, split into words."""
    text = normalize_text(text, _SCORING_OPTS).replace(ZWNJ, " ")
    return _NON_WORD.sub(" ", text).lower().split()


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
