"""Persian-aware text clean-up for raw ASR output.

Whisper's Persian output commonly has Arabic letter variants, inconsistent digits, missing
half-spaces (ZWNJ) and Latin punctuation. Everything here is pure Python (no extra deps);
`use_hazm` optionally runs hazm's normaliser first when it is installed.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace

from goftar.core.config import PostprocessConfig
from goftar.core.types import Transcript

log = logging.getLogger(__name__)

ZWNJ = "‌"
_PERSIAN_LETTERS = "؀-ۿ"

_ARABIC_LETTERS = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه"})
_DIACRITICS = re.compile("[ً-ٰٟـ]")  # harakat, superscript alef, tatweel

_DIGITS_PERSIAN = "۰۱۲۳۴۵۶۷۸۹"
_TO_PERSIAN = str.maketrans("0123456789٠١٢٣٤٥٦٧٨٩", _DIGITS_PERSIAN * 2)
_TO_LATIN = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "0123456789" * 2)

# "می رود" -> "می‌رود", "نمی دانم" -> "نمی‌دانم"
_MI_PREFIX = re.compile(rf"(?<![{_PERSIAN_LETTERS}{ZWNJ}])(ن?می) (?=[{_PERSIAN_LETTERS}])")
# "کتاب ها" -> "کتاب‌ها", "بزرگ تر" -> "بزرگ‌تر"
_SUFFIXES = re.compile(
    rf"(?<=[{_PERSIAN_LETTERS}]) (ها|های|هایی|هایم|هایت|هایش|تر|ترین)(?![{_PERSIAN_LETTERS}])"
)

_Q_MARK = re.compile(rf"(?<=[{_PERSIAN_LETTERS}])[^\S\n]*\?")
_COMMA = re.compile(rf"(?<=[{_PERSIAN_LETTERS}])[^\S\n]*,(?=\s|$)")
_SEMICOLON = re.compile(rf"(?<=[{_PERSIAN_LETTERS}])[^\S\n]*;(?=\s|$)")
_SPACE_BEFORE_PUNCT = re.compile(r"[^\S\n]+([،؛؟!])")
_SPACE_BEFORE_COLON_DOT = re.compile(r"[^\S\n]+([.:])(?=\s|$)")
_NO_SPACE_AFTER = re.compile(r"([،؛؟!])(?=[^\s\d،؛؟!.…\"»)\]])")
_DOT_BETWEEN_LETTERS = re.compile(rf"(?<=[{_PERSIAN_LETTERS}])\.(?=[{_PERSIAN_LETTERS}])")

_MULTI_ZWNJ = re.compile(f"{ZWNJ}{{2,}}")
_ZWNJ_AT_EDGE = re.compile(rf"(?<=\s){ZWNJ}|{ZWNJ}(?=\s)|^{ZWNJ}|{ZWNJ}$")
_SPACES = re.compile(r"[^\S\n]+")


def _hazm_normalize(text: str) -> str:
    try:
        from hazm import Normalizer
    except ImportError:
        log.warning("postprocess.use_hazm is set but hazm is not installed; skipping")
        return text
    return Normalizer().normalize(text)


def normalize_text(text: str, opts: PostprocessConfig | None = None) -> str:
    """Normalise one piece of Persian text. Idempotent: normalising twice changes nothing."""
    opts = opts or PostprocessConfig()
    if not opts.enabled or not text:
        return text

    text = text.replace(" ", " ")
    if opts.use_hazm:
        text = _hazm_normalize(text)
    if opts.fix_arabic_chars:
        text = text.translate(_ARABIC_LETTERS)
    if opts.remove_diacritics:
        text = _DIACRITICS.sub("", text)

    if opts.digits == "persian":
        text = text.translate(_TO_PERSIAN)
    elif opts.digits == "latin":
        text = text.translate(_TO_LATIN)

    if opts.punctuation:
        text = _Q_MARK.sub("؟", text)
        text = _COMMA.sub("،", text)
        text = _SEMICOLON.sub("؛", text)
        text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
        text = _SPACE_BEFORE_COLON_DOT.sub(r"\1", text)
        text = _NO_SPACE_AFTER.sub(r"\1 ", text)
        text = _DOT_BETWEEN_LETTERS.sub(". ", text)

    if opts.half_space:
        text = _MI_PREFIX.sub(rf"\1{ZWNJ}", text)
        text = _SUFFIXES.sub(rf"{ZWNJ}\1", text)

    text = _MULTI_ZWNJ.sub(ZWNJ, text)
    text = _ZWNJ_AT_EDGE.sub("", text)
    text = _SPACES.sub(" ", text)
    return text.strip()


def postprocess_transcript(transcript: Transcript, opts: PostprocessConfig) -> Transcript:
    """Return a copy of the transcript with every segment normalised (empty ones dropped)."""
    if not opts.enabled:
        return transcript
    segments = []
    for seg in transcript.segments:
        text = normalize_text(seg.text, opts)
        if text:
            segments.append(replace(seg, text=text))
    return replace(transcript, segments=segments)
