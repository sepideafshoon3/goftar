"""Persian number words -> digits ("سه هزار و پونصد" -> "3500").

Persian fine-tunes of Whisper often write numbers as words, while the reference style (and most
written Persian) uses digits. Only clear compound numbers are converted: parts joined by "و" in
falling order of size ("صد و بیست و پنج"), or a number with a scale word ("سه هزار").

Single words stay words ("یه", "دو تا", "بیست تا", "صد درصد"): they are often written that way,
and many have other meanings ("نه" no, "ده" village, "سی" the letter C, "بیست" the English
"based"). A run that goes on with "و" and an unknown word ("بیست و هشتا", "سی و پنجم") is also
left alone. Values of a million or more keep their scale words:
"سه میلیون و پونصد هزار" -> "3 میلیون و 500 هزار".
Digits come out as Latin; the digits step of the normaliser converts them afterwards.
"""

from __future__ import annotations

import re

_UNITS = {
    "صفر": 0, "یک": 1, "یه": 1, "دو": 2, "سه": 3, "چهار": 4, "چار": 4, "پنج": 5,
    "شش": 6, "شیش": 6, "هفت": 7, "هف": 7, "هشت": 8, "هش": 8, "نه": 9,
}  # fmt: skip
_TEENS = {
    "ده": 10, "یازده": 11, "دوازده": 12, "سیزده": 13, "چهارده": 14, "چارده": 14,
    "پانزده": 15, "پونزده": 15, "شانزده": 16, "شونزده": 16, "هفده": 17, "هیفده": 17,
    "هجده": 18, "هیجده": 18, "نوزده": 19,
}  # fmt: skip
_TENS = {
    "بیست": 20, "سی": 30, "چهل": 40, "چل": 40, "پنجاه": 50, "پنجا": 50, "شصت": 60,
    "هفتاد": 70, "هشتاد": 80, "نود": 90,
}  # fmt: skip
_HUNDREDS = {
    "صد": 100, "یکصد": 100, "دویست": 200, "سیصد": 300, "چهارصد": 400, "چارصد": 400,
    "پانصد": 500, "پونصد": 500, "ششصد": 600, "شیشصد": 600, "هفتصد": 700, "هفصد": 700,
    "هشتصد": 800, "هشصد": 800, "نهصد": 900,
}  # fmt: skip
_SCALES = {"هزار": 1_000, "میلیون": 1_000_000, "میلیارد": 1_000_000_000}

# Order of size inside a group below a thousand: hundreds (3) > tens and teens (2) > units (1).
_RANK = {**dict.fromkeys(_UNITS, 1), **dict.fromkeys(_TEENS, 2), **dict.fromkeys(_TENS, 2),
         **dict.fromkeys(_HUNDREDS, 3)}  # fmt: skip
_VALUE = {**_UNITS, **_TEENS, **_TENS, **_HUNDREDS}

_WORD = "|".join(sorted([*_VALUE, *_SCALES], key=len, reverse=True))
# A run of number words, each pair separated by " و " or a plain space (before a scale word).
_RUN = re.compile(
    rf"(?<![\w‌])(?:{_WORD})(?:(?: و | )(?:{_WORD}))*(?![\w‌])"
    rf"(?:(?: ممیز )(?:{_WORD})(?:(?: و | )(?:{_WORD}))*(?![\w‌]))?"
    r"(?! و [\w‌])"  # "بیست و سه و هشتا": the number goes on with a word we do not know
)
_TOKEN = re.compile(rf"(?:{_WORD})|و")


def _parse(tokens: list[str]) -> int | None:
    """Value of a number-word sequence (tokens include "و"), or None if it is not one number.

    Rules: "و" joins parts of falling size; a scale word follows its multiplier directly
    ("سه هزار", no "و") and later scales must be smaller.
    """
    total, chunk, last_rank, last_scale, expect_join = 0, 0, 4, None, False
    for tok in tokens:
        if tok == "و":
            if not expect_join:
                return None
            expect_join = False
            continue
        if tok in _SCALES:
            scale = _SCALES[tok]
            if last_scale is not None and scale >= last_scale:
                return None
            total += (chunk or 1) * scale
            chunk, last_rank, last_scale, expect_join = 0, 4, scale, True
            continue
        if expect_join and chunk:  # "دو سه": two numbers in a row without "و"
            return None
        rank = _RANK[tok]
        if rank >= last_rank:
            return None
        chunk += _VALUE[tok]
        last_rank, expect_join = rank, True
    return total + chunk


def _render(value: int) -> str:
    if value < 1_000_000:
        return str(value)
    parts = []
    for name, scale in (("میلیارد", 1_000_000_000), ("میلیون", 1_000_000), ("هزار", 1_000)):
        if value >= scale:
            parts.append(f"{value // scale} {name}")
            value %= scale
    if value:
        parts.append(str(value))
    return " و ".join(parts)


def _convert_number(text: str) -> str | None:
    """Digits for `text` if it is one clear number, else None."""
    tokens = _TOKEN.findall(text)
    words = [t for t in tokens if t != "و"]
    if len(words) == 1:
        return None
    value = _parse(tokens)
    return None if value is None else _render(value)


def _convert_run(text: str, decimal_sep: str) -> str:
    whole, _, frac = text.partition(" ممیز ")
    if frac:
        value, frac_value = _parse(_TOKEN.findall(whole)), _parse(_TOKEN.findall(frac))
        if value is None or frac_value is None or value >= 1_000_000:
            return text
        return f"{value}{decimal_sep}{frac_value}"

    # "صد و بیست و پنج و سه هزار" is two numbers: split at "و" (never at a plain space, so
    # "دو سه هزار" stays a range), taking the longest part that is one number each time.
    parts, out, i = whole.split(" و "), [], 0
    while i < len(parts):
        for j in range(len(parts), i, -1):
            converted = _convert_number(" و ".join(parts[i:j]))
            if converted is not None:
                out.append(converted)
                i = j
                break
        else:
            return whole  # a piece stays words: keep the run as words, not "پنج و 20"
    return " و ".join(out)


def words_to_digits(text: str, decimal_sep: str = ".") -> str:
    """Replace Persian number words with digits where the words clearly form one number."""
    if not text:
        return text
    return _RUN.sub(lambda m: _convert_run(m.group(0), decimal_sep), text)
