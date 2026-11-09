"""Optional spelling stage: fix homophone-letter typos and apply a personal glossary.

Whisper often writes a Persian word with the wrong one of several letters that sound alike
(ز ذ ض ظ, س ص ث, ت ط, ح ه, غ ق, ا ع). This stage fixes a word only when it is unknown, exactly one
letter is swapped within such a group, and the result is a known word that clearly beats the
other candidates. Everything uncertain is left alone. The glossary is an exact mapping for words
you know the model always writes your way (names, transliterated English).
"""

from __future__ import annotations

import functools
import hashlib
import logging
import re
from collections.abc import Iterable
from pathlib import Path

import yaml

from goftar.errors import ConfigError

log = logging.getLogger(__name__)

ZWNJ = "‌"
LETTERS = "آ-غف-یپچژکگ"
_RUN = re.compile(f"[{LETTERS}]+")
_BOUNDARY_CLASS = f"[{LETTERS}{ZWNJ}]"

CONFUSABLE_GROUPS = ("زذضظ", "سصث", "تط", "حه", "غق", "اع")
_SWAPS: dict[str, str] = {c: g for g in CONFUSABLE_GROUPS for c in g}
_SUFFIXES = ("ها", "های", "هایی", "تر", "ترین", "ی", "ام", "ات", "اش", "مان", "تان", "شان", "ایم")
MIN_LENGTH = 3


def file_digest(path: str | Path) -> str:
    """Short content hash of a file ('missing' if it cannot be read); used in cache keys."""
    try:
        return hashlib.sha256(Path(path).expanduser().read_bytes()).hexdigest()[:16]
    except OSError:
        return "missing"


def load_wordlist(path: str | Path) -> dict[str, int]:
    """One word per line, optionally followed by a count: 'کتاب 120'."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise ConfigError(f"word list not found: {p}")
    words: dict[str, int] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        count = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
        words[parts[0]] = words.get(parts[0], 0) + count
    return words


def load_glossary(path: str | Path) -> dict[str, str]:
    p = Path(path).expanduser()
    if not p.is_file():
        raise ConfigError(f"glossary not found: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {p}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in data.items()
    ):
        raise ConfigError(f"{p} must be a mapping of wrong spelling -> right spelling (strings)")
    return {k: v for k, v in data.items() if k}


def _hazm_words() -> dict[str, int]:
    try:
        try:
            from hazm.utils import words_list
        except ImportError:
            from hazm import words_list
        entries = words_list()
    except Exception as exc:  # hazm missing, or data not shipped
        log.warning("spelling is on but no word list is available (%s); skipping", exc)
        return {}
    words: dict[str, int] = {}
    for entry in entries:
        word, count = (entry[0], entry[1]) if isinstance(entry, tuple) else (entry, 1)
        words[word] = words.get(word, 0) + int(count or 1)
    return words


class SpellFixer:
    def __init__(self, words: dict[str, int] | None = None, glossary: dict[str, str] | None = None):
        self.words = words or {}
        self.glossary = glossary or {}
        keys = sorted(self.glossary, key=len, reverse=True)
        self._glossary_re = (
            re.compile(
                f"(?<!{_BOUNDARY_CLASS})(?:" + "|".join(re.escape(k) for k in keys) + ")"
                f"(?!{_BOUNDARY_CLASS})"
            )
            if keys
            else None
        )

    def _known(self, word: str) -> bool:
        if word in self.words:
            return True
        return any(
            word.endswith(s) and len(word) - len(s) >= 2 and word[: -len(s)] in self.words
            for s in _SUFFIXES
        )

    def fix_word(self, word: str) -> str:
        if not self.words or len(word) < MIN_LENGTH or self._known(word):
            return word
        best: list[str] = []
        best_count = 0
        for i, ch in enumerate(word):
            for alt in _SWAPS.get(ch, "").replace(ch, ""):
                cand = word[:i] + alt + word[i + 1 :]
                count = self.words.get(cand, 0) or (1 if self._known(cand) else 0)
                if count > best_count:
                    best, best_count = [cand], count
                elif count == best_count and count and cand not in best:
                    best.append(cand)
        return best[0] if len(best) == 1 else word

    def _fix_spelling(self, text: str) -> str:
        if not self.words:
            return text
        return _RUN.sub(lambda m: self.fix_word(m.group()), text)

    def fix(self, text: str) -> str:
        if self._glossary_re is None:
            return self._fix_spelling(text)
        out: list[str] = []
        pos = 0
        for m in self._glossary_re.finditer(text):
            out.append(self._fix_spelling(text[pos : m.start()]))
            out.append(self.glossary[m.group()])
            pos = m.end()
        out.append(self._fix_spelling(text[pos:]))
        return "".join(out)


@functools.lru_cache(maxsize=8)
def _build(
    spelling: bool, wordlist: str, wordlist_digest: str, glossary: str, glossary_digest: str
):
    del wordlist_digest, glossary_digest  # only part of the cache key
    words: dict[str, int] = {}
    if spelling:
        words = load_wordlist(wordlist) if wordlist else _hazm_words()
    return SpellFixer(words, load_glossary(glossary) if glossary else {})


def get_fixer(spelling: bool, wordlist: str = "", glossary: str = "") -> SpellFixer:
    return _build(
        spelling,
        wordlist,
        file_digest(wordlist) if wordlist else "",
        glossary,
        file_digest(glossary) if glossary else "",
    )


def fix_words(words: Iterable[str], fixer: SpellFixer) -> list[str]:
    return [fixer.fix_word(w) for w in words]
