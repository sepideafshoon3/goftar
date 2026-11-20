"""Transcript writers: txt, srt, vtt, json."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path

from goftar.core.types import Segment, Transcript

# Subtitle layout (common broadcast guidance): at most 2 lines of about 42 characters per cue.
# Whisper segments run up to ~150 characters, which covers half the screen, so long segments
# become several cues, timed in proportion to their length, and each cue is wrapped in 2 lines.
MAX_LINE_CHARS = 42
MAX_CUE_LINES = 2
_BREAK_AFTER = ("،", ".", "؟", "!", "؛", ",", "?", ":")


def _best_break(words: list[str]) -> int:
    """Index to split `words` into two balanced lines, preferring a break after punctuation."""
    total = len(" ".join(words))
    best, best_cost = 1, float("inf")
    for i in range(1, len(words)):
        left = len(" ".join(words[:i]))
        cost = abs(total - 2 * left) - (12 if words[i - 1].endswith(_BREAK_AFTER) else 0)
        if cost < best_cost:
            best, best_cost = i, cost
    return best


def wrap_lines(text: str, width: int = MAX_LINE_CHARS) -> str:
    """Break a cue's text into two balanced lines when it is longer than `width`."""
    words = text.split()
    if len(text) <= width or len(words) < 2:
        return text
    i = _best_break(words)
    return " ".join(words[:i]) + "\n" + " ".join(words[i:])


def subtitle_cues(seg: Segment) -> list[tuple[float, float, str]]:
    """Split one segment into cues of at most 2 lines x MAX_LINE_CHARS, timed by length."""
    limit = MAX_LINE_CHARS * MAX_CUE_LINES
    words = seg.text.split()
    if len(seg.text) <= limit or len(words) < 2:
        return [(seg.start, seg.end, wrap_lines(seg.text))]
    parts = max(2, -(-len(seg.text) // limit))  # ceil
    target = len(seg.text) / parts
    pieces: list[list[str]] = [[]]
    for word in words:
        current = " ".join(pieces[-1])
        if pieces[-1] and (len(current) >= target or len(current) + 1 + len(word) > limit):
            pieces.append([])
        pieces[-1].append(word)
    texts = [" ".join(p) for p in pieces]
    total = sum(len(t) for t in texts)
    cues, start = [], seg.start
    for t in texts:
        end = start + (seg.end - seg.start) * len(t) / total
        cues.append((start, end, wrap_lines(t)))
        start = end
    cues[-1] = (cues[-1][0], seg.end, cues[-1][2])
    return cues


def _cues(t: Transcript) -> list[tuple[float, float, str]]:
    return [cue for s in t.segments if s.text for cue in subtitle_cues(s)]


def format_timestamp(seconds: float, sep: str = ",") -> str:
    total_ms = round(max(seconds, 0.0) * 1000)
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{ms:03d}"


def to_txt(t: Transcript) -> str:
    return "\n".join(s.text for s in t.segments if s.text) + "\n"


def to_srt(t: Transcript) -> str:
    blocks = []
    for i, (a, b, text) in enumerate(_cues(t), start=1):
        blocks.append(f"{i}\n{format_timestamp(a)} --> {format_timestamp(b)}\n{text}\n")
    return "\n".join(blocks)


def to_vtt(t: Transcript) -> str:
    blocks = ["WEBVTT\n"]
    for a, b, text in _cues(t):
        blocks.append(f"{format_timestamp(a, '.')} --> {format_timestamp(b, '.')}\n{text}\n")
    return "\n".join(blocks)


def to_json(t: Transcript) -> str:
    return json.dumps(t.to_dict(), ensure_ascii=False, indent=2) + "\n"


FORMATS: dict[str, Callable[[Transcript], str]] = {
    "txt": to_txt,
    "srt": to_srt,
    "vtt": to_vtt,
    "json": to_json,
}


def render(t: Transcript, fmt: str) -> str:
    try:
        return FORMATS[fmt](t)
    except KeyError:
        raise ValueError(f"unknown format {fmt!r}; choose from {sorted(FORMATS)}") from None


def write_transcript(
    t: Transcript, out_dir: str | Path, stem: str, formats: Iterable[str]
) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = out / f"{stem}.{fmt}"
        path.write_text(render(t, fmt), encoding="utf-8")
        written.append(path)
    return written
