"""Transcript writers: txt, srt, vtt, json."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path

from goftar.core.types import Transcript


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
    for i, s in enumerate((s for s in t.segments if s.text), start=1):
        start, end = format_timestamp(s.start), format_timestamp(s.end)
        blocks.append(f"{i}\n{start} --> {end}\n{s.text}\n")
    return "\n".join(blocks)


def to_vtt(t: Transcript) -> str:
    blocks = ["WEBVTT\n"]
    for s in (s for s in t.segments if s.text):
        start, end = format_timestamp(s.start, "."), format_timestamp(s.end, ".")
        blocks.append(f"{start} --> {end}\n{s.text}\n")
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
