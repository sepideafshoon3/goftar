"""Helpers for building the benchmark dataset.

Typing a reference transcript from scratch is the slowest part of benchmarking. Instead:

1. ``goftar dataset prepare``  writes ``voice01.draft.txt`` next to each audio file;
2. you listen and correct EVERY word of the draft by hand;
3. ``goftar dataset accept``  renames it to ``voice01.txt``, which makes it a reference.

Drafts use a different file name on purpose: the benchmark only reads ``voice01.txt``, so an
uncorrected draft can never silently become a reference.

Bias warning: a draft comes from a model, so a reference you only skimmed will make that model
(and similar ones) look better than it is. Draft with your strongest model and fix carefully.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from goftar.audio.ingest import SUPPORTED_EXTENSIONS
from goftar.core.pipeline import Pipeline
from goftar.export.formats import to_txt

READY = "ready"  # voice.txt exists: used by the benchmark
DRAFT = "draft"  # voice.draft.txt exists: needs your review
MISSING = "missing"  # neither exists

WRITTEN = "written"
HAS_REFERENCE = "has-reference"
HAS_DRAFT = "has-draft"
FAILED = "failed"


@dataclass(frozen=True)
class Entry:
    audio: Path
    state: str


@dataclass(frozen=True)
class Outcome:
    audio: Path
    status: str
    detail: str = ""


def reference_path(audio: Path) -> Path:
    return audio.with_suffix(".txt")


def draft_path(audio: Path) -> Path:
    return audio.with_name(audio.stem + ".draft.txt")


def list_entries(folder: str | Path) -> list[Entry]:
    entries = []
    for audio in sorted(Path(folder).iterdir()):
        if not audio.is_file() or audio.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if reference_path(audio).is_file():
            state = READY
        elif draft_path(audio).is_file():
            state = DRAFT
        else:
            state = MISSING
        entries.append(Entry(audio, state))
    return entries


def prepare_drafts(
    pipeline: Pipeline, folder: str | Path, *, overwrite: bool = False
) -> list[Outcome]:
    """Write a draft transcript (one segment per line) for every audio file without one."""
    outcomes = []
    for entry in list_entries(folder):
        if entry.state == READY:
            outcomes.append(Outcome(entry.audio, HAS_REFERENCE))
            continue
        if entry.state == DRAFT and not overwrite:
            outcomes.append(Outcome(entry.audio, HAS_DRAFT))
            continue
        try:
            transcript = pipeline.transcribe(entry.audio)
        except Exception as exc:
            outcomes.append(Outcome(entry.audio, FAILED, f"{type(exc).__name__}: {exc}"))
            continue
        draft_path(entry.audio).write_text(to_txt(transcript), encoding="utf-8")
        outcomes.append(Outcome(entry.audio, WRITTEN))
    return outcomes


def accept_draft(audio: Path, *, force: bool = False) -> Path:
    """Promote ``voice.draft.txt`` to ``voice.txt`` (call this only after correcting it)."""
    draft, ref = draft_path(audio), reference_path(audio)
    if not draft.is_file():
        raise FileNotFoundError(f"no draft for {audio.name} (expected {draft.name})")
    if ref.exists() and not force:
        raise FileExistsError(f"{ref.name} already exists (use --force to replace it)")
    draft.replace(ref)
    return ref
